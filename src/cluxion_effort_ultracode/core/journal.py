"""Append-only debate journals and replay-aware LLM wrapper."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from cluxion_effort_ultracode.core.journal_records import call_record, decode_response, total_tokens

SCHEMA_VERSION = 1
HOME_ENV = "CLUXION_EFFORT_ULTRACODE_HOME"

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX hosts
    _fcntl = None  # type: ignore[assignment]


class ResumeMismatch(ValueError):
    def __init__(self, fields: Mapping[str, Mapping[str, object]]) -> None:
        super().__init__("resume journal does not match current invocation")
        self.fields = dict(fields)


class ResumeNotFound(FileNotFoundError):
    pass


class JournalBusy(OSError):
    """Raised when another process holds the per-run advisory lock."""

    def __init__(self, run_id: str) -> None:
        super().__init__(run_id)
        self.run_id = run_id


class JournalLockUnsupported(RuntimeError):
    """Raised when resume/GC mutation requires a lock that the host cannot provide."""

    def __init__(self, message: str = "POSIX advisory locks unavailable") -> None:
        super().__init__(message)


class JournalCorrupt(ValueError):
    """Raised when a journal contains an unrepairable corrupt JSONL record."""

    def __init__(self, run_id: str, message: str = "journal contains corrupt record") -> None:
        super().__init__(message)
        self.run_id = run_id


def ultracode_home() -> Path:
    return Path(os.getenv(HOME_ENV, "~/.cluxion-ultracode")).expanduser()


def journals_dir(home: Path | None = None) -> Path:
    return (home or ultracode_home()) / "journals"


def context_hash(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def build_header(
    *,
    run_id: str,
    question: str,
    context: str,
    agents_count: int,
    max_rounds: int,
    models: list[str],
    adapter: str,
    agent_timeout_s: float,
    debate_budget_s: float,
    budget_tokens: int | None,
) -> dict[str, object]:
    return {
        "type": "header",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "question": question,
        "context": context,
        "context_hash": context_hash(context),
        "agents_count": agents_count,
        "max_rounds": max_rounds,
        "models": list(models),
        "adapter": adapter,
        "agent_timeout_s": agent_timeout_s,
        "debate_budget_s": debate_budget_s,
        "budget_tokens": budget_tokens,
    }


class DebateJournal:
    def __init__(
        self,
        path: Path,
        run_id: str,
        replay_calls: list[dict[str, Any]] | None = None,
        header: Mapping[str, object] | None = None,
        *,
        file: TextIO | None = None,
        locked: bool = False,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.replay_calls = replay_calls or []
        self.header = dict(header) if header is not None else None
        self._file: TextIO | None = file
        self._pending_header = dict(header) if header is not None and file is None else None
        self._warned = False
        self._locked = locked
        self._writes_disabled = False
        self._closed = False

    @classmethod
    def start(cls, header: Mapping[str, object], *, home: Path | None = None) -> DebateJournal:
        run_id = str(header["run_id"])
        return cls(journals_dir(home) / f"{run_id}.jsonl", run_id, header=header)

    @classmethod
    def resume(
        cls,
        run_id: str,
        expected_header: Mapping[str, object] | None = None,
        *,
        home: Path | None = None,
    ) -> DebateJournal:
        path = journals_dir(home) / f"{run_id}.jsonl"
        if not path.exists():
            raise ResumeNotFound(run_id)
        if _fcntl is None:
            raise JournalLockUnsupported("POSIX advisory locks unavailable for journal resume")

        fd: int | None = None
        file_obj: TextIO | None = None
        try:
            # Same FD for lock/read/append: kernel O_APPEND retained through close.
            try:
                fd = os.open(path, os.O_RDWR | os.O_APPEND)
            except FileNotFoundError as exc:
                # Exists check passed but path vanished before open (unlink race).
                raise ResumeNotFound(run_id) from exc
            _lock_exclusive_nonblocking(fd, run_id)
            _verify_same_inode(fd, path, run_id)
            _tighten_journal_modes(path)
            records = _read_records_from_fd(fd, run_id=run_id, mutate=True)
            if not records:
                raise ResumeNotFound(run_id)
            header = records[0]
            if header.get("type") != "header":
                raise ValueError("journal_missing_header")
            if expected_header is not None:
                mismatches = _header_mismatches(header, expected_header)
                if mismatches:
                    raise ResumeMismatch(mismatches)
            calls = [record for record in records if record.get("type") == "call"]
            os.lseek(fd, 0, os.SEEK_END)
            file_obj = os.fdopen(fd, "a", encoding="utf-8")
            fd = None
            journal = cls(path, run_id, calls, header=header, file=file_obj, locked=True)
            file_obj = None
            return journal
        except Exception:
            if file_obj is not None:
                with contextlib.suppress(OSError):
                    file_obj.close()
            elif fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            raise

    def ensure_matches(self, expected_header: Mapping[str, object]) -> None:
        if not self.header:
            raise ValueError("journal_missing_header")
        mismatches = _header_mismatches(self.header, expected_header)
        if mismatches:
            raise ResumeMismatch(mismatches)

    def close(self) -> None:
        """Release ownership (advisory lock via fd close). Idempotent."""
        if self._closed:
            return
        self._closed = True
        file_obj = self._file
        self._file = None
        self._locked = False
        if file_obj is not None:
            with contextlib.suppress(Exception):
                file_obj.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort GC release
        with contextlib.suppress(Exception):
            self.close()

    def append(self, record: Mapping[str, object]) -> None:
        if self._closed or self._writes_disabled:
            return
        if self._file is None and not self._open_append():
            return
        if self._pending_header is not None:
            if not self._write(self._pending_header):
                return
            self._pending_header = None
        self._write(record)

    def _write(self, record: Mapping[str, object]) -> bool:
        if self._file is None or self._writes_disabled:
            return False
        payload = dict(record)
        payload.setdefault("schema_version", SCHEMA_VERSION)
        try:
            self._file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            self._file.flush()
            return True
        except OSError as exc:
            self._warn_once(f"warning: ultracode journal write failed: {exc}")
            # Disable further writes but retain ownership until close.
            self._writes_disabled = True
            return False

    def append_result(self, result: object) -> None:
        self.append(
            {
                "type": "result",
                "run_id": self.run_id,
                "status": getattr(result, "status", None),
                "tokens_spent": getattr(result, "tokens_spent", 0),
                "tokens_replayed": getattr(result, "tokens_replayed", 0),
                "rounds": getattr(result, "rounds", None),
            }
        )

    def _open_append(self) -> bool:
        """Lazy open for new runs: exclusive create; fail-open if claim/init fails."""
        if self._closed or self._writes_disabled:
            return False
        fd: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            # The plugin home above journals/ may hold other run artifacts;
            # keep it private too, matching the forgetforge home policy.
            os.chmod(self.path.parent.parent, 0o700)
            # O_EXCL: never write header/call bytes into a colliding existing file
            # (including zero-byte). Debate continues with warning fail-open.
            fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600)
            os.chmod(self.path, 0o600)

            if _fcntl is not None:
                try:
                    _lock_exclusive_nonblocking(fd, self.run_id)
                except JournalBusy:
                    # Terminal for this object: disable so later appends cannot open
                    # again and create seq gaps (e.g. seq=1 without seq=0).
                    return self._disable_writes_terminal(
                        "warning: ultracode journal disabled: run journal busy",
                        fd=fd,
                    )
                except JournalLockUnsupported as exc:
                    # O_EXCL already created a zero-byte file. Close the owned FD once,
                    # leave the empty file in place (do not unlink / write unlocked), and
                    # disable further opens for this journal object so the debate continues.
                    return self._disable_writes_terminal(
                        f"warning: ultracode journal disabled: {exc}",
                        fd=fd,
                    )
                try:
                    _verify_same_inode(fd, self.path, self.run_id)
                except (ResumeNotFound, OSError) as exc:
                    return self._disable_writes_terminal(
                        f"warning: ultracode journal disabled: {exc}",
                        fd=fd,
                    )
                self._locked = True
            # Without fcntl, new-run journaling remains fail-open (no exclusive ownership).

            self._file = os.fdopen(fd, "a", encoding="utf-8")
            fd = None
            return True
        except OSError as exc:
            # EEXIST / other terminal open failures: same-object fail-open disable so
            # retries cannot allocate later seqs without the first write succeeding.
            if exc.errno == errno.EEXIST:
                message = "warning: ultracode journal disabled: run_id collision"
            else:
                message = f"warning: ultracode journal disabled: {exc}"
            return self._disable_writes_terminal(message, fd=fd)

    def _disable_writes_terminal(self, message: str, *, fd: int | None = None) -> bool:
        """Terminal new-run open failure: close any owned FD, clear lock, disable writes."""
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._file = None
        self._locked = False
        self._writes_disabled = True
        self._warn_once(message)
        return False

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(message, file=sys.stderr)
            self._warned = True


class LazyLlm:
    def __init__(self, factory: Callable[[], object]) -> None:
        self._factory = factory
        self._llm: object | None = None

    @property
    def last_usage(self) -> object:
        return getattr(self._llm, "last_usage", None)

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any] | str:
        llm = self._get()
        if model is None:
            return llm.complete(prompt, schema=schema)
        return llm.complete(prompt, schema=schema, model=model)

    def _get(self) -> Any:
        if self._llm is None:
            llm = self._factory()
            if not callable(getattr(llm, "complete", None)):
                raise ValueError("llm_factory must return an object with complete(...)")
            self._llm = llm
        return self._llm


class JournaledLlm:
    serial_complete = True

    def __init__(self, llm: object, journal: DebateJournal) -> None:
        self.llm = llm
        self.journal = journal
        self.seq = 0
        self.tokens_replayed = 0
        self.last_usage: object = None
        self._replaying = bool(journal.replay_calls)

    @property
    def run_id(self) -> str:
        return self.journal.run_id

    @property
    def journal_path(self) -> str:
        return str(self.journal.path)

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any] | str:
        prompt_sha = prompt_hash(prompt)
        replay = self._replay_match(prompt_sha, model)
        if replay is not None:
            self.seq += 1
            self.tokens_replayed += total_tokens(replay.get("tokens"))
            self.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated": False}
            return decode_response(replay)

        self._replaying = False
        started = time.monotonic()
        if model is None:
            raw = self.llm.complete(prompt, schema=schema)
        else:
            raw = self.llm.complete(prompt, schema=schema, model=model)
        duration_ms = int((time.monotonic() - started) * 1000)
        self.last_usage = getattr(self.llm, "last_usage", None)
        self.journal.append(call_record(self.seq, prompt, prompt_sha, model, raw, self.last_usage, duration_ms))
        self.seq += 1
        return raw

    def _replay_match(self, prompt_sha: str, model: str | None) -> dict[str, Any] | None:
        if not self._replaying or self.seq >= len(self.journal.replay_calls):
            return None
        record = self.journal.replay_calls[self.seq]
        if record.get("seq") == self.seq and record.get("prompt_sha256") == prompt_sha and record.get("model") == model:
            return record
        return None


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ResumeNotFound(path.stem)
    return _parse_jsonl_bytes(path.read_bytes(), run_id=path.stem).records


def journal_header(run_id: str, *, home: Path | None = None) -> dict[str, Any]:
    records = read_records(journals_dir(home) / f"{run_id}.jsonl")
    if not records or records[0].get("type") != "header":
        raise ValueError("journal_missing_header")
    return records[0]


def try_lock_journal_fd(fd: int, run_id: str) -> None:
    """Acquire nonblocking exclusive lock on an already-open journal fd."""
    if _fcntl is None:
        raise JournalLockUnsupported("POSIX advisory locks unavailable")
    _lock_exclusive_nonblocking(fd, run_id)


def locks_supported() -> bool:
    return _fcntl is not None


def _lock_exclusive_nonblocking(fd: int, run_id: str) -> None:
    assert _fcntl is not None
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise JournalBusy(run_id) from exc
    except OSError as exc:
        if exc.errno in {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK}:
            raise JournalBusy(run_id) from exc
        if exc.errno == errno.ENOLCK:
            # Transient lock-table exhaustion: same typed surface as ENOTSUP for callers,
            # but message/cause preserve that resources are temporarily unavailable/retryable.
            raise JournalLockUnsupported(
                "POSIX advisory locks temporarily unavailable (ENOLCK): lock resources exhausted; retryable"
            ) from exc
        unsupported = {errno.ENOSYS, errno.ENOTSUP}
        eopnotsupp = getattr(errno, "EOPNOTSUPP", None)
        if eopnotsupp is not None:
            unsupported.add(eopnotsupp)
        if exc.errno in unsupported:
            raise JournalLockUnsupported("POSIX advisory locks unsupported on this filesystem/host") from exc
        raise


def _verify_same_inode(fd: int, path: Path, run_id: str) -> None:
    fd_stat = os.fstat(fd)
    try:
        path_stat = path.stat()
    except FileNotFoundError as exc:
        raise ResumeNotFound(run_id) from exc
    if fd_stat.st_ino != path_stat.st_ino or fd_stat.st_dev != path_stat.st_dev:
        raise ResumeNotFound(run_id)


def _read_fd_bytes(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        piece = os.read(fd, 65536)
        if not piece:
            break
        chunks.append(piece)
    return b"".join(chunks)


def _read_records_from_fd(
    fd: int,
    *,
    run_id: str,
    mutate: bool = False,
) -> list[dict[str, Any]]:
    """Parse journal bytes from an open FD; optionally repair the final non-LF segment."""
    data = _read_fd_bytes(fd)
    parsed = _parse_jsonl_bytes(data, run_id=run_id)
    if mutate:
        if parsed.trailing_action == "truncate":
            os.ftruncate(fd, parsed.validated_end)
        elif parsed.trailing_action == "append_lf":
            os.lseek(fd, 0, os.SEEK_END)
            os.write(fd, b"\n")
    return parsed.records


class _JsonlParseResult:
    __slots__ = ("records", "trailing_action", "validated_end")

    def __init__(
        self,
        records: list[dict[str, Any]],
        validated_end: int,
        trailing_action: str,
    ) -> None:
        self.records = records
        self.validated_end = validated_end
        # "none" | "truncate" | "append_lf"
        self.trailing_action = trailing_action


def _parse_jsonl_bytes(data: bytes, *, run_id: str) -> _JsonlParseResult:
    """Shared bytes-based JSONL parser for path readers and resume FD readers.

    Tracks validated record boundaries. Newline-terminated or mid-file invalid
    records raise JournalCorrupt (callers must not mutate). A final segment
    without LF that fails UTF-8/JSON decode is reported as
    trailing_action='truncate' at the start of that segment; a successfully
    decoded non-object final segment is JournalCorrupt (no mutation); a valid
    final JSON object without LF is kept with trailing_action='append_lf'.
    """
    records: list[dict[str, Any]] = []
    offset = 0
    validated_end = 0
    length = len(data)

    while offset < length:
        nl = data.find(b"\n", offset)
        if nl == -1:
            segment = data[offset:]
            if not segment:
                break
            try:
                text = segment.decode("utf-8")
            except UnicodeDecodeError:
                return _JsonlParseResult(records, offset, "truncate")
            if not text.strip():
                # Trailing whitespace-only segment: leave bytes as-is for readers.
                break
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                return _JsonlParseResult(records, offset, "truncate")
            if not isinstance(record, dict):
                raise JournalCorrupt(run_id, "journal contains non-object JSONL record")
            records.append(record)
            return _JsonlParseResult(records, length, "append_lf")

        line_bytes = data[offset:nl]
        line_end = nl + 1
        try:
            text = line_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JournalCorrupt(run_id, "journal contains corrupt UTF-8 record") from exc
        if not text.strip():
            offset = line_end
            validated_end = line_end
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JournalCorrupt(run_id, "journal contains corrupt JSONL record") from exc
        if not isinstance(record, dict):
            raise JournalCorrupt(run_id, "journal contains non-object JSONL record")
        records.append(record)
        offset = line_end
        validated_end = line_end

    return _JsonlParseResult(records, validated_end if validated_end or not data else 0, "none")


def _tighten_journal_modes(path: Path) -> None:
    try:
        os.chmod(path.parent, 0o700)
        if path.parent.parent.exists():
            os.chmod(path.parent.parent, 0o700)
        os.chmod(path, 0o600)
    except OSError:
        pass


def _header_mismatches(
    header: Mapping[str, object],
    expected: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    fields = (
        "question",
        "context_hash",
        "agents_count",
        "max_rounds",
        "models",
        "adapter",
        "agent_timeout_s",
        "debate_budget_s",
        "budget_tokens",
    )
    return {
        field: {"journal": header.get(field), "current": expected.get(field)}
        for field in fields
        if header.get(field) != expected.get(field)
    }
