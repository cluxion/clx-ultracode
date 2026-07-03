"""Append-only debate journals and replay-aware LLM wrapper."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.core.journal_records import call_record, decode_response, total_tokens

SCHEMA_VERSION = 1
HOME_ENV = "CLUXION_EFFORT_ULTRACODE_HOME"


class ResumeMismatch(ValueError):
    def __init__(self, fields: Mapping[str, Mapping[str, object]]) -> None:
        super().__init__("resume journal does not match current invocation")
        self.fields = dict(fields)


class ResumeNotFound(FileNotFoundError):
    pass


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
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.replay_calls = replay_calls or []
        self._file: object | None = None
        self._pending_header = dict(header) if header is not None else None
        self._warned = False

    @classmethod
    def start(cls, header: Mapping[str, object], *, home: Path | None = None) -> DebateJournal:
        run_id = str(header["run_id"])
        return cls(journals_dir(home) / f"{run_id}.jsonl", run_id, header=header)

    @classmethod
    def resume(
        cls,
        run_id: str,
        expected_header: Mapping[str, object],
        *,
        home: Path | None = None,
    ) -> DebateJournal:
        path = journals_dir(home) / f"{run_id}.jsonl"
        records = read_records(path)
        if not records:
            raise ResumeNotFound(run_id)
        header = records[0]
        if header.get("type") != "header":
            raise ValueError("journal_missing_header")
        mismatches = _header_mismatches(header, expected_header)
        if mismatches:
            raise ResumeMismatch(mismatches)
        calls = [record for record in records if record.get("type") == "call"]
        return cls(path, run_id, calls)

    def append(self, record: Mapping[str, object]) -> None:
        if self._file is None and not self._open_append():
            return
        if self._pending_header is not None:
            if not self._write(self._pending_header):
                return
            self._pending_header = None
        self._write(record)

    def _write(self, record: Mapping[str, object]) -> bool:
        payload = dict(record)
        payload.setdefault("schema_version", SCHEMA_VERSION)
        try:
            self._file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            self._file.flush()
            return True
        except OSError as exc:
            self._warn_once(f"warning: ultracode journal write failed: {exc}")
            self._file = None
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
        fd: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            # The plugin home above journals/ may hold other run artifacts;
            # keep it private too, matching the forgetforge home policy.
            os.chmod(self.path.parent.parent, 0o700)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.chmod(self.path, 0o600)
            self._file = os.fdopen(fd, "a", encoding="utf-8")
            fd = None
            return True
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            self._warn_once(f"warning: ultracode journal disabled: {exc}")
            self._file = None
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
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(record, dict):
            records.append(record)
    return records


def journal_header(run_id: str, *, home: Path | None = None) -> dict[str, Any]:
    records = read_records(journals_dir(home) / f"{run_id}.jsonl")
    if not records or records[0].get("type") != "header":
        raise ValueError("journal_missing_header")
    return records[0]


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
