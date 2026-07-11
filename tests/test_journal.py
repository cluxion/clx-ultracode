"""Regression tests for debate journals and resume replay."""

from __future__ import annotations

import errno
import json
import os
import stat
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cluxion_effort_ultracode.core import ConsensusEngine
from cluxion_effort_ultracode.core.journal import (
    DebateJournal,
    JournaledLlm,
    ResumeMismatch,
    build_header,
)
from cluxion_effort_ultracode.core.journal_lifecycle import gc_journals
from cluxion_effort_ultracode.core.journal_records import decode_response


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class ScriptedLlm:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = deque(outputs)
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any]:
        del schema, model
        self.calls.append(prompt)
        return self.outputs.popleft()


class ExplodingLlm:
    def complete(self, prompt: str, *, schema: object = None, model: str | None = None) -> Mapping[str, Any]:
        del prompt, schema, model
        pytest.fail("backend should not be called during full replay")


def position(stance: str) -> dict[str, Any]:
    return {"stance": stance, "rationale": f"R {stance}", "evidence": [f"E {stance}"], "confidence": 0.8}


def update(stance: str, *, conceded: bool = False) -> dict[str, Any]:
    return {
        **position(stance),
        "conceded": [{"point": "old", "reason": "new evidence wins"}] if conceded else [],
        "maintained": [] if conceded else [{"point": stance, "reason": "still strongest"}],
    }


def header(run_id: str = "abc123", *, question: str = "Q?", home_adapter: str = "test") -> dict[str, object]:
    return build_header(
        run_id=run_id,
        question=question,
        context="ctx",
        agents_count=2,
        max_rounds=1,
        models=[],
        adapter=home_adapter,
        agent_timeout_s=10.0,
        debate_budget_s=30.0,
        budget_tokens=None,
    )


def test_journal_is_created_lazily_with_private_modes(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)

    assert not journal.path.exists()

    ConsensusEngine(
        JournaledLlm(ScriptedLlm([position("Adopt"), position("ADOPT.")]), journal),
        agents_count=2,
        max_rounds=0,
    ).decide("Q?", context="ctx")

    assert mode(journal.path.parent) == 0o700
    assert mode(journal.path) == 0o600
    records = json.loads(journal.path.read_text(encoding="utf-8").splitlines()[0])
    assert records["type"] == "header"


def test_decode_response_degrades_gracefully_at_replay_boundary() -> None:
    assert decode_response({"response_type": "json"}) == ""
    assert decode_response({"response_type": "json", "response_text": "{not json"}) == "{not json"
    assert decode_response({"response_type": "json", "response_text": '{"a": 1}'}) == {"a": 1}
    assert decode_response({"response_type": "text", "response_text": "hi"}) == "hi"


def test_resume_append_tightens_existing_journal_modes(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)
    journal.append({"type": "call", "run_id": "abc123"})
    journal.close()
    os.chmod(journal.path.parent, 0o755)
    os.chmod(journal.path, 0o644)

    resumed = DebateJournal.resume("abc123", run_header, home=tmp_path)
    resumed.append({"type": "result", "run_id": "abc123"})
    resumed.close()

    assert mode(journal.path.parent) == 0o700
    assert mode(journal.path) == 0o600


def test_crash_then_resume_replays_prefix_and_continues(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)
    first_llm = ScriptedLlm([position("Adopt"), position("Delay"), update("Adopt"), update("Adopt", conceded=True)])
    first = ConsensusEngine(JournaledLlm(first_llm, journal), agents_count=2, max_rounds=1).decide("Q?", context="ctx")
    assert first.status == "unanimous"
    journal.close()

    lines = journal.path.read_text(encoding="utf-8").splitlines()
    journal.path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")

    resumed_journal = DebateJournal.resume("abc123", run_header, home=tmp_path)
    suffix_llm = ScriptedLlm([update("Adopt"), update("Adopt", conceded=True)])
    resumed = ConsensusEngine(JournaledLlm(suffix_llm, resumed_journal), agents_count=2, max_rounds=1).decide(
        "Q?",
        context="ctx",
    )
    resumed_journal.close()

    assert resumed.status == "unanimous"
    assert len(suffix_llm.calls) == 2
    assert resumed.tokens_replayed > 0


def test_resume_mismatch_reports_changed_question(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)
    journal.append({"type": "call", "run_id": "abc123"})
    journal.close()
    changed = header(question="Different?")

    with pytest.raises(ResumeMismatch) as exc:
        DebateJournal.resume("abc123", changed, home=tmp_path)

    assert "question" in exc.value.fields


def test_completed_run_full_replay_is_deterministic(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)
    first = ConsensusEngine(
        JournaledLlm(ScriptedLlm([position("Adopt"), position("ADOPT.")]), journal),
        agents_count=2,
        max_rounds=1,
    ).decide("Q?", context="ctx")
    journal.close()

    replayed_journal = DebateJournal.resume("abc123", run_header, home=tmp_path)
    replayed = ConsensusEngine(JournaledLlm(ExplodingLlm(), replayed_journal), agents_count=2, max_rounds=1).decide(
        "Q?",
        context="ctx",
    )
    replayed_journal.close()

    assert replayed.status == first.status
    assert replayed.decision == first.decision
    assert replayed.tokens_spent == 0
    assert replayed.tokens_replayed == first.tokens_spent


def test_journal_write_failure_warns_and_debate_continues(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    class BrokenFile:
        def write(self, value: str) -> int:
            del value
            raise OSError("readonly")

        def flush(self) -> None:
            pass

    journal = DebateJournal.start(header(), home=tmp_path)
    journal._file = BrokenFile()
    result = ConsensusEngine(
        JournaledLlm(ScriptedLlm([position("Adopt"), position("ADOPT.")]), journal),
        agents_count=2,
        max_rounds=0,
    ).decide("Q?", context="ctx")

    assert result.status == "unanimous"
    assert "journal write failed" in capsys.readouterr().err


def test_journals_gc_respects_cutoff_and_dry_run(tmp_path: Path) -> None:
    directory = tmp_path / "journals"
    directory.mkdir()
    old = dict(header("old123"))
    old["created_at"] = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    new = dict(header("new123"))
    new["created_at"] = datetime.now(UTC).isoformat()
    (directory / "old123.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    (directory / "new123.jsonl").write_text(json.dumps(new) + "\n", encoding="utf-8")

    dry = gc_journals(home=tmp_path, older_than_days=7)
    assert [item["run_id"] for item in dry["candidates"]] == ["old123"]
    assert (directory / "old123.jsonl").exists()

    applied = gc_journals(home=tmp_path, older_than_days=7, apply=True)
    assert [item["run_id"] for item in applied["candidates"]] == ["old123"]
    assert not (directory / "old123.jsonl").exists()
    assert (directory / "new123.jsonl").exists()


def test_journals_gc_rejects_huge_older_than_days_before_mutation(tmp_path: Path) -> None:
    directory = tmp_path / "journals"
    directory.mkdir()
    path = directory / "keep123.jsonl"
    path.write_text(json.dumps(header("keep123")) + "\n", encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="older_than_days"):
        gc_journals(home=tmp_path, older_than_days=10**20, apply=True)

    assert path.read_text(encoding="utf-8") == before


def test_journals_gc_rejects_datetime_cutoff_overflow_before_filesystem_access(tmp_path: Path) -> None:
    home = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match="older_than_days"):
        gc_journals(home=home, older_than_days=999_999_999, apply=True)

    assert not home.exists()


def test_debate_journal_close_is_idempotent(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)
    journal.append({"type": "call", "run_id": "abc123", "seq": 0})
    journal.close()
    journal.close()  # must not raise


def test_resume_second_holder_is_journal_busy(tmp_path: Path) -> None:
    from cluxion_effort_ultracode.core.journal import JournalBusy

    run_header = header()
    first = DebateJournal.start(run_header, home=tmp_path)
    first.append({"type": "call", "run_id": "abc123", "seq": 0})
    first.close()

    holder = DebateJournal.resume("abc123", run_header, home=tmp_path)
    try:
        with pytest.raises(JournalBusy) as exc:
            DebateJournal.resume("abc123", run_header, home=tmp_path)
        assert exc.value.run_id == "abc123"
    finally:
        holder.close()


def test_multiprocess_resume_busy_gc_preserves_active_and_exit_releases(tmp_path: Path) -> None:
    """Real OS lock across processes: busy resume, GC skips active, exit releases, then resume+GC work."""
    import multiprocessing as mp

    from cluxion_effort_ultracode.core.journal import JournalBusy
    from mp_helpers import hold_journal_until_release

    run_header = header(run_id="lockrun1")
    seed = DebateJournal.start(run_header, home=tmp_path)
    seed.append({"type": "call", "run_id": "lockrun1", "seq": 0})
    seed.close()

    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Queue()

    proc = ctx.Process(target=hold_journal_until_release, args=(str(tmp_path), "lockrun1", ready, release))
    proc.start()
    assert ready.get(timeout=10) == "ready"

    with pytest.raises(JournalBusy):
        DebateJournal.resume("lockrun1", run_header, home=tmp_path)

    old = dict(run_header)
    old["created_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    journal_path = tmp_path / "journals" / "lockrun1.jsonl"
    body = journal_path.read_text(encoding="utf-8").splitlines()
    body[0] = json.dumps(old)
    journal_path.write_text("\n".join(body) + "\n", encoding="utf-8")

    applied_busy = gc_journals(home=tmp_path, older_than_days=7, apply=True)
    assert journal_path.exists()
    busy_ids = [c["run_id"] for c in applied_busy["candidates"]]
    assert "lockrun1" not in busy_ids

    release.put("done")
    proc.join(timeout=10)
    assert proc.exitcode == 0

    resumed = DebateJournal.resume("lockrun1", run_header, home=tmp_path)
    resumed.close()

    body = journal_path.read_text(encoding="utf-8").splitlines()
    body[0] = json.dumps(old)
    journal_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    applied = gc_journals(home=tmp_path, older_than_days=7, apply=True)
    assert [c["run_id"] for c in applied["candidates"]] == ["lockrun1"]
    assert not journal_path.exists()


def test_journal_write_failure_retains_ownership_until_close(tmp_path: Path) -> None:
    from cluxion_effort_ultracode.core.journal import JournalBusy

    run_header = header()
    seed = DebateJournal.start(run_header, home=tmp_path)
    seed.append({"type": "call", "run_id": "abc123", "seq": 0})
    seed.close()

    class BrokenFile:
        def write(self, value: str) -> int:
            del value
            raise OSError("readonly")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    # Replace write surface but keep underlying locked fd ownership via journal internals.
    locked = DebateJournal.resume("abc123", run_header, home=tmp_path)
    try:
        real_file = locked._file
        locked._file = BrokenFile()
        locked.append({"type": "call", "run_id": "abc123", "seq": 1})
        # Ownership retained: second resume still busy.
        with pytest.raises(JournalBusy):
            DebateJournal.resume("abc123", run_header, home=tmp_path)
        locked._file = real_file
    finally:
        locked.close()

    # After close, resume works again.
    free = DebateJournal.resume("abc123", run_header, home=tmp_path)
    free.close()


def test_new_run_lazy_open_refuses_collision_including_zero_bytes(tmp_path: Path) -> None:
    """O_EXCL new-run create: existing file (even empty) never receives header/call bytes."""
    run_header = header(run_id="collide1")
    journal = DebateJournal.start(run_header, home=tmp_path)
    journal.path.parent.mkdir(parents=True, exist_ok=True)
    journal.path.write_bytes(b"")  # zero-byte collision
    before = journal.path.read_bytes()

    journal.append({"type": "call", "run_id": "collide1", "seq": 0})

    assert journal.path.read_bytes() == before
    assert before == b""


def test_new_run_lazy_open_refuses_nonempty_collision_bytes_unchanged(tmp_path: Path) -> None:
    run_header = header(run_id="collide2")
    journal = DebateJournal.start(run_header, home=tmp_path)
    journal.path.parent.mkdir(parents=True, exist_ok=True)
    preexisting = b'{"type":"header","run_id":"collide2"}\n'
    journal.path.write_bytes(preexisting)

    journal.append({"type": "call", "run_id": "collide2", "seq": 0})

    assert journal.path.read_bytes() == preexisting


@pytest.mark.skipif(not hasattr(os, "O_APPEND"), reason="POSIX open flags required")
def test_resume_opens_same_fd_with_kernel_o_append(tmp_path: Path) -> None:
    import fcntl

    run_header = header()
    seed = DebateJournal.start(run_header, home=tmp_path)
    seed.append({"type": "call", "run_id": "abc123", "seq": 0})
    seed.close()

    resumed = DebateJournal.resume("abc123", run_header, home=tmp_path)
    try:
        assert resumed._file is not None
        flags = fcntl.fcntl(resumed._file.fileno(), fcntl.F_GETFL)
        assert flags & os.O_APPEND, f"resume FD missing O_APPEND (flags={flags:#x})"
        # Same live FD retained through ownership lifetime (not reopened as a second handle).
        fd = resumed._file.fileno()
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_APPEND
    finally:
        resumed.close()


def test_destructive_gc_without_lock_support_raises_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cluxion_effort_ultracode.core.journal_lifecycle as lifecycle
    from cluxion_effort_ultracode.core.journal import JournalLockUnsupported

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "oldrun.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "header",
                "run_id": "oldrun",
                "created_at": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    monkeypatch.setattr(lifecycle, "locks_supported", lambda: False)

    with pytest.raises(JournalLockUnsupported):
        gc_journals(home=tmp_path, older_than_days=7, apply=True)

    assert path.read_bytes() == before
    assert path.exists()


def test_gc_dry_run_without_lock_support_stays_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cluxion_effort_ultracode.core.journal_lifecycle as lifecycle

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "oldrun.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "header",
                "run_id": "oldrun",
                "created_at": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    monkeypatch.setattr(lifecycle, "locks_supported", lambda: False)

    dry = gc_journals(home=tmp_path, older_than_days=7, apply=False)
    assert path.read_bytes() == before
    assert [c["run_id"] for c in dry["candidates"]] == ["oldrun"]


def test_runtime_flock_enotsup_apply_raises_typed_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import JournalLockUnsupported

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "oldrun.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "header",
                "run_id": "oldrun",
                "created_at": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None

    class _FlockEnotsup:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del fd, flags
            raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnotsup())

    with pytest.raises(JournalLockUnsupported) as exc:
        gc_journals(home=tmp_path, older_than_days=7, apply=True)

    assert isinstance(exc.value.__cause__, OSError)
    assert exc.value.__cause__.errno == errno.ENOTSUP
    assert path.read_bytes() == before
    assert path.exists()


def test_runtime_flock_enotsup_dry_run_returns_old_candidate_unlocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "oldrun.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "header",
                "run_id": "oldrun",
                "created_at": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None

    class _FlockEnotsup:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del fd, flags
            raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnotsup())

    dry = gc_journals(home=tmp_path, older_than_days=7, apply=False)
    assert path.read_bytes() == before
    assert [c["run_id"] for c in dry["candidates"]] == ["oldrun"]


def test_resume_runtime_flock_enotsup_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import JournalLockUnsupported

    run_header = header()
    seed = DebateJournal.start(run_header, home=tmp_path)
    seed.append({"type": "call", "run_id": "abc123", "seq": 0})
    seed.close()

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None

    class _FlockEnotsup:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del fd, flags
            raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnotsup())

    with pytest.raises(JournalLockUnsupported) as exc:
        DebateJournal.resume("abc123", run_header, home=tmp_path)
    assert isinstance(exc.value.__cause__, OSError)
    assert exc.value.__cause__.errno == errno.ENOTSUP


def test_new_run_flock_enotsup_fail_open_disables_writes_no_fd_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """fcntl present + flock ENOTSUP after O_EXCL: close FD once, leave zero-byte file, disable writes."""
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None
    real_open = journal_mod.os.open
    real_close = journal_mod.os.close
    captured: dict[str, int | None] = {"fd": None, "closes": 0}

    class _FlockEnotsup:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del flags
            captured["fd"] = fd
            raise OSError(errno.ENOTSUP, "Operation not supported")

    def tracking_open(path_arg, flags, mode=0o777):
        fd = real_open(path_arg, flags, mode)
        return fd

    def tracking_close(fd: int) -> None:
        if captured["fd"] is not None and fd == captured["fd"]:
            captured["closes"] = int(captured["closes"]) + 1
        return real_close(fd)

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnotsup())
    monkeypatch.setattr(journal_mod.os, "open", tracking_open)
    monkeypatch.setattr(journal_mod.os, "close", tracking_close)

    run_header = header(run_id="enotsup1")
    journal = DebateJournal.start(run_header, home=tmp_path)

    journal.append({"type": "call", "run_id": "enotsup1", "seq": 0})
    journal.append({"type": "call", "run_id": "enotsup1", "seq": 1})

    owned_fd = captured["fd"]
    assert owned_fd is not None
    assert captured["closes"] == 1
    with pytest.raises(OSError) as bad:
        os.fstat(owned_fd)
    assert bad.value.errno == errno.EBADF

    assert journal.path.exists()
    assert journal.path.read_bytes() == b""
    assert journal._file is None
    assert journal._locked is False
    assert journal._writes_disabled is True

    err = capsys.readouterr().err
    assert err.count("warning: ultracode journal disabled:") == 1
    assert "unsupported" in err.lower() or "ENOTSUP" in err or "not supported" in err.lower()


def test_lock_exclusive_enolck_maps_to_typed_with_retryable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ENOLCK → JournalLockUnsupported preserving cause errno and retryable message."""
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import JournalLockUnsupported, _lock_exclusive_nonblocking

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None
    cause = OSError(errno.ENOLCK, "No locks available")

    class _FlockEnolck:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del fd, flags
            raise cause

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnolck())

    with pytest.raises(JournalLockUnsupported) as exc:
        _lock_exclusive_nonblocking(0, "run-x")

    assert exc.value.__cause__ is cause
    assert isinstance(exc.value.__cause__, OSError)
    assert exc.value.__cause__.errno == errno.ENOLCK
    message = str(exc.value).lower()
    assert "enolck" in message
    assert "temporarily unavailable" in message or "retryable" in message
    assert "lock resources" in message or "resources" in message


def test_gc_apply_enolck_raises_typed_preserves_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import JournalLockUnsupported

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "oldrun.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "header",
                "run_id": "oldrun",
                "created_at": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None

    class _FlockEnolck:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del fd, flags
            raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnolck())

    with pytest.raises(JournalLockUnsupported) as exc:
        gc_journals(home=tmp_path, older_than_days=7, apply=True)

    assert isinstance(exc.value.__cause__, OSError)
    assert exc.value.__cause__.errno == errno.ENOLCK
    assert "retryable" in str(exc.value).lower() or "temporarily unavailable" in str(exc.value).lower()
    assert path.read_bytes() == before
    assert path.exists()


def test_gc_dry_run_enolck_stays_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "oldrun.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "header",
                "run_id": "oldrun",
                "created_at": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None

    class _FlockEnolck:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del fd, flags
            raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnolck())

    dry = gc_journals(home=tmp_path, older_than_days=7, apply=False)
    assert path.read_bytes() == before
    assert [c["run_id"] for c in dry["candidates"]] == ["oldrun"]


def test_resume_enolck_is_typed_with_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import JournalLockUnsupported

    run_header = header()
    seed = DebateJournal.start(run_header, home=tmp_path)
    seed.append({"type": "call", "run_id": "abc123", "seq": 0})
    seed.close()

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None

    class _FlockEnolck:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del fd, flags
            raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnolck())

    with pytest.raises(JournalLockUnsupported) as exc:
        DebateJournal.resume("abc123", run_header, home=tmp_path)
    assert isinstance(exc.value.__cause__, OSError)
    assert exc.value.__cause__.errno == errno.ENOLCK
    assert "ENOLCK" in str(exc.value)


def test_new_run_enolck_disables_only_this_journal_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ENOLCK on new-run fails open for that journal only; a later run_id can still attempt locks."""
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod

    real_fcntl = journal_mod._fcntl
    assert real_fcntl is not None
    calls = {"n": 0}

    class _FlockEnolckThenOk:
        LOCK_EX = real_fcntl.LOCK_EX
        LOCK_NB = real_fcntl.LOCK_NB

        def flock(self, fd: int, flags: int) -> None:
            del flags
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(errno.ENOLCK, "No locks available")
            # Subsequent run_ids retry normally (real flock on this fd).
            real_fcntl.flock(fd, real_fcntl.LOCK_EX | real_fcntl.LOCK_NB)

    monkeypatch.setattr(journal_mod, "_fcntl", _FlockEnolckThenOk())

    disabled = DebateJournal.start(header(run_id="enolck-a"), home=tmp_path)
    disabled.append({"type": "call", "run_id": "enolck-a", "seq": 0})
    assert disabled._writes_disabled is True
    assert disabled.path.read_bytes() == b""

    # Second append on same object must not re-open / re-lock.
    locks_before = calls["n"]
    disabled.append({"type": "call", "run_id": "enolck-a", "seq": 1})
    assert calls["n"] == locks_before

    other = DebateJournal.start(header(run_id="enolck-b"), home=tmp_path)
    other.append({"type": "call", "run_id": "enolck-b", "seq": 0})
    try:
        assert other._writes_disabled is False
        assert other._file is not None
        assert other.path.stat().st_size > 0
        assert calls["n"] == locks_before + 1
    finally:
        other.close()

    err = capsys.readouterr().err
    assert err.count("warning: ultracode journal disabled:") == 1


def test_resume_open_unlink_race_is_typed_resume_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import ResumeNotFound

    run_header = header()
    seed = DebateJournal.start(run_header, home=tmp_path)
    seed.append({"type": "call", "run_id": "abc123", "seq": 0})
    seed.close()

    path = tmp_path / "journals" / "abc123.jsonl"
    real_open = journal_mod.os.open
    calls = {"n": 0}

    def unlink_before_open(path_arg: int | str | bytes | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        # Resume path only: after exists check, unlink then delegate so open raises FileNotFoundError.
        if os.fspath(path_arg) == os.fspath(path):
            calls["n"] += 1
            path.unlink(missing_ok=True)
        return real_open(path_arg, flags, mode)

    monkeypatch.setattr(journal_mod.os, "open", unlink_before_open)

    with pytest.raises(ResumeNotFound) as exc:
        DebateJournal.resume("abc123", run_header, home=tmp_path)

    assert str(exc.value) == "abc123" or getattr(exc.value, "args", (None,))[0] == "abc123"
    assert isinstance(exc.value.__cause__, FileNotFoundError)
    assert calls["n"] == 1
    assert not path.exists()


def _seed_journal_bytes(tmp_path: Path, run_id: str, *segments: bytes) -> Path:
    journals = tmp_path / "journals"
    journals.mkdir(parents=True, exist_ok=True)
    path = journals / f"{run_id}.jsonl"
    path.write_bytes(b"".join(segments))
    os.chmod(journals, 0o700)
    os.chmod(path, 0o600)
    return path


def test_resume_repairs_torn_json_tail_before_append_and_later_record_visible(tmp_path: Path) -> None:
    """Invalid final segment without LF is truncated; subsequent append remains visible."""
    run_id = "tornjson1"
    run_header = header(run_id=run_id)
    header_line = (json.dumps(dict(run_header), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    call0 = (json.dumps({"type": "call", "run_id": run_id, "seq": 0}, sort_keys=True) + "\n").encode("utf-8")
    torn = b'{"type":"call","run_id":"tornjson1","seq":1'
    path = _seed_journal_bytes(tmp_path, run_id, header_line, call0, torn)
    good_prefix = header_line + call0

    resumed = DebateJournal.resume(run_id, run_header, home=tmp_path)
    try:
        assert len(resumed.replay_calls) == 1
        assert resumed.replay_calls[0]["seq"] == 0
        resumed.append({"type": "call", "run_id": run_id, "seq": 1})
    finally:
        resumed.close()

    data = path.read_bytes()
    assert data.startswith(good_prefix)
    assert torn not in data
    lines = data.decode("utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["type"] == "header"
    assert json.loads(lines[1])["seq"] == 0
    assert json.loads(lines[2])["seq"] == 1
    for line in lines:
        json.loads(line)


def test_resume_repairs_torn_utf8_tail_before_append(tmp_path: Path) -> None:
    """Partial UTF-8 final segment without LF is truncated before future writes."""
    run_id = "tornutf8"
    run_header = header(run_id=run_id)
    header_line = (json.dumps(dict(run_header), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    call0 = (json.dumps({"type": "call", "run_id": run_id, "seq": 0}, sort_keys=True) + "\n").encode("utf-8")
    # Incomplete 2-byte UTF-8 sequence (would be U+00E9 if completed with 0xA9).
    torn_utf8 = b"\xc3"
    path = _seed_journal_bytes(tmp_path, run_id, header_line, call0, torn_utf8)
    good_prefix = header_line + call0

    resumed = DebateJournal.resume(run_id, run_header, home=tmp_path)
    try:
        resumed.append({"type": "call", "run_id": run_id, "seq": 1})
    finally:
        resumed.close()

    data = path.read_bytes()
    assert data.startswith(good_prefix)
    assert not data[len(good_prefix) :].startswith(torn_utf8)
    lines = data.decode("utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["seq"] == 1


def test_resume_valid_json_without_final_lf_gets_delimiter_and_is_preserved(tmp_path: Path) -> None:
    """Valid final JSON object without LF is kept and exactly one LF is appended before writes."""
    run_id = "nofl1"
    run_header = header(run_id=run_id)
    header_obj = json.dumps(dict(run_header), ensure_ascii=False, sort_keys=True)
    call0_obj = json.dumps({"type": "call", "run_id": run_id, "seq": 0}, sort_keys=True)
    # Header ends with LF; final call record has no trailing LF.
    path = _seed_journal_bytes(
        tmp_path,
        run_id,
        (header_obj + "\n").encode("utf-8"),
        call0_obj.encode("utf-8"),
    )
    assert not path.read_bytes().endswith(b"\n")

    resumed = DebateJournal.resume(run_id, run_header, home=tmp_path)
    try:
        assert len(resumed.replay_calls) == 1
        assert resumed.replay_calls[0]["seq"] == 0
        # After resume repair, validated content must end with exactly one LF before appends.
        after_resume = path.read_bytes()
        assert after_resume.endswith(b"\n")
        assert after_resume == (header_obj + "\n" + call0_obj + "\n").encode("utf-8")
        resumed.append({"type": "call", "run_id": run_id, "seq": 1})
    finally:
        resumed.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[1])["seq"] == 0
    assert json.loads(lines[2])["seq"] == 1
    # No doubled blank record between preserved call and append.
    assert path.read_bytes().count(b"\n\n") == 0


def test_newline_terminated_and_midfile_corrupt_fail_without_mutation(tmp_path: Path) -> None:
    """Newline-terminated corrupt and mid-file corrupt records raise typed error; bytes unchanged."""
    from cluxion_effort_ultracode.core.journal import JournalCorrupt

    run_id = "corrupt1"
    run_header = header(run_id=run_id)
    header_line = (json.dumps(dict(run_header), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

    # Newline-terminated corrupt record (final line has LF).
    nl_corrupt = header_line + b"not-json-at-all\n"
    path_nl = _seed_journal_bytes(tmp_path, run_id, nl_corrupt)
    before_nl = path_nl.read_bytes()
    with pytest.raises(JournalCorrupt) as exc_nl:
        DebateJournal.resume(run_id, run_header, home=tmp_path)
    assert exc_nl.value.run_id == run_id
    assert path_nl.read_bytes() == before_nl

    # Mid-file corrupt: bad newline-terminated record, then a valid record after it.
    run_id_mid = "corrupt2"
    run_header_mid = header(run_id=run_id_mid)
    header_mid = (json.dumps(dict(run_header_mid), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    call_mid = (json.dumps({"type": "call", "run_id": run_id_mid, "seq": 0}, sort_keys=True) + "\n").encode("utf-8")
    mid_corrupt = header_mid + b"{bad-json}\n" + call_mid
    path_mid = _seed_journal_bytes(tmp_path, run_id_mid, mid_corrupt)
    before_mid = path_mid.read_bytes()
    with pytest.raises(JournalCorrupt) as exc_mid:
        DebateJournal.resume(run_id_mid, run_header_mid, home=tmp_path)
    assert exc_mid.value.run_id == run_id_mid
    assert path_mid.read_bytes() == before_mid


@pytest.mark.parametrize("tail", [b"[]", b"null"])
def test_complete_nonobject_final_no_lf_is_corrupt_not_torn(tmp_path: Path, tail: bytes) -> None:
    """Complete valid-JSON non-object final no-LF is corrupt, not a torn fragment; bytes exact."""
    from cluxion_effort_ultracode.core.journal import JournalCorrupt, read_records

    run_id = f"nonobj{tail.decode('ascii')}"
    run_header = header(run_id=run_id)
    header_line = (json.dumps(dict(run_header), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    call0 = (json.dumps({"type": "call", "run_id": run_id, "seq": 0}, sort_keys=True) + "\n").encode("utf-8")
    path = _seed_journal_bytes(tmp_path, run_id, header_line, call0, tail)
    before = path.read_bytes()
    assert not before.endswith(b"\n")
    assert before.endswith(tail)

    with pytest.raises(JournalCorrupt) as exc_resume:
        DebateJournal.resume(run_id, run_header, home=tmp_path)
    assert exc_resume.value.run_id == run_id
    assert path.read_bytes() == before

    with pytest.raises(JournalCorrupt) as exc_read:
        read_records(path)
    assert exc_read.value.run_id == run_id
    assert path.read_bytes() == before


# --- Cycle 108: GC preflight atomicity + new-run fail-open disable ---


def _old_header_bytes(run_id: str, *, days: int = 30) -> bytes:
    payload = dict(header(run_id))
    payload["created_at"] = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def test_gc_apply_good_before_corrupt_raises_and_unlinks_nothing(tmp_path: Path) -> None:
    """apply=True must not unlink an earlier good candidate when a later file is corrupt."""
    from cluxion_effort_ultracode.core.journal import JournalCorrupt

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    good = journals / "aaa_good.jsonl"
    corrupt = journals / "zzz_corrupt.jsonl"
    good.write_bytes(_old_header_bytes("aaa_good"))
    corrupt.write_bytes(b'{"type":"header","run_id":"zzz_corrupt"}\nNOT-JSON\n')
    before_good = good.read_bytes()
    before_corrupt = corrupt.read_bytes()

    with pytest.raises(JournalCorrupt) as exc:
        gc_journals(home=tmp_path, older_than_days=7, apply=True)

    assert exc.value.run_id == "zzz_corrupt"
    assert good.exists()
    assert corrupt.exists()
    assert good.read_bytes() == before_good
    assert corrupt.read_bytes() == before_corrupt


def test_gc_empty_claim_uses_mtime_age_only(tmp_path: Path) -> None:
    """size==0 O_EXCL leftovers: stale mtime is a candidate; fresh mtime is kept."""
    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    stale = journals / "empty_stale.jsonl"
    fresh = journals / "empty_fresh.jsonl"
    stale.write_bytes(b"")
    fresh.write_bytes(b"")
    now = datetime.now(UTC).timestamp()
    os.utime(stale, (now - 30 * 86400, now - 30 * 86400))
    os.utime(fresh, (now, now))

    applied = gc_journals(home=tmp_path, older_than_days=7, apply=True)
    ids = [c["run_id"] for c in applied["candidates"]]
    assert "empty_stale" in ids
    assert "empty_fresh" not in ids
    assert not stale.exists()
    assert fresh.exists()
    assert fresh.read_bytes() == b""


def test_gc_nonempty_corrupt_never_mtime_deleted(tmp_path: Path) -> None:
    """Nonempty corrupt journals raise JournalCorrupt and are never deleted by mtime."""
    from cluxion_effort_ultracode.core.journal import JournalCorrupt

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "badmtime.jsonl"
    path.write_bytes(b"not-a-jsonl-record\n")
    now = datetime.now(UTC).timestamp()
    os.utime(path, (now - 30 * 86400, now - 30 * 86400))
    before = path.read_bytes()

    with pytest.raises(JournalCorrupt) as exc:
        gc_journals(home=tmp_path, older_than_days=7, apply=True)

    assert exc.value.run_id == "badmtime"
    assert path.exists()
    assert path.read_bytes() == before


def test_gc_apply_inode_swap_skips_unlink_of_replaced_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After preflight, if path identity changes, unlink phase must not delete the path."""
    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    path = journals / "swap1.jsonl"
    path.write_bytes(_old_header_bytes("swap1"))
    before = path.read_bytes()
    target = os.fspath(path)
    path_stats = {"n": 0}
    real_stat = os.stat

    def reverify_sees_other_inode(path_arg, *args, **kwargs):
        st = real_stat(path_arg, *args, **kwargs)
        if os.fspath(path_arg) == target:
            path_stats["n"] += 1
            # First path.stat is preflight identity check; second is unlink reverify.
            if path_stats["n"] >= 2:
                class _OtherInode:
                    def __init__(self, base: os.stat_result) -> None:
                        self._base = base
                        self.st_ino = base.st_ino + 10_000_001
                        self.st_dev = base.st_dev

                    def __getattr__(self, name: str):
                        return getattr(self._base, name)

                return _OtherInode(st)
        return st

    monkeypatch.setattr(os, "stat", reverify_sees_other_inode)

    result = gc_journals(home=tmp_path, older_than_days=7, apply=True)

    assert path.exists()
    assert path.read_bytes() == before
    assert [c["run_id"] for c in result["candidates"]] == []
    assert path_stats["n"] >= 2


def test_gc_apply_failure_releases_retained_fds(tmp_path: Path) -> None:
    """On preflight corruption, retained candidate locks/FDs are released (file re-lockable)."""
    import fcntl

    from cluxion_effort_ultracode.core.journal import JournalCorrupt

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    good = journals / "aaa_hold.jsonl"
    corrupt = journals / "zzz_bad.jsonl"
    good.write_bytes(_old_header_bytes("aaa_hold"))
    corrupt.write_bytes(b"{bad}\n")

    with pytest.raises(JournalCorrupt):
        gc_journals(home=tmp_path, older_than_days=7, apply=True)

    # After abort, good file must not remain exclusively locked by GC.
    fd = os.open(good, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    assert good.exists()


def test_gc_apply_open_ioerror_aborts_before_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard open error is not a benign race and must leave earlier candidates intact."""
    from cluxion_effort_ultracode.core import journal_lifecycle

    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    good = journals / "aaa_good.jsonl"
    broken = journals / "zzz_ioerr.jsonl"
    good.write_bytes(_old_header_bytes("aaa_good"))
    broken.write_bytes(_old_header_bytes("zzz_ioerr"))
    real_open = os.open

    def fail_target(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(broken):
            raise OSError(errno.EIO, "simulated open I/O failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(journal_lifecycle.os, "open", fail_target)
    with pytest.raises(OSError, match="simulated open I/O failure"):
        gc_journals(home=tmp_path, older_than_days=7, apply=True)

    assert good.exists()
    assert broken.exists()


def test_gc_apply_reverifies_all_candidates_before_any_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later identity-check I/O error must not follow deletion of an earlier candidate."""
    journals = tmp_path / "journals"
    journals.mkdir(parents=True)
    first = journals / "aaa_first.jsonl"
    later = journals / "zzz_later.jsonl"
    first.write_bytes(_old_header_bytes("aaa_first"))
    later.write_bytes(_old_header_bytes("zzz_later"))
    real_stat = os.stat
    later_stats = {"n": 0}

    def fail_later_reverify(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(later):
            later_stats["n"] += 1
            if later_stats["n"] == 2:
                raise OSError(errno.EIO, "simulated reverify I/O failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fail_later_reverify)
    with pytest.raises(OSError, match="simulated reverify I/O failure"):
        gc_journals(home=tmp_path, older_than_days=7, apply=True)
    monkeypatch.setattr(os, "stat", real_stat)

    assert first.exists()
    assert later.exists()


def _assert_no_second_open_and_new_object_ok(
    tmp_path: Path,
    *,
    run_id: str,
    other_run_id: str,
    journal: DebateJournal,
    open_calls: dict[str, int],
) -> None:
    opens_after_first = open_calls["n"]
    journal.append({"type": "call", "run_id": run_id, "seq": 1})
    assert open_calls["n"] == opens_after_first
    assert journal._writes_disabled is True
    assert journal._file is None
    assert journal._locked is False

    other = DebateJournal.start(header(run_id=other_run_id), home=tmp_path)
    other.append({"type": "call", "run_id": other_run_id, "seq": 0})
    try:
        assert other._writes_disabled is False
        assert other._file is not None
        assert other.path.stat().st_size > 0
        lines = other.path.read_text(encoding="utf-8").splitlines()
        # header + call seq=0 — no seq gap on the successful new object
        assert any(json.loads(line).get("seq") == 0 for line in lines if line.strip())
        assert not any(json.loads(line).get("seq") == 1 for line in lines if line.strip())
    finally:
        other.close()


def test_new_run_journal_busy_disables_writes_no_retry_no_seq_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import JournalBusy

    real_open = journal_mod.os.open
    real_lock = journal_mod._lock_exclusive_nonblocking
    open_calls = {"n": 0}
    fail_run = "busyfail1"

    def counting_open(path_arg, flags, mode=0o777):
        open_calls["n"] += 1
        return real_open(path_arg, flags, mode)

    def busy_lock_only_target(fd: int, run_id: str) -> None:
        if run_id == fail_run:
            raise JournalBusy(run_id)
        return real_lock(fd, run_id)

    monkeypatch.setattr(journal_mod.os, "open", counting_open)
    monkeypatch.setattr(journal_mod, "_lock_exclusive_nonblocking", busy_lock_only_target)

    journal = DebateJournal.start(header(run_id=fail_run), home=tmp_path)
    journal.append({"type": "call", "run_id": fail_run, "seq": 0})
    assert journal._writes_disabled is True
    assert open_calls["n"] >= 1

    _assert_no_second_open_and_new_object_ok(
        tmp_path,
        run_id=fail_run,
        other_run_id="busyok2",
        journal=journal,
        open_calls=open_calls,
    )
    err = capsys.readouterr().err
    assert "busy" in err.lower()
    assert err.count("warning: ultracode journal disabled:") == 1


def test_new_run_inode_verify_failure_disables_writes_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import cluxion_effort_ultracode.core.journal as journal_mod
    from cluxion_effort_ultracode.core.journal import ResumeNotFound

    real_open = journal_mod.os.open
    real_verify = journal_mod._verify_same_inode
    open_calls = {"n": 0}
    fail_run = "inodefail1"

    def counting_open(path_arg, flags, mode=0o777):
        open_calls["n"] += 1
        return real_open(path_arg, flags, mode)

    def boom_inode_only_target(fd: int, path: Path, run_id: str) -> None:
        if run_id == fail_run:
            raise ResumeNotFound(run_id)
        return real_verify(fd, path, run_id)

    monkeypatch.setattr(journal_mod.os, "open", counting_open)
    monkeypatch.setattr(journal_mod, "_verify_same_inode", boom_inode_only_target)

    journal = DebateJournal.start(header(run_id=fail_run), home=tmp_path)
    journal.append({"type": "call", "run_id": fail_run, "seq": 0})
    assert journal._writes_disabled is True

    _assert_no_second_open_and_new_object_ok(
        tmp_path,
        run_id=fail_run,
        other_run_id="inodeok2",
        journal=journal,
        open_calls=open_calls,
    )
    assert "warning: ultracode journal disabled:" in capsys.readouterr().err


def test_new_run_eexist_disables_writes_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import cluxion_effort_ultracode.core.journal as journal_mod

    real_open = journal_mod.os.open
    open_calls = {"n": 0}

    def counting_open(path_arg, flags, mode=0o777):
        open_calls["n"] += 1
        return real_open(path_arg, flags, mode)

    monkeypatch.setattr(journal_mod.os, "open", counting_open)

    run_id = "eexist1"
    journal = DebateJournal.start(header(run_id=run_id), home=tmp_path)
    journal.path.parent.mkdir(parents=True, exist_ok=True)
    journal.path.write_bytes(b"")

    journal.append({"type": "call", "run_id": run_id, "seq": 0})
    assert journal._writes_disabled is True
    assert journal.path.read_bytes() == b""

    _assert_no_second_open_and_new_object_ok(
        tmp_path,
        run_id=run_id,
        other_run_id="eexistok2",
        journal=journal,
        open_calls=open_calls,
    )
    assert "collision" in capsys.readouterr().err.lower()


def test_new_run_terminal_open_oserror_disables_writes_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod

    real_open = journal_mod.os.open
    open_calls = {"n": 0}

    def counting_open(path_arg, flags, mode=0o777):
        open_calls["n"] += 1
        # Fail exclusive create for the target journal only.
        if str(path_arg).endswith("openfail1.jsonl"):
            raise OSError(errno.EIO, "simulated open failure")
        return real_open(path_arg, flags, mode)

    monkeypatch.setattr(journal_mod.os, "open", counting_open)

    run_id = "openfail1"
    journal = DebateJournal.start(header(run_id=run_id), home=tmp_path)
    journal.append({"type": "call", "run_id": run_id, "seq": 0})
    assert journal._writes_disabled is True

    _assert_no_second_open_and_new_object_ok(
        tmp_path,
        run_id=run_id,
        other_run_id="openok2",
        journal=journal,
        open_calls=open_calls,
    )
    assert "warning: ultracode journal disabled:" in capsys.readouterr().err
