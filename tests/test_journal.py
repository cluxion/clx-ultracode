"""Regression tests for debate journals and resume replay."""

from __future__ import annotations

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
    os.chmod(journal.path.parent, 0o755)
    os.chmod(journal.path, 0o644)

    resumed = DebateJournal.resume("abc123", run_header, home=tmp_path)
    resumed.append({"type": "result", "run_id": "abc123"})

    assert mode(journal.path.parent) == 0o700
    assert mode(journal.path) == 0o600


def test_crash_then_resume_replays_prefix_and_continues(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)
    first_llm = ScriptedLlm([position("Adopt"), position("Delay"), update("Adopt"), update("Adopt", conceded=True)])
    first = ConsensusEngine(JournaledLlm(first_llm, journal), agents_count=2, max_rounds=1).decide("Q?", context="ctx")
    assert first.status == "unanimous"

    lines = journal.path.read_text(encoding="utf-8").splitlines()
    journal.path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")

    resumed_journal = DebateJournal.resume("abc123", run_header, home=tmp_path)
    suffix_llm = ScriptedLlm([update("Adopt"), update("Adopt", conceded=True)])
    resumed = ConsensusEngine(JournaledLlm(suffix_llm, resumed_journal), agents_count=2, max_rounds=1).decide(
        "Q?",
        context="ctx",
    )

    assert resumed.status == "unanimous"
    assert len(suffix_llm.calls) == 2
    assert resumed.tokens_replayed > 0


def test_resume_mismatch_reports_changed_question(tmp_path: Path) -> None:
    run_header = header()
    journal = DebateJournal.start(run_header, home=tmp_path)
    journal.append({"type": "call", "run_id": "abc123"})
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

    replayed_journal = DebateJournal.resume("abc123", run_header, home=tmp_path)
    replayed = ConsensusEngine(JournaledLlm(ExplodingLlm(), replayed_journal), agents_count=2, max_rounds=1).decide(
        "Q?",
        context="ctx",
    )

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
