"""Tests for the cluxion-ultracode CLI."""

from __future__ import annotations

import json
import os
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from cluxion_effort_ultracode.adapters.codex_llm import CodexExecutableNotFoundError
from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError
from cluxion_effort_ultracode.cli import _prepare_consensus, main
from cluxion_effort_ultracode.core.journal import journals_dir


def test_consensus_mock_unanimous_adapter(capsys):
    exit_code = main(
        [
            "consensus",
            "--question",
            "Adopt?",
            "--adapter",
            "mock-unanimous",
            "--agents",
            "2",
            "--rounds",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["run_id"]
    assert payload["journal_path"].endswith(f"{payload['run_id']}.jsonl")


def test_consensus_hermes_adapter_uses_default_llm_factory():
    class _StubLlm:
        def __init__(self) -> None:
            self.outputs = [
                {"stance": "Yes", "rationale": "r1", "evidence": ["e1"], "confidence": 0.9},
                {"stance": "No", "rationale": "r2", "evidence": ["e2"], "confidence": 0.8},
                {
                    "stance": "Yes",
                    "rationale": "r1",
                    "evidence": ["e1"],
                    "confidence": 0.95,
                    "conceded": [{"point": "No", "reason": "Yes is stronger"}],
                    "maintained": [],
                },
                {
                    "stance": "Yes",
                    "rationale": "r2",
                    "evidence": ["e2"],
                    "confidence": 0.95,
                    "conceded": [{"point": "No", "reason": "Yes is stronger"}],
                    "maintained": [],
                },
            ]
            self.index = 0

        def complete(self, prompt: str, *, schema=None):
            output = self.outputs[self.index]
            self.index += 1
            return output

    with patch("cluxion_effort_ultracode.llm_factory.default_llm", return_value=_StubLlm()):
        exit_code = main(
            [
                "consensus",
                "--question",
                "Adopt?",
                "--adapter",
                "hermes",
                "--agents",
                "2",
                "--rounds",
                "1",
            ]
        )
    assert exit_code == 0


def test_consensus_cli_hermes_missing_returns_json_error(capsys):
    with patch(
        "cluxion_effort_ultracode.llm_factory.default_llm",
        side_effect=HermesExecutableNotFoundError("Hermes executable not found: 'missing-hermes'"),
    ):
        exit_code = main(["consensus", "--question", "Adopt?", "--adapter", "hermes"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "hermes_not_found"
    assert "PATH" in payload["hint"]


def test_consensus_cli_codex_adapter_uses_default_llm_factory():
    class _StubLlm:
        def __init__(self) -> None:
            self.outputs = [
                {"stance": "Yes", "rationale": "r1", "evidence": ["e1"], "confidence": 0.9},
                {"stance": "YES.", "rationale": "r2", "evidence": ["e2"], "confidence": 0.8},
            ]
            self.index = 0

        def complete(self, prompt: str, *, schema=None, model=None):
            del prompt, schema, model
            output = self.outputs[self.index]
            self.index += 1
            return output

    with patch("cluxion_effort_ultracode.llm_factory.default_llm", return_value=_StubLlm()) as default_llm:
        exit_code = main(
            [
                "consensus",
                "--question",
                "Adopt?",
                "--adapter",
                "codex",
                "--agents",
                "2",
                "--rounds",
                "0",
            ]
        )
    assert exit_code == 0
    default_llm.assert_called_once_with("codex", timeout_seconds=180.0)


def test_consensus_cli_codex_missing_returns_json_error(capsys):
    with patch(
        "cluxion_effort_ultracode.llm_factory.default_llm",
        side_effect=CodexExecutableNotFoundError("Codex executable not found: 'missing-codex'"),
    ):
        exit_code = main(["consensus", "--question", "Adopt?", "--adapter", "codex"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "codex_not_found"
    assert "PATH" in payload["hint"]
    assert "CLUXION_EFFORT_ULTRACODE_CODEX_BINARY" in payload["hint"]


def test_consensus_cli_adapter_missing_leaves_no_journal_and_resume_is_clear(capsys):
    with patch(
        "cluxion_effort_ultracode.llm_factory.default_llm",
        side_effect=CodexExecutableNotFoundError("Codex executable not found: 'missing-codex'"),
    ):
        assert main(["consensus", "--question", "Adopt?", "--adapter", "codex"]) == 1

    first = json.loads(capsys.readouterr().out)
    assert first["error"] == "codex_not_found"
    assert not (journals_dir() / f"{first['run_id']}.jsonl").exists()

    assert main(["consensus", "--resume", first["run_id"]]) == 1
    resumed = json.loads(capsys.readouterr().out)
    assert resumed == {"ok": False, "error": "journal_not_found", "run_id": first["run_id"]}


def test_consensus_cli_exposes_timeout_and_budget_flags(capsys):
    exit_code = main(
        [
            "consensus",
            "--question",
            "Adopt?",
            "--adapter",
            "mock-unanimous",
            "--agents",
            "2",
            "--rounds",
            "1",
            "--agent-timeout",
            "3",
            "--debate-budget",
            "20",
            "--budget-tokens",
            "10000",
            "--models",
            "cheap,strong",
        ]
    )
    err = capsys.readouterr().err
    assert exit_code == 0
    assert "round 0 independent start" in err
    assert "round 1 debate start" not in err


def test_consensus_cli_rejects_empty_model_entries(capsys):
    exit_code = main(
        [
            "consensus",
            "--question",
            "Adopt?",
            "--adapter",
            "mock-unanimous",
            "--models",
            "cheap,,strong",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_models"
    assert "models entries" in payload["message"]


@pytest.mark.parametrize("models", ["", " "])
def test_consensus_cli_rejects_explicit_empty_models(capsys, models: str):
    exit_code = main(
        [
            "consensus",
            "--question",
            "Adopt?",
            "--adapter",
            "mock-unanimous",
            "--models",
            models,
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_models"
    assert "models entries" in payload["message"]


def test_consensus_cli_rejects_huge_agents_before_journal(capsys):
    huge = "9" * 500
    started = time.perf_counter()

    exit_code = main(["consensus", "--question", "Adopt?", "--adapter", "mock-unanimous", "--agents", huge])

    elapsed = time.perf_counter() - started
    payload = json.loads(capsys.readouterr().out)
    assert elapsed < 1
    assert exit_code == 1
    assert payload == {"ok": False, "error": "invalid_agents", "message": "agents_count must be <= 8"}
    assert not journals_dir().exists()


@pytest.mark.parametrize("question", ["", " "])
def test_consensus_cli_rejects_empty_question(capsys, question: str):
    exit_code = main(["consensus", "--question", question, "--adapter", "mock-unanimous"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert "question" in payload["message"]


def test_consensus_cli_rejects_invalid_utf8_question_file_before_journal(capsys, tmp_path: Path):
    question_file = tmp_path / "bad.txt"
    question_file.write_bytes(b"\xff\xfe invalid")

    exit_code = main(
        ["consensus", "--question-file", str(question_file), "--adapter", "mock-unanimous", "--agents", "2"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert "run_id" not in payload
    assert "journal_path" not in payload
    assert not journals_dir().exists()


def test_consensus_cli_rejects_surrogate_question_before_journal(capsys):
    exit_code = main(
        ["consensus", "--question", "Adopt?\udcff", "--adapter", "mock-unanimous", "--agents", "2"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert "run_id" not in payload
    assert "journal_path" not in payload
    assert not journals_dir().exists()


def test_consensus_cli_rejects_surrogate_context_before_journal(capsys):
    exit_code = main(
        [
            "consensus",
            "--question",
            "Adopt?",
            "--context",
            "ctx\udcff",
            "--adapter",
            "mock-unanimous",
            "--agents",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert "run_id" not in payload
    assert "journal_path" not in payload
    assert not journals_dir().exists()


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        (["--rounds", "99"], "invalid_rounds"),
        (["--agent-timeout", "0"], "invalid_timeout"),
        (["--agent-timeout", "-1"], "invalid_timeout"),
        (["--agent-timeout", "nan"], "invalid_timeout"),
        (["--agent-timeout", "inf"], "invalid_timeout"),
        (["--agent-timeout=-inf"], "invalid_timeout"),
        (["--debate-budget", "0"], "invalid_budget"),
        (["--debate-budget", "-1"], "invalid_budget"),
        (["--debate-budget", "nan"], "invalid_budget"),
        (["--debate-budget", "inf"], "invalid_budget"),
        (["--debate-budget=-inf"], "invalid_budget"),
        (["--budget-tokens", "0"], "invalid_budget"),
    ],
)
def test_consensus_cli_validation_errors_use_semantic_codes(capsys, argv: list[str], code: str):
    exit_code = main(["consensus", "--question", "Adopt?", "--adapter", "mock-unanimous", *argv])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == code
    assert "run_id" not in payload
    assert not journals_dir().exists()


def test_prepare_consensus_rejects_huge_int_timeout_without_overflow() -> None:
    namespace = type(
        "NS",
        (),
        {
            "resume": None,
            "question": "Adopt?",
            "question_file": None,
            "context": "",
            "rounds": 0,
            "agents": 2,
            "agent_timeout": 10**400,
            "debate_budget": 30.0,
            "budget_tokens": None,
            "models": None,
            "adapter": "mock-unanimous",
        },
    )()

    with pytest.raises(ValueError, match="agent_timeout") as caught:
        _prepare_consensus(namespace)
    assert not isinstance(caught.value, OverflowError)
    assert not journals_dir().exists()


def test_consensus_cli_reads_question_from_stdin(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("Adopt from stdin?"))

    exit_code = main(["consensus", "--question", "-", "--adapter", "mock-unanimous", "--agents", "2", "--rounds", "0"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "unanimous"


def test_consensus_cli_reads_large_question_file(capsys, tmp_path: Path):
    question_file = tmp_path / "question.txt"
    question_file.write_text("Adopt from file?\n" + ("x" * 1_000_000), encoding="utf-8")

    exit_code = main(
        [
            "consensus",
            "--question-file",
            str(question_file),
            "--adapter",
            "mock-unanimous",
            "--agents",
            "2",
            "--rounds",
            "0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "unanimous"


def test_consensus_cli_resume_mismatch_is_structured(capsys):
    assert (
        main(["consensus", "--question", "Adopt?", "--adapter", "mock-unanimous", "--agents", "2", "--rounds", "0"])
        == 0
    )
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    exit_code = main(
        [
            "consensus",
            "--resume",
            run_id,
            "--question",
            "Different?",
            "--adapter",
            "mock-unanimous",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["error"] == "resume_mismatch"
    assert "question" in payload["fields"]


def test_consensus_cli_resume_mismatch_on_adapter_change(capsys):
    assert (
        main(["consensus", "--question", "Adopt?", "--adapter", "mock-unanimous", "--agents", "2", "--rounds", "0"])
        == 0
    )
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    exit_code = main(["consensus", "--resume", run_id, "--adapter", "mock-no-consensus"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["error"] == "resume_mismatch"
    assert payload["fields"]["adapter"] == {"journal": "mock-unanimous", "current": "mock-no-consensus"}


def test_consensus_cli_resume_completed_run_replays_without_question(capsys):
    assert (
        main(["consensus", "--question", "Adopt?", "--adapter", "mock-unanimous", "--agents", "2", "--rounds", "0"])
        == 0
    )
    first = json.loads(capsys.readouterr().out)

    assert main(["consensus", "--resume", first["run_id"]]) == 0
    replayed = json.loads(capsys.readouterr().out)

    assert replayed["status"] == first["status"]
    assert replayed["decision"] == first["decision"]
    assert replayed["tokens_spent"] == 0
    assert replayed["tokens_replayed"] == first["tokens_spent"]


def test_consensus_cli_resume_returns_journal_busy_when_locked(capsys):
    import multiprocessing as mp

    from cluxion_effort_ultracode.core.journal import journals_dir
    from mp_helpers import hold_journal_until_release

    assert (
        main(["consensus", "--question", "Adopt?", "--adapter", "mock-unanimous", "--agents", "2", "--rounds", "0"])
        == 0
    )
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    home = Path(os.environ["CLUXION_EFFORT_ULTRACODE_HOME"])

    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Queue()

    proc = ctx.Process(target=hold_journal_until_release, args=(str(home), run_id, ready, release))
    proc.start()
    assert ready.get(timeout=10) == "ready"

    exit_code = main(["consensus", "--resume", run_id])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload == {"ok": False, "error": "journal_busy", "run_id": run_id}

    release.put("done")
    proc.join(timeout=10)
    assert proc.exitcode == 0
    assert (journals_dir() / f"{run_id}.jsonl").exists()


def test_journals_list_and_show(capsys):
    assert (
        main(["consensus", "--question", "Adopt?", "--adapter", "mock-unanimous", "--agents", "2", "--rounds", "0"])
        == 0
    )
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    assert main(["journals", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["journals"][0]["run_id"] == run_id
    assert listed["journals"][0]["calls_recorded"] == 2

    assert main(["journals", "show", run_id]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["records"][0]["type"] == "header"
    assert shown["records"][1]["type"] == "call"


def test_journals_list_warns_when_total_exceeds_threshold(capsys, monkeypatch):
    directory = journals_dir()
    directory.mkdir(parents=True)
    (directory / "abc123.jsonl").write_text(json.dumps({"type": "header", "run_id": "abc123"}) + "\n", encoding="utf-8")
    monkeypatch.setattr("cluxion_effort_ultracode.core.journal_lifecycle.WARN_SIZE_BYTES", 10)

    assert main(["journals", "list"]) == 0

    captured = capsys.readouterr()
    assert "warning: journal directory exceeds 10 bytes" in captured.err


def test_doctor_json_output_is_stdout_pure(capsys):
    main(["doctor", "--json"])
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert captured.err == ""


def test_consensus_help_documents_cost_formula(capsys):
    with pytest.raises(SystemExit):
        main(["consensus", "--help"])
    out = capsys.readouterr().out
    assert "agents * (rounds + 1)" in out
    assert "--agent-timeout" in out
    assert "--debate-budget" in out
    assert "--budget-tokens" in out
    assert "--models" in out
    assert "--question-file" in out
    assert "codex" in out


def test_journals_gc_huge_days_returns_structured_error(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    directory = journals_dir()
    directory.mkdir(parents=True)
    path = directory / "stay.jsonl"
    path.write_text(
        json.dumps({"type": "header", "run_id": "stay", "created_at": "2020-01-01T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")

    assert main(["journals", "gc", "--older-than-days", str(10**20), "--apply"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "older_than_days" in payload["message"]
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    assert path.read_text(encoding="utf-8") == before


def test_journals_gc_apply_without_lock_support_returns_typed_error(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HOME", str(tmp_path))
    directory = journals_dir()
    directory.mkdir(parents=True)
    path = directory / "stay.jsonl"
    path.write_text(
        json.dumps({"type": "header", "run_id": "stay", "created_at": "2020-01-01T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    monkeypatch.setattr("cluxion_effort_ultracode.core.journal_lifecycle.locks_supported", lambda: False)

    assert main(["journals", "gc", "--older-than-days", "7", "--apply"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"] == "journal_lock_unsupported"
    assert path.read_bytes() == before


def test_journals_gc_apply_runtime_flock_enotsup_is_structured(capsys, tmp_path, monkeypatch):
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod

    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HOME", str(tmp_path))
    directory = journals_dir()
    directory.mkdir(parents=True)
    path = directory / "stay.jsonl"
    path.write_text(
        json.dumps({"type": "header", "run_id": "stay", "created_at": "2020-01-01T00:00:00+00:00"}) + "\n",
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

    assert main(["journals", "gc", "--older-than-days", "7", "--apply"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"] == "journal_lock_unsupported"
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    assert path.read_bytes() == before


def test_journals_gc_apply_enolck_is_structured(capsys, tmp_path, monkeypatch):
    import errno

    import cluxion_effort_ultracode.core.journal as journal_mod

    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HOME", str(tmp_path))
    directory = journals_dir()
    directory.mkdir(parents=True)
    path = directory / "stay.jsonl"
    path.write_text(
        json.dumps({"type": "header", "run_id": "stay", "created_at": "2020-01-01T00:00:00+00:00"}) + "\n",
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

    assert main(["journals", "gc", "--older-than-days", "7", "--apply"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"] == "journal_lock_unsupported"
    assert "ENOLCK" in payload["message"]
    assert "retryable" in payload["message"].lower() or "temporarily unavailable" in payload["message"].lower()
    assert "Traceback" not in captured.out
    assert path.read_bytes() == before


@pytest.mark.skipif(
    os.getenv("CLUXION_EFFORT_ULTRACODE_LIVE") != "1",
    reason="set CLUXION_EFFORT_ULTRACODE_LIVE=1 to run real hermes ultracode-llm consensus via CLI",
)
def test_consensus_hermes_adapter_live_smoke():
    exit_code = main(
        [
            "consensus",
            "--question",
            "Use stance YES. Is YES correct?",
            "--adapter",
            "hermes",
            "--agents",
            "2",
            "--rounds",
            "0",
            "--context",
            "Keep evidence short.",
        ]
    )
    assert exit_code == 0
