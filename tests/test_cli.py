"""Tests for the cluxion-ultracode CLI."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError
from cluxion_effort_ultracode.cli import main


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
    assert payload["error"] == "ValueError"
    assert "models entries" in payload["message"]


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


@pytest.mark.skipif(
    os.getenv("CLUXION_EFFORT_ULTRACODE_LIVE") != "1",
    reason="set CLUXION_EFFORT_ULTRACODE_LIVE=1 to run real hermes -z consensus via CLI",
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
