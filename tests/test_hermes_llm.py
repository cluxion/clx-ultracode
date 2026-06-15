"""Tests for the Hermes subprocess LLM adapter."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError, HermesSubprocessLlm
from cluxion_effort_ultracode.core import ConsensusEngine


def completed(stdout: str, *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["hermes"], returncode, stdout=stdout, stderr=stderr)


def test_structured_complete_parses_json_from_subprocess_stdout() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=12)

    with patch(
        "cluxion_effort_ultracode.adapters.hermes_llm.subprocess.run",
        return_value=completed('```json\n{"stance":"Adopt"}\n```'),
    ) as run:
        result = llm.complete("Prompt", schema={"type": "object"})

    assert result == {"stance": "Adopt"}
    command = run.call_args.args[0]
    assert command[0] == "hermes"
    assert command[1] == "-z"
    assert "Return ONLY one JSON object" in command[2]
    assert run.call_args.kwargs["timeout"] == 12


def test_model_override_is_passed_to_hermes_oneshot() -> None:
    llm = HermesSubprocessLlm(binary="/opt/hermes", timeout_seconds=5, model="grok-test")

    with patch(
        "cluxion_effort_ultracode.adapters.hermes_llm.subprocess.run",
        return_value=completed("raw text"),
    ) as run:
        assert llm.complete("Prompt") == "raw text"

    assert run.call_args.args[0] == ["/opt/hermes", "--model", "grok-test", "-z", "Prompt"]


def test_structured_complete_retries_once_after_malformed_json() -> None:
    llm = HermesSubprocessLlm()

    with patch(
        "cluxion_effort_ultracode.adapters.hermes_llm.subprocess.run",
        side_effect=[completed("not json"), completed('{"stance":"Adopt"}')],
    ) as run:
        result = llm.complete("Prompt", schema={"type": "object"})

    assert result == {"stance": "Adopt"}
    assert run.call_count == 2
    retry_prompt = run.call_args.args[0][-1]
    assert "previous response was not parseable" in retry_prompt


def test_missing_hermes_binary_raises_honest_error() -> None:
    llm = HermesSubprocessLlm(binary="missing-hermes")

    with (
        patch(
            "cluxion_effort_ultracode.adapters.hermes_llm.subprocess.run",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(HermesExecutableNotFoundError, match="missing-hermes"),
    ):
        llm.complete("Prompt")


@pytest.mark.skipif(
    os.getenv("CLUXION_EFFORT_ULTRACODE_LIVE") != "1",
    reason="set CLUXION_EFFORT_ULTRACODE_LIVE=1 to run real hermes -z consensus",
)
def test_live_tiny_consensus_via_hermes_oneshot() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=120)

    result = ConsensusEngine(llm, agents_count=2, max_rounds=1).decide(
        "Use the exact stance YES. Is YES the correct stance for this smoke test?",
        context="Keep evidence short. The intended answer is YES.",
    )

    assert result.agents_count == 2
    assert result.rounds <= 1
