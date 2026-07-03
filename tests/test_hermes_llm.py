"""Tests for the Hermes subprocess LLM adapter."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError, HermesSubprocessLlm
from cluxion_effort_ultracode.core import ConsensusEngine


class FakeProcess:
    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = "", pid: int = 12345) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = pid
        self.timeout: float | None = None

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.timeout = timeout
        return self.stdout, self.stderr


class TimeoutThenExitProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__("", returncode=-15)
        self.calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.calls += 1
        if self.calls == 1:
            raise subprocess.TimeoutExpired(["hermes"], timeout)
        return "", ""


def test_structured_complete_parses_json_from_subprocess_stdout() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=12)

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess('```json\n{"stance":"Adopt"}\n```'),
    ) as popen:
        result = llm.complete("Prompt", schema={"type": "object"})

    assert result == {"stance": "Adopt"}
    command = popen.call_args.args[0]
    assert command[0] == "hermes"
    assert command[1] == "-z"
    assert "Return ONLY one JSON object" in command[2]
    assert popen.return_value.timeout == 12
    assert popen.call_args.kwargs["start_new_session"] is True


def test_default_model_is_passed_to_hermes_oneshot() -> None:
    llm = HermesSubprocessLlm(binary="/opt/hermes", timeout_seconds=5, model="grok-test")

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess("raw text"),
    ) as popen:
        assert llm.complete("Prompt") == "raw text"

    assert popen.call_args.args[0] == ["/opt/hermes", "-m", "grok-test", "-z", "Prompt"]


def test_per_call_model_override_reaches_hermes_m_flag() -> None:
    llm = HermesSubprocessLlm(binary="/opt/hermes", timeout_seconds=5, model="default-model")

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess("raw text"),
    ) as popen:
        assert llm.complete("Prompt", model="seat-model") == "raw text"

    assert popen.call_args.args[0] == ["/opt/hermes", "-m", "seat-model", "-z", "Prompt"]


def test_hermes_usage_is_feature_detected_from_stderr_json() -> None:
    llm = HermesSubprocessLlm()

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess("raw text", stderr='{"usage":{"input_tokens":7,"output_tokens":5}}'),
    ):
        assert llm.complete("Prompt") == "raw text"

    assert llm.last_usage == {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12, "estimated": False}


def test_structured_complete_retries_once_after_malformed_json() -> None:
    llm = HermesSubprocessLlm()

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        side_effect=[FakeProcess("not json"), FakeProcess('{"stance":"Adopt"}')],
    ) as popen:
        result = llm.complete("Prompt", schema={"type": "object"})

    assert result == {"stance": "Adopt"}
    assert popen.call_count == 2
    retry_prompt = popen.call_args.args[0][-1]
    assert "previous response was not parseable" in retry_prompt


def test_structured_complete_logs_first_parse_failure(capsys) -> None:
    llm = HermesSubprocessLlm()

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        side_effect=[FakeProcess("not json"), FakeProcess('{"stance":"Adopt"}')],
    ):
        assert llm.complete("Prompt", schema={"type": "object"}) == {"stance": "Adopt"}

    assert "Hermes structured JSON parse failed" in capsys.readouterr().err


def test_transient_subprocess_error_retries_once() -> None:
    llm = HermesSubprocessLlm()

    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            side_effect=[TimeoutThenExitProcess(), FakeProcess("raw text")],
        ) as popen,
        patch("cluxion_effort_ultracode.adapters.subprocess_common.os.killpg", lambda pid, sig: None),
    ):
        assert llm.complete("Prompt") == "raw text"

    assert popen.call_count == 2


def test_structured_complete_treats_empty_code_fence_as_retryable_parse_failure() -> None:
    llm = HermesSubprocessLlm()

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        side_effect=[FakeProcess("```\n```"), FakeProcess('{"stance":"Adopt"}')],
    ) as popen:
        result = llm.complete("Prompt", schema={"type": "object"})

    assert result == {"stance": "Adopt"}
    assert popen.call_count == 2


def test_missing_hermes_binary_raises_honest_error() -> None:
    llm = HermesSubprocessLlm(binary="missing-hermes")

    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(HermesExecutableNotFoundError, match="missing-hermes"),
    ):
        llm.complete("Prompt")


def test_sigterm_reaps_hermes_process_group(tmp_path: Path) -> None:
    fake_hermes = tmp_path / "hermes"
    pid_file = tmp_path / "child.pid"
    fake_hermes.write_text(
        "#!/usr/bin/env python3\n"
        "import os, time\n"
        f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    fake_hermes.chmod(0o755)
    script = textwrap.dedent(
        """
        import sys
        from cluxion_effort_ultracode.adapters.hermes_llm import HermesSubprocessLlm

        HermesSubprocessLlm(binary=sys.argv[1], timeout_seconds=120).complete("Prompt")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(fake_hermes)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        deadline = time.monotonic() + 10
        child_pid = None
        while time.monotonic() < deadline and child_pid is None:
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
                child_pid = int(pid_file.read_text(encoding="utf-8").strip())
            else:
                time.sleep(0.1)
        assert child_pid is not None, "fake hermes child never started"
        assert child_pid != parent.pid

        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=10)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(child_pid, signal.SIGKILL)
        raise AssertionError(f"hermes child {child_pid} survived parent SIGTERM")
    finally:
        if parent.poll() is None:
            parent.kill()


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
