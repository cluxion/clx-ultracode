"""Tests for the Codex subprocess LLM adapter."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cluxion_effort_ultracode.adapters.codex_llm import CodexExecutableNotFoundError, CodexSubprocessLlm


def _fake_codex(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 10**400])
def test_codex_constructor_rejects_non_finite_or_non_positive_timeout(timeout: float | int) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        CodexSubprocessLlm(timeout_seconds=timeout)


def test_codex_constructor_accepts_positive_finite_timeout() -> None:
    llm = CodexSubprocessLlm(timeout_seconds=12.5)
    assert llm.timeout_seconds == 12.5
    assert CodexSubprocessLlm(timeout_seconds=12).timeout_seconds == 12.0


def test_structured_complete_reads_output_last_message_not_stdout_noise(tmp_path: Path) -> None:
    calls_file = tmp_path / "calls.jsonl"
    fake = _fake_codex(
        tmp_path / "codex",
        textwrap.dedent(
            f"""
            import json, pathlib, sys
            args = sys.argv[1:]
            prompt = sys.stdin.read()
            pathlib.Path({str(calls_file)!r}).write_text(json.dumps({{"args": args, "prompt": prompt}}), encoding="utf-8")
            out = pathlib.Path(args[args.index("--output-last-message") + 1])
            out.write_text('```json\\n{{"stance":"Adopt"}}\\n```', encoding="utf-8")
            print("hook noise that must not be parsed")
            print(json.dumps({{"usage": {{"input_tokens": 8, "output_tokens": 5}}}}))
            """
        ),
    )
    llm = CodexSubprocessLlm(binary=str(fake), timeout_seconds=12)

    result = llm.complete("Prompt", schema={"type": "object"})

    assert result == {"stance": "Adopt"}
    assert llm.last_usage == {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13, "estimated": False}
    call = json.loads(calls_file.read_text(encoding="utf-8"))
    assert call["args"][:3] == ["-a", "never", "exec"]
    assert "--json" in call["args"]
    assert "--output-last-message" in call["args"]
    assert "--output-schema" in call["args"]
    assert "--ephemeral" in call["args"]
    assert "--ignore-rules" in call["args"]
    assert "--skip-git-repo-check" in call["args"]
    assert call["args"][call["args"].index("--sandbox") + 1] == "read-only"
    cwd = Path(call["args"][call["args"].index("--cd") + 1])
    assert cwd.name.startswith("cluxion-codex-")
    assert cwd != Path.cwd()
    assert call["prompt"].startswith("Prompt")
    assert "Return ONLY one JSON object" in call["prompt"]


def test_default_and_per_call_model_reach_codex_m_flag(tmp_path: Path) -> None:
    calls_file = tmp_path / "calls.jsonl"
    fake = _fake_codex(
        tmp_path / "codex",
        textwrap.dedent(
            f"""
            import json, pathlib, sys
            args = sys.argv[1:]
            with pathlib.Path({str(calls_file)!r}).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(args) + "\\n")
            pathlib.Path(args[args.index("--output-last-message") + 1]).write_text("raw text", encoding="utf-8")
            """
        ),
    )
    llm = CodexSubprocessLlm(binary=str(fake), timeout_seconds=5, model="default-model")

    assert llm.complete("Prompt") == "raw text"
    assert llm.complete("Prompt", model="seat-model") == "raw text"

    calls = [json.loads(line) for line in calls_file.read_text(encoding="utf-8").splitlines()]
    assert calls[0][calls[0].index("-m") + 1] == "default-model"
    assert calls[1][calls[1].index("-m") + 1] == "seat-model"


def test_structured_complete_retries_once_after_malformed_json(tmp_path: Path) -> None:
    count_file = tmp_path / "count"
    fake = _fake_codex(
        tmp_path / "codex",
        textwrap.dedent(
            f"""
            import json, pathlib, sys
            args = sys.argv[1:]
            count_path = pathlib.Path({str(count_file)!r})
            count = int(count_path.read_text(encoding="utf-8") or "0") if count_path.exists() else 0
            count_path.write_text(str(count + 1), encoding="utf-8")
            output = "not json" if count == 0 else '{{"stance":"Adopt"}}'
            pathlib.Path(args[args.index("--output-last-message") + 1]).write_text(output, encoding="utf-8")
            print(json.dumps({{"usage": {{"total_tokens": 3 if count == 0 else 5}}}}))
            """
        ),
    )
    llm = CodexSubprocessLlm(binary=str(fake))

    assert llm.complete("Prompt", schema={"type": "object"}) == {"stance": "Adopt"}
    assert count_file.read_text(encoding="utf-8") == "2"
    assert llm.last_usage is not None
    assert llm.last_usage["total_tokens"] == 8


def test_structured_repair_with_unknown_attempt_keeps_usage_unknown(tmp_path: Path) -> None:
    count_file = tmp_path / "count"
    fake = _fake_codex(
        tmp_path / "codex",
        textwrap.dedent(
            f"""
            import json, pathlib, sys
            args = sys.argv[1:]
            count_path = pathlib.Path({str(count_file)!r})
            count = int(count_path.read_text(encoding="utf-8") or "0") if count_path.exists() else 0
            count_path.write_text(str(count + 1), encoding="utf-8")
            output = "not json" if count == 0 else '{{"stance":"Adopt"}}'
            pathlib.Path(args[args.index("--output-last-message") + 1]).write_text(output, encoding="utf-8")
            if count == 1:
                print(json.dumps({{"usage": {{"total_tokens": 5}}}}))
            """
        ),
    )
    llm = CodexSubprocessLlm(binary=str(fake))

    assert llm.complete("Prompt", schema={"type": "object"}) == {"stance": "Adopt"}
    assert llm.last_usage is None


def test_missing_codex_binary_raises_honest_error() -> None:
    with pytest.raises(CodexExecutableNotFoundError, match="missing-codex"):
        CodexSubprocessLlm(binary="missing-codex").complete("Prompt")


def test_sigterm_reaps_codex_process_group(tmp_path: Path) -> None:
    fake_codex = tmp_path / "codex"
    pid_file = tmp_path / "child.pid"
    _fake_codex(
        fake_codex,
        textwrap.dedent(
            f"""
            import os, pathlib, time
            pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
            time.sleep(300)
            """
        ),
    )
    script = textwrap.dedent(
        """
        import sys
        from cluxion_effort_ultracode.adapters.codex_llm import CodexSubprocessLlm

        CodexSubprocessLlm(binary=sys.argv[1], timeout_seconds=120).complete("Prompt")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(fake_codex)],
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
        assert child_pid is not None, "fake codex child never started"
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
        raise AssertionError(f"codex child {child_pid} survived parent SIGTERM")
    finally:
        if parent.poll() is None:
            parent.kill()
