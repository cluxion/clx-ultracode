"""Tests for the Hermes host/bridge LLM adapters."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cluxion_effort_ultracode.adapters import subprocess_common
from cluxion_effort_ultracode.adapters.hermes_llm import (
    BRIDGE_CLI_NAME,
    BRIDGE_MARKER,
    BRIDGE_VERSION,
    HOST_PURPOSE,
    HermesExecutableNotFoundError,
    HermesHostLlm,
    HermesLlmError,
    HermesSubprocessLlm,
    handle_ultracode_llm_cli,
    parse_bridge_envelope,
    setup_ultracode_llm_cli,
    validate_bridge_request,
)
from cluxion_effort_ultracode.core import ConsensusEngine


class FakeProcess:
    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = "", pid: int = 12345) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = pid
        self.timeout: float | None = None
        self.stdin_input: str | None = None

    def communicate(self, input: str | None = None, timeout: float | None = None) -> tuple[str, str]:
        self.stdin_input = input
        self.timeout = timeout
        return self.stdout, self.stderr


def _ok_envelope(output: str = "raw text", **extra: object) -> str:
    payload = {
        "marker": BRIDGE_MARKER,
        "v": BRIDGE_VERSION,
        "ok": True,
        "output": output,
        "error": None,
        "usage": None,
        "provider": "hermes-host",
        "model": "host-model",
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 10**400])
def test_hermes_constructor_rejects_non_finite_or_non_positive_timeout(timeout: float | int) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        HermesSubprocessLlm(timeout_seconds=timeout)


def test_hermes_constructor_accepts_positive_finite_timeout() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=12.5)
    assert llm.timeout_seconds == 12.5
    assert HermesSubprocessLlm(timeout_seconds=12).timeout_seconds == 12.0


def test_subprocess_complete_uses_exact_argv_and_stdin_json() -> None:
    llm = HermesSubprocessLlm(binary="/opt/hermes", timeout_seconds=12)
    envelope = _ok_envelope('{"stance":"Adopt"}')

    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess(envelope),
    ) as popen:
        result = llm.complete("Prompt", schema={"type": "object"}, model="seat-model")

    assert result == {"stance": "Adopt"}
    assert popen.call_args.args[0] == ["/opt/hermes", BRIDGE_CLI_NAME]
    assert "Prompt" not in popen.call_args.args[0]
    stdin = popen.return_value.stdin_input
    assert stdin is not None
    request = json.loads(stdin)
    assert request["v"] == 1
    assert request["prompt"] == "Prompt"
    assert request["schema"] == {"type": "object"}
    assert request["model"] == "seat-model"
    assert request["timeout_s"] > 0
    assert "monotonic" not in request
    assert popen.call_args.kwargs["start_new_session"] is True
    assert popen.call_args.kwargs["cwd"] is not None


def test_relative_binary_path_is_resolved_before_private_temp_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    llm = HermesSubprocessLlm(binary=".venv/bin/hermes", timeout_seconds=5)
    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess(_ok_envelope("ok")),
    ) as popen:
        assert llm.complete("Prompt") == "ok"
    assert popen.call_args.args[0] == [str((tmp_path / ".venv/bin/hermes").resolve()), BRIDGE_CLI_NAME]


def test_large_prompt_never_lands_on_argv() -> None:
    large = "x" * 2_000_000
    llm = HermesSubprocessLlm(timeout_seconds=30)
    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess(_ok_envelope("ok")),
    ) as popen:
        assert llm.complete(large) == "ok"
    command = popen.call_args.args[0]
    assert command == ["hermes", BRIDGE_CLI_NAME]
    assert all(len(part) < 100 for part in command)
    assert json.loads(popen.return_value.stdin_input)["prompt"] == large


def test_bridge_envelope_isolates_reverse_line_noise() -> None:
    noise = "\n".join(
        [
            "discovering tools...",
            json.dumps({"marker": "other", "v": 1, "ok": True, "output": "nope"}),
            json.dumps({"v": 1, "ok": True, "output": "no-marker"}),
            _ok_envelope("accepted"),
            "trailing noise",
        ]
    )
    assert parse_bridge_envelope(noise)["output"] == "accepted"


@pytest.mark.parametrize(
    "payload",
    [
        {**json.loads(_ok_envelope("spoof")), "v": True},
        {**json.loads(_ok_envelope("spoof")), "ok": "false"},
        {**json.loads(_ok_envelope("spoof")), "error": {}},
        {
            **json.loads(_ok_envelope("spoof")),
            "ok": False,
            "output": None,
            "error": None,
        },
        {
            **json.loads(_ok_envelope("spoof")),
            "ok": False,
            "output": None,
            "error": {"code": 1, "message": "failed"},
        },
    ],
)
def test_bridge_envelope_rejects_non_boolean_or_invalid_result_shape(payload: dict[str, object]) -> None:
    with pytest.raises(HermesLlmError) as caught:
        parse_bridge_envelope(json.dumps(payload))
    assert caught.value.code == "hermes_bridge_protocol"


def test_mode_0700_temp_cwd_cleaned_up(tmp_path: Path) -> None:
    llm = HermesSubprocessLlm(timeout_seconds=5)
    seen: dict[str, object] = {}

    def _popen(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        seen["cwd"] = cwd
        seen["mode"] = cwd.stat().st_mode & 0o777
        seen["empty"] = list(cwd.iterdir()) == []
        return FakeProcess(_ok_envelope("raw"))

    with patch("cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen", side_effect=_popen):
        assert llm.complete("Prompt") == "raw"
    assert seen["mode"] == 0o700
    assert seen["empty"] is True
    assert not Path(str(seen["cwd"])).exists()


def test_structured_complete_one_process_across_malformed_json_retry() -> None:
    """Repair retry lives inside the bridge process — outer client launches once."""
    structured = json.dumps({"stance": "Adopt", "rationale": "R", "evidence": ["E"], "confidence": 0.9})
    llm = HermesSubprocessLlm()
    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess(_ok_envelope(structured)),
    ) as popen:
        result = llm.complete("Prompt", schema={"type": "object"})
    assert result == json.loads(structured)
    assert popen.call_count == 1


def test_host_llm_does_schema_repair_without_subprocess() -> None:
    calls: list[dict[str, object]] = []

    class Host:
        def complete(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(text="not json", model="m1", usage={"total_tokens": 3}, provider="p")
            return SimpleNamespace(
                text='{"stance":"Adopt"}',
                model="m2",
                usage={"total_tokens": 5, "input_tokens": 2, "output_tokens": 3},
                provider="p",
            )

    llm = HermesHostLlm(Host(), timeout_seconds=20)
    with patch("cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen") as popen:
        result = llm.complete("Prompt", schema={"type": "object"})
    assert result == {"stance": "Adopt"}
    assert popen.call_count == 0
    assert len(calls) == 2
    assert calls[0]["purpose"] == HOST_PURPOSE
    assert calls[0]["messages"][0]["role"] == "user"
    assert llm.last_usage is not None
    assert llm.last_usage["total_tokens"] == 8
    assert llm.last_model == "m2"
    assert llm.last_provider == "p"


@pytest.mark.parametrize(
    "usages",
    [
        (None, {"total_tokens": 5}),
        ({"total_tokens": 3}, None),
        ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, {"total_tokens": 5}),
    ],
)
def test_host_schema_repair_keeps_mixed_known_unknown_usage_honest(usages: tuple[object, object]) -> None:
    calls = 0

    class Host:
        def complete(self, **kwargs):
            nonlocal calls
            index = calls
            calls += 1
            text = "not json" if index == 0 else '{"stance":"Adopt"}'
            return SimpleNamespace(text=text, model="m", provider="p", usage=usages[index])

    llm = HermesHostLlm(Host())
    assert llm.complete("Prompt", schema={"type": "object"}) == {"stance": "Adopt"}
    assert llm.last_usage is None


def test_host_schema_repair_attribution_comes_from_final_response() -> None:
    calls = 0

    class Host:
        def complete(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(text="not json", model="first-model", provider="first-provider", usage=None)
            return SimpleNamespace(text='{"stance":"Adopt"}', usage=None)

    llm = HermesHostLlm(Host())
    assert llm.complete("Prompt", schema={"type": "object"}) == {"stance": "Adopt"}
    assert llm.last_provider is None
    assert llm.last_model is None


def test_host_model_override_denied_is_typed() -> None:
    class PluginLlmTrustError(PermissionError):
        pass

    class Host:
        def complete(self, **kwargs):
            raise PluginLlmTrustError("model trust denied")

    llm = HermesHostLlm(Host(), timeout_seconds=5)
    with pytest.raises(HermesLlmError) as caught:
        llm.complete("Prompt", model="secret-model")
    assert caught.value.code == "model_override_denied"
    assert "secret-model" not in caught.value.message


def test_plain_host_permission_error_is_not_misclassified_as_model_trust() -> None:
    class Host:
        def complete(self, **kwargs):
            raise PermissionError("credential path /secret/token is unreadable")

    with pytest.raises(HermesLlmError) as caught:
        HermesHostLlm(Host(), timeout_seconds=5).complete("Prompt")
    assert caught.value.code == "hermes_request_failed"
    assert "/secret/token" not in caught.value.message


def test_host_timeout_error_is_typed() -> None:
    class Host:
        def complete(self, **kwargs):
            raise TimeoutError("provider timed out")

    with pytest.raises(HermesLlmError) as caught:
        HermesHostLlm(Host(), timeout_seconds=5).complete("Prompt")
    assert caught.value.code == "hermes_timeout"


def test_host_usage_dataclass_is_preserved() -> None:
    @dataclass
    class Usage:
        input_tokens: int = 3
        output_tokens: int = 5
        total_tokens: int = 8
        cache_read_tokens: int = 2
        cache_write_tokens: int = 1
        cost_usd: float | None = 0.01

    class Host:
        def complete(self, **kwargs):
            return SimpleNamespace(text="ok", model="m", provider="p", usage=Usage())

    llm = HermesHostLlm(Host())
    assert llm.complete("Prompt") == "ok"
    assert llm.last_usage == {
        "input_tokens": 3,
        "output_tokens": 5,
        "total_tokens": 8,
        "cache_read_tokens": 2,
        "cache_write_tokens": 1,
        "cost_usd": 0.01,
    }


@pytest.mark.parametrize("usage", [{}, {"total_tokens": 0}])
def test_host_empty_or_zero_usage_mapping_is_unknown(usage: dict[str, object]) -> None:
    class Host:
        def complete(self, **kwargs):
            return SimpleNamespace(text="ok", model="m", provider="p", usage=usage)

    llm = HermesHostLlm(Host())
    assert llm.complete("Prompt") == "ok"
    assert llm.last_usage is None


def test_host_default_zero_usage_dataclass_is_unknown() -> None:
    @dataclass
    class Usage:
        input_tokens: int = 0
        output_tokens: int = 0
        total_tokens: int = 0
        cache_read_tokens: int = 0
        cache_write_tokens: int = 0
        cost_usd: float | None = None

    class Host:
        def complete(self, **kwargs):
            return SimpleNamespace(text="ok", model="m", provider="p", usage=Usage())

    llm = HermesHostLlm(Host())
    assert llm.complete("Prompt") == "ok"
    assert llm.last_usage is None


def test_cli_setup_is_token_free() -> None:
    class Parser:
        def __init__(self) -> None:
            self.defaults = {}

        def set_defaults(self, **kwargs):
            self.defaults.update(kwargs)

    parser = Parser()
    setup_ultracode_llm_cli(parser)
    assert parser.defaults.get("_ultracode_llm_bridge") is True


def test_validate_bridge_request_rejects_malformed_before_model() -> None:
    with pytest.raises(HermesLlmError) as caught:
        validate_bridge_request('{"v":2,"prompt":"x","timeout_s":1}')
    assert caught.value.code == "hermes_bridge_protocol"

    with pytest.raises(HermesLlmError) as caught:
        validate_bridge_request('{"v":1,"prompt":"x","timeout_s":0}')
    assert caught.value.code == "hermes_bridge_protocol"

    with pytest.raises(HermesLlmError) as caught:
        validate_bridge_request('{"v":1,"prompt":"x","timeout_s":1,"extra":1}')
    assert caught.value.code == "hermes_bridge_protocol"

    with pytest.raises(HermesLlmError) as caught:
        validate_bridge_request('{"v":true,"prompt":"x","timeout_s":1}')
    assert caught.value.code == "hermes_bridge_protocol"


def test_handle_bridge_cli_uses_host_and_emits_envelope(capsys, monkeypatch) -> None:
    class Host:
        def complete(self, **kwargs):
            return SimpleNamespace(text="hello", model="m", usage={"total_tokens": 2}, provider="p")

    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"v": 1, "prompt": "hi", "schema": None, "model": None, "timeout_s": 5})),
    )
    code = handle_ultracode_llm_cli(Host())
    assert code == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    envelope = json.loads(line)
    assert envelope["marker"] == BRIDGE_MARKER
    assert envelope["ok"] is True
    assert envelope["output"] == "hello"
    assert envelope["usage"]["total_tokens"] == 2


def test_hermes_outer_failure_is_not_retried() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=2)
    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            side_effect=OSError("boom"),
        ) as popen,
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_request_failed"
    assert popen.call_count == 1


def test_non_envelope_failure_does_not_echo_stderr_or_secrets() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=2)
    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            return_value=FakeProcess("not an envelope", returncode=1, stderr="Traceback: API_KEY=super-secret"),
        ),
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_request_failed"
    assert "Traceback" not in caught.value.message
    assert "super-secret" not in caught.value.message


def test_unknown_configuration_error_is_not_misclassified_as_missing_bridge() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=2)
    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            return_value=FakeProcess("", returncode=2, stderr="unknown configuration key: llm.provider"),
        ),
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_request_failed"


def test_nonzero_process_cannot_return_success_envelope() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=2)
    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            return_value=FakeProcess(_ok_envelope("spoof"), returncode=1),
        ),
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_request_failed"


def test_non_utf8_bridge_output_is_typed() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=2)
    decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    with (
        patch(
            "cluxion_effort_ultracode.adapters.hermes_llm.run_process",
            side_effect=decode_error,
        ),
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_bridge_protocol"


def test_missing_hermes_binary_raises_honest_error() -> None:
    llm = HermesSubprocessLlm(binary="missing-hermes")
    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            side_effect=FileNotFoundError,
        ),
        patch("cluxion_effort_ultracode.adapters.hermes_llm.shutil.which", return_value=None),
        pytest.raises(HermesExecutableNotFoundError, match="missing-hermes"),
    ):
        llm.complete("Prompt")


def test_absent_bridge_command_is_actionable() -> None:
    llm = HermesSubprocessLlm(binary="/opt/hermes")
    with (
        patch(
            "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
            return_value=FakeProcess(
                "unrecognized arguments: ultracode-llm", returncode=2, stderr="unrecognized arguments"
            ),
        ),
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_bridge_unavailable"


def test_remaining_timeout_decreases_for_host_calls() -> None:
    seen: list[float] = []

    class Host:
        def complete(self, **kwargs):
            seen.append(float(kwargs["timeout"]))
            time.sleep(0.05)
            if len(seen) == 1:
                return SimpleNamespace(text="not-json", model="m", usage=None, provider="p")
            return SimpleNamespace(text='{"stance":"A"}', model="m", usage=None, provider="p")

    llm = HermesHostLlm(Host(), timeout_seconds=1.0)
    llm.complete("Prompt", schema={"type": "object"})
    assert len(seen) == 2
    assert seen[1] < seen[0]


def test_host_result_returned_after_parent_deadline_is_rejected() -> None:
    class Host:
        def complete(self, **kwargs):
            return SimpleNamespace(text="late", model="m", usage=None, provider="p")

    llm = HermesHostLlm(Host(), timeout_seconds=0.5)
    with (
        patch(
            "cluxion_effort_ultracode.adapters.hermes_llm.time.monotonic",
            side_effect=[10.0, 10.1, 10.6],
        ),
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_timeout"


def test_subprocess_result_returned_after_parent_deadline_is_rejected() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=0.5)
    with (
        patch(
            "cluxion_effort_ultracode.adapters.hermes_llm.time.monotonic",
            side_effect=[10.0, 10.1, 10.2, 10.6],
        ),
        patch.object(llm, "_run_bridge_once", return_value=json.loads(_ok_envelope("late"))),
        pytest.raises(HermesLlmError) as caught,
    ):
        llm.complete("Prompt")
    assert caught.value.code == "hermes_timeout"


def test_usage_aggregation_and_provider_model_on_subprocess() -> None:
    envelope = _ok_envelope(
        "raw",
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3, "estimated": False},
        provider="bridge-provider",
        model="bridge-model",
    )
    llm = HermesSubprocessLlm()
    with patch(
        "cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen",
        return_value=FakeProcess(envelope),
    ):
        assert llm.complete("Prompt") == "raw"
    assert llm.last_usage["total_tokens"] == 3
    assert llm.last_provider == "bridge-provider"
    assert llm.last_model == "bridge-model"


def test_sigterm_reaps_hermes_process_group(tmp_path: Path) -> None:
    fake_hermes = tmp_path / "hermes"
    pid_file = tmp_path / "child.pid"
    fake_hermes.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "sys.stdout.flush()\n"
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


def test_term_ignoring_child_is_killed_and_reaped(tmp_path: Path) -> None:
    child = tmp_path / "ignore_term.py"
    ready = tmp_path / "ready"
    child.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8')\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    child.chmod(0o755)
    proc = subprocess.Popen([sys.executable, str(child), str(ready)], start_new_session=True)
    subprocess_common._register_child(proc.pid)
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.01)
        assert ready.exists(), "child did not install SIGTERM handler"
        subprocess_common._reap_live_processes()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        assert proc.poll() is not None
        # not orphaned under init as a live process
        try:
            os.kill(proc.pid, 0)
            raise AssertionError("child still alive after reap")
        except ProcessLookupError:
            pass
        with subprocess_common._live_processes_lock:
            assert proc.pid not in subprocess_common._live_processes
    finally:
        with subprocess_common._live_processes_lock:
            subprocess_common._live_processes.discard(proc.pid)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)


def test_signal_hook_preserves_previous_sig_ign() -> None:
    handlers: dict[int, object] = {}
    with subprocess_common._live_processes_lock:
        saved_pids = set(subprocess_common._live_processes)
        saved_installed = subprocess_common._signal_hooks_installed
        subprocess_common._live_processes.clear()
        subprocess_common._signal_hooks_installed = False

    def _capture_handler(signum, handler):
        handlers[signum] = handler

    try:
        with (
            patch("cluxion_effort_ultracode.adapters.subprocess_common.atexit.register"),
            patch("cluxion_effort_ultracode.adapters.subprocess_common.signal.getsignal", return_value=signal.SIG_IGN),
            patch("cluxion_effort_ultracode.adapters.subprocess_common.signal.signal", side_effect=_capture_handler),
        ):
            subprocess_common._register_child(12345)
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            with (
                patch("cluxion_effort_ultracode.adapters.subprocess_common._reap_live_processes") as reap,
                patch("cluxion_effort_ultracode.adapters.subprocess_common.os.kill") as kill,
            ):
                handler(signal.SIGTERM, None)
            reap.assert_called_once_with()
            kill.assert_not_called()
    finally:
        with subprocess_common._live_processes_lock:
            subprocess_common._live_processes.clear()
            subprocess_common._live_processes.update(saved_pids)
            subprocess_common._signal_hooks_installed = saved_installed


def test_worker_thread_child_registration_adds_one_atexit_hook() -> None:
    with subprocess_common._live_processes_lock:
        saved_pids = set(subprocess_common._live_processes)
        saved_installed = subprocess_common._signal_hooks_installed
        saved_atexit = getattr(subprocess_common, "_atexit_registered", False)
        subprocess_common._live_processes.clear()
        subprocess_common._signal_hooks_installed = False
        subprocess_common._atexit_registered = False

    try:
        with patch("cluxion_effort_ultracode.adapters.subprocess_common.atexit.register") as register:
            for pid in (111, 222, 333):
                worker = threading.Thread(target=subprocess_common._register_child, args=(pid,))
                worker.start()
                worker.join(timeout=2)
                assert not worker.is_alive()
        register.assert_called_once_with(subprocess_common._reap_live_processes)
    finally:
        with subprocess_common._live_processes_lock:
            subprocess_common._live_processes.clear()
            subprocess_common._live_processes.update(saved_pids)
            subprocess_common._signal_hooks_installed = saved_installed
            subprocess_common._atexit_registered = saved_atexit


def test_multiple_children_share_one_global_term_window() -> None:
    clock = 0.0
    sleep_calls: list[float] = []
    pids = {111, 222, 333}

    def _monotonic() -> float:
        return clock

    def _sleep(seconds: float) -> None:
        nonlocal clock
        sleep_calls.append(seconds)
        clock += seconds

    with subprocess_common._live_processes_lock:
        saved_pids = set(subprocess_common._live_processes)
        subprocess_common._live_processes.clear()
        subprocess_common._live_processes.update(pids)
    try:
        with (
            patch.object(subprocess_common, "_TERM_POLL_TIMEOUT_SECONDS", 0.05),
            patch.object(subprocess_common.time, "monotonic", side_effect=_monotonic),
            patch.object(subprocess_common.time, "sleep", side_effect=_sleep),
            patch.object(subprocess_common, "_process_group_alive", return_value=True),
            patch.object(subprocess_common, "_pid_alive", return_value=False),
            patch.object(subprocess_common.os, "waitpid", side_effect=ChildProcessError),
            patch.object(subprocess_common.os, "killpg") as killpg,
        ):
            subprocess_common._reap_live_processes()

        assert 0.05 <= clock < 0.06
        assert len(sleep_calls) <= 6
        for pid in pids:
            killpg.assert_any_call(pid, signal.SIGTERM)
            killpg.assert_any_call(pid, signal.SIGKILL)
    finally:
        with subprocess_common._live_processes_lock:
            subprocess_common._live_processes.clear()
            subprocess_common._live_processes.update(saved_pids)


def test_reap_preserves_pids_registered_after_snapshot() -> None:
    late_pid = 424242
    with subprocess_common._live_processes_lock:
        subprocess_common._live_processes.clear()
        subprocess_common._live_processes.add(111)
    original_killpg = os.killpg

    def _killpg(pid, sig):
        if pid == 111 and sig == signal.SIGTERM:
            with subprocess_common._live_processes_lock:
                subprocess_common._live_processes.add(late_pid)
            raise ProcessLookupError
        if pid == late_pid:
            raise AssertionError("late pid should not be in snapshot cleanup")
        return original_killpg(pid, sig)

    with (
        patch("cluxion_effort_ultracode.adapters.subprocess_common.os.killpg", side_effect=_killpg),
        patch("cluxion_effort_ultracode.adapters.subprocess_common._process_group_alive", return_value=False),
        patch("cluxion_effort_ultracode.adapters.subprocess_common.os.waitpid", side_effect=ChildProcessError),
    ):
        subprocess_common._reap_live_processes()
    with subprocess_common._live_processes_lock:
        assert late_pid in subprocess_common._live_processes
        assert 111 not in subprocess_common._live_processes
        subprocess_common._live_processes.discard(late_pid)


def test_reap_reentry_returns_immediately() -> None:
    entered = threading.Event()
    release = threading.Event()
    results: list[str] = []

    def _blocking_killpg(pid, sig):
        entered.set()
        release.wait(timeout=2)
        raise ProcessLookupError

    with subprocess_common._live_processes_lock:
        subprocess_common._live_processes.add(999001)

    def worker():
        with (
            patch("cluxion_effort_ultracode.adapters.subprocess_common.os.killpg", side_effect=_blocking_killpg),
            patch("cluxion_effort_ultracode.adapters.subprocess_common._process_group_alive", return_value=False),
            patch("cluxion_effort_ultracode.adapters.subprocess_common.os.waitpid", side_effect=ChildProcessError),
        ):
            subprocess_common._reap_live_processes()
            results.append("done")

    t = threading.Thread(target=worker)
    t.start()
    assert entered.wait(timeout=2)
    # re-entry while first cleanup active
    subprocess_common._reap_live_processes()
    results.append("reentered")
    release.set()
    t.join(timeout=3)
    assert "done" in results
    assert "reentered" in results
    with subprocess_common._live_processes_lock:
        subprocess_common._live_processes.discard(999001)
        assert subprocess_common._signal_cleanup_active is False


def test_already_exited_processlookup_is_harmless() -> None:
    with subprocess_common._live_processes_lock:
        subprocess_common._live_processes.add(999002)
    with (
        patch("cluxion_effort_ultracode.adapters.subprocess_common.os.killpg", side_effect=ProcessLookupError),
        patch("cluxion_effort_ultracode.adapters.subprocess_common._process_group_alive", return_value=False),
        patch("cluxion_effort_ultracode.adapters.subprocess_common.os.waitpid", side_effect=ChildProcessError),
    ):
        subprocess_common._reap_live_processes()
    with subprocess_common._live_processes_lock:
        assert 999002 not in subprocess_common._live_processes


@pytest.mark.skipif(
    os.getenv("CLUXION_EFFORT_ULTRACODE_LIVE") != "1",
    reason="set CLUXION_EFFORT_ULTRACODE_LIVE=1 to run real hermes ultracode-llm consensus",
)
def test_live_tiny_consensus_via_hermes_bridge() -> None:
    llm = HermesSubprocessLlm(timeout_seconds=120)

    result = ConsensusEngine(llm, agents_count=2, max_rounds=1).decide(
        "Use the exact stance YES. Is YES the correct stance for this smoke test?",
        context="Keep evidence short. The intended answer is YES.",
    )

    assert result.agents_count == 2
    assert result.rounds <= 1
