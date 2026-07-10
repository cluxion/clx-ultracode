"""Shared subprocess helpers for CLI-backed LLM adapters."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.core import ConsensusProtocolError

_KILL_DRAIN_TIMEOUT_SECONDS = 0.5
_TERM_POLL_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True)
class ProcessOutput:
    stdout: str
    stderr: str
    returncode: int


def complete_with_optional_schema(
    prompt: str,
    *,
    schema: Mapping[str, Any] | None,
    model: str | None,
    adapter_name: str,
    run_text: Callable[[str, Mapping[str, Any] | None, str | None], str],
) -> Mapping[str, Any] | str:
    if schema is None:
        return run_text(prompt, None, model)

    structured = structured_prompt(prompt, schema=schema, retry=False)
    output = run_text(structured, schema, model)
    try:
        return parse_json_object(output, adapter_name=adapter_name)
    except ConsensusProtocolError as exc:
        print(f"{adapter_name} structured JSON parse failed; retrying once: {exc}", file=sys.stderr)

    retry = structured_prompt(prompt, schema=schema, retry=True)
    retry_output = run_text(retry, schema, model)
    try:
        return parse_json_object(retry_output, adapter_name=adapter_name)
    except ConsensusProtocolError as exc:
        raise ConsensusProtocolError(
            f"{adapter_name} structured output was not valid JSON after one retry: {exc}"
        ) from exc


def structured_prompt(prompt: str, *, schema: Mapping[str, Any], retry: bool) -> str:
    schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    instruction = (
        "Return ONLY one JSON object matching the JSON Schema below. "
        "Do not include Markdown fences, commentary, or extra text."
    )
    if retry:
        instruction = (
            "The previous response was not parseable as a single JSON object. "
            "Return ONLY raw JSON. No Markdown fences, no commentary, no prose."
        )
    return f"{prompt}\n\n{instruction}\nJSON Schema:\n{schema_text}"


def parse_json_object(output: str, *, adapter_name: str) -> Mapping[str, Any]:
    candidate = _strip_code_fence(output)
    if not candidate:
        raise ConsensusProtocolError(f"{adapter_name} returned empty JSON content")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ConsensusProtocolError(f"{adapter_name} returned malformed JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ConsensusProtocolError(f"{adapter_name} returned JSON that is not an object")
    return parsed


def run_process(
    command: list[str],
    *,
    timeout_seconds: float,
    label: str,
    cwd: Path | None = None,
    stdin_text: str | None = None,
) -> ProcessOutput:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="strict",
            cwd=str(cwd) if cwd is not None else None,
            start_new_session=True,
        )
    except (OSError, ValueError):
        raise

    _register_child(process.pid)
    try:
        if stdin_text is None:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        else:
            stdout, stderr = process.communicate(input=stdin_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_process_group(
            process,
            label=label,
            grace_seconds=_termination_grace(timeout_seconds),
        )
        raise subprocess.TimeoutExpired(command, timeout_seconds, output=stdout, stderr=stderr) from exc
    finally:
        _unregister_child(process.pid)

    return ProcessOutput(stdout or "", stderr or "", process.returncode or 0)


def run_with_transient_retry(
    run_once: Callable[[], str],
    *,
    label: str,
    binary: str,
    timeout_seconds: float,
    not_found_error: Callable[[str], RuntimeError],
) -> str:
    last_transient: Exception | None = None
    for attempt in range(2):
        try:
            return run_once()
        except (subprocess.TimeoutExpired, OSError) as exc:
            if isinstance(exc, FileNotFoundError):
                if shutil.which(binary) is None:
                    raise not_found_error(binary) from exc
                last_transient = exc
                break
            last_transient = exc
            if attempt == 0:
                time.sleep(0.1)
                continue
            break
    if isinstance(last_transient, subprocess.TimeoutExpired):
        raise ConsensusProtocolError(f"{label} timed out after {timeout_seconds:g} seconds") from last_transient
    raise ConsensusProtocolError(f"{label} failed to start: {last_transient}") from last_transient


def extract_usage(stdout: str, stderr: str) -> Mapping[str, int | bool] | None:
    for text in (stderr, stdout):
        usage = _extract_json_usage(text)
        if usage is not None:
            return usage
        usage = _extract_regex_usage(text)
        if usage is not None:
            return usage
    return None


def truncate(value: str, *, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _strip_code_fence(output: str) -> str:
    stripped = output.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_usage(text: str) -> Mapping[str, int | bool] | None:
    for candidate in [text.strip(), *[line.strip() for line in text.splitlines()]]:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        usage = _usage_from_any(parsed)
        if usage is not None:
            return usage
    return None


def _usage_from_any(value: object) -> Mapping[str, int | bool] | None:
    if not isinstance(value, Mapping):
        return None
    direct = _usage_mapping(value.get("usage") if isinstance(value.get("usage"), Mapping) else value)
    if direct is not None:
        return direct
    for nested_key in ("info", "payload", "data", "message"):
        nested = value.get(nested_key)
        if isinstance(nested, Mapping):
            usage = _usage_from_any(nested)
            if usage is not None:
                return usage
    return None


def _extract_regex_usage(text: str) -> Mapping[str, int | bool] | None:
    values = {
        key: int(match.group(1))
        for key in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens")
        if (match := re.search(rf"{key}\D+(\d+)", text))
    }
    return _usage_mapping(values)


def _usage_mapping(value: object) -> Mapping[str, int | bool] | None:
    if not isinstance(value, Mapping):
        return None
    input_tokens = _token_int(value, "input_tokens", "prompt_tokens")
    output_tokens = _token_int(value, "output_tokens", "completion_tokens")
    total_tokens = _token_int(value, "total_tokens", "tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if total_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens if output_tokens is not None else max(0, total_tokens - input_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated": False,
    }


def _token_int(value: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
    return None


_live_processes: set[int] = set()
_live_processes_lock = threading.RLock()
_atexit_registered = False
_signal_hooks_installed = False
_signal_cleanup_active = False


def _register_child(pid: int) -> None:
    global _atexit_registered, _signal_hooks_installed
    with _live_processes_lock:
        _live_processes.add(pid)
        if not _atexit_registered:
            atexit.register(_reap_live_processes)
            _atexit_registered = True
        already_installed = _signal_hooks_installed
    if already_installed:
        return

    handlers_installed = 0
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(signum)

        def _handler(signo: int, frame: object, _previous=previous) -> None:
            _reap_live_processes()
            if callable(_previous):
                _previous(signo, frame)
            elif _previous == signal.SIG_IGN:
                return
            else:
                signal.signal(signo, signal.SIG_DFL)
                os.kill(os.getpid(), signo)

        try:
            signal.signal(signum, _handler)
            handlers_installed += 1
        except (ValueError, OSError):
            continue

    if handlers_installed:
        with _live_processes_lock:
            _signal_hooks_installed = True


def _unregister_child(pid: int) -> None:
    with _live_processes_lock:
        _live_processes.discard(pid)


def _reap_live_processes() -> None:
    global _signal_cleanup_active
    with _live_processes_lock:
        if _signal_cleanup_active:
            return
        _signal_cleanup_active = True
        snapshot = set(_live_processes)
    try:
        if not snapshot:
            return
        for pid in snapshot:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pid, signal.SIGTERM)

        term_deadline = time.monotonic() + _TERM_POLL_TIMEOUT_SECONDS
        survivors = set(snapshot)
        while survivors and time.monotonic() < term_deadline:
            survivors = {pid for pid in survivors if _process_group_alive(pid)}
            if not survivors:
                break
            time.sleep(0.01)
        survivors = {pid for pid in survivors if _process_group_alive(pid)}

        for pid in survivors:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(pid, signal.SIGKILL)

        drain_deadline = time.monotonic() + _KILL_DRAIN_TIMEOUT_SECONDS
        pending = set(snapshot)
        while pending and time.monotonic() < drain_deadline:
            for pid in list(pending):
                reaped = False
                try:
                    waited, _status = os.waitpid(pid, os.WNOHANG)
                    if waited == pid:
                        reaped = True
                except ChildProcessError:
                    reaped = True
                except ProcessLookupError:
                    reaped = True
                except OSError:
                    reaped = not _pid_alive(pid)
                if reaped or not _pid_alive(pid):
                    pending.discard(pid)
            if pending:
                time.sleep(0.01)

        with _live_processes_lock:
            _live_processes.difference_update(snapshot)
    finally:
        with _live_processes_lock:
            _signal_cleanup_active = False


def _process_group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _termination_grace(timeout_seconds: float) -> float:
    return min(5.0, max(0.5, timeout_seconds * 0.5))


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    label: str,
    grace_seconds: float,
) -> tuple[str, str]:
    stderr_chunks: list[str] = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        stderr_chunks.append(f"failed to terminate {label} process group {process.pid}: {exc}")
    try:
        stdout, stderr = process.communicate(timeout=grace_seconds)
        return stdout or "", _join_stderr(stderr, stderr_chunks)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            stderr_chunks.append(f"failed to kill {label} process group {process.pid}: {exc}")
        try:
            stdout, stderr = process.communicate(timeout=_KILL_DRAIN_TIMEOUT_SECONDS)
            return stdout or "", _join_stderr(stderr, stderr_chunks)
        except subprocess.TimeoutExpired:
            stderr_chunks.append(f"{label} process group {process.pid} did not exit after SIGKILL")
            return "", _join_stderr("", stderr_chunks)


def _join_stderr(stderr: str | None, chunks: list[str]) -> str:
    parts = [part for part in [stderr or "", *chunks] if part]
    return "\n".join(parts)
