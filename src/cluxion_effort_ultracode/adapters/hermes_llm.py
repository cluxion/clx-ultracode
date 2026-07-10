"""Hermes host/plugin LLM adapter and standalone CLI bridge client."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.adapters.subprocess_common import (
    parse_json_object,
    run_process,
    structured_prompt,
    truncate,
)
from cluxion_effort_ultracode.core.errors import require_positive_finite

BRIDGE_MARKER = "cluxion-ultracode-llm"
BRIDGE_VERSION = 1
BRIDGE_CLI_NAME = "ultracode-llm"
HOST_PURPOSE = "cluxion-ultracode"
_HOST_TIMEOUT_RESERVE_CAP_S = 2.0
UsageMapping = Mapping[str, int | float | bool]


class HermesLlmError(Exception):
    """Typed Hermes bridge/host failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class HermesExecutableNotFoundError(RuntimeError):
    """Raised when the configured Hermes executable cannot be launched."""


class HermesHostLlm:
    """Duck-typed adapter over a host `ctx.llm` surface (no Hermes internals)."""

    def __init__(
        self,
        host_llm: object | Callable[[], object],
        *,
        timeout_seconds: float = 120.0,
        model: str | None = None,
    ) -> None:
        self._host_llm = host_llm
        self.timeout_seconds = require_positive_finite(timeout_seconds, "timeout_seconds")
        self.model = model.strip() if model else None
        self._local = threading.local()
        self._last_provider: str | None = None
        self._last_model: str | None = None

    @property
    def last_usage(self) -> UsageMapping | None:
        return getattr(self._local, "last_usage", None)

    @property
    def last_provider(self) -> str | None:
        return self._last_provider

    @property
    def last_model(self) -> str | None:
        return self._last_model

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any] | str:
        deadline = time.monotonic() + self.timeout_seconds
        selected_model = model.strip() if model else self.model
        self._local.last_usage = None
        self._local.usage_unknown = False
        self._last_provider = None
        self._last_model = None

        if schema is None:
            return self._complete_text(prompt, model=selected_model, deadline=deadline)

        structured = structured_prompt(prompt, schema=schema, retry=False)
        first = self._complete_text(structured, model=selected_model, deadline=deadline)
        try:
            return parse_json_object(first, adapter_name="Hermes")
        except Exception as exc:
            print(f"Hermes structured JSON parse failed; retrying once: {exc}", file=sys.stderr)

        retry = structured_prompt(prompt, schema=schema, retry=True)
        second = self._complete_text(retry, model=selected_model, deadline=deadline)
        try:
            return parse_json_object(second, adapter_name="Hermes")
        except Exception as exc:
            raise HermesLlmError(
                "invalid_model_output",
                f"Hermes structured output was not valid JSON after one retry: {exc}",
            ) from exc

    def _complete_text(self, text: str, *, model: str | None, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HermesLlmError("hermes_timeout", "Hermes host call timed out")
        host_timeout = _host_timeout_from_remaining(remaining)
        response = self._call_host(text, model=model, timeout=host_timeout)
        if deadline - time.monotonic() <= 0:
            raise HermesLlmError("hermes_timeout", "Hermes host call timed out")
        usage = _duck_usage(response)
        if usage is None:
            self._local.usage_unknown = True
        self._local.last_usage = None if self._local.usage_unknown else _sum_usage(self.last_usage, usage)
        provider, response_model = _duck_provider_model(response)
        self._last_provider = provider
        self._last_model = response_model if response_model is not None else model
        return _duck_text(response)

    def _call_host(self, text: str, *, model: str | None, timeout: float) -> object:
        host = self._resolve_host()
        complete = getattr(host, "complete", None)
        if not callable(complete):
            raise HermesLlmError("hermes_bridge_unavailable", "host llm.complete is not available")
        kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": text}],
            "timeout": timeout,
            "purpose": HOST_PURPOSE,
        }
        if model is not None:
            kwargs["model"] = model
        try:
            return complete(**kwargs)
        except TimeoutError as exc:
            raise HermesLlmError("hermes_timeout", "Hermes host call timed out") from exc
        except HermesLlmError:
            raise
        except Exception as exc:
            name = type(exc).__name__
            if name == "PluginLlmTrustError" or "trust" in name.lower():
                raise HermesLlmError(
                    "model_override_denied",
                    "Host denied the requested model override; enable explicit model trust or omit model.",
                ) from exc
            raise HermesLlmError("hermes_request_failed", f"Hermes host LLM request failed: {name}") from exc

    def _resolve_host(self) -> object:
        host = self._host_llm
        if callable(host) and not hasattr(host, "complete"):
            host = host()
            self._host_llm = host
        return host


class HermesSubprocessLlm:
    """Standalone client that launches the host-registered ultracode-llm CLI bridge."""

    def __init__(
        self,
        *,
        binary: str = "hermes",
        timeout_seconds: float = 120.0,
        model: str | None = None,
    ) -> None:
        if not binary.strip():
            raise ValueError("binary must not be empty")
        self.binary = binary
        self.timeout_seconds = require_positive_finite(timeout_seconds, "timeout_seconds")
        self.model = model.strip() if model else None
        self._local = threading.local()
        self._last_provider: str | None = None
        self._last_model: str | None = None

    @property
    def last_usage(self) -> UsageMapping | None:
        return getattr(self._local, "last_usage", None)

    @property
    def last_provider(self) -> str | None:
        return self._last_provider

    @property
    def last_model(self) -> str | None:
        return self._last_model

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any] | str:
        deadline = time.monotonic() + self.timeout_seconds
        selected_model = model.strip() if model else self.model
        self._local.last_usage = None
        self._last_provider = None
        self._last_model = None

        remaining_after_setup = deadline - time.monotonic()
        if remaining_after_setup <= 0:
            raise HermesLlmError("hermes_timeout", "Hermes bridge timed out before launch")
        request_timeout = _host_timeout_from_remaining(remaining_after_setup)
        request = {
            "v": BRIDGE_VERSION,
            "prompt": prompt,
            "schema": dict(schema) if schema is not None else None,
            "model": selected_model,
            "timeout_s": request_timeout,
        }
        process_timeout = deadline - time.monotonic()
        if process_timeout <= 0:
            raise HermesLlmError("hermes_timeout", "Hermes bridge timed out before launch")

        envelope = self._run_bridge_once(request, process_timeout=process_timeout)
        if deadline - time.monotonic() <= 0:
            raise HermesLlmError("hermes_timeout", "Hermes bridge timed out")
        if not envelope.get("ok"):
            error = envelope.get("error") if isinstance(envelope.get("error"), Mapping) else {}
            code = str(error.get("code") or "hermes_request_failed")
            message = str(error.get("message") or "Hermes bridge request failed")
            if code == "hermes_not_found":
                raise HermesExecutableNotFoundError(message)
            raise HermesLlmError(code, message)

        usage = envelope.get("usage")
        self._local.last_usage = usage if isinstance(usage, Mapping) else None
        provider = envelope.get("provider")
        response_model = envelope.get("model")
        self._last_provider = str(provider) if provider is not None else None
        self._last_model = str(response_model) if response_model is not None else selected_model

        output = envelope.get("output")
        if schema is None:
            if not isinstance(output, str):
                raise HermesLlmError("hermes_bridge_protocol", "bridge ok envelope missing string output")
            return output
        if isinstance(output, Mapping):
            return output
        if isinstance(output, str):
            try:
                return parse_json_object(output, adapter_name="Hermes")
            except Exception as exc:
                raise HermesLlmError("invalid_model_output", str(exc)) from exc
        raise HermesLlmError("hermes_bridge_protocol", "bridge ok envelope has unsupported output type")

    def _run_bridge_once(self, request: Mapping[str, Any], *, process_timeout: float) -> Mapping[str, Any]:
        launch_binary = _binary_for_launch(self.binary)
        command = [launch_binary, BRIDGE_CLI_NAME]
        stdin_text = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        workdir: str | None = None
        try:
            workdir = tempfile.mkdtemp(prefix="cluxion-hermes-")
            os.chmod(workdir, 0o700)
            try:
                completed = run_process(
                    command,
                    timeout_seconds=process_timeout,
                    label="hermes ultracode-llm",
                    cwd=Path(workdir),
                    stdin_text=stdin_text,
                )
            except FileNotFoundError as exc:
                if shutil.which(launch_binary) is None:
                    raise HermesExecutableNotFoundError(
                        f"Hermes executable not found: {self.binary!r}. Ensure Hermes is installed and on PATH."
                    ) from exc
                raise HermesLlmError(
                    "hermes_bridge_unavailable",
                    f"Hermes bridge command unavailable: {self.binary!r} {BRIDGE_CLI_NAME}",
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise HermesLlmError(
                    "hermes_timeout",
                    f"Hermes bridge timed out after {process_timeout:g} seconds",
                ) from exc
            except UnicodeError as exc:
                raise HermesLlmError(
                    "hermes_bridge_protocol",
                    "Hermes bridge output was not valid UTF-8",
                ) from exc
            except OSError as exc:
                raise HermesLlmError(
                    "hermes_request_failed",
                    f"Hermes bridge failed to start: {type(exc).__name__}",
                ) from exc
        finally:
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)

        if completed.returncode != 0:
            try:
                envelope = parse_bridge_envelope(completed.stdout)
                if envelope["ok"] is False:
                    return envelope
                raise HermesLlmError(
                    "hermes_request_failed",
                    f"hermes ultracode-llm exited with code {completed.returncode} after a success envelope",
                )
            except HermesLlmError as exc:
                if exc.code == "hermes_request_failed":
                    raise
                detail = truncate((completed.stderr or completed.stdout or "no output").strip())
                unavailable_markers = (
                    "unrecognized arguments",
                    "invalid choice",
                    "unknown command",
                    "unknown option",
                    "no such command",
                )
                if any(marker in detail.lower() for marker in unavailable_markers):
                    raise HermesLlmError(
                        "hermes_bridge_unavailable",
                        "Hermes does not provide the ultracode-llm bridge; install/update the plugin on the host.",
                    ) from exc
                raise HermesLlmError(
                    "hermes_request_failed",
                    f"hermes ultracode-llm exited with code {completed.returncode} without a valid bridge envelope",
                ) from exc

        return parse_bridge_envelope(completed.stdout)


def _binary_for_launch(binary: str) -> str:
    if os.sep in binary or (os.altsep is not None and os.altsep in binary):
        return str(Path(binary).expanduser().resolve())
    return binary


def setup_ultracode_llm_cli(parser: object) -> None:
    """Configure argparse only; must not touch host LLM or models."""

    set_defaults = getattr(parser, "set_defaults", None)
    if callable(set_defaults):
        set_defaults(_ultracode_llm_bridge=True)
    # Intentionally no model/timeout flags: the request body owns those fields.


def handle_ultracode_llm_cli(host_llm: object, args: object | None = None) -> int:
    """Read one stdin JSON request, call host LLM, emit one bridge envelope line."""

    del args
    try:
        raw_text = sys.stdin.read()
        request = validate_bridge_request(raw_text)
        adapter = HermesHostLlm(host_llm, timeout_seconds=float(request["timeout_s"]), model=request.get("model"))
        schema = request.get("schema")
        result = adapter.complete(request["prompt"], schema=schema if isinstance(schema, Mapping) else None)
        output: Any = dict(result) if isinstance(result, Mapping) else result
        envelope = {
            "marker": BRIDGE_MARKER,
            "v": BRIDGE_VERSION,
            "ok": True,
            "output": output,
            "error": None,
            "usage": dict(adapter.last_usage) if adapter.last_usage is not None else None,
            "provider": adapter.last_provider,
            "model": adapter.last_model,
        }
        print(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")), flush=True)
        return 0
    except HermesLlmError as exc:
        _print_error_envelope(exc.code, exc.message)
        return 1
    except HermesExecutableNotFoundError as exc:
        _print_error_envelope("hermes_not_found", str(exc))
        return 1
    except Exception as exc:
        _print_error_envelope("hermes_request_failed", f"bridge handler failed: {type(exc).__name__}")
        return 1


def validate_bridge_request(raw_text: str | bytes | object) -> dict[str, Any]:
    """Validate request v1 before any host model access."""

    if isinstance(raw_text, bytes):
        try:
            raw_text = raw_text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HermesLlmError("hermes_bridge_protocol", "request must be UTF-8 JSON") from exc
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise HermesLlmError("hermes_bridge_protocol", "request body is required")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HermesLlmError("hermes_bridge_protocol", "request must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HermesLlmError("hermes_bridge_protocol", "request must be a JSON object")

    allowed = {"v", "prompt", "schema", "model", "timeout_s"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise HermesLlmError("hermes_bridge_protocol", f"unknown request fields: {', '.join(unknown)}")
    if type(payload.get("v")) is not int or payload["v"] != BRIDGE_VERSION:
        raise HermesLlmError("hermes_bridge_protocol", f"unsupported request version: {payload.get('v')!r}")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise HermesLlmError("hermes_bridge_protocol", "prompt must be a string")
    schema = payload.get("schema")
    if schema is not None and not isinstance(schema, dict):
        raise HermesLlmError("hermes_bridge_protocol", "schema must be an object or null")
    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise HermesLlmError("hermes_bridge_protocol", "model must be a string or null")
    try:
        timeout_s = require_positive_finite(payload.get("timeout_s"), "timeout_s")
    except (TypeError, ValueError) as exc:
        raise HermesLlmError("hermes_bridge_protocol", "timeout_s must be a positive finite relative duration") from exc
    return {
        "v": BRIDGE_VERSION,
        "prompt": prompt,
        "schema": schema,
        "model": model.strip() if isinstance(model, str) and model.strip() else None,
        "timeout_s": timeout_s,
    }


def parse_bridge_envelope(stdout: str) -> Mapping[str, Any]:
    """Parse stdout from the end; accept only the exact bridge marker/version."""

    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("marker") != BRIDGE_MARKER:
            continue
        if type(parsed.get("v")) is not int or parsed["v"] != BRIDGE_VERSION:
            continue
        ok = parsed.get("ok")
        if type(ok) is not bool:
            continue
        if ok:
            if "output" not in parsed or parsed.get("error") is not None:
                continue
        else:
            error = parsed.get("error")
            if parsed.get("output") is not None or not isinstance(error, Mapping):
                continue
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not code.strip() or not isinstance(message, str) or not message.strip():
                continue
        return parsed
    raise HermesLlmError(
        "hermes_bridge_protocol",
        "no valid cluxion-ultracode-llm v1 envelope found on bridge stdout",
    )


def _print_error_envelope(code: str, message: str) -> None:
    envelope = {
        "marker": BRIDGE_MARKER,
        "v": BRIDGE_VERSION,
        "ok": False,
        "output": None,
        "error": {"code": code, "message": message},
        "usage": None,
        "provider": None,
        "model": None,
    }
    print(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")), flush=True)


def _host_timeout_from_remaining(remaining: float) -> float:
    if remaining <= 0:
        raise HermesLlmError("hermes_timeout", "Hermes host call timed out")
    reserve = min(_HOST_TIMEOUT_RESERVE_CAP_S, remaining / 2)
    host_timeout = remaining - reserve
    if host_timeout <= 0:
        host_timeout = remaining
    return host_timeout


def _duck_text(response: object) -> str:
    if isinstance(response, str):
        text = response
    else:
        text = getattr(response, "text", None)
        if text is None and isinstance(response, Mapping):
            text = response.get("text")
            if text is None:
                text = response.get("content")
        if text is None:
            raise HermesLlmError("invalid_model_output", "host response missing text")
    if not isinstance(text, str):
        raise HermesLlmError("invalid_model_output", "host response text must be a string")
    if not text.strip():
        raise HermesLlmError("invalid_model_output", "host response text is empty")
    return text


def _duck_usage(response: object) -> UsageMapping | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return None
    if is_dataclass(usage) and not isinstance(usage, type):
        usage = asdict(usage)
    if not isinstance(usage, Mapping):
        return None
    token_values: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        token_values[key] = value
    total = token_values.get("total_tokens", 0)
    if total == 0:
        total = token_values.get("input_tokens", 0) + token_values.get("output_tokens", 0)
    if total <= 0:
        return None
    normalized: dict[str, int | float | bool] = {
        str(key): value for key, value in usage.items() if isinstance(value, (int, float, bool)) and value is not None
    }
    normalized["total_tokens"] = total
    return normalized


def _duck_provider_model(response: object) -> tuple[str | None, str | None]:
    provider = getattr(response, "provider", None)
    model = getattr(response, "model", None)
    if isinstance(response, Mapping):
        if provider is None:
            provider = response.get("provider")
        if model is None:
            model = response.get("model")
    return (
        str(provider) if provider is not None else None,
        str(model) if model is not None else None,
    )


def _sum_usage(
    left: UsageMapping | None,
    right: UsageMapping | None,
) -> UsageMapping | None:
    if left is None:
        return dict(right) if right is not None else None
    if right is None:
        return dict(left)
    merged: dict[str, int | float | bool] = dict(left)
    for key, value in right.items():
        if key == "estimated":
            merged["estimated"] = bool(left.get("estimated")) or bool(value)
            continue
        if isinstance(value, bool):
            continue
        current = merged.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
        ):
            merged[key] = current + value
        elif key not in merged:
            merged[key] = value
    return merged


def _parse_json_object(output: str) -> Mapping[str, Any]:
    return parse_json_object(output, adapter_name="Hermes")


__all__ = [
    "BRIDGE_CLI_NAME",
    "BRIDGE_MARKER",
    "BRIDGE_VERSION",
    "HOST_PURPOSE",
    "HermesExecutableNotFoundError",
    "HermesHostLlm",
    "HermesLlmError",
    "HermesSubprocessLlm",
    "handle_ultracode_llm_cli",
    "parse_bridge_envelope",
    "setup_ultracode_llm_cli",
    "validate_bridge_request",
]
