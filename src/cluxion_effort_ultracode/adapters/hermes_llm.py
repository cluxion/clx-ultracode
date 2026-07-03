"""Hermes subprocess LLM adapter for the portable consensus core."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

from cluxion_effort_ultracode.core import ConsensusProtocolError


class HermesExecutableNotFoundError(RuntimeError):
    """Raised when the configured Hermes executable cannot be launched."""


class HermesSubprocessLlm:
    """Call the configured Hermes model through the official oneshot command."""

    def __init__(
        self,
        *,
        binary: str = "hermes",
        timeout_seconds: float = 120.0,
        model: str | None = None,
    ) -> None:
        if not binary.strip():
            raise ValueError("binary must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.binary = binary
        self.timeout_seconds = timeout_seconds
        self.model = model.strip() if model else None
        self._local = threading.local()

    @property
    def last_usage(self) -> Mapping[str, int | bool] | None:
        return getattr(self._local, "last_usage", None)

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any] | str:
        """Return raw Hermes output or a parsed JSON object for structured calls."""

        if schema is None:
            return self._run_oneshot(prompt, model=model)

        structured_prompt = _structured_prompt(prompt, schema=schema, retry=False)
        output = self._run_oneshot(structured_prompt, model=model)
        try:
            return _parse_json_object(output)
        except ConsensusProtocolError as exc:
            print(f"Hermes structured JSON parse failed; retrying once: {exc}", file=sys.stderr)

        retry_prompt = _structured_prompt(prompt, schema=schema, retry=True)
        retry_output = self._run_oneshot(retry_prompt, model=model)
        try:
            return _parse_json_object(retry_output)
        except ConsensusProtocolError as exc:
            raise ConsensusProtocolError(f"Hermes structured output was not valid JSON after one retry: {exc}") from exc

    def _run_oneshot(self, prompt: str, *, model: str | None = None) -> str:
        command = self._command(prompt, model=model)
        last_transient: Exception | None = None
        for attempt in range(2):
            try:
                return self._run_oneshot_once(command)
            except (subprocess.TimeoutExpired, OSError) as exc:
                if isinstance(exc, FileNotFoundError):
                    raise HermesExecutableNotFoundError(
                        f"Hermes executable not found: {self.binary!r}. Ensure Hermes is installed and on PATH."
                    ) from exc
                last_transient = exc
                if attempt == 0:
                    time.sleep(0.1)
                    continue
                break
        if isinstance(last_transient, subprocess.TimeoutExpired):
            raise ConsensusProtocolError(
                f"hermes -z timed out after {self.timeout_seconds:g} seconds"
            ) from last_transient
        raise ConsensusProtocolError(f"hermes -z failed to start: {last_transient}") from last_transient

    def _run_oneshot_once(self, command: list[str]) -> str:
        self._local.last_usage = None
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError):
            raise

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            detail = _truncate(stderr or stdout or "no output")
            raise ConsensusProtocolError(f"hermes -z exited with code {completed.returncode}: {detail}")
        if not stdout:
            raise ConsensusProtocolError("hermes -z produced empty stdout")
        self._local.last_usage = _extract_usage(stdout, stderr)
        return stdout

    def _command(self, prompt: str, *, model: str | None = None) -> list[str]:
        command = [self.binary]
        selected_model = model.strip() if model else self.model
        if selected_model:
            command.extend(["-m", selected_model])
        command.extend(["-z", prompt])
        return command


def _structured_prompt(prompt: str, *, schema: Mapping[str, Any], retry: bool) -> str:
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


def _parse_json_object(output: str) -> Mapping[str, Any]:
    candidate = _strip_code_fence(output)
    if not candidate:
        raise ConsensusProtocolError("Hermes returned empty JSON content")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ConsensusProtocolError(f"Hermes returned malformed JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ConsensusProtocolError("Hermes returned JSON that is not an object")
    return parsed


def _strip_code_fence(output: str) -> str:
    stripped = output.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _truncate(value: str, *, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _extract_usage(stdout: str, stderr: str) -> Mapping[str, int | bool] | None:
    for text in (stderr, stdout):
        usage = _extract_json_usage(text)
        if usage is not None:
            return usage
        usage = _extract_regex_usage(text)
        if usage is not None:
            return usage
    return None


def _extract_json_usage(text: str) -> Mapping[str, int | bool] | None:
    for candidate in [text.strip(), *[line.strip() for line in text.splitlines()]]:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            usage = _usage_mapping(parsed.get("usage") if isinstance(parsed.get("usage"), Mapping) else parsed)
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


__all__ = ["HermesExecutableNotFoundError", "HermesSubprocessLlm"]
