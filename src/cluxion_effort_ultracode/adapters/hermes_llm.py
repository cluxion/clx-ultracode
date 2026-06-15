"""Hermes subprocess LLM adapter for the portable consensus core."""

from __future__ import annotations

import json
import subprocess
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

    def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> Mapping[str, Any] | str:
        """Return raw Hermes output or a parsed JSON object for structured calls."""

        if schema is None:
            return self._run_oneshot(prompt)

        last_error: ConsensusProtocolError | None = None
        for retry in (False, True):
            structured_prompt = _structured_prompt(prompt, schema=schema, retry=retry)
            output = self._run_oneshot(structured_prompt)
            try:
                return _parse_json_object(output)
            except ConsensusProtocolError as exc:
                last_error = exc
                if not retry:
                    continue
                raise ConsensusProtocolError(
                    f"Hermes structured output was not valid JSON after one retry: {exc}"
                ) from exc
        raise ConsensusProtocolError(f"Hermes structured output could not be parsed: {last_error}")

    def _run_oneshot(self, prompt: str) -> str:
        command = self._command(prompt)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise HermesExecutableNotFoundError(
                f"Hermes executable not found: {self.binary!r}. Ensure Hermes is installed and on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ConsensusProtocolError(f"hermes -z timed out after {self.timeout_seconds:g} seconds") from exc
        except OSError as exc:
            raise ConsensusProtocolError(f"hermes -z failed to start: {exc}") from exc

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            detail = _truncate(stderr or stdout or "no output")
            raise ConsensusProtocolError(f"hermes -z exited with code {completed.returncode}: {detail}")
        if not stdout:
            raise ConsensusProtocolError("hermes -z produced empty stdout")
        return stdout

    def _command(self, prompt: str) -> list[str]:
        command = [self.binary]
        if self.model:
            command.extend(["--model", self.model])
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


__all__ = ["HermesExecutableNotFoundError", "HermesSubprocessLlm"]
