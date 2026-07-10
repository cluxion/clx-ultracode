"""Hermes subprocess LLM adapter for the portable consensus core."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from cluxion_effort_ultracode.adapters.subprocess_common import (
    complete_with_optional_schema,
    extract_usage,
    parse_json_object,
    run_process,
    run_with_transient_retry,
    truncate,
)
from cluxion_effort_ultracode.core import ConsensusProtocolError
from cluxion_effort_ultracode.core.errors import require_positive_finite


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
        self.binary = binary
        self.timeout_seconds = require_positive_finite(timeout_seconds, "timeout_seconds")
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

        return complete_with_optional_schema(
            prompt,
            schema=schema,
            model=model,
            adapter_name="Hermes",
            run_text=lambda text, _schema, selected_model: self._run_oneshot(text, model=selected_model),
        )

    def _run_oneshot(self, prompt: str, *, model: str | None = None) -> str:
        command = self._command(prompt, model=model)
        return run_with_transient_retry(
            lambda: self._run_oneshot_once(command),
            label="hermes -z",
            binary=self.binary,
            timeout_seconds=self.timeout_seconds,
            not_found_error=lambda binary: HermesExecutableNotFoundError(
                f"Hermes executable not found: {binary!r}. Ensure Hermes is installed and on PATH."
            ),
        )

    def _run_oneshot_once(self, command: list[str]) -> str:
        self._local.last_usage = None
        completed = run_process(command, timeout_seconds=self.timeout_seconds, label="hermes")
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            detail = truncate(stderr or stdout or "no output")
            raise ConsensusProtocolError(f"hermes -z exited with code {completed.returncode}: {detail}")
        if not stdout:
            raise ConsensusProtocolError("hermes -z produced empty stdout")
        self._local.last_usage = extract_usage(stdout, stderr)
        return stdout

    def _command(self, prompt: str, *, model: str | None = None) -> list[str]:
        command = [self.binary]
        selected_model = model.strip() if model else self.model
        if selected_model:
            command.extend(["-m", selected_model])
        command.extend(["-z", prompt])
        return command


def _parse_json_object(output: str) -> Mapping[str, Any]:
    return parse_json_object(output, adapter_name="Hermes")


__all__ = ["HermesExecutableNotFoundError", "HermesSubprocessLlm"]
