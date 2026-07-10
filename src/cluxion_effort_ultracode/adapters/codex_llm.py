"""Codex CLI subprocess LLM adapter for the portable consensus core."""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.adapters.subprocess_common import (
    complete_with_optional_schema,
    extract_usage,
    run_process,
    run_with_transient_retry,
    truncate,
)
from cluxion_effort_ultracode.core import ConsensusProtocolError
from cluxion_effort_ultracode.core.errors import require_positive_finite


class CodexExecutableNotFoundError(RuntimeError):
    """Raised when the configured Codex executable cannot be launched."""


def _strict_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Shape a JSON schema for OpenAI strict structured output.

    The API rejects any object without additionalProperties=false and
    demands every property be listed in required; our contract schemas
    already treat all fields as required, so this is a lossless tightening.
    """

    def transform(node: Any) -> Any:
        if isinstance(node, Mapping):
            out = {key: transform(value) for key, value in node.items()}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                properties = out.get("properties")
                if isinstance(properties, dict):
                    out["required"] = list(properties)
            return out
        if isinstance(node, list):
            return [transform(item) for item in node]
        return node

    return transform(dict(schema))


class CodexSubprocessLlm:
    """Call a model through `codex exec` without exposing the user's repo as a workspace."""

    def __init__(
        self,
        *,
        binary: str = "codex",
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
        """Return raw Codex final output or a parsed JSON object for structured calls."""

        return complete_with_optional_schema(
            prompt,
            schema=schema,
            model=model,
            adapter_name="Codex",
            run_text=self._run_exec,
        )

    def _run_exec(self, prompt: str, schema: Mapping[str, Any] | None, model: str | None) -> str:
        return run_with_transient_retry(
            lambda: self._run_exec_once(prompt, schema=schema, model=model),
            label="codex exec",
            binary=self.binary,
            timeout_seconds=self.timeout_seconds,
            not_found_error=lambda binary: CodexExecutableNotFoundError(
                f"Codex executable not found: {binary!r}. Ensure Codex is installed and on PATH."
            ),
        )

    def _run_exec_once(self, prompt: str, *, schema: Mapping[str, Any] | None, model: str | None) -> str:
        self._local.last_usage = None
        with tempfile.TemporaryDirectory(prefix="cluxion-codex-") as tmp:
            workdir = Path(tmp)
            output_path = workdir / "last-message.txt"
            schema_path = workdir / "schema.json" if schema is not None else None
            if schema_path is not None:
                schema_path.write_text(
                    json.dumps(_strict_schema(schema), ensure_ascii=False, sort_keys=True), encoding="utf-8"
                )
            command = self._command(workdir=workdir, output_path=output_path, schema_path=schema_path, model=model)
            completed = run_process(
                command,
                timeout_seconds=self.timeout_seconds,
                label="codex exec",
                cwd=workdir,
                stdin_text=prompt,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            if completed.returncode != 0:
                detail = truncate(stderr or stdout or "no output")
                raise ConsensusProtocolError(f"codex exec exited with code {completed.returncode}: {detail}")
            try:
                message = output_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ConsensusProtocolError("codex exec did not write --output-last-message") from exc
            if not message:
                raise ConsensusProtocolError("codex exec produced empty final message")
            self._local.last_usage = extract_usage(stdout, stderr)
            return message

    def _command(
        self,
        *,
        workdir: Path,
        output_path: Path,
        schema_path: Path | None,
        model: str | None = None,
    ) -> list[str]:
        command = [
            self.binary,
            "-a",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            str(workdir),
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            str(output_path),
        ]
        selected_model = model.strip() if model else self.model
        if selected_model:
            command.extend(["-m", selected_model])
        if schema_path is not None:
            command.extend(["--output-schema", str(schema_path)])
        command.append("-")
        return command


__all__ = ["CodexExecutableNotFoundError", "CodexSubprocessLlm"]
