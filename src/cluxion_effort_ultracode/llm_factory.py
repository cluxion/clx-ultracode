"""Shared LLM factory for CLI and plugin entry points."""

from __future__ import annotations

import os

from cluxion_effort_ultracode.adapters.codex_llm import CodexSubprocessLlm
from cluxion_effort_ultracode.adapters.hermes_llm import HermesSubprocessLlm


def default_llm(adapter: str = "hermes", *, timeout_seconds: float | None = None) -> HermesSubprocessLlm | CodexSubprocessLlm:
    timeout = timeout_from_env() if timeout_seconds is None else _validate_timeout(timeout_seconds)
    if adapter == "hermes":
        binary = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", "hermes")
        model = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_MODEL") or None
        return HermesSubprocessLlm(binary=binary, timeout_seconds=timeout, model=model)
    if adapter == "codex":
        binary = os.getenv("CLUXION_EFFORT_ULTRACODE_CODEX_BINARY", "codex")
        model = os.getenv("CLUXION_EFFORT_ULTRACODE_CODEX_MODEL") or None
        return CodexSubprocessLlm(binary=binary, timeout_seconds=timeout, model=model)
    raise ValueError(f"unknown adapter: {adapter}")


def timeout_from_env() -> float:
    raw = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "").strip()
    if not raw:
        return 120.0
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT must be numeric") from exc
    return _validate_timeout(timeout, env_name="CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT")


def _validate_timeout(timeout: float, *, env_name: str = "timeout_seconds") -> float:
    if timeout <= 0:
        raise ValueError(f"{env_name} must be greater than zero")
    return timeout


__all__ = ["default_llm", "timeout_from_env"]
