"""Shared structured error helpers."""

from __future__ import annotations


def validation_error_code(exc: BaseException) -> str:
    message = str(exc)
    lower = message.lower()
    if "models" in lower:
        return "invalid_models"
    if "agents_count" in lower or lower.startswith("agents "):
        return "invalid_agents"
    if "max_rounds" in lower or lower.startswith("rounds "):
        return "invalid_rounds"
    if "timeout" in lower:
        return "invalid_timeout"
    if "budget" in lower:
        return "invalid_budget"
    if "question" in lower or lower.startswith("unknown arguments"):
        return "invalid_question"
    return type(exc).__name__
