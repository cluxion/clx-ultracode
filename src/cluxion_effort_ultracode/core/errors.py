"""Shared structured error helpers."""

from __future__ import annotations

import math


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


def require_utf8_text(value: str, field: str) -> str:
    """Reject lone surrogates / non-UTF-8 text before journal creation."""
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        # Context is part of the existing question/prompt input contract.
        label = "question context" if field == "context" else field
        raise ValueError(f"{label} must be valid UTF-8 text") from exc
    return value


def require_positive_finite(value: object, field: str) -> float:
    """Coerce int/float to a positive finite float.

    Accepts only int or float (not bool or str). Catches TypeError, ValueError,
    and OverflowError from float conversion. Rejects non-finite values and <= 0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be positive")
    return number
