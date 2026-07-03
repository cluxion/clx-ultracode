"""Ports owned by the portable core and implemented by host adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

JsonMapping = Mapping[str, Any]


class LlmPort(Protocol):
    """Host-neutral LLM completion port.

    The core owns this interface. Host adapters translate it to a concrete LLM,
    but the core never imports host SDKs or calls host APIs directly.
    """

    def complete(
        self,
        prompt: str,
        *,
        schema: JsonMapping | None = None,
        model: str | None = None,
    ) -> JsonMapping | str:
        """Return either a structured object matching schema or raw text."""


class StructuredOutputPort(Protocol):
    """Optional port for hosts that expose native structured output."""

    def complete_structured(self, prompt: str, schema: JsonMapping) -> JsonMapping | None:
        """Return a validated object or None on exhausted retry/terminal failure."""


class LogPort(Protocol):
    """Optional logging port for visible non-silent degradation notices."""

    def log(self, message: str) -> None:
        """Emit a user-visible progress or degradation message."""
