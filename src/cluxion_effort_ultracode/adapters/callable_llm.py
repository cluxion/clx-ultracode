"""Reference LLM adapter that wraps plain Python callables."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

LlmCallable: TypeAlias = Callable[[str], Mapping[str, Any] | str]
StructuredCallable: TypeAlias = Callable[[str, Mapping[str, Any]], Mapping[str, Any] | str]


class CallableLlmAdapter:
    """Wrap callables behind the core LlmPort without any host dependency."""

    def __init__(
        self,
        complete: LlmCallable,
        *,
        structured_complete: StructuredCallable | None = None,
    ) -> None:
        self._complete = complete
        self._structured_complete = structured_complete

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any] | str:
        """Return callable output, parsing JSON text when structured output is requested."""

        del model
        if schema is not None and self._structured_complete is not None:
            return _maybe_parse_json(self._structured_complete(prompt, schema))
        return _maybe_parse_json(self._complete(prompt)) if schema is not None else self._complete(prompt)


def _maybe_parse_json(value: Mapping[str, Any] | str) -> Mapping[str, Any] | str:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, Mapping) else value
