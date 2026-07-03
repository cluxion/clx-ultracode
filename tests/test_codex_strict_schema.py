from __future__ import annotations

from cluxion_effort_ultracode.adapters.codex_llm import _strict_schema
from cluxion_effort_ultracode.core.consensus import DEBATE_SCHEMA, POSITION_SCHEMA


def _assert_strict(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(node.get("properties", {}))
        for value in node.values():
            _assert_strict(value)
    elif isinstance(node, list):
        for item in node:
            _assert_strict(item)


def test_contract_schemas_become_strict() -> None:
    for schema in (POSITION_SCHEMA, DEBATE_SCHEMA):
        _assert_strict(_strict_schema(schema))


def test_original_schema_untouched() -> None:
    before = dict(POSITION_SCHEMA)
    _strict_schema(POSITION_SCHEMA)
    assert before == POSITION_SCHEMA
