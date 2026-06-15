"""Tests for the Hermes plugin registration shim."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from typing import Any

from cluxion_effort_ultracode import plugin
from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError


class RecordingHermesContext:
    def __init__(self) -> None:
        self.tools: list[dict[str, object]] = []

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict[str, object],
        handler: object,
        check_fn: object = None,
        requires_env: list[object] | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        override: bool = False,
    ) -> None:
        self.tools.append(
            {
                "name": name,
                "toolset": toolset,
                "schema": schema,
                "handler": handler,
                "check_fn": check_fn,
                "requires_env": requires_env,
                "is_async": is_async,
                "description": description,
                "emoji": emoji,
                "override": override,
            }
        )


class ScriptedLlm:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = deque(outputs)
        self.calls: list[dict[str, object]] = []

    def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.outputs.popleft()


class MissingHermesLlm:
    def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        raise HermesExecutableNotFoundError("Hermes executable not found: 'hermes'")


def position(stance: str) -> dict[str, Any]:
    return {
        "stance": stance,
        "rationale": f"Rationale for {stance}",
        "evidence": [f"Evidence for {stance}"],
        "confidence": 0.8,
    }


def test_register_uses_real_hermes_tool_contract() -> None:
    ctx = RecordingHermesContext()

    plugin.register(ctx)

    assert len(ctx.tools) == 1
    registered = ctx.tools[0]
    assert registered["name"] == "cluxion_consensus"
    assert registered["toolset"] == "ultracode"
    assert registered["emoji"] == "🧠"
    assert callable(registered["handler"])

    schema = registered["schema"]
    assert schema["name"] == "cluxion_consensus"
    assert isinstance(schema["description"], str)
    assert "agents * (rounds + 1)" in schema["description"]
    assert schema["parameters"]["type"] == "object"
    assert schema["parameters"]["required"] == ["question"]
    assert "question" in schema["parameters"]["properties"]


def test_register_tolerates_host_without_register_tool() -> None:
    plugin.register(object())


def test_consensus_handler_returns_json_string_with_scripted_llm() -> None:
    llm = ScriptedLlm([position("yes"), position("YES.")])
    handler = plugin.build_consensus_handler(lambda: llm)

    raw = handler({"question": "Should we answer yes?", "rounds": 0, "agents": 2})
    payload = json.loads(raw)

    assert isinstance(raw, str)
    assert payload["ok"] is True
    assert payload["result"]["status"] == "unanimous"
    assert payload["result"]["decision"] == "yes"
    assert len(llm.calls) == 2


def test_consensus_handler_returns_honest_missing_hermes_error() -> None:
    handler = plugin.build_consensus_handler(lambda: MissingHermesLlm())

    payload = json.loads(handler({"question": "Question?", "rounds": 0, "agents": 2}))

    assert payload["ok"] is False
    assert payload["error"] == "hermes_not_found"
    assert "PATH" in payload["hint"]
