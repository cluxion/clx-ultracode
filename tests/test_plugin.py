"""Tests for the Hermes plugin registration shim."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from typing import Any

import pytest

from cluxion_effort_ultracode import plugin
from cluxion_effort_ultracode.adapters.codex_llm import CodexExecutableNotFoundError
from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError
from cluxion_effort_ultracode.core.journal import journals_dir


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

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append({"prompt": prompt, "schema": schema, "model": model})
        return self.outputs.popleft()


class MissingHermesLlm:
    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any]:
        del prompt, schema, model
        raise HermesExecutableNotFoundError("Hermes executable not found: 'hermes'")


class MissingCodexLlm:
    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any]:
        del prompt, schema, model
        raise CodexExecutableNotFoundError("Codex executable not found: 'codex'")


class NoCallLlm:
    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any]:
        del prompt, schema, model
        raise AssertionError("backend should not be called")


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

    assert len(ctx.tools) == 2
    registered = next(t for t in ctx.tools if t["name"] == "cluxion_consensus")
    assert registered["name"] == "cluxion_consensus"
    assert registered["toolset"] == "ultracode"
    assert registered["emoji"] == "🧠"
    assert callable(registered["handler"])

    schema = registered["schema"]
    assert schema["name"] == "cluxion_consensus"
    assert isinstance(schema["description"], str)
    assert "agents * (rounds + 1)" in schema["description"]
    assert schema["parameters"]["type"] == "object"
    assert {"required": ["question"]} in schema["parameters"]["anyOf"]
    assert {"required": ["resume"]} in schema["parameters"]["anyOf"]
    assert "question" in schema["parameters"]["properties"]
    assert "resume" in schema["parameters"]["properties"]
    assert "budget_tokens" in schema["parameters"]["properties"]
    assert "models" in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["adapter"]["enum"] == ["hermes", "codex"]


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
    assert payload["result"]["run_id"]
    assert payload["result"]["journal_path"]
    assert len(llm.calls) == 2


def test_consensus_handler_can_resume_completed_journal_without_question() -> None:
    llm = ScriptedLlm([position("yes"), position("YES.")])
    first = json.loads(plugin.build_consensus_handler(lambda: llm)({"question": "Should we answer yes?", "rounds": 0, "agents": 2}))
    replayed = json.loads(plugin.build_consensus_handler(lambda: NoCallLlm())({"resume": first["result"]["run_id"]}))

    assert replayed["ok"] is True
    assert replayed["result"]["status"] == first["result"]["status"]
    assert replayed["result"]["tokens_spent"] == 0
    assert replayed["result"]["tokens_replayed"] == first["result"]["tokens_spent"]


def test_consensus_handler_routes_models_and_rejects_empty_entries() -> None:
    llm = ScriptedLlm([position("yes"), position("YES.")])
    handler = plugin.build_consensus_handler(lambda: llm)

    payload = json.loads(handler({"question": "Should we answer yes?", "rounds": 0, "agents": 2, "models": ["a", "b"]}))

    assert payload["ok"] is True
    assert [call["model"] for call in llm.calls] == ["a", "b"]
    assert payload["result"]["transcript"][0]["positions"][0]["model"] == "a"

    bad = json.loads(handler({"question": "Q?", "models": ["a", ""]}))
    assert bad["ok"] is False
    assert bad["error"] == "invalid_models"
    assert "models entries" in bad["message"]


def test_consensus_handler_routes_codex_adapter_to_default_factory(monkeypatch) -> None:
    llm = ScriptedLlm([position("yes"), position("YES.")])
    calls: list[str] = []

    def _default_llm(adapter: str = "hermes", *, timeout_seconds: float | None = None):
        del timeout_seconds
        calls.append(adapter)
        return llm

    monkeypatch.setattr(plugin, "default_llm", _default_llm)
    payload = json.loads(
        plugin.build_consensus_handler()({"question": "Should we answer yes?", "rounds": 0, "agents": 2, "adapter": "codex"})
    )

    assert payload["ok"] is True
    assert calls == ["codex"]


@pytest.mark.parametrize("question", ["", " "])
def test_consensus_handler_rejects_empty_question(question: str) -> None:
    handler = plugin.build_consensus_handler(lambda: NoCallLlm())

    payload = json.loads(handler({"question": question}))

    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert "question" in payload["message"]


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"question": "Q?", "agents": 1}, "invalid_agents"),
        ({"question": "Q?", "rounds": 99}, "invalid_rounds"),
        ({"question": "Q?", "agent_timeout": 0}, "invalid_timeout"),
        ({"question": "Q?", "debate_budget": 0}, "invalid_budget"),
        ({"question": "Q?", "budget_tokens": 0}, "invalid_budget"),
    ],
)
def test_consensus_handler_validation_errors_use_semantic_codes(args: dict[str, object], code: str) -> None:
    handler = plugin.build_consensus_handler(lambda: NoCallLlm())

    payload = json.loads(handler(args))

    assert payload["ok"] is False
    assert payload["error"] == code


def test_consensus_handler_returns_honest_missing_hermes_error() -> None:
    handler = plugin.build_consensus_handler(lambda: MissingHermesLlm())

    payload = json.loads(handler({"question": "Question?", "rounds": 0, "agents": 2}))

    assert payload["ok"] is False
    assert payload["error"] == "hermes_not_found"
    assert "PATH" in payload["hint"]


def test_consensus_handler_returns_honest_missing_codex_error() -> None:
    handler = plugin.build_consensus_handler(lambda: MissingCodexLlm())

    payload = json.loads(handler({"question": "Question?", "rounds": 0, "agents": 2, "adapter": "codex"}))

    assert payload["ok"] is False
    assert payload["error"] == "codex_not_found"
    assert "PATH" in payload["hint"]


def test_consensus_handler_adapter_missing_leaves_no_journal() -> None:
    handler = plugin.build_consensus_handler(lambda: MissingCodexLlm())

    payload = json.loads(handler({"question": "Question?", "rounds": 0, "agents": 2, "adapter": "codex"}))

    assert payload["error"] == "codex_not_found"
    assert not journals_dir().exists()


def test_consensus_handler_returns_json_error_for_non_mapping_args() -> None:
    handler = plugin.build_consensus_handler(lambda: MissingHermesLlm())

    payload = json.loads(handler("not an object"))

    assert payload == {"ok": False, "error": "invalid_question", "message": "args must be an object"}


def test_consensus_handler_rejects_unknown_args() -> None:
    handler = plugin.build_consensus_handler(lambda: MissingHermesLlm())

    payload = json.loads(handler({"question": "Q?", "surprise": True}))

    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert payload["message"] == "unknown arguments: surprise"
