"""Tests for the Hermes plugin registration shim."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from cluxion_effort_ultracode import plugin
from cluxion_effort_ultracode.adapters.codex_llm import CodexExecutableNotFoundError
from cluxion_effort_ultracode.adapters.hermes_llm import (
    BRIDGE_CLI_NAME,
    HermesExecutableNotFoundError,
    HermesHostLlm,
    HermesLlmError,
)
from cluxion_effort_ultracode.core.journal import journals_dir


class RecordingHermesContext:
    def __init__(self, *, with_llm: bool = False) -> None:
        self.tools: list[dict[str, object]] = []
        self.commands: list[dict[str, object]] = []
        self.cli_commands: list[dict[str, object]] = []
        if with_llm:
            self.llm = SimpleNamespace()

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

    def register_command(self, name: str, handler: object, **kwargs: object) -> None:
        self.commands.append({"name": name, "handler": handler, **kwargs})

    def register_cli_command(self, *, name: str, help: str, setup_fn: object, handler_fn: object) -> None:
        self.cli_commands.append({"name": name, "help": help, "setup_fn": setup_fn, "handler_fn": handler_fn})


def _factory(llm: object):
    def factory(adapter: str = "hermes", *, timeout_seconds: float | None = None):
        del adapter, timeout_seconds
        return llm

    return factory


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
    assert any(cmd["name"] == BRIDGE_CLI_NAME for cmd in ctx.cli_commands)
    assert any(cmd["name"] == "clx-consensus" for cmd in ctx.commands)
    assert not any(cmd["name"] == "cluxion-consensus" for cmd in ctx.commands)


def test_register_tolerates_host_without_register_tool() -> None:
    plugin.register(object())


def test_register_cli_without_register_tool() -> None:
    class CliOnly:
        def __init__(self) -> None:
            self.cli_commands: list[dict[str, object]] = []

        def register_cli_command(self, *, name: str, help: str, setup_fn: object, handler_fn: object) -> None:
            self.cli_commands.append({"name": name, "help": help, "setup_fn": setup_fn, "handler_fn": handler_fn})

    ctx = CliOnly()
    plugin.register(ctx)
    assert len(ctx.cli_commands) == 1
    assert ctx.cli_commands[0]["name"] == BRIDGE_CLI_NAME
    assert callable(ctx.cli_commands[0]["setup_fn"])
    assert callable(ctx.cli_commands[0]["handler_fn"])


def test_registered_tool_and_slash_use_host_llm_not_subprocess() -> None:
    class HostLlm:
        def complete(self, **kwargs):
            return SimpleNamespace(
                text=json.dumps(position("yes")),
                model="host",
                usage={"total_tokens": 1},
                provider="host",
            )

    ctx = RecordingHermesContext()
    ctx.llm = HostLlm()
    plugin.register(ctx)
    handler = next(t for t in ctx.tools if t["name"] == "cluxion_consensus")["handler"]
    with patch("cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen") as popen:
        payload = json.loads(handler({"question": "Should we answer yes?", "rounds": 0, "agents": 2}))
    assert payload["ok"] is True
    assert popen.call_count == 0

    slash = next(c for c in ctx.commands if c["name"] == "clx-consensus")["handler"]
    assert slash("") == "Usage: /clx-consensus <question|--resume run_id>"
    with patch("cluxion_effort_ultracode.adapters.subprocess_common.subprocess.Popen") as popen:
        raw = slash("Should we answer yes?")
    assert '"ok": true' in raw or '"ok":true' in raw.replace(" ", "")
    assert popen.call_count == 0


def test_call_llm_factory_uses_adapter_timeout_contract() -> None:
    seen: list[tuple[str, float]] = []

    def factory(adapter: str, *, timeout_seconds: float):
        seen.append((adapter, timeout_seconds))
        return ScriptedLlm([position("yes"), position("YES.")])

    llm = plugin._call_llm_factory(factory, adapter="hermes", timeout_seconds=33.0)
    assert isinstance(llm, ScriptedLlm)
    assert seen == [("hermes", 33.0)]


def test_call_llm_factory_preserves_legacy_zero_arg_contract() -> None:
    llm = ScriptedLlm([position("yes")])

    def factory():
        return llm

    assert plugin._call_llm_factory(factory, adapter="hermes", timeout_seconds=33.0) is llm


def test_call_llm_factory_does_not_retry_internal_type_error() -> None:
    calls = 0

    def factory(adapter: str, *, timeout_seconds: float):
        nonlocal calls
        calls += 1
        raise TypeError("factory implementation failed")

    with pytest.raises(ValueError, match="llm_factory raised TypeError"):
        plugin._call_llm_factory(factory, adapter="hermes", timeout_seconds=33.0)
    assert calls == 1


def test_host_factory_returns_hermes_host_llm() -> None:
    ctx = SimpleNamespace(llm=SimpleNamespace(complete=lambda **k: None))
    factory = plugin._host_llm_factory(ctx)
    llm = factory("hermes", timeout_seconds=12)
    assert isinstance(llm, HermesHostLlm)


def test_consensus_handler_returns_json_string_with_scripted_llm() -> None:
    llm = ScriptedLlm([position("yes"), position("YES.")])
    handler = plugin.build_consensus_handler(_factory(llm))

    raw = handler({"question": "Should we answer yes?", "rounds": 0, "agents": 2})
    payload = json.loads(raw)

    assert isinstance(raw, str)
    assert payload["ok"] is True
    assert payload["result"]["status"] == "unanimous"
    assert payload["result"]["decision"] == "yes"
    assert payload["result"]["run_id"]
    assert payload["result"]["journal_path"]
    assert len(llm.calls) == 2


def test_consensus_handler_error_includes_resumable_journal_info() -> None:
    handler = plugin.build_consensus_handler(_factory(ScriptedLlm([position("yes"), {}])))

    payload = json.loads(handler({"question": "Should we answer yes?", "rounds": 0, "agents": 2}))

    journal_path = journals_dir() / f"{payload['run_id']}.jsonl"
    assert payload["ok"] is False
    assert payload["error"] == "ConsensusProtocolError"
    assert payload["journal_path"] == str(journal_path)
    assert journal_path.exists()
    assert json.loads(journal_path.read_text(encoding="utf-8").splitlines()[0])["run_id"] == payload["run_id"]


def test_consensus_handler_can_resume_completed_journal_without_question() -> None:
    llm = ScriptedLlm([position("yes"), position("YES.")])
    first = json.loads(
        plugin.build_consensus_handler(_factory(llm))({"question": "Should we answer yes?", "rounds": 0, "agents": 2})
    )
    replayed = json.loads(plugin.build_consensus_handler(_factory(NoCallLlm()))({"resume": first["result"]["run_id"]}))

    assert replayed["ok"] is True
    assert replayed["result"]["status"] == first["result"]["status"]
    assert replayed["result"]["tokens_spent"] == 0
    assert replayed["result"]["tokens_replayed"] == first["result"]["tokens_spent"]


def test_consensus_handler_resume_returns_journal_busy_when_locked() -> None:
    import multiprocessing as mp

    from cluxion_effort_ultracode.core.journal import journals_dir
    from mp_helpers import hold_journal_until_release

    llm = ScriptedLlm([position("yes"), position("YES.")])
    first = json.loads(
        plugin.build_consensus_handler(_factory(llm))({"question": "Should we answer yes?", "rounds": 0, "agents": 2})
    )
    run_id = first["result"]["run_id"]
    home = journals_dir().parent

    ctx = mp.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Queue()

    proc = ctx.Process(target=hold_journal_until_release, args=(str(home), run_id, ready, release))
    proc.start()
    assert ready.get(timeout=10) == "ready"
    try:
        payload = json.loads(plugin.build_consensus_handler(_factory(NoCallLlm()))({"resume": run_id}))
        assert payload["ok"] is False
        assert payload["error"] == "journal_busy"
        assert payload["run_id"] == run_id
    finally:
        release.put("done")
        proc.join(timeout=10)
        assert proc.exitcode == 0


def test_consensus_handler_routes_models_and_rejects_empty_entries() -> None:
    llm = ScriptedLlm([position("yes"), position("YES.")])
    handler = plugin.build_consensus_handler(_factory(llm))

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
        plugin.build_consensus_handler()(
            {"question": "Should we answer yes?", "rounds": 0, "agents": 2, "adapter": "codex"}
        )
    )

    assert payload["ok"] is True
    assert calls == ["codex"]


def test_consensus_handler_returns_hermes_llm_error_code() -> None:
    class Bad:
        def complete(self, *a, **k):
            raise HermesLlmError("model_override_denied", "denied")

    payload = json.loads(plugin.build_consensus_handler(_factory(Bad()))({"question": "Q?", "rounds": 0, "agents": 2}))
    assert payload["ok"] is False
    assert payload["error"] == "model_override_denied"


@pytest.mark.parametrize("question", ["", " "])
def test_consensus_handler_rejects_empty_question(question: str) -> None:
    handler = plugin.build_consensus_handler(_factory(NoCallLlm()))

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
        ({"question": "Q?", "agent_timeout": -1}, "invalid_timeout"),
        ({"question": "Q?", "agent_timeout": float("nan")}, "invalid_timeout"),
        ({"question": "Q?", "agent_timeout": float("inf")}, "invalid_timeout"),
        ({"question": "Q?", "agent_timeout": float("-inf")}, "invalid_timeout"),
        ({"question": "Q?", "agent_timeout": 10**400}, "invalid_timeout"),
        ({"question": "Q?", "debate_budget": 0}, "invalid_budget"),
        ({"question": "Q?", "debate_budget": -1}, "invalid_budget"),
        ({"question": "Q?", "debate_budget": float("nan")}, "invalid_budget"),
        ({"question": "Q?", "debate_budget": float("inf")}, "invalid_budget"),
        ({"question": "Q?", "debate_budget": float("-inf")}, "invalid_budget"),
        ({"question": "Q?", "debate_budget": 10**400}, "invalid_budget"),
        ({"question": "Q?", "budget_tokens": 0}, "invalid_budget"),
    ],
)
def test_consensus_handler_validation_errors_use_semantic_codes(args: dict[str, object], code: str) -> None:
    handler = plugin.build_consensus_handler(_factory(NoCallLlm()))

    payload = json.loads(handler(args))

    assert payload["ok"] is False
    assert payload["error"] == code
    # timeout/budget bounds are rejected before journal creation
    if code in {"invalid_timeout", "invalid_budget"}:
        assert "run_id" not in payload
        assert not journals_dir().exists()


@pytest.mark.parametrize(
    "args",
    [
        {"question": "Q?", "rounds": float("inf")},
        {"question": "Q?", "agents": float("inf")},
        {"question": "Q?", "budget_tokens": float("inf")},
    ],
)
def test_plugin_int_args_map_overflow_to_integer_error(args: dict[str, object]) -> None:
    handler = plugin.build_consensus_handler(_factory(NoCallLlm()))
    payload = json.loads(handler(args))
    assert payload["ok"] is False
    assert "must be an integer" in payload["message"]
    assert "run_id" not in payload
    assert not journals_dir().exists()


@pytest.mark.parametrize("agents", [2.9, True, False])
def test_plugin_rejects_non_integral_or_bool_agents_before_journal(agents: object) -> None:
    handler = plugin.build_consensus_handler(_factory(NoCallLlm()))

    payload = json.loads(handler({"question": "Q?", "agents": agents, "rounds": 0}))

    assert payload["ok"] is False
    assert "must be an integer" in payload["message"]
    assert "run_id" not in payload
    assert "journal_path" not in payload
    assert not journals_dir().exists()


@pytest.mark.parametrize("budget_tokens", [2.9, True])
def test_plugin_rejects_non_integral_or_bool_budget_tokens_before_journal(budget_tokens: object) -> None:
    handler = plugin.build_consensus_handler(_factory(NoCallLlm()))

    payload = json.loads(handler({"question": "Q?", "agents": 2, "rounds": 0, "budget_tokens": budget_tokens}))

    assert payload["ok"] is False
    assert "must be an integer" in payload["message"]
    assert "run_id" not in payload
    assert "journal_path" not in payload
    assert not journals_dir().exists()


@pytest.mark.parametrize("agents", [2.0, "2"])
def test_plugin_int_args_accept_integral_float_and_decimal_string(agents: object) -> None:
    llm = ScriptedLlm([position("yes"), position("yes")])
    handler = plugin.build_consensus_handler(_factory(llm))

    payload = json.loads(handler({"question": "Should we answer yes?", "rounds": 0, "agents": agents}))

    assert payload["ok"] is True
    assert payload["result"]["status"] == "unanimous"
    assert payload["result"]["agents_count"] == 2
    assert len(llm.calls) == 2


def test_consensus_handler_rejects_surrogate_question_before_journal() -> None:
    handler = plugin.build_consensus_handler(_factory(NoCallLlm()))

    payload = json.loads(handler({"question": "Q?\udcff", "agents": 2, "rounds": 0}))

    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert "run_id" not in payload
    assert "journal_path" not in payload
    assert not journals_dir().exists()


def test_consensus_handler_rejects_surrogate_context_before_journal() -> None:
    handler = plugin.build_consensus_handler(_factory(NoCallLlm()))

    payload = json.loads(handler({"question": "Q?", "context": "ctx\udcff", "agents": 2, "rounds": 0}))

    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert "run_id" not in payload
    assert "journal_path" not in payload
    assert not journals_dir().exists()


def test_consensus_handler_returns_honest_missing_hermes_error() -> None:
    handler = plugin.build_consensus_handler(_factory(MissingHermesLlm()))

    payload = json.loads(handler({"question": "Question?", "rounds": 0, "agents": 2}))

    assert payload["ok"] is False
    assert payload["error"] == "hermes_not_found"
    assert payload["run_id"]
    assert payload["journal_path"].endswith(f"{payload['run_id']}.jsonl")
    assert "PATH" in payload["hint"]


def test_consensus_handler_returns_honest_missing_codex_error() -> None:
    handler = plugin.build_consensus_handler(_factory(MissingCodexLlm()))

    payload = json.loads(handler({"question": "Question?", "rounds": 0, "agents": 2, "adapter": "codex"}))

    assert payload["ok"] is False
    assert payload["error"] == "codex_not_found"
    assert payload["run_id"]
    assert payload["journal_path"].endswith(f"{payload['run_id']}.jsonl")
    assert "PATH" in payload["hint"]


def test_consensus_handler_adapter_missing_leaves_no_journal() -> None:
    handler = plugin.build_consensus_handler(_factory(MissingCodexLlm()))

    payload = json.loads(handler({"question": "Question?", "rounds": 0, "agents": 2, "adapter": "codex"}))

    assert payload["error"] == "codex_not_found"
    assert not journals_dir().exists()


def test_consensus_handler_returns_json_error_for_non_mapping_args() -> None:
    handler = plugin.build_consensus_handler(_factory(MissingHermesLlm()))

    payload = json.loads(handler("not an object"))

    assert payload == {"ok": False, "error": "invalid_question", "message": "args must be an object"}


def test_consensus_handler_rejects_unknown_args() -> None:
    handler = plugin.build_consensus_handler(_factory(MissingHermesLlm()))

    payload = json.loads(handler({"question": "Q?", "surprise": True}))

    assert payload["ok"] is False
    assert payload["error"] == "invalid_question"
    assert payload["message"] == "unknown arguments: surprise"
