"""Hermes plugin shim for exposing the cluxion_consensus tool."""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.adapters.codex_llm import CodexExecutableNotFoundError
from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError
from cluxion_effort_ultracode.core import ConsensusEngine, ConsensusProtocolError
from cluxion_effort_ultracode.core.consensus import (
    DEFAULT_AGENT_TIMEOUT_S,
    DEFAULT_DEBATE_BUDGET_S,
    MAX_AGENTS,
    MAX_ROUNDS,
)
from cluxion_effort_ultracode.core.journal import (
    DebateJournal,
    JournaledLlm,
    LazyLlm,
    ResumeMismatch,
    ResumeNotFound,
    build_header,
    journal_header,
    new_run_id,
)
from cluxion_effort_ultracode.core.ports import LlmPort
from cluxion_effort_ultracode.doctor import render_json, run_doctor
from cluxion_effort_ultracode.llm_factory import default_llm

CONSENSUS_ARG_KEYS = {
    "question",
    "resume",
    "context",
    "rounds",
    "agents",
    "agent_timeout",
    "debate_budget",
    "budget_tokens",
    "models",
    "adapter",
}

CONSENSUS_SCHEMA: dict[str, Any] = {
    "name": "cluxion_consensus",
    "description": (
        "Run an opt-in deep-deliberation consensus debate through Hermes or Codex. "
        "Worst-case cost is agents * (rounds + 1) model calls, plus tracked tokens."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Decision, proposal, or question that the agents must settle.",
            },
            "resume": {
                "type": "string",
                "description": "Existing journal run_id to replay or continue.",
            },
            "context": {
                "type": "string",
                "description": "Optional shared context shown to every agent.",
                "default": "",
            },
            "rounds": {
                "type": "integer",
                "description": "Maximum debate rounds after independent round 0.",
                "default": 3,
                "minimum": 0,
                "maximum": MAX_ROUNDS,
            },
            "agents": {
                "type": "integer",
                "description": "Number of independent agents in the debate.",
                "default": 3,
                "minimum": 2,
                "maximum": MAX_AGENTS,
            },
            "agent_timeout": {
                "type": "number",
                "description": "Per-agent timeout in seconds.",
                "default": DEFAULT_AGENT_TIMEOUT_S,
                "exclusiveMinimum": 0,
            },
            "debate_budget": {
                "type": "number",
                "description": "Total debate budget in seconds across all rounds.",
                "default": DEFAULT_DEBATE_BUDGET_S,
                "exclusiveMinimum": 0,
            },
            "budget_tokens": {
                "type": "integer",
                "description": "Optional total token ceiling. Omit for unlimited.",
                "exclusiveMinimum": 0,
            },
            "models": {
                "type": "array",
                "description": "Optional per-agent model list, cycled across agent seats.",
                "items": {"type": "string"},
                "default": [],
            },
            "adapter": {
                "type": "string",
                "description": "Real LLM adapter. Hermes remains the default; Codex is recommended on Codex hosts.",
                "enum": ["hermes", "codex"],
                "default": "hermes",
            },
        },
        "anyOf": [{"required": ["question"]}, {"required": ["resume"]}],
        "required": [],
        "additionalProperties": False,
    },
}


def register(ctx: object) -> None:
    """Register the cluxion_consensus tool with a Hermes-like host context."""

    register_tool = getattr(ctx, "register_tool", None)
    if not callable(register_tool):
        return

    register_tool(
        name="cluxion_consensus",
        toolset="ultracode",
        schema=CONSENSUS_SCHEMA,
        handler=build_consensus_handler(),
        emoji="🧠",
    )
    # doctor tool
    DOCTOR_SCHEMA = {
        "name": "ultracode_doctor",
        "description": "Run the embedded deterministic health checks for this plugin",
        "parameters": {
            "type": "object",
            "properties": {"verbose": {"type": "boolean"}},
            "additionalProperties": False,
        },
    }
    register_tool(
        name="ultracode_doctor",
        toolset="ultracode",
        schema=DOCTOR_SCHEMA,
        handler=lambda args, **kw: _json_result(lambda: _handle_doctor(args, **kw)),
        emoji="🩺",
    )

    register_command = getattr(ctx, "register_command", None)
    if callable(register_command):

        def _slash_cluxion_consensus(raw_args: str) -> str:
            question = raw_args.strip()
            if not question:
                return "Usage: /cluxion-consensus <question|--resume run_id>"
            args = {"resume": question.split(maxsplit=1)[1]} if question.startswith("--resume ") else {"question": question}
            payload = _handle_consensus(args, llm_factory=_default_llm)
            return json.dumps(payload, ensure_ascii=False, indent=2)

        def _slash_ultracode_doctor(raw_args: str) -> str:
            del raw_args
            return _handle_doctor({})

        register_command(
            "cluxion-consensus",
            _slash_cluxion_consensus,
            description="Run 3-agent adversarial consensus debate (ultracode)",
            args_hint="<question>",
        )
        register_command(
            "ultracode-doctor",
            _slash_ultracode_doctor,
            description="Run ultracode plugin doctor checks",
        )


def build_consensus_handler(llm_factory: object | None = None):
    """Build the Hermes registry handler for cluxion_consensus."""

    def handler(args: object, **_: object) -> str:
        payload = _handle_consensus(args, llm_factory=llm_factory or _default_llm)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return handler


def _handle_consensus(args: object, *, llm_factory: object) -> dict[str, object]:
    if not isinstance(args, Mapping):
        return {"ok": False, "error": "ValueError", "message": "args must be an object"}

    try:
        _reject_unknown_args(args)
        resume = _text_arg(args, "resume", default="")
        saved = journal_header(resume) if resume else None
        question = _text_arg(args, "question", default=str((saved or {}).get("question", "")))
        if not question:
            raise ValueError("question is required unless resume points to a journal")
        context = _text_arg(args, "context", default=str((saved or {}).get("context", "")))
        rounds = _int_arg(args, "rounds", default=int((saved or {}).get("max_rounds", 3)))
        agents = _int_arg(args, "agents", default=int((saved or {}).get("agents_count", 3)))
        agent_timeout = _float_arg(
            args,
            "agent_timeout",
            default=float((saved or {}).get("agent_timeout_s", DEFAULT_AGENT_TIMEOUT_S)),
        )
        debate_budget = _float_arg(
            args,
            "debate_budget",
            default=float((saved or {}).get("debate_budget_s", DEFAULT_DEBATE_BUDGET_S)),
        )
        budget_tokens = _optional_int_arg(args, "budget_tokens", default=(saved or {}).get("budget_tokens"))
        models = _models_arg(args, "models") if "models" in args else _saved_models(saved)
        adapter = _adapter_arg(args, saved)
        run_id = resume or new_run_id()
        header = build_header(
            run_id=run_id,
            question=question,
            context=context,
            agents_count=agents,
            max_rounds=rounds,
            models=models or [],
            adapter=adapter,
            agent_timeout_s=agent_timeout,
            debate_budget_s=debate_budget,
            budget_tokens=budget_tokens,
        )
        journal = DebateJournal.resume(run_id, header) if resume else DebateJournal.start(header)
        llm = JournaledLlm(
            LazyLlm(lambda: _call_llm_factory(llm_factory, adapter=adapter, timeout_seconds=agent_timeout)),
            journal,
        )
        result = ConsensusEngine(
            llm,
            agents_count=agents,
            max_rounds=rounds,
            agent_timeout_s=agent_timeout,
            debate_budget_s=debate_budget,
            budget_tokens=budget_tokens,
            models=models,
        ).decide(question, context=context)
        journal.append_result(result)
    except ResumeMismatch as exc:
        return {"ok": False, "error": "resume_mismatch", "fields": exc.fields}
    except ResumeNotFound as exc:
        return {"ok": False, "error": "resume_not_found", "run_id": str(exc)}
    except HermesExecutableNotFoundError as exc:
        return {
            "ok": False,
            "error": "hermes_not_found",
            "message": str(exc),
            "hint": ("Ensure the hermes executable is on PATH, or configure CLUXION_EFFORT_ULTRACODE_HERMES_BINARY."),
        }
    except CodexExecutableNotFoundError as exc:
        return {
            "ok": False,
            "error": "codex_not_found",
            "message": str(exc),
            "hint": ("Ensure the codex executable is on PATH, or configure CLUXION_EFFORT_ULTRACODE_CODEX_BINARY."),
        }
    except (ConsensusProtocolError, ValueError) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    return {"ok": True, "result": asdict(result)}


def _default_llm(adapter: str = "hermes", *, timeout_seconds: float | None = None) -> LlmPort:
    return default_llm(adapter, timeout_seconds=timeout_seconds)


def _call_llm_factory(llm_factory: object, *, adapter: str, timeout_seconds: float) -> LlmPort:
    if not callable(llm_factory):
        raise ValueError("llm_factory must be callable")
    llm = llm_factory(adapter, timeout_seconds=timeout_seconds) if llm_factory is _default_llm else llm_factory()
    complete = getattr(llm, "complete", None)
    if not callable(complete):
        raise ValueError("llm_factory must return an object with complete(...)")
    return llm


def _reject_unknown_args(args: Mapping[str, object]) -> None:
    unknown = sorted(set(args) - CONSENSUS_ARG_KEYS)
    if unknown:
        raise ValueError(f"unknown arguments: {', '.join(unknown)}")


def _text_arg(
    args: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    value = args.get(key, default)
    if required and value is None:
        raise ValueError(f"{key} is required")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{key} is required")
    return value


def _int_arg(args: Mapping[str, object], key: str, *, default: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _float_arg(args: Mapping[str, object], key: str, *, default: float) -> float:
    value = args.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc


def _optional_int_arg(args: Mapping[str, object], key: str, *, default: object = None) -> int | None:
    value = default if key not in args or args[key] is None else args[key]
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _models_arg(args: Mapping[str, object], key: str) -> list[str] | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        models = [item.strip() for item in value.split(",")] if value.strip() else []
    elif isinstance(value, list):
        models = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"{key}[{index}] must be a string")
            models.append(item.strip())
    else:
        raise ValueError(f"{key} must be a list of strings")
    if any(not model for model in models):
        raise ValueError("models entries must be non-empty")
    return models or None


def _adapter_arg(args: Mapping[str, object], saved: Mapping[str, object] | None) -> str:
    adapter = _text_arg(args, "adapter", default=str((saved or {}).get("adapter", "hermes")))
    if adapter not in {"hermes", "codex"}:
        raise ValueError("adapter must be one of: hermes, codex")
    return adapter


def _saved_models(saved: Mapping[str, object] | None) -> list[str] | None:
    if saved is None:
        return None
    models = saved.get("models", [])
    return [str(model) for model in models] if isinstance(models, list) and models else None


def _handle_doctor(args: dict[str, object], **_: object) -> str:
    pkg = "cluxion_effort_ultracode.doctor"
    catalog_path = Path(str(importlib.resources.files(pkg).joinpath("catalog.json")))
    result = run_doctor(
        cwd=Path.cwd(),
        hermes_bin="hermes",
        catalog_path=catalog_path,
        probes=__import__("cluxion_effort_ultracode.doctor.probes", fromlist=["PROBES"]).PROBES,
        plugin="effort-ultracode",
        version=__import__("cluxion_effort_ultracode").__version__,
    )
    return render_json(result)


def _json_result(callback):
    try:
        return callback()
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True)


__all__ = ["CONSENSUS_SCHEMA", "build_consensus_handler", "register"]
