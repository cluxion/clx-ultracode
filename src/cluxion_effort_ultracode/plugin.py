"""Hermes plugin shim for exposing the cluxion_consensus tool."""

from __future__ import annotations

import importlib.resources
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.adapters.hermes_llm import HermesExecutableNotFoundError, HermesSubprocessLlm
from cluxion_effort_ultracode.core import ConsensusEngine, ConsensusProtocolError
from cluxion_effort_ultracode.core.ports import LlmPort
from cluxion_effort_ultracode.doctor import render_json, run_doctor

CONSENSUS_SCHEMA: dict[str, Any] = {
    "name": "cluxion_consensus",
    "description": (
        "Run an opt-in deep-deliberation consensus debate through Hermes oneshot. "
        "Worst-case cost is agents * (rounds + 1) hermes -z model calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Decision, proposal, or question that the agents must settle.",
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
            },
            "agents": {
                "type": "integer",
                "description": "Number of independent agents in the debate.",
                "default": 3,
                "minimum": 2,
            },
        },
        "required": ["question"],
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
        question = _text_arg(args, "question", required=True)
        context = _text_arg(args, "context", default="")
        rounds = _int_arg(args, "rounds", default=3)
        agents = _int_arg(args, "agents", default=3)
        llm = _call_llm_factory(llm_factory)
        result = ConsensusEngine(llm, agents_count=agents, max_rounds=rounds).decide(question, context=context)
    except HermesExecutableNotFoundError as exc:
        return {
            "ok": False,
            "error": "hermes_not_found",
            "message": str(exc),
            "hint": ("Ensure the hermes executable is on PATH, or configure CLUXION_EFFORT_ULTRACODE_HERMES_BINARY."),
        }
    except (ConsensusProtocolError, ValueError) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    return {"ok": True, "result": asdict(result)}


def _default_llm() -> HermesSubprocessLlm:
    binary = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", "hermes")
    model = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_MODEL") or None
    timeout = _timeout_from_env()
    return HermesSubprocessLlm(binary=binary, timeout_seconds=timeout, model=model)


def _timeout_from_env() -> float:
    raw = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "").strip()
    if not raw:
        return 120.0
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT must be numeric") from exc
    if timeout <= 0:
        raise ValueError("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT must be greater than zero")
    return timeout


def _call_llm_factory(llm_factory: object) -> LlmPort:
    if not callable(llm_factory):
        raise ValueError("llm_factory must be callable")
    llm = llm_factory()
    complete = getattr(llm, "complete", None)
    if not callable(complete):
        raise ValueError("llm_factory must return an object with complete(...)")
    return llm


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


def _handle_doctor(args: dict[str, object], **_: object) -> str:
    pkg = "cluxion_effort_ultracode.doctor"
    catalog_path = Path(str(importlib.resources.files(pkg).joinpath("catalog.json")))
    result = run_doctor(
        cwd=Path.cwd(),
        hermes_bin="hermes",
        catalog_path=catalog_path,
        probes=__import__(
            "cluxion_effort_ultracode.doctor.probes", fromlist=["PROBES"]
        ).PROBES,
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
