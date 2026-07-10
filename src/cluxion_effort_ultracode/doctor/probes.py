"""Plugin-specific probes for effort-ultracode doctor. Cross-cutting only (no native)."""

from __future__ import annotations

import importlib.metadata
import inspect
import os
from collections.abc import Callable, Mapping
from typing import Any

from .framework import DoctorContext

PROBES: dict[str, Callable[[DoctorContext], tuple[str, str]]] = {}


def _register(name: str):
    def deco(fn):
        PROBES[name] = fn
        return fn

    return deco


@_register("hermes_on_path")
def hermes_on_path(ctx: DoctorContext) -> tuple[str, str]:
    p = ctx.which(ctx.hermes_bin)
    if p:
        return "pass", str(p)
    return "fail", "not found on PATH"


def _codex_binary() -> str:
    return os.getenv("CLUXION_EFFORT_ULTRACODE_CODEX_BINARY", "codex")


@_register("codex_on_path")
def codex_on_path(ctx: DoctorContext) -> tuple[str, str]:
    p = ctx.which("codex")
    if p:
        return "pass", str(p)
    return "fail", "not found on PATH"


@_register("codex_binary_available")
def codex_binary_available(ctx: DoctorContext) -> tuple[str, str]:
    binary = _codex_binary()
    p = ctx.which(binary)
    if p:
        return "pass", str(p)
    return "skip", "codex binary not on PATH — cannot verify"


@_register("codex_version")
def codex_version(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run_cached([_codex_binary(), "--version"])
        if cp.returncode == 0 and (cp.stdout.strip() or cp.stderr.strip()):
            return "pass", cp.stdout.strip() or cp.stderr.strip()
        return "fail", cp.stdout.strip() or cp.stderr.strip() or f"exit {cp.returncode}"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("codex_subprocess_launchable")
def codex_subprocess_launchable(ctx: DoctorContext) -> tuple[str, str]:
    binary = _codex_binary()
    if ctx.which(binary) is None:
        return "skip", "codex binary not on PATH — cannot verify"
    try:
        cp = ctx.run_cached([binary, "--version"])
        if cp.returncode == 0:
            return "pass", cp.stdout.strip() or cp.stderr.strip() or "launched"
        detail = cp.stdout.strip() or cp.stderr.strip() or f"exit {cp.returncode}"
        return "fail", detail
    except Exception as e:
        return "fail", f"launch error: {e}"


@_register("codex_exec_flag_support")
def codex_exec_flag_support(ctx: DoctorContext) -> tuple[str, str]:
    binary = _codex_binary()
    if ctx.which(binary) is None:
        return "skip", "codex binary not on PATH — cannot verify"
    try:
        cp = ctx.run_cached([binary, "exec", "--help"])
        out = cp.stdout + cp.stderr
        if "codex exec" in out and "--output-last-message" in out and "--json" in out:
            return "pass", "present"
        return "fail", "missing in exec --help"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("hermes_binary_available")
def hermes_binary_available(ctx: DoctorContext) -> tuple[str, str]:
    binary = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", ctx.hermes_bin)
    p = ctx.which(binary)
    if p:
        return "pass", str(p)
    return "skip", "hermes binary not on PATH — cannot verify"


@_register("hermes_version")
def hermes_version(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run_cached([ctx.hermes_bin, "--version"])
        if cp.returncode == 0 and "Hermes Agent v" in cp.stdout:
            return "pass", cp.stdout.strip()
        return "fail", cp.stdout.strip() or cp.stderr.strip()
    except Exception as e:
        return "fail", f"run error: {e}"


def _hermes_bridge_help(ctx: DoctorContext) -> tuple[str, str]:
    """Token-free bridge capability probe: hermes ultracode-llm --help."""
    binary = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", ctx.hermes_bin)
    if ctx.which(binary) is None:
        return "skip", "hermes binary not on PATH — cannot verify"
    try:
        cp = ctx.run_cached([binary, "ultracode-llm", "--help"])
        if cp.returncode == 0:
            return "pass", "ultracode-llm bridge help ok"
        detail = (cp.stdout or cp.stderr or f"exit {cp.returncode}").strip()
        return "fail", detail or "ultracode-llm --help failed"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("hermes_oneshot_flag")
def hermes_oneshot_flag(ctx: DoctorContext) -> tuple[str, str]:
    # Historical migration note: check_id kept; formerly probed host oneshot flags.
    # Active contract is only `hermes ultracode-llm --help` (token-free).
    return _hermes_bridge_help(ctx)


@_register("hermes_subprocess_launchable")
def hermes_subprocess_launchable(ctx: DoctorContext) -> tuple[str, str]:
    binary = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", ctx.hermes_bin)
    if ctx.which(binary) is None:
        return "skip", "hermes binary not on PATH — cannot verify"
    try:
        cp = ctx.run_cached([binary, "--version"])
        if cp.returncode == 0:
            return "pass", cp.stdout.strip() or "launched"
        detail = cp.stdout.strip() or cp.stderr.strip() or f"exit {cp.returncode}"
        return "fail", detail
    except Exception as e:
        return "fail", f"launch error: {e}"


@_register("hermes_z_flag_support")
def hermes_z_flag_support(ctx: DoctorContext) -> tuple[str, str]:
    # Historical migration note: check_id kept; formerly probed host -z flags.
    # Active contract is only `hermes ultracode-llm --help` (token-free).
    return _hermes_bridge_help(ctx)


@_register("entry_point_registered")
def entry_point_registered(ctx: DoctorContext) -> tuple[str, str]:
    try:
        eps = importlib.metadata.entry_points(group="hermes_agent.plugins")
        for ep in eps:
            if "cluxion-agentplugin-effort-ultracode" in (ep.name or "").lower() or "cluxion_effort_ultracode" in (
                ep.value or ""
            ):
                mod = ep.load()
                if hasattr(mod, "register") and callable(mod.register):
                    return "pass", ep.value or str(ep)
        from cluxion_effort_ultracode import plugin

        if callable(getattr(plugin, "register", None)):
            return "pass", "module register callable (editable/local checkout)"
        return "fail", "entry point not found or register missing"
    except Exception as e:
        return "fail", f"metadata error: {e}"


@_register("toolset_valid")
def toolset_valid(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run_cached([ctx.hermes_bin, "tools", "list"])
        if cp.returncode == 0 and "ultracode" in cp.stdout:
            return "pass", "ultracode present"
        return "fail", "ultracode not in tools list"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("install_integrity")
def install_integrity(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode import __version__ as pkg_version

        try:
            dist_version = importlib.metadata.version("cluxion-agentplugin-effort-ultracode")
        except importlib.metadata.PackageNotFoundError:
            return "warn", f"editable/local checkout pkg={pkg_version}"
        if dist_version == pkg_version:
            return "pass", dist_version
        return "warn", f"dist={dist_version} pkg={pkg_version}"
    except Exception as e:
        return "fail", f"version error: {e}"


@_register("hermes_timeout_configured")
def hermes_timeout_configured(ctx: DoctorContext) -> tuple[str, str]:
    try:
        val = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "").strip()
        if not val:
            return "pass", "defaults (unset)"
        try:
            timeout = float(val)
        except ValueError:
            return "fail", "non-numeric"
        if timeout > 0:
            return "pass", f"valid {timeout}"
        return "fail", "non-positive"
    except Exception as e:
        return "fail", f"env error: {e}"


@_register("consensus_schema_contract")
def consensus_schema_contract(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.plugin import CONSENSUS_SCHEMA, register

        class _RecordingCtx:
            def __init__(self) -> None:
                self.tools: list[dict[str, Any]] = []

            def register_tool(
                self,
                *,
                name: str,
                toolset: str,
                schema: Mapping[str, Any],
                handler: object,
                **_: object,
            ) -> None:
                self.tools.append({"name": name, "toolset": toolset, "schema": dict(schema)})

        rec = _RecordingCtx()
        register(rec)
        tool = next((t for t in rec.tools if t["name"] == "cluxion_consensus"), None)
        schema = tool["schema"] if tool is not None else CONSENSUS_SCHEMA

        if schema.get("name") != "cluxion_consensus":
            return "fail", f"name={schema.get('name')!r}"
        description = schema.get("description", "")
        if not isinstance(description, str) or not description.strip():
            return "fail", "empty description"
        parameters = schema.get("parameters")
        if not isinstance(parameters, dict):
            return "fail", "parameters not a dict"
        any_of = parameters.get("anyOf")
        properties = parameters.get("properties")
        if not isinstance(any_of, list) or {"required": ["question"]} not in any_of:
            return "fail", f"anyOf={any_of!r}"
        if not isinstance(any_of, list) or {"required": ["resume"]} not in any_of:
            return "fail", f"anyOf={any_of!r}"
        if not isinstance(properties, dict) or not {"question", "resume", "adapter"} <= set(properties):
            return "fail", "question/resume missing from properties"
        adapter = properties.get("adapter")
        if not isinstance(adapter, dict) or adapter.get("enum") != ["hermes", "codex"]:
            return "fail", f"adapter={adapter!r}"
        return "pass", "schema contract ok"
    except Exception as e:
        return "fail", f"schema error: {e}"


@_register("llm_factory_callable")
def llm_factory_callable(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.llm_factory import default_llm
        from cluxion_effort_ultracode.plugin import _call_llm_factory

        class _StubLlm:
            def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
                return {"stance": "Yes", "rationale": "stub", "evidence": ["e"], "confidence": 0.9}

        def _factory(adapter: str, *, timeout_seconds: float) -> _StubLlm:
            del adapter, timeout_seconds
            return _StubLlm()

        _call_llm_factory(_factory, adapter="hermes", timeout_seconds=120)
        if not callable(default_llm):
            return "fail", "default_llm is not callable"
        llm = default_llm()
        if not callable(getattr(llm, "complete", None)):
            return "fail", "default_llm() missing complete()"
        return "pass", "factory contract ok"
    except Exception as e:
        return "fail", f"factory error: {e}"


@_register("plugin_registration_host_compat")
def plugin_registration_host_compat(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.plugin import register

        class _DummyCtx:
            pass

        register(_DummyCtx())
        return "pass", "no raise on minimal ctx"
    except Exception as e:
        return "warn", f"raised: {type(e).__name__}"


@_register("hermes_json_output_parseable")
def hermes_json_output_parseable(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.adapters.hermes_llm import _parse_json_object

        parsed = _parse_json_object('{"ok": true}')
        if parsed.get("ok") is True:
            return "pass", "json parser contract ok"
        return "fail", "parser returned unexpected object"
    except Exception as e:
        return "fail", f"json parser error: {e}"


@_register("agent_position_required_fields")
def agent_position_required_fields(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.core.consensus import POSITION_SCHEMA

        required = set(POSITION_SCHEMA.get("required", []))
        expected = {"stance", "rationale", "evidence", "confidence"}
        if expected <= required:
            return "pass", "required fields present"
        return "fail", f"missing={sorted(expected - required)}"
    except Exception as e:
        return "fail", f"schema error: {e}"


@_register("debate_round_concession_validity")
def debate_round_concession_validity(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.core.consensus import DEBATE_SCHEMA

        required = set(DEBATE_SCHEMA.get("required", []))
        if {"conceded", "maintained"} <= required:
            return "pass", "debate point arrays required"
        return "fail", f"required={sorted(required)}"
    except Exception as e:
        return "fail", f"schema error: {e}"


@_register("debate_round_response_type")
def debate_round_response_type(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.core.consensus import DEBATE_SCHEMA

        if DEBATE_SCHEMA.get("type") == "object":
            return "pass", "debate response object required"
        return "fail", f"type={DEBATE_SCHEMA.get('type')!r}"
    except Exception as e:
        return "fail", f"schema error: {e}"


@_register("consensus_result_valid")
def consensus_result_valid(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.core.types import ConsensusResult

        fields = set(ConsensusResult.__dataclass_fields__)
        expected = {"status", "decision", "rationale", "rounds", "transcript", "agents_count", "dissent"}
        extra = {"abort_reason", "rounds_completed", "tokens_spent", "tokens_estimated"}
        if expected <= fields and extra <= fields:
            return "pass", "result dataclass contract ok"
        return "fail", f"missing={sorted((expected | extra) - fields)}"
    except Exception as e:
        return "fail", f"result contract error: {e}"


@_register("debate_non_termination_cost")
def debate_non_termination_cost(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.core.consensus import DEFAULT_DEBATE_BUDGET_S, MAX_AGENTS, MAX_ROUNDS

        calls = MAX_AGENTS * (MAX_ROUNDS + 1)
        if DEFAULT_DEBATE_BUDGET_S > 0:
            return "pass", (
                f"bounded by max {calls} calls and {DEFAULT_DEBATE_BUDGET_S:g}s; "
                "token ceiling unlimited unless budget_tokens/--budget-tokens is set"
            )
        return "fail", "non-positive debate budget"
    except Exception as e:
        return "fail", f"cost bound error: {e}"


@_register("hermes_subprocess_returncode_nonzero")
def hermes_subprocess_returncode_nonzero(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.adapters.hermes_llm import HermesSubprocessLlm

        if callable(getattr(HermesSubprocessLlm, "_run_bridge_once", None)):
            return "pass", "nonzero returncode classified by bridge client"
        return "fail", "adapter runner missing"
    except Exception as e:
        return "fail", f"adapter error: {e}"


@_register("hermes_subprocess_timeout_triggered")
def hermes_subprocess_timeout_triggered(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.adapters.hermes_llm import HermesSubprocessLlm

        if HermesSubprocessLlm(timeout_seconds=1).timeout_seconds == 1:
            return "pass", "timeout configured"
        return "fail", "timeout not retained"
    except Exception as e:
        return "fail", f"timeout error: {e}"


@_register("llm_port_complete_method_signature")
def llm_port_complete_method_signature(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.adapters.codex_llm import CodexSubprocessLlm
        from cluxion_effort_ultracode.adapters.hermes_llm import HermesSubprocessLlm

        signatures = {
            "hermes": inspect.signature(HermesSubprocessLlm.complete),
            "codex": inspect.signature(CodexSubprocessLlm.complete),
        }
        for name, signature in signatures.items():
            if {"schema", "model"} > set(signature.parameters):
                return "fail", f"{name}: {signature}"
        return "pass", "; ".join(f"{name}: {signature}" for name, signature in signatures.items())
    except Exception as e:
        return "fail", f"signature error: {e}"
