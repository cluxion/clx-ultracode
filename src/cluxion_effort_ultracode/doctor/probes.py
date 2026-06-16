"""Plugin-specific probes for effort-ultracode doctor. Cross-cutting only (no native)."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
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
    p = shutil.which(ctx.hermes_bin)
    if p:
        return "pass", str(p)
    return "fail", "not found on PATH"


@_register("hermes_binary_available")
def hermes_binary_available(ctx: DoctorContext) -> tuple[str, str]:
    binary = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", ctx.hermes_bin)
    p = shutil.which(binary)
    if p:
        return "pass", str(p)
    return "skip", "hermes binary not on PATH — cannot verify"


@_register("hermes_version")
def hermes_version(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run([ctx.hermes_bin, "--version"])
        if cp.returncode == 0 and "Hermes Agent v" in cp.stdout:
            return "pass", cp.stdout.strip()
        return "fail", cp.stdout.strip() or cp.stderr.strip()
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("hermes_oneshot_flag")
def hermes_oneshot_flag(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run([ctx.hermes_bin, "--help"])
        out = cp.stdout + cp.stderr
        if "-z" in out and "--oneshot" in out:
            return "pass", "present"
        return "fail", "missing in --help"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("hermes_z_flag_support")
def hermes_z_flag_support(ctx: DoctorContext) -> tuple[str, str]:
    binary = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", ctx.hermes_bin)
    if shutil.which(binary) is None:
        return "skip", "hermes binary not on PATH — cannot verify"
    try:
        cp = ctx.run([ctx.hermes_bin, "--help"])
        out = cp.stdout + cp.stderr
        if "-z" in out and "--oneshot" in out:
            return "pass", "present"
        return "fail", "missing in --help"
    except Exception as e:
        return "fail", f"run error: {e}"


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
        return "fail", "entry point not found or register missing"
    except Exception as e:
        return "fail", f"metadata error: {e}"


@_register("toolset_valid")
def toolset_valid(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cp = ctx.run([ctx.hermes_bin, "tools", "list"])
        if cp.returncode == 0 and "ultracode" in cp.stdout:
            return "pass", "ultracode present"
        return "fail", "ultracode not in tools list"
    except Exception as e:
        return "fail", f"run error: {e}"


@_register("install_integrity")
def install_integrity(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode import __version__ as pkg_version

        dist_version = importlib.metadata.version("cluxion-agentplugin-effort-ultracode")
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
        required = parameters.get("required")
        properties = parameters.get("properties")
        if not isinstance(required, list) or "question" not in required:
            return "fail", f"required={required!r}"
        if not isinstance(properties, dict) or "question" not in properties:
            return "fail", "question missing from properties"
        return "pass", "schema contract ok"
    except Exception as e:
        return "fail", f"schema error: {e}"


@_register("llm_factory_callable")
def llm_factory_callable(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.plugin import _call_llm_factory, _default_llm

        class _StubLlm:
            def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
                return {"stance": "Yes", "rationale": "stub", "evidence": ["e"], "confidence": 0.9}

        _call_llm_factory(lambda: _StubLlm())
        if not callable(_default_llm):
            return "fail", "_default_llm is not callable"
        default_llm = _default_llm()
        if not callable(getattr(default_llm, "complete", None)):
            return "fail", "_default_llm() missing complete()"
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
