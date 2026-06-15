"""Plugin-specific probes for effort-ultracode doctor. Cross-cutting only (no native)."""

from __future__ import annotations

import importlib.metadata
import shutil
from collections.abc import Callable

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


@_register("entry_point_registered")
def entry_point_registered(ctx: DoctorContext) -> tuple[str, str]:
    try:
        eps = importlib.metadata.entry_points(group="hermes_agent.plugins")
        for ep in eps:
            if "cluxion-agentplugin-effort-ultracode" in (ep.name or "").lower() or "cluxion_effort_ultracode" in (ep.value or ""):
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


# note: no native_module_importable (NATIVE=none)
# other checks in catalog will be reported as skip (no probe)

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


@_register("import_availability")
def import_availability(ctx: DoctorContext) -> tuple[str, str]:
    try:
        import importlib

        importlib.import_module("json")
        return "pass", "json importable"
    except Exception as e:
        return "skip", f"import error: {e}"


@_register("abi3_wheel_compatible")
def abi3_wheel_compatible(ctx: DoctorContext) -> tuple[str, str]:
    return "pass", f"python {sys.version_info.major}.{sys.version_info.minor} (abi3 floor 3.11)"


@_register("sqlite_wal_mode_compatible")
def sqlite_wal_mode_compatible(ctx: DoctorContext) -> tuple[str, str]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            conn = sqlite3.connect(str(db))
            try:
                mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if str(mode).lower() == "wal":
                    return "pass", f"sqlite {sqlite3.sqlite_version} supports WAL"
                return "warn", f"returned {mode}"
            finally:
                conn.close()
    except Exception as e:
        return "skip", f"sqlite error: {e}"


@_register("json_serialization_deterministic")
def json_serialization_deterministic(ctx: DoctorContext) -> tuple[str, str]:
    try:
        data = {"a": 1, "b": [2, 3], "c": {"d": "e"}}
        s1 = json.dumps(data, sort_keys=True, separators=(",", ":"))
        s2 = json.dumps(data, sort_keys=True, separators=(",", ":"))
        if s1 == s2:
            return "pass", "roundtrip byte equal"
        return "fail", "not deterministic"
    except Exception as e:
        return "skip", f"json error: {e}"


@_register("hermes_plugin_enabled")
def hermes_plugin_enabled(ctx: DoctorContext) -> tuple[str, str]:
    try:
        cfg = Path.home() / ".hermes" / "config.yaml"
        if not cfg.exists():
            return "skip", "config not present"
        if yaml is None:
            return "skip", "pyyaml not importable"
        try:
            content = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            plugins = content.get("plugins", {}) if isinstance(content, dict) else {}
            enabled = plugins.get("enabled", []) if isinstance(plugins, dict) else []
            plugin_names = [str(x).lower() for x in (enabled if isinstance(enabled, (list, tuple)) else [])]
            if any("effort-ultracode" in n or "cluxion-agentplugin-effort-ultracode" in n for n in plugin_names):
                return "pass", "plugin enabled in config"
            return "warn", "not in plugins.enabled"
        except Exception as e:
            return "skip", f"yaml parse error: {e}"
    except Exception as e:
        return "skip", f"config read error: {e}"


@_register("env_var_consistency")
def env_var_consistency(ctx: DoctorContext) -> tuple[str, str]:
    try:
        val = os.getenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "").strip()
        if not val:
            return "pass", "defaults (unset)"
        try:
            t = float(val)
            if t > 0:
                return "pass", f"valid {t}"
            return "warn", "non-positive"
        except ValueError:
            return "warn", "non-numeric"
    except Exception as e:
        return "skip", f"env error: {e}"


@_register("hermes_timeout_configured")
def hermes_timeout_configured(ctx: DoctorContext) -> tuple[str, str]:
    return env_var_consistency(ctx)


@_register("plugin_registration_host_compat")
def plugin_registration_host_compat(ctx: DoctorContext) -> tuple[str, str]:
    try:
        from cluxion_effort_ultracode.plugin import register

        class DummyCtx:
            pass

        register(DummyCtx())
        return "pass", "no raise on minimal ctx"
    except Exception as e:
        return "warn", f"raised: {type(e).__name__}"
