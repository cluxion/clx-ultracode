"""Tests for embedded doctor (determinism + cross-cutting checks)."""

import json
import subprocess
import time
from pathlib import Path

from cluxion_effort_ultracode.doctor import (
    DoctorResult,
    render_json,
    run_doctor,
)
from cluxion_effort_ultracode.doctor.framework import DoctorContext
from cluxion_effort_ultracode.doctor.probes import PROBES


def _catalog_path() -> Path:
    import importlib.resources

    pkg = "cluxion_effort_ultracode.doctor"
    return Path(str(importlib.resources.files(pkg).joinpath("catalog.json")))


def _doctor_ctx() -> DoctorContext:
    def _dummy_run(cmd):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return DoctorContext(Path.cwd(), "hermes", _dummy_run)


def test_run_doctor_returns_result_and_deterministic():
    cat = _catalog_path()
    r1 = run_doctor(
        cwd=Path.cwd(),
        catalog_path=cat,
        probes=PROBES,
        plugin="effort-ultracode",
        version="0.1.4",
    )
    assert isinstance(r1, DoctorResult)
    j1 = render_json(r1)
    r2 = run_doctor(
        cwd=Path.cwd(),
        catalog_path=cat,
        probes=PROBES,
        plugin="effort-ultracode",
        version="0.1.4",
    )
    j2 = render_json(r2)
    assert j1 == j2  # byte identical
    ids = [c.check_id for c in r1.checks]
    assert len(ids) > 0


def test_every_catalog_check_has_registered_probe():
    catalog_ids = {entry["check_id"] for entry in json.loads(_catalog_path().read_text())}
    assert catalog_ids <= set(PROBES)


def test_cross_cutting_checks_present():
    cat = _catalog_path()
    result = run_doctor(
        cwd=Path.cwd(),
        catalog_path=cat,
        probes=PROBES,
        plugin="effort-ultracode",
        version="0.1.4",
    )
    statuses = {c.check_id: c.status for c in result.checks}
    for key in ("hermes_on_path", "codex_on_path", "entry_point_registered", "toolset_valid"):
        assert key in statuses
        assert statuses[key] in ("pass", "warn", "fail", "skip")


def test_probe_exception_becomes_fail():
    def bad_probe(ctx):
        raise RuntimeError("boom")

    result = run_doctor(
        cwd=Path.cwd(),
        catalog_path=_catalog_path(),
        probes={"hermes_on_path": bad_probe},
        plugin="effort-ultracode",
        version="0.1.4",
    )
    statuses = {c.check_id: c.status for c in result.checks}
    assert statuses["hermes_on_path"] == "fail"


def test_warn_only_is_ok():
    from cluxion_effort_ultracode.doctor.framework import CheckResult, DoctorResult

    checks = (CheckResult(check_id="x", category="c", severity="medium", status="warn", detail="w"),)
    r = DoctorResult(plugin="p", version="0.1.4", checks=checks)
    assert r.ok is True
    assert r.summary == "ok"


def test_critical_skip_does_not_degrade_summary(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: None,
    )
    # Exclude probes that FAIL (not skip) when hermes is absent; we test probe-level SKIP here.
    hermes_absent_fail_probes = {
        "codex_on_path",
        "codex_version",
        "hermes_on_path",
        "hermes_version",
        "hermes_oneshot_flag",
        "toolset_valid",
    }
    probes = {k: v for k, v in PROBES.items() if k not in hermes_absent_fail_probes}
    cat = _catalog_path()
    result = run_doctor(
        cwd=Path.cwd(),
        catalog_path=cat,
        probes=probes,
        plugin="effort-ultracode",
        version="0.1.4",
    )
    statuses = {c.check_id: c.status for c in result.checks}
    assert statuses["hermes_binary_available"] == "skip"
    assert statuses["hermes_subprocess_launchable"] == "skip"
    assert result.summary == "ok"
    assert result.ok is True
    payload = json.loads(render_json(result))
    assert payload["summary"] == "ok"
    assert payload["ok"] is True


def test_hermes_binary_available_passes_when_present(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/hermes",
    )
    status, detail = PROBES["hermes_binary_available"](_doctor_ctx())
    assert status == "pass"
    assert detail == "/usr/local/bin/hermes"


def test_hermes_static_critical_probes_registered():
    for name in ("hermes_binary_available", "hermes_subprocess_launchable", "hermes_z_flag_support"):
        assert name in PROBES


def test_codex_static_critical_probes_registered():
    for name in ("codex_binary_available", "codex_subprocess_launchable", "codex_exec_flag_support"):
        assert name in PROBES


def test_hermes_z_flag_support_parses_help(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/hermes",
    )

    def _help_run(cmd):
        assert cmd[-2:] == ["ultracode-llm", "--help"] or cmd[1:] == ["ultracode-llm", "--help"]
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="usage: hermes ultracode-llm",
            stderr="",
        )

    ctx = DoctorContext(Path.cwd(), "hermes", _help_run)
    status, detail = PROBES["hermes_z_flag_support"](ctx)
    assert status == "pass"
    assert "ultracode-llm" in detail


def test_codex_exec_flag_support_parses_help(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/codex",
    )

    def _help_run(cmd):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Usage: codex exec [OPTIONS] [PROMPT]\n      --output-last-message <FILE>\n      --json",
            stderr="",
        )

    ctx = DoctorContext(Path.cwd(), "hermes", _help_run)
    status, detail = PROBES["codex_exec_flag_support"](ctx)
    assert status == "pass"
    assert detail == "present"


def test_hermes_z_flag_support_skips_when_absent(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: None,
    )
    status, detail = PROBES["hermes_z_flag_support"](_doctor_ctx())
    assert status == "skip"
    assert "cannot verify" in detail


def test_codex_exec_flag_support_skips_when_absent(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: None,
    )
    status, detail = PROBES["codex_exec_flag_support"](_doctor_ctx())
    assert status == "skip"
    assert "cannot verify" in detail


def test_static_probes_do_not_skip(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/hermes",
    )
    ctx = _doctor_ctx()
    static_probes = (
        "consensus_schema_contract",
        "codex_binary_available",
        "hermes_binary_available",
        "hermes_timeout_configured",
        "llm_factory_callable",
        "plugin_registration_host_compat",
    )
    for name in static_probes:
        assert name in PROBES
        status, _ = PROBES[name](ctx)
        assert status != "skip", f"{name} should not skip"


def test_consensus_schema_contract_detects_missing_question_route(monkeypatch):
    from cluxion_effort_ultracode import plugin

    broken = dict(plugin.CONSENSUS_SCHEMA)
    params = dict(broken["parameters"])
    params["anyOf"] = [{"required": ["resume"]}]
    broken["parameters"] = params
    monkeypatch.setattr(plugin, "CONSENSUS_SCHEMA", broken)

    status, detail = PROBES["consensus_schema_contract"](_doctor_ctx())
    assert status == "fail"
    assert "anyOf" in detail


def test_hermes_timeout_configured_rejects_invalid(monkeypatch):
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "not-a-number")
    status, detail = PROBES["hermes_timeout_configured"](_doctor_ctx())
    assert status == "fail"
    assert "non-numeric" in detail


def test_codex_binary_available_passes_when_present(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/codex",
    )
    status, detail = PROBES["codex_binary_available"](_doctor_ctx())
    assert status == "pass"
    assert detail == "/usr/local/bin/codex"


def test_debate_non_termination_cost_mentions_token_ceiling():
    status, detail = PROBES["debate_non_termination_cost"](_doctor_ctx())
    assert status == "pass"
    assert "token ceiling" in detail


def test_dead_probes_removed():
    for dead in ("abi3_wheel_compatible", "sqlite_wal_mode_compatible", "import_availability"):
        assert dead not in PROBES


def _mock_healthy_hermes_run(cmd):
    if cmd[0] == "codex" and "--version" in cmd:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="codex 1.2.3", stderr="")
    if cmd[0] == "codex" and "--help" in cmd:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Usage: codex exec [OPTIONS] [PROMPT]\n      --output-last-message <FILE>\n      --json",
            stderr="",
        )
    if "--version" in cmd:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Hermes Agent v0.1.7",
            stderr="",
        )
    if len(cmd) >= 3 and cmd[1] == "ultracode-llm" and cmd[2] == "--help":
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="usage: hermes ultracode-llm\nHidden-purpose cluxion ultracode LLM bridge",
            stderr="",
        )
    if "--help" in cmd:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="usage: hermes",
            stderr="",
        )
    if len(cmd) >= 2 and cmd[1] == "tools":
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ultracode", stderr="")
    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")


def test_doctor_run_cache_is_per_context_not_process_global() -> None:
    """Same callable identity+argv across contexts must not share stale results."""
    responses = [
        subprocess.CompletedProcess(args=["probe"], returncode=0, stdout="first", stderr=""),
        subprocess.CompletedProcess(args=["probe"], returncode=1, stdout="second", stderr=""),
    ]
    calls = 0

    def shared_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        result = responses[calls]
        calls += 1
        return result

    ctx_a = DoctorContext(Path.cwd(), "hermes", shared_run)
    first = ctx_a.run_cached(["probe"])
    cached = ctx_a.run_cached(["probe"])
    assert first.stdout == "first"
    assert cached.stdout == "first"
    assert calls == 1

    ctx_b = DoctorContext(Path.cwd(), "hermes", shared_run)
    second = ctx_b.run_cached(["probe"])
    assert second.stdout == "second"
    assert second.returncode == 1
    assert calls == 2


def test_doctor_run_memoizes_duplicate_commands(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/hermes",
    )
    invocations: list[list[str]] = []

    def _counting_run(cmd):
        invocations.append(list(cmd))
        return _mock_healthy_hermes_run(cmd)

    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework._make_runner",
        lambda timeout=8.0: _counting_run,
    )
    result = run_doctor(
        cwd=Path.cwd(),
        catalog_path=_catalog_path(),
        probes=PROBES,
        plugin="effort-ultracode",
        version="0.1.4",
    )
    statuses = {c.check_id: c.status for c in result.checks}
    assert statuses["hermes_version"] == "pass"
    assert statuses["hermes_subprocess_launchable"] == "pass"
    assert statuses["hermes_oneshot_flag"] == "pass"
    assert statuses["hermes_z_flag_support"] == "pass"

    hermes_version_calls = [cmd for cmd in invocations if cmd[0] == "hermes" and cmd[-1] == "--version"]
    codex_version_calls = [cmd for cmd in invocations if cmd[0] == "codex" and cmd[-1] == "--version"]
    hermes_bridge_help_calls = [
        cmd for cmd in invocations if cmd[0] == "hermes" and cmd[1:] == ["ultracode-llm", "--help"]
    ]
    codex_help_calls = [cmd for cmd in invocations if cmd[0] == "codex" and cmd[-1] == "--help"]
    assert len(hermes_version_calls) == 1
    assert len(codex_version_calls) == 1
    assert len(hermes_bridge_help_calls) == 1
    assert len(codex_help_calls) == 1


def test_doctor_warm_run_under_400ms(monkeypatch):
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/hermes",
    )
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework._make_runner",
        lambda timeout=8.0: _mock_healthy_hermes_run,
    )
    kwargs = {
        "cwd": Path.cwd(),
        "catalog_path": _catalog_path(),
        "probes": PROBES,
        "plugin": "effort-ultracode",
        "version": "0.1.4",
    }
    run_doctor(**kwargs)
    start = time.perf_counter()
    run_doctor(**kwargs)
    assert time.perf_counter() - start < 0.4


def test_hermes_subprocess_launchable_doctor_invariants(monkeypatch):
    cat = _catalog_path()
    doctor_kwargs = {
        "cwd": Path.cwd(),
        "catalog_path": cat,
        "probes": PROBES,
        "plugin": "effort-ultracode",
        "version": "0.1.4",
    }

    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/hermes",
    )
    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework._make_runner",
        lambda timeout=8.0: _mock_healthy_hermes_run,
    )
    healthy = run_doctor(**doctor_kwargs)
    statuses = {c.check_id: c.status for c in healthy.checks}
    assert statuses["hermes_subprocess_launchable"] == "pass"
    assert healthy.ok is True
    assert json.loads(render_json(healthy))["ok"] is True

    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: None,
    )
    hermes_absent_fail_probes = {
        "codex_on_path",
        "codex_version",
        "hermes_on_path",
        "hermes_version",
        "hermes_oneshot_flag",
        "toolset_valid",
    }
    absent_probes = {k: v for k, v in PROBES.items() if k not in hermes_absent_fail_probes}
    absent = run_doctor(**{**doctor_kwargs, "probes": absent_probes})
    statuses = {c.check_id: c.status for c in absent.checks}
    assert statuses["hermes_subprocess_launchable"] == "skip"
    assert absent.ok is True

    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework.shutil.which",
        lambda _: "/usr/local/bin/hermes",
    )

    def _broken_launch(cmd):
        if "--version" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=127, stdout="", stderr="Permission denied")
        return _mock_healthy_hermes_run(cmd)

    monkeypatch.setattr(
        "cluxion_effort_ultracode.doctor.framework._make_runner",
        lambda timeout=8.0: _broken_launch,
    )
    failed = run_doctor(**doctor_kwargs)
    statuses = {c.check_id: c.status for c in failed.checks}
    assert statuses["hermes_subprocess_launchable"] == "fail"
    assert failed.ok is False
    assert failed.summary == "fail"
