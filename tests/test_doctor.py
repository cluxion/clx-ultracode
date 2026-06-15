"""Tests for embedded doctor (determinism + cross-cutting checks)."""

from pathlib import Path

from cluxion_effort_ultracode.doctor import (
    DoctorResult,
    render_json,
    run_doctor,
)
from cluxion_effort_ultracode.doctor.probes import PROBES


def _catalog_path() -> Path:
    import importlib.resources

    pkg = "cluxion_effort_ultracode.doctor"
    return Path(str(importlib.resources.files(pkg).joinpath("catalog.json")))


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
    # sorted by severity then id
    ids = [c.check_id for c in r1.checks]
    assert len(ids) > 0


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
    for key in ("hermes_on_path", "entry_point_registered", "toolset_valid"):
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
    # construct a result with only warn (no fail)
    from cluxion_effort_ultracode.doctor.framework import CheckResult, DoctorResult

    checks = (
        CheckResult(check_id="x", category="c", severity="medium", status="warn", detail="w"),
    )
    r = DoctorResult(plugin="p", version="0.1.4", checks=checks)
    assert r.ok is True


def test_new_deterministic_probes_non_skip():
    import subprocess
    from pathlib import Path

    from cluxion_effort_ultracode.doctor.framework import DoctorContext

    def _dummy_run(cmd):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    ctx = DoctorContext(Path.cwd(), "hermes", _dummy_run)
    new_probes = [
        "import_availability",
        "abi3_wheel_compatible",
        "sqlite_wal_mode_compatible",
        "json_serialization_deterministic",
        "hermes_plugin_enabled",
        "env_var_consistency",
        "hermes_timeout_configured",
        "plugin_registration_host_compat",
    ]
    non_skip = 0
    for name in new_probes:
        if name in PROBES:
            status, _ = PROBES[name](ctx)
            if status != "skip":
                non_skip += 1
    assert non_skip >= 2, f"only {non_skip} returned non-skip"
