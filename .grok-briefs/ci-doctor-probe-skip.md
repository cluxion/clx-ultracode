# Task: doctor hermes-dependent critical probes must SKIP (not fail) when hermes is absent (CI fix)

## Context
CI (no `hermes` binary on PATH) failed: `tests/test_doctor.py:105` expected summary=="degraded" but
got "fail". The critical `hermes_binary_available` probe FAILS when hermes isn't on PATH, instead of
SKIP. In a clean build/CI env hermes isn't installed, so this should SKIP (cannot verify) → summary
"degraded", NOT "fail". A "fail" should mean a definite defect, not a missing runtime dependency at
build/CI time.

## Fix (doctor/probes.py + tests/test_doctor.py)
- `hermes_binary_available` (and `hermes_z_flag_support`, which also needs the hermes binary): if
  `shutil.which("hermes")` is None, return SKIP ("hermes binary not on PATH — cannot verify")
  instead of fail. Align with `hermes_on_path` behavior. FAIL only for a definite contract violation
  when hermes IS present.
- Keep summary logic: critical SKIP → "degraded"; critical FAIL → "fail".
- Make the env-dependent test robust: the critical-skip→degraded test should run with hermes ABSENT
  (monkeypatch `shutil.which` to return None) so the probe skips → degraded, independent of the CI
  machine. Add a test with hermes "present" (monkeypatched) asserting the probe PASSes.

## Invariants (MUST hold)
- hermes absent: hermes probes SKIP → summary "degraded". hermes present: PASS.
- No change to the consensus algorithm or CLI behavior.

## Tests
- `uv run pytest` green REGARDLESS of whether hermes is installed (this is what CI needs). The
  degraded-summary test no longer depends on the CI machine having hermes.

## Out of scope
- No version bump / publish.

## Done
hermes-dependent probes SKIP (not fail) when hermes absent; summary degraded in clean env; tests
pass without hermes. Concise diff summary.
