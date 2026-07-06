# Task: ultracode doctor — honest summary + register statically-checkable critical hermes probes

## Context
After the P2 cleanup, ultracode doctor still shows critical checks as "skip (no probe registered)" —
`hermes_binary_available`, `hermes_subprocess_launchable`, `hermes_z_flag_support` — yet the overall
`ok` stays True (green), hiding that critical checks never ran. The sibling supercoder plugin was
given an honest summary (ok|degraded|fail) that downgrades when a critical check is unregistered;
ultracode should match for consistency.

## Implement
1. Add the SAME honest summary to ultracode's doctor/framework.py: a `summary` field
   ("ok"|"degraded"|"fail") where summary is "degraded" (and top-level ok False) when ANY CRITICAL
   catalog check is `skip` (unregistered). Include it in BOTH text and `--json` output. Mirror
   supercoder's doctor/framework.py approach.
2. Register the statically-checkable critical probes that DON'T need a live model/subprocess:
   - `hermes_binary_available`: assert the hermes binary resolves on PATH (shutil.which) — static.
   - `hermes_z_flag_support`: assert `hermes --help` advertises `-z`/`--oneshot` (parse --help) —
     static, cheap, no consensus run.
   Leave `hermes_subprocess_launchable` (needs a real subprocess) unregistered — the honest summary
   will correctly mark it.

## Invariants (MUST hold)
- Consensus core behavior UNCHANGED. NO live model calls added to doctor. Existing probes/tests green.

## Tests
- `uv run pytest` green; new probes + summary have tests, including one asserting summary=="degraded"
  when a critical check is unregistered (mirror supercoder's test_critical_skip_marks_degraded_summary).

## Out of scope
- No version bump / build / publish. No consensus algorithm change.

## Done
doctor reports an honest ok|degraded|fail summary (degraded when a critical check is unregistered);
hermes_binary_available + hermes_z_flag_support registered as static probes; tests green. Concise diff.
