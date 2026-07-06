# Task: effort-ultracode — fix 1 P2 defect (false-degraded self-doctor)

## Context
Installed build is 0.1.7. A live adversarial audit (running the installed site-packages build) confirmed 1 REAL P2 defect. Fix in repo `src/`. Do NOT regress the 13 live-verified working functions. Do NOT bump the version in pyproject (deploy handled separately). Do NOT touch `.grok-briefs/`.

## Defect (P2 — FALSE HEALTH ALARM): `ultracode_doctor` always reports 'degraded' / ok:false / exit-1 on a fully-healthy install
A critical catalog check (`hermes_subprocess_launchable`) is marked 'critical' but has NO registered probe, so it is skipped, and a critical-skip is treated as a failure → overall 'degraded' / ok:false / exit 1 even when hermes is correctly installed and everything verifiable passes. This is a false-negative health report and contradicts the user's standing requirement for a trustworthy, environment-robust built-in self-doctor.
Fix (pick the correct one and apply the principle consistently):
  - PREFERRED: register a real probe for `hermes_subprocess_launchable` — attempt a minimal/no-op hermes launch (e.g. `hermes --version`, or a cheap no-op) and report pass/fail. If the hermes binary is ABSENT, report SKIP (not fail). Do NOT make a real expensive LLM call.
  - OR demote that catalog entry below 'critical'; OR change the overall-health logic so a SKIPPED critical (not a FAILED one) does not degrade overall status.
Core principle: a SKIP is not a FAIL; overall status is 'degraded' only on a real FAIL.
Invariant + test (add, environment-independent via monkeypatch): healthy install (hermes present) → `ultracode_doctor` overall ok=True, exit 0; hermes absent → that probe SKIPs and overall stays ok; a genuine failure still → degraded/exit 1.

## Done criteria
- `uv run pytest` GREEN. `uv run ruff check .` pass. Environment-independent doctor test added.
- No version bump in pyproject. No edits under `.grok-briefs/`. Concise diff summary.
