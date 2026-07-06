# Task: Clean up ultracode doctor coverage + CLI consensus path (P2)

## Context (from inspection — core 3-agent consensus is verified working end-to-end; these are P2 hygiene)
- doctor: 14 of 22 catalog checks skip "no probe registered". Several are statically verifiable
  WITHOUT any LLM/hermes call. Separately, 6 REGISTERED probes are dead (not referenced by
  catalog.json so never run), and some are irrelevant to this pure-python plugin
  (`sqlite_wal_mode_compatible`, `abi3_wheel_compatible` — deps=[], no sqlite/native).
- CLI `consensus` forces `--adapter {mock-unanimous,mock-no-consensus}` (required); the real Hermes
  path exists only in the plugin handler (`_default_llm`). README implies the CLI runs real
  consensus, but it can't.

## Implement
1. Register the statically-checkable probes currently skipped (NO live LLM/subprocess needed), e.g.:
   `consensus_schema_contract` (tool schema has required=['question']), `hermes_timeout_configured`
   (env parse/default), `llm_factory_callable`, `plugin_registration_host_compat` (register on a
   minimal ctx without register_tool). Implement only those that need no live model.
2. Remove or re-scope the dead/irrelevant registered probes: DROP `sqlite_wal_mode_compatible` and
   `abi3_wheel_compatible` for this pure-python plugin; for the others, either add a catalog entry
   (if meaningful) or remove.
3. CLI consensus: EITHER add `--adapter hermes` that routes through the real `_default_llm` path, OR
   clearly state in `consensus --help` + README that CLI adapters are mock-only and real consensus
   runs via the hermes plugin tool. Prefer adding the real adapter if low-risk.

## Invariants (MUST hold)
- The 3-agent adversarial consensus CORE behavior is UNCHANGED. Add NO live LLM calls to doctor.
- Existing tests stay green.

## Tests (must pass)
- `uv run pytest` green; new probes have unit tests. If you add `--adapter hermes`, gate any live
  test behind the existing `CLUXION_EFFORT_ULTRACODE_LIVE` env so CI never calls real models.

## Out of scope
- No version bump / build / publish. No change to the consensus algorithm.

## Done
doctor coverage materially improved (skipped static checks registered; dead/irrelevant probes
removed), CLI consensus real-path added or clearly documented, tests green. Concise diff summary.
