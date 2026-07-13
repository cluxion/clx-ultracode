---
name: clx-ultracode
description: Use Cluxion Ultracode when a decision needs bounded multi-agent adversarial debate, unanimous consensus, or honest no_consensus output.
disable-model-invocation: true
---

# Cluxion Ultracode

Call the package CLI. The host agent owns model calls, shell execution, and final answers.

## Consensus

```bash
cluxion-ultracode consensus --question "<decision or proposal>" --context "<optional context>"
```

On Codex hosts, prefer the first-class Codex adapter:

```bash
cluxion-ultracode consensus --question "<decision or proposal>" --adapter codex
```

Useful options:

```bash
cluxion-ultracode consensus --question "<decision>" --rounds 3 --agents 3
cluxion-ultracode consensus --question "<decision>" --agent-timeout 180 --debate-budget 600
cluxion-ultracode consensus --question "<decision>" --budget-tokens 120000 --models cheap,strong,cheap
cluxion-ultracode consensus --question "<decision>" --adapter hermes
cluxion-ultracode consensus --question-file <path>
cat <path> | cluxion-ultracode consensus --question -
cluxion-ultracode consensus --question "<decision>" --adapter mock-unanimous
cluxion-ultracode consensus --resume <run_id>
```

Worst-case cost: `agents * (rounds + 1)` model calls plus `tokens_spent`. Token usage is real when
the backend reports usage, otherwise `estimated: true` via chars/4. Every result includes `run_id` and
`journal_path`; resume replays matching calls into `tokens_replayed` and only live suffix calls
consume `tokens_spent`/`--budget-tokens`. Completed journals can be replayed for deterministic
debugging. Journaled runs do NOT drop timed-out agents: a timeout or completion failure aborts the
invocation; `--resume` replays the recorded prefix and retries the unrecorded failed call, which may
be billed again. Timeout-drop + `MIN_QUORUM` continuation apply only to the non-journaled parallel path. Validation errors use `invalid_question`, `invalid_models`, `invalid_agents`,
`invalid_rounds`, `invalid_budget`, or `invalid_timeout`; missing journals on resume return
`journal_not_found`, while newline-terminated or mid-file corruption returns `journal_corrupt`
without mutation (only a torn final no-newline fragment is repaired).

## Journals

```bash
cluxion-ultracode journals list
cluxion-ultracode journals show <run_id>
cluxion-ultracode journals gc --older-than-days 7
cluxion-ultracode journals gc --older-than-days 7 --apply
```

Rules:

1. Treat the CLI output as the JSON contract.
2. If `status` is `unanimous`, report the decision, rationale, and evidence trail.
3. If `status` is `no_consensus`, report the dissent instead of fabricating agreement.
4. If `status` is `aborted`, report `abort_reason`, `rounds_completed`, and the partial transcript.
5. Treat `abort_reason: "token_budget_exceeded"` as an honest budget stop, not a failed consensus.
6. Do not raise `--rounds` or `--agents` past the CLI hard caps.
7. On resume errors, report `resume_mismatch` fields, `journal_not_found`, or `journal_corrupt` instead of mixing runs.
8. Never claim checks were run unless the host actually ran them.

## Doctor

```bash
cluxion-ultracode doctor
cluxion-ultracode doctor --json
```
