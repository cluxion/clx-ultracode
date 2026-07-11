---
description: Run Cluxion Ultracode adversarial consensus.
disable-model-invocation: true
---

Run on Codex hosts:

```bash
cluxion-ultracode consensus --question "$ARGUMENTS" --adapter codex
```

Run elsewhere / preserve legacy Hermes backend:

```bash
cluxion-ultracode consensus --question "$ARGUMENTS" --adapter hermes
```

Useful flags:

```bash
cluxion-ultracode consensus --question "$ARGUMENTS" --rounds 3 --agents 3 --agent-timeout 180 --debate-budget 600 --budget-tokens 120000 --models cheap,strong,cheap
cluxion-ultracode consensus --question "$ARGUMENTS" --adapter codex
cluxion-ultracode consensus --question-file <path>
cat <path> | cluxion-ultracode consensus --question -
cluxion-ultracode consensus --resume <run_id>
cluxion-ultracode journals list
cluxion-ultracode journals show <run_id>
cluxion-ultracode journals gc --older-than-days 7 --apply
```

Worst-case cost: `agents * (rounds + 1)` model calls plus `tokens_spent`. Token usage is real when
the backend reports usage, otherwise `estimated: true` via chars/4. Budget aborts (and parallel-path
quorum loss) return JSON with `status: "aborted"` and a partial transcript. In JOURNALED runs an agent
timeout/completion failure aborts the invocation instead (no graceful quorum-drop); `--resume` replays
the recorded prefix and retries the unrecorded failed call, which may be billed again. Every result includes `run_id` and `journal_path`;
resume replays matching recorded calls into `tokens_replayed` and only live suffix calls consume
`tokens_spent`/`--budget-tokens`. Completed journals can be replayed for deterministic debugging.
Validation errors use `invalid_question`, `invalid_models`, `invalid_agents`, `invalid_rounds`,
`invalid_budget`, or `invalid_timeout`; missing journals on resume return `journal_not_found`.
