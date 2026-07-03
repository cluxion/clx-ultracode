---
name: cluxion-ultracode
description: Use Cluxion Ultracode when a decision needs bounded multi-agent adversarial debate, unanimous consensus, or honest no_consensus output.
---

# Cluxion Ultracode

Call the package CLI. The host agent owns model calls, shell execution, and final answers.

## Consensus

```bash
cluxion-ultracode consensus --question "<decision or proposal>" --context "<optional context>"
```

Useful options:

```bash
cluxion-ultracode consensus --question "<decision>" --rounds 3 --agents 3
cluxion-ultracode consensus --question "<decision>" --agent-timeout 180 --debate-budget 600
cluxion-ultracode consensus --question "<decision>" --budget-tokens 120000 --models cheap,strong,cheap
cluxion-ultracode consensus --question "<decision>" --adapter mock-unanimous
```

Worst-case cost: `agents * (rounds + 1)` model calls plus `tokens_spent`. Token usage is real when
Hermes reports usage, otherwise `estimated: true` via chars/4.

Rules:

1. Treat the CLI output as the JSON contract.
2. If `status` is `unanimous`, report the decision, rationale, and evidence trail.
3. If `status` is `no_consensus`, report the dissent instead of fabricating agreement.
4. If `status` is `aborted`, report `abort_reason`, `rounds_completed`, and the partial transcript.
5. Treat `abort_reason: "token_budget_exceeded"` as an honest budget stop, not a failed consensus.
6. Do not raise `--rounds` or `--agents` past the CLI hard caps.
7. Never claim checks were run unless the host actually ran them.

## Doctor

```bash
cluxion-ultracode doctor
cluxion-ultracode doctor --json
```
