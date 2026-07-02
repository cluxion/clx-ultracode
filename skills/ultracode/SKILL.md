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
cluxion-ultracode consensus --question "<decision>" --adapter mock-unanimous
```

Rules:

1. Treat the CLI output as the JSON contract.
2. If `status` is `unanimous`, report the decision, rationale, and evidence trail.
3. If `status` is `no_consensus`, report the dissent instead of fabricating agreement.
4. Do not raise `--rounds` or `--agents` past the CLI hard caps.
5. Never claim checks were run unless the host actually ran them.

## Doctor

```bash
cluxion-ultracode doctor
cluxion-ultracode doctor --json
```
