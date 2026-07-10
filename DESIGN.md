# cluxion Effort-Ultracode v0.1 Design

## Source Spec Mapping

The authoritative source is the Ultracode master spec (original local design doc; not tracked in this repo).
v0.1 implements one additive quality pattern from that spec: a 3-agent adversarial debate that
can be layered into the broader ultracode loop. It does not replace the loop. The intended future
shape remains:

1. understand
2. design
3. implement
4. review
5. apply quality patterns such as adversarial verification, judge panels, loop-until-dry, and now
   unanimous consensus debate where a decision must be settled.

The important mapping to Part 1 §3.8 is that the model controls only content. Code controls:

- agent count
- round count
- prompt fan-out order
- transcript shape
- stance normalization
- vote/convergence checks
- termination on unanimity or honest no-consensus
- concession validation

The engine never asks the model whether consensus exists. It computes consensus by normalizing the
agents' stances and checking whether every normalized stance is identical.

## Core, Ports, and Adapters

The package follows the Part 2 dependency-inversion requirement:

- `src/cluxion_effort_ultracode/core/types.py` contains dataclasses only.
- `src/cluxion_effort_ultracode/core/ports.py` owns the `LlmPort` protocol and optional ports.
- `src/cluxion_effort_ultracode/core/consensus.py` contains the deterministic algorithm.
- `src/cluxion_effort_ultracode/adapters/callable_llm.py` is a reference adapter for tests and
  plain Python use.
- `src/cluxion_effort_ultracode/adapters/hermes_llm.py` uses the host plugin LLM surface
  (`ctx.llm`) in-process, or the standalone `hermes ultracode-llm` stdin/stdout bridge.
- `src/cluxion_effort_ultracode/plugin.py` is a thin Hermes-facing shim.

The portable core imports no host SDK and knows no Hermes, Claude, Codex, OpenAI, or workflow host
API. The only runtime dependency it needs is an object that satisfies:

```python
complete(
    prompt: str,
    *,
    schema: Mapping[str, Any] | None = None,
    model: str | None = None,
) -> Mapping[str, Any] | str
```

Adapters are intentionally translation layers. If Hermes later exposes a native structured-output
or workflow-agent API, that support should be implemented behind the port, not inside
`ConsensusEngine`.

## Consensus Algorithm

Inputs:

- `question`: the decision, proposal, or question.
- `context`: shared context shown to every agent.
- `agents_count`: default 3.
- `max_rounds`: debate rounds after independent round 0.

State:

- `AgentPosition`: `{agent_id, stance, rationale, evidence, confidence, conceded, maintained}`.
- `ConsensusRound`: one transcript entry for round 0 or a debate round.
- `ConsensusResult`: final structured result.

Round 0:

- The engine calls the LLM port once per agent.
- Each prompt contains only the question, context, and that agent's identifier.
- Round 0 prompts do not include other agents' positions.
- The response must include stance, rationale, evidence, and confidence.

Debate rounds:

- Each agent sees all current positions.
- Each agent must either maintain/rebut a specific point with a reason, or concede a specific point
  with a reason.
- A changed stance requires at least one explicit concession.
- Any conceded or maintained point without a non-empty reason raises `ConsensusProtocolError`.

Convergence:

- After every round, code normalizes stances using Unicode normalization, case-folding, punctuation
  removal, and whitespace collapse.
- Unanimity means exactly one non-empty normalized stance remains.
- The decision is the first agent's display stance for that unanimous normalized value.

Termination:

- On unanimity: `status="unanimous"`, `decision` is set, rationale merges the agent rationales,
  and `evidence_trail` contains deduplicated evidence in deterministic order.
- On max rounds without unanimity: `status="no_consensus"`, `decision=None`, each final stance is
  returned in `dissent`, and `points_of_disagreement` records the remaining split.
- The engine never fabricates agreement from rationales such as "we agree"; only stance equality
  computed by code can produce unanimity.

## Anti-Groupthink Safeguards

v0.1 makes the safeguards explicit and testable:

- Independent round 0.
- Evidence is required and must contain at least one non-empty item.
- Concession requires a stated reason.
- A stance change without concession is invalid.
- Convergence is deterministic code, not model self-reporting.
- A rotating devil's-advocate instruction is included in debate prompts by default.
- No-consensus is a first-class honest result with dissent preserved.

## CLI

The CLI entry point is:

```bash
cluxion-ultracode consensus --question "..." --adapter mock-unanimous
```

The CLI boundary supports the real `hermes` and `codex` adapters plus deterministic test adapters:

- `mock-unanimous`
- `mock-no-consensus`

The mock adapters keep tests and local smoke runs network-free. Inside Hermes, the plugin injects
`HermesHostLlm` over `ctx.llm`; standalone `--adapter hermes` uses `HermesSubprocessLlm`. Other
hosts can add adapters by implementing the `LlmPort` contract.

## Hermes Shim

`plugin.py` exposes `register(ctx)` and registers `cluxion_consensus` through the real Hermes
PluginContext contract:

```python
ctx.register_tool(
    name="cluxion_consensus",
    toolset="ultracode",
    schema=CONSENSUS_SCHEMA,
    handler=handler,
    emoji="🧠",
)
```

`CONSENSUS_SCHEMA` is a full function spec: `{name, description, parameters}`. Hermes wraps this
verbatim as `{"type": "function", "function": schema}` before exposing it to the model, so the
schema is not a partial parameter object.

The handler accepts `(args: dict, **kwargs)` and returns a JSON string:

- success: `{"ok": true, "result": {...}}`
- failure: `{"ok": false, "error": "...", "message": "..."}`

For real LLM work on a Hermes host, the registered tool/slash command injects `HermesHostLlm`,
a duck-typed adapter over the host's lazy `ctx.llm.complete(messages=..., model=..., timeout=...,
purpose="cluxion-ultracode")` surface. It must not spawn a nested full Hermes agent. Standalone
CLI `adapter=hermes` uses `HermesSubprocessLlm`, which launches exactly:

```bash
hermes ultracode-llm
```

with one UTF-8 JSON request on stdin (`v`, `prompt`, `schema`, `model`, `timeout_s`) and a single
stdout envelope line (`marker="cluxion-ultracode-llm"`, `v=1`, `ok`, `output`/`error`, `usage`,
`provider`, `model`). Prompt and schema never appear on argv. The bridge process owns optional-schema
structured prompting plus one JSON repair retry, so one logical `complete()` is one external process.
`hermes ultracode-llm --help` is token-free (setup_fn only configures argparse). Explicit model
overrides require `plugins.entries.cluxion-agentplugin-effort-ultracode.llm.allow_model_override`
and, when restricted, `allowed_models`; the plugin never enables this trust itself. A denied
override is a typed `model_override_denied` error with no silent substitution. Codex selection
remains `CodexSubprocessLlm`.

Runtime knobs:

- `CLUXION_EFFORT_ULTRACODE_HERMES_BINARY`: defaults to `hermes`.
- `CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT`: defaults to 120 seconds per call.
- `CLUXION_EFFORT_ULTRACODE_HERMES_MODEL`: optional default model for host/bridge requests (requires host trust when overriding).

Cost note: `cluxion_consensus` is an opt-in deep-deliberation tool. Maximum logical fan-out is
`agents_count * (max_rounds + 1)` agent/adapter calls. The default `agents=3, rounds=3` can therefore
make up to 12 logical calls, though the engine stops early if round-0 or a debate round reaches
unanimity. A malformed first Hermes structured response can add at most one provider repair call
per logical call. Token usage is tracked per logical call and per round: aggregated real host usage
wins when complete, otherwise the engine marks `estimated: true` and uses chars/4.

## Broader Ultracode Porting Deferred

The rest of the master spec should be added as separate portable core modules and ports:

- Scheduler: concurrency caps, queue rotation, per-call and lifetime fan-out limits.
- Workflow primitives: `agent`, `parallel`, `pipeline`, `phase`, `log`, `budget`, nested workflow.
- Journal/resume: run IDs, immutable-prefix replay, transcript persistence.
- Script runtime: deterministic JavaScript subset, meta literal validation, nondeterminism guards.
- Quality patterns: adversarial verify, perspective-diverse verify, judge panel, loop-until-dry,
  completeness critic, no-silent-caps logging.
- Capability negotiation: background execution, structured output, token metering, model resolution,
  worktree isolation, and graceful degradation warnings.

Recommended build order:

1. Add a `RuntimeProfile` and `LogPort` so all degraded paths are visible.
2. Replace subprocess JSON parsing with native structured output if Hermes exposes it.
3. Add a workflow-agent adapter if the host can spawn independent sessions.
4. Add journal/transcript persistence before long-running orchestration.
5. Add scheduler and fan-out primitives.
6. Promote this consensus engine into the quality-pattern library so the broader loop can call it
   when a design or implementation decision needs unanimous settlement.

## Open Questions

- Does Hermes expose native JSON schema enforcement that should replace subprocess JSON parsing?
- Can Hermes spawn isolated sub-agent contexts, or is the first real adapter a single-session
  completion adapter?
- Where should durable journals live for Cluxion/Hermes sessions?
- How should token accounting be reported if the host only exposes post-hoc usage?
