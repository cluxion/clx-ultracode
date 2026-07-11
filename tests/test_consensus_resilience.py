from __future__ import annotations

import concurrent.futures
import time

import pytest

from cluxion_effort_ultracode.core.consensus import ConsensusEngine, _PostCallDeadline
from cluxion_effort_ultracode.core.types import AgentPosition, TokenUsage
from test_consensus import ScriptedLlm, _agent_id_from_prompt, position


class SlowThenFineLlm:
    """First agent call hangs past the per-agent timeout; the rest answer."""

    def __init__(self, hang_seconds: float) -> None:
        self.hang_seconds = hang_seconds
        self.calls = 0

    def complete(self, prompt: str, *, schema: object = None) -> str:
        self.calls += 1
        if self.calls == 1:
            time.sleep(self.hang_seconds)
        return position("Adopt proposal")


def test_hung_agent_is_dropped_and_quorum_continues() -> None:
    llm = SlowThenFineLlm(hang_seconds=2.0)
    engine = ConsensusEngine(llm, agents_count=3, max_rounds=1, agent_timeout_s=0.3, debate_budget_s=30.0)
    result = engine.decide("Should we adopt the proposal?")
    assert result.status == "unanimous"
    assert len(result.transcript[0].positions) == 2


class SlowScriptedLlm(ScriptedLlm):
    def complete(self, prompt: str, *, schema: object = None) -> str:
        time.sleep(0.01)
        return super().complete(prompt, schema=schema)


def test_total_debate_budget_is_enforced() -> None:
    llm = SlowScriptedLlm([position("Adopt"), position("Delay"), position("Reject")] * 10)
    engine = ConsensusEngine(llm, agents_count=3, max_rounds=8, debate_budget_s=0.005)
    result = engine.decide("Q?")
    assert result.status == "aborted"
    assert result.abort_reason is not None
    assert "debate_budget_s" in result.abort_reason
    assert result.rounds_completed == 0
    # Serial path aborts mid-round once the budget is crossed; incomplete round is not marked completed.
    assert len(result.transcript) == 0
    assert result.tokens_spent > 0


def test_serial_budget_checked_before_and_after_each_adapter_call() -> None:
    """Production serial path bounds overrun to one adapter call and aborts honestly."""

    class SerialSlowLlm:
        serial_complete = True

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, *, schema: object = None, model: str | None = None) -> dict[str, object]:
            del prompt, schema, model
            self.calls += 1
            time.sleep(0.04)
            return position(f"Stance-{self.calls}")

    llm = SerialSlowLlm()
    engine = ConsensusEngine(llm, agents_count=3, max_rounds=1, debate_budget_s=0.03)
    result = engine.decide("Q?")

    assert result.status == "aborted"
    assert "debate_budget_s" in (result.abort_reason or "")
    assert result.rounds_completed == 0
    assert len(result.transcript) == 0
    # First call allowed to finish (overrun bound); second must not start after deadline.
    assert llm.calls == 1
    assert result.tokens_spent > 0


def test_serial_budget_abort_counts_completed_call_tokens_once() -> None:
    class TokenSerialLlm:
        serial_complete = True

        def __init__(self) -> None:
            self.calls = 0
            self.last_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "estimated": False}

        def complete(self, prompt: str, *, schema: object = None, model: str | None = None) -> dict[str, object]:
            del prompt, schema, model
            self.calls += 1
            time.sleep(0.04)
            return position("Adopt" if self.calls == 1 else "Delay")

    llm = TokenSerialLlm()
    engine = ConsensusEngine(llm, agents_count=2, max_rounds=0, debate_budget_s=0.02)
    result = engine.decide("Q?")

    assert result.status == "aborted"
    assert result.rounds_completed == 0
    assert len(result.transcript) == 0
    assert llm.calls == 1
    assert result.tokens_spent == 15


def test_serial_slow_malformed_after_deadline_aborts_budget_not_protocol() -> None:
    """Deadline is checked right after llm.complete; malformed output must not mask timeout."""

    class SlowMalformedLlm:
        serial_complete = True

        def __init__(self) -> None:
            self.calls = 0
            self.last_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "estimated": False}

        def complete(self, prompt: str, *, schema: object = None, model: str | None = None) -> str:
            del prompt, schema, model
            self.calls += 1
            time.sleep(0.04)
            return "{not-valid-json"

    llm = SlowMalformedLlm()
    engine = ConsensusEngine(llm, agents_count=2, max_rounds=0, debate_budget_s=0.02)
    result = engine.decide("Q?")

    assert result.status == "aborted"
    assert "debate_budget_s" in (result.abort_reason or "")
    assert result.rounds_completed == 0
    assert len(result.transcript) == 0
    assert llm.calls == 1
    assert result.tokens_spent == 15


def _tok(n: int = 15) -> TokenUsage:
    return TokenUsage(input_tokens=n - 5, output_tokens=5, total_tokens=n, estimated=False)


def _pos(agent_id: str = "agent-2", tokens: int = 15) -> AgentPosition:
    return AgentPosition(
        agent_id=agent_id,
        stance="Delay",
        rationale="R",
        evidence=["E"],
        confidence=0.8,
        tokens=_tok(tokens),
    )


def _engine_for_account() -> ConsensusEngine:
    class _Stub:
        def complete(self, prompt: str, *, schema: object = None, model: str | None = None) -> dict[str, object]:
            del prompt, schema, model
            raise AssertionError("complete must not run during unit accounting tests")

    return ConsensusEngine(_Stub(), agents_count=2, max_rounds=0, debate_budget_s=10.0)


def test_account_post_deadline_parallel_dual_deadline_counts_both() -> None:
    """Both current + snapshotted sibling are _PostCallDeadline: 30 tokens, 2 completed calls."""
    engine = _engine_for_account()
    sibling = concurrent.futures.Future()
    sibling.set_exception(_PostCallDeadline(_tok(15)))
    futures = {object(): 0, sibling: 1}

    completed = engine._account_post_deadline_parallel(
        _PostCallDeadline(_tok(15)),
        futures=futures,
        results={},
        current_index=0,
    )

    assert completed == 2
    assert engine._tokens_spent == 30


def test_account_post_deadline_parallel_successful_sibling() -> None:
    """Snapshotted successful AgentPosition sibling contributes tokens + completed_calls."""
    engine = _engine_for_account()
    sibling = concurrent.futures.Future()
    sibling.set_result(_pos(tokens=15))
    futures = {object(): 0, sibling: 1}

    completed = engine._account_post_deadline_parallel(
        _PostCallDeadline(_tok(15)),
        futures=futures,
        results={},
        current_index=0,
    )

    assert completed == 2
    assert engine._tokens_spent == 30


def test_account_post_deadline_parallel_generic_pre_adapter_failure_not_counted() -> None:
    """Generic Exception sibling proves no adapter completion: no tokens, no completed_calls."""
    engine = _engine_for_account()
    sibling = concurrent.futures.Future()
    sibling.set_exception(RuntimeError("pre-adapter failure"))
    futures = {object(): 0, sibling: 1}

    completed = engine._account_post_deadline_parallel(
        _PostCallDeadline(_tok(15)),
        futures=futures,
        results={},
        current_index=0,
    )

    assert completed == 1
    assert engine._tokens_spent == 15


def test_account_post_deadline_parallel_late_done_not_in_snapshot_ignored() -> None:
    """Future that becomes done only during current-token accounting is outside the frozen snapshot."""
    engine = _engine_for_account()
    late = concurrent.futures.Future()
    futures = {object(): 0, late: 1}
    late_deadline = _PostCallDeadline(_tok(15))

    original_account = engine._account_call_tokens

    def account_and_complete_late(tokens: TokenUsage) -> None:
        # During current-token accounting the late sibling finishes, but it was
        # not done when the snapshot was taken, so it must not be drained.
        if not late.done():
            late.set_exception(late_deadline)
        original_account(tokens)

    engine._account_call_tokens = account_and_complete_late  # type: ignore[method-assign]

    completed = engine._account_post_deadline_parallel(
        _PostCallDeadline(_tok(15)),
        futures=futures,
        results={},
        current_index=0,
    )

    assert late.done()
    assert completed == 1
    assert engine._tokens_spent == 15


def test_budget_checked_before_unanimous_success_return(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("cluxion_effort_ultracode.core.consensus.time.monotonic", lambda: clock["now"])

    class UnanimousLlm:
        serial_complete = True

        def complete(self, prompt: str, *, schema: object = None, model: str | None = None) -> dict[str, object]:
            del prompt, schema, model
            return position("Adopt")

    original_append = ConsensusEngine._append_round

    def _append_and_expire(self, transcript, *, round_index, phase, positions):
        original_append(self, transcript, round_index=round_index, phase=phase, positions=positions)
        clock["now"] = 100.0

    monkeypatch.setattr(ConsensusEngine, "_append_round", _append_and_expire)
    engine = ConsensusEngine(UnanimousLlm(), agents_count=2, max_rounds=0, debate_budget_s=10.0)
    result = engine.decide("Q?")

    assert result.status == "aborted"
    assert "debate_budget_s" in (result.abort_reason or "")
    assert result.rounds_completed == 0
    assert len(result.transcript) == 1


def test_budget_checked_before_no_consensus_success_return(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("cluxion_effort_ultracode.core.consensus.time.monotonic", lambda: clock["now"])

    class DivergentLlm:
        serial_complete = True

        def complete(self, prompt: str, *, schema: object = None, model: str | None = None) -> dict[str, object]:
            del schema, model
            agent_id = _agent_id_from_prompt(prompt)
            stance = "Adopt" if agent_id == "agent-1" else "Delay"
            if "Round: 0" in prompt:
                return position(stance)
            return {
                **position(stance),
                "maintained": [{"point": "current", "reason": "still stronger"}],
                "conceded": [],
            }

    original_append = ConsensusEngine._append_round

    def _append_and_expire(self, transcript, *, round_index, phase, positions):
        original_append(self, transcript, round_index=round_index, phase=phase, positions=positions)
        if round_index >= 1:
            clock["now"] = 100.0

    monkeypatch.setattr(ConsensusEngine, "_append_round", _append_and_expire)
    engine = ConsensusEngine(DivergentLlm(), agents_count=2, max_rounds=1, debate_budget_s=10.0)
    result = engine.decide("Q?")

    assert result.status == "aborted"
    assert "debate_budget_s" in (result.abort_reason or "")
    assert result.rounds_completed == 1


def test_quorum_abort_returns_partial_transcript() -> None:
    class SlowAfterInitialLlm:
        def complete(self, prompt: str, *, schema: object = None) -> dict[str, object]:
            if "Round: 0" in prompt:
                return position("Adopt" if "agent-1" in prompt else "Delay")
            time.sleep(1.0)
            return {
                **position("Adopt"),
                "conceded": [{"point": "Delay", "reason": "Adopt has stronger evidence"}],
                "maintained": [],
            }

    engine = ConsensusEngine(SlowAfterInitialLlm(), agents_count=3, max_rounds=1, agent_timeout_s=0.05)
    result = engine.decide("Q?")
    assert result.status == "aborted"
    assert result.rounds_completed == 0
    assert len(result.transcript) == 1
    assert "quorum lost" in (result.abort_reason or "")


def test_devils_advocate_rotates_over_surviving_agents_after_drop() -> None:
    class DropThenRecordLlm:
        def __init__(self) -> None:
            self.debate_prompts: list[str] = []

        def complete(self, prompt: str, *, schema: object = None) -> dict[str, object]:
            agent_id = _agent_id_from_prompt(prompt)
            if "Round: 0" in prompt:
                if agent_id == "agent-1":
                    time.sleep(0.2)
                return position("Adopt" if agent_id == "agent-2" else "Delay")
            self.debate_prompts.append(prompt)
            return {
                **position("Adopt" if agent_id == "agent-2" else "Delay"),
                "maintained": [{"point": "current stance", "reason": "Evidence remains stronger"}],
                "conceded": [],
            }

    llm = DropThenRecordLlm()
    engine = ConsensusEngine(llm, agents_count=3, max_rounds=2, agent_timeout_s=0.05)
    result = engine.decide("Q?")

    assert result.status == "no_consensus"
    round_2 = [p for p in llm.debate_prompts if "Round: 2 " in p]
    assert len(round_2) == 2
    assert sum("Temporary role: devil's advocate" in p for p in round_2) == 1


def test_invalid_timeouts_rejected() -> None:
    llm = ScriptedLlm([])
    with pytest.raises(ValueError):
        ConsensusEngine(llm, agent_timeout_s=0)
    with pytest.raises(ValueError):
        ConsensusEngine(llm, debate_budget_s=-1)
    with pytest.raises(ValueError):
        ConsensusEngine(llm, budget_tokens=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 10**400])
def test_engine_rejects_non_finite_or_non_positive_timeout_budget(value: float | int) -> None:
    llm = ScriptedLlm([])
    with pytest.raises(ValueError, match="agent_timeout_s"):
        ConsensusEngine(llm, agent_timeout_s=value)
    with pytest.raises(ValueError, match="debate_budget_s"):
        ConsensusEngine(llm, debate_budget_s=value)


def test_engine_accepts_positive_int_and_float_timeout_budget() -> None:
    llm = ScriptedLlm([])
    engine = ConsensusEngine(llm, agent_timeout_s=12, debate_budget_s=30.5)
    assert engine.agent_timeout_s == 12.0
    assert engine.debate_budget_s == 30.5
