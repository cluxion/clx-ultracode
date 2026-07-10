from __future__ import annotations

import time

import pytest

from cluxion_effort_ultracode.core.consensus import ConsensusEngine
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
    assert len(result.transcript) == 1


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
