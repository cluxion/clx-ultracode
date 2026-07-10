"""Deterministic tests for the portable consensus engine."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict
from typing import Any

import pytest

import cluxion_effort_ultracode.core.consensus as consensus_module
from cluxion_effort_ultracode.adapters import CallableLlmAdapter
from cluxion_effort_ultracode.core import ConsensusEngine, ConsensusProtocolError, normalize_stance


def test_consensus_engine_rejects_rounds_above_hard_cap() -> None:
    with pytest.raises(ValueError, match="max_rounds must be <= "):
        ConsensusEngine(ScriptedLlm([]), max_rounds=999)


def test_consensus_engine_rejects_agent_counts_above_hard_cap() -> None:
    with pytest.raises(ValueError, match="agents_count must be <= "):
        ConsensusEngine(ScriptedLlm([]), agents_count=999)


@pytest.mark.parametrize("question", ["", " "])
def test_consensus_engine_rejects_empty_question(question: str) -> None:
    with pytest.raises(ValueError, match="question"):
        ConsensusEngine(ScriptedLlm([])).decide(question)


class ScriptedLlm:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = deque(outputs)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "schema": schema, "model": model})
        if not self.outputs:
            pytest.fail("scripted LLM exhausted")
        return self.outputs.popleft()


def position(stance: str, evidence: list[str] | None = None, rationale: str | None = None) -> dict[str, Any]:
    return {
        "stance": stance,
        "rationale": rationale or f"Rationale for {stance}",
        "evidence": evidence or [f"Evidence for {stance}"],
        "confidence": 0.7,
    }


def update(
    stance: str,
    *,
    conceded: list[dict[str, str]] | None = None,
    maintained: list[dict[str, str]] | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        **position(stance, evidence=evidence),
        "conceded": conceded or [],
        "maintained": maintained or [],
    }


def test_round_zero_prompts_are_independent() -> None:
    llm = ScriptedLlm([position("Adopt"), position("Delay"), position("Reject")])
    result = ConsensusEngine(llm, max_rounds=0).decide("Should we ship?")

    assert result.status == "no_consensus"
    assert len(llm.calls) == 3
    for index, call in enumerate(llm.calls):
        prompt = call["prompt"]
        own_agent = f"agent-{index + 1}"
        other_agents = {f"agent-{i}" for i in range(1, 4)} - {own_agent}
        assert own_agent in prompt
        assert "Current positions:" not in prompt
        assert all(other_agent not in prompt for other_agent in other_agents)
        assert call["schema"] is not None


def test_debate_converges_to_unanimity_with_concessions() -> None:
    llm = ScriptedLlm(
        [
            position("Adopt proposal", ["A1"]),
            position("Delay proposal", ["D1"]),
            position("Reject proposal", ["R1"]),
            update(
                "Adopt proposal",
                maintained=[{"point": "Adopt", "reason": "Mitigation evidence remains strongest"}],
                evidence=["A1", "A2"],
            ),
            update(
                "Adopt proposal",
                conceded=[{"point": "Delay", "reason": "The risk is directly mitigated"}],
                evidence=["D1", "A2"],
            ),
            update(
                "Adopt proposal",
                conceded=[{"point": "Reject", "reason": "The blocking concern has a bounded workaround"}],
                evidence=["R1", "A2"],
            ),
        ]
    )

    result = ConsensusEngine(llm, max_rounds=2).decide("Should we adopt the proposal?")
    result_shape = asdict(result)

    assert result.status == "unanimous"
    assert result.decision == "Adopt proposal"
    assert result.rounds == 1
    assert result.dissent == []
    assert result.evidence_trail == ["A1", "A2", "D1", "R1"]
    assert len(result.transcript) == 2
    assert result_shape["status"] == "unanimous"
    assert result_shape["agents_count"] == 3
    assert len(result_shape["transcript"][1]["positions"]) == 3
    assert result_shape["tokens_spent"] > 0
    assert result_shape["tokens_estimated"] is True
    assert result_shape["transcript"][0]["tokens_spent"] > 0
    assert result_shape["transcript"][0]["positions"][0]["tokens"]["estimated"] is True


def test_token_budget_abort_preserves_partial_transcript() -> None:
    llm = ScriptedLlm([position("Adopt"), position("Delay")])

    result = ConsensusEngine(llm, agents_count=2, max_rounds=0, budget_tokens=1).decide("Should we adopt?")

    assert result.status == "aborted"
    assert result.abort_reason == "token_budget_exceeded"
    assert result.rounds_completed == 0
    assert result.tokens_spent > 1
    assert result.tokens_estimated is True
    assert len(result.transcript) == 1
    assert result.transcript[0].tokens_spent == result.tokens_spent
    assert result.transcript[0].positions[0].tokens is not None
    assert result.transcript[0].positions[0].tokens.estimated is True


def test_real_token_usage_beats_estimator() -> None:
    class UsageLlm:
        def __init__(self) -> None:
            self.outputs = [position("Adopt"), position("Adopt")]
            self.last_usage: dict[str, int | bool] | None = None

        def complete(
            self,
            prompt: str,
            *,
            schema: dict[str, Any] | None = None,
            model: str | None = None,
        ) -> dict[str, Any]:
            del prompt, schema, model
            self.last_usage = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "estimated": False}
            return self.outputs.pop(0)

    result = ConsensusEngine(UsageLlm(), agents_count=2, max_rounds=0).decide("Q?")

    assert result.status == "unanimous"
    assert result.tokens_spent == 10
    assert result.tokens_estimated is False
    assert result.transcript[0].positions[0].tokens is not None
    assert result.transcript[0].positions[0].tokens.total_tokens == 5


def test_models_are_cycled_across_agent_seats() -> None:
    llm = ScriptedLlm([position("Adopt") for _ in range(5)])

    result = ConsensusEngine(llm, agents_count=5, max_rounds=0, models=["cheap", "strong"]).decide("Q?")

    assert result.status == "unanimous"
    assert [call["model"] for call in llm.calls] == ["cheap", "strong", "cheap", "strong", "cheap"]
    assert [position.model for position in result.transcript[0].positions] == [
        "cheap",
        "strong",
        "cheap",
        "strong",
        "cheap",
    ]


def test_empty_model_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="models entries must be non-empty"):
        ConsensusEngine(ScriptedLlm([]), models=["cheap", " "])


def test_no_unanimous_consensus_records_dissent() -> None:
    llm = ScriptedLlm(
        [
            position("Adopt proposal", ["A1"]),
            position("Delay proposal", ["D1"]),
            position("Reject proposal", ["R1"]),
            update("Adopt proposal", maintained=[{"point": "Adopt", "reason": "Evidence A still dominates"}]),
            update("Delay proposal", maintained=[{"point": "Delay", "reason": "Evidence D still dominates"}]),
            update("Reject proposal", maintained=[{"point": "Reject", "reason": "Evidence R still dominates"}]),
        ]
    )

    result = ConsensusEngine(llm, max_rounds=1).decide("Should we adopt the proposal?")

    assert result.status == "no_consensus"
    assert result.decision is None
    assert result.majority_stance is None
    assert len(result.dissent) == 3
    assert {dissent.stance for dissent in result.dissent} == {
        "Adopt proposal",
        "Delay proposal",
        "Reject proposal",
    }
    assert len(result.points_of_disagreement) == 3


def test_no_consensus_majority_stance_requires_strict_majority() -> None:
    plurality = ConsensusEngine(
        ScriptedLlm([position("ship"), position("ship"), position("wait"), position("rewrite")]),
        agents_count=4,
        max_rounds=0,
    ).decide("Should we ship?")
    majority = ConsensusEngine(
        ScriptedLlm([position("ship"), position("ship"), position("ship"), position("wait")]),
        agents_count=4,
        max_rounds=0,
    ).decide("Should we ship?")

    assert plurality.majority_stance is None
    assert majority.majority_stance == "ship"


def test_conceding_requires_a_stated_reason() -> None:
    llm = ScriptedLlm(
        [
            position("Adopt proposal"),
            position("Delay proposal"),
            position("Reject proposal"),
            update("Adopt proposal", maintained=[{"point": "Adopt", "reason": "Evidence remains strongest"}]),
            update("Adopt proposal", conceded=[{"point": "Delay", "reason": ""}]),
        ]
    )

    with pytest.raises(ConsensusProtocolError, match="without a reason"):
        ConsensusEngine(llm, max_rounds=1).decide("Should we adopt the proposal?")


def test_convergence_is_decided_by_code_not_claimed_agreement() -> None:
    llm = ScriptedLlm(
        [
            position("Adopt proposal"),
            position("Delay proposal"),
            position("Reject proposal"),
            update("Adopt proposal", maintained=[{"point": "Adopt", "reason": "I still hold this stance"}]),
            update("Delay proposal", maintained=[{"point": "Delay", "reason": "I still hold this stance"}]),
            update("Reject proposal", maintained=[{"point": "Reject", "reason": "I still hold this stance"}]),
        ]
    )

    result = ConsensusEngine(llm, max_rounds=1).decide("Should we adopt the proposal?")

    assert result.status == "no_consensus"
    assert "agreement was not fabricated" in result.rationale


def test_normalized_stances_can_reach_round_zero_unanimity() -> None:
    llm = ScriptedLlm(
        [
            position("Ship it!"),
            position("ship it"),
            position("SHIP IT."),
        ]
    )

    result = ConsensusEngine(llm).decide("Should we ship?")

    assert result.status == "unanimous"
    assert result.rounds == 0
    assert result.decision == "Ship it!"
    assert normalize_stance("Ship it!") == normalize_stance("SHIP IT.")
    assert len(llm.calls) == 3


def test_callable_adapter_parses_structured_json_text() -> None:
    adapter = CallableLlmAdapter(lambda _prompt: '{"stance":"Adopt","rationale":"R","evidence":["E"],"confidence":1}')

    result = ConsensusEngine(adapter, agents_count=2).decide("Question?")

    assert result.status == "unanimous"
    assert result.decision == "Adopt"


def test_confidence_overflow_maps_to_finite_number_error() -> None:
    with pytest.raises(ConsensusProtocolError, match="confidence must be a finite number"):
        consensus_module._parse_position(
            {"stance": "A", "rationale": "R", "evidence": ["E"], "confidence": 10**400},
            agent_id="agent-1",
            debate=False,
        )


def test_confidence_json_1e999_maps_to_finite_number_error() -> None:
    with pytest.raises(ConsensusProtocolError, match="confidence must be a finite number"):
        consensus_module._parse_position(
            {"stance": "A", "rationale": "R", "evidence": ["E"], "confidence": float("1e999")},
            agent_id="agent-1",
            debate=False,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_confidence_nan_inf_map_to_finite_number_error(value: float) -> None:
    with pytest.raises(ConsensusProtocolError, match="confidence must be a finite number"):
        consensus_module._parse_position(
            {"stance": "A", "rationale": "R", "evidence": ["E"], "confidence": value},
            agent_id="agent-1",
            debate=False,
        )


def test_confidence_non_numeric_keeps_must_be_numeric_message() -> None:
    with pytest.raises(ConsensusProtocolError, match="confidence must be numeric"):
        consensus_module._parse_position(
            {"stance": "A", "rationale": "R", "evidence": ["E"], "confidence": "high"},
            agent_id="agent-1",
            debate=False,
        )


def test_confidence_finite_out_of_range_is_preserved() -> None:
    position = consensus_module._parse_position(
        {"stance": "A", "rationale": "R", "evidence": ["E"], "confidence": 1.5},
        agent_id="agent-1",
        debate=False,
    )
    assert position.confidence == 1.5
    position = consensus_module._parse_position(
        {"stance": "A", "rationale": "R", "evidence": ["E"], "confidence": -0.25},
        agent_id="agent-1",
        debate=False,
    )
    assert position.confidence == -0.25


def test_confidence_ordinary_finite_values() -> None:
    position = consensus_module._parse_position(
        {"stance": "A", "rationale": "R", "evidence": ["E"], "confidence": 0.75},
        agent_id="agent-1",
        debate=False,
    )
    assert position.confidence == 0.75


def _agent_id_from_prompt(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Agent: "):
            return line.removeprefix("Agent: ").strip()
    pytest.fail("prompt missing Agent line")


class AgentKeyedScriptedLlm:
    """Scripted LLM keyed by agent id so parallel rounds stay deterministic."""

    def __init__(
        self,
        phases: list[dict[str, dict[str, Any]]],
        *,
        latencies: dict[str, float] | None = None,
    ) -> None:
        self.phases = deque(phases)
        self.latencies = latencies or {}
        self._calls_in_phase = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self._lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        del model
        agent_id = _agent_id_from_prompt(prompt)
        with self._lock:
            if not self.phases:
                pytest.fail("scripted LLM exhausted")
            phase = self.phases[0]
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self._calls_in_phase += 1
            if self._calls_in_phase == len(phase):
                self.phases.popleft()
                self._calls_in_phase = 0
        try:
            latency = self.latencies.get(agent_id, 0.0)
            if latency:
                time.sleep(latency)
            return phase[agent_id]
        finally:
            with self._lock:
                self.active_calls -= 1


def test_parallel_round_matches_serial_outcome_and_order() -> None:
    phases = [
        {
            "agent-1": position("Adopt proposal", ["A1"]),
            "agent-2": position("Delay proposal", ["D1"]),
            "agent-3": position("Reject proposal", ["R1"]),
        },
        {
            "agent-1": update(
                "Adopt proposal",
                maintained=[{"point": "Adopt", "reason": "Mitigation evidence remains strongest"}],
                evidence=["A1", "A2"],
            ),
            "agent-2": update(
                "Adopt proposal",
                conceded=[{"point": "Delay", "reason": "The risk is directly mitigated"}],
                evidence=["D1", "A2"],
            ),
            "agent-3": update(
                "Adopt proposal",
                conceded=[{"point": "Reject", "reason": "The blocking concern has a bounded workaround"}],
                evidence=["R1", "A2"],
            ),
        },
    ]
    latencies = {"agent-1": 0.03, "agent-2": 0.01, "agent-3": 0.02}

    parallel_llm = AgentKeyedScriptedLlm(phases, latencies=latencies)
    parallel_result = ConsensusEngine(parallel_llm, max_rounds=2).decide("Should we adopt the proposal?")

    serial_outputs = [
        phases[0]["agent-1"],
        phases[0]["agent-2"],
        phases[0]["agent-3"],
        phases[1]["agent-1"],
        phases[1]["agent-2"],
        phases[1]["agent-3"],
    ]
    serial_llm = ScriptedLlm(serial_outputs)
    serial_result = ConsensusEngine(serial_llm, max_rounds=2).decide("Should we adopt the proposal?")

    assert parallel_result.status == serial_result.status == "unanimous"
    assert parallel_result.rounds == serial_result.rounds == 1
    assert parallel_result.decision == serial_result.decision == "Adopt proposal"
    assert parallel_result.evidence_trail == serial_result.evidence_trail

    parallel_positions = [p.agent_id for p in parallel_result.transcript[-1].positions]
    serial_positions = [p.agent_id for p in serial_result.transcript[-1].positions]
    assert parallel_positions == serial_positions == ["agent-1", "agent-2", "agent-3"]

    parallel_stances = [p.stance for p in parallel_result.transcript[-1].positions]
    serial_stances = [p.stance for p in serial_result.transcript[-1].positions]
    assert parallel_stances == serial_stances

    assert parallel_llm.max_active_calls >= 2


def test_parallel_worker_cap_leaves_cpu_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(consensus_module.os, "cpu_count", lambda: 3)
    phase = {f"agent-{index}": position("Adopt proposal", [f"E{index}"]) for index in range(1, 6)}
    llm = AgentKeyedScriptedLlm([phase], latencies={agent_id: 0.03 for agent_id in phase})

    result = ConsensusEngine(llm, agents_count=5, max_rounds=0).decide("Should we adopt?")

    assert result.status == "unanimous"
    assert llm.max_active_calls == 2
