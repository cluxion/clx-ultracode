"""Deterministic unanimous-consensus debate engine for the portable core."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cluxion_effort_ultracode.core.ports import LlmPort
from cluxion_effort_ultracode.core.types import (
    AgentPosition,
    ConsensusResult,
    ConsensusRound,
    DebatePoint,
    Dissent,
)

MAX_AGENTS = 8
MAX_ROUNDS = 8

POSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stance": {"type": "string"},
        "rationale": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["stance", "rationale", "evidence", "confidence"],
}

DEBATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **POSITION_SCHEMA["properties"],
        "conceded": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["point", "reason"],
            },
        },
        "maintained": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["point", "reason"],
            },
        },
    },
    "required": ["stance", "rationale", "evidence", "confidence", "conceded", "maintained"],
}


class ConsensusProtocolError(ValueError):
    """Raised when an LLM response violates the consensus debate protocol."""


class ConsensusEngine:
    """Run an adversarial debate until code-detected unanimity or honest dissent."""

    def __init__(
        self,
        llm: LlmPort,
        *,
        agents_count: int = 3,
        max_rounds: int = 3,
        rotate_devils_advocate: bool = True,
    ) -> None:
        if agents_count < 2:
            raise ValueError("agents_count must be at least 2")
        if agents_count > MAX_AGENTS:
            raise ValueError(f"agents_count must be <= {MAX_AGENTS}")
        if max_rounds < 0:
            raise ValueError("max_rounds must be non-negative")
        if max_rounds > MAX_ROUNDS:
            raise ValueError(f"max_rounds must be <= {MAX_ROUNDS}")
        self.llm = llm
        self.agents_count = agents_count
        self.max_rounds = max_rounds
        self.rotate_devils_advocate = rotate_devils_advocate

    def decide(self, question: str, *, context: str = "") -> ConsensusResult:
        """Run independent positions, debate revisions, and deterministic convergence checks."""

        transcript: list[ConsensusRound] = []
        current = self._initial_positions(question, context)
        transcript.append(ConsensusRound(round_index=0, phase="independent", positions=current))
        if self._is_unanimous(current):
            return self._unanimous_result(current, transcript, rounds=0)

        for round_index in range(1, self.max_rounds + 1):
            current = self._debate_round(question, context, round_index, current)
            transcript.append(ConsensusRound(round_index=round_index, phase="debate", positions=current))
            if self._is_unanimous(current):
                return self._unanimous_result(current, transcript, rounds=round_index)

        return self._no_consensus_result(current, transcript, rounds=self.max_rounds)

    def _initial_positions(self, question: str, context: str) -> list[AgentPosition]:
        def _run_agent(index: int) -> AgentPosition:
            agent_id = self._agent_id(index)
            prompt = self._build_initial_prompt(question, context, agent_id)
            raw = self.llm.complete(prompt, schema=POSITION_SCHEMA)
            return _parse_position(raw, agent_id=agent_id, debate=False)

        with ThreadPoolExecutor(max_workers=self.agents_count) as executor:
            futures = [executor.submit(_run_agent, index) for index in range(self.agents_count)]
            return [future.result() for future in futures]

    def _debate_round(
        self,
        question: str,
        context: str,
        round_index: int,
        previous: Sequence[AgentPosition],
    ) -> list[AgentPosition]:
        def _run_agent(index: int, prior: AgentPosition) -> AgentPosition:
            prompt = self._build_debate_prompt(
                question=question,
                context=context,
                round_index=round_index,
                agent_id=prior.agent_id,
                positions=previous,
                devil_advocate=index == ((round_index - 1) % self.agents_count)
                if self.rotate_devils_advocate
                else False,
            )
            raw = self.llm.complete(prompt, schema=DEBATE_SCHEMA)
            position = _parse_position(raw, agent_id=prior.agent_id, debate=True)
            self._validate_debate_update(prior, position)
            return position

        with ThreadPoolExecutor(max_workers=self.agents_count) as executor:
            futures = [
                executor.submit(_run_agent, index, prior) for index, prior in enumerate(previous)
            ]
            return [future.result() for future in futures]

    def _validate_debate_update(self, prior: AgentPosition, position: AgentPosition) -> None:
        if not position.conceded and not position.maintained:
            raise ConsensusProtocolError(
                f"{position.agent_id} must either concede specific points or maintain/rebut with reasons"
            )
        for point in [*position.conceded, *position.maintained]:
            if not point.point.strip() or not point.reason.strip():
                raise ConsensusProtocolError(f"{position.agent_id} returned a debate point without a reason")
        if normalize_stance(prior.stance) != normalize_stance(position.stance) and not position.conceded:
            raise ConsensusProtocolError(
                f"{position.agent_id} changed stance without conceding a specific point and reason"
            )

    def _is_unanimous(self, positions: Sequence[AgentPosition]) -> bool:
        if not positions:
            return False
        stances = {normalize_stance(position.stance) for position in positions}
        return len(stances) == 1 and "" not in stances

    def _unanimous_result(
        self,
        positions: Sequence[AgentPosition],
        transcript: list[ConsensusRound],
        *,
        rounds: int,
    ) -> ConsensusResult:
        decision = positions[0].stance
        evidence = _merge_evidence(positions)
        rationale = "Unanimous stance reached by deterministic stance normalization.\n" + "\n".join(
            f"{position.agent_id}: {position.rationale}" for position in positions
        )
        return ConsensusResult(
            status="unanimous",
            decision=decision,
            rationale=rationale,
            rounds=rounds,
            transcript=transcript,
            agents_count=self.agents_count,
            dissent=[],
            evidence_trail=evidence,
        )

    def _no_consensus_result(
        self,
        positions: Sequence[AgentPosition],
        transcript: list[ConsensusRound],
        *,
        rounds: int,
    ) -> ConsensusResult:
        normalized_to_display: dict[str, str] = {}
        normalized_counts: Counter[str] = Counter()
        for position in positions:
            normalized = normalize_stance(position.stance)
            normalized_counts[normalized] += 1
            normalized_to_display.setdefault(normalized, position.stance)

        most = normalized_counts.most_common(2)
        if len(most) >= 2 and most[0][1] > most[1][1]:
            majority_normalized = most[0][0]
            majority = normalized_to_display[majority_normalized]
        else:
            majority = None
        points = [
            f"{display}: {', '.join(p.agent_id for p in positions if normalize_stance(p.stance) == normalized)}"
            for normalized, display in normalized_to_display.items()
        ]
        dissent = [
            Dissent(
                agent_id=position.agent_id,
                stance=position.stance,
                rationale=position.rationale,
                evidence=list(position.evidence),
            )
            for position in positions
        ]
        return ConsensusResult(
            status="no_consensus",
            decision=None,
            rationale="No unanimous consensus after max_rounds; agreement was not fabricated.",
            rounds=rounds,
            transcript=transcript,
            agents_count=self.agents_count,
            dissent=dissent,
            evidence_trail=_merge_evidence(positions),
            points_of_disagreement=points,
            majority_stance=majority,
        )

    def _build_initial_prompt(self, question: str, context: str, agent_id: str) -> str:
        return (
            f"Agent: {agent_id}\n"
            "Round: 0 independent position\n"
            "You must decide your own position without seeing or inferring any other agent's answer.\n"
            "Return a structured object with stance, rationale, evidence, and confidence.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context or '(none)'}"
        )

    def _build_debate_prompt(
        self,
        *,
        question: str,
        context: str,
        round_index: int,
        agent_id: str,
        positions: Sequence[AgentPosition],
        devil_advocate: bool,
    ) -> str:
        role = (
            "Temporary role: devil's advocate. Prefer the strongest evidence, not agreement.\n"
            if devil_advocate
            else ""
        )
        return (
            f"Agent: {agent_id}\n"
            f"Round: {round_index} adversarial debate revision\n"
            f"{role}"
            "You see all current positions below. You must either rebut/maintain specific points with stronger "
            "reasons or concede specific points with reasons. Agreement without a concession reason is invalid.\n"
            "Return stance, rationale, evidence, confidence, conceded[{point, reason}], and "
            "maintained[{point, reason}].\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"Current positions:\n{_format_positions(positions)}"
        )

    def _agent_id(self, index: int) -> str:
        return f"agent-{index + 1}"


def normalize_stance(stance: str) -> str:
    """Normalize stance text for deterministic code-level unanimity checks."""

    normalized = unicodedata.normalize("NFKC", stance).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_position(raw: Mapping[str, Any] | str, *, agent_id: str, debate: bool) -> AgentPosition:
    data = _coerce_mapping(raw, agent_id=agent_id)
    required = ("stance", "rationale", "evidence", "confidence")
    missing = [key for key in required if key not in data]
    if missing:
        raise ConsensusProtocolError(f"{agent_id} response missing required fields: {', '.join(missing)}")

    evidence = data["evidence"]
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ConsensusProtocolError(f"{agent_id} evidence must be a list of strings")
    clean_evidence = [item.strip() for item in evidence if item.strip()]
    if not clean_evidence:
        raise ConsensusProtocolError(f"{agent_id} evidence must include at least one non-empty item")

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ConsensusProtocolError(f"{agent_id} confidence must be numeric") from exc

    conceded = _parse_debate_points(data.get("conceded", []), agent_id=agent_id, field_name="conceded")
    maintained = _parse_debate_points(data.get("maintained", []), agent_id=agent_id, field_name="maintained")
    if not debate and (conceded or maintained):
        raise ConsensusProtocolError(f"{agent_id} round-0 position must not include debate concessions")

    return AgentPosition(
        agent_id=agent_id,
        stance=_require_text(data["stance"], agent_id=agent_id, field_name="stance"),
        rationale=_require_text(data["rationale"], agent_id=agent_id, field_name="rationale"),
        evidence=clean_evidence,
        confidence=confidence,
        conceded=conceded,
        maintained=maintained,
    )


def _coerce_mapping(raw: Mapping[str, Any] | str, *, agent_id: str) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConsensusProtocolError(f"{agent_id} returned non-JSON text for structured output") from exc
        if not isinstance(loaded, dict):
            raise ConsensusProtocolError(f"{agent_id} returned JSON that is not an object")
        return loaded
    if not isinstance(raw, Mapping):
        raise ConsensusProtocolError(f"{agent_id} returned unsupported response type: {type(raw).__name__}")
    return raw


def _parse_debate_points(raw: Any, *, agent_id: str, field_name: str) -> list[DebatePoint]:
    if not isinstance(raw, list):
        raise ConsensusProtocolError(f"{agent_id} {field_name} must be a list")
    points: list[DebatePoint] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ConsensusProtocolError(f"{agent_id} {field_name}[{index}] must be an object")
        point = _require_text(item.get("point", ""), agent_id=agent_id, field_name=f"{field_name}[{index}].point")
        reason = _require_text(
            item.get("reason", ""),
            agent_id=agent_id,
            field_name=f"{field_name}[{index}].reason",
            allow_empty=True,
        )
        points.append(DebatePoint(point=point, reason=reason))
    return points


def _require_text(value: Any, *, agent_id: str, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConsensusProtocolError(f"{agent_id} {field_name} must be a string")
    stripped = value.strip()
    if not allow_empty and not stripped:
        raise ConsensusProtocolError(f"{agent_id} {field_name} must not be empty")
    return stripped


def _merge_evidence(positions: Sequence[AgentPosition]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for position in positions:
        for item in position.evidence:
            key = item.strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(key)
    return merged


def _format_positions(positions: Sequence[AgentPosition]) -> str:
    blocks = []
    for position in positions:
        evidence = "\n".join(f"- {item}" for item in position.evidence) or "- (none)"
        blocks.append(
            f"{position.agent_id}\n"
            f"stance: {position.stance}\n"
            f"rationale: {position.rationale}\n"
            f"evidence:\n{evidence}\n"
            f"confidence: {position.confidence:.2f}"
        )
    return "\n\n---\n\n".join(blocks)
