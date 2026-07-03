"""Structured types for the host-agnostic Ultracode portable core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ConsensusStatus = Literal["unanimous", "no_consensus", "aborted"]
RoundPhase = Literal["independent", "debate"]


@dataclass(frozen=True)
class DebatePoint:
    """A maintained or conceded debate point with its explicit reason."""

    point: str
    reason: str


@dataclass(frozen=True)
class TokenUsage:
    """Token use for one LLM call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated: bool


@dataclass(frozen=True)
class AgentPosition:
    """One agent's structured stance in an independent or debate round."""

    agent_id: str
    stance: str
    rationale: str
    evidence: list[str]
    confidence: float
    conceded: list[DebatePoint] = field(default_factory=list)
    maintained: list[DebatePoint] = field(default_factory=list)
    model: str | None = None
    tokens: TokenUsage | None = None


@dataclass(frozen=True)
class ConsensusRound:
    """Transcript entry for one consensus round."""

    round_index: int
    phase: RoundPhase
    positions: list[AgentPosition]
    tokens_spent: int = 0


@dataclass(frozen=True)
class Dissent:
    """Final non-unanimous position from one agent."""

    agent_id: str
    stance: str
    rationale: str
    evidence: list[str]


@dataclass(frozen=True)
class ConsensusResult:
    """Final structured result returned by the consensus engine."""

    status: ConsensusStatus
    decision: str | None
    rationale: str
    rounds: int
    transcript: list[ConsensusRound]
    agents_count: int
    dissent: list[Dissent]
    evidence_trail: list[str] = field(default_factory=list)
    points_of_disagreement: list[str] = field(default_factory=list)
    majority_stance: str | None = None
    abort_reason: str | None = None
    rounds_completed: int | None = None
    tokens_spent: int = 0
    tokens_replayed: int = 0
    tokens_estimated: bool = False
    run_id: str | None = None
    journal_path: str | None = None
