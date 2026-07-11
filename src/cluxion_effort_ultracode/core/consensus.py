"""Deterministic unanimous-consensus debate engine for the portable core."""

from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from cluxion_effort_ultracode.core.errors import require_positive_finite
from cluxion_effort_ultracode.core.ports import LlmPort
from cluxion_effort_ultracode.core.types import (
    AgentPosition,
    ConsensusResult,
    ConsensusRound,
    DebatePoint,
    Dissent,
    RoundPhase,
    TokenUsage,
)

MAX_AGENTS = 8
MAX_ROUNDS = 8
DEFAULT_AGENT_TIMEOUT_S = 180.0
DEFAULT_DEBATE_BUDGET_S = 600.0
MIN_QUORUM = 2

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


class _ConsensusAbort(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _PostCallDeadline(Exception):
    """Adapter call finished after debate deadline; tokens must be accounted once."""

    def __init__(self, tokens: TokenUsage) -> None:
        super().__init__("adapter call completed after debate deadline")
        self.tokens = tokens


class ConsensusEngine:
    """Run an adversarial debate until code-detected unanimity or honest dissent."""

    def __init__(
        self,
        llm: LlmPort,
        *,
        agents_count: int = 3,
        max_rounds: int = 3,
        rotate_devils_advocate: bool = True,
        agent_timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
        debate_budget_s: float = DEFAULT_DEBATE_BUDGET_S,
        budget_tokens: int | None = None,
        models: Sequence[str] | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
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
        agent_timeout_s = require_positive_finite(agent_timeout_s, "agent_timeout_s")
        debate_budget_s = require_positive_finite(debate_budget_s, "debate_budget_s")
        if budget_tokens is not None and budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        clean_models = _validate_models(models)
        self.agent_timeout_s = agent_timeout_s
        self.debate_budget_s = debate_budget_s
        self.budget_tokens = budget_tokens
        self.models = clean_models
        self.progress_callback = progress_callback
        self._tokens_spent = 0
        self._tokens_estimated = False

    def decide(self, question: str, *, context: str = "") -> ConsensusResult:
        """Run independent positions, debate revisions, and deterministic convergence checks."""

        question = question.strip()
        if not question:
            raise ValueError("question must be non-empty")
        self._tokens_spent = 0
        self._tokens_estimated = False
        transcript: list[ConsensusRound] = []
        deadline = time.monotonic() + self.debate_budget_s
        try:
            self._emit_progress(0, "independent")
            current = self._initial_positions(question, context, deadline=deadline)
            self._append_round(transcript, round_index=0, phase="independent", positions=current)
        except _ConsensusAbort as exc:
            return self._aborted_result([], transcript, reason=exc.reason, rounds_completed=0)
        if self._token_budget_exceeded():
            return self._aborted_result(
                current,
                transcript,
                reason="token_budget_exceeded",
                rounds_completed=0,
            )
        if self._is_unanimous(current):
            if time.monotonic() >= deadline:
                return self._aborted_result(
                    current,
                    transcript,
                    reason=self._budget_reason(rounds_completed=0),
                    rounds_completed=0,
                )
            return self._unanimous_result(current, transcript, rounds=0)

        for round_index in range(1, self.max_rounds + 1):
            if time.monotonic() >= deadline:
                return self._aborted_result(
                    current,
                    transcript,
                    reason=self._budget_reason(rounds_completed=round_index - 1),
                    rounds_completed=round_index - 1,
                )
            self._emit_progress(round_index, "debate")
            try:
                current = self._debate_round(question, context, round_index, current, deadline=deadline)
                self._append_round(transcript, round_index=round_index, phase="debate", positions=current)
            except _ConsensusAbort as exc:
                return self._aborted_result(
                    current,
                    transcript,
                    reason=exc.reason,
                    rounds_completed=round_index - 1,
                )
            if self._token_budget_exceeded():
                return self._aborted_result(
                    current,
                    transcript,
                    reason="token_budget_exceeded",
                    rounds_completed=round_index,
                )
            if self._is_unanimous(current):
                if time.monotonic() >= deadline:
                    return self._aborted_result(
                        current,
                        transcript,
                        reason=self._budget_reason(rounds_completed=round_index),
                        rounds_completed=round_index,
                    )
                return self._unanimous_result(current, transcript, rounds=round_index)

        if time.monotonic() >= deadline:
            return self._aborted_result(
                current,
                transcript,
                reason=self._budget_reason(rounds_completed=self.max_rounds),
                rounds_completed=self.max_rounds,
            )
        return self._no_consensus_result(current, transcript, rounds=self.max_rounds)

    def _initial_positions(self, question: str, context: str, *, deadline: float) -> list[AgentPosition]:
        def _run_agent(index: int) -> AgentPosition:
            agent_id = self._agent_id(index)
            model = self._model_for(index)
            prompt = self._build_initial_prompt(question, context, agent_id)
            raw, tokens = self._complete(prompt, schema=POSITION_SCHEMA, model=model, deadline=deadline)
            return _parse_position(raw, agent_id=agent_id, debate=False, model=model, tokens=tokens)

        return self._gather_positions(
            [(index, None) for index in range(self.agents_count)],
            lambda index, _prior: _run_agent(index),
            deadline=deadline,
            phase="independent",
        )

    def _debate_round(
        self,
        question: str,
        context: str,
        round_index: int,
        previous: Sequence[AgentPosition],
        *,
        deadline: float,
    ) -> list[AgentPosition]:
        def _run_agent(index: int, prior: AgentPosition) -> AgentPosition:
            model = self._model_for(index)
            prompt = self._build_debate_prompt(
                question=question,
                context=context,
                round_index=round_index,
                agent_id=prior.agent_id,
                positions=previous,
                devil_advocate=index == ((round_index - 1) % len(previous)) if self.rotate_devils_advocate else False,
            )
            raw, tokens = self._complete(prompt, schema=DEBATE_SCHEMA, model=model, deadline=deadline)
            position = _parse_position(raw, agent_id=prior.agent_id, debate=True, model=model, tokens=tokens)
            self._validate_debate_update(prior, position)
            return position

        return self._gather_positions(
            list(enumerate(previous)),
            _run_agent,
            deadline=deadline,
            phase=f"debate round {round_index}",
        )

    def _complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any],
        model: str | None,
        deadline: float | None = None,
    ) -> tuple[Mapping[str, Any] | str, TokenUsage]:
        if model is None:
            raw = self.llm.complete(prompt, schema=schema)
        else:
            try:
                raw = self.llm.complete(prompt, schema=schema, model=model)
            except TypeError as exc:
                raise ConsensusProtocolError("llm.complete must accept model= when models are configured") from exc
        tokens = _token_usage(prompt, raw, getattr(self.llm, "last_usage", None))
        # Check budget immediately after the adapter returns, before parse/validation,
        # so malformed output cannot mask a crossed debate deadline.
        if deadline is not None and time.monotonic() >= deadline:
            raise _PostCallDeadline(tokens)
        return raw, tokens

    def _gather_positions(self, tasks, run_agent, *, deadline: float, phase: str) -> list[AgentPosition]:
        """Collect agent positions with per-agent and total deadlines.

        Journaled CLI/plugin runs are serialized to preserve replay order; in that
        (production) mode an adapter timeout or completion error ABORTS the current
        invocation — it is not dropped-and-continued. Only the non-journaled parallel
        path drops a hung/failed agent and continues while at least MIN_QUORUM
        positions survive (worker count is capped to leave CPU headroom under high fan-out).
        """
        if hasattr(self.llm, "outputs") or getattr(self.llm, "serial_complete", False):
            return self._gather_positions_serial(tasks, run_agent, deadline=deadline, phase=phase)

        results: dict[int, AgentPosition] = {}
        failures: list[str] = []
        executor = ThreadPoolExecutor(max_workers=min(len(tasks), max(2, (os.cpu_count() or 4) - 2), MAX_AGENTS))
        try:
            futures = {executor.submit(run_agent, index, prior): index for index, prior in tasks}
            for future, index in futures.items():
                remaining = deadline - time.monotonic()
                per_agent = min(self.agent_timeout_s, max(0.1, remaining))
                try:
                    results[index] = future.result(timeout=per_agent)
                except FutureTimeoutError:
                    # Hangs are isolated: one stuck agent must not block siblings.
                    # Protocol violations and backend errors still propagate -
                    # hiding them would fake a healthier debate than happened.
                    future.cancel()
                    failures.append(f"agent {index}: timed out after {per_agent:.0f}s in {phase}")
                except _PostCallDeadline as exc:
                    # Completed after debate budget: account this call + already-done siblings, then abort.
                    completed_calls = self._account_post_deadline_parallel(
                        exc,
                        futures=futures,
                        results=results,
                        current_index=index,
                    )
                    raise _ConsensusAbort(self._budget_reason_during(phase, completed_calls=completed_calls)) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if len(results) < MIN_QUORUM:
            detail = "; ".join(failures) or "no agent completed"
            raise _ConsensusAbort(f"quorum lost in {phase} ({len(results)}/{len(tasks)} survived): {detail}")
        return [results[index] for index in sorted(results)]

    def _gather_positions_serial(self, tasks, run_agent, *, deadline: float, phase: str) -> list[AgentPosition]:
        """Serial production path: check debate budget before/after every adapter call.

        Overrun is bounded to one in-flight logical call. A completed call that crosses
        the deadline before parse/validation (or before _append_round) contributes its
        tokens once without marking the incomplete round as completed.
        """
        results: list[AgentPosition] = []
        for index, prior in tasks:
            if time.monotonic() >= deadline:
                self._account_partial_tokens(results)
                raise _ConsensusAbort(self._budget_reason_during(phase, completed_calls=len(results)))
            try:
                position = run_agent(index, prior)
            except _PostCallDeadline as exc:
                # Call finished after deadline: account prior positions + this call once.
                self._account_partial_tokens(results)
                self._account_call_tokens(exc.tokens)
                raise _ConsensusAbort(self._budget_reason_during(phase, completed_calls=len(results) + 1)) from exc
            results.append(position)
            if time.monotonic() >= deadline:
                self._account_partial_tokens(results)
                raise _ConsensusAbort(self._budget_reason_during(phase, completed_calls=len(results)))
        return results

    def _account_partial_tokens(self, positions: Sequence[AgentPosition]) -> None:
        tokens_spent = sum(position.tokens.total_tokens for position in positions if position.tokens is not None)
        self._tokens_spent += tokens_spent
        self._tokens_estimated = self._tokens_estimated or any(
            position.tokens is not None and position.tokens.estimated for position in positions
        )

    def _account_call_tokens(self, tokens: TokenUsage) -> None:
        self._tokens_spent += tokens.total_tokens
        self._tokens_estimated = self._tokens_estimated or tokens.estimated

    def _account_post_deadline_parallel(
        self,
        exc: _PostCallDeadline,
        *,
        futures: Mapping[Any, int],
        results: Mapping[int, AgentPosition],
        current_index: int,
    ) -> int:
        """Account prior results + current deadline call + already-done siblings exactly once.

        Freezes the eligible already-done sibling list once at entry (before any token
        accounting) so a race that finishes mid-accounting cannot inflate the count.
        Only prior parsed results, the current ``_PostCallDeadline``, snapshotted
        successful ``AgentPosition``s, and snapshotted ``_PostCallDeadline`` siblings
        contribute tokens/completed_calls. Generic Exception/cancelled/unfinished
        siblings prove neither adapter completion nor usage. Never waits on unfinished
        futures. Returns known completed-call count for the user-visible abort reason
        (not parsed-position count).
        """
        # Snapshot once: only siblings already done at entry are eligible to drain.
        already_done = [
            (other_future, other_index)
            for other_future, other_index in futures.items()
            if other_index != current_index and other_index not in results and other_future.done()
        ]

        # Prior successful positions already stored in results.
        self._account_partial_tokens(list(results.values()))
        self._account_call_tokens(exc.tokens)
        completed_calls = len(results) + 1

        for other_future, _other_index in already_done:
            try:
                sibling = other_future.result(timeout=0)
            except _PostCallDeadline as sibling_exc:
                self._account_call_tokens(sibling_exc.tokens)
                completed_calls += 1
            except Exception:
                # Cancellation / timeout / pre-adapter failure: no tokens, no completed_calls.
                # Only adapter-proven outcomes (AgentPosition / _PostCallDeadline) count.
                pass
            else:
                self._account_partial_tokens([sibling])
                completed_calls += 1
        return completed_calls

    def _budget_reason(self, *, rounds_completed: int) -> str:
        return f"debate exceeded debate_budget_s={self.debate_budget_s:.0f}s after round {rounds_completed}"

    def _budget_reason_during(self, phase: str, *, completed_calls: int) -> str:
        return (
            f"debate exceeded debate_budget_s={self.debate_budget_s:.0f}s "
            f"during {phase} after {completed_calls} adapter call(s)"
        )

    def _append_round(
        self,
        transcript: list[ConsensusRound],
        *,
        round_index: int,
        phase: RoundPhase,
        positions: list[AgentPosition],
    ) -> None:
        tokens_spent = sum(position.tokens.total_tokens for position in positions if position.tokens is not None)
        self._tokens_spent += tokens_spent
        self._tokens_estimated = self._tokens_estimated or any(
            position.tokens is not None and position.tokens.estimated for position in positions
        )
        transcript.append(
            ConsensusRound(
                round_index=round_index,
                phase=phase,
                positions=positions,
                tokens_spent=tokens_spent,
            )
        )

    def _token_budget_exceeded(self) -> bool:
        return self.budget_tokens is not None and self._tokens_spent > self.budget_tokens

    def _model_for(self, index: int) -> str | None:
        if not self.models:
            return None
        return self.models[index % len(self.models)]

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
            rounds_completed=rounds,
            tokens_spent=self._tokens_spent,
            tokens_replayed=self._tokens_replayed(),
            tokens_estimated=self._tokens_estimated,
            run_id=self._run_id(),
            journal_path=self._journal_path(),
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

        most = normalized_counts.most_common(1)
        if most and most[0][1] > len(positions) / 2:
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
            rounds_completed=rounds,
            tokens_spent=self._tokens_spent,
            tokens_replayed=self._tokens_replayed(),
            tokens_estimated=self._tokens_estimated,
            run_id=self._run_id(),
            journal_path=self._journal_path(),
        )

    def _aborted_result(
        self,
        positions: Sequence[AgentPosition],
        transcript: list[ConsensusRound],
        *,
        reason: str,
        rounds_completed: int,
    ) -> ConsensusResult:
        return ConsensusResult(
            status="aborted",
            decision=None,
            rationale="Debate aborted before a final consensus result; partial transcript is preserved.",
            rounds=rounds_completed,
            transcript=transcript,
            agents_count=self.agents_count,
            dissent=[
                Dissent(
                    agent_id=position.agent_id,
                    stance=position.stance,
                    rationale=position.rationale,
                    evidence=list(position.evidence),
                )
                for position in positions
            ],
            evidence_trail=_merge_evidence(positions),
            abort_reason=reason,
            rounds_completed=rounds_completed,
            tokens_spent=self._tokens_spent,
            tokens_replayed=self._tokens_replayed(),
            tokens_estimated=self._tokens_estimated,
            run_id=self._run_id(),
            journal_path=self._journal_path(),
        )

    def _emit_progress(self, round_index: int, phase: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(round_index, phase)

    def _tokens_replayed(self) -> int:
        return int(getattr(self.llm, "tokens_replayed", 0) or 0)

    def _run_id(self) -> str | None:
        value = getattr(self.llm, "run_id", None)
        return str(value) if value is not None else None

    def _journal_path(self) -> str | None:
        value = getattr(self.llm, "journal_path", None)
        return str(value) if value is not None else None

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


def _validate_models(models: Sequence[str] | None) -> list[str]:
    if models is None:
        return []
    clean: list[str] = []
    for index, model in enumerate(models):
        if not isinstance(model, str):
            raise ValueError(f"models[{index}] must be a string")
        stripped = model.strip()
        if not stripped:
            raise ValueError("models entries must be non-empty")
        clean.append(stripped)
    return clean


def _token_usage(prompt: str, raw: Mapping[str, Any] | str, usage: object) -> TokenUsage:
    real = _real_token_usage(usage)
    if real is not None:
        return real
    real = _real_token_usage(raw.get("usage") if isinstance(raw, Mapping) else None)
    if real is not None:
        return real
    input_tokens = _estimate_tokens(prompt)
    output_tokens = _estimate_tokens(_response_text(raw))
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated=True,
    )


def _real_token_usage(value: object) -> TokenUsage | None:
    if not isinstance(value, Mapping):
        return None
    usage = value.get("usage")
    if isinstance(usage, Mapping):
        value = usage
    input_tokens = _int_token(value, "input_tokens", "prompt_tokens")
    output_tokens = _int_token(value, "output_tokens", "completion_tokens")
    total_tokens = _int_token(value, "total_tokens", "tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if total_tokens is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens if output_tokens is not None else max(0, total_tokens - input_tokens)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated=False,
    )


def _int_token(value: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
    return None


def _estimate_tokens(value: str) -> int:
    # ponytail: chars/4 estimator, replace with host-native usage when every adapter exposes it.
    return max(1, (len(value) + 3) // 4)


def _response_text(raw: Mapping[str, Any] | str) -> str:
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_stance(stance: str) -> str:
    """Normalize stance text for deterministic code-level unanimity checks."""

    normalized = unicodedata.normalize("NFKC", stance).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _parse_position(
    raw: Mapping[str, Any] | str,
    *,
    agent_id: str,
    debate: bool,
    model: str | None = None,
    tokens: TokenUsage | None = None,
) -> AgentPosition:
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
    except OverflowError as exc:
        raise ConsensusProtocolError(f"{agent_id} confidence must be a finite number") from exc
    if not math.isfinite(confidence):
        raise ConsensusProtocolError(f"{agent_id} confidence must be a finite number")

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
        model=model,
        tokens=tokens,
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
            empty_message=f"{agent_id} returned a debate point without a reason",
        )
        points.append(DebatePoint(point=point, reason=reason))
    return points


def _require_text(
    value: Any,
    *,
    agent_id: str,
    field_name: str,
    allow_empty: bool = False,
    empty_message: str | None = None,
) -> str:
    if not isinstance(value, str):
        raise ConsensusProtocolError(f"{agent_id} {field_name} must be a string")
    stripped = value.strip()
    if not allow_empty and not stripped:
        raise ConsensusProtocolError(empty_message or f"{agent_id} {field_name} must not be empty")
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
