"""Command line interface for cluxion Effort-Ultracode."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.adapters import (
    CallableLlmAdapter,
    CodexExecutableNotFoundError,
    HermesExecutableNotFoundError,
    HermesLlmError,
)
from cluxion_effort_ultracode.core import ConsensusEngine, ConsensusProtocolError
from cluxion_effort_ultracode.core.consensus import (
    DEFAULT_AGENT_TIMEOUT_S,
    DEFAULT_DEBATE_BUDGET_S,
    MAX_AGENTS,
    MAX_ROUNDS,
)
from cluxion_effort_ultracode.core.errors import require_positive_finite, require_utf8_text, validation_error_code
from cluxion_effort_ultracode.core.journal import (
    DebateJournal,
    JournalBusy,
    JournalCorrupt,
    JournaledLlm,
    JournalLockUnsupported,
    LazyLlm,
    ResumeMismatch,
    ResumeNotFound,
    build_header,
    journals_dir,
    new_run_id,
    read_records,
)
from cluxion_effort_ultracode.core.journal_lifecycle import gc_journals, list_journals
from cluxion_effort_ultracode.doctor import render_json, render_text, run_doctor


@dataclass(frozen=True)
class _ConsensusConfig:
    question: str
    context: str
    rounds: int
    agents: int
    agent_timeout: float
    debate_budget: float
    budget_tokens: int | None
    models: list[str] | None
    adapter: str
    journal: DebateJournal


class _ScriptedConsensusLlm:
    def __init__(self, outputs: list[Mapping[str, Any]]) -> None:
        self.outputs = deque(outputs)

    def complete(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> Mapping[str, Any]:
        del model
        if not self.outputs:
            raise ConsensusProtocolError("mock adapter exhausted before consensus engine completed")
        return self.outputs.popleft()


def main(argv: list[str] | None = None) -> int:
    """Run the cluxion-ultracode command."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--version", "-V"):
        from cluxion_effort_ultracode import __version__

        print(f"cluxion-ultracode {__version__}")
        return 0
    parser = _build_parser()
    namespace = parser.parse_args(args)
    if namespace.command == "consensus":
        return _run_consensus(namespace)
    if namespace.command == "doctor":
        return _doctor(namespace)
    if namespace.command == "journals":
        return _journals(namespace)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cluxion-ultracode")
    subparsers = parser.add_subparsers(dest="command")

    consensus = subparsers.add_parser(
        "consensus",
        help="Run an adversarial unanimous-consensus debate",
        description=("Fan-out: agents * (rounds + 1) logical adapter calls; structured repair can add provider calls."),
    )
    consensus.add_argument("--question", help="Decision, proposal, or question to decide; use '-' to read stdin")
    consensus.add_argument("--question-file", help="Read the decision question from a UTF-8 text file")
    consensus.add_argument("--resume", help="Replay/continue a saved journal run_id")
    consensus.add_argument("--context", default=None, help="Optional context supplied to every agent")
    consensus.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=f"Maximum debate rounds after round 0, default 3, capped at {MAX_ROUNDS}",
    )
    consensus.add_argument("--agents", default=None, help=f"Number of agents, default 3, capped at {MAX_AGENTS}")
    consensus.add_argument(
        "--agent-timeout",
        type=float,
        default=None,
        help=f"Per-agent timeout in seconds, default {DEFAULT_AGENT_TIMEOUT_S:g}",
    )
    consensus.add_argument(
        "--debate-budget",
        type=float,
        default=None,
        help=f"Total debate budget in seconds across all rounds, default {DEFAULT_DEBATE_BUDGET_S:g}",
    )
    consensus.add_argument("--budget-tokens", type=int, default=None, help="Optional total token ceiling")
    consensus.add_argument("--models", default=None, help="Comma-separated per-agent models, cycled across agents")
    consensus.add_argument(
        "--adapter",
        choices=["hermes", "codex", "mock-unanimous", "mock-no-consensus"],
        default=None,
        help=(
            "LLM adapter: hermes uses the host ultracode-llm bridge, codex runs codex exec; "
            "mock-* adapters are deterministic for local testing."
        ),
    )
    doctor = subparsers.add_parser("doctor", help="Run embedded health checks")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--verbose", action="store_true")

    journals = subparsers.add_parser("journals", help="Inspect or clean debate journals")
    journal_commands = journals.add_subparsers(dest="journal_command", required=True)
    journal_commands.add_parser("list", help="List recorded debate journals")
    show = journal_commands.add_parser("show", help="Show one journal")
    show.add_argument("run_id")
    gc = journal_commands.add_parser("gc", help="Garbage-collect old journals")
    gc.add_argument("--older-than-days", type=int, default=7)
    gc.add_argument("--apply", action="store_true")
    return parser


def _run_consensus(namespace: argparse.Namespace) -> int:
    journal_info: dict[str, object] = {}
    config: _ConsensusConfig | None = None
    try:
        config = _prepare_consensus(namespace)
        journal_info = {"run_id": config.journal.run_id, "journal_path": str(config.journal.path)}
        adapter = LazyLlm(
            lambda: _resolve_adapter(
                config.adapter,
                agents=config.agents,
                rounds=config.rounds,
                timeout_seconds=config.agent_timeout,
            )
        )
        journaled = JournaledLlm(adapter, config.journal)
        engine = ConsensusEngine(
            journaled,
            agents_count=config.agents,
            max_rounds=config.rounds,
            agent_timeout_s=config.agent_timeout,
            debate_budget_s=config.debate_budget,
            budget_tokens=config.budget_tokens,
            models=config.models,
            progress_callback=lambda round_index, phase: print(f"round {round_index} {phase} start", file=sys.stderr),
        )
        result = engine.decide(config.question, context=config.context)
        config.journal.append_result(result)
    except JournalBusy as exc:
        print(json.dumps({"ok": False, "error": "journal_busy", "run_id": exc.run_id}, ensure_ascii=False))
        return 1
    except JournalLockUnsupported as exc:
        print(
            json.dumps(
                {"ok": False, "error": "journal_lock_unsupported", "message": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1
    except ResumeMismatch as exc:
        print(json.dumps({"ok": False, "error": "resume_mismatch", "fields": exc.fields}, ensure_ascii=False))
        return 1
    except ResumeNotFound as exc:
        print(json.dumps({"ok": False, "error": "journal_not_found", "run_id": str(exc)}, ensure_ascii=False))
        return 1
    except JournalCorrupt as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "journal_corrupt",
                    "run_id": exc.run_id,
                    "message": str(exc),
                    **journal_info,
                },
                ensure_ascii=False,
            )
        )
        return 1
    except HermesLlmError as exc:
        print(
            json.dumps(
                {"ok": False, "error": exc.code, "message": exc.message, **journal_info},
                ensure_ascii=False,
            )
        )
        return 1
    except HermesExecutableNotFoundError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "hermes_not_found",
                    "message": str(exc),
                    **journal_info,
                    "hint": (
                        "Ensure the hermes executable is on PATH, or configure CLUXION_EFFORT_ULTRACODE_HERMES_BINARY."
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 1
    except CodexExecutableNotFoundError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "codex_not_found",
                    "message": str(exc),
                    **journal_info,
                    "hint": (
                        "Ensure the codex executable is on PATH, or configure CLUXION_EFFORT_ULTRACODE_CODEX_BINARY."
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 1
    except (ConsensusProtocolError, ValueError) as exc:
        error = type(exc).__name__ if isinstance(exc, ConsensusProtocolError) else validation_error_code(exc)
        print(
            json.dumps(
                {"ok": False, "error": error, "message": str(exc), **journal_info},
                ensure_ascii=False,
            )
        )
        return 1
    finally:
        if config is not None:
            config.journal.close()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


def _prepare_consensus(namespace: argparse.Namespace) -> _ConsensusConfig:
    journal: DebateJournal | None = None
    try:
        if namespace.resume:
            # Claim resume first (open+lock), then derive defaults from the locked header.
            journal = DebateJournal.resume(namespace.resume)
            saved = journal.header
        else:
            saved = None
        question = _question_arg(namespace, saved)
        context = namespace.context if namespace.context is not None else str((saved or {}).get("context", ""))
        require_utf8_text(question, "question")
        require_utf8_text(context, "context")
        rounds = int(_saved_or(namespace.rounds, saved, "max_rounds", 3))
        agents = _bounded_int(_saved_or(namespace.agents, saved, "agents_count", 3), "agents_count", 2, MAX_AGENTS)
        budget_tokens = _optional_int(_saved_or(namespace.budget_tokens, saved, "budget_tokens", None))
        agent_timeout, debate_budget = _validate_pre_journal(
            rounds,
            _saved_or(namespace.agent_timeout, saved, "agent_timeout_s", DEFAULT_AGENT_TIMEOUT_S),
            _saved_or(namespace.debate_budget, saved, "debate_budget_s", DEFAULT_DEBATE_BUDGET_S),
            budget_tokens,
        )
        models = _parse_models(namespace.models) if namespace.models is not None else _saved_models(saved)
        adapter = namespace.adapter or str((saved or {}).get("adapter") or "hermes")
        run_id = namespace.resume or new_run_id()
        header = build_header(
            run_id=run_id,
            question=question,
            context=context,
            agents_count=agents,
            max_rounds=rounds,
            models=models or [],
            adapter=adapter,
            agent_timeout_s=agent_timeout,
            debate_budget_s=debate_budget,
            budget_tokens=budget_tokens,
        )
        if journal is not None:
            journal.ensure_matches(header)
        else:
            journal = DebateJournal.start(header)
        return _ConsensusConfig(
            question=question,
            context=context,
            rounds=rounds,
            agents=agents,
            agent_timeout=agent_timeout,
            debate_budget=debate_budget,
            budget_tokens=budget_tokens,
            models=models,
            adapter=adapter,
            journal=journal,
        )
    except Exception:
        if journal is not None:
            journal.close()
        raise


def _saved_or(value: object, saved: Mapping[str, object] | None, key: str, default: object) -> object:
    if value is not None:
        return value
    if saved is not None and key in saved:
        return saved[key]
    return default


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _question_arg(namespace: argparse.Namespace, saved: Mapping[str, object] | None) -> str:
    if namespace.question_file and namespace.question is not None:
        raise ValueError("use either --question or --question-file, not both")
    if namespace.question_file:
        try:
            question = Path(namespace.question_file).read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise ValueError("question file is not valid UTF-8") from exc
        except OSError as exc:
            raise ValueError(f"question file could not be read: {exc}") from exc
    elif namespace.question == "-":
        question = sys.stdin.read()
    elif namespace.question is not None:
        question = namespace.question
    else:
        question = str((saved or {}).get("question", ""))
    question = question.strip()
    if not question:
        raise ValueError("question is required unless --resume points to a journal")
    return question


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    raw = str(value).strip()
    sign = -1 if raw.startswith("-") else 1
    digits = raw[1:] if raw[:1] in "+-" else raw
    if not digits.isdecimal():
        raise ValueError(f"{name} must be an integer")
    significant = digits.lstrip("0") or "0"
    if sign < 0 and significant != "0":
        raise ValueError(f"{name} must be at least {minimum}")
    if len(significant) > len(str(maximum)):
        raise ValueError(f"{name} must be <= {maximum}")
    parsed = sign * int(significant)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def _validate_pre_journal(
    rounds: int,
    agent_timeout: object,
    debate_budget: object,
    budget_tokens: int | None,
) -> tuple[float, float]:
    if rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if rounds > MAX_ROUNDS:
        raise ValueError(f"max_rounds must be <= {MAX_ROUNDS}")
    timeout = require_positive_finite(agent_timeout, "agent_timeout_s")
    budget = require_positive_finite(debate_budget, "debate_budget_s")
    if budget_tokens is not None and budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    return timeout, budget


def _saved_models(saved: Mapping[str, object] | None) -> list[str] | None:
    if saved is None:
        return None
    models = saved.get("models", [])
    return [str(model) for model in models] if isinstance(models, list) and models else None


def _resolve_adapter(
    name: str,
    *,
    agents: int,
    rounds: int,
    timeout_seconds: float,
) -> CallableLlmAdapter | _ScriptedConsensusLlm:
    if name in {"hermes", "codex"}:
        from cluxion_effort_ultracode.llm_factory import default_llm

        return default_llm(name, timeout_seconds=timeout_seconds)
    return _mock_adapter(name, agents=agents, rounds=rounds)


def _mock_adapter(name: str, *, agents: int, rounds: int) -> CallableLlmAdapter | _ScriptedConsensusLlm:
    if name == "mock-unanimous":
        return _ScriptedConsensusLlm(_mock_unanimous_outputs(agents))
    if name == "mock-no-consensus":
        return _ScriptedConsensusLlm(_mock_no_consensus_outputs(agents, rounds))
    raise ValueError(f"unknown adapter: {name}")


def _parse_models(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    models = [item.strip() for item in raw.split(",")]
    if any(not model for model in models):
        raise ValueError("models entries must be non-empty")
    return models


def _journals(namespace: argparse.Namespace) -> int:
    try:
        if namespace.journal_command == "list":
            payload = list_journals()
            warn_size = int(payload["warn_size_bytes"])
            if int(payload["total_bytes"]) > warn_size:
                print(
                    f"warning: journal directory exceeds {warn_size} bytes: {payload['total_bytes']}",
                    file=sys.stderr,
                )
        elif namespace.journal_command == "show":
            run_id = namespace.run_id
            payload = {"run_id": run_id, "records": read_records(journals_dir() / f"{run_id}.jsonl")}
        elif namespace.journal_command == "gc":
            if namespace.older_than_days < 0:
                raise ValueError("--older-than-days must be non-negative")
            payload = gc_journals(older_than_days=namespace.older_than_days, apply=namespace.apply)
        else:
            raise ValueError(f"unknown journals command: {namespace.journal_command}")
    except ResumeNotFound as exc:
        print(json.dumps({"ok": False, "error": "journal_not_found", "run_id": str(exc)}, ensure_ascii=False))
        return 1
    except JournalCorrupt as exc:
        print(
            json.dumps(
                {"ok": False, "error": "journal_corrupt", "run_id": exc.run_id, "message": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1
    except JournalLockUnsupported as exc:
        print(
            json.dumps(
                {"ok": False, "error": "journal_lock_unsupported", "message": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": validation_error_code(exc), "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _position(stance: str, rationale: str, evidence: list[str], confidence: float = 0.75) -> dict[str, Any]:
    return {"stance": stance, "rationale": rationale, "evidence": evidence, "confidence": confidence}


def _update(
    stance: str,
    rationale: str,
    evidence: list[str],
    *,
    conceded: list[dict[str, str]] | None = None,
    maintained: list[dict[str, str]] | None = None,
    confidence: float = 0.82,
) -> dict[str, Any]:
    return {
        **_position(stance, rationale, evidence, confidence),
        "conceded": conceded or [],
        "maintained": maintained or [],
    }


def _mock_unanimous_outputs(agents: int) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index in range(agents):
        outputs.append(
            _position(
                "Adopt proposal",
                "The deterministic mock starts unanimous for local smoke tests.",
                [f"E{index + 1}"],
            )
        )
    return outputs


def _mock_no_consensus_outputs(agents: int, rounds: int) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    stances = ["Adopt proposal", "Delay proposal", "Reject proposal"]
    for index in range(agents):
        stance = stances[index % len(stances)]
        outputs.append(_position(stance, f"{stance} has the strongest local evidence.", [f"E{index + 1}"]))
    for _round_index in range(rounds):
        for index in range(agents):
            stance = stances[index % len(stances)]
            outputs.append(
                _update(
                    stance,
                    f"Maintaining {stance} because the contrary evidence remains weaker.",
                    [f"E{index + 1}"],
                    maintained=[{"point": stance, "reason": "No opposing evidence outweighed it"}],
                )
            )
    return outputs


def _doctor(namespace):
    pkg = "cluxion_effort_ultracode.doctor"
    catalog_path = Path(str(importlib.resources.files(pkg).joinpath("catalog.json")))
    result = run_doctor(
        cwd=Path.cwd(),
        hermes_bin="hermes",
        catalog_path=catalog_path,
        probes=__import__("cluxion_effort_ultracode.doctor.probes", fromlist=["PROBES"]).PROBES,
        plugin="effort-ultracode",
        version=__import__("cluxion_effort_ultracode").__version__,
    )
    if getattr(namespace, "json", False):
        print(render_json(result))
    else:
        text = render_text(
            result,
            __import__(
                "cluxion_effort_ultracode.doctor.framework",
                fromlist=["load_catalog"],
            ).load_catalog(catalog_path),
            verbose=bool(getattr(namespace, "verbose", False)),
        )
        print(text)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
