"""Command line interface for cluxion Effort-Ultracode."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cluxion_effort_ultracode.adapters import CallableLlmAdapter
from cluxion_effort_ultracode.core import ConsensusEngine, ConsensusProtocolError
from cluxion_effort_ultracode.doctor import render_json, render_text, run_doctor


class _ScriptedConsensusLlm:
    def __init__(self, outputs: list[Mapping[str, Any]]) -> None:
        self.outputs = deque(outputs)

    def complete(self, prompt: str, *, schema: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
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
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cluxion-ultracode")
    subparsers = parser.add_subparsers(dest="command")

    consensus = subparsers.add_parser("consensus", help="Run an adversarial unanimous-consensus debate")
    consensus.add_argument("--question", required=True, help="Decision, proposal, or question to decide")
    consensus.add_argument("--context", default="", help="Optional context supplied to every agent")
    consensus.add_argument("--rounds", type=int, default=3, help="Maximum debate rounds after round 0")
    consensus.add_argument("--agents", type=int, default=3, help="Number of agents, default 3")
    consensus.add_argument(
        "--adapter",
        choices=["mock-unanimous", "mock-no-consensus"],
        required=True,
        help="LLM adapter to use. v0.1 ships deterministic mock adapters for local testing.",
    )
    doctor = subparsers.add_parser("doctor", help="Run embedded health checks")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--verbose", action="store_true")
    return parser


def _run_consensus(namespace: argparse.Namespace) -> int:
    try:
        adapter = _mock_adapter(namespace.adapter, agents=namespace.agents, rounds=namespace.rounds)
        engine = ConsensusEngine(adapter, agents_count=namespace.agents, max_rounds=namespace.rounds)
        result = engine.decide(namespace.question, context=namespace.context)
    except (ConsensusProtocolError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


def _mock_adapter(name: str, *, agents: int, rounds: int) -> CallableLlmAdapter | _ScriptedConsensusLlm:
    if name == "mock-unanimous":
        return _ScriptedConsensusLlm(_mock_unanimous_outputs(agents))
    if name == "mock-no-consensus":
        return _ScriptedConsensusLlm(_mock_no_consensus_outputs(agents, rounds))
    raise ValueError(f"unknown adapter: {name}")


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
        if index == 0:
            outputs.append(_position("Adopt proposal", "The proposal has direct evidence and bounded risk.", ["E1"]))
        else:
            outputs.append(_position("Delay proposal", "The initial concern is unresolved risk.", [f"E{index + 1}"]))
    for index in range(agents):
        if index == 0:
            outputs.append(
                _update(
                    "Adopt proposal",
                    "Maintaining adoption because the opposing risks are mitigated by the evidence trail.",
                    ["E1", "E-merged"],
                    maintained=[{"point": "Adoption remains justified", "reason": "Risk evidence was bounded"}],
                )
            )
        else:
            outputs.append(
                _update(
                    "Adopt proposal",
                    "Conceding the delay stance because the mitigation evidence is stronger.",
                    [f"E{index + 1}", "E-merged"],
                    conceded=[{"point": "Delay until more data", "reason": "The shared evidence addresses the risk"}],
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
        probes=__import__(
            "cluxion_effort_ultracode.doctor.probes", fromlist=["PROBES"]
        ).PROBES,
        plugin="effort-ultracode",
        version=__import__("cluxion_effort_ultracode").__version__,
    )
    text = render_text(result, __import__(
        "cluxion_effort_ultracode.doctor.framework", fromlist=["load_catalog"]
    ).load_catalog(catalog_path), verbose=bool(getattr(namespace, "verbose", False)))
    print(text, file=sys.stderr)
    if getattr(namespace, "json", False):
        print(render_json(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
