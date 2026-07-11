"""Spawn-safe multiprocessing workers for journal lock regressions."""

from __future__ import annotations

import os
from pathlib import Path


def hold_journal_until_release(home_s: str, run_id: str, q_ready: object, q_release: object) -> None:
    from cluxion_effort_ultracode.core.journal import DebateJournal, build_header, journal_header

    home = Path(home_s)
    hdr = journal_header(run_id, home=home)
    expected = build_header(
        run_id=run_id,
        question=str(hdr["question"]),
        context=str(hdr.get("context", "")),
        agents_count=int(hdr["agents_count"]),  # type: ignore[arg-type]
        max_rounds=int(hdr["max_rounds"]),  # type: ignore[arg-type]
        models=list(hdr.get("models") or []),
        adapter=str(hdr["adapter"]),
        agent_timeout_s=float(hdr["agent_timeout_s"]),  # type: ignore[arg-type]
        debate_budget_s=float(hdr["debate_budget_s"]),  # type: ignore[arg-type]
        budget_tokens=hdr.get("budget_tokens"),  # type: ignore[arg-type]
    )
    # Keep a live reference so __del__/close does not release the lock early.
    journal = DebateJournal.resume(run_id, expected, home=home)
    del expected, hdr, home  # keep only the live journal FD
    q_ready.put("ready")  # type: ignore[attr-defined]
    q_release.get()  # type: ignore[attr-defined]
    # Terminate while the journal FD remains live. Do not null _file (that can
    # close/release before exit) and do not return normally (triggers __del__).
    # OS releases the advisory lock with the process.
    _ = journal  # retain reference until hard exit
    os._exit(0)
