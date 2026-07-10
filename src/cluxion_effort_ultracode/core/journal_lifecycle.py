"""List/show/gc helpers for debate journal files."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from cluxion_effort_ultracode.core.journal import journals_dir, read_records

WARN_SIZE_BYTES = 50 * 1024 * 1024


def list_journals(*, home: Path | None = None) -> dict[str, object]:
    directory = journals_dir(home)
    files = sorted(directory.glob("*.jsonl")) if directory.exists() else []
    return {
        "journals": [_summary(path) for path in files],
        "total_bytes": sum(path.stat().st_size for path in files),
        "warn_size_bytes": WARN_SIZE_BYTES,
    }


def gc_journals(*, older_than_days: int = 7, apply: bool = False, home: Path | None = None) -> dict[str, object]:
    try:
        age = timedelta(days=older_than_days)
        cutoff = datetime.now(UTC) - age
    except OverflowError as exc:
        raise ValueError("older_than_days is outside the supported datetime range") from exc
    directory = journals_dir(home)
    candidates = []
    for path in sorted(directory.glob("*.jsonl")) if directory.exists() else []:
        summary = _summary(path)
        created = _parse_time(str(summary.get("created_at") or ""))
        if created is not None and created < cutoff:
            candidates.append(summary)
            if apply:
                path.unlink(missing_ok=True)
    return {"apply": apply, "older_than_days": older_than_days, "candidates": candidates}


def _summary(path: Path) -> dict[str, object]:
    records = read_records(path)
    header = records[0] if records else {}
    result = next((record for record in reversed(records) if record.get("type") == "result"), None)
    calls = [record for record in records if record.get("type") == "call"]
    return {
        "run_id": header.get("run_id", path.stem),
        "created_at": header.get("created_at"),
        "question": _preview(str(header.get("question", ""))),
        "status": result.get("status") if result else "incomplete",
        "calls_recorded": len(calls),
        "path": str(path),
    }


def _preview(value: str, limit: int = 80) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
