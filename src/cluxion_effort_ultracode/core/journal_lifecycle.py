"""List/show/gc helpers for debate journal files."""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cluxion_effort_ultracode.core.journal import (
    JournalBusy,
    JournalLockUnsupported,
    journals_dir,
    locks_supported,
    read_records,
    try_lock_journal_fd,
)

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

    # Destructive GC without lock support must refuse (typed error), not skip as success.
    if apply and not locks_supported():
        raise JournalLockUnsupported("POSIX advisory locks unavailable for destructive journal GC")

    directory = journals_dir(home)
    candidates = []
    for path in sorted(directory.glob("*.jsonl")) if directory.exists() else []:
        summary = _gc_inspect(path, cutoff=cutoff, apply=apply)
        if summary is not None:
            candidates.append(summary)
    return {"apply": apply, "older_than_days": older_than_days, "candidates": candidates}


def _gc_inspect(path: Path, *, cutoff: datetime, apply: bool) -> dict[str, object] | None:
    """Read (and optionally delete) one journal while holding the inode lock.

    Busy journals are skipped. Without lock support, non-destructive dry-run falls
    back to unlocked read; apply is handled by the caller.
    """
    if not locks_supported():
        summary = _summary(path)
        created = _parse_time(str(summary.get("created_at") or ""))
        if created is not None and created < cutoff:
            return summary
        return None

    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDWR)
        try:
            try_lock_journal_fd(fd, path.stem)
        except JournalBusy:
            return None
        except JournalLockUnsupported:
            # Destructive apply must never report success/empty after unsupported lock.
            # Dry-run may continue read-only/unlocked on the already-open FD (matches
            # the locks_supported() is False contract).
            if apply:
                raise

        # Fail closed on GC/unlink inode races.
        fd_stat = os.fstat(fd)
        try:
            path_stat = path.stat()
        except FileNotFoundError:
            return None
        if fd_stat.st_ino != path_stat.st_ino or fd_stat.st_dev != path_stat.st_dev:
            return None

        records = _read_records_fd(fd)
        summary = _summary_from_records(path, records)
        created = _parse_time(str(summary.get("created_at") or ""))
        if created is None or created >= cutoff:
            return None
        if apply:
            try:
                os.unlink(path)
            except FileNotFoundError:
                return None
        return summary
    except OSError:
        return None
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)


def _summary(path: Path) -> dict[str, object]:
    records = read_records(path)
    return _summary_from_records(path, records)


def _summary_from_records(path: Path, records: list[dict[str, object]]) -> dict[str, object]:
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


def _read_records_fd(fd: int) -> list[dict[str, object]]:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        piece = os.read(fd, 65536)
        if not piece:
            break
        chunks.append(piece)

    records: list[dict[str, object]] = []
    for line in b"".join(chunks).decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(record, dict):
            records.append(record)
    return records


def _preview(value: str, limit: int = 80) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
