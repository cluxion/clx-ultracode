"""List/show/gc helpers for debate journal files."""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cluxion_effort_ultracode.core.journal import (
    JournalBusy,
    JournalLockUnsupported,
    _parse_jsonl_bytes,
    _read_fd_bytes,
    journals_dir,
    locks_supported,
    read_records,
    try_lock_journal_fd,
)

WARN_SIZE_BYTES = 50 * 1024 * 1024

# Retained GC candidate: path, locked fd, verified inode/dev, summary for report.
_GcRetained = tuple[Path, int, int, int, dict[str, object]]


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
    paths = sorted(directory.glob("*.jsonl")) if directory.exists() else []

    if not apply:
        candidates = []
        for path in paths:
            summary = _gc_inspect(path, cutoff=cutoff, apply=False)
            if summary is not None:
                candidates.append(summary)
        return {"apply": False, "older_than_days": older_than_days, "candidates": candidates}

    # apply=True: full preflight under per-file nonblocking locks, then unlink.
    # Never unlink mid-scan — a later JournalCorrupt / inspection failure must leave
    # zero destructive work (release retained FDs, raise, no partial GC).
    retained: list[_GcRetained] = []
    try:
        for path in paths:
            item = _gc_preflight_one(path, cutoff=cutoff)
            if item is not None:
                retained.append(item)

        verified: list[_GcRetained] = []
        for item in retained:
            path, _fd, ino, dev, _summary = item
            try:
                path_stat = path.stat()
            except FileNotFoundError:
                continue
            if path_stat.st_ino != ino or path_stat.st_dev != dev:
                continue
            verified.append(item)

        candidates: list[dict[str, object]] = []
        for path, _fd, _ino, _dev, summary in verified:
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            candidates.append(summary)
        return {"apply": True, "older_than_days": older_than_days, "candidates": candidates}
    finally:
        for _path, fd, _ino, _dev, _summary in retained:
            with contextlib.suppress(OSError):
                os.close(fd)


def _gc_preflight_one(path: Path, *, cutoff: datetime) -> _GcRetained | None:
    """Inspect one snapshot under a nonblocking lock; retain FD only if stale candidate.

    Busy files skip. Fresh/non-candidate clean FDs close immediately. Stale deletion
    candidates keep the locked FD + verified inode until the caller finishes preflight.
    Non-busy corruption / lock-unsupported / other inspection errors propagate after
    releasing this file's FD (caller releases any previously retained candidates).
    """
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDWR)
        try:
            try_lock_journal_fd(fd, path.stem)
        except JournalBusy:
            os.close(fd)
            fd = None
            return None
        except JournalLockUnsupported:
            # Destructive apply must never report success after unsupported lock.
            raise

        fd_stat = os.fstat(fd)
        try:
            path_stat = path.stat()
        except FileNotFoundError:
            os.close(fd)
            fd = None
            return None
        if fd_stat.st_ino != path_stat.st_ino or fd_stat.st_dev != path_stat.st_dev:
            os.close(fd)
            fd = None
            return None

        # size==0 O_EXCL leftovers: age by mtime only (no created_at header).
        if fd_stat.st_size == 0:
            mtime = datetime.fromtimestamp(fd_stat.st_mtime, tz=UTC)
            if mtime >= cutoff:
                os.close(fd)
                fd = None
                return None
            summary = _empty_claim_summary(path)
            retained = (path, fd, fd_stat.st_ino, fd_stat.st_dev, summary)
            fd = None
            return retained

        # Nonempty: parse under lock. JournalCorrupt and similar propagate — never
        # mtime-delete corrupt content.
        records = _read_records_fd(fd, run_id=path.stem)
        summary = _summary_from_records(path, records)
        created = _parse_time(str(summary.get("created_at") or ""))
        if created is None or created >= cutoff:
            os.close(fd)
            fd = None
            return None
        retained = (path, fd, fd_stat.st_ino, fd_stat.st_dev, summary)
        fd = None
        return retained
    except FileNotFoundError:
        # A path removed during the snapshot is a benign race; hard I/O and
        # permission failures must abort before any destructive pass begins.
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        return None
    except Exception:
        # JournalCorrupt / JournalLockUnsupported / other hard failures: release this
        # FD and propagate so the caller can release retained candidates with zero unlink.
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        raise


def _gc_inspect(path: Path, *, cutoff: datetime, apply: bool) -> dict[str, object] | None:
    """Read (and, for legacy single-file apply, optionally delete) one journal.

    Destructive multi-file apply uses `_gc_preflight_one` + batch unlink instead.
    Busy journals are skipped. Without lock support, non-destructive dry-run falls
    back to unlocked read; apply is handled by the caller.
    """
    if not locks_supported():
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        if st.st_size == 0:
            mtime = datetime.fromtimestamp(st.st_mtime, tz=UTC)
            if mtime < cutoff:
                return _empty_claim_summary(path)
            return None
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

        if fd_stat.st_size == 0:
            mtime = datetime.fromtimestamp(fd_stat.st_mtime, tz=UTC)
            if mtime >= cutoff:
                return None
            summary = _empty_claim_summary(path)
            if apply:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    return None
            return summary

        records = _read_records_fd(fd, run_id=path.stem)
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


def _empty_claim_summary(path: Path) -> dict[str, object]:
    return {
        "run_id": path.stem,
        "created_at": None,
        "question": "",
        "status": "incomplete",
        "calls_recorded": 0,
        "path": str(path),
    }


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


def _read_records_fd(fd: int, *, run_id: str = "") -> list[dict[str, object]]:
    """Read-only shared JSONL parse (no tail repair mutation)."""
    data = _read_fd_bytes(fd)
    return _parse_jsonl_bytes(data, run_id=run_id or "unknown").records


def _preview(value: str, limit: int = 80) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
