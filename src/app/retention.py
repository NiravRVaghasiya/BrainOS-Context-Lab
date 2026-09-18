"""Retention cleanup for browser-initiated evaluation artifacts.

The Evaluation tab writes sensitive, session-scoped reports under
``results/ui/<session>/<run>``. ``End session`` deletes them immediately, but a
browser can disappear without sending that callback. This module provides the
bounded, path-safe sweep used at application startup and before later browser
runs so abandoned sessions do not accumulate forever on a hosted Space.

Only direct child directories of the configured ``ui`` root are eligible. A
session directory must have a safe single-segment name, symlinked directories
are never followed, and the root itself is never removed. The sweep is
best-effort: one unreadable or concurrently removed session is reported by
exception type and does not prevent the remaining sessions from being checked.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import time

# Keep this module independent from ``app.evaluation`` so the runner can import
# the sweep without creating an evaluation/retention import cycle. It is the
# same path-segment contract used by the runner for session and run ids.
_SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")

RETENTION_VERSION = "retention-v1"
DEFAULT_UI_RETENTION_SECONDS = 24 * 60 * 60
RETENTION_ENV = "BRAINOS_LAB_UI_RETENTION_SECONDS"


@dataclass(frozen=True)
class RetentionReport:
    """JSON-ready receipt for one artifact-retention sweep."""

    root: str
    retention_seconds: float
    now: float
    sessions_scanned: int = 0
    sessions_protected: int = 0
    sessions_skipped: int = 0
    sessions_removed: int = 0
    files_removed: int = 0
    bytes_removed: int = 0
    errors: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "retention_version": RETENTION_VERSION,
            "root": self.root,
            "retention_seconds": self.retention_seconds,
            "now": self.now,
            "sessions_scanned": self.sessions_scanned,
            "sessions_protected": self.sessions_protected,
            "sessions_skipped": self.sessions_skipped,
            "sessions_removed": self.sessions_removed,
            "files_removed": self.files_removed,
            "bytes_removed": self.bytes_removed,
            "errors": list(self.errors),
            "clean": self.clean,
        }


def configured_retention_seconds(value: float | int | str | None = None) -> float:
    """Return the configured positive retention window.

    An invalid or non-positive environment value falls back to the safe
    one-day default rather than disabling cleanup or making application startup
    fail. Callers that need a different value may pass it explicitly.
    """

    raw = os.getenv(RETENTION_ENV) if value is None else value
    try:
        seconds = float(raw) if raw is not None and str(raw).strip() else float(
            DEFAULT_UI_RETENTION_SECONDS
        )
    except (TypeError, ValueError):
        return float(DEFAULT_UI_RETENTION_SECONDS)
    if not isfinite(seconds) or seconds <= 0:
        return float(DEFAULT_UI_RETENTION_SECONDS)
    return seconds


def sweep_session_artifacts(
    results_root: str | Path,
    *,
    retention_seconds: float | int | str | None = None,
    now: float | None = None,
    protected_sessions: Iterable[str] = (),
) -> RetentionReport:
    """Remove stale session directories below ``results_root/ui``.

    ``now`` is injectable for deterministic tests. ``protected_sessions`` is
    used by the in-process runner to avoid removing a session that is currently
    active even if an older run within it exceeds the retention window.
    """

    seconds = configured_retention_seconds(retention_seconds)
    current = float(time() if now is None else now)
    root_candidate = Path(results_root).expanduser() / "ui"
    if root_candidate.is_symlink():
        return RetentionReport(
            root=str(root_candidate.resolve()),
            retention_seconds=seconds,
            now=current,
            errors=("invalid_root",),
        )
    root = root_candidate.resolve()
    protected = {
        token
        for token in (str(value).strip() for value in protected_sessions)
        if _SAFE_TOKEN_RE.fullmatch(token)
    }
    report = RetentionReport(
        root=str(root),
        retention_seconds=seconds,
        now=current,
    )
    if not root.exists():
        return report
    if not root.is_dir() or root.is_symlink():
        return RetentionReport(
            root=str(root),
            retention_seconds=seconds,
            now=current,
            errors=("invalid_root",),
        )

    scanned = protected_count = skipped = removed = files = bytes_removed = 0
    errors: list[str] = []
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        return RetentionReport(
            root=str(root),
            retention_seconds=seconds,
            now=current,
            errors=(type(exc).__name__,),
        )

    cutoff = current - seconds
    for session_root in entries:
        if session_root.is_symlink() or not session_root.is_dir():
            skipped += 1
            continue
        if not _SAFE_TOKEN_RE.fullmatch(session_root.name):
            skipped += 1
            continue
        scanned += 1
        if session_root.name in protected:
            protected_count += 1
            continue
        try:
            latest, session_files, session_bytes = _tree_activity(session_root)
            if latest > cutoff:
                continue
            shutil.rmtree(session_root)
            removed += 1
            files += session_files
            bytes_removed += session_bytes
        except OSError as exc:
            errors.append(type(exc).__name__)

    return RetentionReport(
        root=str(root),
        retention_seconds=seconds,
        now=current,
        sessions_scanned=scanned,
        sessions_protected=protected_count,
        sessions_skipped=skipped,
        sessions_removed=removed,
        files_removed=files,
        bytes_removed=bytes_removed,
        errors=tuple(errors),
    )


def _tree_activity(root: Path) -> tuple[float, int, int]:
    """Return ``(latest_mtime, regular_file_count, byte_count)`` safely."""

    latest = root.stat().st_mtime
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        latest = max(latest, stat.st_mtime)
        if path.is_file():
            file_count += 1
            total_bytes += stat.st_size
    return latest, file_count, total_bytes


__all__ = [
    "DEFAULT_UI_RETENTION_SECONDS",
    "RETENTION_ENV",
    "RETENTION_VERSION",
    "RetentionReport",
    "configured_retention_seconds",
    "sweep_session_artifacts",
]
