"""Retention for browser-run evaluation artifacts (Phase 19, plan §19 / §21).

Phase 17 gave every browser session its own artifact tree,
``results/ui/<session>/<run>/``, and its **End session** control deletes it. What
that leaves uncovered is the session nobody ends: a visitor closes the tab, and
their run's raw files, figures, reports, and answer records stay on the host's
disk for as long as the host exists. On a public Space that is unbounded growth
owned by nobody — the Phase 17 hand-off named it as the gap to close before the
Evaluation tab is offered publicly with generation on.

This module is the missing half. It sweeps on age, not on count:

* a session directory is removed only when its **newest** file is older than the
  retention window, so a session that is still running can never be swept out
  from under its owner, and a burst of runs cannot evict each other;
* the sweep is bounded (``max_sessions_per_sweep``) so a pathological root
  cannot turn one page load into thousands of deletes;
* every candidate must be a direct child of ``<root>/ui``, must be a single safe
  path segment, must not be a symlink, and must still resolve inside that
  directory — the same containment rules the runner's ``discard`` uses, because
  a retention job with a path bug is a deletion tool.

The sweep is deliberately boring and reportable: it returns a
:class:`RetentionReport` (counts, bytes, what it skipped and why), and
``python -m app.retention`` runs it from a terminal or a cron job. A dry run is
one flag away, because the first time an operator runs a deletion tool they
should be able to see what it would have done.

Nothing here reads or writes a credential: artifacts are counted and unlinked,
never opened.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from checkout import repo_anchored

#: Identifies the retention semantics recorded in a report or a run artifact.
RETENTION_VERSION = "retention-v1"

#: The plan's results root, and the subdirectory browser runs live in (Phase 17).
#: Declared here, next to the sweep that owns them, and imported by
#: :mod:`app.evaluation` so the layout cannot drift between writer and sweeper.
RESULTS_ROOT = Path("results")
RUNS_SUBDIR = "ui"

#: A week. Long enough that a visitor can come back to a report they ran, short
#: enough that an abandoned session's files do not accumulate forever.
DEFAULT_RETENTION_SECONDS = 7 * 24 * 60 * 60

#: Deletions one sweep may perform. A public Space with a year of abandoned
#: sessions should not turn the next visitor's run into a filesystem purge.
MAX_SESSIONS_PER_SWEEP = 200

#: Session ids are path segments; the runner's own tokens are uuid4-shaped.
#: Anything with a separator, a parent reference, or a leading dot is refused
#: rather than guessed at.
_SAFE_SEGMENT_RE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_MAX_SEGMENT_LENGTH = 128


@dataclass(frozen=True)
class RetentionPolicy:
    """How long browser-run artifacts are kept, and how much one sweep may do."""

    #: Age at which a session's artifacts are swept. ``None`` keeps them until
    #: the visitor ends the session (or forever, if they never do).
    max_age_seconds: float | None = DEFAULT_RETENTION_SECONDS
    #: Upper bound on session directories removed by one sweep.
    max_sessions_per_sweep: int = MAX_SESSIONS_PER_SWEEP

    def __post_init__(self) -> None:
        age = self.max_age_seconds
        if age is not None and (not isinstance(age, (int, float)) or age <= 0):
            raise ValueError("max_age_seconds must be a positive number of seconds, or None.")
        if (
            not isinstance(self.max_sessions_per_sweep, int)
            or isinstance(self.max_sessions_per_sweep, bool)
            or self.max_sessions_per_sweep < 1
        ):
            raise ValueError("max_sessions_per_sweep must be a positive integer.")

    @property
    def enabled(self) -> bool:
        return self.max_age_seconds is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retention_version": RETENTION_VERSION,
            "max_age_seconds": self.max_age_seconds,
            "max_sessions_per_sweep": self.max_sessions_per_sweep,
        }


@dataclass(frozen=True)
class RetentionReport:
    """What one sweep did — or, in a dry run, what it would have done."""

    root: str
    swept_at: float
    policy: RetentionPolicy
    dry_run: bool = False
    sessions_scanned: int = 0
    sessions_removed: int = 0
    files_removed: int = 0
    bytes_removed: int = 0
    #: Candidate session directories left in place: too young to sweep, or not
    #: reached because the sweep hit its bound (see ``truncated``).
    kept: int = 0
    #: Directories that were refused rather than considered — an unusable
    #: session id, a symlink, a path that resolves outside ``<root>/ui``.
    skipped: tuple[str, ...] = field(default_factory=tuple)
    truncated: bool = False
    sweep_root_exists: bool = True

    @property
    def removed(self) -> int:
        return self.sessions_removed

    def summary(self) -> str:
        verbs = "would remove" if self.dry_run else "removed"
        lines = [
            f"retention_version={RETENTION_VERSION} root={self.root or '<unset>'} "
            f"sessions={self.sessions_scanned} kept={self.kept} "
            f"{verbs}={self.sessions_removed} files={self.files_removed} "
            f"bytes={self.bytes_removed} dry_run={self.dry_run}"
        ]
        age = self.policy.max_age_seconds
        if age is None:
            lines.append(
                "retention disabled (max_age_seconds=None): artifacts live "
                "until End session"
            )
        else:
            lines.append(
                f"max_age_seconds={age:g} "
                f"max_sessions_per_sweep={self.policy.max_sessions_per_sweep}"
            )
        if not self.sweep_root_exists:
            lines.append("nothing to sweep: the results/ui directory does not exist")
        if self.truncated:
            lines.append(
                "stopped at max_sessions_per_sweep: more expired sessions remain for the next sweep"
            )
        for note in self.skipped:
            lines.append(f"skipped: {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retention_version": RETENTION_VERSION,
            "root": self.root,
            "swept_at": self.swept_at,
            "dry_run": self.dry_run,
            "policy": self.policy.to_dict(),
            "sessions_scanned": self.sessions_scanned,
            "sessions_removed": self.sessions_removed,
            "files_removed": self.files_removed,
            "bytes_removed": self.bytes_removed,
            "kept": self.kept,
            "skipped": list(self.skipped),
            "truncated": self.truncated,
            "sweep_root_exists": self.sweep_root_exists,
        }


def _safe_segment(name: str) -> bool:
    return (
        bool(name)
        and len(name) <= _MAX_SEGMENT_LENGTH
        and not name.startswith(".")
        and set(name) <= _SAFE_SEGMENT_RE_CHARS
    )


def _newest_mtime(path: Path) -> float:
    """Return the newest modification time under ``path``.

    Directory mtimes move when entries are added, not when the files inside them
    are rewritten, so the *files* decide a session's age. Symlinks are not
    followed: a link is not evidence that the target is this session's work, and
    a retention job has no business measuring (or deleting) anything outside the
    directory it was pointed at.
    """

    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if not _is_symlink(Path(dirpath) / name)]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if _is_symlink(candidate):
                continue
            try:
                newest = max(newest, candidate.stat().st_mtime)
            except OSError:  # pragma: no cover - a file removed mid-walk
                continue
    try:
        newest = max(newest, path.stat().st_mtime)
    except OSError:  # pragma: no cover - a directory removed mid-walk
        pass
    return newest


def _is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:  # pragma: no cover - depends on the filesystem
        return True


def _tree_size(path: Path) -> tuple[int, int]:
    """Return ``(files, bytes)`` under ``path`` without following symlinks."""

    files = 0
    size = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if not _is_symlink(Path(dirpath) / name)]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if _is_symlink(candidate):
                continue
            files += 1
            try:
                size += candidate.stat().st_size
            except OSError:  # pragma: no cover - a file removed mid-walk
                continue
    return files, size


def sweep_results(
    root: str | Path | None = None,
    *,
    policy: RetentionPolicy | None = None,
    now: float | None = None,
    dry_run: bool = False,
) -> RetentionReport:
    """Remove expired session artifact trees under ``<root>/ui``.

    ``root`` defaults to the plan's ``results/`` directory, anchored to the
    checkout when the process runs outside it. A missing sweep root is not an
    error: it is the normal state of a deployment where nobody has run a
    benchmark yet, and the report says so.
    """

    policy = policy or RetentionPolicy()
    resolved_root = Path(root) if root is not None else repo_anchored(RESULTS_ROOT)
    moment = float(now if now is not None else time.time())
    base = Path(resolved_root) / RUNS_SUBDIR
    report = RetentionReport(
        root=str(resolved_root),
        swept_at=moment,
        policy=policy,
        dry_run=bool(dry_run),
    )
    if not policy.enabled:
        return report
    if not base.is_dir():
        return RetentionReport(
            root=report.root,
            swept_at=moment,
            policy=policy,
            dry_run=bool(dry_run),
            sweep_root_exists=False,
        )

    base_resolved = base.resolve()
    scanned = 0
    removed = 0
    files_removed = 0
    bytes_removed = 0
    skipped: list[str] = []
    truncated = False

    candidates: list[tuple[float, Path]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not _safe_segment(name):
            skipped.append(f"{name!r} is not a usable session id")
            continue
        if _is_symlink(child):
            skipped.append(f"{name!r} is a symlink (not followed)")
            continue
        try:
            if not child.resolve().is_relative_to(base_resolved):
                skipped.append(f"{name!r} resolves outside {base}")
                continue
        except OSError:  # pragma: no cover - depends on the filesystem
            skipped.append(f"{name!r} could not be resolved")
            continue
        candidates.append((_newest_mtime(child), child))

    # Oldest first, so a truncated sweep removes the artefacts closest to expiry.
    candidates.sort(key=lambda item: item[0])
    age = float(policy.max_age_seconds or 0.0)
    for newest, child in candidates:
        # ``<=`` rather than ``<``: a directory whose newest file is exactly at
        # the window is inside it. Removal needs the age to have passed.
        if moment - newest <= age:
            continue
        if removed >= policy.max_sessions_per_sweep:
            truncated = True
            break
        scanned += 1
        files, size = _tree_size(child)
        if dry_run:
            removed += 1
            files_removed += files
            bytes_removed += size
            continue
        shutil.rmtree(child, ignore_errors=True)
        if child.exists():  # pragma: no cover - a permission failure
            skipped.append(f"{child.name!r} could not be fully removed")
            removed -= 1
            continue
        removed += 1
        files_removed += files
        bytes_removed += size

    report = RetentionReport(
        root=report.root,
        swept_at=moment,
        policy=policy,
        dry_run=bool(dry_run),
        sessions_scanned=scanned,
        sessions_removed=removed,
        files_removed=files_removed,
        bytes_removed=bytes_removed,
        # Everything still on disk afterwards: never expired, or expired but
        # not reached before the sweep hit its bound.
        kept=len(candidates) - removed,
        skipped=tuple(skipped),
        truncated=truncated,
        sweep_root_exists=True,
    )
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep expired browser-run evaluation artifacts (plan Phase 19).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Results root to sweep (default: this checkout's results/). The "
            "sweep only ever touches <root>/ui/<session>/."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=DEFAULT_RETENTION_SECONDS / 86_400,
        metavar="DAYS",
        help="Age at which a session's artifacts are removed (default: 7).",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=MAX_SESSIONS_PER_SWEEP,
        metavar="N",
        help=f"Sessions one sweep may remove (default: {MAX_SESSIONS_PER_SWEEP}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without removing anything.",
    )
    parser.add_argument("--output", type=Path, help="Write the report as JSON here.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--quiet", action="store_true", help="Print only the status line.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = RetentionPolicy(
        max_age_seconds=args.max_age_days * 86_400,
        max_sessions_per_sweep=args.max_sessions,
    )
    report = sweep_results(args.root, policy=policy, dry_run=args.dry_run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(report.summary())
    else:
        print(
            f"removed={report.sessions_removed} files={report.files_removed} "
            f"dry_run={report.dry_run}"
        )
    if not report.sweep_root_exists:
        # Nothing was scanned because there was nothing to scan. Like
        # ``security.scan``, that is a distinct outcome from "swept cleanly".
        print("Nothing to sweep: the results/ui directory does not exist.", flush=True)
        return 0
    return 0


__all__ = [
    "DEFAULT_RETENTION_SECONDS",
    "MAX_SESSIONS_PER_SWEEP",
    "RESULTS_ROOT",
    "RETENTION_VERSION",
    "RUNS_SUBDIR",
    "RetentionPolicy",
    "RetentionReport",
    "build_parser",
    "main",
    "sweep_results",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
