"""Phase 19: bounded cleanup for abandoned browser evaluation artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from app.evaluation import UIEvaluationRunner
from app.retention import (
    DEFAULT_UI_RETENTION_SECONDS,
    RETENTION_ENV,
    configured_retention_seconds,
    sweep_session_artifacts,
)


def _set_tree_mtime(root: Path, timestamp: float) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in (*paths, root):
        os.utime(path, (timestamp, timestamp), follow_symlinks=False)


def test_sweep_removes_stale_sessions_and_reports_removed_bytes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    stale = results / "ui" / "stale-session"
    stale.mkdir(parents=True)
    payload = stale / "run-1" / "report" / "report.md"
    payload.parent.mkdir(parents=True)
    payload.write_text("artifact", encoding="utf-8")
    _set_tree_mtime(stale, 100.0)

    fresh = results / "ui" / "fresh-session"
    fresh.mkdir(parents=True)
    fresh_payload = fresh / "run.json"
    fresh_payload.write_text("keep", encoding="utf-8")
    _set_tree_mtime(fresh, 950.0)

    report = sweep_session_artifacts(results, retention_seconds=100, now=1_000.0)

    assert report.clean
    assert report.sessions_scanned == 2
    assert report.sessions_removed == 1
    assert report.files_removed == 1
    assert report.bytes_removed == len("artifact")
    assert not stale.exists()
    assert fresh_payload.exists()
    assert (results / "ui").is_dir()


def test_sweep_does_not_follow_symlinked_or_unsafe_session_directories(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    ui_root = results / "ui"
    ui_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "secret.txt"
    external.write_text("must stay", encoding="utf-8")
    unsafe = ui_root / "not a session"
    unsafe.mkdir()
    _set_tree_mtime(unsafe, 0.0)
    try:
        (ui_root / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        # Symlink creation is unavailable on a few supported Windows runners;
        # the direct-directory safety assertion still exercises the sweep.
        linked = None
    else:
        linked = ui_root / "linked"

    report = sweep_session_artifacts(results, retention_seconds=1, now=10_000.0)

    assert report.sessions_removed == 0
    assert unsafe.exists()
    assert external.read_text(encoding="utf-8") == "must stay"
    if linked is not None:
        assert linked.is_symlink()


def test_sweep_protects_the_session_currently_being_served(tmp_path: Path) -> None:
    results = tmp_path / "results"
    protected = results / "ui" / "active"
    protected.mkdir(parents=True)
    (protected / "old.json").write_text("old", encoding="utf-8")
    _set_tree_mtime(protected, 0.0)

    report = sweep_session_artifacts(
        results,
        retention_seconds=1,
        now=10_000.0,
        protected_sessions=("active",),
    )

    assert report.sessions_scanned == 1
    assert report.sessions_protected == 1
    assert report.sessions_removed == 0
    assert protected.exists()


def test_invalid_retention_configuration_falls_back_to_one_day(
    monkeypatch,
) -> None:
    monkeypatch.setenv(RETENTION_ENV, "not-a-duration")
    assert configured_retention_seconds() == DEFAULT_UI_RETENTION_SECONDS
    assert configured_retention_seconds("0") == DEFAULT_UI_RETENTION_SECONDS
    assert configured_retention_seconds("inf") == DEFAULT_UI_RETENTION_SECONDS


def test_runner_sweeps_on_startup_and_exposes_the_receipt(tmp_path: Path) -> None:
    results = tmp_path / "results"
    abandoned = results / "ui" / "abandoned"
    abandoned.mkdir(parents=True)
    (abandoned / "artifact.txt").write_text("old", encoding="utf-8")
    _set_tree_mtime(abandoned, 0.0)

    runner = UIEvaluationRunner(
        results_root=results,
        dataset_root=tmp_path / "benchmarks",
        retention_seconds=1,
    )

    assert not abandoned.exists()
    assert runner.last_retention_report.sessions_removed == 1
    assert runner.last_retention_report.to_dict()["retention_version"] == "retention-v1"


def test_runner_sweep_keeps_the_session_for_a_run_then_cleans_it_later(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    session = results / "ui" / "active"
    session.mkdir(parents=True)
    (session / "old.json").write_text("old", encoding="utf-8")
    _set_tree_mtime(session, 0.0)
    runner = UIEvaluationRunner(
        results_root=results,
        dataset_root=tmp_path / "benchmarks",
        retention_seconds=1,
    )

    # The startup sweep already removed it; recreate it to model a run that
    # became old while the process stayed alive.
    session.mkdir(parents=True)
    (session / "old.json").write_text("old", encoding="utf-8")
    _set_tree_mtime(session, 0.0)
    protected_report = runner.sweep_stale(now=10_000.0, protected_sessions=("active",))
    assert protected_report.sessions_protected == 1
    assert session.exists()

    removed_report = runner.sweep_stale(now=10_000.0)
    assert removed_report.sessions_removed == 1
    assert not session.exists()
