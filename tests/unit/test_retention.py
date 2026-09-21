"""Phase 19: retention for the artifacts of a session nobody ends.

Phase 17's ``End session`` deletes a visitor's own artifact tree. The session
that never calls it — the closed tab, the abandoned run — was left on disk
forever, which is the gap this module closes.

These tests are about the two properties that decide whether an automatic
deletion job is safe to run on a public host:

* **it never removes anything live.** Age is measured from the newest file, so a
  session still writing is never swept, and a session on the boundary is kept.
* **it never removes anything it was not pointed at.** Candidates are direct
  children of ``<root>/ui``, single safe path segments, not symlinks, and
  re-checked after resolution. A retention tool with a path bug is a deletion
  tool.

The rest is reporting: a sweep says what it removed, what it kept, what it
skipped and why, and ``--dry-run`` shows the same thing without touching disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app import retention
from app.retention import (
    DEFAULT_RETENTION_SECONDS,
    MAX_SESSIONS_PER_SWEEP,
    RETENTION_VERSION,
    RetentionPolicy,
    sweep_results,
)

DAY = 86_400
SESSION_A = "11111111-1111-1111-1111-111111111111"
SESSION_B = "22222222-2222-2222-2222-222222222222"
NOW = 1_800_000_000.0


def _session(
    root: Path, name: str, *, age_days: float, files: int = 2, now: float | None = None
) -> Path:
    """Create ``<root>/ui/<name>/<run>/file`` with a chosen age.

    ``now`` defaults to the real clock, which is what the CLI tests sweep with;
    the deterministic tests pass the same fixed ``now`` to the sweep.
    """

    directory = root / "ui" / name
    run = directory / "run-0001"
    run.mkdir(parents=True)
    for index in range(files):
        target = run / f"artifact-{index}.json"
        target.write_text("{}", encoding="utf-8")
    stamp = (time.time() if now is None else now) - age_days * DAY
    for target in sorted(run.rglob("*"), reverse=True) + [run, directory]:
        os.utime(target, (stamp, stamp))
    return directory


def _policy(days: float = 7.0) -> RetentionPolicy:
    """A retention policy expressed in days, the unit the CLI flag uses."""

    return RetentionPolicy(max_age_seconds=days * DAY)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def test_the_default_window_is_a_week_and_a_sweep_is_bounded() -> None:
    policy = RetentionPolicy()

    assert policy.max_age_seconds == DEFAULT_RETENTION_SECONDS == 7 * DAY
    assert policy.enabled is True
    assert policy.max_sessions_per_sweep == MAX_SESSIONS_PER_SWEEP


def test_retention_can_be_disabled_for_a_deployment_that_wants_to_keep_everything() -> None:
    policy = RetentionPolicy(max_age_seconds=None)

    assert policy.enabled is False
    assert policy.to_dict()["max_age_seconds"] is None


@pytest.mark.parametrize("value", [0, -1, "week"])
def test_a_nonsensical_window_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match="max_age_seconds"):
        RetentionPolicy(max_age_seconds=value)  # type: ignore[arg-type]


def test_a_nonsensical_sweep_bound_is_refused() -> None:
    with pytest.raises(ValueError, match="max_sessions_per_sweep"):
        RetentionPolicy(max_sessions_per_sweep=0)


# --------------------------------------------------------------------------- #
# Age, not count
# --------------------------------------------------------------------------- #


def test_an_expired_session_is_removed_and_a_live_one_is_kept(tmp_path: Path) -> None:
    root = tmp_path
    _session(root, SESSION_A, age_days=30, now=NOW)
    _session(root, SESSION_B, age_days=0.1, now=NOW)

    report = sweep_results(root, policy=_policy(), now=NOW)

    assert report.sessions_removed == 1
    assert report.kept == 1
    assert not (root / "ui" / SESSION_A).exists()
    assert (root / "ui" / SESSION_B).exists()
    assert report.files_removed == 2
    assert report.bytes_removed == 4  # two files of "{}"


def test_a_session_on_the_boundary_is_kept(tmp_path: Path) -> None:
    """Expiry is not a race: exactly-at-the-window is still inside it."""

    _session(tmp_path, SESSION_A, age_days=7, now=NOW)

    report = sweep_results(tmp_path, policy=_policy(7), now=NOW)

    assert report.sessions_removed == 0
    assert (tmp_path / "ui" / SESSION_A).is_dir()


def test_a_single_recent_file_protects_a_session_whose_older_files_are_ancient(
    tmp_path: Path,
) -> None:
    """Age is the newest file, so a long study's early raw files do not expire it."""

    directory = _session(tmp_path, SESSION_A, age_days=90, files=1, now=NOW)
    recent = directory / "run-0001" / "report.md"
    recent.write_text("# report", encoding="utf-8")
    os.utime(recent, (NOW - 60, NOW - 60))

    report = sweep_results(tmp_path, policy=_policy(), now=NOW)

    assert report.sessions_removed == 0
    assert recent.is_file()


def test_an_empty_session_directory_is_removed_once_it_is_old(tmp_path: Path) -> None:
    directory = tmp_path / "ui" / SESSION_A
    directory.mkdir(parents=True)
    os.utime(directory, (NOW - 30 * DAY, NOW - 30 * DAY))

    report = sweep_results(tmp_path, policy=_policy(), now=NOW)

    assert report.sessions_removed == 1
    assert not directory.exists()


def test_a_missing_results_root_is_reported_rather_than_invented(tmp_path: Path) -> None:
    report = sweep_results(tmp_path / "never-ran", policy=_policy(), now=NOW)

    assert report.sweep_root_exists is False
    assert report.sessions_removed == 0
    assert "nothing to sweep" in report.summary()


def test_a_disabled_policy_touches_nothing(tmp_path: Path) -> None:
    _session(tmp_path, SESSION_A, age_days=999, now=NOW)

    report = sweep_results(tmp_path, policy=RetentionPolicy(max_age_seconds=None), now=NOW)

    assert report.sessions_removed == 0
    assert (tmp_path / "ui" / SESSION_A).is_dir()


def test_a_sweep_stops_at_its_bound_and_says_so(tmp_path: Path) -> None:
    for index in range(5):
        _session(tmp_path, f"{index:032d}", age_days=30, now=NOW)

    report = sweep_results(tmp_path, policy=RetentionPolicy(max_sessions_per_sweep=2), now=NOW)

    assert report.sessions_removed == 2
    assert report.kept == 3  # the three the sweep did not reach are still there
    assert report.truncated is True
    assert "max_sessions_per_sweep" in report.summary()
    # The oldest go first, so a truncated sweep still makes progress.
    assert len(list((tmp_path / "ui").iterdir())) == 3


def test_a_dry_run_reports_without_removing(tmp_path: Path) -> None:
    _session(tmp_path, SESSION_A, age_days=30, now=NOW)

    report = sweep_results(tmp_path, policy=_policy(), now=NOW, dry_run=True)

    assert report.dry_run is True
    assert report.sessions_removed == 1
    assert report.files_removed == 2
    assert (tmp_path / "ui" / SESSION_A).is_dir()


# --------------------------------------------------------------------------- #
# Containment
# --------------------------------------------------------------------------- #


def test_a_symlinked_session_is_skipped_not_followed(tmp_path: Path) -> None:
    outside = tmp_path.parent / "precious"
    outside.mkdir()
    (outside / "keep-me.txt").write_text("data", encoding="utf-8")
    ui = tmp_path / "ui"
    ui.mkdir()
    link = ui / SESSION_A
    link.symlink_to(outside, target_is_directory=True)

    report = sweep_results(tmp_path, policy=_policy(), now=NOW)

    assert report.sessions_removed == 0
    assert any("symlink" in note for note in report.skipped)
    assert (outside / "keep-me.txt").is_file()


def test_a_session_id_that_is_not_a_safe_segment_is_skipped(tmp_path: Path) -> None:
    ui = tmp_path / "ui"
    (ui / "..").mkdir(parents=True) if False else None
    ui.mkdir(parents=True)
    (ui / ".hidden").mkdir()
    report = sweep_results(tmp_path, policy=_policy(), now=NOW)

    assert report.sessions_removed == 0
    assert any("not a usable session id" in note for note in report.skipped)
    assert (ui / ".hidden").is_dir()


def test_the_sweep_never_touches_a_sibling_of_the_runs_subdirectory(tmp_path: Path) -> None:
    """``results/raw``, ``results/report`` and a CLI run's own output are not ours."""

    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "brainos.json").write_text("{}", encoding="utf-8")
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "report.md").write_text("kept", encoding="utf-8")
    _session(tmp_path, SESSION_A, age_days=90, now=NOW)

    sweep_results(tmp_path, policy=_policy(), now=NOW)

    assert (tmp_path / "raw" / "brainos.json").is_file()
    assert (tmp_path / "report" / "report.md").is_file()


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_the_report_is_json_serializable_and_carries_its_scope(tmp_path: Path) -> None:
    _session(tmp_path, SESSION_A, age_days=30, now=NOW)
    _session(tmp_path, SESSION_B, age_days=1, now=NOW)

    report = sweep_results(tmp_path, policy=_policy(), now=NOW)
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["retention_version"] == RETENTION_VERSION
    assert payload["root"] == str(tmp_path)
    assert payload["policy"]["max_age_seconds"] == 7 * DAY
    assert payload["sessions_scanned"] == 1
    assert payload["sessions_removed"] == 1
    assert payload["kept"] == 1
    assert payload["dry_run"] is False


def test_a_report_never_contains_a_credential_shaped_value(tmp_path: Path) -> None:
    """Retention counts files; it must not start quoting what is inside them."""

    directory = _session(tmp_path, SESSION_A, age_days=30, files=1, now=NOW)
    (directory / "run-0001" / "artifact-0.json").write_text(
        '{"api_key": "sk-retention-should-never-appear-0001"}', encoding="utf-8"
    )
    os.utime(directory / "run-0001" / "artifact-0.json", (NOW - 30 * DAY, NOW - 30 * DAY))

    report = sweep_results(tmp_path, policy=_policy(), now=NOW)

    assert "sk-retention" not in json.dumps(report.to_dict())
    assert "sk-retention" not in report.summary()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_the_cli_sweeps_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _session(tmp_path, SESSION_A, age_days=30)  # the CLI reads the real clock

    code = retention.main(["--root", str(tmp_path), "--max-age-days", "7"])

    assert code == 0
    assert "removed=1" in capsys.readouterr().out
    assert not (tmp_path / "ui" / SESSION_A).exists()


def test_the_cli_dry_run_removes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _session(tmp_path, SESSION_A, age_days=30)

    code = retention.main(["--root", str(tmp_path), "--dry-run"])

    assert code == 0
    assert "dry_run=True" in capsys.readouterr().out
    assert (tmp_path / "ui" / SESSION_A).is_dir()


def test_the_cli_writes_a_json_report_when_asked(tmp_path: Path) -> None:
    target = tmp_path / "retention.json"

    code = retention.main(
        ["--root", str(tmp_path), "--format", "json", "--output", str(target)]
    )

    assert code == 0
    assert json.loads(target.read_text(encoding="utf-8"))["retention_version"] == RETENTION_VERSION


def test_the_cli_tolerates_a_host_that_has_never_run_a_benchmark(tmp_path: Path) -> None:
    assert retention.main(["--root", str(tmp_path / "nothing-here"), "--quiet"]) == 0


# --------------------------------------------------------------------------- #
# The runner sweeps before it runs, under the deployment's own policy
# --------------------------------------------------------------------------- #


def test_the_runner_sweep_uses_the_policy_and_leaves_live_sessions_alone(tmp_path: Path) -> None:
    from app.evaluation import EvaluationPolicy, UIEvaluationRunner

    root = tmp_path / "results"
    _session(root, SESSION_A, age_days=30, now=NOW)
    _session(root, SESSION_B, age_days=0.5, now=NOW)

    runner = UIEvaluationRunner(
        results_root=root,
        policy=EvaluationPolicy(results_retention_seconds=7 * DAY),
    )
    report = runner.sweep(now=NOW)

    assert runner.policy.retention_policy().max_age_seconds == 7 * DAY
    assert report.sessions_removed == 1
    assert (root / "ui" / SESSION_B).is_dir()


def test_a_deployment_can_turn_retention_off_and_the_notice_says_so(tmp_path: Path) -> None:
    from app.evaluation import EvaluationPolicy, UIEvaluationRunner

    policy = EvaluationPolicy(results_retention_seconds=None)
    runner = UIEvaluationRunner(results_root=tmp_path / "results", policy=policy)

    notices = " ".join(runner.policy_notice())

    assert "no automatic sweep" in notices
    assert runner.sweep(now=time.time()).sessions_removed == 0


def test_the_runner_notice_discloses_the_window(tmp_path: Path) -> None:
    from app.evaluation import EvaluationPolicy, UIEvaluationRunner

    runner = UIEvaluationRunner(
        results_root=tmp_path / "results",
        policy=EvaluationPolicy(results_retention_seconds=2 * DAY),
    )

    notices = " ".join(runner.policy_notice())

    assert "2 days" in notices
    assert "End session" in notices
