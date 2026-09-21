"""Phase 19: repository paths survive a working directory that is not the checkout.

The audit that opened Phase 19 found the Evaluation tab resolving the committed
dataset, the dataset root, and the whole ``results/ui/`` tree against
``os.getcwd()``. Run the controller from anywhere else and the tab answered
``Dataset not found: benchmarks/context_rot/dataset.jsonl`` — and, had it got
past that, it would have written artifacts into whatever directory the process
happened to start in.

These tests pin the fix from both ends:

* :mod:`checkout` decides *when* to anchor, and only ever anchors to the
  checkout — never to a guessed location;
* the runner, the storage default, and the CLIs are wired to it, so a process
  started outside the repository still finds the dataset and still writes under
  the repository it belongs to;
* inside the checkout nothing changed, because the plan's relative layout is
  what makes a re-run command portable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import checkout
from app.evaluation import (
    DATASET_ROOT,
    DEFAULT_DATASET,
    UI_RESULTS_ROOT,
    EvaluationPolicy,
    UIEvaluationRunner,
)
from tests.fakes import FakeLLMProvider, InMemoryEvaluationStore

REPO_ROOT = Path(__file__).resolve().parents[2]
KEY = "sk-anchored-SECRET-4242"
SESSION = "99999999-8888-7777-6666-555555555555"


@pytest.fixture()
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the rest of the test from a directory outside the checkout."""

    monkeypatch.chdir(tmp_path)
    assert checkout.in_checkout() is False
    return tmp_path


def _runner(root: Path, **kwargs: object) -> UIEvaluationRunner:
    kwargs.setdefault("results_root", root / "results")
    kwargs.setdefault("evaluation_store", InMemoryEvaluationStore())
    kwargs.setdefault("provider_factory", FakeLLMProvider)
    return UIEvaluationRunner(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# checkout
# --------------------------------------------------------------------------- #


def test_the_repository_root_is_the_checkout_the_module_lives_in() -> None:
    assert checkout.REPO_ROOT == REPO_ROOT
    assert (checkout.REPO_ROOT / "BrainOS_Context_Lab_Implementation_Plan.md").is_file()


def test_in_checkout_accepts_the_root_and_its_children(monkeypatch: pytest.MonkeyPatch) -> None:
    assert checkout.in_checkout(REPO_ROOT) is True
    assert checkout.in_checkout(REPO_ROOT / "src" / "app") is True

    monkeypatch.chdir(REPO_ROOT / "tests")
    assert checkout.in_checkout() is True


def test_in_checkout_rejects_everything_else(tmp_path: Path) -> None:
    assert checkout.in_checkout(tmp_path) is False
    assert checkout.in_checkout(Path("/")) is False


def test_inside_the_checkout_paths_are_left_exactly_as_the_plan_writes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)

    # Not just equal — the same relative path the plan documents, which is what
    # keeps an artifact's re-run command portable between machines.
    assert checkout.repo_anchored(UI_RESULTS_ROOT) == Path("results")
    assert checkout.repo_anchored(DEFAULT_DATASET, must_exist=True) == DEFAULT_DATASET
    assert not checkout.repo_anchored(DEFAULT_DATASET).is_absolute()


def test_outside_the_checkout_paths_anchor_to_it(elsewhere: Path) -> None:
    assert checkout.repo_anchored(UI_RESULTS_ROOT) == REPO_ROOT / UI_RESULTS_ROOT
    assert checkout.repo_anchored(DATASET_ROOT) == REPO_ROOT / DATASET_ROOT
    assert checkout.repo_anchored(DEFAULT_DATASET, must_exist=True).exists()


def test_an_absent_input_falls_back_to_the_documented_relative_path(elsewhere: Path) -> None:
    """A missing benchmark is reported where the plan says it lives.

    ``must_exist`` exists so the failure message names
    ``benchmarks/context_rot/dataset.jsonl`` rather than an invented absolute
    path — an installed wheel has no ``benchmarks/`` at all.
    """

    missing = Path("benchmarks/context_rot/not-a-real-tier.jsonl")

    assert checkout.repo_anchored(missing, must_exist=True) == missing
    assert checkout.repo_anchored(missing) == REPO_ROOT / missing


# --------------------------------------------------------------------------- #
# The runner the Evaluation tab drives
# --------------------------------------------------------------------------- #


def test_the_runner_anchors_its_roots_when_started_outside_the_checkout(
    elsewhere: Path,
) -> None:
    runner = UIEvaluationRunner(
        evaluation_store=InMemoryEvaluationStore(), provider_factory=FakeLLMProvider
    )

    assert runner.results_root == REPO_ROOT / "results"
    assert runner.dataset_root == REPO_ROOT / "benchmarks"
    dataset = runner._default_dataset()
    assert dataset is not None and dataset.exists()
    assert dataset.resolve() == (REPO_ROOT / DEFAULT_DATASET).resolve()


def test_the_tabs_dataset_menu_is_populated_from_any_directory(elsewhere: Path) -> None:
    runner = UIEvaluationRunner(
        evaluation_store=InMemoryEvaluationStore(), provider_factory=FakeLLMProvider
    )

    choices = runner.dataset_choices()

    assert choices, "the committed dataset must be offered"
    assert Path(choices[0]).resolve() == (REPO_ROOT / DEFAULT_DATASET).resolve()


def test_a_preview_works_from_any_directory_and_records_a_portable_path(
    elsewhere: Path,
) -> None:
    """The bug the audit reproduced, in one assertion.

    Before the fix this raised ``EvaluationRequestError: Dataset not found:
    benchmarks/context_rot/dataset.jsonl`` as soon as the process was started
    outside the checkout.
    """

    runner = UIEvaluationRunner(
        results_root=elsewhere / "results",
        evaluation_store=InMemoryEvaluationStore(),
        provider_factory=FakeLLMProvider,
    )

    preview = runner.preview(session_id=SESSION, preset_name="quick", limit=1)

    assert preview.tasks_selected == 1
    assert preview.dataset_sha256
    # Anchored for I/O, portable in the artifact: a reader re-running this run
    # should not be handed this machine's directory layout.
    assert preview.dataset == str(DEFAULT_DATASET)
    assert not Path(preview.dataset).is_absolute()


def test_a_run_completes_from_any_directory_and_writes_under_its_own_root(
    elsewhere: Path,
) -> None:
    root = elsewhere / "results"
    runner = UIEvaluationRunner(
        results_root=root,
        dataset_root=REPO_ROOT / "benchmarks",
        policy=EvaluationPolicy(max_tasks=1, max_requests=10),
        evaluation_store=InMemoryEvaluationStore(),
        provider_factory=FakeLLMProvider,
    )

    view = runner.run(session_id=SESSION, preset_name="quick", modes=["brainos"], limit=1)

    assert view.ok is True
    assert (root / "ui" / SESSION).is_dir()
    # Nothing was written into the checkout by a run that was pointed elsewhere.
    assert not (REPO_ROOT / "results" / "ui" / SESSION).exists()


# --------------------------------------------------------------------------- #
# Storage default and the CLIs
# --------------------------------------------------------------------------- #


def test_the_database_default_anchors_without_breaking_the_documented_one(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import storage.sqlite as sqlite_mod

    monkeypatch.delenv("BRAINOS_LAB_DB", raising=False)

    assert sqlite_mod.default_database_path() == REPO_ROOT / "data" / "brainos_lab.sqlite3"

    monkeypatch.chdir(REPO_ROOT)
    assert sqlite_mod.default_database_path() == Path("data") / "brainos_lab.sqlite3"


def test_the_single_mode_cli_defaults_to_the_committed_dataset_from_anywhere(
    elsewhere: Path,
) -> None:
    """The installed console script is the case the audit could not rule out."""

    from evaluation import run as run_cli

    default = next(
        action.default for action in run_cli.build_parser()._actions if action.dest == "dataset"
    )

    assert default == REPO_ROOT / DEFAULT_DATASET
    assert Path(default).is_file()


def test_the_pipeline_cli_is_usable_from_outside_the_checkout(elsewhere: Path) -> None:
    """End to end: no path argument, a working directory that is not the repo."""

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.pipeline",
            "--dry-run",
            "--limit",
            "1",
            "--no-plots",
            "--output-dir",
            str(elsewhere / "out"),
        ],
        cwd=elsewhere,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dataset" not in result.stderr
    assert (elsewhere / "out" / "report" / "report.md").is_file()
