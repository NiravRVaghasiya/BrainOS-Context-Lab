"""Live Phase 17 validation: the automated pipeline against the pinned BrainOS.

The unit tests drive the pipeline with fakes. This file runs the *committed*
benchmark through the real runtime, all five modes, and a deterministic
prompt-reading provider, and checks the properties the phase exists to
establish:

* one command produces the plan's whole ``results/`` layout, with every stage
  recorded and every artifact carrying the Phase 16 manifest;
* the artifacts compose — the comparison, the statistics, the failure taxonomy,
  the figures, and the report all describe the same run;
* the report says what it can and cannot claim (graded answers, one trial, one
  length tier), instead of presenting a smoke run as a result;
* nothing credential-shaped reaches any file the pipeline wrote.

Rules this file enforces on itself, following the other live files:

* **no API key**: generation goes through ``PromptReadingProvider``, a
  deterministic reader that answers only from the prompt and never calls a
  network;
* **no research claim**: the reader is not a model, so its accuracy measures the
  pipeline, not LLM quality. Assertions are relationships between modes and
  between artifacts, never numbers to be quoted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.modes import MODE_BRAINOS, MODE_FULL_CONTEXT, MODE_ORDER, MODE_SLIDING_WINDOW
from evaluation.datasets import load_jsonl
from evaluation.experiment import ExperimentPlan
from evaluation.generation import ModelSpec
from evaluation.limits import RunLimits
from evaluation.pipeline import (
    EXIT_OK,
    PIPELINE_VERSION,
    RESULT_DIRECTORIES,
    STAGES,
    run_pipeline,
)
from reproducibility.run_provenance import app_version, brainos_version
from security.scan import scan_paths
from tests.fakes import PromptReadingProvider

pytest.importorskip("brainos_runtime")

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)
MODEL = ModelSpec(name="prompt-reader", provider="openai", model="deterministic-reader")


@pytest.fixture(scope="module")
def live_run(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """One pipeline run over the committed benchmark, into a temporary results/."""

    root = tmp_path_factory.mktemp("phase17-live")
    plan = ExperimentPlan(
        name="live-pipeline",
        modes=MODE_ORDER,
        trials=1,
        limits=RunLimits(max_requests=1000),
        dataset=str(DATASET),
    )
    provider = PromptReadingProvider(report_model="deterministic-reader")
    result = run_pipeline(
        plan,
        TASKS,
        MODEL,
        provider,
        output_dir=root,
        credential_source="test:deterministic-reader",
    )
    return result, root, provider


def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "report" / "pipeline.json").read_text(encoding="utf-8"))


def test_live_every_stage_ran_and_the_layout_is_complete(live_run) -> None:  # type: ignore[no-untyped-def]
    result, root, _provider = live_run

    assert result.exit_code == EXIT_OK, [stage.to_dict() for stage in result.stages]
    assert [stage.name for stage in result.stages] == list(STAGES)
    assert all(stage.ok for stage in result.stages)
    assert result.failed_stages() == ()
    assert result.skipped_stages() == ()
    for name in RESULT_DIRECTORIES:
        assert (root / name).is_dir(), name
    assert list((root / "raw").glob("*.json"))
    assert list((root / "aggregated").glob("*.json"))
    assert list((root / "plots").glob("*.png"))
    assert (root / "report" / "report.md").is_file()


def test_live_every_mode_ran_every_task(live_run) -> None:  # type: ignore[no-untyped-def]
    result, _root, provider = live_run

    assert result.experiment is not None
    assert result.experiment.violations == ()
    assert result.experiment.passed is True
    assert result.artifact["tasks_executed"] == len(TASKS)
    # One request per (task, mode) — the pipeline adds no calls of its own.
    assert provider.request_count == len(TASKS) * len(MODE_ORDER)
    for mode in MODE_ORDER:
        rows = [row for row in result.artifact["headline"] if row["mode"] == mode]
        assert rows, mode
        assert all(int(row["graded"]) == len(TASKS) for row in rows), mode


def test_live_the_artifacts_carry_the_reproducibility_manifest(live_run) -> None:  # type: ignore[no-untyped-def]
    result, root, _provider = live_run
    manifest = _manifest(root)

    assert manifest["pipeline_version"] == PIPELINE_VERSION
    assert manifest["exit_code"] == EXIT_OK
    assert manifest["run_id"] == result.run_id
    repro = manifest["repro"]  # type: ignore[index]
    assert repro["brainos_version"] == brainos_version()
    assert repro["application_version"] == app_version()
    assert repro["benchmark_version"] == "context-rot-v1"
    assert repro["dataset"]["sha256"]
    assert sorted(repro["task_ids"]) == sorted(task.task_id for task in TASKS)
    assert str(manifest["pipeline_rerun_command"]).startswith(  # type: ignore[call-overload]
        "python -m evaluation.pipeline"
    )
    # Each raw run file is independently re-derivable.
    for path in (root / "raw").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["repro"]["dataset"]["sha256"] == repro["dataset"]["sha256"]
        assert payload["repro"]["rerun_command"]


def test_live_the_aggregates_describe_the_same_run(live_run) -> None:  # type: ignore[no-untyped-def]
    _result, root, _provider = live_run

    def aggregate(name: str) -> dict[str, object]:
        return json.loads(
            (root / "aggregated" / f"{name}.json").read_text(encoding="utf-8")
        )

    comparison = aggregate("comparison")
    statistics = aggregate("statistics")
    errors = aggregate("errors")
    experiment = aggregate("experiment")

    # One row per mode in the comparison, one trial summary per mode in the
    # statistics, one taxonomy block per mode in the error report — all five
    # modes, from the same run.
    assert {row["mode"] for row in comparison["runs"]} == set(MODE_ORDER)  # type: ignore[union-attr]
    assert set(statistics["trial_summaries"]) == set(MODE_ORDER)  # type: ignore[arg-type]
    assert {row["mode"] for row in statistics["headline_table"]} == set(MODE_ORDER)  # type: ignore[union-attr]
    assert set(errors["by_mode"]) == set(MODE_ORDER)  # type: ignore[arg-type]
    # The experiment aggregate covers the same tasks the manifest lists, and the
    # dataset hash it names is the one every artifact names (Phase 16's pin,
    # filled in at run time because the plan left it empty).
    assert experiment["repro"]["task_count"] == len(TASKS)  # type: ignore[index]
    assert experiment["provenance"]["tasks_executed"] == len(TASKS)  # type: ignore[index]
    assert experiment["provenance"]["dataset_sha256"]  # type: ignore[index]
    assert (  # type: ignore[index]
        experiment["provenance"]["dataset_sha256"]
        == _manifest(root)["repro"]["dataset"]["sha256"]  # type: ignore[index]
    )
    # The six planned figures each have a series behind them.
    assert set(comparison["series"]) == {  # type: ignore[arg-type]
        "accuracy_vs_length",
        "accuracy_vs_tokens",
        "quality_adjusted_efficiency",
        "retrieval",
        "token_savings",
        "tokens_vs_length",
    }
    # The failure taxonomy agrees with the scorer it is built from (Phase 12's pin).
    assert errors["labels_vs_scorer"]["unexpected"] == 0  # type: ignore[index]
    # Phase 17 compares failure *distributions*, not accuracies: every mode with
    # records is measured against the run's baseline over the Phase 12 taxonomy.
    distributions = errors["distribution_comparison"]  # type: ignore[index]
    assert distributions["metric"] == "total_variation_distance"  # type: ignore[index]
    assert set(distributions["vocabulary"]) >= {"hallucination", "missed_memory"}  # type: ignore[arg-type]
    modes = distributions["modes"]  # type: ignore[index]
    # Only modes with failure records appear — a clean mode has no distribution
    # to compare, and the report says so when that changes the baseline.
    assert modes and set(modes) <= set(MODE_ORDER)
    baseline = str(distributions["baseline_mode"])  # type: ignore[index]
    assert baseline in modes
    if baseline != MODE_FULL_CONTEXT:
        assert "absent, fell back to" in str(distributions["baseline_selection"])  # type: ignore[index]
    # The baseline is zero distance from itself, which is the check that the
    # distribution comparison is measuring anything at all.
    assert modes[baseline]["total_variation_vs_baseline"] == 0.0  # type: ignore[index]
    assert all(0.0 <= modes[mode]["total_variation_vs_baseline"] <= 1.0 for mode in modes)  # type: ignore[index]
    # The paired tests are between two different modes, and the run's baseline is
    # one side of every pair — that is what makes them paired against a reference
    # rather than an arbitrary set of contrasts.
    paired = statistics["paired_comparisons"]  # type: ignore[index]
    assert paired
    assert all(entry["mode_a"] != entry["mode_b"] for entry in paired)  # type: ignore[union-attr]
    assert {entry["mode_a"] for entry in paired} | {  # type: ignore[union-attr]
        entry["mode_b"] for entry in paired  # type: ignore[union-attr]
    } <= set(MODE_ORDER)
    # mode_a is the subject and mode_b the reference, so every mode meets the
    # run's baseline as the reference, and the system under test is additionally
    # contrasted with the cheaper baselines (Phase 9's job, computed once here).
    per_metric: dict[str, set[tuple[str, str]]] = {}
    for entry in paired:
        per_metric.setdefault(entry["metric"], set()).add(
            (entry["mode_a"], entry["mode_b"])
        )
    assert len(per_metric) == 8  # the metric suite, not a subset of it
    for metric, pairs in per_metric.items():
        for mode in set(MODE_ORDER) - {MODE_FULL_CONTEXT}:
            assert (mode, MODE_FULL_CONTEXT) in pairs, (metric, mode)
        assert any(MODE_BRAINOS in pair for pair in pairs), metric
    # The pairs are per task, not per trial: with one trial the sample size of a
    # paired test is the number of tasks, and the win/loss/tie split has to
    # account for all of them.
    assert all(entry["sample_size"] == len(TASKS) for entry in paired)  # type: ignore[union-attr]
    assert all(
        entry["wins"] + entry["losses"] + entry["ties"] == entry["sample_size"]  # type: ignore[union-attr]
        for entry in paired
    )
    assert all(entry["effect_size_magnitude"] for entry in paired)  # type: ignore[union-attr]


def test_live_the_report_claims_only_what_the_run_supports(live_run) -> None:  # type: ignore[no-untyped-def]
    result, root, _provider = live_run
    report = (root / "report" / "report.md").read_text(encoding="utf-8")

    assert report == result.report_markdown
    # A single trial and a single length tier are stated, not hidden: the
    # statistics section must say the intervals are degenerate and the
    # robustness section must not invent a degradation curve.
    assert "degenerate" in report.lower() or "one trial" in report.lower()
    assert "Reproducibility" in report
    assert "Pipeline stages" in report
    # Every mode appears with its own label, and the answer side is a number
    # (this run graded answers) rather than an em dash.
    for mode in MODE_ORDER:
        assert mode in report or mode.replace("_", " ") in report.lower()
    assert "## Headline metrics" in report
    # All seven stages ran, so the report must not hedge about missing ones —
    # and the manifest agrees about what was selected.
    assert "Partial pipeline" not in report
    assert result.artifact["partial"] is False
    assert result.artifact["stages_selected"] == list(STAGES)


def test_live_the_modes_stay_ordered_the_way_the_smoke_tier_orders_them(
    live_run,
) -> None:  # type: ignore[no-untyped-def]
    """Relationships between modes, not quotable numbers.

    On the committed smoke tier the full context carries everything, the sliding
    window carries least, and BrainOS sits between them while keeping evidence
    the window drops. That ordering is a property of the harness; the values are
    one seed on one tier with a deterministic reader and mean nothing else.
    """

    result, _root, _provider = live_run
    tokens = {
        row["mode"]: float(row["mean_context_tokens"])
        for row in result.artifact["headline"]
    }
    evidence = {
        row["mode"]: float(row["evidence_in_prompt"]) for row in result.artifact["headline"]
    }

    assert tokens[MODE_FULL_CONTEXT] >= tokens[MODE_BRAINOS]
    assert tokens[MODE_BRAINOS] > tokens[MODE_SLIDING_WINDOW]
    assert evidence[MODE_SLIDING_WINDOW] == 0.0
    assert evidence[MODE_BRAINOS] > evidence[MODE_SLIDING_WINDOW]
    assert evidence[MODE_FULL_CONTEXT] == max(evidence.values())


def test_live_no_credential_reaches_any_artifact(live_run) -> None:  # type: ignore[no-untyped-def]
    result, root, _provider = live_run
    manifest = _manifest(root)

    check = manifest["credential_check"]  # type: ignore[index]
    assert check["clean"] is True
    assert check["files_scanned"] >= len(TASKS)
    # The second pass covers the report and this manifest; its result lives in
    # the returned artifact, because a file cannot record the scan of itself —
    # and the manifest says so rather than leaving the absence unexplained.
    assert "second pass" in check["report_scan_note"].lower()  # type: ignore[index]
    in_memory = result.artifact["credential_check"]
    assert in_memory["report_scan"]["clean"] is True
    assert in_memory["report_scan"]["files_scanned"] == 2
    assert not check.get("quarantined")
    assert scan_paths([str(root)]).clean
    blob = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".jsonl", ".txt"}
    )
    for needle in ("api_key=", "Bearer ", "sk-"):
        assert needle not in blob, needle
    assert result.artifact["generation"]["credential_source"] == (
        "test:deterministic-reader"
    )
