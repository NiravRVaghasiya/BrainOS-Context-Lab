"""Phase 17: one command that produces the plan's whole ``results/`` layout.

These tests run the pipeline as a **dry run** against the committed dataset, so
they need no API key, no provider, and no BrainOS runtime. What they pin is not
a metric — it is the set of properties that make an automated pipeline
trustworthy:

* it writes the four directories the plan names, in the schemas the earlier
  phases defined (raw run files, aggregates, figures, report);
* every artifact carries the reproducibility manifest, so a number in the report
  can be traced to a dataset digest and a re-run command;
* a stage that fails never silently produces a report that looks complete — the
  report says which stages ran;
* the CLI refuses the two mistakes that cost money or leak a credential
  (a model with no dry run and no key environment, ``--api-key`` as an argument).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.modes import MODE_BRAINOS, MODE_FULL_CONTEXT, MODE_RAG
from evaluation import pipeline as pipeline_module
from evaluation.datasets import load_jsonl
from evaluation.experiment import ExperimentPlan
from evaluation.limits import PRESETS, RunLimits
from evaluation.pipeline import (
    EXIT_OK,
    EXIT_STAGE_FAILED,
    PIPELINE_VERSION,
    REPORT_PREVIEW_CHARS,
    RESULT_DIRECTORIES,
    STAGES,
    PipelineError,
    build_parser,
    expand_stages,
    main,
    pipeline_layout,
    pipeline_rerun_command,
    run_pipeline,
)
from evaluation.reports import markdown_report
from tests.fakes import RecordingProvider

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)[:2]
MODES = (MODE_FULL_CONTEXT, MODE_RAG, MODE_BRAINOS)


def _plan(**overrides: object) -> ExperimentPlan:
    defaults: dict[str, object] = {
        "name": "pipeline",
        "modes": MODES,
        "dataset": DATASET,
        "limits": RunLimits(max_requests=200),
    }
    defaults.update(overrides)
    return ExperimentPlan(**defaults)  # type: ignore[arg-type]


def _dry_run(tmp_path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
    """A retrieval-only pipeline run into ``tmp_path`` (no plots unless asked)."""

    kwargs.setdefault("render_plots", False)
    return run_pipeline(_plan(), TASKS, output_dir=tmp_path, **kwargs)  # type: ignore[arg-type]


def _manifest(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / "report" / "pipeline.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_layout_creates_exactly_the_four_directories_the_plan_names(tmp_path: Path) -> None:
    layout = pipeline_layout(tmp_path / "results").ensure()

    assert [name for name in RESULT_DIRECTORIES] == ["raw", "aggregated", "plots", "report"]
    for name in RESULT_DIRECTORIES:
        assert layout.directory(name).is_dir()
    with pytest.raises(PipelineError, match="Unknown result directory"):
        layout.directory("secrets")


def test_layout_to_dict_records_every_directory(tmp_path: Path) -> None:
    payload = pipeline_layout(tmp_path).to_dict()

    assert payload["root"] == str(tmp_path)
    assert payload["report"].endswith("report")


# --------------------------------------------------------------------------- #
# A dry run writes the whole layout
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_dry_run_writes_raw_aggregated_and_report_artifacts(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert result.exit_code == EXIT_OK
    assert result.dry_run is True
    raw = sorted(path.name for path in (tmp_path / "raw").glob("*.json"))
    assert raw == ["brainos.json", "full_context.json", "rag.json"]
    assert (tmp_path / "raw" / "error-records.jsonl").exists()
    aggregated = sorted(path.name for path in (tmp_path / "aggregated").glob("*.json"))
    assert aggregated == [
        "comparison.json",
        "errors.json",
        "experiment.json",
        "statistics.json",
    ]
    assert (tmp_path / "report" / "pipeline.json").exists()
    assert (tmp_path / "report" / "report.md").exists()


@pytest.mark.requires_runtime
def test_every_stage_is_recorded_with_a_status_and_artifacts(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)

    assert [stage.name for stage in result.stages] == list(STAGES)
    assert result.failed_stages() == ()
    # The figure stage was asked not to render, so it is *skipped* — a distinct
    # status from "ok", because a reader must be able to tell "no figures were
    # wanted" from "figures were produced".
    assert result.skipped_stages() == ("plots",)
    skipped = result.stage("plots")
    assert skipped is not None and not skipped.ok
    assert skipped.detail
    for name in ("experiment", "raw", "comparison", "statistics", "errors", "report"):
        stage = result.stage(name)
        assert stage is not None and stage.ok, name
        assert stage.artifacts, f"{name} wrote nothing"
        assert all(Path(path).exists() for path in stage.artifacts)


@pytest.mark.requires_runtime
def test_raw_run_files_carry_their_own_reproducibility_manifest(tmp_path: Path) -> None:
    _dry_run(tmp_path)

    for name in ("full_context", "rag", "brainos"):
        payload = json.loads((tmp_path / "raw" / f"{name}.json").read_text(encoding="utf-8"))
        repro = payload["repro"]
        assert repro["dataset"]["sha256"]
        assert repro["application_version"]
        assert repro["brainos_version"]
        # The raw file's own re-run command re-derives that one mode.
        assert repro["rerun_command"].startswith("python -m evaluation.run")
        assert f"--mode {name}" in repro["rerun_command"]


@pytest.mark.requires_runtime
def test_plots_stage_renders_one_figure_per_report_chart(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib", reason="figures need matplotlib")

    result = run_pipeline(_plan(), TASKS, output_dir=tmp_path, render_plots=True)

    figures = sorted(path.name for path in (tmp_path / "plots").glob("*.png"))
    assert len(figures) == 6
    assert result.artifact["plots"]
    assert result.skipped_stages() == ()


@pytest.mark.requires_runtime
def test_report_markdown_renders_the_sections_a_reader_needs(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)
    report = result.report_markdown

    # Every section the report renderer owns, in order: what the run may claim,
    # its identity, the matrix, the statistics, the failure taxonomy, length
    # robustness, cost, the controlled-comparison checks, security, the stages
    # that ran, and how to reproduce it.
    headings = [line for line in report.splitlines() if line.startswith("#")]
    assert headings == [
        "# BrainOS Context Lab — evaluation report",
        "## What this report can claim",
        "## Run identity",
        "## Headline metrics",
        "## Statistical evaluation",
        "## Failure analysis",
        "## Context-length robustness",
        "## Cost",
        "## Controlled-comparison checks",
        "## Security",
        "## Pipeline stages",
        "## Reproducibility",
    ]
    # A retrieval-only run grades nothing, so the answer-side metrics are unset
    # rather than zero — the report must not invent a 0.000.
    assert "—" in report
    written = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert written == report


@pytest.mark.requires_runtime
def test_report_discloses_a_changed_distribution_baseline(tmp_path: Path) -> None:
    """A mode that failed nothing cannot anchor a failure distribution.

    Phase 12 falls back to another mode and records why in ``baseline_selection``;
    the report has to say it too, because every distance in that table is
    relative to the choice — and the mode a reader would expect (the full
    context) is the one that is missing precisely when it was clean.
    """

    result = _dry_run(tmp_path)
    artifact = dict(result.artifact)
    errors = dict(artifact["errors"])
    distributions = dict(errors["distribution_comparison"])
    distributions["baseline_mode"] = MODE_BRAINOS
    distributions["baseline_selection"] = (
        f"requested:{MODE_FULL_CONTEXT} (absent, fell back to {MODE_BRAINOS})"
    )
    errors["distribution_comparison"] = distributions
    artifact["errors"] = errors

    report = markdown_report(artifact)

    assert f"Distance is total variation against `{MODE_BRAINOS}`" in report
    assert f"The run asked for `{MODE_FULL_CONTEXT}`" in report
    assert "cannot anchor a failure distribution" in report
    # When the requested baseline did report failures, the table names it with no
    # caveat at all.
    honest = dict(result.artifact)
    honest_errors = dict(honest["errors"])
    honest_distributions = dict(honest_errors["distribution_comparison"])
    honest_distributions["baseline_selection"] = "explicit"
    honest_distributions["baseline_mode"] = str(
        sorted(honest_distributions["modes"])[0]
    )
    honest_errors["distribution_comparison"] = honest_distributions
    honest["errors"] = honest_errors

    clean = markdown_report(honest)
    assert "cannot anchor a failure distribution" not in clean
    assert "The run asked for" not in clean


@pytest.mark.requires_runtime
def test_manifest_preview_is_capped_and_the_full_report_is_on_disk(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)
    manifest = _manifest(tmp_path)

    assert len(str(manifest["report_markdown_preview"])) <= REPORT_PREVIEW_CHARS
    assert len(result.report_markdown) >= len(str(manifest["report_markdown_preview"]))


def test_manifest_records_versions_stages_and_the_exit_code(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)
    manifest = _manifest(tmp_path)

    assert manifest["pipeline_version"] == PIPELINE_VERSION
    assert manifest["exit_code"] == result.exit_code
    assert manifest["run_id"] == result.run_id
    assert manifest["repro"]["dataset"]["path"] == str(DATASET)
    assert manifest["repro"]["task_count"] == len(TASKS)
    assert [stage["name"] for stage in manifest["stages"]] == list(STAGES)  # type: ignore[index]
    assert manifest["dry_run"] is True
    assert manifest["generation"] == {  # type: ignore[comparison-overlap]
        "enabled": False,
        "credential_source": "none (dry run)",
        "settings": {},
    }


@pytest.mark.requires_runtime
def test_generated_run_records_the_model_and_the_credential_route(tmp_path: Path) -> None:
    from evaluation.generation import ModelSpec

    spec = ModelSpec(model="fake-model", temperature=0.0, max_tokens=32)
    result = run_pipeline(
        _plan(),
        TASKS,
        spec,
        RecordingProvider(),
        output_dir=tmp_path,
        render_plots=False,
        credential_source="environment:TEST_KEY",
    )
    manifest = _manifest(tmp_path)

    assert result.dry_run is False
    assert manifest["dry_run"] is False
    assert manifest["generation"]["enabled"] is True  # type: ignore[index]
    assert manifest["generation"]["credential_source"] == "environment:TEST_KEY"  # type: ignore[index]
    assert manifest["model"]["model"] == "fake-model"  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Stage selection and failure
# --------------------------------------------------------------------------- #


def test_expand_stages_adds_the_dependencies_a_stage_is_made_of() -> None:
    assert expand_stages(["report"]) == ("experiment", "report")
    # The error taxonomy grades answers, so it needs the experiment, not the raw
    # run files (which the comparison stage is the one that consumes).
    assert expand_stages(["errors"]) == ("experiment", "errors")
    assert expand_stages(["comparison"]) == ("experiment", "raw", "comparison")
    assert expand_stages(list(STAGES)) == tuple(STAGES)
    # Order is the pipeline's, not the caller's: a report cannot precede the
    # experiment it summarises.
    assert expand_stages(["report", "raw"]) == ("experiment", "raw", "report")


def test_expand_stages_rejects_an_unknown_stage() -> None:
    with pytest.raises(PipelineError, match="Unknown pipeline stage"):
        expand_stages(["vibes"])


@pytest.mark.requires_runtime
def test_a_partial_run_discloses_which_stages_ran(tmp_path: Path) -> None:
    result = run_pipeline(
        _plan(),
        TASKS,
        output_dir=tmp_path,
        render_plots=False,
        stages=expand_stages(["report"]),
    )

    assert [stage.name for stage in result.stages] == ["experiment", "report"]
    assert "Partial pipeline" in result.report_markdown
    assert "Only these stages ran" in result.report_markdown
    # The experiment aggregate exists (that stage ran); the ones that were not
    # selected do not, and the report's stage table is what says so.
    aggregates = sorted(path.name for path in (tmp_path / "aggregated").glob("*.json"))
    assert aggregates == ["experiment.json"]
    assert list((tmp_path / "plots").glob("*.png")) == []


@pytest.mark.requires_runtime
def test_a_failing_stage_does_not_raise_and_fails_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**kwargs: object):  # type: ignore[no-untyped-def]
        raise RuntimeError("matplotlib exploded")

    monkeypatch.setattr(pipeline_module, "_stage_plots", boom)

    result = run_pipeline(_plan(), TASKS, output_dir=tmp_path, render_plots=True)

    assert result.exit_code == EXIT_STAGE_FAILED
    assert result.failed_stages() == ("plots",)
    failed = result.stage("plots")
    assert failed is not None and not failed.ok
    assert "matplotlib exploded" in failed.error
    # The report still renders, and says the figure stage failed rather than
    # presenting a run that looks complete.
    assert result.report_markdown
    assert "failed" in result.report_markdown.lower()
    stages = {stage["name"]: stage["status"] for stage in _manifest(tmp_path)["stages"]}  # type: ignore[index]
    assert stages["plots"] == "failed"
    assert stages["report"] == "ok"


def test_a_later_stage_is_skipped_when_its_dependency_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**kwargs: object):  # type: ignore[no-untyped-def]
        raise RuntimeError("no experiment for you")

    monkeypatch.setattr(pipeline_module, "_stage_experiment", boom)

    result = run_pipeline(_plan(), TASKS, output_dir=tmp_path, render_plots=False)

    assert result.exit_code == EXIT_STAGE_FAILED
    assert result.failed_stages() == ("experiment",)
    assert set(result.skipped_stages()) == {
        "raw",
        "comparison",
        "statistics",
        "errors",
        "plots",
        "report",
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_cli_dry_run_writes_the_layout_and_exits_zero(tmp_path: Path) -> None:
    code = main(
        [
            "--dry-run",
            "--preset",
            "quick",
            "--limit",
            "1",
            "--no-plots",
            "--quiet",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert code == EXIT_OK
    assert (tmp_path / "report" / "report.md").exists()
    assert (tmp_path / "aggregated" / "comparison.json").exists()


@pytest.mark.requires_runtime
def test_cli_prints_a_summary_with_the_caveats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--dry-run", "--limit", "1", "--no-plots", "--output-dir", str(tmp_path)])

    printed = capsys.readouterr().out
    assert code == EXIT_OK
    assert "results/" in printed or "report" in printed
    assert "dry run" in printed.lower()
    assert "unset" in printed.lower() or "—" in printed


def test_cli_refuses_to_generate_without_a_model(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--output-dir", str(tmp_path), "--quiet"])

    assert "--model is required unless --dry-run" in str(excinfo.value)


def test_cli_reports_a_missing_credential_by_variable_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main(["--model", "gpt-4o-mini", "--output-dir", str(tmp_path), "--quiet"])

    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "never written to a run artifact" in message


def test_cli_rejects_trials_below_one(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--trials must be at least 1"):
        main(["--dry-run", "--trials", "0", "--output-dir", str(tmp_path)])


def test_cli_rejects_an_unknown_preset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--dry-run", "--preset", "yolo", "--output-dir", str(tmp_path)])


def test_cli_does_not_accept_an_api_key_argument() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--api-key", "sk-secret"])
    # ``allow_abbrev=False`` is what stops ``--api-key`` being read as
    # ``--api-key-env`` and a pasted credential being recorded as a name.
    assert parser.allow_abbrev is False
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--api-key" not in options
    assert "--api-key-env" in options


def test_cli_rejects_an_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Unknown pipeline stage"):
        main(["--dry-run", "--stages", "vibes", "--output-dir", str(tmp_path)])


@pytest.mark.requires_runtime
def test_cli_modes_ablations_always_include_the_full_system(tmp_path: Path) -> None:
    code = main(
        [
            "--dry-run",
            "--modes",
            "ablations",
            "--limit",
            "1",
            "--no-plots",
            "--quiet",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert code == EXIT_OK
    assert sorted(path.stem for path in (tmp_path / "raw").glob("*.json")) == [
        "brainos",
        "brainos_no_conflict",
        "brainos_no_memory",
        "brainos_no_relevance",
        "brainos_no_temporal",
    ]


@pytest.mark.requires_runtime
def test_cli_baseline_mode_is_recorded_and_validated(tmp_path: Path) -> None:
    code = main(
        [
            "--dry-run",
            "--limit",
            "1",
            "--no-plots",
            "--quiet",
            "--baseline-mode",
            "rag",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert code == EXIT_OK
    assert _manifest(tmp_path)["baseline_mode"] == "rag"
    with pytest.raises(SystemExit, match="mode"):
        main(
            [
                "--dry-run",
                "--limit",
                "1",
                "--baseline-mode",
                "not_a_mode",
                "--output-dir",
                str(tmp_path),
            ]
        )


# --------------------------------------------------------------------------- #
# Re-run commands and presets
# --------------------------------------------------------------------------- #


def test_rerun_command_spells_out_a_plan_that_is_not_a_preset(tmp_path: Path) -> None:
    plan = _plan(limits=RunLimits(max_tasks=2, max_requests=9))
    command = pipeline_rerun_command(plan, layout=pipeline_layout(tmp_path), dry_run=True)

    assert command.startswith("python -m evaluation.pipeline")
    assert "--dry-run" in command
    assert str(tmp_path) in command
    # No preset is named, because none was used: the budget is reproduced
    # flag by flag instead of by a name that would not resolve.
    assert "--preset" not in command
    assert f"--modes {','.join(MODES)}" in command
    assert "--limit 2" in command
    assert "--max-requests 9" in command


def test_rerun_command_names_the_preset_and_only_its_overrides(tmp_path: Path) -> None:
    plan = ExperimentPlan.from_preset("quick", dataset=DATASET, max_tasks=2)
    command = pipeline_rerun_command(plan, layout=pipeline_layout(tmp_path), dry_run=True)

    assert "--preset quick" in command
    assert "--limit 2" in command
    # The preset's own request ceiling needs no flag: the preset carries it.
    assert "--max-requests" not in command
    assert f"--modes {','.join(plan.resolved_modes())}" not in command


def test_rerun_command_never_contains_a_credential(tmp_path: Path) -> None:
    from evaluation.generation import ModelSpec

    plan = ExperimentPlan.from_preset("quick", dataset=DATASET)
    spec = ModelSpec(model="fake-model", api_key_env="MY_KEY_VAR")
    command = pipeline_rerun_command(
        plan, layout=pipeline_layout(tmp_path), model=spec, dry_run=False
    )

    assert "--api-key-env MY_KEY_VAR" in command
    assert "--api-key " not in command
    assert "--model fake-model" in command
    assert "--dry-run" not in command


def test_every_preset_reports_a_planned_request_ceiling() -> None:
    for name in sorted(PRESETS):
        entry = PRESETS[name]
        assert entry.planned_requests is None or entry.planned_requests >= 0
        assert entry.limits.max_requests is None or entry.limits.max_requests > 0
        assert entry.modes, name


@pytest.mark.requires_runtime
def test_artifact_lists_every_path_the_pipeline_wrote(tmp_path: Path) -> None:
    result = _dry_run(tmp_path)
    listing = result.artifact["artifacts"]
    assert isinstance(listing, dict)

    listed = {
        str(path)
        for group in listing.values()
        if isinstance(group, list)
        for path in group
    }
    written = {str(path) for path in result.artifacts()}
    # Everything the result says it wrote is in the manifest's own inventory,
    # and every inventory entry is a file that exists.
    assert written <= listed | {str(listing["pipeline_manifest"])}
    assert all(Path(path).is_file() for path in written)
    assert str(tmp_path / "report" / "report.md") in listed
    assert listing["layout"]["report"] == str(tmp_path / "report")
