"""The automated evaluation pipeline (plan Phase 17 / §23).

Phases 7–16 built the instruments: a benchmark, a scorer, five baseline modes,
four ablations, cost ceilings, trial statistics, an error taxonomy, a prompt
guard, and a reproducibility manifest. Each one shipped behind its own command,
so producing the plan's result set meant chaining three commands by hand and
hoping the flags agreed:

```bash
python -m evaluation.experiment --preset quick --output exp.json --runs-dir raw/
python -m evaluation.compare raw/*.json --stats --output agg/stats.json --plots-dir plots/
python -m evaluation.errors exp.json --output agg/errors.json --records raw/records.jsonl
```

Phase 17 replaces that with one command that writes the plan's four directories:

```text
benchmarks/context_rot/dataset.jsonl
        |
        v
 [1] experiment   controlled comparison (Phase 9):    -> aggregated/experiment.json
        |         same tasks / model / settings, cost
        |         ceilings, constants check
        v
 [2] raw          one Phase 6/7 run file per          -> raw/{mode}[-tN].json
        |         (mode, trial), each stamped with its
        |         own Phase 16 ``repro`` manifest
        v
 [3] comparison   cross-mode headline + plot series   -> aggregated/comparison.json
 [4] statistics   trials, CIs, paired tests, effects  -> aggregated/statistics.json
 [5] errors       Phase 12 taxonomy plus the failure  -> aggregated/errors.json
        |         *distribution* comparison            + raw/error-records.jsonl
        v
 [6] plots        the six planned figures             -> plots/*.png
 [7] report       human-readable, rendered *from*     -> report/report.md
                  the artifact                         + report/pipeline.json
```

Design rules this module exists to hold:

* **A stage that cannot run is recorded, not fatal.** matplotlib missing, a
  dataset with one length tier, a budget abort — each leaves a ``skipped`` or
  ``failed`` stage with a reason, beside the stages that did produce artifacts.
  A pipeline that dies at stage 6 loses stages 1–5, which is the opposite of
  automated.
* **The report renders from the artifact.** :func:`evaluation.reports.markdown_report`
  is a pure function of ``report/pipeline.json``, so the markdown cannot claim
  anything the JSON does not contain, and an artifact written weeks ago can be
  re-rendered without re-running anything.
* **Provenance is imported, never restated.** Every version string comes from
  :mod:`reproducibility.run_provenance` (Phase 16's first constraint); this
  module defines no ``APPLICATION_VERSION`` of its own.
* **No credential reaches an artifact.** The key is read from an environment
  variable (CLI) or handed in by a caller (UI), given to the generator's
  redaction list, and — as the last line of defence — every file this pipeline
  writes is scanned with :mod:`security.scan` before the run is called clean. A
  finding fails the run.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from baselines.modes import ABLATION_ORDER, MODE_BRAINOS, MODE_ORDER, resolve_mode
from brain.tokenizers import (
    TokenCounter,
    TokenizerUnavailableError,
    estimate_tokens,
    tiktoken_counter,
)
from brain.trace import sanitize_value
from checkout import repo_anchored
from reproducibility.run_provenance import build_manifest
from security.findings import summarize_security
from security.scan import scan_paths

from .analysis import compare_error_distributions
from .datasets import BenchmarkTask, load_jsonl
from .errors import collect_failures, error_report, write_failure_records
from .experiment import (
    BILLING_NOTICE,
    CONTEXT_BUDGET,
    EXPERIMENT_VERSION,
    ExperimentError,
    ExperimentPlan,
    ExperimentRun,
    dataset_sha256,
    run_controlled_experiment,
)
from .generation import (
    MissingCredentialError,
    ModelSpec,
    api_key_from_environment,
    build_provider,
)
from .limits import PRESETS, RunLimits, RunPreset, preset
from .plots import plot_all
from .reports import (
    comparison_report,
    markdown_report,
    statistical_report,
    write_json_report,
    write_markdown_report,
)

#: Identifies the pipeline semantics recorded on its artifact.
PIPELINE_VERSION = "pipeline-v1"

#: The plan's §23 output directories, created under ``--output-dir``.
RESULT_DIRECTORIES: tuple[str, ...] = ("raw", "aggregated", "plots", "report")

#: Default root, matching the repository's ``results/`` (contents Git-ignored).
DEFAULT_OUTPUT_DIR = Path("results")

#: Default dataset: the committed smoke tier, as the plan names it.
DEFAULT_DATASET = Path("benchmarks/context_rot/dataset.jsonl")


def default_dataset_path() -> Path:
    """The committed dataset, anchored to the checkout when the CLI runs outside it.

    Phase 19: resolved at call time, not at import time, so the console script
    works from any working directory — and so the value an artifact records is
    still the plan's relative path when the process *is* in the checkout (see
    :mod:`checkout`).
    """

    return repo_anchored(DEFAULT_DATASET, must_exist=True)

# Stage names, in execution order. ``STAGES`` is also the default selection.
STAGE_EXPERIMENT = "experiment"
STAGE_RAW = "raw"
STAGE_COMPARISON = "comparison"
STAGE_STATISTICS = "statistics"
STAGE_ERRORS = "errors"
STAGE_PLOTS = "plots"
STAGE_REPORT = "report"

STAGES: tuple[str, ...] = (
    STAGE_EXPERIMENT,
    STAGE_RAW,
    STAGE_COMPARISON,
    STAGE_STATISTICS,
    STAGE_ERRORS,
    STAGE_PLOTS,
    STAGE_REPORT,
)

#: Which stage each stage needs. A stage whose dependency produced nothing is
#: *skipped* with a reason instead of failing on an empty input, so
#: ``--stages report`` on a broken experiment reports the breakage rather than
#: writing an empty report that looks like a result.
STAGE_DEPENDENCIES: dict[str, str] = {
    STAGE_RAW: STAGE_EXPERIMENT,
    STAGE_COMPARISON: STAGE_RAW,
    STAGE_STATISTICS: STAGE_EXPERIMENT,
    STAGE_ERRORS: STAGE_EXPERIMENT,
    STAGE_PLOTS: STAGE_EXPERIMENT,
    STAGE_REPORT: STAGE_EXPERIMENT,
}

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

#: Exit codes. ``2`` and ``3`` mean the *comparison* is untrustworthy (the
#: experiment CLI's own codes, kept identical so one script checks both);
#: ``4`` means the *pipeline* is incomplete or wrote something it should not have.
EXIT_OK = 0
EXIT_VIOLATIONS = 2
EXIT_ABORTED = 3
EXIT_STAGE_FAILED = 4

#: How much of the rendered markdown is echoed into the JSON artifact. The report
#: file on disk is the full text; the artifact carries a bounded preview so a
#: reader of the JSON sees the report's shape without a second 40 KB blob.
REPORT_PREVIEW_CHARS = 4000


class PipelineError(RuntimeError):
    """The pipeline cannot run as configured.

    Raised for configuration problems only. A stage that fails mid-run is
    recorded on the result instead, because losing the artifacts of the stages
    that succeeded costs more than the diagnostic is worth.
    """


@dataclass(frozen=True)
class PipelineLayout:
    """The plan's four result directories under one root."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def aggregated(self) -> Path:
        return self.root / "aggregated"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def report(self) -> Path:
        return self.root / "report"

    def directory(self, name: str) -> Path:
        """Return one of the plan's directories, rejecting any other name."""

        if name not in RESULT_DIRECTORIES:
            raise PipelineError(
                f"Unknown result directory {name!r}. Expected {', '.join(RESULT_DIRECTORIES)}."
            )
        return self.root / name

    def ensure(self) -> PipelineLayout:
        """Create every directory and return ``self`` (idempotent)."""

        for name in RESULT_DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            **{name: str(self.root / name) for name in RESULT_DIRECTORIES},
        }


def pipeline_layout(root: str | Path = DEFAULT_OUTPUT_DIR) -> PipelineLayout:
    """Return the layout for ``root`` without creating anything."""

    return PipelineLayout(root=Path(root))


@dataclass(frozen=True)
class StageResult:
    """What one stage did, what it wrote, and how long it took."""

    name: str
    status: str
    duration_seconds: float = 0.0
    artifacts: tuple[str, ...] = ()
    digests: dict[str, str] = field(default_factory=dict)
    detail: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_seconds": round(float(self.duration_seconds), 4),
            "artifacts": list(self.artifacts),
            "digests": dict(self.digests),
            "detail": self.detail,
            "error": self.error,
        }


@dataclass
class _StageContext:
    """Values one stage produces and a later stage consumes."""

    experiment: ExperimentRun | None = None
    run_dicts: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, str] = field(default_factory=dict)
    report_text: str = ""
    #: The credential scan of everything stages 1–6 wrote, computed *before* the
    #: report renders so the report can quote it.
    credential_check: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """One pipeline execution: its stages, its numbers, and its provenance."""

    run_id: str
    timestamp: str
    layout: PipelineLayout
    plan: ExperimentPlan
    stages: tuple[StageResult, ...]
    experiment: ExperimentRun | None
    artifact: dict[str, Any]
    exit_code: int
    #: The full rendered report, kept off the artifact (which carries a preview).
    report_markdown: str = ""
    #: Where ``report/pipeline.json`` was written.
    manifest_path: str = ""

    @property
    def dry_run(self) -> bool:
        return self.experiment is None or self.experiment.dry_run

    @property
    def violations(self) -> tuple[str, ...]:
        return () if self.experiment is None else self.experiment.violations

    @property
    def aborted(self) -> dict[str, Any] | None:
        return None if self.experiment is None else self.experiment.aborted

    def stage(self, name: str) -> StageResult | None:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def artifacts(self) -> list[str]:
        """Every path this pipeline wrote, in stage order."""

        paths = [path for stage in self.stages for path in stage.artifacts]
        if self.manifest_path:
            paths.append(self.manifest_path)
        return paths

    def failed_stages(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.status == STATUS_FAILED)

    def skipped_stages(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.status == STATUS_SKIPPED)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.artifact)


def run_pipeline(
    plan: ExperimentPlan,
    tasks: Sequence[BenchmarkTask],
    model: ModelSpec | None = None,
    provider: Any | None = None,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    token_counter: TokenCounter = estimate_tokens,
    counter_name: str = "",
    secrets: Sequence[str] = (),
    baseline_mode: str = "full_context",
    render_plots: bool = True,
    examples_per_type: int = 1,
    stages: Sequence[str] = STAGES,
    clock: Callable[[], float] = time.perf_counter,
    credential_source: str = "",
    notes: Mapping[str, Any] | None = None,
    run_id: str = "",
) -> PipelineResult:
    """Run every selected stage and write the plan's ``results/`` layout.

    ``model`` / ``provider`` are optional exactly as they are for
    :func:`~evaluation.experiment.run_controlled_experiment`: without them the
    pipeline is a **dry run** that measures prompts, tokens, and retrieval with
    no credential and no spend.

    The function does not raise for a failing stage. It returns a
    :class:`PipelineResult` whose ``stages`` say what happened and whose
    ``exit_code`` is non-zero when the comparison cannot be trusted
    (violations), when the run stopped at a ceiling (abort), or when the
    pipeline is incomplete (a failed stage or a credential finding).

    ``run_id`` lets a caller that has already named the destination directory
    (the Evaluation tab writes to ``results/ui/<session>/<run id>``) keep the
    directory name, the manifest's ``run_id``, and the history row the same
    identifier. Left empty, the pipeline generates one.
    """

    selected = _ordered_stages(stages)
    layout = pipeline_layout(output_dir).ensure()
    resolved_run_id = str(run_id or "").strip() or str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    secret_tuple = tuple(str(value) for value in secrets if value)

    # Hash the dataset once, before any stage runs, so every artifact in this run
    # names the same digest. ``build_manifest`` computes it for the pipeline
    # manifest on its own; a plan handed straight to the experiment would
    # otherwise record ``dataset_sha256: ""`` in the run files and aggregates, and
    # one run would describe its own dataset two ways (Phase 16's pin, which the
    # UI runner inherits because it builds a plan without the digest).
    if not plan.dataset_sha256:
        plan = replace(plan, dataset_sha256=dataset_sha256(plan.dataset))

    context = _StageContext()
    results: list[StageResult] = []
    for name in selected:
        started = clock()
        try:
            stage = _run_stage(
                name,
                context=context,
                completed=tuple(results),
                plan=plan,
                tasks=list(tasks),
                model=model,
                provider=provider,
                layout=layout,
                token_counter=token_counter,
                counter_name=counter_name,
                secret_tuple=secret_tuple,
                baseline_mode=baseline_mode,
                render_plots=render_plots,
                examples_per_type=examples_per_type,
                started=started,
                clock=clock,
                run_id=resolved_run_id,
                started_at=started_at,
                credential_source=credential_source,
                notes=dict(notes or {}),
                selected=selected,
            )
        except Exception as exc:  # noqa: BLE001 - one stage must not lose the run
            stage = StageResult(
                name=name,
                status=STATUS_FAILED,
                duration_seconds=clock() - started,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(stage)

    artifact = _artifact_payload(
        run_id=resolved_run_id,
        timestamp=started_at,
        layout=layout,
        plan=plan,
        tasks=list(tasks),
        model=model,
        context=context,
        stages=tuple(results),
        baseline_mode=baseline_mode,
        credential_source=credential_source,
        notes=dict(notes or {}),
        selected=selected,
    )
    # Two scans, because a file cannot contain the result of scanning itself.
    # Pass 1 covers every artifact stages 1–6 wrote and is computed before the
    # report renders, so the report can quote it. Pass 2 covers the report and
    # the manifest; a finding there fails the run and is reported to the caller,
    # but cannot be recorded inside the file that contains it — the regress is
    # documented rather than hidden.
    credential_check = artifact.get("credential_check") or _scan(
        _stage_paths(tuple(results)),
        secret_tuple,
        note=(
            "Scanned with security.scan (the Phase 13 artifact scanner) over "
            "exactly the files this pipeline's stages wrote. The report stage did "
            "not run, so this block was built after the fact."
        ),
    )
    # A reader of this file alone cannot see the second pass, because the second
    # pass scans this file: the result is reported to the caller and fails the
    # run, and the file says so rather than letting the absence read as "scanned
    # once, clean".
    credential_check["report_scan_note"] = (
        "A second pass scans this manifest and report/report.md after they are "
        "written. Its result cannot be stored inside a file it scans, so it is "
        "returned to the caller (`credential_check.report_scan`) and fails the "
        "run with exit code 4 if it finds anything."
    )
    artifact["credential_check"] = credential_check
    exit_code = _exit_code(context.experiment, tuple(results), credential_check)
    artifact["exit_code"] = exit_code

    manifest_path = write_json_report(artifact, layout.report / "pipeline.json")
    report_paths = [str(manifest_path)]
    report_stage = next((stage for stage in results if stage.name == STAGE_REPORT), None)
    if report_stage is not None:
        report_paths.extend(report_stage.artifacts)
    report_scan = scan_paths(report_paths, secrets=secret_tuple).to_dict()
    report_scan["note"] = (
        "Second pass, over the report and this manifest — the files written after "
        "the first pass ran."
    )
    artifact["credential_check"]["report_scan"] = report_scan
    if not report_scan.get("clean", True):
        exit_code = EXIT_STAGE_FAILED
    artifact["exit_code"] = exit_code
    artifact["artifacts"]["pipeline_manifest"] = str(manifest_path)

    quarantine = _quarantine(layout, artifact, secrets=secret_tuple)
    if quarantine["deleted"]:
        exit_code = EXIT_STAGE_FAILED
        artifact["exit_code"] = exit_code
    # If the report file itself was quarantined, the text handed back to the
    # caller (which the Evaluation tab renders straight into a browser) is the
    # quarantine note *instead of* the report — returning the deleted prose
    # would publish the credential the deletion just removed.
    report_text = (
        quarantine["report_note"]
        if quarantine["report_replaced"]
        else context.report_text + quarantine["report_note"]
    )

    return PipelineResult(
        run_id=resolved_run_id,
        timestamp=started_at,
        layout=layout,
        plan=plan,
        stages=tuple(results),
        experiment=context.experiment,
        artifact=artifact,
        exit_code=exit_code,
        report_markdown=report_text,
        manifest_path=str(manifest_path),
    )


# --------------------------------------------------------------------------- #
# Stage dispatch
# --------------------------------------------------------------------------- #


def _run_stage(name: str, **kwargs: Any) -> StageResult:
    handlers: dict[str, Callable[..., StageResult]] = {
        STAGE_EXPERIMENT: _stage_experiment,
        STAGE_RAW: _stage_raw,
        STAGE_COMPARISON: _stage_comparison,
        STAGE_STATISTICS: _stage_statistics,
        STAGE_ERRORS: _stage_errors,
        STAGE_PLOTS: _stage_plots,
        STAGE_REPORT: _stage_report,
    }
    handler = handlers.get(name)
    if handler is None:  # pragma: no cover - _ordered_stages rejects unknown names
        raise PipelineError(f"Unknown pipeline stage {name!r}.")
    return handler(**kwargs)


def _skipped(name: str, reason: str, started: float, clock: Callable[[], float]) -> StageResult:
    return StageResult(
        name=name,
        status=STATUS_SKIPPED,
        duration_seconds=clock() - started,
        detail=reason,
    )


def _missing_dependency(context: _StageContext, name: str) -> str:
    """Return ``""`` when a stage may run, or the reason it may not."""

    requires = STAGE_DEPENDENCIES.get(name)
    if requires is None:
        return ""
    if requires == STAGE_EXPERIMENT and context.experiment is None:
        return f"the `{requires}` stage produced no experiment"
    if requires == STAGE_RAW and not context.run_dicts:
        return f"the `{requires}` stage produced no run files"
    return ""


def _stage_experiment(
    *,
    context: _StageContext,
    plan: ExperimentPlan,
    tasks: list[BenchmarkTask],
    model: ModelSpec | None,
    provider: Any,
    layout: PipelineLayout,
    token_counter: TokenCounter,
    counter_name: str,
    secret_tuple: tuple[str, ...],
    baseline_mode: str,
    started: float,
    clock: Callable[[], float],
    **_: Any,
) -> StageResult:
    """Stage 1: the Phase 9 controlled comparison every other stage reads."""

    run = run_controlled_experiment(
        plan,
        tasks,
        model,
        provider,
        token_counter=token_counter,
        counter_name=counter_name,
        secrets=secret_tuple,
        baseline_mode=baseline_mode,
        clock=clock,
    )
    context.experiment = run
    destination = layout.aggregated / "experiment.json"
    write_json_report(run.to_dict(), destination)
    detail = (
        f"{run.tasks_executed}/{run.tasks_available} task(s) × "
        f"{len(plan.resolved_modes())} mode(s) × {plan.trials} trial(s); "
        f"{'dry run' if run.dry_run else 'generated'}; "
        f"{len(run.violations)} controlled-comparison violation(s)"
        + (f"; aborted at {run.aborted.get('limit')}" if run.aborted else "")
    )
    return StageResult(
        name=STAGE_EXPERIMENT,
        status=STATUS_OK,
        duration_seconds=clock() - started,
        artifacts=(str(destination),),
        digests={str(destination): _digest(destination)},
        detail=detail,
    )


def _stage_raw(
    *,
    context: _StageContext,
    plan: ExperimentPlan,
    layout: PipelineLayout,
    started: float,
    clock: Callable[[], float],
    **_: Any,
) -> StageResult:
    """Stage 2: one Phase 6/7 run file per (mode, trial), each with a manifest.

    The experiment CLI's ``--runs-dir`` already wrote files in this schema; what
    the pipeline adds is the Phase 16 ``repro`` block and the ``config`` version
    fields, so a raw file read on its own — ``python -m evaluation.compare
    results/raw/*.json`` — carries the same provenance a single-mode run does.
    """

    run = context.experiment
    reason = _missing_dependency(context, STAGE_RAW)
    if run is None or reason:
        return _skipped(STAGE_RAW, reason or "no experiment", started, clock)

    written: list[str] = []
    digests: dict[str, str] = {}
    multi_trial = plan.trials > 1
    for result in run.mode_results:
        payload = result.to_run_dict()
        suffix = f"-t{result.trial}" if multi_trial else ""
        destination = layout.raw / f"{result.mode}{suffix}.json"
        config = dict(payload.get("config") or {})
        config["output"] = str(destination)
        payload["config"] = config
        payload["repro"] = build_manifest(
            run_id=str(payload.get("run_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            task_ids=tuple(str(item.get("task_id", "")) for item in result.task_results),
            dataset_path=plan.dataset,
            config=config,
            aggregate_metrics=result.aggregate_metrics,
            extra={"notes": {"pipeline_stage": STAGE_RAW, "trial": result.trial}},
        )
        write_json_report(payload, destination)
        context.run_dicts.append(payload)
        written.append(str(destination))
        digests[str(destination)] = _digest(destination)
    return StageResult(
        name=STAGE_RAW,
        status=STATUS_OK,
        duration_seconds=clock() - started,
        artifacts=tuple(written),
        digests=digests,
        detail=(
            f"{len(written)} run file(s) in the Phase 6/7 schema, each with its own "
            "repro manifest"
        ),
    )


def _stage_comparison(
    *,
    context: _StageContext,
    layout: PipelineLayout,
    started: float,
    clock: Callable[[], float],
    **_: Any,
) -> StageResult:
    """Stage 3: the cross-mode headline table and the six plot series."""

    reason = _missing_dependency(context, STAGE_COMPARISON)
    if reason:
        return _skipped(STAGE_COMPARISON, reason, started, clock)
    context.comparison = comparison_report(list(context.run_dicts))
    destination = layout.aggregated / "comparison.json"
    write_json_report(context.comparison, destination)
    return StageResult(
        name=STAGE_COMPARISON,
        status=STATUS_OK,
        duration_seconds=clock() - started,
        artifacts=(str(destination),),
        digests={str(destination): _digest(destination)},
        detail=(
            f"{len(context.comparison.get('runs') or ())} mode row(s), "
            f"{len(context.comparison.get('series') or {})} plot series"
        ),
    )


def _stage_statistics(
    *,
    context: _StageContext,
    layout: PipelineLayout,
    baseline_mode: str,
    started: float,
    clock: Callable[[], float],
    **_: Any,
) -> StageResult:
    """Stage 4: trial statistics and paired comparisons (Phase 10)."""

    run = context.experiment
    reason = _missing_dependency(context, STAGE_STATISTICS)
    if run is None or reason:
        return _skipped(STAGE_STATISTICS, reason or "no experiment", started, clock)
    context.statistics = statistical_report(run, baseline_mode=baseline_mode)
    destination = layout.aggregated / "statistics.json"
    write_json_report(context.statistics, destination)
    detail = (
        f"{len(context.statistics.get('trial_summaries') or {})} mode(s), "
        f"{run.plan.trials} trial(s), "
        f"{len(context.statistics.get('paired_comparisons') or ())} paired "
        f"comparison(s) against `{baseline_mode}`"
    )
    if run.plan.trials <= 1:
        detail += "; one trial, so the confidence intervals are degenerate"
    return StageResult(
        name=STAGE_STATISTICS,
        status=STATUS_OK,
        duration_seconds=clock() - started,
        artifacts=(str(destination),),
        digests={str(destination): _digest(destination)},
        detail=detail,
    )


def _stage_errors(
    *,
    context: _StageContext,
    layout: PipelineLayout,
    tasks: list[BenchmarkTask],
    baseline_mode: str,
    examples_per_type: int,
    started: float,
    clock: Callable[[], float],
    **_: Any,
) -> StageResult:
    """Stage 5: the Phase 12 taxonomy plus the comparison it asked this phase for.

    Phase 12's carried constraint names Phase 17 directly: *"Phase 17 must run
    this report over generated answers and compare distributions, not
    accuracies."* The distribution comparison is computed from the report's own
    label counts, so nothing is re-graded and ``labels_vs_scorer.unexpected`` is
    untouched.
    """

    run = context.experiment
    reason = _missing_dependency(context, STAGE_ERRORS)
    if run is None or reason:
        return _skipped(STAGE_ERRORS, reason or "no experiment", started, clock)
    records = collect_failures(run, tasks=tasks)
    report = error_report(
        run,
        tasks=tasks,
        records=records,
        examples_per_type=int(examples_per_type),
        sources=[{"path": str(layout.aggregated / "experiment.json"), "kind": "experiment"}],
        dataset={
            "path": run.plan.dataset,
            "sha256": run.plan.dataset_sha256,
            "task_count": len(tasks),
        },
    )
    report["distribution_comparison"] = compare_error_distributions(
        report, baseline_mode=baseline_mode
    )
    context.errors = report
    destination = layout.aggregated / "errors.json"
    write_json_report(report, destination)
    records_path = layout.raw / "error-records.jsonl"
    write_failure_records(records, records_path)
    unexpected = int((report.get("labels_vs_scorer") or {}).get("unexpected", 0) or 0)
    return StageResult(
        name=STAGE_ERRORS,
        status=STATUS_OK,
        duration_seconds=clock() - started,
        artifacts=(str(destination), str(records_path)),
        digests={
            str(destination): _digest(destination),
            str(records_path): _digest(records_path),
        },
        detail=(
            f"{report.get('failure_record_count', 0)} failure record(s) "
            f"({report.get('observed_failure_count', 0)} observed, "
            f"{report.get('latent_defect_count', 0)} latent); "
            f"labels_vs_scorer.unexpected={unexpected}"
        ),
    )


def _stage_plots(
    *,
    context: _StageContext,
    layout: PipelineLayout,
    render_plots: bool,
    started: float,
    clock: Callable[[], float],
    **_: Any,
) -> StageResult:
    """Stage 6: the six planned figures.

    matplotlib is an optional extra, so a missing renderer is a *skipped* stage:
    the series it would have drawn are already on disk in
    ``aggregated/comparison.json``, and the numbers survive the missing picture.
    """

    run = context.experiment
    reason = _missing_dependency(context, STAGE_PLOTS)
    if run is None or reason:
        return _skipped(STAGE_PLOTS, reason or "no experiment", started, clock)
    if not render_plots:
        return _skipped(STAGE_PLOTS, "--no-plots: rendering was not requested", started, clock)
    try:
        written = plot_all(run, layout.plots)
    except RuntimeError as exc:
        return _skipped(
            STAGE_PLOTS,
            f"{exc} The plot series are in aggregated/comparison.json; install the "
            "`evaluation` extra (matplotlib) to render them.",
            started,
            clock,
        )
    context.plots = {name: str(path) for name, path in sorted(written.items())}
    return StageResult(
        name=STAGE_PLOTS,
        status=STATUS_OK,
        duration_seconds=clock() - started,
        artifacts=tuple(str(path) for path in written.values()),
        digests={str(path): _digest(path) for path in written.values()},
        detail=f"{len(written)} figure(s): {', '.join(sorted(written))}",
    )


def _stage_report(
    *,
    context: _StageContext,
    completed: Sequence[StageResult],
    layout: PipelineLayout,
    plan: ExperimentPlan,
    tasks: list[BenchmarkTask],
    model: ModelSpec | None,
    baseline_mode: str,
    credential_source: str,
    notes: Mapping[str, Any],
    secret_tuple: tuple[str, ...],
    run_id: str,
    started_at: str,
    started: float,
    clock: Callable[[], float],
    selected: Sequence[str] = STAGES,
    **_: Any,
) -> StageResult:
    """Stage 7: the human-readable report, rendered from the machine artifact.

    The artifact handed to the renderer holds stages 1–6. Stage 7's own row is
    added afterwards, when ``report/pipeline.json`` is written: a report that
    described its own writing would have to guess its own duration. The
    *selection* is passed separately for the same reason — a complete run must
    not read as a partial one just because the report's own row is not in yet.
    """

    run = context.experiment
    reason = _missing_dependency(context, STAGE_REPORT)
    if run is None or reason:
        return _skipped(STAGE_REPORT, reason or "no experiment", started, clock)

    context.credential_check = _scan(
        _stage_paths(tuple(completed)),
        secret_tuple,
        note=(
            "Scanned with security.scan (the Phase 13 artifact scanner) over "
            "exactly the files this pipeline's stages wrote. `report_scan` covers "
            "this report and the manifest, which are written after this block."
        ),
    )
    partial = _artifact_payload(
        run_id=run_id,
        timestamp=started_at,
        layout=layout,
        plan=plan,
        tasks=tasks,
        model=model,
        context=context,
        stages=tuple(completed),
        baseline_mode=baseline_mode,
        credential_source=credential_source,
        notes=dict(notes or {}),
        selected=selected,
    )
    text = markdown_report(partial)
    destination = layout.report / "report.md"
    write_markdown_report(text, destination)
    context.report_text = text
    return StageResult(
        name=STAGE_REPORT,
        status=STATUS_OK,
        duration_seconds=clock() - started,
        artifacts=(str(destination),),
        digests={str(destination): _digest(destination)},
        detail=f"{len(text)} characters rendered from the pipeline artifact",
    )


# --------------------------------------------------------------------------- #
# Artifact assembly
# --------------------------------------------------------------------------- #


def _artifact_payload(  # noqa: PLR0913 - one artifact, every input it records
    *,
    run_id: str,
    timestamp: str,
    layout: PipelineLayout,
    plan: ExperimentPlan,
    tasks: Sequence[BenchmarkTask],
    model: ModelSpec | None,
    context: _StageContext,
    stages: Sequence[StageResult],
    baseline_mode: str,
    credential_source: str,
    notes: Mapping[str, Any],
    selected: Sequence[str] = STAGES,
) -> dict[str, Any]:
    """Build the machine-readable pipeline artifact (``report/pipeline.json``)."""

    run = context.experiment
    modes = plan.resolved_modes()
    # The manifest's ``rerun_command`` re-derives one mode's raw run file, so it
    # names the system under test rather than the reference: that is the run a
    # reader would want back. ``pipeline_rerun_command`` re-derives everything.
    primary_mode = MODE_BRAINOS if MODE_BRAINOS in modes else (modes[0] if modes else "")
    config = {
        "provider": model.provider if model is not None else "",
        "model": model.model if model is not None else "",
        "mode": primary_mode,
        "benchmark": "context_rot",
        "temperature": model.temperature if model is not None else None,
        "context_budget": CONTEXT_BUDGET,
        "dataset": plan.dataset,
        "dataset_sha256": plan.dataset_sha256,
        "output": str(layout.raw / f"{primary_mode}.json") if primary_mode else "",
    }
    repro = build_manifest(
        run_id=run_id,
        timestamp=timestamp,
        task_ids=tuple(task.task_id for task in tasks),
        dataset_path=plan.dataset,
        config=config,
        aggregate_metrics={},
        extra={"notes": dict(notes)},
    )
    artifacts: dict[str, Any] = {"layout": layout.to_dict()}
    for stage in stages:
        if stage.artifacts:
            artifacts[stage.name] = list(stage.artifacts)
    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "run_id": run_id,
        "timestamp": timestamp,
        "preset": plan.name,
        "dry_run": bool(run.dry_run) if run is not None else True,
        "passed": bool(run.passed) if run is not None else False,
        "plan": plan.to_dict(),
        "layout": layout.to_dict(),
        "dataset": _dataset_block(plan, tasks),
        "model": model.to_dict() if model is not None else None,
        "generation": {
            "enabled": model is not None,
            "credential_source": credential_source
            or ("none (dry run)" if model is None else ""),
            "settings": _settings_block(run),
        },
        "billing_notice": BILLING_NOTICE,
        "baseline_mode": baseline_mode,
        "tasks_executed": run.tasks_executed if run is not None else 0,
        "tasks_available": run.tasks_available if run is not None else len(tasks),
        "headline": run.headline_rows() if run is not None else [],
        "budget": run.budget if run is not None else {},
        "cost_estimate": run.cost_estimate if run is not None else {},
        "constants": run.constants if run is not None else {},
        "violations": list(run.violations) if run is not None else [],
        "warnings": list(run.warnings) if run is not None else [],
        "dataset_issues": list(run.dataset_issues) if run is not None else [],
        "aborted": run.aborted if run is not None else None,
        "security": _security_block(run),
        "credential_check": dict(context.credential_check),
        "comparison": context.comparison,
        "statistics": context.statistics,
        "errors": context.errors,
        "plots": dict(context.plots),
        "stages": [stage.to_dict() for stage in stages],
        # ``stages`` holds what *completed* — when the report renders, its own
        # row is not in yet — so the selection is recorded separately and the
        # report reads that, rather than concluding a complete run was partial.
        "stages_selected": list(selected),
        "partial": list(selected) != list(STAGES),
        "artifacts": artifacts,
        "repro": repro,
        "pipeline_rerun_command": pipeline_rerun_command(
            plan,
            layout=layout,
            model=model,
            baseline_mode=baseline_mode,
            dry_run=model is None,
        ),
        "notes": dict(notes or {}),
    }
    if context.report_text:
        payload["report_markdown_preview"] = context.report_text[:REPORT_PREVIEW_CHARS]
    return payload


def _dataset_block(plan: ExperimentPlan, tasks: Sequence[BenchmarkTask]) -> dict[str, Any]:
    """What the dataset was, so a report can say what it measured."""

    return {
        "path": plan.dataset,
        "sha256": plan.dataset_sha256,
        "task_count": len(tasks),
        "categories": sorted({str(task.category) for task in tasks}),
        "length_tiers": sorted(
            {
                int(task.metadata.get("length_tier", 0) or 0)
                for task in tasks
                if "length_tier" in task.metadata
            }
        ),
        "seeds": sorted(
            {
                int(task.metadata.get("seed", 0) or 0)
                for task in tasks
                if "seed" in task.metadata
            }
        ),
    }


def _security_block(run: ExperimentRun | None) -> dict[str, Any]:
    """The guard audit behind these numbers (Phase 13's artifact block)."""

    if run is None:
        return summarize_security(())
    return summarize_security(
        record.get("security") for result in run.mode_results for record in result.task_results
    )


def _settings_block(run: ExperimentRun | None) -> dict[str, Any]:
    if run is None or run.model is None:
        return {}
    return run.model.generation_settings().to_dict()


def _scan(paths: Sequence[str], secrets: Sequence[str], *, note: str = "") -> dict[str, Any]:
    """Scan written artifacts for a credential (defence in depth).

    The generator redacts, the artifacts are assembled from credential-free
    objects, and Phase 13's scanner already answers this question for any
    directory. This asks it about *this run's own output*, so a leak introduced
    by a future stage fails the pipeline instead of shipping.
    """

    unique = _unique_paths(paths)
    if not unique:
        return {
            "scanned": 0,
            "files_scanned": 0,
            "finding_count": 0,
            "clean": True,
            "note": note or "No artifacts were written, so there was nothing to scan.",
        }
    payload = scan_paths(unique, secrets=tuple(secrets)).to_dict()
    payload["scanned"] = len(unique)
    payload["note"] = note
    return payload


def _quarantine(
    layout: PipelineLayout,
    artifact: dict[str, Any],
    *,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    """Delete this run's own artifacts that contain a credential, and say so.

    Detection is not remediation. A key that reached
    ``aggregated/errors.json`` — because a provider echoed it into an answer, and
    an answer is exactly what the failure examples quote — is still on the host's
    disk, still readable by the next process, and still in every backup. So the
    pipeline removes the files it just wrote.

    The blast radius is deliberately tiny: only paths the scanner named, only
    when they resolve inside this run's own output directory. Nothing else is
    touched, and what was removed is recorded by *name* in the manifest, in the
    report, and in a ``QUARANTINED.txt`` left where the files were.

    If the manifest itself was implicated it is rewritten without the findings'
    previews, so the run stays traceable; if the rewrite is still implicated it
    is deleted too, and the note file is all that remains.
    """

    check = artifact.get("credential_check")
    if not isinstance(check, dict):
        return {"deleted": [], "report_note": "", "report_replaced": False}
    scans = [check, check.get("report_scan") if isinstance(check.get("report_scan"), dict) else {}]
    findings = [
        finding
        for scan in scans
        if isinstance(scan, dict)
        for finding in scan.get("findings") or ()
        if isinstance(finding, dict)
    ]
    if not findings:
        return {"deleted": [], "report_note": "", "report_replaced": False}

    root = layout.root.resolve()
    named = {str(finding.get("path", "")) for finding in findings if finding.get("path")}
    deleted: list[str] = []
    for candidate in sorted(named):
        path = Path(candidate)
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - an unreadable path is not ours to delete
            continue
        if not resolved.is_relative_to(root) or not resolved.is_file():
            continue
        try:
            resolved.unlink()
        except OSError:  # pragma: no cover - deletion is best effort
            continue
        deleted.append(str(path))

    # The surviving record keeps categories and pointers, not previews: a
    # truncated credential is still a credential prefix.
    for finding in findings:
        finding.pop("preview", None)
    check["quarantined"] = deleted
    artifact["quarantined"] = True
    check["quarantine_note"] = (
        f"{len(deleted)} artifact file(s) contained a credential and were deleted "
        "from this run's output directory. The run's numbers cannot be trusted and "
        "its artifacts are incomplete by design."
    )

    if secrets:
        # The files are gone; the dict the caller still holds is not. A known
        # credential is scrubbed from it too, so an application that renders the
        # artifact (the Evaluation tab) cannot republish what was just deleted.
        scrubbed = sanitize_value(artifact, secrets=tuple(secrets))
        if isinstance(scrubbed, dict):
            artifact.clear()
            artifact.update(scrubbed)

    report_file = layout.report / "report.md"
    report_replaced = str(report_file) in {str(Path(path)) for path in deleted}

    manifest = layout.report / "pipeline.json"
    if str(manifest) in {str(Path(path)) for path in deleted} or not manifest.is_file():
        _rewrite_quarantined_manifest(manifest, artifact, secret_paths=deleted)

    note_path = layout.report / "QUARANTINED.txt"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "\n".join(
            [
                "This run's artifacts were quarantined.",
                "",
                check["quarantine_note"],
                "",
                "Deleted:",
                *(f"  - {Path(path).name}" for path in deleted),
                "",
                "Nothing in this file is a credential. Re-run with a provider that",
                "does not echo credentials into its answers, and treat any key that",
                "reached a model response as compromised: rotate it.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lines = [
        "",
        "## Quarantine",
        "",
        f"**{len(deleted)} artifact file(s) contained a credential and were deleted.** "
        "The scan that found them is the same Phase 13 scanner this repository runs "
        "over every artifact directory; a credential in a model's answer is a "
        "credential in the failure examples that quote it, so the files are removed "
        "rather than shipped.",
        "",
    ]
    lines.extend(f"- deleted: `{Path(path).name}`" for path in deleted)
    lines.extend(
        [
            "",
            "Treat any key that reached a model response as compromised and rotate "
            "it. This report's own numbers are incomplete: the artifacts they were "
            "computed from are partly gone.",
            "",
        ]
    )
    return {
        "deleted": deleted,
        "report_note": "\n".join(lines),
        "report_replaced": report_replaced,
    }


def _rewrite_quarantined_manifest(
    manifest: Path, artifact: Mapping[str, Any], *, secret_paths: Sequence[str]
) -> None:
    """Rewrite a manifest that was itself quarantined, without the report text.

    The report preview is the most likely carrier (it quotes failure examples),
    so the replacement keeps the provenance and the scan result and drops the
    prose. If even that is implicated, the file is removed and the caller's
    ``QUARANTINED.txt`` is the only record.
    """

    replacement = {
        key: value for key, value in artifact.items() if key != "report_markdown_preview"
    }
    replacement["quarantined_manifest"] = True
    replacement["quarantined_files"] = [str(path) for path in secret_paths]
    try:
        write_json_report(replacement, manifest)
    except OSError:  # pragma: no cover - an unwritable directory is already fatal
        return
    if not scan_paths([str(manifest)]).clean:
        manifest.unlink(missing_ok=True)


def _stage_paths(stages: Sequence[StageResult]) -> list[str]:
    """Every path the given stages wrote, de-duplicated and in stage order."""

    return _unique_paths([path for stage in stages for path in stage.artifacts])


def _unique_paths(paths: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for path in paths:
        text = str(path or "")
        if text and text not in unique:
            unique.append(text)
    return unique


def _exit_code(
    experiment: ExperimentRun | None,
    stages: Sequence[StageResult],
    credential_check: Mapping[str, Any],
) -> int:
    """Return the pipeline's exit code.

    Precedence follows the experiment CLI — an untrustworthy comparison outranks
    an incomplete pipeline — with one addition: a credential finding is the most
    severe outcome available, so it is treated as a failed pipeline.
    """

    if not bool(credential_check.get("clean", True)):
        return EXIT_STAGE_FAILED
    if experiment is not None and experiment.violations:
        return EXIT_VIOLATIONS
    if experiment is not None and experiment.aborted is not None:
        return EXIT_ABORTED
    if any(stage.status == STATUS_FAILED for stage in stages):
        return EXIT_STAGE_FAILED
    return EXIT_OK


def _digest(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return ""


def _ordered_stages(stages: Sequence[str]) -> tuple[str, ...]:
    """Validate a stage selection and return it in execution order."""

    requested = {str(stage).strip() for stage in stages if str(stage).strip()}
    unknown = sorted(requested - set(STAGES))
    if unknown:
        raise PipelineError(
            f"Unknown pipeline stage(s): {', '.join(unknown)}. "
            f"Expected {', '.join(STAGES)}."
        )
    if not requested:
        raise PipelineError("At least one pipeline stage is required.")
    return tuple(stage for stage in STAGES if stage in requested)


def expand_stages(stages: Sequence[str]) -> tuple[str, ...]:
    """Return ``stages`` plus every dependency, in execution order.

    Selecting ``report`` alone would render a report about nothing; the honest
    reading of "I want the report" is "run what the report is made of".
    """

    expanded = set(_ordered_stages(stages))
    for name in tuple(expanded):
        dependency = STAGE_DEPENDENCIES.get(name)
        while dependency:
            expanded.add(dependency)
            dependency = STAGE_DEPENDENCIES.get(dependency)
    return tuple(stage for stage in STAGES if stage in expanded)


#: Which CLI flag reproduces which budget field. ``max_tasks`` is ``--limit``
#: because that is the flag a reader reaches for; the rest keep their names.
_LIMIT_FLAGS: tuple[tuple[str, str], ...] = (
    ("max_tasks", "--limit"),
    ("max_requests", "--max-requests"),
    ("max_input_tokens", "--max-input-tokens"),
    ("max_output_tokens", "--max-output-tokens"),
    ("max_total_tokens", "--max-total-tokens"),
    ("max_generation_failures", "--max-failures"),
    ("timeout_seconds", "--timeout"),
)


def _limit_flags(limits: RunLimits, reference: RunLimits | None = None) -> list[str]:
    """The flags that reproduce ``limits``, minus any ``reference`` already sets."""

    flags: list[str] = []
    for name, flag in _LIMIT_FLAGS:
        value = getattr(limits, name)
        if value is None:
            continue
        if reference is not None and getattr(reference, name) == value:
            continue
        rendered = value if isinstance(value, int) else repr(float(value))
        flags.append(f"{flag} {rendered}")
    return flags


def pipeline_rerun_command(
    plan: ExperimentPlan,
    *,
    layout: PipelineLayout | None = None,
    model: ModelSpec | None = None,
    baseline_mode: str = "full_context",
    dry_run: bool = True,
    dataset: str | Path | None = None,
) -> str:
    """Render the command that re-derives this pipeline's artifacts.

    Credential-free by construction: a generated run names the environment
    variable its key came from, never a value. Only the flags that differ from
    the preset's own defaults are emitted, so the command stays readable.

    A plan whose name is *not* a preset — one built by a test, by an embedding
    application, or by the Evaluation tab, whose policy may have tightened a
    ceiling — gets its budget spelled out flag by flag instead. Recording how to
    re-run a finished pipeline must never be the reason it fails.
    """

    destination = layout or pipeline_layout(DEFAULT_OUTPUT_DIR)
    try:
        reference: RunPreset | None = preset(plan.name)
    except ValueError:
        reference = None
    modes = plan.resolved_modes()
    lines = ["python -m evaluation.pipeline"]
    if reference is None:
        if modes:
            lines.append(f"--modes {','.join(modes)}")
        lines.append(f"--trials {plan.trials}")
        lines.extend(_limit_flags(plan.limits))
    else:
        lines.append(f"--preset {plan.name}")
        if modes and modes != tuple(reference.modes):
            lines.append(f"--modes {','.join(modes)}")
        if plan.trials != reference.trials:
            lines.append(f"--trials {plan.trials}")
        lines.extend(_limit_flags(plan.limits, reference.limits))
    dataset_path = str(dataset if dataset is not None else plan.dataset)
    if dataset_path and dataset_path != str(DEFAULT_DATASET):
        lines.append(f"--dataset {dataset_path}")
    if plan.session_isolation:
        lines.append("--session-isolation")
    if dry_run:
        lines.append("--dry-run")
    elif model is not None:
        lines.append(f"--model {model.model}")
        lines.append(f"--provider {model.provider}")
        lines.append(f"--api-key-env {model.api_key_env}")
    lines.append(f"--baseline-mode {baseline_mode}")
    lines.append(f"--output-dir {destination.root}")
    return " \\\n  ".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    # ``allow_abbrev=False`` for the reason the other CLIs give: ``--api-key``
    # must not be accepted as an abbreviation of ``--api-key-env``, or a pasted
    # credential would be recorded as a variable name and written to the artifact.
    parser = argparse.ArgumentParser(
        description=(
            "Run the plan's automated evaluation pipeline: controlled comparison, "
            "raw run files, aggregates, statistics, error analysis, plots, and a "
            "report, into results/{raw,aggregated,plots,report}."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument(
        "--modes",
        help=(
            "Comma-separated mode selectors, overriding the preset's selection. "
            "'all' runs the five baselines; 'ablations' runs BrainOS plus the four "
            "Phase 11 ablations."
        ),
    )
    parser.add_argument("--trials", type=int, help="Repeated trials per (task, mode).")
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N tasks (0 = the preset's own max_tasks).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root for the plan's raw/ aggregated/ plots/ report/ directories.",
    )
    parser.add_argument(
        "--stages",
        help=(
            "Comma-separated stage subset "
            f"({', '.join(STAGES)}). Dependencies are added automatically: "
            "selecting `report` still runs the stages a report is made of."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure rendering (the plot series are still written as JSON).",
    )
    parser.add_argument(
        "--examples-per-type",
        type=int,
        default=1,
        help="Failure examples per label in the error report.",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="", help="Provider model identifier.")
    parser.add_argument("--model-role", default="primary", help="Matrix role label.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint.")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Name of the environment variable holding the credential.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout.")
    parser.add_argument(
        "--session-isolation",
        action="store_true",
        help="Start a fresh session at every transcript session boundary.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-failures", type=int, default=None)
    parser.add_argument(
        "--token-counter",
        choices=("estimate", "tiktoken"),
        default="estimate",
        help="How prompts are counted for the cost ceilings.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="full_context",
        help="Baseline for the paired comparisons and the failure-distribution distance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build every prompt and report cost without calling a provider.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary.")
    return parser


def _parse_modes(value: str | None) -> tuple[str, ...] | None:
    """Parse ``--modes`` exactly as the experiment CLI does."""

    if value is None:
        return None
    keyword = value.strip().lower()
    if keyword == "all":
        return MODE_ORDER
    if keyword == "ablations":
        return (MODE_BRAINOS, *ABLATION_ORDER)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _limit_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.limit:
        overrides["max_tasks"] = args.limit
    if args.max_input_tokens is not None:
        overrides["max_input_tokens"] = args.max_input_tokens
    if args.max_output_tokens is not None:
        overrides["max_output_tokens"] = args.max_output_tokens
    if args.max_requests is not None:
        overrides["max_requests"] = args.max_requests
    if args.max_total_tokens is not None:
        overrides["max_total_tokens"] = args.max_total_tokens
    if args.max_failures is not None:
        overrides["max_generation_failures"] = args.max_failures
    if args.timeout is not None:
        overrides["timeout_seconds"] = args.timeout
    return overrides


def _resolve_counter(name: str, model: str) -> tuple[TokenCounter, str]:
    if name == "tiktoken":
        return tiktoken_counter(model), f"tiktoken:{model}"
    return estimate_tokens, "estimate_tokens"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials is not None and args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if not args.dry_run and not args.model.strip():
        raise SystemExit(
            "--model is required unless --dry-run is used. A dry run measures "
            "prompts and retrieval with no credential and no spend."
        )

    tasks = load_jsonl(args.dataset)
    digest = dataset_sha256(args.dataset)
    try:
        stages = expand_stages(
            [part for part in str(args.stages or "").split(",") if part.strip()] or list(STAGES)
        )
        plan = ExperimentPlan.from_preset(
            args.preset,
            modes=_parse_modes(args.modes),
            trials=args.trials,
            dataset=args.dataset,
            dataset_sha256=digest,
            notes={"cli": "evaluation.pipeline", "dry_run": bool(args.dry_run)},
            **_limit_overrides(args),
        )
        plan.resolved_modes()
        resolve_mode(args.baseline_mode)
    except (ValueError, PipelineError) as exc:
        raise SystemExit(str(exc)) from None
    plan = replace(plan, session_isolation=bool(args.session_isolation))

    try:
        counter, counter_name = _resolve_counter(args.token_counter, args.model)
    except TokenizerUnavailableError as exc:
        raise SystemExit(str(exc)) from None

    model: ModelSpec | None = None
    provider: Any | None = None
    api_key = ""
    credential_source = "none (dry run)"
    if not args.dry_run:
        try:
            candidate = ModelSpec(
                name=args.model_role,
                provider=args.provider,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                seed=args.seed,
                timeout_seconds=float(
                    args.timeout if args.timeout is not None else plan.limits.timeout_seconds
                ),
                base_url=args.base_url,
                api_key_env=args.api_key_env,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        model = candidate.with_limits(plan.limits)
        try:
            api_key = api_key_from_environment(model)
            provider = build_provider(model, api_key=api_key)
        except MissingCredentialError as exc:
            raise SystemExit(str(exc)) from None
        credential_source = f"environment:{args.api_key_env}"

    try:
        result = run_pipeline(
            plan,
            tasks,
            model,
            provider,
            output_dir=args.output_dir,
            token_counter=counter,
            counter_name=counter_name,
            secrets=(api_key,) if api_key else (),
            baseline_mode=args.baseline_mode,
            render_plots=not args.no_plots,
            examples_per_type=args.examples_per_type,
            stages=stages,
            credential_source=credential_source,
        )
    except (PipelineError, ExperimentError) as exc:
        raise SystemExit(str(exc)) from None

    if not args.quiet:
        _print_summary(result)
    return result.exit_code


def _print_summary(result: PipelineResult) -> None:
    """Print the stage table, the layout, the headline numbers, and every caveat."""

    artifact = result.artifact
    print(BILLING_NOTICE)
    if result.dry_run:
        print("Dry run: no provider configured, so no generation happened and no key was used.")
    print(
        f"pipeline={PIPELINE_VERSION} preset={artifact.get('preset')} "
        f"run={result.run_id} "
        f"tasks={artifact.get('tasks_executed')}/{artifact.get('tasks_available')} "
        f"modes={len(result.plan.resolved_modes())} trials={result.plan.trials}"
    )
    print()
    _print_table(
        ("stage", "status", "seconds", "detail"),
        [
            (stage.name, stage.status, f"{stage.duration_seconds:.2f}", stage.detail or stage.error)
            for stage in result.stages
        ],
    )
    print()
    layout = result.layout
    for name in RESULT_DIRECTORIES:
        print(f"{name:<11}{layout.directory(name)}")
    budget = artifact.get("budget") or {}
    print(
        f"cost: requests={budget.get('requests', 0)} "
        f"total_tokens={budget.get('total_tokens', 0)} "
        f"failures={budget.get('failures', 0)} skips={budget.get('skips', 0)} "
        f"within_limits={budget.get('within_limits', True)}"
    )
    headline = list(artifact.get("headline") or ())
    for row in headline:
        graded = int(row.get("graded", 0) or 0)
        print(
            f"  {str(row.get('mode', '')):<16} tasks={row.get('tasks', 0)} graded={graded} "
            f"accuracy={'—' if not graded else _fmt(row.get('accuracy'))} "
            f"recall={_fmt(row.get('recall'))} "
            f"evidence={_fmt(row.get('evidence_in_prompt'))} "
            f"tokens={_fmt(row.get('mean_context_tokens'), digits=1)} "
            f"reduction={_fmt(row.get('mean_reduction'))}"
        )
    if headline and not any(int(row.get("graded", 0) or 0) for row in headline):
        print("No answers were graded: accuracy, faithfulness, and QAE are unset, not zero.")
    for violation in result.violations:
        print(f"violation: {violation}")
    if result.aborted:
        print(f"aborted: {result.aborted.get('reason', '')}")
    for warning in artifact.get("warnings") or ():
        print(f"warning: {warning}")
    check = artifact.get("credential_check") or {}
    report_scan = check.get("report_scan") or {}
    files = int(check.get("files_scanned", 0) or 0) + int(
        report_scan.get("files_scanned", 0) or 0
    )
    findings = int(check.get("finding_count", 0) or 0) + int(
        report_scan.get("finding_count", 0) or 0
    )
    print(f"credential scan: files={files} findings={findings} clean={findings == 0}")
    for block in (check, report_scan):
        for finding in list(block.get("findings") or [])[:5]:
            print(f"  {finding.get('path')} {finding.get('pointer')} {finding.get('category')}")
    quarantined = list(check.get("quarantined") or ())
    if quarantined:
        # The names are printed, not the values: the point of the quarantine is
        # that nothing credential-shaped survives the run, including its summary.
        print(f"QUARANTINED: {len(quarantined)} artifact file(s) deleted:")
        for path in quarantined:
            print(f"  - {Path(str(path)).name}")
        print("  rotate any key that reached a model response; see report/QUARANTINED.txt")
    print(f"report: {layout.report / 'report.md'}")
    print(f"exit_code={result.exit_code}")


def _print_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    widths = [len(str(column)) for column in columns]
    for row in rows:
        for index, value in enumerate(row[: len(columns)]):
            widths[index] = max(widths[index], len(str(value)))
    print("  ".join(str(column).ljust(widths[index]) for index, column in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        cells = [str(value) for value in row]
        cells.extend([""] * (len(columns) - len(cells)))
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)))


def _fmt(value: Any, *, digits: int = 3) -> str:
    """Render a missing metric as an em dash; ``None`` is never zero."""

    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


__all__ = [
    "DEFAULT_DATASET",
    "default_dataset_path",
    "DEFAULT_OUTPUT_DIR",
    "EXIT_ABORTED",
    "EXIT_OK",
    "EXIT_STAGE_FAILED",
    "EXIT_VIOLATIONS",
    "PIPELINE_VERSION",
    "REPORT_PREVIEW_CHARS",
    "RESULT_DIRECTORIES",
    "STAGES",
    "STAGE_COMPARISON",
    "STAGE_DEPENDENCIES",
    "STAGE_ERRORS",
    "STAGE_EXPERIMENT",
    "STAGE_PLOTS",
    "STAGE_RAW",
    "STAGE_REPORT",
    "STAGE_STATISTICS",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "PipelineError",
    "PipelineLayout",
    "PipelineResult",
    "StageResult",
    "expand_stages",
    "pipeline_layout",
    "pipeline_rerun_command",
    "run_pipeline",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
