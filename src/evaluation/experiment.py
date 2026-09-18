"""Controlled experiments across the baseline modes (plan Phase 9).

Phase 7 measured what each mode *put in a prompt*. Phase 8 scored those prompts
with scripted answers. Phase 9 closes the loop: the same tasks, the same model,
the same sampling parameters, and the same scoring procedure are run through
every context-management strategy, and only the strategy changes.

```text
                        same tasks, same model, same parameters
                                        |
        +---------------+-------+-------+-------+---------------+
        |               |               |               |       |
   Mode A          Mode B          Mode C          Mode D   Mode E
 full context   sliding window   lexical RAG    BrainOS    BrainOS+RAG
        |               |               |               |       |
        +---------------+-------+-------+-------+---------------+
                                |
                        one prompt per (task, mode)
                                |
                        provider generation (budgeted)
                                |
                     Phase 7/8 scoring, per mode
```

The plan's rule for this phase is "keep constant: model, temperature, generation
parameters, benchmark examples, task wording, evaluation procedure. Only change
the context-management strategy." That rule is not left to discipline here —
:func:`run_controlled_experiment` checks it after the run and records the result:

* every request carries the same :meth:`GenerationSettings.fingerprint`;
* the system instructions (the first system message) are byte-identical across
  modes;
* the current request (the final user message) is identical across modes for a
  task;
* every mode ran the same task ids, with the same dataset hash and the same
  token counter.

A violation is reported in ``constants.violations`` rather than thrown away,
and the CLI exits non-zero on one: a comparison whose controls failed is worse
than no comparison, because it looks like a result.

Two ordering decisions matter for validity:

1. **Task-outer, mode-inner.** All modes run one task before any mode starts the
   next, so provider drift (rate limits, latency, a model silently updating
   mid-run) lands on every mode rather than on whichever mode happened to run
   last. It also makes the records paired by task, which is what Phase 10's
   paired comparisons need.
2. **Fresh session per (task, mode, trial).** A mode never benefits from a
   runtime warmed by another mode, and each trial starts from nothing.

Cost controls are enforced by the shared :class:`~evaluation.limits.RunBudget`,
including its per-request prompt ceiling: a prompt that does not fit is
*skipped* and recorded, because at the plan's longer lengths "Mode A cannot fit
inside the input budget" is a finding, not a crash.

The CLI (``python -m evaluation.experiment``) is the plan's controlled
comparison entry point. It reads the credential from an environment variable,
never from an argument, and writes one artifact plus optional per-mode run files
that :mod:`evaluation.compare` can consume unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.service import DEFAULT_SYSTEM_INSTRUCTIONS
from baselines.modes import (
    ABLATION_ORDER,
    MODE_BRAINOS,
    MODE_ORDER,
    mode_label,
    resolve_mode,
)
from brain.tokenizers import (
    TokenCounter,
    TokenizerUnavailableError,
    estimate_tokens,
    tiktoken_counter,
)
from reproducibility.run_provenance import (
    BENCHMARK_VERSION as REPRO_BENCHMARK_VERSION,
)
from reproducibility.run_provenance import app_version, brainos_version
from security.findings import summarize_security

from .datasets import BenchmarkTask, dataset_issues, load_jsonl
from .generation import (
    GenerationSettings,
    Generator,
    MissingCredentialError,
    ModelSpec,
    api_key_from_environment,
    budgeted_generator,
    build_provider,
)
from .limits import (
    PRESETS,
    BudgetExceeded,
    RunBudget,
    RunLimits,
    RunPreset,
    preset,
)
from .modes import replay_task
from .reports import write_json_report
from .scoring import aggregate_scores, score_record

#: Identifies the experiment semantics recorded on every artifact.
EXPERIMENT_VERSION = "experiment-v1"

#: Application version recorded on a run, matching :mod:`evaluation.runner`.
APPLICATION_VERSION = "0.1.0"  # noqa: F841 - legacy export field, kept for schema stability

#: Benchmark revision recorded on a run, matching the Phase 7 generator
#: (``benchmarks/context_rot``, which pins ``context-rot-v1`` in its spec).
BENCHMARK_VERSION = "context-rot-v1"

#: The context budget the controlled runs are built with (plan Phase 16 lists
#: it as provenance because every token figure depends on it).
CONTEXT_BUDGET = 4096

DEFAULT_DATASET = Path("benchmarks/context_rot/dataset.jsonl")


class ExperimentError(RuntimeError):
    """The experiment cannot be run as configured."""


def runtime_version() -> str:
    """Return the installed BrainOS runtime version, or ``"unvalidated"``.

    Recorded on every run for the plan's reproducibility block. The import is
    optional on purpose: a controlled experiment with a dry run or a fake
    provider must still work on a machine without the runtime installed.
    """

    try:
        import brainos_runtime  # noqa: PLC0415 - optional dependency, checked at runtime
    except Exception:
        return "unvalidated"
    version = getattr(brainos_runtime, "__version__", "")
    return str(version or "unknown")


def dataset_sha256(path: Path | str) -> str:
    """Return the content hash of a dataset file (or ``""`` when absent)."""

    source = Path(path)
    if not source.exists():
        return ""
    return hashlib.sha256(source.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ExperimentPlan:
    """What will be run, under which limits, against which dataset."""

    name: str = "controlled"
    modes: tuple[str, ...] = MODE_ORDER
    trials: int = 1
    session_isolation: bool = False
    limits: RunLimits = field(default_factory=RunLimits)
    dataset: str = str(DEFAULT_DATASET)
    dataset_sha256: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trials, int) or isinstance(self.trials, bool) or self.trials < 1:
            raise ValueError("trials must be at least 1.")
        if not self.modes:
            raise ValueError("At least one mode is required.")

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        modes: Sequence[str] | None = None,
        trials: int | None = None,
        dataset: str | Path = DEFAULT_DATASET,
        dataset_sha256: str = "",
        notes: dict[str, Any] | None = None,
        **limit_overrides: Any,
    ) -> ExperimentPlan:
        """Build a plan from a named preset (plan Phase 15's Quick/Standard/Research)."""

        selected: RunPreset = preset(name)
        if limit_overrides:
            selected = selected.with_overrides(**limit_overrides)
        return cls(
            name=selected.name,
            modes=tuple(modes) if modes else selected.modes,
            trials=selected.trials if trials is None else int(trials),
            limits=selected.limits,
            dataset=str(dataset),
            dataset_sha256=dataset_sha256,
            notes=dict(notes or {}),
        )

    def with_modes(self, modes: Sequence[str]) -> ExperimentPlan:
        return replace(self, modes=tuple(modes))

    def with_dataset(self, dataset: str | Path, dataset_sha256: str = "") -> ExperimentPlan:
        return replace(self, dataset=str(dataset), dataset_sha256=dataset_sha256)

    def resolved_modes(self) -> tuple[str, ...]:
        """Return the modes in plan order, normalized and de-duplicated."""

        resolved: list[str] = []
        for mode in self.modes:
            normalized = resolve_mode(mode)
            if normalized not in resolved:
                resolved.append(normalized)
        return tuple(resolved)

    def planned_requests(self, task_count: int) -> int:
        """Return how many provider requests this plan implies."""

        return int(task_count) * len(self.resolved_modes()) * self.trials

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modes": list(self.resolved_modes()),
            "mode_labels": [mode_label(mode) for mode in self.resolved_modes()],
            "trials": self.trials,
            "session_isolation": self.session_isolation,
            "limits": self.limits.to_dict(),
            "dataset": self.dataset,
            "dataset_sha256": self.dataset_sha256,
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class ModeResult:
    """One mode's results for one trial, in both aggregate and run-file form."""

    mode: str
    label: str
    trial: int
    task_results: tuple[dict[str, Any], ...]
    scores: tuple[dict[str, Any], ...]
    aggregate_metrics: dict[str, Any]
    generation_usage: dict[str, Any]
    config: dict[str, Any]
    run_id: str
    timestamp: str
    complete: bool = True

    @property
    def skipped_task_ids(self) -> tuple[str, ...]:
        """Tasks whose prompt exceeded the run's input ceiling (never sent)."""

        return tuple(
            str(record.get("task_id", ""))
            for record in self.task_results
            if (record.get("generation") or {}).get("skipped")
        )

    def security_summary(self) -> dict[str, Any]:
        """Phase 13: the guard audit for this mode's trial.

        Summed from the per-task blocks the replays already carry, so a mode
        whose prompts were assembled with quarantines or redactions says so next
        to its metrics instead of only inside its task records.
        """

        return summarize_security(record.get("security") for record in self.task_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "trial": self.trial,
            "complete": self.complete,
            "aggregate_metrics": self.aggregate_metrics,
            "generation_usage": self.generation_usage,
            "security": self.security_summary(),
            "skipped_task_ids": list(self.skipped_task_ids),
            "task_results": [dict(record) for record in self.task_results],
            "scores": [dict(score) for score in self.scores],
        }

    def to_run_dict(self) -> dict[str, Any]:
        """Return this mode's trial in the Phase 6/7 run-file schema.

        The shape matches :meth:`evaluation.runner.EvaluationRun.to_dict`, so
        ``python -m evaluation.compare`` accepts an experiment's per-mode files
        without a translation step.
        """

        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": dict(self.config),
            "task_results": [dict(record) for record in self.task_results],
            "aggregate_metrics": self.aggregate_metrics,
            "security": self.security_summary(),
            "dataset_issues": list(self.config.get("dataset_issues") or []),
        }


@dataclass(frozen=True)
class ExperimentRun:
    """One controlled comparison: every mode, the same tasks, one model."""

    run_id: str
    timestamp: str
    plan: ExperimentPlan
    model: ModelSpec | None
    mode_results: tuple[ModeResult, ...]
    constants: dict[str, Any]
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    dataset_issues: tuple[str, ...]
    budget: dict[str, Any]
    cost_estimate: dict[str, Any]
    tasks_executed: int
    tasks_available: int
    dry_run: bool
    aborted: dict[str, Any] | None = None
    #: Baseline for the paired comparisons in ``statistical_summary``. Ablation
    #: runs compare against the full system (``brainos``); the five-mode
    #: comparison keeps the full-context reference.
    baseline_mode: str = "full_context"

    @property
    def passed(self) -> bool:
        """Whether the controls held and the run finished within its budget."""

        return not self.violations and self.aborted is None

    def mode_result(self, mode: str, trial: int = 0) -> ModeResult | None:
        """Return one (mode, trial) result, or ``None`` when it did not run."""

        for result in self.mode_results:
            if result.mode == mode and result.trial == trial:
                return result
        return None

    def headline_rows(self) -> list[dict[str, Any]]:
        """Return one compact row per (mode, trial) for tables and plots."""

        rows: list[dict[str, Any]] = []
        for result in self.mode_results:
            aggregate = result.aggregate_metrics
            rows.append(
                {
                    "mode": result.mode,
                    "mode_label": result.label,
                    "trial": result.trial,
                    "tasks": aggregate.get("task_count", 0),
                    "graded": aggregate.get("graded_answer_count", 0),
                    "accuracy": aggregate.get("answer_accuracy"),
                    "faithfulness": aggregate.get("faithfulness"),
                    "recall": aggregate.get("retrieval_recall"),
                    "evidence_in_prompt": aggregate.get("evidence_in_prompt_rate"),
                    "conflict_resolution": aggregate.get("conflict_resolution_accuracy"),
                    "abstention": aggregate.get("abstention_accuracy"),
                    "mean_context_tokens": aggregate.get("mean_final_context_tokens"),
                    "mean_reduction": aggregate.get("mean_context_reduction"),
                    "mean_latency_ms": aggregate.get("mean_latency_ms"),
                    "failures": result.generation_usage.get("failures", 0),
                    "skips": result.generation_usage.get("skips", 0),
                }
            )
        return rows

    def statistical_summary(
        self, *, baseline_mode: str | None = None
    ) -> dict[str, Any]:
        """Compute the Phase 10 statistical analysis across trials and paired tasks."""

        from .analysis import statistical_analysis

        return statistical_analysis(
            self, baseline_mode=baseline_mode or self.baseline_mode
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "repro": {
                "repro_version": "repro-v1",
                "application_version": app_version(),
                "benchmark_version": REPRO_BENCHMARK_VERSION,
                "brainos_version": brainos_version(),
                "experiment_version": EXPERIMENT_VERSION,
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "dry_run": self.dry_run,
                "passed": self.passed,
                "trials": self.plan.trials,
                "session_isolation": self.plan.session_isolation,
                "modes": list(self.plan.resolved_modes()),
                "task_count": self.tasks_executed,
                "dataset": self.plan.dataset,
                "dataset_sha256": self.plan.dataset_sha256,
                "limits": self.plan.limits.to_dict(),
                "model": (
                    {
                        "provider": self.model.provider,
                        "model": self.model.model,
                        "temperature": self.model.temperature,
                    }
                    if self.model is not None
                    else None
                ),
                "violations": list(self.violations),
            },
            "dry_run": self.dry_run,
            "passed": self.passed,
            "plan": self.plan.to_dict(),
            "model": self.model.to_dict() if self.model is not None else None,
            "provenance": {
                "application_version": app_version(),
                "brainos_version": brainos_version(),
                "benchmark_version": BENCHMARK_VERSION,
                "dataset_sha256": self.plan.dataset_sha256,
                "context_budget": CONTEXT_BUDGET,
                "tasks_executed": self.tasks_executed,
                "tasks_available": self.tasks_available,
                "provider": self.model.provider if self.model is not None else "",
                "model": self.model.model if self.model is not None else "",
                "temperature": self.model.temperature if self.model is not None else None,
            },
            "constants": self.constants,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "dataset_issues": list(self.dataset_issues),
            "budget": self.budget,
            "cost_estimate": self.cost_estimate,
            "aborted": self.aborted,
            "modes": [result.to_dict() for result in self.mode_results],
            "security": summarize_security(
                record.get("security")
                for result in self.mode_results
                for record in result.task_results
            ),
            "headline": self.headline_rows(),
            "baseline_mode": self.baseline_mode,
            "statistical_summary": self.statistical_summary(),
        }


def run_controlled_experiment(
    plan: ExperimentPlan,
    tasks: Sequence[BenchmarkTask],
    model: ModelSpec | None = None,
    provider: Any | None = None,
    *,
    token_counter: TokenCounter = estimate_tokens,
    counter_name: str = "",
    clock: Callable[[], float] = time.perf_counter,
    secrets: tuple[str, ...] = (),
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    baseline_mode: str = "full_context",
) -> ExperimentRun:
    """Run every mode of ``plan`` over ``tasks`` with one model and one budget.

    ``model`` and ``provider`` are both optional and must be supplied together:
    without them the experiment is a **dry run** that measures prompts, tokens,
    and retrieval with no generation and no credential. A dry run is the plan's
    "show estimated usage where possible": it costs nothing and reports what a
    real run would ask the provider to do.

    Never raises for a provider failure — a failed request is a record with a
    redacted error. It *does* stop at a run-level cost limit, returning the
    partial run with ``aborted`` set rather than spending past the user's
    configured ceiling.
    """

    modes = plan.resolved_modes()
    if (model is None) != (provider is None):
        raise ExperimentError("Supply both a model spec and a provider, or neither.")

    issues = dataset_issues(list(tasks))
    available = len(tasks)
    selected = list(tasks)
    warnings: list[str] = []
    if plan.limits.max_tasks is not None and len(selected) > plan.limits.max_tasks:
        selected = selected[: plan.limits.max_tasks]
        warnings.append(
            f"Dataset truncated to {len(selected)} of {available} tasks by the run's "
            "max_tasks limit."
        )
    if not selected:
        raise ExperimentError("At least one benchmark task is required.")
    if not plan.dataset_sha256:
        warnings.append(
            "Dataset hash not recorded: this run cannot be tied to the exact text it ran on."
        )

    settings: GenerationSettings | None = (
        model.with_limits(plan.limits).generation_settings() if model is not None else None
    )
    if settings is not None and settings.temperature > 0 and plan.trials == 1:
        warnings.append(
            "temperature > 0 with a single trial: stochastic sampling needs repeated trials "
            "before any number from this run is reportable (plan Phase 10)."
        )

    budget = RunBudget(
        limits=plan.limits,
        counter=token_counter,
        counter_name=counter_name or getattr(token_counter, "__name__", "custom_counter"),
    )
    generator: Generator | None = None
    if provider is not None and settings is not None:
        generator = budgeted_generator(
            provider,
            settings,
            budget,
            counter=token_counter,
            clock=clock,
            secrets=secrets,
        )
    planned_requests = plan.planned_requests(len(selected))
    collected: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {
        (mode, trial): {"records": [], "scores": []}
        for trial in range(plan.trials)
        for mode in modes
    }
    aborted: dict[str, Any] | None = None

    try:
        # Task-outer, mode-inner: provider drift lands on every mode equally and
        # the records come out paired by task.
        for trial in range(plan.trials):
            for task in selected:
                for mode in modes:
                    replay = replay_task(
                        task,
                        mode,
                        session_isolation=plan.session_isolation,
                        generate=generator,
                    )
                    record = replay.to_dict()
                    score = score_record(
                        task,
                        record,
                        answer=replay.answer,
                        session_isolation=plan.session_isolation,
                    )
                    collected[(mode, trial)]["records"].append(record)
                    collected[(mode, trial)]["scores"].append(score)
    except BudgetExceeded as exc:
        aborted = {
            "limit": exc.limit,
            "allowed": exc.allowed,
            "used": exc.used,
            "reason": str(exc),
            "completed_requests": budget.requests,
        }
        warnings.append(f"Run stopped at a cost limit: {exc}")

    timestamp = datetime.now(timezone.utc).isoformat()
    mode_results = tuple(
        _build_mode_result(
            plan=plan,
            model=model,
            mode=mode,
            trial=trial,
            records=collected[(mode, trial)]["records"],
            scores=collected[(mode, trial)]["scores"],
            issues=issues,
            timestamp=timestamp,
            complete=len(collected[(mode, trial)]["records"]) == len(selected),
        )
        for trial in range(plan.trials)
        for mode in modes
        if collected[(mode, trial)]["records"]
    )
    constants, violations = check_constants(
        plan, model, selected, collected, system_instructions=system_instructions
    )
    skipped = sum(result.generation_usage.get("skips", 0) for result in mode_results)
    if skipped:
        warnings.append(
            f"{skipped} request(s) were skipped because the prompt exceeded the run's "
            "max_input_tokens ceiling; those tasks are ungraded, not wrong."
        )

    return ExperimentRun(
        run_id=str(uuid4()),
        timestamp=timestamp,
        plan=plan,
        model=None if model is None else model.with_limits(plan.limits),
        mode_results=mode_results,
        constants=constants,
        violations=tuple(violations),
        warnings=tuple(warnings),
        dataset_issues=tuple(issues),
        budget=budget.snapshot(),
        cost_estimate=budget.estimate_ceiling(planned_requests),
        tasks_executed=len(selected),
        tasks_available=available,
        dry_run=model is None,
        aborted=aborted,
        baseline_mode=baseline_mode,
    )


def check_constants(
    plan: ExperimentPlan,
    model: ModelSpec | None,
    tasks: Sequence[BenchmarkTask],
    collected: dict[tuple[str, int], dict[str, list[dict[str, Any]]]],
    *,
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
) -> tuple[dict[str, Any], list[str]]:
    """Verify the plan's controlled-comparison rule and describe the controls.

    Returns ``(constants, violations)``. The constants block is provenance: it
    records what was held constant, so a reader does not have to trust the
    procedure description. A violation is a statement that the comparison is
    not controlled, which is a result about the run rather than about BrainOS.
    """

    task_ids = [task.task_id for task in tasks]
    expected_modes = list(plan.resolved_modes())
    violations: list[str] = []

    actual_modes: set[str] = set()
    questions_by_task: dict[str, set[str]] = {task_id: set() for task_id in task_ids}
    task_ids_by_mode: dict[str, set[str]] = {mode: set() for mode in expected_modes}
    system_prompts: set[str] = set()
    fingerprints: set[str] = set()
    reported_models: set[str] = set()
    counters: set[str] = set()
    generated = 0

    for (mode, _trial), bucket in collected.items():
        for record in bucket["records"]:
            actual_modes.add(str(record.get("mode", "")))
            task_id = str(record.get("task_id", ""))
            questions_by_task.setdefault(task_id, set()).add(
                _final_user_message(record.get("prompt_messages") or ())
            )
            task_ids_by_mode.setdefault(mode, set()).add(task_id)
            first_system = _first_system_message(record.get("prompt_messages") or ())
            if first_system is not None:
                system_prompts.add(first_system)
            generation = record.get("generation") or {}
            if generation:
                if generation.get("skipped"):
                    continue
                if generation.get("settings_fingerprint"):
                    fingerprints.add(str(generation["settings_fingerprint"]))
                # ``model`` is what the provider said it served, which a gateway
                # can silently route away from ``requested_model``.
                if generation.get("model"):
                    reported_models.add(str(generation["model"]))
                generated += 1
            stats = record.get("stats") or {}
            if stats.get("token_counter"):
                counters.add(str(stats["token_counter"]))

    if set(expected_modes) != actual_modes:
        violations.append(
            f"Modes executed {sorted(actual_modes)} differ from the plan's "
            f"{sorted(expected_modes)}."
        )
    for task_id, questions in questions_by_task.items():
        if len(questions) > 1:
            violations.append(
                f"Task wording differs between modes for {task_id}: {sorted(questions)}."
            )
    for mode in expected_modes:
        recorded = task_ids_by_mode.get(mode, set())
        if recorded and recorded != set(task_ids):
            missing = sorted(set(task_ids) - recorded)
            violations.append(f"Mode {mode} is missing tasks: {missing}.")
    if len(system_prompts) > 1:
        violations.append(
            "The system instructions differ between modes "
            f"({len(system_prompts)} distinct prompts); only retrieval may change."
        )
    if system_prompts and str(system_instructions or "").strip():
        expected_prompt = str(system_instructions).strip()
        # The context builder may truncate the system prompt to its budget and
        # marks the cut with an ellipsis, so a stored prompt is either the
        # configured text or a prefix of it. Anything else means a mode sent
        # different instructions, which would make the comparison meaningless.
        mismatched = [
            prompt
            for prompt in system_prompts
            if not expected_prompt.startswith(prompt.rstrip(" …"))
        ]
        if mismatched:
            violations.append(
                "The system instructions stored in the prompt are not the configured "
                "application prompt."
            )
    if model is not None and generated and len(fingerprints) > 1:
        violations.append(
            "Generation settings fingerprints differ between requests "
            f"({sorted(fingerprints)}); the comparison is not controlled."
        )
    if model is not None and generated and reported_models and reported_models != {model.model}:
        violations.append(
            f"Responses reported model(s) {sorted(reported_models)} but the run requested "
            f"{model.model!r}."
        )
    if len(counters) > 1:
        violations.append(
            f"Token counters differ across records ({sorted(counters)}); token figures "
            "are not comparable."
        )

    constants = {
        "checked": [
            "modes",
            "task_set",
            "task_wording",
            "system_instructions",
            "generation_parameters",
            "requested_model",
            "token_counter",
        ],
        "passed": not violations,
        "modes": expected_modes,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "dataset_sha256": plan.dataset_sha256,
        "trials": plan.trials,
        "session_isolation": plan.session_isolation,
        "system_prompt_characters": sorted(len(prompt) for prompt in system_prompts),
        "system_prompt_sha256": sorted(
            _sha256(prompt)[:16] for prompt in system_prompts
        ),
        "generation_settings": settings_dict(model),
        "generation_fingerprints": sorted(fingerprints),
        "reported_models": sorted(reported_models),
        "token_counters": sorted(counters),
        "generated_requests": generated,
        "violations": violations,
    }
    return constants, violations


def settings_dict(model: ModelSpec | None) -> dict[str, Any]:
    """Return the constant generation parameters, or an empty block for a dry run."""

    if model is None:
        return {}
    return model.generation_settings().to_dict()


def _build_mode_result(
    *,
    plan: ExperimentPlan,
    model: ModelSpec | None,
    mode: str,
    trial: int,
    records: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    issues: list[str],
    timestamp: str,
    complete: bool,
) -> ModeResult:
    """Assemble one (mode, trial) result, including the per-mode cost report."""

    config: dict[str, Any] = {
        "provider": model.provider if model is not None else "",
        "model": model.model if model is not None else "",
        "mode": mode,
        "benchmark": "context_rot",
        "benchmark_version": BENCHMARK_VERSION,
        "application_version": app_version(),
        "brainos_version": brainos_version(),
        "context_budget": CONTEXT_BUDGET,
        "temperature": model.temperature if model is not None else None,
        "session_isolation": plan.session_isolation,
        "dataset": plan.dataset,
        "dataset_sha256": plan.dataset_sha256,
        "dataset_issues": list(issues),
        "trial": trial,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_run": plan.name,
        "limits": plan.limits.to_dict(),
        "dry_run": model is None,
    }
    return ModeResult(
        mode=mode,
        label=mode_label(mode),
        trial=trial,
        task_results=tuple(record for record in records),
        scores=tuple(score for score in scores),
        aggregate_metrics=aggregate_scores(scores) if scores else {},
        generation_usage=generation_usage(records),
        config=config,
        run_id=str(uuid4()),
        timestamp=timestamp,
        complete=complete,
    )


def generation_usage(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize generation cost and latency for one mode's records.

    Tokens are summed from what the provider reported. A record whose provider
    reported nothing still contributes its counted prompt tokens, so a missing
    usage block cannot make a mode look free.
    """

    requests = 0
    failures = 0
    skips = 0
    generated = 0
    latencies: list[float] = []
    input_tokens = 0
    output_tokens = 0
    for record in records:
        generation = record.get("generation") or {}
        if not generation:
            continue
        requests += 1
        if generation.get("skipped"):
            skips += 1
            continue
        if not generation.get("generated"):
            failures += 1
        else:
            generated += 1
        usage = generation.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        if prompt is None:
            prompt = generation.get("prompt_token_count")
        completion = usage.get("completion_tokens")
        input_tokens += int(prompt or 0)
        output_tokens += int(completion or 0)
        latency = generation.get("latency_ms")
        if latency is not None:
            latencies.append(float(latency))
    return {
        "requests": requests,
        "generated": generated,
        "failures": failures,
        "skips": skips,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "mean_latency_ms": (
            round(sum(latencies) / len(latencies), 3) if latencies else None
        ),
        "max_latency_ms": (round(max(latencies), 3) if latencies else None),
    }


def compare_modes_across_models(runs: Sequence[ExperimentRun]) -> dict[str, Any]:
    """Build the plan's cross-model table ("does it work across models?").

    Rows are ``(model, mode)`` pairs with the headline metrics, so a report can
    show whether a mode's advantage survives a model change instead of reporting
    one model's numbers as general.
    """

    rows: list[dict[str, Any]] = []
    for run in runs:
        model_name = run.model.name if run.model is not None else "dry-run"
        model_id = run.model.model if run.model is not None else ""
        for result in run.mode_results:
            aggregate = result.aggregate_metrics
            rows.append(
                {
                    "matrix_role": model_name,
                    "model": model_id,
                    "provider": run.model.provider if run.model is not None else "",
                    "mode": result.mode,
                    "mode_label": result.label,
                    "trial": result.trial,
                    "tasks": aggregate.get("task_count", 0),
                    "graded": aggregate.get("graded_answer_count", 0),
                    "accuracy": aggregate.get("answer_accuracy"),
                    "faithfulness": aggregate.get("faithfulness"),
                    "recall": aggregate.get("retrieval_recall"),
                    "mean_context_tokens": aggregate.get("mean_final_context_tokens"),
                    "mean_reduction": aggregate.get("mean_context_reduction"),
                    "mean_latency_ms": aggregate.get("mean_latency_ms"),
                }
            )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "models": [
            {
                "matrix_role": run.model.name if run.model is not None else "dry-run",
                "model": run.model.model if run.model is not None else "",
                "provider": run.model.provider if run.model is not None else "",
                "dry_run": run.dry_run,
                "passed": run.passed,
            }
            for run in runs
        ],
        "rows": rows,
        "note": (
            "Cross-model rows are descriptive. Statistical comparison across models "
            "belongs to the trial-level analysis phase."
        ),
    }


def _first_system_message(messages: Sequence[dict[str, Any]]) -> str | None:
    """Return the first system message's content, or ``None``.

    The context builder emits the system instructions first and the retrieved
    evidence blocks after it, so the *first* system message is the one the
    controlled comparison must hold constant. Comparing later system messages
    would flag exactly the difference the experiment exists to measure.
    """

    for message in messages:
        if str(message.get("role", "")).lower() == "system":
            return str(message.get("content", ""))
    return None


def _final_user_message(messages: Sequence[dict[str, Any]]) -> str:
    """Return the last user message's content — the task's wording."""

    for message in reversed(list(messages)):
        if str(message.get("role", "")).lower() == "user":
            return str(message.get("content", ""))
    return ""


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

#: The plan requires the user to be told whose account pays for a run.
BILLING_NOTICE = (
    "Requests are sent with your provider credential: your provider account is "
    "responsible for all usage and cost."
)


def build_parser() -> argparse.ArgumentParser:
    # ``allow_abbrev=False`` is a security setting here, not a style choice:
    # without it ``--api-key`` would be accepted as an abbreviation of
    # ``--api-key-env`` and a pasted credential would end up in the artifact.
    parser = argparse.ArgumentParser(
        description="Run the plan's controlled comparison across context-management modes.",
        allow_abbrev=False,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument(
        "--modes",
        help=(
            "Comma-separated mode selectors, overriding the preset's selection. "
            "'all' runs the five baselines; 'ablations' runs BrainOS plus the "
            "four Phase 11 ablations."
        ),
    )
    parser.add_argument("--trials", type=int, help="Repeated trials per (task, mode).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N tasks (0 = the preset's own max_tasks).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Also write one Phase 6/7 run file per (mode, trial) for evaluation.compare.",
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
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Skip a request whose prompt exceeds this many tokens.",
    )
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
        "--dry-run",
        action="store_true",
        help="Build every prompt and report cost without calling a provider.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory to save the six evaluation plots.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="full_context",
        help=(
            "Baseline for the paired comparisons in the statistical summary "
            "(use 'brainos' for Phase 11 ablation runs)."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the summary table.")
    return parser


def _parse_modes(value: str | None) -> tuple[str, ...] | None:
    """Parse ``--modes``.

    Unset means the preset's selection, ``all`` means the five baselines, and
    ``ablations`` means the full system plus the four Phase 11 ablations (an
    ablation without its full reference is not interpretable, so the keyword
    always includes ``brainos``).
    """

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


def _resolve_counter(args: argparse.Namespace, model: str) -> tuple[TokenCounter, str]:
    if args.token_counter == "tiktoken":
        counter = tiktoken_counter(model)
        return counter, f"tiktoken:{model}"
    return estimate_tokens, "estimate_tokens"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials is not None and args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if not args.dry_run and not args.model.strip():
        raise SystemExit("--model is required unless --dry-run is used.")

    tasks = load_jsonl(args.dataset)
    digest = dataset_sha256(args.dataset)
    try:
        plan = ExperimentPlan.from_preset(
            args.preset,
            modes=_parse_modes(args.modes),
            trials=args.trials,
            dataset=args.dataset,
            dataset_sha256=digest,
            notes={"cli": True, "dry_run": bool(args.dry_run)},
            **_limit_overrides(args),
        )
        plan.resolved_modes()
        resolve_mode(args.baseline_mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    plan = replace(plan, session_isolation=bool(args.session_isolation))

    try:
        counter, counter_name = _resolve_counter(args, args.model)
    except TokenizerUnavailableError as exc:
        raise SystemExit(str(exc)) from None

    model = None
    provider = None
    api_key = ""
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
                timeout_seconds=min(
                    float(
                        args.timeout
                        if args.timeout is not None
                        else plan.limits.timeout_seconds
                    ),
                    float(plan.limits.timeout_seconds),
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

    try:
        run = run_controlled_experiment(
            plan,
            tasks,
            model,
            provider,
            token_counter=counter,
            counter_name=counter_name,
            secrets=(api_key,) if api_key else (),
            baseline_mode=args.baseline_mode,
        )
    except ExperimentError as exc:
        raise SystemExit(str(exc)) from None

    write_json_report(run.to_dict(), args.output)
    if args.runs_dir:
        for result in run.mode_results:
            suffix = f"-t{result.trial}" if plan.trials > 1 else ""
            write_json_report(
                result.to_run_dict(), Path(args.runs_dir) / f"{result.mode}{suffix}.json"
            )

    if args.plots_dir:
        from .plots import plot_all

        plot_all(run, args.plots_dir)

    if not args.quiet:
        _print_summary(run, output=args.output, runs_dir=args.runs_dir)
    return _exit_code(run)


def _exit_code(run: ExperimentRun) -> int:
    if run.violations:
        return 2
    if run.aborted is not None:
        return 3
    return 0


def _print_summary(run: ExperimentRun, *, output: Path, runs_dir: Path | None) -> None:
    """Print the run's table, cost report, and every control message."""

    print(f"{BILLING_NOTICE}")
    if run.dry_run:
        print("Dry run: no provider configured, so no generation happened and no key was used.")
    print(
        f"experiment={run.plan.name} dry_run={run.dry_run} tasks={run.tasks_executed}"
        f"/{run.tasks_available} modes={len(run.plan.resolved_modes())} "
        f"trials={run.plan.trials} requests={run.budget.get('requests', 0)}"
    )
    columns = (
        "mode",
        "tasks",
        "graded",
        "accuracy",
        "faithfulness",
        "recall",
        "evidence",
        "mean_tokens",
        "mean_ms",
    )
    rows: list[dict[str, str]] = []
    ungraded = True
    for result in run.mode_results:
        aggregate = result.aggregate_metrics
        graded = int(aggregate.get("graded_answer_count", 0) or 0)
        ungraded = ungraded and graded == 0
        # An ungraded run has no accuracy; printing 0.000 would read as "every
        # answer was wrong" when in fact no answer was ever graded.
        answer_metrics = (
            {"accuracy": _fmt(aggregate.get("answer_accuracy")),
             "faithfulness": _fmt(aggregate.get("faithfulness"))}
            if graded
            else {"accuracy": "—", "faithfulness": "—"}
        )
        rows.append(
            {
                "mode": result.mode,
                "tasks": str(aggregate.get("task_count", 0)),
                "graded": str(graded),
                **answer_metrics,
                "recall": _fmt(aggregate.get("retrieval_recall")),
                "evidence": _fmt(aggregate.get("evidence_in_prompt_rate")),
                "mean_tokens": _fmt(aggregate.get("mean_final_context_tokens"), digits=1),
                "mean_ms": _fmt(aggregate.get("mean_latency_ms"), digits=1),
            }
        )
    _print_table(columns, rows)
    if ungraded and rows:
        print("No answers were graded: accuracy and faithfulness are unset, not zero.")
    if run.plan.trials > 1:
        from .compare import _print_statistical_summary

        _print_statistical_summary(run.statistical_summary())
    budget = run.budget
    print(
        f"cost: requests={budget.get('requests', 0)} input={budget.get('input_tokens', 0)} "
        f"output={budget.get('output_tokens', 0)} total={budget.get('total_tokens', 0)} "
        f"failures={budget.get('failures', 0)} skips={budget.get('skips', 0)} "
        f"within_limits={budget.get('within_limits')}"
    )
    print(f"artifact: {output}")
    if runs_dir is not None:
        print(f"per-mode run files: {runs_dir}")
    for name in ("warnings", "violations", "dataset_issues"):
        messages = getattr(run, name)
        for message in messages:
            print(f"{name[:-1]}: {message}", file=sys.stderr)
    if run.aborted:
        print(f"aborted: {run.aborted['reason']}", file=sys.stderr)
    if run.violations:
        print(
            "The controlled-comparison checks failed: do not report these numbers as "
            "a mode comparison.",
            file=sys.stderr,
        )


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _print_table(columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        return
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


__all__ = [
    "APPLICATION_VERSION",
    "BENCHMARK_VERSION",
    "BILLING_NOTICE",
    "CONTEXT_BUDGET",
    "DEFAULT_DATASET",
    "EXPERIMENT_VERSION",
    "ExperimentError",
    "ExperimentPlan",
    "ExperimentRun",
    "ModeResult",
    "check_constants",
    "compare_modes_across_models",
    "dataset_sha256",
    "generation_usage",
    "run_controlled_experiment",
    "runtime_version",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
