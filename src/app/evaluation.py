"""The Evaluation tab's pipeline runner (plan Phase 17).

Phase 4 shipped the Evaluation tab as a placeholder and every phase since left
it a note: Phase 7 said grading answers needed cost controls before the tab
could expose a run, Phase 15 built those controls and asked the tab to *"let the
visitor pick a preset and see the ``estimate_ceiling`` before running"*, and
Phase 16 asked it to surface the reproducibility manifest and to persist through
:func:`reproducibility.run_provenance.persist_run`. This module is the tab's
half of Phase 17: the same automated pipeline the CLI runs, driven from a
browser session.

```text
Evaluation tab
   |  preset / modes / limit / dataset / generate?
   v
UIEvaluationRunner
   |  policy check -> cost preview (before anything runs)
   |  session-scoped output dir -> results/ui/<session>/<run>/
   v
evaluation.pipeline.run_pipeline      (the CLI's own entry point)
   |
   +-> raw/ aggregated/ plots/ report/
   +-> persist_run(...) through the session's EvaluationStore
   +-> EvaluationRunView (status, tables, figures, manifest, re-run command)
```

Four properties this module exists to hold, all of them consequences of the tab
being reachable by an anonymous visitor on a public Space:

1. **Nothing is written where the visitor points.** The output directory is
   derived from the session id under ``results/ui/`` and re-checked after
   resolution; a dataset must live under ``benchmarks/``. A text box is not a
   path oracle (:func:`UIEvaluationRunner.output_dir`,
   :meth:`UIEvaluationRunner.resolve_dataset`).
2. **A dry run is the default and needs no key.** Retrieval, prompt cost, the
   taxonomy, the statistics, the figures, and the report are all measurable
   without a model. Generation is an explicit opt-in that reuses the *session's*
   bring-your-own-key credential — never an operator key, never the environment.
3. **The visitor sees the ceiling before the spend.** :meth:`UIEvaluationRunner.preview`
   answers "what would this cost" from the preset's own limits, which is what
   makes the run button an informed choice rather than a surprise.
4. **No credential reaches the browser or the disk.** The key is handed to the
   pipeline's redaction list, the recorded credential source is the *name* of the
   session route rather than a value, and the controller redacts every field of
   the returned view against the active key as defence in depth.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.retention import (
    DEFAULT_RETENTION_SECONDS,
    MAX_SESSIONS_PER_SWEEP,
    RESULTS_ROOT,
    RETENTION_VERSION,
    RUNS_SUBDIR,
    RetentionPolicy,
    RetentionReport,
    sweep_results,
)
from baselines.modes import ABLATION_ORDER, MODE_BRAINOS, MODE_ORDER, mode_label, resolve_mode
from brain.tokenizers import (
    TokenCounter,
    TokenizerUnavailableError,
    estimate_tokens,
    tiktoken_counter,
)
from checkout import REPO_ROOT, repo_anchored
from evaluation.datasets import load_jsonl
from evaluation.experiment import ExperimentPlan, dataset_sha256
from evaluation.generation import ModelSpec, build_provider
from evaluation.limits import PRESETS, RunBudget, RunLimits, preset
from evaluation.pipeline import (
    PIPELINE_VERSION,
    PipelineError,
    PipelineResult,
    run_pipeline,
)
from providers import create_provider
from providers.base import ProviderConfig
from reproducibility.run_provenance import persist_run

#: Where a browser-initiated run writes, under the plan's ``results/`` root.
#: Session-scoped so two visitors cannot overwrite each other's artifacts.
#: The constants are the *declared* layout; :class:`UIEvaluationRunner` resolves
#: them at construction so a process started outside the checkout still writes
#: here instead of creating a stray ``results/`` (Phase 19, see
#: :mod:`checkout`). They are the same objects :mod:`app.retention` sweeps by, so
#: the writer and the sweeper cannot disagree about where runs live.
UI_RESULTS_ROOT = RESULTS_ROOT
UI_RESULTS_SUBDIR = RUNS_SUBDIR

#: Datasets the tab may load. Anything outside this root is refused: the widget
#: is a text box, and a text box that resolves arbitrary paths is a file oracle.
DATASET_ROOT = Path("benchmarks")
DEFAULT_DATASET = Path("benchmarks/context_rot/dataset.jsonl")

#: The *name* recorded as a generated run's credential source. It is a label, not
#: a variable this code reads: the UI takes the key from the session, and writing
#: it into the environment so a name would be honest would put a secret where
#: every subprocess could read it.
SESSION_CREDENTIAL_LABEL = "BRAINOS_LAB_SESSION_KEY"

#: Session identifiers become path segments, so they are constrained to the
#: characters ``uuid4`` produces (plus ``.``, ``_``, ``-`` for operator-set ids).
_SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")

#: How many generated datasets the dataset dropdown offers before it stops
#: looking. A directory with thousands of tiers is a listing problem, not a UI.
MAX_DATASET_CHOICES = 24


class EvaluationRequestError(ValueError):
    """A run the tab asked for and the runner will not perform.

    The message is written for a visitor: it names the rule that was broken and
    what to do instead. It never contains a path outside the allowed roots, a
    credential, or a stack trace.
    """


@dataclass(frozen=True)
class EvaluationPolicy:
    """What a deployment allows the Evaluation tab to do.

    The defaults are the plan's own presets — a bring-your-own-key visitor spends
    their own money, so the preset ceilings are the honest limit. A public Space
    operator who would rather not host 7 500-request runs tightens this object
    once, at construction, instead of editing the tab::

        UIEvaluationRunner(policy=EvaluationPolicy(
            allowed_presets=("quick",), max_tasks=20, max_requests=60,
        ))

    ``apply`` only ever **tightens** a plan: a policy cannot raise a ceiling the
    preset set, because then the preset's recorded limits would not be the limits
    the run obeyed.
    """

    #: Presets a visitor may select.
    allowed_presets: tuple[str, ...] = tuple(sorted(PRESETS))
    #: Whether the tab may send prompts to the visitor's model at all.
    allow_generation: bool = True
    #: Hard task ceiling, applied on top of the preset's ``max_tasks``.
    max_tasks: int = 500
    #: Hard request ceiling, applied on top of the preset's ``max_requests``.
    max_requests: int = 7_500
    #: Runs one session may start before it must be ended and restarted.
    max_runs_per_session: int = 5
    #: Phase 19: age at which an *abandoned* session's artifacts are swept.
    #: ``End session`` still deletes a visitor's own files immediately; this
    #: covers the session nobody ends. ``None`` keeps artifacts until then.
    results_retention_seconds: float | None = DEFAULT_RETENTION_SECONDS
    #: Sessions one sweep may remove, so a page load cannot become a purge.
    max_sessions_per_sweep: int = MAX_SESSIONS_PER_SWEEP

    def __post_init__(self) -> None:
        unknown = sorted(set(self.allowed_presets) - set(PRESETS))
        if unknown:
            raise ValueError(
                f"Unknown preset(s) in policy: {', '.join(unknown)}. "
                f"Expected {', '.join(sorted(PRESETS))}."
            )
        if not self.allowed_presets:
            raise ValueError("A policy must allow at least one preset.")
        for name in ("max_tasks", "max_requests", "max_runs_per_session"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.retention_policy()  # validates max_age_seconds / max_sessions_per_sweep

    def retention_policy(self) -> RetentionPolicy:
        """The chat-path-style retention object this policy describes."""

        return RetentionPolicy(
            max_age_seconds=self.results_retention_seconds,
            max_sessions_per_sweep=self.max_sessions_per_sweep,
        )

    def retention_notice(self) -> str:
        """One sentence a visitor can act on, for the Evaluation tab."""

        age = self.results_retention_seconds
        if age is None:
            return (
                "Benchmark artifacts stay on this host until you press **End "
                "session**. This deployment has no automatic sweep."
            )
        days = age / 86_400
        human = f"{days:g} day" + ("" if days == 1 else "s")
        return (
            f"Benchmark artifacts under `results/ui/` are kept for {human} after "
            "the last file was written, then swept automatically — including the "
            "artifacts of a session nobody ended. **End session** removes yours "
            "immediately."
        )

    def check(self, *, preset_name: str, generate: bool, runs_so_far: int = 0) -> None:
        """Refuse a request the deployment does not allow, before any work."""

        key = str(preset_name or "").strip().lower()
        if key not in self.allowed_presets:
            raise EvaluationRequestError(
                f"This deployment does not allow the `{preset_name}` preset. "
                f"Allowed: {', '.join(self.allowed_presets)}."
            )
        if generate and not self.allow_generation:
            raise EvaluationRequestError(
                "This deployment runs benchmarks in retrieval-only mode: no prompt "
                "is sent to a model. Uncheck generation and re-run to measure "
                "prompts, tokens, and retrieval."
            )
        if runs_so_far >= self.max_runs_per_session:
            raise EvaluationRequestError(
                f"This session has already run {runs_so_far} benchmark(s), the "
                f"per-session ceiling. Press **End session** to start a fresh one."
            )

    def apply(self, plan: ExperimentPlan) -> ExperimentPlan:
        """Return ``plan`` with the policy's ceilings applied (tightening only)."""

        limits = plan.limits
        overrides: dict[str, int] = {}
        if limits.max_tasks is None or limits.max_tasks > self.max_tasks:
            overrides["max_tasks"] = self.max_tasks
        if limits.max_requests is None or limits.max_requests > self.max_requests:
            overrides["max_requests"] = self.max_requests
        if not overrides:
            return plan
        return replace(plan, limits=replace(limits, **overrides))


@dataclass(frozen=True)
class PresetView:
    """One preset as the tab shows it, including what it would cost."""

    name: str
    label: str
    description: str
    modes: tuple[str, ...]
    mode_labels: tuple[str, ...]
    trials: int
    limits: dict[str, Any]
    planned_requests: int | None
    allowed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "modes": list(self.modes),
            "mode_labels": list(self.mode_labels),
            "trials": self.trials,
            "limits": dict(self.limits),
            "planned_requests": self.planned_requests,
            "allowed": self.allowed,
        }

@dataclass(frozen=True)
class CostPreview:
    """What a run *would* cost, computed before it is allowed to cost anything.

    The plan asks the application to "show estimated usage where possible". The
    honest answer for input tokens is that they are unknown until the prompts are
    built, so the preview reports the output ceiling and the hard caps rather
    than guessing low.
    """

    preset: str
    label: str
    description: str
    modes: tuple[str, ...]
    mode_labels: tuple[str, ...]
    trials: int
    tasks_available: int
    tasks_selected: int
    planned_requests: int
    limits: dict[str, Any]
    estimate: dict[str, Any]
    generate: bool
    credential_source: str
    dataset: str
    dataset_sha256: str
    output_dir: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "label": self.label,
            "description": self.description,
            "modes": list(self.modes),
            "mode_labels": list(self.mode_labels),
            "trials": self.trials,
            "tasks_available": self.tasks_available,
            "tasks_selected": self.tasks_selected,
            "planned_requests": self.planned_requests,
            "limits": dict(self.limits),
            "estimate": dict(self.estimate),
            "generate": self.generate,
            "credential_source": self.credential_source,
            "dataset": self.dataset,
            "dataset_sha256": self.dataset_sha256,
            "output_dir": self.output_dir,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EvaluationRunView:
    """What a run produced, in the pipeline's own vocabulary.

    This object carries *data*, not rendered cells: the controller hands
    ``headline`` and ``stages`` to :mod:`app.panels`, which is the only place
    that turns application values into browser-safe text. Everything here is
    already credential-free (the pipeline scans its own artifacts before
    returning), and the controller redacts it once more against the session key.
    """

    ok: bool
    status: str
    run_id: str = ""
    exit_code: int = 0
    dry_run: bool = True
    output_dir: str = ""
    #: The artifact's headline rows, one per (mode, trial).
    headline: tuple[dict[str, Any], ...] = ()
    #: ``StageResult.to_dict()`` for every stage that was selected.
    stages: tuple[dict[str, Any], ...] = ()
    plots: tuple[str, ...] = ()
    report_markdown: str = ""
    repro: dict[str, Any] = field(default_factory=dict)
    rerun_command: str = ""
    pipeline_rerun_command: str = ""
    artifact_path: str = ""
    report_path: str = ""
    persisted: bool = False
    warnings: tuple[str, ...] = ()
    #: What the run *actually* spent, in :class:`CostPreview`'s vocabulary plus a
    #: ``used`` block — one renderer (:func:`app.panels.evaluation_cost_markdown`)
    #: serves both the preview and the receipt.
    cost: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "run_id": self.run_id,
            "exit_code": self.exit_code,
            "dry_run": self.dry_run,
            "output_dir": self.output_dir,
            "headline": [dict(row) for row in self.headline],
            "stages": [dict(stage) for stage in self.stages],
            "plots": list(self.plots),
            "repro": dict(self.repro),
            "rerun_command": self.rerun_command,
            "pipeline_rerun_command": self.pipeline_rerun_command,
            "artifact_path": self.artifact_path,
            "report_path": self.report_path,
            "persisted": self.persisted,
            "warnings": list(self.warnings),
            "cost": dict(self.cost),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class RunRecord:
    """One entry in a session's run history (never a credential, never a prompt).

    Rendered by :func:`app.panels.evaluation_history_rows`; this object is the
    data, so there is one place that decides what a history row says.
    """

    run_id: str
    session_id: str
    timestamp: str
    preset: str
    modes: tuple[str, ...]
    trials: int
    dry_run: bool
    exit_code: int
    tasks: int
    requests: int
    total_tokens: int
    persisted: bool
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "preset": self.preset,
            "modes": list(self.modes),
            "trials": self.trials,
            "dry_run": self.dry_run,
            "exit_code": self.exit_code,
            "tasks": self.tasks,
            "requests": self.requests,
            "total_tokens": self.total_tokens,
            "persisted": self.persisted,
            "output_dir": self.output_dir,
        }

class UIEvaluationRunner:
    """Run the automated pipeline on behalf of one browser session."""

    def __init__(
        self,
        *,
        results_root: str | Path | None = None,
        dataset_root: str | Path | None = None,
        policy: EvaluationPolicy | None = None,
        evaluation_store: Any = None,
        provider_factory: Callable[[ProviderConfig], Any] | None = None,
    ) -> None:
        """Create a runner.

        The two roots default to the plan's layout (``results/`` and
        ``benchmarks/``) **resolved at construction time**: inside the checkout
        they stay relative, so artifacts and re-run commands read the way the
        plan writes them; started from anywhere else (an installed console
        script, a supervisor with its own working directory) they anchor to the
        checkout rather than inventing a second ``results/`` wherever the
        process happens to be. Phase 19 fixed this: the tab used to raise
        ``Dataset not found: benchmarks/context_rot/dataset.jsonl`` when the
        controller was built outside the repository.
        """

        self.results_root = (
            Path(results_root)
            if results_root is not None
            else repo_anchored(UI_RESULTS_ROOT)
        )
        self.dataset_root = (
            Path(dataset_root) if dataset_root is not None else repo_anchored(DATASET_ROOT)
        )
        self.policy = policy or EvaluationPolicy()
        self.evaluation_store = evaluation_store
        self.provider_factory = provider_factory or create_provider
        self._runs: dict[str, list[RunRecord]] = {}

    # ------------------------------------------------------------------ #
    # Catalogue
    # ------------------------------------------------------------------ #

    def preset_views(self) -> tuple[PresetView, ...]:
        """Every preset, marked with whether this deployment allows it."""

        views: list[PresetView] = []
        for name in sorted(PRESETS):
            entry = preset(name)
            views.append(
                PresetView(
                    name=entry.name,
                    label=name.capitalize(),
                    description=entry.description,
                    modes=tuple(entry.modes),
                    mode_labels=tuple(mode_label(mode) for mode in entry.modes),
                    trials=entry.trials,
                    limits=entry.limits.to_dict(),
                    planned_requests=entry.planned_requests,
                    allowed=name in self.policy.allowed_presets,
                )
            )
        return tuple(views)

    def policy_notice(self) -> tuple[str, ...]:
        """What this deployment restricts, in sentences a visitor can act on.

        The tab shows these before anything runs: a visitor who cannot select
        the Research preset or generate answers should learn that from the
        interface, not from a refusal after they have configured a run.
        """

        policy = self.policy
        notices: list[str] = []
        if not policy.allow_generation:
            notices.append(
                "This deployment runs benchmarks in retrieval-only mode — no "
                "prompt is sent to a model, so accuracy and faithfulness stay "
                "unset (not zero)."
            )
        allowed = tuple(sorted(policy.allowed_presets))
        if allowed != tuple(sorted(PRESETS)):
            notices.append(
                f"Allowed presets: {', '.join(allowed)} — the others are "
                "disabled for this deployment."
            )
        notices.append(
            f"Hard ceilings: {policy.max_tasks} task(s) and "
            f"{policy.max_requests} provider request(s) per run; "
            f"{policy.max_runs_per_session} run(s) per session."
        )
        notices.append(policy.retention_notice())
        return tuple(notices)

    def preset_choices(self) -> tuple[tuple[str, str], ...]:
        """Dropdown ``(label, value)`` pairs for the allowed presets."""

        return tuple(
            (
                f"{view.label} — {view.limits.get('max_tasks')} tasks × "
                f"{len(view.modes)} modes × {view.trials} trial(s)",
                view.name,
            )
            for view in self.preset_views()
            if view.allowed
        )

    def mode_choices(self) -> tuple[tuple[str, str], ...]:
        """Selectable modes: the five baselines plus the four ablations."""

        return tuple(
            (f"{mode_label(mode)} (`{mode}`)", mode) for mode in (*MODE_ORDER, *ABLATION_ORDER)
        )

    def baseline_choices(self) -> tuple[tuple[str, str], ...]:
        """Selectable paired-comparison baselines."""

        return tuple((mode_label(mode), mode) for mode in MODE_ORDER)

    def dataset_choices(self) -> tuple[str, ...]:
        """The committed dataset plus any generated tiers under ``benchmarks/``."""

        choices: list[str] = []
        default = self._default_dataset()
        if default is not None:
            choices.append(str(default))
        generated = self.dataset_root / "context_rot" / "generated"
        if generated.is_dir():
            for path in sorted(generated.glob("*.jsonl"))[:MAX_DATASET_CHOICES]:
                if str(path) not in choices:
                    choices.append(str(path))
        return tuple(choices) or (str(DEFAULT_DATASET),)

    def _default_dataset(self) -> Path | None:
        """The committed dataset, or ``None`` when this checkout has no copy.

        The declared path is tried first, so the documented workflow keeps
        naming ``benchmarks/context_rot/dataset.jsonl``; when the process is not
        running from the checkout, the anchored copy is what makes the tab work
        instead of reporting the committed dataset missing.
        """

        for candidate in (Path(DEFAULT_DATASET), repo_anchored(DEFAULT_DATASET, must_exist=True)):
            if candidate.exists():
                return candidate
        fallback = self.dataset_root / "context_rot" / "dataset.jsonl"
        return fallback if fallback.exists() else None

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def resolve_dataset(self, value: Any) -> Path:
        """Return the dataset to run, refusing anything outside ``benchmarks/``.

        Two checks, because one is not enough: the resolved path must sit under
        the dataset root (no ``../`` escape, no absolute path to ``/etc``), and
        it must exist and look like a JSONL dataset. Both are reported in the
        visitor's vocabulary.
        """

        text = str(value or "").strip()
        if text:
            candidate = Path(text)
        else:
            candidate = self._default_dataset() or repo_anchored(
                DEFAULT_DATASET, must_exist=True
            )
        resolved = self._contained(candidate, self.dataset_root, "Benchmark datasets")
        if not resolved.exists():
            raise EvaluationRequestError(
                f"Dataset not found: `{candidate}`. Regenerate it with "
                "`python benchmarks/context_rot/generation.py`."
            )
        if resolved.suffix != ".jsonl":
            raise EvaluationRequestError(
                f"`{resolved.name}` is not a `.jsonl` benchmark dataset."
            )
        return resolved

    def output_root(self, session_id: str) -> Path:
        """Return the session's private results root (created on demand)."""

        token = self._safe_token(session_id, "session")
        return self._contained(
            self.results_root / UI_RESULTS_SUBDIR / token,
            self.results_root,
            "Run output",
        )

    def output_dir(self, session_id: str, run_id: str | None = None) -> Path:
        """Return one run's output root: ``results/ui/<session>/<run>``."""

        token = self._safe_token(run_id or str(uuid4()), "run")
        return self._contained(
            self.output_root(session_id) / token, self.results_root, "Run output"
        )

    def _contained(self, candidate: Path, root: Path, label: str) -> Path:
        """Resolve ``candidate`` and refuse it unless it stays under ``root``.

        ``Path.resolve()`` is non-strict, so a path that does not exist yet is
        still normalized (``..`` folded away) before the containment check — the
        check therefore sees where the file *would* be written, not where the
        string points.
        """

        base = Path(root).expanduser().resolve()
        resolved = Path(candidate).expanduser().resolve()
        if resolved != base and not resolved.is_relative_to(base):
            raise EvaluationRequestError(
                f"{label} must stay inside `{root}` — `{candidate}` resolves outside it."
            )
        return resolved

    @staticmethod
    def _safe_token(value: Any, label: str) -> str:
        text = str(value or "").strip()
        if not text or not _SAFE_TOKEN_RE.fullmatch(text):
            raise EvaluationRequestError(
                f"The {label} identifier is not usable as a path segment."
            )
        return text

    # ------------------------------------------------------------------ #
    # Preview
    # ------------------------------------------------------------------ #

    def preview(
        self,
        *,
        session_id: str,
        preset_name: str = "quick",
        modes: Sequence[str] | None = None,
        limit: int = 0,
        dataset: Any = None,
        generate: bool = False,
        provider: ProviderConfig | None = None,
        runs_so_far: int | None = None,
    ) -> CostPreview:
        """Describe a run's cost before it is allowed to cost anything."""

        plan, tasks, path = self._plan(
            session_id=session_id,
            preset_name=preset_name,
            modes=modes,
            limit=limit,
            dataset=dataset,
            generate=generate,
            provider=provider,
            runs_so_far=runs_so_far,
        )
        entry = preset(plan.name)
        selected = _selected_task_count(len(tasks), plan.limits)
        planned = plan.planned_requests(selected)
        counter, counter_name = _resolve_counter("estimate", "")
        budget = RunBudget(limits=plan.limits, counter=counter, counter_name=counter_name)
        warnings = _preview_warnings(
            plan=plan,
            tasks_available=len(tasks),
            tasks_selected=selected,
            generate=generate,
            provider=provider,
            entry=entry,
        )
        return CostPreview(
            preset=plan.name,
            label=preset_name.capitalize(),
            description=entry.description,
            modes=plan.resolved_modes(),
            mode_labels=tuple(mode_label(mode) for mode in plan.resolved_modes()),
            trials=plan.trials,
            tasks_available=len(tasks),
            tasks_selected=selected,
            planned_requests=planned,
            limits=plan.limits.to_dict(),
            estimate=budget.estimate_ceiling(planned),
            generate=bool(generate),
            credential_source=_credential_source(generate, provider),
            dataset=_portable(path),
            dataset_sha256=plan.dataset_sha256,
            output_dir=str(self.output_root(session_id)),
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        session_id: str,
        preset_name: str = "quick",
        modes: Sequence[str] | None = None,
        limit: int = 0,
        dataset: Any = None,
        generate: bool = False,
        render_plots: bool = True,
        baseline_mode: str = "full_context",
        token_counter: str = "estimate",
        provider: ProviderConfig | None = None,
        runs_so_far: int | None = None,
    ) -> EvaluationRunView:
        """Run the pipeline for this session and return the tab's view of it.

        Never raises for a pipeline problem: a failed stage, an exhausted budget,
        or a violated control comes back as an :class:`EvaluationRunView` whose
        ``status`` says what happened. Configuration mistakes the visitor can fix
        (an unknown preset, a dataset outside ``benchmarks/``, generation without
        a connected key) raise :class:`EvaluationRequestError`, which the
        controller turns into the same kind of status line.

        Phase 19: a run also sweeps *expired* sessions out of ``results/ui/``
        before it starts, so abandoned artifacts are reclaimed by ordinary use
        rather than by a cron job somebody has to remember to install. The sweep
        is bounded, age-based, and never touches the session that asked for the
        run — see :mod:`app.retention`.
        """

        retention = self.sweep()
        plan, tasks, path = self._plan(
            session_id=session_id,
            preset_name=preset_name,
            modes=modes,
            limit=limit,
            dataset=dataset,
            generate=generate,
            provider=provider,
            runs_so_far=runs_so_far,
        )
        try:
            resolve_mode(baseline_mode)
        except ValueError as exc:
            raise EvaluationRequestError(str(exc)) from None

        run_id = str(uuid4())
        output_dir = self.output_dir(session_id, run_id)
        model: ModelSpec | None = None
        built_provider: Any | None = None
        api_key = ""
        if generate and provider is not None:
            api_key = str(provider.api_key or "")
            model = self._model_spec(provider, plan.limits)
            built_provider = build_provider(
                model, api_key=api_key, provider_factory=self.provider_factory
            )
        try:
            counter, counter_name = _resolve_counter(token_counter, model.model if model else "")
        except TokenizerUnavailableError as exc:
            raise EvaluationRequestError(str(exc)) from None

        try:
            result = run_pipeline(
                plan,
                tasks,
                model,
                built_provider,
                output_dir=output_dir,
                token_counter=counter,
                counter_name=counter_name,
                secrets=(api_key,) if api_key else (),
                baseline_mode=baseline_mode,
                render_plots=bool(render_plots),
                run_id=run_id,
                credential_source=_credential_source(generate, provider),
                notes={
                    "origin": "ui",
                    "session_id": self._safe_token(session_id, "session"),
                    "policy": {
                        "max_tasks": self.policy.max_tasks,
                        "max_requests": self.policy.max_requests,
                        "allow_generation": self.policy.allow_generation,
                        "allowed_presets": list(self.policy.allowed_presets),
                        "results_retention_seconds": self.policy.results_retention_seconds,
                    },
                    # What this run swept before it started, so an artifact says
                    # how the host it was produced on was being maintained.
                    "retention": {
                        "retention_version": RETENTION_VERSION,
                        "sessions_removed": retention.sessions_removed,
                        "files_removed": retention.files_removed,
                        "bytes_removed": retention.bytes_removed,
                        "dry_run": retention.dry_run,
                    },
                },
            )
        except (PipelineError, ValueError) as exc:
            return EvaluationRunView(
                ok=False,
                status=f"**The pipeline could not run.** {exc}",
                dry_run=not generate,
                output_dir=str(output_dir),
            )

        persisted = self._persist(session_id, result)
        record = self._record(result, persisted=persisted, session_id=session_id)
        self._runs.setdefault(str(session_id or ""), []).append(record)
        return self._view(result, persisted=persisted, output_dir=output_dir)

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #

    def runs(self, session_id: str) -> tuple[RunRecord, ...]:
        """This session's run history, newest last."""

        return tuple(self._runs.get(str(session_id or ""), ()))

    def run_count(self, session_id: str) -> int:
        return len(self._runs.get(str(session_id or ""), ()))

    def sweep(self, *, now: float | None = None, dry_run: bool = False) -> RetentionReport:
        """Reclaim expired session artifacts, without touching a live session.

        Called by :meth:`run` before a run starts and available directly to an
        operator or a test. It is a no-op (and says so in its report) when the
        policy disables retention or nothing has ever run on this host.
        """

        return sweep_results(
            self.results_root,
            policy=self.policy.retention_policy(),
            now=now,
            dry_run=dry_run,
        )

    def forget(self, session_id: str) -> None:
        """Drop a session's in-memory history (called when the session ends).

        The persisted rows are removed by the controller's own deletion path;
        this clears the index that would otherwise keep summaries of runs whose
        artifacts the visitor asked to forget.
        """

        self._runs.pop(str(session_id or ""), None)

    def discard(self, session_id: str) -> int:
        """Delete a session's artifacts and forget its index; return files removed.

        ``results/ui/<session>/`` exists only because that session asked for a
        run, so ending the session removes it. The alternative is a tab that says
        "your session data is deleted" while leaving the session's reports,
        figures, and raw answer files on the host's disk — and, on a hosted
        Space, leaving them there forever.

        Deletion is deliberately boring: the session id must be a single safe
        path segment, the directory must sit under the runner's own results root
        after resolution, and anything else is a no-op rather than a guess.
        """

        self._runs.pop(str(session_id or ""), None)
        token = str(session_id or "").strip()
        if not token or not _SAFE_TOKEN_RE.fullmatch(token):
            return 0
        base = (Path(self.results_root) / UI_RESULTS_SUBDIR).resolve()
        root = (base / token).resolve()
        if not root.is_relative_to(base) or not root.is_dir():
            return 0
        files = sum(1 for path in root.rglob("*") if path.is_file())
        shutil.rmtree(root, ignore_errors=True)
        return files

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _plan(
        self,
        *,
        session_id: str,
        preset_name: str,
        modes: Sequence[str] | None,
        limit: int,
        dataset: Any,
        generate: bool,
        provider: ProviderConfig | None,
        runs_so_far: int | None,
    ) -> tuple[ExperimentPlan, list[Any], Path]:
        """Validate a request and build the plan it describes."""

        self.policy.check(
            preset_name=str(preset_name or "").strip().lower(),
            generate=bool(generate),
            runs_so_far=self.run_count(session_id) if runs_so_far is None else int(runs_so_far),
        )
        if generate:
            if provider is None or not str(provider.api_key or "").strip():
                raise EvaluationRequestError(
                    "Generation needs a connected provider: choose one in the "
                    "sidebar, paste your key, and press **Connect**. Or leave "
                    "generation off — retrieval, prompt cost, the taxonomy, and "
                    "the report all work without a key."
                )
            if not str(provider.model or "").strip():
                raise EvaluationRequestError("Generation needs a model identifier.")
        path = self.resolve_dataset(dataset)
        try:
            tasks = load_jsonl(path)
        except (OSError, ValueError) as exc:
            raise EvaluationRequestError(f"`{path.name}` could not be read: {exc}") from None
        if not tasks:
            raise EvaluationRequestError(f"`{path.name}` contains no benchmark tasks.")

        selected_modes = _resolve_modes(modes)
        try:
            plan = ExperimentPlan.from_preset(
                str(preset_name or "quick").strip().lower(),
                modes=selected_modes,
                dataset=_portable(path),
                dataset_sha256=dataset_sha256(path),
                notes={"origin": "ui"},
                **({"max_tasks": int(limit)} if int(limit or 0) > 0 else {}),
            )
            plan.resolved_modes()
        except ValueError as exc:
            raise EvaluationRequestError(str(exc)) from None
        return self.policy.apply(plan), tasks, path

    def _model_spec(self, provider: ProviderConfig, limits: RunLimits) -> ModelSpec:
        """Build the model spec for a session-credentialed run.

        ``api_key_env`` carries the *label* of the credential route, never a
        value, and ``notes`` records that the key came from the session rather
        than the environment — so the artifact says the truth about where the
        credential came from without containing it.
        """

        return ModelSpec(
            name="ui-session",
            provider=str(provider.provider or "openai"),
            model=str(provider.model or ""),
            temperature=float(provider.temperature),
            max_tokens=provider.max_tokens,
            base_url=provider.base_url,
            timeout_seconds=float(provider.timeout_seconds),
            api_key_env=SESSION_CREDENTIAL_LABEL,
            notes=(
                "credential source: the active browser session (bring-your-own-key). "
                "No environment variable was read and no credential is recorded."
            ),
        ).with_limits(limits)

    def _persist(self, session_id: str, result: PipelineResult) -> bool:
        """Persist the run through the Phase 16 storage seam (best-effort)."""

        artifact = result.artifact
        repro = artifact.get("repro") or {}
        metadata = {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "origin": "ui",
            "pipeline_version": PIPELINE_VERSION,
            "preset": artifact.get("preset", ""),
            "modes": list(result.plan.resolved_modes()),
            "trials": result.plan.trials,
            "dry_run": bool(artifact.get("dry_run")),
            "exit_code": result.exit_code,
            "dataset": artifact.get("dataset", {}),
            "output_dir": str(result.layout.root),
            "artifacts": artifact.get("artifacts", {}),
            "versions": {
                "application": repro.get("application_version", ""),
                "brainos": repro.get("brainos_version", ""),
                "benchmark": repro.get("benchmark_version", ""),
            },
            "violations": list(artifact.get("violations") or ()),
        }
        metrics = {
            "headline": list(artifact.get("headline") or ()),
            "budget": dict(artifact.get("budget") or {}),
            "cost_estimate": dict(artifact.get("cost_estimate") or {}),
            "stages": [
                {"name": stage.name, "status": stage.status} for stage in result.stages
            ],
        }
        return persist_run(
            result.run_id,
            metadata,
            metrics,
            store=self.evaluation_store,
            session_id=str(session_id or ""),
        )

    def _record(
        self, result: PipelineResult, *, persisted: bool, session_id: str
    ) -> RunRecord:
        artifact = result.artifact
        budget = artifact.get("budget") or {}
        return RunRecord(
            run_id=result.run_id,
            session_id=str(session_id or ""),
            timestamp=result.timestamp,
            preset=str(artifact.get("preset", "")),
            modes=tuple(result.plan.resolved_modes()),
            trials=result.plan.trials,
            dry_run=bool(artifact.get("dry_run")),
            exit_code=result.exit_code,
            tasks=int(artifact.get("tasks_executed", 0) or 0),
            requests=int(budget.get("requests", 0) or 0),
            total_tokens=int(budget.get("total_tokens", 0) or 0),
            persisted=persisted,
            output_dir=str(result.layout.root),
        )

    def _view(
        self, result: PipelineResult, *, persisted: bool, output_dir: Path
    ) -> EvaluationRunView:
        artifact = result.artifact
        budget = artifact.get("budget") or {}
        return EvaluationRunView(
            ok=result.exit_code == 0,
            status=_status_line(result),
            run_id=result.run_id,
            exit_code=result.exit_code,
            dry_run=bool(artifact.get("dry_run")),
            output_dir=str(output_dir),
            headline=tuple(
                dict(row) for row in artifact.get("headline") or () if isinstance(row, Mapping)
            ),
            stages=tuple(stage.to_dict() for stage in result.stages),
            plots=tuple(str(path) for path in sorted((artifact.get("plots") or {}).values())),
            report_markdown=result.report_markdown,
            repro=dict(artifact.get("repro") or {}),
            rerun_command=str(artifact.get("repro", {}).get("rerun_command", "")),
            pipeline_rerun_command=str(artifact.get("pipeline_rerun_command", "")),
            artifact_path=result.manifest_path,
            report_path=str(output_dir / "report" / "report.md"),
            persisted=persisted,
            cost=_executed_cost(artifact, output_dir),
            warnings=tuple(artifact.get("warnings") or ()),
            summary={
                "preset": artifact.get("preset", ""),
                "tasks_executed": artifact.get("tasks_executed", 0),
                "tasks_available": artifact.get("tasks_available", 0),
                "modes": list(result.plan.resolved_modes()),
                "trials": result.plan.trials,
                "requests": budget.get("requests", 0),
                "total_tokens": budget.get("total_tokens", 0),
                "skips": budget.get("skips", 0),
                "failures": budget.get("failures", 0),
                "within_limits": budget.get("within_limits", True),
                "violations": list(artifact.get("violations") or ()),
                "aborted": artifact.get("aborted"),
                "failed_stages": list(result.failed_stages()),
                "skipped_stages": list(result.skipped_stages()),
                "credential_check": artifact.get("credential_check", {}),
                "security": artifact.get("security", {}),
                "layout": artifact.get("layout", {}),
                "dataset": artifact.get("dataset", {}),
            },
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_modes(modes: Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize a mode selection, accepting the CLI's keywords."""

    if modes is None:
        return None
    if isinstance(modes, str):
        raw: Sequence[str] = [part for part in modes.split(",")]
    else:
        raw = list(modes)
    cleaned = [str(mode).strip() for mode in raw if str(mode).strip()]
    if not cleaned:
        return None
    lowered = {mode.lower() for mode in cleaned}
    if "all" in lowered:
        return MODE_ORDER
    if "ablations" in lowered:
        # An ablation without its full reference is not interpretable, so the
        # keyword always includes the system under test (the CLI's own rule).
        return tuple(dict.fromkeys((MODE_BRAINOS, *ABLATION_ORDER)))
    resolved: list[str] = []
    for mode in cleaned:
        try:
            name = resolve_mode(mode)
        except ValueError as exc:
            raise EvaluationRequestError(str(exc)) from None
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)



def _selected_task_count(available: int, limits: RunLimits) -> int:
    if limits.max_tasks is None:
        return int(available)
    return min(int(available), int(limits.max_tasks))


def _resolve_counter(name: str, model: str) -> tuple[TokenCounter, str]:
    if str(name) == "tiktoken":
        return tiktoken_counter(model), f"tiktoken:{model}"
    return estimate_tokens, "estimate_tokens"


def _credential_source(generate: bool, provider: ProviderConfig | None) -> str:
    """Name the credential route without naming a credential."""

    if not generate:
        return "none (retrieval-only dry run)"
    if provider is None:
        return "none requested"
    return f"the active browser session ({provider.provider or 'provider'} bring-your-own-key)"


def _preview_warnings(
    *,
    plan: ExperimentPlan,
    tasks_available: int,
    tasks_selected: int,
    generate: bool,
    provider: ProviderConfig | None,
    entry: Any,
) -> list[str]:
    """What a visitor should know *before* pressing run."""

    warnings: list[str] = []
    if tasks_selected < tasks_available:
        warnings.append(
            f"The dataset holds {tasks_available} task(s); the run's `max_tasks` "
            f"ceiling selects the first {tasks_selected}."
        )
    if not generate:
        warnings.append(
            "Retrieval-only: no prompt is sent to a model, so accuracy, "
            "faithfulness, and quality-adjusted efficiency stay unset (not zero)."
        )
    else:
        warnings.append(
            "Generation sends one request per (task, mode, trial) with your "
            "session key. Your provider account is billed for all of it."
        )
        if provider is not None and float(provider.temperature or 0.0) > 0 and plan.trials == 1:
            warnings.append(
                "Temperature is above zero with a single trial: stochastic "
                "sampling needs repeated trials before any number is reportable."
            )
    if plan.trials == 1:
        warnings.append("One trial: confidence intervals are degenerate.")
    return warnings


def _status_line(result: PipelineResult) -> str:
    """The visitor-facing summary of what just happened."""

    artifact = result.artifact
    budget = artifact.get("budget") or {}
    modes = len(result.plan.resolved_modes())
    base = (
        f"**Pipeline finished** — {artifact.get('tasks_executed', 0)} task(s) × "
        f"{modes} mode(s) × {result.plan.trials} trial(s)"
    )
    if result.dry_run:
        base += ", retrieval-only (no model was called, no key was used)"
    else:
        base += (
            f", {budget.get('requests', 0)} request(s) and "
            f"{budget.get('total_tokens', 0)} token(s) charged to your provider account"
        )
    notes: list[str] = []
    if result.violations:
        notes.append(
            f"**{len(result.violations)} controlled-comparison violation(s)** — these "
            "numbers are not a controlled comparison."
        )
    if result.aborted:
        notes.append(
            f"Stopped at a cost ceiling (`{result.aborted.get('limit', '')}`): "
            f"{result.aborted.get('reason', '')}"
        )
    failed = result.failed_stages()
    if failed:
        notes.append(f"Stage(s) failed: {', '.join(failed)} — see the stage table.")
    skipped = result.skipped_stages()
    if skipped:
        notes.append(f"Stage(s) skipped: {', '.join(skipped)}.")
    check = artifact.get("credential_check") or {}
    report_scan = check.get("report_scan") or {}
    findings = int(check.get("finding_count", 0) or 0) + int(
        report_scan.get("finding_count", 0) or 0
    )
    quarantined = list(check.get("quarantined") or ())
    if findings or quarantined:
        notes.append(
            f"**{findings} credential-shaped value(s) found in the artifacts"
            + (
                f"; {len(quarantined)} file(s) were deleted from this run's "
                "output directory** — the run is not safe to share and its "
                "numbers are incomplete. Rotate any key that reached a model "
                "response."
                if quarantined
                else "** — the run is not safe to share."
            )
        )
    else:
        scanned = int(check.get("files_scanned", 0) or 0) + int(
            report_scan.get("files_scanned", 0) or 0
        )
        notes.append(f"Artifact credential scan clean ({scanned} file(s)).")
    if not any(int(row.get("graded", 0) or 0) for row in artifact.get("headline") or ()):
        notes.append("No answers were graded: accuracy, faithfulness, and QAE are unset, not zero.")
    return base + ".\n\n" + "\n".join(f"- {note}" for note in notes)


def _executed_cost(artifact: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """The receipt for a run that already happened.

    Same field names as :class:`CostPreview` so one renderer handles both, plus
    the ``used`` block only an executed run can have. The point of keeping the
    planned numbers next to the used ones is that a visitor can see a ceiling
    that was *not* reached, which is the difference between "the run was cheap"
    and "the run was cut short".
    """

    budget = _mapping(artifact.get("budget"))
    plan = _mapping(artifact.get("plan"))
    estimate = _mapping(artifact.get("cost_estimate"))
    generation = _mapping(artifact.get("generation"))
    return {
        "executed": True,
        "preset": str(artifact.get("preset", "")),
        "label": str(artifact.get("preset", "")).capitalize(),
        "description": (
            f"Run {str(artifact.get('run_id', ''))[:8]} · exit "
            f"{artifact.get('exit_code', 0)}"
        ),
        "modes": list(plan.get("modes") or ()),
        "mode_labels": list(plan.get("mode_labels") or ()),
        "trials": int(plan.get("trials", 1) or 1),
        "tasks_available": int(artifact.get("tasks_available", 0) or 0),
        "tasks_selected": int(artifact.get("tasks_executed", 0) or 0),
        "planned_requests": estimate.get("planned_requests"),
        "limits": dict(_mapping(budget.get("limits"))),
        "estimate": dict(estimate),
        "used": {
            "requests": int(budget.get("requests", 0) or 0),
            "input_tokens": int(budget.get("input_tokens", 0) or 0),
            "output_tokens": int(budget.get("output_tokens", 0) or 0),
            "total_tokens": int(budget.get("total_tokens", 0) or 0),
            "skips": int(budget.get("skips", 0) or 0),
            "failures": int(budget.get("failures", 0) or 0),
            "remaining_requests": budget.get("remaining_requests"),
            "remaining_total_tokens": budget.get("remaining_total_tokens"),
            "within_limits": bool(budget.get("within_limits", True)),
        },
        "generate": not bool(artifact.get("dry_run", True)),
        "credential_source": str(generation.get("credential_source", "")),
        "dataset": str(plan.get("dataset", "")),
        "dataset_sha256": str(plan.get("dataset_sha256", "")),
        "output_dir": str(output_dir),
        "warnings": [str(item) for item in artifact.get("warnings") or ()],
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` as a mapping, or an empty one (artifact fields may be absent)."""

    return value if isinstance(value, Mapping) else {}


def _portable(path: Path) -> str:
    """Record a dataset path a reader can re-run, not this machine's layout.

    The containment check needs an absolute path; the artifact needs a portable
    one. A dataset inside the working directory is recorded relative to it, so
    the re-run command works from any checkout. Phase 19 adds the second base: a
    process started outside the checkout anchors its dataset to the checkout, and
    the recorded path is still the plan's relative one rather than this
    machine's directory layout.
    """

    for base in (Path.cwd(), REPO_ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


__all__ = [
    "DATASET_ROOT",
    "DEFAULT_DATASET",
    "RESULTS_ROOT",
    "RETENTION_VERSION",
    "RUNS_SUBDIR",
    "RetentionPolicy",
    "RetentionReport",
    "MAX_DATASET_CHOICES",
    "PIPELINE_VERSION",
    "SESSION_CREDENTIAL_LABEL",
    "UI_RESULTS_ROOT",
    "UI_RESULTS_SUBDIR",
    "CostPreview",
    "EvaluationPolicy",
    "EvaluationRequestError",
    "EvaluationRunView",
    "PresetView",
    "RunRecord",
    "UIEvaluationRunner",
]
