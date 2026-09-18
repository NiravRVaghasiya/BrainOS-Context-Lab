"""Phase 17 in the browser: the Evaluation tab drives the same pipeline.

The tab is the first surface where a *visitor* can spend money on a benchmark,
so these tests are about the guard rails rather than the numbers:

* the cost ceiling is visible before a run is allowed to start;
* a dataset path or session id cannot be used to write outside the runner's own
  directories;
* a deployment policy can narrow what the tab offers, and only ever narrows it;
* generation uses the session's own key — never an environment variable, and
  never recorded anywhere a reader could find it;
* a run is persisted through the same ``persist_run`` seam as everything else,
  and ending the session removes it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.controller import UIController
from app.evaluation import (
    SESSION_CREDENTIAL_LABEL,
    UI_RESULTS_SUBDIR,
    EvaluationPolicy,
    EvaluationRequestError,
    UIEvaluationRunner,
    public_evaluation_policy,
)
from evaluation.limits import PRESETS
from providers.base import ProviderConfig
from tests.fakes import FakeLLMProvider, InMemoryEvaluationStore, RecordingProvider

KEY = "sk-evaluation-SECRET-7777"
SESSION = "11111111-2222-3333-4444-555555555555"


def _runner(tmp_path: Path, **kwargs: object) -> UIEvaluationRunner:
    kwargs.setdefault("results_root", tmp_path / "results")
    kwargs.setdefault("evaluation_store", InMemoryEvaluationStore())
    # ``FakeLLMProvider`` takes a :class:`ProviderConfig` positionally, exactly
    # like the real ``create_provider`` the runner defaults to.
    kwargs.setdefault("provider_factory", FakeLLMProvider)
    return UIEvaluationRunner(**kwargs)  # type: ignore[arg-type]


def _config(key: str = KEY) -> ProviderConfig:
    return ProviderConfig(provider="openai", model="gpt-4o-mini", api_key=key)


def _paths(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #


def test_catalogue_lists_every_preset_with_its_ceilings(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    views = {view.name: view for view in runner.preset_views()}
    assert set(views) == set(PRESETS)
    quick = views["quick"]
    assert quick.limits["max_requests"] == PRESETS["quick"].limits.max_requests
    assert quick.planned_requests is not None
    assert quick.allowed is True
    assert quick.mode_labels
    assert [value for _, value in runner.preset_choices()] == sorted(PRESETS)


def test_catalogue_offers_the_modes_and_baselines_the_benchmark_defines(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)

    modes = {value for _, value in runner.mode_choices()}
    assert {"brainos", "full_context", "rag", "sliding_window", "brainos_rag"} <= modes
    assert "brainos_no_memory" in modes  # the Phase 11 ablations are selectable
    baselines = {value for _, value in runner.baseline_choices()}
    assert "full_context" in baselines
    # An ablation is not a baseline: pairing against it would compare a broken
    # system with a more broken system.
    assert "brainos_no_memory" not in baselines
    assert all("`" in label for label, _ in runner.mode_choices())


def test_catalogue_lists_the_committed_dataset_and_generated_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = tmp_path / "benchmarks"
    (datasets / "context_rot" / "generated").mkdir(parents=True)
    (datasets / "context_rot" / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    (datasets / "context_rot" / "generated" / "tier-1.jsonl").write_text("{}\n", encoding="utf-8")
    (datasets / "context_rot" / "generated" / "notes.txt").write_text("x", encoding="utf-8")
    runner = UIEvaluationRunner(results_root=tmp_path / "results", dataset_root=datasets)

    choices = runner.dataset_choices()
    assert any(choice.endswith("dataset.jsonl") for choice in choices)
    assert any(choice.endswith("tier-1.jsonl") for choice in choices)
    # Only datasets are offered: a stray text file is not a benchmark.
    assert not any(choice.endswith("notes.txt") for choice in choices)


# --------------------------------------------------------------------------- #
# Deployment policy
# --------------------------------------------------------------------------- #


def test_public_policy_is_conservative_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BRAINOS_LAB_EVAL_PRESETS",
        "BRAINOS_LAB_EVAL_ALLOW_GENERATION",
        "BRAINOS_LAB_EVAL_MAX_TASKS",
        "BRAINOS_LAB_EVAL_MAX_REQUESTS",
    ):
        monkeypatch.delenv(name, raising=False)

    policy = public_evaluation_policy()

    assert policy == EvaluationPolicy(
        allowed_presets=("quick",),
        allow_generation=False,
        max_tasks=20,
        max_requests=60,
    )


def test_ui_controller_uses_the_public_policy_when_none_is_injected() -> None:
    controller = UIController()

    assert controller.evaluation.policy == EvaluationPolicy(
        allowed_presets=("quick",),
        allow_generation=False,
        max_tasks=20,
        max_requests=60,
    )


def test_public_policy_environment_overrides_are_explicit_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINOS_LAB_EVAL_PRESETS", "quick,standard")
    monkeypatch.setenv("BRAINOS_LAB_EVAL_ALLOW_GENERATION", "yes")
    monkeypatch.setenv("BRAINOS_LAB_EVAL_MAX_TASKS", "7")
    monkeypatch.setenv("BRAINOS_LAB_EVAL_MAX_REQUESTS", "19")

    policy = public_evaluation_policy()

    assert policy.allowed_presets == ("quick", "standard")
    assert policy.allow_generation is True
    assert policy.max_tasks == 7
    assert policy.max_requests == 19

    monkeypatch.setenv("BRAINOS_LAB_EVAL_PRESETS", "quick,unknown")
    monkeypatch.setenv("BRAINOS_LAB_EVAL_MAX_TASKS", "0")
    assert public_evaluation_policy() == EvaluationPolicy(
        allowed_presets=("quick",),
        allow_generation=True,
        max_tasks=20,
        max_requests=19,
    )


def test_policy_marks_disallowed_presets_and_hides_them_from_the_dropdown(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, policy=EvaluationPolicy(allowed_presets=("quick",)))

    allowed = {view.name: view.allowed for view in runner.preset_views()}
    assert allowed == {"quick": True, "research": False, "standard": False}
    assert [value for _, value in runner.preset_choices()] == ["quick"]
    with pytest.raises(EvaluationRequestError, match="does not allow the `research` preset"):
        runner.preview(session_id=SESSION, preset_name="research")


def test_policy_can_disable_generation_and_says_so_before_a_run(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, policy=EvaluationPolicy(allow_generation=False))

    notices = " ".join(runner.policy_notice())
    assert "retrieval-only" in notices
    with pytest.raises(EvaluationRequestError, match="retrieval-only"):
        runner.run(
            session_id=SESSION, preset_name="quick", generate=True, provider=_config()
        )
    # The same run without generation is allowed.
    view = runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)
    assert view.ok


def test_policy_caps_runs_per_session(tmp_path: Path) -> None:
    runner = _runner(tmp_path, policy=EvaluationPolicy(max_runs_per_session=2))

    for _ in range(2):
        runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)
    with pytest.raises(EvaluationRequestError, match="per-session ceiling"):
        runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)
    # A different session is unaffected.
    assert runner.run_count("other-session") == 0


def test_policy_notice_reports_the_ceilings_it_enforces(tmp_path: Path) -> None:
    runner = _runner(tmp_path, policy=EvaluationPolicy(max_tasks=4, max_requests=12))

    notice = " ".join(runner.policy_notice())
    assert "4 task(s)" in notice
    assert "12 provider request(s)" in notice


def test_policy_only_ever_tightens_a_plan(tmp_path: Path) -> None:
    from evaluation.experiment import ExperimentPlan

    policy = EvaluationPolicy(max_tasks=3, max_requests=9)
    tighter = ExperimentPlan.from_preset("quick", max_tasks=1, max_requests=2)
    looser = ExperimentPlan.from_preset("research")

    assert policy.apply(tighter).limits.max_tasks == 1
    assert policy.apply(tighter).limits.max_requests == 2
    applied = policy.apply(looser)
    assert applied.limits.max_tasks == 3
    assert applied.limits.max_requests == 9


def test_policy_rejects_an_unknown_preset_and_a_non_positive_ceiling() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        EvaluationPolicy(allowed_presets=("turbo",))
    with pytest.raises(ValueError, match="at least one preset"):
        EvaluationPolicy(allowed_presets=())
    with pytest.raises(ValueError, match="max_tasks must be a positive integer"):
        EvaluationPolicy(max_tasks=0)


def test_a_run_tightened_by_policy_records_the_tightened_ceiling(tmp_path: Path) -> None:
    runner = _runner(tmp_path, policy=EvaluationPolicy(max_tasks=1, max_requests=3))

    view = runner.run(session_id=SESSION, preset_name="quick", render_plots=False)
    manifest = json.loads(Path(view.artifact_path).read_text(encoding="utf-8"))

    assert view.ok
    assert manifest["budget"]["limits"]["max_tasks"] == 1
    assert manifest["budget"]["limits"]["max_requests"] == 3
    assert view.cost["limits"]["max_tasks"] == 1


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def test_session_artifacts_live_under_results_ui_and_the_session_id(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    root = runner.output_root(SESSION)
    assert root == Path(runner.results_root) / UI_RESULTS_SUBDIR / SESSION
    run_dir = runner.output_dir(SESSION, "run-1")
    assert run_dir.parent == root
    assert runner.output_dir(SESSION) != run_dir  # a fresh run id per call


@pytest.mark.parametrize(
    "session_id",
    ["../../escape", "a/b", "", "   ", "x" * 200, "sess;rm -rf /"],
)
def test_an_unusable_session_id_is_refused_rather_than_sanitized(
    tmp_path: Path, session_id: str
) -> None:
    runner = _runner(tmp_path)

    with pytest.raises(EvaluationRequestError, match="not usable as a path segment"):
        runner.output_root(session_id)


def test_a_dataset_outside_the_benchmark_root_is_refused(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    for candidate in ("/etc/passwd", "../../secret.jsonl", str(tmp_path / "elsewhere.jsonl")):
        with pytest.raises(EvaluationRequestError, match="must stay inside"):
            runner.resolve_dataset(candidate)


def test_a_dataset_that_is_missing_or_not_jsonl_is_refused(tmp_path: Path) -> None:
    datasets = tmp_path / "benchmarks" / "context_rot"
    datasets.mkdir(parents=True)
    (datasets / "notes.md").write_text("# not a dataset", encoding="utf-8")
    runner = UIEvaluationRunner(results_root=tmp_path / "r", dataset_root=tmp_path / "benchmarks")

    with pytest.raises(EvaluationRequestError, match="Dataset not found"):
        runner.resolve_dataset(str(datasets / "missing.jsonl"))
    with pytest.raises(EvaluationRequestError, match=".jsonl"):
        runner.resolve_dataset(str(datasets / "notes.md"))


def test_a_traversal_dataset_path_cannot_escape_even_when_it_exists(
    tmp_path: Path,
) -> None:
    datasets = tmp_path / "benchmarks" / "context_rot"
    datasets.mkdir(parents=True)
    (datasets / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    runner = UIEvaluationRunner(results_root=tmp_path / "r", dataset_root=tmp_path / "benchmarks")

    # ``..`` that resolves back inside the root is fine; ``..`` that leaves it is not.
    assert runner.resolve_dataset(str(datasets / ".." / "context_rot" / "dataset.jsonl")).exists()
    with pytest.raises(EvaluationRequestError, match="must stay inside"):
        runner.resolve_dataset(str(datasets / ".." / ".." / "outside.jsonl"))


# --------------------------------------------------------------------------- #
# Preview: the ceiling before the spend
# --------------------------------------------------------------------------- #


def test_preview_reports_the_planned_requests_and_the_ceiling(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    preview = runner.preview(session_id=SESSION, preset_name="quick", limit=2)

    assert preview.tasks_selected == 2
    assert preview.planned_requests == 2 * len(PRESETS["quick"].modes) * PRESETS["quick"].trials
    assert preview.estimate["output_token_ceiling"] > 0
    # Input tokens are honestly unknown before the prompts exist.
    assert isinstance(preview.estimate["input_tokens"], str)
    assert preview.limits["max_requests"] == PRESETS["quick"].limits.max_requests
    assert preview.generate is False
    assert "none" in preview.credential_source


def test_preview_writes_nothing_and_calls_nothing(tmp_path: Path) -> None:
    provider = RecordingProvider()
    runner = _runner(tmp_path, provider_factory=lambda config: provider)

    runner.preview(session_id=SESSION, preset_name="research")

    assert _paths(tmp_path) == []
    assert provider.requests == []


def test_preview_warns_that_a_retrieval_only_run_grades_no_answer(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    warnings = " ".join(runner.preview(session_id=SESSION, preset_name="quick").warnings)
    assert "unset (not zero)" in warnings
    assert "One trial" in warnings


def test_preview_warns_when_a_limit_truncates_the_dataset(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    preview = runner.preview(session_id=SESSION, preset_name="quick", limit=1)
    assert preview.tasks_available > preview.tasks_selected
    assert any("max_tasks" in warning for warning in preview.warnings)


def test_preview_names_the_credential_route_generation_would_use(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    preview = runner.preview(
        session_id=SESSION, preset_name="quick", generate=True, provider=_config()
    )
    assert preview.generate is True
    assert "browser session" in preview.credential_source
    assert "bring-your-own-key" in preview.credential_source
    # The route is named; the key is not in the preview.
    assert KEY not in json.dumps(preview.to_dict(), default=str)
    assert preview.planned_requests > 0


def test_preview_refuses_to_price_a_generated_run_it_cannot_price(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    # Refusing beats showing a retrieval-only ceiling under a heading that says
    # "what this run would cost": a visitor who asked to price a generated run
    # would read `0 requests` as "generation is free".
    with pytest.raises(EvaluationRequestError, match="Connect"):
        runner.preview(session_id=SESSION, preset_name="quick", generate=True)
    # The same visitor, asking for the run that *can* happen, gets a price.
    assert runner.preview(session_id=SESSION, preset_name="quick").planned_requests > 0


def test_preview_accepts_an_explicit_mode_selection(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    preview = runner.preview(
        session_id=SESSION, preset_name="quick", modes=["full_context", "rag"]
    )
    assert list(preview.modes) == ["full_context", "rag"]
    assert preview.planned_requests == 2 * preview.tasks_selected


def test_the_ablations_keyword_always_includes_the_full_system(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    preview = runner.preview(session_id=SESSION, preset_name="quick", modes=["ablations"])
    assert "brainos" in preview.modes
    assert "brainos_no_memory" in preview.modes


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_a_dry_run_writes_the_four_directories_and_persists_a_summary(
    tmp_path: Path,
) -> None:
    store = InMemoryEvaluationStore()
    runner = _runner(tmp_path, evaluation_store=store)

    view = runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)

    assert view.ok and view.exit_code == 0 and view.dry_run
    # One identifier names the directory, the manifest's run id, and the history
    # row, so a visitor can go from a row in the table to the files on disk.
    assert view.output_dir.endswith(view.run_id)
    written = _paths(Path(view.output_dir))
    assert any(path.startswith("raw/") for path in written)
    assert any(path.startswith("aggregated/") for path in written)
    assert any(path.startswith("report/") for path in written)
    assert Path(view.report_path).is_file()
    assert Path(view.artifact_path).is_file()
    assert view.persisted is True
    assert len(store.runs) == 1
    saved = next(iter(store.runs.values()))
    assert saved["session_id"] == SESSION
    assert saved["metadata"]["origin"] == "ui"
    assert saved["metadata"]["exit_code"] == 0
    assert saved["metadata"]["pipeline_version"]
    assert saved["metadata"]["versions"]["brainos"]
    assert saved["metrics"]["headline"]
    assert saved["metrics"]["budget"]["limits"]["max_requests"]
    # The persisted row names the artifacts but contains none of them.
    assert saved["metadata"]["output_dir"] == view.output_dir
    assert "report_markdown" not in json.dumps(saved, default=str)


def test_a_run_renders_figures_into_the_plots_directory(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib", reason="figures need matplotlib")
    runner = _runner(tmp_path)

    view = runner.run(session_id=SESSION, preset_name="quick", limit=1)

    assert len(view.plots) == 6
    assert all(Path(path).is_file() for path in view.plots)
    assert all(path.endswith(".png") for path in view.plots)


def test_a_run_records_history_for_its_session_only(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)
    runner.run(session_id="other", preset_name="quick", limit=1, render_plots=False)

    assert runner.run_count(SESSION) == 1
    assert runner.run_count("other") == 1
    record = runner.runs(SESSION)[0]
    assert record.session_id == SESSION
    assert record.preset == "quick"
    assert record.dry_run is True
    assert record.tasks == 1
    assert record.output_dir


def test_forget_drops_the_history_index(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)

    runner.forget(SESSION)

    assert runner.runs(SESSION) == ()
    assert runner.run_count(SESSION) == 0


def test_discard_removes_the_session_artifacts_from_disk(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    view = runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)
    assert Path(view.report_path).is_file()

    removed = runner.discard(SESSION)

    assert removed > 0
    assert not (Path(runner.results_root) / UI_RESULTS_SUBDIR / SESSION).exists()
    assert runner.runs(SESSION) == ()
    # Another session's artifacts are untouched, and a junk id removes nothing.
    assert runner.discard("../../escape") == 0
    assert runner.discard("") == 0


def test_a_run_without_a_store_is_still_a_successful_run(tmp_path: Path) -> None:
    runner = UIEvaluationRunner(results_root=tmp_path / "results")

    view = runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)

    assert view.ok
    assert view.persisted is False


def test_a_failing_store_does_not_lose_the_run(tmp_path: Path) -> None:
    from tests.fakes import RaisingStore

    runner = _runner(tmp_path, evaluation_store=RaisingStore())

    view = runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)

    assert view.ok
    assert view.persisted is False
    assert Path(view.report_path).is_file()


def test_a_generated_run_uses_the_session_key_and_never_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[ProviderConfig] = []
    provider = RecordingProvider()

    def factory(config: ProviderConfig) -> RecordingProvider:
        built.append(config)
        return provider

    runner = _runner(tmp_path, provider_factory=factory)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    view = runner.run(
        session_id=SESSION,
        preset_name="quick",
        limit=1,
        generate=True,
        provider=_config(),
        render_plots=False,
    )

    assert view.ok and not view.dry_run
    assert provider.requests, "the run should have called the model"
    assert SESSION_CREDENTIAL_LABEL not in os.environ
    assert built and built[0].api_key == KEY
    # The key travelled as a value, not as an environment variable.
    assert "OPENAI_API_KEY" not in os.environ
    assert SESSION_CREDENTIAL_LABEL not in os.environ
    manifest = json.loads(Path(view.artifact_path).read_text(encoding="utf-8"))
    assert manifest["generation"]["enabled"] is True
    assert "browser session" in manifest["generation"]["credential_source"]
    assert KEY not in json.dumps(manifest, default=str)
    assert KEY not in view.report_markdown
    # The manifest names the *route* a key arrived by, which is the half a
    # reader needs and the half that is safe to keep.
    assert manifest["generation"]["settings"]


def test_a_generated_run_grades_answers_so_accuracy_is_a_number(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    view = runner.run(
        session_id=SESSION,
        preset_name="quick",
        limit=1,
        generate=True,
        provider=_config(),
        render_plots=False,
    )

    graded = [row for row in view.headline if row.get("graded")]
    assert graded
    assert all(row["accuracy"] is not None for row in graded)


def test_a_retrieval_only_run_leaves_answer_metrics_unset(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    view = runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)

    assert view.headline
    assert all(int(row["graded"]) == 0 for row in view.headline)
    assert "unset, not zero" in view.status


def test_the_view_exposes_the_stages_and_the_reproducibility_manifest(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)

    view = runner.run(session_id=SESSION, preset_name="quick", limit=1, render_plots=False)

    assert [stage["name"] for stage in view.stages][:2] == ["experiment", "raw"]
    assert view.repro["dataset"]["sha256"]
    assert view.pipeline_rerun_command.startswith("python -m evaluation.pipeline")
    assert "--dry-run" in view.pipeline_rerun_command
    assert view.summary["tasks_executed"] == 1
    assert view.cost["used"]["requests"] == 0


def test_an_unusable_baseline_or_counter_is_refused_before_any_work(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)

    with pytest.raises(EvaluationRequestError):
        runner.run(
            session_id=SESSION,
            preset_name="quick",
            limit=1,
            baseline_mode="not_a_mode",
            render_plots=False,
        )
    assert _paths(tmp_path) == []


# --------------------------------------------------------------------------- #
# Through the controller
# --------------------------------------------------------------------------- #


def _controller(tmp_path: Path, **kwargs: object) -> UIController:
    controller = UIController(
        provider_factory=FakeLLMProvider, evaluation_store=InMemoryEvaluationStore()
    )
    controller.evaluation = _runner(tmp_path, **kwargs)
    return controller


def test_controller_renders_the_preset_catalogue_without_a_session(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    view = controller.evaluation_presets()

    assert view.ok
    assert view.preset_rows
    assert "ceiling" in view.status
    assert view.headline_rows == []


def test_controller_preview_is_a_cost_disclosure_not_a_run(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    sid = controller.ensure_session(None).session_id

    view = controller.evaluation_preview(sid, preset="quick", limit=1)

    assert view.ok and not view.ran
    assert "What this run would cost" in view.cost_markdown
    assert view.cost_payload["planned_requests"]
    assert _paths(tmp_path) == []


def test_controller_run_renders_tables_figures_and_the_report(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib", reason="figures need matplotlib")
    controller = _controller(tmp_path)
    sid = controller.ensure_session(None).session_id

    view = controller.run_evaluation(sid, preset="quick", limit=1)

    assert view.ok and view.ran
    assert view.headline_rows and view.stage_rows
    assert len(view.headline_rows[0]) == 10
    assert view.plots and all(Path(path).is_file() for path in view.plots)
    assert "# BrainOS Context Lab — evaluation report" in view.report_markdown
    assert "### Artifacts" in view.artifacts_markdown
    assert "### Reproducibility" in view.repro_markdown
    assert "What this run cost" in view.cost_markdown
    assert view.history_rows


def test_controller_turns_a_refusal_into_a_readable_status(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    sid = controller.ensure_session(None).session_id

    view = controller.run_evaluation(sid, preset="quick", dataset="/etc/passwd")

    assert not view.ok and not view.ran
    assert view.status.startswith("**Not run**")
    assert "must stay inside" in view.status
    assert _paths(tmp_path) == []


def test_controller_refuses_generation_without_a_sidebar_key(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    sid = controller.ensure_session(None).session_id

    view = controller.run_evaluation(sid, preset="quick", generate=True)

    assert not view.ok
    assert "API key" in view.status
    assert "retrieval-only" in view.status


def test_controller_never_renders_the_session_key(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    sid = controller.ensure_session(None).session_id
    controller.connect(sid, provider="openai", model="gpt-4o-mini", api_key=KEY)

    view = controller.run_evaluation(
        sid, preset="quick", limit=1, generate=True, render_plots=False
    )
    blob = json.dumps(view.__dict__, default=str) + view.report_markdown

    assert view.ok
    assert KEY not in blob
    assert KEY.lower() not in blob.lower()
    # The credential *route* is recorded, which is the useful half.
    assert "bring-your-own-key" in blob


def test_controller_history_and_export_cover_the_session_runs(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    sid = controller.ensure_session(None).session_id
    controller.run_evaluation(sid, preset="quick", limit=1, render_plots=False)

    history = controller.evaluation_history(sid)
    exported = controller.export_session(sid)

    assert len(history.history_rows) == 1
    assert "1 run(s) in this session" in history.status
    assert len(exported["evaluation_runs"]) == 1
    assert exported["evaluation_runs"][0]["preset"] == "quick"
    assert KEY not in json.dumps(exported, default=str)


def test_ending_a_session_deletes_its_runs_and_its_artifacts(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    sid = controller.ensure_session(None).session_id
    controller.run_evaluation(sid, preset="quick", limit=1, render_plots=False)
    assert _paths(tmp_path)

    ended = controller.end_session(sid)

    assert "benchmark artifact file(s) deleted" in ended.status
    assert _paths(tmp_path) == []
    assert controller.evaluation_history(sid).history_rows == []
    assert controller.evaluation.run_count(sid) == 0
