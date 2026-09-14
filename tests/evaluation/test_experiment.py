"""Phase 9 controlled experiments: the controls, the cost stop, the artifact.

These tests use the committed dataset and a deterministic provider, so they run
without the BrainOS runtime installed and without an API key. What they pin is
not an accuracy number — it is the set of properties that make a mode comparison
mean something: one model, one parameter set, one system prompt, one task
wording, one dataset revision, and a run-level cost ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.modes import MODE_BRAINOS, MODE_FULL_CONTEXT, MODE_SLIDING_WINDOW
from evaluation.analysis import compare_runs
from evaluation.datasets import load_jsonl
from evaluation.experiment import (
    ExperimentError,
    ExperimentPlan,
    check_constants,
    compare_modes_across_models,
    main,
    run_controlled_experiment,
)
from evaluation.generation import ModelSpec
from evaluation.limits import RunLimits
from tests.fakes import RecordingProvider

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)[:2]
SPEC = ModelSpec(model="fake-model", temperature=0.0, max_tokens=64)


def _clock() -> float:
    """Monotonic-ish fake clock: 100 ms between calls, so latency is never zero."""

    state = {"t": 0.0}

    def tick() -> float:
        state["t"] += 0.1
        return state["t"]

    return tick


def _plan(**overrides) -> ExperimentPlan:  # type: ignore[no-untyped-def]
    defaults = {
        "name": "test",
        "modes": (MODE_FULL_CONTEXT, MODE_BRAINOS),
        "limits": RunLimits(max_requests=1000),
    }
    defaults.update(overrides)
    return ExperimentPlan(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


def test_plan_from_preset_uses_the_preset_selection() -> None:
    plan = ExperimentPlan.from_preset("quick")

    assert plan.name == "quick"
    assert plan.trials == 1
    assert plan.limits.max_tasks == 20
    assert plan.resolved_modes() == (MODE_FULL_CONTEXT, "rag", MODE_BRAINOS)
    assert plan.dataset == str(DATASET)


def test_plan_normalizes_modes_and_drops_duplicates() -> None:
    plan = _plan(modes=("no_memory", MODE_SLIDING_WINDOW, MODE_BRAINOS))

    assert plan.resolved_modes() == (MODE_SLIDING_WINDOW, MODE_BRAINOS)
    with pytest.raises(ValueError):
        _plan(modes=("not-a-mode",)).resolved_modes()


def test_plan_rejects_a_useless_trial_count() -> None:
    with pytest.raises(ValueError):
        ExperimentPlan(trials=0)
    with pytest.raises(ValueError):
        ExperimentPlan(modes=())


def test_planned_requests_counts_tasks_modes_and_trials() -> None:
    plan = _plan(modes=(MODE_FULL_CONTEXT, MODE_BRAINOS), trials=3)

    assert plan.planned_requests(7) == 7 * 2 * 3


# --------------------------------------------------------------------------- #
# Dry run and generated run
# --------------------------------------------------------------------------- #


def test_dry_run_measures_prompts_without_calling_a_model() -> None:
    run = run_controlled_experiment(_plan(), TASKS)

    assert run.dry_run is True
    assert run.passed is True
    assert run.budget["requests"] == 0
    assert run.constants["generated_requests"] == 0
    for result in run.mode_results:
        assert len(result.task_results) == len(TASKS)
        assert result.aggregate_metrics["task_count"] == len(TASKS)
        # No model, so nothing is graded and no latency is invented.
        assert result.aggregate_metrics["graded_answer_count"] == 0
        assert result.aggregate_metrics["mean_latency_ms"] is None
        assert all(record["generation"] is None for record in result.task_results)
    # Retrieval is still measured: a dry run is a real experiment except for the
    # model call, which is exactly what makes it a cost check.
    assert run.mode_result(MODE_FULL_CONTEXT).aggregate_metrics["retrieval_recall"] == 0.0


def test_generated_run_grades_answers_and_records_latency() -> None:
    # One task whose accepted answers contain the text the provider returns, so
    # the graded outcome is deterministic rather than a property of the dataset.
    tasks = [TASKS[0]]
    provider = RecordingProvider(text="MongoDB 7")

    run = run_controlled_experiment(
        _plan(), tasks, SPEC, provider, clock=_clock(), counter_name="estimate_tokens"
    )

    assert run.passed is True
    result = run.mode_result(MODE_BRAINOS)
    assert result is not None
    assert result.aggregate_metrics["graded_answer_count"] == 1
    assert result.aggregate_metrics["mean_latency_ms"] is not None
    assert result.generation_usage["requests"] == 1
    assert result.generation_usage["generated"] == 1
    assert result.generation_usage["total_tokens"] > 0
    assert result.aggregate_metrics["answer_accuracy"] == 1.0
    assert provider.request_count == 2  # both modes ran the task
    # One model, one parameter set, for every request of every mode.
    parameters = {json.dumps(seen, sort_keys=True) for seen in provider.settings_seen()}
    assert len(parameters) == 1


def test_run_requires_a_model_and_a_provider_together() -> None:
    with pytest.raises(ExperimentError):
        run_controlled_experiment(_plan(), TASKS, SPEC, None)


# --------------------------------------------------------------------------- #
# Controlled-comparison checks
# --------------------------------------------------------------------------- #


def test_constants_record_the_controls_that_held() -> None:
    provider = RecordingProvider(text="MongoDB 7")

    run = run_controlled_experiment(_plan(), TASKS, SPEC, provider, clock=_clock())
    constants = run.constants

    assert run.violations == ()
    assert constants["passed"] is True
    assert len(constants["generation_fingerprints"]) == 1
    assert constants["generation_fingerprints"][0] == SPEC.generation_settings().fingerprint()
    assert constants["reported_models"] == ["fake-model"]
    assert constants["token_counters"] == ["estimate_tokens"]
    assert constants["system_prompt_sha256"] and len(constants["system_prompt_characters"]) == 1
    assert constants["task_ids"] == [task.task_id for task in TASKS]
    assert constants["generated_requests"] == len(TASKS) * 2


def test_a_gateway_that_routes_elsewhere_is_a_violation() -> None:
    provider = RecordingProvider(text="MongoDB 7", report_model="some-other-model")

    run = run_controlled_experiment(_plan(), TASKS, SPEC, provider, clock=_clock())

    assert run.passed is False
    assert any("some-other-model" in message for message in run.violations)
    assert run.constants["passed"] is False


def test_check_constants_detects_a_system_prompt_difference() -> None:
    collected = {
        (MODE_FULL_CONTEXT, 0): {
            "records": [_record(prompt="You are an assistant.", mode=MODE_FULL_CONTEXT)],
            "scores": [],
        },
        (MODE_BRAINOS, 0): {
            "records": [_record(prompt="You are a different assistant.", mode=MODE_BRAINOS)],
            "scores": [],
        },
    }
    plan = _plan(modes=(MODE_FULL_CONTEXT, MODE_BRAINOS))

    constants, violations = check_constants(plan, None, TASKS[:1], collected)

    assert constants["passed"] is False
    assert any("system instructions" in message for message in violations)


def test_check_constants_detects_task_wording_drift() -> None:
    collected = {
        (MODE_FULL_CONTEXT, 0): {
            "records": [_record(question="What database?", mode=MODE_FULL_CONTEXT)],
            "scores": [],
        },
        (MODE_BRAINOS, 0): {
            "records": [_record(question="Which database?", mode=MODE_BRAINOS)],
            "scores": [],
        },
    }
    plan = _plan(modes=(MODE_FULL_CONTEXT, MODE_BRAINOS))

    _constants, violations = check_constants(plan, None, TASKS[:1], collected)

    assert any("Task wording differs" in message for message in violations)


def test_check_constants_detects_missing_tasks_and_extra_modes() -> None:
    collected = {
        (MODE_FULL_CONTEXT, 0): {
            "records": [_record(mode=MODE_FULL_CONTEXT), _record(mode=MODE_FULL_CONTEXT)],
            "scores": [],
        },
        (MODE_BRAINOS, 0): {
            "records": [_record(mode=MODE_BRAINOS)],  # one task short
            "scores": [],
        },
        ("rag", 0): {"records": [], "scores": []},  # planned but never ran
    }
    plan = _plan(modes=(MODE_FULL_CONTEXT, MODE_BRAINOS, "rag"))

    _constants, violations = check_constants(plan, None, TASKS[:2], collected)

    assert any("missing tasks" in message for message in violations)
    assert any("differ from the plan" in message for message in violations)


def test_check_constants_detects_mixed_token_counters() -> None:
    collected = {
        (MODE_FULL_CONTEXT, 0): {
            "records": [_record(mode=MODE_FULL_CONTEXT, counter="estimate_tokens")],
            "scores": [],
        },
        (MODE_BRAINOS, 0): {
            "records": [_record(mode=MODE_BRAINOS, counter="tiktoken:gpt-4o-mini")],
            "scores": [],
        },
    }
    plan = _plan(modes=(MODE_FULL_CONTEXT, MODE_BRAINOS))

    _constants, violations = check_constants(plan, None, TASKS[:1], collected)

    assert any("Token counters differ" in message for message in violations)


# --------------------------------------------------------------------------- #
# Cost controls
# --------------------------------------------------------------------------- #


def test_dataset_is_truncated_by_the_task_limit() -> None:
    plan = _plan(limits=RunLimits(max_tasks=1))

    run = run_controlled_experiment(plan, TASKS)

    assert run.tasks_executed == 1
    assert run.tasks_available == len(TASKS)
    assert any("truncated" in message for message in run.warnings)


def test_a_run_level_limit_stops_the_run_and_keeps_partial_results() -> None:
    plan = _plan(modes=(MODE_FULL_CONTEXT, MODE_BRAINOS), limits=RunLimits(max_requests=3))
    provider = RecordingProvider(text="MongoDB 7")

    run = run_controlled_experiment(plan, TASKS, SPEC, provider, clock=_clock())

    assert run.aborted is not None
    assert run.aborted["limit"] == "max_requests"
    assert run.passed is False
    # Task-outer ordering means the first task's two modes completed before the
    # limit stopped the second task, so partial results are still analyzable.
    assert run.budget["requests"] == 3
    # The 2-task run stopped inside the second task: Mode A finished both of its
    # records, Mode D was cut short, and both are still reported.
    assert run.mode_result(MODE_FULL_CONTEXT).complete is True
    assert run.mode_result(MODE_BRAINOS).complete is False
    assert len(run.mode_result(MODE_BRAINOS).task_results) == 1
    assert any("cost limit" in message for message in run.warnings)


def test_an_oversized_prompt_is_skipped_not_sent() -> None:
    plan = _plan(modes=(MODE_FULL_CONTEXT,), limits=RunLimits(max_input_tokens=1))
    provider = RecordingProvider(text="MongoDB 7")

    run = run_controlled_experiment(plan, TASKS, SPEC, provider, clock=_clock())

    result = run.mode_result(MODE_FULL_CONTEXT)
    assert provider.request_count == 0
    assert result.skipped_task_ids == tuple(task.task_id for task in TASKS)
    assert result.aggregate_metrics["graded_answer_count"] == 0
    assert run.budget["skips"] == len(TASKS)
    assert any("skipped" in message for message in run.warnings)


def test_stochastic_sampling_without_trials_is_flagged() -> None:
    provider = RecordingProvider(text="MongoDB 7")
    warm = ModelSpec(model="fake-model", temperature=0.7)

    run = run_controlled_experiment(_plan(), TASKS, warm, provider, clock=_clock())

    assert any("single trial" in message for message in run.warnings)


def test_a_run_without_a_dataset_hash_says_so() -> None:
    run = run_controlled_experiment(_plan(), TASKS)

    assert any("Dataset hash not recorded" in message for message in run.warnings)


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #


def test_mode_run_files_match_the_run_file_schema() -> None:
    provider = RecordingProvider(text="MongoDB 7")

    run = run_controlled_experiment(_plan(), TASKS, SPEC, provider, clock=_clock())
    payload = run.mode_result(MODE_BRAINOS).to_run_dict()

    assert payload["config"]["mode"] == MODE_BRAINOS
    assert payload["config"]["model"] == "fake-model"
    assert payload["config"]["temperature"] == 0.0
    assert payload["aggregate_metrics"]["task_count"] == len(TASKS)
    # The exported shape is the Phase 6/7 one, so the existing comparison tool
    # consumes an experiment's per-mode files without a translation step.
    comparison = compare_runs(
        [result.to_run_dict() for result in run.mode_results]
    )
    assert [row["mode"] for row in comparison] == [MODE_FULL_CONTEXT, MODE_BRAINOS]


def test_artifact_carries_provenance_and_no_credential_fields() -> None:
    provider = RecordingProvider(text="MongoDB 7")

    payload = run_controlled_experiment(
        _plan(), TASKS, SPEC, provider, clock=_clock()
    ).to_dict()
    rendered = json.dumps(payload)

    assert payload["provenance"]["model"] == "fake-model"
    assert payload["provenance"]["application_version"]
    assert payload["provenance"]["benchmark_version"] == "context_rot-v1"
    assert payload["provenance"]["context_budget"] > 0
    assert payload["experiment_version"] == "experiment-v1"
    # ``api_key_env`` is the *name* of an environment variable, which is
    # provenance, not a credential: what must never appear is a key value.
    assert payload["model"]["api_key_env"] == "OPENAI_API_KEY"
    assert '"api_key":' not in rendered
    assert "sk-" not in rendered
    assert payload["headline"][0]["tasks"] == len(TASKS)


def test_cross_model_matrix_reports_a_row_per_model_and_mode() -> None:
    provider = RecordingProvider(text="MongoDB 7")
    small = ModelSpec(name="small", model="fake-small")
    medium = ModelSpec(name="medium", model="fake-medium")

    runs = [
        run_controlled_experiment(_plan(), TASKS, small, provider, clock=_clock()),
        run_controlled_experiment(_plan(), TASKS, medium, provider, clock=_clock()),
    ]
    matrix = compare_modes_across_models(runs)

    assert [model["matrix_role"] for model in matrix["models"]] == ["small", "medium"]
    assert {(row["matrix_role"], row["mode"]) for row in matrix["rows"]} == {
        ("small", MODE_FULL_CONTEXT),
        ("small", MODE_BRAINOS),
        ("medium", MODE_FULL_CONTEXT),
        ("medium", MODE_BRAINOS),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_dry_run_writes_an_artifact_without_a_credential(tmp_path) -> None:
    output = tmp_path / "dry.json"

    code = main(
        [
            "--dry-run",
            "--modes",
            "all",
            "--dataset",
            str(DATASET),
            "--output",
            str(output),
            "--quiet",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["passed"] is True
    assert len(payload["plan"]["modes"]) == 5
    assert payload["budget"]["requests"] == 0


def test_cli_requires_a_model_without_a_dry_run(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "out.json"), "--quiet"])

    assert "--model is required" in str(error.value)


def test_cli_rejects_an_unknown_mode(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--dry-run",
                "--modes",
                "nonsense",
                "--output",
                str(tmp_path / "out.json"),
                "--quiet",
            ]
        )

    assert "Unknown baseline mode" in str(error.value)


def test_cli_writes_per_mode_run_files(tmp_path) -> None:
    runs_dir = tmp_path / "raw"

    code = main(
        [
            "--dry-run",
            "--modes",
            f"{MODE_FULL_CONTEXT},{MODE_BRAINOS}",
            "--output",
            str(tmp_path / "artifact.json"),
            "--runs-dir",
            str(runs_dir),
            "--quiet",
        ]
    )

    written = sorted(path.name for path in runs_dir.glob("*.json"))
    assert code == 0
    assert written == [f"{MODE_BRAINOS}.json", f"{MODE_FULL_CONTEXT}.json"]
    assert json.loads((runs_dir / f"{MODE_BRAINOS}.json").read_text())["config"][
        "mode"
    ] == MODE_BRAINOS


def _record(
    *,
    prompt: str = "You are an assistant.",
    question: str = "What database?",
    mode: str = MODE_FULL_CONTEXT,
    counter: str = "estimate_tokens",
) -> dict:
    """Return a minimal replay-shaped record for ``check_constants`` tests."""

    return {
        "task_id": TASKS[0].task_id,
        "mode": mode,
        "prompt_messages": (
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ),
        "stats": {"token_counter": counter},
        "generation": None,
    }
