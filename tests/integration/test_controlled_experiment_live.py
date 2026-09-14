"""Live Phase 9 validation: the controlled comparison against the pinned BrainOS.

The unit tests drive the experiment with synthetic records. This file runs the
*committed* benchmark through the real runtime, all five modes, and a
deterministic prompt-reading provider, and checks the properties the phase
exists to establish:

* every mode ran every task with the same model, one parameter set, one system
  prompt, and one dataset revision;
* latency and token usage are recorded, not estimated after the fact;
* the evidence a mode retrieved is what its answer was made of — a mode whose
  prompt carries no evidence cannot produce an evidence-based answer;
* no credential appears anywhere in the artifact.

Rules this file enforces on itself, following the Phase 6/7 live files:

* **no API key**: generation goes through ``PromptReadingProvider``, a
  deterministic reader that answers only from the prompt and never calls a
  network;
* **no research claim**: the reader is not a model, so its "accuracy" measures
  the pipeline (does evidence reach the prompt, does a graded answer come back),
  not LLM quality. Assertions are relationships between modes, never numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.modes import (
    MODE_BRAINOS,
    MODE_FULL_CONTEXT,
    MODE_ORDER,
    MODE_SLIDING_WINDOW,
)
from evaluation.analysis import compare_runs
from evaluation.datasets import load_jsonl
from evaluation.experiment import ExperimentPlan, run_controlled_experiment
from evaluation.generation import ModelSpec
from evaluation.limits import RunLimits
from tests.fakes import PromptReadingProvider

pytest.importorskip("brainos_runtime")

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)
MODEL = ModelSpec(name="prompt-reader", provider="openai", model="deterministic-reader")


def _clock() -> float:
    state = {"t": 0.0}

    def tick() -> float:
        state["t"] += 0.05
        return state["t"]

    return tick


@pytest.fixture(scope="module")
def run():  # type: ignore[no-untyped-def]
    plan = ExperimentPlan(
        name="live-controlled",
        modes=MODE_ORDER,
        trials=1,
        limits=RunLimits(max_requests=1000),
        dataset=str(DATASET),
        dataset_sha256="",
    )
    provider = PromptReadingProvider(report_model="deterministic-reader")
    experiment = run_controlled_experiment(
        plan, TASKS, MODEL, provider, clock=_clock()
    )
    return experiment, provider


def test_live_the_comparison_is_controlled(run) -> None:  # type: ignore[no-untyped-def]
    experiment, provider = run

    assert experiment.violations == ()
    assert experiment.passed is True
    constants = experiment.constants
    assert len(constants["generation_fingerprints"]) == 1
    assert constants["reported_models"] == ["deterministic-reader"]
    assert constants["task_ids"] == [task.task_id for task in TASKS]
    assert len(constants["system_prompt_characters"]) == 1
    # One request per (task, mode): the same five modes, the same tasks.
    assert provider.request_count == len(TASKS) * len(MODE_ORDER)
    assert all(
        seen["temperature"] == 0.0 for seen in provider.settings_seen()
    )


def test_live_every_mode_ran_every_task_with_latency_and_usage(run) -> None:  # type: ignore[no-untyped-def]
    experiment, _provider = run

    for mode in MODE_ORDER:
        result = experiment.mode_result(mode)
        assert result is not None, mode
        assert result.complete is True
        assert result.aggregate_metrics["task_count"] == len(TASKS)
        assert result.aggregate_metrics["graded_answer_count"] == len(TASKS)
        assert result.aggregate_metrics["mean_latency_ms"] is not None
        assert result.generation_usage["generated"] == len(TASKS)
        assert all(
            record["generation"]["latency_ms"] > 0 for record in result.task_results
        )
        assert all(record["answer"] for record in result.task_results)


def test_live_evidence_is_what_the_answers_are_made_of(run) -> None:  # type: ignore[no-untyped-def]
    experiment, _provider = run

    window = experiment.mode_result(MODE_SLIDING_WINDOW)
    brainos = experiment.mode_result(MODE_BRAINOS)
    full = experiment.mode_result(MODE_FULL_CONTEXT)

    # The sliding window keeps no evidence at all, so a prompt-only reader can
    # only decline; BrainOS's prompt carries the fact. This is the mechanism the
    # benchmark measures, not an accuracy claim about any model.
    assert window.aggregate_metrics["evidence_in_prompt_rate"] == 0.0
    assert brainos.aggregate_metrics["evidence_in_prompt_rate"] > 0.0
    assert (
        brainos.aggregate_metrics["answer_accuracy"]
        >= window.aggregate_metrics["answer_accuracy"]
    )
    # Full context carries the transcript but retrieves nothing: the retrieval
    # half and the prompt half stay separate numbers.
    assert full.aggregate_metrics["retrieval_recall"] == 0.0
    assert full.aggregate_metrics["evidence_in_prompt_rate"] == 1.0
    # The prompt-only reader reaches at least one correct, graded answer, which
    # is what proves the generation → scoring round trip works end to end.
    assert any(score["answer"]["verdict"] == "correct" for score in brainos.scores)


def test_live_artifact_has_no_credential_and_feeds_the_comparison_tool(run) -> None:  # type: ignore[no-untyped-def]
    experiment, _provider = run

    rendered = json.dumps(experiment.to_dict())
    assert "sk-" not in rendered
    assert '"api_key":' not in rendered
    assert experiment.to_dict()["provenance"]["brainos_version"] not in {"", "unvalidated"}

    comparison = compare_runs(
        [result.to_run_dict() for result in experiment.mode_results]
    )
    assert [row["mode"] for row in comparison] == list(MODE_ORDER)
    assert all(row["headline"]["mean_final_context_tokens"] for row in comparison)
