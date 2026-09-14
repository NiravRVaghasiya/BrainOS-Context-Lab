"""Live Phase 7 validation: the generated benchmark against the pinned BrainOS.

The unit tests check the dataset and the scoring rules. This file runs the
*committed* benchmark through the real runtime and asserts the relationships the
benchmark exists to expose — which mode keeps a fact, which one loses it, whether
an earlier session leaks into a later one. It is the Phase 7 analogue of
``test_baseline_modes_live.py``.

Rules this file enforces on itself (same as the Phase 6 live file):

* no provider API key: the scored run grades scripted answers, so the whole
  pipeline — replay, context construction, scoring, aggregation — is exercised
  without a model;
* assertions are about relationships, never about a specific token count, which
  would be a property of one synthetic dataset rather than a result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.datasets import dataset_issues, load_jsonl
from evaluation.modes import compare_modes, replay_task, task_evaluator
from evaluation.runner import EvaluationConfig, EvaluationRunner
from evaluation.scoring import (
    expects_abstention,
    score_record,
    score_retrieval,
)

pytest.importorskip("brainos_runtime")

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)
BY_CATEGORY = {task.category: task for task in TASKS}


def _score(task, mode: str, *, session_isolation: bool = False):
    """Replay one task in one mode and score its retrieval half."""

    replay = replay_task(task, mode, session_isolation=session_isolation)
    score = score_retrieval(
        task,
        retrieved_texts=[*replay.retrieved_memory_texts, *replay.retrieved_chunk_texts],
        prompt_messages=replay.prompt_messages,
        mode=mode,
    )
    return replay, score


def test_live_the_committed_benchmark_is_valid_and_complete() -> None:
    assert dataset_issues(TASKS) == []
    assert {task.category for task in TASKS} == {
        "single_hop",
        "multi_hop",
        "temporal",
        "conflict",
        "distractor",
        "cross_session",
        "abstention",
    }


def test_live_every_task_runs_through_every_mode() -> None:
    task = BY_CATEGORY["single_hop"]

    replays = compare_modes(task)

    assert len(replays) == 5
    for replay in replays:
        assert replay.prompt_messages
        assert replay.final_context_tokens > 0
        # BrainOS observes in every mode, including the memory-free controls.
        assert replay.stored_memory_count > 0


def test_live_the_window_loses_the_fact_that_brainos_keeps() -> None:
    """The comparison the benchmark exists to make, on generated data."""

    task = BY_CATEGORY["single_hop"]

    windowed, windowed_score = _score(task, "sliding_window")
    managed, managed_score = _score(task, "brainos")
    full, full_score = _score(task, "full_context")

    assert full_score.evidence_in_prompt is True  # everything is replayed
    assert windowed_score.evidence_in_prompt is False, (
        "the fact is older than the window and Mode B has nothing to retrieve it"
    )
    assert managed_score.evidence_in_prompt is True, (
        "BrainOS memory must still carry the fact the window dropped"
    )
    assert managed.context_reduction > 0
    assert windowed.context_reduction > 0


def test_live_the_full_context_reference_price_is_the_ceiling() -> None:
    task = BY_CATEGORY["distractor"]

    full, _ = _score(task, "full_context")
    managed, _ = _score(task, "brainos")

    assert full.context_reduction == 0.0
    assert full.final_context_tokens == full.full_context_reference_tokens
    assert managed.final_context_tokens < full.full_context_reference_tokens


def test_live_both_hops_are_retrievable() -> None:
    """A multi-hop failure must be a budget failure, not a missing fact."""

    task = BY_CATEGORY["multi_hop"]

    managed, score = _score(task, "brainos")

    assert len(task.evidence.required) == 2
    assert score.recall == 1.0, "both required facts must be recalled by the runtime"
    assert managed.selected_memory_count > 0


def test_live_the_current_value_is_what_reaches_a_brainos_prompt() -> None:
    """Temporal/conflict tasks: the newer value is present, not merely recalled."""

    for category in ("temporal", "conflict"):
        task = BY_CATEGORY[category]
        _, score = _score(task, "brainos")

        assert score.expected_answer_in_prompt is True, category
        assert score.missing_fact_ids == (), category


def test_live_the_abstention_task_has_no_evidence_in_any_prompt() -> None:
    """Nothing in the conversation answers the question, in any mode."""

    task = BY_CATEGORY["abstention"]
    assert expects_abstention(task) is True

    for replay in compare_modes(task):
        score = score_retrieval(
            task,
            retrieved_texts=[
                *replay.retrieved_memory_texts,
                *replay.retrieved_chunk_texts,
            ],
            prompt_messages=replay.prompt_messages,
            mode=replay.mode,
        )
        assert score.evidence_in_prompt is False, replay.mode
        assert score.required_fact_ids == (), replay.mode


def test_live_session_isolation_discards_the_earlier_session() -> None:
    """The product invariant, measured on the cross-session category."""

    task = BY_CATEGORY["cross_session"]

    single, single_score = _score(task, "brainos")
    isolated, isolated_score = _score(task, "brainos", session_isolation=True)

    assert single.sessions_used == 1
    assert single_score.evidence_in_prompt is True

    assert isolated.sessions_used == 2
    assert isolated_score.evidence_in_prompt is False, (
        "a fact from a closed session must not reach a new one"
    )
    assert expects_abstention(task, session_isolation=True) is True


def _scripted_answers(tasks) -> dict[tuple[str, str], str]:
    """Mock-model answers: every task answered as its contract prefers."""

    answers: dict[tuple[str, str], str] = {}
    for task in tasks:
        if task.evidence.abstention_expected:
            answers[(task.task_id, "brainos")] = "I don't know; that was not mentioned."
        else:
            answers[(task.task_id, "brainos")] = task.answers()[0]
    return answers


def test_live_a_scored_run_produces_aggregates_without_a_provider() -> None:
    """Phase 7's end-to-end path: dataset → replay → score → aggregate."""

    config = EvaluationConfig(
        mode="brainos",
        dataset_sha256="test",
        dataset_notes={"dataset": str(DATASET), "task_count": len(TASKS)},
    )
    run = EvaluationRunner(config).run(
        TASKS,
        evaluator=task_evaluator(),
        scorer=lambda task, record, answer: score_record(task, record, answer=answer),
        answers=_scripted_answers(TASKS),
    )

    metrics = run.aggregate_metrics
    assert run.dataset_issues == []
    assert metrics["task_count"] == len(TASKS)
    assert metrics["graded_answer_count"] == len(TASKS)
    assert metrics["answer_accuracy"] == 1.0, metrics["verdict_counts"]
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["stale_answer_rate"] == 0.0
    # Retrieval ran without a key, and every record carries its scoring block.
    assert metrics["retrieval_recall"] > 0
    assert metrics["mean_context_reduction"] > 0
    assert metrics["mean_token_savings"] > 0
    assert "faithfulness" in metrics
    assert "quality_per_token" in metrics
    assert metrics["degradation"]["point_count"] == 1
    assert metrics["degradation"]["area_under_degradation_curve"] is None
    for result in run.task_results:
        assert "scores" in result
        assert result["scores"]["answer"]["verdict"] in {"correct", "ungraded"}
    # No credential can appear: the run was never given one.
    assert "api_key" not in run.to_dict()["config"]


def test_live_a_stale_answer_is_penalised_by_the_scorer() -> None:
    """The scoring half must be able to fail a mode, not only pass it."""

    task = BY_CATEGORY["temporal"]
    stale_value = next(
        fact.value for fact in task.planted_facts() if fact.fact_id in task.evidence.forbidden
    )
    record = replay_task(task, "brainos").to_dict()

    scored = score_record(task, record, answer=stale_value)

    assert scored["answer"]["verdict"] == "stale_answer"
    assert scored["error_type"] in {"stale_memory", "conflicting_memory"}
