"""Live Phase 12 validation: the taxonomy against the pinned BrainOS revision.

The unit tests build scored records by hand to pin each rule. This file runs the
same taxonomy over *replayed* conversations with the real runtime, the committed
smoke dataset, and the committed scripted-answer fixture — so the failure
distribution quoted in the phase log comes from the code that ships.

The scripted answers are deliberately not all correct: the fixture makes
``brainos`` guess on the multi-hop question whose second hop the relevance filter
drops, makes the temporal and conflict ablations answer with the superseded
value, and makes an isolated cross-session replay assert a fact its session can
no longer reach. Those are the four failures Phase 11 predicted the taxonomy
would need to separate.

Assertions are about relationships — which label owns which measured failure,
and that the taxonomy never contradicts the scorer — never about a token count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from baselines.modes import (
    ABLATION_NO_CONFLICT,
    ABLATION_NO_MEMORY,
    ABLATION_NO_RELEVANCE,
    ABLATION_NO_TEMPORAL,
    MODE_BRAINOS,
    MODE_FULL_CONTEXT,
    MODE_SLIDING_WINDOW,
)
from evaluation.datasets import load_jsonl
from evaluation.errors import collect_failures, error_report
from evaluation.modes import task_evaluator
from evaluation.run import load_answers
from evaluation.runner import EvaluationConfig, EvaluationRunner
from evaluation.scoring import score_record

pytest.importorskip("brainos_runtime")

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
ANSWERS = Path("benchmarks/fixtures/scripted_answers.jsonl")
TASKS = load_jsonl(DATASET)
ANSWER_SET = load_answers(ANSWERS)

MULTI_HOP = "cr-multi-hop-800-00"
TEMPORAL = "cr-temporal-800-00"
CONFLICT = "cr-conflict-800-00"
CROSS_SESSION = "cr-cross-session-800-00"
ABSTENTION = "cr-abstention-800-00"


def _run(mode: str, *, session_isolation: bool = False) -> dict:
    """Score one mode's replay of the whole smoke tier against the fixture."""

    isolation = session_isolation
    config = EvaluationConfig(
        mode=mode,
        dataset_sha256="phase12-test",
        session_isolation=isolation,
    )
    run = EvaluationRunner(config).run(
        TASKS,
        evaluator=task_evaluator(session_isolation=isolation),
        scorer=lambda task, record, answer: score_record(
            task, record, answer=answer, session_isolation=isolation
        ),
        answers=ANSWER_SET.answers,
    )
    return run.to_dict()


def _failures(mode: str, *, session_isolation: bool = False) -> dict[str, dict]:
    """Classify one mode's replay, enriched with the benchmark's fact ledger."""

    return {
        record["task_id"]: record
        for record in collect_failures(
            _run(mode, session_isolation=session_isolation), tasks=TASKS
        )
    }


def _assert_compression_is_always_reported(records: list[dict]) -> None:
    for record in records:
        if record["absent_from_prompt_fact_ids"]:
            assert "over_compression" in record["labels"], record["task_id"]
            assert record["evidence_lost"] is True, record["task_id"]


def test_live_brainos_multi_hop_failure_is_the_relevance_filter() -> None:
    """The Phase 7/11 finding, now attributed: selected, then judged irrelevant."""

    record = _failures(MODE_BRAINOS)[MULTI_HOP]

    assert record["observed_failure"] is True
    assert record["failure_type"] == "irrelevant_memory"
    assert record["contributors"] == ["over_compression"]
    assert record["verdict"] == "incorrect"
    assert record["retrieval"]["evidence_in_prompt"] is False
    # The evidence trail proves *which* hop was lost and how.
    detail = {item["fact_id"]: item for item in record["expected_memory_detail"]}
    lost = [item for item in detail.values() if not item["in_prompt"]]
    assert lost, record["expected_memory_detail"]
    assert lost[0]["dropped_reason"] == "low_relevance"
    assert lost[0]["blame"] == "irrelevant_memory"
    assert any(entry["reason"] == "low_relevance" for entry in record["dropped_evidence"])


def test_live_removing_the_relevance_filter_clears_the_error() -> None:
    """D2's repair, measured on the taxonomy rather than only on recall."""

    full = _failures(MODE_BRAINOS)
    repaired = _failures(ABLATION_NO_RELEVANCE)

    assert "over_compression" in full[MULTI_HOP]["labels"]
    assert MULTI_HOP not in repaired, repaired[MULTI_HOP]
    assert not any(
        record["evidence_lost"] for record in repaired.values()
    ), "the ablation keeps every required fact in the prompt"


def test_live_a_window_that_never_reaches_the_fact_is_a_recall_failure() -> None:
    records = _failures(MODE_SLIDING_WINDOW)

    answerable = [
        record for record in records.values() if record["expected_memory"]
    ]
    assert answerable
    assert {record["failure_type"] for record in answerable} == {"missed_memory"}
    assert {record["stage"] for record in answerable} == {"recall"}
    _assert_compression_is_always_reported(list(records.values()))


def test_live_no_memory_at_all_is_a_recall_failure_with_a_wrong_guess() -> None:
    records = _failures(ABLATION_NO_MEMORY)

    assert records[MULTI_HOP]["failure_type"] == "missed_memory"
    assert records[MULTI_HOP]["observed_failure"] is True
    assert "over_compression" in records[MULTI_HOP]["contributors"]
    _assert_compression_is_always_reported(list(records.values()))


def test_live_keeping_the_superseded_value_is_harmful_retention() -> None:
    records = _failures(MODE_BRAINOS)

    for task_id in (TEMPORAL, CONFLICT):
        record = records[task_id]
        assert record["failure_type"] == "under_compression", task_id
        assert record["stage"] == "prompt"
        assert record["latent"] is True  # the scripted answer is the current value
        assert record["retrieval"]["prompt_contains_forbidden"] is True
        assert record["retrieval"]["prompt_forbidden_fact_ids"], task_id


def test_live_losing_temporal_signals_reintroduces_staleness() -> None:
    records = _failures(ABLATION_NO_TEMPORAL)

    record = records[TEMPORAL]

    assert record["failure_type"] == "stale_memory"
    assert record["verdict"] == "stale_answer"
    assert record["observed_failure"] is True
    assert "under_compression" in record["contributors"]


def test_live_losing_conflict_handling_fails_the_conflict_category() -> None:
    records = _failures(ABLATION_NO_CONFLICT)

    record = records[CONFLICT]

    assert record["failure_type"] == "conflicting_memory"
    assert record["verdict"] == "stale_answer"
    assert record["observed_failure"] is True


def test_live_an_isolated_session_asserting_a_fact_is_a_hallucination() -> None:
    """With session isolation the fact is unreachable, so asserting it cannot be a retrieval win."""

    record = _failures(MODE_BRAINOS, session_isolation=True)[CROSS_SESSION]

    assert record["failure_type"] == "hallucination"
    assert record["observed_failure"] is True
    assert record["retrieval"]["evidence_in_prompt"] is False
    assert "missed_memory" in record["contributors"]


def test_live_asserting_where_abstention_was_expected_is_a_hallucination() -> None:
    record = _failures(MODE_BRAINOS)[ABSTENTION]

    assert record["failure_type"] == "hallucination"
    assert record["abstention_expected"] is True


def test_live_full_context_owns_its_own_errors() -> None:
    """Mode A retrieves nothing, so its failures are never blamed on a retriever."""

    records = _failures(MODE_FULL_CONTEXT)

    for record in records.values():
        assert "missed_memory" not in record["labels"], record["task_id"]
        assert record["retrieval"]["evidence_in_prompt"] is True or record["mode"] == (
            MODE_FULL_CONTEXT
        )
        assert record["failure_type"] in {"under_compression", "wrong_memory", "hallucination"}


def test_live_the_taxonomy_never_contradicts_the_scorer() -> None:
    artifacts = [
        _run(mode)
        for mode in (
            MODE_FULL_CONTEXT,
            MODE_SLIDING_WINDOW,
            MODE_BRAINOS,
            ABLATION_NO_MEMORY,
            ABLATION_NO_RELEVANCE,
            ABLATION_NO_TEMPORAL,
            ABLATION_NO_CONFLICT,
        )
    ]

    report = error_report(artifacts, tasks=TASKS)

    assert report["labels_vs_scorer"]["unexpected"] == 0, report["labels_vs_scorer"]
    assert report["labels_vs_scorer"]["agree"] > 0
    assert report["labels_vs_scorer"]["refined"] > 0
    assert report["graded_record_count"] == len(TASKS) * len(artifacts)
    assert report["failure_record_count"] > 0
    # Every defect names a stage from the fixed vocabulary.
    assert set(report["stage_counts"]) <= {"recall", "selection", "prompt", "generation"}
    for metrics in report["by_mode"].values():
        assert metrics["defect_count"] >= metrics["observed_failure_count"]


def test_live_the_report_shows_where_the_failures_concentrate() -> None:
    artifacts = [_run(MODE_BRAINOS), _run(ABLATION_NO_MEMORY), _run(MODE_SLIDING_WINDOW)]

    report = error_report(artifacts, tasks=TASKS, examples_per_type=1, top=0)
    rows = {(row["mode"], row["category"]): row for row in report["concentration"]}

    assert rows[(MODE_BRAINOS, "multi_hop")]["failure_type"] == "irrelevant_memory"
    assert rows[(ABLATION_NO_MEMORY, "multi_hop")]["failure_type"] == "missed_memory"
    assert rows[(MODE_SLIDING_WINDOW, "cross_session")]["failure_type"] == "missed_memory"
    assert rows[(MODE_BRAINOS, "temporal")]["failure_type"] == "under_compression"
    assert report["examples"], "the report embeds at least one example per label"
    # The taxonomy travels with the report so the labels stay readable.
    assert report["taxonomy"]["labels"]
    assert report["taxonomy"]["drop_reason_labels"]["low_relevance"] == "irrelevant_memory"
