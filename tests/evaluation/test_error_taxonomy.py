"""Phase 12: the error taxonomy, the failure records, and the report.

Two properties matter more than any single label here, and both are asserted
directly rather than implied:

* **The taxonomy never contradicts the scorer.** Where a label was already
  decided (`stale_answer`, `wrong_abstention`, the answer-side consequences of an
  ``incorrect`` verdict), the classification reproduces it; it only refines the
  two labels the scorer could not split. ``labels_vs_scorer.unexpected`` is the
  count of disagreements and must stay zero.
* **Losing required evidence is always reported as compression.** Every record
  whose prompt failed to carry a required fact lists ``over_compression`` among
  its labels, whichever stage lost it. Without that, Phase 11's finding — the
  filter dropping needed evidence — would vanish from the report whenever a
  reason-specific label took the primary slot.

The synthetic records below are built by hand so each rule is exercised in
isolation; ``tests/integration/test_error_analysis_live.py`` runs the same
taxonomy over real replayed conversations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import errors
from evaluation.errors import (
    DROP_REASON_LABELS,
    ERRORS_VERSION,
    PLAN_ERROR_TYPES,
    classify_failure,
    collect_failures,
    error_report,
    failure_record,
    label_for_drop_reason,
    mode_has_selection_stage,
)
from evaluation.scoring import score_record
from tests.fakes import FakeLLMProvider  # noqa: F401  (import-time dependency check)

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
ANSWERS = Path("benchmarks/fixtures/scripted_answers.jsonl")


def _retrieval(**overrides: object) -> dict:
    base = {
        "task_id": "task-1",
        "category": "single_hop",
        "mode": "brainos",
        "required_fact_ids": ["fact-1"],
        "supporting_fact_ids": [],
        "retrieved_fact_ids": ["fact-1"],
        "missing_fact_ids": [],
        "forbidden_retrieved": [],
        "prompt_fact_ids": ["fact-1"],
        "prompt_forbidden_fact_ids": [],
        "absent_from_prompt_fact_ids": [],
        "lost_after_selection_fact_ids": [],
        "unneeded_prompt_fact_ids": [],
        "recall": 1.0,
        "precision": 1.0,
        "evidence_in_prompt": True,
        "expected_answer_in_prompt": True,
        "prompt_contains_forbidden": False,
    }
    base.update(overrides)
    return base


def _score(
    *,
    task_id: str = "task-1",
    mode: str = "brainos",
    category: str = "single_hop",
    verdict: str = "correct",
    error_type: str = "",
    retrieval: dict | None = None,
    answer: dict | None = None,
    audit: dict | None = None,
    faithfulness: float | None = None,
) -> dict:
    retrieval_block = retrieval if retrieval is not None else _retrieval(task_id=task_id)
    answer_block = {
        "task_id": task_id,
        "category": category,
        "mode": mode,
        "verdict": verdict,
        "expected_answer": "PostgreSQL 16",
        "answer": "PostgreSQL 16" if verdict == "correct" else "",
        "abstention_expected": False,
        "matched_answer": "",
        "stale_answer": "",
        "error_type": error_type,
    }
    if answer:
        answer_block.update(answer)
    return {
        "task_id": task_id,
        "category": category,
        "mode": mode,
        "retrieval": retrieval_block,
        "answer": answer_block,
        "error_type": error_type or answer_block["error_type"],
        "context_reduction": 0.5,
        "final_context_tokens": 100,
        "full_context_reference_tokens": 200,
        "token_savings": 100,
        "quality_adjusted_efficiency": None,
        "faithfulness": faithfulness,
        "conflict_task": category == "conflict",
        "conversation_length": 80,
        "length_tier": 800,
        "latency_ms": None,
        "token_counter": "estimate_tokens",
        "retrieval_audit": audit
        or {"available": False, "candidate_count": 0, "selected_count": 0,
            "reason_counts": {}, "conflicts": [], "dropped": []},
    }


def _audit(*drops: tuple[str, str], available: bool = True) -> dict:
    """Build a retrieval audit from ``(reason, text)`` pairs."""

    dropped = [
        {"memory_id": f"m{index}", "reason": reason, "detail": "", "score": 0.1, "text": text}
        for index, (reason, text) in enumerate(drops)
    ]
    counts: dict[str, int] = {}
    for item in dropped:
        counts[item["reason"]] = counts.get(item["reason"], 0) + 1
    return {
        "available": available,
        "candidate_count": len(dropped) + 1,
        "selected_count": 1,
        "reason_counts": counts,
        "conflicts": [],
        "dropped": dropped,
    }


def _run_file(mode: str, scores: list[dict], *, task_results: list[dict] | None = None) -> dict:
    """Build the artifact shape ``evaluation.run`` writes."""

    results = []
    for score in scores:
        record = {"task_id": score["task_id"], "category": score["category"], "mode": mode}
        record["scores"] = score
        results.append(record)
    if task_results is not None:
        results = [
            {**record, **extra}
            for record, extra in zip(results, task_results, strict=False)
        ]
    return {
        "run_id": "run-1",
        "timestamp": "2026-09-14T00:00:00+00:00",
        "config": {"mode": mode, "benchmark": "context_rot", "model": "test-model"},
        "task_results": results,
        "aggregate_metrics": {},
        "dataset_issues": [],
    }


# --------------------------------------------------------------------------- #
# The reason table
# --------------------------------------------------------------------------- #


def test_every_documented_drop_reason_maps_to_a_plan_label_or_to_nothing() -> None:
    documented = {
        "low_relevance": "irrelevant_memory",
        "weak_relevance": "irrelevant_memory",
        "stale": "stale_memory",
        "expired": "stale_memory",
        "superseded": "conflicting_memory",
        "cap": "over_compression",
        "budget": "over_compression",
        "memory_budget": "over_compression",
        "chunk_budget": "over_compression",
        "chunk_ceiling": "over_compression",
        "duplicate": "",
        "duplicate_history": "",
        "empty": "",
        "suspicious": "",
    }

    assert documented == DROP_REASON_LABELS
    for reason, label in documented.items():
        assert label_for_drop_reason(reason) == label
        if label:
            assert label in PLAN_ERROR_TYPES
    # An unknown reason is not guessed at: a fabricated label is worse than none.
    assert label_for_drop_reason("invented_reason") == ""
    assert label_for_drop_reason(None) == ""


def test_benign_and_guarded_reasons_are_named_not_labeled() -> None:
    for reason in errors.BENIGN_DROP_REASONS:
        assert DROP_REASON_LABELS[reason] == ""
    for reason in errors.GUARDED_DROP_REASONS:
        assert DROP_REASON_LABELS[reason] == ""
    for reason in errors.VOLUME_DROP_REASONS + errors.RELEVANCE_DROP_REASONS:
        assert DROP_REASON_LABELS[reason] in PLAN_ERROR_TYPES


# --------------------------------------------------------------------------- #
# Stage detection
# --------------------------------------------------------------------------- #


def test_only_a_wholesale_replay_mode_has_no_selection_stage() -> None:
    assert mode_has_selection_stage("full_context") is False
    for mode in ("sliding_window", "rag", "brainos", "brainos_rag", "brainos_no_memory"):
        assert mode_has_selection_stage(mode) is True, mode
    # A legacy alias resolves; an unknown selector is treated as selecting,
    # because claiming over_compression asserts a truncation nobody observed.
    assert mode_has_selection_stage("no_memory") is True
    assert mode_has_selection_stage("some_future_mode") is True


# --------------------------------------------------------------------------- #
# One label at a time
# --------------------------------------------------------------------------- #


def test_a_correct_record_with_nothing_wrong_produces_no_failure() -> None:
    assert classify_failure(_score()) is None


def test_a_required_fact_never_selected_is_a_recall_failure() -> None:
    score = _score(
        verdict="incorrect",
        error_type="missed_memory",
        retrieval=_retrieval(
            retrieved_fact_ids=[],
            missing_fact_ids=["fact-1"],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
        ),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "missed_memory"
    assert classification.stage == errors.STAGE_RECALL
    assert classification.observed_failure is True
    assert classification.evidence_lost is True  # the compression event is reported too
    assert "required fact fact-1 was never selected" in " ".join(classification.notes)


def test_a_selected_fact_the_prompt_lost_is_a_relevance_failure() -> None:
    score = _score(
        error_type="missed_memory",
        verdict="incorrect",
        retrieval=_retrieval(
            retrieved_fact_ids=["fact-1", "fact-2"],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
            lost_after_selection_fact_ids=["fact-1"],
        ),
        audit=_audit(("low_relevance", "a memory that was dropped")),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "irrelevant_memory"
    assert classification.stage == errors.STAGE_SELECTION
    assert classification.contributors == ("over_compression",)


def test_a_budget_drop_is_over_compression() -> None:
    score = _score(
        verdict="incorrect",
        error_type="missed_memory",
        retrieval=_retrieval(
            retrieved_fact_ids=["fact-1"],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
            lost_after_selection_fact_ids=["fact-1"],
        ),
        audit=_audit(("cap", "the memory that carried the fact")),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "over_compression"
    assert "cap" in classification.notes[0]


def test_a_wholesale_replay_cannot_blame_a_retriever() -> None:
    score = _score(
        mode="full_context",
        verdict="incorrect",
        retrieval=_retrieval(
            mode="full_context",
            retrieved_fact_ids=[],
            missing_fact_ids=["fact-1"],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
        ),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "over_compression"
    assert "full-transcript prompt" in " ".join(classification.notes)


def test_a_fact_dropped_by_the_audit_is_attributed_by_its_markers() -> None:
    tasks = {task.task_id: task for task in _tasks()}
    task = tasks["cr-multi-hop-800-00"]
    fact_two = task.facts_by_id()["fact-2"]
    score = _score(
        category=task.category,
        verdict="incorrect",
        error_type="missed_memory",
        retrieval=_retrieval(
            category=task.category,
            required_fact_ids=list(task.evidence.required),
            retrieved_fact_ids=list(task.evidence.required),
            prompt_fact_ids=[task.evidence.required[0]],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=[task.evidence.required[1]],
        ),
        audit=_audit(("low_relevance", fact_two.text)),
    )

    classification = classify_failure(score, task=task)

    assert classification is not None
    assert classification.fact_blame == {"fact-2": "irrelevant_memory"}


def test_keeping_a_forbidden_value_is_harmful_retention() -> None:
    score = _score(
        category="conflict",
        retrieval=_retrieval(
            category="conflict",
            required_fact_ids=["fact-2"],
            prompt_fact_ids=["fact-1", "fact-2"],
            prompt_forbidden_fact_ids=["fact-1"],
            prompt_contains_forbidden=True,
            unneeded_prompt_fact_ids=["fact-1"],
        ),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "under_compression"
    assert classification.stage == errors.STAGE_PROMPT
    assert classification.latent is True  # the answer was right; the prompt was not
    assert classification.answer_correct is True


def test_excess_context_alone_is_reported_not_labeled() -> None:
    """A wrong answer over an over-full prompt is the model's failure first."""

    score = _score(
        verdict="incorrect",
        error_type="hallucination",
        retrieval=_retrieval(
            required_fact_ids=["fact-1", "fact-2"],
            retrieved_fact_ids=["fact-1", "fact-2", "fact-9"],
            prompt_fact_ids=["fact-1", "fact-2", "fact-9"],
            unneeded_prompt_fact_ids=["fact-9"],
            supporting_fact_ids=[],
        ),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "wrong_memory"
    assert "under_compression" not in classification.labels
    assert "fact-9" in " ".join(classification.notes)


def test_wasteful_retention_is_noted_when_another_label_owns_the_failure() -> None:
    score = _score(
        verdict="incorrect",
        error_type="missed_memory",
        retrieval=_retrieval(
            required_fact_ids=["fact-1", "fact-2"],
            retrieved_fact_ids=["fact-1", "fact-9"],
            prompt_fact_ids=["fact-9"],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1", "fact-2"],
            missing_fact_ids=["fact-2"],
            unneeded_prompt_fact_ids=["fact-9"],
        ),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "missed_memory"
    notes = " ".join(classification.notes)
    assert "fact-9" in notes
    assert "not attributed as the failure" in notes


def test_a_clean_prompt_and_a_wrong_value_stays_a_hallucination() -> None:
    score = _score(verdict="incorrect", error_type="hallucination")

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "hallucination"
    assert classification.contributors == ()


def test_a_stale_answer_on_a_conflict_task_is_a_conflict_failure() -> None:
    score = _score(
        category="conflict",
        verdict="stale_answer",
        error_type="conflicting_memory",
        answer={"stale_answer": "Wednesday at 09:30 UTC"},
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "conflicting_memory"
    assert classification.stage == errors.STAGE_GENERATION


def test_a_stale_answer_on_a_temporal_task_is_a_staleness_failure() -> None:
    """The scorer's own split: only the conflict category means a correction."""

    score = _score(
        category="temporal",
        verdict="stale_answer",
        error_type="stale_memory",
        answer={"stale_answer": "Couchbase 7"},
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "stale_memory"
    assert classification.failure_type == score["error_type"]


def test_declining_with_the_evidence_present_is_wrong_abstention() -> None:
    score = _score(
        verdict="abstained",
        error_type="missed_memory",
        answer={"answer": "I don't know."},
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "wrong_abstention"
    assert classification.observed_failure is True


def test_asserting_where_abstention_was_expected_is_a_hallucination() -> None:
    score = _score(
        category="abstention",
        verdict="incorrect",
        error_type="hallucination",
        retrieval=_retrieval(
            category="abstention",
            required_fact_ids=[],
            retrieved_fact_ids=[],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
        ),
        answer={"abstention_expected": True, "answer": "Priya Raman."},
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "hallucination"
    assert classification.graded is True
    assert classification.answer_correct is False


def test_an_ungraded_record_still_reports_its_context_defects() -> None:
    score = _score(
        verdict="ungraded",
        error_type="ungraded",
        retrieval=_retrieval(
            retrieved_fact_ids=["fact-1"],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
            lost_after_selection_fact_ids=["fact-1"],
        ),
        audit=_audit(("low_relevance", "dropped")),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.graded is False
    assert classification.answer_correct is None
    assert classification.latent is True
    assert classification.observed_failure is False


def test_the_primary_label_follows_the_documented_attribution_order() -> None:
    score = _score(
        category="conflict",
        verdict="stale_answer",
        error_type="conflicting_memory",
        answer={"stale_answer": "Wednesday at 09:30 UTC"},
        retrieval=_retrieval(
            category="conflict",
            required_fact_ids=["fact-2"],
            prompt_fact_ids=["fact-1", "fact-2"],
            prompt_forbidden_fact_ids=["fact-1"],
            prompt_contains_forbidden=True,
        ),
    )

    classification = classify_failure(score)

    assert classification is not None
    assert classification.failure_type == "conflicting_memory"
    assert classification.contributors == ("under_compression",)
    assert classification.labels == ("conflicting_memory", "under_compression")


def test_the_order_is_a_permutation_of_the_plan_vocabulary() -> None:
    assert sorted(errors.ATTRIBUTION_ORDER) == sorted(PLAN_ERROR_TYPES)
    assert set(errors.ERROR_STAGE) == set(PLAN_ERROR_TYPES)
    assert set(errors.ERROR_DEFINITION) == set(PLAN_ERROR_TYPES)


# --------------------------------------------------------------------------- #
# Records and reports
# --------------------------------------------------------------------------- #


def _tasks():
    from evaluation.datasets import load_jsonl

    return load_jsonl(DATASET)


def test_the_failure_record_carries_the_plans_fields_and_the_four_contexts() -> None:
    score = _score(verdict="incorrect", error_type="hallucination")
    replay = {
        "question": "What database does Project Atlas use?",
        "retrieved_memory_texts": ["For Project Atlas, the production database is PostgreSQL 16."],
        "retrieved_chunk_texts": [],
    }

    record = failure_record(score, replay=replay)

    assert record is not None
    # The plan's §18 shape.
    for key in (
        "task_id",
        "mode",
        "conversation_length",
        "expected_memory",
        "retrieved_memories",
        "answer",
        "failure_type",
    ):
        assert key in record, key
    assert record["failure_type"] == "hallucination"
    # The four metric contexts an attribution needs.
    assert record["retrieval"]["evidence_in_prompt"] is True
    assert record["verdict"] == "incorrect"
    assert record["grounding"]["faithfulness"] is None
    assert record["context"]["final_context_tokens"] == 100
    assert record["question"] == replay["question"]


def test_expected_memory_is_enriched_from_the_ledger() -> None:
    task = _tasks()[0]
    score = _score(category=task.category, retrieval=_retrieval(category=task.category))

    record = failure_record(
        score,
        task=task,
        classification=errors.FailureClassification("under_compression", "prompt"),
    )

    assert record is not None
    assert "Project Atlas" in record["expected_memory"]
    assert record["expected_memory_detail"][0]["fact_id"] == "fact-1"
    assert record["expected_memory_detail"][0]["in_prompt"] is True


def test_dropped_evidence_is_labeled_with_the_taxonomy() -> None:
    score = _score(
        verdict="incorrect",
        error_type="missed_memory",
        retrieval=_retrieval(
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
        ),
        audit=_audit(("low_relevance", "the needed memory"), ("duplicate", "a repeat")),
    )

    record = failure_record(score)

    assert record is not None
    labels = {entry["reason"]: entry["label"] for entry in record["dropped_evidence"]}
    assert labels == {"low_relevance": "irrelevant_memory", "duplicate": ""}
    assert record["drop_reason_counts"] == {"duplicate": 1, "low_relevance": 1}


def test_collect_failures_reads_a_run_file_and_pairs_replays_with_scores() -> None:
    artifact = _run_file(
        "brainos",
        [_score(verdict="incorrect", error_type="hallucination")],
        task_results=[
            {
                "question": "What database?",
                "retrieved_memory_texts": ["a memory"],
                "retrieved_chunk_texts": [],
            }
        ],
    )

    records = collect_failures(artifact)

    assert len(records) == 1
    assert records[0]["retrieved_memories"] == ["a memory"]
    assert records[0]["question"] == "What database?"
    assert records[0]["trial"] == 0


def test_collect_failures_reads_an_exported_experiment_artifact() -> None:
    artifact = {
        "experiment_version": "experiment-v1",
        "run_id": "exp-1",
        "timestamp": "2026-09-14T00:00:00+00:00",
        "dry_run": True,
        "modes": [
            {
                "mode": "brainos",
                "label": "Mode D — BrainOS memory",
                "trial": 0,
                "aggregate_metrics": {},
                "scores": [_score(verdict="incorrect", error_type="hallucination")],
                "task_results": [{"task_id": "task-1", "retrieved_memory_texts": []}],
            }
        ],
    }

    records = collect_failures([artifact])

    assert len(records) == 1
    assert records[0]["mode"] == "brainos"


def test_a_defect_free_run_reports_no_failures() -> None:
    report = error_report(_run_file("brainos", [_score()]))

    assert report["errors_version"] == ERRORS_VERSION
    assert report["failure_record_count"] == 0
    assert report["labels_vs_scorer"]["unexpected"] == 0
    assert report["scored_record_count"] == 1
    assert report["graded_record_count"] == 1


def test_the_report_aggregates_by_mode_category_and_length() -> None:
    scores = [
        _score(verdict="incorrect", error_type="hallucination"),
        _score(
            task_id="task-2",
            category="multi_hop",
            verdict="incorrect",
            error_type="missed_memory",
            retrieval=_retrieval(
                task_id="task-2",
                category="multi_hop",
                retrieved_fact_ids=["fact-1"],
                prompt_fact_ids=[],
                evidence_in_prompt=False,
                absent_from_prompt_fact_ids=["fact-1"],
                lost_after_selection_fact_ids=["fact-1"],
            ),
            audit=_audit(("cap", "dropped by the item cap")),
        ),
    ]
    report = error_report(_run_file("brainos", scores))

    assert report["failure_record_count"] == 2
    assert report["primary_counts"] == {
        "hallucination": 1,
        "over_compression": 1,
    }
    assert report["by_mode"]["brainos"]["defect_count"] == 2
    assert report["by_category"]["multi_hop"]["evidence_lost_count"] == 1
    assert report["by_length"]["800"]["observed_failure_count"] == 2
    assert report["by_mode_category"]["brainos"]["multi_hop"]["primary_counts"] == {
        "over_compression": 1
    }
    assert report["stage_counts"] == {"generation": 1, "selection": 1}
    concentration = report["concentration"]
    assert {row["failure_type"] for row in concentration} == {
        "hallucination",
        "over_compression",
    }
    assert report["taxonomy"]["labels"] == list(PLAN_ERROR_TYPES)


def test_contributors_are_counted_even_when_they_are_not_the_primary() -> None:
    score = _score(
        verdict="incorrect",
        error_type="missed_memory",
        retrieval=_retrieval(
            retrieved_fact_ids=["fact-1"],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
            lost_after_selection_fact_ids=["fact-1"],
        ),
        audit=_audit(("low_relevance", "dropped")),
    )

    report = error_report(_run_file("brainos", [score]))

    assert report["primary_counts"] == {"irrelevant_memory": 1}
    assert report["contributor_counts"] == {"over_compression": 1}
    assert report["by_mode"]["brainos"]["evidence_lost_count"] == 1


def test_a_scorer_label_the_taxonomy_cannot_refine_is_reported_not_hidden() -> None:
    """The check that keeps "aggregation, not a second scorer" honest."""

    score = _score(
        category="conflict",
        verdict="stale_answer",
        error_type="stale_memory",  # the scorer's label for a conflict task
        answer={"stale_answer": "Wednesday at 09:30 UTC"},
    )

    report = error_report(_run_file("brainos", [score]))

    assert report["labels_vs_scorer"]["unexpected"] == 1
    assert report["labels_vs_scorer"]["unexpected_records"][0]["scorer_error_type"] == (
        "stale_memory"
    )


def test_documented_refinements_are_counted_as_refinements() -> None:
    score = _score(
        verdict="incorrect",
        error_type="missed_memory",
        retrieval=_retrieval(
            retrieved_fact_ids=["fact-1"],
            prompt_fact_ids=[],
            evidence_in_prompt=False,
            absent_from_prompt_fact_ids=["fact-1"],
            lost_after_selection_fact_ids=["fact-1"],
        ),
        audit=_audit(("low_relevance", "dropped")),
    )

    report = error_report(_run_file("brainos", [score]))

    assert report["labels_vs_scorer"]["refined"] == 1
    assert report["labels_vs_scorer"]["refinements"] == {
        "missed_memory": {"irrelevant_memory": 1}
    }
    assert report["labels_vs_scorer"]["unexpected"] == 0


def test_examples_are_capped_per_mode_and_label() -> None:
    scores = [
        _score(task_id="task-1", verdict="incorrect", error_type="hallucination"),
        _score(task_id="task-2", verdict="incorrect", error_type="hallucination"),
    ]
    artifact = _run_file("brainos", scores)

    assert len(error_report(artifact, examples_per_type=1)["examples"]) == 1
    assert len(error_report(artifact, examples_per_type=2)["examples"]) == 2
    assert error_report(artifact, examples_per_type=0)["examples"] == []


def test_unneeded_prompt_facts_are_derived_from_the_ledger_sets() -> None:
    score = _score(
        retrieval=_retrieval(
            supporting_fact_ids=["fact-2"],
            prompt_fact_ids=["fact-1", "fact-2", "fact-3"],
        )
    )

    assert errors.unneeded_prompt_fact_ids(score) == ("fact-3",)


def test_provenance_records_the_sources_and_the_dataset() -> None:
    report = error_report(
        _run_file("brainos", [_score(verdict="incorrect", error_type="hallucination")]),
        sources=[{"path": "results/run.json", "sha256": "abc"}],
        dataset={"path": "benchmarks/context_rot/dataset.jsonl", "sha256": "def"},
    )

    assert report["provenance"]["sources"] == [{"path": "results/run.json", "sha256": "abc"}]
    assert report["provenance"]["dataset"]["sha256"] == "def"
    assert report["provenance"]["modes"] == ["brainos"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_writes_a_report_and_a_jsonl_of_failure_records(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(
            _run_file("brainos", [_score(verdict="incorrect", error_type="hallucination")])
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    records = tmp_path / "records.jsonl"

    code = errors.main(
        [
            str(run_path),
            "--output",
            str(output),
            "--records",
            str(records),
            "--dataset",
            str(DATASET),
        ]
    )

    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["failure_record_count"] == 1
    assert report["provenance"]["sources"][0]["path"] == str(run_path)
    lines = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["failure_type"] == "hallucination"


def test_cli_accepts_a_missing_dataset_without_losing_the_analysis(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(_run_file("brainos", [_score(error_type="")])), encoding="utf-8")

    code = errors.main(
        [
            str(run_path),
            "--dataset",
            str(tmp_path / "absent.jsonl"),
            "--output",
            str(tmp_path / "r.json"),
        ]
    )

    assert code == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["provenance"]["dataset"]["task_count"] == 0


def test_the_committed_answer_fixture_is_loadable_and_keyed_by_task() -> None:
    answers = [
        json.loads(line)
        for line in ANSWERS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert answers, "the fixture exists so the Phase 12 validation is reproducible"
    task_ids = {task.task_id for task in _tasks()}
    assert {answer["task_id"] for answer in answers} <= task_ids
    assert all("mode" in answer or "answer" in answer for answer in answers)


def test_scoring_a_replayed_record_still_works_without_a_retrieval_report() -> None:
    """Older artifacts have no audit; the record degrades to fact-level evidence."""

    task = _tasks()[0]
    replay = {"task_id": task.task_id, "retrieved_memory_texts": [], "retrieved_chunk_texts": []}
    replay["prompt_messages"] = [{"role": "user", "content": task.question}]

    score = score_record(task, replay, answer=None)

    assert score["retrieval_audit"]["available"] is False
    assert score["retrieval_audit"]["dropped"] == []
    assert errors.drop_reason_counts(score) == {}


@pytest.mark.parametrize("label", PLAN_ERROR_TYPES)
def test_every_plan_label_has_a_stage_and_a_definition(label: str) -> None:
    assert label in errors.ERROR_STAGE
    assert label in errors.ERROR_DEFINITION
    assert label in errors.ATTRIBUTION_ORDER


def test_a_bare_path_is_rejected_instead_of_yielding_no_failures() -> None:
    """A path argument used to normalize to nothing, which reads as "no errors".

    ``collect_failures("results/run.json")`` and
    ``error_report("results/run.json")`` silently produced an empty report,
    because a path is not a mapping and the sequence branch treats ``str`` as
    text. Reading files belongs to the CLI, so the library refuses the argument
    instead of returning a report that cannot be told apart from a clean run.
    """

    with pytest.raises(TypeError, match="parsed artifacts"):
        errors.mode_result_items("results/run.json")

    with pytest.raises(TypeError, match="parsed artifacts"):
        errors.collect_failures(Path("results/run.json"))

    with pytest.raises(TypeError, match="parsed artifacts"):
        error_report([Path("results/run.json")])
