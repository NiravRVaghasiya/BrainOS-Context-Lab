"""Scoring-rule tests for the Phase 7 benchmark.

The scorer decides what every headline number means, so these tests are written
against the *rules* (what is correct, what is stale, what is a retrieval failure)
rather than against generated data. A deterministic mock answer plays the model's
role, which is what makes the scoring path testable without credentials.
"""

from __future__ import annotations

import pytest

from evaluation.datasets import BenchmarkFact, BenchmarkTask, EvidenceContract
from evaluation.scoring import (
    ABSTENTION_PATTERNS,
    ANSWER_VERDICTS,
    AnswerSet,
    abstains,
    aggregate_scores,
    contains_phrase,
    detect_facts_in_texts,
    expects_abstention,
    normalize_answer,
    score_answer,
    score_record,
    score_retrieval,
    unavailable_required_facts,
)


def _fact(fact_id: str, value: str, markers: tuple[str, ...], **kwargs) -> BenchmarkFact:
    return BenchmarkFact(
        fact_id=fact_id,
        text=f"For the record, {value}.",
        markers=markers,
        value=value,
        **kwargs,
    )


def _task(**overrides) -> BenchmarkTask:
    facts = overrides.pop("facts", None) or (
        _fact("fact-1", "PostgreSQL 16", ("Project Atlas", "PostgreSQL 16")),
        _fact("fact-2", "MySQL 8", ("Project Atlas", "MySQL 8"), supersedes="fact-1"),
        _fact("fact-3", "MariaDB 11", ("Project Beacon", "MariaDB 11"), fact_type="irrelevant"),
    )
    evidence = overrides.pop(
        "evidence",
        EvidenceContract(required=("fact-2",), forbidden=("fact-1",)),
    )
    base = {
        "task_id": "t1",
        "category": "temporal",
        "conversation": [{"role": "user", "content": "For the record, MySQL 8."}],
        "question": "Which database?",
        "expected_answer": "MySQL 8",
        "required_fact_ids": evidence.required,
        "facts": facts,
        "evidence": evidence,
    }
    base.update(overrides)
    return BenchmarkTask.from_dict({**base, "facts": [f.to_dict() for f in base["facts"]]})


# --------------------------------------------------------------------- #
# Text normalization and matching
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("MySQL 8", "mysql 8"),
        ("eu-west-1", "eu west 1"),
        ("EU_West_1", "eu west 1"),
        ("  Friday   at 17:00 UTC. ", "friday at 17 00 utc"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalization_is_consistent_across_separators(answer, expected: str) -> None:
    assert normalize_answer(answer) == expected


def test_phrase_matching_respects_word_boundaries() -> None:
    assert contains_phrase("The database is MySQL 8 now.", "MySQL 8")
    assert contains_phrase("region: EU West 1", "eu-west-1")
    assert not contains_phrase("eu-west-10", "eu-west-1")
    assert not contains_phrase("MySQL", "MySQL 8")
    assert not contains_phrase("anything", "")


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "I do not know which database that is.",
        "That was not mentioned in the conversation.",
        "There is no information about it in the context.",
        "I cannot find anything about it.",
        "unknown",
    ],
)
def test_abstention_phrasings_are_recognised(answer: str) -> None:
    assert abstains(answer)


def test_abstention_patterns_are_all_reachable_after_normalization() -> None:
    """A pattern containing an apostrophe must still match normalized text."""

    for pattern in ABSTENTION_PATTERNS:
        assert abstains(pattern), pattern


def test_an_answer_that_asserts_a_value_is_not_an_abstention() -> None:
    assert not abstains("The database is MySQL 8.")
    assert not abstains("")


# --------------------------------------------------------------------- #
# Retrieval scoring
# --------------------------------------------------------------------- #


def test_markers_must_co_occur_in_one_evidence_item() -> None:
    facts = _task().planted_facts()

    # The subject and the value are in different memories: not evidence for a
    # fact that asserts both together.
    split = detect_facts_in_texts(["Project Atlas", "PostgreSQL 16"], facts)
    together = detect_facts_in_texts(
        ["Project Atlas production database PostgreSQL 16"], facts
    )

    assert "fact-1" not in split
    assert "fact-1" in together


def test_recall_counts_only_required_facts_and_precision_counts_all_relevant() -> None:
    task = _task()
    score = score_retrieval(
        task,
        retrieved_texts=[
            "Project Atlas production database MySQL 8",  # required
            "Project Beacon MariaDB 11",  # relevant? no: irrelevant fact
        ],
    )

    assert score.recall == 1.0
    assert score.precision == pytest.approx(0.5)
    assert score.missing_fact_ids == ()
    assert score.forbidden_retrieved == ()


def test_a_retrieved_stale_fact_is_reported_as_forbidden() -> None:
    task = _task()
    score = score_retrieval(
        task, retrieved_texts=["Project Atlas production database PostgreSQL 16"]
    )

    assert score.forbidden_retrieved == ("fact-1",)
    assert score.recall == 0.0


def test_prompt_availability_is_scored_per_message() -> None:
    task = _task()
    score = score_retrieval(
        task,
        prompt_messages=[
            {"role": "system", "content": "Project Atlas was mentioned"},
            {"role": "user", "content": "MySQL 8 was the answer"},
        ],
    )

    assert score.evidence_in_prompt is False
    assert score.expected_answer_in_prompt is True

    same_message = score_retrieval(
        task,
        prompt_messages=[
            {"role": "system", "content": "Project Atlas production database MySQL 8"}
        ],
    )
    assert same_message.evidence_in_prompt is True


def test_full_context_retrieves_nothing_and_still_puts_evidence_in_the_prompt() -> None:
    """Mode A has no retriever: recall is 0 by construction, availability is not."""

    task = _task()
    score = score_retrieval(
        task,
        retrieved_texts=[],
        prompt_messages=[
            {"role": "user", "content": "For the record, Project Atlas runs MySQL 8."}
        ],
    )

    assert score.recall == 0.0
    assert score.evidence_in_prompt is True


def test_abstention_tasks_have_no_required_evidence() -> None:
    task = _task(
        evidence=EvidenceContract(abstention_expected=True),
        expected_answer="",
    )
    score = score_retrieval(task, retrieved_texts=["Project Beacon MariaDB 11"])

    assert score.required_fact_ids == ()
    assert score.precision == 0.0
    assert score.evidence_in_prompt is False


# --------------------------------------------------------------------- #
# Answer scoring
# --------------------------------------------------------------------- #


def test_an_accepted_answer_grades_correct() -> None:
    score = score_answer(_task(), "The current database is MySQL 8.")

    assert score.verdict == "correct"
    assert score.matched_answer == "MySQL 8"
    assert score.error_type == ""


def test_an_accepted_answer_with_a_hedge_still_grades_correct() -> None:
    score = score_answer(
        _task(), "MySQL 8 — although I don't have information about the region."
    )

    assert score.verdict == "correct"


def test_a_superseded_value_grades_stale() -> None:
    score = score_answer(_task(), "PostgreSQL 16")

    assert score.verdict == "stale_answer"
    assert score.stale_answer == "PostgreSQL 16"
    assert score.error_type == "stale_memory"


def test_a_conflict_task_labels_a_stale_answer_as_conflicting_memory() -> None:
    task = _task(category="conflict")

    assert score_answer(task, "PostgreSQL 16").verdict == "stale_answer"
    assert score_answer(task, "PostgreSQL 16").error_type == "conflicting_memory"


def test_a_declined_answer_to_an_answerable_task_is_not_correct() -> None:
    score = score_answer(_task(), "I don't know.")

    assert score.verdict == "abstained"
    assert score.error_type == "missed_memory"


def test_abstaining_when_the_evidence_was_present_is_wrong_abstention() -> None:
    task = _task()
    record = {
        "mode": "brainos",
        "prompt_messages": [
            {"role": "system", "content": "Project Atlas production database MySQL 8"}
        ],
        "retrieved_memory_texts": ["Project Atlas production database MySQL 8"],
    }

    scored = score_record(task, record, answer="I don't know.")

    assert scored["answer"]["verdict"] == "wrong_abstention"
    assert scored["error_type"] == "wrong_abstention"


def test_an_unexpected_value_grades_incorrect() -> None:
    score = score_answer(_task(), "SQLite 3")

    assert score.verdict == "incorrect"
    assert score.error_type == "wrong_answer"


def test_no_answer_grades_ungraded() -> None:
    score = score_answer(_task(), None)

    assert score.verdict == "ungraded"
    assert not score.graded


def test_abstention_is_correct_when_the_answer_is_absent() -> None:
    task = _task(
        category="abstention",
        evidence=EvidenceContract(abstention_expected=True),
        expected_answer="",
    )

    assert score_answer(task, "I don't know.").verdict == "correct"
    assert score_answer(task, "The vendor is Baker & Co.").verdict == "incorrect"
    assert expects_abstention(task) is True


# --------------------------------------------------------------------- #
# Session isolation
# --------------------------------------------------------------------- #


def _cross_session_task() -> BenchmarkTask:
    return BenchmarkTask.from_dict(
        {
            "task_id": "cross",
            "category": "cross_session",
            "conversation": [
                {"role": "user", "content": "For the record, the data centre is eu-west-1."},
                {"role": "user", "content": "Anything else?", "session": 1},
            ],
            "question": "Which region?",
            "expected_answer": "eu-west-1",
            "required_fact_ids": ["fact-1"],
            "facts": [
                _fact(
                    "fact-1",
                    "eu-west-1",
                    ("Project Ionic", "eu-west-1"),
                    session_index=0,
                ).to_dict()
            ],
            "evidence": {"required": ["fact-1"]},
            "session_count": 2,
        }
    )


def test_a_required_fact_from_an_earlier_session_is_unreachable_when_isolated() -> None:
    task = _cross_session_task()

    assert unavailable_required_facts(task, session_isolation=False) == ()
    assert unavailable_required_facts(task, session_isolation=True) == ("fact-1",)
    assert expects_abstention(task, session_isolation=False) is False
    assert expects_abstention(task, session_isolation=True) is True


def test_abstention_is_the_only_correct_behaviour_when_isolated() -> None:
    """A fact from a closed session is unreachable, so a value answer is a guess."""

    task = _cross_session_task()

    assert score_answer(task, "eu-west-1").verdict == "correct"
    isolated = score_answer(task, "eu-west-1", session_isolation=True)
    assert isolated.verdict == "incorrect"
    assert isolated.error_type == "hallucination"
    assert score_answer(task, "I don't know.", session_isolation=True).verdict == "correct"


# --------------------------------------------------------------------- #
# score_record and aggregation
# --------------------------------------------------------------------- #


def _record(verdict_answer: str | None, *, evidence: bool = True, mode: str = "brainos") -> dict:
    content = (
        "Project Atlas production database MySQL 8"
        if evidence
        else "Nothing relevant here"
    )
    return {
        "task_id": "t1",
        "mode": mode,
        "context_reduction": 0.5,
        "final_context_tokens": 100,
        "full_context_reference_tokens": 200,
        "retrieved_memory_texts": [content],
        "retrieved_chunk_texts": [],
        "prompt_messages": [{"role": "system", "content": content}],
    }


def test_score_record_attributes_an_incorrect_answer_with_evidence_to_hallucination() -> None:
    scored = score_record(_task(), _record("MySQL 8"), answer="SQLite 3")

    assert scored["error_type"] == "hallucination"
    assert scored["retrieval"]["evidence_in_prompt"] is True


def test_score_record_attributes_an_incorrect_answer_without_evidence_to_missed_memory() -> None:
    scored = score_record(
        _task(), _record("MySQL 8", evidence=False), answer="SQLite 3"
    )

    assert scored["error_type"] == "missed_memory"


def test_score_record_is_json_ready_and_carries_the_cost_numbers() -> None:
    scored = score_record(_task(), _record("MySQL 8"), answer="MySQL 8")

    assert scored["context_reduction"] == 0.5
    assert scored["final_context_tokens"] == 100
    assert scored["answer"]["verdict"] == "correct"
    assert set(scored) == {
        "task_id",
        "category",
        "mode",
        "retrieval",
        "answer",
        "error_type",
        "context_reduction",
        "final_context_tokens",
        "full_context_reference_tokens",
    }


def test_aggregate_reports_rates_with_their_own_denominators() -> None:
    task = _task()
    scores = [
        score_record(task, _record("MySQL 8"), answer="MySQL 8"),
        score_record(task, _record("MySQL 8"), answer="PostgreSQL 16"),
        score_record(task, _record("MySQL 8"), answer=None),
    ]

    summary = aggregate_scores(scores)

    assert summary["task_count"] == 3
    assert summary["graded_answer_count"] == 2
    assert summary["answer_accuracy"] == pytest.approx(0.5)
    assert summary["stale_answer_rate"] == pytest.approx(0.5)
    assert summary["retrieval_recall"] == pytest.approx(1.0)
    assert summary["evidence_in_prompt_rate"] == pytest.approx(1.0)
    assert summary["verdict_counts"] == {"correct": 1, "stale_answer": 1, "ungraded": 1}


def test_aggregate_excludes_abstention_tasks_from_recall() -> None:
    """A task with nothing to retrieve must not contribute a vacuous 1.0."""

    answerable = _task()
    abstention = _task(
        task_id="t2",
        category="abstention",
        evidence=EvidenceContract(abstention_expected=True),
        expected_answer="",
    )
    scores = [
        score_record(answerable, _record("MySQL 8"), answer="MySQL 8"),
        score_record(
            abstention,
            {
                "task_id": "t2",
                "mode": "brainos",
                "retrieved_memory_texts": [],
                "prompt_messages": [],
            },
            answer="I don't know.",
        ),
    ]

    summary = aggregate_scores(scores)

    assert summary["retrieval_recall"] == pytest.approx(1.0)
    assert summary["abstention_accuracy"] == pytest.approx(1.0)
    assert summary["answer_accuracy"] == pytest.approx(1.0)


def test_aggregate_is_empty_safe() -> None:
    summary = aggregate_scores([])

    assert summary["task_count"] == 0
    assert summary["answer_accuracy"] == 0.0
    assert summary["by_category"] == {}


def test_aggregate_groups_by_category() -> None:
    temporal = score_record(_task(), _record("MySQL 8"), answer="MySQL 8")
    conflict = score_record(
        _task(category="conflict"), _record("MySQL 8"), answer="MySQL 8"
    )

    summary = aggregate_scores([temporal, conflict])

    assert set(summary["by_category"]) == {"conflict", "temporal"}
    assert summary["by_category"]["temporal"]["answer_accuracy"] == 1.0


def test_every_verdict_is_in_the_documented_vocabulary() -> None:
    assert set(ANSWER_VERDICTS) == {
        "correct",
        "abstained",
        "wrong_abstention",
        "stale_answer",
        "incorrect",
        "ungraded",
    }


# --------------------------------------------------------------------- #
# Answer files
# --------------------------------------------------------------------- #


def test_answer_set_prefers_a_mode_specific_entry() -> None:
    answers = AnswerSet.from_records(
        [
            {"task_id": "t1", "answer": "generic"},
            {"task_id": "t1", "mode": "brainos", "answer": "specific"},
        ]
    )

    assert answers.get("t1", "brainos") == "specific"
    assert answers.get("t1", "rag") == "generic"
    assert answers.get("t9", "rag") is None
    assert len(answers) == 2


def test_answer_records_must_carry_a_task_id_and_an_answer() -> None:
    with pytest.raises(ValueError, match="task_id"):
        AnswerSet.from_records([{"answer": "x"}])
    with pytest.raises(ValueError, match="no 'answer'"):
        AnswerSet.from_records([{"task_id": "t1"}])
