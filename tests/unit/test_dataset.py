"""Schema tests for the Phase 7 benchmark data contract.

These pin the contract itself — ids, the ledger, the evidence contract, and
validation — rather than the generated text, which
``tests/evaluation/test_context_rot_dataset.py`` covers.
"""

from __future__ import annotations

import json

import pytest

from evaluation.datasets import (
    BenchmarkFact,
    BenchmarkTask,
    EvidenceContract,
    dataset_issues,
    load_jsonl,
    validate_task,
    write_jsonl,
)

DATASET = "benchmarks/context_rot/dataset.jsonl"


def _record(**overrides: object) -> dict:
    base = {
        "task_id": "t1",
        "category": "single_hop",
        "conversation": [
            {"role": "user", "content": "For the record, the database is MySQL 8."},
            {"role": "assistant", "content": "Understood."},
        ],
        "question": "Which database?",
        "expected_answer": "MySQL 8",
    }
    base.update(overrides)
    return base


def test_the_committed_dataset_loads_and_validates() -> None:
    tasks = load_jsonl(DATASET)

    assert tasks
    assert dataset_issues(tasks) == []
    assert {task.category for task in tasks} == {
        "single_hop",
        "multi_hop",
        "temporal",
        "conflict",
        "distractor",
        "cross_session",
        "abstention",
    }


def test_every_task_carries_a_fact_ledger() -> None:
    for task in load_jsonl(DATASET):
        assert task.facts, task.task_id
        for fact in task.facts:
            assert fact.fact_id
            if fact.is_planted():
                assert fact.markers, fact.fact_id
                assert fact.introduced_turn >= 0


def test_a_required_fact_must_exist_in_the_ledger() -> None:
    task = BenchmarkTask.from_dict(
        _record(required_fact_ids=["fact-9"], facts=[], evidence={"required": ["fact-9"]})
    )

    assert any("unknown fact" in problem for problem in validate_task(task))


def test_an_abstention_task_cannot_require_evidence() -> None:
    task = BenchmarkTask.from_dict(
        _record(
            required_fact_ids=["fact-1"],
            facts=[BenchmarkFact(fact_id="fact-1", text="x", markers=("x",)).to_dict()],
            evidence={"required": ["fact-1"], "abstention_expected": True},
        )
    )

    assert any("abstention" in problem for problem in validate_task(task))


def test_legacy_required_memory_ids_records_still_load() -> None:
    """Phase 6 records used a different field name for the same concept."""

    task = BenchmarkTask.from_dict(
        {
            "task_id": "legacy",
            "category": "single_hop",
            "conversation": [{"role": "user", "content": "hi there friend"}],
            "question": "q",
            "expected_answer": "a",
            "required_memory_ids": ["fact-1"],
        }
    )

    assert task.required_fact_ids == ("fact-1",)


def test_evidence_required_wins_when_the_field_is_absent() -> None:
    task = BenchmarkTask.from_dict(_record(evidence={"required": ["fact-7"]}))

    assert task.required_fact_ids == ("fact-7",)


def test_answers_collect_expected_acceptable_and_fact_values() -> None:
    task = BenchmarkTask.from_dict(
        _record(
            expected_answer="MySQL 8",
            acceptable_answers=["MySQL"],
            facts=[
                BenchmarkFact(
                    fact_id="fact-1",
                    text="db fact",
                    markers=("db", "MySQL 8"),
                    value="MySQL 8",
                ).to_dict()
            ],
            evidence={"required": ["fact-1"]},
        )
    )

    # expected first, then explicit variants, then the ledger value (no dupes)
    assert task.answers() == ("MySQL 8", "MySQL")
    assert task.required_facts()[0].fact_id == "fact-1"


def test_absent_facts_are_ledger_only() -> None:
    task = BenchmarkTask.from_dict(
        _record(
            facts=[
                BenchmarkFact(fact_id="fact-1", fact_type="absent", text="").to_dict(),
                BenchmarkFact(
                    fact_id="fact-2", text="real", markers=("real",)
                ).to_dict(),
            ],
            evidence={"abstention_expected": True},
        )
    )

    assert [fact.fact_id for fact in task.planted_facts()] == ["fact-2"]


def test_unknown_roles_are_rejected() -> None:
    task = BenchmarkTask.from_dict(
        _record(conversation=[{"role": "wizard", "content": "hello there"}])
    )

    assert any("unknown role" in problem for problem in validate_task(task))


def test_dataset_issues_prefix_every_problem_with_its_task() -> None:
    task = BenchmarkTask.from_dict(_record(evidence={"required": ["nope"]}))

    issues = dataset_issues([task, task])

    assert any(issue.startswith("t1: ") for issue in issues)
    assert any("duplicate task id" in issue for issue in issues)


def test_round_trip_through_disk_preserves_the_ledger(tmp_path) -> None:
    original = load_jsonl(DATASET)[0]
    path = tmp_path / "dataset.jsonl"

    write_jsonl(path, [original])
    reloaded = load_jsonl(path)

    assert reloaded[0].to_dict() == original.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["task_id"] == original.task_id


def test_session_count_must_match_the_transcript_marks() -> None:
    task = BenchmarkTask.from_dict(
        _record(
            session_count=2,
            conversation=[{"role": "user", "content": "a" * 20}],
        )
    )

    assert any("session_count" in problem for problem in validate_task(task))


@pytest.mark.parametrize("field", ["task_id", "question"])
def test_empty_required_fields_are_reported(field: str) -> None:
    task = BenchmarkTask.from_dict(_record(**{field: ""}))

    assert validate_task(task)


def test_evidence_contract_round_trips() -> None:
    contract = EvidenceContract(
        required=("a",), supporting=("b",), forbidden=("c",), abstention_expected=False
    )

    assert EvidenceContract.from_dict(contract.to_dict()) == contract
