"""Validity tests for the generated context-rot benchmark (plan Phase 7).

A benchmark can be wrong in ways that produce plausible-looking numbers: facts
the memory layer never stores, filler that pollutes memory, questions whose
answer is a guess, lengths that never reach the tier they claim. These tests
check the *properties the experiment depends on* — not the generated text — so a
future change to the generator cannot quietly invalidate a result.

The three load-bearing properties:

1. every planted fact is storable by the application's own memory policy;
2. no filler turn is storable (otherwise modes are compared on noise);
3. the committed dataset matches its manifest hash, so a result can be tied to
   the exact text it was measured on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.context_rot import spec
from benchmarks.context_rot.generation import (
    build_manifest,
    dataset_sha256,
    generate_dataset,
    generate_task,
)
from brain.memory_policy import MemoryPolicy, extract_candidates
from evaluation.datasets import dataset_issues, load_jsonl

COMMITTED = Path("benchmarks/context_rot/dataset.jsonl")
MANIFEST = Path("benchmarks/context_rot/MANIFEST.json")

COMMITTED_TASKS = load_jsonl(COMMITTED)
POLICY = MemoryPolicy()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _stored_texts(task) -> set[str]:
    """Every text the application's memory policy would store, from the transcript."""

    stored: set[str] = set()
    for message in task.conversation:
        content = str(message.get("content", ""))
        candidates = extract_candidates(content, source=str(message.get("role", "user")))
        for candidate in POLICY.select(candidates):
            stored.add(candidate.text)
    return stored


# --------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------- #


def test_the_committed_dataset_matches_its_manifest(manifest: dict) -> None:
    assert manifest["dataset_sha256"] == dataset_sha256(COMMITTED_TASKS)
    assert manifest["generator_version"] == spec.GENERATOR_VERSION
    assert manifest["task_count"] == len(COMMITTED_TASKS)
    assert manifest["token_counter"] == "estimate_tokens"


def test_the_committed_dataset_is_structurally_valid() -> None:
    assert dataset_issues(COMMITTED_TASKS) == []


def test_the_manifest_describes_the_regeneration_command(manifest: dict) -> None:
    assert "generation.py" in manifest["regenerate"]
    assert f"--seed {manifest['seed']}" in manifest["regenerate"]


def test_every_category_is_present_in_the_committed_dataset(manifest: dict) -> None:
    assert tuple(manifest["categories"]) == spec.CATEGORIES
    assert {task.category for task in COMMITTED_TASKS} == set(spec.CATEGORIES)


def test_regenerating_with_the_manifest_parameters_reproduces_the_file(
    manifest: dict,
) -> None:
    tasks = generate_dataset(
        categories=tuple(manifest["categories"]),
        lengths=tuple(manifest["lengths"]),
        seed=manifest["seed"],
        variants=manifest["variants"],
    )

    assert dataset_sha256(tasks) == manifest["dataset_sha256"]


# --------------------------------------------------------------------- #
# Benchmark validity: what enters memory
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("task", COMMITTED_TASKS, ids=lambda task: task.task_id)
def test_every_planted_fact_is_storable(task) -> None:
    """A fact the memory policy rejects could never be retrieved by any mode."""

    stored = _stored_texts(task)

    for fact in task.planted_facts():
        assert fact.text in stored, fact.fact_id


@pytest.mark.parametrize("task", COMMITTED_TASKS, ids=lambda task: task.task_id)
def test_filler_never_enters_memory(task) -> None:
    """Filler is length, not evidence: if it were stored, memory fills with noise."""

    planted = {fact.text for fact in task.planted_facts()}
    stray = _stored_texts(task) - planted

    assert not stray, sorted(stray)[:3]


@pytest.mark.parametrize("task", COMMITTED_TASKS, ids=lambda task: task.task_id)
def test_fact_markers_appear_where_the_ledger_says_they_do(task) -> None:
    for fact in task.planted_facts():
        message = task.conversation[fact.introduced_turn]
        assert message.get("role") == "user", fact.fact_id
        assert fact.text == str(message.get("content")), fact.fact_id
        for marker in fact.markers:
            assert marker.casefold() in fact.text.casefold(), (fact.fact_id, marker)


@pytest.mark.parametrize("task", COMMITTED_TASKS, ids=lambda task: task.task_id)
def test_no_two_facts_share_a_marker_set(task) -> None:
    """Detection is per-marker, so identical marker sets would be ambiguous.

    The single exception is deliberate: the repeated fact repeats the same
    statement, so it carries the same markers. It is a *supporting* fact, never a
    required one.
    """

    by_signature: dict[tuple[str, ...], list] = {}
    for fact in task.planted_facts():
        by_signature.setdefault(tuple(sorted(fact.markers)), []).append(fact)

    for signature, facts in by_signature.items():
        if len(facts) < 2:
            continue
        assert task.category == "single_hop", (task.task_id, signature)
        assert {fact.fact_type for fact in facts} == {"durable", "repeated"}
        assert not any(fact.fact_id in task.evidence.required for fact in facts[1:])


# --------------------------------------------------------------------- #
# Benchmark validity: placement and length
# --------------------------------------------------------------------- #


def test_the_evidence_is_out_of_reach_for_a_sliding_window() -> None:
    """The question must not be answerable from the last couple of turns."""

    for task in COMMITTED_TASKS:
        distance = task.metadata.get("required_fact_distance_turns")
        if task.evidence.abstention_expected:
            assert distance is None
            continue
        assert distance is not None and distance >= 4, task.task_id


def test_generated_lengths_track_the_requested_tier() -> None:
    for target in (800, 2000):
        for category in spec.CATEGORIES:
            task = generate_task(category, target)
            achieved = int(task.metadata["achieved_tokens"])
            assert 0.8 * target <= achieved <= 1.2 * target, (task.task_id, achieved)


def test_longer_tiers_produce_longer_conversations() -> None:
    short = generate_task(spec.CATEGORY_SINGLE_HOP, 800)
    long = generate_task(spec.CATEGORY_SINGLE_HOP, 4000)

    assert long.conversation_length > short.conversation_length
    assert long.metadata["achieved_tokens"] > short.metadata["achieved_tokens"]


def test_generation_is_deterministic_and_seed_sensitive() -> None:
    first = generate_task(spec.CATEGORY_CONFLICT, 800, seed=7)
    again = generate_task(spec.CATEGORY_CONFLICT, 800, seed=7)
    other = generate_task(spec.CATEGORY_CONFLICT, 800, seed=8)

    assert first.to_dict() == again.to_dict()
    assert first.to_dict() != other.to_dict()


def test_the_research_tier_is_the_plans_length_ladder() -> None:
    assert spec.TIERS["research"] == spec.PLAN_LENGTH_LADDER
    assert spec.TIERS["smoke"] == (800,)


# --------------------------------------------------------------------- #
# Category contracts
# --------------------------------------------------------------------- #


def _task(category: str):
    return next(task for task in COMMITTED_TASKS if task.category == category)


def test_single_hop_requires_one_fact_and_repeats_it() -> None:
    task = _task("single_hop")

    assert len(task.evidence.required) == 1
    assert not task.evidence.forbidden
    assert any(fact.fact_type == "repeated" for fact in task.facts)
    assert task.planted_facts()


def test_multi_hop_requires_both_facts() -> None:
    task = _task("multi_hop")

    assert len(task.evidence.required) == 2
    turns = [task.facts_by_id()[fact_id].introduced_turn for fact_id in task.evidence.required]
    assert max(turns) - min(turns) > 10, "the hops must be far apart"


def test_temporal_and_conflict_tasks_have_a_superseded_pair() -> None:
    for category in ("temporal", "conflict"):
        task = _task(category)
        (fact_id,) = task.evidence.forbidden
        forbidden = task.facts_by_id()[fact_id]
        (current_id,) = task.evidence.required
        current = task.facts_by_id()[current_id]

        assert len(task.evidence.required) == 1
        assert forbidden.superseded_by == current_id
        assert current.supersedes == fact_id
        assert forbidden.introduced_turn < current.introduced_turn
        # the stale value must not be an accepted answer
        assert not any(
            forbidden.value.casefold() in accepted.casefold()
            for accepted in task.answers()
        )


def test_conflict_task_uses_an_explicit_correction() -> None:
    task = _task("conflict")
    correction = task.facts_by_id()[task.evidence.required[0]]

    assert correction.fact_type == "correction"
    assert "Correction" in correction.text


def test_distractor_task_is_dominated_by_irrelevant_facts() -> None:
    task = _task("distractor")

    irrelevant = [fact for fact in task.facts if fact.fact_type == "irrelevant"]
    assert len(irrelevant) >= 3
    assert len(task.evidence.required) == 1


def test_cross_session_task_marks_a_boundary_and_puts_the_fact_before_it() -> None:
    task = _task("cross_session")

    (fact_id,) = task.evidence.required
    fact = task.facts_by_id()[fact_id]
    boundary = task.metadata["session_boundaries"][0]

    assert task.session_count == 2
    assert fact.introduced_turn < boundary
    assert task.conversation[boundary].get("session") == 1
    assert all(
        message.get("session", 0) == 0 for message in task.conversation[:boundary]
    )


def test_abstention_task_never_mentions_its_probe_topic() -> None:
    for task in COMMITTED_TASKS:
        if not task.evidence.abstention_expected:
            continue
        topic = str(task.metadata["probe_topic"])
        transcript = " ".join(
            str(message.get("content", "")).casefold() for message in task.conversation
        )

        for word in topic.split():
            assert word.casefold() not in transcript, (task.task_id, word)
        assert topic.casefold() in task.question.casefold()
        assert not task.evidence.required
        assert task.expected_answer == ""
        # memory is not empty: the question is not trivially unanswerable
        assert task.planted_facts()


def test_every_task_ends_with_a_question_about_the_planted_evidence() -> None:
    for task in COMMITTED_TASKS:
        assert task.question.endswith("?")
        assert task.expected_answer or task.evidence.abstention_expected


# --------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------- #


def test_manifest_records_the_provenance_a_run_needs() -> None:
    tasks = generate_dataset(categories=("single_hop",), lengths=(800,), seed=5)
    manifest = build_manifest(
        tasks,
        tier="smoke",
        lengths=(800,),
        categories=("single_hop",),
        variants=1,
        seed=5,
    )

    assert manifest["task_count"] == 1
    assert manifest["category_counts"] == {"single_hop": 1}
    assert manifest["dataset_sha256"] == dataset_sha256(tasks)
    assert manifest["mean_achieved_tokens"] > 0


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown category"):
        generate_task("telepathy", 800)


def test_zero_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        generate_task(spec.CATEGORY_SINGLE_HOP, 0)
