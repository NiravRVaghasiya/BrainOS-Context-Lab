"""Deterministic generator for the Phase 7 context-rot benchmark.

The benchmark's research question is whether performance degrades more slowly
when BrainOS — rather than a raw transcript window — decides what history the
model sees. Answering it needs conversations where the answer is *old*, where a
newer fact contradicts an older one, and where the same conversation is long
enough that no window can hold it.

What this module produces, per task:

```text
filler turns            non-storable chatter (questions, greetings, acks)
planted facts           declarative sentences the memory policy stores
distractor facts        irrelevant-but-durable facts that compete for recall
question                asked at the very end, after the fact is out of reach
fact ledger             every planted fact + where it sits + what replaced what
evidence contract       required / supporting / forbidden facts, abstention flag
```

Three properties are non-negotiable, and the test suite asserts all three:

1. **Determinism.** Everything is derived from ``(seed, category, length,
   variant)``. Same inputs, byte-identical dataset; the manifest records the
   hash so a result can be tied to the exact text it was computed on.
2. **The facts are retrievable in principle.** Every planted fact goes through
   the application's own memory policy in the tests. A fact the policy rejects
   would make BrainOS mode fail for a reason the benchmark is not measuring.
3. **Filler is not storable.** Filler is there to make the conversation long,
   not to become memory. If it were stored, the memory block would fill with
   noise and the comparison would stop being about selection.

The generator never calls a model and never needs a provider key. Lengths are
estimated with the repository's dependency-free counter; ``--tier research``
produces the plan's 5k–120k ladder and is expensive to run, not to generate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Keep the generator usable directly from a source checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
for _path in (_SRC_DIR, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from benchmarks.context_rot import spec  # noqa: E402  (path bootstrap first)
from brain.tokenizers import estimate_tokens  # noqa: E402  (path bootstrap first)
from evaluation.datasets import (  # noqa: E402  (path bootstrap first)
    BenchmarkFact,
    BenchmarkTask,
    EvidenceContract,
    validate_task,
    write_jsonl,
)

#: Fraction of the target length a conversation must reach before it is accepted.
_LENGTH_FLOOR = 0.9
#: Fraction above which trailing filler is trimmed back toward the target.
_LENGTH_CEILING = 1.15
#: Facts are never planted in the last stretch of a conversation: the question
#: has to be separated from its evidence, otherwise the sliding-window baseline
#: would answer from the same two turns a memory layer would.
_TAIL_RESERVE = 0.06


@dataclass(frozen=True)
class PlannedFact:
    """A fact plus the turns that will carry it into the conversation."""

    fact: BenchmarkFact
    user_text: str
    ack: str
    position: float
    role: str = "irrelevant"  # required | supporting | forbidden | irrelevant | absent
    session: int = 0


@dataclass(frozen=True)
class TaskPlan:
    """Everything a category decides before the conversation is assembled."""

    category: str
    planned: tuple[PlannedFact, ...]
    question: str
    expected_answer: str
    acceptable: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    supporting: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    abstention: bool = False
    session_count: int = 1
    session_fraction: float = 0.0
    probe_topic: str = ""
    #: Ledger entries that are never spoken (the abstention category's absent
    #: probe). They exist so the contract can name the thing that has no answer.
    extra_facts: tuple[BenchmarkFact, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Planning: what facts a category needs and where they sit
# ---------------------------------------------------------------------- #


def distractor_count(target_tokens: int, *, factor: float = 1.0) -> int:
    """How many irrelevant durable facts to plant alongside the answer.

    Scales with length — a 40k-token conversation with three stored facts would
    make retrieval trivial — but stays bounded, because the point is to stress
    selection, not to make recall impossible.
    """

    raw = 1 + int(target_tokens // 3000)
    return max(1, min(10, round(raw * factor)))


def _spread(count: int, *, low: float = 0.04, high: float = 0.78) -> list[float]:
    """Evenly spread ``count`` positions across ``[low, high]``, in order."""

    if count <= 0:
        return []
    if count == 1:
        return [round((low + high) / 2, 4)]
    step = (high - low) / (count - 1)
    return [round(low + index * step, 4) for index in range(count)]


def _distractors(
    rng: random.Random,
    count: int,
    *,
    used_projects: set[str],
    used_values: set[str],
    start_index: int,
) -> list[PlannedFact]:
    """Build irrelevant-but-durable facts from the unused vocabulary."""

    projects = [value for value in spec.PROJECTS if value not in used_projects]
    rng.shuffle(projects)
    databases = [value for value in spec.DATABASES if value not in used_values]
    rng.shuffle(databases)
    periods = list(spec.RETENTION_PERIODS)

    planned: list[PlannedFact] = []
    count = min(count, len(projects))
    for offset, position in enumerate(_spread(count)):
        project = projects[offset % len(projects)]
        database = databases[offset % len(databases)]
        period = periods[offset % len(periods)]
        text = spec.distractor_fact_statement(rng, project, database, period)
        fact_id = f"fact-{start_index + offset}"
        marker = f"Project {project}"
        planned.append(
            PlannedFact(
                fact=BenchmarkFact(
                    fact_id=fact_id,
                    text=text,
                    fact_type="irrelevant",
                    markers=(marker, database if database in text else period),
                    subject=marker,
                    value="",
                ),
                user_text=text,
                ack=rng.choice(spec.FACT_ACKS),
                position=position,
                role="irrelevant",
            )
        )
    return planned


def _base(rng: random.Random, target_tokens: int, *, factor: float = 1.0) -> dict[str, Any]:
    """Draw the answer fact's entities and the distractors for one task."""

    project = rng.choice(spec.PROJECTS)
    database = rng.choice(spec.DATABASES)
    return {
        "project": project,
        "database": database,
        "distractors": distractor_count(target_tokens, factor=factor),
        "used_projects": {project},
        "used_values": {database},
    }


def _plan_single_hop(rng: random.Random, target_tokens: int) -> TaskPlan:
    """A: one stored fact answers the question directly, plus a repetition."""

    base = _base(rng, target_tokens)
    project, database = base["project"], base["database"]
    marker = f"Project {project}"
    main = BenchmarkFact(
        fact_id="fact-1",
        text=spec.database_statement(project, database),
        fact_type="durable",
        markers=(marker, database),
        subject=marker,
        value=database,
    )
    repeat = BenchmarkFact(
        fact_id="fact-2",
        text=spec.repeated_database_statement(project, database),
        fact_type="repeated",
        markers=(marker, database),
        subject=marker,
        value=database,
    )
    planned = [
        PlannedFact(main, main.text, rng.choice(spec.FACT_ACKS), 0.2, "required"),
        PlannedFact(repeat, repeat.text, rng.choice(spec.FACT_ACKS), 0.5, "supporting"),
        *_distractors(
            rng,
            base["distractors"],
            used_projects=base["used_projects"],
            used_values=base["used_values"],
            start_index=3,
        ),
    ]
    return TaskPlan(
        category=spec.CATEGORY_SINGLE_HOP,
        planned=tuple(planned),
        question=spec.database_question(project),
        expected_answer=database,
        acceptable=spec.answers_for(database),
        required=("fact-1",),
        supporting=("fact-2",),
        notes={"repeated_fact": True, "distractor_facts": base["distractors"]},
    )


def _plan_multi_hop(rng: random.Random, target_tokens: int) -> TaskPlan:
    """B: two facts stated far apart, neither sufficient alone."""

    base = _base(rng, target_tokens)
    project, database = base["project"], base["database"]
    regions = [value for value in spec.REGIONS]
    rng.shuffle(regions)
    region = regions[0]
    marker = f"Project {project}"

    link = BenchmarkFact(
        fact_id="fact-1",
        text=spec.multi_hop_database_statement(project, database),
        fact_type="durable",
        markers=(marker, database),
        subject=f"telemetry service of {marker}",
        value=database,
    )
    location = BenchmarkFact(
        fact_id="fact-2",
        text=spec.region_statement(project, region),
        fact_type="durable",
        markers=(marker, region),
        subject=marker,
        value=region,
    )
    planned = [
        PlannedFact(link, link.text, rng.choice(spec.FACT_ACKS), 0.12, "required"),
        PlannedFact(location, location.text, rng.choice(spec.FACT_ACKS), 0.74, "required"),
        *_distractors(
            rng,
            base["distractors"],
            used_projects=base["used_projects"],
            used_values={database, region},
            start_index=3,
        ),
    ]
    return TaskPlan(
        category=spec.CATEGORY_MULTI_HOP,
        planned=tuple(planned),
        question=spec.multi_hop_question(database),
        expected_answer=region,
        acceptable=spec.answers_for(region),
        required=("fact-1", "fact-2"),
        notes={"hops": 2, "distractor_facts": base["distractors"]},
    )


def _plan_temporal(rng: random.Random, target_tokens: int) -> TaskPlan:
    """C: the older value was replaced; the newer one is correct."""

    base = _base(rng, target_tokens)
    project = base["project"]
    databases = [value for value in spec.DATABASES if value != base["database"]]
    rng.shuffle(databases)
    old_database, new_database = base["database"], databases[0]
    marker = f"Project {project}"

    older = BenchmarkFact(
        fact_id="fact-1",
        text=spec.earlier_database_statement(project, old_database),
        fact_type="temporal",
        markers=(marker, old_database),
        subject=marker,
        value=old_database,
        superseded_by="fact-2",
    )
    newer = BenchmarkFact(
        fact_id="fact-2",
        text=spec.migration_statement(project, old_database, new_database),
        fact_type="temporal",
        markers=(marker, new_database),
        subject=marker,
        value=new_database,
        supersedes="fact-1",
    )
    planned = [
        PlannedFact(older, older.text, rng.choice(spec.FACT_ACKS), 0.18, "forbidden"),
        PlannedFact(newer, newer.text, rng.choice(spec.FACT_ACKS), 0.72, "required"),
        *_distractors(
            rng,
            base["distractors"],
            used_projects=base["used_projects"],
            used_values={old_database, new_database},
            start_index=3,
        ),
    ]
    return TaskPlan(
        category=spec.CATEGORY_TEMPORAL,
        planned=tuple(planned),
        question=spec.migration_question(project),
        expected_answer=new_database,
        acceptable=spec.answers_for(new_database),
        required=("fact-2",),
        forbidden=("fact-1",),
        notes={"superseded_fact": "fact-1", "distractor_facts": base["distractors"]},
    )


def _plan_conflict(rng: random.Random, target_tokens: int) -> TaskPlan:
    """D: the user explicitly corrects an earlier statement."""

    base = _base(rng, target_tokens)
    project = base["project"]
    days = list(spec.DEPLOY_DAYS)
    times = list(spec.DEPLOY_TIMES)
    rng.shuffle(days)
    rng.shuffle(times)
    old_day, new_day = days[0], days[1]
    old_time, new_time = times[0], times[1]
    marker = f"Project {project}"

    older = BenchmarkFact(
        fact_id="fact-1",
        text=spec.deploy_statement(project, old_day, old_time),
        fact_type="temporal",
        markers=(marker, old_day, old_time),
        subject=marker,
        value=f"{old_day} at {old_time} UTC",
        superseded_by="fact-2",
    )
    newer = BenchmarkFact(
        fact_id="fact-2",
        text=spec.deploy_correction(project, new_day, new_time),
        fact_type="correction",
        markers=(marker, new_day, new_time),
        subject=marker,
        value=f"{new_day} at {new_time} UTC",
        supersedes="fact-1",
    )
    planned = [
        PlannedFact(older, older.text, rng.choice(spec.FACT_ACKS), 0.16, "forbidden"),
        PlannedFact(newer, newer.text, rng.choice(spec.FACT_ACKS), 0.7, "required"),
        *_distractors(
            rng,
            base["distractors"],
            used_projects=base["used_projects"],
            used_values={old_day, new_day},
            start_index=3,
        ),
    ]
    return TaskPlan(
        category=spec.CATEGORY_CONFLICT,
        planned=tuple(planned),
        question=spec.deploy_question(project),
        expected_answer=newer.value,
        acceptable=(newer.value, f"{new_day} {new_time}", new_day),
        required=("fact-2",),
        forbidden=("fact-1",),
        notes={"correction_fact": "fact-2", "distractor_facts": base["distractors"]},
    )


def _plan_distractor(rng: random.Random, target_tokens: int) -> TaskPlan:
    """E: the fact is buried among many irrelevant durable facts."""

    base = _base(rng, target_tokens)
    project, database = base["project"], base["database"]
    marker = f"Project {project}"
    main = BenchmarkFact(
        fact_id="fact-1",
        text=spec.database_statement(project, database),
        fact_type="durable",
        markers=(marker, database),
        subject=marker,
        value=database,
    )
    planned = [
        PlannedFact(main, main.text, rng.choice(spec.FACT_ACKS), 0.08, "required"),
        *_distractors(
            rng,
            3 * base["distractors"],
            used_projects=base["used_projects"],
            used_values=base["used_values"],
            start_index=2,
        ),
    ]
    return TaskPlan(
        category=spec.CATEGORY_DISTRACTOR,
        planned=tuple(planned),
        question=spec.database_question(project),
        expected_answer=database,
        acceptable=spec.answers_for(database),
        required=("fact-1",),
        notes={
            "distractor_facts": 3 * base["distractors"],
            "fact_position": 0.08,
        },
    )


def _plan_cross_session(rng: random.Random, target_tokens: int) -> TaskPlan:
    """F: a fact from an earlier session is needed in a later one."""

    base = _base(rng, target_tokens)
    project = base["project"]
    regions = list(spec.REGIONS)
    rng.shuffle(regions)
    region = regions[0]
    marker = f"Project {project}"

    fact = BenchmarkFact(
        fact_id="fact-1",
        text=spec.datacentre_statement(project, region),
        fact_type="durable",
        markers=(marker, region),
        subject=marker,
        value=region,
        session_index=0,
    )
    planned = [
        PlannedFact(fact, fact.text, rng.choice(spec.FACT_ACKS), 0.15, "required"),
        *_distractors(
            rng,
            base["distractors"],
            used_projects=base["used_projects"],
            used_values={region},
            start_index=2,
        ),
    ]
    return TaskPlan(
        category=spec.CATEGORY_CROSS_SESSION,
        planned=tuple(planned),
        question=spec.cross_session_question(project),
        expected_answer=region,
        acceptable=spec.answers_for(region),
        required=("fact-1",),
        session_count=2,
        session_fraction=0.6,
        notes={"session_boundary_count": 1, "distractor_facts": base["distractors"]},
    )


def _plan_abstention(rng: random.Random, target_tokens: int) -> TaskPlan:
    """G: nothing in the conversation answers the question."""

    base = _base(rng, target_tokens)
    project = base["project"]
    topics = [topic for topic in spec.PROBE_TOPICS]
    rng.shuffle(topics)
    topic = topics[0]
    absent = BenchmarkFact(
        fact_id="fact-1",
        text="",
        fact_type="absent",
        markers=(),
        subject=f"the {topic} for Project {project}",
        value="",
    )
    planned = [
        *_distractors(
            rng,
            max(2, base["distractors"]),
            used_projects=base["used_projects"],
            used_values=base["used_values"],
            start_index=2,
        ),
    ]
    return TaskPlan(
        category=spec.CATEGORY_ABSTENTION,
        planned=tuple(planned),
        question=spec.abstention_question(project, topic),
        expected_answer="",
        abstention=True,
        probe_topic=topic,
        extra_facts=(absent,),
        notes={"probe_topic": topic, "distractor_facts": len(planned)},
    )


_PLANNERS = {
    spec.CATEGORY_SINGLE_HOP: _plan_single_hop,
    spec.CATEGORY_MULTI_HOP: _plan_multi_hop,
    spec.CATEGORY_TEMPORAL: _plan_temporal,
    spec.CATEGORY_CONFLICT: _plan_conflict,
    spec.CATEGORY_DISTRACTOR: _plan_distractor,
    spec.CATEGORY_CROSS_SESSION: _plan_cross_session,
    spec.CATEGORY_ABSTENTION: _plan_abstention,
}


# ---------------------------------------------------------------------- #
# Assembly
# ---------------------------------------------------------------------- #


def _filler_block(rng: random.Random, index: int) -> tuple[str, str]:
    user = spec.FILLER_USER_TURNS[index % len(spec.FILLER_USER_TURNS)].format(index=index)
    offset = rng.randrange(len(spec.FILLER_ASSISTANT_TURNS))
    assistant = spec.FILLER_ASSISTANT_TURNS[
        (index + offset) % len(spec.FILLER_ASSISTANT_TURNS)
    ].format(index=index)
    return user, assistant


def _slot_map(planned: tuple[PlannedFact, ...], slots: int) -> dict[int, PlannedFact]:
    """Assign each fact a distinct slot, preserving position order."""

    assigned: dict[int, PlannedFact] = {}
    previous = -1
    for item in sorted(planned, key=lambda entry: entry.position):
        slot = min(slots - 1, max(previous + 1, round(item.position * slots)))
        assigned[slot] = item
        previous = slot
    return assigned


def _achieved_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(str(message.get("content", ""))) for message in messages)


def _assemble(
    plan: TaskPlan,
    *,
    task_id: str,
    seed: int,
    variant: int,
    target_tokens: int,
) -> BenchmarkTask:
    """Build the transcript, then trim or extend it toward the target length."""

    rng = random.Random(f"{seed}:{plan.category}:{target_tokens}:{variant}:{task_id}")
    filler_index = 0
    probe_user, probe_assistant = _filler_block(rng, filler_index)
    block_tokens = max(
        1, estimate_tokens(probe_user) + estimate_tokens(probe_assistant)
    )
    slots = max(len(plan.planned) + 2, round(target_tokens / block_tokens))
    fact_slots = _slot_map(plan.planned, slots)

    messages: list[dict[str, Any]] = []
    recorded_turn: dict[str, int] = {}
    for slot in range(slots):
        item = fact_slots.get(slot)
        if item is not None:
            recorded_turn[item.fact.fact_id] = len(messages)
            messages.append({"role": "user", "content": item.user_text})
            messages.append({"role": "assistant", "content": item.ack})
            continue
        user, assistant = _filler_block(rng, filler_index)
        filler_index += 1
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    # Grow toward the target, then trim trailing filler back toward it. Facts are
    # never trimmed: the ledger says where they are and the transcript must agree.
    guard = 0
    while _achieved_tokens(messages) < target_tokens * _LENGTH_FLOOR and guard < 5000:
        user, assistant = _filler_block(rng, filler_index)
        filler_index += 1
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
        guard += 1
    while (
        _achieved_tokens(messages) > target_tokens * _LENGTH_CEILING
        and len(messages) > 4
        and messages[-2].get("role") == "user"
        and messages[-2]["content"] not in {item.fact.text for item in plan.planned}
    ):
        messages.pop()
        messages.pop()

    # Keep the question away from its evidence: a two-turn gap is the minimum a
    # sliding window must still miss.
    last_fact_turn = max(recorded_turn.values(), default=0)
    while len(messages) - last_fact_turn < 4:
        user, assistant = _filler_block(rng, filler_index)
        filler_index += 1
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    session_boundaries: list[int] = []
    if plan.session_count > 1:
        boundary = max(1, round(plan.session_fraction * len(messages)))
        while boundary < len(messages) and messages[boundary].get("role") != "user":
            boundary += 1
        session_boundaries.append(boundary)
        for message in messages[boundary:]:
            message["session"] = 1

    facts = tuple(
        replace(item.fact, introduced_turn=recorded_turn.get(item.fact.fact_id, 0))
        for item in plan.planned
    ) + tuple(plan.extra_facts)

    required_fact = next(
        (item.fact for item in plan.planned if item.role == "required"), None
    )
    last_fact_turn = max(recorded_turn.values(), default=0)
    achieved = _achieved_tokens(messages)
    filler_messages = len(messages) - 2 * len(plan.planned)
    metadata: dict[str, Any] = {
        "generator_version": spec.GENERATOR_VERSION,
        "seed": seed,
        "variant": variant,
        "target_tokens": target_tokens,
        "achieved_tokens": achieved,
        "length_tier": target_tokens,
        "category_label": spec.CATEGORY_LABELS[plan.category],
        "category_description": spec.CATEGORY_DESCRIPTIONS[plan.category],
        "filler_messages": filler_messages,
        "filler_turns": max(0, filler_messages // 2),
        "planted_facts": len(plan.planned),
        "fact_turns": dict(sorted(recorded_turn.items())),
        "distance_turns": len(messages) - last_fact_turn - 1,
        "session_boundaries": session_boundaries,
        "token_counter": "estimate_tokens",
        **plan.notes,
    }
    if required_fact is not None:
        metadata["required_fact_turn"] = recorded_turn.get(required_fact.fact_id, 0)
        metadata["required_fact_distance_turns"] = (
            len(messages) - metadata["required_fact_turn"] - 1
        )
    if plan.probe_topic:
        metadata["probe_topic"] = plan.probe_topic

    task = BenchmarkTask(
        task_id=task_id,
        category=plan.category,
        conversation=messages,
        question=plan.question,
        expected_answer=plan.expected_answer,
        required_fact_ids=plan.required,
        conversation_length=len(messages),
        facts=facts,
        evidence=EvidenceContract(
            required=plan.required,
            supporting=plan.supporting,
            forbidden=plan.forbidden,
            abstention_expected=plan.abstention,
        ),
        acceptable_answers=plan.acceptable,
        session_count=plan.session_count,
        metadata=metadata,
    )
    problems = validate_task(task)
    if problems:
        raise ValueError(f"{task_id}: generated task is invalid: {'; '.join(problems)}")
    return task


def generate_task(
    category: str,
    target_tokens: int,
    *,
    seed: int = spec.DEFAULT_SEED,
    variant: int = 0,
) -> BenchmarkTask:
    """Generate one task for a category at a target estimated length."""

    if category not in _PLANNERS:
        raise ValueError(
            f"Unknown category {category!r}. Expected one of {', '.join(spec.CATEGORIES)}."
        )
    if target_tokens <= 0:
        raise ValueError("target_tokens must be greater than zero.")
    rng = random.Random(f"{seed}:{category}:{target_tokens}:{variant}")
    plan = _PLANNERS[category](rng, target_tokens)
    task_id = f"cr-{category.replace('_', '-')}-{target_tokens}-{variant:02d}"
    return _assemble(
        plan, task_id=task_id, seed=seed, variant=variant, target_tokens=target_tokens
    )


def generate_dataset(
    categories: tuple[str, ...] = spec.CATEGORIES,
    lengths: tuple[int, ...] = spec.TIERS[spec.DEFAULT_TIER],
    *,
    seed: int = spec.DEFAULT_SEED,
    variants: int = 1,
) -> list[BenchmarkTask]:
    """Generate the cartesian product of categories, lengths, and variants."""

    tasks: list[BenchmarkTask] = []
    for category in categories:
        for length in lengths:
            for variant in range(variants):
                tasks.append(
                    generate_task(category, length, seed=seed, variant=variant)
                )
    return tasks


def dataset_sha256(tasks: list[BenchmarkTask]) -> str:
    """Hash the dataset as it will be written, so the manifest can pin it."""

    digest = hashlib.sha256()
    for task in tasks:
        digest.update(json.dumps(task.to_dict(), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    tasks: list[BenchmarkTask],
    *,
    tier: str,
    lengths: tuple[int, ...],
    categories: tuple[str, ...],
    variants: int,
    seed: int,
) -> dict[str, Any]:
    """Describe a generated dataset well enough to reproduce and cite it."""

    return {
        "generator_version": spec.GENERATOR_VERSION,
        "tier": tier,
        "lengths": list(lengths),
        "categories": list(categories),
        "variants": variants,
        "seed": seed,
        "task_count": len(tasks),
        "token_counter": "estimate_tokens",
        "dataset_sha256": dataset_sha256(tasks),
        "category_counts": {
            category: sum(1 for task in tasks if task.category == category)
            for category in categories
        },
        "mean_achieved_tokens": round(
            sum(int(task.metadata.get("achieved_tokens", 0)) for task in tasks)
            / len(tasks),
            1,
        )
        if tasks
        else 0.0,
        "regenerate": (
            "python benchmarks/context_rot/generation.py "
            f"--tier {tier} --variants {variants} --seed {seed}"
        ),
    }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Phase 7 context-rot benchmark dataset."
    )
    parser.add_argument("--tier", choices=sorted(spec.TIERS), default=spec.DEFAULT_TIER)
    parser.add_argument(
        "--lengths",
        help="Comma-separated target token lengths, overriding the tier.",
    )
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated category names, or 'all'.",
    )
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=spec.DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/context_rot/dataset.jsonl")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Where to write the run manifest. Defaults to MANIFEST.json beside "
            "--output, so regenerating another tier (e.g. into "
            "benchmarks/context_rot/generated/) cannot overwrite the committed "
            "smoke-tier manifest."
        ),
    )
    return parser


def _parse_categories(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return spec.CATEGORIES
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [name for name in names if name not in spec.CATEGORIES]
    if unknown:
        raise SystemExit(
            f"Unknown categories: {', '.join(unknown)}. "
            f"Expected one of {', '.join(spec.CATEGORIES)} or 'all'."
        )
    return names


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lengths = spec.TIERS[args.tier]
    if args.lengths:
        lengths = tuple(int(part) for part in args.lengths.split(",") if part.strip())
    if args.variants < 1:
        raise SystemExit("--variants must be at least 1.")

    categories = _parse_categories(args.categories)
    tasks = generate_dataset(
        categories=categories,
        lengths=lengths,
        seed=args.seed,
        variants=args.variants,
    )
    write_jsonl(args.output, tasks)
    manifest = build_manifest(
        tasks,
        tier=args.tier if not args.lengths else "custom",
        lengths=lengths,
        categories=categories,
        variants=args.variants,
        seed=args.seed,
    )
    # Default the manifest beside the dataset it describes: regenerating a
    # non-default tier must not overwrite the committed smoke-tier manifest.
    manifest_path = args.manifest or args.output.parent / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"{len(tasks)} tasks → {args.output} "
        f"(tier={manifest['tier']}, mean≈{manifest['mean_achieved_tokens']} tokens, "
        f"sha256={manifest['dataset_sha256'][:12]}…)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
