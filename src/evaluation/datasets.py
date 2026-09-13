"""Benchmark task schema and JSONL loading helpers.

Phase 7 turned this module from a scorer-free record carrier into the benchmark's
data contract. A task is no longer just a transcript plus a question: it carries
a **fact ledger** (every fact the conversation planted, where it was planted, and
how it relates to the others) and an **evidence contract** (which facts a correct
answer needs, which facts are stale and must not be used, and whether the only
correct behaviour is to abstain).

The ledger is what makes the benchmark scorable. The runtime mints its own
memory ids during a replay, so a task cannot refer to them; instead every fact
declares distinctive ``markers`` that can be found in retrieved evidence, in a
prompt, or in a model answer. :mod:`evaluation.scoring` is the only consumer of
that convention.

Nothing in this module imports BrainOS, a provider, or the benchmark generator:
the schema is the stable interface between the dataset producer
(``benchmarks/context_rot``) and every consumer (runner, scorer, tests).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Fact roles used by the Phase 7 context-rot benchmark. They describe how a
#: fact behaves over the conversation's lifetime, which is what the task
#: categories probe.
FACT_TYPES: tuple[str, ...] = (
    "durable",
    "repeated",
    "irrelevant",
    "temporal",
    "correction",
    "absent",
)


@dataclass(frozen=True)
class BenchmarkFact:
    """One fact the benchmark planted in a conversation.

    ``markers`` are the distinctive, case-insensitive substrings that prove this
    fact's evidence is present. All of a fact's markers must appear for the fact
    to count as retrieved, so a marker set usually pairs the subject
    (``Project Cobalt``) with the value (``eu-west-1``).

    ``supersedes`` / ``superseded_by`` record replacement chains: a temporal or
    correction task has an older fact that is superseded by a newer one. The
    older fact is *not* deleted from the transcript — forgetting it is exactly
    what a context-management strategy has to get right.
    """

    fact_id: str
    text: str
    fact_type: str = "durable"
    introduced_turn: int = 0
    markers: tuple[str, ...] = ()
    subject: str = ""
    value: str = ""
    supersedes: str = ""
    superseded_by: str = ""
    session_index: int = 0

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> BenchmarkFact:
        """Build from a mapping, or return an already-built fact unchanged."""

        if isinstance(item, cls):
            return item
        return cls(
            fact_id=str(item["fact_id"]),
            text=str(item.get("text", "")),
            fact_type=str(item.get("fact_type", "durable")),
            introduced_turn=int(item.get("introduced_turn", 0) or 0),
            markers=tuple(str(marker) for marker in item.get("markers", [])),
            subject=str(item.get("subject", "")),
            value=str(item.get("value", "")),
            supersedes=str(item.get("supersedes", "")),
            superseded_by=str(item.get("superseded_by", "")),
            session_index=int(item.get("session_index", 0) or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "fact_type": self.fact_type,
            "introduced_turn": self.introduced_turn,
            "markers": list(self.markers),
            "subject": self.subject,
            "value": self.value,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "session_index": self.session_index,
        }

    def is_planted(self) -> bool:
        """Whether the fact is actually spoken in the conversation.

        The abstention category needs a fact-shaped question with no evidence
        behind it, so its "absent" fact is a ledger entry, never a turn.
        """

        return self.fact_type != "absent"


@dataclass(frozen=True)
class EvidenceContract:
    """What a correct answer to a task requires.

    ``required`` are the fact ids the answer depends on (the Recall@K
    denominator). ``supporting`` facts are acceptable extra evidence that is not
    necessary. ``forbidden`` facts are stale, corrected, or superseded values
    that must not be presented as current — the conflict-resolution and
    temporal-reasoning signal. ``abstention_expected`` marks tasks where the only
    correct behaviour is to say the information is unavailable.
    """

    required: tuple[str, ...] = ()
    supporting: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    abstention_expected: bool = False

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> EvidenceContract:
        """Build from a mapping, or return an already-built contract unchanged."""

        if isinstance(item, cls):
            return item
        return cls(
            required=tuple(str(value) for value in item.get("required", [])),
            supporting=tuple(str(value) for value in item.get("supporting", [])),
            forbidden=tuple(str(value) for value in item.get("forbidden", [])),
            abstention_expected=bool(item.get("abstention_expected", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": list(self.required),
            "supporting": list(self.supporting),
            "forbidden": list(self.forbidden),
            "abstention_expected": self.abstention_expected,
        }


@dataclass(frozen=True)
class BenchmarkTask:
    """One context-rot task with explicit evidence expectations.

    ``conversation`` messages may carry an optional ``session`` index (default
    ``0``). A replay that honours session isolation starts a fresh session when
    that index changes; the default replay keeps one session, which is the
    long-range memory case the benchmark primarily measures.
    """

    task_id: str
    category: str
    conversation: list[dict[str, Any]]
    question: str
    expected_answer: str
    required_fact_ids: tuple[str, ...] = ()
    conversation_length: int | None = None
    facts: tuple[BenchmarkFact, ...] = ()
    evidence: EvidenceContract = field(default_factory=EvidenceContract)
    acceptable_answers: tuple[str, ...] = ()
    session_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> BenchmarkTask:
        evidence = EvidenceContract.from_dict(item.get("evidence", {}) or {})
        # Older records (and the pre-Phase-7 fixture) used ``required_memory_ids``
        # for the same concept; accept it so a saved dataset keeps loading.
        required = item.get("required_fact_ids", item.get("required_memory_ids", []))
        return cls(
            task_id=str(item["task_id"]),
            category=str(item["category"]),
            conversation=list(item.get("conversation", [])),
            question=str(item["question"]),
            expected_answer=str(item["expected_answer"]),
            required_fact_ids=tuple(str(value) for value in required or evidence.required),
            conversation_length=item.get("conversation_length"),
            facts=tuple(
                BenchmarkFact.from_dict(fact) for fact in item.get("facts", []) or []
            ),
            evidence=evidence,
            acceptable_answers=tuple(
                str(value) for value in item.get("acceptable_answers", [])
            ),
            session_count=int(item.get("session_count", 1) or 1),
            metadata=dict(item.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "conversation": self.conversation,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "required_fact_ids": list(self.required_fact_ids),
            "conversation_length": self.conversation_length,
            "facts": [fact.to_dict() for fact in self.facts],
            "evidence": self.evidence.to_dict(),
            "acceptable_answers": list(self.acceptable_answers),
            "session_count": self.session_count,
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------ #
    # Convenience accessors used by the scorer
    # ------------------------------------------------------------------ #

    def facts_by_id(self) -> dict[str, BenchmarkFact]:
        return {fact.fact_id: fact for fact in self.facts}

    def planted_facts(self) -> tuple[BenchmarkFact, ...]:
        """Facts that appear in the transcript (everything except ``absent``)."""

        return tuple(fact for fact in self.facts if fact.is_planted())

    def required_facts(self) -> tuple[BenchmarkFact, ...]:
        ledger = self.facts_by_id()
        return tuple(ledger[fact_id] for fact_id in self.evidence.required if fact_id in ledger)

    def answers(self) -> tuple[str, ...]:
        """Accepted answer strings, the explicit list first, then the fact values."""

        values: list[str] = []
        for value in (self.expected_answer, *self.acceptable_answers):
            if value and value not in values:
                values.append(value)
        for fact in self.required_facts():
            if fact.value and fact.value not in values:
                values.append(fact.value)
        return tuple(values)


def load_jsonl(path: str | Path) -> list[BenchmarkTask]:
    """Load non-empty JSONL records and reject malformed task definitions."""

    tasks: list[BenchmarkTask] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(BenchmarkTask.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid benchmark record at line {line_number}") from exc
    return tasks


def write_jsonl(path: str | Path, tasks: Iterable[BenchmarkTask]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")


def validate_task(task: BenchmarkTask) -> list[str]:
    """Return the structural problems that make a task unscorable.

    Producer-side checks (category vocabulary, marker distinctness) live with the
    generator; these are the invariants *every* consumer relies on, so the loader
    and the runner can check them without importing the benchmark package.
    """

    problems: list[str] = []
    if not task.task_id:
        problems.append("task_id is empty")
    if not task.question.strip():
        problems.append("question is empty")
    if not task.conversation:
        problems.append("conversation is empty")

    ledger = task.facts_by_id()
    if len(ledger) != len(task.facts):
        problems.append("duplicate fact ids")
    for fact in task.facts:
        if fact.is_planted() and not fact.markers:
            problems.append(f"fact {fact.fact_id} has no markers")
        for related in (fact.supersedes, fact.superseded_by):
            if related and related not in ledger:
                problems.append(f"fact {fact.fact_id} references unknown fact {related}")

    for role in ("required", "supporting", "forbidden"):
        for fact_id in getattr(task.evidence, role):
            if fact_id not in ledger:
                problems.append(f"evidence.{role} references unknown fact {fact_id}")
    if task.evidence.abstention_expected and task.evidence.required:
        problems.append("an abstention task cannot require evidence")

    for index, message in enumerate(task.conversation):
        if not isinstance(message, Mapping):
            problems.append(f"conversation message {index} is not an object")
            continue
        if str(message.get("role", "")) not in ("user", "assistant", "system", "tool"):
            problems.append(f"conversation message {index} has an unknown role")

    if task.session_count > 1:
        sessions = {
            int(message.get("session", 0) or 0)
            for message in task.conversation
            if isinstance(message, Mapping)
        }
        if len(sessions) != task.session_count:
            problems.append(
                f"session_count is {task.session_count} but the transcript marks {len(sessions)}"
            )
    return problems


def dataset_issues(tasks: Iterable[BenchmarkTask]) -> list[str]:
    """Validate a whole dataset, prefixing every problem with its task id."""

    issues: list[str] = []
    tasks = list(tasks)
    if not tasks:
        issues.append("dataset is empty")
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            issues.append(f"{task.task_id}: duplicate task id")
        seen.add(task.task_id)
        issues.extend(f"{task.task_id}: {problem}" for problem in validate_task(task))
    return issues


__all__ = [
    "FACT_TYPES",
    "BenchmarkFact",
    "BenchmarkTask",
    "EvidenceContract",
    "dataset_issues",
    "load_jsonl",
    "validate_task",
    "write_jsonl",
]
