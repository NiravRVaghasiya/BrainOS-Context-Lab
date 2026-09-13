"""Phase 3 adapter mapping: lifecycle fields, conflicts, and retrieval signals."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from brain.adapter import (
    BrainOSAdapter,
    Conflict,
    MemoryRecord,
    _normalize_key,
)
from brain.memory_policy import MemoryPolicy, MemoryType
from tests.fakes import FakeRuntime

OBSERVED = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)


class DictMemoryRuntime:
    """Runtime double that returns dict memories carrying lifecycle fields."""

    def __init__(self, memories: list[dict[str, Any]]) -> None:
        self._memories = memories
        self.recall_calls: list[dict[str, Any]] = []

    def observe(
        self, event: Any, *, source: str = "user", event_type: Any = "user_message"
    ) -> dict:
        return {"stored_ids": ["new"]}

    def recall(self, query: str, top_k: int = 8) -> list[str]:
        self.recall_calls.append({"query": query, "top_k": top_k})
        return [str(memory["content"]) for memory in self._memories[:top_k]]

    def active_memories(self) -> list[dict[str, Any]]:
        return list(self._memories)

    def contradictions(self) -> list[Any]:
        return []

    def stale_memories(self) -> list[Any]:
        return []


def rich_memory(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "mem-1",
        "content": "The production database is PostgreSQL 16.",
        "type": "episodic",
        "status": "active",
        "source": "user",
        "entities": ["PostgreSQL", "production"],
        "created_at": OBSERVED,
        "observed_at": OBSERVED,
        "valid_from": None,
        "valid_until": None,
        "confidence": 0.9,
        "salience": 0.55,
        "utility": 0.25,
        "access_count": 3,
        "supersedes": None,
        "contradicts": [],
    }
    payload.update(overrides)
    return payload


def test_recall_records_carry_runtime_lifecycle_fields() -> None:
    runtime = DictMemoryRuntime(
        [
            rich_memory(),
            rich_memory(
                id="mem-2",
                content="The staging database is MySQL 8.",
                valid_until=OBSERVED + timedelta(days=30),
                supersedes="mem-1",
                contradicts=["mem-1"],
                status="superseded",
            ),
        ]
    )
    adapter = BrainOSAdapter(runtime)
    records = adapter.recall("Which database do we use?", limit=5)

    first, second = records
    assert first.memory_id == "mem-1"
    assert first.entities == ("PostgreSQL", "production")
    assert first.confidence == 0.9
    assert first.salience == 0.55
    assert first.utility == 0.25
    assert first.retrieval_count == 3
    assert first.observed_at == OBSERVED.isoformat()
    assert first.valid_until is None
    assert first.status == "active"
    assert second.supersedes == ("mem-1",)
    assert second.contradicts == ("mem-1",)
    assert second.status == "superseded"
    assert second.valid_until == (OBSERVED + timedelta(days=30)).isoformat()


def test_missing_timestamps_stay_unknown() -> None:
    """Fabricating ``now`` would make every memory look maximally recent."""

    runtime = DictMemoryRuntime([rich_memory(created_at=None, observed_at=None)])
    record = BrainOSAdapter(runtime).recall("database")[0]
    assert record.created_at == ""
    assert record.observed_at == ""


def test_recall_from_plain_strings_still_produces_usable_records() -> None:
    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime)
    adapter.observe("The production database is PostgreSQL 16.")
    record = adapter.recall("Which database?")[0]
    assert record.text == "The production database is PostgreSQL 16."
    assert record.memory_id == "session-a-m1"
    assert record.entities == ()


def test_conflicts_maps_runtime_contradictions() -> None:
    class ContradictionRuntime(FakeRuntime):
        def contradictions(self) -> list[Any]:
            return [
                {
                    "subject": "production database",
                    "older_id": "old-1",
                    "newer_id": "new-1",
                    "older_content": "The production database is PostgreSQL 16.",
                    "newer_content": "The production database is MySQL 8.",
                }
            ]

    conflicts = BrainOSAdapter(ContradictionRuntime()).conflicts()
    assert conflicts == [
        Conflict(
            subject="production database",
            older_id="old-1",
            newer_id="new-1",
            older_text="The production database is PostgreSQL 16.",
            newer_text="The production database is MySQL 8.",
            source="runtime",
        )
    ]


def test_conflicts_maps_object_shaped_contradictions() -> None:
    class Contradiction:
        subject = "deployment window"
        older_id = "a"
        newer_id = "b"
        older_content = "Deployments are Friday."
        newer_content = "Deployments are Saturday."

    class Runtime(FakeRuntime):
        def contradictions(self) -> list[Any]:
            return [Contradiction()]

    conflicts = BrainOSAdapter(Runtime()).conflicts()
    assert conflicts[0].subject == "deployment window"
    assert conflicts[0].older_text == "Deployments are Friday."


def test_conflicts_is_empty_when_the_runtime_has_no_support() -> None:
    class MinimalRuntime:
        def recall(self, query: str, top_k: int = 8) -> list[str]:
            return []

    adapter = BrainOSAdapter(MinimalRuntime())
    assert adapter.conflicts() == []
    assert adapter.stale_memory_ids() == set()


def test_conflicts_survives_a_runtime_error() -> None:
    class BrokenRuntime(FakeRuntime):
        def contradictions(self) -> list[Any]:
            raise RuntimeError("temporal index unavailable")

        def stale_memories(self) -> list[Any]:
            raise RuntimeError("temporal index unavailable")

    adapter = BrainOSAdapter(BrokenRuntime())
    assert adapter.conflicts() == []
    assert adapter.stale_memory_ids() == set()


def test_stale_memory_ids_are_collected() -> None:
    runtime = FakeRuntime(stale=["session-a-m1"])
    adapter = BrainOSAdapter(runtime)
    adapter.observe("The production database is PostgreSQL 16.")
    assert adapter.stale_memory_ids() == {"session-a-m1"}


def test_enrich_with_explanation_attaches_scores_and_signals() -> None:
    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime)
    adapter.observe("The production database is PostgreSQL 16.")
    records = adapter.recall("Which database?")
    explanation = adapter.explain("Which database?")

    enriched = adapter.enrich_with_explanation(records, explanation)
    assert enriched[0].relevance == 0.91
    assert enriched[0].signals == {"lexical": 1.0}
    # Salience is the runtime's query-independent signal and is left untouched
    # when the explanation does not report one.
    assert enriched[0].salience == 0.8
    # The original records are not mutated in place.
    assert records[0].signals == {}


def test_enrich_with_explanation_matches_by_content_when_ids_differ() -> None:
    records = [
        MemoryRecord(memory_id="unknown", text="Production uses PostgreSQL 16.", created_at="")
    ]
    explanation = {
        "selected": [
            {
                "id": "runtime-id",
                "content": "production  uses PostgreSQL 16.",
                "score": 2.5,
                "signals": {"lexical": 0.4, "semantic": 0.0},
            }
        ]
    }
    enriched = BrainOSAdapter(None).enrich_with_explanation(records, explanation)
    assert enriched[0].relevance == 2.5
    assert enriched[0].signals == {"lexical": 0.4, "semantic": 0.0}


def test_enrich_with_explanation_is_a_noop_without_selected_entries() -> None:
    records = [MemoryRecord(memory_id="a", text="x")]
    adapter = BrainOSAdapter(None)
    assert adapter.enrich_with_explanation(records, {}) == records
    assert adapter.enrich_with_explanation(records, {"selected": []}) == records


def test_explanation_never_carries_secret_named_signals() -> None:
    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime)
    adapter.observe("The production database is PostgreSQL 16.")
    explanation = adapter.explain("Which database?")
    assert "api_key" not in explanation
    assert "token" not in explanation
    assert "should-never-leak" not in str(explanation)


def test_observation_bookkeeping_restores_type_and_turn() -> None:
    """BrainOS stores content; the application remembers its own classification."""

    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime, policy=MemoryPolicy())
    adapter.observe(
        "Actually, we migrated the production database to MySQL 8.",
        metadata={
            "role": "user",
            "memory_type": MemoryType.CORRECTION.value,
            "confidence": 0.92,
            "turn": 7,
        },
    )
    record = adapter.recall("Which database do we use?")[0]
    assert record.memory_type == "CORRECTION"
    assert record.type_source == "policy"
    assert record.source_turn == 7
    assert record.observed_at != ""
    # A confidence reported by the runtime always wins over the policy's.
    assert record.confidence == 0.9


def test_policy_confidence_is_used_only_when_the_runtime_reports_none() -> None:
    runtime = DictMemoryRuntime([rich_memory(confidence=None)])
    adapter = BrainOSAdapter(runtime, policy=MemoryPolicy())
    adapter.observe(
        "The production database is PostgreSQL 16.",
        metadata={"memory_type": MemoryType.PROJECT_STATE.value, "confidence": 0.9, "turn": 2},
    )
    record = adapter.recall("Which database do we use?")[0]
    assert record.confidence == 0.9
    assert record.source_turn == 2


def test_observation_bookkeeping_never_overrides_a_specific_runtime_type() -> None:
    runtime = DictMemoryRuntime([rich_memory(type="preference")])
    adapter = BrainOSAdapter(runtime, policy=MemoryPolicy())
    adapter.observe(
        "The production database is PostgreSQL 16.",
        metadata={"memory_type": MemoryType.CORRECTION.value, "turn": 3},
    )
    record = adapter.recall("Which database do we use?")[0]
    assert record.memory_type == "PREFERENCE"
    assert record.type_source == "runtime"


def test_list_memories_reports_the_same_enriched_records() -> None:
    runtime = DictMemoryRuntime([rich_memory()])
    adapter = BrainOSAdapter(runtime)
    listed = adapter.list_memories()
    assert listed[0].entities == ("PostgreSQL", "production")
    assert listed[0].status == "active"


def test_normalize_key_is_punctuation_and_case_insensitive() -> None:
    assert _normalize_key("  The   Database IS PostgreSQL ") == "the database is postgresql"
