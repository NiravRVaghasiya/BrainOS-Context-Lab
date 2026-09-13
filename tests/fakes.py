"""Shared test doubles for the application boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeMemory:
    id: str
    content: str
    type: str = "semantic"
    source: str = "user"
    salience: float = 0.8
    confidence: float = 0.9
    access_count: int = 0
    status: str = "active"


class FakeRuntime:
    """Injected runtime whose signatures match the pinned BrainOS facade.

    ``contradictions`` and ``stale`` let tests exercise the Phase 3 conflict and
    staleness paths without the live runtime; both default to empty so the fake
    behaves like a runtime that reports nothing.
    """

    def __init__(
        self,
        session_id: str = "session-a",
        actor_id: str = "actor-a",
        *,
        contradictions: list[dict[str, Any]] | None = None,
        stale: list[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.actor_id = actor_id
        self.stored: list[FakeMemory] = []
        self.observe_calls: list[dict[str, Any]] = []
        self.recall_calls: list[dict[str, Any]] = []
        self.cycles: list[dict[str, Any]] = []
        self.why_calls: list[str] = []
        self.contradiction_records: list[dict[str, Any]] = list(contradictions or [])
        self.stale_ids: list[str] = list(stale or [])

    def observe(
        self, event: Any, *, source: str = "user", event_type: Any = "user_message"
    ) -> dict[str, Any]:
        text = event if isinstance(event, str) else str(event)
        memory = FakeMemory(
            id=f"{self.session_id}-m{len(self.stored) + 1}",
            content=text,
            source=source,
        )
        self.stored.append(memory)
        self.observe_calls.append(
            {"event": event, "source": source, "event_type": event_type}
        )
        record = {
            "event_type": str(getattr(event_type, "value", event_type)),
            "tokens": "10 -> 10",
            "retrieved": [],
            "candidates_considered": 0,
            "stored_ids": [memory.id],
            "steps": ["OBSERVE", "FILTER", "RETRIEVE"],
            "confidence": 0.4,
            "decision": "ask",
        }
        self.cycles.append(record)
        return record

    def recall(self, query: str, top_k: int = 8) -> list[str]:
        self.recall_calls.append({"query": query, "top_k": top_k})
        tokens = [token.lower() for token in query.split() if len(token) > 3]
        contents = [
            memory.content
            for memory in self.stored
            if not tokens or any(token in memory.content.lower() for token in tokens)
        ]
        if not contents:
            contents = [memory.content for memory in self.stored]
        return contents[:top_k]

    def decide(self, query: str) -> str:
        return "act" if self.stored else "ask"

    def assess(self, query: str) -> dict[str, Any]:
        decision = self.decide(query)
        return {
            "decision": decision,
            "confidence": 0.82 if self.stored else 0.1,
            "rationale": (
                ["matched stored memory"]
                if self.stored
                else ["no relevant memories retrieved"]
            ),
        }

    def why(self, query: str) -> dict[str, Any]:
        self.why_calls.append(query)
        assessment = self.assess(query)
        return {
            "query": query,
            "selected": [
                {
                    "id": memory.id,
                    "content": memory.content,
                    "score": 0.91,
                    "signals": {"lexical": 1.0},
                    "source": memory.source,
                    "api_key": "should-never-leak",
                }
                for memory in self.stored
            ],
            "decision": assessment["decision"],
            "confidence": assessment["confidence"],
            "rationale": assessment["rationale"],
            "token": "secret-token",
        }

    def trace(self, *, formatted: bool = False) -> dict[str, Any] | str:
        payload = {
            "session_id": self.session_id,
            "cycles": len(self.cycles),
            "records": list(self.cycles),
            "state": {
                "session_id": self.session_id,
                "actor_id": self.actor_id,
                "turn": len(self.cycles),
            },
        }
        if formatted:
            return f"TRACE session={self.session_id} cycles={len(self.cycles)}"
        return payload

    def contradictions(self) -> list[dict[str, Any]]:
        return list(self.contradiction_records)

    def stale_memories(self) -> list[FakeMemory]:
        return [memory for memory in self.stored if memory.id in self.stale_ids]

    def active_memories(self) -> list[FakeMemory]:
        return list(self.stored)

    def memories(self) -> list[FakeMemory]:
        return list(self.stored)


class FakeProvider:
    """Deterministic provider used by application-service tests."""

    def __init__(self, text: str = "You use PostgreSQL 16.") -> None:
        self.text = text
        self.requests: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        from providers.base import ProviderResponse

        self.requests.append(messages)
        return ProviderResponse(text=self.text, model="fake-model", usage={"total_tokens": 4})


class LooseFakeRuntime(FakeRuntime):
    """Fake whose recall is deliberately imprecise but whose ``why()`` is honest.

    Used by security and pipeline tests that need irrelevant candidates to reach
    the context engine, which is what the relevance filter exists for.
    """

    def recall(self, query: str, top_k: int = 8) -> list[str]:
        self.recall_calls.append({"query": query, "top_k": top_k})
        return [memory.content for memory in self.stored[:top_k]]

    def why(self, query: str) -> dict[str, Any]:
        query_tokens = {token for token in query.lower().split() if len(token) > 2}
        selected = []
        for memory in self.stored:
            memory_tokens = {
                token.strip(".,!?") for token in memory.content.lower().split() if len(token) > 2
            }
            union = query_tokens | memory_tokens
            lexical = len(query_tokens & memory_tokens) / len(union) if union else 0.0
            selected.append(
                {
                    "id": memory.id,
                    "content": memory.content,
                    "score": round(3.0 * lexical, 4),
                    "signals": {"lexical": round(lexical, 4), "semantic": 0.0},
                }
            )
        return {"query": query, "selected": selected, "decision": "retrieve", "confidence": 0.5}
