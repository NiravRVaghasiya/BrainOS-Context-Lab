"""Shared test doubles for the application boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from providers.base import ProviderError, ProviderResponse


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


class FakeLLMProvider:
    """Deterministic provider with the full :class:`LLMProvider` surface.

    ``FakeProvider`` covers generation only. The UI controller also lists models
    and validates credentials, so this double records the configuration it was
    built from and can fail on demand.
    """

    def __init__(
        self,
        config: Any,
        *,
        models: list[str] | None = None,
        text: str = "You use PostgreSQL 16.",
        fail: bool = False,
    ) -> None:
        self.config = config
        self.models = list(models or ["gpt-4o-mini", "gpt-4o"])
        self.text = text
        self.fail = fail
        self.requests: list[list[dict[str, str]]] = []

    def list_models(self) -> list[str]:
        if self.fail:
            raise ProviderError(f"401 unauthorized for key {self.config.api_key}")
        return list(self.models)

    def validate_credentials(self) -> bool:
        if self.fail:
            raise ProviderError(f"401 unauthorized for key {self.config.api_key}")
        return True

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        from providers.base import ProviderResponse

        self.requests.append(messages)
        return ProviderResponse(text=self.text, model=self.config.model, usage={"total_tokens": 4})


class RecordingProvider:
    """Provider double for the Phase 9 generation tests.

    Records every request (messages *and* the keyword arguments) so a test can
    assert that a controlled experiment sent identical sampling parameters, and
    returns a configurable, deterministic response. ``report_model`` can differ
    from the requested model on purpose: a gateway that routes elsewhere is
    exactly the drift the controlled-comparison check has to catch.
    """

    def __init__(
        self,
        *,
        text: str = "Recorded answer.",
        report_model: str | None = None,
        usage: Mapping[str, int] | None = None,
        fail: str | None = None,
    ) -> None:
        self.text = text
        self.report_model = report_model
        self.usage = dict(usage) if usage is not None else {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
        }
        self.fail = fail
        self.requests: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "kwargs": dict(kwargs),
            }
        )
        if self.fail:
            raise ProviderError(self.fail)
        return ProviderResponse(
            text=self.respond(messages, kwargs),
            model=self.report_model or str(kwargs.get("model", "")) or "fake-model",
            usage=dict(self.usage),
        )

    def respond(self, messages: Sequence[Mapping[str, str]], kwargs: Mapping[str, Any]) -> str:
        """Return the text for one request; subclasses make it prompt-dependent."""

        return self.text

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def settings_seen(self) -> list[dict[str, Any]]:
        """Return the keyword arguments of every request, for invariant checks."""

        return [dict(request["kwargs"]) for request in self.requests]


_BULLET_RE = re.compile(r"^- ", re.MULTILINE)
_WORD_RE = re.compile(r"[a-z0-9]+")
#: Words too common to identify a question's subject.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "does", "did", "what", "which", "where", "when", "who",
        "how", "why", "is", "are", "was", "were", "use", "uses", "used", "our", "we",
        "you", "your", "its", "it", "of", "in", "on", "to", "a", "an", "at", "by",
        "with", "from", "that", "this", "project", "production", "current", "please",
        "tell", "me", "about", "again", "still", "s", "t",
    }
)


class PromptReadingProvider(RecordingProvider):
    """Deterministic "model" that answers only from the prompt's evidence.

    It is not a research model and its answers mean nothing about LLM quality.
    It exists to prove the *pipeline*: that a mode's retrieved evidence reaches
    the prompt, that the prompt can be turned into a graded answer, and that a
    mode with no evidence at all cannot produce one. The reader picks the
    evidence bullet with the largest word overlap with the question and says
    "I don't know." when the prompt carries no evidence block.
    """

    def respond(self, messages: Sequence[Mapping[str, str]], kwargs: Mapping[str, Any]) -> str:
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        bullets = [
            line for line in re.split(_BULLET_RE, prompt)[1:] if line.strip()
        ]
        if not bullets:
            return "I don't know."
        question = ""
        for message in reversed(list(messages)):
            if str(message.get("role", "")).lower() == "user":
                question = str(message.get("content", ""))
                break
        return max(bullets, key=lambda bullet: _overlap(bullet, question)).strip()


def _overlap(text: str, question: str) -> int:
    """Count question words (stopwords removed) present in ``text``."""

    haystack = set(_WORD_RE.findall(text.lower()))
    needle = {
        word
        for word in _WORD_RE.findall(question.lower())
        if word not in _STOPWORDS and len(word) > 2
    }
    return len(haystack & needle)


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


class InMemoryConversationStore:
    """Dict-backed ``ConversationStore`` double for persistence unit tests."""

    def __init__(self) -> None:
        self.rows: list[Any] = []
        self.append_calls = 0

    def append(self, message: Any) -> None:
        self.append_calls += 1
        self.rows.append(message)

    def list_messages(self, session_id: str, conversation_id: str) -> list[Any]:
        return [
            row
            for row in self.rows
            if row.session_id == session_id and row.conversation_id == conversation_id
        ]

    def clear(self, session_id: str, conversation_id: str) -> None:
        self.rows = [
            row
            for row in self.rows
            if not (row.session_id == session_id and row.conversation_id == conversation_id)
        ]

    def list_conversations(self, session_id: str) -> list[str]:
        return sorted(
            {row.conversation_id for row in self.rows if row.session_id == session_id}
        )

    def delete_session(self, session_id: str) -> None:
        self.rows = [row for row in self.rows if row.session_id != session_id]


class InMemoryMemoryStore:
    """Dict-backed ``MemoryStore`` double; upserts by ``memory_id``."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def save_memories(self, session_id: str, memories: list[dict[str, Any]]) -> None:
        for record in memories:
            key = f"{session_id}:{record.get('memory_id', '')}"
            self.rows[key] = dict(record)

    def list_memories(self, session_id: str) -> list[dict[str, Any]]:
        prefix = f"{session_id}:"
        return [dict(row) for key, row in self.rows.items() if key.startswith(prefix)]

    def clear(self, session_id: str) -> None:
        prefix = f"{session_id}:"
        self.rows = {key: row for key, row in self.rows.items() if not key.startswith(prefix)}


class InMemoryEvaluationStore:
    """Dict-backed ``EvaluationStore`` double."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}

    def save_run(self, run_id: str, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.runs[run_id] = {
            "run_id": run_id,
            "session_id": metadata.get("session_id", ""),
            "metadata": dict(metadata),
            "metrics": dict(metrics),
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.runs.get(run_id)

    def delete_session_data(self, session_id: str) -> None:
        self.runs = {
            key: run for key, run in self.runs.items() if run["session_id"] != session_id
        }


class RaisingStore:
    """Store double whose every operation fails, for resilience tests."""

    def __getattr__(self, name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"storage backend unavailable ({name})")

        return fail
