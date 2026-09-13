"""Stable application-facing interface for the upstream BrainOS runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryRecord:
    """A sanitized memory returned to the application layer."""

    memory_id: str
    text: str
    memory_type: str = "FACT"
    relevance: float | None = None
    source_turn: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retrieval_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """Decision signal used to explain whether context appears sufficient."""

    sufficient: bool
    reason: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class TraceEvent:
    """Sanitized event suitable for rendering in the UI."""

    name: str
    detail: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BrainMemoryAdapter(Protocol):
    """The only BrainOS contract the application should depend on."""

    def observe(self, text: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Observe a user or assistant text using the configured memory policy."""

    def recall(self, query: str, *, limit: int = 8) -> list[MemoryRecord]:
        """Return memories relevant to the current query."""

    def decide(self, query: str) -> Decision:
        """Determine whether available memory/context is sufficient."""

    def explain(self, query: str) -> dict[str, Any]:
        """Return sanitized retrieval and reasoning signals for the UI."""

    def trace(self) -> list[TraceEvent]:
        """Return a sanitized cognitive trace without credentials or raw secrets."""


class BrainOSAdapter:
    """Placeholder adapter for the validated upstream BrainOS v2 runtime.

    The constructor accepts an injected runtime so the rest of the application
    can be tested without importing BrainOS. The concrete mapping from
    ``observe/recall/decide/why`` to this interface belongs in Phase 0 after the
    upstream API and revision are pinned.
    """

    def __init__(self, runtime: Any | None = None) -> None:
        self.runtime = runtime
        self._trace: list[TraceEvent] = []

    def _require_runtime(self) -> Any:
        if self.runtime is None:
            raise RuntimeError(
                "BrainOS is not configured. Validate and inject an upstream BrainOS "
                "runtime during Phase 0."
            )
        return self.runtime

    def observe(self, text: str, *, metadata: dict[str, Any] | None = None) -> None:
        runtime = self._require_runtime()
        # Keep this mapping deliberately explicit rather than hiding API drift.
        runtime.observe(text, metadata=metadata or {})
        self._trace.append(TraceEvent("observe"))

    def recall(self, query: str, *, limit: int = 8) -> list[MemoryRecord]:
        runtime = self._require_runtime()
        raw_memories = runtime.recall(query, limit=limit)
        self._trace.append(TraceEvent("recall", f"returned={len(raw_memories)}"))
        return [self._to_memory_record(item) for item in raw_memories]

    def decide(self, query: str) -> Decision:
        runtime = self._require_runtime()
        result = runtime.decide(query)
        self._trace.append(TraceEvent("decide"))
        if isinstance(result, Decision):
            return result
        if isinstance(result, bool):
            return Decision(sufficient=result)
        return Decision(
            sufficient=bool(getattr(result, "sufficient", False)),
            reason=str(getattr(result, "reason", "")),
            confidence=getattr(result, "confidence", None),
        )

    def explain(self, query: str) -> dict[str, Any]:
        runtime = self._require_runtime()
        why = getattr(runtime, "why", None)
        if why is None:
            return {}
        explanation = why(query)
        self._trace.append(TraceEvent("explain"))
        return self._sanitize_mapping(explanation)

    def trace(self) -> list[TraceEvent]:
        """Return a copy so callers cannot mutate the adapter's trace."""

        return list(self._trace)

    @staticmethod
    def _to_memory_record(item: Any) -> MemoryRecord:
        if isinstance(item, MemoryRecord):
            return item
        if isinstance(item, str):
            return MemoryRecord(memory_id="unknown", text=item)
        if isinstance(item, dict):
            return MemoryRecord(
                memory_id=str(item.get("memory_id", item.get("id", "unknown"))),
                text=str(item.get("text", item.get("content", ""))),
                memory_type=str(item.get("memory_type", item.get("type", "FACT"))),
                relevance=item.get("relevance"),
                source_turn=item.get("source_turn"),
                metadata={"source": "brainos"},
            )
        return MemoryRecord(memory_id="unknown", text=str(item), metadata={"source": "brainos"})

    @staticmethod
    def _sanitize_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {
                str(key): str(item) if str(key).lower() in {"api_key", "key", "token", "secret"} else item
                for key, item in value.items()
            }
        return {"explanation": str(value)}
