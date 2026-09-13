"""Stable application-facing interface for the upstream BrainOS runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from .memory_policy import MemoryCandidate, MemoryPolicy, MemoryType

_SUFFICIENT_DECISIONS = frozenset({"act"})
_EVENT_TYPE_ALIASES = {
    "user": "user_message",
    "user_message": "user_message",
    "assistant": "assistant_message",
    "assistant_message": "assistant_message",
    "tool": "tool_result",
    "tool_result": "tool_result",
    "tool_call": "tool_call",
    "observation": "observation",
    "memory_candidate": "memory_candidate",
    "memory_retrieved": "memory_retrieved",
}
_BRAINOS_TYPE_TO_APP = {
    "semantic": MemoryType.FACT.value,
    "episodic": MemoryType.FACT.value,
    "procedural": MemoryType.TASK.value,
    "preference": MemoryType.PREFERENCE.value,
    "goal": MemoryType.GOAL.value,
    "constraint": MemoryType.CONSTRAINT.value,
    "reflection": MemoryType.DECISION.value,
    "belief": MemoryType.FACT.value,
    "fact": MemoryType.FACT.value,
}


class BrainOSNotConfiguredError(RuntimeError):
    """Raised when the application needs a BrainOS runtime that is not available."""


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
    action: str = ""


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


def create_runtime(
    *,
    session_id: str,
    actor_id: str,
    token_budget: int = 2000,
    wm_slots: int = 6,
    prefer_generated: bool = False,
    storage_backend: Any | None = None,
) -> Any:
    """Construct one isolated BrainOS runtime for a single application session.

    ``prefer_generated`` defaults to False so the application uses the pinned
    runtime's offline inline plugins rather than an optional generated plugin
    package. Callers must not share the returned runtime across sessions.
    """

    try:
        from brainos_runtime import BrainOS
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise BrainOSNotConfiguredError(
            "BrainOS is not installed. Install the integration extra with "
            "`pip install -e '.[integration]'`."
        ) from exc
    return BrainOS(
        session_id=session_id,
        actor_id=actor_id,
        token_budget=token_budget,
        wm_slots=wm_slots,
        prefer_generated=prefer_generated,
        storage_backend=storage_backend,
    )


def create_brain_adapter(
    *,
    session_id: str,
    actor_id: str,
    policy: MemoryPolicy | None = None,
    runtime: Any | None = None,
    token_budget: int = 2000,
    wm_slots: int = 6,
) -> BrainOSAdapter:
    """Build a session-scoped adapter, creating a runtime when one is not injected."""

    if runtime is None:
        runtime = create_runtime(
            session_id=session_id,
            actor_id=actor_id,
            token_budget=token_budget,
            wm_slots=wm_slots,
        )
    return BrainOSAdapter(
        runtime,
        policy=policy,
        session_id=session_id,
        actor_id=actor_id,
    )


class BrainOSAdapter:
    """Explicit mapping from the pinned BrainOS v2 runtime onto the app contract.

    Application code must not call BrainOS methods directly. This adapter is the
    translation layer for:

    * ``observe(event, source=..., event_type=...)``
    * ``recall(query, top_k=...)``
    * decision strings / ``assess()``
    * ``why()`` explanations
    * structured ``trace()`` payloads
    """

    def __init__(
        self,
        runtime: Any | None = None,
        *,
        policy: MemoryPolicy | None = None,
        session_id: str | None = None,
        actor_id: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.policy = policy
        self.session_id = session_id
        self.actor_id = actor_id
        self._trace: list[TraceEvent] = []

    def _require_runtime(self) -> Any:
        if self.runtime is None:
            raise BrainOSNotConfiguredError(
                "BrainOS is not configured. Inject a runtime or call "
                "create_brain_adapter() with a session and actor id."
            )
        return self.runtime

    def observe(self, text: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Store useful text in BrainOS after applying the optional memory policy.

        Application ``metadata`` is translated into ``source`` / ``event_type``.
        It is never forwarded as a ``metadata=`` keyword to the runtime.
        """

        payload = dict(metadata or {})
        stripped = text.strip()
        if not stripped:
            self._trace.append(TraceEvent("observe_skipped", "empty"))
            return

        candidate = _candidate_from_text(stripped, payload)
        if self.policy is not None and not self.policy.accepts(candidate):
            self._trace.append(
                TraceEvent("observe_skipped", f"policy:{candidate.memory_type.value}")
            )
            return

        runtime = self._require_runtime()
        source = _source_from_metadata(payload)
        event_type = _event_type_from_metadata(payload, source)
        result = runtime.observe(stripped, source=source, event_type=event_type)
        stored = 0
        if isinstance(result, dict):
            stored = len(result.get("stored_ids") or [])
        self._trace.append(TraceEvent("observe", f"source={source} stored={stored}"))

    def recall(self, query: str, *, limit: int = 8, top_k: int | None = None) -> list[MemoryRecord]:
        """Retrieve memories, mapping ``limit`` to BrainOS ``top_k``."""

        runtime = self._require_runtime()
        count = top_k if top_k is not None else limit
        raw = runtime.recall(query, top_k=count)
        items = list(raw) if isinstance(raw, list) else []
        records = self._records_from_recall(items)
        self._trace.append(TraceEvent("recall", f"returned={len(records)}"))
        return records

    def decide(self, query: str) -> Decision:
        """Normalize BrainOS decision strings / assessments into ``Decision``."""

        runtime = self._require_runtime()
        assess = getattr(runtime, "assess", None)
        raw: Any
        if callable(assess):
            raw = assess(query)
        else:
            raw = runtime.decide(query)
        decision = _to_decision(raw)
        self._trace.append(
            TraceEvent(
                "decide",
                f"action={decision.action or ('act' if decision.sufficient else 'ask')}",
            )
        )
        return decision

    def explain(self, query: str) -> dict[str, Any]:
        """Return a sanitized ``why()`` explanation for the UI."""

        from .trace import sanitize_value

        runtime = self._require_runtime()
        why = getattr(runtime, "why", None)
        if not callable(why):
            explain = getattr(runtime, "explain", None)
            if not callable(explain):
                return {}
            why = explain
        explanation = why(query)
        self._trace.append(TraceEvent("explain"))
        sanitized = sanitize_value(explanation)
        return sanitized if isinstance(sanitized, dict) else {"explanation": str(sanitized)}

    def list_memories(self) -> list[MemoryRecord]:
        """Return currently stored memories for inspection, if the runtime exposes them."""

        runtime = self._require_runtime()
        records = [
            self._to_memory_record(item) for item in self._iter_runtime_memories(runtime)
        ]
        return records

    def trace(self) -> list[TraceEvent]:
        """Return mapped runtime cycle events followed by adapter-local events."""

        from .trace import map_runtime_trace

        events: list[TraceEvent] = []
        runtime = self.runtime
        tracer = getattr(runtime, "trace", None) if runtime is not None else None
        if callable(tracer):
            try:
                payload = tracer()
            except TypeError:
                payload = tracer(formatted=False)
            events.extend(map_runtime_trace(payload))
        events.extend(self._trace)
        return events

    def _records_from_recall(self, items: list[Any]) -> list[MemoryRecord]:
        if not items:
            return []
        if not all(isinstance(item, str) for item in items):
            return [self._to_memory_record(item) for item in items]

        index = self._memory_index_by_content()
        records: list[MemoryRecord] = []
        for content in items:
            bucket = index.get(content) or index.get(content.strip())
            source_item = bucket.pop(0) if bucket else content
            records.append(self._to_memory_record(source_item, fallback_text=content))
        return records

    def _memory_index_by_content(self) -> dict[str, list[Any]]:
        runtime = self.runtime
        if runtime is None:
            return {}
        index: dict[str, list[Any]] = {}
        for item in self._iter_runtime_memories(runtime):
            index.setdefault(_text_of(item), []).append(item)
        return index

    @staticmethod
    def _iter_runtime_memories(runtime: Any) -> list[Any]:
        for name in ("active_memories", "memories"):
            getter = getattr(runtime, name, None)
            if not callable(getter):
                continue
            try:
                values = getter()
            except Exception:
                continue
            if isinstance(values, dict):
                return list(values.values())
            if isinstance(values, list):
                return values
        return []

    @staticmethod
    def _to_memory_record(item: Any, *, fallback_text: str | None = None) -> MemoryRecord:
        if isinstance(item, MemoryRecord):
            return item
        if isinstance(item, str):
            text = item
            return MemoryRecord(
                memory_id=_stable_id(text),
                text=text,
                metadata={"source": "brainos"},
            )
        if isinstance(item, dict):
            text = str(item.get("text", item.get("content", fallback_text or "")))
            memory_id = str(item.get("memory_id", item.get("id", _stable_id(text))))
            raw_type = item.get("memory_type", item.get("type", "FACT"))
            relevance = item.get("relevance", item.get("score", item.get("salience")))
            created = item.get("created_at")
            retrieval_count = int(item.get("retrieval_count", item.get("access_count", 0)) or 0)
            return MemoryRecord(
                memory_id=memory_id,
                text=text,
                memory_type=_map_memory_type(raw_type),
                relevance=_as_float(relevance),
                source_turn=item.get("source_turn"),
                created_at=_as_iso(created),
                retrieval_count=retrieval_count,
                metadata=_public_memory_metadata(item),
            )
        text = fallback_text or _text_of(item)
        raw_type = getattr(item, "type", getattr(item, "memory_type", "FACT"))
        relevance = getattr(
            item, "salience", getattr(item, "score", getattr(item, "relevance", None))
        )
        created = getattr(item, "created_at", None)
        retrieval_count = int(
            getattr(item, "access_count", getattr(item, "retrieval_count", 0)) or 0
        )
        memory_id = str(getattr(item, "id", getattr(item, "memory_id", _stable_id(text))))
        return MemoryRecord(
            memory_id=memory_id,
            text=text,
            memory_type=_map_memory_type(raw_type),
            relevance=_as_float(relevance),
            created_at=_as_iso(created),
            retrieval_count=retrieval_count,
            metadata=_public_memory_metadata(item),
        )


def _candidate_from_text(text: str, metadata: dict[str, Any]) -> MemoryCandidate:
    raw_type = metadata.get("memory_type", MemoryType.FACT)
    if isinstance(raw_type, MemoryType):
        memory_type = raw_type
    else:
        try:
            memory_type = MemoryType(str(raw_type).upper())
        except ValueError:
            memory_type = MemoryType.FACT
    confidence = metadata.get("confidence", 1.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 1.0
    source = str(metadata.get("source", "conversation"))
    return MemoryCandidate(
        text=text,
        memory_type=memory_type,
        confidence=confidence_value,
        source=source,
        metadata=metadata,
    )


def _source_from_metadata(metadata: dict[str, Any]) -> str:
    source = metadata.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    role = metadata.get("role")
    if isinstance(role, str) and role.strip():
        return role.strip()
    return "user"


def _event_type_from_metadata(metadata: dict[str, Any], source: str) -> str:
    raw = metadata.get("event_type")
    if raw is not None:
        value = str(getattr(raw, "value", raw)).strip().lower()
        return _EVENT_TYPE_ALIASES.get(value, value or "user_message")
    role = str(metadata.get("role", source)).strip().lower()
    return _EVENT_TYPE_ALIASES.get(role, "user_message")


def _to_decision(result: Any) -> Decision:
    if isinstance(result, Decision):
        return result
    if isinstance(result, bool):
        return Decision(sufficient=result, action="act" if result else "ask")
    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            result = result.to_dict()
        except Exception:
            pass
    if isinstance(result, str):
        action = result.strip().lower()
        return Decision(
            sufficient=action in _SUFFICIENT_DECISIONS,
            reason=action,
            action=action,
        )
    if isinstance(result, dict):
        action = str(result.get("decision", result.get("action", ""))).strip().lower()
        reason = result.get("rationale", result.get("reason", action))
        if isinstance(reason, list):
            reason_text = "; ".join(str(item) for item in reason)
        else:
            reason_text = str(reason or action)
        confidence = _as_float(result.get("confidence"))
        sufficient = (
            action in _SUFFICIENT_DECISIONS
            if action
            else bool(result.get("sufficient", False))
        )
        return Decision(
            sufficient=sufficient,
            reason=reason_text,
            confidence=confidence,
            action=action,
        )
    sufficient = bool(getattr(result, "sufficient", False))
    action = str(getattr(result, "action", getattr(result, "decision", "")))
    return Decision(
        sufficient=sufficient,
        reason=str(getattr(result, "reason", "")),
        confidence=_as_float(getattr(result, "confidence", None)),
        action=action,
    )


def _map_memory_type(raw: Any) -> str:
    value = str(getattr(raw, "value", raw) or "FACT").strip()
    mapped = _BRAINOS_TYPE_TO_APP.get(value.lower())
    if mapped:
        return mapped
    upper = value.upper()
    try:
        return MemoryType(upper).value
    except ValueError:
        return MemoryType.FACT.value


def _text_of(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("text", item.get("content", "")))
    content = getattr(item, "content", None)
    if content is not None:
        return str(content)
    text = getattr(item, "text", None)
    if text is not None:
        return str(text)
    return str(item)


def _public_memory_metadata(item: Any) -> dict[str, Any]:
    from .trace import sanitize_value

    if isinstance(item, dict):
        source = item.get("source")
        status = item.get("status")
        brainos_type = item.get("type", item.get("memory_type"))
    else:
        source = getattr(item, "source", None)
        status = getattr(item, "status", None)
        brainos_type = getattr(item, "type", None)
    payload = {
        "source": "brainos",
        "runtime_source": None if source is None else str(getattr(source, "value", source)),
        "status": None if status is None else str(getattr(status, "value", status)),
        "brainos_type": None
        if brainos_type is None
        else str(getattr(brainos_type, "value", brainos_type)),
    }
    sanitized = sanitize_value(payload)
    return sanitized if isinstance(sanitized, dict) else {"source": "brainos"}


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_iso(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())
    return str(value)
