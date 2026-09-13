"""Stable application-facing interface for the upstream BrainOS runtime."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
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
    """A sanitized memory returned to the application layer.

    Phase 3 added the retrieval/lifecycle signals the context construction
    engine needs (entities, scoring signals, temporal validity, supersession).
    Every field is optional and provider-neutral: BrainOS enums and objects are
    converted here so no upstream type reaches the context builder, UI, or
    evaluation runner.
    """

    memory_id: str
    text: str
    memory_type: str = "FACT"
    relevance: float | None = None
    source_turn: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retrieval_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- Phase 3 retrieval signals ------------------------------------- #
    observed_at: str = ""
    entities: tuple[str, ...] = ()
    confidence: float | None = None
    salience: float | None = None
    utility: float | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    status: str | None = None
    signals: dict[str, float] = field(default_factory=dict)
    type_source: str = "runtime"

    @property
    def timestamp(self) -> str:
        """Best available event time, used for recency weighting."""

        return self.observed_at or self.created_at


@dataclass(frozen=True)
class Conflict:
    """A runtime-reported contradiction between two memories.

    Mapped from BrainOS ``contradictions()`` so the context builder can prefer
    the newer claim without importing any upstream type.
    """

    subject: str = ""
    older_id: str = ""
    newer_id: str = ""
    older_text: str = ""
    newer_text: str = ""
    source: str = "runtime"


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

    # --- Phase 3 context-construction extensions ------------------------ #
    def list_memories(self) -> list[MemoryRecord]:
        """Return currently stored memories for inspection."""

    def conflicts(self) -> list[Conflict]:
        """Return runtime-detected contradictions between memories."""

    def stale_memory_ids(self) -> set[str]:
        """Return ids the runtime reports as superseded or expired."""

    def enrich_with_explanation(
        self, records: list[MemoryRecord], explanation: dict[str, Any]
    ) -> list[MemoryRecord]:
        """Attach per-memory retrieval scores/signals from ``explain()``."""


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
        # Application-side observation bookkeeping. BrainOS keeps the memories;
        # this only remembers *which* application memory type and conversation
        # turn produced each observation, so recalled content can be annotated
        # and recency-weighted without asking the runtime for anything it does
        # not model.
        self._observations: dict[str, dict[str, Any]] = {}
        self._observe_count = 0

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
        self._observe_count += 1
        self._observations[_normalize_key(stripped)] = {
            "memory_type": candidate.memory_type.value,
            "confidence": candidate.confidence,
            "source": source,
            "turn": _as_int(payload.get("turn")) or self._observe_count,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "stored": stored,
        }
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

    def conflicts(self) -> list[Conflict]:
        """Return runtime-detected same-subject contradictions.

        Maps BrainOS ``contradictions()`` onto the application ``Conflict``
        record. Returns an empty list when the runtime (or an injected test
        double) does not expose contradiction detection, so callers can fall
        back to the application-level heuristic in ``retrieval_policy``.
        """

        runtime = self.runtime
        getter = getattr(runtime, "contradictions", None) if runtime is not None else None
        if not callable(getter):
            return []
        try:
            raw = getter()
        except Exception:
            return []
        conflicts = [_to_conflict(item) for item in (raw or []) if item is not None]
        self._trace.append(TraceEvent("conflict_check", f"contradictions={len(conflicts)}"))
        return conflicts

    def stale_memory_ids(self) -> set[str]:
        """Return ids the runtime reports as superseded or expired."""

        runtime = self.runtime
        getter = getattr(runtime, "stale_memories", None) if runtime is not None else None
        if not callable(getter):
            return set()
        try:
            raw = getter()
        except Exception:
            return set()
        ids: set[str] = set()
        for item in raw or []:
            if isinstance(item, str):
                ids.add(item)
                continue
            memory_id: Any
            if isinstance(item, dict):
                memory_id = item.get("memory_id", item.get("id"))
            else:
                memory_id = getattr(item, "id", getattr(item, "memory_id", None))
            if memory_id:
                ids.add(str(memory_id))
        self._trace.append(TraceEvent("stale_check", f"stale={len(ids)}"))
        return ids

    def enrich_with_explanation(
        self, records: list[MemoryRecord], explanation: dict[str, Any]
    ) -> list[MemoryRecord]:
        """Attach query-specific retrieval scores and signals from ``explain()``.

        ``recall()`` returns content only. ``why()``/``explain()`` returns the
        runtime's per-signal evidence for the same query, so the context engine
        can rank with BrainOS's own signals instead of re-deriving relevance.
        After enrichment ``relevance`` is the query-specific runtime score and
        ``salience`` remains the query-independent importance signal.
        """

        selected = explanation.get("selected") if isinstance(explanation, dict) else None
        if not isinstance(selected, list) or not selected:
            return list(records)
        by_id: dict[str, dict[str, Any]] = {}
        by_content: dict[str, dict[str, Any]] = {}
        for entry in selected:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id", entry.get("memory_id", "")) or "")
            content = str(entry.get("content", entry.get("text", "")) or "")
            if entry_id:
                by_id.setdefault(entry_id, entry)
            if content:
                by_content.setdefault(_normalize_key(content), entry)

        enriched: list[MemoryRecord] = []
        for record in records:
            entry = by_id.get(record.memory_id) or by_content.get(_normalize_key(record.text))
            if entry is None:
                enriched.append(record)
                continue
            enriched.append(_with_explanation(record, entry))
        return enriched

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

    def _to_memory_record(self, item: Any, *, fallback_text: str | None = None) -> MemoryRecord:
        if isinstance(item, MemoryRecord):
            return self._apply_observation(item)
        if isinstance(item, str):
            return self._apply_observation(
                MemoryRecord(
                    memory_id=_stable_id(item),
                    text=item,
                    created_at="",
                    metadata={"source": "brainos"},
                )
            )
        if isinstance(item, dict):
            text = str(item.get("text", item.get("content", fallback_text or "")))
            provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
            supersedes = _id_tuple(item.get("supersedes")) or _id_tuple(
                provenance.get("superseded_id")
            )
            return self._apply_observation(
                MemoryRecord(
                    memory_id=str(item.get("memory_id", item.get("id", _stable_id(text)))),
                    text=text,
                    memory_type=_map_memory_type(item.get("memory_type", item.get("type", "FACT"))),
                    relevance=_as_float(item.get("relevance", item.get("score"))),
                    source_turn=_as_int(item.get("source_turn", item.get("turn"))),
                    created_at=_iso_or_empty(item.get("created_at")),
                    retrieval_count=_as_int(
                        item.get("retrieval_count", item.get("access_count"))
                    )
                    or 0,
                    metadata=_public_memory_metadata(item),
                    observed_at=_iso_or_empty(
                        item.get("observed_at", provenance.get("observed_at"))
                    ),
                    entities=_text_tuple(item.get("entities")),
                    confidence=_as_float(item.get("confidence", provenance.get("confidence"))),
                    salience=_as_float(item.get("salience")),
                    utility=_as_float(item.get("utility")),
                    valid_from=_iso_or_none(item.get("valid_from")),
                    valid_until=_iso_or_none(item.get("valid_until")),
                    supersedes=supersedes,
                    contradicts=_id_tuple(item.get("contradicts")),
                    status=_status_text(item.get("status")),
                    signals=_signal_map(item.get("signals")),
                )
            )

        text = fallback_text or _text_of(item)
        provenance = getattr(item, "provenance", None)
        supersedes = _id_tuple(getattr(item, "supersedes", None)) or _id_tuple(
            getattr(provenance, "superseded_id", None)
        )
        return self._apply_observation(
            MemoryRecord(
                memory_id=str(getattr(item, "id", getattr(item, "memory_id", _stable_id(text)))),
                text=text,
                memory_type=_map_memory_type(
                    getattr(item, "type", getattr(item, "memory_type", "FACT"))
                ),
                relevance=_as_float(getattr(item, "score", getattr(item, "relevance", None))),
                source_turn=_as_int(getattr(item, "source_turn", getattr(item, "turn", None))),
                created_at=_iso_or_empty(getattr(item, "created_at", None)),
                retrieval_count=_as_int(
                    getattr(item, "access_count", getattr(item, "retrieval_count", None))
                )
                or 0,
                metadata=_public_memory_metadata(item),
                observed_at=_iso_or_empty(
                    getattr(item, "observed_at", getattr(provenance, "observed_at", None))
                ),
                entities=_text_tuple(getattr(item, "entities", None)),
                confidence=_as_float(
                    getattr(item, "confidence", getattr(provenance, "confidence", None))
                ),
                salience=_as_float(getattr(item, "salience", None)),
                utility=_as_float(getattr(item, "utility", None)),
                valid_from=_iso_or_none(getattr(item, "valid_from", None)),
                valid_until=_iso_or_none(getattr(item, "valid_until", None)),
                supersedes=supersedes,
                contradicts=_id_tuple(getattr(item, "contradicts", None)),
                status=_status_text(getattr(item, "status", None)),
                signals=_signal_map(getattr(item, "signals", None)),
            )
        )

    def _apply_observation(self, record: MemoryRecord) -> MemoryRecord:
        """Re-attach application observation metadata to a recalled memory.

        BrainOS stores content, not the application's memory-policy label or
        conversation turn. Restoring them here (keyed by normalized text) gives
        the context engine a usable ``source_turn`` for recency weighting and a
        specific memory type for annotation. Runtime-reported types are only
        refined when they collapsed to the generic ``FACT`` mapping.
        """

        observation = self._observations.get(_normalize_key(record.text))
        if not observation:
            return record
        updates: dict[str, Any] = {}
        if record.memory_type == MemoryType.FACT.value and observation.get("memory_type"):
            updates["memory_type"] = str(observation["memory_type"])
            updates["type_source"] = "policy"
        if record.source_turn is None and observation.get("turn") is not None:
            updates["source_turn"] = int(observation["turn"])
        if not record.observed_at and observation.get("observed_at"):
            updates["observed_at"] = str(observation["observed_at"])
        if record.confidence is None and observation.get("confidence") is not None:
            updates["confidence"] = _as_float(observation["confidence"])
        return replace(record, **updates) if updates else record


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


def _iso_or_empty(value: Any) -> str:
    """Return an ISO timestamp, or ``""`` when the runtime did not report one.

    Unknown times stay unknown. Fabricating ``now`` would make every memory look
    maximally recent and silently disable recency weighting.
    """

    if value is None or value == "":
        return ""
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())
    return str(value)


def _iso_or_none(value: Any) -> str | None:
    """Return an optional ISO timestamp for validity windows."""

    if value is None or value == "":
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())
    return str(value)


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_key(text: str) -> str:
    """Casefold and collapse whitespace for observation bookkeeping keys."""

    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def _text_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a runtime collection of strings into a tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _id_tuple(value: Any) -> tuple[str, ...]:
    """Normalize one id, a list of ids, or nothing into a tuple of ids."""

    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def _status_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _signal_map(value: Any) -> dict[str, float]:
    """Keep only numeric retrieval signals, dropping anything non-numeric."""

    if not isinstance(value, dict):
        return {}
    signals: dict[str, float] = {}
    for key, item in value.items():
        numeric = _as_float(item)
        if numeric is not None:
            signals[str(key)] = numeric
    return signals


def _to_conflict(item: Any) -> Conflict:
    """Map a runtime contradiction record onto the application ``Conflict``."""

    if isinstance(item, Conflict):
        return item
    if isinstance(item, dict):
        return Conflict(
            subject=str(item.get("subject", "") or ""),
            older_id=str(item.get("older_id", item.get("older_memory_id", "")) or ""),
            newer_id=str(item.get("newer_id", item.get("newer_memory_id", "")) or ""),
            older_text=str(item.get("older_content", item.get("older_text", "")) or ""),
            newer_text=str(item.get("newer_content", item.get("newer_text", "")) or ""),
            source="runtime",
        )
    return Conflict(
        subject=str(getattr(item, "subject", "") or ""),
        older_id=str(getattr(item, "older_id", "") or ""),
        newer_id=str(getattr(item, "newer_id", "") or ""),
        older_text=str(getattr(item, "older_content", getattr(item, "older_text", "")) or ""),
        newer_text=str(getattr(item, "newer_content", getattr(item, "newer_text", "")) or ""),
        source="runtime",
    )


def _with_explanation(record: MemoryRecord, entry: dict[str, Any]) -> MemoryRecord:
    """Return a copy of ``record`` carrying the runtime's retrieval evidence."""

    score = _as_float(entry.get("score", entry.get("relevance")))
    signals = _signal_map(entry.get("signals"))
    updates: dict[str, Any] = {}
    if score is not None:
        updates["relevance"] = score
    if signals:
        updates["signals"] = signals
    if not record.salience and signals.get("salience") is not None:
        updates["salience"] = signals["salience"]
    return replace(record, **updates) if updates else record

