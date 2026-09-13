"""Helpers for mapping and rendering sanitized BrainOS/application traces."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .adapter import TraceEvent

TRACE_STAGES = (
    "query",
    "observe",
    "recall",
    "relevance_filtering",
    "context_construction",
    "llm",
    "observation",
)

_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "client_secret",
        "refresh_token",
    }
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)\S+")
_KEY_VALUE_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|authorization|password|secret|token|access[_-]?token|"
    r"auth[_-]?token|client[_-]?secret|refresh[_-]?token)\b\s*[=:]\s*)\S+"
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{3,}\b")


def redact_text(text: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Remove credential-shaped values from a trace or explanation string.

    ``secrets`` are credential values the caller knows are live (for example the
    active session's provider key). They are removed by exact match first,
    because pattern-based scrubbing cannot recognise an arbitrary key that a
    user pasted into conversation content and that BrainOS then stored.
    """

    result = str(text)
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        result = result.replace(secret, "[redacted]")
    result = _BEARER_RE.sub(r"\1[redacted]", result)
    result = _KEY_VALUE_RE.sub(r"\1[redacted]", result)
    result = _OPENAI_KEY_RE.sub("[redacted]", result)
    return result


def sanitize_value(value: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    """Return a JSON-like value with secret fields omitted and strings scrubbed."""

    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower().replace("-", "_") in _SECRET_FIELD_NAMES:
                continue
            safe[key_text] = sanitize_value(item, secrets=secrets)
        return safe
    if isinstance(value, (list, tuple)):
        return [sanitize_value(item, secrets=secrets) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return redact_text(value, secrets=secrets)
    for method_name in ("to_dict", "model_dump", "dict"):
        dump = getattr(value, method_name, None)
        if callable(dump):
            try:
                return sanitize_value(dump(), secrets=secrets)
            except Exception:
                break
    return redact_text(str(value), secrets=secrets)


def map_runtime_trace(payload: Any) -> list[TraceEvent]:
    """Map a BrainOS structured session-trace dict into application events.

    Cycle records are summarized by stage name and counts. Retrieved memory
    text is not copied into the trace so credentials and prompt-injection
    payloads cannot ride along in diagnostic output.
    """

    if payload is None:
        return []
    if isinstance(payload, str):
        return [TraceEvent(name="runtime", detail=redact_text(payload)[:300])]
    if not isinstance(payload, Mapping):
        return []

    events: list[TraceEvent] = []
    session_id = payload.get("session_id")
    cycles = payload.get("cycles")
    if session_id or cycles is not None:
        events.append(
            TraceEvent(
                name="session",
                detail=f"cycles={cycles if cycles is not None else 0}",
            )
        )

    records = payload.get("records") or []
    if isinstance(records, list):
        for index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                continue
            event_type = redact_text(str(record.get("event_type", "cycle")))
            retrieved = record.get("retrieved") or []
            selected = len(retrieved) if isinstance(retrieved, list) else 0
            candidates = record.get("candidates_considered", selected)
            events.append(
                TraceEvent(
                    name="observe",
                    detail=f"cycle={index} event_type={event_type}",
                )
            )
            events.append(
                TraceEvent(
                    name="recall",
                    detail=f"cycle={index} selected={selected} candidates={candidates}",
                )
            )
            decision = record.get("decision")
            if decision:
                events.append(
                    TraceEvent(name="decide", detail=f"cycle={index} decision={decision}")
                )
            steps = record.get("steps") or []
            if isinstance(steps, list) and steps:
                events.append(
                    TraceEvent(
                        name="cycle",
                        detail=f"cycle={index} {' → '.join(str(step) for step in steps)}",
                    )
                )
    return events


def sanitize_trace(
    events: Iterable[TraceEvent | dict[str, Any]], *, secrets: tuple[str, ...] = ()
) -> list[dict[str, str]]:
    """Convert trace events to browser-safe, non-secret dictionaries."""

    output: list[dict[str, str]] = []
    for event in events:
        if isinstance(event, TraceEvent):
            name, detail, timestamp = event.name, event.detail, event.timestamp
        else:
            name = str(event.get("name", "unknown"))
            detail = str(event.get("detail", ""))
            timestamp = str(event.get("timestamp", ""))
            if any(str(key).lower().replace("-", "_") in _SECRET_FIELD_NAMES for key in event):
                detail = "[redacted]"
        output.append(
            {
                "name": redact_text(name, secrets=secrets),
                "detail": redact_text(detail, secrets=secrets),
                "timestamp": timestamp,
            }
        )
    return output
