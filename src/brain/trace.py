"""Helpers for rendering sanitized BrainOS/application traces."""

from __future__ import annotations

from collections.abc import Iterable
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


def sanitize_trace(events: Iterable[TraceEvent | dict[str, Any]]) -> list[dict[str, str]]:
    """Convert trace events to browser-safe, non-secret dictionaries."""

    output: list[dict[str, str]] = []
    secret_names = {"api_key", "key", "token", "secret", "authorization"}
    for event in events:
        if isinstance(event, TraceEvent):
            name, detail, timestamp = event.name, event.detail, event.timestamp
        else:
            name = str(event.get("name", "unknown"))
            detail = str(event.get("detail", ""))
            timestamp = str(event.get("timestamp", ""))
            if any(str(key).lower() in secret_names for key in event):
                detail = "[redacted]"
        output.append({"name": name, "detail": detail, "timestamp": timestamp})
    return output
