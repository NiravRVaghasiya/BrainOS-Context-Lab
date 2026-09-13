"""Conversation and memory persistence abstraction.

SQLite is the planned MVP backend. The interface is defined first so storage
choices cannot leak into the UI, BrainOS adapter, or evaluation runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class ConversationMessage:
    session_id: str
    conversation_id: str
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationStore(Protocol):
    """Storage contract with explicit session and conversation identifiers."""

    def append(self, message: ConversationMessage) -> None:
        """Persist one message without accepting provider credentials."""

    def list_messages(self, session_id: str, conversation_id: str) -> list[ConversationMessage]:
        """Return messages belonging to exactly one isolated conversation."""

    def clear(self, session_id: str, conversation_id: str) -> None:
        """Delete one conversation."""


class MemoryStore(Protocol):
    """Storage contract for user-visible memory records."""

    def list_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Return only memories for the requested session."""

    def clear(self, session_id: str) -> None:
        """Delete all memories for one session."""
