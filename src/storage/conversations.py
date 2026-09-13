"""Conversation and memory persistence abstraction.

SQLite is the planned MVP backend. The interface is defined first so storage
choices cannot leak into the UI, BrainOS adapter, or evaluation runner.

Phase 5 finalized the protocols with the lifecycle operations the plan's data
controls require (*Clear conversation / Clear memory / Delete session / Export
session*, plan §19): ``list_conversations`` and ``delete_session`` on the
conversation store, and ``save_memories`` on the memory store. The memory
store is a **mirror** for inspection and export — the BrainOS runtime stays
authoritative for recall.
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

    def list_conversations(self, session_id: str) -> list[str]:
        """Return the conversation identifiers stored for exactly one session."""

    def delete_session(self, session_id: str) -> None:
        """Delete every conversation persisted for one session."""


class MemoryStore(Protocol):
    """Storage contract for user-visible memory records.

    Records are sanitized ``MemoryRecord`` mappings (as produced by
    ``dataclasses.asdict``). Implementations must upsert by ``memory_id`` so
    repeated mirrors of the same memory update counters instead of duplicating
    rows, and must strip secret-named fields defensively even though callers
    already sanitize.
    """

    def save_memories(self, session_id: str, memories: list[dict[str, Any]]) -> None:
        """Upsert mirrored memory records for one session."""

    def list_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Return only memories for the requested session."""

    def clear(self, session_id: str) -> None:
        """Delete all memories for one session."""
