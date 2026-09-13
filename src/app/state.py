"""Typed state objects shared by the UI and application service layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def new_id() -> str:
    """Return a non-secret identifier for a session or conversation."""

    return str(uuid4())


@dataclass
class ProviderConfig:
    """Configuration for one active provider session.

    ``api_key`` is deliberately held only in process memory. Its repr is
    suppressed so accidental diagnostic output cannot expose it.
    """

    provider: str = "openai"
    model: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None


@dataclass
class ContextSettings:
    """Budgets and mode settings used by the context construction layer."""

    mode: str = "brainos"
    max_tokens: int = 4096
    recent_turn_budget: int = 2048
    memory_budget: int = 1536
    system_budget: int = 512


@dataclass
class SessionState:
    """In-memory state associated with one browser session."""

    session_id: str = field(default_factory=new_id)
    actor_id: str = field(default_factory=new_id)
    conversation_id: str = field(default_factory=new_id)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    context: ContextSettings = field(default_factory=ContextSettings)
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_context: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def clear_conversation(self) -> None:
        """Remove conversation content while retaining session configuration."""

        self.messages.clear()
        self.last_context.clear()
        self.diagnostics.clear()
        self.conversation_id = new_id()

    def clear_credentials(self) -> None:
        """Remove the active provider key from process memory."""

        self.provider.api_key = ""
