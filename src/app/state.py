"""Typed state objects shared by the UI and application service layer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from brain.context_builder import ContextBudget
from brain.retrieval_policy import RetrievalPolicy
from providers.base import ProviderConfig


def new_id() -> str:
    """Return a non-secret identifier for a session or conversation."""

    return str(uuid4())


@dataclass
class ContextSettings:
    """Budgets and retrieval settings used by the context construction layer.

    ``mode`` is the baseline-mode selector (Phase 6). The remaining fields are
    the Phase 3 knobs: token budgets per section, and the deduplication,
    relevance, conflict, and recency parameters of the retrieval policy. They
    live on session state so the UI and the evaluation runner configure context
    construction through one object instead of constructing dataclasses inline.
    """

    mode: str = "brainos"
    max_tokens: int = 4096
    recent_turn_budget: int = 2048
    memory_budget: int = 1536
    system_budget: int = 512
    max_recent_turns: int | None = None
    per_message_overhead: int = 4
    max_memories: int = 12
    relevance_floor: float = 0.12
    near_duplicate_threshold: float = 0.88
    weight_runtime_signals: float = 0.45
    weight_lexical: float = 0.35
    weight_recency: float = 0.20
    recency_half_life_turns: float = 20.0
    recency_half_life_hours: float = 48.0
    annotate_memories: bool = True
    resolve_conflicts: bool = True
    drop_stale_memories: bool = True
    drop_suspicious_memories: bool = False

    def context_budget(self) -> ContextBudget:
        """Return the token budget for one context construction."""

        return ContextBudget(
            max_tokens=self.max_tokens,
            recent_turn_budget=self.recent_turn_budget,
            memory_budget=self.memory_budget,
            system_budget=self.system_budget,
            max_recent_turns=self.max_recent_turns,
            per_message_overhead=self.per_message_overhead,
        )

    def retrieval_policy(self) -> RetrievalPolicy:
        """Return the retrieval policy for one context construction."""

        return RetrievalPolicy(
            relevance_floor=self.relevance_floor,
            max_memories=self.max_memories,
            near_duplicate_threshold=self.near_duplicate_threshold,
            weight_runtime_signals=self.weight_runtime_signals,
            weight_lexical=self.weight_lexical,
            weight_recency=self.weight_recency,
            recency_half_life_turns=self.recency_half_life_turns,
            recency_half_life_hours=self.recency_half_life_hours,
            annotate_memories=self.annotate_memories,
            resolve_conflicts=self.resolve_conflicts,
            drop_stale_memories=self.drop_stale_memories,
            drop_suspicious_memories=self.drop_suspicious_memories,
        )


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
    brain: Any = field(default=None, repr=False, compare=False)

    def clear_conversation(self) -> None:
        """Remove conversation content while retaining session configuration.

        The session-scoped BrainOS adapter is dropped so the next turn cannot
        reuse another conversation's runtime.
        """

        self.messages.clear()
        self.last_context.clear()
        self.diagnostics.clear()
        self.conversation_id = new_id()
        self.brain = None

    def clear_credentials(self) -> None:
        """Remove the active provider key from process memory."""

        self.provider = replace(self.provider, api_key="")
