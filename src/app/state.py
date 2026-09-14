"""Typed state objects shared by the UI and application service layer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from baselines.ablations import ABLATIONS, POLICY_KNOB_FIELDS
from baselines.modes import (
    DEFAULT_MODE,
    EVIDENCE_BUDGET,
    EVIDENCE_ITEMS,
    MODE_BRAINOS,
    MODE_ORDER,
    MODE_SLIDING_WINDOW,
    MODES,
    RECENT_WINDOW_BUDGET,
    RECENT_WINDOW_TURNS,
    BaselineMode,
    mode_profile,
    resolve_mode,
)
from brain.context_builder import ContextBudget
from brain.retrieval_policy import RetrievalPolicy
from providers.base import ProviderConfig

#: Canonical baseline-mode selector values (plan Phase 6, modes A–E).
CONTEXT_MODES: tuple[str, ...] = MODE_ORDER
#: The Phase 4/5 name for the same tuple, kept so existing imports keep working.
#: The vocabulary itself grew from two memory selectors to the plan's five
#: context-management strategies; ``no_memory`` is accepted as an alias for
#: Mode B (see :data:`baselines.modes.LEGACY_MODE_ALIASES`).
MEMORY_MODES: tuple[str, ...] = MODE_ORDER
BRAINOS_MODE = MODE_BRAINOS
NO_MEMORY_MODE = MODE_SLIDING_WINDOW


def new_id() -> str:
    """Return a non-secret identifier for a session or conversation."""

    return str(uuid4())


@dataclass
class ContextSettings:
    """Budgets and retrieval settings used by the context construction layer.

    ``mode`` selects one of the plan's five baseline strategies (Phase 6); the
    remaining fields are the knobs that strategy runs with. They live on session
    state so the UI and the evaluation runner configure context construction
    through one object instead of constructing dataclasses inline.

    The defaults *are* Mode D's canonical profile, not a neutral starting point.
    Phase 3 measured that a BrainOS session with a history window larger than
    the conversation degenerates into full context plus memory overhead (0.0%
    reduction, 95 tokens worse than the baseline), so the default window is the
    small one the strategy needs. :meth:`with_mode_defaults` rewrites the same
    fields whenever the mode changes.
    """

    mode: str = DEFAULT_MODE
    max_tokens: int = 4096
    recent_turn_budget: int = RECENT_WINDOW_BUDGET
    memory_budget: int = EVIDENCE_BUDGET
    chunk_budget: int = 0
    system_budget: int = 512
    max_recent_turns: int | None = RECENT_WINDOW_TURNS
    per_message_overhead: int = 4
    max_memories: int = EVIDENCE_ITEMS
    rag_top_k: int = EVIDENCE_ITEMS
    relevance_floor: float = 0.12
    relative_relevance_ratio: float = 0.55
    near_duplicate_threshold: float = 0.88
    weight_runtime_signals: float = 0.45
    weight_lexical: float = 0.35
    weight_recency: float = 0.20
    recency_half_life_turns: float = 20.0
    recency_half_life_hours: float = 48.0
    annotate_memories: bool = True
    resolve_conflicts: bool = True
    drop_stale_memories: bool = True
    heuristic_conflict_detection: bool = True
    drop_suspicious_memories: bool = False

    def mode_profile(self) -> BaselineMode:
        """Return the :class:`~baselines.modes.BaselineMode` this session runs.

        Raises :class:`ValueError` for an unknown selector: a typo must surface
        as an error rather than silently run a different strategy.
        """

        return mode_profile(self.mode)

    def uses_memory(self) -> bool:
        """Whether this mode injects BrainOS memories into the prompt.

        BrainOS still *observes* in every mode; this answers only "does recalled
        memory reach the model". Modes A, B, and C are the memory-free controls.
        """

        return self.mode_profile().uses_memory

    def uses_rag(self) -> bool:
        """Whether this mode injects lexically retrieved transcript chunks."""

        return self.mode_profile().uses_rag

    def with_mode_defaults(self, mode: Any) -> ContextSettings:
        """Return a copy switched to ``mode`` with that mode's budgets applied.

        This is the seam the Phase 3/4 finding demands: switching strategy
        rewrites the history window and the evidence budgets, so Mode A and
        Mode D cannot accidentally run with the same window and produce a
        comparison that measures nothing. The values stay editable afterwards —
        the mode sets the experiment's starting point, and the sidebar shows
        exactly what was applied.

        Switching to a Phase 11 ablation additionally applies that ablation's
        retrieval-policy overrides (the component removal), while switching
        between the five baselines keeps the caller's policy tuning untouched.
        Switching *away* from an ablation restores the overridden knobs to
        their defaults: the knobs are the ablation, not user tuning, so
        carrying them into ``brainos`` would silently run a different system
        than the label claims.
        """

        resolved = resolve_mode(mode)
        profile = MODES[resolved]
        switched = replace(
            self, mode=resolved, **profile.budget_overrides(self.max_tokens)
        )
        ablation = ABLATIONS.get(resolved)
        if ablation is not None:
            switched = replace(switched, **ablation.policy_overrides)
        elif self.mode in ABLATIONS:
            defaults = ContextSettings()
            switched = replace(
                switched,
                **{name: getattr(defaults, name) for name in POLICY_KNOB_FIELDS},
            )
        return switched

    def context_budget(self) -> ContextBudget:
        """Return the token budget for one context construction."""

        return ContextBudget(
            max_tokens=self.max_tokens,
            recent_turn_budget=self.recent_turn_budget,
            memory_budget=self.memory_budget,
            chunk_budget=self.chunk_budget,
            system_budget=self.system_budget,
            max_recent_turns=self.max_recent_turns,
            per_message_overhead=self.per_message_overhead,
        )

    def retrieval_policy(self) -> RetrievalPolicy:
        """Return the retrieval policy for one context construction."""

        return RetrievalPolicy(
            relevance_floor=self.relevance_floor,
            relative_relevance_ratio=self.relative_relevance_ratio,
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
            heuristic_conflict_detection=self.heuristic_conflict_detection,
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
