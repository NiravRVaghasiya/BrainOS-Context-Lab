"""Baseline context-management modes (plan Phase 6).

The plan requires five ways of turning a long conversation into a prompt, so
that "does BrainOS help?" is answered against real alternatives instead of a
straw man::

    Mode A  full_context      conversation ────────────────────────────→ LLM
    Mode B  sliding_window    conversation → last N turns ─────────────→ LLM
    Mode C  rag               conversation → lexical retriever → top-k ─→ LLM
    Mode D  brainos           conversation → BrainOS recall ───────────→ LLM
    Mode E  brainos_rag       BrainOS recall + lexical top-k ──────────→ LLM

Two rules make the comparison meaningful, and both come from measurements
rather than taste:

1. **Every mode fixes its own history window.** The Phase 3 sweep showed that
   context reduction is driven almost entirely by how much raw history is
   replayed: with a window larger than the conversation, "BrainOS mode"
   degenerated into full context *plus* memory overhead (0.0% reduction, 95
   tokens worse than the baseline). A mode is therefore defined by its window
   as much as by its retriever, and switching modes rewrites the budgets.
2. **The evidence budget is identical across C, D, and E.** Those three modes
   differ in *what selects* the evidence, so they are given the same token
   allowance for it (:data:`EVIDENCE_BUDGET`) and the same recent window. Mode
   E splits the allowance between its two sources instead of doubling it. Any
   accuracy difference is then attributable to selection, not to volume.

Modes A and B inject no retrieved evidence at all — they are the controls the
other three are compared against.

BrainOS still *observes* every turn in every mode. Observation is a
side process, not context construction, and keeping the runtime warm means a
user can switch modes mid-conversation without losing memory that the new mode
would have used. What a mode controls is what reaches the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Mode A — the plan's full-context baseline: replay the whole conversation.
MODE_FULL_CONTEXT = "full_context"
#: Mode B — the plan's sliding window: only the most recent N turns.
MODE_SLIDING_WINDOW = "sliding_window"
#: Mode C — conventional retrieval baseline, no BrainOS in the prompt path.
MODE_RAG = "rag"
#: Mode D — BrainOS recall supplies the evidence.
MODE_BRAINOS = "brainos"
#: Mode E — BrainOS recall plus lexical retrieval, sharing one budget.
MODE_BRAINOS_RAG = "brainos_rag"

#: Tokens of retrieved evidence Modes C/D/E may add to the prompt. Kept equal
#: across the three so the comparison isolates selection, not volume.
EVIDENCE_BUDGET = 1024
#: Tokens of raw recent history Modes C/D/E keep for conversational coherence.
RECENT_WINDOW_BUDGET = 256
#: Messages of raw recent history Modes C/D/E keep (one exchange).
RECENT_WINDOW_TURNS = 2
#: Turns Mode B retains. Deliberately much smaller than a benchmark
#: conversation, which is the entire point of a sliding-window control.
SLIDING_WINDOW_TURNS = 8
#: Evidence items each source may contribute.
EVIDENCE_ITEMS = 6
HYBRID_EVIDENCE_ITEMS = 3

#: The Phase 4/5 sidebar offered ``no_memory`` as the only memory-free control.
#: Mode B is that control with a defined window, so the old value is accepted
#: as an alias rather than rejected: an existing session or a saved
#: configuration keeps working after the vocabulary changed.
LEGACY_MODE_ALIASES: dict[str, str] = {"no_memory": MODE_SLIDING_WINDOW}


@dataclass(frozen=True)
class BaselineMode:
    """One context-management strategy and the budgets that define it.

    ``recent_turn_budget`` is ``None`` when the mode's history is limited by
    turn count rather than tokens (Modes A and B): the budget is then derived
    from the session's ``max_tokens`` ceiling, because "replay everything" and
    "replay the last N turns" must never be silently clipped by a window the
    user set for a different mode.
    """

    mode: str
    letter: str
    label: str
    description: str
    uses_memory: bool
    uses_rag: bool
    max_recent_turns: int | None
    recent_turn_budget: int | None
    memory_budget: int = 0
    chunk_budget: int = 0
    max_memories: int = EVIDENCE_ITEMS
    rag_top_k: int = EVIDENCE_ITEMS

    @property
    def evidence_budget(self) -> int:
        """Total tokens this mode allows for retrieved evidence."""

        return self.memory_budget + self.chunk_budget

    def budget_overrides(self, max_tokens: int) -> dict[str, Any]:
        """Return the ``ContextSettings`` fields this mode fixes.

        Applied when the mode changes, so the sidebar shows the budgets that
        will actually run. They stay editable afterwards — the mode defines the
        starting point of an experiment, not a hard ceiling on curiosity.
        """

        return {
            "max_recent_turns": self.max_recent_turns,
            "recent_turn_budget": (
                max(0, int(max_tokens)) if self.recent_turn_budget is None
                else self.recent_turn_budget
            ),
            "memory_budget": self.memory_budget,
            "chunk_budget": self.chunk_budget,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready description for diagnostics and export."""

        return {
            "mode": self.mode,
            "letter": self.letter,
            "label": self.label,
            "description": self.description,
            "uses_memory": self.uses_memory,
            "uses_rag": self.uses_rag,
            "max_recent_turns": self.max_recent_turns,
            "recent_turn_budget": self.recent_turn_budget,
            "memory_budget": self.memory_budget,
            "chunk_budget": self.chunk_budget,
            "evidence_budget": self.evidence_budget,
            "max_memories": self.max_memories,
            "rag_top_k": self.rag_top_k,
        }


MODES: dict[str, BaselineMode] = {
    MODE_FULL_CONTEXT: BaselineMode(
        mode=MODE_FULL_CONTEXT,
        letter="A",
        label="Mode A — Full context",
        description=(
            "Replay the entire conversation with no retrieval. The reference "
            "price every other mode is measured against."
        ),
        uses_memory=False,
        uses_rag=False,
        max_recent_turns=None,
        recent_turn_budget=None,
    ),
    MODE_SLIDING_WINDOW: BaselineMode(
        mode=MODE_SLIDING_WINDOW,
        letter="B",
        label="Mode B — Sliding window",
        description=(
            f"Keep only the last {SLIDING_WINDOW_TURNS} turns. Cheap and "
            "memory-free; forgets anything older."
        ),
        uses_memory=False,
        uses_rag=False,
        max_recent_turns=SLIDING_WINDOW_TURNS,
        recent_turn_budget=None,
    ),
    MODE_RAG: BaselineMode(
        mode=MODE_RAG,
        letter="C",
        label="Mode C — Lexical RAG",
        description=(
            "Conventional retrieval baseline: lexical top-k over transcript "
            "chunks, no BrainOS, no vector store."
        ),
        uses_memory=False,
        uses_rag=True,
        max_recent_turns=RECENT_WINDOW_TURNS,
        recent_turn_budget=RECENT_WINDOW_BUDGET,
        chunk_budget=EVIDENCE_BUDGET,
        rag_top_k=EVIDENCE_ITEMS,
    ),
    MODE_BRAINOS: BaselineMode(
        mode=MODE_BRAINOS,
        letter="D",
        label="Mode D — BrainOS memory",
        description=(
            "BrainOS observe/recall supplies the evidence; the recent window "
            "stays small so the prompt leans on memory."
        ),
        uses_memory=True,
        uses_rag=False,
        max_recent_turns=RECENT_WINDOW_TURNS,
        recent_turn_budget=RECENT_WINDOW_BUDGET,
        memory_budget=EVIDENCE_BUDGET,
    ),
    MODE_BRAINOS_RAG: BaselineMode(
        mode=MODE_BRAINOS_RAG,
        letter="E",
        label="Mode E — BrainOS + RAG",
        description=(
            "BrainOS memory and lexical chunks share one evidence budget, "
            "split evenly between the two sources."
        ),
        uses_memory=True,
        uses_rag=True,
        max_recent_turns=RECENT_WINDOW_TURNS,
        recent_turn_budget=RECENT_WINDOW_BUDGET,
        memory_budget=EVIDENCE_BUDGET // 2,
        chunk_budget=EVIDENCE_BUDGET - EVIDENCE_BUDGET // 2,
        max_memories=HYBRID_EVIDENCE_ITEMS,
        rag_top_k=HYBRID_EVIDENCE_ITEMS,
    ),
}

#: Plan order (A→E), which is also the order the UI lists them in.
MODE_ORDER: tuple[str, ...] = (
    MODE_FULL_CONTEXT,
    MODE_SLIDING_WINDOW,
    MODE_RAG,
    MODE_BRAINOS,
    MODE_BRAINOS_RAG,
)

#: The default mode. BrainOS is the system under test, so a new session starts
#: in the configuration the project exists to evaluate.
DEFAULT_MODE = MODE_BRAINOS


def resolve_mode(value: Any) -> str:
    """Normalize a mode selector, following legacy aliases.

    Raises :class:`ValueError` for anything else. An unknown mode is rejected
    rather than quietly treated as BrainOS: a typo in an evaluation
    configuration must not silently produce results labelled with a strategy
    that never ran.
    """

    text = str(value or "").strip()
    if text in LEGACY_MODE_ALIASES:
        text = LEGACY_MODE_ALIASES[text]
    if text not in MODES:
        raise ValueError(
            f"Unknown baseline mode {str(value)!r}. Expected one of "
            f"{', '.join(MODE_ORDER)}."
        )
    return text


def mode_profile(value: Any) -> BaselineMode:
    """Return the :class:`BaselineMode` for a selector value."""

    return MODES[resolve_mode(value)]


def mode_choices() -> tuple[tuple[str, str], ...]:
    """Return ``(label, value)`` pairs for a UI selector, in plan order."""

    return tuple((MODES[mode].label, mode) for mode in MODE_ORDER)


def mode_label(value: Any) -> str:
    """Return the human-readable label for a mode selector."""

    try:
        return mode_profile(value).label
    except ValueError:
        return str(value)


__all__ = [
    "DEFAULT_MODE",
    "EVIDENCE_BUDGET",
    "EVIDENCE_ITEMS",
    "HYBRID_EVIDENCE_ITEMS",
    "LEGACY_MODE_ALIASES",
    "MODES",
    "MODE_BRAINOS",
    "MODE_BRAINOS_RAG",
    "MODE_FULL_CONTEXT",
    "MODE_ORDER",
    "MODE_RAG",
    "MODE_SLIDING_WINDOW",
    "RECENT_WINDOW_BUDGET",
    "RECENT_WINDOW_TURNS",
    "SLIDING_WINDOW_TURNS",
    "BaselineMode",
    "mode_choices",
    "mode_label",
    "mode_profile",
    "resolve_mode",
]
