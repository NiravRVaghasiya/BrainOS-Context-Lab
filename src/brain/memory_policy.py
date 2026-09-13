"""Conservative policy for deciding what conversation content becomes memory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    PROJECT_STATE = "PROJECT_STATE"
    DECISION = "DECISION"
    GOAL = "GOAL"
    TASK = "TASK"
    TEMPORAL_EVENT = "TEMPORAL_EVENT"
    CORRECTION = "CORRECTION"
    CONSTRAINT = "CONSTRAINT"


@dataclass(frozen=True)
class MemoryCandidate:
    """Candidate extracted from a turn before it is sent to BrainOS."""

    text: str
    memory_type: MemoryType
    confidence: float = 0.0
    source: str = "conversation"
    metadata: dict[str, Any] | None = None


_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yo|good morning|good evening)\b",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"\b(actually|correction|instead of|changed to|not\s+\w+.+\b(it's|it is|use))\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(prefer|preference|like to|don't like|do not like|always use|never use)\b",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(must not|must|cannot|can't|required to|constraint|never)\b",
    re.IGNORECASE,
)
_GOAL_RE = re.compile(r"\b(goal|objective|aim to|we need to|deadline)\b", re.IGNORECASE)
_TASK_RE = re.compile(
    r"\b(todo|task|please (do|implement|fix|add)|need to (implement|fix|add))\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(decided|decision|we'll use|we will use|going with)\b",
    re.IGNORECASE,
)
_PROJECT_RE = re.compile(
    r"\b(project|database|deploy|deployment|production|staging|provider)\b",
    re.IGNORECASE,
)
_TEMPORAL_RE = re.compile(
    r"\b(friday|monday|tuesday|wednesday|thursday|saturday|sunday|at \d{1,2}:\d{2}|"
    r"deadline|every (week|day|friday)|valid (from|until))\b",
    re.IGNORECASE,
)
_FACT_RE = re.compile(r"\b(is|are|uses|used|has|have|was|were)\b", re.IGNORECASE)


@dataclass(frozen=True)
class MemoryPolicy:
    """Initial conservative policy configuration.

    Extraction is heuristic and intentionally narrow: greetings, questions, and
    low-confidence chatter are not stored. Callers that already know a turn is
    worth remembering can pass an explicit candidate.
    """

    minimum_confidence: float = 0.75
    allowed_types: frozenset[MemoryType] = frozenset(MemoryType)
    max_items_per_turn: int = 5

    def accepts(self, candidate: MemoryCandidate) -> bool:
        """Return whether a candidate is safe to pass to the memory runtime."""

        return (
            candidate.memory_type in self.allowed_types
            and candidate.confidence >= self.minimum_confidence
            and bool(candidate.text.strip())
        )

    def select(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        """Filter and cap candidates while preserving their original order."""

        accepted = [candidate for candidate in candidates if self.accepts(candidate)]
        return accepted[: self.max_items_per_turn]


def extract_candidates(text: str, *, source: str = "conversation") -> list[MemoryCandidate]:
    """Return zero or one conservative memory candidate for a turn.

    The first version classifies the whole turn rather than splitting sentences.
    Questions, greetings, and short chatter produce no candidates.
    """

    stripped = text.strip()
    if len(stripped) < 12:
        return []
    if stripped.endswith("?"):
        return []
    if _GREETING_RE.match(stripped) and len(stripped) < 48:
        return []

    memory_type, confidence = _classify(stripped)
    if memory_type is None:
        return []
    return [
        MemoryCandidate(
            text=stripped,
            memory_type=memory_type,
            confidence=confidence,
            source=source,
        )
    ]


def _classify(text: str) -> tuple[MemoryType | None, float]:
    if _CORRECTION_RE.search(text):
        return MemoryType.CORRECTION, 0.92
    if _PREFERENCE_RE.search(text):
        return MemoryType.PREFERENCE, 0.88
    if _CONSTRAINT_RE.search(text):
        return MemoryType.CONSTRAINT, 0.86
    if _DECISION_RE.search(text):
        return MemoryType.DECISION, 0.86
    if _GOAL_RE.search(text):
        return MemoryType.GOAL, 0.84
    if _TASK_RE.search(text):
        return MemoryType.TASK, 0.84
    if _TEMPORAL_RE.search(text) and _FACT_RE.search(text):
        return MemoryType.TEMPORAL_EVENT, 0.86
    if _PROJECT_RE.search(text) and _FACT_RE.search(text):
        return MemoryType.PROJECT_STATE, 0.9
    if _FACT_RE.search(text) and len(text) >= 20:
        return MemoryType.FACT, 0.8
    return None, 0.0
