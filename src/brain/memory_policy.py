"""Conservative policy for deciding what conversation content becomes memory."""

from __future__ import annotations

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


@dataclass(frozen=True)
class MemoryPolicy:
    """Initial conservative policy configuration.

    Extraction is intentionally left to a later implementation. This policy
    provides one place to enforce minimum confidence and supported categories.
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
