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
# Declarative-verb evidence. The original list (is/are/uses/used/has/have/was/
# were) silently rejected real project facts such as "Deployments happen every
# Friday at 17:00 UTC" and "The retention policy requires 400 days of logs",
# while accepting small talk that happened to contain "are". A fact that is never
# stored cannot be retrieved, so this set covers the common declarative verbs
# that state how something is configured, scheduled, or constrained.
_STATEMENT_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|use|uses|used|run|runs|running|"
    r"require|requires|required|need|needs|happen|happens|occur|occurs|"
    r"contain|contains|include|includes|support|supports|expire|expires|"
    r"last|lasts|take|takes|cost|costs|mean|means|start|starts|begin|begins|"
    r"end|ends|deploy|deploys|deployed|migrate|migrates|migrated|switch|switched|"
    r"move|moves|moved|schedule|scheduled|configure|configured|limit|limited|"
    r"cap|capped|retain|retains|keep|keeps|store|stores)\b",
    re.IGNORECASE,
)
# Kept as an alias: the temporal/project rules below read as "states something".
_FACT_RE = _STATEMENT_RE

# Interrogative and request openers. A question is a request for retrieval, not a
# candidate for storage; storing it would pollute memory with the user's own
# queries and make every later recall match its own question.
_INTERROGATIVE_OPENER_RE = re.compile(
    r"^(?:what|when|where|why|how|who|whom|whose|which|can|could|would|should|"
    r"shall|may|might|do|does|did|is|are|was|were|am|tell me|remind me|explain|"
    r"summarise|summarize|help me|i wonder|let'?s|let us|please could)\b",
    re.IGNORECASE,
)

# Conversational filler that carries no durable information.
_FILLER_RE = re.compile(
    r"\b(?:i am thinking|i'?m thinking|just making conversation|i wonder|"
    r"let'?s talk about|let us talk about|by the way|anyway|small talk|"
    r"chit[- ]?chat|nice weather|how are you|thanks for|thank you for)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


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


def is_question_or_filler(sentence: str) -> bool:
    """Return whether a sentence is a question, a request, or conversational filler.

    Applied per sentence so a turn that mixes a question with a statement
    ("What do we use? PostgreSQL 16.") still contributes its durable half.
    """

    stripped = sentence.strip()
    if not stripped:
        return True
    if "?" in stripped:
        return True
    if _INTERROGATIVE_OPENER_RE.match(stripped):
        return True
    return bool(_FILLER_RE.search(stripped))


def extract_candidates(text: str, *, source: str = "conversation") -> list[MemoryCandidate]:
    """Return conservative memory candidates for a turn.

    A single-sentence turn is classified as a whole, preserving the original
    behaviour. Multi-sentence turns are classified per sentence so one durable
    fact is not lost because it shares a turn with a question or filler.
    Questions, greetings, requests, and short chatter produce no candidates.
    """

    stripped = text.strip()
    if len(stripped) < 12:
        return []
    if _GREETING_RE.match(stripped) and len(stripped) < 48:
        return []

    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
    if len(sentences) <= 1:
        if is_question_or_filler(stripped):
            return []
        return _candidate_for(stripped, source=source)

    candidates: list[MemoryCandidate] = []
    for sentence in sentences:
        if len(sentence) < 12 or is_question_or_filler(sentence):
            continue
        candidates.extend(_candidate_for(sentence, source=source))
    return candidates


def _candidate_for(text: str, *, source: str) -> list[MemoryCandidate]:
    """Classify one declarative statement into zero or one candidate."""

    memory_type, confidence = _classify(text)
    if memory_type is None:
        return []
    return [
        MemoryCandidate(
            text=text,
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
