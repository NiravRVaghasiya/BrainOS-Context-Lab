"""Retrieval policy for the context construction engine.

This module implements the pipeline defined by the implementation plan between
BrainOS recall and the token budget::

    current query
       → BrainOS recall
       → deduplicate
       → relevance filter
       → conflict check
       → recency weighting
       → (token budget, applied by ``context_builder``)
       → final context

Design rules

* **BrainOS stays authoritative.** When the pinned runtime reports retrieval
  signals (``why()`` → ``score``/``signals``), lifecycle status, supersession,
  or contradictions, those values are consumed as-is. The heuristics below are
  only a fallback for runtimes (or test doubles) that do not expose them.
* **Deterministic.** Scoring, tie-breaking, and dropping never depend on set
  iteration order or wall-clock time unless a timestamp is injected, so an
  evaluation run can be replayed exactly.
* **Auditable.** Every dropped memory records a reason. Those reasons are the
  raw material for the later metrics (Precision@K) and error-analysis phases
  (``irrelevant_memory``, ``stale_memory``, ``conflicting_memory``).
* **Dependency-free.** No embeddings, vector stores, or network calls. Lexical
  overlap plus runtime signals is enough to measure the research question.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from security.guard import GuardedText, detect_injection, neutralize

from .adapter import Conflict, MemoryRecord

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    {
        "a", "about", "above", "again", "all", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him",
        "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
        "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
        "or", "other", "our", "ours", "out", "over", "own", "s", "same", "she", "should", "so",
        "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "us",
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "would", "you", "your", "yours",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_WHITESPACE_RE = re.compile(r"\s+")

#: Markers that make an explicit replacement/correction claim.
CORRECTION_MARKERS = (
    "actually",
    "correction",
    "correct that",
    "instead of",
    "no longer",
    "not anymore",
    "changed to",
    "change to",
    "switched to",
    "migrated to",
    "moved to",
    "upgraded to",
    "now uses",
    "now use",
    "as of",
    "replaced",
    "superseded",
    "was wrong",
    "my mistake",
)

#: Lifecycle statuses that must never reach a model prompt.
_INACTIVE_STATUSES = frozenset(
    {"superseded", "expired", "deleted", "quarantined", "archived", "tombstoned"}
)


def normalize_text(text: str) -> str:
    """Casefold, strip accents, and collapse whitespace for comparison purposes."""

    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return _WHITESPACE_RE.sub(" ", folded.casefold()).strip()


def stem(token: str) -> str:
    """Apply a tiny deterministic suffix stemmer.

    Only the inflections that matter for fact retrieval are handled
    (``uses``/``use``, ``databases``/``database``, ``deployed``/``deploy``,
    ``running``/``run``). A real stemmer is deliberately avoided: it would add a
    dependency and change benchmark results without being measurable here.
    """

    word = token
    if len(word) > 4 and word.endswith("ing"):
        word = word[:-3]
        if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiou":
            word = word[:-1]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def content_tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Return stemmed content tokens in order of appearance."""

    tokens = [stem(match.group(0)) for match in _TOKEN_RE.finditer(normalize_text(text))]
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in _STOPWORDS]


def token_set(text: str) -> set[str]:
    """Return the deduplicated content-token set of ``text``."""

    return set(content_tokens(text))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """Jaccard similarity of two token collections."""

    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    """Return adjacent token pairs, which catch phrases token overlap misses."""

    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


# --------------------------------------------------------------------------- #
# Memory text guard
# --------------------------------------------------------------------------- #


def inspect_memory_text(text: str, *, max_chars: int = 600) -> GuardedText:
    """Guard one retrieved string and return the full audit of what happened.

    Phase 13 moved the mechanics into :mod:`security.guard` so one vocabulary
    describes every route into a prompt (memory, transcript chunk, history) and
    so a guard action can be *reported* rather than only applied. The returned
    :class:`~security.guard.GuardedText` carries the guarded text, the attack
    families seen in the original, and which structural categories were removed
    — the raw material for the security findings ledger.
    """

    return neutralize(text, max_chars=max_chars)


def neutralize_memory_text(text: str, *, max_chars: int = 600) -> tuple[str, bool]:
    """Make one retrieved memory safe to embed inside a delimited prompt block.

    Returns the guarded text and whether it looked like an instruction-override
    attempt. Guarding is intentionally lossless for ordinary prose: it removes
    invisible characters, delimiter and chat-template breakouts, control
    characters, and leading role labels, collapses newlines so a memory renders
    as one bullet, and truncates pathological length. Suspicious text is
    *flagged*, not silently deleted — dropping it is a policy decision
    (:attr:`RetrievalPolicy.drop_suspicious_memories`) so the security phase can
    measure both behaviours.
    """

    return neutralize(text, max_chars=max_chars).as_tuple()


# --------------------------------------------------------------------------- #
# Policy configuration
# --------------------------------------------------------------------------- #

#: Blend of the runtime's own ``[0, 1]`` retrieval signals. ``semantic`` is only
#: informative when an embedding plugin is active; weights of signals that are
#: uniformly zero across a candidate set are redistributed (see
#: :func:`runtime_component`) so the relevance floor keeps its meaning offline.
_RUNTIME_SIGNAL_WEIGHTS: Mapping[str, float] = {
    "lexical": 0.30,
    "semantic": 0.25,
    "task_relevance": 0.10,
    "salience": 0.15,
    "confidence": 0.05,
    "recency": 0.10,
    "temporal_relevance": 0.05,
}
#: The subset of runtime signals that actually depend on the query. Only these
#: count as relevance evidence for the floor: salience, confidence, and the
#: runtime's own recency decay are worth ranking on, but they must not let an
#: off-topic memory pass a relevance filter.
_QUERY_DEPENDENT_SIGNALS: Mapping[str, float] = {
    "lexical": 0.60,
    "semantic": 0.25,
    "task_relevance": 0.15,
}
_RUNTIME_PENALTIES: Mapping[str, float] = {"redundancy": 0.50, "token_cost": 0.25}


@dataclass(frozen=True)
class RetrievalPolicy:
    """Knobs for deduplication, relevance, conflict, and recency handling.

    Defaults are conservative: they keep BrainOS's own ranking dominant, drop
    only clearly irrelevant or clearly superseded memories, and never delete
    suspicious text without an explicit opt-in.
    """

    relevance_floor: float = 0.12
    #: Scale-free tail trim: drop candidates whose relevance falls below this
    #: fraction of the best candidate's relevance. An absolute floor alone cannot
    #: separate "one strong match plus noise" from "nothing matches", because the
    #: score distribution shifts with query and corpus; the ratio adapts per query.
    relative_relevance_ratio: float = 0.55
    max_memories: int = 12
    near_duplicate_threshold: float = 0.88
    weight_runtime_signals: float = 0.45
    weight_lexical: float = 0.35
    weight_recency: float = 0.20
    entity_match_bonus: float = 0.15
    recency_half_life_hours: float = 48.0
    recency_half_life_turns: float = 20.0
    recency_half_life_positions: float = 4.0
    max_memory_chars: int = 600
    annotate_memories: bool = True
    resolve_conflicts: bool = True
    drop_stale_memories: bool = True
    heuristic_conflict_detection: bool = True
    drop_suspicious_memories: bool = False
    neutralize_text: bool = True

    def __post_init__(self) -> None:
        for name in (
            "relevance_floor",
            "near_duplicate_threshold",
            "weight_runtime_signals",
            "weight_lexical",
            "weight_recency",
            "entity_match_bonus",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a number.")
        if not 0.0 <= self.relevance_floor <= 1.0:
            raise ValueError("relevance_floor must be between 0 and 1.")
        if not 0.0 <= self.relative_relevance_ratio <= 1.0:
            raise ValueError("relative_relevance_ratio must be between 0 and 1.")
        if not 0.0 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be between 0 and 1.")
        if self.weight_runtime_signals < 0 or self.weight_lexical < 0 or self.weight_recency < 0:
            raise ValueError("Retrieval weights cannot be negative.")
        if (
            self.weight_runtime_signals
            + self.weight_lexical
            + self.weight_recency
        ) <= 0:
            raise ValueError("At least one retrieval weight must be positive.")
        if self.max_memories < 0:
            raise ValueError("max_memories cannot be negative.")
        if self.recency_half_life_hours <= 0 or self.recency_half_life_turns <= 0:
            raise ValueError("Recency half-lives must be greater than zero.")
        if self.recency_half_life_positions <= 0:
            raise ValueError("recency_half_life_positions must be greater than zero.")


@dataclass(frozen=True)
class ScoredMemory:
    """One retained memory with the components behind its score."""

    record: MemoryRecord
    text: str
    score: float
    rank: int
    lexical: float = 0.0
    relevance: float = 0.0
    runtime: float | None = None
    recency: float = 0.0
    merged_ids: tuple[str, ...] = ()
    contested: bool = False
    suspicious: bool = False
    position: int = 0
    #: Phase 13: the attack families the guard saw in this memory's original
    #: text, and the structural categories it removed. Empty for ordinary prose.
    #: ``suspicious`` stays the single boolean the drop policy reads; these two
    #: are what make the flag explainable in a panel and in a security report.
    suspicious_families: tuple[str, ...] = ()
    neutralized: tuple[str, ...] = ()

    @property
    def memory_id(self) -> str:
        return self.record.memory_id

    def components(self) -> dict[str, Any]:
        """Return the score breakdown for UI inspection and evaluation exports."""

        return {
            "memory_id": self.record.memory_id,
            "memory_type": self.record.memory_type,
            "score": round(self.score, 6),
            "relevance": round(self.relevance, 6),
            "lexical": round(self.lexical, 6),
            "runtime": None if self.runtime is None else round(self.runtime, 6),
            "recency": round(self.recency, 6),
            "rank": self.rank,
            "position": self.position,
            "merged_ids": list(self.merged_ids),
            "contested": self.contested,
            "suspicious": self.suspicious,
            "suspicious_families": list(self.suspicious_families),
            "neutralized": list(self.neutralized),
        }


@dataclass(frozen=True)
class DroppedMemory:
    """An audit record for one memory that did not reach the prompt."""

    memory_id: str
    reason: str
    text: str = ""
    detail: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "reason": self.reason,
            "text": self.text,
            "detail": self.detail,
            "score": round(self.score, 6),
        }


@dataclass(frozen=True)
class ConflictResolution:
    """How one contradictory pair was resolved.

    ``dropped_memory_id`` is empty when the pair was flagged ``contested`` and
    both memories were kept — an unresolved conflict is still worth reporting,
    because silently presenting two contradictory facts as equally authoritative
    is its own failure mode.
    """

    subject: str
    kept_memory_id: str = ""
    dropped_memory_id: str = ""
    source: str = "runtime"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "kept_memory_id": self.kept_memory_id,
            "dropped_memory_id": self.dropped_memory_id,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RetrievalReport:
    """Per-stage accounting for one retrieval decision.

    The counts mirror the plan's pipeline so the evaluation phase can compute
    Precision@K and populate the error taxonomy without re-running retrieval.
    """

    query: str = ""
    candidate_count: int = 0
    empty_count: int = 0
    duplicate_count: int = 0
    stale_count: int = 0
    conflict_count: int = 0
    low_relevance_count: int = 0
    cap_count: int = 0
    suspicious_count: int = 0
    selected_count: int = 0
    dropped: tuple[DroppedMemory, ...] = ()
    conflicts: tuple[ConflictResolution, ...] = ()
    ranking: tuple[dict[str, Any], ...] = ()
    selected_ids: tuple[str, ...] = ()
    #: Phase 13 guard accounting, over every *candidate* the policy scored (not
    #: only the survivors): how many candidates matched each attack family, how
    #: many needed each structural rewrite, and how many were flagged at all.
    #: ``suspicious_count`` above stays the prompt-level number it always was.
    guard_family_counts: Mapping[str, int] = field(default_factory=dict)
    guard_action_counts: Mapping[str, int] = field(default_factory=dict)
    suspicious_candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "candidate_count": self.candidate_count,
            "empty_count": self.empty_count,
            "duplicate_count": self.duplicate_count,
            "stale_count": self.stale_count,
            "conflict_count": self.conflict_count,
            "low_relevance_count": self.low_relevance_count,
            "cap_count": self.cap_count,
            "suspicious_count": self.suspicious_count,
            "suspicious_candidate_count": self.suspicious_candidate_count,
            "guard_family_counts": dict(sorted(self.guard_family_counts.items())),
            "guard_action_counts": dict(sorted(self.guard_action_counts.items())),
            "selected_count": self.selected_count,
            "selected_ids": list(self.selected_ids),
            "dropped": [item.to_dict() for item in self.dropped],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "ranking": list(self.ranking),
        }

    def reason_counts(self) -> dict[str, int]:
        """Return how many memories were dropped for each reason."""

        counts: dict[str, int] = {}
        for item in self.dropped:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return counts


@dataclass(frozen=True)
class RetrievalSelection:
    """Ordered memories plus the audit report for one query."""

    ordered: tuple[ScoredMemory, ...] = ()
    report: RetrievalReport = field(default_factory=RetrievalReport)

    @property
    def records(self) -> list[MemoryRecord]:
        return [item.record for item in self.ordered]

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.ordered]


# --------------------------------------------------------------------------- #
# Scoring components
# --------------------------------------------------------------------------- #


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp into an aware datetime, or return ``None``."""

    if not value or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def document_frequencies(texts: Iterable[str]) -> dict[str, int]:
    """Count how many candidate memories contain each token.

    Used to discount query terms that every candidate shares. Without this a
    project name that appears in all memories ("Project Atlas") gives every
    memory the same relevance floor, so an off-topic question retrieves
    everything and Precision@K collapses.
    """

    frequencies: dict[str, int] = {}
    for text in texts:
        for token in set(content_tokens(text)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def inverse_document_frequency(term: str, frequencies: Mapping[str, int], count: int) -> float:
    """Return a smooth, strictly positive IDF weight for one term."""

    if count <= 0 or not frequencies:
        return 1.0
    return math.log(1.0 + count / (1.0 + frequencies.get(term, 0)))


def lexical_score(
    query_tokens: Sequence[str],
    memory_tokens: Sequence[str],
    *,
    entities: Iterable[str] = (),
    entity_bonus: float = 0.15,
    frequencies: Mapping[str, int] | None = None,
    candidate_count: int = 0,
) -> float:
    """Dependency-free lexical relevance in ``[0, 1]``.

    Query-term coverage dominates (a memory that answers the question must
    mention what was asked), with Jaccard and phrase-bigram overlap as secondary
    terms and an entity-match bonus when the runtime extracted entities.

    When ``frequencies`` is supplied, coverage and similarity are weighted by
    inverse document frequency over the candidate set, so a term shared by every
    memory contributes almost nothing and a discriminative term ("cache",
    "retention") dominates. Terms that no memory contains only inflate the
    denominator, which is what makes abstention fall out naturally.
    """

    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    memory_set = set(memory_tokens)
    if not memory_set:
        return 0.0
    shared = query_set & memory_set

    if frequencies:
        query_weight = sum(
            inverse_document_frequency(term, frequencies, candidate_count) for term in query_set
        )
        shared_weight = sum(
            inverse_document_frequency(term, frequencies, candidate_count) for term in shared
        )
        coverage = shared_weight / query_weight if query_weight > 0 else 0.0
        union_weight = sum(
            inverse_document_frequency(term, frequencies, candidate_count)
            for term in query_set | memory_set
        )
        similarity = shared_weight / union_weight if union_weight > 0 else 0.0
    else:
        coverage = len(shared) / len(query_set)
        similarity = len(shared) / len(query_set | memory_set)

    query_bigrams = bigrams(list(query_tokens))
    phrase = (
        len(query_bigrams & bigrams(list(memory_tokens))) / len(query_bigrams)
        if query_bigrams
        else 0.0
    )
    score = 0.60 * coverage + 0.20 * similarity + 0.20 * phrase

    entity_tokens = {token for entity in entities for token in content_tokens(str(entity))}
    if entity_tokens:
        matched = entity_tokens & query_set
        if matched:
            matched_weight = sum(
                inverse_document_frequency(term, frequencies or {}, candidate_count)
                for term in matched
            ) if frequencies else len(matched)
            entity_weight = sum(
                inverse_document_frequency(term, frequencies or {}, candidate_count)
                for term in entity_tokens
            ) if frequencies else len(entity_tokens)
            if entity_weight > 0:
                score += entity_bonus * min(1.0, matched_weight / entity_weight)
    return max(0.0, min(1.0, score))


def recency_score(
    record: MemoryRecord,
    *,
    position: int,
    current_turn: int | None = None,
    now: datetime | None = None,
    half_life_hours: float = 48.0,
    half_life_turns: float = 20.0,
    half_life_positions: float = 4.0,
) -> float:
    """Exponential recency in ``(0, 1]`` using the best available clock.

    Preference order: an observed/created timestamp, then a conversation turn,
    then the recall position (BrainOS returns its best match first, so position
    is a usable proxy when nothing else is known). Half-lives default to the
    pinned runtime's own 48-hour retrieval decay.
    """

    reference = now or datetime.now(timezone.utc)
    timestamp = parse_timestamp(record.observed_at) or parse_timestamp(record.created_at)
    if timestamp is not None:
        age_hours = max(0.0, (reference - timestamp).total_seconds() / 3600.0)
        return 0.5 ** (age_hours / max(half_life_hours, 1e-9))
    if record.source_turn is not None and current_turn is not None:
        age_turns = max(0, int(current_turn) - int(record.source_turn))
        return 0.5 ** (age_turns / max(half_life_turns, 1e-9))
    return 0.5 ** (max(0, position) / max(half_life_positions, 1e-9))


def query_dependent_component(
    record: MemoryRecord, *, informative: Iterable[str] = ()
) -> float | None:
    """Blend only the runtime signals that depend on the query.

    Returns ``None`` when the runtime reported no query-dependent evidence, so
    the caller can fall back to the application's own lexical score instead of
    treating a missing signal as zero relevance.
    """

    signals = record.signals or {}
    if not signals:
        return None
    keep = set(informative)
    total_weight = 0.0
    total = 0.0
    for name, weight in _QUERY_DEPENDENT_SIGNALS.items():
        if name not in signals:
            continue
        if keep and name not in keep:
            continue
        try:
            value = float(signals[name])
        except (TypeError, ValueError):
            continue
        total_weight += weight
        total += weight * max(0.0, min(1.0, value))
    if total_weight <= 0:
        return None
    return max(0.0, min(1.0, total / total_weight))


def relevance_evidence(
    *, lexical: float, runtime: float | None, policy: RetrievalPolicy
) -> float:
    """Return the query relevance used by the floor, in ``[0, 1]``.

    Recency is deliberately excluded: it ranks memories but must not rescue an
    off-topic one, otherwise the relevance filter cannot improve Precision@K.
    """

    total_weight = 0.0
    total = 0.0
    if runtime is not None and policy.weight_runtime_signals > 0:
        total_weight += policy.weight_runtime_signals
        total += policy.weight_runtime_signals * runtime
    if policy.weight_lexical > 0:
        total_weight += policy.weight_lexical
        total += policy.weight_lexical * lexical
    if total_weight <= 0:
        return 0.0
    return max(0.0, min(1.0, total / total_weight))


def _informative_signals(candidates: Sequence[MemoryRecord]) -> set[str]:
    """Return signal names that are non-zero for at least one candidate."""

    informative: set[str] = set()
    for record in candidates:
        for name, value in (record.signals or {}).items():
            try:
                if float(value) != 0.0:
                    informative.add(str(name))
            except (TypeError, ValueError):
                continue
    return informative


def runtime_component(
    record: MemoryRecord, *, informative: Iterable[str] = ()
) -> float | None:
    """Blend the runtime's own ``[0, 1]`` retrieval signals into one value.

    Returns ``None`` when the record carries no signals at all, which lets the
    caller renormalise the remaining weights instead of penalising runtimes that
    do not expose per-signal evidence. Signals that are uniformly zero across
    the candidate set (for example ``semantic`` without an embedding plugin) are
    excluded so the relevance floor stays comparable offline and online.
    """

    signals = record.signals or {}
    if not signals:
        return None
    keep = set(informative)
    total_weight = 0.0
    total = 0.0
    for name, weight in _RUNTIME_SIGNAL_WEIGHTS.items():
        if name not in signals:
            continue
        if keep and name not in keep:
            continue
        try:
            value = float(signals[name])
        except (TypeError, ValueError):
            continue
        total_weight += weight
        total += weight * max(0.0, min(1.0, value))
    if total_weight <= 0:
        return None
    component = total / total_weight
    for name, weight in _RUNTIME_PENALTIES.items():
        if name not in signals:
            continue
        try:
            component -= weight * max(0.0, min(1.0, float(signals[name])))
        except (TypeError, ValueError):
            continue
    return max(0.0, min(1.0, component))


def _composite_score(
    *,
    lexical: float,
    runtime: float | None,
    recency: float,
    policy: RetrievalPolicy,
) -> float:
    """Weighted mean over the components that are actually available."""

    total_weight = 0.0
    total = 0.0
    if runtime is not None and policy.weight_runtime_signals > 0:
        total_weight += policy.weight_runtime_signals
        total += policy.weight_runtime_signals * runtime
    if policy.weight_lexical > 0:
        total_weight += policy.weight_lexical
        total += policy.weight_lexical * lexical
    if policy.weight_recency > 0:
        total_weight += policy.weight_recency
        total += policy.weight_recency * recency
    if total_weight <= 0:
        return 0.0
    return max(0.0, min(1.0, total / total_weight))


def has_correction_marker(text: str) -> bool:
    """Return whether the text explicitly claims to replace earlier information."""

    normalized = normalize_text(text)
    return any(marker in normalized for marker in CORRECTION_MARKERS)


# --------------------------------------------------------------------------- #
# Conflict detection
# --------------------------------------------------------------------------- #

_SUBJECT_VALUE_PATTERNS = (
    re.compile(
        r"^(?P<subject>[^.?!]{2,60}?)\s+\b(?:is|are|uses|use|runs on|runs|hosted on|based on)\b"
        r"\s+(?P<value>[^.?!]{1,60})$",
        re.I,
    ),
    re.compile(
        r"^(?:we\s+|i\s+)?\b(?:migrated|switched|moved|changed|upgraded|downgraded)\b\s+"
        r"(?:the\s+|our\s+)?(?P<subject>[^.?!]{1,50}?)\s+\bto\b\s+(?P<value>[^.?!]{1,50})$",
        re.I,
    ),
    re.compile(
        r"^(?P<subject>[^:=?!]{2,50}?)(?<![\d:])\s*[:=]\s*(?!\d)(?P<value>[^:=?!]{1,50})$"
    ),
)
_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:actually|also|btw|by the way|note|correction|update|updated|fyi|hey|well|so|"
    r"now|today|yesterday|as of [^,]{1,40}|heads up)\s*[,:]?\s+",
    re.I,
)
_VALUE_KIND_PATTERNS = {
    "number": re.compile(r"\d"),
    "proper": re.compile(r"\b[A-Z][A-Za-z0-9_.+-]{2,}\b"),
    "day": re.compile(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|"
        r"april|may|june|july|august|september|october|november|december)\b",
        re.I,
    ),
    "time": re.compile(r"\b\d{1,2}:\d{2}\b|\b(?:utc|est|pst|cet)\b", re.I),
    "boolean": re.compile(r"\b(true|false|enabled|disabled|on|off|yes|no)\b", re.I),
}


@dataclass(frozen=True)
class SubjectClaim:
    """An ``attribute → value`` claim extracted from one memory."""

    subject: str
    value: str
    kinds: frozenset[str]

    @property
    def subject_tokens(self) -> frozenset[str]:
        return frozenset(content_tokens(self.subject))

    @property
    def value_tokens(self) -> frozenset[str]:
        return frozenset(content_tokens(self.value))


def split_sentences(text: str) -> list[str]:
    """Split a memory into sentences so multi-sentence turns yield claims."""

    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [part.strip() for part in parts if part.strip()]


def strip_discourse_markers(sentence: str) -> str:
    """Remove leading fillers that would hide a declarative claim.

    ``"Actually, we migrated the database to MySQL 8."`` carries the strongest
    replacement signal in a conversation, so the matcher must see past the
    discourse marker rather than fail on it.
    """

    current = sentence.strip()
    for _ in range(3):
        stripped = _DISCOURSE_PREFIX_RE.sub("", current, count=1).strip(" ,;:")
        if stripped == current.strip(" ,;:") or not stripped:
            break
        current = stripped
    return current


def extract_claims(text: str) -> list[SubjectClaim]:
    """Extract zero or more ``subject → value`` claims from a memory.

    Extraction is intentionally narrow: a claim must be a short declarative
    statement whose value looks like a value (a proper noun, number, day, time,
    or boolean). That restriction is what keeps unrelated statements about the
    same subject — "the database is PostgreSQL" versus "the database is
    encrypted at rest" — from being reported as contradictions.

    Value kinds are detected on the original casing because proper nouns are
    only visible before casefolding, while token comparison uses the normalized
    form.
    """

    claims: list[SubjectClaim] = []
    for sentence in split_sentences(text):
        candidate = strip_discourse_markers(sentence).strip(".!?;:, ").strip()
        if not candidate or len(candidate) > 160:
            continue
        for pattern in _SUBJECT_VALUE_PATTERNS:
            match = pattern.match(candidate)
            if not match:
                continue
            subject = match.group("subject").strip(" ,;")
            value = match.group("value").strip(" ,;")
            subject_tokens = content_tokens(subject)
            value_tokens = content_tokens(value)
            if not subject_tokens or not value_tokens:
                continue
            if len(value_tokens) > 8:
                continue
            kinds = frozenset(
                name
                for name, kind_pattern in _VALUE_KIND_PATTERNS.items()
                if kind_pattern.search(value)
            )
            if not kinds:
                continue
            if set(value_tokens) & set(subject_tokens):
                continue
            claims.append(SubjectClaim(subject=subject, value=value, kinds=kinds))
            break
    return claims


def _claims_conflict(left: SubjectClaim, right: SubjectClaim) -> bool:
    """Return whether two claims describe the same attribute with different values."""

    if not (left.kinds & right.kinds):
        return False
    left_tokens = set(left.subject_tokens)
    right_tokens = set(right.subject_tokens)
    if not (left_tokens & right_tokens):
        return False
    subject_similarity = jaccard(left_tokens, right_tokens)
    if subject_similarity < 0.6:
        return False
    left_value = set(left.value_tokens)
    right_value = set(right.value_tokens)
    if left_value & right_value:
        return False
    return jaccard(left_value, right_value) < 0.6


@dataclass(frozen=True)
class ClaimConflict:
    """A heuristically detected same-subject conflict between two memories.

    ``older``/``newer`` are ``None`` when the chronology cannot be established
    from a timestamp or conversation turn. Recall position is deliberately *not*
    used as an age proxy: BrainOS returns its best match first, so position
    expresses priority, not time. Guessing an order would let the heuristic
    delete the wrong memory.
    """

    left: ScoredMemory
    right: ScoredMemory
    subject: str
    older: ScoredMemory | None = None
    newer: ScoredMemory | None = None

    @property
    def ordered(self) -> bool:
        return self.older is not None and self.newer is not None


def detect_conflicts(scored: Sequence[ScoredMemory]) -> list[ClaimConflict]:
    """Heuristic same-subject conflict detection over already-ranked memories.

    Used only for pairs the runtime did not classify, so BrainOS remains
    authoritative.
    """

    found: list[ClaimConflict] = []
    for index, left in enumerate(scored):
        left_claims = extract_claims(left.text)
        if not left_claims:
            continue
        for right in scored[index + 1 :]:
            for left_claim in left_claims:
                for right_claim in extract_claims(right.text):
                    if not _claims_conflict(left_claim, right_claim):
                        continue
                    older, newer = _order_by_recency(left, right)
                    found.append(
                        ClaimConflict(
                            left=left,
                            right=right,
                            subject=left_claim.subject,
                            older=older,
                            newer=newer,
                        )
                    )
                    break
                else:
                    continue
                break
    return found


def _order_by_recency(
    left: ScoredMemory, right: ScoredMemory
) -> tuple[ScoredMemory | None, ScoredMemory | None]:
    """Return ``(older, newer)`` from timestamps or turns, or ``(None, None)``."""

    left_time = parse_timestamp(left.record.observed_at) or parse_timestamp(
        left.record.created_at
    )
    right_time = parse_timestamp(right.record.observed_at) or parse_timestamp(
        right.record.created_at
    )
    if left_time is not None and right_time is not None and left_time != right_time:
        return (left, right) if left_time < right_time else (right, left)
    left_turn = left.record.source_turn
    right_turn = right.record.source_turn
    if left_turn is not None and right_turn is not None and left_turn != right_turn:
        return (left, right) if left_turn < right_turn else (right, left)
    return None, None


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


def select_memories(
    query: str,
    records: Iterable[MemoryRecord],
    *,
    policy: RetrievalPolicy | None = None,
    conflicts: Iterable[Conflict] = (),
    stale_ids: Iterable[str] = (),
    current_turn: int | None = None,
    now: datetime | None = None,
) -> RetrievalSelection:
    """Run dedupe → relevance → conflict → recency → rank over recalled memories.

    ``conflicts`` and ``stale_ids`` come from the BrainOS runtime when it
    exposes them (``contradictions()`` / ``stale_memories()``); the heuristic
    fallback covers pairs the runtime did not classify. The token budget is
    applied later by the context builder so this function stays pure ranking.
    """

    active_policy = policy or RetrievalPolicy()
    materialized = list(records)
    dropped: list[DroppedMemory] = []
    resolutions: list[ConflictResolution] = []
    # Phase 13 guard accounting, over every candidate that gets scored.
    guard_families: dict[str, int] = {}
    guard_actions: dict[str, int] = {}
    suspicious_candidates = 0

    usable: list[tuple[int, MemoryRecord]] = []
    for position, record in enumerate(materialized):
        if not str(record.text or "").strip():
            dropped.append(
                DroppedMemory(
                    memory_id=record.memory_id,
                    reason="empty",
                    detail="memory has no text",
                )
            )
            continue
        status = str(record.status or "").strip().lower()
        if status in _INACTIVE_STATUSES:
            dropped.append(
                DroppedMemory(
                    memory_id=record.memory_id,
                    reason="stale",
                    text=preview_text(record.text),
                    detail=f"runtime status={status}",
                )
            )
            continue
        expired = _expired_at(record, now)
        if expired:
            dropped.append(
                DroppedMemory(
                    memory_id=record.memory_id,
                    reason="expired",
                    text=preview_text(record.text),
                    detail=f"valid_until={expired}",
                )
            )
            continue
        usable.append((position, record))

    candidates = [record for _, record in usable]
    informative = _informative_signals(candidates)
    query_tokens = content_tokens(query)
    frequencies = document_frequencies(record.text for record in candidates)

    scored: list[ScoredMemory] = []
    for position, record in usable:
        lexical = lexical_score(
            query_tokens,
            content_tokens(record.text),
            entities=record.entities,
            entity_bonus=active_policy.entity_match_bonus,
            frequencies=frequencies,
            candidate_count=len(candidates),
        )
        runtime = runtime_component(record, informative=informative)
        query_runtime = query_dependent_component(record, informative=informative)
        relevance = relevance_evidence(
            lexical=lexical, runtime=query_runtime, policy=active_policy
        )
        recency = recency_score(
            record,
            position=position,
            current_turn=current_turn,
            now=now,
            half_life_hours=active_policy.recency_half_life_hours,
            half_life_turns=active_policy.recency_half_life_turns,
            half_life_positions=active_policy.recency_half_life_positions,
        )
        score = _composite_score(
            lexical=lexical,
            runtime=runtime,
            recency=recency,
            policy=active_policy,
        )
        if active_policy.neutralize_text:
            guarded = inspect_memory_text(
                record.text, max_chars=active_policy.max_memory_chars
            )
            text, suspicious = guarded.text, guarded.suspicious
            families, neutralized = guarded.families, guarded.neutralized
        else:
            # Text is passed through untouched, but detection still runs: a
            # policy that disables rewriting must not also disable *seeing*.
            text = str(record.text).strip()
            families = detect_injection(record.text)
            suspicious, neutralized = bool(families), ()
        for family in families:
            guard_families[family] = guard_families.get(family, 0) + 1
        for category in neutralized:
            guard_actions[category] = guard_actions.get(category, 0) + 1
        suspicious_candidates += 1 if suspicious else 0
        scored.append(
            ScoredMemory(
                record=record,
                text=text,
                score=score,
                rank=0,
                lexical=lexical,
                relevance=relevance,
                runtime=runtime,
                recency=recency,
                suspicious=suspicious,
                position=position,
                suspicious_families=families,
                neutralized=neutralized,
            )
        )

    # Highest score first; ties break on the runtime's own recall order so the
    # result never depends on dict/set iteration.
    scored.sort(key=lambda item: (-item.score, item.position, item.record.memory_id))

    deduplicated = _deduplicate(scored, active_policy, dropped)
    stale_dropped, contested = _apply_stale_and_conflicts(
        deduplicated,
        active_policy,
        list(conflicts),
        {str(value) for value in stale_ids if value},
        dropped,
        resolutions,
    )

    relevant = _apply_relevance_floor(stale_dropped, active_policy, dropped)
    capped, cap_dropped = _apply_cap(relevant, active_policy, dropped)

    ranked = tuple(
        replace(item, rank=index + 1, contested=item.memory_id in contested)
        for index, item in enumerate(capped)
    )
    report = RetrievalReport(
        query=query,
        candidate_count=len(materialized),
        empty_count=sum(1 for item in dropped if item.reason == "empty"),
        duplicate_count=sum(1 for item in dropped if item.reason == "duplicate"),
        stale_count=sum(1 for item in dropped if item.reason in {"stale", "expired"}),
        conflict_count=sum(1 for item in dropped if item.reason == "superseded"),
        low_relevance_count=sum(
            1 for item in dropped if item.reason in {"low_relevance", "weak_relevance"}
        ),
        cap_count=sum(1 for item in dropped if item.reason in {"cap", "suspicious"}),
        suspicious_count=sum(1 for item in ranked if item.suspicious) + cap_dropped,
        selected_count=len(ranked),
        guard_family_counts=dict(sorted(guard_families.items())),
        guard_action_counts=dict(sorted(guard_actions.items())),
        suspicious_candidate_count=suspicious_candidates,
        dropped=tuple(dropped),
        conflicts=tuple(resolutions),
        ranking=tuple(item.components() for item in ranked),
        selected_ids=tuple(item.memory_id for item in ranked),
    )
    return RetrievalSelection(ordered=ranked, report=report)


def preview_text(text: str, limit: int = 160) -> str:
    """Return a short single-line excerpt for audit records."""

    collapsed = _WHITESPACE_RE.sub(" ", str(text)).strip()
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def _expired_at(record: MemoryRecord, now: datetime | None) -> str | None:
    """Return the validity stamp when a memory is past its ``valid_until``."""

    valid_until = parse_timestamp(record.valid_until)
    if valid_until is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return record.valid_until if valid_until < reference else None


def _deduplicate(
    scored: Sequence[ScoredMemory],
    policy: RetrievalPolicy,
    dropped: list[DroppedMemory],
) -> list[ScoredMemory]:
    """Merge exact and near-duplicate memories into their best representative."""

    kept: list[ScoredMemory] = []
    for item in scored:
        tokens = token_set(item.text)
        match: ScoredMemory | None = None
        item_tokens = content_tokens(item.text)
        for existing in kept:
            if content_tokens(existing.text) == item_tokens:
                match = existing
                detail = "exact duplicate"
                break
            if jaccard(tokens, token_set(existing.text)) >= policy.near_duplicate_threshold:
                match = existing
                detail = "near duplicate"
                break
        if match is None:
            kept.append(item)
            continue
        index = kept.index(match)
        merged = replace(match, merged_ids=(*match.merged_ids, item.memory_id))
        kept[index] = merged
        dropped.append(
            DroppedMemory(
                memory_id=item.memory_id,
                reason="duplicate",
                text=preview_text(item.text),
                detail=f"{detail} of {match.memory_id}",
                score=item.score,
            )
        )
    return kept


def _apply_stale_and_conflicts(
    scored: Sequence[ScoredMemory],
    policy: RetrievalPolicy,
    conflicts: list[Conflict],
    stale_ids: set[str],
    dropped: list[DroppedMemory],
    resolutions: list[ConflictResolution],
) -> tuple[list[ScoredMemory], set[str]]:
    """Drop superseded/stale memories and resolve contradictory pairs.

    Two stages, in this order:

    1. **Runtime evidence.** BrainOS contradictions and stale reports are
       applied directly, and the pairs they cover are excluded from stage 2.
    2. **Application heuristic.** For pairs the runtime did not classify, an
       older claim is dropped only when the newer claim explicitly presents
       itself as a replacement (a correction marker or a ``CORRECTION`` memory
       type). Otherwise both survive and are flagged ``contested``, so a false
       positive can never silently delete a user fact.

    Drop reasons are kept distinct — ``stale`` for a runtime staleness report
    and ``superseded`` for a resolved contradiction — because the evaluation
    phase scores them separately.
    """

    if not policy.drop_stale_memories and not policy.resolve_conflicts:
        return list(scored), set()

    contested: set[str] = set()
    by_id = {item.memory_id: item for item in scored}
    runtime_drops: dict[str, tuple[str, str, str]] = {}
    runtime_pairs: set[frozenset[str]] = set()

    if policy.resolve_conflicts:
        for conflict in conflicts:
            older = by_id.get(str(conflict.older_id or "")) or _find_by_text(
                scored, conflict.older_text
            )
            newer = by_id.get(str(conflict.newer_id or "")) or _find_by_text(
                scored, conflict.newer_text
            )
            if older is None:
                continue
            if newer is None:
                # The runtime says a newer version exists but it was not
                # recalled: keep the older claim visible and flag it.
                contested.add(older.memory_id)
                continue
            runtime_pairs.add(frozenset({older.memory_id, newer.memory_id}))
            runtime_drops[older.memory_id] = (newer.memory_id, conflict.subject, "superseded")

    if policy.drop_stale_memories:
        for stale_id in sorted(stale_ids):
            item = by_id.get(stale_id)
            if item is None or stale_id in runtime_drops:
                continue
            runtime_drops[stale_id] = ("", "", "stale")

    survivors: list[ScoredMemory] = []
    for item in scored:
        drop = runtime_drops.get(item.memory_id)
        if drop is None:
            survivors.append(item)
            continue
        newer_id, subject, reason = drop
        detail = f"superseded by {newer_id}" if newer_id else "reported stale by runtime"
        dropped.append(
            DroppedMemory(
                memory_id=item.memory_id,
                reason=reason,
                text=preview_text(item.text),
                detail=f"{detail} (subject={subject or 'unknown'}, source=runtime)",
                score=item.score,
            )
        )
        if newer_id:
            resolutions.append(
                ConflictResolution(
                    subject=subject,
                    kept_memory_id=newer_id,
                    dropped_memory_id=item.memory_id,
                    source="runtime",
                    reason=detail,
                )
            )

    if not (policy.heuristic_conflict_detection and policy.resolve_conflicts):
        return survivors, contested

    heuristic_drops: dict[str, tuple[str, str]] = {}
    for found in detect_conflicts(survivors):
        if frozenset({found.left.memory_id, found.right.memory_id}) in runtime_pairs:
            continue
        if not found.ordered:
            contested.update({found.left.memory_id, found.right.memory_id})
            resolutions.append(
                ConflictResolution(
                    subject=found.subject,
                    source="heuristic",
                    reason=(
                        "contested: no timestamp or conversation turn establishes "
                        "which claim is newer, both memories kept"
                    ),
                )
            )
            continue
        older, newer = found.older, found.newer
        assert older is not None and newer is not None  # narrowed by ``found.ordered``
        if not _is_replacement(newer):
            contested.update({older.memory_id, newer.memory_id})
            resolutions.append(
                ConflictResolution(
                    subject=found.subject,
                    kept_memory_id=newer.memory_id,
                    source="heuristic",
                    reason=(
                        "contested: newer claim lacks an explicit correction marker, "
                        "both memories kept"
                    ),
                )
            )
            continue
        heuristic_drops[older.memory_id] = (newer.memory_id, found.subject)

    resolved: list[ScoredMemory] = []
    for item in survivors:
        replacement = heuristic_drops.get(item.memory_id)
        if replacement is None:
            resolved.append(item)
            continue
        newer_id, subject = replacement
        detail = f"superseded by {newer_id}"
        dropped.append(
            DroppedMemory(
                memory_id=item.memory_id,
                reason="superseded",
                text=preview_text(item.text),
                detail=f"{detail} (subject={subject or 'unknown'}, source=heuristic)",
                score=item.score,
            )
        )
        resolutions.append(
            ConflictResolution(
                subject=subject,
                kept_memory_id=newer_id,
                dropped_memory_id=item.memory_id,
                source="heuristic",
                reason=detail,
            )
        )
    return resolved, contested


def _is_replacement(item: ScoredMemory) -> bool:
    """Return whether a memory explicitly claims to replace earlier information.

    Either the text carries a correction marker or the application memory policy
    already classified the turn as a ``CORRECTION``. This gate is what keeps the
    heuristic from deleting a user fact on a weak signal.
    """

    return has_correction_marker(item.text) or item.record.memory_type == "CORRECTION"


def _find_by_text(scored: Sequence[ScoredMemory], text: str) -> ScoredMemory | None:
    """Locate a scored memory by content when its runtime id is not available."""

    target = normalize_text(text)
    if not target:
        return None
    for item in scored:
        if normalize_text(item.text) == target:
            return item
    return None


def _apply_relevance_floor(
    scored: Sequence[ScoredMemory],
    policy: RetrievalPolicy,
    dropped: list[DroppedMemory],
) -> list[ScoredMemory]:
    """Drop memories below the relevance floor (Precision@K protection).

    The floor is applied to query relevance only, after deduplication and
    conflict resolution, exactly where the plan puts it in the pipeline. Two
    tests are applied, and the report distinguishes them:

    ``low_relevance``
        below the absolute floor — no meaningful evidence for this query.
    ``weak_relevance``
        above the absolute floor but far below the best candidate — the tail of
        a distribution that clearly has a stronger answer.
    """

    best = max((item.relevance for item in scored), default=0.0)
    relative_floor = best * policy.relative_relevance_ratio
    survivors: list[ScoredMemory] = []
    for item in scored:
        if item.relevance < policy.relevance_floor:
            dropped.append(
                DroppedMemory(
                    memory_id=item.memory_id,
                    reason="low_relevance",
                    text=preview_text(item.text),
                    detail=(
                        f"relevance={item.relevance:.4f} below "
                        f"floor={policy.relevance_floor:.4f}"
                    ),
                    score=item.score,
                )
            )
            continue
        if item.relevance < relative_floor:
            dropped.append(
                DroppedMemory(
                    memory_id=item.memory_id,
                    reason="weak_relevance",
                    text=preview_text(item.text),
                    detail=(
                        f"relevance={item.relevance:.4f} below "
                        f"{policy.relative_relevance_ratio:.2f} of best={best:.4f}"
                    ),
                    score=item.score,
                )
            )
            continue
        survivors.append(item)
    return survivors


def _apply_cap(
    scored: Sequence[ScoredMemory],
    policy: RetrievalPolicy,
    dropped: list[DroppedMemory],
) -> tuple[list[ScoredMemory], int]:
    """Enforce ``max_memories`` and the optional suspicious-content drop."""

    survivors: list[ScoredMemory] = []
    suspicious_dropped = 0
    for item in scored:
        if policy.drop_suspicious_memories and item.suspicious:
            suspicious_dropped += 1
            dropped.append(
                DroppedMemory(
                    memory_id=item.memory_id,
                    reason="suspicious",
                    text=preview_text(item.text),
                    detail="instruction-override pattern detected in retrieved memory",
                    score=item.score,
                )
            )
            continue
        survivors.append(item)
    if policy.max_memories and len(survivors) > policy.max_memories:
        for item in survivors[policy.max_memories :]:
            dropped.append(
                DroppedMemory(
                    memory_id=item.memory_id,
                    reason="cap",
                    text=preview_text(item.text),
                    detail=f"max_memories={policy.max_memories}",
                    score=item.score,
                )
            )
        survivors = survivors[: policy.max_memories]
    return survivors, suspicious_dropped


__all__ = [
    "CORRECTION_MARKERS",
    "ClaimConflict",
    "GuardedText",
    "Conflict",
    "ConflictResolution",
    "DroppedMemory",
    "RetrievalPolicy",
    "RetrievalReport",
    "RetrievalSelection",
    "ScoredMemory",
    "SubjectClaim",
    "bigrams",
    "content_tokens",
    "detect_conflicts",
    "document_frequencies",
    "extract_claims",
    "inverse_document_frequency",
    "has_correction_marker",
    "inspect_memory_text",
    "jaccard",
    "lexical_score",
    "neutralize_memory_text",
    "normalize_text",
    "parse_timestamp",
    "preview_text",
    "query_dependent_component",
    "recency_score",
    "relevance_evidence",
    "runtime_component",
    "select_memories",
    "split_sentences",
    "stem",
    "strip_discourse_markers",
    "token_set",
]
