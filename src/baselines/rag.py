"""Conventional retrieval baseline: lexical top-k over transcript chunks.

This is the Mode C / Mode E retriever the plan asks for — "history →
embedding/retriever → top-k → LLM" — implemented without BrainOS and without a
vector store. Both omissions are deliberate:

* **No BrainOS.** Mode C exists to show what the same conversation yields when
  an ordinary retriever selects the evidence. Importing the runtime here would
  make the baseline a re-skin of the system under test.
* **No embeddings.** The plan's "what not to build" list rules out complex
  vector databases, and a lexical retriever is reproducible with zero
  dependencies: the same conversation and question always produce the same
  top-k, so a benchmark run can be replayed exactly.

The retriever reuses the Phase 3 lexical machinery
(:mod:`brain.retrieval_policy`) — tokenization, stemming, IDF weighting, and
the coverage/similarity/phrase score — so Mode C and Mode D are compared on the
same notion of lexical relevance and differ in *what* is scored (transcript
chunks versus BrainOS memories) and *how* the winner list is assembled.

Known, intentional difference from BrainOS mode: this retriever has no
abstention behaviour. It returns ``top_k`` chunks whenever the conversation has
any content, even for a question nothing in it answers. That is what ordinary
RAG does, and it is one of the effects the benchmark is meant to expose rather
than hide.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from brain.retrieval_policy import (
    content_tokens,
    document_frequencies,
    inspect_memory_text,
    jaccard,
    lexical_score,
    normalize_text,
    split_sentences,
    token_set,
)

#: Characters a chunk may hold before the next one starts. Roughly a
#: paragraph: long enough to keep a fact with its surrounding sentence, short
#: enough that one turn's small talk cannot crowd out a fact from another turn.
DEFAULT_CHUNK_CHARS = 400
#: Jaccard threshold above which a chunk is a restatement of one already taken.
DEFAULT_DEDUPE_THRESHOLD = 0.88
#: Guard bound for chunk text; chunks are already bounded by
#: :data:`DEFAULT_CHUNK_CHARS`, so this only has to be generous.
_GUARD_CHARS = 1200
_ALLOWED_ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class HistoryChunk:
    """One retrievable slice of the transcript.

    ``text`` is the guarded text that will be rendered into the prompt — the
    retriever neutralizes it at selection time so the inspection panels and the
    prompt can never disagree about what the model saw. ``turn`` is the index of
    the source message in the transcript, which is what makes chunk ordering
    (and therefore tie-breaking) deterministic.
    """

    chunk_id: str
    text: str
    role: str = "user"
    turn: int = 0
    score: float = 0.0
    suspicious: bool = False
    #: Injection families detected in the raw text before guarding. Carried on
    #: the chunk so the prompt builder can report what was neutralized on this
    #: route without re-deriving it from text that is already clean.
    families: tuple[str, ...] = ()
    #: Neutralization categories applied to this chunk's text.
    neutralized: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready view for panels, export, and evaluation records."""

        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "role": self.role,
            "turn": self.turn,
            "score": round(self.score, 6),
            "suspicious": self.suspicious,
            "families": list(self.families),
            "neutralized": list(self.neutralized),
        }


def chunk_history(
    messages: Iterable[Mapping[str, object]] | Iterable[object],
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[HistoryChunk]:
    """Split a transcript into retrievable chunks, in transcript order.

    Each message is broken at sentence boundaries and sentences are packed up
    to ``max_chars``. Messages with no lexical content (greetings, empty turns)
    are skipped: they can never answer a question, and letting them compete
    would only push real evidence out of the top-k.

    Chunk ids are positional (``chunk-<index>``), so the same transcript always
    produces the same identifiers and an evaluation run can be diffed.
    """

    chunks: list[HistoryChunk] = []
    for turn, item in enumerate(messages):
        if not isinstance(item, Mapping):
            continue
        content = item.get("content", "")
        text = "" if content is None else str(content)
        if not normalize_text(text):
            continue
        role = str(item.get("role", "user")).strip().lower()
        if role not in _ALLOWED_ROLES:
            role = "user"
        for piece in _pack_sentences(text, max_chars):
            if not content_tokens(piece):
                continue
            chunks.append(
                HistoryChunk(
                    chunk_id=f"chunk-{len(chunks)}",
                    text=piece,
                    role=role,
                    turn=turn,
                )
            )
    return chunks


def retrieve_chunks(
    query: str,
    chunks: Sequence[HistoryChunk],
    *,
    top_k: int = 6,
    min_score: float = 0.0,
    dedupe_threshold: float = DEFAULT_DEDUPE_THRESHOLD,
) -> list[HistoryChunk]:
    """Return the top-k chunks for ``query``, most relevant first.

    Scoring is IDF-weighted lexical overlap over the candidate set, so a term
    shared by every chunk (a project name mentioned throughout) contributes
    almost nothing and a discriminative term dominates. Ties break on transcript
    position and then chunk id, which keeps the ranking reproducible.

    ``min_score`` defaults to ``0`` on purpose — see the module docstring. A
    caller that wants abstention can raise it; the benchmark should record
    whichever value ran.
    """

    candidates = [chunk for chunk in chunks if normalize_text(chunk.text)]
    if not candidates or top_k <= 0:
        return []

    query_tokens = content_tokens(query)
    frequencies = document_frequencies(chunk.text for chunk in candidates)
    scored: list[tuple[float, HistoryChunk]] = []
    for chunk in candidates:
        score = lexical_score(
            query_tokens,
            content_tokens(chunk.text),
            frequencies=frequencies,
            candidate_count=len(candidates),
        )
        if score < min_score:
            continue
        scored.append((score, chunk))

    # Highest score first; transcript order breaks ties, then the stable id.
    scored.sort(key=lambda pair: (-pair[0], pair[1].turn, pair[1].chunk_id))

    selected: list[HistoryChunk] = []
    taken_tokens: list[set[str]] = []
    for score, chunk in scored:
        if len(selected) >= top_k:
            break
        tokens = token_set(chunk.text)
        if dedupe_threshold > 0 and any(
            jaccard(tokens, other) >= dedupe_threshold for other in taken_tokens
        ):
            continue
        guarded = inspect_memory_text(chunk.text, max_chars=_GUARD_CHARS)
        if not normalize_text(guarded.text):
            continue
        selected.append(
            replace(
                chunk,
                text=guarded.text,
                score=score,
                suspicious=guarded.suspicious,
                families=guarded.families,
                neutralized=guarded.neutralized,
            )
        )
        taken_tokens.append(tokens)
    return selected


def _pack_sentences(text: str, max_chars: int) -> list[str]:
    """Pack sentences into chunks of at most ``max_chars`` characters.

    A single sentence longer than the bound is emitted whole rather than cut in
    half: splitting mid-sentence would separate a fact from the words that make
    it retrievable, and the builder's token budget is the real length control.
    """

    sentences = split_sentences(text)
    if not sentences:
        stripped = normalize_text(text)
        return [str(text).strip()] if stripped else []

    limit = max(1, int(max_chars))
    packed: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and length + addition > limit:
            packed.append(" ".join(current))
            current = [sentence]
            length = len(sentence)
            continue
        current.append(sentence)
        length += addition
    if current:
        packed.append(" ".join(current))
    return packed


__all__ = [
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_DEDUPE_THRESHOLD",
    "HistoryChunk",
    "chunk_history",
    "retrieve_chunks",
]
