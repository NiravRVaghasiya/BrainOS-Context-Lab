"""Benchmark scoring rules for the context-rot dataset (plan Phase 7).

Two halves, kept deliberately separate:

**Retrieval scoring is model-free.** Given the evidence a mode selected and the
prompt it produced, this module says which ledger facts are visible. It rides on
the marker convention from :mod:`evaluation.datasets` — a fact counts as
retrieved when *all* of its markers appear — because a replay cannot know the
runtime's memory ids in advance. The Phase 7 finding that motivated the split:
a benchmark whose retrieval side needs a model cannot be run at all without
credentials, and the whole point of the mode comparison is that most of it can
be measured before generation.

**Answer scoring needs a model.** It grades an answer string against the task's
acceptable answers, its stale answers, and the abstention expectation. Nothing
here calls a model: a caller supplies the answers (a real run from Phase 8, or a
deterministic mock from the Phase 18 tests). That keeps the scoring rules
testable in isolation and keeps a provider key out of the benchmark path.

Phase 8 extends each scored record with the rest of the plan's metric suite
(faithfulness, token savings, quality-adjusted efficiency, length-tier, optional
latency) and turns :func:`aggregate_scores` from a descriptive summary into the
run-level report: quality rates, efficiency means, per-length curves, and the
degradation/AUC measures. The verdict rules themselves do not change.

The verdict vocabulary is fixed here because later phases consume it:

```text
correct           the answer matches an accepted answer
abstained         the answer declines to answer, and an answer was expected
wrong_abstention  the answer declines although the evidence was in the prompt
stale_answer      the answer asserts a superseded or corrected value
incorrect         neither accepted nor stale (hallucination or retrieval failure)
ungraded          no answer was supplied
```

Every scored record also carries an ``error_type`` drawn from the plan's Phase 12
taxonomy when the verdict is not ``correct``, so error analysis starts from the
same record the aggregate metrics do.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .datasets import BenchmarkFact, BenchmarkTask
from .metrics import (
    CONFLICT_CATEGORIES,
    METRICS_VERSION,
    accuracy_degradation,
    area_under_curve,
    area_under_degradation_curve,
    mean_degradation,
    precision_at_k,
    quality_adjusted_efficiency,
    rate,
    recall_at_k,
    relative_degradation,
    summarize,
    token_savings,
)

#: Verdicts, most specific first. ``stale_answer`` outranks ``incorrect`` because
#: a run that confidently repeats a superseded value is a different failure from
#: one that says nothing useful.
ANSWER_VERDICTS: tuple[str, ...] = (
    "correct",
    "abstained",
    "wrong_abstention",
    "stale_answer",
    "incorrect",
    "ungraded",
)

#: Phrases that count as declining to answer. Checked *after* the acceptable
#: answers, so "the database is MySQL 8, but I have no data on the region" still
#: grades as correct.
ABSTENTION_PATTERNS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "i dont know",
    "i don't have",
    "i do not have",
    "i have no",
    "i don't recall",
    "i do not recall",
    "i can't find",
    "i cannot find",
    "can't find",
    "cannot find",
    "can't determine",
    "cannot determine",
    "unable to find",
    "unable to determine",
    "no information",
    "no data",
    "no record",
    "no evidence",
    "no memory",
    "not mentioned",
    "wasn't mentioned",
    "was not mentioned",
    "not specified",
    "isn't specified",
    "is not specified",
    "not in the",
    "nothing in",
    "doesn't contain",
    "does not contain",
    "not available",
    "unknown",
    "unsure",
    "not sure",
)

_PUNCT_RE = re.compile(r"[^\w\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: Any) -> str:
    """Casefold, strip punctuation, and collapse separators for comparison.

    Hyphens and underscores become spaces so ``eu-west-1``, ``EU West 1``, and
    ``eu_west_1`` compare equal; every other punctuation mark is dropped. The
    transformation is applied to both sides of every comparison, so it never has
    to be exactly right — only consistent.
    """

    lowered = str(text or "").casefold().replace("-", " ").replace("_", " ")
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub(" ", lowered)).strip()


def contains_phrase(haystack: str, needle: str) -> bool:
    """Whether ``needle`` appears in ``haystack`` as whole words.

    Both arguments are normalized first. Empty needles never match, so a fact
    with no declared value cannot make every answer correct.
    """

    normalized_needle = normalize_answer(needle)
    if not normalized_needle:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_needle)}(?!\w)"
    return re.search(pattern, normalize_answer(haystack)) is not None


#: Patterns in the same normalized space the answers are compared in. The
#: normalization strips apostrophes, so ``i don't know`` becomes ``i don t
#: know``; normalizing the patterns the same way is what keeps the two sides
#: comparable.
_NORMALIZED_ABSTENTION_PATTERNS: tuple[str, ...] = tuple(
    normalize_answer(pattern) for pattern in ABSTENTION_PATTERNS
)


def abstains(answer: Any) -> bool:
    """Whether an answer declines to answer rather than asserting a value."""

    normalized = normalize_answer(answer)
    if not normalized:
        return False
    return any(pattern in normalized for pattern in _NORMALIZED_ABSTENTION_PATTERNS)


def fact_markers_present(text: str, fact: BenchmarkFact) -> bool:
    """Whether every marker of ``fact`` appears in ``text``.

    All markers must match: a marker set pairs the subject with the value, so a
    conversation that mentions the project but never its database must not count
    as evidence for the fact.
    """

    return bool(fact.markers) and all(contains_phrase(text, marker) for marker in fact.markers)


def detect_facts(text: str, facts: Iterable[BenchmarkFact]) -> set[str]:
    """Return the ids of every ledger fact whose evidence is present in ``text``."""

    return {fact.fact_id for fact in facts if fact_markers_present(text, fact)}


def detect_facts_in_texts(texts: Iterable[str], facts: Iterable[BenchmarkFact]) -> set[str]:
    """Detect facts across independent evidence items, one item at a time.

    Each item is checked separately so a fact's markers must co-occur in the
    *same* memory or chunk. Joining the items first would count "Project Cobalt"
    in one memory and "eu-west-1" in another as evidence for a fact that says
    both, which is exactly the false positive a strict recall metric must not
    have.
    """

    ledger = list(facts)
    found: set[str] = set()
    for text in texts:
        found |= detect_facts(str(text), ledger)
    return found


def latest_session_index(task: BenchmarkTask) -> int:
    """Return the highest session index the transcript marks."""

    sessions = [
        int(message.get("session", 0) or 0)
        for message in task.conversation
        if isinstance(message, Mapping)
    ]
    return max(sessions) if sessions else 0


def unavailable_required_facts(
    task: BenchmarkTask, *, session_isolation: bool
) -> tuple[str, ...]:
    """Required facts a *session-isolated* replay provably cannot reach.

    Session isolation is a product invariant (BrainOS memory must never cross a
    session boundary), so a cross-session task replayed in isolated sessions is
    answerable only in its final session. The scorer needs that fact to know
    whether declining to answer is the correct behaviour there.
    """

    if not session_isolation:
        return ()
    final_session = latest_session_index(task)
    ledger = task.facts_by_id()
    return tuple(
        fact_id
        for fact_id in task.evidence.required
        if fact_id in ledger and ledger[fact_id].session_index < final_session
    )


def expects_abstention(task: BenchmarkTask, *, session_isolation: bool = False) -> bool:
    """Whether the only correct answer to ``task`` is "I don't know"."""

    if task.evidence.abstention_expected:
        return True
    required = tuple(task.evidence.required)
    return bool(required) and len(unavailable_required_facts(
        task, session_isolation=session_isolation
    )) == len(required)


@dataclass(frozen=True)
class RetrievalScore:
    """What a mode's retrieval put in front of the model for one task."""

    task_id: str
    category: str
    mode: str = ""
    required_fact_ids: tuple[str, ...] = ()
    retrieved_fact_ids: tuple[str, ...] = ()
    missing_fact_ids: tuple[str, ...] = ()
    forbidden_retrieved: tuple[str, ...] = ()
    recall: float = 0.0
    precision: float = 0.0
    evidence_in_prompt: bool = False
    expected_answer_in_prompt: bool = False
    prompt_contains_forbidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "mode": self.mode,
            "required_fact_ids": list(self.required_fact_ids),
            "retrieved_fact_ids": list(self.retrieved_fact_ids),
            "missing_fact_ids": list(self.missing_fact_ids),
            "forbidden_retrieved": list(self.forbidden_retrieved),
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "evidence_in_prompt": self.evidence_in_prompt,
            "expected_answer_in_prompt": self.expected_answer_in_prompt,
            "prompt_contains_forbidden": self.prompt_contains_forbidden,
        }


@dataclass(frozen=True)
class AnswerScore:
    """How an answer compares to the task's contract."""

    task_id: str
    category: str
    mode: str = ""
    verdict: str = "ungraded"
    expected_answer: str = ""
    answer: str = ""
    abstention_expected: bool = False
    matched_answer: str = ""
    stale_answer: str = ""
    error_type: str = ""

    @property
    def correct(self) -> bool:
        return self.verdict == "correct"

    @property
    def graded(self) -> bool:
        return self.verdict != "ungraded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "mode": self.mode,
            "verdict": self.verdict,
            "expected_answer": self.expected_answer,
            "answer": self.answer,
            "abstention_expected": self.abstention_expected,
            "matched_answer": self.matched_answer,
            "stale_answer": self.stale_answer,
            "error_type": self.error_type,
        }


def score_retrieval(
    task: BenchmarkTask,
    *,
    retrieved_texts: Iterable[str] = (),
    prompt_messages: Iterable[Mapping[str, Any]] = (),
    mode: str = "",
) -> RetrievalScore:
    """Score retrieval and prompt availability against the task's ledger.

    ``recall`` and ``precision`` describe the *retriever*: facts whose evidence
    the mode actually selected. ``evidence_in_prompt`` describes the *prompt*:
    every required fact's markers are present there. The two differ by design —
    Mode A retrieves nothing and still puts every fact in the prompt, and a
    retriever can select the right evidence and have the budget evict it.
    """

    planted = task.planted_facts()
    retrieved = detect_facts_in_texts(retrieved_texts, planted)
    required = tuple(task.evidence.required)
    forbidden = tuple(task.evidence.forbidden)
    prompt_items = [str(message.get("content", "")) for message in prompt_messages]
    in_prompt = detect_facts_in_texts(prompt_items, planted)

    prompt_required = tuple(fact_id for fact_id in required if fact_id in in_prompt)
    prompt_forbidden = tuple(fact_id for fact_id in forbidden if fact_id in in_prompt)
    accepted = task.answers()

    return RetrievalScore(
        task_id=task.task_id,
        category=task.category,
        mode=mode,
        required_fact_ids=required,
        retrieved_fact_ids=tuple(sorted(retrieved)),
        missing_fact_ids=tuple(fact_id for fact_id in required if fact_id not in retrieved),
        forbidden_retrieved=tuple(sorted(set(forbidden) & retrieved)),
        recall=recall_at_k(retrieved, required),
        precision=precision_at_k(retrieved, (*required, *task.evidence.supporting)),
        evidence_in_prompt=bool(required) and len(prompt_required) == len(required),
        expected_answer_in_prompt=bool(accepted)
        and any(
            contains_phrase(item, value) for item in prompt_items for value in accepted
        ),
        prompt_contains_forbidden=bool(prompt_forbidden),
    )


def score_answer(
    task: BenchmarkTask,
    answer: Any,
    *,
    mode: str = "",
    session_isolation: bool = False,
) -> AnswerScore:
    """Grade one answer string against the task's evidence contract.

    When an answer is expected, accepted answers are checked first, then the
    stale (superseded/corrected) values, then abstention. An answer that mentions
    a stale value *and* the current one is correct: it resolved the conflict and
    said so.

    When **abstention is expected** — the abstention category, or a cross-session
    task replayed with isolation — the order is inverted: declining is the only
    correct behaviour, and asserting a value is a hallucination. That is the
    stricter rule on purpose; under isolation a "correct-looking" answer can only
    come from a guess or from memory leaking across sessions, and the benchmark
    must not score either as success.
    """

    text = str(answer or "").strip()
    abstention_expected = expects_abstention(task, session_isolation=session_isolation)
    accepted = task.answers()
    stale_values = [
        fact.value
        for fact_id in task.evidence.forbidden
        for fact in (task.facts_by_id().get(fact_id),)
        if fact is not None and fact.value
    ]

    matched = next((value for value in accepted if contains_phrase(text, value)), "")
    declined = abstains(text)
    stale = next((value for value in stale_values if contains_phrase(text, value)), "")

    if not text:
        verdict = "ungraded"
    elif abstention_expected:
        verdict = "correct" if declined else "incorrect"
    elif matched:
        verdict = "correct"
    elif declined:
        verdict = "abstained"
    elif stale:
        verdict = "stale_answer"
    else:
        verdict = "incorrect"

    return AnswerScore(
        task_id=task.task_id,
        category=task.category,
        mode=mode,
        verdict=verdict,
        expected_answer=task.expected_answer,
        answer=text,
        abstention_expected=abstention_expected,
        matched_answer=matched,
        stale_answer=stale,
        error_type=_error_type(
            verdict,
            category=task.category,
            abstention_expected=abstention_expected,
        ),
    )


def score_faithfulness(
    *,
    verdict: str,
    evidence_in_prompt: bool,
    expected_answer_in_prompt: bool,
    prompt_contains_forbidden: bool,
    abstention_expected: bool,
) -> float | None:
    """Whether the answer is grounded in the evidence the model actually saw.

    Faithfulness is *not* accuracy. A stale answer that repeats a superseded
    value present in the prompt is faithful to retrieved memory and still a
    conflict-resolution failure; a correct answer whose evidence never reached
    the prompt is accurate and unfaithful (a guess). Ungraded records return
    ``None`` and are excluded from the rate.

    The prompt, not the retriever, is the grounding surface: Mode A retrieves
    nothing and can still be faithful because the full transcript is in the
    prompt.
    """

    if verdict == "ungraded":
        return None
    if abstention_expected:
        return 1.0 if verdict == "correct" else 0.0
    if verdict in {"abstained", "wrong_abstention"}:
        return 0.0 if evidence_in_prompt else 1.0
    if verdict == "correct":
        # All required evidence must have reached the prompt. A correct answer
        # whose evidence was missing is a guess — the multi-hop case where
        # Recall@K is 1.0 but evidence-in-prompt is 0.0. The unused
        # ``expected_answer_in_prompt`` stays on the signature so callers can
        # pass the whole retrieval record through; it does not relax this.
        return 1.0 if evidence_in_prompt else 0.0
    if verdict == "stale_answer":
        return 1.0 if prompt_contains_forbidden else 0.0
    return 0.0


def is_conflict_task(category: str) -> bool:
    """Whether conflict-resolution accuracy is defined for this category."""

    return category in CONFLICT_CATEGORIES


def _error_type(verdict: str, *, category: str, abstention_expected: bool = False) -> str:
    """Map a verdict onto the plan's Phase 12 error taxonomy.

    Retrieval-dependent distinctions (was the evidence in the prompt?) are
    applied in :func:`score_record`, which sees both halves.
    """

    if verdict == "incorrect" and abstention_expected:
        return "hallucination"
    return {
        "correct": "",
        "stale_answer": "conflicting_memory" if category == "conflict" else "stale_memory",
        "incorrect": "wrong_answer",
        "wrong_abstention": "wrong_abstention",
        "abstained": "missed_memory",
        "ungraded": "ungraded",
    }.get(verdict, "")


def score_record(
    task: BenchmarkTask,
    record: Mapping[str, Any],
    *,
    answer: Any | None = None,
    session_isolation: bool = False,
) -> dict[str, Any]:
    """Score one mode replay, combining retrieval and answer evidence.

    ``record`` is a :class:`~evaluation.modes.ModeReplay` view (or its
    ``to_dict``). The returned mapping is JSON-ready and is what the runner
    attaches to each task result and what the aggregate is computed over.

    Two refinements need both halves of the record, which is why they happen
    here rather than in :func:`score_answer`:

    * declining although the evidence was in the prompt is ``wrong_abstention``
      (the information was there and the model did not use it);
    * an incorrect answer whose evidence was missing from the prompt is
      ``missed_memory`` rather than ``hallucination``: the model was never given
      the fact, so the failure belongs to context construction. That split is the
      reason the retrieval half is scored separately at all.
    """

    retrieved_texts = [
        *list(record.get("retrieved_memory_texts") or ()),
        *list(record.get("retrieved_chunk_texts") or ()),
    ]
    retrieval = score_retrieval(
        task,
        retrieved_texts=retrieved_texts,
        prompt_messages=record.get("prompt_messages") or (),
        mode=str(record.get("mode", "")),
    )
    answer_score = score_answer(
        task,
        answer,
        mode=str(record.get("mode", "")),
        session_isolation=session_isolation,
    )
    verdict = answer_score.verdict
    error_type = answer_score.error_type
    if verdict == "incorrect":
        # Asserting a value when abstention was expected is already a
        # hallucination; otherwise the prompt decides whether the model was
        # given the fact at all.
        if error_type != "hallucination":
            error_type = "hallucination" if retrieval.evidence_in_prompt else "missed_memory"
    elif verdict == "abstained" and retrieval.evidence_in_prompt:
        error_type = "wrong_abstention"
        answer_score = replace(
            answer_score, verdict="wrong_abstention", error_type=error_type
        )
    # An ungraded record stays ungraded: "nobody graded this task" is not an
    # observed failure, and the retrieval half already reports whether the
    # evidence was missing.

    conversation_length, length_tier = _length_fields(task, record)
    final_tokens = int(record.get("final_context_tokens", 0) or 0)
    full_tokens = int(record.get("full_context_reference_tokens", 0) or 0)
    savings = token_savings(full_tokens, final_tokens)
    quality: float | None
    if answer_score.verdict == "ungraded":
        quality = None
    else:
        quality = 1.0 if answer_score.verdict == "correct" else 0.0
    qae = (
        None
        if quality is None
        else quality_adjusted_efficiency(quality, final_tokens)
    )
    stats = record.get("stats") if isinstance(record.get("stats"), Mapping) else {}
    latency = _optional_float(record.get("latency_ms", record.get("generation_latency_ms")))
    return {
        "task_id": task.task_id,
        "category": task.category,
        "mode": answer_score.mode,
        "retrieval": retrieval.to_dict(),
        "answer": answer_score.to_dict(),
        "error_type": error_type,
        "context_reduction": float(record.get("context_reduction", 0.0) or 0.0),
        "final_context_tokens": final_tokens,
        "full_context_reference_tokens": full_tokens,
        "token_savings": savings,
        "quality_adjusted_efficiency": qae,
        "faithfulness": score_faithfulness(
            verdict=answer_score.verdict,
            evidence_in_prompt=retrieval.evidence_in_prompt,
            expected_answer_in_prompt=retrieval.expected_answer_in_prompt,
            prompt_contains_forbidden=retrieval.prompt_contains_forbidden,
            abstention_expected=answer_score.abstention_expected,
        ),
        "conflict_task": is_conflict_task(task.category),
        "conversation_length": conversation_length,
        "length_tier": length_tier,
        "latency_ms": latency,
        "token_counter": str(stats.get("token_counter") or record.get("token_counter") or ""),
    }


def aggregate_scores(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate scored records into the headline numbers for one run.

    Phase 7 reported counts, rates, and means, each with its own denominator.
    Phase 8 keeps those and adds the rest of the plan's suite: faithfulness,
    conflict-resolution accuracy, token savings, quality-adjusted efficiency,
    optional latency, per-length curves, and the degradation/AUC measures.

    Paired trials, effect sizes, and trial-level confidence intervals still
    belong to Phase 10 — a single run's task-level mean is not a trial CI.
    Each rate still carries its own denominator so a partially graded run
    cannot flatter itself.
    """

    records = [dict(score) for score in scores]
    graded = [record for record in records if record["answer"]["verdict"] != "ungraded"]
    abstention_tasks = [
        record for record in records if record["answer"]["abstention_expected"]
    ]
    conflict_tasks = [
        record
        for record in graded
        if record.get("conflict_task") or is_conflict_task(str(record.get("category", "")))
    ]
    faithful_defined = [
        record for record in records if record.get("faithfulness") is not None
    ]
    # Forbidden-in-prompt is only meaningful on tasks that declare a stale
    # value. Using the conflict/temporal categories keeps the denominator honest.
    forbidden_scope = [
        record
        for record in records
        if is_conflict_task(str(record.get("category", "")))
        or record.get("conflict_task")
    ]
    latencies = [
        float(record["latency_ms"])
        for record in records
        if record.get("latency_ms") is not None
    ]
    qae_values = [
        float(record["quality_adjusted_efficiency"])
        for record in graded
        if record.get("quality_adjusted_efficiency") is not None
    ]
    answerable = [
        record for record in records if record["retrieval"]["required_fact_ids"]
    ]
    mean_tokens = _mean(record.get("final_context_tokens", 0) for record in records)
    accuracy = rate(
        sum(1 for record in graded if record["answer"]["verdict"] == "correct"),
        len(graded),
    )
    by_length = _by_length(records)
    token_counters = sorted(
        {
            str(record.get("token_counter") or "")
            for record in records
            if record.get("token_counter")
        }
    )

    summary = {
        "metrics_version": METRICS_VERSION,
        "task_count": len(records),
        "graded_answer_count": len(graded),
        # Recall is undefined for tasks with no required evidence (the
        # abstention category): averaging in their vacuous 1.0 would flatter
        # every mode, so they are excluded from the denominator instead.
        "retrieval_recall": _mean(
            record["retrieval"]["recall"] for record in answerable
        ),
        "retrieval_precision": _mean(
            record["retrieval"]["precision"] for record in records
        ),
        "evidence_in_prompt_rate": rate(
            sum(1 for record in records if record["retrieval"]["evidence_in_prompt"]),
            len(answerable),
        ),
        "answer_accuracy": accuracy,
        "faithfulness": rate(
            sum(1 for record in faithful_defined if record.get("faithfulness") == 1.0),
            len(faithful_defined),
        ),
        "faithfulness_count": len(faithful_defined),
        "conflict_resolution_accuracy": rate(
            sum(1 for record in conflict_tasks if record["answer"]["verdict"] == "correct"),
            len(conflict_tasks),
        ),
        "conflict_task_count": len(conflict_tasks),
        "abstention_accuracy": rate(
            sum(
                1
                for record in abstention_tasks
                if record["answer"]["verdict"] == "correct"
            ),
            len(abstention_tasks),
        ),
        "stale_answer_rate": rate(
            sum(1 for record in graded if record["answer"]["verdict"] == "stale_answer"),
            len(graded),
        ),
        "forbidden_in_prompt_rate": rate(
            sum(
                1
                for record in forbidden_scope
                if record.get("retrieval", {}).get("prompt_contains_forbidden")
            ),
            len(forbidden_scope),
        ),
        "mean_context_reduction": _mean(
            record.get("context_reduction", 0.0) for record in records
        ),
        "mean_final_context_tokens": mean_tokens,
        "mean_full_context_reference_tokens": _mean(
            record.get("full_context_reference_tokens", 0) for record in records
        ),
        "mean_token_savings": _mean(record.get("token_savings", 0) for record in records),
        "mean_quality_adjusted_efficiency": _mean(qae_values),
        "quality_per_token": (
            accuracy / mean_tokens if mean_tokens else 0.0
        ),
        "mean_latency_ms": _mean(latencies) if latencies else None,
        "latency_count": len(latencies),
        "token_counters": token_counters,
        "error_counts": _counts(record.get("error_type", "") or "none" for record in records),
        "verdict_counts": _counts(record["answer"]["verdict"] for record in records),
        "by_category": _by_category(records),
        "by_length": by_length,
        "degradation": _degradation_report(by_length),
    }
    return summary


def _by_category(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("category", "")), []).append(record)
    return {
        category: _group_metrics(items, extra={"conflict_task": is_conflict_task(category)})
        for category, items in sorted(grouped.items())
    }


def _by_length(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        length = int(record.get("length_tier") or record.get("conversation_length") or 0)
        grouped.setdefault(length, []).append(record)
    return {
        str(length): {**_group_metrics(items), "length": length}
        for length, items in sorted(grouped.items())
    }


def _group_metrics(
    items: Sequence[Mapping[str, Any]], extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    graded = [item for item in items if item["answer"]["verdict"] != "ungraded"]
    answerable = [item for item in items if item["retrieval"]["required_fact_ids"]]
    faithful_defined = [item for item in items if item.get("faithfulness") is not None]
    payload: dict[str, Any] = {
        "task_count": len(items),
        "graded_answer_count": len(graded),
        "retrieval_recall": _mean(item["retrieval"]["recall"] for item in answerable),
        "retrieval_precision": _mean(item["retrieval"]["precision"] for item in items),
        "evidence_in_prompt_rate": rate(
            sum(1 for item in items if item["retrieval"]["evidence_in_prompt"]),
            len(answerable),
        ),
        "answer_accuracy": rate(
            sum(1 for item in graded if item["answer"]["verdict"] == "correct"),
            len(graded),
        ),
        "faithfulness": rate(
            sum(1 for item in faithful_defined if item.get("faithfulness") == 1.0),
            len(faithful_defined),
        ),
        "mean_final_context_tokens": _mean(
            item.get("final_context_tokens", 0) for item in items
        ),
        "mean_context_reduction": _mean(
            item.get("context_reduction", 0.0) for item in items
        ),
        "mean_token_savings": _mean(item.get("token_savings", 0) for item in items),
        "mean_quality_adjusted_efficiency": _mean(
            float(item["quality_adjusted_efficiency"])
            for item in graded
            if item.get("quality_adjusted_efficiency") is not None
        ),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _degradation_report(by_length: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Accuracy-vs-length degradation from the shortest length as reference.

    A single length (the committed smoke tier) cannot support a curve: the AUC
    fields are ``None`` and a note says so, rather than reporting a silent 0
    that looks like "no degradation".
    """

    points = sorted(by_length.values(), key=lambda item: int(item.get("length", 0)))
    if not points:
        return {
            "reference_length": 0,
            "reference_accuracy": 0.0,
            "point_count": 0,
            "by_length": [],
            "area_under_degradation_curve": None,
            "area_under_accuracy_curve": None,
            "mean_degradation": None,
            "note": "No length points.",
        }
    reference = points[0]
    reference_accuracy = float(reference.get("answer_accuracy", 0.0) or 0.0)
    rows = []
    for point in points:
        accuracy = float(point.get("answer_accuracy", 0.0) or 0.0)
        rows.append(
            {
                "length": int(point.get("length", 0) or 0),
                "accuracy": round(accuracy, 6),
                "absolute_degradation": round(
                    accuracy_degradation(reference_accuracy, accuracy), 6
                ),
                "relative_degradation": round(
                    relative_degradation(reference_accuracy, accuracy), 6
                ),
                "mean_final_context_tokens": point.get("mean_final_context_tokens", 0.0),
            }
        )
    if len(points) < 2:
        return {
            "reference_length": int(reference.get("length", 0) or 0),
            "reference_accuracy": round(reference_accuracy, 6),
            "point_count": 1,
            "by_length": rows,
            "area_under_degradation_curve": None,
            "area_under_accuracy_curve": None,
            "mean_degradation": None,
            "note": "A single length cannot support a degradation curve.",
        }
    lengths = [row["length"] for row in rows]
    accuracies = [row["accuracy"] for row in rows]
    return {
        "reference_length": int(reference.get("length", 0) or 0),
        "reference_accuracy": round(reference_accuracy, 6),
        "point_count": len(points),
        "by_length": rows,
        "area_under_degradation_curve": round(
            area_under_degradation_curve(
                lengths, accuracies, reference_accuracy=reference_accuracy
            ),
            6,
        ),
        "area_under_accuracy_curve": round(area_under_curve(lengths, accuracies), 6),
        "mean_degradation": round(
            mean_degradation(lengths, accuracies, reference_accuracy=reference_accuracy),
            6,
        ),
        "note": "",
    }


def _length_fields(task: BenchmarkTask, record: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(conversation_length, length_tier)`` for a scored record.

    The length ladder is the plan's token-tier (5k, 10k, …), not the message
    count. Message count is still stored because some fixtures have no tier.
    """

    conversation_length = record.get("conversation_length")
    if conversation_length is None:
        conversation_length = task.conversation_length
    if conversation_length is None:
        conversation_length = len(task.conversation)
    metadata = task.metadata or {}
    length_tier = record.get("length_tier")
    if length_tier is None:
        length_tier = metadata.get("length_tier") or metadata.get("target_tokens")
    if not length_tier:
        length_tier = conversation_length
    return int(conversation_length or 0), int(length_tier or 0)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float:
    summary = summarize(list(values))
    return round(summary.mean, 6)


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


@dataclass
class AnswerSet:
    """Answers keyed by task id, optionally scoped to one mode.

    The benchmark is scored from a file rather than from a model call, so a run
    can be graded later, re-graded with different rules, or graded from a
    deterministic mock in a test.
    """

    answers: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> AnswerSet:
        answers: dict[tuple[str, str], str] = {}
        for record in records:
            task_id = str(record.get("task_id", ""))
            if not task_id:
                raise ValueError("Every answer record needs a task_id.")
            if "answer" not in record:
                raise ValueError(f"Answer record for {task_id} has no 'answer' field.")
            mode = str(record.get("mode", ""))
            answers[(task_id, mode)] = str(record.get("answer") or "")
        return cls(answers)

    def get(self, task_id: str, mode: str) -> str | None:
        """Return the answer for a task, preferring a mode-specific entry."""

        if (task_id, mode) in self.answers:
            return self.answers[(task_id, mode)]
        return self.answers.get((task_id, ""))

    def __len__(self) -> int:
        return len(self.answers)


__all__ = [
    "ABSTENTION_PATTERNS",
    "ANSWER_VERDICTS",
    "AnswerScore",
    "AnswerSet",
    "RetrievalScore",
    "abstains",
    "aggregate_scores",
    "contains_phrase",
    "detect_facts",
    "detect_facts_in_texts",
    "expects_abstention",
    "fact_markers_present",
    "is_conflict_task",
    "latest_session_index",
    "normalize_answer",
    "score_answer",
    "score_faithfulness",
    "score_record",
    "score_retrieval",
    "unavailable_required_facts",
]
