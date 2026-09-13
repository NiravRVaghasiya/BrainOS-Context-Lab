"""Evaluation metric primitives.

Functions here are dependency-free and deterministic so benchmark calculations
can be tested independently of any provider or model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Iterable


@dataclass(frozen=True)
class Summary:
    """Descriptive statistics for repeated benchmark measurements."""

    count: int
    mean: float
    standard_deviation: float
    confidence_interval_95: tuple[float, float]


def recall_at_k(retrieved_ids: Iterable[str], required_ids: Iterable[str]) -> float:
    required = set(required_ids)
    if not required:
        return 1.0
    return len(set(retrieved_ids) & required) / len(required)


def precision_at_k(retrieved_ids: Iterable[str], required_ids: Iterable[str]) -> float:
    retrieved = list(retrieved_ids)
    if not retrieved:
        return 0.0 if set(required_ids) else 1.0
    return len(set(retrieved) & set(required_ids)) / len(set(retrieved))


def context_reduction(full_context_tokens: int, managed_context_tokens: int) -> float:
    """Return the fraction of full-context tokens removed."""

    if full_context_tokens <= 0:
        return 0.0
    return max(0.0, 1.0 - managed_context_tokens / full_context_tokens)


def token_savings(full_context_tokens: int, managed_context_tokens: int) -> int:
    return max(0, full_context_tokens - managed_context_tokens)


def quality_adjusted_efficiency(answer_quality: float, context_tokens: int) -> float:
    """Quality per context token; callers define the quality scale."""

    if context_tokens <= 0:
        return 0.0
    return answer_quality / context_tokens


def summarize(values: Iterable[float]) -> Summary:
    observations = [float(value) for value in values]
    if not observations:
        return Summary(0, 0.0, 0.0, (0.0, 0.0))
    average = mean(observations)
    deviation = stdev(observations) if len(observations) > 1 else 0.0
    margin = 1.96 * deviation / sqrt(len(observations))
    return Summary(len(observations), average, deviation, (average - margin, average + margin))
