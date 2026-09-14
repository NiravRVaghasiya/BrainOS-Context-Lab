"""Evaluation metric primitives.

Functions here are dependency-free and deterministic so benchmark calculations
can be tested independently of any provider or model.

Phase 7 implemented retrieval Recall@K / Precision@K and the token-reduction
primitives. Phase 8 fills the rest of the plan's suite:

* quality — faithfulness to retrieved evidence, conflict-resolution accuracy
  (the abstention/accuracy rates live with the scorer, which owns the verdicts);
* efficiency — token savings and quality-adjusted efficiency, now called by the
  aggregate rather than sitting unused;
* robustness — accuracy as a function of conversation length, absolute and
  relative degradation from a reference length, and the area under those curves.

Descriptive statistics (:func:`summarize`) remain the trial-level building
block for Phase 10. A single run's task-level mean is not a trial CI; callers
must not present it as one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Any

#: Identifies the Phase 8 metric definitions. Stored on every aggregate so a
#: later phase can tell which formula produced a number.
METRICS_VERSION = "metrics-v1"

#: Categories whose correct behaviour is "use the newer value, not the older
#: one". Temporal replacement and explicit correction are scored the same way;
#: they differ in *how* the conversation stated the change, not in the metric.
CONFLICT_CATEGORIES: frozenset[str] = frozenset({"conflict", "temporal"})


@dataclass(frozen=True)
class Summary:
    """Descriptive statistics for repeated benchmark measurements."""

    count: int
    mean: float
    standard_deviation: float
    confidence_interval_95: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.confidence_interval_95
        return {
            "count": self.count,
            "mean": round(self.mean, 6),
            "standard_deviation": round(self.standard_deviation, 6),
            "confidence_interval_95": (round(lo, 6), round(hi, 6)),
        }


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
    """Quality per context token; callers define the quality scale.

    A system that saves 95% of tokens but loses 50% accuracy is not successful:
    this number falls when either quality drops or the prompt grows. ``0``
    tokens is treated as undefined and returns ``0.0`` rather than raising, so a
    missing accounting field cannot crash a run.
    """

    if context_tokens <= 0:
        return 0.0
    return answer_quality / context_tokens


def accuracy_degradation(reference_accuracy: float, accuracy_at_length: float) -> float:
    """``reference_accuracy - accuracy_at_length``.

    Positive means the longer conversation did worse than the reference.
    Negative means it did better — that is a result, not a clamp.
    """

    return float(reference_accuracy) - float(accuracy_at_length)


def relative_degradation(reference_accuracy: float, accuracy_at_length: float) -> float:
    """``(acc_ref - acc_L) / acc_ref``.

    When the reference accuracy is ``0`` the ratio is undefined; this returns
    ``0.0`` so a silent-zero reference cannot manufacture a huge relative drop.
    Accuracy cannot be worse than ``0``, so a zero reference can only stay
    flat or improve.
    """

    reference = float(reference_accuracy)
    if reference <= 0.0:
        return 0.0
    return (reference - float(accuracy_at_length)) / reference


def area_under_curve(xs: Iterable[float], ys: Iterable[float]) -> float:
    """Trapezoidal area under ``(x, y)``, sorted by ``x``.

    Duplicate or unsorted ``x`` values are accepted: points are sorted, and
    zero-width segments contribute nothing. A curve of fewer than two points
    has no area and returns ``0.0`` — the aggregate layer is responsible for
    recording that a single length cannot support a degradation claim.
    """

    points = sorted(
        ((float(x), float(y)) for x, y in zip(xs, ys)),
        key=lambda point: point[0],
    )
    if len(points) < 2:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        width = x1 - x0
        if width <= 0:
            continue
        area += width * (y0 + y1) / 2.0
    return area


def area_under_degradation_curve(
    lengths: Iterable[float],
    accuracies: Iterable[float],
    *,
    reference_accuracy: float | None = None,
) -> float:
    """Area under ``degradation(L)`` across the length ladder.

    The reference accuracy defaults to the accuracy at the shortest length,
    which is the plan's "accuracy at a reference length". Callers that want a
    fixed baseline (for example Mode A's accuracy at 5k) pass it explicitly.
    """

    length_list = [float(value) for value in lengths]
    accuracy_list = [float(value) for value in accuracies]
    if not length_list or not accuracy_list:
        return 0.0
    if reference_accuracy is None:
        shortest = min(range(len(length_list)), key=lambda index: length_list[index])
        reference_accuracy = accuracy_list[shortest]
    degradations = [
        accuracy_degradation(reference_accuracy, accuracy) for accuracy in accuracy_list
    ]
    return area_under_curve(length_list, degradations)


def mean_degradation(
    lengths: Iterable[float],
    accuracies: Iterable[float],
    *,
    reference_accuracy: float | None = None,
) -> float:
    """Length-normalized degradation AUC: mean drop across the ladder.

    Raw AUC has units of accuracy × tokens and grows with the ladder span, so
    a 5k–40k run is not comparable to a 5k–120k run on the raw number. Dividing
    by the length span gives a mean degradation that is.
    """

    length_list = [float(value) for value in lengths]
    if len(length_list) < 2:
        return 0.0
    span = max(length_list) - min(length_list)
    if span <= 0:
        return 0.0
    return (
        area_under_degradation_curve(
            length_list, accuracies, reference_accuracy=reference_accuracy
        )
        / span
    )


def summarize(values: Iterable[float]) -> Summary:
    observations = [float(value) for value in values]
    if not observations:
        return Summary(0, 0.0, 0.0, (0.0, 0.0))
    average = mean(observations)
    deviation = stdev(observations) if len(observations) > 1 else 0.0
    margin = 1.96 * deviation / sqrt(len(observations))
    return Summary(
        len(observations), average, deviation, (average - margin, average + margin)
    )


def rate(numerator: int, denominator: int) -> float:
    """``numerator / denominator``, or ``0.0`` when the denominator is empty.

    Every Phase 8 rate carries its own denominator in the aggregate so a
    partially graded run cannot flatter itself. This helper is the one division.
    """

    if denominator <= 0:
        return 0.0
    return numerator / denominator


__all__ = [
    "CONFLICT_CATEGORIES",
    "METRICS_VERSION",
    "Summary",
    "accuracy_degradation",
    "area_under_curve",
    "area_under_degradation_curve",
    "context_reduction",
    "mean_degradation",
    "precision_at_k",
    "quality_adjusted_efficiency",
    "rate",
    "recall_at_k",
    "relative_degradation",
    "summarize",
    "token_savings",
]
