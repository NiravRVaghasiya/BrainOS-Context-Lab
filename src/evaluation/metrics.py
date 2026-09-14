"""Evaluation metric primitives.

Functions here are dependency-free and deterministic so benchmark calculations
can be tested independently of any provider or model.

Phase 7 implemented retrieval Recall@K / Precision@K and the token-reduction
primitives. Phase 8 filled the quality, efficiency, and robustness suites.
Phase 10 adds the statistical evaluation layer:

* trial-level mean, standard deviation, and 95% confidence intervals;
* exact Student's t critical values and p-values for small sample sizes;
* paired comparisons across benchmark tasks with mean differences and standard error;
* effect sizes: Cohen's d (paired and independent) and bias-corrected Hedges' g;
* sign test and win/loss/tie counting for paired evaluations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp, lgamma, log, sqrt
from statistics import mean, stdev
from typing import Any

#: Identifies the metric definitions version. Stored on every aggregate so a
#: later phase can tell which formula produced a number.
METRICS_VERSION = "metrics-v1"

#: Categories whose correct behaviour is "use the newer value, not the older
#: one". Temporal replacement and explicit correction are scored the same way;
#: they differ in *how* the conversation stated the change, not in the metric.
CONFLICT_CATEGORIES: frozenset[str] = frozenset({"conflict", "temporal"})

#: Two-tailed Student's t critical values for 95% confidence (alpha = 0.05).
STUDENT_T_CRITICAL_95: dict[int, float] = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    14: 2.144787,
    15: 2.131450,
    16: 2.119905,
    17: 2.109816,
    18: 2.100922,
    19: 2.093024,
    20: 2.085963,
    21: 2.079614,
    22: 2.073873,
    23: 2.068658,
    24: 2.063899,
    25: 2.059539,
    26: 2.055529,
    27: 2.051831,
    28: 2.048407,
    29: 2.045230,
    30: 2.042272,
    40: 2.021075,
    50: 2.008559,
    60: 2.000298,
    80: 1.989972,
    100: 1.983972,
    120: 1.979930,
}


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


@dataclass(frozen=True)
class PairedDifference:
    """Statistical summary of paired differences across benchmark tasks."""

    metric: str
    sample_size: int
    mean_a: float
    mean_b: float
    mean_difference: float
    standard_deviation_difference: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    t_statistic: float
    p_value: float
    cohens_d: float
    hedges_g: float
    effect_size_magnitude: str
    wins: int
    losses: int
    ties: int
    win_rate: float
    sign_test_p_value: float

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.confidence_interval_95
        return {
            "metric": self.metric,
            "sample_size": self.sample_size,
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "mean_difference": round(self.mean_difference, 6),
            "standard_deviation_difference": round(self.standard_deviation_difference, 6),
            "standard_error": round(self.standard_error, 6),
            "confidence_interval_95": (round(lo, 6), round(hi, 6)),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "cohens_d": round(self.cohens_d, 4),
            "hedges_g": round(self.hedges_g, 4),
            "effect_size_magnitude": self.effect_size_magnitude,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": round(self.win_rate, 4),
            "sign_test_p_value": round(self.sign_test_p_value, 6),
        }


def t_critical(df: int, confidence: float = 0.95) -> float:
    """Return the two-tailed critical t value for degrees of freedom ``df``."""

    if df <= 0:
        return 1.96
    if confidence == 0.95 and df in STUDENT_T_CRITICAL_95:
        return STUDENT_T_CRITICAL_95[df]
    z = 1.959963984540054
    # Cornish-Fisher expansion for t quantile
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * (df**2))
    )


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued fraction evaluation for incomplete beta function."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _incbeta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_p_value(t: float, df: int) -> float:
    """Return the two-tailed p-value for Student's t statistic and degrees of freedom."""

    if df <= 0:
        return 1.0
    t_val = float(t)
    if t_val == 0.0:
        return 1.0
    x = df / (df + t_val * t_val)
    return max(0.0, min(1.0, _incbeta(0.5 * df, 0.5, x)))


def sign_test_p_value(wins: int, losses: int) -> float:
    """Return the two-sided binomial p-value under H0: p = 0.5."""

    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return max(0.0, min(1.0, 2.0 * prob))


def effect_size_magnitude(d: float) -> str:
    """Classify Cohen's d effect size according to standard thresholds."""

    abs_d = abs(float(d))
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def cohens_d(
    sample_a: Iterable[float],
    sample_b: Iterable[float],
    *,
    paired: bool = True,
) -> float:
    """Compute Cohen's d effect size between two samples."""

    a_list = [float(x) for x in sample_a]
    b_list = [float(x) for x in sample_b]
    if not a_list or not b_list:
        return 0.0
    if paired:
        if len(a_list) != len(b_list):
            raise ValueError("Paired samples must have equal length.")
        diffs = [a - b for a, b in zip(a_list, b_list)]
        if len(diffs) <= 1:
            return 0.0
        sd = stdev(diffs)
        if sd <= 0.0:
            return 0.0
        return mean(diffs) / sd
    else:
        if len(a_list) <= 1 and len(b_list) <= 1:
            return 0.0
        mean_a = mean(a_list)
        mean_b = mean(b_list)
        var_a = stdev(a_list) ** 2 if len(a_list) > 1 else 0.0
        var_b = stdev(b_list) ** 2 if len(b_list) > 1 else 0.0
        n_a, n_b = len(a_list), len(b_list)
        df = n_a + n_b - 2
        if df <= 0:
            return 0.0
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
        if pooled_var <= 0.0:
            return 0.0
        return (mean_a - mean_b) / sqrt(pooled_var)


def hedges_g(
    sample_a: Iterable[float],
    sample_b: Iterable[float],
    *,
    paired: bool = True,
) -> float:
    """Compute Hedges' g bias-corrected effect size."""

    a_list = [float(x) for x in sample_a]
    b_list = [float(x) for x in sample_b]
    d = cohens_d(a_list, b_list, paired=paired)
    if d == 0.0:
        return 0.0
    df = (len(a_list) - 1) if paired else (len(a_list) + len(b_list) - 2)
    if df <= 1:
        return d
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * correction


def paired_difference_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    metric: str = "",
    use_t: bool = True,
) -> PairedDifference:
    """Perform a paired comparison test between sample A and sample B."""

    a_list = [float(x) for x in sample_a]
    b_list = [float(x) for x in sample_b]
    if len(a_list) != len(b_list):
        raise ValueError("Paired samples must have equal length.")
    n = len(a_list)
    if n == 0:
        return PairedDifference(
            metric=metric,
            sample_size=0,
            mean_a=0.0,
            mean_b=0.0,
            mean_difference=0.0,
            standard_deviation_difference=0.0,
            standard_error=0.0,
            confidence_interval_95=(0.0, 0.0),
            t_statistic=0.0,
            p_value=1.0,
            cohens_d=0.0,
            hedges_g=0.0,
            effect_size_magnitude="negligible",
            wins=0,
            losses=0,
            ties=0,
            win_rate=0.0,
            sign_test_p_value=1.0,
        )

    mean_a = mean(a_list)
    mean_b = mean(b_list)
    diffs = [a - b for a, b in zip(a_list, b_list)]
    mean_diff = mean(diffs)

    wins = sum(1 for a, b in zip(a_list, b_list) if a > b)
    losses = sum(1 for a, b in zip(a_list, b_list) if a < b)
    ties = sum(1 for a, b in zip(a_list, b_list) if a == b)
    win_rate = wins / n
    sign_p = sign_test_p_value(wins, losses)

    if n == 1:
        return PairedDifference(
            metric=metric,
            sample_size=1,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_difference=mean_diff,
            standard_deviation_difference=0.0,
            standard_error=0.0,
            confidence_interval_95=(mean_diff, mean_diff),
            t_statistic=0.0,
            p_value=1.0,
            cohens_d=0.0,
            hedges_g=0.0,
            effect_size_magnitude="negligible",
            wins=wins,
            losses=losses,
            ties=ties,
            win_rate=win_rate,
            sign_test_p_value=sign_p,
        )

    sd_diff = stdev(diffs)
    se = sd_diff / sqrt(n)
    df = n - 1
    t_crit = t_critical(df) if use_t else 1.96
    margin = t_crit * se
    ci_95 = (mean_diff - margin, mean_diff + margin)

    if sd_diff <= 0.0:
        t_stat = 0.0
        p_val = 1.0 if mean_diff == 0.0 else 0.0
        d = 0.0
        g = 0.0
    else:
        t_stat = mean_diff / se
        p_val = student_t_p_value(t_stat, df)
        d = mean_diff / sd_diff
        g = hedges_g(a_list, b_list, paired=True)

    mag = effect_size_magnitude(d)
    return PairedDifference(
        metric=metric,
        sample_size=n,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_difference=mean_diff,
        standard_deviation_difference=sd_diff,
        standard_error=se,
        confidence_interval_95=ci_95,
        t_statistic=t_stat,
        p_value=p_val,
        cohens_d=d,
        hedges_g=g,
        effect_size_magnitude=mag,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=win_rate,
        sign_test_p_value=sign_p,
    )


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


def summarize(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    use_t: bool = False,
) -> Summary:
    """Compute descriptive statistics and 95% confidence interval for observations."""

    observations = [float(value) for value in values]
    if not observations:
        return Summary(0, 0.0, 0.0, (0.0, 0.0))
    n = len(observations)
    average = mean(observations)
    if n == 1:
        return Summary(1, average, 0.0, (average, average))
    deviation = stdev(observations)
    multiplier = t_critical(n - 1, confidence) if use_t else 1.96
    margin = multiplier * deviation / sqrt(n)
    return Summary(
        n, average, deviation, (average - margin, average + margin)
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
    "STUDENT_T_CRITICAL_95",
    "PairedDifference",
    "Summary",
    "accuracy_degradation",
    "area_under_curve",
    "area_under_degradation_curve",
    "cohens_d",
    "context_reduction",
    "effect_size_magnitude",
    "hedges_g",
    "mean_degradation",
    "paired_difference_test",
    "precision_at_k",
    "quality_adjusted_efficiency",
    "rate",
    "recall_at_k",
    "relative_degradation",
    "sign_test_p_value",
    "student_t_p_value",
    "summarize",
    "t_critical",
    "token_savings",
]
