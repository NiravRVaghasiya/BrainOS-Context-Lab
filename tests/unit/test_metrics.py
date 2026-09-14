import pytest

from evaluation.metrics import (
    METRICS_VERSION,
    Summary,
    accuracy_degradation,
    area_under_curve,
    area_under_degradation_curve,
    context_reduction,
    mean_degradation,
    precision_at_k,
    quality_adjusted_efficiency,
    rate,
    recall_at_k,
    relative_degradation,
    summarize,
    token_savings,
)


def test_retrieval_metrics() -> None:
    assert recall_at_k(["a", "x"], ["a", "b"]) == 0.5
    assert precision_at_k(["a", "x"], ["a", "b"]) == 0.5


def test_context_reduction_and_summary() -> None:
    assert context_reduction(100, 25) == 0.75
    summary = summarize([1.0, 2.0, 3.0])
    assert summary.count == 3
    assert summary.mean == 2.0
    assert summary.confidence_interval_95[0] < 2.0 < summary.confidence_interval_95[1]


def test_token_savings_and_quality_adjusted_efficiency() -> None:
    assert token_savings(1000, 250) == 750
    assert token_savings(100, 150) == 0
    assert quality_adjusted_efficiency(1.0, 100) == 0.01
    assert quality_adjusted_efficiency(0.0, 100) == 0.0
    assert quality_adjusted_efficiency(1.0, 0) == 0.0


def test_degradation_is_signed() -> None:
    assert accuracy_degradation(0.9, 0.6) == pytest.approx(0.3)
    assert accuracy_degradation(0.5, 0.8) == pytest.approx(-0.3)
    assert relative_degradation(0.8, 0.4) == pytest.approx(0.5)
    assert relative_degradation(0.0, 0.0) == 0.0
    assert relative_degradation(0.0, 0.4) == 0.0


def test_area_under_curve_is_trapezoidal_and_sorts_inputs() -> None:
    assert area_under_curve([0, 10], [0, 0]) == 0.0
    assert area_under_curve([0], [1]) == 0.0
    # Unsorted 5k→1.0, 15k→0.0: triangle of width 10000 height 1 → 5000.
    assert area_under_curve([15000, 5000], [0.0, 1.0]) == 5000.0
    # Duplicate x is a zero-width segment and contributes nothing; the
    # remaining 1→2 at y=1 is a rectangle of area 1.
    assert area_under_curve([1, 1, 2], [1, 1, 1]) == 1.0


def test_degradation_auc_uses_the_shortest_length_as_reference() -> None:
    lengths = [5000, 10000, 20000]
    accuracies = [1.0, 0.8, 0.4]
    # degradations: 0, 0.2, 0.6. Trapezoids: 5000*0.1 + 10000*0.4 = 4500.
    assert area_under_degradation_curve(lengths, accuracies) == pytest.approx(4500.0)
    assert mean_degradation(lengths, accuracies) == pytest.approx(4500.0 / 15000)
    assert mean_degradation([800], [1.0]) == 0.0


def test_rate_is_zero_on_an_empty_denominator() -> None:
    assert rate(3, 4) == 0.75
    assert rate(1, 0) == 0.0


def test_summary_to_dict_is_json_ready() -> None:
    payload = summarize([1.0, 2.0, 3.0]).to_dict()

    assert payload["count"] == 3
    assert payload["mean"] == 2.0
    assert isinstance(payload["confidence_interval_95"], tuple)
    assert METRICS_VERSION == "metrics-v1"
    assert isinstance(Summary(0, 0.0, 0.0, (0.0, 0.0)), Summary)
