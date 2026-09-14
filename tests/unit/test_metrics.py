import pytest

from evaluation.metrics import (
    METRICS_VERSION,
    PairedDifference,
    Summary,
    accuracy_degradation,
    area_under_curve,
    area_under_degradation_curve,
    cohens_d,
    context_reduction,
    effect_size_magnitude,
    hedges_g,
    mean_degradation,
    paired_difference_test,
    precision_at_k,
    quality_adjusted_efficiency,
    rate,
    recall_at_k,
    relative_degradation,
    sign_test_p_value,
    student_t_p_value,
    summarize,
    t_critical,
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


# --------------------------------------------------------------------------- #
# Phase 10: Statistical Evaluation Primitives
# --------------------------------------------------------------------------- #


def test_t_critical_values_and_summarize_with_t() -> None:
    # Small sample degrees of freedom (df = 2 for n = 3)
    assert t_critical(2) == pytest.approx(4.302653, rel=1e-4)
    assert t_critical(10) == pytest.approx(2.228139, rel=1e-4)
    assert t_critical(100) == pytest.approx(1.984, rel=1e-2)
    assert t_critical(0) == 1.96

    summ_z = summarize([1.0, 2.0, 3.0], use_t=False)
    summ_t = summarize([1.0, 2.0, 3.0], use_t=True)
    assert summ_t.count == 3
    assert summ_t.mean == 2.0
    # t CI is wider than normal approx for small n
    margin_z = summ_z.confidence_interval_95[1] - summ_z.mean
    margin_t = summ_t.confidence_interval_95[1] - summ_t.mean
    assert margin_t > margin_z

    # Edge cases
    assert summarize([]).count == 0
    assert summarize([5.0]).confidence_interval_95 == (5.0, 5.0)


def test_student_t_p_value_and_sign_test() -> None:
    assert student_t_p_value(0.0, 10) == 1.0
    assert student_t_p_value(2.0, 10) == pytest.approx(0.073388, rel=1e-3)
    assert student_t_p_value(4.0, 10) < 0.01
    assert student_t_p_value(1.0, 0) == 1.0

    # Sign test (binomial)
    assert sign_test_p_value(7, 0) == pytest.approx(2.0 * (0.5**7))
    assert sign_test_p_value(5, 2) == pytest.approx(0.453125)
    assert sign_test_p_value(0, 0) == 1.0


def test_cohens_d_and_hedges_g() -> None:
    a = [1.0, 0.9, 0.8, 1.0, 0.9]
    b = [0.4, 0.5, 0.3, 0.4, 0.5]
    d_paired = cohens_d(a, b, paired=True)
    g_paired = hedges_g(a, b, paired=True)

    assert d_paired > 0.8  # Large effect size
    assert g_paired < d_paired  # Hedges' g applies small-sample correction (J < 1)
    assert effect_size_magnitude(d_paired) == "large"
    assert effect_size_magnitude(0.1) == "negligible"
    assert effect_size_magnitude(0.3) == "small"
    assert effect_size_magnitude(0.6) == "medium"

    d_indep = cohens_d(a, b, paired=False)
    assert d_indep > 0.8
    assert hedges_g(a, b, paired=False) > 0.0

    # Edge cases
    assert cohens_d([], []) == 0.0
    assert cohens_d([1.0], [1.0]) == 0.0
    with pytest.raises(ValueError):
        cohens_d([1.0, 2.0], [1.0], paired=True)


def test_paired_difference_test_and_edge_cases() -> None:
    a = [1.0, 1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0, 0.0, 0.0]
    res = paired_difference_test(a, b, metric="accuracy")

    assert isinstance(res, PairedDifference)
    assert res.metric == "accuracy"
    assert res.sample_size == 5
    assert res.mean_a == 0.8
    assert res.mean_b == 0.2
    assert res.mean_difference == pytest.approx(0.6)
    assert res.wins == 3
    assert res.losses == 0
    assert res.ties == 2
    assert res.win_rate == 0.6
    assert res.t_statistic > 0.0
    assert res.p_value < 0.10
    assert res.cohens_d > 1.0
    assert res.effect_size_magnitude == "large"

    # Edge cases: empty, n=1, zero variance
    empty_res = paired_difference_test([], [])
    assert empty_res.sample_size == 0
    assert empty_res.p_value == 1.0

    single_res = paired_difference_test([1.0], [0.5])
    assert single_res.sample_size == 1
    assert single_res.mean_difference == 0.5
    assert single_res.confidence_interval_95 == (0.5, 0.5)

    flat_res = paired_difference_test([1.0, 1.0], [1.0, 1.0])
    assert flat_res.mean_difference == 0.0
    assert flat_res.p_value == 1.0

    with pytest.raises(ValueError):
        paired_difference_test([1.0], [1.0, 2.0])

    payload = res.to_dict()
    assert payload["metric"] == "accuracy"
    assert payload["effect_size_magnitude"] == "large"
    assert isinstance(payload["confidence_interval_95"], tuple)
