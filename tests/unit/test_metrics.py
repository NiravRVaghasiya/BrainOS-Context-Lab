from evaluation.metrics import (
    context_reduction,
    precision_at_k,
    recall_at_k,
    summarize,
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
