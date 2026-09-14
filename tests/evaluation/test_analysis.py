"""Phase 8 analysis: headline comparison and plot-ready series."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.analysis import HEADLINE_KEYS, compare_runs, plot_series
from evaluation.compare import main as compare_main
from evaluation.metrics import METRICS_VERSION
from evaluation.plots import PLOT_NAMES, plot_accuracy_by_length
from evaluation.reports import comparison_report, metrics_report, write_json_report


def _run(mode: str, *, accuracy: float, tokens: float, length: int = 800) -> dict:
    return {
        "run_id": f"run-{mode}",
        "timestamp": "2026-09-14T00:00:00+00:00",
        "config": {"mode": mode, "model": "mock"},
        "aggregate_metrics": {
            "metrics_version": METRICS_VERSION,
            "answer_accuracy": accuracy,
            "retrieval_recall": 1.0 if mode != "full_context" else 0.0,
            "retrieval_precision": 0.4,
            "evidence_in_prompt_rate": 1.0 if mode != "sliding_window" else 0.0,
            "faithfulness": accuracy,
            "conflict_resolution_accuracy": accuracy,
            "abstention_accuracy": 1.0,
            "mean_final_context_tokens": tokens,
            "mean_full_context_reference_tokens": 1000.0,
            "mean_context_reduction": max(0.0, 1.0 - tokens / 1000.0),
            "mean_token_savings": max(0.0, 1000.0 - tokens),
            "mean_quality_adjusted_efficiency": accuracy / tokens if tokens else 0.0,
            "quality_per_token": accuracy / tokens if tokens else 0.0,
            "mean_latency_ms": None,
            "by_length": {
                str(length): {
                    "length": length,
                    "answer_accuracy": accuracy,
                    "mean_final_context_tokens": tokens,
                }
            },
            "degradation": {
                "point_count": 1,
                "area_under_degradation_curve": None,
                "note": "A single length cannot support a degradation curve.",
            },
        },
        "dataset_issues": [],
    }


def test_compare_runs_exposes_a_headline_row_per_mode() -> None:
    comparison = compare_runs(
        [_run("full_context", accuracy=1.0, tokens=1000), _run("brainos", accuracy=1.0, tokens=200)]
    )

    assert [row["mode"] for row in comparison] == ["full_context", "brainos"]
    assert set(comparison[0]["headline"]) == set(HEADLINE_KEYS)
    assert comparison[1]["headline"]["mean_token_savings"] == 800.0
    assert comparison[0]["aggregate_metrics"]["answer_accuracy"] == 1.0


def test_plot_series_covers_the_six_planned_plots() -> None:
    series = plot_series(
        [
            _run("sliding_window", accuracy=0.0, tokens=180),
            _run("brainos", accuracy=1.0, tokens=190),
        ]
    )

    assert set(series) == set(PLOT_NAMES)
    assert {point["mode"] for point in series["accuracy_vs_length"]} == {
        "sliding_window",
        "brainos",
    }
    brainos = next(point for point in series["token_savings"] if point["mode"] == "brainos")
    window = next(
        point for point in series["token_savings"] if point["mode"] == "sliding_window"
    )
    assert brainos["token_savings"] == 810.0
    assert window["context_reduction"] == pytest.approx(0.82)


def test_metrics_report_is_credential_free_and_json_ready(tmp_path: Path) -> None:
    run = _run("brainos", accuracy=1.0, tokens=200)
    report = metrics_report(run)

    assert report["mode"] == "brainos"
    assert report["headline"]["answer_accuracy"] == 1.0
    assert "api_key" not in json.dumps(report)
    path = write_json_report(report, tmp_path / "metrics.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metrics_version"] == METRICS_VERSION


def test_comparison_report_includes_series() -> None:
    report = comparison_report([_run("brainos", accuracy=1.0, tokens=200)])

    assert "series" in report
    assert "runs" in report
    assert report["series"]["retrieval"][0]["recall"] == 1.0


def test_compare_cli_writes_the_structured_report(tmp_path: Path) -> None:
    left = tmp_path / "full.json"
    right = tmp_path / "brainos.json"
    left.write_text(json.dumps(_run("full_context", accuracy=1.0, tokens=1000)), encoding="utf-8")
    right.write_text(json.dumps(_run("brainos", accuracy=1.0, tokens=200)), encoding="utf-8")
    output = tmp_path / "compare.json"

    assert compare_main([str(left), str(right), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [row["mode"] for row in payload["runs"]] == ["full_context", "brainos"]
    assert set(payload["series"]) == set(PLOT_NAMES)


def test_plotting_without_matplotlib_raises_a_clear_error(tmp_path: Path, monkeypatch) -> None:
    import evaluation.plots as plots

    def _boom():
        raise RuntimeError("Plotting requires the `evaluation` optional dependency.")

    monkeypatch.setattr(plots, "_pyplot", _boom)

    try:
        plot_accuracy_by_length(
            [{"mode": "brainos", "conversation_length": 800, "accuracy": 1.0}],
            tmp_path / "plot.png",
        )
    except RuntimeError as exc:
        assert "optional dependency" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when matplotlib is unavailable")
