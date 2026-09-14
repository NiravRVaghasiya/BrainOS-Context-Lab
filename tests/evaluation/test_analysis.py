"""Phase 8 & Phase 10 analysis: headline comparison, statistical evaluation, and plots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.analysis import (
    HEADLINE_KEYS,
    compare_modes_paired,
    compare_runs,
    pairwise_comparisons,
    plot_series,
    statistical_analysis,
    statistical_plot_series,
    summarize_trial_modes,
)
from evaluation.compare import main as compare_main
from evaluation.metrics import METRICS_VERSION
from evaluation.plots import (
    PLOT_NAMES,
    plot_accuracy_by_length,
    plot_all,
)
from evaluation.reports import (
    comparison_report,
    metrics_report,
    statistical_report,
    write_json_report,
)


def _run(
    mode: str,
    *,
    accuracy: float,
    tokens: float,
    length: int = 800,
    trial: int = 0,
    tasks_count: int = 3,
) -> dict:
    scores = [
        {
            "task_id": f"task-{i}",
            "mode": mode,
            "answer": {"verdict": "correct" if (i == 0 or accuracy >= 0.5) else "incorrect"},
            "retrieval": {
                "recall": 1.0 if mode != "full_context" else 0.0,
                "precision": 0.5,
                "evidence_in_prompt": mode != "sliding_window",
            },
            "faithfulness": accuracy,
            "final_context_tokens": tokens + (i * 10),
            "full_context_reference_tokens": 1000.0,
            "token_savings": max(0.0, 1000.0 - (tokens + (i * 10))),
            "quality_adjusted_efficiency": accuracy / (tokens + 10),
            "latency_ms": 120.0 + i,
        }
        for i in range(tasks_count)
    ]

    return {
        "run_id": f"run-{mode}-t{trial}",
        "trial": trial,
        "timestamp": "2026-09-14T00:00:00+00:00",
        "config": {"mode": mode, "model": "mock", "trial": trial},
        "scores": scores,
        "aggregate_metrics": {
            "metrics_version": METRICS_VERSION,
            "mode": mode,
            "trial": trial,
            "task_count": tasks_count,
            "graded_answer_count": tasks_count,
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
            "mean_latency_ms": 120.0,
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
        [
            _run("full_context", accuracy=1.0, tokens=1000),
            _run("brainos", accuracy=1.0, tokens=200),
        ]
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
    left.write_text(
        json.dumps(_run("full_context", accuracy=1.0, tokens=1000)), encoding="utf-8"
    )
    right.write_text(
        json.dumps(_run("brainos", accuracy=1.0, tokens=200)), encoding="utf-8"
    )
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


# --------------------------------------------------------------------------- #
# Phase 10: Statistical Evaluation Tests
# --------------------------------------------------------------------------- #


def test_summarize_trial_modes_aggregates_across_repeated_trials() -> None:
    trial_runs = [
        _run("brainos", accuracy=0.90, tokens=200, trial=0),
        _run("brainos", accuracy=0.85, tokens=210, trial=1),
        _run("brainos", accuracy=0.95, tokens=190, trial=2),
        _run("full_context", accuracy=0.70, tokens=1000, trial=0),
        _run("full_context", accuracy=0.65, tokens=1000, trial=1),
        _run("full_context", accuracy=0.75, tokens=1000, trial=2),
    ]

    summaries = summarize_trial_modes(trial_runs)
    assert "brainos" in summaries
    assert "full_context" in summaries

    brainos_summary = summaries["brainos"]
    assert brainos_summary["trials"] == 3
    acc_stat = brainos_summary["metrics"]["answer_accuracy"]
    assert acc_stat["count"] == 3
    assert acc_stat["mean"] == pytest.approx(0.90)
    assert acc_stat["standard_deviation"] == pytest.approx(0.05)
    ci = acc_stat["confidence_interval_95"]
    assert ci[0] < 0.90 < ci[1]


def test_compare_modes_paired_and_pairwise_comparisons() -> None:
    runs = [
        _run("brainos", accuracy=1.0, tokens=200, trial=0, tasks_count=5),
        _run("sliding_window", accuracy=0.2, tokens=180, trial=0, tasks_count=5),
        _run("full_context", accuracy=0.8, tokens=1000, trial=0, tasks_count=5),
    ]

    paired_b_vs_sw = compare_modes_paired(runs, mode_a="brainos", mode_b="sliding_window")
    assert paired_b_vs_sw
    acc_diff = next(p for p in paired_b_vs_sw if p["metric"] == "accuracy")
    assert acc_diff["mode_a"] == "brainos"
    assert acc_diff["mode_b"] == "sliding_window"
    assert acc_diff["mean_difference"] > 0.0
    assert acc_diff["wins"] > 0
    assert acc_diff["effect_size_magnitude"] in ("medium", "large")

    all_pairs = pairwise_comparisons(runs, baseline_mode="full_context")
    pair_names = {(p["mode_a"], p["mode_b"]) for p in all_pairs}
    assert ("brainos", "full_context") in pair_names
    assert ("sliding_window", "full_context") in pair_names


def test_statistical_analysis_and_report_schema() -> None:
    runs = [
        _run("brainos", accuracy=0.9, tokens=200, trial=0),
        _run("brainos", accuracy=0.95, tokens=200, trial=1),
        _run("full_context", accuracy=0.7, tokens=1000, trial=0),
        _run("full_context", accuracy=0.75, tokens=1000, trial=1),
    ]

    analysis = statistical_analysis(runs, baseline_mode="full_context")
    assert analysis["metrics_version"] == METRICS_VERSION
    assert "brainos" in analysis["trial_summaries"]
    assert "full_context" in analysis["trial_summaries"]
    assert analysis["paired_comparisons"]
    assert analysis["headline_table"]

    report = statistical_report(runs, baseline_mode="full_context")
    assert report["metrics_version"] == METRICS_VERSION
    assert "trial_summaries" in report
    assert "paired_comparisons" in report
    assert "series" in report
    assert set(report["series"]) == set(PLOT_NAMES)


def test_statistical_plot_series_and_rendering_all_six_plots(tmp_path: Path) -> None:
    runs = [
        _run("brainos", accuracy=0.9, tokens=200, trial=0, length=800),
        _run("brainos", accuracy=0.95, tokens=210, trial=1, length=800),
        _run("full_context", accuracy=0.7, tokens=1000, trial=0, length=800),
        _run("full_context", accuracy=0.75, tokens=1010, trial=1, length=800),
    ]

    series = statistical_plot_series(runs)
    assert set(series) == set(PLOT_NAMES)
    acc_pt = next(pt for pt in series["accuracy_vs_length"] if pt["mode"] == "brainos")
    assert acc_pt["accuracy"] == pytest.approx(0.925)
    assert acc_pt["accuracy_sd"] > 0.0

    # Render all 6 plots to disk
    plots_dir = tmp_path / "plots"
    written = plot_all(runs, plots_dir)
    assert len(written) == 6
    for name, p in written.items():
        assert p.exists()
        assert p.stat().st_size > 0


def test_compare_cli_with_stats_and_plots_dir(tmp_path: Path) -> None:
    run1 = tmp_path / "brainos-t0.json"
    run2 = tmp_path / "brainos-t1.json"
    run3 = tmp_path / "full-t0.json"
    run1.write_text(
        json.dumps(_run("brainos", accuracy=0.9, tokens=200, trial=0)), encoding="utf-8"
    )
    run2.write_text(
        json.dumps(_run("brainos", accuracy=0.95, tokens=210, trial=1)), encoding="utf-8"
    )
    run3.write_text(
        json.dumps(_run("full_context", accuracy=0.7, tokens=1000, trial=0)), encoding="utf-8"
    )

    out_report = tmp_path / "stat_report.json"
    plots_dir = tmp_path / "cli_plots"

    exit_code = compare_main(
        [
            str(run1),
            str(run2),
            str(run3),
            "--stats",
            "--output",
            str(out_report),
            "--plots-dir",
            str(plots_dir),
        ]
    )
    assert exit_code == 0
    assert out_report.exists()
    payload = json.loads(out_report.read_text(encoding="utf-8"))
    assert "trial_summaries" in payload
    assert "paired_comparisons" in payload
    assert len(list(plots_dir.glob("*.png"))) == 6
