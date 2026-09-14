"""Report generation for evaluation results.

Phase 8 writes a structured metrics report (the plan's quality / efficiency /
robustness numbers) without credentials or provider clients. Phase 10 adds
statistical evaluation reports with trial-level confidence intervals and
paired comparison effect sizes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import (
    compare_runs,
    headline_from_aggregate,
    plot_series,
    statistical_analysis,
    statistical_plot_series,
)
from .metrics import METRICS_VERSION


def write_json_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write a structured report without credentials or provider clients."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    destination.write_text(rendered, encoding="utf-8")
    return destination


def metrics_report(run: dict[str, Any]) -> dict[str, Any]:
    """Build the Phase 8 metrics report from one exported run."""

    config = run.get("config", {}) or {}
    aggregate = run.get("aggregate_metrics", {}) or {}
    return {
        "run_id": run.get("run_id", ""),
        "timestamp": run.get("timestamp", ""),
        "mode": config.get("mode", ""),
        "model": config.get("model", ""),
        "metrics_version": aggregate.get("metrics_version", METRICS_VERSION),
        "headline": headline_from_aggregate(aggregate),
        "by_category": aggregate.get("by_category", {}),
        "by_length": aggregate.get("by_length", {}),
        "degradation": aggregate.get("degradation", {}),
        "error_counts": aggregate.get("error_counts", {}),
        "verdict_counts": aggregate.get("verdict_counts", {}),
        "token_counters": aggregate.get("token_counters", []),
        "dataset_issues": list(run.get("dataset_issues") or []),
    }


def comparison_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the cross-mode comparison the evaluation matrix needs."""

    return {
        "metrics_version": METRICS_VERSION,
        "runs": compare_runs(runs),
        "series": plot_series(runs),
    }


def statistical_report(
    runs_or_experiment: Any,
    *,
    baseline_mode: str = "full_context",
) -> dict[str, Any]:
    """Build the Phase 10 statistical evaluation report."""

    analysis = statistical_analysis(runs_or_experiment, baseline_mode=baseline_mode)
    series = statistical_plot_series(runs_or_experiment)
    return {
        "metrics_version": METRICS_VERSION,
        "trial_summaries": analysis.get("trial_summaries", {}),
        "paired_comparisons": analysis.get("paired_comparisons", []),
        "headline_table": analysis.get("headline_table", []),
        "series": series,
    }


__all__ = [
    "comparison_report",
    "metrics_report",
    "statistical_report",
    "write_json_report",
]
