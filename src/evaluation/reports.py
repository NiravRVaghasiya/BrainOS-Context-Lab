"""Report generation for evaluation results.

Phase 8 writes a structured metrics report (the plan's quality / efficiency /
robustness numbers) without credentials or provider clients. Markdown and plot
rendering stay optional: JSON is the archival form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import compare_runs, headline_from_aggregate, plot_series
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


__all__ = [
    "comparison_report",
    "metrics_report",
    "write_json_report",
]
