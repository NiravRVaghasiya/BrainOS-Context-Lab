"""Analysis helpers for comparing exported evaluation runs."""

from __future__ import annotations

from typing import Any


def compare_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact mode-oriented comparison table.

    The function does not infer statistical significance; that belongs to the
    paired-trial analysis phase.
    """

    comparison = []
    for run in runs:
        config = run.get("config", {})
        comparison.append(
            {
                "run_id": run.get("run_id", ""),
                "mode": config.get("mode", ""),
                "model": config.get("model", ""),
                "aggregate_metrics": run.get("aggregate_metrics", {}),
            }
        )
    return comparison
