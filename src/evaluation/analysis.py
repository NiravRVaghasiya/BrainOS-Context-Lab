"""Analysis helpers for comparing exported evaluation runs.

Phase 8 turns the comparison from "dump each run's aggregate" into the
plan's headline table: quality, efficiency, and robustness side by side, plus
the plot-ready series Phase 10 renders. Statistical significance, paired
tests, and effect sizes still belong to Phase 10 — this module does not infer
them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .metrics import METRICS_VERSION

#: Compact columns the plan's final evaluation matrix actually needs. A run
#: that predates Phase 8 simply omits the new keys.
HEADLINE_KEYS: tuple[str, ...] = (
    "answer_accuracy",
    "retrieval_recall",
    "retrieval_precision",
    "evidence_in_prompt_rate",
    "faithfulness",
    "conflict_resolution_accuracy",
    "abstention_accuracy",
    "mean_final_context_tokens",
    "mean_full_context_reference_tokens",
    "mean_context_reduction",
    "mean_token_savings",
    "mean_quality_adjusted_efficiency",
    "quality_per_token",
    "mean_latency_ms",
)


def headline_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact metric row used in cross-mode tables."""

    return {key: aggregate.get(key) for key in HEADLINE_KEYS}


def compare_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact mode-oriented comparison table.

    The function does not infer statistical significance; that belongs to the
    paired-trial analysis phase.
    """

    comparison = []
    for run in runs:
        config = run.get("config", {})
        aggregate = run.get("aggregate_metrics", {}) or {}
        comparison.append(
            {
                "run_id": run.get("run_id", ""),
                "mode": config.get("mode", ""),
                "model": config.get("model", ""),
                "metrics_version": aggregate.get("metrics_version", METRICS_VERSION),
                "headline": headline_from_aggregate(aggregate),
                "degradation": aggregate.get("degradation", {}),
                "aggregate_metrics": aggregate,
            }
        )
    return comparison


def plot_series(runs: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the six plan-required plot series from exported runs.

    Each series is a list of ``{mode, x, y, ...}`` points so a renderer (or a
    test) can consume them without matplotlib. Missing length ladders produce
    one point per run rather than an empty series: a smoke-tier plot is a
    degenerate scatter, not a missing plot.
    """

    accuracy_vs_length: list[dict[str, Any]] = []
    tokens_vs_length: list[dict[str, Any]] = []
    accuracy_vs_tokens: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    savings: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []

    for run in runs:
        config = run.get("config", {}) if isinstance(run.get("config"), Mapping) else {}
        aggregate = run.get("aggregate_metrics", {}) or {}
        mode = str(config.get("mode") or aggregate.get("mode") or "")
        by_length = aggregate.get("by_length") or {}
        if by_length:
            for point in by_length.values():
                length = point.get("length", 0)
                accuracy = point.get("answer_accuracy", 0.0)
                tokens = point.get("mean_final_context_tokens", 0.0)
                accuracy_vs_length.append(
                    {"mode": mode, "conversation_length": length, "accuracy": accuracy}
                )
                tokens_vs_length.append(
                    {
                        "mode": mode,
                        "conversation_length": length,
                        "context_tokens": tokens,
                    }
                )
                accuracy_vs_tokens.append(
                    {"mode": mode, "context_tokens": tokens, "accuracy": accuracy}
                )
        else:
            tokens = aggregate.get("mean_final_context_tokens", 0.0) or 0.0
            accuracy = aggregate.get("answer_accuracy", 0.0) or 0.0
            accuracy_vs_length.append(
                {"mode": mode, "conversation_length": 0, "accuracy": accuracy}
            )
            tokens_vs_length.append(
                {"mode": mode, "conversation_length": 0, "context_tokens": tokens}
            )
            accuracy_vs_tokens.append(
                {"mode": mode, "context_tokens": tokens, "accuracy": accuracy}
            )
        retrieval.append(
            {
                "mode": mode,
                "recall": aggregate.get("retrieval_recall", 0.0) or 0.0,
                "precision": aggregate.get("retrieval_precision", 0.0) or 0.0,
                "evidence_in_prompt": aggregate.get("evidence_in_prompt_rate", 0.0) or 0.0,
            }
        )
        savings.append(
            {
                "mode": mode,
                "token_savings": aggregate.get("mean_token_savings", 0.0) or 0.0,
                "context_reduction": aggregate.get("mean_context_reduction", 0.0) or 0.0,
            }
        )
        efficiency.append(
            {
                "mode": mode,
                "quality_adjusted_efficiency": aggregate.get(
                    "mean_quality_adjusted_efficiency", 0.0
                )
                or 0.0,
                "quality_per_token": aggregate.get("quality_per_token", 0.0) or 0.0,
                "accuracy": aggregate.get("answer_accuracy", 0.0) or 0.0,
                "context_tokens": aggregate.get("mean_final_context_tokens", 0.0) or 0.0,
            }
        )

    return {
        "accuracy_vs_length": accuracy_vs_length,
        "tokens_vs_length": tokens_vs_length,
        "accuracy_vs_tokens": accuracy_vs_tokens,
        "retrieval": retrieval,
        "token_savings": savings,
        "quality_adjusted_efficiency": efficiency,
    }


def length_curve_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows the accuracy-vs-length plot historically accepted."""

    return list(plot_series(list(runs))["accuracy_vs_length"])


__all__ = [
    "HEADLINE_KEYS",
    "compare_runs",
    "headline_from_aggregate",
    "length_curve_rows",
    "plot_series",
]
