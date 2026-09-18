"""Analysis helpers for comparing exported evaluation runs and statistical evaluation.

Phase 8 established the headline comparison table, degradation metrics, and plot series.
Phase 10 adds the full statistical evaluation suite:

* trial-level descriptive statistics (mean, SD, 95% CI) across repeated stochastic trials;
* paired comparison tests across benchmark tasks (paired t-test, Cohen's d, Hedges' g,
  win/loss/tie sign tests);
* statistical plot series with confidence intervals and standard deviations;
* multi-length ladder robustness statistics across trials.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from baselines.ablations import ablation_pairs
from baselines.modes import ABLATION_ORDER, MODE_ORDER, mode_label

from .metrics import (
    METRICS_VERSION,
    paired_difference_test,
    summarize,
)

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

#: Metrics summarized across trials in statistical evaluations.
STATISTICAL_METRICS: tuple[str, ...] = HEADLINE_KEYS


def headline_from_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact metric row used in cross-mode tables."""

    return {key: aggregate.get(key) for key in HEADLINE_KEYS}


def compare_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact mode-oriented comparison table."""

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


def extract_mode_result_items(source: Any) -> list[dict[str, Any]]:
    """Normalize run structures into a uniform list of mode-trial dicts.

    Public because Phase 12's error analysis consumes the same three shapes the
    statistical analysis does — a live ``ExperimentRun``, an exported
    experiment artifact, or a list of run files — and a second normalizer would
    be a second place for the shapes to drift apart.
    """

    if hasattr(source, "mode_results"):
        return [
            {
                "mode": getattr(res, "mode", ""),
                "label": getattr(res, "label", ""),
                "trial": getattr(res, "trial", 0),
                "aggregate_metrics": getattr(res, "aggregate_metrics", {}) or {},
                "scores": list(getattr(res, "scores", ()) or ()),
                "config": getattr(res, "config", {}) or {},
                "task_results": list(getattr(res, "task_results", ()) or ()),
            }
            for res in source.mode_results
        ]
    if isinstance(source, Mapping) and "modes" in source and isinstance(source["modes"], list):
        return [
            {
                "mode": str(m.get("mode", "")),
                "label": str(m.get("label", "")),
                "trial": int(m.get("trial", 0) or 0),
                "aggregate_metrics": m.get("aggregate_metrics", {}) or {},
                "scores": list(m.get("scores") or ()),
                "config": m.get("config", {}) or {},
                "task_results": list(m.get("task_results") or ()),
            }
            for m in source["modes"]
        ]
    if isinstance(source, Sequence):
        items: list[dict[str, Any]] = []
        for elem in source:
            if hasattr(elem, "mode"):
                items.append(
                    {
                        "mode": getattr(elem, "mode", ""),
                        "label": getattr(elem, "label", ""),
                        "trial": getattr(elem, "trial", 0),
                        "aggregate_metrics": getattr(elem, "aggregate_metrics", {}) or {},
                        "scores": list(getattr(elem, "scores", ()) or ()),
                        "config": getattr(elem, "config", {}) or {},
                        "task_results": list(getattr(elem, "task_results", ()) or ()),
                    }
                )
            elif isinstance(elem, Mapping):
                cfg = elem.get("config") if isinstance(elem.get("config"), Mapping) else {}
                agg = (
                    elem.get("aggregate_metrics")
                    if isinstance(elem.get("aggregate_metrics"), Mapping)
                    else {}
                )
                mode_str = str(cfg.get("mode") or agg.get("mode") or elem.get("mode") or "")
                trial_num = int(elem.get("trial") or cfg.get("trial") or 0)
                items.append(
                    {
                        "mode": mode_str,
                        "label": str(elem.get("label") or mode_label(mode_str)),
                        "trial": trial_num,
                        "aggregate_metrics": agg or dict(elem),
                        "scores": list(elem.get("scores") or ()),
                        "config": cfg,
                        "task_results": list(elem.get("task_results") or ()),
                    }
                )
        return items
    return []


def summarize_trial_modes(runs_or_results: Any) -> dict[str, dict[str, Any]]:
    """Summarize metrics across trials for each baseline mode."""

    items = extract_mode_result_items(runs_or_results)
    if not items:
        return {}

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        m = item["mode"]
        if not m:
            continue
        by_mode.setdefault(m, []).append(item)

    ordered_modes = [m for m in (*MODE_ORDER, *ABLATION_ORDER) if m in by_mode] + [
        m for m in by_mode if m not in MODE_ORDER and m not in ABLATION_ORDER
    ]

    summaries: dict[str, dict[str, Any]] = {}
    for mode in ordered_modes:
        mode_items = by_mode[mode]
        trials_count = len(mode_items)
        label = mode_items[0].get("label") or mode_label(mode)

        metric_summaries: dict[str, dict[str, Any]] = {}
        for key in STATISTICAL_METRICS:
            values = [
                float(item["aggregate_metrics"][key])
                for item in mode_items
                if key in item["aggregate_metrics"]
                and item["aggregate_metrics"][key] is not None
            ]
            if values:
                metric_summaries[key] = summarize(values, use_t=True).to_dict()

        # Group by_length across trials
        length_tiers: set[int] = set()
        for item in mode_items:
            by_len = item["aggregate_metrics"].get("by_length") or {}
            for k in by_len:
                try:
                    length_tiers.add(int(k))
                except (ValueError, TypeError):
                    pass

        by_length_summaries: dict[str, dict[str, Any]] = {}
        for length in sorted(length_tiers):
            str_len = str(length)
            acc_vals: list[float] = []
            tok_vals: list[float] = []
            for item in mode_items:
                point = (item["aggregate_metrics"].get("by_length") or {}).get(str_len)
                if point:
                    if point.get("answer_accuracy") is not None:
                        acc_vals.append(float(point["answer_accuracy"]))
                    if point.get("mean_final_context_tokens") is not None:
                        tok_vals.append(float(point["mean_final_context_tokens"]))
            by_length_summaries[str_len] = {
                "length": length,
                "trials": len(acc_vals),
                "accuracy": summarize(acc_vals, use_t=True).to_dict() if acc_vals else None,
                "context_tokens": summarize(tok_vals, use_t=True).to_dict() if tok_vals else None,
            }

        # Degradation AUC summary across trials
        auc_vals = [
            float(item["aggregate_metrics"]["degradation"]["area_under_degradation_curve"])
            for item in mode_items
            if item["aggregate_metrics"].get("degradation")
            and item["aggregate_metrics"]["degradation"].get("area_under_degradation_curve")
            is not None
        ]
        mean_deg_vals = [
            float(item["aggregate_metrics"]["degradation"]["mean_degradation"])
            for item in mode_items
            if item["aggregate_metrics"].get("degradation")
            and item["aggregate_metrics"]["degradation"].get("mean_degradation") is not None
        ]

        degradation_summary = {
            "trials_with_curve": len(auc_vals),
            "area_under_curve": summarize(auc_vals, use_t=True).to_dict() if auc_vals else None,
            "mean_degradation": (
                summarize(mean_deg_vals, use_t=True).to_dict() if mean_deg_vals else None
            ),
        }

        summaries[mode] = {
            "mode": mode,
            "label": label,
            "trials": trials_count,
            "metrics": metric_summaries,
            "by_length": by_length_summaries,
            "degradation": degradation_summary,
        }

    return summaries


def compare_modes_paired(
    runs_or_results: Any,
    *,
    mode_a: str,
    mode_b: str,
    metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Perform paired task comparisons between mode A and mode B."""

    items = extract_mode_result_items(runs_or_results)
    items_a = [it for it in items if it["mode"] == mode_a]
    items_b = [it for it in items if it["mode"] == mode_b]
    if not items_a or not items_b:
        return []

    # Map task_id -> list of trial scores
    def _extract_task_map(mode_items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        task_map: dict[str, list[dict[str, Any]]] = {}
        for it in mode_items:
            for sc in it["scores"]:
                tid = sc.get("task_id")
                if tid:
                    task_map.setdefault(tid, []).append(sc)
        return task_map

    tasks_a = _extract_task_map(items_a)
    tasks_b = _extract_task_map(items_b)
    common_task_ids = sorted(set(tasks_a) & set(tasks_b))
    if not common_task_ids:
        return []

    metric_extractors = {
        "accuracy": lambda sc: (
            1.0
            if sc.get("answer", {}).get("verdict") == "correct"
            else (0.0 if sc.get("answer", {}).get("verdict") not in (None, "ungraded") else None)
        ),
        "faithfulness": lambda sc: (
            float(sc["faithfulness"]) if sc.get("faithfulness") is not None else None
        ),
        "evidence_in_prompt": lambda sc: (
            1.0 if sc.get("retrieval", {}).get("evidence_in_prompt") else 0.0
        ),
        "retrieval_recall": lambda sc: (
            float(sc["retrieval"]["recall"])
            if sc.get("retrieval") and sc["retrieval"].get("recall") is not None
            else None
        ),
        "final_context_tokens": lambda sc: (
            float(sc["final_context_tokens"])
            if sc.get("final_context_tokens") is not None
            else None
        ),
        "token_savings": lambda sc: (
            float(sc["token_savings"]) if sc.get("token_savings") is not None else None
        ),
        "quality_adjusted_efficiency": lambda sc: (
            float(sc["quality_adjusted_efficiency"])
            if sc.get("quality_adjusted_efficiency") is not None
            else None
        ),
        "latency_ms": lambda sc: (
            float(sc["latency_ms"]) if sc.get("latency_ms") is not None else None
        ),
    }

    selected_metrics = list(metrics) if metrics else list(metric_extractors.keys())
    results: list[dict[str, Any]] = []

    for met in selected_metrics:
        extractor = metric_extractors.get(met)
        if not extractor:
            continue
        pairs_a: list[float] = []
        pairs_b: list[float] = []
        for tid in common_task_ids:
            vals_a = [extractor(sc) for sc in tasks_a[tid]]
            vals_b = [extractor(sc) for sc in tasks_b[tid]]
            clean_a = [v for v in vals_a if v is not None]
            clean_b = [v for v in vals_b if v is not None]
            if clean_a and clean_b:
                pairs_a.append(sum(clean_a) / len(clean_a))
                pairs_b.append(sum(clean_b) / len(clean_b))

        if pairs_a and len(pairs_a) == len(pairs_b):
            test_res = paired_difference_test(pairs_a, pairs_b, metric=met, use_t=True)
            res_dict = test_res.to_dict()
            res_dict["mode_a"] = mode_a
            res_dict["mode_b"] = mode_b
            res_dict["label_a"] = mode_label(mode_a)
            res_dict["label_b"] = mode_label(mode_b)
            results.append(res_dict)

    return results


def pairwise_comparisons(
    runs_or_results: Any,
    *,
    baseline_mode: str = "full_context",
    target_modes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute standard paired comparisons against a baseline and between key modes."""

    items = extract_mode_result_items(runs_or_results)
    available_modes = {it["mode"] for it in items if it["mode"]}
    if not available_modes:
        return []

    targets = (
        list(target_modes)
        if target_modes
        else [m for m in (*MODE_ORDER, *ABLATION_ORDER) if m in available_modes]
    )
    comparisons: list[dict[str, Any]] = []

    # Baseline comparisons: (Target vs Baseline)
    if baseline_mode in available_modes:
        for tgt in targets:
            if tgt != baseline_mode and tgt in available_modes:
                comparisons.extend(compare_modes_paired(items, mode_a=tgt, mode_b=baseline_mode))

    # Specific informative pairs (BrainOS vs Sliding Window, BrainOS vs RAG, BrainOS+RAG vs BrainOS)
    special_pairs = [
        ("brainos", "sliding_window"),
        ("brainos", "rag"),
        ("brainos_rag", "brainos"),
    ]
    # Phase 11: every ablation present is paired against the full system, so an
    # ablation run is interpretable whatever baseline the caller chose.
    special_pairs.extend(ablation_pairs(available_modes))
    for a, b in special_pairs:
        if a in available_modes and b in available_modes:
            # Check if not already added
            existing = any(c.get("mode_a") == a and c.get("mode_b") == b for c in comparisons)
            if not existing:
                comparisons.extend(compare_modes_paired(items, mode_a=a, mode_b=b))

    return comparisons


def statistical_analysis(
    experiment_or_runs: Any,
    *,
    baseline_mode: str = "full_context",
) -> dict[str, Any]:
    """Build the comprehensive Phase 10 statistical analysis report."""

    items = extract_mode_result_items(experiment_or_runs)
    summaries = summarize_trial_modes(items)
    paired = pairwise_comparisons(items, baseline_mode=baseline_mode)

    # Build compact headline table with trial stats (mean, SD, 95% CI)
    headline_table: list[dict[str, Any]] = []
    for mode, summ in summaries.items():
        metrics = summ.get("metrics", {})
        headline_table.append(
            {
                "mode": mode,
                "label": summ.get("label", mode),
                "trials": summ.get("trials", 1),
                "accuracy": metrics.get("answer_accuracy"),
                "faithfulness": metrics.get("faithfulness"),
                "recall": metrics.get("retrieval_recall"),
                "evidence_in_prompt": metrics.get("evidence_in_prompt_rate"),
                "conflict_resolution": metrics.get("conflict_resolution_accuracy"),
                "abstention": metrics.get("abstention_accuracy"),
                "mean_context_tokens": metrics.get("mean_final_context_tokens"),
                "mean_reduction": metrics.get("mean_context_reduction"),
                "mean_savings": metrics.get("mean_token_savings"),
                "mean_qae": metrics.get("mean_quality_adjusted_efficiency"),
                "latency_ms": metrics.get("mean_latency_ms"),
            }
        )

    return {
        "metrics_version": METRICS_VERSION,
        "modes": list(summaries.keys()),
        "trial_summaries": summaries,
        "paired_comparisons": paired,
        "headline_table": headline_table,
    }


def statistical_plot_series(
    runs_or_results: Sequence[Any] | Any,
) -> dict[str, list[dict[str, Any]]]:
    """Build the six plan-required plot series with trial statistics and error bars."""

    items = extract_mode_result_items(runs_or_results)
    if not items:
        return plot_series([])

    summaries = summarize_trial_modes(items)
    accuracy_vs_length: list[dict[str, Any]] = []
    tokens_vs_length: list[dict[str, Any]] = []
    accuracy_vs_tokens: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    savings: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []

    for mode, summ in summaries.items():
        metrics = summ.get("metrics", {})
        by_length = summ.get("by_length", {})
        if by_length:
            for pt in by_length.values():
                length = pt.get("length", 0)
                acc_stat = pt.get("accuracy") or {}
                tok_stat = pt.get("context_tokens") or {}
                acc_mean = acc_stat.get("mean", 0.0)
                tok_mean = tok_stat.get("mean", 0.0)

                accuracy_vs_length.append(
                    {
                        "mode": mode,
                        "conversation_length": length,
                        "accuracy": acc_mean,
                        "accuracy_sd": acc_stat.get("standard_deviation", 0.0),
                        "accuracy_ci": acc_stat.get("confidence_interval_95"),
                        "trials": pt.get("trials", 1),
                    }
                )
                tokens_vs_length.append(
                    {
                        "mode": mode,
                        "conversation_length": length,
                        "context_tokens": tok_mean,
                        "context_tokens_sd": tok_stat.get("standard_deviation", 0.0),
                        "context_tokens_ci": tok_stat.get("confidence_interval_95"),
                        "trials": pt.get("trials", 1),
                    }
                )
                accuracy_vs_tokens.append(
                    {
                        "mode": mode,
                        "context_tokens": tok_mean,
                        "accuracy": acc_mean,
                        "accuracy_sd": acc_stat.get("standard_deviation", 0.0),
                        "context_tokens_sd": tok_stat.get("standard_deviation", 0.0),
                        "trials": pt.get("trials", 1),
                    }
                )
        else:
            acc_stat = metrics.get("answer_accuracy") or {}
            tok_stat = metrics.get("mean_final_context_tokens") or {}
            acc_mean = acc_stat.get("mean", 0.0)
            tok_mean = tok_stat.get("mean", 0.0)
            accuracy_vs_length.append(
                {
                    "mode": mode,
                    "conversation_length": 0,
                    "accuracy": acc_mean,
                    "accuracy_sd": acc_stat.get("standard_deviation", 0.0),
                    "trials": summ.get("trials", 1),
                }
            )
            tokens_vs_length.append(
                {
                    "mode": mode,
                    "conversation_length": 0,
                    "context_tokens": tok_mean,
                    "context_tokens_sd": tok_stat.get("standard_deviation", 0.0),
                    "trials": summ.get("trials", 1),
                }
            )
            accuracy_vs_tokens.append(
                {
                    "mode": mode,
                    "context_tokens": tok_mean,
                    "accuracy": acc_mean,
                    "accuracy_sd": acc_stat.get("standard_deviation", 0.0),
                    "trials": summ.get("trials", 1),
                }
            )

        rec_stat = metrics.get("retrieval_recall") or {}
        prec_stat = metrics.get("retrieval_precision") or {}
        ev_stat = metrics.get("evidence_in_prompt_rate") or {}
        retrieval.append(
            {
                "mode": mode,
                "recall": rec_stat.get("mean", 0.0),
                "recall_sd": rec_stat.get("standard_deviation", 0.0),
                "precision": prec_stat.get("mean", 0.0),
                "precision_sd": prec_stat.get("standard_deviation", 0.0),
                "evidence_in_prompt": ev_stat.get("mean", 0.0),
                "evidence_in_prompt_sd": ev_stat.get("standard_deviation", 0.0),
                "trials": summ.get("trials", 1),
            }
        )

        sav_stat = metrics.get("mean_token_savings") or {}
        red_stat = metrics.get("mean_context_reduction") or {}
        savings.append(
            {
                "mode": mode,
                "token_savings": sav_stat.get("mean", 0.0),
                "token_savings_sd": sav_stat.get("standard_deviation", 0.0),
                "context_reduction": red_stat.get("mean", 0.0),
                "context_reduction_sd": red_stat.get("standard_deviation", 0.0),
                "trials": summ.get("trials", 1),
            }
        )

        qae_stat = metrics.get("mean_quality_adjusted_efficiency") or {}
        qpt_stat = metrics.get("quality_per_token") or {}
        efficiency.append(
            {
                "mode": mode,
                "quality_adjusted_efficiency": qae_stat.get("mean", 0.0),
                "quality_adjusted_efficiency_sd": qae_stat.get("standard_deviation", 0.0),
                "quality_per_token": qpt_stat.get("mean", 0.0),
                "quality_per_token_sd": qpt_stat.get("standard_deviation", 0.0),
                "accuracy": acc_stat.get("mean", 0.0),
                "context_tokens": tok_stat.get("mean", 0.0),
                "trials": summ.get("trials", 1),
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


def plot_series(runs: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the six plan-required plot series from exported runs."""

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


# --------------------------------------------------------------------------- #
# Phase 17: failure-distribution comparison
# --------------------------------------------------------------------------- #


def total_variation_distance(
    left: Mapping[str, float], right: Mapping[str, float], vocabulary: Sequence[str]
) -> float | None:
    """Half the L1 distance between two distributions over ``vocabulary``.

    ``0.0`` means the two mixes are identical, ``1.0`` means they share no label.
    Returns ``None`` when either side is not a distribution (empty or negative
    mass), because a distance to nothing is not a small distance — it is no
    distance at all.

    Shares are re-normalized over ``vocabulary`` so a label present in one mode's
    report and absent from the other's is counted as ``0.0`` rather than ignored,
    which is the whole point of comparing over a fixed support.
    """

    keys = [str(label) for label in vocabulary]
    if not keys:
        return None
    left_total = sum(max(0.0, float(left.get(key, 0.0) or 0.0)) for key in keys)
    right_total = sum(max(0.0, float(right.get(key, 0.0) or 0.0)) for key in keys)
    if left_total <= 0.0 or right_total <= 0.0:
        return None
    distance = 0.0
    for key in keys:
        left_share = max(0.0, float(left.get(key, 0.0) or 0.0)) / left_total
        right_share = max(0.0, float(right.get(key, 0.0) or 0.0)) / right_total
        distance += abs(left_share - right_share)
    return round(distance / 2.0, 6)


def compare_error_distributions(
    report: Mapping[str, Any], *, baseline_mode: str = ""
) -> dict[str, Any]:
    """Compare how modes fail, rather than how often (plan Phase 12 → Phase 17).

    Phase 12's carried constraint for this phase is explicit: *"Phase 17 must run
    this report over generated answers and compare distributions, not
    accuracies."* Two modes can share an accuracy and fail in entirely different
    ways — one losing evidence at selection, another keeping a superseded fact in
    the prompt — and the accuracy column cannot tell them apart. This function is
    the comparison that can.

    It reads a Phase 12 :func:`evaluation.errors.error_report` artifact and never
    re-grades anything: label counts come from ``by_mode[*].label_counts``, volume
    from ``defect_count`` / ``graded_count`` / ``failure_rate``.

    The distance is total variation over the taxonomy's own fixed vocabulary, so
    it describes the *shape* of a mode's failures. It is deliberately **not** a
    significance test: with a handful of failure records a distance of 0.5 says
    "a different mix", not "a reliably different mix", and the interpretation
    string says so.
    """

    by_mode = report.get("by_mode") if isinstance(report.get("by_mode"), Mapping) else {}
    taxonomy = report.get("taxonomy") if isinstance(report.get("taxonomy"), Mapping) else {}
    declared = taxonomy.get("labels")
    seen = sorted(
        {
            str(label)
            for block in by_mode.values()
            if isinstance(block, Mapping)
            for label in (block.get("label_counts") or {})
        }
    )
    vocabulary = [str(label) for label in (declared or ())] or seen
    # A label the taxonomy declares but no mode produced still belongs to the
    # support: its absence is the finding, and dropping it would compare two
    # different vocabularies.
    vocabulary = sorted(set(vocabulary) | set(seen))

    modes: dict[str, Any] = {}
    for mode, block in sorted(by_mode.items()):
        if not isinstance(block, Mapping):
            continue
        counts = {
            str(label): int(value or 0)
            for label, value in (block.get("label_counts") or {}).items()
        }
        defects = int(block.get("defect_count", 0) or 0)
        total = sum(counts.get(label, 0) for label in vocabulary)
        modes[str(mode)] = {
            "mode_label": mode_label(str(mode)),
            "defect_count": defects,
            "graded_count": int(block.get("graded_count", 0) or 0),
            "failure_rate": block.get("failure_rate"),
            "observed_failure_count": int(block.get("observed_failure_count", 0) or 0),
            "latent_defect_count": int(block.get("latent_defect_count", 0) or 0),
            "evidence_lost_count": int(block.get("evidence_lost_count", 0) or 0),
            "label_counts": {label: counts.get(label, 0) for label in vocabulary},
            "shares": {
                label: round(counts.get(label, 0) / total, 6) if total else 0.0
                for label in vocabulary
            },
            "label_total": total,
            "top_label": (
                max(vocabulary, key=lambda label: (counts.get(label, 0), label))
                if total
                else None
            ),
        }

    resolved_baseline, selection = _resolve_baseline(baseline_mode, tuple(modes))
    pairs: list[dict[str, Any]] = []
    baseline_counts = (modes.get(resolved_baseline) or {}).get("label_counts") or {}
    # A baseline's distance to itself is exactly zero, not "unknown": rendering
    # it as a gap would invite the question the number already answers.
    if resolved_baseline in modes:
        modes[resolved_baseline]["total_variation_vs_baseline"] = 0.0
    for mode, block in modes.items():
        if mode == resolved_baseline:
            continue
        distance = total_variation_distance(baseline_counts, block["label_counts"], vocabulary)
        block["total_variation_vs_baseline"] = distance
        divergent = _largest_divergence(baseline_counts, block["label_counts"], vocabulary)
        pairs.append(
            {
                "mode": mode,
                "mode_label": block["mode_label"],
                "baseline_mode": resolved_baseline,
                "total_variation_distance": distance,
                "comparable": distance is not None,
                "largest_divergence": divergent,
                "note": (
                    ""
                    if distance is not None
                    else (
                        "No failure records on one side: a distribution over zero "
                        "records is not a distribution, so no distance is reported."
                    )
                ),
            }
        )
    for block in modes.values():
        block.setdefault("total_variation_vs_baseline", None)

    return {
        "vocabulary": vocabulary,
        "baseline_mode": resolved_baseline,
        "baseline_selection": selection,
        "metric": "total_variation_distance",
        "modes": modes,
        "pairs": pairs,
        "interpretation": (
            "Shares describe the *mix* of a mode's failure labels, not how often "
            "it fails; `defect_count` / `graded_count` / `failure_rate` carry the "
            "volume. Total variation is 0 for an identical mix and 1 for a "
            "disjoint one, over the taxonomy's fixed vocabulary. It is a "
            "description, not a significance test: on a smoke tier a large "
            "distance can be one record."
        ),
    }


def _resolve_baseline(requested: str, modes: Sequence[str]) -> tuple[str, str]:
    """Pick the distribution baseline, recording why."""

    wanted = str(requested or "").strip()
    if wanted:
        if wanted in modes:
            return wanted, "explicit"
        if modes:
            return modes[0], f"requested:{wanted} (absent, fell back to {modes[0]})"
        return wanted, f"requested:{wanted} (no modes reported)"
    if "full_context" in modes:
        return "full_context", "default:full_context"
    if modes:
        return modes[0], f"first:{modes[0]}"
    return "", "none"


def _largest_divergence(
    baseline: Mapping[str, int], other: Mapping[str, int], vocabulary: Sequence[str]
) -> dict[str, Any]:
    """Return the label whose share differs most from the baseline's."""

    baseline_total = sum(int(baseline.get(label, 0) or 0) for label in vocabulary)
    other_total = sum(int(other.get(label, 0) or 0) for label in vocabulary)
    if not baseline_total or not other_total:
        return {}
    best: dict[str, Any] = {}
    best_gap = -1.0
    for label in vocabulary:
        left = int(baseline.get(label, 0) or 0) / baseline_total
        right = int(other.get(label, 0) or 0) / other_total
        gap = abs(left - right)
        if gap > best_gap:
            best_gap = gap
            best = {
                "label": label,
                "baseline_share": round(left, 6),
                "mode_share": round(right, 6),
                "difference": round(right - left, 6),
            }
    # Two identical mixes have no "largest divergence": naming a label whose share
    # differs by exactly zero would read as a finding where there is none.
    return best if best_gap > 0.0 else {}


__all__ = [
    "HEADLINE_KEYS",
    "STATISTICAL_METRICS",
    "compare_error_distributions",
    "compare_modes_paired",
    "compare_runs",
    "extract_mode_result_items",
    "headline_from_aggregate",
    "length_curve_rows",
    "pairwise_comparisons",
    "plot_series",
    "statistical_analysis",
    "statistical_plot_series",
    "summarize_trial_modes",
    "total_variation_distance",
]
