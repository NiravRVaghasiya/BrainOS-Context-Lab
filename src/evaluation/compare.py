"""CLI entry point for comparing exported runs and generating statistical reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analysis import (
    HEADLINE_KEYS,
    compare_runs,
)
from .plots import plot_all
from .reports import comparison_report, statistical_report, write_json_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Context Lab evaluation runs.")
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Paths to run JSON or experiment JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write comparison/statistical report JSON",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute trial statistics and paired comparisons",
    )
    parser.add_argument(
        "--baseline-mode",
        default="full_context",
        help="Baseline mode for paired comparisons",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory to save the six evaluation plots",
    )
    args = parser.parse_args(argv)

    raw_records = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]

    # Normalize if an experiment artifact was supplied
    records: list[dict[str, Any]] = []
    is_experiment = False
    for rec in raw_records:
        if isinstance(rec, dict) and "experiment_version" in rec and "modes" in rec:
            is_experiment = True
            for mode_res in rec["modes"]:
                # Reconstruct run dict if to_run_dict format exists or use directly
                records.append(mode_res)
        else:
            records.append(rec)

    has_multiple_trials = False
    trial_counts = {int(r.get("trial", 0) or 0) for r in records if isinstance(r, dict)}
    if len(trial_counts) > 1 or max(trial_counts, default=0) > 0:
        has_multiple_trials = True

    include_stats = args.stats or is_experiment or has_multiple_trials

    if include_stats:
        stat_report = statistical_report(records, baseline_mode=args.baseline_mode)
        if args.output:
            write_json_report(stat_report, args.output)
        else:
            rendered = json.dumps(stat_report, indent=2) + "\n"
            print(rendered, end="")

        _print_statistical_summary(stat_report)
    else:
        comparison = compare_runs(records)
        report = comparison_report(records)
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            write_json_report(report, args.output)
        else:
            print(rendered, end="")
        _print_table(comparison)

    if args.plots_dir:
        written_plots = plot_all(records, args.plots_dir)
        print(f"\nRendered {len(written_plots)} plots to {args.plots_dir}")

    return 0


def _print_table(comparison: list[dict]) -> None:
    """Print a one-line-per-mode summary of the headline metrics."""

    if not comparison:
        return
    columns = ("mode",) + HEADLINE_KEYS[:8]
    widths = {column: len(column) for column in columns}
    rows: list[dict[str, str]] = []
    for item in comparison:
        headline = item.get("headline") or {}
        row = {"mode": str(item.get("mode") or "")}
        for key in HEADLINE_KEYS[:8]:
            value = headline.get(key)
            row[key] = _fmt(value)
            widths[key] = max(widths[key], len(row[key]))
        widths["mode"] = max(widths["mode"], len(row["mode"]))
        rows.append(row)
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def _print_statistical_summary(report: dict[str, Any]) -> None:
    """Print formatted trial-level statistics and paired comparisons."""

    summaries = report.get("trial_summaries", {})
    if summaries:
        print("\n=== Trial Statistics (Mean ± SD [95% CI]) ===")
        metrics_to_show = (
            "answer_accuracy",
            "faithfulness",
            "retrieval_recall",
            "evidence_in_prompt_rate",
            "mean_final_context_tokens",
            "mean_token_savings",
            "mean_quality_adjusted_efficiency",
        )
        for mode, data in summaries.items():
            trials = data.get("trials", 1)
            label = data.get("label", mode)
            print(f"\nMode: {mode} ({label}) — {trials} trial(s)")
            for met in metrics_to_show:
                stat = (data.get("metrics") or {}).get(met)
                if stat:
                    m = stat.get("mean", 0.0)
                    sd = stat.get("standard_deviation", 0.0)
                    ci = stat.get("confidence_interval_95", (m, m))
                    print(f"  {met:<32}: {m:.4f} ± {sd:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    paired = report.get("paired_comparisons", [])
    if paired:
        print("\n=== Paired Comparisons Across Benchmark Tasks ===")
        headers = [
            ("Pair", 28),
            ("Metric", 18),
            ("Diff (A-B)", 12),
            ("95% CI", 20),
            ("t-stat", 8),
            ("p-val", 8),
            ("Cohen's d", 10),
            ("Effect", 10),
            ("W/L/T", 8),
        ]
        hdr_line = " ".join(title.ljust(w) for title, w in headers)
        print(hdr_line)
        print("-" * len(hdr_line))
        for p in paired:
            pair_name = f"{p.get('mode_a')} vs {p.get('mode_b')}"
            met = str(p.get("metric", ""))
            diff = f"{p.get('mean_difference', 0.0):+.4f}"
            ci = p.get("confidence_interval_95", (0, 0))
            ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            t_str = f"{p.get('t_statistic', 0.0):.2f}"
            pval = p.get("p_value", 1.0)
            p_str = f"{pval:.4f}" if pval >= 0.0001 else "<0.0001"
            d_str = f"{p.get('cohens_d', 0.0):.2f}"
            mag = str(p.get("effect_size_magnitude", ""))
            wlt = f"{p.get('wins')}/{p.get('losses')}/{p.get('ties')}"
            row = (
                f"{pair_name:<28} {met:<18} {diff:<12} {ci_str:<20} "
                f"{t_str:<8} {p_str:<8} {d_str:<10} {mag:<10} {wlt:<8}"
            )
            print(row)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
