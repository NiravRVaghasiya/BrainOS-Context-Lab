"""Report generation for evaluation results.

Phase 8 writes a structured metrics report (the plan's quality / efficiency /
robustness numbers) without credentials or provider clients. Phase 10 adds
statistical evaluation reports with trial-level confidence intervals and
paired comparison effect sizes.

Phase 17 adds the half the plan's ``results/report/`` directory asks for: a
**human-readable** report rendered from the machine-readable pipeline artifact.
:func:`markdown_report` is a pure function of that artifact — it runs no
benchmark, calls no provider, and reads no dataset — so a report can be
re-rendered from an artifact written weeks earlier without re-deriving the
numbers it quotes. That is also what keeps the two honest with each other: the
markdown cannot say anything the JSON does not contain.

Three rules the renderer follows, all inherited from earlier phases:

* **unset is not zero.** An ungraded run has no accuracy, faithfulness, or
  quality-adjusted efficiency; a single length has no degradation curve. Those
  render as an em dash plus a sentence saying why, never as ``0.000``.
* **the scope travels with the numbers.** A dry run says it measured prompts and
  not answers; a smoke tier says it cannot support a claim.
* **no credential, ever.** The artifact the renderer reads is already
  credential-free (Phase 13/16), and the renderer adds only what it was given.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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

#: Headline keys that describe *answers*, so they exist only when something was
#: graded. An ungraded run reports them as unset; printing ``0.000`` would read
#: as "every answer was wrong" when nothing was ever asked of a model.
ANSWER_SIDE_HEADLINE_KEYS: frozenset[str] = frozenset(
    {"accuracy", "faithfulness", "conflict_resolution", "abstention"}
)

#: The same distinction in the aggregate-metric vocabulary (statistical tables)
#: and in the paired-comparison vocabulary (Phase 10's metric names).
ANSWER_SIDE_METRICS: frozenset[str] = frozenset(
    {"answer_accuracy", "faithfulness", "mean_quality_adjusted_efficiency"}
)
ANSWER_SIDE_PAIRED_METRICS: frozenset[str] = frozenset(
    {"accuracy", "faithfulness", "quality_adjusted_efficiency"}
)

#: Columns of the report's headline table, in the order the plan's final
#: evaluation matrix asks the questions (quality first, then efficiency).
REPORT_HEADLINE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("mode_label", "Mode"),
    ("tasks", "Tasks"),
    ("graded", "Graded"),
    ("accuracy", "Accuracy"),
    ("faithfulness", "Faithful"),
    ("recall", "Recall@K"),
    ("evidence_in_prompt", "Evidence"),
    ("conflict_resolution", "Conflict"),
    ("abstention", "Abstain"),
    ("mean_context_tokens", "Tokens"),
    ("mean_reduction", "Reduction"),
    ("mean_latency_ms", "Latency ms"),
)


def write_json_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write a structured report without credentials or provider clients.

    ``default=str`` is load-bearing: a payload that reaches this point has
    already been produced by a run that cost time and possibly money, and losing
    it because one value is a :class:`~pathlib.Path` (or a datetime, or an enum)
    rather than a ``str`` would be the worst possible moment to raise. Stringify
    and write.
    """

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    destination.write_text(rendered, encoding="utf-8")
    return destination


def write_markdown_report(text: str, output: str | Path) -> Path:
    """Write the human-readable report (the plan's ``results/report/``)."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = text if text.endswith("\n") else text + "\n"
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


# --------------------------------------------------------------------------- #
# Phase 17: the human-readable report
# --------------------------------------------------------------------------- #


def markdown_report(artifact: Mapping[str, Any]) -> str:
    """Render the pipeline artifact as the plan's ``results/report/report.md``.

    The renderer is deliberately dumb: it formats what the artifact says and
    adds no analysis of its own, so re-rendering an old artifact reproduces the
    report that artifact deserved rather than today's interpretation of it.
    """

    sections = [
        _report_header(artifact),
        _report_scope(artifact),
        _report_identity(artifact),
        _report_headline(artifact),
        _report_statistics(artifact),
        _report_failures(artifact),
        _report_robustness(artifact),
        _report_cost(artifact),
        _report_controls(artifact),
        _report_security(artifact),
        _report_stages(artifact),
        _report_reproducibility(artifact),
    ]
    return "\n\n".join(section for section in sections if section.strip()) + "\n"


def _report_header(artifact: Mapping[str, Any]) -> str:
    preset = str(artifact.get("preset") or artifact.get("plan", {}).get("name") or "")
    dry_run = bool(artifact.get("dry_run"))
    kind = "retrieval-only dry run" if dry_run else "generated run"
    lines = [
        "# BrainOS Context Lab — evaluation report",
        "",
        f"Run `{artifact.get('run_id', '')}` · preset **{preset or 'custom'}** · {kind}",
        f"Produced {artifact.get('timestamp', '')} by the automated pipeline "
        f"(`{artifact.get('pipeline_version', '')}`).",
    ]
    return "\n".join(lines)


def _report_scope(artifact: Mapping[str, Any]) -> str:
    """What this report can and cannot support — before any number."""

    dry_run = bool(artifact.get("dry_run"))
    headline = list(artifact.get("headline") or ())
    graded = sum(int(row.get("graded", 0) or 0) for row in headline)
    lengths = _length_tiers(artifact)
    trials = int((artifact.get("plan") or {}).get("trials", 1) or 1)
    aborted = artifact.get("aborted")
    violations = list(artifact.get("violations") or ())

    lines = ["## What this report can claim", ""]
    if dry_run:
        lines.append(
            "- **No model was called.** This is a dry run: it measures what each "
            "mode put in a prompt and what that cost. Accuracy, faithfulness, and "
            "quality-adjusted efficiency are unset because nothing was graded."
        )
    elif graded:
        lines.append(
            f"- {graded} answer(s) were graded, so the quality columns are "
            "measurements of this model on this dataset."
        )
    else:
        lines.append(
            "- A model was called but **no answer was graded**, so the quality "
            "columns stay unset rather than reading as zero."
        )
    if len(lengths) <= 1:
        lines.append(
            "- **One length tier only.** There is no accuracy-vs-length curve and "
            "no degradation AUC; those fields are `null`, not zero."
        )
    else:
        lines.append(
            f"- {len(lengths)} length tiers ({', '.join(str(tier) for tier in lengths)}), "
            "so the degradation curve and its AUC are defined."
        )
    if trials <= 1:
        lines.append(
            "- **One trial per (task, mode).** Confidence intervals are degenerate "
            "and no repeated-trial variance is reported."
        )
    if violations:
        lines.append(
            f"- **The controlled-comparison checks did not hold** "
            f"({len(violations)} violation(s)). The mode comparison below is not "
            "a controlled comparison and must not be quoted as one."
        )
    if aborted:
        lines.append(
            "- **The run stopped at a cost ceiling** "
            f"(`{_mapping(aborted).get('limit', '')}`): the numbers cover the tasks "
            "that completed, not the whole dataset."
        )
    # The pipeline records which stages were *selected*; ``stages`` holds the rows
    # that completed, which while the report is rendering excludes the report
    # itself. Reading the rows alone would call every complete run partial.
    selected = [str(name) for name in artifact.get("stages_selected") or ()]
    rows = [str(_mapping(stage).get("name", "")) for stage in artifact.get("stages") or ()]
    ran = selected or sorted({*rows, "report"}, key=rows.index if rows else str)
    partial = artifact.get("partial")
    if partial is None:
        partial = bool(ran) and len(set(ran) | {"report"}) < 7
    if partial:
        lines.append(
            f"- **Partial pipeline.** Only these stages ran: {', '.join(ran)}. A "
            "section whose stage did not run is absent from this report — that is "
            "a stage selection, not an empty result."
        )
    seeds = _mapping(artifact.get("dataset")).get("seeds") or ()
    seed_note = (
        f"{len(seeds)} generator seed(s)" if seeds else "no generator seed recorded"
    )
    lines.append(
        f"- Synthetic benchmark, {seed_note}. These are measurements of this "
        "repository's pipeline on this dataset, not a claim about BrainOS in "
        "general (plan §20: let the benchmark decide the result)."
    )
    return "\n".join(lines)


def _report_identity(artifact: Mapping[str, Any]) -> str:
    repro = _mapping(artifact.get("repro"))
    dataset = _mapping(artifact.get("dataset"))
    plan = _mapping(artifact.get("plan"))
    model = artifact.get("model")
    generation = _mapping(artifact.get("generation"))
    modes = list(plan.get("modes") or [])
    labels = list(plan.get("mode_labels") or modes)
    rows = [
        ("Application version", str(repro.get("application_version") or "—")),
        ("BrainOS revision", str(repro.get("brainos_version") or "—")),
        ("Benchmark revision", str(repro.get("benchmark_version") or "—")),
        ("Dataset", f"`{dataset.get('path') or '—'}`"),
        ("Dataset SHA-256", _digest_or_dash(dataset.get("sha256"))),
        (
            "Tasks",
            f"{artifact.get('tasks_executed', '—')} of "
            f"{artifact.get('tasks_available', '—')}",
        ),
        ("Length tiers", ", ".join(str(tier) for tier in _length_tiers(artifact)) or "—"),
        ("Modes", ", ".join(f"`{label}`" for label in labels) or "—"),
        ("Trials", str(plan.get("trials") or "—")),
        ("Session isolation", "yes" if plan.get("session_isolation") else "no"),
        ("Model", _model_line(model)),
        ("Credential source", _credential_source(artifact, generation)),
        ("Token counter", str(_mapping(artifact.get("budget")).get("token_counter") or "—")),
    ]
    return "\n".join(
        ["## Run identity", "", _markdown_table(("Field", "Value"), rows)]
    )


def _report_headline(artifact: Mapping[str, Any]) -> str:
    rows = list(artifact.get("headline") or ())
    lines = ["## Headline metrics", ""]
    if not rows:
        lines.append("_No mode results were produced._")
        return "\n".join(lines)
    table: list[tuple[str, ...]] = []
    for row in rows:
        cells = []
        ungraded = not int(row.get("graded", 0) or 0)
        for key, _ in REPORT_HEADLINE_COLUMNS:
            value = row.get(key)
            if ungraded and key in ANSWER_SIDE_HEADLINE_KEYS:
                cells.append("—")
            elif key == "mode_label":
                cells.append(str(value or row.get("mode") or ""))
            elif key in ("tasks", "graded"):
                cells.append(str(int(value or 0)))
            elif key == "mean_context_tokens":
                cells.append(_fmt(value, digits=1))
            elif key == "mean_reduction":
                cells.append(_percent(value))
            else:
                cells.append(_fmt(value))
        trial = row.get("trial")
        label = cells[0]
        if trial:
            label = f"{label} (t{trial})"
        table.append((label, *cells[1:]))
    lines.append(
        _markdown_table(tuple(label for _, label in REPORT_HEADLINE_COLUMNS), table)
    )
    graded = sum(int(row.get("graded", 0) or 0) for row in rows)
    if not graded:
        lines.append("")
        lines.append(
            "_No answers were graded: accuracy, faithfulness, and QAE are unset, "
            "not zero. Recall@K and Evidence are retrieval measurements and are "
            "real numbers either way._"
        )
    return "\n".join(lines)


def _report_statistics(artifact: Mapping[str, Any]) -> str:
    statistics = _mapping(artifact.get("statistics"))
    summaries = _mapping(statistics.get("trial_summaries"))
    paired = list(statistics.get("paired_comparisons") or ())
    if not summaries and not paired:
        return ""
    graded_by_mode = _graded_by_mode(artifact)
    total_graded = sum(graded_by_mode.values())
    lines = ["## Statistical evaluation", ""]
    if summaries:
        rows: list[tuple[str, ...]] = []
        for mode, block in sorted(summaries.items()):
            metrics = _mapping(_mapping(block).get("metrics"))
            trials = int(_mapping(block).get("trials", 1) or 1)
            ungraded = not graded_by_mode.get(str(mode), 0)
            for key in (
                "answer_accuracy",
                "faithfulness",
                "retrieval_recall",
                "mean_final_context_tokens",
                "mean_quality_adjusted_efficiency",
            ):
                stat = _mapping(metrics.get(key))
                if not stat or (ungraded and key in ANSWER_SIDE_METRICS):
                    continue
                interval = stat.get("confidence_interval_95") or (None, None)
                rows.append(
                    (
                        str(_mapping(block).get("label") or mode),
                        key,
                        str(trials),
                        _fmt(stat.get("mean"), digits=4),
                        _fmt(stat.get("standard_deviation"), digits=4),
                        f"[{_fmt(interval[0], digits=4)}, {_fmt(interval[1], digits=4)}]",
                    )
                )
        if rows:
            lines.append(
                _markdown_table(
                    ("Mode", "Metric", "Trials", "Mean", "SD", "95% CI"), rows
                )
            )
            lines.append("")
        if any(not count for count in graded_by_mode.values()):
            lines.append(
                "_Answer-side metrics (accuracy, faithfulness, QAE) are omitted "
                "for modes that graded no answer: a mean over zero graded answers "
                "is unset, not 0.0000._"
            )
            lines.append("")
    if paired:
        reportable = [
            comparison
            for comparison in paired
            if total_graded
            or str(_mapping(comparison).get("metric", "")) not in ANSWER_SIDE_PAIRED_METRICS
        ]
        suppressed = len(paired) - len(reportable)
        lines.append(f"{len(reportable)} paired comparison(s) across benchmark tasks:")
        lines.append("")
        rows = []
        for comparison in reportable[:24]:
            block = _mapping(comparison)
            interval = block.get("confidence_interval_95") or (None, None)
            rows.append(
                (
                    f"{block.get('mode_a', '')} vs {block.get('mode_b', '')}",
                    str(block.get("metric", "")),
                    _fmt(block.get("mean_difference"), digits=4, signed=True),
                    f"[{_fmt(interval[0], digits=4, signed=True)}, "
                    f"{_fmt(interval[1], digits=4, signed=True)}]",
                    _fmt(block.get("p_value"), digits=4),
                    _fmt(block.get("cohens_d"), digits=2),
                    str(block.get("effect_size_magnitude") or "—"),
                    f"{block.get('wins', 0)}/{block.get('losses', 0)}/{block.get('ties', 0)}",
                )
            )
        lines.append(
            _markdown_table(
                ("Pair", "Metric", "Diff", "95% CI", "p", "Cohen's d", "Effect", "W/L/T"),
                rows,
            )
        )
        if len(reportable) > 24:
            lines.append("")
            lines.append(f"_… {len(reportable) - 24} more in `aggregated/statistics.json`._")
        if suppressed:
            lines.append("")
            lines.append(
                f"_{suppressed} answer-side comparison(s) are omitted because no "
                "answer was graded; they are still in `aggregated/statistics.json` "
                "as zeros, which is why this report does not print them._"
            )
    return "\n".join(lines)


def _graded_by_mode(artifact: Mapping[str, Any]) -> dict[str, int]:
    """Graded-answer counts per mode, from the headline rows (0 when absent)."""

    counts: dict[str, int] = {}
    for row in artifact.get("headline") or ():
        block = _mapping(row)
        mode = str(block.get("mode") or "")
        if mode:
            counts[mode] = counts.get(mode, 0) + int(block.get("graded", 0) or 0)
    return counts


def _report_failures(artifact: Mapping[str, Any]) -> str:
    errors = _mapping(artifact.get("errors"))
    if not errors:
        return ""
    distributions = _mapping(errors.get("distribution_comparison"))
    modes = _mapping(distributions.get("modes"))
    lines = [
        "## Failure analysis",
        "",
        f"{errors.get('failure_record_count', 0)} failure record(s) over "
        f"{errors.get('scored_record_count', 0)} scored replay(s); "
        f"{errors.get('observed_failure_count', 0)} observed, "
        f"{errors.get('latent_defect_count', 0)} latent (defect in the prompt, "
        "answer still correct).",
    ]
    scorer = _mapping(errors.get("labels_vs_scorer"))
    lines.append(
        f"Taxonomy vs scorer: agree={scorer.get('agree', 0)} "
        f"refined={scorer.get('refined', 0)} **unexpected={scorer.get('unexpected', 0)}** "
        "(must stay zero — the taxonomy never re-grades)."
    )
    if modes:
        declared = [str(label) for label in distributions.get("vocabulary") or ()]
        # A nine-column taxonomy makes an unreadable table when one tier produced
        # two labels. Show the labels something actually produced, and say which
        # were never seen — an absent label is a finding, not a formatting choice.
        vocabulary = [
            label
            for label in declared
            if any(
                int(_mapping(_mapping(block).get("label_counts")).get(label, 0) or 0)
                for block in modes.values()
            )
        ] or declared
        silent = [label for label in declared if label not in vocabulary]
        rows: list[tuple[str, ...]] = []
        for mode, block in sorted(modes.items()):
            counts = _mapping(_mapping(block).get("label_counts"))
            shares = _mapping(_mapping(block).get("shares"))
            rows.append(
                (
                    str(_mapping(block).get("mode_label") or mode),
                    str(_mapping(block).get("defect_count", 0)),
                    _fmt(_mapping(block).get("failure_rate")),
                    str(_mapping(block).get("top_label") or "—"),
                    _fmt(_mapping(block).get("total_variation_vs_baseline")),
                    *(_label_cell(counts, shares, label) for label in vocabulary),
                )
            )
        lines.append("")
        # The effective baseline is not always the one the run asked for: a mode
        # with no failure records at all cannot anchor a distribution. Phase 12
        # records why in ``baseline_selection``; the report has to say it too,
        # because every distance in this table is relative to that choice.
        selection = str(distributions.get("baseline_selection") or "")
        requested = ""
        if selection.startswith("requested:"):
            requested = selection.split("(", 1)[0][len("requested:") :].strip()
        baseline_note = ""
        if requested and requested != str(distributions.get("baseline_mode", "")):
            baseline_note = (
                f" The run asked for `{requested}`, which reported no failure "
                f"records, so the distance is against "
                f"`{distributions.get('baseline_mode', '')}` instead — a mode that "
                "failed nothing cannot anchor a failure distribution."
            )
        elif selection and selection not in {"explicit", "default:full_context"}:
            baseline_note = f" (baseline selection: `{selection}`)"
        lines.append(
            "Failure mix per mode (counts and share of that mode's labels). "
            f"Distance is total variation against "
            f"`{distributions.get('baseline_mode', '')}`:{baseline_note}"
        )
        lines.append("")
        lines.append(
            _markdown_table(
                (
                    "Mode",
                    "Defects",
                    "Failure rate",
                    "Top label",
                    "TV distance",
                    *(str(label) for label in vocabulary),
                ),
                rows,
            )
        )
        if silent:
            lines.append("")
            lines.append(
                "_Labels never produced on this run (columns omitted): "
                + ", ".join(f"`{label}`" for label in silent)
                + "._"
            )
        if all(_mapping(block).get("failure_rate") is None for block in modes.values()):
            lines.append("")
            lines.append(
                "_Failure rates are unset because no answer was graded; every "
                "defect above is a **latent** one — a defect in the prompt that "
                "the (ungraded) answer did not contradict._"
            )
        lines.append("")
        lines.append(f"_{distributions.get('interpretation', '')}_")
    pairs = list(distributions.get("pairs") or ())
    notable = [
        pair
        for pair in pairs
        if _mapping(pair).get("comparable") and _mapping(pair).get("largest_divergence")
    ]
    if notable:
        lines.append("")
        lines.append("Where the mixes diverge most from the baseline:")
        lines.append("")
        for pair in notable:
            block = _mapping(pair)
            divergence = _mapping(block.get("largest_divergence"))
            lines.append(
                f"- `{block.get('mode', '')}` vs `{block.get('baseline_mode', '')}`: "
                f"TV {_fmt(block.get('total_variation_distance'))} — largest gap on "
                f"`{divergence.get('label', '')}` "
                f"({_percent(divergence.get('baseline_share'))} baseline → "
                f"{_percent(divergence.get('mode_share'))} mode)."
            )
    concentration = list(errors.get("concentration") or [])[:8]
    if concentration:
        lines.append("")
        lines.append("Where failures concentrate:")
        lines.append("")
        rows = [
            (
                str(_mapping(row).get("mode", "")),
                str(_mapping(row).get("category", "")),
                str(_mapping(row).get("label", "") or _mapping(row).get("failure_type", "")),
                str(_mapping(row).get("defect_count", "") or _mapping(row).get("count", "")),
                str(_mapping(row).get("stage", "")),
            )
            for row in concentration
        ]
        lines.append(_markdown_table(("Mode", "Category", "Label", "n", "Stage"), rows))
    return "\n".join(lines)


def _report_robustness(artifact: Mapping[str, Any]) -> str:
    comparison = _mapping(artifact.get("comparison"))
    runs = list(comparison.get("runs") or [])
    if not runs:
        return ""
    lines = ["## Context-length robustness", ""]
    rows: list[tuple[str, ...]] = []
    for run in runs:
        block = _mapping(run)
        degradation = _mapping(block.get("degradation"))
        points = [
            _mapping(point) for point in (degradation.get("by_length") or ())
        ]
        longest = max((point.get("length", 0) or 0 for point in points), default=None)
        rows.append(
            (
                str(block.get("mode") or ""),
                _fmt(degradation.get("point_count")),
                _fmt(degradation.get("reference_length")),
                _fmt(degradation.get("reference_accuracy")),
                _fmt(longest),
                _fmt(degradation.get("mean_degradation")),
                _fmt(degradation.get("area_under_degradation_curve")),
            )
        )
    lines.append(
        _markdown_table(
            (
                "Mode",
                "Lengths",
                "Reference length",
                "Reference accuracy",
                "Longest",
                "Mean degradation",
                "Degradation AUC",
            ),
            rows,
        )
    )
    note = next(
        (
            str(_mapping(run.get("degradation")).get("note"))
            for run in runs
            if _mapping(run.get("degradation")).get("note")
        ),
        "",
    )
    if note:
        lines.append("")
        lines.append(f"_{note}_")
    lengths = _length_tiers(artifact)
    if len(lengths) <= 1 and not note:
        lines.append("")
        lines.append(
            "_`null` above means \"one length tier cannot support a curve\", not "
            "\"no degradation\". Generate a longer tier "
            "(`python benchmarks/context_rot/generation.py --tier standard`) and "
            "re-run to get one._"
        )
    return "\n".join(lines)


def _report_cost(artifact: Mapping[str, Any]) -> str:
    budget = _mapping(artifact.get("budget"))
    limits = _mapping(budget.get("limits"))
    estimate = _mapping(artifact.get("cost_estimate"))
    if not budget:
        return ""
    lines = [
        "## Cost",
        "",
        str(artifact.get("billing_notice") or ""),
        "",
        _markdown_table(
            ("Ceiling", "Limit", "Used", "Remaining"),
            [
                (
                    "Requests",
                    limits.get("max_requests"),
                    budget.get("requests"),
                    budget.get("remaining_requests"),
                ),
                (
                    "Total tokens",
                    limits.get("max_total_tokens"),
                    budget.get("total_tokens"),
                    budget.get("remaining_total_tokens"),
                ),
                (
                    "Generation failures",
                    limits.get("max_generation_failures"),
                    budget.get("failures"),
                    budget.get("remaining_failures"),
                ),
                (
                    "Tasks",
                    limits.get("max_tasks"),
                    artifact.get("tasks_executed"),
                    _remaining(limits.get("max_tasks"), artifact.get("tasks_executed")),
                ),
            ],
        ),
        "",
        f"Input tokens {budget.get('input_tokens', 0)} · output tokens "
        f"{budget.get('output_tokens', 0)} · skipped requests {budget.get('skips', 0)} "
        f"(a prompt larger than the per-request ceiling is skipped, not sent) · "
        f"within limits: **{budget.get('within_limits', True)}**.",
    ]
    if estimate:
        lines.append(
            f"Planned requests {estimate.get('planned_requests', '—')} · output-token "
            f"ceiling {estimate.get('output_token_ceiling', '—')} · input tokens "
            f"{estimate.get('input_tokens', '—')}."
        )
    return "\n".join(lines)


def _report_controls(artifact: Mapping[str, Any]) -> str:
    constants = _mapping(artifact.get("constants"))
    violations = list(artifact.get("violations") or ())
    warnings = list(artifact.get("warnings") or [])
    issues = list(artifact.get("dataset_issues") or [])
    lines = ["## Controlled-comparison checks", ""]
    if constants:
        rows = [
            ("Passed", "yes" if constants.get("passed") else "no"),
            ("Checks run", _unique_list(constants.get("checked"))),
            ("Task count identical across modes", str(constants.get("task_count", "—"))),
            ("Dataset SHA-256", _digest_or_dash(constants.get("dataset_sha256"))),
            ("Trials", str(constants.get("trials", "—"))),
            ("System prompt hash(es)", _unique_list(constants.get("system_prompt_sha256"))),
            ("Sampling fingerprint(s)", _unique_list(constants.get("generation_fingerprints"))),
            ("Provider-reported model(s)", _unique_list(constants.get("reported_models"))),
            ("Token counter(s)", _unique_list(constants.get("token_counters"))),
            ("Generated requests", str(constants.get("generated_requests", 0))),
        ]
        lines.append(_markdown_table(("Check", "Value"), rows))
        lines.append("")
    lines.append(
        f"**{len(violations)} violation(s)** — "
        + (
            "the only thing that changed between modes was the context-management "
            "strategy."
            if not violations
            else "these numbers are **not** a controlled comparison and must not be "
            "quoted as one."
        )
    )
    for violation in violations:
        lines.append(f"- violation: {violation}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
    if issues:
        lines.append("")
        lines.append("Dataset issues:")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
    return "\n".join(lines)


def _report_security(artifact: Mapping[str, Any]) -> str:
    security = _mapping(artifact.get("security"))
    check = _mapping(artifact.get("credential_check"))
    if not security and not check:
        return ""
    lines = ["## Security", ""]
    if security:
        totals = _mapping(security.get("totals"))
        summary = ", ".join(f"{name}={count}" for name, count in sorted(totals.items()))
        lines.append(
            f"Prompt guard: records={security.get('record_count', 0)} "
            f"with findings={security.get('records_with_findings', 0)} "
            f"quarantined={security.get('quarantined', 0)} "
            f"credentials redacted={security.get('credentials_redacted', 0)}"
            + (f" — {summary}" if summary else " — no guard fired")
            + "."
        )
        lines.append(
            "A clean guard block means nothing matched the detector's patterns; "
            "the structural controls (delimiters, the untrusted-data preamble, "
            "role containment) are what do not depend on recognising an attack."
        )
    if check:
        lines.append(
            f"Artifact credential scan: {check.get('files_scanned', 0)} file(s), "
            f"{check.get('finding_count', 0)} finding(s), clean={check.get('clean', False)}."
        )
    return "\n".join(lines)


def _report_stages(artifact: Mapping[str, Any]) -> str:
    stages = list(artifact.get("stages") or ())
    layout = _mapping(artifact.get("layout"))
    lines = ["## Pipeline stages", ""]
    if layout:
        directories = ", ".join(
            f"`{layout.get(name, '')}`" for name in ("raw", "aggregated", "plots", "report")
        )
        lines.append(f"Output layout (the plan's §23 directories): {directories}")
        lines.append("")
    rows: list[tuple[str, ...]] = []
    for stage in stages:
        block = _mapping(stage)
        artifacts = block.get("artifacts") or ()
        digests = _mapping(block.get("digests"))
        rendered = ", ".join(
            f"`{Path(str(item)).name}` ({_short_digest(digests.get(str(item), ''))})"
            for item in artifacts
        )
        rows.append(
            (
                str(block.get("name", "")),
                str(block.get("status", "")),
                f"{float(block.get('duration_seconds', 0.0) or 0.0):.2f}",
                rendered or "—",
                str(block.get("detail") or block.get("error") or ""),
            )
        )
    lines.append(
        _markdown_table(("Stage", "Status", "Seconds", "Artifacts", "Detail"), rows)
    )
    if not any(str(_mapping(stage).get("name")) == "report" for stage in stages):
        lines.append("")
        lines.append(
            "_The `report` stage renders this document, so its own row is recorded "
            "in `report/pipeline.json` rather than guessed here._"
        )
    return "\n".join(lines)


def _report_reproducibility(artifact: Mapping[str, Any]) -> str:
    repro = _mapping(artifact.get("repro"))
    lines = ["## Reproducibility", ""]
    if repro:
        lines.append(
            f"Manifest `{repro.get('repro_version', '')}` · run `{repro.get('run_id', '')}` · "
            f"{repro.get('task_count', 0)} task id(s) recorded."
        )
        lines.append("")
    pipeline_command = str(artifact.get("pipeline_rerun_command") or "")
    single_command = str(repro.get("rerun_command") or "")
    if pipeline_command:
        lines.append("Re-run this whole pipeline:")
        lines.append("")
        lines.append("```bash")
        lines.append(pipeline_command)
        lines.append("```")
    if single_command:
        lines.append("")
        lines.append("Re-derive one mode's raw run file:")
        lines.append("")
        lines.append("```bash")
        lines.append(single_command)
        lines.append("```")
    lines.append("")
    lines.append(
        "Every artifact carries the versions, dataset digest, task ids, and limits "
        "that produced it; no artifact contains a credential."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured markdown table with aligned columns."""

    headers = [str(column) for column in columns]
    rendered_rows = [[_cell(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for index, value in enumerate(row[: len(headers)]):
            widths[index] = max(widths[index], len(value))
    header_cells = [header.ljust(widths[index]) for index, header in enumerate(headers)]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in rendered_rows:
        padded = list(row[: len(headers)])
        padded.extend([""] * (len(headers) - len(padded)))
        body_cells = [value.ljust(widths[index]) for index, value in enumerate(padded)]
        lines.append("| " + " | ".join(body_cells) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _unique_list(value: Any) -> str:
    """Render a controlled-comparison constant as its distinct values.

    One distinct value is the point of the check: a fingerprint list with two
    entries means two different sampling settings reached the provider, which is
    exactly the violation the table is there to show.
    """

    if not isinstance(value, (list, tuple)):
        return "—" if value is None else str(value)
    seen: list[str] = []
    for item in value:
        text = str(item)
        if text and text not in seen:
            seen.append(text)
    if not seen:
        return "none"
    rendered = ", ".join(f"`{item}`" for item in seen[:4])
    if len(seen) > 4:
        rendered += f" … ({len(seen)} distinct)"
    return rendered


def _model_line(model: Any) -> str:
    if not isinstance(model, Mapping) or not model.get("model"):
        return "none (dry run — no provider was configured)"
    provider = str(model.get("provider") or "")
    name = str(model.get("model") or "")
    temperature = model.get("temperature")
    fingerprint = str(model.get("settings_fingerprint") or "")
    line = f"`{provider}/{name}` at temperature {_fmt(temperature, digits=2)}"
    if fingerprint:
        line += f" (settings fingerprint `{fingerprint}`)"
    return line


def _length_tiers(artifact: Mapping[str, Any]) -> list[Any]:
    dataset = _mapping(artifact.get("dataset"))
    tiers = dataset.get("length_tiers")
    if isinstance(tiers, (list, tuple)):
        return list(tiers)
    return []


def _digest_or_dash(value: Any) -> str:
    text = str(value or "")
    return f"`{text[:16]}…`" if text else "—"


def _short_digest(value: Any) -> str:
    text = str(value or "")
    return text[:8] if text else "no digest"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fmt(value: Any, *, digits: int = 3, signed: bool = False) -> str:
    """Render a missing metric as an em dash; ``None`` is never zero."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return str(value)


def _credential_source(artifact: Mapping[str, Any], generation: Mapping[str, Any]) -> str:
    """Name the credential route the run used, or why there was none."""

    source = str(generation.get("credential_source") or "")
    if source:
        return source
    return "none (dry run)" if artifact.get("dry_run") else "—"


def _label_cell(
    counts: Mapping[str, Any], shares: Mapping[str, Any], label: str
) -> str:
    """``n (share%)`` for one taxonomy label in one mode's failure mix."""

    count = int(counts.get(label, 0) or 0)
    share = float(shares.get(label, 0.0) or 0.0)
    return f"{count} ({share:.0%})"


def _remaining(limit: Any, used: Any) -> Any:
    """Remaining allowance, or ``None`` when either side is not a number."""

    try:
        if limit is None or used is None:
            return None
        return max(0, int(limit) - int(used))
    except (TypeError, ValueError):
        return None


def _percent(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "REPORT_HEADLINE_COLUMNS",
    "comparison_report",
    "markdown_report",
    "metrics_report",
    "statistical_report",
    "write_json_report",
    "write_markdown_report",
]
