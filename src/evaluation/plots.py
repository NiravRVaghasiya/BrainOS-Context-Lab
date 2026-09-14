"""Plotting extension points for benchmark reports and statistical evaluation.

Phase 8 produces the six plan-required series (accuracy vs length, tokens vs
length, accuracy vs tokens, retrieval precision/recall, token savings,
quality-adjusted efficiency). Rendering them needs matplotlib, which is an
optional extra; the series themselves are built in :mod:`evaluation.analysis`
and are tested without that dependency.

Phase 10 adds trial-level confidence intervals and error bars to the rendered plots.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .analysis import plot_series, statistical_plot_series

PLOT_NAMES: tuple[str, ...] = (
    "accuracy_vs_length",
    "tokens_vs_length",
    "accuracy_vs_tokens",
    "retrieval",
    "token_savings",
    "quality_adjusted_efficiency",
)


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Plotting requires the `evaluation` optional dependency."
        ) from exc
    return plt


def plot_accuracy_by_length(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Create Plot 1: Accuracy vs conversation length with error bars."""

    return _line_plot(
        rows,
        output,
        x="conversation_length",
        y="accuracy",
        xlabel="Conversation length",
        ylabel="Accuracy",
        title="Accuracy vs conversation length",
    )


def plot_tokens_by_length(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Create Plot 2: Context tokens vs conversation length with error bars."""

    return _line_plot(
        rows,
        output,
        x="conversation_length",
        y="context_tokens",
        xlabel="Conversation length",
        ylabel="Context tokens",
        title="Context tokens vs conversation length",
    )


def plot_accuracy_vs_tokens(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Create Plot 3: Accuracy vs context tokens."""

    return _line_plot(
        rows,
        output,
        x="context_tokens",
        y="accuracy",
        xlabel="Context tokens",
        ylabel="Accuracy",
        title="Accuracy vs context tokens",
    )


def plot_retrieval(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Create Plot 4: Grouped bars of recall, precision, and evidence-in-prompt per mode."""

    return _bar_plot(
        rows,
        output,
        values=("recall", "precision", "evidence_in_prompt"),
        ylabel="Rate",
        title="Retrieval precision / recall / evidence-in-prompt",
    )


def plot_token_savings(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Create Plot 5: Token savings vs full context."""

    return _bar_plot(
        rows,
        output,
        values=("token_savings",),
        ylabel="Token savings",
        title="Token savings vs full context",
    )


def plot_quality_adjusted_efficiency(
    rows: Iterable[dict[str, Any]], output: str | Path
) -> Path:
    """Create Plot 6: Quality-adjusted efficiency."""

    return _bar_plot(
        rows,
        output,
        values=("quality_adjusted_efficiency",),
        ylabel="Quality / tokens",
        title="Quality-adjusted efficiency",
    )


def plot_all(
    runs_or_series: Any,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Render every planned plot from exported runs or pre-built series into ``output_dir``."""

    if isinstance(runs_or_series, Mapping) and all(k in runs_or_series for k in PLOT_NAMES):
        series = dict(runs_or_series)
    else:
        # Check if source contains trial results or multi-trial structure
        try:
            series = statistical_plot_series(runs_or_series)
            if not any(series.values()):
                series = plot_series(runs_or_series)
        except Exception:
            series = plot_series(runs_or_series)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    writers = {
        "accuracy_vs_length": plot_accuracy_by_length,
        "tokens_vs_length": plot_tokens_by_length,
        "accuracy_vs_tokens": plot_accuracy_vs_tokens,
        "retrieval": plot_retrieval,
        "token_savings": plot_token_savings,
        "quality_adjusted_efficiency": plot_quality_adjusted_efficiency,
    }
    written: dict[str, Path] = {}
    for name, writer in writers.items():
        written[name] = writer(series[name], destination / f"{name}.png")
    return written


def _line_plot(
    rows: Iterable[dict[str, Any]],
    output: str | Path,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path:
    plt = _pyplot()
    rows = list(rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for mode in sorted({str(row.get("mode", "")) for row in rows}):
        selected = [row for row in rows if row.get("mode") == mode]
        selected.sort(key=lambda row: row.get(x, 0) or 0)
        x_vals = [row.get(x, 0) for row in selected]
        y_vals = [float(row.get(y, 0.0) or 0.0) for row in selected]
        y_err_key = f"{y}_sd"
        y_errs = [float(row.get(y_err_key, row.get("y_err", 0.0)) or 0.0) for row in selected]
        has_err = any(err > 0.0 for err in y_errs)
        if has_err:
            plt.errorbar(
                x_vals,
                y_vals,
                yerr=y_errs,
                capsize=3,
                marker="o",
                label=mode or "(unknown)",
            )
        else:
            plt.plot(
                x_vals,
                y_vals,
                marker="o",
                label=mode or "(unknown)",
            )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if any(row.get("mode") for row in rows):
        plt.legend()
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
    return destination


def _bar_plot(
    rows: Iterable[dict[str, Any]],
    output: str | Path,
    *,
    values: Sequence[str],
    ylabel: str,
    title: str,
) -> Path:
    plt = _pyplot()
    rows = list(rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    modes = [str(row.get("mode", "") or "(unknown)") for row in rows]
    plt.figure()
    if len(values) == 1:
        val_key = values[0]
        y_vals = [float(row.get(val_key, 0.0) or 0.0) for row in rows]
        y_err_key = f"{val_key}_sd"
        y_errs = [float(row.get(y_err_key, row.get("y_err", 0.0)) or 0.0) for row in rows]
        has_err = any(err > 0.0 for err in y_errs)
        if has_err:
            plt.bar(modes, y_vals, yerr=y_errs, capsize=3)
        else:
            plt.bar(modes, y_vals)
    else:
        width = 0.8 / max(len(values), 1)
        indexes = list(range(len(rows)))
        for offset, key in enumerate(values):
            shifted = [index + offset * width for index in indexes]
            y_vals = [float(row.get(key, 0.0) or 0.0) for row in rows]
            y_err_key = f"{key}_sd"
            y_errs = [float(row.get(y_err_key, 0.0) or 0.0) for row in rows]
            has_err = any(err > 0.0 for err in y_errs)
            if has_err:
                plt.bar(
                    shifted,
                    y_vals,
                    width=width,
                    yerr=y_errs,
                    capsize=3,
                    label=key,
                )
            else:
                plt.bar(
                    shifted,
                    y_vals,
                    width=width,
                    label=key,
                )
        tick_centers = [index + width * (len(values) - 1) / 2 for index in indexes]
        plt.xticks(tick_centers, modes)
        plt.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
    return destination


__all__ = [
    "PLOT_NAMES",
    "plot_accuracy_by_length",
    "plot_accuracy_vs_tokens",
    "plot_all",
    "plot_quality_adjusted_efficiency",
    "plot_retrieval",
    "plot_token_savings",
    "plot_tokens_by_length",
]
