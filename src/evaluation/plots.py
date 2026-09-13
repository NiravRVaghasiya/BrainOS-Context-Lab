"""Plotting extension points for benchmark reports."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def plot_accuracy_by_length(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Create the first planned plot once plotting dependencies are installed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Plotting requires the `evaluation` optional dependency.") from exc

    rows = list(rows)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for mode in sorted({str(row.get("mode", "")) for row in rows}):
        selected = [row for row in rows if row.get("mode") == mode]
        plt.plot(
            [row.get("conversation_length", 0) for row in selected],
            [row.get("accuracy", 0.0) for row in selected],
            marker="o",
            label=mode,
        )
    plt.xlabel("Conversation length")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs conversation length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()
    return destination
