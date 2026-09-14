"""CLI entry point for comparing exported runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import HEADLINE_KEYS, compare_runs
from .reports import comparison_report, write_json_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Context Lab evaluation runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    comparison = compare_runs(records)
    report = comparison_report(records)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        write_json_report(report, args.output)
    else:
        print(rendered, end="")
    _print_table(comparison)
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


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
