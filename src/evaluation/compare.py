"""CLI entry point for comparing exported runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import compare_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Context Lab evaluation runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    comparison = compare_runs(records)
    rendered = json.dumps(comparison, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
