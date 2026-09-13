"""CLI entry point for one benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import load_jsonl
from .runner import EvaluationConfig, EvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BrainOS Context Lab evaluation.")
    parser.add_argument("--model", default="", help="Provider model identifier.")
    parser.add_argument("--provider", default="", help="Provider name.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full_context", "sliding_window", "rag", "brainos", "brainos_rag"],
    )
    parser.add_argument("--benchmark", default="context_rot")
    parser.add_argument(
        "--dataset", type=Path, default=Path("benchmarks/context_rot/dataset.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = load_jsonl(args.dataset)
    config = EvaluationConfig(
        provider=args.provider,
        model=args.model,
        mode=args.mode,
        benchmark=args.benchmark,
    )
    try:
        run = EvaluationRunner(config).run(tasks)
    except NotImplementedError as exc:
        raise SystemExit(f"Evaluation scaffold: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
