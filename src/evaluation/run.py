"""CLI entry point for one benchmark run.

Phase 6 made this command execute a task through any baseline mode. Phase 7 adds
the other half of a run:

* the dataset is validated before anything executes, and any structural problem
  is recorded in the run instead of being silently averaged into a metric;
* ``--answers`` grades a supplied answer per task, so answer quality can be
  scored from a model run (or a deterministic mock) without this process holding
  a provider key;
* the run always reports retrieval and context aggregates, and reports answer
  metrics only for the answers it was given.

The dataset's own revision is recorded as a content hash, which is what makes a
result reproducible: the same hash, the same generator seed, and the same mode
describe the same experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .datasets import load_jsonl
from .modes import task_evaluator
from .runner import EvaluationConfig, EvaluationRunner
from .scoring import AnswerSet, score_record

DEFAULT_DATASET = Path("benchmarks/context_rot/dataset.jsonl")


def dataset_sha256(path: Path) -> str:
    """Return the content hash of a dataset file (or ``""`` when absent)."""

    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_answers(path: Path | None) -> AnswerSet:
    """Load a JSONL answer file keyed by ``task_id`` (optionally ``mode``)."""

    if path is None:
        return AnswerSet()
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid answer record at line {line_number}") from exc
    try:
        return AnswerSet.from_records(records)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--answers",
        type=Path,
        help="Optional JSONL of {task_id, mode?, answer} records used to grade answers.",
    )
    parser.add_argument(
        "--session-isolation",
        action="store_true",
        help="Start a fresh session at every transcript session boundary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N tasks (0 = all). A cost control, not a result.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = load_jsonl(args.dataset)
    if args.limit:
        tasks = tasks[: args.limit]

    answers = load_answers(args.answers)
    evaluator = task_evaluator(session_isolation=args.session_isolation)

    def scorer(task, record, answer):  # type: ignore[no-untyped-def]
        return score_record(
            task,
            record,
            answer=answer,
            session_isolation=args.session_isolation,
        )

    config = EvaluationConfig(
        provider=args.provider,
        model=args.model,
        mode=args.mode,
        benchmark=args.benchmark,
        dataset_sha256=dataset_sha256(args.dataset),
        session_isolation=args.session_isolation,
        dataset_notes={
            "dataset": str(args.dataset),
            "task_count": len(tasks),
            "categories": sorted({task.category for task in tasks}),
            "reference_lengths": sorted(
                {
                    int(task.metadata.get("length_tier", 0) or 0)
                    for task in tasks
                }
            ),
            "seeds": sorted(
                {
                    int(task.metadata.get("seed", 0) or 0)
                    for task in tasks
                    if "seed" in task.metadata
                }
            ),
            "answers_supplied": len(answers),
        },
    )
    try:
        run = EvaluationRunner(config).run(
            tasks, evaluator=evaluator, scorer=scorer, answers=answers.answers
        )
    except NotImplementedError as exc:
        raise SystemExit(f"Evaluation scaffold: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run.to_dict(), indent=2) + "\n", encoding="utf-8")

    metrics = run.aggregate_metrics
    print(
        f"mode={args.mode} tasks={metrics.get('task_count', 0)} "
        f"graded={metrics.get('graded_answer_count', 0)} "
        f"recall={metrics.get('retrieval_recall', 0.0):.3f} "
        f"evidence_in_prompt={metrics.get('evidence_in_prompt_rate', 0.0):.3f} "
        f"accuracy={metrics.get('answer_accuracy', 0.0):.3f} "
        f"reduction={metrics.get('mean_context_reduction', 0.0):.3f}"
    )
    for issue in run.dataset_issues:
        print(f"dataset issue: {issue}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
