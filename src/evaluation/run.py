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

Phase 13 adds a ``security`` block beside the metrics: the aggregate of every
guard action taken while the run's prompts were assembled (injection families
detected, structural rewrites, credentials redacted, history roles downgraded).
A clean block is the run's evidence that nothing was quarantined or redacted; it
is not a claim that the dataset was benign.

Phase 9 adds ``--generate``: the prompt each task built is sent to the model and
the answer is graded in place, so a single mode can be measured with a real model
in the loop (latency included). The credential is read from an environment
variable and never appears on the command line or in the artifact. Cost ceilings
apply to the whole run, and provider usage is recorded in a ``generation_usage``
block beside the Phase 8 metrics.

The dataset's own revision is recorded as a content hash, which is what makes a
result reproducible: the same hash, the same generator seed, and the same mode
describe the same experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from baselines.modes import ABLATION_ORDER, MODE_ORDER
from brain.tokenizers import TokenizerUnavailableError, estimate_tokens, tiktoken_counter
from reproducibility.run_provenance import (
    app_version,
    brainos_version,
    build_manifest,
)
from security.findings import summarize_security

from .datasets import load_jsonl
from .generation import (
    MissingCredentialError,
    ModelSpec,
    api_key_from_environment,
    budgeted_generator,
    build_provider,
)
from .limits import RunBudget, RunLimits
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
    # ``allow_abbrev=False`` keeps ``--api-key`` from abbreviating
    # ``--api-key-env``: a pasted credential would then be recorded as a
    # variable name and exported with the run.
    parser = argparse.ArgumentParser(
        description="Run a BrainOS Context Lab evaluation.", allow_abbrev=False
    )
    parser.add_argument("--model", default="", help="Provider model identifier.")
    parser.add_argument("--provider", default="", help="Provider name.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=list(MODE_ORDER) + list(ABLATION_ORDER),
        help=(
            "Baseline mode or Phase 11 ablation "
            f"({', '.join(ABLATION_ORDER)})."
        ),
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
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Send each task's prompt to the configured model and grade its answer. "
            "Requires --model and a credential in --api-key-env."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout.")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Name of the environment variable holding the credential.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-failures", type=int, default=None)
    parser.add_argument(
        "--token-counter",
        choices=("estimate", "tiktoken"),
        default="estimate",
        help="How prompts are counted for the cost ceilings.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _generation_setup(
    args: argparse.Namespace, limits: RunLimits
) -> tuple[ModelSpec, object, RunBudget, str]:
    """Build the model spec, provider, shared budget, and key for a generated run.

    The key is returned only so the caller can hand it to the redaction list; it
    is never written to the artifact.
    """

    try:
        counter, counter_name = _resolve_counter(args)
    except TokenizerUnavailableError as exc:
        raise SystemExit(str(exc)) from None
    try:
        spec = ModelSpec(
            name="single-mode",
            provider=args.provider or "openai",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            seed=args.seed,
            timeout_seconds=min(float(args.timeout), float(limits.timeout_seconds)),
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        ).with_limits(limits)
        api_key = api_key_from_environment(spec)
        provider = build_provider(spec, api_key=api_key)
    except MissingCredentialError as exc:
        raise SystemExit(str(exc)) from None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    budget = RunBudget(limits=limits, counter=counter, counter_name=counter_name)
    return spec, provider, budget, api_key


def _resolve_counter(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    if args.token_counter == "tiktoken":
        return tiktoken_counter(args.model), f"tiktoken:{args.model}"
    return estimate_tokens, "estimate_tokens"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generate and not args.model.strip():
        raise SystemExit("--generate requires --model.")
    tasks = load_jsonl(args.dataset)
    if args.limit:
        tasks = tasks[: args.limit]

    limits = RunLimits(
        max_tasks=args.limit or None,
        max_requests=args.max_requests,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_total_tokens=args.max_total_tokens,
        max_generation_failures=args.max_failures,
        timeout_seconds=float(args.timeout),
    )
    generator = None
    budget = None
    spec = None
    if args.generate:
        spec, provider, budget, api_key = _generation_setup(args, limits)
        generator = budgeted_generator(
            provider,
            spec.generation_settings(),
            budget,
            counter=budget.counter,
            secrets=(api_key,),
        )

    answers = load_answers(args.answers)
    evaluator = task_evaluator(
        session_isolation=args.session_isolation, generate=generator
    )

    def scorer(task, record, answer):  # type: ignore[no-untyped-def]
        # A supplied answer (``--answers``) grades a replayed or externally
        # generated answer; otherwise a generated run grades its own output.
        supplied = answer if answer is not None else record.get("answer")
        return score_record(
            task,
            record,
            answer=supplied,
            session_isolation=args.session_isolation,
        )

    config = EvaluationConfig(
        provider=args.provider,
        model=args.model,
        mode=args.mode,
        benchmark=args.benchmark,
        dataset_sha256=dataset_sha256(args.dataset),
        session_isolation=args.session_isolation,
        temperature=float(args.temperature),
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

    payload = run.to_dict()
    # Phase 13: the guard audit travels with the numbers. A run whose prompts
    # were assembled with quarantines or credential redactions is not the same
    # run as one whose guards never fired, and an artifact that carries no guard
    # block cannot claim it held nothing sensitive.
    payload["security"] = summarize_security(
        record.get("security") for record in run.task_results
    )
    # Phase 16 (reproducibility): a top-level ``repro`` block and the fallback
    # version fields on ``config``, so a run artifact produced or consumed by
    # older tooling still carries the §22 provenance vocabulary. The manifest
    # is the full authority.
    config_dict = asdict(run.config)
    config_dict["output"] = str(args.output)
    payload["repro"] = build_manifest(
        run_id=run.run_id,
        timestamp=run.timestamp,
        task_ids=tuple(str(item.get("task_id", "")) for item in run.task_results),
        dataset_path=args.dataset,
        config=config_dict,
        aggregate_metrics=run.aggregate_metrics,
    )
    payload["config"]["application_version"] = app_version()
    payload["config"]["brainos_version"] = brainos_version()
    payload["config"]["output"] = str(args.output)
    if budget is not None and spec is not None:
        # The Phase 6/7 run schema stays as it is; the cost report is an
        # addition beside it so older consumers keep working unchanged.
        payload["generation_settings"] = spec.to_dict()
        payload["generation_usage"] = budget.snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    metrics = run.aggregate_metrics
    graded = int(metrics.get("graded_answer_count", 0) or 0)
    # An ungraded run has no accuracy, faithfulness, or QAE. Printing 0.000
    # would read as "every answer was wrong" when nothing was graded at all —
    # the same convention the controlled-comparison summary already uses.
    ungraded = not graded
    print(
        f"mode={args.mode} tasks={metrics.get('task_count', 0)} "
        f"graded={graded} "
        f"recall={_fmt(metrics.get('retrieval_recall'))} "
        f"evidence_in_prompt={_fmt(metrics.get('evidence_in_prompt_rate'))} "
        f"accuracy={'—' if ungraded else _fmt(metrics.get('answer_accuracy'))} "
        f"faithfulness={'—' if ungraded else _fmt(metrics.get('faithfulness'))} "
        f"reduction={_fmt(metrics.get('mean_context_reduction'))} "
        f"qae="
        f"{'—' if ungraded else _fmt(metrics.get('mean_quality_adjusted_efficiency'), digits=6)}"
    )
    if ungraded:
        print("No answers were graded: accuracy, faithfulness, and QAE are unset, not zero.")
    if budget is not None:
        snapshot = budget.snapshot()
        print(
            f"generated={snapshot['requests'] - snapshot['failures'] - snapshot['skips']} "
            f"failures={snapshot['failures']} skips={snapshot['skips']} "
            f"tokens={snapshot['total_tokens']} within_limits={snapshot['within_limits']}"
        )
    security = payload["security"]
    if not security.get("clean", True):
        totals = security.get("totals") or {}
        summary = ", ".join(f"{name}={count}" for name, count in sorted(totals.items()))
        print(f"security findings: {summary} (quarantined={security.get('quarantined', 0)})")
    for issue in run.dataset_issues:
        print(f"dataset issue: {issue}", file=sys.stderr)
    return 0


def _fmt(value: object, *, digits: int = 3) -> str:
    """Render a missing metric as an em dash; ``None`` is not zero."""

    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
