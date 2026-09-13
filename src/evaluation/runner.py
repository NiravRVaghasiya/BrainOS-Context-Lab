"""Evaluation orchestration boundary.

The runner deliberately accepts strategy, scoring, and model callables instead of
coupling evaluation to a particular provider. Phase 6 wired the strategy half
(the five baseline modes) and Phase 7 wired the scoring half, so a run now
produces per-task records *and* an aggregate without any provider configured —
`--mode brainos` measures retrieval, context cost, and (when answers are
supplied) answer quality.

What the runner still does not do: call a model. Generation belongs to a caller
with credentials and the plan's cost controls (Phase 15); the runner grades
whatever answers it is handed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .datasets import BenchmarkTask, dataset_issues
from .scoring import aggregate_scores

#: Scoring callable shape: ``(task, strategy_record, answer) -> scored record``.
Scorer = Callable[[BenchmarkTask, dict[str, Any], str | None], dict[str, Any]]


@dataclass(frozen=True)
class EvaluationConfig:
    provider: str = ""
    model: str = ""
    mode: str = "brainos"
    benchmark: str = "context_rot"
    context_budget: int = 4096
    temperature: float = 0.0
    benchmark_version: str = "context_rot-v1"
    brainos_version: str = "unvalidated"
    application_version: str = "0.1.0"
    dataset_sha256: str = ""
    session_isolation: bool = False
    #: Free-form provenance for a run: length tier, seed, category filter, and
    #: anything else Phase 16 needs to reproduce it.
    dataset_notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    timestamp: str
    config: EvaluationConfig
    task_results: list[dict[str, Any]] = field(default_factory=list)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    dataset_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "task_results": self.task_results,
            "aggregate_metrics": self.aggregate_metrics,
            "dataset_issues": list(self.dataset_issues),
        }


class EvaluationRunner:
    """Provider- and strategy-neutral benchmark runner shell."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def run(
        self,
        tasks: list[BenchmarkTask],
        evaluator: Callable[[BenchmarkTask, str], dict[str, Any]] | None = None,
        *,
        scorer: Scorer | None = None,
        answers: Mapping[tuple[str, str], str] | None = None,
    ) -> EvaluationRun:
        """Execute ``tasks`` through ``evaluator`` and optionally score them.

        ``answers`` is keyed by ``(task_id, mode)`` with ``(task_id, "")`` as the
        mode-agnostic fallback; an unscored task is still reported, but its
        answer verdict is ``ungraded`` so it cannot inflate a run's accuracy.
        """

        if not tasks:
            raise ValueError("At least one benchmark task is required.")
        if evaluator is None:
            raise NotImplementedError(
                "Strategy execution is not wired yet. Implement the baseline mode "
                "adapters before running evaluations."
            )
        issues = dataset_issues(tasks)
        results: list[dict[str, Any]] = []
        scored: list[dict[str, Any]] = []
        for task in tasks:
            record = evaluator(task, self.config.mode)
            if scorer is not None:
                answer = None
                if answers is not None:
                    answer = answers.get((task.task_id, self.config.mode))
                    if answer is None:
                        answer = answers.get((task.task_id, ""))
                score = scorer(task, record, answer)
                record = dict(record)
                record["scores"] = score
                scored.append(score)
            results.append(record)
        return EvaluationRun(
            run_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
            task_results=results,
            aggregate_metrics=aggregate_scores(scored) if scored else {},
            dataset_issues=issues,
        )


__all__ = ["EvaluationConfig", "EvaluationRun", "EvaluationRunner", "Scorer"]
