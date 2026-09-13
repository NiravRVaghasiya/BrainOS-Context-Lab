"""Evaluation orchestration boundary.

The runner deliberately accepts strategy and model callables instead of
coupling evaluation to a particular provider. Full execution is added after
the baseline modes and benchmark dataset are validated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from .datasets import BenchmarkTask


@dataclass(frozen=True)
class EvaluationConfig:
    provider: str = ""
    model: str = ""
    mode: str = "brainos"
    benchmark: str = "context_rot"
    context_budget: int = 4096
    temperature: float = 0.0
    benchmark_version: str = "scaffold"
    brainos_version: str = "unvalidated"
    application_version: str = "0.1.0"


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    timestamp: str
    config: EvaluationConfig
    task_results: list[dict[str, Any]] = field(default_factory=list)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "task_results": self.task_results,
            "aggregate_metrics": self.aggregate_metrics,
        }


class EvaluationRunner:
    """Provider- and strategy-neutral benchmark runner shell."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def run(
        self,
        tasks: list[BenchmarkTask],
        evaluator: Callable[[BenchmarkTask, str], dict[str, Any]] | None = None,
    ) -> EvaluationRun:
        if not tasks:
            raise ValueError("At least one benchmark task is required.")
        if evaluator is None:
            raise NotImplementedError(
                "Strategy execution is not wired yet. Implement the baseline mode "
                "adapters before running evaluations."
            )
        results = [evaluator(task, self.config.mode) for task in tasks]
        return EvaluationRun(
            run_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
            task_results=results,
        )
