import pytest

from evaluation.datasets import load_jsonl
from evaluation.runner import EvaluationConfig, EvaluationRunner


def test_runner_requires_an_execution_strategy() -> None:
    tasks = load_jsonl("benchmarks/context_rot/dataset.jsonl")
    runner = EvaluationRunner(EvaluationConfig(mode="brainos"))

    with pytest.raises(NotImplementedError):
        runner.run(tasks)
