from pathlib import Path

from evaluation.datasets import load_jsonl


def test_scaffold_dataset_loads() -> None:
    path = Path("benchmarks/context_rot/dataset.jsonl")
    tasks = load_jsonl(path)

    assert len(tasks) == 3
    assert tasks[0].task_id == "fixture-1"
    assert tasks[0].required_memory_ids == ("fact-1",)
