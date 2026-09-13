"""Benchmark task schema and JSONL loading helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    """One context-rot task with explicit evidence expectations."""

    task_id: str
    category: str
    conversation: list[dict[str, str]]
    question: str
    expected_answer: str
    required_memory_ids: tuple[str, ...] = ()
    conversation_length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> BenchmarkTask:
        return cls(
            task_id=str(item["task_id"]),
            category=str(item["category"]),
            conversation=list(item.get("conversation", [])),
            question=str(item["question"]),
            expected_answer=str(item["expected_answer"]),
            required_memory_ids=tuple(item.get("required_memory_ids", [])),
            conversation_length=item.get("conversation_length"),
            metadata=dict(item.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "conversation": self.conversation,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "required_memory_ids": list(self.required_memory_ids),
            "conversation_length": self.conversation_length,
            "metadata": self.metadata,
        }


def load_jsonl(path: str | Path) -> list[BenchmarkTask]:
    """Load non-empty JSONL records and reject malformed task definitions."""

    tasks: list[BenchmarkTask] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(BenchmarkTask.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid benchmark record at line {line_number}") from exc
    return tasks


def write_jsonl(path: str | Path, tasks: Iterable[BenchmarkTask]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
