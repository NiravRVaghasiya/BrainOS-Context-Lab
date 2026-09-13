"""Deterministic context-rot fixture generator.

The research benchmark generator will add richer temporal, conflict, distractor,
and abstention cases after the initial BrainOS API validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep the fixture generator usable directly from a source checkout.
src_dir = Path(__file__).resolve().parents[2] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from evaluation.datasets import BenchmarkTask, write_jsonl


def generate_fixture(count: int = 3) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    for index in range(count):
        memory_id = f"fact-{index + 1}"
        database = "PostgreSQL" if index % 2 == 0 else "SQLite"
        tasks.append(
            BenchmarkTask(
                task_id=f"fixture-{index + 1}",
                category="single_hop",
                conversation=[
                    {
                        "role": "user",
                        "content": f"For project {index + 1}, the database is {database}.",
                    },
                    {
                        "role": "assistant",
                        "content": "Understood; I will retain that project detail.",
                    },
                ],
                question=f"What database does project {index + 1} use?",
                expected_answer=database,
                required_memory_ids=(memory_id,),
                conversation_length=2,
                metadata={"fixture": True, "memory_id": memory_id},
            )
        )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic context-rot fixtures.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/context_rot/dataset.jsonl"))
    args = parser.parse_args()
    write_jsonl(args.output, generate_fixture(args.count))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
