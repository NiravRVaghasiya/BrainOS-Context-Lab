# Context-rot benchmark

This directory contains the initial deterministic fixture and the planned home for the long-conversation benchmark.

The full dataset should distribute facts across increasing conversation lengths and include:

- recent and old facts,
- repeated and irrelevant content,
- temporal replacement,
- explicit corrections and conflicts,
- multi-hop questions,
- distractor-heavy conversations,
- cross-session cases, and
- abstention cases where the answer is absent.

`generation.py` currently creates a small single-hop fixture for validating dataset loading and evaluation plumbing. It is not a research result and must not be used to support claims about BrainOS performance.

Generate the fixture with:

```bash
python benchmarks/context_rot/generation.py --count 3
```

Each JSONL record contains a task ID, category, conversation, question, expected answer, required memory IDs, and reproducibility metadata.
