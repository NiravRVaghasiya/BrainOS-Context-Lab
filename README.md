# BrainOS Context Lab

> Bring your model. Give it memory. Measure context efficiency.

BrainOS Context Lab is a standalone experimental web platform for investigating whether external cognitive memory can reduce unnecessary historical context while preserving task performance in long-running LLM interactions.

This repository integrates BrainOS as an upstream dependency. It does **not** modify or reimplement BrainOS. The application owns the provider abstraction, session management, UI, context orchestration, benchmarking, and evaluation harness.

## Current status

The repository is currently at the initial scaffolding stage. The package boundaries, public interfaces, documentation locations, benchmark layout, and test layout are in place. BrainOS integration, provider calls, persistence, and the complete evaluation pipeline will be implemented incrementally after the Phase 0 dependency and API validation.

## Repository layout

```text
app.py                         # Local/Hugging Face Space entry point
pyproject.toml                 # Package metadata and optional dependencies
src/
  app/                         # UI, session lifecycle, and application state
  brain/                       # BrainOS adapter and context construction
  providers/                   # LLM provider interfaces and adapters
  storage/                     # Conversation and evaluation persistence
  evaluation/                  # Benchmark runners, metrics, and reports
benchmarks/
  context_rot/                 # Long-conversation benchmark definition
  fixtures/                    # Small deterministic test fixtures
docs/                          # Architecture, integration, evaluation, and security notes
tests/                         # Unit, integration, security, and evaluation tests
results/                       # Generated results (not committed by default)
```

## Local setup

Python 3.10 or newer is required. The base package intentionally has no mandatory third-party dependency while the upstream BrainOS API is being validated.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ui,providers,dev]"
```

Start the initial UI scaffold with:

```bash
python app.py
```

The scaffold displays the planned application surfaces and does not yet make provider requests.

To enable the validated upstream BrainOS revision later, install the `integration` extra only after Phase 0 has confirmed the compatible commit:

```bash
pip install -e ".[integration]"
```

## Evaluation commands

The planned command-line entry points are already reserved:

```bash
python -m evaluation.run --mode brainos --benchmark context_rot --output results/run.json
python -m evaluation.compare results/full_context.json results/rag.json results/brainos.json
```

At scaffolding stage these commands provide interfaces and validation errors; benchmark execution will be added in the evaluation phases.

## Security principles

- Provider API keys are session-only secrets and must never be persisted or logged.
- Conversation and memory data are isolated by session.
- Retrieved memory is data, not a higher-priority instruction.
- Evaluation artifacts never contain provider credentials.
- Users remain responsible for usage and costs charged by their provider account.

See [`docs/security.md`](docs/security.md) for the intended threat model and [`docs/architecture.md`](docs/architecture.md) for the system boundaries.

## Research positioning

The project evaluates the following question rather than assuming a result:

> Does BrainOS-style external cognitive memory reduce historical context while maintaining task performance in long-running LLM interactions?

The initial release will compare Full Context, Sliding Window, Conventional RAG, and BrainOS under controlled benchmark conditions. Results should report both quality and efficiency, including retrieval quality, answer accuracy, context tokens, latency, and robustness as conversation length increases.
