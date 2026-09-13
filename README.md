# BrainOS Context Lab

> Bring your model. Give it memory. Measure context efficiency.

BrainOS Context Lab is a standalone experimental web platform for investigating whether external cognitive memory can reduce unnecessary historical context while preserving task performance in long-running LLM interactions.

This repository integrates BrainOS as an upstream dependency. It does **not** modify or reimplement BrainOS. The application owns the provider abstraction, session management, UI, context orchestration, benchmarking, and evaluation harness.

## Current status

Phase 2, the BrainOS adapter, is complete at the adapter/service/test level.
The application now maps the pinned BrainOS v2 runtime (`observe(source=,
event_type=)`, `recall(top_k=)`, decision strings / `assess()`, `why()`, and
structured `trace()`) behind a stable `BrainMemoryAdapter`. A session-scoped
`ConversationService` combines that adapter with the Phase 1 provider factory
and the existing context builder. Deterministic fakes cover the mapping
without BrainOS installed; optional live tests exercise the pinned runtime.

The Gradio scaffold is not yet wired to chat callbacks (Phase 4), and the
context builder is still a first-pass implementation (Phase 3). See
[`CONTEXT.md`](CONTEXT.md) for the living implementation state and the Phase
0/1/2 logs in `docs/`.

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

The upstream BrainOS revision is pinned and the Phase 2 adapter mapping is
wired. Install the optional integration extra to use the live runtime:

```bash
pip install -e ".[integration]"
```

### Provider adapter smoke test

The provider layer is usable without a live request by injecting a client in
unit tests. In an application service, construct a session-only configuration
and use the factory:

```python
from providers import ProviderConfig, create_provider

config = ProviderConfig(
    provider="openai",
    model="gpt-4o-mini",
    api_key="provided-for-this-session",
)
provider = create_provider(config)
provider.validate_credentials()
reply = provider.generate([{"role": "user", "content": "Hello"}])
print(reply.text)
```

Do not put a real key in source code, logs, benchmark records, or committed
examples. Generic OpenAI-compatible endpoints use
`provider="openai-compatible"` and a session-local `base_url`.

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
