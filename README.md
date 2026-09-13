# BrainOS Context Lab

> Bring your model. Give it memory. Measure context efficiency.

BrainOS Context Lab is a standalone experimental web platform for investigating whether external cognitive memory can reduce unnecessary historical context while preserving task performance in long-running LLM interactions.

This repository integrates BrainOS as an upstream dependency. It does **not** modify or reimplement BrainOS. The application owns the provider abstraction, session management, UI, context orchestration, benchmarking, and evaluation harness.

## Current status

Phases 1–6 are complete and validated against the pinned BrainOS runtime. The
app persists conversations and memory mirrors to SQLite with hard session
isolation, offers the full data-control set (clear conversation, clear memory,
export session, end/delete session), and can now run the same conversation
through all five of the plan's baseline context-management modes.

- **Phase 1** maps the provider abstraction (OpenAI and OpenAI-compatible)
  behind `LLMProvider`, with secret-safe errors and diagnostics.
- **Phase 2** maps the pinned BrainOS v2 runtime (`observe(source=, event_type=)`,
  `recall(top_k=)`, decision strings / `assess()`, `why()`, structured `trace()`,
  `contradictions()`, `stale_memories()`) behind a stable `BrainMemoryAdapter`.
- **Phase 3** turns recall results into a model-ready prompt through the full
  retrieval pipeline — deduplicate, IDF-weighted relevance filter, conflict
  check, recency weighting, enforced token budget — with complete accounting and
  a per-memory audit trail.
- **Phase 4** wires the Gradio UI to a web-framework-free `UIController`:
  bring-your-own-key provider connection, one chat turn, and the Memory,
  Context, Cognitive Trace, and Evaluation tabs.
- **Phase 5** persists conversations and memory mirrors to SQLite behind the
  `ConversationStore` / `MemoryStore` / `EvaluationStore` protocols:
  best-effort writes that never break a turn, session-key redaction at the
  write site, row-level session isolation, and export/delete controls.
- **Phase 6** implements the plan's five baseline modes — A full context,
  B sliding window, C lexical RAG, D BrainOS, E BrainOS + RAG — each fixing its
  own history window, with a BrainOS-free lexical chunk retriever, a second
  delimited evidence block, and the evaluation-runner seam that makes
  `python -m evaluation.run --mode …` execute.

A session-scoped `ConversationService` combines the adapter, the retrieval
policy, the context builder, and the provider factory. Deterministic fakes cover
the whole pipeline without BrainOS installed; optional live tests exercise the
pinned runtime. 379 tests pass and `ruff check .` is clean repository-wide.

Chat is usable without an API key: BrainOS still observes and retrieves memory,
and the panels show exactly what the model *would* have been sent. See
[`CONTEXT.md`](CONTEXT.md) for the living implementation state and the Phase
0–6 logs in `docs/`.

### Measured behaviour so far

On a synthetic 60-turn conversation with six durable facts (one corrected mid-
conversation), validated with the live pinned runtime and an estimated token
counter:

| Context configuration | mean tokens sent | full-context baseline | reduction |
| --- | --- | --- | --- |
| wide history window | 1486 | 1391 | 0.0% |
| memory-first | 387 | 1391 | **72.2%** |
| memory-first, tight | 308 | 1391 | **77.8%** |

6/6 probe questions received exactly the evidence they needed, and the corrected
fact never re-entered a prompt. The reduction is driven mainly by the
history-window setting rather than by memory selection — which is exactly why
Phase 6 gives every baseline mode its own window.

Phase 6 then ran one 28-message conversation through all five modes with the
live pinned runtime:

| Mode | tokens sent | full-context baseline | reduction | fact still in prompt |
| --- | --- | --- | --- | --- |
| A full context | 510 | 510 | 0.0% | yes |
| B sliding window | 189 | 510 | 62.9% | **no** |
| C lexical RAG | 256 | 510 | 49.8% | yes |
| D BrainOS | 182 | 510 | **64.3%** | yes |
| E BrainOS + RAG | 345 | 510 | 32.4% | yes |

Mode D cost fewer tokens than the sliding window *and* still carried the fact
the window had dropped. These are integration-validation observations, **not**
research results: one synthetic conversation, an estimated token counter, no
model in the loop, single trial, and the benchmark dataset (Phase 7) and scoring
(Phase 8) do not exist yet.

## Repository layout

```text
app.py                         # Local/Hugging Face Space entry point
pyproject.toml                 # Package metadata and optional dependencies
src/
  app/                         # UI, controller, session lifecycle, service, and state
  baselines/                   # Phase 6 baseline modes A-E and the BrainOS-free
                               #   lexical retriever Mode C/E use
  brain/                       # BrainOS adapter, retrieval policy, context
                               #   builder, memory policy, tokenizers, traces
  providers/                   # LLM provider interfaces and adapters
  storage/                     # Conversation and evaluation persistence
  evaluation/                  # Benchmark runners, mode strategies, metrics, reports
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

Start the UI with:

```bash
python app.py          # http://localhost:7860
```

Select a provider, paste a session-only key, pick a model, and press **Connect**
to list the models that key can reach. The key is held in server memory, the key
box is cleared immediately, and **Forget key** drops it at any time. Without a
key the app still observes, retrieves, and builds context — the panels show the
exact prompt, but no model is called.

The five tabs are Chat, Memory (stored / in-prompt / filtered-out / conflicts),
Context (summary, statistics, final prompt), Cognitive Trace, and Evaluation
(placeholder until Phase 17).

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
- The API key box is cleared on every connect attempt; the key is never rendered
  back to the browser, not even masked, and every panel value is redacted
  against it.
- Conversation and memory data are isolated by session, persisted only with
  credentials redacted at the write site, and deleted from disk by the clear
  and end-session controls.
- Retrieved memory is data, not a higher-priority instruction: it is delimited,
  stripped of delimiter breakouts and role prefixes, and flagged when it matches
  an instruction-override pattern.
- The active session key is redacted by exact match from browser-visible
  diagnostics and from recalled memory text before it can reach a prompt.
- Evaluation artifacts never contain provider credentials.
- Users remain responsible for usage and costs charged by their provider account.

See [`docs/security.md`](docs/security.md) for the intended threat model and [`docs/architecture.md`](docs/architecture.md) for the system boundaries.

## Research positioning

The project evaluates the following question rather than assuming a result:

> Does BrainOS-style external cognitive memory reduce historical context while maintaining task performance in long-running LLM interactions?

The initial release will compare Full Context, Sliding Window, Conventional RAG, and BrainOS under controlled benchmark conditions. Results should report both quality and efficiency, including retrieval quality, answer accuracy, context tokens, latency, and robustness as conversation length increases.
