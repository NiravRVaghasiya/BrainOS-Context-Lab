# BrainOS Context Lab — Working Context

> Living hand-off context for implementation work. This file records the
> repository state and decisions that should be preserved between phases.

**Last updated:** 2026-09-13
**Branch:** `arena/01a09b29-brainos-context-lab`
**Baseline commit:** `199329b59a482226ed3e0f5e5a8b241bff3fe5f7`
**Implementation plan:** [`BrainOS_Context_Lab_Implementation_Plan.md`](BrainOS_Context_Lab_Implementation_Plan.md)

## Product boundary

BrainOS Context Lab is a standalone application around the upstream BrainOS
runtime. It owns provider adapters, session handling, context orchestration,
UI, storage, and evaluation. It must not reimplement or modify BrainOS. The
LLM remains responsible for generation; BrainOS is an external cognitive-memory
layer that helps select historical context.

## Phase ledger

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 0 — Requirements/research baseline | Complete | BrainOS commit `1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc` is pinned; upstream tests (316), eval, long-run benchmark, and `observe → recall` smoke test passed. |
| Phase 1 — Provider abstraction | Complete | OpenAI and OpenAI-compatible adapters implement listing, credential validation, generation, normalization, a factory, and secret-safe errors. |
| **Phase 2 — BrainOS adapter** | **Complete in this turn** | Explicit mapping of `observe(source/event_type)`, `recall(top_k)`, decision strings/`assess()`, `why()`, and structured `trace()`. Session-isolated factory, conservative memory policy, and `ConversationService` wiring. |
| Phase 3 — Context construction engine | Scaffold present | Existing builder is a dependency-free first pass used by the service; budget/conflict/relevance/recency work remains. |
| Phase 4 — Chat web UI | Pending | UI surfaces exist, but callbacks are not wired to `ConversationService`. |
| Phases 5–20 | Pending | Persistence, baselines, benchmark, evaluation, security hardening, deployment, and research release follow the plan. |

Detailed logs are available in
[`docs/phase-0-research-baseline.md`](docs/phase-0-research-baseline.md),
[`docs/phase-1-provider-abstraction.md`](docs/phase-1-provider-abstraction.md),
and [`docs/phase-2-brainos-adapter.md`](docs/phase-2-brainos-adapter.md).

## Phase 2 implementation contract

The application-facing BrainOS contract remains in
[`src/brain/adapter.py`](src/brain/adapter.py):

```python
class BrainMemoryAdapter:
    def observe(self, text, *, metadata=None) -> None: ...
    def recall(self, query, *, limit=8) -> list[MemoryRecord]: ...
    def decide(self, query) -> Decision: ...
    def explain(self, query) -> dict: ...
    def trace(self) -> list[TraceEvent]: ...
```

The adapter translates that contract onto the pinned BrainOS facade:

| Application | Pinned BrainOS runtime |
| --- | --- |
| `observe(text, metadata={source, role, event_type, ...})` | `observe(text, source=..., event_type=...)` — never `metadata=` |
| `recall(query, limit=8)` / `top_k=` | `recall(query, top_k=...)` returning `list[str]`, enriched with IDs from `active_memories()` / `memories()` |
| `decide(query)` | prefers `assess(query)` and falls back to `decide()` strings `act/retrieve/search/ask/clarify`; only `act` is `Decision.sufficient=True` |
| `explain(query)` | sanitized `why(query)` |
| `trace()` | mapped structured `{session_id, cycles, records, state}` plus adapter-local events |

`create_brain_adapter(session_id=..., actor_id=...)` constructs **one runtime
per application session** with `prefer_generated=False`. Runtimes and storage
backends must not be shared across sessions.

`ConversationService` in [`src/app/service.py`](src/app/service.py) is the
application service layer: memory-policy extraction, BrainOS observe/recall,
the existing context builder, and `providers.create_provider`. Credentials
stay in session memory and are omitted from diagnostics.

## Security decisions carried forward

- API keys remain excluded from `ProviderConfig` representations and
  `safe_dict()` diagnostics.
- Provider error messages are redacted using the active key and common bearer,
  API-key, token, and OpenAI-key patterns.
- BrainOS explanations and traces drop secret-named fields and scrub
  credential-shaped strings. Trace mapping records counts, not retrieved
  memory text.
- Session cleanup replaces the shared frozen provider configuration with a
  credential-free copy and drops the session BrainOS adapter.
- Retrieved memory is still wrapped as data, not instructions, by the context
  builder.
- The provider endpoint is supplied by the active session. Never expose the
  session key in browser diagnostics, evaluation artifacts, or traces.

## Validation baseline

From the repository root:

```bash
.venv/bin/pytest -q
# 37 passed

.venv/bin/ruff check src/brain/adapter.py src/brain/trace.py \
  src/brain/memory_policy.py src/app/service.py src/app/state.py \
  src/app/session.py tests/fakes.py tests/unit/test_adapter.py \
  tests/unit/test_service.py tests/unit/test_trace.py \
  tests/integration tests/security
# All checks passed
```

Live adapter tests in `tests/integration/test_brainos_runtime.py` require the
optional integration extra and ran successfully in this environment against
the pinned BrainOS revision. They skip when `brainos_runtime` is not
installed. No provider API key was used.

The `.venv` directory is ignored and is only a local test environment. A
full-repository lint run still reports pre-existing scaffold lint findings
outside the Phase 2 files; those are not silently presented as Phase 2
failures.

## Next safe step

Implement Phase 3's context construction engine on top of the now-mapped
adapter. Keep the existing delimited-memory builder, then add relevance
filtering, conflict checks, recency weighting, and stricter token-budget
accounting. Do not wire Gradio callbacks until that context contract is
stable; the service already produces `BuiltContext` for the UI phase to
consume.
