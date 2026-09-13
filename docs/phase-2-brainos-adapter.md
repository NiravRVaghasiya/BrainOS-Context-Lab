# Phase 2 log — BrainOS adapter

**Date:** 2026-09-13
**Branch:** `arena/01a09b29-brainos-context-lab`
**Starting commit:** `199329b59a482226ed3e0f5e5a8b241bff3fe5f7`
**Plan section:** [Phase 2 — BrainOS Adapter](../BrainOS_Context_Lab_Implementation_Plan.md#6-phase-2--brainos-adapter)

## Objective

Create a clean, explicit boundary between the application and the pinned
BrainOS v2 runtime. The adapter must translate the validated signatures
recorded in Phase 0 rather than guessing or silently adapting incompatible
shapes:

```text
observe(event, source=..., event_type=...)
recall(query, top_k=...)
decide(query) -> decision string
assess(query) -> grounding dict
why(query) -> explanation dict
trace() -> structured session-trace mapping
```

The application-facing contract remains `observe` / `recall` / `decide` /
`explain` / `trace`. BrainOS types must not leak into the UI, provider layer,
or evaluation runner.

## Starting-state audit

Phase 0 pinned BrainOS commit `1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc` and
documented the real API. Phase 1 completed the provider abstraction. The
application adapter still used the pre-validation fake-runtime assumptions:

| Scaffold assumption | Validated upstream API |
| --- | --- |
| `observe(text, metadata=...)` | `observe(event, source=..., event_type=...)` |
| `recall(query, limit=...)` | `recall(query, top_k=...)` |
| boolean/object `decide` | decision string (`act` / `retrieve` / `search` / `ask` / `clarify`) |
| trace list | structured `{session_id, cycles, records, state}` mapping |

`memory_policy.py` filtered candidates but did not extract them.
`context_builder.py` already existed as a Phase 3 scaffold and was reused, not
rewritten. Gradio callbacks remain unwired (Phase 4).

## Work completed

### 1. Explicit runtime mapping

Rewrote [`src/brain/adapter.py`](../src/brain/adapter.py):

- Application `metadata` is translated into `source` and `event_type`. The
  runtime is never called with `metadata=`.
- Application `limit` (and optional `top_k`) maps to BrainOS `top_k`.
- `recall()` enriches `list[str]` results with stable IDs from
  `active_memories()` / `memories()` when the runtime exposes them.
- `decide()` prefers `assess()` and normalizes decision strings into the
  application `Decision` dataclass. Only `act` is treated as sufficient.
- `explain()` calls `why()` and sanitizes the result.
- `trace()` maps the structured BrainOS session trace into `TraceEvent`
  values, then appends adapter-local events.
- `list_memories()` exposes stored memories for inspection.
- `create_runtime()` / `create_brain_adapter()` construct **one runtime per
  session** with `session_id` and `actor_id`, using `prefer_generated=False`
  so the offline inline plugins are used.

### 2. Conservative memory policy

Updated [`src/brain/memory_policy.py`](../src/brain/memory_policy.py) with
`extract_candidates()`. Greetings, questions, and short chatter are not
stored. Project facts, preferences, corrections, constraints, goals, tasks,
decisions, and temporal statements are classified conservatively. The adapter
can additionally refuse candidates below the policy threshold.

### 3. Trace mapping and secret scrubbing

[`src/brain/trace.py`](../src/brain/trace.py) now maps BrainOS cycle records
by stage name and counts. Retrieved memory text is not copied into traces.
Secret-named fields are dropped and credential-shaped strings are redacted.

### 4. Application service

Added [`src/app/service.py`](../src/app/service.py). `ConversationService` is
the only place that combines:

1. memory-policy extraction,
2. BrainOS observe / recall / decide / explain,
3. the existing context builder,
4. the Phase 1 provider factory.

Session diagnostics use `ProviderConfig.safe_dict()` and never include API
keys. Clearing a conversation drops the session-scoped runtime so a later
turn cannot reuse another conversation's memory.

### 5. Tests

- Injected `FakeRuntime` now matches the pinned BrainOS signatures.
- Unit tests cover source/event_type mapping, `top_k`, decision strings,
  policy skips, session isolation, and explanation redaction.
- Service tests cover observe → later recall, optional generation, and
  credential-free diagnostics.
- Optional live tests against the pinned `brainos_runtime` prove the real
  `observe → recall` round-trip and per-session isolation. They skip when
  the integration extra is not installed.

## Phase 2 exit assessment

| Exit criterion | Status |
| --- | --- |
| Observe information | **Complete** — policy-accepted text is stored via `observe(source=, event_type=)`. |
| Retrieve it later | **Complete** — `recall(top_k=)` returns `MemoryRecord` values with stable IDs when the runtime exposes them. |
| Construct a relevant-memory set | **Complete at the adapter/service boundary** — the service recalls memories and the existing context builder delimits them. Conflict/recency work remains Phase 3. |
| Show the retrieved items | **Complete** — `list_memories()`, `explain()`, mapped traces, and service diagnostics expose sanitized inspection data. |

The Gradio UI is still a scaffold. Chat callbacks are Phase 4.

## Validation performed

```text
.venv/bin/pytest -q
37 passed in 0.11s
```

The two live tests
`tests/integration/test_brainos_runtime.py` ran against the pinned BrainOS
revision installed from the `integration` extra. No provider API key was
used.

```text
.venv/bin/ruff check src/brain/adapter.py src/brain/trace.py \
  src/brain/memory_policy.py src/app/service.py src/app/state.py \
  src/app/session.py tests/fakes.py tests/unit/test_adapter.py \
  tests/unit/test_service.py tests/unit/test_trace.py \
  tests/integration tests/security
# All checks passed
```

A full-repository Ruff run still reports pre-existing scaffold findings in
`context_builder.py`, `evaluation/`, and `ui.py`. Those are outside this
phase.

## Files changed in Phase 2

```text
CONTEXT.md
README.md
docs/brainos-integration.md
docs/phase-2-brainos-adapter.md
src/app/__init__.py
src/app/service.py
src/app/session.py
src/app/state.py
src/app/ui.py
src/brain/__init__.py
src/brain/adapter.py
src/brain/memory_policy.py
src/brain/trace.py
tests/__init__.py
tests/conftest.py
tests/fakes.py
tests/integration/test_adapter_boundary.py
tests/integration/test_brainos_runtime.py
tests/security/test_secrets.py
tests/unit/test_adapter.py
tests/unit/test_memory_policy.py
tests/unit/test_service.py
tests/unit/test_trace.py
```

## Follow-up work

1. Phase 3: relevance filtering, conflict checks, recency weighting, and
   tokenizer-aware budgets in the context construction engine.
2. Phase 4: wire Gradio callbacks to `ConversationService`.
3. Keep BrainOS isolated behind `BrainMemoryAdapter`. Do not import
   `brainos_runtime` from the UI, providers, or evaluation metrics.
