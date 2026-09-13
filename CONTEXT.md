# BrainOS Context Lab — Working Context

> Living hand-off context for implementation work. This file records the
> repository state and decisions that should be preserved between phases.

**Last updated:** 2026-09-13
**Branch:** `arena/01a09b1f-brainos-context-lab`
**Baseline commit:** `ec1f29445764f2f58884ee3c5c1afdc0ae6411e5`
**Implementation plan:** [`BrainOS_Context_Lab_Implementation_Plan.md`](BrainOS_Context_Lab_Implementation_Plan.md)

## Product boundary

BrainOS Context Lab is a standalone application around the upstream BrainOS
runtime. It owns provider adapters, session handling, context orchestration,
UI, storage, and evaluation. It must not reimplement or modify BrainOS. The
LLM remains responsible for generation; BrainOS is an external cognitive-memory
layer that helps select historical context.

## Repository audit before Phase 1

The baseline commit was a scaffold rather than a completed implementation.
The following pieces were already present:

- the planned `src/app`, `src/brain`, `src/providers`, `src/storage`, and
  `src/evaluation` package boundaries;
- a session-local state and cleanup boundary;
- a BrainOS adapter contract with an injected fake-runtime integration test;
- a dependency-free context builder, memory policy, benchmark fixture, metric
  primitives, and evaluation runner shell;
- architecture, security, threat-model, and evaluation documentation; and
- provider files containing only a partial OpenAI SDK skeleton.

The Phase 0 upstream audit has now been completed. The validated BrainOS
revision is pinned in `pyproject.toml`, the upstream suite/evaluation/long-run
checks passed, and the actual runtime signatures are recorded in
[`docs/brainos-integration.md`](docs/brainos-integration.md). The application
adapter still needs the explicit signature/return-shape mapping in Phase 2;
that work is not silently claimed as complete here.

## Phase ledger

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 0 — Requirements/research baseline | **Complete as an upstream/API baseline** | BrainOS commit `1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc` is pinned; upstream tests (316), eval, long-run benchmark, and `observe → recall` smoke test passed. The application adapter mapping is Phase 2. |
| **Phase 1 — Provider abstraction** | **Complete in this turn** | OpenAI and generic OpenAI-compatible adapters now implement model listing, credential validation, generation, normalization, a provider factory, and secret-safe errors/configuration. |
| Phase 2 — BrainOS adapter | Next | Map the validated `BrainOS` API (`source`, `event_type`, `top_k`, decision strings, structured traces) into the application boundary with session isolation. |
| Phase 3 — Context construction engine | Scaffold present | Existing builder is a dependency-free first pass; budget/conflict/relevance work remains. |
| Phase 4 — Chat web UI | Pending | UI surfaces exist, but callbacks are not wired to provider/BrainOS services. |
| Phases 5–20 | Pending | Persistence, baselines, benchmark, evaluation, security hardening, deployment, and research release follow the plan. |

Detailed logs are available in
[`docs/phase-0-research-baseline.md`](docs/phase-0-research-baseline.md) and
[`docs/phase-1-provider-abstraction.md`](docs/phase-1-provider-abstraction.md).

## Phase 1 implementation contract

The stable application-facing provider contract is in
[`src/providers/base.py`](src/providers/base.py):

```python
class LLMProvider:
    def list_models(self) -> list[str]: ...
    def validate_credentials(self) -> bool: ...
    def generate(self, messages, **kwargs) -> ProviderResponse: ...
```

`ProviderConfig` contains the provider name, model, session-only API key,
optional compatible endpoint, temperature, output-token limit, and timeout.
`ProviderResponse` contains normalized text, model, token usage, and sanitized
optional raw metadata. SDK objects and SDK exception types do not cross the
boundary.

The factory in `providers.create_provider` currently recognizes:

- `openai` → `OpenAIProvider`;
- `openai-compatible`, `openai_compatible`, or `compatible` →
  `OpenAICompatibleProvider`.

Both adapters use the official OpenAI Python client lazily. Tests inject a
fake client, so provider behavior is testable without credentials, network
access, or the optional SDK.

## Security decisions carried forward

- API keys are excluded from `ProviderConfig` representations and
  `safe_dict()` diagnostics.
- Provider error messages are redacted using the active key and common bearer,
  API-key, token, and OpenAI-key patterns.
- Credential overrides (`api_key`, `authorization`, and `headers`) are rejected
  from per-request generation kwargs.
- Normalized response metadata drops secret-named fields and scrubs configured
  secret values.
- Session cleanup replaces the shared frozen provider configuration with a
  credential-free copy; keys remain process-local and are not persisted.
- The provider endpoint is supplied by the active session. Never expose the
  session key in browser diagnostics, evaluation artifacts, or traces.

## Validation baseline

From the repository root:

```bash
.venv/bin/pytest -q
# 18 passed

.venv/bin/ruff check src/providers src/app/state.py tests/unit/test_providers.py
# All checks passed
```

Phase 0 upstream evidence is recorded separately: the pinned BrainOS checkout
passed `316` upstream tests, its offline evaluation and 200-event long-run
benchmark passed, and a clean temporary environment installed
`.[integration]` and imported `brainos_runtime.BrainOS`.

The `.venv` directory is ignored and is only a local test environment. The
provider tests use deterministic injected clients and do not make network
requests. A full-repository lint run still reports pre-existing scaffold lint
findings outside the Phase 1 files; those are not silently presented as Phase 1
failures and should be handled in the relevant later cleanup.

## Next safe step

Implement Phase 2's explicit BrainOS adapter mapping against the pinned
revision. Replace the injected `FakeRuntime` assumptions only with tested
translations for `observe(source/event_type)`, `recall(top_k)`, decision
strings, `why()`, and structured `trace()`. Then wire the provider factory and
BrainOS adapter into an application service without moving credentials into
persistent storage or browser-visible diagnostics.
