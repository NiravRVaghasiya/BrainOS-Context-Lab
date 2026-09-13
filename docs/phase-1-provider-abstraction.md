# Phase 1 log — Provider abstraction

**Date:** 2026-09-13
**Branch:** `arena/01a09b1f-brainos-context-lab`
**Starting commit:** `ec1f29445764f2f58884ee3c5c1afdc0ae6411e5`
**Plan section:** [Phase 1 — Provider Abstraction](../BrainOS_Context_Lab_Implementation_Plan.md#5-phase-1--provider-abstraction)

## Objective

Make model invocation provider-agnostic while keeping BrainOS independent from
provider SDKs. The phase target was the application contract:

```python
class LLMProvider:
    def list_models(self): ...
    def validate_credentials(self): ...
    def generate(self, messages, **kwargs): ...
```

The initial provider set is OpenAI plus a generic OpenAI-compatible endpoint.
The implementation must support session-only BYOK credentials, normalize SDK
responses, and never put API keys into logs, traces, diagnostics, or persisted
results.

## Starting-state audit

Before changing code, the repository was inspected against the full
implementation plan.

### Already present

- The repository was a clean scaffold at the starting commit.
- `src/providers/base.py` declared a partial `ProviderConfig`,
  `ProviderResponse`, and `LLMProvider` protocol.
- `src/providers/openai.py` had lazy SDK import and a basic chat-completions
  call, but no complete normalization/error/security boundary.
- `src/providers/openai_compatible.py` only checked that `base_url` was
  truthy.
- `src/app/session.py` already removed session state and attempted to clear the
  active key.
- Existing tests covered session cleanup, trace redaction, the BrainOS adapter
  boundary, context construction, metrics, dataset loading, and the evaluation
  scaffold.
- The UI showed provider fields but explicitly did not make provider calls.

### Deliberately not claimed as part of Phase 1

The real BrainOS runtime was not wired into the provider work. A separate
Phase 0 audit later validated and pinned the upstream revision, but the
application-side signature/return-shape mapping remains Phase 2. The existing
injected runtime test continues to validate the provider-independent
application boundary.

## Work completed

### 1. Strengthened the provider-neutral contract

Updated [`src/providers/base.py`](../src/providers/base.py) to provide:

- a session-safe, frozen `ProviderConfig` with provider/model/endpoint,
  temperature, output-token, and timeout settings;
- `safe_dict()` diagnostics that omit `api_key` entirely;
- `ProviderResponse` as the normalized result type;
- `ProviderError`, `ProviderConfigurationError`,
  `ProviderResponseError`, and `UnsupportedProviderError` boundaries;
- common redaction helpers for configured secrets, bearer values,
  `api_key`/token/secret assignments, and OpenAI-style `sk-...` values; and
- recursive sanitization for response metadata with secret-named fields
  removed.

The API key remains a field only because the active provider needs it in
memory. It is excluded from `repr` and safe diagnostics.

### 2. Completed the OpenAI adapter

Updated [`src/providers/openai.py`](../src/providers/openai.py) to:

- import the OpenAI SDK lazily;
- support deterministic client injection for tests;
- pass the configured API key and timeout, and an optional `base_url`, to the
  SDK client;
- implement `list_models()` and `validate_credentials()` through an
  authenticated model-list request;
- implement `generate()` with per-request model, temperature, and
  `max_tokens` overrides without mutating session configuration;
- reject per-request credential/header overrides;
- validate message shape without echoing message contents in errors;
- normalize object or mapping responses, content parts, model IDs, and token
  usage into `ProviderResponse`; and
- convert SDK failures and malformed responses into safe application errors
  without exception chaining or credential-bearing messages.

### 3. Completed the generic compatible adapter

Updated [`src/providers/openai_compatible.py`](../src/providers/openai_compatible.py)
to reuse the OpenAI implementation while requiring a caller-supplied
`base_url`. This keeps provider-specific behavior out of the context builder
and leaves room for hosted gateways and local OpenAI-compatible servers.

### 4. Added an explicit provider factory

Updated [`src/providers/__init__.py`](../src/providers/__init__.py) with
`create_provider(config)`. It maps the normalized provider names
`openai`, `openai-compatible`, `openai_compatible`, and `compatible` to the
adapters and rejects unknown names explicitly.

### 5. Unified session configuration and cleanup

`src/app/state.py` now reuses the provider-layer `ProviderConfig` instead of
maintaining a second configuration type. Because the shared configuration is
frozen to prevent accidental in-place secret mutation, session cleanup replaces
it with a credential-free copy. Existing session isolation behavior remains
intact.

### 6. Added automated coverage

Added [`tests/unit/test_providers.py`](../tests/unit/test_providers.py) covering:

- model listing and credential validation;
- response and usage normalization;
- request-level overrides and immutability of session settings;
- compatible endpoint requirements and factory selection;
- unknown provider rejection;
- API-key redaction in provider errors and configuration diagnostics; and
- required model/message validation.

## Security checklist for this phase

| Requirement from the plan | Result |
| --- | --- |
| Never write API keys to logs | No provider logging was added. |
| Never include API keys in traces | Provider calls do not create traces or expose credentials; existing trace sanitization remains in place. |
| Never store keys in a database | The provider config is process/session memory only; no storage code was changed to accept keys. |
| Keep keys in server memory for the active session | `ProviderConfig` is held by session state and is replaced during session end. |
| Clear keys when a session ends | Preserved and adapted `SessionManager.end()` cleanup. |
| Redact secrets from exception messages | Centralized configured-key and credential-pattern redaction. |
| Do not expose credentials in browser diagnostics | `safe_dict()` omits the key and provider raw metadata is sanitized. |
| Tell users they pay their provider | Existing UI copy already states that provider-account usage/cost is the user's responsibility. |

## Validation performed

The system Python did not have `pytest` installed. An ignored local `.venv`
was created for validation, then the suite was run without network calls:

```text
.venv/bin/pytest -q
18 passed in 0.05s
```

The Phase 1 files also pass the configured lint rules:

```text
.venv/bin/ruff check src/providers src/app/state.py tests/unit/test_providers.py
All checks passed!
```

A full-repository Ruff run still reports pre-existing scaffold findings in
`app.py`, `src/app/ui.py`, `src/brain/`, and `src/evaluation/`; those are
outside this phase's implementation files and are recorded in `CONTEXT.md`
instead of being hidden.

## Phase 1 exit assessment

The provider abstraction exit criteria are met at the adapter/test level:

- OpenAI and OpenAI-compatible configurations can be constructed.
- Credentials can be validated with `list_models()`.
- A model-ready `Hello` message can be sent through `generate()` and returned
  as a normalized response, as demonstrated by the deterministic fake-client
  test.
- The application, BrainOS adapter, context builder, and evaluation layers do
  not depend on an OpenAI SDK response object.

A live network smoke test was intentionally not run because no user credential
was supplied and this repository must not request or persist one. The next
phase should add a live/manual smoke-test procedure only after the user
chooses to run it with their own session key.

## Files changed in Phase 1

```text
CONTEXT.md
README.md                                  (status/setup wording)
docs/phase-1-provider-abstraction.md      (this log)
src/app/state.py
src/providers/__init__.py
src/providers/base.py
src/providers/openai.py
src/providers/openai_compatible.py
tests/unit/test_providers.py
```

## Follow-up work

1. Finish Phase 0 by inspecting and pinning the upstream BrainOS revision.
2. Add an application service that combines `create_provider()` with the
   context builder and session state.
3. Wire that service into the UI in the planned chat phase.
4. Add timeout/cost controls and provider-specific tokenizer accounting before
   research evaluation.
5. Keep all future provider additions behind `LLMProvider`; do not import an
   SDK from BrainOS, the context builder, or evaluation metrics.
