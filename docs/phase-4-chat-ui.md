# Phase 4 log — Chat web UI

**Date:** 2026-09-13
**Branch:** `arena/01a09bc4-brainos-context-lab`
**Starting commit:** `2d63a3bdc4da1af0c33fa7a98d8b934b4634664c`
**Plan section:** [Phase 4 — Chat Web UI](../BrainOS_Context_Lab_Implementation_Plan.md#8-phase-4--chat-web-ui)

## Objective

Wire the Gradio shell to `ConversationService` following the plan's layout —
configuration sidebar (provider, model, API key, endpoint, temperature, context
budget, memory mode), chat in the middle, and an inspection column with
Memory / Context / Cognitive Trace / Evaluation tabs — while honouring the five
constraints the Phase 3 hand-off carried forward:

1. Configure context through `ContextSettings` only.
2. Render diagnostics from the sanitized values the service already returns.
3. Surface the `contested` label and the drop reasons.
4. Keep BrainOS behind `BrainMemoryAdapter`; no `brainos_runtime` import from
   the UI.
5. Add cost/turn limits before exposing benchmark controls.

## Starting-state audit

| Scaffold state | Phase 4 requirement |
| --- | --- |
| 89-line static `ui.py`: every component constructed, zero `.click`/`.submit` handlers | every planned control drives the service and every panel renders live turn data |
| status banner: "Chat callbacks are not wired yet" | honest phase status |
| no session plumbing: components shared by all visitors | per-browser `ChatController` in `gr.State`, registered in `SessionManager` |
| `pyproject.toml` floor `gradio>=4.0` with `Chatbot(type="messages")` | verified floor; Gradio 6 removed the `type` parameter entirely |
| `FakeProvider` implemented only `generate()` | UI exercises `list_models()` / `validate_credentials()` too |

Everything the panels need was already produced per turn by Phase 3
(`context_stats`, `context_report`, `memory_ranking`, `trace`, `inspect()`), so
Phase 4 is pure application-layer work: no brain/ or providers/ module changed.

## Work completed

### 1. `ChatController` (`src/app/controller.py`, new, ~530 lines)

All UI behaviour lives in a controller that never imports Gradio, so the whole
phase is unit-testable headlessly and the callbacks stay thin.

| Responsibility | Behaviour |
| --- | --- |
| Lazy service creation | the BrainOS runtime is imported on first *use*, not at page load; when it is missing, `send_message` returns the install hint (`pip install -e '.[integration]'`) instead of a traceback, and `_stored_rows` never constructs a runtime just to render an empty table |
| `apply_provider` | the browser's key field travels server-side only: an empty field *keeps* the credential already in server memory, so no view ever has to echo a key back. The frozen `ProviderConfig` is replaced atomically; invalid values (negative temperature, bad max_tokens) are rejected with a message and change nothing. The cached service — and with it the cached provider client — is dropped **only when the config actually changed**, so the session's BrainOS runtime (which lives on `state.brain`) survives provider edits |
| `apply_context_settings` | writes to `ContextSettings` only (constraint 1) and validates a candidate by *constructing* both `context_budget()` and `retrieval_policy()` before accepting it, so a bad slider value can never reach context construction mid-turn |
| `set_memory_mode` | records the Phase 6 selector on `ContextSettings.mode`; choosing `full_context`/`sliding_window`/`rag` today returns an explicit "arrives with Phase 6, turns still run the BrainOS pipeline" notice instead of silently mislabeling runs |
| `views()` | one call produces **every** browser-visible value (history, status, stored/retrieved/dropped/conflict rows, ranking JSON, context summary + full accounting, final prompt, trace rows, decision, connection status, mode notice, turn counters) from the sanitized service outputs (constraint 2), with runtime-derived table text additionally scrubbed against the session key |
| `validate_connection` / `list_models` | prefer the service's provider seam (keeps injected/test providers authoritative) and fall back to `create_provider` only when the BrainOS runtime is unavailable, so credentials remain validatable on a server without BrainOS; failures go through `safe_error_message` with the session secret |
| Session lifecycle | `clear_conversation()` (drops transcript + BrainOS memory, keeps settings), `end_session()` (clears credentials via `SessionManager.end`, returns a *fresh* controller), and an early turn-limit guard (`DEFAULT_MAX_TURNS = 400`) as the Phase 15 down-payment required by constraint 5 — the Evaluation tab stays a placeholder |
| Error discipline | `BrainOSNotConfiguredError` → install hint; `ProviderError`/unexpected `Exception` → redacted one-line status; no traceback ever reaches the browser |

### 2. Gradio wiring (`src/app/ui.py`, rewritten, 89 → ~380 lines)

Plan §8 layout, mapped component-for-component:

```text
Sidebar (scale 1)          Main (scale 2)          Inspection (scale 1)
├─ Provider dropdown       ├─ Chatbot (messages)   ├─ Tab Memory
├─ Model (dropdown with    ├─ Status line          │   ├─ Stored memories table
│  allow_custom_value)     └─ Message + Send       │   │   (type, text, timestamp,
├─ API key (password)                              │   │    retrievals, source turn, status)
├─ Endpoint                                        │   ├─ Retrieved table (rank, score,
├─ Temperature slider                              │   │   relevance, type, text, flags:
├─ Max output tokens                               │   │   contested/suspicious)
├─ Validate / List models                          │   └─ Ranking components JSON
├─ Connection status                               ├─ Tab Context
├─ Context budget slider                           │   ├─ Accounting digest (the plan's
├─ Memory mode + notice                            │   │   five required fields, budget
├─ Accordion: recent-history                       │   │   utilization, full-context
│   budget, memory budget,                         │   │   reference, reduction %)
│   max memories                                   │   ├─ Full 26-field stats JSON
├─ Clear conversation /                            │   └─ Final prompt (verbatim)
│   End session                                    ├─ Tab Cognitive Trace
└─ BYOK cost + key-lifetime                        │   ├─ Pipeline diagram
    notice (plan §5 security)                      │   ├─ Decision (sufficient/action/
                                                   │   │   confidence/reason)
                                                   │   ├─ Sanitized trace table
                                                   │   ├─ Dropped memories + reasons
                                                   │   └─ Conflict resolutions
                                                   └─ Tab Evaluation (Phase 17
                                                       placeholder, turn limit shown)
```

Eight wired events: `demo.load` (attach controller), send `click` + `submit`,
validate, list models, clear, end, mode change. All sidebar settings are
re-applied on **every** send, so an edited budget can never be forgotten; any
rejection is prepended to the status line.

**Gradio version tolerance.** Gradio 5 needs `Chatbot(type="messages")` for
role/content dicts; Gradio 6 removed the parameter (messages are the only
format). `_make_chatbot()` inspects the signature and passes `type` only when
it exists — verified against Gradio 6.27.0. `pyproject.toml`'s UI floor moved
`gradio>=4.0` → `gradio>=5.0` accordingly (the messages format arrived in
4.44; below that the dict history would not render).

### 3. Repairs found by Phase 4 tests

**Repair 1 — `turn.error` was rendered unredacted.** `ConversationService`
stores `error = str(exc)` from a caught `ProviderError`. Real adapters redact
before raising (Phase 1), but the service does not re-verify, and a fake or
buggy adapter can carry a credential (`authorization: Bearer <key>`). The test
`test_provider_error_surfaces_redacted_in_status` pasted the session key into
an error message and found it verbatim in the status line. The controller —
the browser boundary — now scrubs `turn.error` against the session secret and
the usual credential shapes before composing the headline. Defence in depth:
even if a future provider forgets to redact, the UI cannot leak.

**Repair 2 — validation bypassed the service's provider seam.** The first
implementation of `validate_connection`/`list_models` called `create_provider`
directly whenever no service existed yet. That ignored an injected provider
(tests, and later any wrapper a deployment injects) and, on a server without
the OpenAI SDK, reported the dependency error instead of validating. Both now
prefer `service.provider()` and fall back to the factory only when the BrainOS
runtime itself is unavailable.

### 4. Test-fake upgrade (`tests/fakes.py`)

`FakeProvider` implements the full `LLMProvider` protocol: `list_models()`
(configurable catalogue), `validate_credentials()` (configurable verdict), and
an `error` mode in which every call raises `ProviderError(error)` — the seam
the redaction tests use. Existing positional construction is unchanged, so no
prior test moved.

## Security posture (what the browser can and cannot see)

| Surface | Rule |
| --- | --- |
| API key textbox | browser → server once per apply; **no** output list contains it; empty resubmit keeps the server-side value |
| All panel values | produced by `controller.views()` from service-sanitized data; runtime-derived strings (retrieved/dropped/conflict text, decision reason) scrubbed against the session key |
| Error text | `safe_error_message` + session-secret scrubbing; no tracebacks |
| Chat transcript and final prompt | deliberately **not** pattern-scrubbed: they are faithful records of what the user typed and what was sent to their own provider. The session key is still removed from memory text *before* prompt construction (Phase 3 `_guard_memories`), so a key can never be replayed to a provider |
| Connection/model status | counts and identifiers only |
| Session end | `SessionManager.end` clears credentials, conversation, and BrainOS runtime; the browser receives a fresh controller |

New tests pin all of the above (`tests/security/test_ui_secrets.py`), including
that credential-shaped strings (`sk-proj-…`) are scrubbed from diagnostic
tables while the transcript stays faithful.

## Validation

```bash
.venv/bin/pytest -q      # 184 passed   (154 → 184)
.venv/bin/ruff check .   # All checks passed!  (whole repository)
```

New tests: 21 controller unit cases (`tests/unit/test_ui_controller.py`),
3 structural Gradio-wiring cases that skip without Gradio
(`tests/unit/test_ui_wiring.py`), 4 UI-secret security cases
(`tests/security/test_ui_secrets.py`), and 2 live-runtime integration cases on
the pinned BrainOS revision that skip without it
(`tests/integration/test_ui_live.py`).

**Live HTTP validation** (real server, pinned BrainOS runtime, no API key, via
the Gradio client API — the same path a browser takes):

- `on_send` stored "For Project Atlas, the production database is PostgreSQL
  16." as `PROJECT_STATE` with timestamp, source turn, and retrieval count;
- the follow-up question retrieved it at **rank 1** (score 0.6066, relevance
  0.3661), rendered `<retrieved_memory>` delimiters in the final prompt, and
  filled the 26-field accounting (`final_context_tokens=155` of a 4096 budget);
- 24 sanitized trace rows rendered; `on_clear`, `on_end`, `on_mode_change`,
  and `on_validate` all returned their expected views, and a fresh session
  worked after `on_end`;
- session state persisted across HTTP calls (gradio queue session), confirming
  the `gr.State` + `SessionManager` plumbing.

Observed in the live numbers: with the default wide history window on a
two-message chat, `final_context_tokens` (155) exceeds the full-context
reference (81) — the Phase 3 degenerate case ("reduction is driven by the
history window, not memory selection") is now visible in the UI's own
accounting, rendered honestly (reduction clamps at 0%). This is a short-chat
artifact, not a regression; memory-first budgets show 72–78% reduction at
60 turns (Phase 3 measurements).

## Decisions carried forward

- **Controller is the seam.** Phase 6 baseline modes plug into
  `ContextSettings.mode` (already recorded per turn) and Phase 15 cost
  controls extend `max_turns` — neither should require touching `ui.py`.
- **History renders from `state.messages`** (the single source of truth) every
  turn; the chatbot is not editable, so no divergence path exists.
- **Page load is BrainOS-optional**: a visitor without the runtime installed
  gets a working UI and a precise install hint on first send.
- Single-process in-memory sessions are intentional for Phase 4; HF Spaces
  multi-worker deployments need Phase 5 persistence or sticky routing.

## Known limitations

- No response streaming: `LLMProvider.generate` is one-shot (Phase 1 contract).
- Evaluation tab is a placeholder until Phase 17, gated by Phase 15.
- Stored-memory table text is clipped at 240 characters for readability; the
  full text remains in the ranking JSON and final prompt.
- Model dropdown persists its choices only for the session; there is no
  cross-session model cache.

## Next

Phase 5 — Conversation persistence: implement the already-defined
`ConversationStore` / `MemoryStore` / `EvaluationStore` protocols
(`src/storage/`) with the SQLite MVP backend, keeping session isolation and
the never-persist-keys rule.
