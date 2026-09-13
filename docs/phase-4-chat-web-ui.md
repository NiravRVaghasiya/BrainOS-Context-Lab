# Phase 4 log — Chat web UI

**Date:** 2026-09-13
**Branch:** `arena/01a09b6a-brainos-context-lab`
**Starting commit:** `2d63a3bdc4da1af0c33fa7a98d8b934b4634664c`
**Plan section:** [Phase 4 — Chat Web UI](../BrainOS_Context_Lab_Implementation_Plan.md#8-phase-4--chat-web-ui)

## Objective

Wire the Gradio shell to the application so a user can bring a provider, chat,
and inspect what BrainOS remembered, what was sent to the model, and what it
would have cost to send everything.

The plan specifies four surfaces:

| Surface | Plan requirement |
| --- | --- |
| Sidebar | provider, model, API key, endpoint, temperature, context budget, memory mode |
| Main | chat |
| Right panel | memory, retrieved memories, context statistics, cognitive trace |
| Tabs | Chat, Memory, Context, Cognitive Trace, Evaluation |

Phase 3 already produced every value those panels need. The work of this phase
is the boundary between "the service can produce it" and "a browser can see it
without leaking a credential".

## Starting-state audit

| Scaffold behaviour (end of Phase 3) | Phase 4 requirement |
| --- | --- |
| `app/ui.py` built widgets with no callbacks | every widget wired to a controller call |
| Provider config had nowhere to come from | connect / validate / list models / forget key |
| One shared `ProviderConfig` default | one isolated `SessionState` per browser session |
| No place to hold a session's service | cached `ConversationService` per session id |
| Panel values existed only as dicts on a turn | rendered, bounded, redacted rows and markdown |
| No session controls | clear conversation / clear memory / end session / export |
| `SessionManager` existed but was unused by the UI | session lifecycle owned by the UI controller |
| No runaway-usage guard on a BYOK surface | turn and message limits (Phase 15, minimal subset) |

## Work completed

### 1. `src/app/controller.py` (new) — the UI's behaviour, without the UI

A Gradio-free controller that owns session lifecycle, provider connection, one
chat turn, panel rendering, and export. Keeping it free of any web dependency
means the whole surface is testable without a browser and the layout can change
without touching behaviour.

| Method | Responsibility |
| --- | --- |
| `ensure_session(id)` | return the session, or start a fresh one when the id is stale |
| `service(id)` | cached `ConversationService` per session (one BrainOS runtime per session) |
| `connect(...)` | build/validate `ProviderConfig`, list models, store the key, clear the key box |
| `disconnect(id)` | forget the key and drop the cached provider client |
| `refresh_models(id)` | re-list models for the key already in memory |
| `update_context(id, **fields)` | apply budget/policy knobs; reject unknown and invalid fields |
| `chat(id, message)` | one turn → transcript + every panel |
| `refresh(id)` | re-render the last turn without sending anything |
| `clear_conversation(id)` / `clear_memory(id)` | the plan's two separate user-data controls |
| `end_session(id)` | drop transcript, memory, key; return a new session id |
| `export_session(id)` / `export_text(id)` | sanitized JSON snapshot |

Design decisions worth recording:

- **The session id is the only thing the browser holds.** `gr.State` carries a
  non-secret uuid; the transcript, the memories, and the key stay server-side.
  A stale id never restores another session's memory — it starts a new one.
- **Connect never builds a BrainOS runtime.** The controller resets an
  *existing* service instead of creating one, so a missing `brainos_runtime`
  installation breaks the chat turn (with an install hint) and not the connect
  button.
- **Unknown context fields are rejected, not ignored.** A typo in a slider name
  would otherwise leave the user looking at controls that do nothing.
- **Limits are the two that matter before any network surface exists**:
  `max_turns` (200) and `max_message_chars` (8000). Benchmark controls — the
  rest of Phase 15 — are still unexposed.

### 2. `src/app/panels.py` (new) — rendering, and the only path to the browser

Pure functions from application data to browser-safe values: table rows, the
context summary, the prompt rendering, the trace, and the status line. Panels
import no web framework and no service, so they cannot accidentally widen what
gets rendered.

- **Column schemas are module constants** (`STORED_MEMORY_COLUMNS`, …) shared by
  the layout and the tests, so a column cannot drift out of sync with its rows.
- **Cells are bounded**: memory text is clipped to 240 characters, prompt text
  to 20 000, trace details to 160, and the runtime trace to the newest 40
  events.
- **Missing signals render as `—`, not `None` or `0`.** A memory with no
  runtime score must not look like a memory that scored zero.
- **The context summary always reports the comparison, not just the count** —
  tokens sent, the full-context baseline, the reduction, budget utilization, and
  the per-stage token breakdown.

#### Panel fix found during validation

The first implementation built the "retrieved memories" table from
`turn.retrieved_memories` — everything BrainOS recalled — while the audit table
listed what the policy removed. On a live run this made both tables describe the
same memories: six rows "retrieved" while the statistics said `1 of 6` were
selected. The table is now driven by the builder's **post-budget ranking**, so
it answers "what did the model actually see"; everything else appears once, in
the audit table, with its reason. When no ranking exists at all (a caller that
did not score) the records are listed in recall order, so the panel is never
silently empty.

### 3. `src/app/ui.py` (rewritten) — layout plus thin callbacks

```
Header (positioning + BYOK cost notice)
┌──────────────┬────────────────────────┬──────────────────────────┐
│ Provider     │ Chat                   │ Inspection               │
│  provider    │  transcript            │  Memory                  │
│  model       │  message + Send        │   stored / in-prompt /   │
│  api key     │  status line           │   filtered out /         │
│  endpoint    │  clear conversation    │   conflicts              │
│  temperature │  clear memory          │  Context                 │
│  max tokens  │  refresh panels        │   summary / stats /      │
│  connect     │  end session           │   final prompt           │
│  forget key  │  export session JSON   │  Cognitive Trace         │
│ Context      │                        │   stages / runtime       │
│  mode,       │                        │  Evaluation (placeholder)│
│  budgets,    │                        │                          │
│  policy      │                        │                          │
└──────────────┴────────────────────────┴──────────────────────────┘
```

- Callbacks are two-line adapters over the controller; the controller decides
  what a `TurnView` contains and the UI only unpacks it.
- `demo.load` starts a session per page load; every callback also self-heals a
  stale id.
- Callback functions are named (`connect`, `chat`, `export_session`, …) so the
  auto-generated API endpoints are readable; the Enter-to-send duplicate of the
  chat callback is registered with `api_name=False`.
- The layout builders return component dataclasses and `_wire()` attaches the
  events, so the component set and the callback arity are inspectable in tests.
- **Gradio 6 API notes** (the installed version dropped several kwargs the plan
  era used): `Dropdown` has no `placeholder` (use `info`), `Chatbot` no longer
  takes `type="messages"` — the messages format is native — `Textbox` has no
  `show_copy_button`, and `launch()` has no `allowed_origins`; cross-origin
  embedding is controlled with `strict_cors`, which the app leaves at Gradio's
  strict default unless `GRADIO_STRICT_CORS` is set.
- Default sidebar budgets are deliberately **memory-first** (history window 512
  tokens) because the Phase 3 measurement showed a window larger than the
  conversation makes BrainOS mode indistinguishable from full context.

### 4. Service and state seams

| Change | Why |
| --- | --- |
| `ConversationService(adapter_factory=…, provider_factory=…)` | lets the UI (and later the evaluation runner) construct both from session state, while tests inject fakes without weakening the production path |
| `ConversationService.stored_memories()` | public, credential-guarded listing for the Memory panel |
| `ConversationService.reset_memory()` | "clear memory" without clearing the transcript — two distinct controls in the plan |
| `ConversationService.reset_provider()` | a client built from an old key must not keep serving after the key is replaced or cleared |
| `ContextSettings.uses_memory()` | the memory-mode selector maps to one behaviour the application can already perform honestly |
| `MEMORY_MODES = ("brainos", "no_memory")` | the selector offers the modes that exist today; unknown values keep memory enabled rather than silently disabling it |

Diagnostics now also carry `mode`, `turn`, and `stored_memory_count`.

### 5. Tests

| File | Cases | Covers |
| --- | --- | --- |
| `tests/unit/test_controller.py` | 29 | connect/disconnect, model listing, redacted provider errors, turn/char limits, session isolation, stale-id recovery, no-memory mode, clear/end/refresh, export, missing-runtime hint |
| `tests/unit/test_panels.py` | 23 | column schemas, bounded cells, missing-signal rendering, summary/trace/status variants, conflict counting |
| `tests/ui/test_ui.py` | 6 | the app builds against the installed Gradio, every callback's declared outputs match what it returns, the key box is cleared, export writes parseable JSON |
| `tests/integration/test_chat_controller_live.py` | 4 | the same controller against the **pinned BrainOS runtime**, deterministic provider, no API key |

Test count: **154 → 216**. `ruff check .` is clean repository-wide.

`tests/ui/` skips when Gradio is absent, and the live file skips when
`brainos_runtime` is absent, so the suite still runs with no optional
dependency installed.

## Security decisions for the browser boundary

- **The key box is emptied on every connect attempt.** `ConnectionView.key_value`
  is always `""`; the value lives on the session's `ProviderConfig` and is
  absent from `safe_dict()`.
- **Every returned value is redacted against the active key.** The controller
  runs `sanitize_value(..., secrets=(key,))` over the transcript, the rows, the
  summary, the prompt, the trace, and the export — so a key pasted into a
  conversation cannot be handed back by a panel. Redaction is applied to the
  *rendered* values, not to the `MemoryRecord`s (sanitizing a dataclass turns it
  into a dict and the panel would then read the wrong shape).
- **Model choices and status text come from `safe_dict()`**; the endpoint is
  redacted with it too.
- **`gr.State` holds only the session uuid.**
- Memory still renders inside `<retrieved_memory>` delimiters as untrusted data,
  and the prompt panel shows the exact text the model received.

## Measured behaviour (live pinned BrainOS, 41-turn conversation)

Driven through `UIController` — the same path a browser uses — with a
deterministic provider double (`FakeLLMProvider`, reply `Noted.`) and no
provider API key. Six durable facts were planted among filler turns; no chit-chat
was stored.

Sidebar sweep at `memory_budget=512`, `max_memories=6`, probing three facts:

| Recent-history budget | mean tokens sent | full-context baseline | mean reduction | probes answered | history kept |
| --- | --- | --- | --- | --- | --- |
| 1024 | 881.7 | 811.0 | 0.0% | 3/3 | 84 of 84 |
| 256 | 590.0 | 811.0 | **27.2%** | 3/3 | 51 of 84 |
| 64 | 248.0 | 811.0 | **69.4%** | 3/3 | 13 of 84 |

This reproduces the Phase 3 finding through the UI's own controls: **the
reduction comes from the history window, not from memory selection**, and
retrieval quality held at 3/3 in every configuration tested.

Other measured properties:

- 6/6 durable facts stored; 0/15 chit-chat turns stored (the Phase 2 policy
  repair continues to hold).
- `no_memory` control mode: 0 memories retrieved, no `<retrieved_memory>` block,
  510 tokens sent.
- The export snapshot parses as JSON and contains transcript, memories, provider
  (key-free), and context settings.
- The session key never appeared in any panel, prompt, trace, or export.

**These are integration-validation observations, not research results.** One
synthetic conversation, an estimated token counter, no model in the loop, and no
baseline modes yet (Phase 6) or benchmark (Phase 7).

### Finding for Phases 7–8: a correction can be filtered out before it can win

For `What database does Project Atlas use?` after the turn
`Correction: the production database is MySQL 8 now.`, the prompt contained the
**stale** `PostgreSQL 16` memory and the `CORRECTION` memory was dropped as
`low_relevance`. Two independent mechanisms, both now visible in the audit
table:

1. The heuristic claim matcher requires subject Jaccard ≥ 0.6. The two subjects
   are `{production, database}` and `{project, atla, production, database}` —
   Jaccard 0.5 — so the pair is never detected as a conflict, even though the
   newer memory carries the `CORRECTION` type.
2. Conflict resolution runs *before* the relevance floor, but the correction
   text scores below the floor against the question regardless
   (`Correction: the production database is MySQL 8 now.` shares only
   `database` with `What database does Project Atlas use?`). Detecting the
   conflict alone would therefore not put the correction in the prompt — it
   would produce an empty memory block.

Deliberately **not** tuned here: changing the subject-similarity rule or the
relevance floor to fix one synthetic conversation is the overfitting the Phase 3
log warned about. Conflict-resolution accuracy and abstention accuracy are
scored benchmark categories (plan §11 D, §12); this is the concrete,
reproducible failure case for them to aim at.

### Constraint: the auto-generated HTTP API is stateless

Gradio exposes every callback as an API endpoint (`/chat`, `/connect`,
`/export_session`, …; the callbacks are named so the endpoints are readable).
Verified against the running server: a call to `/chat` returns a full, correctly
serialized set of panels.

What it does **not** do is carry `gr.State` across calls — an API client gets a
fresh session per request (verified: two consecutive `/chat` calls both report
`Turn 1`, with and without a `session_hash` query parameter). The browser UI is
unaffected: there the state is stored server-side per session hash. Programmatic
multi-turn use therefore needs the "own Space API" the plan reserves for
Phase 14, not the auto-generated endpoints.

### Finding: abstention is still not achieved (unchanged from Phase 3)

The absent-answer probe (`What is the Project Atlas payroll vendor?`) still
selected one memory — the only one sharing `Project Atlas`. Nothing in this
phase changed the threshold; it remains a Phase 7–8 calibration item.

## Validation baseline

```bash
python3 -m venv .venv
.venv/bin/pip install pytest ruff gradio
.venv/bin/pip install "brainos-cli @ git+https://github.com/NiravRVaghasiya/BrainOS.git@1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc"

.venv/bin/pytest -q          # 216 passed
.venv/bin/ruff check .       # All checks passed!
.venv/bin/python app.py      # http://localhost:7860
```

The live tests in `tests/integration/test_chat_controller_live.py`,
`tests/integration/test_context_pipeline.py`, and
`tests/integration/test_brainos_runtime.py` run against the pinned revision and
skip when `brainos_runtime` is not installed. No provider API key is used
anywhere in the suite.

## Next safe step

**Phase 5 — Conversation persistence** (SQLite-backed `ConversationStore`,
`MemoryStore`, `EvaluationStore`). Everything it needs is already isolated: a
session is a `SessionState` with a `conversation_id`, the transcript is a list of
`{role, content}` dicts, and the controller is the single place that mutates
them. Two constraints to carry forward:

1. Persistence must never widen the credential surface — the API key lives on
   `ProviderConfig` in process memory and is already excluded from every
   serialized view; a store must be given the sanitized payloads, not the state
   object.
2. Phase 6 (baseline modes) must set `recent_turn_budget` **per mode**, or Mode A
   and Mode D stay indistinguishable — now measurable from the sidebar sweep
   above.
