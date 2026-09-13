# BrainOS Context Lab — Working Context

> Living hand-off context for implementation work. This file records the
> repository state and decisions that should be preserved between phases.

**Last updated:** 2026-09-13
**Branch:** `arena/01a09bc4-brainos-context-lab`
**Baseline commit:** `2d63a3bdc4da1af0c33fa7a98d8b934b4634664c`
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
| Phase 2 — BrainOS adapter | Complete | Explicit mapping of `observe(source/event_type)`, `recall(top_k)`, decision strings/`assess()`, `why()`, and structured `trace()`. Session-isolated factory, conservative memory policy, and `ConversationService` wiring. **Memory-policy classifier repaired during Phase 3** (see below). |
| Phase 3 — Context construction engine | Complete | Full retrieval pipeline (dedupe → relevance → conflict → recency → budget), enforced token budgets with documented eviction order, 26-field accounting, runtime signal consumption, injection hardening, and a credential-leak repair. Validated live at 60 turns. |
| Phase 4 — Chat web UI | Complete | `ChatController` + rewritten Gradio UI: BYOK sidebar, chat, Memory / Context / Cognitive Trace tabs, session lifecycle, turn-limit guard. Two browser-boundary repairs (error redaction, provider seam). Validated live over HTTP against the pinned runtime. |
| **Phase 5 — Conversation persistence** | **Complete in this turn** | SQLite backend behind finalized store protocols (thread-safe per-op connections, lazy schema, upsert mirrors); the service persists turns best-effort with session-key redaction at the write site; the controller owns Clear memory / Delete session / Export session; the backend is chosen at deployment wiring, so headless runs never touch disk. Validated live: real server + real SQLite + pinned runtime. |
| Phase 6 — Baseline modes | Pending | Modes A–E behind `ContextSettings.mode` (already recorded per selection); the Phase 3 finding requires per-mode `recent_turn_budget`, and `evaluation/runner.py` is still a shell. |
| Phases 7–20 | Pending | Benchmark, metrics, experiments, statistics, ablations, error analysis, security hardening, deployment, cost controls, reproducibility, evaluation pipeline, tests, MVP, release. |

Detailed logs are available in
[`docs/phase-0-research-baseline.md`](docs/phase-0-research-baseline.md),
[`docs/phase-1-provider-abstraction.md`](docs/phase-1-provider-abstraction.md),
[`docs/phase-2-brainos-adapter.md`](docs/phase-2-brainos-adapter.md),
[`docs/phase-3-context-construction.md`](docs/phase-3-context-construction.md),
[`docs/phase-4-chat-ui.md`](docs/phase-4-chat-ui.md), and
[`docs/phase-5-conversation-persistence.md`](docs/phase-5-conversation-persistence.md).

## What was done in Phase 5 (this turn)

### New / changed modules

| File | Purpose |
| --- | --- |
| [`src/storage/sqlite.py`](src/storage/sqlite.py) (new) | `SqliteConversationStore` / `SqliteMemoryStore` / `SqliteEvaluationStore`: per-operation connections (Gradio-thread safe), lazy schema (`messages` seq-ordered + indexed, `memories` keyed `(session_id, memory_id)` with JSON payload, `evaluation_runs` session-indexed), recursive `strip_secret_fields`, `BRAINOS_LAB_DB` env path (default `data/brainos_lab.sqlite3`, Git-ignored). |
| [`src/storage/conversations.py`](src/storage/conversations.py) | Protocols finalized: `list_conversations` + `delete_session` (conversation store), `save_memories` upsert (memory store, mirror semantics documented). |
| [`src/app/service.py`](src/app/service.py) | Optional `conversation_store` / `memory_store`; `_persist_message` (redacts the exact session key at the write site) and `_persist_memories` (whole-set guarded upsert each turn); `clear_conversation()` deletes the old conversation's rows + memory mirror; every store call best-effort — failures log only the exception *type* server-side and never break a turn. |
| [`src/app/controller.py`](src/app/controller.py) | Stores are controller-owned **protocol references** attached to the service at creation; `clear_memory()` (runtime + mirror dropped, transcript kept); `end_session()` = Delete session (all persisted rows removed before credentials are cleared, never materializing a store); `export_session()` / `export_session_file()` (sanitized JSON: ids, provider `safe_dict`, context settings, transcript, memories, last-turn accounting). |
| [`src/app/ui.py`](src/app/ui.py) | *Clear memory* + *Export session* buttons and a `gr.File` download slot (10 wired events); `create_app()`'s default factory builds one server-wide SQLite store set — deployment wiring owns the backend choice; security notice updated. |
| [`tests/fakes.py`](tests/fakes.py) | `InMemoryConversationStore`, `InMemoryMemoryStore`, `InMemoryEvaluationStore`, `RaisingStore` (every operation fails — resilience tests). |
| New tests | 16 SQLite-store, 12 persistence-wiring, 4 storage-security, 1 live SQLite integration, extended wiring labels: **184 → 216**. |

### Key design decisions

- **Backend at the edge, protocols at the core.** The first draft had the
  controller lazily materialize SQLite defaults; that would have made every
  headless test write into the repository working tree and coupled the
  controller to an infrastructure choice. Final layering: controller and
  service know only the protocols (`None` = purely in-memory, Phase 4
  behaviour unchanged); `create_app` picks SQLite.
- **Mirror, not source.** The memory table exists for inspection/export; no
  code path hydrates BrainOS memory from it, and `state.messages` remains the
  render source. Storage is written after a turn, never read back mid-turn.
- **Data-control semantics** (all four now exist, as `docs/security.md`
  promised): *Clear conversation* = transcript + runtime + that conversation's
  rows + mirror; *Clear memory* = runtime + mirror, transcript kept; *Export*
  = sanitized read-only JSON; *End session* = credentials + every persisted
  row of the session.
- **Documented caveat for Phase 13:** SQLite `DELETE` frees pages without
  physically scrubbing bytes; the tested contract is that no query can reach a
  deleted session's rows. `VACUUM` / per-session files are hardening
  candidates.

### Live HTTP validation (running server + pinned BrainOS + real SQLite)

Two turns persisted 2 transcript rows and 1 memory row classified
`PROJECT_STATE` by the live runtime; a second session added a `TEMPORAL_EVENT`
row. `/on_export` returned a downloadable JSON (transcript, memories, provider
`safe_dict` without `api_key`, settings, last-turn accounting).
`/on_clear_memory` kept the transcript and emptied the stored panel;
`/on_end` deleted exactly the ended session's rows — a per-session `GROUP BY`
proved another session's rows were untouched.

## What was done in Phase 4

### New modules

| File | Purpose |
| --- | --- |
| [`src/app/controller.py`](src/app/controller.py) | `ChatController`: the session-scoped seam between Gradio callbacks and `ConversationService`. Lazy service creation (page load is BrainOS-optional; a missing runtime yields the install hint on first send), validated settings application, `views()` producing **every** browser-visible value from sanitized service output, session lifecycle (clear / end), and an early turn-limit guard (`DEFAULT_MAX_TURNS = 400` — the Phase 15 down-payment required before benchmark controls). Never imports Gradio, so the whole UI is unit-testable headlessly. |
| [`src/app/ui.py`](src/app/ui.py) (rewritten, 89 → ~380 lines) | Plan §8 layout: BYOK sidebar (provider, model, key, endpoint, temperature, budgets, memory mode), chat, and Memory / Context / Cognitive Trace / Evaluation-placeholder tabs. 8 wired events; all sidebar settings re-applied on every send so edits cannot be forgotten; `_make_chatbot` tolerates Gradio 5 (`type="messages"`) and Gradio 6 (parameter removed). |
| [`tests/unit/test_ui_controller.py`](tests/unit/test_ui_controller.py) | 21 controller cases (panels, settings, gating, errors, lifecycle, isolation, limits). |
| [`tests/unit/test_ui_wiring.py`](tests/unit/test_ui_wiring.py) | 3 structural Gradio-wiring cases (skip without Gradio). |
| [`tests/security/test_ui_secrets.py`](tests/security/test_ui_secrets.py) | 4 browser-boundary secret cases. |
| [`tests/integration/test_ui_live.py`](tests/integration/test_ui_live.py) | 2 live cases on the pinned runtime (skip without it). |

### Behaviour worth remembering

- **The API key travels browser → server only.** No UI output list contains
  it; an empty key field *keeps* the server-side credential instead of
  clearing or echoing it. `apply_provider` drops the cached service (provider
  client) only when the config actually changed, so the BrainOS runtime —
  which lives on `state.brain` — survives provider edits.
- **`apply_context_settings` validates by construction**: a candidate
  `ContextSettings` must build both its `ContextBudget` and its
  `RetrievalPolicy` before it is accepted, so a bad slider value can never
  reach context construction mid-turn (constraint 1: `ContextSettings` is the
  only configuration path).
- **Baseline modes announce themselves.** Selecting `full_context` /
  `sliding_window` / `rag` records the choice on `ContextSettings.mode` and
  returns an explicit "arrives with Phase 6" notice instead of silently
  running (and mislabeling) the BrainOS pipeline.
- **Transcript fidelity vs scrubbing:** chat history and the final prompt are
  deliberately *not* pattern-scrubbed — they are faithful records of what the
  user typed and what was sent to their own provider. The session key is still
  removed from memory text before prompt construction (Phase 3
  `_guard_memories`), and every diagnostic table string is scrubbed.

### Repairs found by Phase 4 tests

- **`turn.error` reached the browser unredacted.** The service stores
  `str(exc)` from caught `ProviderError`s; real adapters redact before
  raising, but fakes or buggy adapters may not — a test pasted the session key
  into an error message and found it verbatim in the status line. The
  controller (the browser boundary) now scrubs error text against the session
  secret and credential shapes: defence in depth even if a future provider
  forgets.
- **`validate_connection` / `list_models` bypassed the service's provider
  seam** when no service existed yet, ignoring injected providers and hitting
  SDK-dependency errors. They now prefer `service.provider()` and fall back to
  `create_provider` only when the BrainOS runtime itself is unavailable.

### Packaging

`gradio>=4.0` → `gradio>=5.0` in `pyproject.toml` (the messages-format
Chatbot arrived in 4.44; validated against Gradio 6.27.0). `tests/fakes.py`:
`FakeProvider` now implements the full `LLMProvider` protocol
(`list_models`, `validate_credentials`, and an `error` mode) — existing
positional construction unchanged.

## What was done in Phase 3

### New modules

| File | Purpose |
| --- | --- |
| [`src/brain/retrieval_policy.py`](src/brain/retrieval_policy.py) | The retrieval pipeline: deduplication, IDF-weighted relevance, conflict resolution, recency weighting, ranking, memory-text guarding, and the audit report. |
| [`src/brain/tokenizers.py`](src/brain/tokenizers.py) | Token counting strategies (`estimate_tokens`, `char_ratio_counter`, `whitespace_counter`, optional `tiktoken_counter`, `provider_counter`) and token-aware `truncate_to_tokens`. |
| [`tests/unit/test_retrieval_policy.py`](tests/unit/test_retrieval_policy.py) | 34 policy cases. |
| [`tests/unit/test_tokenizers.py`](tests/unit/test_tokenizers.py) | 13 counting/truncation cases. |
| [`tests/unit/test_adapter_signals.py`](tests/unit/test_adapter_signals.py) | 17 adapter-mapping cases. |
| [`tests/integration/test_context_pipeline.py`](tests/integration/test_context_pipeline.py) | 12 end-to-end cases, 2 against the live pinned runtime. |

### Rewritten / extended

- **`src/brain/context_builder.py` (rewritten).** `ContextBudget` is validated
  and *enforced*: `max_tokens` is a hard ceiling with the eviction order
  *current message never dropped → oldest history → lowest-ranked memories →
  system prompt truncated last*. `memory_budget` bounds the memory block as sent
  (preamble and delimiters included), filled strictly in rank order.
  `ContextStats` grew from 6 to 26 fields, including
  `full_context_reference_tokens` (the Mode A baseline price for the same system
  prompt and question) and per-stage drop counts. `BuiltContext` now carries the
  `RetrievalReport` and the ranking, plus `final_prompt()` and `to_dict()`.
- **`src/brain/adapter.py`.** `MemoryRecord` gained `entities`, `observed_at`,
  `confidence`, `salience`, `utility`, `valid_from/valid_until`, `supersedes`,
  `contradicts`, `status`, `signals`, `type_source`. New `Conflict` record and
  `conflicts()`, `stale_memory_ids()`, `enrich_with_explanation()` methods, plus
  observation bookkeeping that restores the application's memory type and
  conversation turn onto recalled records.
- **`src/app/state.py`.** `ContextSettings` holds the budget *and* policy knobs
  and builds both (`context_budget()` / `retrieval_policy()`), so the UI and the
  evaluation runner configure context construction through one object.
- **`src/app/service.py`.** recall → enrich with `why()` signals → pass runtime
  conflicts/stale ids → build context with the current turn; `context_stats`,
  `context_report`, and `memory_ranking` are exposed on the turn and in
  sanitized diagnostics.
- **`src/brain/trace.py`, `src/brain/memory_policy.py`, `tests/fakes.py`** — see
  the two repairs below.

### Retrieval policy behaviour (defaults)

```text
dedupe          exact on stemmed token sequence, plus Jaccard ≥ 0.88 near-duplicates
relevance       floor 0.12 absolute (query relevance only — recency cannot rescue
                an off-topic memory) + 0.55 relative tail trim against the best
lexical         IDF-weighted query coverage 0.60 / Jaccard 0.20 / bigram 0.20,
                + up to 0.15 entity bonus
runtime signal  BrainOS why() signals blended; query-dependent subset
                (lexical/semantic/task_relevance) gates the floor, the rest ranks
ranking score   0.45 runtime / 0.35 lexical / 0.20 recency, renormalized over
                the components actually available
recency         0.5 ** (age / half_life); 48 h timestamp (matches the pinned
                runtime), 20 turns, then 4 recall positions
conflicts       runtime contradictions()/stale_memories() first; heuristic
                subject→value claims only drop an older claim when the newer one
                carries a correction marker or a CORRECTION type, else "contested"
cap             max_memories 12
guard           delimiter breakouts, control chars, and role prefixes removed;
                instruction-override patterns flagged (kept unless opted out)
```

### Repair 1 — credential leak (found by a Phase 3 security test)

Phase 3 added browser-visible surfaces (dropped-memory audit text, ranking
components). A test that pasted a credential into memory showed it echoed back
in diagnostics: `brain/trace.py` scrubbed credential *shapes* but cannot
recognise an arbitrary session key. `redact_text` / `sanitize_value` /
`sanitize_trace` now accept the caller's known `secrets`, and
`ConversationService` redacts the active key from every browser-visible value
**and** from recalled memory text before it enters a prompt — otherwise a key
pasted into one conversation could be replayed to a different provider later.

### Repair 2 — Phase 2 memory policy (prerequisite, out of Phase 3 scope)

Live validation exposed that durable facts were never stored. `_classify`
required an `is/are/uses/has/was/were` verb, so "Deployments happen every Friday
at 17:00 UTC" and "The retention policy requires 400 days of audit logs"
produced no candidate — while chit-chat containing "are" was stored eleven
times. **No context engine can retrieve a fact that was never stored**, so this
Phase 2 file was repaired during Phase 3 and is logged as such:

- `_STATEMENT_RE` covers common declarative verbs (`happen`, `requires`, `runs`,
  `retains`, `scheduled`, `migrated`, …);
- questions and requests are rejected (`?` anywhere, interrogative openers);
- conversational filler is rejected by pattern;
- multi-sentence turns are classified per sentence.

Measured on the 60-turn validation conversation:

```text
durable facts stored     4/6  →  6/6
chit-chat stored         5/5  →  0/5
memories after 60 turns    15 →   6   (11 were duplicate small talk)
```

### Repository lint baseline

The 13 pre-existing scaffold findings (import sorting, `typing.Callable`,
`E501`, one `E402`) were fixed in this turn. `ruff check .` is now clean across
the **whole repository**, not just the touched files.

## Phase 3 implementation contract

Context construction is one call, used identically by the chat service and (in
later phases) by every baseline mode:

```python
build_context(
    system_instructions=...,      # str
    current_user_message=...,     # str
    recent_conversation=...,      # Iterable[{role, content}]
    memories=...,                 # Iterable[MemoryRecord] from adapter.recall()
    budget=ContextBudget(...),    # enforced, validated
    policy=RetrievalPolicy(...),  # dedupe/relevance/conflict/recency knobs
    conflicts=adapter.conflicts(),        # BrainOS contradictions()
    stale_ids=adapter.stale_memory_ids(), # BrainOS stale_memories()
    current_turn=...,             # recency fallback when timestamps are absent
    token_counter=...,            # optional; named in the accounting
    now=...,                      # optional; inject for reproducible runs
) -> BuiltContext(messages, selected_memories, stats, report, ranking)
```

Invariants other phases may rely on:

- `messages` is provider-neutral (`{role, content}`), roles normalized to
  `system|user|assistant|tool`, and the last message is always the current user
  message — never dropped, never truncated.
- Memory lives in exactly one `system` message between `<retrieved_memory>`
  delimiters, introduced as untrusted data. The delimiters appear once each and
  close the block, even against hostile memory text.
- `stats.to_dict()` always contains the plan's five required fields plus the
  full-context reference and the derived reduction ratios.
- `report.to_dict()` lists every dropped memory with a reason, so Precision@K
  and the error taxonomy can be computed without re-running retrieval.
- Ranking is deterministic: ties break on recall position, then memory id.

Audit reasons → Phase 12 error taxonomy mapping:

| Reason | Taxonomy |
| --- | --- |
| `low_relevance`, `weak_relevance` | `irrelevant_memory` |
| `stale`, `expired` | `stale_memory` |
| `superseded` | `conflicting_memory` |
| `memory_budget`, `budget`, `cap` | `over_compression` |
| `duplicate`, `empty` | (not a failure — noise removed) |
| `suspicious` | injection guard, scored in Phase 13 |

## Measured behaviour (live pinned BrainOS, 60-turn conversation)

Six durable facts distributed across 60 turns, including a correction at turn
44, between five recurring chit-chat turns. Dependency-free token estimator; no
provider API key used.

| Configuration | mean tokens sent | full-context baseline | mean reduction |
| --- | --- | --- | --- |
| wide window (`recent_turn_budget=2048`) | 1486 | 1391 | **0.0%** |
| memory-first (`192`, `memory_budget=512`, `max_memories=6`) | 387 | 1391 | **72.2%** |
| memory-first tight (`128`, `384`, `4`) | 308 | 1391 | **77.8%** |

Retrieval quality: **6/6** questions had exactly the evidence they needed, with
the correct memory at rank 1 every time. The corrected fact (PostgreSQL →
MySQL 8) never leaked into a production-database prompt at any length.

**These are integration-validation observations, not research results.** They
come from one synthetic conversation with an estimated token counter and no
model in the loop.

### Finding that constrains Phase 6

**Context reduction is driven almost entirely by the recent-history window, not
by memory selection.** With `recent_turn_budget` larger than the conversation,
"BrainOS mode" degenerates into full context *plus* memory overhead: 0.0%
reduction, 95 tokens *worse* than the baseline. Phase 6 must set
`recent_turn_budget` per baseline mode, or Mode A and Mode D will not be
distinguishable and the comparison will be meaningless.

### Known limitation: abstention is not yet achieved

For a question whose answer is absent ("What is the Project Atlas payroll
vendor?"), the best candidate scored relevance `0.344` against the `0.12`
absolute floor, so 4 memories were selected instead of none. The relative floor
trimmed the tail (a true-match query went from 5 selected to 2) but cannot
produce abstention: when nothing matches, there is no strong head to compare
against.

The absolute floor was deliberately **not** tuned to fix this. The measured
separation (true match ≈ `0.56` relevance / `0.90` lexical vs best-of-nothing
≈ `0.34` / `0.32`) comes from one synthetic conversation; calibrating a
threshold on it would be overfitting. Threshold calibration belongs to Phases
7–8, where abstention accuracy is a scored category.

## Security decisions carried forward

- API keys remain excluded from `ProviderConfig` representations and
  `safe_dict()` diagnostics.
- Provider error messages are redacted using the active key and common bearer,
  API-key, token, and OpenAI-key patterns.
- **New:** the active session key is redacted by exact match from every
  browser-visible value the service produces (diagnostics, inspection, reports,
  rankings, traces) and from recalled memory text before it enters a prompt.
- **New:** retrieved memory is guarded before rendering — delimiter breakouts,
  control characters, and `system:`-style role prefixes are stripped, text is
  collapsed to one bullet and length-bounded, and instruction-override patterns
  are flagged (`drop_suspicious_memories` opts into removal).
- **New (Phase 4):** the API key travels browser → server only — no UI output
  contains it, and an empty key field keeps the server-side credential instead
  of echoing one back.
- **New (Phase 4):** the controller redacts `turn.error` and every
  runtime-derived table string (retrieved/dropped/conflict text, decision
  reason) against the session key and credential shapes before rendering. Chat
  transcript and final prompt stay faithful by design (see above).
- **New (Phase 4):** no traceback ever reaches the browser — missing BrainOS
  becomes an install hint, provider failures become redacted one-line statuses,
  and page load never constructs a runtime.
- **New (Phase 5):** the active session key is redacted from message and
  memory text **at the write site** before anything reaches a store; stores
  additionally strip secret-named fields recursively from every metadata and
  payload blob. Raw-database-byte tests prove a session that types its key
  into chat cannot persist it.
- **New (Phase 5):** all store reads and deletes filter by exact
  `session_id` (+ `conversation_id`); *End session* deletes every persisted
  row of the session (transcript, memory mirror, evaluation runs). A failing
  storage backend never breaks a turn and is logged server-side by exception
  type only — never content.
- BrainOS explanations and traces drop secret-named fields and scrub
  credential-shaped strings. Trace mapping records counts, not retrieved memory
  text.
- Session cleanup replaces the shared frozen provider configuration with a
  credential-free copy and drops the session BrainOS adapter. An adapter the
  service created is replaced on `clear_conversation()`; an injected adapter is
  retained (test seam), so clearing memory is a separate operation from clearing
  a conversation.
- The provider endpoint is supplied by the active session. Never expose the
  session key in browser diagnostics, evaluation artifacts, or traces.

## Validation baseline

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install pytest ruff "gradio>=5.0"
.venv/bin/pip install "brainos-cli @ git+https://github.com/NiravRVaghasiya/BrainOS.git@1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc"

.venv/bin/pytest -q
# 216 passed

.venv/bin/ruff check .
# All checks passed!   (whole repository, no exclusions)

.venv/bin/python app.py
# Gradio UI on 0.0.0.0:7860 — works without BrainOS (install hint on send)
```

Test count went 184 → **216** in this phase (37 → 154 → 184 → 216 across
Phases 3–5). The Gradio-dependent wiring tests skip without the `ui` extra;
the live tests in `tests/integration/` run against the pinned BrainOS revision
and skip when `brainos_runtime` is not installed. Unit tests never touch disk:
SQLite-store tests use `tmp_path`, wiring tests use in-memory doubles, and a
bare controller has no stores. No provider API key was used anywhere;
generation is exercised through `FakeProvider`.

The running server was additionally validated **over real HTTP** (Gradio
client API + real SQLite on disk + live BrainOS runtime): turns persisted with
live memory classification (`PROJECT_STATE`, `TEMPORAL_EVENT`), export
downloaded as sanitized JSON, clear-memory kept the transcript, and
end-session deleted exactly its own session's rows.

`.venv` and `data/*.sqlite3` are ignored and are only local runtime artifacts.

## Next safe step

Implement **Phase 6 — Baseline modes**: the five context-management strategies
the scientific comparison depends on (plan §10), routed through the mode
selector that Phase 4 already records and Phase 5 already persists:

```text
Mode A  Full Context      whole history, no memory
Mode B  Sliding Window    last N turns only
Mode C  Conventional RAG  embedding/lexical top-k over history, no BrainOS
Mode D  BrainOS           observe/recall + retrieval policy (current pipeline)
Mode E  BrainOS + RAG     D plus semantic retrieval over raw history
```

| Piece | Available now |
| --- | --- |
| Mode selector | `ContextSettings.mode` — set by the UI dropdown, recorded per session, exported; the UI already announces pending modes honestly |
| Context construction | `build_context(...)` is mode-agnostic: modes differ in *what they feed it* (history window, memory source), not in the builder |
| The constraining finding | Phase 3 measured that reduction is driven by `recent_turn_budget`: with a wide window, Mode D degenerates into full context *plus* memory overhead (0.0% reduction, 95 tokens worse). **Each mode must configure its own budget/window**, or A and D are indistinguishable |
| Runner shell | `evaluation/runner.py` raises `NotImplementedError("Strategy execution is not wired yet...")` and accepts strategy callables — the seam modes should plug into |
| Persistence for results | `EvaluationStore.save_run` (SQLite, session-indexed) awaits the Phase 16 metadata block |
| Mode C retrieval | `retrieval_policy.lexical_score` / IDF helpers are dependency-free building blocks; the plan forbids complex vector DBs in v1 |

Carry these constraints into Phase 6:

1. Only the context-management strategy may differ between modes — model,
   temperature, generation parameters, task wording, and evaluation procedure
   stay constant (plan §15).
2. Keep modes behind one strategy interface consumed by both the chat service
   (the UI dropdown) and the evaluation runner; no `brainos_runtime` import
   outside `brain/adapter.py`.
3. Mode A must send the *same* system prompt as the other modes so the
   full-context reference stays comparable with `full_context_reference_tokens`.
4. Modes C/E need a corpus: derive it from the same transcript/memory data the
   service already holds; do not add a vector database (plan §31).
5. Update the UI's mode notice only when a mode becomes real — the honest
   "arrives with Phase 6" notice must disappear mode-by-mode as they land.
6. Per-mode accounting must populate the same 26-field `ContextStats` so
   Phase 8 metrics can compare modes without special cases.
