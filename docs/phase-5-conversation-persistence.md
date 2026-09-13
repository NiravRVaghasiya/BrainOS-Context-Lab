# Phase 5 log — Conversation persistence

**Date:** 2026-09-13
**Branch:** `arena/01a09bc4-brainos-context-lab`
**Starting commit:** `2dfd5b3b8ed7e438cd0ae382d6f8773846d56bf7`
**Plan section:** [Phase 5 — Conversation Persistence](../BrainOS_Context_Lab_Implementation_Plan.md#9-phase-5--conversation-persistence)

## Objective

Implement the plan's session-local persistence MVP — SQLite behind the
`ConversationStore` / `MemoryStore` / `EvaluationStore` abstractions — with
hard session isolation, the plan's absolute rule that **keys are never stored
in the database**, and the data-control operations `docs/security.md` already
promised: *Clear conversation, Clear memory, Delete session, Export session*.

The Phase 4 hand-off added five constraints: persistence must not change the
Phase 3/4 contracts; the controller and service stay the only writers; the
BrainOS runtime remains authoritative for recall (the store is a mirror);
SQLite files stay out of Git; and the data-control semantics get defined here,
before Phase 14 deployment needs them.

## Starting-state audit

| Available at Phase 4 | Missing for Phase 5 |
| --- | --- |
| `ConversationStore` / `MemoryStore` protocols (`src/storage/conversations.py`, Phase 0 scaffold) | no implementation of any protocol |
| `EvaluationStore` protocol | no writer/reader; no `session_id` index |
| Protocols covered `append` / `list` / `clear` only | no `list_conversations`, `delete_session`, `save_memories` (needed by Delete-session and the mirror upsert) |
| `SessionState` ids + `conversation_id` rotation on clear | nothing persisted them |
| `security.md` promised four data controls | only *Clear conversation* and *End session* existed |

## Work completed

### 1. Protocol finalization (`src/storage/conversations.py`)

Minimal, documented extensions so the plan's data controls are expressible:

- `ConversationStore.list_conversations(session_id)` and
  `ConversationStore.delete_session(session_id)` — *Delete session* removes
  every conversation of a session, not just the active one.
- `MemoryStore.save_memories(session_id, memories)` — an **upsert by
  `memory_id`**, so mirroring the same memory each turn updates counters
  instead of duplicating rows. The docstring pins the mirror semantics:
  BrainOS stays authoritative for recall.

### 2. SQLite backend (`src/storage/sqlite.py`, new)

Three stores over one shared base:

- **Thread safety by construction** — Gradio runs callbacks in worker threads,
  so every operation opens and closes its own connection (`timeout=30`); no
  shared connection, no locking bugs. The database file is server-wide;
  isolation is row-level via exact `session_id` / `conversation_id` filters on
  every read and delete.
- **Lazy schema** (`CREATE TABLE IF NOT EXISTS` on connect): `messages`
  (append-only, `seq AUTOINCREMENT` preserves order, indexed by
  session+conversation), `memories` (primary key `(session_id, memory_id)`,
  indexed columns plus the full sanitized record as JSON `payload`), and
  `evaluation_runs` (run id primary key, session-indexed for
  `delete_session_data`) — the Phase 16/17 writer already has its home.
- **Secrets never reach disk** — `strip_secret_fields` recursively drops
  secret-named keys from every metadata/payload blob at the store level
  (defence in depth over service-layer sanitization). Strings are deliberately
  *not* pattern-scrubbed here: transcript fidelity is a documented decision,
  and the session key itself is redacted by the caller before writing.
- **Location** — `BRAINOS_LAB_DB` env override, default
  `data/brainos_lab.sqlite3`; the existing `*.sqlite3` ignore rule keeps
  runtime data out of Git, and the env var gives HF Spaces a writable path.
- Unicode, JSON round-trips, and reopened-database persistence are tested.

### 3. Service persistence hooks (`src/app/service.py`)

`ConversationService` gained optional `conversation_store` / `memory_store`
parameters — `None` keeps the Phase 4 purely-in-memory behaviour, so no
existing caller or test changed semantics.

- `_persist_message` appends user and assistant messages with
  `{turn: n}` metadata, after `_redact_for_storage` replaces the **exact
  session key** in content (the plan's "never store keys in the database",
  enforced at the write site).
- `_persist_memories` mirrors the whole current memory set each turn through
  `_guard_memories` (same session-key guard as recalled memories), upserted by
  id — counters and status transitions (stale/superseded) stay current without
  diffing.
- `clear_conversation()` now captures the pre-rotation `conversation_id` and
  deletes that conversation's rows plus the session's memory mirror, so
  *Clear conversation* means the same thing in the store as in process memory.
- **Best-effort discipline**: every store call is wrapped; a failing backend
  logs one safe server-side line (`_warn_storage` prints the exception *type*
  only — never content) and the turn proceeds. Storage never drives behaviour.

### 4. Controller lifecycle + export (`src/app/controller.py`)

- Stores are controller-owned and attached to the service at creation (also
  when a test factory built the service without them). The controller knows
  only the **protocols** — backend choice is deployment wiring, so a bare
  `ChatController` (scripts, unit tests) never touches disk. This replaced the
  first design, in which the controller lazily materialized SQLite defaults:
  it would have made every headless test write `data/…sqlite3` into the
  repository working tree.
- **`clear_memory()`** — the plan's distinct *Clear memory* control: drops the
  BrainOS runtime (`state.brain = None`, service cache released) and the
  persisted mirror, **keeps the transcript**; the next turn lazily creates a
  fresh runtime through the existing seam.
- **`end_session()` = Delete session** — before clearing credentials, deletes
  every persisted row for the session (transcript, memory mirror, evaluation
  runs). Deletion never materializes a store: a session that persisted nothing
  touches no disk.
- **`export_session()` / `export_session_file()`** — reads the persisted
  mirrors (falling back to process state when none exist) and returns a
  sanitized JSON payload: ids, provider `safe_dict()`, full context settings,
  turn counters, transcript, memories, and the last turn's stats/ranking/drop
  audit. The file variant writes a temp JSON for browser download.
- `create_app()`'s default factory now constructs one server-wide SQLite store
  set and passes it to every session's controller.

### 5. UI (`src/app/ui.py`)

Sidebar Session section grew *Clear memory* and *Export session* buttons plus
a `gr.File` download slot (two new wired events, ten total); the security
notice now states what is persisted, that it is row-isolated, and that *End
session* deletes it together with the key.

## Data-control semantics (defined for Phase 13/14)

| Control | Process memory | SQLite |
| --- | --- | --- |
| Clear conversation | transcript, diagnostics, BrainOS runtime dropped; `conversation_id` rotates; settings kept | that conversation's rows + session memory mirror deleted |
| Clear memory | BrainOS runtime dropped, transcript kept | memory mirror deleted, transcript rows kept |
| Export session | unchanged | read-only; sanitized JSON download |
| End/Delete session | credentials cleared, session unregistered, fresh session issued | every row of the session deleted (transcript, memories, evaluation runs) |

Known caveat, documented for Phase 13 hardening: SQLite `DELETE` frees pages
but does not physically scrub them; byte-level removal would need a `VACUUM`
(or per-session files). The isolation contract — no query can reach a deleted
session's rows — is tested.

## Security tests (the phase's hard requirement)

`tests/security/test_storage_secrets.py` asserts at the strongest level
available: after a session that *types its live credential into the chat*,

- the **raw bytes** of the SQLite file do not contain the key, while
  `[redacted]` does (transcript survives, key does not);
- mirrored memory rows and the export JSON/file are key-free;
- every diagnostic view surface is key-free — with `history` and
  `final_prompt` as the documented faithful surfaces (they show the user what
  they typed and what was sent to their own provider, per the Phase 4
  decision);
- secret-named metadata cannot be smuggled onto disk even when nested
  (`{"note": {"authorization": …}}` → `{"note": {}}`);
- deleted sessions are unreachable through every store API while other
  sessions' rows survive the delete.

## Validation

```bash
.venv/bin/pytest -q      # 216 passed   (184 → 216)
.venv/bin/ruff check .   # All checks passed!  (whole repository)
```

New tests: 16 SQLite-store cases (`tests/unit/test_storage_sqlite.py`), 12
persistence-wiring cases on in-memory doubles
(`tests/unit/test_persistence.py`, including `RaisingStore` proving a failing
backend never breaks a turn), 4 storage-security cases, 1 live-runtime
SQLite round-trip (`tests/integration/test_ui_live.py`), extended wiring
labels (10 events). `tests/fakes.py` gained `InMemoryConversationStore`,
`InMemoryMemoryStore`, `InMemoryEvaluationStore`, and `RaisingStore`.

**Live HTTP validation** (running Gradio server + pinned BrainOS + real
SQLite on disk, no API key):

- two turns persisted 2 transcript rows and 1 memory row classified
  `PROJECT_STATE` by the live runtime; a follow-up session added a
  `TEMPORAL_EVENT` row ("Deployments happen every Friday at 17:00 UTC");
- `/on_export` returned a downloadable JSON with transcript, memories,
  provider `safe_dict` (no `api_key` field), context settings, and last-turn
  accounting;
- `/on_clear_memory` kept the transcript (history intact, stored rows empty);
- `/on_end` deleted exactly the ended session's rows — a per-session `GROUP
  BY` proved the orphaned earlier session's rows were untouched and the ended
  session's were gone.

## Decisions carried forward

- **Protocol-first storage**: the evaluation runner (Phase 17) should write
  through `EvaluationStore.save_run` with the Phase 16 reproducibility
  metadata block; the schema already carries `session_id` for
  `delete_session_data`.
- **Mirror, not source**: no code path may hydrate BrainOS memory from the
  store; rows exist for inspection, export, and later offline analysis.
- The in-process turn transcript (`state.messages`) remains the render source;
  the store is written after, never read back mid-turn.
- A bare `ChatController`/`ConversationService` (no stores) is fully
  functional in memory — evaluation harnesses can opt out of disk entirely.

## Next

Phase 6 — Baseline modes: implement Mode A (full context), B (sliding
window), C (conventional RAG), D (BrainOS), E (BrainOS + RAG) behind
`ContextSettings.mode`, **setting `recent_turn_budget` per mode** — the Phase
3 finding showed Mode A and Mode D are otherwise indistinguishable — and wire
the strategies into the `evaluation/runner.py` shell.
