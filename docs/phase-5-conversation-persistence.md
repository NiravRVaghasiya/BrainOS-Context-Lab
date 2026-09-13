# Phase 5 log — Conversation persistence

**Status:** complete (branch `arena/01a09bc4-brainos-context-lab`, PR #6)
**Plan section:** Phase 5 — Conversation persistence
**Depends on:** Phase 4 (chat web UI, merged to `main` via PR #5) and every
phase before it.

> **Merge note.** PR #6 originally carried this branch's own Phase 4
> implementation (`ChatController` + rewritten Gradio UI) with Phase 5 built on
> top of it. While the PR was open, PR #5 merged a parallel Phase 4 — the
> Gradio-free `UIController` + `panels` stack — into `main`. The conflicts were
> resolved by treating **upstream PR #5 as the canonical Phase 4**: its
> controller, panels, UI, and tests were adopted as merged, this branch's
> superseded Phase 4 modules, tests, and log were retired, and Phase 5 was
> ported onto the upstream seams. Everything below describes the *ported*
> result, which is what the branch now contains.

## Objective

From the plan: persist conversations and BrainOS memory so a session survives
process boundaries, give users real data controls (clear / export / delete),
and keep the storage layer strictly credential-free and session-isolated.
Persistence must be best-effort: a storage failure may never break a
conversation turn, and headless runs (CLI, evaluation, unit tests) must never
touch disk unless they opt in.

## Starting-state audit (after the merge)

- Upstream Phase 4 was already complete on `main`: `UIController` with
  `provider_factory`/`adapter_factory` seams, session lifecycle actions
  (clear conversation, clear memory, end session), an in-memory
  `export_session()` snapshot, `panels.py` rendering, and the Gradio UI with a
  `DownloadButton` export. 216 tests passed.
- `src/storage/conversations.py` held this branch's Phase 5 protocol
  finalization (`list_conversations`, `delete_session`, `save_memories`
  upsert) — kept unchanged by the merge.
- `src/storage/sqlite.py`, the persistence test files, and this log are this
  branch's Phase 5 contribution — kept, with the app-layer wiring and the
  tests retargeted to the upstream controller/service API.

## Work completed

### 1. Protocol finalization (`src/storage/conversations.py`)

Unchanged from this branch's Phase 5 work and kept by the merge:
`ConversationMessage` dataclass; `ConversationStore` protocol with `append`,
`list_messages`, `clear(session_id, conversation_id)`, `list_conversations`,
`delete_session`; `MemoryStore` protocol with `save_memories` (upsert mirror
semantics documented on the protocol), `list_memories`, `clear`.

### 2. SQLite backend (`src/storage/sqlite.py`, new)

- `SqliteConversationStore`, `SqliteMemoryStore`, `SqliteEvaluationStore` on a
  shared `SqliteStore` base: **per-operation connections** (safe across
  Gradio's threadpool; no cross-thread connection reuse), lazy schema creation
  on first use.
- Schema: `messages` (sequence-ordered, indexed by
  `(session_id, conversation_id)`), `memories` (mirror keyed
  `(session_id, memory_id)` with the full sanitized record as JSON payload),
  `evaluation_runs` (session-indexed, ready for Phase 16).
- `strip_secret_fields` recursively removes secret-named keys (`api_key`,
  `authorization`, `token`, …) from any metadata/payload before it is written.
- Database path: `BRAINOS_LAB_DB` environment variable, default
  `data/brainos_lab.sqlite3` (Git-ignored). Constructed once per deployment in
  `create_app`, never lazily inside the controller.

### 3. Service persistence hooks (`src/app/service.py`)

Ported onto the upstream service (which already had `reset_provider()` /
`reset_memory()` / `stored_memories()`):

- Optional `conversation_store` / `memory_store` constructor parameters;
  `None` (the default) keeps the service purely in-memory — upstream Phase 4
  behaviour is byte-for-byte unchanged without stores.
- `handle_user_message` persists the user message on append and the assistant
  reply when generated (`_persist_message`), then mirrors the whole guarded
  memory set (`_persist_memories`, upsert by `memory_id`, so retrieval
  counters and status transitions stay current without per-turn growth).
- `_redact_for_storage` replaces the exact session key with `[redacted]` at
  the write site; memory records pass through the same `_guard_memories`
  credential guard as recalled memories.
- `clear_conversation()` captures the old `conversation_id`, deletes that
  conversation's rows, then resets memory; `reset_memory()` also clears the
  persisted mirror **regardless of the injected-adapter test seam** — when the
  user asks the app to forget, the durable copy forgets too.
- Every store call is best-effort: failures are logged server-side with the
  exception **type only** (`_warn_storage` — exception text could carry user
  content) and never break a turn.

### 4. Controller lifecycle + export (`src/app/controller.py`)

Ported onto the upstream `UIController`:

- The controller owns the three store protocols (inject-or-`None`) and passes
  them to each session's `ConversationService` at creation. It never
  materializes a backend itself.
- `end_session()` now deletes **every persisted row of the ended session**
  (conversation rows, memory mirror, evaluation runs) before dropping the
  in-process session, and its status says so.
- `export_session()` keeps the upstream payload (ids, provider `safe_dict`,
  context payload, live transcript, live memories) and adds a `persistence`
  block: `enabled`, every persisted conversation of the session with its
  messages, and the memory mirror. Reads are best-effort — a failing backend
  downgrades the export instead of failing it. The UI's existing
  `DownloadButton` serves it as JSON with no UI changes.

### 5. UI (`src/app/ui.py`)

- `create_app()`'s **default** controller is constructed with one server-wide
  SQLite store set — the backend choice lives at the deployment edge; an
  injected controller keeps exactly the stores its caller gave it (tests pass
  fakes and never touch the real database).
- Header text now discloses persistence honestly: keys are still never written
  to disk, transcripts and memory mirrors are persisted with credentials
  redacted, and the clear/end controls delete the matching rows.

### 6. Merge repairs (small, deliberate changes to upstream code)

- `ConversationService.reset_memory()` recreates the runtime through
  `_new_adapter()` so an injected `adapter_factory` is honoured after a memory
  clear (upstream called `create_brain_adapter` directly there, bypassing its
  own seam).
- `ui` extra floor raised to `gradio>=6.0`: the upstream UI renders message
  dicts natively (Gradio 6 default), while Gradio 5 requires the
  `type="messages"` parameter it no longer passes.
- Lifecycle status strings now mention persisted-row deletion ("persisted rows
  deleted too", "persisted mirror too", "all persisted session rows dropped").

## Data-control semantics (defined for Phase 13/14)

| Control | In-process effect | Persisted effect |
| --- | --- | --- |
| Clear conversation | transcript emptied, runtime dropped, new `conversation_id` | that conversation's rows deleted + memory mirror cleared |
| Clear memory | runtime dropped, transcript kept | memory mirror cleared, transcript rows kept |
| Export session | — | read-only sanitized JSON: live views + `persistence` block |
| End session | credentials cleared, session dropped, fresh session started | **every** row of the ended session deleted (messages, memories, evaluation runs) |

Known caveat (documented, deferred to Phase 13 hardening): SQLite `DELETE`
frees pages without physically scrubbing bytes. The tested contract is
API-level isolation — no query can reach a deleted session's rows; `VACUUM` or
per-session files are the hardening candidates.

## Security tests (the phase's hard requirement)

`tests/security/test_storage_secrets.py` runs the upstream `UIController` on
the **real SQLite backends** with a session that actively tries to persist its
credential:

- raw database **bytes** never contain the session key after it is typed into
  chat; `[redacted]` appears in the transcript instead;
- mirrored memory rows never contain the key (the guarded record is what gets
  written);
- the export JSON (live views *and* the `persistence` block) and every
  rendered `TurnView` surface are key-free;
- secret-named metadata fields cannot be smuggled onto disk
  (`strip_secret_fields` is recursive);
- ending a session removes exactly that session's rows — another session's
  rows provably survive.

## Validation

- `.venv/bin/pytest -q` → **249 passed** (upstream Phase 4's 216 + 15
  SQLite-store + 12 persistence-wiring + 4 storage-security + 2 live-SQLite
  integration tests). `ruff check .` clean.
- Live tests run against the pinned BrainOS runtime and skip without it.
- **Live HTTP validation** (running Gradio server + pinned runtime + real
  SQLite at `data/brainos_lab.sqlite3`, via `gradio_client`): two chat turns
  persisted 2 transcript rows and 1 memory row classified `PROJECT_STATE` by
  the live runtime; the recall panel retrieved the fact; `/export_session`
  downloaded `brainos-context-lab-session-*.json` containing the live payload
  plus `persistence` (1 conversation, 2 messages, 1 memory) with no `api_key`
  anywhere; `/clear_memory` kept the 2-message transcript, emptied the stored
  panel, and left 2 message rows / 0 memory rows on disk; `/end_session`
  deleted the session's rows exactly (0 / 0 afterwards). Dev database rows
  were removed after validation.

## Decisions carried forward

- Backend at the edge, protocols at the core: controller/service know only the
  store protocols; `create_app` picks SQLite. A bare `UIController` /
  `ConversationService` (no stores) is fully functional and purely in-memory —
  the evaluation runner and CLI paths stay disk-free by default.
- Mirror, not source: nothing rehydrates BrainOS memory from the store;
  `state.messages` remains the render source; storage is written after a turn,
  never read back mid-turn.
- Storage failures are invisible to users and typed-only in server logs.
- Upstream UI gap noted for a follow-up: the `/end_session` handler renders
  the fresh session's panels but discards the controller's "Session ended …"
  confirmation string — the richer status exists, the UI just does not surface
  it yet.

## Next

Phase 6 — Baseline modes A–E behind `ContextSettings.mode`, extending the
upstream `MEMORY_MODES` / `uses_memory()` seam, with each mode setting its own
`recent_turn_budget` (the Phase 3/4 finding). See `CONTEXT.md` →
"Next safe step".
