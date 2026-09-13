# Architecture

## Scope

BrainOS Context Lab is an application around the upstream BrainOS runtime. BrainOS remains responsible for cognitive memory operations; this repository is responsible for providers, sessions, context orchestration, UI, storage, and evaluation.

```text
Browser
  │
  ▼
Gradio UI
  │
  ▼
Application/session layer
  ├── BrainMemoryAdapter ──► BrainOS runtime
  ├── LLMProvider ─────────► Selected provider
  ├── Context builder
  ├── Conversation/memory stores
  └── Evaluation logger
```

## Request lifecycle

1. Receive a user message inside an isolated session.
2. Apply the memory policy to information eligible for observation.
3. Ask the BrainOS adapter to observe eligible content.
4. Recall memories relevant to the current query.
5. Deduplicate, filter, account for conflicts/recency, and apply the context budget.
6. Send the resulting provider-neutral message list to the selected LLM provider.
7. Render the answer and sanitized inspection data.
8. Observe the assistant response when the memory policy permits it.
9. Record token accounting and evaluation metadata without secrets.

## Boundaries

### BrainOS adapter

The adapter is the only application dependency on BrainOS. It exposes `observe`, `recall`, `decide`, `explain`, and `trace`. Phase 2 maps those methods onto the pinned v2 runtime (`source`/`event_type`, `top_k`, decision strings, `why()`, structured traces) and constructs one runtime per `session_id`/`actor_id`. `ConversationService` is the application service layer that combines the adapter with the provider factory and context builder.

### Provider adapter

The provider interface normalizes model listing, credential validation, and generation. Provider-specific SDK objects must not leak into the context builder or evaluation runner.

### Context builder

`brain/retrieval_policy.py` and `brain/context_builder.py` together form the
context construction engine (Phase 3). The policy runs the retrieval pipeline;
the builder allocates the token budget and renders provider-neutral messages.

```text
BrainOS recall
  → deduplicate (exact + near-duplicate Jaccard)
  → relevance filter (IDF-weighted lexical + runtime signals; absolute and
    relative floors — recency ranks but never rescues an off-topic memory)
  → conflict check (runtime contradictions()/stale_memories() first, then a
    conservative subject→value heuristic; unresolved pairs are kept "contested")
  → recency weighting (timestamp → conversation turn → recall position)
  → token budget (memory_budget in rank order, then the max_tokens ceiling)
  → delimited message list + accounting + audit report
```

BrainOS stays authoritative: the engine consumes the runtime's own retrieval
signals, lifecycle status, supersession links, and contradiction reports, and
only falls back to application heuristics when the runtime reports nothing.

The builder returns model-ready messages and explicit accounting fields:

- raw history tokens,
- selected recent-history tokens,
- retrieved-memory tokens,
- system tokens,
- final context tokens,
- the full-context (Mode A) reference price for the same prompt, and
- per-stage drop counts with a reason for every dropped memory.

`max_tokens` is a hard ceiling with a documented eviction order: the current
user message is never dropped or truncated, then oldest history, then
lowest-ranked memories, then the system prompt is truncated.

Retrieved memory is delimited and explicitly described as data rather than
instructions. Memory text is additionally stripped of delimiter breakouts,
control characters, and role prefixes before rendering.

Because context reduction depends mostly on how much raw history is replayed,
the baseline modes (Phase 6) must configure `recent_turn_budget` per mode —
otherwise "BrainOS mode" is indistinguishable from full context plus overhead.

### Storage

Session-local persistence is implemented with SQLite behind the
`ConversationStore` / `MemoryStore` / `EvaluationStore` protocols
(`src/storage/`); the backend is chosen at deployment wiring
(`app.ui.create_app`), and the database location is configurable via
`BRAINOS_LAB_DB` (default `data/brainos_lab.sqlite3`, Git-ignored). API keys
are never part of persisted state: the active session key is redacted from
message and memory text before writing, and secret-named metadata fields are
stripped recursively at the store level. Every record carries exact
session/conversation identity, and all reads and deletes filter on it to
prevent cross-session leakage. The memory table is a mirror for inspection
and export — the BrainOS runtime remains authoritative for recall.

## Baseline modes

The evaluation layer must support the same provider and task protocol for:

- Full Context,
- Sliding Window,
- Conventional RAG,
- BrainOS, and optionally
- BrainOS + RAG.

Only the context-management strategy should change in a controlled comparison.
