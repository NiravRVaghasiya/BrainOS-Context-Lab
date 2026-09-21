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

### Locations and retention

Every repository-owned location — the committed dataset, the dataset root, the
browser run artifacts under `results/ui/`, the CLIs' `--dataset` default, and the
unset SQLite default — is resolved through `src/checkout.py`. Inside the checkout
the paths stay exactly as the plan writes them (relative), which is what keeps
artifacts and re-run commands portable between machines; a process started
elsewhere (an installed console script, a supervisor with its own working
directory) anchors them to the checkout instead of creating a second `results/`
wherever it happened to start. `BRAINOS_LAB_DB` still overrides the database
location, and `:memory:` still means "no file".

`src/app/retention.py` owns the lifetime of `results/ui/`: a session directory is
removed when its **newest** file is older than the retention window (7 days by
default, `None` disables), oldest first and capped per sweep, and only if it is a
direct child of `ui/`, a single safe path segment, not a symlink, and still
resolves inside that directory. The sweep runs when a browser run starts and from
`python -m app.retention`; each run's manifest records the window it ran under and
what the sweep removed. **End session** still deletes a visitor's own rows and
artifacts immediately.

## Baseline modes

The evaluation layer supports the same provider and task protocol for all five
of the plan's strategies (Phase 6), selected by `ContextSettings.mode` and
registered in `src/baselines/modes.py`:

| Mode | Selector | Evidence | History window |
| --- | --- | --- | --- |
| A | `full_context` | none | unlimited (whole `max_tokens` ceiling¹) |
| B | `sliding_window` | none | last 8 turns |
| C | `rag` | lexical top-k transcript chunks | last 2 turns / 256 tokens |
| D | `brainos` | BrainOS recall | last 2 turns / 256 tokens |
| E | `brainos_rag` | BrainOS recall + lexical chunks | last 2 turns / 256 tokens |

¹ In the product the ceiling is the session's `max_tokens` (4,096 by default).
A benchmark replay sizes that ceiling to the transcript it is about to replay
(`evaluation.modes.replay_ceiling`), because a reference that stops growing at
4k tokens makes every reduction above 4k an artefact — Phase 20 measured exactly
that before the fix.

Only the context-management strategy changes in a controlled comparison. Two
rules keep the comparison meaningful, both derived from the Phase 3 measurement
that reduction is driven by the history window rather than by memory selection:

1. **Each mode fixes its own history window.** Switching modes rewrites
   `recent_turn_budget`, `max_recent_turns`, and the evidence budgets through
   `ContextSettings.with_mode_defaults()`, so Mode A and Mode D can never
   accidentally share a window larger than the conversation.
2. **Modes C, D, and E share one evidence allowance** (`EVIDENCE_BUDGET =
   1024` tokens; Mode E splits it between its two sources). They differ in
   *what selects* the evidence, so volume is held constant.

Mode C's retriever (`src/baselines/rag.py`) is deliberately BrainOS-free and
embedding-free: it chunks the transcript at sentence boundaries and ranks chunks
by IDF-weighted lexical overlap, reusing the same lexical machinery as memory
retrieval. `src/baselines/` never imports the BrainOS runtime — asserted in
`tests/security/test_mode_secrets.py` — because Mode C is only a baseline if it
does not use the system under test.

BrainOS still observes in every mode; only *injection* is mode-dependent, so a
user can switch strategies mid-conversation without losing memory.

`evaluation/modes.py` replays a benchmark task through any mode using the same
service and builder the chat path uses, which is the seam the Phase 17 pipeline
plugs into. Phase 7 added the two halves on either side of that seam:

- `benchmarks/context_rot/` generates the conversations (`spec.py` for content,
  `generation.py` for structure) and pins the committed dataset with a manifest;
- `evaluation/datasets.py` carries the fact ledger and evidence contract, and
  `evaluation/scoring.py` turns a replay into `retrieval` + `answer` verdicts,
  which `evaluation/runner.py` aggregates into a run's metrics.

The dependency direction is one-way: `benchmarks` imports `evaluation.datasets`
(the schema), never the other way round, and nothing under `src/` imports the
benchmark package. A benchmark is data; the application does not know it exists.
