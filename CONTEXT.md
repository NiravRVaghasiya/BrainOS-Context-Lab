# BrainOS Context Lab — Working Context

> Living hand-off context for implementation work. This file records the
> repository state and decisions that should be preserved between phases.

**Last updated:** 2026-09-13
**Branch:** `arena/01a09c46-brainos-context-lab`
**Baseline:** `147c189` (`origin/main`, includes PR #7's Phase 6); this branch adds Phase 7 on top
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
| Phase 4 — Chat web UI | Complete | Gradio callbacks wired to a Gradio-free `UIController`: provider connect/validate/list-models, per-session chat, the five planned tabs, session controls (clear conversation / clear memory / end session / export), and a memory-mode selector. Landed on `main` via PR #5; 216 tests, live-validated at 41 turns. |
| Phase 5 — Conversation persistence | Complete | SQLite backend behind the finalized store protocols (per-op thread-safe connections, lazy schema, upsert memory mirrors); the service persists turns best-effort with session-key redaction at the write site; `UIController` owns lifecycle deletes and a persisted-view export block; `create_app` picks the backend, so headless runs never touch disk. Landed on `main` via PR #6. 249 tests; live-validated over HTTP with real SQLite. |
| Phase 6 — Baseline modes | Complete | The plan's five modes (A full context, B sliding window, C lexical RAG, D BrainOS, E BrainOS + RAG) behind `ContextSettings.mode`, each fixing its own history window; a BrainOS-free lexical chunk retriever; a second delimited evidence block with its own budget, accounting, and eviction slot; mode-aware UI with a chunk panel; and the `evaluation/runner.py` seam wired, so `python -m evaluation.run --mode …` executes. 379 tests; live-validated over HTTP. |
| **Phase 7 — Context-rot benchmark** | **Complete in this turn** | The seven plan categories are generated (`spec.py` + `generation.py`) with a fact ledger and evidence contract; retrieval scoring is model-free (markers → Recall@K / precision / evidence-in-prompt) and answer scoring grades supplied answers (`--answers`) into a fixed verdict vocabulary with the Phase 12 error labels; the runner reports aggregates and the dataset hash; the committed smoke dataset is 7 tasks, SHA-256 pinned by a manifest. 494 tests; live-validated on the pinned runtime. |
| Phase 8 — Metrics | Pending | `aggregate_scores` is descriptive only (counts/rates/means with per-rate denominators). Missing: the full metric suite (faithfulness, latency), per-length curves, quality-adjusted efficiency reported per mode, and the degradation/AUC measures. |
| Phases 9–20 | Pending | Controlled experiments, statistics, ablations, error analysis, security hardening, deployment, cost controls, reproducibility, evaluation pipeline, tests, MVP, release. |

Detailed logs are available in
[`docs/phase-0-research-baseline.md`](docs/phase-0-research-baseline.md),
[`docs/phase-1-provider-abstraction.md`](docs/phase-1-provider-abstraction.md),
[`docs/phase-2-brainos-adapter.md`](docs/phase-2-brainos-adapter.md),
[`docs/phase-3-context-construction.md`](docs/phase-3-context-construction.md),
[`docs/phase-4-chat-web-ui.md`](docs/phase-4-chat-web-ui.md),
[`docs/phase-5-conversation-persistence.md`](docs/phase-5-conversation-persistence.md),
[`docs/phase-6-baseline-modes.md`](docs/phase-6-baseline-modes.md), and
[`docs/phase-7-context-rot-benchmark.md`](docs/phase-7-context-rot-benchmark.md).

## What was done in Phase 7 (this turn)

### New / changed modules

| File | Purpose |
| --- | --- |
| [`benchmarks/context_rot/spec.py`](benchmarks/context_rot/spec.py) (new) | The declarative half of the benchmark: `GENERATOR_VERSION = "context-rot-v1"`, the seven categories with plan letters, `TIERS` (`smoke` 800 / `quick` 2k,4k / `standard` 5k…40k / `research` = the plan's 5k–120k ladder), `DEFAULT_SEED = 20260913`, the fact and distractor vocabulary, filler corpora, every sentence/question template, `answers_for()` for accepted answer variants. |
| [`benchmarks/context_rot/generation.py`](benchmarks/context_rot/generation.py) (rewritten) | The structural half: per-category fact plans (one function per category), deterministic packing of facts and filler to a target length, session boundaries, a supersession chain per temporal/conflict task, the fact ledger, and the manifest. CLI: `--tier/--lengths/--categories/--variants/--seed/--output/--manifest`. |
| [`benchmarks/context_rot/dataset.jsonl`](benchmarks/context_rot/dataset.jsonl) (regenerated) | 7 tasks (one per category, 800-token tier, 66–80 messages), 53 KB, SHA-256 `4c4a4d37…` pinned by the manifest. |
| [`benchmarks/context_rot/MANIFEST.json`](benchmarks/context_rot/MANIFEST.json) (new) | Generator version, tier, lengths, categories, variants, seed, task count, mean achieved tokens, dataset hash, and the regeneration command. |
| [`src/evaluation/datasets.py`](src/evaluation/datasets.py) (rewritten) | The benchmark data contract: `BenchmarkFact` (markers, `fact_type`, `introduced_turn`, subject/value, `supersedes`/`superseded_by`, `session_index`), `EvidenceContract` (required / supporting / forbidden / `abstention_expected`), `BenchmarkTask` (ledger + contract + `acceptable_answers` + `session_count`, with `answers()` / `facts_by_id()` / `planted_facts()` / `required_facts()`), `validate_task` / `dataset_issues`, JSONL load/write. `required_memory_ids` records still load. |
| [`src/evaluation/scoring.py`](src/evaluation/scoring.py) (new) | Normalization (`normalize_answer`), boundary-aware matching (`contains_phrase`), `abstains`, marker detection (`detect_facts`, `detect_facts_in_texts`), `RetrievalScore` / `AnswerScore`, `score_retrieval` / `score_answer` / `score_record`, `aggregate_scores`, `AnswerSet`, and the session-isolation helpers (`latest_session_index`, `unavailable_required_facts`, `expects_abstention`). |
| [`src/evaluation/modes.py`](src/evaluation/modes.py) (extended) | `ModeReplay` gained `retrieved_memory_texts`, `retrieved_chunk_texts`, `stored_memory_count`, `sessions_used`, and `prompt_text()`; `replay_task` / `compare_modes` / `task_evaluator` accept `session_isolation=True`, which starts a fresh service at each transcript `session` boundary. |
| [`src/evaluation/runner.py`](src/evaluation/runner.py) (extended) | `run(tasks, evaluator, scorer=…, answers=…)`: attaches a `scores` block per result, fills `aggregate_metrics` from `aggregate_scores`, validates the dataset into `dataset_issues`, and records provenance (`benchmark_version`, `dataset_sha256`, `session_isolation`, `dataset_notes`). |
| [`src/evaluation/run.py`](src/evaluation/run.py) (extended) | `--answers`, `--session-isolation`, `--limit`; hashes the dataset; prints a one-line summary; sends dataset problems to stderr. Default dataset is now `benchmarks/context_rot/dataset.jsonl`. |
| [`benchmarks/context_rot/README.md`](benchmarks/context_rot/README.md) (rewritten) | Task anatomy, the marker convention, the categories, the tiers, generation and scoring commands, and what the benchmark is not. |
| [`docs/evaluation.md`](docs/evaluation.md) | New "Dataset (Phase 7)" and "Scoring (Phase 7)" sections. |
| [`docs/limitations.md`](docs/limitations.md) | New benchmark-specific limits: template filler, lexically scored retrieval/answers, smoke tier is plumbing not evidence, estimator counter. |
| [`README.md`](README.md) | Status, measured table, repository layout, and evaluation commands updated. |
| Tests | `tests/unit/test_dataset.py` rewritten (15), `tests/evaluation/test_context_rot_dataset.py` (49), `tests/evaluation/test_scoring.py` (42), `tests/integration/test_context_rot_live.py` (10): **379 → 494**, `ruff check .` clean. |
| `.gitignore` | Ignores `benchmarks/context_rot/generated/` — larger tiers are build artifacts, not source. |

### The benchmark contract (what later phases build on)

```text
task = conversation + question + expected_answer
     + facts      (the ledger: markers, type, turn, supersession, session)
     + evidence   (required / supporting / forbidden / abstention_expected)
     + acceptable_answers, session_count, metadata (seed, lengths, distances)
```

- **Markers are the scoring interface.** All of a fact's markers must appear, and
  they must co-occur in one memory/chunk/prompt message; the runtime's own memory
  ids are minted per replay and cannot be referenced by a dataset.
- **Two validity properties are enforced by tests**: every planted fact is
  storable by the app's memory policy, and no filler turn is storable.
- **Scoring is split.** Retrieval scoring is model-free (Recall@K, Precision@K
  over required∪supporting, `evidence_in_prompt`, `expected_answer_in_prompt`,
  `prompt_contains_forbidden`). Answer scoring takes a supplied string
  (`--answers` JSONL `{task_id, mode?, answer}`) and never calls a model.
- **Verdicts**: `correct | abstained | wrong_abstention | stale_answer |
  incorrect | ungraded`. When abstention is expected (abstention category, or a
  cross-session task under `--session-isolation`), declining is the *only*
  correct behaviour; asserting a value there is a `hallucination`.
- **Error labels** (Phase 12 seam) come from the same record: `incorrect` with
  evidence in the prompt → `hallucination`, without → `missed_memory`;
  `stale_answer` → `conflicting_memory` (conflict category) else `stale_memory`;
  abstained with evidence present → `wrong_abstention`. `ungraded` stays
  `ungraded` — missing data is not an observed failure.
- **Aggregates** are descriptive: counts, rates, and means each with their own
  denominator, plus `by_category`. Retrieval recall excludes tasks with no
  required facts, so an abstention task cannot contribute a vacuous 1.0.
- **Session isolation is a flag, not an assumption.** Default replay = one
  session (the long-range case). `--session-isolation` starts a fresh session at
  each boundary, which measures the product invariant and flips the
  cross-session expectation to abstention.
- **Determinism is hash-pinned**: tasks derive from `(seed, category, length,
  variant)`, and `tests/…/test_context_rot_dataset.py` regenerates from the
  manifest parameters and asserts the same SHA-256.

### Measured behaviour (live pinned BrainOS, committed smoke dataset, no provider)

7 tasks × 5 modes, estimated token counter. "Evidence in prompt" is over the 6
answerable tasks.

| Mode | mean tokens sent | full-context ref | reduction | Recall@K | Precision@K | evidence in prompt |
| --- | --- | --- | --- | --- | --- | --- |
| A `full_context` | 1152.7 | 1152.7 | 0.0% | 0.00 | 0.14 | 100% |
| B `sliding_window` | 189.4 | 1152.7 | 83.5% | 0.00 | 0.14 | **0%** |
| C `rag` | 270.6 | 1152.7 | 76.5% | 1.00 | 0.39 | 100% |
| D `brainos` | 193.6 | 1152.7 | **83.2%** | 1.00 | 0.39 | 83.3% |
| E `brainos_rag` | 367.1 | 1152.7 | 68.1% | 1.00 | 0.39 | 100% |

Per-category evidence availability: B loses every category; D loses exactly
`multi_hop`; C, E and A carry all six; the abstention task has no evidence in any
mode (correct — nothing answers it).

Scored end-to-end run (scripted answers, no model):
`task_count=7 graded=7 recall=1.00 precision=0.39 in_prompt=0.83 accuracy=1.00
abstention=1.00 stale=0.00 reduction=0.832`. Accuracy 1.0 validates the plumbing
only: the answers were scripted from the contract the scorer grades against.

**Integration-validation observations, not research results** — one seed, one
length tier, estimated counter, no model in the loop.

### Findings this phase produced (carry into Phases 8–11)

1. **Mode D fails multi-hop.** Recall@K is 1.0 (both facts recalled) but
   `selected_memory_count=1` and `memory_block_tokens=77` of a 1024-token
   allowance → the Phase 3 retrieval policy's relevance filtering/ranking is the
   binding constraint, not the budget; the second hop falls below the relative
   tail trim against a question that matches the first hop better. Report
   Recall@K and evidence-in-prompt separately in Phase 8; ablate the relative
   trim and `max_memories` in Phase 11.
2. **A stale value reaches every mode's prompt** on temporal and conflict
   (`prompt_contains_forbidden` true for A, C, D, E). Conflict resolution is only
   visible in the answer verdict (`stale_answer`) — which is why the two halves
   are scored separately.
3. **Abstention is still not achieved by BrainOS mode** (2 irrelevant memories
   selected on the abstention task). Now measured by a category instead of a
   one-off probe.
4. **Mode E earned its cost on this dataset** (both hops for 1.9× Mode D's
   tokens). A hypothesis for Phase 9, not a claim.
5. **The harness is CI-cheap**: the 7-task × 5-mode sweep plus the scored run is
   ~4 s against the live runtime.

### Bugs found and fixed while building it

1. **Filler was being stored as memory** — `Nothing new since the last check…`
   matched the policy's statement verbs through `\blast\b`. Reworded to
   "previous"; the "filler never enters memory" test guards the corpus.
2. **A distractor template was never storable** ("archives" is not a statement
   verb) → the distractor was invisible to every mode. Reworded to "retains".
3. **`session_boundaries` metadata was overwritten by the plan's notes** (int
   replaced the list), breaking the cross-session lookup.
4. **`score_record` dropped retrieved memory texts** through operator precedence
   (`a or [] + b`), so retrieval scoring silently saw only chunks.
5. **No abstention pattern was reachable**: normalization strips apostrophes, so
   `i don't know` could never match. Patterns are now normalized the same way —
   before this, *every* abstention answer graded `incorrect`.
6. **Marker detection over a joined blob produced false positives**; detection is
   now per evidence item and per prompt message.
7. **`wrong_abstention` was documented but never produced**, and the first fix
   updated the dataclass without the payload.
8. **Ungraded records leaked into the error taxonomy** as `missed_memory`.
9. **`distractor_count` could exceed the project vocabulary**, letting two ledger
   facts share a marker set. Capped; detection stays unambiguous.
10. **`EvidenceContract.from_dict` crashed on a built contract**; it now accepts
    either form.

## What was done in Phase 6 (PR #7, previous turn)

### New / changed modules

| File | Purpose |
| --- | --- |
| [`src/baselines/modes.py`](src/baselines/modes.py) (new) | The mode registry: one frozen `BaselineMode` per strategy carrying its evidence sources, history window, and section budgets. `resolve_mode()` normalizes selectors (`no_memory` → Mode B) and **rejects** anything unknown; `budget_overrides()` is what a mode change writes. |
| [`src/baselines/rag.py`](src/baselines/rag.py) (new) | The BrainOS-free lexical baseline: sentence-packed ~400-char chunks ranked by IDF-weighted overlap, reusing the Phase 3 lexical helpers. Deterministic tie-breaks, near-duplicate suppression, and no abstention — by design, because that is what ordinary RAG does. |
| [`src/brain/context_builder.py`](src/brain/context_builder.py) | `build_context(..., retrieved_chunks=)` renders a second `<retrieved_history>` block under a new `chunk_budget` (default `0`, so pre-Phase-6 callers are unaffected); chunks carry `[role #turn]` provenance; a chunk duplicating the retained window is dropped; seven new `ContextStats` fields; eviction order is now history → chunks → memories → system. |
| [`src/evaluation/modes.py`](src/evaluation/modes.py) (new) | `replay_task` / `compare_modes` / `task_evaluator` run a benchmark task through any mode on the application's own service and builder — fresh session per (task, mode), no provider, no storage. `evaluation/run.py` now passes the evaluator to the runner. |
| [`src/app/state.py`](src/app/state.py) | Five-mode vocabulary plus `chunk_budget` / `rag_top_k`; `mode_profile()`, `uses_rag()`, `with_mode_defaults()`. **Defaults are Mode D's canonical profile**, not neutral values. |
| [`src/app/service.py`](src/app/service.py) | `_retrieve_chunks()` (credential-guarded before the builder sees it), chunks into `build_context`, `ConversationTurn.retrieved_chunks`, and a new `record_assistant_message()` so scripted benchmark turns enter the transcript and memory as generated ones do. |
| [`src/app/controller.py`](src/app/controller.py) | `apply_mode()`; `update_context()` validates the mode and lets **the mode's budgets win when the mode changes**; `context_payload()` reports the mode profile; `TurnView.chunk_rows`. |
| [`src/app/panels.py`](src/app/panels.py), [`src/app/ui.py`](src/app/ui.py) | Chunk table and evidence lines in the panels; five-mode dropdown with a live description, a `mode.change` handler that pushes the applied budgets back into the sliders, and new "recent turns kept" / "chunk budget" controls. |
| Tests | 25 mode-registry, 16 retriever, 20 builder-chunk, 28 service/controller mode, 16 evaluation-seam, 11 security, 11 live-runtime: **249 → 379**. |

### Key design decisions

- **A mode is defined by its window as much as by its retriever.** This is the
  Phase 3 finding turned into a constraint: `recent_turn_budget` is `None` for
  Modes A/B (derived from the session's `max_tokens` ceiling) and 256 tokens for
  C/D/E. Two modes sharing a window larger than the conversation are
  indistinguishable, so switching modes rewrites the window.
- **Modes C, D, and E share one evidence allowance** (`EVIDENCE_BUDGET = 1024`;
  Mode E splits it 512/512). They differ in *what selects* the evidence, so
  volume is held constant and any accuracy difference is attributable to
  selection. Any new retrieval mode must fit inside that allowance.
- **The mode's budgets win over sliders submitted in the same call**, but only
  when the mode actually changes. Tuning inside a mode still works; the sidebar
  just cannot carry the old mode's window into the new one.
- **BrainOS still observes in every mode.** Observation is a side process, not
  context construction, and a warm runtime means switching modes mid-session
  does not lose memory the new mode would use. Only *injection* is
  mode-dependent.
- **The builder guards chunk text itself.** The retriever already neutralizes
  what it selects, but the prompt is the one surface where a missed guard is a
  security failure, so it does not depend on any caller remembering. Guarding is
  idempotent.
- **`full_context_reference_tokens` still prices only system + full history +
  question.** Mode A has no evidence blocks; including them would compare every
  mode against something Mode A never was.

### Measured behaviour (live pinned BrainOS, 28-message transcript)

Two facts separated by 24 filler turns, asked at the end. Dependency-free token
estimator; no model in the loop; no provider API key.

| Mode | Sent | Full-context ref | Reduction | Mem | Chunks | History | Fact in prompt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A `full_context` | 510 | 510 | 0.0% | 0 | 0 | 28 | yes |
| B `sliding_window` | 189 | 510 | 62.9% | 0 | 0 | 8 | **no** |
| C `rag` | 256 | 510 | 49.8% | 0 | 6 | 2 | yes |
| D `brainos` | 182 | 510 | **64.3%** | 2 | 0 | 2 | yes |
| E `brainos_rag` | 345 | 510 | 32.4% | 2 | 6 | 2 | yes |

**Integration-validation observations, not research results** — one synthetic
conversation, an estimated counter, no model, single trial.

Mode D cost fewer tokens than Mode B *and still carried the fact B lost*: that
is the comparison the project exists to make, and Phase 6 makes it expressible
for the first time. Mode E is not free — combining the sources cost 1.9× Mode D
because both blocks fill their budgets, so the hybrid's value has to be earned
on quality rather than assumed.

### Bugs found and fixed while testing

1. **`_apply_mode` returned the wrong session.** It resolved the session id from
   its input *after* acting, so with an empty browser state it minted a second
   session and handed the browser an id whose mode had never changed — the
   switch would have been silently lost on the next turn. Now resolved before
   acting, matching `_update_context`.
2. **The builder trusted the retriever to guard chunk text**, so a hand-built
   chunk containing `</retrieved_history>` broke out of the block. Fixed in the
   builder (twice, idempotently).
3. **`_DELIMITER_RE` did not cover `retrieved_history`.** The Phase 3 guard
   stripped `<retrieved_memory>` breakouts but not the new delimiter. Extended.

### Live HTTP validation and the limitation it exposed

Verified against a running `python app.py` with the pinned runtime and real
SQLite: `/apply_mode` writes and returns exactly the per-mode budgets (Mode A
came back with `recent turns = None` and `window = 4096`; Mode E with
`512 / 512`); the chat status and Context summary name the mode that ran; the
chunk panel is registered with its headers; `/export_session` recorded
`mode=rag`, `letter=C`, `chunk_budget=1024`, `evidence_budget=1024` with no
`api_key` field; and the session key appeared in no panel, response, or raw
database byte.

**Limitation found, pre-existing:** Gradio does not expose `gr.State` to API
clients, so `/chat` accepts only `message` and an HTTP client gets a fresh
session per request. The browser path is unaffected. Consequence: the
*multi-turn* cross-mode comparison cannot be driven over HTTP and is covered
instead by `tests/integration/test_baseline_modes_live.py`. Generation was not
exercised over HTTP either (no `providers` extra installed, no real API key);
it is covered by the fake-provider tests.

## What was done in Phase 5 (previous turns)

### How this branch merged with upstream Phase 4

PR #6 originally carried this branch's own Phase 4 implementation
(`ChatController` + a rewritten Gradio UI) with Phase 5 on top. While it was
open, PR #5 merged a parallel Phase 4 — the `UIController` + `panels` stack
documented in the next section — into `main`. The conflict resolution treats
**upstream PR #5 as the canonical Phase 4**: its controller, panels, UI, and
tests were adopted as merged; this branch's superseded Phase 4 modules, tests,
and log (`docs/phase-4-chat-ui.md`) were retired; and Phase 5 was ported onto
the upstream seams (`provider_factory` / `adapter_factory`, `reset_memory()`,
`export_session()`). Phase 5's storage layer, protocols, and SQLite backend
are unique to this branch and were kept unchanged.

### New / changed modules

| File | Purpose |
| --- | --- |
| [`src/storage/sqlite.py`](src/storage/sqlite.py) (new) | `SqliteConversationStore` / `SqliteMemoryStore` / `SqliteEvaluationStore`: per-operation connections (Gradio-thread safe), lazy schema (`messages` seq-ordered + indexed, `memories` keyed `(session_id, memory_id)` with JSON payload, `evaluation_runs` session-indexed), recursive `strip_secret_fields`, `BRAINOS_LAB_DB` env path (default `data/brainos_lab.sqlite3`, Git-ignored). |
| [`src/storage/conversations.py`](src/storage/conversations.py) | Protocols finalized: `list_conversations` + `delete_session` (conversation store), `save_memories` upsert (memory store, mirror semantics documented). |
| [`src/app/service.py`](src/app/service.py) | Optional `conversation_store` / `memory_store` on the upstream service; `_persist_message` (redacts the exact session key at the write site) and `_persist_memories` (whole-set guarded upsert each turn); `clear_conversation()` deletes the old conversation's rows; `reset_memory()` also clears the persisted mirror regardless of the injected-adapter seam; every store call best-effort — failures log only the exception *type* server-side (`_warn_storage`) and never break a turn. |
| [`src/app/controller.py`](src/app/controller.py) | `UIController` owns the three store protocols (inject-or-`None`) and passes them to each session's service; `end_session()` deletes every persisted row of the ended session first; `export_session()` gains a `persistence` block — every persisted conversation of the session plus the memory mirror — read best-effort, served through the existing `DownloadButton`. |
| [`src/app/ui.py`](src/app/ui.py) | `create_app()`'s default controller builds one server-wide SQLite store set (backend choice lives at the deployment edge; injected controllers keep their caller's stores); header text discloses persistence and the delete controls. |
| [`tests/fakes.py`](tests/fakes.py) | `InMemoryConversationStore`, `InMemoryMemoryStore`, `InMemoryEvaluationStore`, `RaisingStore` (every operation fails — resilience tests) added alongside the upstream fakes. |
| Tests | 15 SQLite-store, 12 persistence-wiring, 4 storage-security, 2 live-SQLite integration, all retargeted to the upstream `UIController` API: **216 → 249**. |

### Key design decisions

- **Backend at the edge, protocols at the core.** Controller and service know
  only the protocols (`None` = purely in-memory, upstream Phase 4 behaviour
  unchanged); `create_app` picks SQLite. Headless runs and tests never touch
  disk unless they opt in.
- **Mirror, not source.** The memory table exists for inspection/export; no
  code path hydrates BrainOS memory from it, and `state.messages` remains the
  render source. Storage is written after a turn, never read back mid-turn.
- **Data-control semantics** (all four exist, as `docs/security.md` promised):
  *Clear conversation* = transcript + runtime + that conversation's rows +
  mirror; *Clear memory* = runtime + mirror, transcript kept; *Export* =
  sanitized read-only JSON (live views + persisted views); *End session* =
  credentials + every persisted row of the session.
- **Merge repairs to upstream code** (small and deliberate):
  `reset_memory()` now rebuilds the runtime via `_new_adapter()` so an
  injected `adapter_factory` is honoured after a memory clear; the `ui` extra
  floor is `gradio>=6.0` (the upstream UI renders message dicts natively,
  which Gradio 5 requires `type="messages"` for); lifecycle status strings
  mention the persisted-row deletes.
- **Documented caveat for Phase 13:** SQLite `DELETE` frees pages without
  physically scrubbing bytes; the tested contract is that no query can reach a
  deleted session's rows. `VACUUM` / per-session files are hardening
  candidates.
- **Upstream UI gap noted:** the `/end_session` handler renders the fresh
  session's panels but discards the controller's "Session ended …"
  confirmation string. Follow-up candidate, not a persistence defect.

### Live HTTP validation (running server + pinned BrainOS + real SQLite)

Two chat turns through the live Gradio server persisted 2 transcript rows and
1 memory row classified `PROJECT_STATE` by the live runtime; the recall panel
retrieved the fact. `/export_session` downloaded a JSON file containing the
live payload plus `persistence` (1 conversation, 2 messages, 1 memory) with no
`api_key` anywhere. `/clear_memory` kept the 2-message transcript, emptied the
stored panel, and left 2 message rows / 0 memory rows on disk;
`/end_session` deleted the session's rows exactly (0 / 0 afterwards). Dev
database rows were removed after validation.

Full log: [`docs/phase-5-conversation-persistence.md`](docs/phase-5-conversation-persistence.md).

## What was done in Phase 4 (upstream PR #5)

### New modules

| File | Purpose |
| --- | --- |
| [`src/app/controller.py`](src/app/controller.py) | Gradio-free UI controller: session lifecycle, provider connect/validate/list-models, one chat turn, panel rendering, export, and the turn/message limits. |
| [`src/app/panels.py`](src/app/panels.py) | Pure formatting for the browser surface: table rows, context summary, prompt rendering, cognitive trace, status line. |
| [`tests/unit/test_controller.py`](tests/unit/test_controller.py) | 29 controller cases (connection, turn, isolation, limits, lifecycle, export, degraded runtime). |
| [`tests/unit/test_panels.py`](tests/unit/test_panels.py) | 23 rendering cases. |
| [`tests/ui/test_ui.py`](tests/ui/test_ui.py) | 6 Gradio cases; skipped when Gradio is absent. |
| [`tests/integration/test_chat_controller_live.py`](tests/integration/test_chat_controller_live.py) | 4 live cases against the pinned runtime, deterministic provider, no API key. |

### Rewritten / extended

- **`src/app/ui.py` (rewritten).** Layout builders return component dataclasses
  and `_wire()` attaches events, so callback arity is inspectable. Memory tab
  (stored / in-prompt / filtered-out / conflicts), Context tab (summary, stats,
  final prompt), Cognitive Trace tab (staged markdown + runtime events),
  Evaluation tab (placeholder until Phase 17).
- **`src/app/service.py`.** `adapter_factory` / `provider_factory` seams,
  `stored_memories()`, `reset_memory()` (clear memory without clearing the
  transcript), `reset_provider()` (a client built from an old key must stop
  serving), and diagnostics now include `mode`, `turn`, `stored_memory_count`.
- **`src/app/state.py`.** `ContextSettings.uses_memory()` and `MEMORY_MODES`
  (`brainos`, `no_memory`) — the mode selector only offers behaviour the
  application can already perform honestly.
- **`tests/fakes.py`.** `FakeLLMProvider` covers the whole `LLMProvider`
  surface (list, validate, generate) and can fail on demand.

### Panel fix found during live validation

The first "retrieved memories" table was built from everything BrainOS recalled,
so it listed six memories while the statistics said `1 of 6` were selected. The
table is now driven by the builder's **post-budget ranking** — it answers "what
did the model see" — and everything else appears once, in the audit table, with
its reason.

### Security properties at the browser boundary

- The key box is emptied on every connect attempt; `ConnectionView.key_value` is
  always `""`.
- Every returned value is redacted against the active key — applied to the
  *rendered* values, because sanitizing a `MemoryRecord` turns it into a dict.
- `gr.State` holds only the non-secret session uuid; a stale id starts a fresh
  session rather than restoring another session's memory.

### Measured behaviour (live pinned BrainOS, 41-turn conversation)

Driven through `UIController` with a deterministic provider double and no
provider API key. Six durable facts among filler turns.

| Recent-history budget | mean tokens sent | full-context baseline | reduction | probes answered | history kept |
| --- | --- | --- | --- | --- | --- |
| 1024 | 881.7 | 811.0 | 0.0% | 3/3 | 84 of 84 |
| 256 | 590.0 | 811.0 | **27.2%** | 3/3 | 51 of 84 |
| 64 | 248.0 | 811.0 | **69.4%** | 3/3 | 13 of 84 |

Reproduces the Phase 3 finding through the UI's own controls: **the reduction
comes from the history window, not from memory selection**. Also measured: 6/6
durable facts stored, 0/15 chit-chat turns stored, `no_memory` mode sends no
memory block, and the session key never appeared in any panel, prompt, trace, or
export.

**These are integration-validation observations, not research results** — one
synthetic conversation, estimated token counter, no model in the loop.

### Findings carried into later phases

1. **A correction can be filtered out before it can win.** For
   `What database does Project Atlas use?` after
   `Correction: the production database is MySQL 8 now.`, the prompt contained
   the stale `PostgreSQL 16` memory. Subject Jaccard is 0.5
   (`{production, database}` vs `{project, atla, production, database}`) against
   a 0.6 threshold, so no conflict is detected; and the correction text scores
   below the relevance floor anyway, so detecting the pair would still not put it
   in the prompt. Left untouched deliberately — conflict-resolution and
   abstention accuracy are scored categories in Phases 7–8, and tuning on one
   synthetic conversation is the overfitting the Phase 3 log warned about.
2. **Abstention is still not achieved** (unchanged): the absent-answer probe
   still selects one memory.

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
`E501`, one `E402`) were fixed in that phase. `ruff check .` is now clean across
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
    retrieved_chunks=...,         # Phase 6: Iterable[RetrievedChunk] for Modes C/E
    budget=ContextBudget(...),    # enforced, validated (chunk_budget added in Phase 6)
    policy=RetrievalPolicy(...),  # dedupe/relevance/conflict/recency knobs
    conflicts=adapter.conflicts(),        # BrainOS contradictions()
    stale_ids=adapter.stale_memory_ids(), # BrainOS stale_memories()
    current_turn=...,             # recency fallback when timestamps are absent
    token_counter=...,            # optional; named in the accounting
    now=...,                      # optional; inject for reproducible runs
) -> BuiltContext(messages, selected_memories, stats, report, ranking, selected_chunks)
```

Phase 6 additions to the contract, which every baseline mode relies on:

- `retrieved_chunks` defaults to empty and `chunk_budget` to `0`, so a caller
  that passes neither builds byte-for-byte the pre-Phase-6 prompt.
- Chunks are duck-typed through a `RetrievedChunk` `Protocol`, so `brain` never
  imports `baselines` and the dependency stays one-directional.
- Chunk text is guarded by the builder itself (idempotently, twice), so no
  caller can forget and break out of `<retrieved_history>`.
- A chunk duplicating a message the retained window already carries is dropped
  with reason `duplicate_history` rather than paid for twice.
- Message order is `system`, memory block, chunk block, history…, question; the
  ceiling evicts history → chunks → memories → system.
- `full_context_reference_tokens` prices only system + full history + question,
  because Mode A has no evidence blocks.

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
| `chunk_budget`, `chunk_ceiling` | `over_compression`, **retrieved-chunk source only** (Phase 6) |
| `duplicate_history` | (not a failure — chunk already inside the history window) |

Phase 6 namespaced the chunk reasons (`chunk_` prefix) precisely so a
Precision@K computation over `report.dropped` can still isolate *memory* drops;
do not fold them into the memory counts.

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

### Finding that constrained Phase 6 — now discharged

**Context reduction is driven almost entirely by the recent-history window, not
by memory selection.** With `recent_turn_budget` larger than the conversation,
"BrainOS mode" degenerates into full context *plus* memory overhead: 0.0%
reduction, 95 tokens *worse* than the baseline.

*Resolved in Phase 6:* every baseline mode now fixes its own
`recent_turn_budget` and `max_recent_turns`, switching modes rewrites them, and
the active window is reported in `context_payload()` and the UI. The modes are
separable as a result — on a 28-message conversation Mode A sent 510 tokens,
Mode B 189, Mode D 182 (see the Phase 6 table above).

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
- **New (Phase 5):** nothing reaches disk unredacted: transcript text has the
  exact session key replaced at the write site, memory rows pass through the
  same credential guard as recalled memories, and the SQLite backend strips
  secret-named fields recursively (`strip_secret_fields`) from every
  metadata/payload. The raw bytes of the database file are asserted key-free
  in `tests/security/test_storage_secrets.py`.
- **New (Phase 5):** persistence is best-effort — storage exceptions are
  logged server-side by exception *type* only and never break a turn or leak
  stored content into browser-visible errors.
- **New (Phase 5):** user-data deletes are row-scoped and tested: clear
  conversation / clear memory / end session remove exactly the matching
  conversation, mirror, or session rows, and another session's rows provably
  survive.
- **New (Phase 6):** retrieved transcript chunks — the second route Phase 6
  added from conversation content into a prompt — are credential-guarded by the
  service *before* the builder sees them, and guarded again by the builder
  itself (idempotently) so no caller can forget. `_DELIMITER_RE` was extended to
  cover `<retrieved_history>` breakouts, which the Phase 3 guard did not.
- **New (Phase 6):** the evaluation replay path configures no provider, so a
  replay holds no credential at all. A key inside a *dataset's* conversation is
  that dataset's content and is replayed faithfully — `tests/security/`
  documents this boundary explicitly rather than leaving it implicit.
- **New (Phase 7):** the benchmark path holds no credential at all. A replay
  configures no provider; answer grading takes an answer *string* (from a file),
  never a client; the run's provenance block carries the dataset hash, counts,
  and categories, never a key; and the committed dataset was generated offline
  (`generation.py` imports no provider and no runtime).
- **Restated (Phase 6):** raw history inside the recent window is replayed
  verbatim. The credential guard covers recalled memory, retrieved chunks,
  diagnostics, and persisted rows — not the transcript Mode A must send as
  written. A key a user pasted into a conversation therefore still reaches their
  own provider in Mode A/B history. Pre-existing Phase 3 behaviour, restated
  because Phase 6 added a guarded retrieval route beside it.
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
.venv/bin/pip install pytest ruff gradio
.venv/bin/pip install "brainos-cli @ git+https://github.com/NiravRVaghasiya/BrainOS.git@1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc"

.venv/bin/pytest -q
# 494 passed

.venv/bin/ruff check .
# All checks passed!   (whole repository, no exclusions)

.venv/bin/python app.py
# http://localhost:7860

# Phase 6/7: the CLI executes every task in a dataset through any mode,
# scoring retrieval with no provider key; --answers adds answer grading
PYTHONPATH=src .venv/bin/python -m evaluation.run --mode brainos --output results/run.json
PYTHONPATH=src .venv/bin/python -m evaluation.run --mode brainos \
  --answers answers.jsonl --output results/brainos.json

# Phase 7: regenerate the committed benchmark (byte-identical; the manifest
# records the SHA-256). Larger tiers are generated on demand and Git-ignored.
.venv/bin/python benchmarks/context_rot/generation.py
.venv/bin/python benchmarks/context_rot/generation.py --tier standard --variants 2 \
  --output benchmarks/context_rot/generated/standard.jsonl
```

Test count went 154 → 216 (Phase 4) → 249 (Phase 5) → 379 (Phase 6) → **494**
in this phase (115 new: 14 dataset-schema, 49 benchmark-validity, 42 scoring,
10 live benchmark).
The live tests in `tests/integration/test_chat_controller_live.py`,
`tests/integration/test_persistence_live.py`,
`tests/integration/test_context_pipeline.py`,
`tests/integration/test_baseline_modes_live.py`, and
`tests/integration/test_brainos_runtime.py` run against the pinned BrainOS
revision and skip when `brainos_runtime` is not installed; `tests/ui/` skips
when Gradio is absent. No provider API key was used; generation is exercised
through `FakeProvider` / `FakeLLMProvider`.

`.venv` is ignored and is only a local test environment.

## Next safe step

Implement **Phase 8 — Metrics** on top of the Phase 7 scorer, then Phase 9
(controlled experiments) and Phase 10 (statistics/plots).

What already exists and should be reused rather than rebuilt:

* `evaluation/scoring.py` — `score_record` produces one JSON-ready record per
  (task, mode) with `retrieval`, `answer`, `error_type`, and cost fields;
  `aggregate_scores` already reports counts/rates/means with per-rate
  denominators and a `by_category` block. Phase 8 extends this, it does not
  replace it.
* `evaluation/metrics.py` — `recall_at_k`, `precision_at_k`,
  `context_reduction`, `token_savings`, `quality_adjusted_efficiency`,
  `summarize` (mean/SD/95% CI). Only `summarize` is used by the aggregate today;
  the efficiency metrics are still uncalled.
* `evaluation/modes.py` — `replay_task` / `compare_modes` (with
  `session_isolation`) run any task through any mode on the real service; the
  replay carries the retrieved texts, the prompt, the stored-memory count, and
  the sessions used.
* `evaluation/runner.py` + `evaluation/run.py` — a run already scores,
  aggregates, records the dataset hash and notes, and writes JSON.
* `benchmarks/context_rot` — `--tier standard` (5k/10k/20k/40k) and
  `--variants N` generate the length ladder; `reports.py`, `analysis.py`,
  `plots.py`, and `compare.py` are thin scaffolding waiting for Phase 8/10.

Carry these constraints into Phase 8:

1. **Recall@K and evidence-in-prompt must both be reported** — the multi-hop
   case has 1.0 and 0.0 respectively, and that disagreement is the finding.
2. **Abstention is inverted-scored on purpose.** Declining is correct when
   abstention is expected; asserting a value there is a hallucination. Do not
   "normalise" the verdicts.
3. **The committed smoke tier cannot support a claim** (7 tasks, one length, one
   seed). Use `--tier standard`/`research` with multiple variants, write into the
   Git-ignored `benchmarks/context_rot/generated/`, and treat every number from
   the smoke tier as plumbing validation.
4. **Use an exact token counter for research runs.** Every number in this file is
   `estimate_tokens`; `tiktoken` is not installed here.
5. **Grading needs the Phase 15 cost controls** before the Evaluation tab
   (Phase 17) exposes a benchmark run; `--limit` is the only control today.
6. **Do not tune thresholds on the smoke tier** (Phase 3/4 rule, still in force).
7. **Keep the isolated replay in the suite**: it is the only test that would
   catch memory leaking across sessions after a future adapter/storage change.
8. Small UI follow-up candidate, still open: surface the controller's
   "Session ended …" confirmation in the `/end_session` handler (currently
   dropped by the upstream UI wiring).
