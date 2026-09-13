# Phase 3 log — Context construction engine

**Date:** 2026-09-13
**Branch:** `arena/01a09b44-brainos-context-lab`
**Starting commit:** `f1827e1c897b53dc869f186e392b80cb54c1c901`
**Plan section:** [Phase 3 — Context Construction Engine](../BrainOS_Context_Lab_Implementation_Plan.md#7-phase-3--context-construction-engine)

## Objective

Build the most important application layer: turn *system instructions + current
user message + recent conversation + BrainOS memories* into a model-ready
message list, by running the plan's retrieval policy

```text
current query → BrainOS recall → deduplicate → relevance filter
              → conflict check → recency weighting → token budget → final context
```

and logging the accounting every later phase depends on:

```text
raw_history_tokens, recent_history_tokens, retrieved_memory_tokens,
system_tokens, final_context_tokens
```

## Starting-state audit

Phase 2 left a working adapter and a `ConversationService`, but
`context_builder.py` was still the Phase 0 scaffold:

| Scaffold behaviour | Phase 3 requirement |
| --- | --- |
| Memories taken in recall order, truncated only by a token count | deduplicate → relevance → conflict → recency → rank |
| No deduplication | repeated memories were sent verbatim |
| No conflict handling | a corrected fact was sent alongside its correction |
| No recency model | stale and fresh memories ranked identically |
| `max_tokens` read then discarded (`_ = budget.max_tokens`) | hard ceiling with a documented eviction order |
| `system_budget * 4` characters as a token proxy | token-aware truncation for any counter |
| 6 accounting fields | 26, including the full-context reference |
| `MemoryRecord` carried text, type, and a relevance float | entities, timestamps, validity, status, supersession, per-signal evidence |

`memory_policy.py` extracted candidates but classified only through a narrow
`is/are/uses/has/was/were` verb list. Gradio callbacks remain unwired (Phase 4).

## Work completed

### 1. Retrieval policy (`src/brain/retrieval_policy.py`, new)

The whole pipeline between recall and the token budget, dependency-free and
deterministic:

- **Deduplication** — exact matches on the stemmed token sequence plus
  near-duplicates at a configurable Jaccard threshold (default `0.88`), merged
  into the highest-scoring representative with `merged_ids` recorded.
- **Relevance filtering** — a composite of the application's own lexical score
  and BrainOS's retrieval signals. The floor is applied to *query relevance
  only*: recency ranks memories but must not rescue an off-topic one.
- **Inverse document frequency** — query terms are weighted by how many
  candidates contain them. Without this, an entity that appears in every memory
  ("Project Atlas") gives every memory the same relevance floor and an off-topic
  question retrieves everything.
- **Conflict check** — BrainOS `contradictions()` and `stale_memories()` are
  authoritative and applied first. For pairs the runtime did not classify, an
  application heuristic extracts `subject → value` claims and resolves them,
  dropping the older claim **only** when the newer one explicitly presents
  itself as a replacement (a correction marker or a `CORRECTION` memory type).
  Otherwise both survive, flagged `contested`, and are labelled as such in the
  prompt.
- **Recency weighting** — exponential decay over the best available clock:
  observed/created timestamp, then conversation turn, then recall position. The
  48-hour default half-life matches the pinned runtime's own retrieval decay.
  Recall position is deliberately **not** used to infer age, because BrainOS
  returns its best match first — position expresses priority, not chronology.
- **Audit report** — every dropped memory records a reason
  (`empty`, `duplicate`, `stale`, `expired`, `superseded`, `low_relevance`,
  `weak_relevance`, `cap`, `suspicious`, `memory_budget`, `budget`). These are
  the raw material for Phase 8's Precision@K and Phase 12's error taxonomy.

Deliberate conservatism: when the chronology of a contradictory pair cannot be
established, the policy flags the pair rather than guessing a winner. Deleting
the wrong memory is a worse failure than presenting a labelled conflict.

### 2. Context builder rewritten (`src/brain/context_builder.py`)

- **Hard ceiling.** `max_tokens` is now enforced. Eviction priority: the current
  user message is never dropped or truncated → oldest recent history first →
  lowest-ranked memories next → the system prompt truncated last.
- **Section budgets.** `memory_budget` is enforced on the memory block *as sent*
  (delimiters and preamble included), filling strictly in rank order. Skipping
  ahead to a shorter memory would pack more text but silently reorder BrainOS's
  ranking, which would make retrieval quality unmeasurable.
- **Token-aware truncation** through `truncate_to_tokens`, correct for any
  monotonic counter instead of a `× 4` character guess.
- **Per-message overhead** (default 4 tokens) approximates chat-template framing
  so accounting reflects what is actually billed.
- **Prompt-injection hardening.** Memory text is stripped of delimiter breakouts,
  control characters, and `system:`-style role prefixes, collapsed to one bullet,
  length-bounded, and flagged when it matches an instruction-override pattern.
  The preamble is kept short on purpose: it is paid for on every request, so a
  verbose preamble would work against the token-efficiency question the project
  exists to measure.
- **Full-context reference.** `full_context_reference_tokens` prices the Mode A
  baseline (identical system prompt and question, every history message
  replayed), so `context_reduction_vs_full_context` measures only the
  context-management strategy.

### 3. Token counting (`src/brain/tokenizers.py`, new)

`estimate_tokens` (dependency-free default), `char_ratio_counter`,
`whitespace_counter`, `tiktoken_counter` (optional, raising
`TokenizerUnavailableError` rather than silently mixing counters),
`provider_counter` (a provider's own `count_tokens` hook), `resolve_counter`,
and `truncate_to_tokens`. Every accounting record now names the counter that
produced it, because a run is only comparable to itself.

### 4. Adapter signals (`src/brain/adapter.py`)

`MemoryRecord` gained the lifecycle fields the engine needs — `entities`,
`observed_at`, `confidence`, `salience`, `utility`, `valid_from/valid_until`,
`supersedes`, `contradicts`, `status`, `signals` — all mapped from BrainOS types
so none leak past the adapter. New methods: `conflicts()` (maps
`contradictions()`), `stale_memory_ids()` (maps `stale_memories()`), and
`enrich_with_explanation()` (attaches the `why()` per-memory `score`/`signals`,
matched by id or content).

Two mapping decisions worth recording:

- **Unknown timestamps stay unknown.** The scaffold filled a missing
  `created_at` with `now`, which made every memory look maximally recent and
  would have silently disabled recency weighting.
- **Observation bookkeeping.** BrainOS stores content, not the application's
  memory-policy label or conversation turn. The adapter remembers both, keyed by
  normalized text, so recalled records recover a specific `memory_type` and a
  `source_turn`. A runtime-reported type always wins; the policy label only
  refines the generic `FACT` mapping.

### 5. Service and session wiring

`ContextSettings` now carries the budget and policy knobs and builds both
objects (`context_budget()` / `retrieval_policy()`), so the UI and the
evaluation runner configure context construction through one object.
`ConversationService` recalls → enriches with the runtime's explanation → passes
runtime conflicts and stale ids → builds context with the current turn, and
exposes `context_stats`, `context_report`, and `memory_ranking` on the turn and
in sanitized diagnostics.

### 6. Credential-leak repair (found by a Phase 3 test)

Phase 3 added new browser-visible surfaces (dropped-memory audit text, ranking
components). A test that pasted a credential into memory showed it echoed back
in diagnostics: `brain/trace.py` scrubbed credential *shapes* but could not
recognise an arbitrary session key. `redact_text` / `sanitize_value` /
`sanitize_trace` now accept the caller's known `secrets`, and the service
redacts the active key from every browser-visible value **and** from recalled
memory text before it is placed in a prompt — otherwise a key pasted into one
conversation could be replayed to a different provider later.

### 7. Phase 2 memory-policy repair (prerequisite, outside Phase 3 scope)

Validation against the live runtime exposed that durable facts were never being
stored. `_classify` required an `is/are/uses/has/was/were` verb, so
"Deployments happen every Friday at 17:00 UTC" and "The retention policy
requires 400 days of audit logs" produced no candidate — while chit-chat
containing "are" ("What is the weather like where you are? Just making
conversation here.") was stored, eleven times over.

This is Phase 2 code, changed during Phase 3 because no context engine can
retrieve a fact that was never stored. `src/brain/memory_policy.py` now:

- covers common declarative verbs (`happen`, `requires`, `runs`, `retains`,
  `scheduled`, `migrated`, …) via `_STATEMENT_RE`;
- rejects questions and requests (`?` anywhere, interrogative/request openers);
- rejects conversational filler by pattern;
- classifies multi-sentence turns per sentence, so a durable half of
  "What do we use for caching? Our cache layer uses Redis 7." is not lost.

Measured effect on the 60-turn validation conversation:

```text
durable facts stored   4/6  →  6/6
chit-chat stored       5/5  →  0/5
memories after 60 turns  15 →   6   (11 were duplicate small talk)
```

### 8. Tests

37 → **154 tests**.

New files: `tests/unit/test_retrieval_policy.py` (38),
`tests/unit/test_tokenizers.py` (12), `tests/unit/test_adapter_signals.py` (17),
and `tests/integration/test_context_pipeline.py` (12, two of them against the
live pinned runtime).

Extended: `tests/unit/test_context_builder.py` (29 cases, 2 retained from the
scaffold), `tests/security/test_secrets.py` (+5: delimiter escape, role
smuggling, suspicious-memory exclusion, credential redaction), and
`tests/unit/test_memory_policy.py` (+6 regression cases for the classifier
repair).

`tests/fakes.py` gained `LooseFakeRuntime` (imprecise recall, honest `why()`)
plus `contradictions()` / `stale_memories()` support.

### 9. Repository lint baseline

The 13 pre-existing scaffold findings (import sorting, `typing.Callable`,
`E501`, an `E402`) were fixed alongside this phase, so `ruff check .` is now
clean across the whole repository rather than only across touched files.

## Validation performed

```text
.venv/bin/pytest -q
154 passed in 0.30s          (includes 4 live pinned-BrainOS tests)

.venv/bin/ruff check .
All checks passed!
```

No provider API key was used at any point; generation is exercised through
`FakeProvider`.

### Live runtime validation (60-turn conversation, pinned BrainOS)

Six durable facts were distributed across 60 turns (including a correction at
turn 44) between five recurring chit-chat turns. Token counts use the
dependency-free estimator.

| Configuration | mean tokens sent | full-context baseline | mean reduction |
| --- | --- | --- | --- |
| wide window (`recent_turn_budget=2048`) | 1486 | 1391 | **0.0%** |
| memory-first (`192` / `memory_budget=512` / `max_memories=6`) | 387 | 1391 | **72.2%** |
| memory-first tight (`128` / `384` / `4`) | 308 | 1391 | **77.8%** |

Retrieval quality on the same conversation: **6/6** questions had exactly the
evidence they needed in the memory block, with the correct memory at rank 1 in
every case. The corrected fact (PostgreSQL → MySQL 8) never leaked into a
production-database prompt at any length.

### Finding that constrains Phase 6

**Context reduction is driven almost entirely by the recent-history window, not
by memory selection.** With `recent_turn_budget` larger than the conversation,
"BrainOS mode" degenerates into full context *plus* memory overhead — 0.0%
reduction, 95 tokens worse than the baseline. The 72–78% figures only appear
once the window is smaller than the conversation.

Phase 6's baseline modes must therefore set `recent_turn_budget` per mode, or
Mode A and Mode D will not be distinguishable and the comparison will be
meaningless. This is recorded here because it is a property of the engine, not a
tuning preference.

### Known limitation: abstention is not yet achieved

For a question whose answer is absent ("What is the Project Atlas payroll
vendor?"), the best candidate scored relevance `0.344` against an absolute floor
of `0.12`, so 4 memories were still selected instead of none. The scale-free
relative floor (`0.55` of best) trimmed the tail — a true-match query went from
5 selected to 2 — but it cannot produce abstention, because when nothing matches
there is no strong head to compare against.

The absolute floor was deliberately **not** tuned to fix this. The measured
separation (true match ≈ `0.56` relevance / `0.90` lexical versus
best-of-nothing ≈ `0.34` / `0.32`) comes from one synthetic 60-turn
conversation; calibrating a threshold on it would be overfitting. Threshold
calibration belongs to the benchmark phases (7 and 8) where abstention accuracy
is a scored category.

## Phase 3 exit assessment

| Plan requirement | Status |
| --- | --- |
| Input: system + current message + recent conversation + memories | **Complete** |
| Output: model-ready message list | **Complete** — delimited, role-normalized, provider-neutral |
| `ContextBudget(max_tokens, recent_turn_budget, memory_budget, system_budget)` | **Complete** — plus `max_recent_turns` and `per_message_overhead`; validated and enforced |
| Retrieval policy: dedupe → relevance → conflict → recency → budget | **Complete** — every stage implemented, configurable, and audited |
| Accounting: the five required token fields | **Complete** — plus 21 more, including the full-context reference and per-stage drop counts |
| Retrieved memory is data, not instructions | **Complete** — delimiters, preamble, delimiter-breakout and role-prefix stripping, suspicious flagging |

## Files changed in Phase 3

```text
CONTEXT.md
README.md
docs/architecture.md
docs/brainos-integration.md
docs/evaluation.md
docs/limitations.md
docs/phase-3-context-construction.md   (this log)
docs/security.md
benchmarks/context_rot/generation.py   (lint only)
src/app/service.py
src/app/state.py
src/app/ui.py                          (lint only)
src/brain/__init__.py
src/brain/adapter.py
src/brain/context_builder.py           (rewritten)
src/brain/memory_policy.py             (Phase 2 repair)
src/brain/retrieval_policy.py          (new)
src/brain/tokenizers.py                (new)
src/brain/trace.py
src/evaluation/reports.py              (lint only)
src/evaluation/run.py                  (lint only)
tests/fakes.py
tests/integration/test_context_pipeline.py  (new)
tests/security/test_secrets.py
tests/unit/test_adapter_signals.py     (new)
tests/unit/test_context_builder.py
tests/unit/test_memory_policy.py
tests/unit/test_retrieval_policy.py    (new)
tests/unit/test_tokenizers.py          (new)
```

## Follow-up work

1. **Phase 4** — wire Gradio callbacks to `ConversationService`. Everything the
   panels need already exists: `context_messages`, `context_stats`,
   `context_report`, `memory_ranking`, `trace`, `inspect()`, and
   `BuiltContext.final_prompt()` for the "final prompt" view.
2. **Phase 6** — baseline modes must set `recent_turn_budget` per mode (see the
   finding above). Mode C/D/E need a retriever that shares this builder.
3. **Phases 7–8** — calibrate `relevance_floor` and `relative_relevance_ratio`
   against the benchmark, and score abstention explicitly. The audit reasons map
   onto the error taxonomy: `low_relevance`/`weak_relevance` → `irrelevant_memory`,
   `stale`/`expired` → `stale_memory`, `superseded` → `conflicting_memory`,
   `memory_budget`/`budget` → `over_compression`.
4. Consider whether recent history that duplicates a selected memory should be
   sent at all. Left out of Phase 3 on purpose: removing a user turn while
   keeping the assistant's reply risks incoherent conversation flow, and history
   window strategy is Phase 6 territory.
5. Optional exact tokenizer wiring per provider before research runs, so reported
   tokens match what the provider bills.
