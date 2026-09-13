# BrainOS Context Lab — Working Context

> Living hand-off context for implementation work. This file records the
> repository state and decisions that should be preserved between phases.

**Last updated:** 2026-09-13
**Branch:** `arena/01a09b6a-brainos-context-lab`
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
| **Phase 4 — Chat web UI** | **Complete in this turn** | Gradio callbacks wired to a Gradio-free `UIController`: provider connect/validate/list-models, per-session chat, the five planned tabs, session controls (clear conversation / clear memory / end session / export), and a memory-mode selector. 216 tests, live-validated at 41 turns. |
| Phases 5–20 | Pending | Persistence, baselines, benchmark, evaluation, security hardening, deployment, and research release follow the plan. |

Detailed logs are available in
[`docs/phase-0-research-baseline.md`](docs/phase-0-research-baseline.md),
[`docs/phase-1-provider-abstraction.md`](docs/phase-1-provider-abstraction.md),
[`docs/phase-2-brainos-adapter.md`](docs/phase-2-brainos-adapter.md),
[`docs/phase-3-context-construction.md`](docs/phase-3-context-construction.md),
and [`docs/phase-4-chat-web-ui.md`](docs/phase-4-chat-web-ui.md).

## What was done in Phase 4 (this turn)

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
# 216 passed

.venv/bin/ruff check .
# All checks passed!   (whole repository, no exclusions)

.venv/bin/python app.py
# http://localhost:7860
```

Test count went 154 → **216** in this phase. The live tests in
`tests/integration/test_chat_controller_live.py`,
`tests/integration/test_context_pipeline.py`, and
`tests/integration/test_brainos_runtime.py` run against the pinned BrainOS
revision and skip when `brainos_runtime` is not installed; `tests/ui/` skips
when Gradio is absent. No provider API key was used; generation is exercised
through `FakeProvider` / `FakeLLMProvider`.

`.venv` is ignored and is only a local test environment.

## Next safe step

Implement **Phase 5 — Conversation persistence** (SQLite-backed
`ConversationStore`, `MemoryStore`, `EvaluationStore`). Everything it needs is
already isolated: a session is a `SessionState` carrying a `conversation_id`,
the transcript is a list of `{role, content}` dicts, and `UIController` is the
single place that mutates them.

Carry these constraints into Phase 5:

1. **Persistence must not widen the credential surface.** The API key lives on
   `ProviderConfig` in process memory and is excluded from every serialized
   view; a store must be handed sanitized payloads (`export_session()`,
   `safe_dict()`), never the state object itself.
2. **Phase 6 must set `recent_turn_budget` per baseline mode.** The Phase 4
   sweep shows Mode A and Mode D are indistinguishable while the history window
   is larger than the conversation (0.0% reduction at 1024 tokens).
3. Keep the service as the only writer of session state, so the UI, the
   evaluation runner, and the store cannot disagree about what a turn was.
4. Keep BrainOS behind `BrainMemoryAdapter`: no `brainos_runtime` import from
   the UI, providers, storage, or evaluation metrics.
5. Finish the Phase 15 cost controls (max input tokens, max output tokens,
   per-session budget, benchmark tiers) before the Evaluation tab exposes
   benchmark runs in Phase 17.
