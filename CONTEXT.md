# BrainOS Context Lab — Working Context

> Living hand-off context for implementation work. This file records the
> repository state and decisions that should be preserved between phases.

**Last updated:** 2026-09-13
**Branch:** `arena/01a09b44-brainos-context-lab`
**Baseline commit:** `f1827e1c897b53dc869f186e392b80cb54c1c901`
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
| **Phase 3 — Context construction engine** | **Complete in this turn** | Full retrieval pipeline (dedupe → relevance → conflict → recency → budget), enforced token budgets with documented eviction order, 26-field accounting, runtime signal consumption, injection hardening, and a credential-leak repair. Validated live at 60 turns. |
| Phase 4 — Chat web UI | Pending | UI surfaces exist; callbacks are not wired to `ConversationService`. Every value the panels need is now produced (see "Next safe step"). |
| Phases 5–20 | Pending | Persistence, baselines, benchmark, evaluation, security hardening, deployment, and research release follow the plan. |

Detailed logs are available in
[`docs/phase-0-research-baseline.md`](docs/phase-0-research-baseline.md),
[`docs/phase-1-provider-abstraction.md`](docs/phase-1-provider-abstraction.md),
[`docs/phase-2-brainos-adapter.md`](docs/phase-2-brainos-adapter.md), and
[`docs/phase-3-context-construction.md`](docs/phase-3-context-construction.md).

## What was done in Phase 3 (this turn)

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
.venv/bin/pip install pytest ruff
.venv/bin/pip install "brainos-cli @ git+https://github.com/NiravRVaghasiya/BrainOS.git@1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc"

.venv/bin/pytest -q
# 154 passed

.venv/bin/ruff check .
# All checks passed!   (whole repository, no exclusions)
```

Test count went 37 → **154** in this phase. The two live tests in
`tests/integration/test_context_pipeline.py` and the two in
`tests/integration/test_brainos_runtime.py` run against the pinned BrainOS
revision and skip when `brainos_runtime` is not installed. No provider API key
was used; generation is exercised through `FakeProvider`.

`.venv` is ignored and is only a local test environment.

## Next safe step

Implement **Phase 4 — Chat web UI** by wiring Gradio callbacks to
`ConversationService`. The context contract is now stable and everything the
planned panels need is already produced per turn:

| Planned panel | Available now |
| --- | --- |
| Chat | `ConversationTurn.reply` / `state.messages` |
| Memory (stored, type, timestamp, retrieval count, relevance, source turn) | `service.inspect()["memories"]`, `MemoryRecord` fields |
| Memory (retrieved) | `turn.retrieved_memories`, `turn.memory_ranking` |
| Context (raw size, selected memories, final prompt, token savings) | `turn.context_stats`, `BuiltContext.final_prompt()` |
| Cognitive Trace | `turn.trace` + `turn.context_report` (stage-by-stage drops) |
| Evaluation | still Phase 17; `evaluation/runner.py` remains a shell |

Carry these constraints into Phase 4:

1. Configure context through `ContextSettings` only — it builds both the budget
   and the policy, so the UI cannot desynchronize them.
2. Render diagnostics from the sanitized values the service already returns;
   never re-serialize `MemoryRecord.metadata` or provider config by hand.
3. Surface the `contested` label and the drop reasons; they are the user-visible
   form of the conflict and relevance stages.
4. Keep BrainOS behind `BrainMemoryAdapter`: no `brainos_runtime` import from
   the UI, providers, or evaluation metrics.
5. Add cost/turn limits (Phase 15) before exposing benchmark controls.
