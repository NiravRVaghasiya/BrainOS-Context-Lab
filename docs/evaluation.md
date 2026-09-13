# Evaluation protocol

## Research question

Does external BrainOS cognitive memory reduce historical context while preserving task performance during long-running LLM interactions?

## Required comparisons

For each task and model, compare:

1. Full Context
2. Sliding Window
3. Conventional RAG
4. BrainOS
5. BrainOS + RAG, when enabled

Keep the model, temperature, generation parameters, benchmark tasks, task wording, and scoring procedure constant.

## Task categories

The context-rot dataset should include:

- single-hop retrieval,
- multi-hop retrieval,
- temporal reasoning,
- conflict resolution,
- distractor resistance,
- cross-session memory, and
- correct abstention when information is absent.

Evaluate at increasing conversation lengths, such as 5k, 10k, 20k, 40k, 80k, and 120k tokens where model limits and budget permit.

## Dataset (Phase 7)

The dataset is generated, not hand-written:
[`benchmarks/context_rot/spec.py`](../benchmarks/context_rot/spec.py) holds the
vocabulary, templates, categories, and tiers;
[`generation.py`](../benchmarks/context_rot/generation.py) turns them into
deterministic tasks; [`MANIFEST.json`](../benchmarks/context_rot/MANIFEST.json)
pins the committed file by generator version, seed, and SHA-256. Each task
carries a **fact ledger** (every fact, its markers, where it was planted, and its
supersession chain) and an **evidence contract** (required / supporting /
forbidden facts, plus whether abstention is the correct behaviour).

Two validity properties are enforced by tests, because a benchmark that violates
them produces plausible-looking numbers that mean nothing:

1. every planted fact is storable by the application's own memory policy — a fact
   that is never stored cannot be retrieved by any mode;
2. no filler turn is storable — filler is there to make the conversation long,
   not to pollute memory.

Tiers: `smoke` (800 tokens, committed), `quick` (2k/4k), `standard`
(5k/10k/20k/40k), `research` (5k/10k/20k/40k/80k/120k). Lengths are estimated,
and every run records which token counter produced the numbers.

## Scoring (Phase 7)

Scoring is split in two so that most of a run needs no credentials:

- **Retrieval scoring is model-free.** Given the evidence a mode selected and
  the prompt it produced, facts are detected by marker co-occurrence and scored
  with Recall@K / Precision@K, plus whether every required fact reached the
  prompt and whether a forbidden (stale) fact was present. Mode A retrieves
  nothing and still reports evidence-in-prompt — the two are deliberately
  separate fields.
- **Answer scoring needs an answer string, never a model call.** Answers are
  supplied with `--answers` (JSONL: `{task_id, mode?, answer}`), so a real model
  run, a re-grade, or a deterministic mock all go through the same rules.

Verdicts: `correct`, `abstained`, `wrong_abstention`, `stale_answer`,
`incorrect`, `ungraded`. When abstention is expected — the abstention category,
or a cross-session task replayed with `--session-isolation` — declining is the
only correct behaviour; asserting a value there is a hallucination. Every
non-correct verdict carries a label from the error taxonomy above, with two
refinements that need both halves of a record: declining although the evidence
was in the prompt is `wrong_abstention`, and an incorrect answer whose evidence
was missing from the prompt is `missed_memory` rather than `hallucination`.

The aggregate a run reports is descriptive only: counts, rates, and means, each
with its own denominator, plus a per-category breakdown. Confidence intervals
across trials, paired comparisons, effect sizes, and the planned plots belong to
Phases 8 and 10.

## Metrics

### Quality

- Retrieval Recall@K
- Retrieval Precision@K
- answer accuracy
- faithfulness to retrieved evidence
- conflict-resolution accuracy
- abstention accuracy

### Efficiency

- context tokens sent to the model,
- context reduction percentage,
- absolute token savings,
- latency, and
- quality-adjusted efficiency.

Phase 3 emits these per request from `ContextStats.to_dict()`, so no later phase
has to reconstruct them:

```text
raw_history_tokens            recent_history_tokens      retrieved_memory_tokens
system_tokens                 final_context_tokens       memory_block_tokens
current_message_tokens        candidate_memory_tokens    overhead_tokens
full_context_reference_tokens token_savings_vs_full_context
context_reduction_vs_full_context
token_savings_vs_raw_history  context_reduction_vs_raw_history
budget_utilization            max_tokens                 token_counter
history_messages_considered   history_messages_selected  selected_memory_count
candidate_memory_count        dropped_history_for_budget dropped_memories_for_budget
system_truncated              exceeds_max_tokens
```

Phase 6 added the retrieved-transcript fields, so a mode's evidence spend can be
attributed to its source. Every mode reports the same set; a mode that uses no
chunks reports zeros, so a cross-mode table needs no special cases:

```text
selected_chunk_count          retrieved_chunk_tokens     chunk_block_tokens
candidate_chunk_count         candidate_chunk_tokens     dropped_chunks_for_budget
evidence_tokens               (= memory_block_tokens + chunk_block_tokens)
```

`full_context_reference_tokens` prices the Mode A baseline for the *same* system
prompt and question (every history message replayed, no evidence blocks — Mode A
has none), so `context_reduction_vs_full_context` isolates the
context-management strategy. `token_counter` names the counter that produced the
numbers: a run is only comparable to runs that used the same counter.

Retrieval quality is reconstructable from `RetrievalReport` without re-running
retrieval. It records `candidate_count`, `selected_ids`, the per-memory ranking
components (`score`, `relevance`, `lexical`, `runtime`, `recency`, `rank`), and
a reason for every dropped memory. Reasons map onto the error taxonomy:

| Drop reason | Taxonomy |
| --- | --- |
| `low_relevance`, `weak_relevance` | `irrelevant_memory` |
| `stale`, `expired` | `stale_memory` |
| `superseded` | `conflicting_memory` |
| `memory_budget`, `budget`, `cap` | `over_compression` |
| `duplicate`, `empty` | noise removed, not a failure |
| `suspicious` | injection guard (Phase 13) |
| `chunk_budget`, `chunk_ceiling` | `over_compression`, retrieved-chunk source only |
| `duplicate_history` | noise removed — the chunk was already in the history window |

Chunk reasons are namespaced so a Precision@K computation over `dropped` can
still isolate *memory* drops. Do not fold them into the memory counts.

Two measurement cautions established during Phase 3 validation:

1. **Reduction depends on the history window.** With `recent_turn_budget` larger
   than the conversation, BrainOS mode costs *more* than full context (observed:
   1486 vs 1391 tokens, 0.0% reduction). Memory-first settings on the same
   conversation reached 72–78% reduction. Phase 6 acted on this: every
   baseline mode now fixes its own `recent_turn_budget` and `max_recent_turns`,
   switching modes rewrites them, and the active window is reported in
   `context_payload()` and the UI.
2. **Token estimates are not provider tokens.** Validation used the
   dependency-free estimator (~4 chars/token). Research runs should use an exact
   counter and report which one.

### Robustness

Report accuracy as a function of conversation length, degradation from a reference length, relative degradation, and area under the degradation curve.

## Repeated trials

Stochastic generation requires multiple trials. Report mean, standard deviation, 95% confidence intervals, paired comparisons, and effect sizes where appropriate. Do not report only a best run.

## Reproducibility metadata

Each run must record a run ID, timestamp, application version, BrainOS revision, provider, model, temperature, benchmark revision, mode, context budget, task IDs, raw metrics, and aggregate metrics. Never record API keys.

## Error analysis

Every failure should be classifiable as one of:

```text
missed_memory, wrong_memory, stale_memory, conflicting_memory,
irrelevant_memory, hallucination, over_compression,
under_compression, wrong_abstention
```

Aggregate metrics are not sufficient without representative failure records.
