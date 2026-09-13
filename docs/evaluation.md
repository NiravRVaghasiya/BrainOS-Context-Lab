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

`full_context_reference_tokens` prices the Mode A baseline for the *same* system
prompt and question (every history message replayed), so
`context_reduction_vs_full_context` isolates the context-management strategy.
`token_counter` names the counter that produced the numbers: a run is only
comparable to runs that used the same counter.

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

Two measurement cautions established during Phase 3 validation:

1. **Reduction depends on the history window.** With `recent_turn_budget` larger
   than the conversation, BrainOS mode costs *more* than full context (observed:
   1486 vs 1391 tokens, 0.0% reduction). Memory-first settings on the same
   conversation reached 72–78% reduction. Baseline modes must therefore fix
   `recent_turn_budget` per mode and report it.
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
