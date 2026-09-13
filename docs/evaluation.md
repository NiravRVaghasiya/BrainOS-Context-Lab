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
