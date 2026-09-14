# Phase 8 log — Metrics

**Status:** complete (branch `arena/01a09f1b-brainos-context-lab`)
**Plan section:** Phase 8 — Metrics (§12), Context Efficiency Metrics (§13),
Context-Rot Metrics (§14). Plot *series* are produced here; trial-level
statistics and rendered figures belong to Phase 10.
**Depends on:** Phase 7 (the scorer, the fact ledger, and `aggregate_scores`).

## Objective

From the plan: do not evaluate only token reduction. Measure quality (Recall@K,
Precision@K, answer accuracy, faithfulness to memory, conflict-resolution
accuracy, abstention accuracy), efficiency (context tokens, reduction, token
savings, quality-adjusted compression), and robustness (accuracy as a function
of conversation length, absolute and relative degradation, area under the
degradation curve).

Phase 7 left `aggregate_scores` descriptive only (counts/rates/means) and left
`quality_adjusted_efficiency` / `token_savings` uncalled. Phase 8 fills the
suite on top of the existing scorer — it does not replace the verdict rules.

## Starting state

- `evaluation/scoring.py` produced per-task `retrieval` / `answer` / `error_type`
  plus three cost fields. `aggregate_scores` reported rates with per-rate
  denominators and `by_category`.
- `evaluation/metrics.py` already had `recall_at_k`, `precision_at_k`,
  `context_reduction`, `token_savings`, `quality_adjusted_efficiency`, and
  `summarize`. Only `summarize` was used by the aggregate.
- `analysis.py`, `plots.py`, `reports.py`, and `compare.py` were thin
  scaffolding.
- Constraints carried in: report Recall@K **and** evidence-in-prompt (they
  disagree on multi-hop); do not "normalise" inverted abstention scoring; a
  single length cannot support a degradation claim; do not tune thresholds on
  the smoke tier; latency is optional because no model is in the loop.

## Work completed

### New / changed modules

| File | Purpose |
| --- | --- |
| [`src/evaluation/metrics.py`](../src/evaluation/metrics.py) | Phase 8 primitives: `METRICS_VERSION`, `CONFLICT_CATEGORIES`, `accuracy_degradation`, `relative_degradation`, `area_under_curve`, `area_under_degradation_curve`, `mean_degradation`, `rate`, `Summary.to_dict()`. Existing efficiency helpers are now called. |
| [`src/evaluation/scoring.py`](../src/evaluation/scoring.py) | `score_faithfulness`, `is_conflict_task`; `score_record` carries token savings, QAE, faithfulness, length tier, optional latency; `aggregate_scores` reports the full suite plus `by_length` and a degradation block that is `None` on a single length. |
| [`src/evaluation/analysis.py`](../src/evaluation/analysis.py) | Headline rows (`HEADLINE_KEYS`) and the six plan-required plot series, built without matplotlib. |
| [`src/evaluation/plots.py`](../src/evaluation/plots.py) | Renderers for those six series. Matplotlib remains the `evaluation` extra. |
| [`src/evaluation/reports.py`](../src/evaluation/reports.py) | `metrics_report` / `comparison_report` around the existing JSON writer. |
| [`src/evaluation/compare.py`](../src/evaluation/compare.py) | Writes the structured comparison and prints a compact table. |
| [`src/evaluation/run.py`](../src/evaluation/run.py) | Summary line now includes faithfulness and QAE. |
| Tests | `tests/unit/test_metrics.py` (2 → 8), `tests/evaluation/test_scoring.py` (+5), `tests/evaluation/test_analysis.py` (new, 6), live aggregate assertions: **494 → 503**. `ruff check .` clean. |

### Key design decisions

- **Extend the scorer, do not replace it.** Verdicts, error labels, inverted
  abstention scoring, and the Recall@K / evidence-in-prompt split are unchanged.
  Phase 8 adds fields and aggregates.
- **Faithfulness is grounding, not accuracy.** A correct answer whose required
  evidence never reached the prompt is a guess (unfaithful). A stale answer that
  repeats a superseded value that *was* in the prompt is faithful to retrieved
  memory and still a conflict-resolution failure. Ungraded records are `None`
  and excluded from the rate. The prompt is the grounding surface, so Mode A
  (no retriever) can still be faithful.
- **Conflict-resolution accuracy is scoped.** Only `conflict` and `temporal`
  tasks (the categories that plant a superseded value) enter the denominator.
- **A single length cannot support a curve.** The smoke tier has one
  `length_tier`. AUC fields are JSON `null` with an explicit note, not a silent
  `0.0` that looks like "no degradation".
- **Reference length is the shortest length in the run.** Callers that want a
  fixed baseline (Mode A at 5k) can pass `reference_accuracy` into the
  primitive; the aggregate uses the shortest tier present.
- **Two efficiency numbers, documented.** `mean_quality_adjusted_efficiency` is
  `E[quality_i / tokens_i]`; `quality_per_token` is `accuracy / mean_tokens`.
  They diverge when token counts vary. Neither is a research result on the
  smoke tier.
- **Latency is optional.** Replays call no model, so `mean_latency_ms` is
  `null` unless a future generation path puts `latency_ms` on the record.
  Missing latency is not reported as `0`.
- **Token counters are named.** The aggregate lists every `token_counter` seen
  so runs measured with different counters are not compared silently.
- **Plot series are data, plots are optional.** The six plan plots can be
  tested without matplotlib; rendering raises a clear error when the extra is
  absent. Trial-level CIs and effect sizes remain Phase 10.

### Faithfulness rules (the contract)

| Verdict | Evidence in prompt | Faithfulness |
| --- | --- | --- |
| `ungraded` | anything | `None` (excluded) |
| `correct` and abstention expected | n/a | `1.0` |
| `incorrect` and abstention expected | n/a | `0.0` (hallucination) |
| `correct` | yes | `1.0` |
| `correct` | no | `0.0` (guess) |
| `stale_answer` | forbidden value present | `1.0` (echoed retrieved stale memory) |
| `stale_answer` | forbidden value absent | `0.0` |
| `abstained` / `wrong_abstention` | no | `1.0` (honest decline) |
| `abstained` / `wrong_abstention` | yes | `0.0` |
| `incorrect` | anything | `0.0` |

## Measured behaviour (live pinned BrainOS, committed smoke dataset, no provider)

Scripted answers from the contract (the same plumbing check Phase 7 ran).
Estimated token counter. **Not a research result.**

```text
mode=brainos tasks=7 graded=7
recall=1.000  evidence_in_prompt=0.833  accuracy=1.000  faithfulness=0.857
conflict_resolution=1.000  abstention=1.000
reduction=0.832  mean_tokens=193.6  savings=959.1  qae=0.005200
latency=null  degradation.auc=null
("A single length cannot support a degradation curve.")
```

Accuracy 1.0 is the scripted-answer plumbing check. Faithfulness is **0.857**
because the multi-hop task has Recall@K 1.0 and evidence-in-prompt 0.0: the
scripted answer is correct and unfaithful — the metric split the Phase 7
finding required. `forbidden_in_prompt_rate` is 1.0 on the two conflict/temporal
tasks (stale values still reach the prompt); conflict-resolution accuracy is
1.0 only because the answers were scripted as the current value.

## Constraints carried into later phases

1. **Faithfulness, Recall@K, and evidence-in-prompt are three numbers.** They
   disagree on multi-hop; do not collapse them.
2. **The smoke tier still cannot support a claim**, and now the degradation
   block says so in the artifact (`auc: null`). Use `--tier standard`/`research`
   with multiple variants before reporting a curve.
3. **Do not present a single run's task-level mean as a trial CI.** `summarize`
   is the Phase 10 building block; this phase does not invent trials.
4. **Latency will stay null until a generation path exists** (Phase 9/15). Do
   not fill it with replay wall-clock.
5. **Do not tune thresholds on the smoke tier** (still in force).
6. **Grading still needs Phase 15 cost controls** before the Evaluation tab
   exposes a run.
7. **Keep the isolated replay in the suite.**

## Validation

```bash
.venv/bin/pytest -q          # 503 passed, 1 skipped (Gradio)
.venv/bin/ruff check .       # All checks passed!

PYTHONPATH=src .venv/bin/python -m evaluation.run \
  --mode brainos --dataset benchmarks/context_rot/dataset.jsonl \
  --answers /tmp/answers.jsonl --output results/brainos.json

PYTHONPATH=src .venv/bin/python -m evaluation.compare \
  results/full_context.json results/brainos.json --output results/compare.json
```

No provider API key was used. Answer quality on the smoke run is scripted.
Token numbers are `estimate_tokens`.
