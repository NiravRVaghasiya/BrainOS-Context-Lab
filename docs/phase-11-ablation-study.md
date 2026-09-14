# Phase 11 log — Ablation study

**Status:** complete (branch `arena/01a09f7a-brainos-context-lab`)
**Plan section:** Phase 11 — Ablation Study (§17).
**Depends on:** Phase 6 (mode profiles and windows), Phase 7 (benchmark and
scorer), Phase 8 (metric suite), Phase 9 (controlled experiment harness),
Phase 10 (paired statistics and effect sizes).

## Objective

From the plan:

> Test: BrainOS full, no temporal signal, no working memory, no
> consolidation, no relevance filtering, no conflict handling, no memory.
> Only include components actually available and stable in the integrated
> BrainOS version.
>
> Goal: which BrainOS components are responsible for the observed improvement?

Phase 10 left two concrete targets: the multi-hop evidence gap (Recall@K 1.0
but evidence-in-prompt 0.83 — the retrieval policy's relevance filtering
delivering only one of two recalled hops) and the open question of what each
pipeline stage contributes. Phase 11 answers by running Mode D with exactly
one component removed at a time, through the same controlled comparison and
paired statistics as the baselines.

## Starting state

- `baselines/modes.py` owned five modes; `ContextSettings.with_mode_defaults`
  rewrote budgets on a mode switch but had no notion of a component removal.
- Two retrieval knobs the ablations need (`relative_relevance_ratio`,
  `heuristic_conflict_detection`) existed on `RetrievalPolicy` but were not
  reachable from `ContextSettings`.
- The experiment runner accepted only the five modes (`--modes all`), and its
  statistical summary always paired against `full_context`.
- Upstream BrainOS at the pinned revision exposes `features` flags
  (`temporal_supersession`, `exclude_stale`, `consolidation`, …) and a
  `wm_slots` constructor argument — but the application never varies them.

## Work completed

### New / changed modules

| File | Purpose |
| --- | --- |
| [`src/baselines/ablations.py`](../src/baselines/ablations.py) (new) | The four runnable `AblationProfile` records (removal, hypothesis, policy overrides, runtime-report and record-stripping flags), `EXCLUDED_ABLATIONS` with the reason each plan component is *not* run, `prepare_records` / `strip_temporal_signals` / `strip_lifecycle`, and `ablation_pairs` (every ablation vs `brainos`). |
| [`src/baselines/modes.py`](../src/baselines/modes.py) | Four `BaselineMode` entries (letters D1–D4) sharing Mode D's window and evidence budget; `ABLATION_ORDER`; `resolve_mode` accepts ablations; `mode_choices()` stays the five baselines (ablations are evaluation-only). |
| [`src/app/state.py`](../src/app/state.py) | `ContextSettings` gains `relative_relevance_ratio` and `heuristic_conflict_detection` with pass-through to `retrieval_policy()`; `with_mode_defaults` applies an ablation's policy overrides on entry and restores the overridden knobs to defaults when leaving an ablation for a baseline. |
| [`src/app/service.py`](../src/app/service.py) | `_build_context` honours the active ablation: runtime contradiction/stale reports can be ignored and temporal/lifecycle record preparation applied before the builder runs. Baselines pass through unchanged. |
| [`src/evaluation/run.py`](../src/evaluation/run.py) | `--mode` accepts the four ablation ids. |
| [`src/evaluation/experiment.py`](../src/evaluation/experiment.py) | `--modes ablations` runs the full system plus the four ablations (an ablation without its reference is not interpretable, so the keyword always includes `brainos`); `--baseline-mode` (recorded on the artifact) selects the paired-comparison reference — `brainos` for ablation runs. |
| [`src/evaluation/analysis.py`](../src/evaluation/analysis.py) | Trial summaries order ablations after the baselines; `pairwise_comparisons` targets ablations and always pairs each present ablation against `brainos`, whatever baseline the caller chose. |
| Tests | `tests/unit/test_ablations.py` (19: registry, shared budgets, routing, knob application/restoration, stripping, pairing), `tests/evaluation/test_ablation_modes.py` (8: removal behaviour on fakes, routing, experiment pairing, CLI), `tests/integration/test_ablation_live.py` (7: the same removals against the pinned runtime, no key). **598 → 632**; `ruff check .` clean. |

### The ablation contract

```text
brainos               Mode D, the full reference every ablation is compared to
brainos_no_temporal   D1  weight_recency=0 + runtime recency/temporal_relevance
                          signals stripped before ranking
brainos_no_relevance  D2  relevance_floor=0 + relative_relevance_ratio=0
                          (ranking, cap, and token budget still apply)
brainos_no_conflict   D3  resolve/drop_stale/heuristic off + runtime reports
                          ignored + status/valid_until neutralized
brainos_no_memory     D4  Mode D's window with no memory injected
                          (unlike Mode B, which keeps eight turns)
```

Rules the implementation enforces:

- Every ablation keeps Mode D's history window and evidence budget
  (`EVIDENCE_BUDGET = 1024`; D4 keeps the window with a zero memory budget),
  keeps observing, and flows through the same service, builder, scorer, and
  paired statistics as the baselines. A measured gap is attributable to the
  removed component, not to prompt volume.
- `brainos` is the comparison reference. `ablation_pairs` never pairs two
  ablations, and `--modes ablations` always includes the full system.
- Leaving an ablation for a baseline restores the overridden knobs: the knobs
  *are* the ablation, so carrying `relevance_floor=0` into `brainos` would
  silently run a different system than the label claims. Baseline-to-baseline
  switches still keep the caller's tuning.
- The pinned runtime exposes no retrieval-weight configuration, so its
  internal top-k pre-ranking still uses temporal signals under D1; the
  ablation removes temporal influence from the application's selection stage,
  where the prompt is decided. This is documented on the profile, not hidden.

### Excluded components (documented, not silent)

| Plan component | Why it is not run at the pinned revision |
| --- | --- |
| no working memory | The runtime writes working memory (topic, entities, recalled fact) but retrieval never reads it back, and the adapter never calls `working_memory()`. Varying `wm_slots` changes trace snapshots only. |
| no consolidation | Consolidation is an explicit offline pass (`consolidate()`) the application never invokes; the observe → recall path runs no consolidation either way. A periodic-consolidation variant would be an addition to the pipeline, not an ablation of it. |

Both are recorded in `EXCLUDED_ABLATIONS` with these reasons. An ablation
that cannot move a metric must not be reported as a null result.

A related measurement pins the premise: after observing a fact and its
correction through the live runtime, `adapter.conflicts()` is `[]` and
`stale_memory_ids()` is empty — the observe path never versions by subject,
so the runtime's contradiction/stale reports are inert here and the
application's heuristic is the operative conflict handling D3 removes
(`test_live_runtime_reports_no_conflicts_on_the_observe_path`).

## Measured behaviour

Live pinned runtime, committed 7-task smoke dataset, estimated counter, no
provider. Retrieval side first (dry run, `constants.passed`, no violations):

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick \
  --modes ablations --dry-run --output results/phase11-dry.json \
  --baseline-mode brainos
PYTHONPATH=src .venv/bin/python -m evaluation.compare results/phase11-dry.json \
  --stats --baseline-mode brainos --output results/phase11-stat-report.json
```

```text
mode                  recall  evidence  mean_tokens
brainos               1.000   0.833     193.6
brainos_no_temporal   1.000   0.833     193.6
brainos_no_relevance  1.000   1.000     215.7
brainos_no_conflict   1.000   0.833     193.6
brainos_no_memory     0.000   0.000     97.0
```

Paired against the full system across the 7 tasks:

```text
Pair                         Metric               Diff      95% CI          p-val    d      W/L/T
no_temporal vs brainos       evidence_in_prompt   +0.0000   [±0.0000]       1.0000   0.00   0/0/7
no_temporal vs brainos       final_context_tokens +0.0000   [±0.0000]       1.0000   0.00   0/0/7
no_relevance vs brainos      evidence_in_prompt   +0.1429   [-0.21, +0.49]  0.3559   0.38   1/0/6
no_relevance vs brainos      final_context_tokens +22.1429  [+5.79, +38.49] 0.0161   1.25   5/0/2
no_conflict vs brainos       evidence_in_prompt   +0.0000   [±0.0000]       1.0000   0.00   0/0/7
no_conflict vs brainos       final_context_tokens +0.0000   [±0.0000]       1.0000   0.00   0/0/7
no_memory vs brainos         evidence_in_prompt   -0.7143   [-1.17, -0.26]  0.0082  -1.46   0/5/2
no_memory vs brainos         retrieval_recall     -0.8571   [-1.21, -0.51]  0.0010  -2.27   0/6/1
no_memory vs brainos         final_context_tokens -96.5714  [-113.11, -80.04] <0.0001 -5.40  0/7/0
```

Per-task, D2 repairs exactly the multi-hop gap (selected memories 1 → 3,
evidence 0 → 1, 187 → 235 tokens) while also selecting more on single-hop,
conflict, distractor, and cross-session — the precision cost of no filter.
D1 and D3 are byte-identical to full on all seven tasks. D4 selects nothing
anywhere: with the window alone, every distant fact is unreachable.

Answer-side plumbing check with scripted answers (same convention as
Phase 8 — validates the metrics move, not answer quality):

```text
mode                  graded  accuracy  faithfulness  reduction
brainos               7       1.000     0.857         0.832
brainos_no_relevance  7       1.000     1.000         0.813
brainos_no_memory     7       1.000     0.143         0.916
```

D2's extra evidence moves multi-hop faithfulness 0 → 1. D4's scripted
accuracy stays 1.0 while faithfulness collapses to 0.143 — six correct
guesses with no evidence in the prompt — and its quality-adjusted efficiency
is meaninglessly high (scripted accuracy over tiny prompts), which is why
QAE needs real generations before it means anything.

**Integration-validation observations, not research results** — one seed,
one length tier, estimated counter, no model in the loop.

## Findings this phase produced (carry into Phase 12)

1. **The multi-hop failure is the relevance filter, confirmed by removal.**
   Disabling the floor and tail trim restores the second hop (evidence 0.83 →
   1.0, faithfulness 0.86 → 1.0) for +22 tokens. The filter is simultaneously
   the precision mechanism (D2 selects more almost everywhere), so Phase 12's
   error taxonomy should separate `over_compression` (filter dropped needed
   evidence) from `missed_memory` (never recalled) — they now have different
   owners.
2. **Temporal and conflict handling are unexercised by the smoke tier, not
   unimportant.** D1/D3 are identical to full because one short session gives
   recency nothing to decide and retrieval-stage conflicts nothing to
   resolve — while the crafted live tests prove both removals fire when the
   condition exists. Longer tiers and multi-session transcripts are needed
   before these ablations can move.
3. **D4 bounds the memory contribution: all of it, on this dataset.**
   Evidence 0.83 → 0.0 with the window held constant (p = 0.008, d = −1.46).
   The window alone keeps nothing distant — which also means any future
   window change must re-run D4 rather than assume the split.
4. **The runtime's conflict reports are inert on the observe path.**
   Contradictions only populate for subject-versioned memories
   (`remember(subject=...)`), which the application never creates. If a
   future phase versions observations by subject, D3 must be re-run: it would
   then remove runtime *and* application handling together.

## Constraints carried into later phases

1. **Ablations are modes to the runner, not to the user.** They resolve
   through `resolve_mode`, replay through `task_evaluator`, and pair through
   `pairwise_comparisons` — but `MODE_ORDER`, the UI dropdown, and the preset
   defaults stay the plan's five baselines.
2. **Always run ablations with their reference.** `--modes ablations`
   includes `brainos`; `ablation_pairs` returns nothing without it.
3. **Knob restoration is load-bearing.** Leaving an ablation restores the
   overridden policy knobs to defaults; do not "simplify" `with_mode_defaults`
   back to budgets-only without replacing that guarantee.
4. **The smoke tier cannot support an ablation claim** (7 tasks, one length).
   Report ablations from `--tier standard`/`research` with a model in the
   loop and repeated trials, like any other comparison.
5. **Skips stay visible for ablations too.** D2 selects more memories per
   task; at the plan's longer lengths it will hit the input ceiling first,
   and that is a finding about the filter, not a crash to trim.

## Validation

```bash
.venv/bin/python -m pytest -q
# 632 passed  (598 before this phase; +34)

.venv/bin/ruff check .
# All checks passed!

PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick \
  --modes ablations --dry-run --output results/phase11-dry.json \
  --baseline-mode brainos
# table above; exit 0; constants passed

PYTHONPATH=src .venv/bin/python -m evaluation.run --mode brainos_no_relevance \
  --output results/run.json
# ablation ids work on the single-mode CLI too
```

No provider API key was used anywhere in this phase: the dry run builds
prompts only, scripted answers grade strings from a file, and the live tests
run the pinned runtime with no provider configured.
