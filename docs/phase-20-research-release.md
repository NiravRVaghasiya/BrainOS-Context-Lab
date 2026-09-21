# Phase 20 — Research Release (in progress)

**Plan reference:** §26 (Research Release), §27 Milestone 6, §29 (Benchmark Matrix), §14 (length ladder)
**Status:** **in progress** — the dataset half and the model-free half are done; two findings below have to be
resolved before the model half is worth paying for. **Nothing here is a model result.**
**Datasets:** committed `smoke` tier (7 tasks) · generated `standard` (56 tasks, 5k–40k) · `research` (42 tasks,
5k–120k) · `top-rung` (7 tasks, 120k) · `quick` (14 tasks, 2k–4k) · `curve` (9 tasks, 5k–20k). Manifests are
committed; datasets are local.

## Why this phase is being done in two halves

The audit (`docs/status-audit-2026-09-21.md`) ranked the gaps: first, **the experiment has never run with a
model in the loop**; second, the committed dataset is a smoke tier that cannot answer any of §29's questions.
The first needs a key and a budget; the second needs neither. Everything short of the model call is
deterministic and local, so it can be done, verified and recorded now — and it is what makes the model call a
single command later:

1. generate the tiers §29 needs (more than one length, more than one variant per task);
2. run the pipeline over them **without** a model, which exercises the retrieval side, the token accounting,
   the aggregation, the statistics, the plots and the report at research scale;
3. record what these numbers do **not** support, the cost basis of the remaining run, and what has to change
   before that run is worth its money.

Point 3 is why this phase is not just preparation: the model-free runs found a measurement bug that would have
made §29's central comparison meaningless.

## What was generated

```bash
python benchmarks/context_rot/generation.py --tier standard --variants 2 --seed 20260913 \
  --output benchmarks/context_rot/generated/standard.jsonl \
  --manifest benchmarks/context_rot/generated/standard.manifest.json
python benchmarks/context_rot/generation.py --tier research --variants 1 --seed 20260913 \
  --output benchmarks/context_rot/generated/research.jsonl \
  --manifest benchmarks/context_rot/generated/research.manifest.json
python benchmarks/context_rot/generation.py --lengths 120000 --variants 1 --seed 20260913 \
  --output benchmarks/context_rot/generated/top-rung.jsonl \
  --manifest benchmarks/context_rot/generated/top-rung.manifest.json
python benchmarks/context_rot/generation.py --tier quick --variants 1 --seed 20260913 \
  --output benchmarks/context_rot/generated/quick.jsonl --manifest benchmarks/context_rot/generated/quick.manifest.json
python benchmarks/context_rot/generation.py --lengths 5000,10000,20000 \
  --categories single_hop,temporal,abstention --variants 1 --seed 20260913 \
  --output benchmarks/context_rot/generated/curve.jsonl --manifest benchmarks/context_rot/generated/curve.manifest.json
```

| Tier | Tasks | Lengths (target tokens) | Variants | Mean achieved | SHA-256 (first 16) | Size |
| --- | --- | --- | --- | --- | --- | --- |
| `quick` | 14 | 2k / 4k | 1 | 3,120.1 | `b377ba7da3d9f9be` | 1.3 MB |
| `standard` | 56 | 5k / 10k / 20k / 40k | 2 | 19,848.0 | `4f9d0925e12a6e74` | 8.1 MB |
| `research` | 42 | 5k … 120k (plan §14 ladder) | 1 | 49,300.5 | `acf712d76569839d` | 15.0 MB |
| `curve` | 9 | 5k / 10k / 20k | 1 | 12,639.7 | `4fce52c4093cedc0` | 2.0 MB |
| `top-rung` | 7 | 120k | 1 | 133,419.7 | `7eea7eadbbe244d5` | 6.7 MB |

Generation is fast — all of the above together took under 10 seconds — and the achieved length **overshoots**
the target at the top of the ladder (a 120k target lands at a mean of 133,419.7 estimated tokens, because
packing rounds up). The `standard` tier alone contains 99,266 messages (3,204–4,124 per 40k task).

`generated/` holds the datasets and their manifests; `.gitignore` keeps the JSONL local and **commits the
manifests**, so a run's `dataset_sha256` can be tied to the parameters that produced it.

## The model-free runs

Both runs are `--dry-run`: no provider, no key, no generation, and `graded = 0` throughout — the answer-side
metrics are `—`, not zero. Retrieval, token accounting, aggregation and the report all still run.

```bash
.venv/bin/python -m evaluation.pipeline --dry-run --no-plots \
  --dataset benchmarks/context_rot/generated/quick.jsonl --preset standard --output-dir /tmp/quick-run
# 14/14 task(s) × 5 mode(s) × 1 trial(s), exit 0; 5 run files + 4 aggregates; report 19,997 bytes

.venv/bin/python -m evaluation.pipeline --dry-run --no-plots \
  --dataset benchmarks/context_rot/generated/curve.jsonl --preset standard --output-dir /tmp/curve-run
# 9/9 task(s) × 5 mode(s) × 1 trial(s); experiment 710.58 s; 5 run files; 28 paired comparison(s);
# 20 failure record(s) (0 observed, 20 latent); report 19,727 characters; scan files=12 findings=0; exit 0
```

Both runs write the Phase 6/7 schema per mode, four aggregates (`experiment`, `comparison`, `statistics`,
`errors` — six plot series each) and a report; the quick run's figures were skipped with `--no-plots`.

Mean prompt tokens per mode, and the same figure per conversation length:

| Mode | `quick` (2k / 4k) | `curve` (5k / 10k / 20k) | mean reduction, `curve` | Recall@K | evidence in prompt |
| --- | --- | --- | --- | --- | --- |
| A `full_context` | 2,907 → 4,089 | 4,086 / 4,090 / 4,090 | 0.670 | 0.000 | 0.500 |
| B `sliding_window` | 185 → 189 | 189 / 189 / 183 | 0.985 | 0.000 | 0.000 |
| C `rag` | 273 → 274 | 279 / 290 / 290 | 0.977 | 1.000 | 1.000 |
| D `brainos` | 196 → 190 | 208 / 237 / 236 | 0.982 | 1.000 | 1.000 |
| E `brainos_rag` | 371 → 368 | 388 / 431 / 433 | 0.967 | 1.000 | 1.000 |

The retrieval side behaves as the earlier phases predicted: the four retrieval modes stay nearly flat (B, C, D
within ~50 tokens of their 5k value at 20k) while Mode A's prompt would normally grow with the conversation,
and only D/C/E reach the required evidence (`recall = 1.000`) — B, the window control, reaches none of it.
The `quick` tier shows the same shape across all seven categories at 2k and 4k.

## Finding 1 — Mode A is capped at 4,096 tokens, so the comparison above is not the one §29 asks for

Mode A is defined as "replay the entire conversation with no retrieval — **the reference price every other
mode is measured against**" (`src/baselines/modes.py`). Its history budget is `None`, which
`BaselineMode.budget_overrides` turns into `max_tokens` — the *session* ceiling — and the benchmark's replay
factory builds its session with a plain `ContextSettings()`, whose default is `max_tokens = 4096`
(`src/app/state.py:63`, used at `src/evaluation/modes.py:171`).

The measurement shows the cap exactly: Mode A's prompt grows 2,907 → 4,089 tokens between 2k and 4k and then
**stops**: 4,086 / 4,090 / 4,090 at 5k / 10k / 20k. Every `mean_reduction` of 0.97–0.99 above is therefore
measured against a *truncated* reference, and a degradation curve over the 5k–20k rungs would be measuring
where Mode A stops truncating, not where the retrieval modes start degrading. On the plan's ladder
(up to 120k) it would be worse.

This is the one thing that must be settled before the model half is worth paying for. The options:

1. size the replay session's ceiling to the task (`max_tokens = max(4096, target_tokens + slack)`) so Mode A
   really replays the conversation, and keep the run-level `--max-input-tokens` ceiling as the guard;
2. declare Mode A a *4k window* baseline for long contexts, rename it accordingly, and add a separate
   whole-conversation reference — which changes what §29's matrix means;
3. cap the dataset at lengths where Mode A is not truncated (≤4k), which abandons the length curve.

Option 1 is the smallest change that makes the plan's comparison real; it costs replay time (Mode A's prompt
becomes the conversation, on a harness that rebuilds context every turn — see finding 2).

## Finding 2 — the harness replays every message, so cost grows faster than the dataset

`src/evaluation/modes.py::replay_task` runs each message of the conversation through the normal service path
and builds a context for each one, which is what makes its numbers comparable to the product. The cost of that
choice only shows past the smoke tier:

| Run | Tasks | Cost |
| --- | --- | --- |
| `quick` (2k / 4k) | 14 × 5 modes | ~2 minutes |
| `curve` (5k / 10k / 20k) | 9 × 5 modes | **710.58 s** |
| `standard` (5k … 40k) | 56 × 5 modes | started, stopped after **>70 minutes of CPU**, 40k band unfinished |
| `top-rung` (120k) | 1 task × 5 modes | stopped after **>50 minutes of CPU**, unfinished |

That is local CPU, not API cost — but it bounds the real run: the `standard` tier is 99,266 messages ≈
**496,000 context builds**, and the 80k / 120k rungs are not practical on one core until the replay is made
cheaper (skipping the context rebuild for turns that only need the transcript, or sharding tasks across
workers). The plan's length curve can be reported from `standard` meanwhile. Nothing about this affects the
Space, which replays one visitor's conversation rather than a benchmark task.

## What these numbers are not

- **Not accuracy.** With no model in the loop, `graded = 0` and every answer-level metric (accuracy,
  faithfulness, conflict resolution, abstention, degradation AUC) is `—`.
- **Not §26.** The release document needs the §29 matrix, which needs the model.
- **Not a comparison yet**, for the reason in finding 1: Mode A's reference is truncated above ~4k tokens.
- **Not a leaderboard.** The dataset is template-generated and the run is retrieval-only.

## The one command that remains (after finding 1 is settled)

```bash
brainos-context-pipeline --preset standard --generate \
  --model <model> --api-key-env OPENAI_API_KEY \
  --dataset benchmarks/context_rot/generated/standard.jsonl
```

The pipeline reports token ceilings, not dollars (`cost_estimate` is "ceilings, not estimates"), so the
budget is arithmetic on the run's own numbers: 56 tasks × 5 modes = **280 requests**, input tokens on the
order of the sums in the table above, output bounded by `--max-output-tokens` (1,024 in the `standard`
preset). The repository deliberately ships no price table — multiply by the provider's current price.

## Constraints carried forward

1. **Do not publish retrieval-only numbers as §26 results**, and do not publish the `mean_reduction` figures
   until finding 1 is resolved — they are measurements against a truncated reference.
2. **Keep the manifests committed** when a tier changes, and regenerate rather than edit a dataset.
3. **Preset ceilings are part of the dataset contract.** `standard` allows 64k input tokens, so the 80k /
   120k rungs need `--preset research` (200k); mixing them under `standard` would be silently truncated at
   the ceiling.
4. **The plan's phases are still the contract.** §26's document is written from run manifests, and the audit's
   independent verdict stands until a model run changes it.
