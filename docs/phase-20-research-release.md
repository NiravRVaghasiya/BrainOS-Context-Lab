# Phase 20 — Research Release (in progress)

**Plan reference:** §26 (Research Release), §27 Milestone 6, §29 (Benchmark Matrix), §14 (length ladder)
**Tests:** 1335 → **1349** (+14) · coverage **94.19%** (floor 90) · `ruff check .` clean
**CI:** run `35597471402` (commit `25e2e56`) green — `lint, test, coverage floor` 6m43s · `suite runs without the optional extras` 35s
**Status:** **in progress** — the dataset half and the model-free half are done, and the measurement bug the
model-free half exposed is **fixed and re-measured**. The model runs, §26's document and the Space remain.
**Nothing here is a model result.**
**Datasets:** committed `smoke` tier (7 tasks) · generated `quick` (14, 2k–4k) · `standard` (56, 5k–40k) ·
`curve` (9, 5k–20k) · `research` (42, 5k–120k) · `top-rung` (7, 120k). Manifests are committed; datasets are
local.

## Why this phase is being done in two halves

The audit (`docs/status-audit-2026-09-21.md`) ranked the gaps: first, **the experiment has never run with a
model in the loop**; second, the committed dataset is a smoke tier that cannot answer any of §29's questions.
The first needs a key and a budget; the second needs neither. Everything short of the model call is
deterministic and local, so it can be done, verified and recorded now:

1. generate the tiers §29 needs (more than one length, more than one variant per task);
2. run the pipeline over them **without** a model, which exercises the retrieval side, the token accounting,
   the aggregation, the statistics, the plots and the report at research scale;
3. record what these numbers do **not** support, the cost basis of the remaining run, and what had to change
   before that run is worth its money.

Point 3 is why this phase is not just preparation: the model-free runs found a measurement bug that would
have made §29's central comparison meaningless, and the fix is part of this phase.

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
| `curve` | 9 | 5k / 10k / 20k | 1 | 12,639.7 | `4fce52c4093cedc0` | 2.0 MB |
| `research` | 42 | 5k … 120k (plan §14 ladder) | 1 | 49,300.5 | `acf712d76569839d` | 15.0 MB |
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
  --dataset benchmarks/context_rot/generated/quick.jsonl --preset standard --output-dir /tmp/quick-run-fixed
# 14/14 task(s) × 5 mode(s) × 1 trial(s), exit 0; 5 run files + 4 aggregates; report 19,667 characters

.venv/bin/python -m evaluation.pipeline --dry-run --no-plots \
  --dataset benchmarks/context_rot/generated/curve.jsonl --preset standard --output-dir /tmp/curve-run-fixed
# 9/9 task(s) × 5 mode(s) × 1 trial(s); experiment 759.03 s; 5 run files; 28 paired comparison(s);
# 18 failure record(s) (0 observed, 18 latent); report 19,385 characters; scan files=12 findings=0; exit 0
```

Mean prompt tokens per mode, per conversation length:

| Mode | `quick` (2k / 4k) | `curve` (5k / 10k / 20k) | mean reduction, `curve` | Recall@K | evidence in prompt |
| --- | --- | --- | --- | --- | --- |
| A `full_context` | 2,907 → 5,707 | **7,307 / 13,956 / 30,417** | 0.000 | 0.000 | **1.000** |
| B `sliding_window` | 185 → 189 | 189 / 189 / 183 | 0.985 | 0.000 | 0.000 |
| C `rag` | 273 → 274 | 279 / 290 / 290 | 0.977 | 1.000 | 1.000 |
| D `brainos` | 196 → 190 | 208 / 237 / 236 | 0.982 | 1.000 | 1.000 |
| E `brainos_rag` | 371 → 368 | 388 / 431 / 433 | 0.967 | 1.000 | 1.000 |

This is the shape §29 asks for: the reference grows with the conversation (7.3k → 14.0k → 30.4k tokens while
the nominal length doubles) and the four retrieval modes stay flat — B, C and D within ~50 tokens of their 5k
value at 20k. Only C, D and E reach the required evidence (`recall = 1.000`); B, the window control, reaches
none of it, and D reaches it for 83% of the quick tier's tasks against the window's 0%.

## Finding 1 — Mode A was capped at 4,096 tokens (fixed in this phase)

Mode A is defined as "replay the entire conversation with no retrieval — **the reference price every other
mode is measured against**" (`src/baselines/modes.py`). Its history budget is `None`, which
`BaselineMode.budget_overrides` turns into `max_tokens` — the session ceiling — and the benchmark's replay
factory built its session with a plain `ContextSettings()`, whose default is `max_tokens = 4096`
(`src/app/state.py:63`).

The measurement showed the cap exactly, and it was not cosmetic:

| | before the fix | after the fix |
| --- | --- | --- |
| Mode A, `curve` (5k / 10k / 20k) | 4,086 / 4,090 / 4,090 | 7,307 / 13,956 / 30,417 |
| Mode A, `quick` (2k / 4k) | 2,907 / 4,089 | 2,907 / 5,707 |
| Mode A mean reduction | 0.670 (truncation, not saving) | 0.000 (it is the reference) |
| Mode A evidence in prompt | 0.500 (`curve`) / 0.750 (`quick`) | **1.000** / **1.000** |

The last two rows are why this mattered: a task in four was **losing its evidence to the truncation**, and
every other mode's `mean_reduction` of 0.97–0.99 was partly the reference shrinking rather than the mode
saving. The retrieval-side numbers for B/C/D/E did not move at all (186.9 / 286.1 / 226.9 / 417.0 mean tokens
before and after) — which is the check that the fix changed the reference and nothing else.

The fix, in `src/evaluation/modes.py`:

* `replay_ceiling(task)` returns the ceiling a replay needs — the transcript's estimated tokens (the same
  counter and per-message overhead the cost accounting uses) plus 1,024 tokens of room for the system prompt
  and the question, never below the product default. Malformed transcript entries are skipped the way the
  replay skips them.
* `_raise_session_ceiling(service, ceiling)` applies it to whatever service the factory returned — the default
  factory or an injected one — through the same settings object the sidebar's mode switch edits, and re-derives
  the mode's budgets from the new ceiling. That second step is the load-bearing one: Mode A's history window
  *is* the ceiling, so raising only `max_tokens` would have moved the wall without moving the window.
* Regression tests: `tests/evaluation/test_mode_strategies.py` (the ceiling covers the transcript, is never
  lowered below the default, is raised only when needed, and Mode A carries a transcript larger than 4k with
  its whole history intact) and `tests/integration/test_baseline_modes_live.py` (the same, against the pinned
  runtime).

The product's 4,096-token ceiling is untouched for live sessions; only a benchmark replay sizes its own.

## Finding 2 — the harness replays every message, so cost grows faster than the dataset

`src/evaluation/modes.py::replay_task` runs each message of the conversation through the normal service path
and builds a context for each one, which is what makes its numbers comparable to the product. The cost of that
choice only shows past the smoke tier:

| Run | Tasks | Cost |
| --- | --- | --- |
| `quick` (2k / 4k) | 14 × 5 modes | ~2 minutes |
| `curve` (5k / 10k / 20k) | 9 × 5 modes | 711 s before the fix, **759 s** after |
| `standard` (5k … 40k) | 56 × 5 modes | started, stopped after **>70 minutes of CPU**, 40k band unfinished |
| `top-rung` (120k) | 1 task × 5 modes | stopped after **>50 minutes of CPU**, unfinished |

That is local CPU, not API cost — but it bounds the real run: the `standard` tier is 99,266 messages ≈
**496,000 context builds**, and the 80k / 120k rungs are not practical on one core until the replay is made
cheaper (skipping the context rebuild for turns that only need the transcript, or sharding tasks across
workers). The plan's length curve can be reported from `standard` meanwhile, and the `curve` tier already
gives three points. Nothing about this affects the Space, which replays one visitor's conversation rather than
a benchmark task.

## What these numbers are not

- **Not accuracy.** With no model in the loop, `graded = 0` and every answer-level metric (accuracy,
  faithfulness, conflict resolution, abstention, degradation AUC) is `—`.
- **Not §26.** The release document needs the §29 matrix, which needs the model. These are the retrieval
  half of it.
- **Not a leaderboard.** The dataset is template-generated, one seed, one trial, an estimated token counter.

## The one command that remains

```bash
brainos-context-pipeline --preset standard --generate \
  --model <model> --api-key-env OPENAI_API_KEY \
  --dataset benchmarks/context_rot/generated/standard.jsonl
```

The pipeline reports token ceilings, not dollars (`cost_estimate` is "ceilings, not estimates"), so the
budget is arithmetic on the run's own numbers: 56 tasks × 5 modes = **280 requests**, input tokens on the
order of the sums in the table above (Mode A carries the conversation, the retrieval modes carry a few hundred
tokens), output bounded by `--max-output-tokens` (1,024 in the `standard` preset). The repository deliberately
ships no price table — multiply by the provider's current price.

## Milestone 6: the Space half of the release

The audit's fourth gap was that §27's "HF Release" milestone was not a release:
`app.py`, `requirements.txt` and `packages.txt` were Space-ready, but `README.md`
carried **no Hugging Face front matter**, so a Space created from this repository
would have had nothing to build, and a Space operator had no way to narrow the
Evaluation tab without editing source — the kind of edit that drifts from the
version the tests pin.

What changed:

1. `README.md` opens with the manifest a Space build reads — `sdk: gradio`,
   `sdk_version: 6.28.0` (the version this checkout is validated against),
   `app_file: app.py`, title, emoji, licence, short description.
2. [`docs/deployment.md`](deployment.md) is the operator document: the variable
   table, two recommended postures (a public retrieval-only Space; a research
   deployment with the plan's own defaults), and the checklist that ends with
   "record the Space URL in the README and CONTEXT".
3. `EvaluationPolicy.from_environment()` (`src/app/evaluation.py`) reads five
   deployment variables — `BRAINOS_LAB_EVAL_PRESETS`, `..._MAX_TASKS`,
   `..._MAX_REQUESTS`, `..._ALLOW_GENERATION`, `..._RETENTION_DAYS` — and
   `create_app()` builds its controller with them. They can only **tighten**:
   `apply` still enforces the preset's ceilings underneath. A value that cannot
   be obeyed (an unknown preset, `twenty` where a number belongs) raises at
   startup with the variable's name in the message, because a public deployment
   should fail where its operator can see it rather than quietly host the widest
   policy the code allows.
4. Ten new tests in `tests/unit/test_deployment_config.py` pin the manifest (its
   keys, that `app_file` exposes the entry point, and that the manifest's SDK
   major matches `requirements.txt`), the variable table against the code, and
   every branch of the narrowing — including that `..._ALLOW_GENERATION=0` makes
   the tab refuse a run that would spend a visitor's key, and that a policy can
   never widen a preset's ceiling.

**No Space exists yet.** The MVP checklist's first item is the one item this
repository cannot satisfy by itself; the checklist in `docs/deployment.md` ends
with the two files that must name the URL once it does.

## Verification of the fix

```bash
# The suite, every extra installed (after the ceiling fix and the deployment work)
# Required test coverage of 90.0% reached. Total coverage: 94.19%
# 1349 passed in 256.70s (0:04:16)

# The deployment tests do not need the optional extras (the `without-extras` job
# runs them on a base install)
$ PYTHONPATH=<base-install-simulator> pytest tests/unit/test_deployment_config.py -q
# 21 passed in 0.12s

# The live replay, pinned runtime (the new regression test included)
$ pytest tests/integration/test_baseline_modes_live.py -q
# 12 passed in 5.67 s

# The shipped surface (114 files once docs/deployment.md joined it)
# scan_version=scan-v1 files=114 bytes=32850981 findings=0 clean=True

# The store change that goes with it: one full suite failed this test once and
# never again, so the retry budget is now a constant with a name (1 s, was 0.25 s)
# and the test reports which of its two invariants broke first
$ pytest tests/unit/test_sqlite_concurrency.py tests/security/test_artifact_scan.py -q
# 86 passed in 3.94 s
```

## Constraints carried forward

1. **Do not publish retrieval-only numbers as §26 results.** They are the retrieval half: no model, no
   accuracy, one trial, one seed. The report says what it can claim.
2. **Keep the manifests committed** when a tier changes, and regenerate rather than edit a dataset.
3. **Preset ceilings are part of the dataset contract.** `standard` allows 64k input tokens, and Mode A now
   really carries the conversation, so the 80k / 120k rungs need `--preset research` (200k); a run that mixes
   them under `standard` is reported as a ceiling violation rather than silently truncated.
4. **The replay ceiling is a benchmark decision, not a product one.** Raising `ContextSettings.max_tokens`
   inside a replay is what makes Mode A a reference; it must not leak into the app's live default.
