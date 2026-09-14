# Phase 9 log — Controlled experiments

**Status:** complete (branch `arena/01a09f31-brainos-context-lab`)
**Plan section:** Phase 9 — Controlled Experiments (§15).
**Depends on:** Phase 6 (the five mode profiles), Phase 7 (the benchmark and the
scorer), Phase 8 (the metric suite), and Phase 15's cost controls — the plan
lists those controls as a prerequisite for putting a model in the loop, so the
part of Phase 15 that a run cannot exist without (ceilings, budget accounting,
skips, per-request timeouts) is implemented here and the *UI-side* controls stay
in Phase 15.

## Objective

From the plan:

> For every benchmark task, compare Full Context / Sliding Window / RAG /
> BrainOS / BrainOS + RAG. Keep constant: model, temperature, generation
> parameters, benchmark examples, task wording, evaluation procedure. Only
> change the context-management strategy.

Phase 7 measured what each mode *put in a prompt*; Phase 8 scored prompts against
scripted answers. Phase 9 closes the loop: a real prompt per (task, mode) goes to
a provider, the answer it returns is graded by the same Phase 7/8 procedure, and
the claim "only the strategy changed" is **checked and recorded** rather than
asserted in prose.

## Starting state

- `baselines/modes.py` already fixed each mode's history window and `evaluation/modes.py`
  already replayed a task through the real `ConversationService` in memory, with
  no provider and no disk.
- Answers had to be supplied out of band (`--answers`), so no metric that depends
  on generation (QAE, latency, real accuracy) meant anything.
- Phase 8's ledger carried four constraints into this phase: QAE is meaningless
  while answers are scripted; latency stays `null` until generation exists; answer
  grading needs cost controls before it can be exposed; use an exact token counter
  for research runs.
- There was no artifact that recorded *what was held constant* across modes.

## Work completed

### New / changed modules

| File | Purpose |
| --- | --- |
| `src/evaluation/generation.py` | The model-in-the-loop path. `GenerationSettings` (model, temperature, max_tokens, top_p, seed, timeout) with `fingerprint()` — a 16-hex SHA-256 over exactly those fields; `GenerationResult` (text, model, requested_model, latency_ms, usage, prompt token count, skipped/error, `.ok`); immutable `ModelSpec` with `with_limits` / `provider_config` / `to_dict`; `count_messages` (+4 tokens per message); `budgeted_generator(...)`, which wraps a provider so every request is counted, gated, timed, charged, and redacted before it reaches the score; `api_key_from_environment`; `MissingCredentialError`; `build_provider`. |
| `src/evaluation/limits.py` | The cost controls: `RunLimits` (tasks, requests, input/output/total tokens, generation failures, timeout), `RunBudget` (per-request prompt ceiling, request gate, reported usage preferred over the local estimate, skip recording, `snapshot`, `estimate_ceiling`), `BudgetExceeded(limit, allowed, used)`, and the `quick` / `standard` / `research` presets. |
| `src/evaluation/experiment.py` | `python -m evaluation.experiment`: `ExperimentPlan`, `run_controlled_experiment`, `check_constants`, `compare_modes_across_models`, the artifact writer, and the summary table. Exit codes: `0` clean, `2` constants violated, `3` aborted by a ceiling. |
| `src/evaluation/modes.py` | `task_evaluator(..., generate=...)`: after the prompt is built, the finished message list goes to the generator and the answer is graded in place. The replay path and the generated path share one prompt builder, so a mode cannot behave differently in an experiment than in a replay. |
| `src/evaluation/run.py` | `--generate` (single mode, model in the loop) and `--base-url`; the artifact gains `generation_settings` and `generation_usage` beside the unchanged Phase 6/7 schema. The summary prints `—`, not `0.000`, for accuracy/faithfulness/QAE when nothing was graded. |
| `tests/fakes.py` | `RecordingProvider` records messages *and* sampling kwargs and can report a different model than the one requested; `PromptReadingProvider` is a deterministic reader that answers only from the evidence in the prompt and says "I don't know." when there is none. |

### Tests added (75 in the Phase 9 files, 503 → 586 overall)

| File | Tests | What it pins |
| --- | --- | --- |
| `tests/unit/test_limits.py` | 14 | Preset shapes, request/token ceilings, charge-with-reported-usage, skip recording, `estimate_ceiling`, `BudgetExceeded` fields. |
| `tests/unit/test_generation.py` | 19 | Settings validation, fingerprint stability and sensitivity (including `max_tokens`), `ModelSpec` credential hygiene, `count_messages`, budgeted generation, failure/blank/skip handling. |
| `tests/evaluation/test_experiment.py` | 25 | The committed dataset through `RecordingProvider`: constants checking, ordering, trials, session isolation, aborts, artifact schema, run-file compatibility. |
| `tests/security/test_experiment_secrets.py` | 6 | No credential in the artifact or the summary; `api_key_env` validation; the CLI rejects `--api-key`. |
| `tests/evaluation/test_run_cli.py` | 5 | `--generate` wiring, ungraded runs printing `—` instead of `0.000`, fail-before-request behavior. |
| `tests/integration/test_controlled_experiment_live.py` | 4 | The real pinned runtime: one fingerprint over 35 requests, per-request latency/usage, mode evidence signatures, artifact hygiene, `compare_runs` compatibility. |
| `tests/integration/test_provider_http_live.py` | 2 | The real OpenAI SDK over a localhost stub: base URL, payload, usage normalization, credential in the header only. |

### Key design decisions

1. **Task-outer, mode-inner.** All five modes finish one task before any mode
   starts the next. Provider drift (rate limits, latency, a model silently
   updating mid-run) then lands on every mode instead of on whichever mode ran
   last, and the records come out paired by task — which is what Phase 10's paired
   comparisons consume.
2. **Fresh session per (task, mode, trial).** No mode inherits a runtime warmed by
   another mode, and each trial starts from nothing.
3. **One fingerprint, checked after the run.** Every record carries the settings
   fingerprint, the first system message, and the final user message. A
   comparison with more than one fingerprint, a differing system prompt, or a
   differing task wording is not controlled, and the run says so.
4. **The reported model is what is checked.** A gateway can silently serve a
   different model than the one requested; the check reads the provider's
   reported model, and a mismatch is a violation.
5. **Violations do not throw the run away.** They are recorded in
   `constants.violations`, `constants.passed` goes false, the CLI exits `2` and
   prints "do not report these numbers as a mode comparison" — a broken control
   set is worse than no comparison, because it still looks like a result.
6. **A budget ceiling produces a partial result, not a traceback.** Exhausting a
   limit sets `aborted` (exit `3`) and keeps the records collected so far. A
   provider failure is recorded as an errored generation and never raised into
   the run.
7. **A prompt that does not fit the input ceiling is a skip, and is recorded.**
   At the plan's longer lengths "Mode A cannot fit inside the budget" is a finding
   about the mode; silently truncating or crashing would destroy it.
8. **A dry run is still a measurement.** With no model configured the run builds
   every prompt and reports the retrieval side of the comparison — recall,
   evidence-in-prompt, and per-mode context tokens — without a key or a request.
   The generation columns render as `—`, and the artifact records
   `dry_run: true` rather than pretending the run was complete.
9. **The credential comes from the environment, never from the command line.**
   Both CLIs construct the parser with `allow_abbrev=False`, so `--api-key` can no
   longer abbreviate `--api-key-env` and a pasted secret can no longer be recorded
   as a variable name. `ModelSpec.api_key_env` must match
   `[A-Za-z_][A-Za-z0-9_]{0,127}`, and the error for a bad value never echoes it.
   The key is passed to the SDK, excluded from `repr`, and handed to the
   generator's redaction list; no artifact containing `sk-` or `"api_key":` is
   accepted by the test suite.
10. **The artifact stays compatible with the Phase 6/7 run schema.**
    `--runs-dir` writes one run file per (mode, trial) that
    `evaluation.compare` / `analysis.compare_runs` consume unchanged, so the
    existing reports and plot series keep working on a controlled experiment.

### The controlled-comparison contract

```text
one artifact
  plan                 experiment_version, application_version, benchmark_version,
                       modes, trials, session_isolation, dataset path + sha256
  model                provider, model, temperature, max_tokens, top_p, seed,
                       timeout, base_url, api_key_env, settings_fingerprint
  limits               the ceilings the run was allowed to spend
  constants            checked: modes, task_set, task_wording,
                       system_instructions, generation_parameters,
                       requested_model, token_counter
                       passed, violations[], generation_fingerprints[],
                       reported_models[], token_counters[], generated_requests
  budget               requests, input/output/total tokens, failures, skips,
                       remaining, within_limits
  modes[]              per (mode, trial): scored records, aggregate metrics,
                       dataset issues, and a Phase 6/7-compatible run dict
  cost_estimate        requests x ceiling, before a request is sent
  headline[]           the comparison table (accuracy, faithfulness, recall,
                       evidence-in-prompt, mean tokens, mean latency)
  warnings, dataset_issues, aborted
```

Rules that hold for every artifact:

- `constants.passed` is `false` ⇒ the run is not a comparison, whatever the
  headline says;
- `aborted` is set ⇒ every number covers only the part of the plan that ran;
- `graded_answer_count == 0` ⇒ accuracy and faithfulness are unset (`—` in the
  summary and in the ungraded rows), **not** zero;
- `dry_run: true` ⇒ no provider was called, so no answer metric exists;
- latency is per request and comes from the generation path only. A replay
  never fills it in.

### Cost controls (the part of Phase 15 this phase cannot run without)

| Preset | Modes | Trials | Requests | Input / output / total ceilings |
| --- | --- | --- | --- | --- |
| `quick` | full context, RAG, BrainOS | 1 | 60 | 16k prompt / 512 out / 250k |
| `standard` | all five | 1 | 500 | 64k prompt / 1024 out / 5M |
| `research` | all five | 3 | 7,500 | 200k prompt / 2048 out / 100M |

`--limit` caps tasks; `--max-input-tokens`, `--max-output-tokens`,
`--max-requests`, `--max-total-tokens`, `--max-failures`, and `--timeout`
override the preset. `RunBudget` prefers the provider's reported usage over the
local estimate when charging, which keeps the ledger honest for
`tiktoken`-counted runs as well as estimated ones.

## Bugs found and fixed while building it

1. **`check_constants` invented a system-prompt violation.** With no system
   instructions configured it compared against `DEFAULT_SYSTEM_INSTRUCTIONS`
   anyway and failed the run. The check now runs only when a prompt was actually
   configured, and it compares with a prefix rule because `_fit_system` may
   truncate the stored prompt with a `" …"` marker under budget pressure.
2. **Gateway routing was invisible.** The requested-model check read
   `requested_model` — the value the caller asked for — so a gateway that served
   another model passed. It now reads the provider-reported `model`.
3. **`--api-key` abbreviated `--api-key-env` in both CLIs.** `argparse`'s default
   abbreviation would have accepted `--api-key sk-…` as the *name* of the
   environment variable and exported the credential into the artifact. Both
   parsers now set `allow_abbrev=False`; a pinning test asserts the flag still
   has `dest="api_key_env"` and that the abbreviation is rejected.
4. **`api_key_env` accepted any string,** so a pasted credential was a legal
   variable name. It is now validated against `[A-Za-z_][A-Za-z0-9_]{0,127}` and
   the `ValueError` does not contain the value; both CLIs surface it as
   `SystemExit`.

Two test bugs of the same phase were fixed alongside them: a grading test that
asserted `accuracy == 1.0` across two committed tasks while its fake answered only
one, and a credential test that failed on legitimate `api_key_env` provenance
(it asserts value shapes — `sk-`, `"api_key":` — instead of the field name).

## Measured behaviour

### Dry run — retrieval side, no provider, committed smoke dataset (reproducible)

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick --modes all \
  --dry-run --output results/phase9-dry.json
```

```text
experiment=quick dry_run=True tasks=7/7 modes=5 trials=1 requests=0
mode            tasks  graded  accuracy  faithfulness  recall  evidence  mean_tokens  mean_ms
full_context    7      0       —         —             0.000   1.000     1152.7       —
sliding_window  7      0       —         —             0.000   0.000     189.4        —
rag             7      0       —         —             1.000   1.000     270.6        —
brainos         7      0       —         —             1.000   0.833     193.6        —
brainos_rag     7      0       —         —             1.000   1.000     367.1        —
No answers were graded: accuracy and faithfulness are unset, not zero.
```

This is the comparison's retrieval half on the 7-task smoke tier with the
estimated counter: full context spends 1152.7 tokens per prompt, BrainOS 193.6
(≈6× smaller), and the multi-hop task keeps evidence-in-prompt at 0.833 for
BrainOS. **It is not a result about answer quality** — nothing was generated.
Longer/claim-bearing runs need `--tier standard` and `--token-counter tiktoken`.

### Live properties (pinned BrainOS, `PromptReadingProvider`, no key)

`tests/integration/test_controlled_experiment_live.py` runs the committed dataset
through all five modes with a deterministic reader and asserts the phase's
properties rather than numbers:

- one fingerprint across 35 requests, temperature `0.0`, one system prompt;
- every (task, mode) recorded latency and provider usage;
- sliding window's evidence-in-prompt is `0.0`, BrainOS's is `> 0`, and full
  context's Recall@K is `0.0` with evidence `1.0` — the mode signatures;
- the artifact contains no `sk-` and no `"api_key":`, and its per-mode run dicts
  feed `evaluation.compare`.

`tests/integration/test_provider_http_live.py` additionally drives the real
OpenAI SDK over a throwaway `127.0.0.1` stub server: the credential travels as an
`Authorization` header (never in the request body), the base URL, model and
`max_tokens` reach the wire, usage is normalized, and a `--generate` CLI run
writes `generation_usage` without the key.

## Constraints carried into later phases

1. **A run is not a trial.** The `research` preset runs three trials per
   (task, mode) because stochastic sampling without repeats cannot support a
   claim; Phase 10 must summarize with `metrics.summarize` and compare paired
   records — never present one run's task-level mean as a trial CI.
2. **Latency is real now, and it is a provider metric.** Keep it per request;
   never backfill a replay with wall-clock.
3. **The smoke tier still cannot support a claim** (7 tasks, one length):
   `--tier standard`/`research` and `--token-counter tiktoken` for anything
   reported.
4. **Never report a comparison with `constants.passed == false` or a set
   `aborted`.** Both are exit-code-visible for that reason.
5. **Skips are a mode property.** Full context will hit the input ceiling at the
   plan's longer lengths; keep skips in the denominators and in the report.
6. **Phase 15 keeps these objects.** When the UI exposes a benchmark run it must
   reuse `RunPreset` / `RunLimits` / `RunBudget` rather than inventing limits.
7. **Phase 11's ablations slot in as modes**, not as a second runner: a new mode
   profile plus a `MODE_ORDER` entry is enough to appear in the controlled
   comparison.
8. Keep the offline paths in the suite: `PromptReadingProvider` (no key, real
   runtime) and the localhost HTTP stub (real SDK, no network).

## Validation

```bash
.venv/bin/python -m pytest -q
# 586 passed  (503 before this phase; +83)

.venv/bin/ruff check .
# All checks passed!

.venv/bin/python -m pytest \
  tests/unit/test_limits.py tests/unit/test_generation.py \
  tests/evaluation/test_experiment.py tests/evaluation/test_run_cli.py \
  tests/security/test_experiment_secrets.py \
  tests/integration/test_controlled_experiment_live.py \
  tests/integration/test_provider_http_live.py -q
# 75 passed

PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick --modes all \
  --dry-run --output results/phase9-dry.json
# table above; exit 0
```

No provider API key was used anywhere in this phase: generation is exercised
through `RecordingProvider` (unit), `PromptReadingProvider` on the pinned runtime
(live), and a localhost OpenAI-compatible stub (transport). Real-provider runs
need only an exported key — `openai` is installed as the `providers` extra.
