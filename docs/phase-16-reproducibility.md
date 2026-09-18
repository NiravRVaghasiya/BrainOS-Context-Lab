# Phase 16 — Reproducibility

> Plan reference: §16 ("Reproducibility") and §22 ("Reproducibility"), and the
> Phase 15 log's carried-forward constraints 1–3, which named this as the next
> step after cost controls.

Phase 16 makes every evaluation result re-derivable and every chat export
re-inspectable: one reproducibility vocabulary, one manifest builder, and one
credential boundary the cost controls had left open.

## Goals

From the plan §16 / §22:

1. **Every evaluation run gets a run id.** Already true (Phase 6/9 mint a
   `uuid4` per run); this phase makes it travel with the full provenance.
2. **Record the §22 block.** `run_id`, `timestamp`, `brainos_version`,
   `application_version`, `provider`, `model`, `temperature`,
   `benchmark_version`, `mode`, `context_budget`.
3. **Save configuration, task ids, raw metrics, aggregate metrics, model
   identifier, BrainOS revision, benchmark revision.** The manifest's inventory.
4. **Never save API keys.** The manifest records `api_key_env` *names* only
   where a caller supplies them; the run path records none at all.
5. **Honest unknowns.** A missing runtime is `unvalidated`, a source checkout
   is `fallback:0.1.0` — never an invented number.

Constraints from the Phase 15 log:

- **Record the chat limits in the export** (constraint 1).
- **No credential may leave the process**, now including the one route that was
  still technically open: a provider that echoes a pasted key *inside a model
  id*, which `list_models()` would have rendered back into the model dropdown.

## What was built

### New package: `src/reproducibility/`

| Module | Purpose |
| --- | --- |
| `run_provenance.py` | The single source of the §22 vocabulary. `app_version()` (installed-metadata read with a `fallback:` origin tag), `brainos_version()` (`unvalidated` / `pinned:` / live read plus the `pin:` commit), `build_manifest()` (the full manifest: versions, environment block, config, task ids, dataset path + digest, aggregate metrics, `--rerun` string), `rerun_command()` (credential-free copy-pasteable command), `global_overrides()` (the version fields a consumer can merge into any artifact), `persist_run()` (best-effort write through the secret-stripping evaluation store protocol). |

The package imports only the standard library; `persist_run` imports the
storage layer lazily, so the evaluation CLI and the chat controller can both
use it without pulling in the UI or the BrainOS runtime.

### Single-mode run CLI — `evaluation/run.py`

Every `python -m evaluation.run` artifact now carries:

* a top-level **`repro`** block — the §22 manifest built by `build_manifest`,
  including the dataset digest already recorded in `config`;
* **`config.application_version`**, **`config.brainos_version`**,
  **`config.benchmark_version`**, and **`config.output`** — the fallback
  fields, so a consumer written before Phase 16 keeps reading the version
  vocabulary at its existing path.

### Controlled experiments — `evaluation/experiment.py`

* `ExperimentRun.to_dict()` gains a top-level **`repro`** block whose
  application/benchmark/BrainOS fields are built with the *same* functions the
  run manifest uses, so the two can never drift.
* `provenance.application_version` / `provenance.brainos_version` now call
  `app_version()` / `brainos_version()` directly (was: the local
  `APPLICATION_VERSION` constant and a local `runtime_version()` that only
  survived as an internal helper).
* `ModeResult` config carries the same version pair.

### Error analysis — `evaluation/errors.py`

The Phase 12 report's `provenance` block gains `application_version` and
`brainos_version`, so a report aggregating several artifacts also records the
code revision that classified its failures.

### Chat export — `app/controller.py`

The session export gains three reproducibility fields:

* **`limits`** — the session's `ChatLimits.to_dict()` (the Phase 15 constraint).
* **`versions`** — `application`, `benchmark`, `brainos`.
* **`chat_history_sha256`** — a digest of the transcript in message order, so
  the exported conversation can be tied back to a result and re-verified.

### Credential boundary — `app/controller.py`

`connect()` now drops any listed model id that contains the active session key
before the list reaches the browser. A provider that echoes a credential inside
a model identifier can no longer use `list_models()` to leak it through the
dropdown.

## Decisions later phases must not undo

1. **One vocabulary, three surfaces.** `reproducibility.run_provenance` is the
   only place version strings are produced. Phase 17 (UI evaluation pipeline)
   and any future runner must call `build_manifest` / `app_version` /
   `brainos_version` rather than defining its own `APPLICATION_VERSION`
   constant.
2. **`benchmark_version` is `context-rot-v1`** (hyphen), matching the dataset
   generator's `spec.GENERATOR_VERSION`. The experiment's underscore spelling
   was a latent split-vocabulary bug that the tests now pin as fixed.
3. **Versions carry their origin.** `fallback:` / `installed:` / `pinned:` /
   `unvalidated` are data about how the read was made. A later phase must not
   strip the tags or a live read will be indistinguishable from a constant.
4. **The `repro` block is an addition, not a replacement.** `config` keeps the
   fallback version fields so older consumers stay intact; the top-level
   manifest is authoritative.
5. **Manifest never fails a run.** `build_manifest` takes `run_id` and returns
   a dictionary; it never raises, so provenance recording cannot be the thing
   that loses an experiment.
6. **Model-id guard is exact-match on the session key.** It is the same
   redaction the rest of the UI uses; it is not a pattern scrub of arbitrary
   "key-looking" model ids (a provider may legitimately name a model strangely,
   and the sidebar must not silently drop it).

## Bugs and gaps found while building it

1. **The benchmark revision was spelled two ways.** The dataset generator and
   manifest pinned `context-rot-v1`; `evaluation.runner` and
   `evaluation.experiment` recorded `context_rot-v1`. Two artifacts from the
   same run could not share a revision string — exactly the reproducibility
   failure this phase exists to catch. Unified on `context-rot-v1` and pinned
   by `test_every_producer_shares_one_benchmark_revision`.
2. **The experiment had two BrainOS-version readers.** `runtime_version()` and
   the manifest would have read differently once one was updated; both now
   route through `brainos_version()`.
3. **`output` was never recorded.** A run file had no reminder of where it was
   meant to live; `config.output` now records the path, and the manifest's
   `rerun_command` reconstructs from it.

## Measured behaviour

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.run --mode brainos --limit 1 \
  --output /tmp/repro-demo.json
# → mode=brainos tasks=1 graded=0 recall=1.000 … reduction=0.830
# artifact repro block (excerpt):
#   brainos_version: 2.0.0-alpha.1 (pin:1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc)
#   application_version: fallback:0.1.0
#   benchmark_version: context-rot-v1
#   task_ids: ["cr-single-hop-800-00"]
#   dataset.sha256: 4c4a4d37… (matches the committed manifest)

PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick \
  --modes brainos,full_context --dry-run --limit 1 --output /tmp/repro-exp.json --quiet
# → repro.application_version == provenance.application_version (True)
# → repro.brainos_version  == provenance.brainos_version  (True)

# chat export (fakes):
#   limits.max_input_tokens = 32000, limits.request_timeout_seconds = 120.0
#   versions = {application, benchmark: context-rot-v1, brainos}
#   chat_history_sha256 = 10a22845…
#   credential in export: False
```

The full suite — 1070 tests (1048 → +22) — passes, `ruff check .` is clean,
and the shipped-surface scan reports 87 files with 0 findings.

## Validation

```bash
.venv/bin/pytest -q                       # 1070 passed
.venv/bin/ruff check .                    # All checks passed!
PYTHONPATH=src .venv/bin/python -m security.scan <all shipped>  # 87 files, 0 findings
```

## Constraints carried into Phase 17+

1. **Phase 17 (evaluation pipeline in the UI) should surface the manifest.**
   When the Evaluation tab runs a benchmark, the `repro` block is the shape it
   renders to the user ("how to re-run this") and the shape it persists through
   `persist_run`.
2. **Sugar the presets with the manifest.** The Evaluation tab already lists
   Quick/Standard/Research; the manifest's `limits` block is what a visible
   "expected cost" widget should read.
3. **Phase 18 (test suite completion) keeps the manifest tests green.** The
   `SlowProvider` timeout test and the multi-threaded stress test are still to
   come; neither may weaken the credential-free artifact pins added here.
4. **`persist_run` is the UI runner's storage seam.** The CLI deliberately does
   not persist by default (the artifact is the record); the UI runner should.
