# Phase 17 — Automated Evaluation Pipeline

**Plan reference:** §23 (Phase 17 — Automated Evaluation Pipeline)
**Status:** complete in this turn
**Tests:** 1070 → **1179** (+109) · `ruff check .` clean · shipped surfaces scan clean (91 files, 0 findings)

## Goals

§23 asks for one thing and then lists it three ways: an automated pipeline that
takes a dataset and a model configuration and produces

```text
results/
  raw/          # one run file per mode, in the Phase 6/7 schema
  aggregated/   # comparison, statistics, error analysis
  plots/        # the six Phase 8/10 figures
  report/       # the human-readable report + the run manifest
```

The building blocks all existed by Phase 16 — `evaluation.run` (one mode),
`evaluation.experiment` (a controlled comparison), `evaluation.compare`,
`evaluation.errors`, `evaluation.plots`, `evaluation.analysis`,
`reproducibility.run_provenance` — but each was a separate command a researcher
had to run in the right order, with the right flags, into the right directory.
Nothing composed them, nothing checked that the composition was complete, and
the browser had an **Evaluation** tab that was a paragraph of prose about
commands the visitor could not run.

Phase 17 adds the composition (`evaluation.pipeline`) and puts it in the tab
(`app.evaluation`), carrying the constraints the earlier phases set for it:

* Phase 15 — the run's cost ceiling is visible **before** the run can spend;
* Phase 16 — every artifact carries `build_manifest`'s `repro` block, and the UI
  persists through `persist_run`;
* Phase 13 — no artifact contains a credential;
* Phase 12 — failures are classified with the existing taxonomy, and the
  comparison is of *distributions*, not of accuracies;
* the repository's honesty rule — an unset metric is `—`, never `0.000`.

## What was built

### New modules

| File | Purpose |
| --- | --- |
| [`src/evaluation/pipeline.py`](../src/evaluation/pipeline.py) (new, 1725 lines) | The seven-stage pipeline, its layout, its exit codes, its two-pass credential scan with quarantine, and the CLI (`python -m evaluation.pipeline`, console script `brainos-context-pipeline`). |
| [`src/app/evaluation.py`](../src/app/evaluation.py) (new, 1196 lines) | `UIEvaluationRunner` — the Gradio-free runner the tab drives: preset catalogue, cost preview, session-scoped output paths, dataset allowlist, generation with the session key, history, and `persist_run`. Plus `EvaluationPolicy`, the operator's knob for what a public Space will host. |
| [`docs/phase-17-automated-pipeline.md`](phase-17-automated-pipeline.md) (this file) | Phase log, decisions, measured behaviour, constraints carried forward. |

### Changed modules

| File | Change |
| --- | --- |
| [`src/evaluation/reports.py`](../src/evaluation/reports.py) | `markdown_report(artifact)` and twelve section renderers: scope, identity, headline, statistics, failures, robustness, cost, controls, security, stages, reproducibility. Answer-side honesty (`ANSWER_SIDE_HEADLINE_KEYS`, `_graded_by_mode`), task-ceiling `_remaining`, `_unique_list`, the partial-pipeline disclosure, and `write_json_report(..., default=str)`. |
| [`src/evaluation/analysis.py`](../src/evaluation/analysis.py) | `compare_error_distributions(report, baseline_mode)` — per-mode label shares plus total variation against the baseline, so the report compares *failure mixes* rather than ranking accuracies. |
| [`src/app/panels.py`](../src/app/panels.py) | Four column schemas (`PRESET_COLUMNS`, `EVALUATION_HEADLINE_COLUMNS`, `EVALUATION_STAGE_COLUMNS`, `EVALUATION_HISTORY_COLUMNS`) and six renderers, including one cost table that serves both the preview and the receipt. |
| [`src/app/controller.py`](../src/app/controller.py) | `EvaluationView` plus `evaluation_presets()`, `evaluation_preview()`, `run_evaluation()`, `evaluation_history()`; every value rendered by `panels` and redacted again against the session key. `end_session` now discards the session's artifacts; the export carries `evaluation_runs`. |
| [`src/app/ui.py`](../src/app/ui.py) | The full-width **Evaluation** tab (`EvaluationComponents`, `_build_evaluation`, three callbacks, `_evaluation_values`) and a rewritten `EVALUATION_MARKDOWN` that states the billing and unset-metric rules up front. |
| [`app.py`](../app.py) | The UI import moved under `if __name__ == "__main__"` — see bug 1. |
| [`pyproject.toml`](../pyproject.toml) | `brainos-context-pipeline = "evaluation.pipeline:main"`. |
| [`.gitignore`](../.gitignore) | `results/ui/*` (session-scoped runs from the tab). |

### The seven stages

```text
experiment → raw → comparison → statistics → errors → plots → report
```

`STAGE_DEPENDENCIES` records what each stage is made of, and `expand_stages`
adds the dependencies a selected stage needs: `--stages report` runs
`experiment` and `report`, because a report about nothing is worse than no
report. A stage whose dependency produced nothing is **skipped with a reason**,
not run against an empty input.

`run_pipeline` never raises for a failing stage. It returns a `PipelineResult`
whose `stages` say what happened and whose `exit_code` says whether the result
can be trusted:

| Code | Meaning |
| --- | --- |
| `0` | Complete and controlled. |
| `2` | Controlled-comparison violations — the numbers are not a comparison (the experiment CLI's own code). |
| `3` | The run stopped at a cost ceiling, so it describes part of the dataset. |
| `4` | The pipeline is incomplete (a stage failed) **or** a credential reached an artifact. |

### The tab

The Evaluation tab is the same pipeline, driven from a browser:

```text
preset · modes · task limit · dataset · baseline · token counter
[Preview cost]  [Run benchmark]  [History]
```

* **Preview** renders `RunBudget.estimate_ceiling` — planned requests, the
  output-token ceiling, every hard cap, and the honest
  `unknown before the prompts are built` for input tokens — and writes nothing.
* **Run** executes the pipeline into `results/ui/<session>/<run id>`, renders the
  headline matrix, the stage table, the six figures, the full report, the
  artifact/reproducibility panels, and persists a summary through `persist_run`.
* **History** lists this session's runs (summaries, never artifacts).
* **End session** deletes the persisted rows *and* the artifacts.

Generation is opt-in and uses the sidebar key: it is passed to
`build_provider(spec, api_key=…)` as a value, never exported to the environment,
and the artifact records the route (`the active browser session (openai
bring-your-own-key)`), not the value.

`EvaluationPolicy` is the operator's narrowing knob — allowed presets, whether
generation is offered at all, hard task/request ceilings, runs per session.
`apply()` only ever tightens a plan, so a preset's recorded limits are always the
limits the run obeyed.

## Decisions later phases must not undo

1. **The tab and the CLI share one entry point.** `app.evaluation` calls
   `evaluation.pipeline.run_pipeline`; it does not re-implement staging,
   artifacts, or the manifest. A browser run and a terminal run must stay
   byte-comparable in shape.
2. **One identifier per run.** `run_pipeline(run_id=…)` exists so the output
   directory, the manifest's `run_id`, the history row, and the re-run command
   are the same string. Do not let a caller generate a second id.
3. **`pipeline_rerun_command` never fails.** A plan that is not named after a
   preset gets its budget spelled out flag by flag (`_limit_flags`). Recording
   how to re-run a finished pipeline must never be the reason it fails.
4. **Artifact writing is serialization-tolerant.** `write_json_report` passes
   `default=str`. A `Path`, a datetime, or an enum in a payload must not lose a
   run that already cost money.
5. **Detection is not remediation.** When the credential scan finds something,
   the pipeline deletes the implicated files *inside its own output directory*,
   scrubs known secrets from the in-memory artifact, suppresses the report text
   if `report.md` was itself implicated, leaves a credential-free
   `QUARANTINED.txt`, appends a `## Quarantine` section, marks
   `artifact["quarantined"]`, and exits `4`. A later phase may add redaction in
   place; it may not go back to reporting and keeping.
6. **Unset is not zero, in the report too.** A mode that graded no answer shows
   `—` for accuracy, faithfulness, and QAE. `_graded_by_mode` is the single place
   that decides, and `ANSWER_SIDE_*` names the columns it governs.
7. **A partial run says it is partial.** Fewer than seven stages renders
   `Partial pipeline — Only these stages ran: …` in the report's scope section.
   The stages table already discloses which ran; the scope line is what a reader
   of the headline sees first.
8. **Session-scoped artifacts are session data.** `end_session` deletes them
   (`UIEvaluationRunner.discard`). A tab that promises deletion must delete the
   files, not only the rows.
9. **Panels render, the runner computes.** `EvaluationRunView` carries data
   (`headline`, `stages`, `cost`); `app.panels` owns every cell. Adding a second
   row-builder is how a table and its renderer drift.
10. **The root `app.py` shim must stay import-light.** `src/evaluation` imports
    `app.service` (a benchmark must run the application's real context builder),
    so the shim importing `app.ui` at module scope makes the benchmark CLI load
    Gradio and closes a cycle the moment the controller imports anything from
    `evaluation`.

## Bugs and gaps found while building it

1. **An import cycle through the repository's own entry point.**
   `evaluation.experiment` imports `app.service`; in a source checkout `app`
   resolves to the root `app.py` shim, which imported `app.ui` at module scope →
   `app.controller` → (new) `app.evaluation` → `evaluation.experiment`, partially
   initialized: `ImportError`. It also meant `python -m evaluation.pipeline`
   loaded Gradio, an *optional* dependency. Fixed by importing the UI only under
   `if __name__ == "__main__"`; `python -c "import evaluation.pipeline"` now
   leaves `gradio` out of `sys.modules`.
2. **A finished run could be lost by its own re-run command.**
   `pipeline_rerun_command` called `preset(plan.name)` and raised
   `ValueError: Unknown limit preset 'pipeline'` for any plan not built by
   `from_preset` — i.e. every hand-built plan, every test, and any embedding
   application. The exception fired while assembling the manifest, after all the
   work was done.
3. **`json.dumps` without `default=str`.** An `ExperimentPlan(dataset=Path(...))`
   crashed `write_json_report` with `Object of type PosixPath is not JSON
   serializable` — the same failure mode as bug 2: the run is complete, the
   artifact is not written.
4. **The report rendered `0.000` for ungraded modes.** A retrieval-only run
   grades no answers, so accuracy and faithfulness are unset; the first draft of
   `markdown_report` printed `0.000`, which reads as "every answer was wrong".
5. **A divergence finding that said nothing.** `_largest_divergence` reported
   `rag vs full_context: TV 0.000 — largest gap on conflicting_memory (0.0% →
   0.0%)`. A zero gap is not a finding; it now returns `{}`.
6. **The credential scan detected leaks and left them on disk.** A provider that
   echoes a key into an answer is enough: failure examples quote answers, so the
   key landed in `raw/*.json`, `aggregated/errors.json`, and the manifest. The
   scan reported 30 findings and exit `4` — and the files stayed. Quarantine
   (§ decision 5) now removes them; `security.scan` over the whole run directory
   reports clean afterwards.
7. **`python -m evaluation.pipeline` exited 0 and printed nothing.** The module
   had no `if __name__ == "__main__"` guard.
8. **The report implied completeness.** `--stages report` renders a report with
   no aggregates behind it; nothing said so. The scope line now names the stages
   that ran.
9. **The output directory and the manifest disagreed about the run id.** The
   runner generated a UUID for the directory and the pipeline generated another
   for the manifest, so a history row's id did not name its own artifacts.
10. **`End session` deleted rows and kept files.** The status line promised the
    session's data was dropped while `results/ui/<session>/` stayed on the host —
    on a Space, forever.
11. **Per-request and run-level ceilings shared a column.** The first cost table
    put `max_input_tokens` (per request, the one that causes skips) and
    `max_total_tokens` (per run) in the same "ceiling" column, so the output
    ceiling of 1536 appeared to be capped at 512. The per-request ceiling is now
    labelled `16000 per request`.
12. **A complete run reported itself as a partial pipeline.** The report renders
    from an artifact that holds stages 1–6 — stage 7's own row is added when the
    manifest is written, because a report cannot time its own writing — and the
    scope line inferred completeness from `len(stages) < 7`. Every full run
    therefore printed "**Partial pipeline.** Only these stages ran: experiment,
    raw, comparison, statistics, errors, plots", which is the one warning a
    reader must be able to trust. The artifact now records the *selection*
    (`stages_selected`, `partial`) and the report reads that, falling back to the
    completed rows (plus the report itself, which is obviously rendering) for
    hand-built artifacts.
13. **One run described its own dataset two ways.** `build_manifest` hashes the
    dataset path when the plan left `dataset_sha256` empty, so `pipeline.json`'s
    `repro.dataset.sha256` carried the digest while `plan.dataset_sha256`,
    `raw/*.json`, and `aggregated/experiment.json` all carried `""` — the same
    dataset, named and unnamed in one run. The UI runner hit this on every run
    because it builds a plan from form values without a digest. `run_pipeline`
    now hashes once, before stage 1, so every artifact in a run names the same
    bytes.
14. **The report named a baseline the run had not chosen.** A mode that produced
    no failure records cannot anchor a failure distribution, so Phase 12 falls
    back to another mode and records why in `baseline_selection` — but the report
    printed only the effective name, and every total-variation distance in the
    table is relative to that choice. The missing mode is usually `full_context`,
    i.e. the fallback happens exactly when the reference mode was clean. The
    report now says which mode was requested, that it reported nothing, and why
    the distance is against another one.
15. **A credential-freedom test read only part of the database.** Phase 14 put
    SQLite in WAL mode, so committed rows can still be in `<db>-wal` while the
    main file is one page long.
    `test_raw_database_bytes_never_contain_the_session_key` read `db.read_bytes()`
    and asserted *both* that the key was absent and that `[redacted]` was
    present — so it depended on when the last connection happened to close (that
    close auto-checkpoints): green alone, red after unrelated tests, with the
    redaction itself correct in both cases. It now checkpoints and reads the main
    file plus its sidecars. The same blind spot was in the scanner: the documented
    `python -m security.scan data/brainos_lab.sqlite3` scanned one file and could
    report clean while the newest rows sat in the WAL beside it (one Phase 13 live
    test had already worked around this by scanning the parent directory instead).
    `iter_files` now expands a named database's `-wal`, `-shm`, and `-journal`
    sidecars, so naming a database scans the database.

## Measured behaviour

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.pipeline --dry-run --limit 2 \
  --output-dir /tmp/p17
# pipeline=pipeline-v1 preset=quick run=3e6611ef-… tasks=2/7 modes=3 trials=1
# stage       status  seconds  detail
# experiment  ok      0.95     2/7 task(s) × 3 mode(s) × 1 trial(s); dry run; 0 violations
# raw         ok      0.00     3 run file(s) in the Phase 6/7 schema, each with its own repro manifest
# comparison  ok      0.00     3 mode row(s), 6 plot series
# statistics  ok      0.00     3 mode(s), 1 trial(s), 12 paired comparison(s); intervals degenerate
# errors      ok      0.00     1 failure record(s) (0 observed, 1 latent); unexpected=0
# plots       ok      1.04     6 figure(s)
# report      ok      0.04     15347 characters rendered from the pipeline artifact
#   full_context  tasks=2 graded=0 accuracy=— recall=0.000 evidence=1.000 tokens=1226.5 reduction=0.000
#   rag           tasks=2 graded=0 accuracy=— recall=1.000 evidence=1.000 tokens=281.0  reduction=0.771
#   brainos       tasks=2 graded=0 accuracy=— recall=1.000 evidence=0.500 tokens=191.0  reduction=0.843
# No answers were graded: accuracy, faithfulness, and QAE are unset, not zero.
# credential scan: files=16 findings=0 clean=True
# exit_code=0

PYTHONPATH=src .venv/bin/python -m security.scan /tmp/p17
# scan_version=scan-v1 files=16 bytes=478105 findings=0 clean=True
```

16 files, in the plan's four directories: 3 raw run files + `error-records.jsonl`,
4 aggregates, 6 figures, `report.md` + `pipeline.json`.

The same pipeline through the tab, with `FakeLLMProvider` and a session key
(2 tasks × 3 modes, generation on):

```text
**Pipeline finished** — 2 task(s) × 3 mode(s) × 1 trial(s), 6 request(s) and
3433 token(s) charged to your provider account.

- Artifact credential scan clean (10 file(s)).
Mode A — Full context   tasks=2 graded=2 accuracy=0.000 tokens=1226.5 reduction=0.0%
Mode C — Lexical RAG    tasks=2 graded=2 accuracy=0.000 tokens=281.0  reduction=77.1%
Mode D — BrainOS memory tasks=2 graded=2 accuracy=0.000 tokens=191.0  reduction=84.3%
```

`graded=2` is why accuracy is a number here and `—` in the dry run above; the
value is `0.000` because the fake provider's canned answer does not match the
dataset's expected facts, which is the correct result for that provider.
The session key appears in no artifact, no rendered field, and no environment
variable (`tests/security/test_pipeline_secrets.py`).

Quarantine, driven by a provider that echoes a key into its answer:

```text
credential scan: files=8 findings=12 clean=False
QUARANTINED: 5 artifact file(s) deleted:
  - errors.json  - experiment.json  - error-records.jsonl  - full_context.json  - pipeline.json
  rotate any key that reached a model response; see report/QUARANTINED.txt
exit_code=4
# security.scan over the run directory afterwards: clean=True
```

## Validation

```bash
.venv/bin/pytest -q          # 1179 passed (two consecutive full runs, both green)
.venv/bin/ruff check .       # All checks passed!
PYTHONPATH=src .venv/bin/python -m security.scan src docs README.md CONTEXT.md \
    benchmarks app.py requirements.txt packages.txt pyproject.toml \
    BrainOS_Context_Lab_Implementation_Plan.md   # 91 files, 0 findings
PYTHONPATH=src .venv/bin/python -m evaluation.pipeline --dry-run --limit 2 \
    --output-dir /tmp/p17    # exit 0, 16 files, scan clean
```

New tests: `tests/evaluation/test_pipeline.py` (31), `tests/unit/
test_evaluation_ui.py` (48), `tests/security/test_pipeline_secrets.py` (14),
`tests/integration/test_automated_pipeline_live.py` (7), `tests/ui/test_ui.py`
(+8, and the registered-callback count 16 → 19), plus one scanner test for the
sidecar rule. No provider key was used anywhere: the unit tests drive generation
through `RecordingProvider` and `FakeLLMProvider`, and the live file through
`PromptReadingProvider`, which answers only from the prompt it was given.

The live file is the phase's integration proof, in the shape the earlier phases
use: the committed benchmark, all five modes, the pinned `brainos_runtime`
(`pytest.importorskip`), one command. It asserts the properties rather than the
numbers — every stage ran and the layout is complete, every mode ran every task
with exactly one request per (task, mode), every artifact carries the same
dataset digest and a rerun command, the four aggregates describe the same run,
the report claims only what the run supports, the token/evidence ordering between
modes holds, and no credential reaches any file. It is where bugs 12–14 were
caught: a fake-driven unit test builds its own artifact and so never renders the
scope line a real seven-stage run renders.

## Constraints carried into Phase 18+

1. **Phase 18 (test suite) keeps these pins green.** In particular: the
   credential-freedom pins in `tests/security/test_pipeline_secrets.py`, the
   quarantine behaviour, the `—`-not-`0.000` rule, and the callback count in
   `tests/ui/test_ui.py` (19). A new tab callback changes that number on purpose
   or not at all.
2. **Phase 18's "evaluation tests use deterministic mock models"** now has a home:
   `run_pipeline(plan, tasks, ModelSpec(...), RecordingProvider(...))` is the
   one-call way to verify a benchmark calculation end to end.
3. **`EvaluationPolicy` is the deployment knob.** Phase 19/20 (MVP polish,
   research release) should tighten it for a public Space rather than editing the
   tab: `EvaluationPolicy(allowed_presets=("quick",), allow_generation=False,
   max_tasks=20, max_requests=60)` is a defensible public default.
4. **A hosted Space accumulates `results/ui/`.** Runs are deleted with their
   session, but an abandoned session is not ended by anyone. Phase 19 should add
   a retention sweep (or write to a temp root) before the tab is enabled on a
   public Space with generation on.
5. **Nothing here validates a model's quality.** Every number in this phase came
   from a dry run or a deterministic fake. The research claim still needs a real
   run at `--preset research` with a real key, which is Phase 20's job.
