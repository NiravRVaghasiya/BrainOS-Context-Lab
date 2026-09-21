# Phase 19 — MVP Definition

**Plan reference:** §25 (Phase 19 — MVP Definition), §21 (UI), §23 (automated pipeline), §19 (privacy)
**Status:** complete in this turn
**Tests:** 1248 → **1335** (+87) · coverage 94% → **94%** (floor 90) · `ruff check .` clean · shipped-surface credential scan: 102 files, 0 findings

## Goals

§25 states the MVP as fourteen things a user can do:

```text
1  Open the HF Space.           8  Retrieve the fact later.
2  Select OpenAI.               9  Inspect BrainOS memory.
3  Enter an API key.           10  Inspect retrieved context.
4  Select a model.             11  Compare BrainOS against full-context mode.
5  Start a conversation.       12  See token usage.
6  Store a fact.               13  Run a small benchmark.
7  Continue the conversation.  14  Export results.
```

Phases 4–18 built every one of those, and every one has tests. What no phase had
done was run them **as a sequence**, which is what the audit released as
`docs/status-audit-2026-09-21.md` found: a checklist of fourteen implemented
features is not a claim that a visitor can walk from "enter a key" to "export
results" without hitting something that only exists in isolation.

The audit also found three gaps with names attached, and this phase closes them
rather than leaving them as prose:

| Gap from the audit | Where it is closed |
| --- | --- |
| The checklist was never walked end to end | `tests/integration/test_mvp_walkthrough.py` — one session, items 2–14, in the plan's order |
| The §25 items were registered nowhere | `tests/unit/test_plan_traceability.py` — MVP rows, plan-text checks, item 1 recorded as external |
| Timeout and multi-visitor claims had no test that exercised them | `tests/unit/test_chat_timeout.py` (a provider that really blocks) and `tests/unit/test_sqlite_concurrency.py` (real threads, one database) |
| Budget was disclosed, retention was not | `src/app/retention.py` — age-based sweep for the session nobody ends, announced in the tab and recorded in each run's manifest |

## What was built

| File | Purpose |
| --- | --- |
| [`tests/integration/test_mvp_walkthrough.py`](../tests/integration/test_mvp_walkthrough.py) (new, 19 tests) | §25 walked once, in order, against the pinned runtime, with a deterministic provider double: one controller, one session, connect → chat → store a fact → four unrelated turns → retrieve it → inspect memory → inspect context → compare with `full_context` → usage → a 2-task retrieval-only pipeline run → export. Each step is a separate test asserting the artefact the step is supposed to produce (the prompt contains the fact from six turns earlier, BrainOS's final context is smaller than the full-context reference, the run's rerun command is portable, the export is key-free). |
| [`tests/unit/test_plan_traceability.py`](../tests/unit/test_plan_traceability.py) (+8 → 20 tests) | §25 becomes a second table (`MVP_ITEMS`): one ordered row per item, each mapped to the walkthrough test *and* the focused test that pins it, verified against the suite's own collection. The plan's §25 text is parsed and compared row by row, so a renumbered or reworded checklist fails loudly. Item 1 ("open the HF Space") is marked **external** with its reason and is the only row allowed to be. |
| [`tests/unit/test_chat_timeout.py`](../tests/unit/test_chat_timeout.py) (new, 10 tests) | The Phase 15 timeout, tested with a provider that blocks on a thread event instead of a fake that returns instantly: the turn returns in milliseconds, the status is redacted and names the timeout, the message is still in the transcript and still observed by the memory layer, the request is charged to the budget exactly once (`timeouts == 1`), the session is usable afterwards, and a *generous* timeout does not fire early on a merely slow call. |
| [`tests/unit/test_sqlite_concurrency.py`](../tests/unit/test_sqlite_concurrency.py) (new, 4 tests) | The Phase 14 claim ("WAL plus a busy timeout, so a multi-visitor Space serializes instead of failing") exercised with 8 threads sharing one database: concurrent transcript appends lose no row and cross no session boundary; the three stores write to one file at once; a `clear` racing writers takes only its own session's rows; and a delete-then-vacuum under load leaves no deleted bytes in the file or its WAL. |
| [`src/app/retention.py`](../src/app/retention.py) (new) · [`tests/unit/test_retention.py`](../tests/unit/test_retention.py) (new, 26 tests) | Retention for the artifacts of a session nobody ends. Age is the **newest** file in the session directory, so a live session is never swept and a burst of runs cannot evict a long study; the sweep is capped (`max_sessions_per_sweep`), oldest first; only direct children of `<root>/ui` that are single safe path segments, are not symlinks, and still resolve inside that directory; every candidate is reported, dry runs change nothing, and `python -m app.retention` runs it from cron with `--dry-run`, `--output`, `--format json`. |
| [`tests/unit/test_checkout_paths.py`](../tests/unit/test_checkout_paths.py) (new, 13 tests) | Audit bug #8: the Evaluation tab resolved the dataset and the whole `results/ui/` tree against `os.getcwd()`. These tests pin the fix from both ends — outside the checkout the runner anchors to the checkout and still records the plan's *relative* dataset path in artifacts; inside it, nothing changed. |
| [`src/checkout.py`](../src/checkout.py) (new) | The one place that answers "where is this checkout, and does this path belong to it?": `REPO_ROOT`, `in_checkout()`, `repo_anchored(relative, must_exist=…)`. Dependency-free, because `storage`, `app`, and `evaluation` all import it. |
| [`src/app/evaluation.py`](../src/app/evaluation.py) | The runner resolves its two roots and the default dataset at construction; `EvaluationPolicy` gains `results_retention_seconds` / `max_sessions_per_sweep` (validated, `None` disables) and a `retention_notice()` that `policy_notice()` appends; `run()` sweeps before planning and records the policy's window plus what the sweep did in the run manifest. |
| [`src/evaluation/{run,experiment,pipeline,errors}.py`](../src/evaluation/) | The console scripts resolve their committed-dataset default at call time, so `brainos-context-evaluate` / `brainos-context-pipeline` work from any working directory; the pipeline's rerun-command builder treats the anchored default as *the* default, so an artifact still carries a portable relative path. |
| [`src/storage/sqlite.py`](../src/storage/sqlite.py) | The unset database default is anchored the same way: a process started outside the checkout keeps persisting to its own `data/` instead of creating one wherever it happened to start. Inside the checkout it is still the documented relative path. |
| [`tests/unit/test_evaluation_ui.py`](../tests/unit/test_evaluation_ui.py) (+3 tests) | The wiring, not just the sweeper: a run sweeps an expired session before it starts, leaves the live one alone, and writes the retention policy and result into its manifest. |

## The audit

### §25 item by item

| §25 | Walked by | Focused tests | State |
| --- | --- | --- | --- |
| 1 Open the HF Space | — (external) | `test_deployment_config.py` (app entry point, requirements), `tests/ui/test_ui.py::test_app_builds_and_registers_every_callback` | **Deployable, not deployed.** The repository checks the half it owns; publishing the Space is an operator action and the ledger records it as external. |
| 2 Select OpenAI | `test_item_2_through_4…` | `test_controller.py::test_connect_lists_models_and_clears_the_key_box`, `::test_connect_rejects_an_invalid_configuration` | Walked |
| 3 Enter an API key | `test_item_3_the_key_is_never_returned_to_the_browser` | `test_connect_keeps_the_key_in_server_memory_only`, `tests/ui/…::test_connect_callback_clears_the_key_box_and_names_no_secret` | Walked |
| 4 Select a model | `test_item_4_the_selected_model_is_the_one_the_turn_used` | `test_baseline_modes_live.py::test_live_the_controller_reports_the_mode_that_ran` | Walked |
| 5 Start a conversation | `test_item_5_a_conversation_starts_and_the_provider_is_called` | `test_chat_controller_live.py::test_live_panels_show_a_fact_recalled_many_turns_later` | Walked |
| 6 Store a fact | `test_item_6_the_fact_is_stored_in_brainos_memory` | `test_controller.py::test_chat_records_the_fact_in_the_memory_panel`, `test_memory_policy.py::test_extract_candidates_keeps_project_facts_and_skips_chatter` | Walked |
| 7 Continue the conversation | `test_item_7_the_conversation_continues_past_the_fact` | `test_baseline_modes.py::test_full_context_replays_the_whole_conversation` | Walked |
| 8 Retrieve the fact later | `test_item_8_the_fact_is_retrieved_later_into_the_prompt` | `test_brainos_keeps_the_fact_the_sliding_window_lost`, `test_baseline_modes_live.py::test_live_brainos_keeps_the_fact_for_the_same_cost_as_the_window` | Walked |
| 9 Inspect BrainOS memory | `test_item_9_the_memory_panel_shows_what_brainos_holds` | `test_panels.py::test_memory_rows_match_the_declared_columns`, `test_controller.py::test_chat_records_the_fact_in_the_memory_panel` | Walked |
| 10 Inspect retrieved context | `test_item_10_…_accounting`, `test_item_10_the_retrieved_context_is_attributed` | `test_retrieved_rows_show_only_what_reached_the_prompt`, `test_context_summary_reports_the_baseline_next_to_the_savings` | Walked |
| 11 Compare with full context | `test_item_11_brainos_and_full_context_are_compared_on_the_same_session`, `::test_item_11_the_mode_is_visible_in_the_session_payload` | `test_baseline_modes_live.py::test_live_full_context_replays_everything_and_is_the_reference` | Walked |
| 12 See token usage | `test_item_12_token_usage_is_visible_per_turn_and_per_session` | `test_turn_view_carries_usage_summary_and_report`, `test_turn_view_usage_never_carries_the_key` | Walked |
| 13 Run a small benchmark | `test_item_13_a_small_benchmark_runs_from_the_tab` (+3) | `tests/ui/…::test_run_callback_feeds_the_tables_the_figures_and_the_report`, `::test_preview_callback_shows_a_ceiling_and_writes_nothing` | Walked |
| 14 Export results | `test_item_14_the_session_exports_with_its_transcript_usage_and_runs`, `::test_item_14_the_export_is_key_free` | `tests/ui/…::test_export_callback_writes_a_credential_free_json_file`, `test_chat_controller_live.py::test_live_export_contains_the_conversation` | Walked |

### What the walkthrough actually showed

From the run that the new integration file performs on every suite run
(7 chat turns, one 2-task × 2-mode retrieval-only pipeline run):

| Observation | Value |
| --- | --- |
| Provider calls for 7 chat turns | 7 — one per turn, each carrying the system instructions and a delimited memory block |
| Fact stated in turn 1, asked for in turn 6 | Present in the prompt (`PostgreSQL 16`), and the retrieved-memory panel attributes it |
| Final context, BrainOS | 197 tokens vs 277 for the full-context reference (**reduction 0.289**) |
| History considered / selected | 10 / 2 messages — the reduction is real work, not a claim |
| Pipeline run | 2 tasks × 2 modes, all stages `ok` (plots skipped, and the status says so) |
| Artifact credential scan | clean; the rerun command carries the plan's relative dataset path |
| Export | transcript, usage (7 requests) and the run's history, with no `sk-` substring anywhere |

### Bug #8 — cwd-relative roots

The audit reproduced the failure by starting the controller outside the
checkout: `EvaluationRequestError: Dataset not found:
benchmarks/context_rot/dataset.jsonl`, and an output root that would have been
created wherever the process started. Reproduced and fixed in this phase:

```bash
$ cd /tmp/elsewhere-test
$ PYTHONPATH=/home/user/BrainOS-Context-Lab/src python -c "
    ... UIEvaluationRunner().preview(...) ..."
dataset: /home/user/BrainOS-Context-Lab/benchmarks/context_rot/dataset.jsonl
preview dataset (recorded): benchmarks/context_rot/dataset.jsonl
run ok: True | rerun: python -m evaluation.run --mode brainos --benchmark context_rot
                    --dataset benchmarks/context_rot/dataset.jsonl …
```

Note the two paths in that output: **anchored for I/O, relative in the
artifact**. `_portable()` prefers the working directory, then the checkout, and
only then the absolute path, so a run made outside the checkout still produces
the re-run command the plan documents.

## Decisions later phases must not undo

1. **The MVP walkthrough is one ordered session, not fourteen snapshots.** The
   fixture performs the steps in the plan's order and shares the artifacts; a
   test that re-created each step independently would pass with retrieval broken
   across turns, which is the only interesting case.
2. **Item 1 stays external.** No test may claim to open a Space. What can be
   tested is that the checkout is deployable, and that is what the row maps to.
3. **Retention is age-based** (newest file), **bounded** (per-sweep cap), and
   **announced** (tab notice + run manifest). Retention that depends on a count
   of sessions evicts a running study; retention that silently deletes is a
   support incident.
4. **A sweep never follows a symlink and never leaves `<root>/ui`.** The
   containment rules in `retention.py` mirror `discard()`'s. Do not "simplify"
   them into a single `rmtree` on a glob.
5. **`checkout.repo_anchored` is the only place that decides where "the
   repository" is.** Inside the checkout, paths stay relative — that is what
   keeps artifacts and rerun commands portable, and it is asserted in
   `tests/unit/test_checkout_paths.py`.
6. **A timeout is a recorded turn, not a crash and not a lost request.** The
   message is in the transcript, the provider saw exactly one call, and the
   budget counts it (`timeouts == 1`).
7. **The stores are the concurrency boundary.** WAL + `busy_timeout` +
   `secure_delete` are re-applied on every connection; the threaded tests fail
   if a future change drops one.
8. **§25 rows in the ledger are ordered and parsed from the plan.** Adding a
   fifteenth MVP item means adding a row, a walkthrough assertion, and nothing
   else.

## Bugs and gaps found while building it

1. **The timeout had never been triggered.** Phase 15's tests asserted the
   controller passes `request_timeout_seconds` down and that an instant fake does
   not time out. A provider that actually blocks reveals the parts that matter
   (redaction, accounting, session reuse) and they all held — but the phase's
   claim was untested, which is the same grade of gap the Phase 18 audit found in
   the log surface.
2. **`sweep()` did not exist when `run()` started calling it.** Wiring before
   the method is a reminder that "the code is written" and "the code has run" are
   different states; the runner test in `test_evaluation_ui.py` is what proves
   the sweep happens on the run path.
3. **The dataset default was resolved at import time** in the first draft of this
   phase, which passes a "start from elsewhere" test only by accident of import
   order. Moved to call time (`default_dataset_path()`) — the failing test that
   caught it is `tests/unit/test_checkout_paths.py::test_the_single_mode_cli_defaults_to_the_committed_dataset_from_anywhere`.
4. **The pipeline's rerun command would have carried a machine-specific path**
   once the CLI anchored its default. Fixed by treating the anchored default as
   the default in `_pipeline_rerun_command`.
5. **Two of the new checkout tests needed the pinned runtime and did not say
   so** — a pipeline run builds a BrainOS adapter for every mode, including the
   baselines, so a base install fails the experiment stage (exit 4) rather than
   skipping it. The `without-extras` CI job caught this on the first push; the
   two tests now carry `@pytest.mark.requires_runtime`, and the path resolution
   they are about is still exercised without the runtime by the preview and
   sweeper tests beside them (`1136 passed, 104 skipped, 0 failed` in a base
   install).
6. **The retention notice made `policy_notice()` longer than the tab's header
   budget.** The sentence is written to be the shortest honest version of the
   policy; if a deployment turns retention off, it says that instead.
7. **Every SQLite connection re-issued `PRAGMA journal_mode=WAL` — and that
   pragma can fail on a healthy database.** Found by chasing a `verify` job that
   failed once on GitHub and could not be reproduced locally: the threaded
   concurrency tests flaked in about 5% of solo runs, always with
   `sqlite3.OperationalError('database is locked')` raised inside `_connect`. An
   8-thread repro (`conversation.append` + 25 memory writes + `save_run`, fresh
   file per iteration) failed at iteration 38 and pointed at the line:
   `_connect` ran the WAL switch *before* `PRAGMA busy_timeout=30000`, and SQLite
   does not route a journal-mode change through the busy handler — two Gradio
   workers opening a fresh Store-2 database at the same moment could therefore
   raise before doing any work. Journal mode is a property of the *file*, so
   `_ensure_wal` now reads the mode (lock-free in WAL) and only switches when the
   file is not already WAL, retrying a lost race five times and then proceeding:
   a caller's write must not fail over a mode switch a later connection will
   complete. The old code was in every store since Phase 14; the tests that found
   it are this phase's own.

## Measured behaviour

```bash
# The new files, on their own
$ .venv/bin/python -m pytest tests/integration/test_mvp_walkthrough.py \
      tests/unit/test_retention.py tests/unit/test_checkout_paths.py \
      tests/unit/test_chat_timeout.py tests/unit/test_sqlite_concurrency.py \
      tests/unit/test_plan_traceability.py -q
# 96 passed

# The concurrency bug, before and after the fix
$ for i in $(seq 1 40); do .venv/bin/python -m pytest -q \
      tests/unit/test_sqlite_concurrency.py; done
# before (4 tests): 2 failed runs in 40, both sqlite3.OperationalError: database is locked
# after  (8 tests): 0 failed runs in 60

# The regression the fix added, run against the old code
$ for i in $(seq 1 20); do .venv/bin/python -m pytest -q \
      tests/unit/test_sqlite_concurrency.py::test_concurrent_first_opens_do_not_race_the_journal_mode_switch; done
# old code: 1 failed run in 20 (database is locked, raised by the WAL preamble)
# new code: 0 failed runs in 25 + this file's thread tests

# The 8-thread repro (8 appends + 25 memory writes + save_run, fresh file each round)
$ .venv/bin/python /tmp/repro_locked.py
# before: iteration 38 -> PRAGMA journal_mode=WAL -> database is locked
# after:  no failures in 60 iterations

# The retention tool, by hand — dry run, then the real sweep
$ .venv/bin/python -m app.retention --root /tmp/retention-demo --dry-run
retention_version=retention-v1 root=/tmp/retention-demo sessions=1 kept=1 would remove=1 files=1 bytes=209 dry_run=True
max_age_seconds=604800 max_sessions_per_sweep=200
$ .venv/bin/python -m app.retention --root /tmp/retention-demo
retention_version=retention-v1 root=/tmp/retention-demo sessions=1 kept=1 removed=1 files=1 bytes=209 dry_run=False
$ ls /tmp/retention-demo/ui
bbbbbbbb-1111-2222-3333-444444444444          # the 1-day-old session survived
$ .venv/bin/python -m app.retention --root /tmp/never-ran-here
nothing to sweep: the results/ui directory does not exist
Nothing to sweep: the results/ui directory does not exist.
exit=0                                          # an empty host is not an error

# The pipeline CLI from a directory that is not the checkout
$ cd /tmp/elsewhere-test
$ PYTHONPATH=/home/user/BrainOS-Context-Lab/src python -m evaluation.pipeline \
      --dry-run --limit 1 --no-plots --output-dir /tmp/elsewhere-test/out
  brainos          tasks=1 graded=0 accuracy=— recall=1.000 evidence=1.000 tokens=195.0 reduction=0.830
credential scan: files=10 findings=0 clean=True
exit_code=0

# The suite and the shipped-surface scan
$ .venv/bin/python -m pytest -q --cov --cov-report=term
# Required test coverage of 90.0% reached. Total coverage: 94.19%
# 1335 passed in 219.98s (0:03:39)
$ .venv/bin/ruff check .
# All checks passed!
$ .venv/bin/python -m security.scan src docs README.md CONTEXT.md benchmarks \
      app.py requirements.txt packages.txt pyproject.toml \
      BrainOS_Context_Lab_Implementation_Plan.md
scan_version=scan-v1 files=102 bytes=1787889 findings=0 clean=True
```

New modules: `src/checkout.py` 100%, `src/app/retention.py` 95%,
`src/app/evaluation.py` 94% (all above the 90% floor).

Both CI jobs pass on the pushed branch — the pre-fix run
[`35582619453`](https://github.com/NiravRVaghasiya/BrainOS-Context-Lab/actions/runs/35582619453)
and the run that carries the WAL fix
[`35585990332`](https://github.com/NiravRVaghasiya/BrainOS-Context-Lab/actions/runs/35585990332):

```text
35582619453  (9cb5338 → 50f5289)
lint, test, coverage floor                pass    6m17s
suite runs without the optional extras    pass      25s

35585990332  (d1a8f6a, finding 7)
lint, test, coverage floor                pass    5m57s
suite runs without the optional extras    pass      27s
```

The `verify` job failed once more on the commit that recorded the run above
([`35583466031`](https://github.com/NiravRVaghasiya/BrainOS-Context-Lab/actions/runs/35583466031):
`Test with the coverage floor`, 3m32s; `without-extras` passed in 32s). Its logs
were not retrievable — the log endpoint returned `results-receiver: unexpected
EOF` and the job annotations only carried "Process completed with exit code 1."
— so the failure was chased from the one reproducible signal instead: the local
suite at that commit passes in full (1335 tests, 94.19%), so the difference had
to be a flake. Forty solo runs of the new threaded test file produced two
failures, and both were the same real bug (finding 7), not a test artefact.
The tests for it were hardened alongside the fix: the delete-under-load test now
asserts the rows are gone *under* load and only then scans the file and its WAL
sidecar for deleted bytes after `vacuum()`, and the generous-timeout test moved
from a 5-second ceiling to 60 seconds with an explicit `elapsed < 30` bound.

The `without-extras` job failed on the first push and is worth recording: a
pipeline run builds a BrainOS adapter for *every* mode, including the baselines,
so in a base install the experiment stage fails (exit 4) rather than skipping —
two of this phase's checkout tests asserted a successful run without declaring
the dependency. They now carry `@pytest.mark.requires_runtime`, and the local
base-install run is `1136 passed, 104 skipped, 0 failed`.

## Constraints carried into Phase 20+

1. **The MVP is walkable but not published.** §25 item 1 is the only unmet item,
   and it is unmet because nobody has created the Space — not because the code
   cannot. Phase 20's release checklist should record the Space URL when it
   exists, and the ledger row should stay external until then.
2. **No number in this phase is a research result.** The walkthrough's reduction
   figure (0.289 over seven turns of a stub conversation) and the pipeline's
   numbers come from the smoke tier (7 tasks, one length, `variants=1`). §29's
   accuracy / temporal / abstention / length-curve / cross-model rows still
   require a real model run; faking them here would poison §26.
3. **Retention is a policy, not a promise.** The sweep runs when a run starts and
   when an operator runs the CLI. A Space with no traffic and no cron keeps its
   files; `docs/limitations.md` says so, and Phase 20 should state the window in
   the Space description rather than implying immediate deletion.
4. **The database default is anchored but still operator-controlled.**
   `BRAINOS_LAB_DB` wins over everything, `:memory:` still means "no file", and
   `tests/unit/test_deployment_config.py` pins the in-checkout relative default.
   A Phase 20 deployment that wants its data elsewhere sets the variable.
5. **The ledger is the contract for §25.** Adding a feature to the tab does not
   add an MVP item; changing the plan's checklist does, and both the walkthrough
   and the ledger must move together.
