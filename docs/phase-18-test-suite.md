# Phase 18 — Test Suite

**Plan reference:** §24 (Phase 18 — Test Suite), §28 (test layout)
**Status:** complete in this turn
**Tests:** 1179 → **1248** (+69) · coverage 93% → **94%** (floor 90) · `ruff check .` clean · base install: 1073 passed, 98 skipped, 0 failed

## Goals

§24 asks for four categories of test:

```text
unit         provider adapters · API key redaction · context builder ·
             token budgeting · memory filtering · session isolation ·
             conflict handling
integration  user input → observe → recall → context → provider → response →
             memory update
evaluation   deterministic mock models that verify benchmark calculations
security     key never in logs · Session A cannot read Session B's memory ·
             retrieved memory cannot override system instructions ·
             user data is not returned in diagnostics
```

Phases 1–17 each wrote the tests their own phase required, so when the plan
reached §24 there were already 1179 tests. That changes the phase's job. It is
not "write a test suite"; it is:

1. **prove** each of §24's items is covered, by name, not by a coverage
   percentage — a line can be executed by a test that asserts nothing about the
   requirement;
2. **close the gaps that proof finds**;
3. make the suite's shape **part of the build**, so the proof cannot quietly
   rot.

The audit produced three real findings: the log surface was untested, the
provider adapter was the least-covered module in the project, and a base install
**failed 86 tests** rather than skipping them.

## What was built

| File | Purpose |
| --- | --- |
| [`tests/unit/test_plan_traceability.py`](../tests/unit/test_plan_traceability.py) (new, 12 tests) | The §24 ledger: every plan item mapped to named tests, each mapping checked against the suite's own collection, and the plan text itself checked so a renumbered §24 fails loudly instead of silently validating a stale list. |
| [`tests/security/test_log_hygiene.py`](../tests/security/test_log_hygiene.py) (new, 9 tests) | §24's "API key never appears in logs" and §13's first rule, pinned on every surface a log can reach: logging records, stdout, stderr, warnings. Static (no shipped module configures logging or emits a record; importing every package installs no sink) and dynamic (a credentialed session that fails a connect, a chat, and a storage write leaves no trace of the key). |
| [`tests/unit/test_provider_adapter_edges.py`](../tests/unit/test_provider_adapter_edges.py) (new, 21 tests → 34 collected) | §24's first unit item, at its edges: the optional SDK missing, client construction from the session configuration, a gateway that rejects the key, model lists with unusable entries, credential kwargs refused, message validation, missing/empty choices, content-part and usage normalization, and raw metadata that drops secret-named fields. |
| [`tests/evaluation/test_experiment_cli.py`](../tests/evaluation/test_experiment_cli.py) (new, 11 tests) | The controlled experiment's CLI: the summary a researcher reads, the exit codes that decide whether numbers may be reported (0 / 2 / 3, including the precedence rule), and credential handling on the command line. |
| [`tests/conftest.py`](../tests/conftest.py) (was a docstring) | The optional-dependency policy: `@pytest.mark.requires_runtime` / `@pytest.mark.requires_figures` become skips when the module is absent, so a base install reports skips instead of failures. |
| [`tests/unit/test_tokenizers.py`](../tests/unit/test_tokenizers.py) (+2) · [`tests/security/test_secrets.py`](../tests/security/test_secrets.py) (+1) | The tiktoken setup path (encoding by name, by model, unknown encoding) with the dependency stubbed; `SessionManager.clear_all` clearing every session's credential. |
| [`pyproject.toml`](../pyproject.toml) | `pytest-cov` in the `dev` extra; `[tool.coverage.run] source = ["src"]`; `[tool.coverage.report] fail_under = 90`; the two optional-dependency markers registered. |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) (new) | Two jobs: `verify` (ruff + the full suite with the coverage floor, `integration` extra installed) and `without-extras` (the base package only, where optional tests skip themselves). The shipped-surface credential scan is already a test, so it runs inside both. |

Nothing in `src/` changed. Phase 18 is the phase that measures the others; a
test-suite phase that needs a production edit to pass is a bug report, and none
was needed here.

## The audit

### §24 item by item

| §24 | Where it is verified | State |
| --- | --- | --- |
| unit · provider adapters | `tests/unit/test_providers.py`, `test_provider_adapter_edges.py`, `tests/integration/test_provider_http_live.py` | gap closed (79% → 100% on `providers/openai.py`) |
| unit · API key redaction | `tests/unit/test_providers.py`, `tests/security/test_secrets.py`, `test_log_hygiene.py`, `tests/unit/test_trace.py` | already covered; log route added |
| unit · context builder | `tests/unit/test_context_builder.py` (29 tests) | covered |
| unit · token budgeting | `test_context_builder.py`, `tests/unit/test_tokenizers.py` | covered; tiktoken path closed (85% → 99%) |
| unit · memory filtering | `tests/unit/test_memory_policy.py`, `test_retrieval_policy.py` | covered |
| unit · session isolation | `test_storage_sqlite.py`, `test_adapter.py`, `test_service.py`, `tests/integration/test_context_pipeline.py`, `test_brainos_runtime.py` | covered |
| unit · conflict handling | `test_retrieval_policy.py`, `test_context_pipeline.py` | covered |
| integration · the full chain | `test_service.py`, `test_controller.py`, `tests/integration/test_chat_controller_live.py`, `test_context_pipeline.py` | covered |
| evaluation · deterministic mock models | `test_scoring.py`, `test_metrics.py`, `test_experiment.py` over `RecordingProvider` / `PromptReadingProvider` / `FakeLLMProvider` | covered |
| security · key never in logs | `test_log_hygiene.py` + the artifact pins in `test_pipeline_secrets.py`, `test_artifact_scan.py` | **gap closed** |
| security · session isolation | `test_controller.py`, `tests/integration/test_security_live.py`, `tests/security/test_storage_secrets.py` | covered |
| security · memory cannot override instructions | `test_injection_corpus.py`, `test_history_guard.py`, `tests/integration/test_security_live.py` | covered |
| security · user data not in diagnostics | `test_controller.py`, `test_mode_secrets.py`, `test_error_records.py`, `test_storage_secrets.py` | covered |

### Coverage before and after

| Measurement | Before | After |
| --- | --- | --- |
| collected tests | 1179 | 1248 |
| statement coverage (`src`) | 93% (9068 stmts, 643 missed) | **94%** (9067 stmts, 543 missed) |
| `src/providers/openai.py` | 79% | **100%** |
| `src/evaluation/experiment.py` | 83% | 94% |
| `src/brain/tokenizers.py` | 85% | 99% |
| `src/app/session.py` | 81% | 100% |
| files skipped as fully covered | 11 | 13 |

The two files still below 90% are honest gaps rather than oversights:
`src/brain/adapter.py` (87%) is dominated by defensive branches that the pinned
runtime's own contract makes unreachable in this environment, and
`src/evaluation/analysis.py` (89%) carries report-formatting branches for run
shapes no committed fixture produces. `src/app/__init__.py` and
`src/storage/evaluations.py` report 0% because they contain no executable logic:
a re-export module and a `Protocol`.

### The base install failed 86 tests

The README claims "deterministic fakes cover the whole pipeline without BrainOS
installed; optional live tests exercise the pinned runtime". In a base install
that claim was false:

```bash
python -m venv /tmp/base && /tmp/base/bin/pip install "pytest>=8.0"
/tmp/base/bin/python -m pytest -q
# → 82 failed, 1071 passed, 18 skipped
```

With the UI and provider extras present but the pinned runtime absent, 86 tests
failed. They were concentrated in the files that run the application's *real*
composition — `evaluation.pipeline`, `app.evaluation`, the experiment CLI, the
reproducibility artifacts — where the default adapter factory builds a BrainOS
adapter:

| File | Failing tests |
| --- | --- |
| `tests/evaluation/test_experiment.py` | 16 |
| `tests/evaluation/test_pipeline.py` | 15 |
| `tests/unit/test_evaluation_ui.py` | 14 |
| `tests/security/test_pipeline_secrets.py` | 12 |
| `tests/evaluation/test_experiment_cli.py` | 8 |
| `tests/evaluation/test_reproducibility.py` | 4 |
| `tests/security/test_error_records.py` | 3 |
| `test_mode_secrets.py`, `test_experiment_secrets.py`, `test_run_cli.py`, `test_mode_strategies.py` | 2 each |
| `test_run_provenance.py`, `test_ui.py`, `test_log_hygiene.py`, `test_provider_http_live.py` | 1 each |

A red suite that is really a missing optional dependency teaches a contributor
to ignore failures, which is worse than the failure. Two of the 86 were better
fixed than skipped, and were: the controller's limits test only needed an
injected fake adapter, and the traceability ledger now falls back to source
inspection for a module that was skipped wholesale. The other 84 carry
`@pytest.mark.requires_runtime`; two matplotlib-only tests carry
`@pytest.mark.requires_figures`.

## Decisions later phases must not undo

1. **An optional dependency skips; it never fails.** A new test that needs the
   pinned runtime carries `@pytest.mark.requires_runtime`; one that renders a
   figure carries `@pytest.mark.requires_figures`. Inline
   `pytest.importorskip("brainos_runtime")` remains the convention for a module
   that is live end to end (`tests/integration/*`).
2. **The ledger names tests, and it reads the plan.** §24 is verified by
   `node_id`, so renaming or deleting a mapped test fails the build. Adding a
   requirement to §24 without a row here fails too.
3. **A skipped module is checked differently from a renamed test.** The ledger
   accepts a mapped id when it is either collected, or *defined in the source of
   a module that contributed no ids at all*. A rename inside a collected module
   still fails.
4. **The application has no logger, and that is now a tested property.**
   Diagnostics go through `print(..., file=sys.stderr)` in
   `app.service._warn_storage`, which reports an exception *type*. Adding a
   logger is a security change: `test_no_shipped_module_configures_logging_or_emits_a_record`
   fails until the redaction route is reviewed.
5. **Coverage is an instrument with a floor, not a target.** `fail_under = 90`
   sits below the measured 94% so a regression trips it while ordinary edits do
   not. Do not raise it to chase the number; raise it only after the missing
   lines are covered for a reason.
6. **CI runs the suite twice, on purpose.** `verify` proves the pinned-runtime
   path; `without-extras` proves the fake-backed path a Space or a contributor
   with no extras gets. Deleting the second job deletes the claim.
7. **A base install stays supported.** `pip install -e ".[dev]"` and
   `pytest -q` must stay green (with skips). Do not "fix" a future failure by
   making the extras mandatory.

## Bugs and gaps found while building it

1. **No test in the repository touched the log surface.** 1179 tests, and not
   one used `caplog`, a logging handler, or a captured stderr for a *credential*
   claim. §24's "API key never appears in logs" was satisfied by the absence of
   any logger rather than by an assertion — a property that would have gone
   unnoticed the day someone added one.
2. **The provider adapter was the least-covered module in the project (79%).**
   The uncovered branches were the ones that decide what a user sees when the
   SDK is missing, a gateway rejects the key, or the vendor answers with an
   unexpected shape. Those are exactly the paths a *test-suite* phase should
   find.
3. **A base install failed 86 tests.** See above. The suite had a
   documentation claim (`README.md`) that no test enforced.
4. **`tests/unit/test_controller.py::test_turn_and_message_limits_are_enforced`
   needed BrainOS for no reason.** It built its second controller with the
   default adapter factory while the limits under test were the controller's.
   Fixed by injecting the fake adapter the rest of the module already uses.
5. **Two figure tests sat outside the suite's `importorskip("matplotlib")`
   convention**, so they failed (rather than skipped) in the documented minimal
   `[dev]` environment. Both are now marker-guarded.
6. **The traceability ledger's first design was wrong**, and the base install
   found it: `--collect-only` reports no ids for a module skipped for a missing
   dependency, so a healthy skip looked like a deleted test. The ledger now
   distinguishes "module contributed nothing" from "module was collected but
   this test is gone".
7. **My own new CLI test asserted the wrong exit code** while probing the
   ceiling path: a budget stop that truncates mode/task coverage produces *both*
   a ceiling abort (3) and a controlled-comparison violation (2), and 2 wins by
   design. The behaviour is right and is now pinned as its own test (a repeated
   trial is the case where a ceiling stop stays controlled and exits 3).

## Measured behaviour

```bash
# Full environment (all extras): the suite and the coverage floor
$ .venv/bin/python -m pytest -q --cov --cov-report=term-missing:skip-covered
# 13 files skipped due to complete coverage.
# TOTAL   9067   543   94%
# Required test coverage of 90.0% reached. Total coverage: 94.01%
# 1248 passed in 325.77s (0:05:25)

# Base install — the CI `without-extras` job. The same suite, no extras, no failures.
$ python -m venv /tmp/civenv
$ /tmp/civenv/bin/pip install "pytest>=8.0" "pytest-cov>=5.0"
$ /tmp/civenv/bin/python -m pytest -q --tb=line -rs
# SKIPPED [1] tests/integration/test_brainos_runtime.py:13: could not import
#              'brainos_runtime': No module named 'brainos_runtime'
# SKIPPED [1] tests/unit/test_run_provenance.py:61: the pinned BrainOS runtime is
#              not installed (pip install -e '.[integration]')
# 1073 passed, 98 skipped in 10.56s

# The ledger, on its own
$ .venv/bin/python -m pytest tests/unit/test_plan_traceability.py -q
# 12 passed in 3.72s

$ .venv/bin/ruff check .
# All checks passed!
```

The 98 skips are 85 marker-guarded tests plus 13 whole modules that skip
themselves at import (`tests/integration/*`, `tests/ui`); those modules' 77
tests are not collected at all without their extras, which is why the base
install collects 1171 items instead of 1248.

Live proof that the new guarantees hold against the pinned runtime comes from
the existing live files, which run in the full environment and skip in the base
one: `tests/integration/test_security_live.py` (10), `test_pipeline`'s live
sibling `tests/integration/test_automated_pipeline_live.py` (7),
`tests/integration/test_chat_controller_live.py` (4), and the adapter boundary
files. No provider key was used anywhere in this phase: every new test drives a
deterministic double, and the two CLI tests monkeypatch the provider seam rather
than reaching a network.

## Constraints carried into Phase 19+

1. **Phase 19 (MVP definition) inherits a green, self-describing suite.** The
   §25 checklist (open the Space, connect, chat, store a fact, retrieve it,
   inspect memory and context, compare with full context, see token usage, run a
   small benchmark, export) has no single test today — it is spread across unit,
   UI, and integration files. A checklist file that walks the 14 items in one
   session is the natural Phase 19 artifact, and this phase's ledger is where it
   gets registered.
2. **CI is now the enforcement point.** `.github/workflows/ci.yml` runs on every
   PR; a phase that changes a guarantee should change the test that names it in
   the ledger, not just the code.
3. **Phase 20 (research release) should publish the numbers, not the suite.**
   The measured behaviour sections in `docs/phase-*.md` are the raw material;
   the release documents the benchmark methodology and results, and it should
   cite the run manifests rather than re-running tests.
4. **The 86-failure finding is a warning about documentation claims.** The
   README asserted a property no test enforced. Phase 19's MVP section should be
   checked the same way: if the README says a visitor can do it, something
   should fail when they cannot.
