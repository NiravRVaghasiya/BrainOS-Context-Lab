# Phase 19 — MVP polish

Phase 19 closes the deployment gap between a correct Evaluation tab and a tab
that can run for a long time on a hosted Space. The product boundary is
unchanged: BrainOS remains the upstream memory dependency; this application
owns the provider route, browser sessions, context orchestration, storage, and
evaluation pipeline.

## Retention contract

The browser writes credential-free evaluation artifacts below:

```text
results/ui/<session>/<run>/
```

`End session` still removes the active session's rows and files immediately.
That callback cannot run when a browser disappears, so `UIEvaluationRunner`
now performs a bounded retention sweep:

* once when the runner is constructed;
* before a preview; and
* before a run.

The default window is **24 hours since the newest file or directory activity**.
The deployment may set `BRAINOS_LAB_UI_RETENTION_SECONDS` to another positive
number of seconds. Invalid, zero, negative, or non-finite values fall back to
24 hours rather than disabling cleanup. A run operation protects the session it
is currently serving, while an inactive session is eligible after the window.

The sweep is intentionally narrower than a general-purpose janitor:

* only direct child directories of the resolved `results/ui` root are eligible;
* session names must satisfy the same safe path-segment contract as the runner;
* non-directories, unsafe names, and symlinked directories are skipped;
* the `ui` root is never removed and a symlinked `ui` root is refused;
* one filesystem error is recorded and does not abort the remaining scan.

`RetentionReport` is an in-process operational receipt with counts and byte
totals. It contains no prompts, provider configuration, or credentials and is
available as `runner.last_retention_report` for diagnostics. The sweep does not
claim to solve multi-process races: deployment workers must share a deliberate
artifact root and should not run concurrent janitors against the same volume.

## Public EvaluationPolicy

The browser-facing defaults are now explicit and conservative:

```python
EvaluationPolicy(
    allowed_presets=("quick",),
    allow_generation=False,
    max_tasks=20,
    max_requests=60,
)
```

`UIController()` and `create_app()` use `public_evaluation_policy()` unless a
caller injects an explicit policy or runner. This keeps local tests and the CLI
on the same `run_pipeline` entry point without silently enabling anonymous
model calls on the hosted UI. Retrieval-only runs can still measure prompt
construction, retrieval, tokens, and report artifacts; answer-side quality
metrics remain unset (`—`), not zero.

An operator can make an intentional deployment choice with environment
variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `BRAINOS_LAB_EVAL_PRESETS` | `quick` | Comma-separated known presets; an unknown or empty value falls back to `quick`. |
| `BRAINOS_LAB_EVAL_ALLOW_GENERATION` | `false` | `1`, `true`, `yes`, or `on` explicitly enables session-key generation. |
| `BRAINOS_LAB_EVAL_MAX_TASKS` | `20` | Positive hard task ceiling; invalid values fall back safely. |
| `BRAINOS_LAB_EVAL_MAX_REQUESTS` | `60` | Positive hard provider-request ceiling; invalid values fall back safely. |
| `BRAINOS_LAB_UI_RETENTION_SECONDS` | `86400` | Positive artifact-retention window. |

Enabling generation does not provide an operator key: the visitor's connected
session key remains the only credential route, and the existing quarantine and
credential-free artifact checks remain in force. The policy only tightens a
preset plan; it cannot raise the preset's own limits.

## Persistence and deployment boundary

The existing `BRAINOS_LAB_DB` setting still controls the SQLite persistence
choice. A persistent path retains server-side transcript/memory/evaluation
rows across application restarts; `:memory:` is ephemeral. Evaluation artifact
retention is separate: ignored `results/ui/*` files are bounded by the sweep,
while `End session` remains the immediate deletion path.

SQLite uses per-operation connections, WAL, and a busy timeout. Phase 18 proves
concurrent operations within one process and one database. Phase 19 does not
turn that into a multi-process queue or claim that multiple replicas can safely
share an arbitrary filesystem/database volume. A deployment that scales beyond
one process should choose a shared storage/database design and an external
coordination or cleanup mechanism before increasing evaluation concurrency.

## Validation

The phase adds focused tests for stale/fresh retention, protected active
sessions, invalid configuration, symlink/path safety, startup cleanup, public
policy defaults, and explicit policy overrides. The full repository validation
remains the release gate:

```text
.venv/bin/pytest -q                         # 1192 passed
.venv/bin/ruff check .                     # clean
git diff --check                            # clean
PYTHONPATH=src .venv/bin/python -m security.scan ...  # 94 files, 0 findings
```

Phase 20 remains the research release. Dry runs and deterministic fakes in this
phase validate plumbing and safety only; they are not evidence for a model
quality or context-rot research claim.
