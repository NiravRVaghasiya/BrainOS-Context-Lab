# Phase 14 — HF Deployment

> Plan reference: §20 ("HF Deployment"), §19 ("MVP Definition") for the BYOK
> flow, and §13 ("Security") for the credential rules a public Space must not
> weaken.

Phase 14 turns the working application into something that can run as a
Hugging Face Gradio Space (or any other public, multi-visitor Gradio
deployment) without violating the security posture Phases 0–13 built up. It
is the first phase that treats **the repository itself** as a deployment
artifact: requirements files, concurrency, database location, queueing, and
the header's persistence disclosure all become code that the test suite
holds to the same standard as a retrieval decision.

## Goals

From the plan §20:

1. A Space-ready `app.py` (already existed from Phase 4/5; made deployment
   aware in this phase).
2. A `requirements.txt` that installs on a fresh Space runner without local
   patches.
3. A `packages.txt` (apt packages; empty but present, because the Space SDK
   treats a missing file as an error).
4. A Space README (`README.md`, which the repo already uses as its primary
   documentation — it doubles as the Space's About tab copy).
5. The BYOK flow: user enters key → server-memory-only → provider call →
   discarded at session end — already implemented in Phase 4/5, re-verified
   under concurrent access.
6. No shared provider key anywhere in source, Space configuration, logs, or
   deployment metadata.

From §15 / §19 and the Phase 13 "next safe step":

7. A public Space is a multi-visitor process, so SQLite and BrainOS access
   must be safe under concurrent Gradio callbacks.
8. Ephemeral hosts (HF Spaces free tier, where the filesystem may be reset)
   must not silently lose data because of a misconfigured database path.
9. The app must expose an operator-controlled "run fully in memory" mode so
   a stateless demo can be deployed with a single env var and the UI must
   tell the visitor honestly whether messages are persisted.

## What was built

### Deployment files

| File | Purpose |
| --- | --- |
| [`app.py`](../app.py) (existing, unchanged shape) | Source-path bootstrap + `app.ui:main` entry point; the Space runs this file directly. |
| [`requirements.txt`](../requirements.txt) | Flat dependency list for Space build: `gradio>=6.0`, `openai>=1.0`, pinned BrainOS commit `1d9eb7a0…`, matplotlib, pandas, plus `-e .` so the project's own `src/` package layout is installed. |
| [`packages.txt`](../packages.txt) (new) | Apt packages; comment-only (the app is pure Python and ships matplotlib wheels) so the Space SDK does not reject the build. |

The previous `requirements.txt` was `-e .[ui,providers]` alone. That is fine
for a local checkout but is fragile on a Space build because: (a) Space
builders resolve optional extras after the editable install, (b) the pinned
BrainOS commit was hidden inside an optional extra (`[integration]`) not
installed by that line, and (c) the evaluation extras (matplotlib, pandas)
were omitted. The new file lists everything explicitly.

**No shared provider key** is anywhere in these files, and the artifact
scanner now covers `app.py`, `requirements.txt`, `packages.txt`, and
`pyproject.toml` in addition to the previous surfaces.

### SQLite concurrency hardening (`src/storage/sqlite.py`)

Phase 5's SQLite stores opened one connection per operation but used SQLite's
default (rollback) journal mode. That works for a single local user; two
Gradio worker threads writing at the same time get `database is locked`.
Phase 14 changes three things:

1. **WAL journal mode** on every disk-backed connection
   (`PRAGMA journal_mode=WAL`). Readers do not block writers and writers do
   not block readers, which is what a multi-visitor server needs.
2. **Busy timeout raised to 30s** (`PRAGMA busy_timeout=30000`) so two
   concurrent writers serialize instead of raising, and
   `PRAGMA synchronous=NORMAL` (safe with WAL; still durable against power
   loss).
3. **Destructive writes (`DELETE`, `clear`, `delete_session`) commit and
   `wal_checkpoint(TRUNCATE)` immediately** so deleted content is not left
   in the WAL for a raw byte scan to find. The `VACUUM` path also checkpoints
   before and after, so the Phase 13 secure-deletion contract holds under
   WAL.

The helper `tests/security/test_secure_deletion.py::raw()` was extended to
read the `-wal` and `-shm` sidecars as well as the main file. Without that
change the byte-level deletion assertions under-counted (data in WAL, not in
main file) and over-counted (deleted page images) once WAL was turned on.

### In-memory deployment mode (`BRAINOS_LAB_DB=:memory:`)

Setting `BRAINOS_LAB_DB=:memory:` now switches the stores to a shared-cache
in-memory database (URI `file:brainos_lab_shared?mode=memory&cache=shared`)
rather than treating the literal string `:memory:` as a filename. This means
an operator can deploy a fully stateless demo with one env var: every store
connection attaches to the same shared cache and the controller's
per-operation connection pattern still works, but nothing is written to disk
and a process restart clears everything.

Two helpers accompany this:

- `default_database_path()` returns `":memory:"` (a `str`) when the env var
  is set, instead of `Path(":memory:")` (which would resolve to a
  filesystem entry).
- `persistence_enabled()` returns `False` in that case, and the UI reads it
  to decide what the header says.

### Gradio queue and concurrency (`src/app/ui.py`)

`main()` now calls `demo.queue(default_concurrency_limit=3, max_size=32,
api_open=False)` before launching. Two reasons:

1. A public demo must not let an unbounded number of visitors drive
   concurrent BrainOS+provider calls from one process. `default_concurrency_limit`
   bounds Gradio's worker pool per callback; excess visitors wait in the
   queue rather than 502'ing the process. The limit is tunable through
   `BRAINOS_LAB_CONCURRENCY` and `BRAINOS_LAB_MAX_QUEUE`.
2. `api_open=False` disables Gradio's auto-generated REST API page. The
   internal routes used by the UI and the DownloadButton still work; random
   internet traffic cannot script the Space as a proxy.

`share=False` is set explicitly so the launch cannot create an Gradio share
tunnel behind an operator's back.

### Honest header (`_header_markdown()`)

The header previously hard-coded the "messages are persisted to a
server-side SQLite database" sentence. That becomes a lie on an in-memory
deployment. Phase 14 replaces the constant with a function that reads
`persistence_enabled()` and renders one of two sentences:

- disk-backed: "Transcript messages and BrainOS memories *are* persisted to a
  server-side SQLite database …"
- in-memory: "This deployment runs **fully in memory**: closing the tab,
  ending the session, or restarting the Space discards every transcript and
  every memory."

This is what makes the data-disclosure control real rather than a
hard-coded claim.

### New tests: `tests/unit/test_deployment_config.py`

11 tests covering:

- `app.py` bootstraps the source path and imports `main`.
- `requirements.txt` lists the five required runtime deps, contains no
  credential shape, and pins the BrainOS commit hash.
- `packages.txt` exists and contains only comments / valid apt package
  names.
- `BRAINOS_LAB_DB=:memory:` enables shared-cache in-memory mode with
  `persistence_enabled() == False` and journal_mode `memory`.
- Disk-backed stores open in WAL mode with a ≥30s busy timeout,
  `synchronous=NORMAL|FULL`, and `secure_delete=ON`.
- Two in-memory stores on the same sentinel share data (the shared-cache URI
  actually works).
- `delete_session` truncates the WAL so deleted bytes cannot be found by a
  raw scan.
- `main()` calls `demo.queue()` with `default_concurrency_limit`,
  `max_size`, and `api_open=False`, and does so before `demo.launch()`.
- The header renders the "fully in memory" copy when persistence is off and
  the "server-side SQLite" copy when it is on.
- The default database path is `data/brainos_lab.sqlite3` when the env is
  unset.

The artifact scanner (`tests/security/test_artifact_scan.py`) had its
`SHIPPED_PATHS` extended to include `app.py`, `requirements.txt`,
`packages.txt`, and `pyproject.toml` — so a future commit that accidentally
adds a placeholder key to any of those files fails the same test that
protects `src/` and `docs/`.

## Bugs and gaps found while building it

1. **WAL breaks "database.read_bytes() hides nothing".** Data written by one
   connection lands in the `-wal` file, not the main file, until a
   checkpoint; a vacuum that didn't checkpoint left markers in the WAL.
   Fixed by (a) running `wal_checkpoint(TRUNCATE)` after every destructive
   write and inside `vacuum()`, and (b) teaching the test helper to read the
   WAL and SHM files too so byte-level assertions are honest under both
   journal modes.
2. **`wal_checkpoint` needs its own transaction.** Issuing
   `PRAGMA wal_checkpoint(TRUNCATE)` inside the `with connection:` block
   (which opens an implicit transaction) raised `database table is locked`;
   moved to an explicit `commit()` followed by the checkpoint on a
   connection closed in `finally:` rather than relying on the context
   manager's auto-commit.
3. **`Path(":memory:")` is a filesystem path.** `default_database_path()`
   used to return `Path(configured)` for any non-empty env string, so
   `BRAINOS_LAB_DB=:memory:` created a file literally named `:memory:` in
   the working directory. The sentinel is now detected before `Path()`
   wraps it.
4. **The requirements file was `pip install -e .[ui,providers]` which did
   not include BrainOS.** A clean Space would install the app and its UI /
   provider extras but not the pinned `brainos-cli` distribution (that lived
   in the separate `[integration]` extra), so the first chat turn would hit
   the "install brainos" notice instead of running. The new requirements
   list BrainOS explicitly, which is what a deployment has to do.

## Measured behaviour

```bash
.venv/bin/pytest -q --ignore=tests/integration     # 925 → 936 passed
.venv/bin/ruff check .                              # All checks passed!
.venv/bin/python -m security.scan \
    src docs README.md CONTEXT.md benchmarks \
    app.py requirements.txt packages.txt pyproject.toml \
    BrainOS_Context_Lab_Implementation_Plan.md      # 81 files, 0 findings

# Server boots and serves HTTP
PYTHONPATH=src BRAINOS_LAB_DB=:memory: python app.py
# → Running on local URL: http://0.0.0.0:7860
# curl http://127.0.0.1:7860/ -> HTTP 200, Gradio HTML
```

No provider API key was used; BrainOS is installed for the live integration
suites which are excluded from the quick count above.

## Constraints carried into Phase 15+

1. **`requirements.txt` is a deployment surface.** New runtime dependencies
   must be added there as well as to `pyproject.toml`, otherwise local dev
   works and Space builds break.
2. **WAL + TRUNCATE is part of the deletion contract now.** Any future
   destructive write that skips the checkpoint leaves a deleted page in the
   WAL and regresses Phase 13's byte-level guarantee; the
   `test_deleting_a_session_checkpoints_the_wal` test pins that behaviour.
3. **Concurrency is bounded, not unlimited.** `default_concurrency_limit=3`
   is a starting point for a CPU-basic Space, not a research result. Phase
   15 will layer per-turn token/turn/timeout ceilings on top of the queue —
   concurrency alone is not a cost control.
4. **Phase 15 is still the remaining abuse mitigation.** Phase 14 enables
   deployment but does not yet implement per-request token caps, benchmark
   limits, or request timeouts at the UI layer (the evaluation runner's
   `RunLimits` do not apply to chat). Phase 14 CONTEXT notes this and
   recommends not exposing a public Space with BYOK billing until Phase 15
   ships.
5. **Ephemeral persistence must be a conscious choice.** The default is
   still `data/brainos_lab.sqlite3`; `BRAINOS_LAB_DB=:memory:` is opt-in for
   stateless demos. Do not flip the default silently — visitors expecting
   persistence would be surprised by a restart.
6. **The header copy is code, not static text.** Any future change to
   persistence behaviour (encrypted at-rest storage, server-side session
   sync, multi-user accounts in later phases) must update both the code and
   the `test_header_discloses_*` tests together.
