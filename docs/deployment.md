# Deployment

How BrainOS Context Lab is run somewhere other than a developer's checkout — a
Hugging Face Space, a container, or a local server — and which decisions belong
to the deployment rather than to the code.

Phase 14 built the deployment surface (a Space-shaped entry point, dependency
files, bounded concurrency, an honest header, WAL-backed SQLite, an in-memory
mode). Phase 20 closes the gap the audit found in that milestone: the README had
no Hugging Face front matter, so a Space created from this repository would not
have known what to build, and a Space operator had no way to narrow the
Evaluation tab without editing source.

## Space manifest

The first block of [`README.md`](../README.md) is the Space manifest. Hugging Face
reads it when a Space is created from this repository; GitHub renders it as a
small table, which is the usual trade for a repository that is also a Space.

```yaml
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
```

`app.py` bootstraps `src/` onto `sys.path` and calls `app.ui.main()` under
`__main__`, which is how the Gradio SDK starts a Space: the runtime executes
`app_file` as the main module. Two constraints the Space server enforces are
worth knowing before editing that block: `short_description` must be 60
characters or fewer (a longer one is rejected when the Space is created) and the
two colours come from Hugging Face's fixed list. Both are pinned by
`tests/unit/test_deployment_config.py`, because neither failure appears until a
Space is created.

`requirements.txt` and `packages.txt` are read by the build; both are pinned by
the same test file, including the BrainOS commit this repository is validated
against.

CPU basic hardware is enough. The application is a UI over a SQLite file and a
provider API; there is no model on the host.

## Variables

| Variable | Default | What it decides |
| --- | --- | --- |
| `BRAINOS_LAB_DB` | `data/brainos_lab.sqlite3` | Database path. `:memory:` runs everything in a shared in-memory cache and turns the header's persistence sentence off — the right choice for a Space that must not accumulate visitors' transcripts. |
| `BRAINOS_LAB_CONCURRENCY` | `3` | Gradio queue concurrency. Every turn touches SQLite and the pinned BrainOS runtime, so this is a load limit, not a preference. |
| `BRAINOS_LAB_MAX_QUEUE` | `32` | Queue depth before Gradio refuses work. |
| `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` | `0.0.0.0` / `7860` | Where the server binds. |
| `GRADIO_STRICT_CORS` | Gradio's own default (on) | Set `0` only to embed the UI in a frame. |
| `BRAINOS_LAB_EVAL_PRESETS` | all three presets | Comma-separated subset of `quick`, `standard`, `research`. |
| `BRAINOS_LAB_EVAL_MAX_TASKS` | `500` | Hard task ceiling, applied underneath the preset's own limit. |
| `BRAINOS_LAB_EVAL_MAX_REQUESTS` | `7500` | Hard request ceiling, applied underneath the preset's own limit. |
| `BRAINOS_LAB_EVAL_ALLOW_GENERATION` | `1` | `0`, `false`, `no` or `off` makes the Evaluation tab retrieval-only: the visitor can still measure prompts, tokens, retrieval, figures and the report, but nothing is sent to a model. |
| `BRAINOS_LAB_EVAL_RETENTION_DAYS` | `7` | Age at which an *abandoned* session's `results/ui/` artifacts are swept. `0` means **no sweep** — an operator has to write that deliberately. |

The five `BRAINOS_LAB_EVAL_*` variables are read once, at startup, by
`EvaluationPolicy.from_environment()` (`src/app/evaluation.py`). They can only
**tighten**: `EvaluationPolicy.apply` still enforces the preset's own ceilings,
so a variable cannot talk a run past the limits the preset records. A value that
cannot be obeyed — an unknown preset, `twenty` where a number belongs — raises a
`ValueError` naming the variable, because a public deployment should fail where
its operator can see it rather than quietly host the widest policy the code
allows.

## Two postures

**A public Space.** Retrieval-only by default, small ceilings, retention decided
explicitly:

```bash
BRAINOS_LAB_DB=":memory:"
BRAINOS_LAB_CONCURRENCY="2"
BRAINOS_LAB_MAX_QUEUE="16"
BRAINOS_LAB_EVAL_PRESETS="quick"
BRAINOS_LAB_EVAL_MAX_TASKS="20"
BRAINOS_LAB_EVAL_MAX_REQUESTS="60"
BRAINOS_LAB_EVAL_ALLOW_GENERATION="0"
BRAINOS_LAB_EVAL_RETENTION_DAYS="0"
```

With generation off, a visitor can still walk every MVP item that does not need a
model and read the whole reporting pipeline; the only thing the deployment
refuses is spending the visitor's key on the host's queue. If the operator wants
generation on, the visitor's own key is what pays — this application never reads
an operator key.

**A research deployment** (this repository's own runs) is the default posture:
all presets, the plan's ceilings, generation on, seven-day retention for
abandoned sessions. The command-line pipeline

```bash
brainos-context-pipeline --preset standard --generate \
  --model <model> --api-key-env OPENAI_API_KEY \
  --dataset benchmarks/context_rot/generated/standard.jsonl
```

reads its credential from the named environment variable and never from
`BRAINOS_LAB_*`.

## Operator checklist

1. Create the Space from this repository (Gradio SDK, `app_file: app.py`).
2. Set the variables above for the posture you want — nothing else is required.
3. Paste no secret into the Space settings. The application is bring-your-own-key:
   the visitor's key lives in server memory for their session and `End session`
   discards it.
4. Decide retention (`BRAINOS_LAB_EVAL_RETENTION_DAYS`) and say so in the Space
   card, or set it to `0` and rely on `End session`.
5. Run `python -m security.scan .` on the tree you deploy (the repository's own
   scan covers the shipped surface and reports zero findings).
6. Record the Space URL in [`README.md`](../README.md) and
   [`CONTEXT.md`](../CONTEXT.md). The MVP checklist's first item is
   "open the HF Space"; it is the one item this repository cannot satisfy by
   itself, and it is still unmet — no Space URL exists yet.

## What the manifest does not claim

A Space built from this repository runs the same code the tests run, but a
running Space is not verified by this suite: `tests/integration/test_mvp_walkthrough.py`
walks the checklist against the pinned runtime locally, and
`tests/unit/test_deployment_config.py` pins the files a Space build reads. The
repository's habit is to say which half is checked (see
[`limitations.md`](limitations.md)).
