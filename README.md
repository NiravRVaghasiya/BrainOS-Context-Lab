---
title: BrainOS Context Lab
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
short_description: Bring your model, give it memory, measure context cost.
---

# BrainOS Context Lab

[![CI](https://github.com/NiravRVaghasiya/BrainOS-Context-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/NiravRVaghasiya/BrainOS-Context-Lab/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-lightgrey.svg)](./pyproject.toml)

> Bring your model. Give it memory. Measure context cost.

An experimental web platform for evaluating whether external cognitive memory ([BrainOS](https://github.com/NiravRVaghasiya/BrainOS)) can reduce historical context sent to an LLM while preserving task performance in long conversations. BrainOS is used as a pinned upstream dependency — this repository does not modify or reimplement it. It owns the provider abstraction, session lifecycle, context orchestration, UI, storage, and evaluation harness.

## Overview

Long-running LLM chats accumulate history linearly: every turn inflates the prompt, cost, and latency. BrainOS Context Lab tests a different path — let BrainOS observe, store, and recall only the durable facts, and build the prompt from recalled memory instead of the raw transcript.

**Who it's for:** researchers and engineers evaluating memory-augmented LLMs, benchmarking retrieval strategies, or prototyping BYOK (bring-your-own-key) chat systems that need auditable context accounting.

**Research question:**

> Does BrainOS-style external cognitive memory reduce historical context while maintaining task performance in long-running LLM interactions?

Every comparison is controlled — same tasks, same model, same sampling parameters, same scoring — and every artifact carries a reproducibility manifest so a result can be re-derived.

## Features

- **Bring-your-own-key chat** — OpenAI and any OpenAI-compatible endpoint; key held in server memory only, never persisted or logged, cleared after connect and on **Forget key / End session**
- **Session-isolated memory** — `BrainMemoryAdapter` maps BrainOS v2 (`observe`/`recall`/`assess`/`why`/`trace`/`contradictions`) per `session_id` with SQLite persistence (`ConversationStore` / `MemoryStore` mirrors, WAL, `secure_delete`)
- **Auditable context construction** — deduplication → IDF-weighted relevance filter → conflict/staleness check → recency weighting → hard token budget; every drop reason and token count is accounted and shown in the UI
- **Five baseline modes** — `full_context` (A), `sliding_window` (B), `rag` (C, lexical chunks), `brainos` (D), `brainos_rag` (E) — each with its own history window and a delimited evidence block
- **Context-rot benchmark** — deterministic generator for 7 categories (single-hop, multi-hop, temporal, conflict, distractor, cross-session, abstention) across tiers from 800 tokens to 120k; fact ledger + required/supporting/forbidden evidence contract; marker-based retrieval scoring needs no model call
- **Full metric suite** — retrieval (Recall@K, precision, evidence-in-prompt), answer accuracy, faithfulness (grounding), conflict-resolution, token savings, quality-adjusted efficiency, latency, per-length curves, and degradation/AUC (null on a single length, not zero)
- **Controlled experiments & statistics** — `python -m evaluation.experiment` runs all modes identically; paired tests, 95% CI (exact t), Cohen's d / Hedges' g, win/loss/tie; constants check fails an uncontrolled comparison
- **Ablations (D1–D4)** — BrainOS mode with one component removed (no temporal, no relevance filter, no conflict handling, no memory) to isolate what matters
- **Failure taxonomy** — 9-label classification (`missed_memory`, `hallucination`, `stale_memory`, …) attributed to a pipeline stage; per-failure JSON records aggregated by mode, category, and length
- **Security guard** — 7-family injection guard (intent vs. structural), delimiter isolation, role containment, credential redaction on every route into a prompt, per-session **Security** tab, `python -m security.scan` over any artifact directory (quarantines findings inside its own output)
- **Cost controls** — 7 ceilings (`max_input_tokens`, `max_output_tokens`, `max_turns`, `max_session_tokens`, `max_session_requests`, `max_message_chars`, `request_timeout_seconds`); wall-clock timeout on provider calls; live **Usage** tab; budgets enforced before any spend
- **Reproducibility** — run manifest on every artifact (run id, BrainOS revision, app version, provider/model/temperature, dataset path + SHA-256, `rerun_command`); anchored repo paths so `results/` never strays
- **One-command pipeline** — `python -m evaluation.pipeline` → `results/{raw,aggregated,plots,report}/` plus an in-browser **Evaluation** tab (preset catalogue + cost preview before spend)
- **Deployable** — Hugging Face Space (`app.py`, `gradio>=6.0`) and Vercel serverless (`api/index.py` → FastAPI + mounted Gradio, `:memory:` DB, `/tmp` artifacts)

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/NiravRVaghasiya/BrainOS-Context-Lab.git
cd BrainOS-Context-Lab

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Full local runtime (UI + providers + plots)
pip install -e ".[ui,providers,dev]"

# Optional: live BrainOS runtime (pinned commit)
pip install -e ".[integration]"

# Optional: plotting for the evaluation runner
pip install -e ".[evaluation]"
```

Flat installs for deployment use `requirements.txt` (HF Space) or `requirements-vercel.txt` (Vercel lean bundle without matplotlib).

## Usage / Quick Start

### 1. Launch the chat UI

```bash
python app.py
# → http://localhost:7860
```

1. In the sidebar, select **Provider** → `openai` or `openai-compatible` (with `base_url`).
2. Paste your **API key** → pick a **Model** → **Connect**. The key box clears immediately; the key lives in server memory only.
3. Chat. Without a key, BrainOS still observes and builds context — panels show the exact prompt that *would* be sent, but no model is called.

**Tabs:** Chat · Memory (stored / in-prompt / filtered / conflicts) · Context (summary, stats, final prompt) · Cognitive Trace · Security · Usage · **Evaluation** (full-width, below chat).

Session controls: **Clear conversation**, **Clear memory**, **Export session** (transcript + usage + runs), **End session / Delete** (rows + artifacts removed, `VACUUM` applied).

### 2. Provider adapter (headless)

```python
from providers import ProviderConfig, create_provider

config = ProviderConfig(
    provider="openai",              # or "openai-compatible" with base_url="..."
    model="gpt-4o-mini",
    api_key="provided-for-this-session",
)
provider = create_provider(config)
provider.validate_credentials()
reply = provider.generate([{"role": "user", "content": "Hello"}])
print(reply.text)
```

Do not put real keys in source, logs, or committed artifacts.

### 3. Evaluation CLI

```bash
# Regenerate the committed smoke dataset (byte-identical, SHA-256 pinned)
python benchmarks/context_rot/generation.py

# Generate a research-scale tier (local only, manifests committed)
python benchmarks/context_rot/generation.py --tier standard --variants 2 \
  --output benchmarks/context_rot/generated/standard.jsonl

# Retrieval-only run — no API key needed, measures tokens + evidence
python -m evaluation.run --mode brainos --output results/run.json

# Grade supplied answers
python -m evaluation.run --mode brainos --answers results/model_answers.jsonl \
  --output results/graded.json

# Compare two runs
python -m evaluation.compare results/full_context.json results/brainos.json

# Controlled multi-mode experiment (dry run)
python -m evaluation.experiment --preset quick --modes all --dry-run \
  --output results/exp.json

# Ablations: Mode D with one component removed
python -m evaluation.experiment --preset quick --modes ablations --dry-run \
  --baseline-mode brainos --output results/ablations.json

# Failure taxonomy
python -m evaluation.errors results/exp.json \
  --output results/report/error-report.json \
  --records results/raw/error-records.jsonl

# One-command pipeline: experiment → raw → comparison → statistics → errors → plots → report
python -m evaluation.pipeline --dry-run --limit 2 --output-dir results/pipeline-demo
# → results/pipeline-demo/{raw,aggregated,plots,report}/

# Live model run (key from environment, never as an argument)
python -m evaluation.pipeline --preset quick --provider openai \
  --model gpt-4o-mini --api-key-env OPENAI_API_KEY --output-dir results/live

# Credential scan over any artifact directory
python -m security.scan results/ --secrets-env OPENAI_API_KEY

# Retention sweep for abandoned browser sessions (default 7 days)
python -m app.retention --dry-run
```

Common flags: `--session-isolation` (cross-session replay), `--limit N`, `--generate` (call the model), `--baseline-mode full_context`, `--stages report` (partial pipeline). Exit codes: `0` complete · `2` uncontrolled comparison · `3` cost ceiling hit · `4` stage failed or credential found (quarantined).

## Configuration

Environment variables narrow what a deployment offers — they can only tighten preset ceilings, never loosen them.

| Variable | Default | Description |
|---|---|---|
| `BRAINOS_LAB_DB` | `data/brainos_lab.sqlite3` | SQLite path; `:memory:` for stateless/ephemeral (Vercel) |
| `BRAINOS_LAB_EVAL_PRESETS` | `all` | Allowed presets (`quick`, `standard`, `research`, `all`) |
| `BRAINOS_LAB_EVAL_MAX_TASKS` | preset default | Cap tasks per run |
| `BRAINOS_LAB_EVAL_MAX_REQUESTS` | preset default | Cap provider requests |
| `BRAINOS_LAB_EVAL_ALLOW_GENERATION` | `1` | `0` = retrieval-only, no model calls |
| `BRAINOS_LAB_EVAL_RETENTION_DAYS` | `7` | Days to keep `results/ui/<session>/`; `0` disables sweep (End session still deletes) |
| `RESULTS_ROOT` | `results` | Output root (`/tmp/results` on Vercel) |

See [`docs/deployment.md`](./docs/deployment.md) for the two recommended postures (public Space vs. research deployment) and the full operator checklist.

## Project Structure

```text
app.py                         # HF Space / local entry point
api/index.py                   # Vercel serverless entry (FastAPI + Gradio mount)
pyproject.toml                 # package metadata & optional dependencies
requirements.txt               # HF Space flat bundle
requirements-vercel.txt        # Vercel lean bundle (no matplotlib)
packages.txt                   # apt packages for HF Space image
src/
  app/                         # UI, controller, session, service, state, limits, retention
  brain/                       # adapter, context_builder, retrieval_policy, memory_policy, trace
  providers/                   # LLM provider interface (openai, openai_compatible)
  baselines/                   # modes A–E, lexical RAG, ablation profiles D1–D4
  evaluation/                  # runner, experiment, metrics, scoring, analysis, pipeline, reports
  reproducibility/             # run manifest, version reads, rerun command
  security/                    # injection guard, findings ledger, artifact scanner
  storage/                     # SQLite stores (conversations, memory mirrors, evaluations)
benchmarks/
  context_rot/                 # spec, generation, dataset.jsonl, MANIFEST.json, generated/ tiers
  fixtures/                    # deterministic fixtures (e.g. scripted_answers.jsonl)
docs/                          # architecture, evaluation protocol, threat model & per-phase logs
tests/                         # unit · integration · evaluation · security · ui
results/                       # generated outputs (raw/ aggregated/ plots/ report/ ui/) — not committed
```

## Evaluation & Benchmark

The **context-rot** benchmark (`benchmarks/context_rot/`) is the instrument for the research question. Each task is a long conversation (filler + planted facts) ending in a question that is out of reach for a small window.

| Tier | Target length | Use |
|---|---|---|
| `smoke` | 800 tokens | Committed dataset (7 tasks), runs in seconds |
| `quick` | 2k / 4k | Sanity check before a real run |
| `standard` | 5k / 10k / 20k / 40k | Paper-scale comparison (56 tasks, 2 variants) |
| `research` | 5k – 120k | Full ladder (§14); cheap to generate, expensive to run |

Generation is deterministic — `MANIFEST.json` pins generator version + seed + SHA-256, and `python benchmarks/context_rot/generation.py` reproduces the file byte-for-byte. Every run records `dataset_sha256` and `token_counter` so mismatched runs never compare silently.

Retrieval scoring is model-free (marker co-occurrence → Recall@K / evidence-in-prompt); answer scoring grades supplied `--answers` JSONL into `correct` / `abstained` / `wrong_abstention` / `stale_answer` / `incorrect` / `ungraded`.

Further reading: [`docs/evaluation.md`](./docs/evaluation.md) · [`benchmarks/context_rot/README.md`](./benchmarks/context_rot/README.md) · [`docs/evaluation-protocol.md`](./docs/evaluation-protocol.md)

## Security

- Keys are session-only, never persisted, never rendered back (even masked), and redacted from prompts, recalled memory, retrieved chunks, replayed history, and diagnostics by exact match.
- Retrieved memory and transcript chunks are **data, not instructions**: rendered inside delimiters with structural breakouts rewritten; intent-bearing patterns are flagged and optionally quarantined.
- Replayed history cannot claim privileged roles (`system`/`tool`/`developer`/`function` → `user`).
- Every guard action is reported as `PromptGuardReport` (category/action/route/stage/families) in the **Security** tab, session export, and run artifacts — masked previews only, never payloads.
- Deletes are byte-level (`PRAGMA secure_delete=ON` + `VACUUM`; WAL sidecars included in scans).
- `python -m security.scan <dir>` checks shipped surfaces and run outputs (exit `0` clean / `1` findings / `2` nothing scannable); a finding inside `results/` quarantines the file (`QUARANTINED.txt` + report section) and fails the pipeline.

See [`docs/threat-model.md`](./docs/threat-model.md) and [`docs/security.md`](./docs/security.md).

## Testing

```bash
pip install -e ".[dev,ui,providers,evaluation,integration]"
pytest -q                          # full suite
pytest -q --cov --cov-report=term-missing   # with 90% floor (see pyproject.toml)
ruff check .
```

- Tests that need the pinned BrainOS runtime are marked `requires_runtime` and **skip** with an install hint on a base install (`pip install -e ".[dev]"`).
- Tests that need matplotlib are marked `requires_figures`.
- `tests/unit/test_plan_traceability.py` is the §24 ledger — it maps every plan requirement to a named test and fails the build if a test is renamed or deleted.
- CI runs two jobs on every PR: `verify` (full suite + coverage) and `without-extras` (base install only). See [`.github/workflows/ci.yml`](./.github/workflows/ci.yml).

## Deployment

**Hugging Face Space** — the repo is shaped as a Space already: `app.py` + `requirements.txt` + `packages.txt` + the YAML frontmatter at the top of this file. `docs/deployment.md` is the operator checklist.

**Vercel** — `api/index.py` is the serverless entry (`[tool.vercel] entrypoint` in `pyproject.toml`). Build: `uv pip install -r requirements-vercel.txt` → `scripts/prepare_vercel_bundle.py` → `scripts/check_vercel_runtime.py`. Env defaults in `vercel.json` (`:memory:` DB, retrieval-only, `quick` preset). Sessions are per-function-instance; for multi-instance persistence replace `SessionManager` with Redis (see [`VERCEL_DEPLOYMENT_ANALYSIS.md`](./VERCEL_DEPLOYMENT_ANALYSIS.md) and [`docs/vercel-deployment.md`](./docs/vercel-deployment.md)).

No Space is published yet — the first item of the §25 MVP checklist. When one is, its URL belongs here and in [`CONTEXT.md`](./CONTEXT.md).

## Roadmap

Phases 1–19 are complete and validated against the pinned BrainOS runtime. Phase 20 (research release) is in progress:

- [x] Provider abstraction (OpenAI + OpenAI-compatible)
- [x] BrainOS adapter & retrieval pipeline
- [x] Gradio chat UI with inspection panels
- [x] SQLite persistence & session isolation
- [x] Five baseline modes + lexical RAG
- [x] Context-rot benchmark & scoring
- [x] Metric suite & degradation curves
- [x] Controlled experiments with budgets
- [x] Statistics (CI, paired tests, effect sizes)
- [x] Ablations (D1–D4)
- [x] Failure taxonomy & error analysis
- [x] Security guard, redaction & artifact scan
- [x] HF Space packaging & deployment config
- [x] Cost controls & timeouts
- [x] Reproducibility manifests
- [x] Automated pipeline + Evaluation tab
- [x] Test suite as contract (coverage floor, traceability ledger)
- [x] MVP walkthrough (§25)
- [ ] Research-scale generated runs with a live model (§26 / Phase 20)
- [ ] Published HF Space URL + tagged release

See [`CONTEXT.md`](./CONTEXT.md) for the living phase ledger and `docs/phase-*.md` for per-phase design notes. [`docs/limitations.md`](./docs/limitations.md) tracks known limitations.

## Contributing

Issues and pull requests are welcome. CI must stay green:

```bash
ruff check .
pytest -q --cov --cov-report=term-missing
```

Please do not commit API keys, `data/*.sqlite3`, or `results/` artifacts. The credential scanner (`python -m security.scan`) runs as a test over shipped surfaces — a finding there fails CI.

## License

MIT — see [LICENSE](./LICENSE). Copyright (c) 2026 BrainOS Context Lab contributors.

## Acknowledgments

- [BrainOS](https://github.com/NiravRVaghasiya/BrainOS) (pinned at `1d9eb7a`) as the upstream cognitive-memory runtime.
- Gradio, FastAPI, and the OpenAI SDK for the UI and provider layers.
