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

> Bring your model. Give it memory. Measure context efficiency.

BrainOS Context Lab is a standalone experimental web platform for investigating whether external cognitive memory can reduce unnecessary historical context while preserving task performance in long-running LLM interactions.

This repository integrates BrainOS as an upstream dependency. It does **not** modify or reimplement BrainOS. The application owns the provider abstraction, session management, UI, context orchestration, benchmarking, and evaluation harness.

## Current status

Phases 1–19 are complete and validated against the pinned BrainOS runtime. The
app persists conversations and memory mirrors to SQLite with hard session
isolation, offers the full data-control set (clear conversation, clear memory,
export session, end/delete session), can run the same conversation through all
five of the plan's baseline context-management modes, ships a generated
context-rot benchmark with scoring, reports the plan's quality / efficiency /
robustness metric suite (including faithfulness and a degradation curve that
stays unset until more than one length is present), runs controlled
multi-mode experiments with cost budgets, summarizes trials with paired
statistics and effect sizes, ablates the BrainOS pipeline one component at a
time, turns the scored records into a structured failure taxonomy, guards
every route untrusted text takes into a prompt — with the guard's own findings
reported in the UI, in exports, and in evaluation artifacts, enforces
per-request and per-session cost ceilings on the chat path so a public BYOK
deployment cannot run away, stamps every run with a reproducibility
manifest (versions, environment, task ids, dataset digest, and a re-run
command) so a result can be re-derived and re-inspected, and runs the whole
benchmark end to end — controlled experiment, raw run files, aggregates,
statistics, failure taxonomy, figures, and a rendered report — from one command
or one button in the browser.

- **Phase 1** maps the provider abstraction (OpenAI and OpenAI-compatible)
  behind `LLMProvider`, with secret-safe errors and diagnostics.
- **Phase 2** maps the pinned BrainOS v2 runtime (`observe(source=, event_type=)`,
  `recall(top_k=)`, decision strings / `assess()`, `why()`, structured `trace()`,
  `contradictions()`, `stale_memories()`) behind a stable `BrainMemoryAdapter`.
- **Phase 3** turns recall results into a model-ready prompt through the full
  retrieval pipeline — deduplicate, IDF-weighted relevance filter, conflict
  check, recency weighting, enforced token budget — with complete accounting and
  a per-memory audit trail.
- **Phase 4** wires the Gradio UI to a web-framework-free `UIController`:
  bring-your-own-key provider connection, one chat turn, and the Memory,
  Context, Cognitive Trace, and Evaluation tabs.
- **Phase 5** persists conversations and memory mirrors to SQLite behind the
  `ConversationStore` / `MemoryStore` / `EvaluationStore` protocols:
  best-effort writes that never break a turn, session-key redaction at the
  write site, row-level session isolation, and export/delete controls.
- **Phase 6** implements the plan's five baseline modes — A full context,
  B sliding window, C lexical RAG, D BrainOS, E BrainOS + RAG — each fixing its
  own history window, with a BrainOS-free lexical chunk retriever, a second
  delimited evidence block, and the evaluation-runner seam that makes
  `python -m evaluation.run --mode …` execute.
- **Phase 7** turns the benchmark into an instrument: a deterministic generator
  for the plan's seven categories (single-hop, multi-hop, temporal, conflict,
  distractor, cross-session, abstention) across a 800-token smoke tier up to the
  plan's 5k–120k ladder, a fact ledger with a required/supporting/forbidden
  evidence contract, model-free retrieval scoring (Recall@K, precision,
  evidence-in-prompt), answer verdicts with an error taxonomy, and a runner that
  reports aggregates and records the dataset hash.
- **Phase 8** fills the metric suite on top of that scorer: faithfulness to
  retrieved evidence (grounding, not accuracy), conflict-resolution accuracy,
  token savings, quality-adjusted efficiency, optional latency, per-length
  curves, and degradation/AUC that is `null` on a single length rather than a
  silent zero. Plot-ready series for the six planned figures ship without
  requiring matplotlib.
- **Phase 9** runs the controlled comparison: the same tasks, model, sampling
  parameters, and scoring through every mode, with the credential read from
  the environment, run budgets (`quick` / `standard` / `research`), and a
  post-run constants check that fails the run instead of reporting an
  uncontrolled comparison.
- **Phase 10** adds trial statistics (mean, SD, 95% CI with exact t critical
  values), paired task tests with p-values, Cohen's d / Hedges' g effect
  sizes, win/loss/tie sign tests, and the six planned plots with error bars.
- **Phase 11** ablates Mode D one component at a time — no temporal signals,
  no relevance filtering, no conflict handling, no memory — pairing each
  ablation against the full system. Working memory and consolidation are
  excluded with documented pinned-revision reasons rather than run as null
  ablations.
- **Phase 12** classifies every defective record with the plan's nine-label
  taxonomy (`missed_memory`, `irrelevant_memory`, `over_compression`,
  `under_compression`, `conflicting_memory`, `stale_memory`, `hallucination`,
  `wrong_memory`, `wrong_abstention`), each attributed to a pipeline stage, and
  emits the plan's per-failure JSON records plus a report that aggregates them
  by mode (baselines *and* ablations), category, and length. Failure attribution
  builds on the existing scorer and the Phase 3 drop-reason audit — it never
  re-grades — and `labels_vs_scorer.unexpected` is printed on every run and must
  stay zero.

- **Phase 13** hardens the whole path: one shared injection guard (seven
  families, split into *intent* — which flags and can quarantine — and
  *structural* — delimiter and chat-template breakouts, role prefixes, invisible
  and control characters, which are rewritten), credential redaction on the two
  routes that were still open (replayed history and the current message), role
  containment so a pasted transcript cannot speak as the system, a per-session
  findings ledger rendered in a **Security** tab, `PRAGMA secure_delete=ON` plus
  `VACUUM` so a delete removes bytes and not just rows, and
  `python -m security.scan` to answer "is there a credential in this artifact?"
  over any directory. A quarantine is a security finding, not a tenth error
  label, so `labels_vs_scorer.unexpected` stays zero.
- **Phase 14** turns the repo into a deployable HF Space: a flat
  `requirements.txt` (pinned BrainOS commit, no `-e .[ui,providers]` only), a
  present `packages.txt`, WAL-mode SQLite with TRUNCATE checkpoints after
  every destructive write, a shared-cache `:memory:` mode for stateless
  deployments, a bounded Gradio queue (concurrency 3, queue depth 32, public
  API disabled) for multi-visitor safety, an honest header that discloses
  whether messages are persisted or run fully in memory, and deployment
  configuration tests that pin all of the above.
- **Phase 15** layers the plan's full cost-control surface onto the chat
  path: `ChatLimits` (7 ceilings — per-request input/output tokens, per-
  session turns/requests/tokens, request timeout), `ChatBudget` (mutable
  session accounting with turn gating, usage charging from provider-reported
  or estimated tokens, refusal counting, and a JSON snapshot), a wall-clock
  timeout wrapping the provider call, a Usage tab in the inspection panels,
  cost control widgets in the sidebar, usage in the session export, and
  budget reset on clear-conversation. A refusal is a user-visible status,
  not a security finding.
- **Phase 16** makes results reproducible: one `reproducibility` package is
  the single source of the plan's §22 vocabulary, and every
  `python -m evaluation.run` artifact, every controlled experiment, and every
  error report carries a `repro` manifest (run id, timestamp, pinned BrainOS
  revision, application version, provider/model/temperature, benchmark
  revision, mode, context budget, task ids, dataset path + SHA-256, and a
  copy-pasteable `--rerun` command). The chat export now records the session
  cost ceilings, the application/BrainOS/benchmark revisions, and a
  `chat_history_sha256` transcript digest, and the model dropdown refuses
  identifiers that echo the session key. A latent split-vocabulary bug — the
  runner/experiment recorded `context_rot-v1` where the dataset generator
  pins `context-rot-v1` — is fixed and pinned by a test.
- **Phase 17** automates the evaluation: `python -m evaluation.pipeline` runs
  the seven stages (`experiment → raw → comparison → statistics → errors →
  plots → report`) into the plan's `results/{raw,aggregated,plots,report}`
  layout, records a `StageResult` per stage, and exits non-zero when the
  comparison is uncontrolled (2), when a ceiling truncated the run (3), or when
  the pipeline is incomplete or a credential reached an artifact (4). Its
  credential scan does not just report: implicated files inside the run's own
  output directory are **quarantined** (deleted, with a `QUARANTINED.txt` and a
  `## Quarantine` report section left behind), because a detected key that stays
  on disk is a key that leaked. The same pipeline drives a full-width
  **Evaluation** tab — preset catalogue, a cost preview that renders
  `estimate_ceiling` *before* anything can be spent, the run itself, figures, the
  rendered report, per-session history, and optional generation with the
  visitor's own sidebar key (never an environment variable). `EvaluationPolicy`
  lets a deployment narrow what the tab offers; `End session` deletes the
  session's persisted rows *and* its artifacts.

- **Phase 18** turns the suite into a checked contract: a §24 ledger that maps
  every test requirement the plan names to a named test and fails when one is
  renamed or deleted, a log-hygiene module for "the API key never appears in
  logs" (the one §24 security rule no test had ever exercised), the provider
  adapter's untested edge paths (79% → 100%), the controlled experiment's CLI
  exit codes, a `requires_runtime` marker so a base install *skips* instead of
  failing the 86 tests that need the pinned runtime, a 90% coverage floor, and
  CI that runs the suite both with the pinned runtime and without any of the
  optional extras.
- **Phase 19** walks the plan's §25 MVP checklist instead of asserting it: one
  ordered session (connect → chat → store a fact → four unrelated turns →
  retrieve it → inspect memory → inspect context → compare with full-context
  mode → token usage → a small benchmark → export), one test per item, plus a
  ledger table that parses the checklist out of the plan. It also closes the
  three gaps the 2026-09-21 audit named — the request timeout is now triggered by
  a provider that really blocks, the SQLite stores are exercised by eight
  concurrent threads rather than by a comment, and `results/ui/` gains retention
  for the session nobody ends (`python -m app.retention`, age-based, capped,
  dry-runnable). Audit bug: the Evaluation tab resolved the dataset and its
  output root against the working directory, so starting the app from anywhere
  but the checkout raised `Dataset not found` and wrote a stray `results/`; the
  repository now knows where it is (`src/checkout.py`) and anchors its own paths
  without changing the relative paths the plan documents.

A session-scoped `ConversationService` combines the adapter, the retrieval
policy, the context builder, and the provider factory. Deterministic fakes cover
the whole pipeline without BrainOS installed; optional live tests exercise the
pinned runtime. 1349 tests pass and `ruff check .` is clean repository-wide.

Chat is usable without an API key: BrainOS still observes and retrieves memory,
and the panels show exactly what the model *would* have been sent. See
[`CONTEXT.md`](CONTEXT.md) for the living implementation state, the Phase 0–19
logs in `docs/`, and [`docs/status-audit-2026-09-21.md`](docs/status-audit-2026-09-21.md)
for the audit that motivates Phase 19.

### The MVP checklist (plan §25)

Every item below is walkable today and walked by
`tests/integration/test_mvp_walkthrough.py`, with the exception of the first —
publishing a Space is an operator action, and the repository checks the half it
owns (the app entry point, the declared dependencies, a Gradio app that builds).

| §25 item | How to do it |
| --- | --- |
| 1 Open the HF Space | **Not published** — the only unmet MVP item. The repository-side half is in place: the Space manifest is the first block of this README, and [`docs/deployment.md`](docs/deployment.md) carries the operator checklist. |
| 2–4 Select OpenAI, enter a key, select a model | Sidebar → provider, API key, model → **Connect**. The key stays in server memory and is never returned to the browser. |
| 5 Start a conversation | Type in the chat box. |
| 6 Store a fact | State it in a turn; the memory policy keeps durable project facts. |
| 7 Continue the conversation | Keep talking — the fact is no longer in recent history. |
| 8 Retrieve the fact later | Ask for it; the prompt contains the memory block. |
| 9–10 Inspect BrainOS memory / retrieved context | Memory and Context tabs, including why each memory was selected and what was dropped. |
| 11 Compare with full-context mode | Sidebar → context mode → `full_context`, ask the same question, compare the token accounting. |
| 12 See token usage | Usage tab: per-turn and per-session ceilings, requests, and timeouts. |
| 13 Run a small benchmark | Evaluation tab → preset → **Preview** (cost ceiling) → **Run**. A dry run needs no key. |
| 14 Export results | Session export (transcript, usage, runs) and the run's own `results/ui/<session>/<run>/` artifacts. |

Benchmark artifacts under `results/ui/` are kept for **7 days** after the last
file in that session was written and then swept automatically; **End session**
deletes a visitor's own artifacts (and persisted rows) immediately.
`python -m app.retention --dry-run` shows what a sweep would remove.

### Measured behaviour so far

On a synthetic 60-turn conversation with six durable facts (one corrected mid-
conversation), validated with the live pinned runtime and an estimated token
counter:

| Context configuration | mean tokens sent | full-context baseline | reduction |
| --- | --- | --- | --- |
| wide history window | 1486 | 1391 | 0.0% |
| memory-first | 387 | 1391 | **72.2%** |
| memory-first, tight | 308 | 1391 | **77.8%** |

6/6 probe questions received exactly the evidence they needed, and the corrected
fact never re-entered a prompt. The reduction is driven mainly by the
history-window setting rather than by memory selection — which is exactly why
Phase 6 gives every baseline mode its own window.

Phase 6 then ran one 28-message conversation through all five modes with the
live pinned runtime:

| Mode | tokens sent | full-context baseline | reduction | fact still in prompt |
| --- | --- | --- | --- | --- |
| A full context | 510 | 510 | 0.0% | yes |
| B sliding window | 189 | 510 | 62.9% | **no** |
| C lexical RAG | 256 | 510 | 49.8% | yes |
| D BrainOS | 182 | 510 | **64.3%** | yes |
| E BrainOS + RAG | 345 | 510 | 32.4% | yes |

Mode D cost fewer tokens than the sliding window *and* still carried the fact
the window had dropped.

Phase 7 then ran the committed benchmark (7 tasks, one per category, 800-token
tier) through all five modes with the live pinned runtime and no provider:

| Mode | mean tokens sent | full-context reference | reduction | Recall@K | evidence in prompt |
| --- | --- | --- | --- | --- | --- |
| A full context | 1152.7 | 1152.7 | 0.0% | 0.00 | 100% |
| B sliding window | 189.4 | 1152.7 | 83.5% | 0.00 | **0%** |
| C lexical RAG | 270.6 | 1152.7 | 76.5% | 1.00 | 100% |
| D BrainOS | 193.6 | 1152.7 | **83.2%** | 1.00 | 83.3% |
| E BrainOS + RAG | 367.1 | 1152.7 | 68.1% | 1.00 | 100% |

Mode D again cost about what the sliding window cost and carried evidence the
window lost — 5 of 6 answerable tasks against 0 of 6. It also **failed the
multi-hop category**: BrainOS recalled both required facts, but the retrieval
policy's relevance filtering delivered only one to the prompt. That is a
measurement, not a failure of the harness, and it is the first concrete target
for the Phase 11 ablation.

Phase 8 then scored the same smoke run through the full metric suite. Scripted
answers still produce accuracy 1.0 (plumbing). Faithfulness is **0.857**: the
multi-hop task is a correct guess whose evidence never reached the prompt, so
accuracy, Recall@K, evidence-in-prompt, and faithfulness are four different
numbers (1.00 / 1.00 / 0.83 / 0.86). The degradation AUC is `null` — one length
cannot support a curve.

Phase 11 ablated Mode D on the same smoke tier (dry run, no provider):

| Condition | Recall@K | evidence in prompt | mean tokens |
| --- | --- | --- | --- |
| D full | 1.00 | 0.833 | 193.6 |
| D1 no temporal | 1.00 | 0.833 | 193.6 |
| D2 no relevance | 1.00 | **1.000** | 215.7 |
| D3 no conflict | 1.00 | 0.833 | 193.6 |
| D4 no memory | 0.00 | 0.000 | 97.0 |

Removing the relevance filter repairs exactly the multi-hop gap (at +22
tokens) — confirming the filter as the component behind the Phase 7 finding —
while removing memory collapses evidence to zero with the window held
constant. D1/D3 are byte-identical to full: one short session gives recency
and retrieval-stage conflict handling nothing to decide.

Phase 17 then ran the whole benchmark through one command — two tasks of the
committed smoke tier, three modes, no provider:

```text
stage       status  detail
experiment  ok      2/7 task(s) × 3 mode(s) × 1 trial(s); dry run; 0 violations
raw         ok      3 run file(s) in the Phase 6/7 schema, each with its own repro manifest
comparison  ok      3 mode row(s), 6 plot series
statistics  ok      12 paired comparison(s) against `full_context`; one trial, intervals degenerate
errors      ok      1 failure record(s) (0 observed, 1 latent); labels_vs_scorer.unexpected=0
plots       ok      6 figure(s)
report      ok      15347 characters rendered from the pipeline artifact

  full_context  graded=0 accuracy=—  tokens=1226.5 reduction=0.000
  rag           graded=0 accuracy=—  tokens=281.0  reduction=0.771
  brainos       graded=0 accuracy=—  tokens=191.0  reduction=0.843
credential scan: files=16 findings=0 clean=True · exit_code=0
```

16 files across `results/{raw,aggregated,plots,report}`, and `accuracy=—` rather
than `0.000` because no model was asked anything: an ungraded answer-side metric
is unset, not zero.

**Integration-validation observations, not research results**: one seed, one
length tier, an estimated token counter, and no model in the loop — answer
accuracy is only measurable once real generations are graded. Every number in
this README came from a dry run or a deterministic fake.

Phase 20 (in progress) prepares the run the research question needs. The tiers
§29 asks for are generated — `standard` (56 tasks, 5k–40k, two variants), the
plan's `research` ladder (5k–120k) and three smaller slices — with their manifests
committed and the datasets local (`benchmarks/context_rot/generated/`). The
model-free runs over the new tiers, recorded in
[`docs/phase-20-research-release.md`](docs/phase-20-research-release.md), found
something the smoke tier could not show: the benchmark's replay session took the
product's default `max_tokens` ceiling, so above ~4k tokens Mode A — the "full
context" reference every reduction is priced against — silently became a 4k
window (4,086 / 4,090 / 4,090 tokens for 5k / 10k / 20k tasks, and one task in
four *lost its evidence to the truncation*). A replay now sizes its session
ceiling to the transcript being replayed (`evaluation.modes.replay_ceiling`),
and the re-measured runs show Mode A carrying the whole conversation again
(mean 4,307 tokens on the 2k/4k tier, evidence in prompt 1.000) while BrainOS
stays at 193 tokens. That fix is what makes the model half worth its budget; the
phase log also records what the harness costs and what is still open.

## Repository layout

```text
app.py                         # Local/Hugging Face Space entry point
pyproject.toml                 # Package metadata and optional dependencies
src/
  app/                         # UI, controller, session lifecycle, service, state,
                               #   and the Phase 17 Evaluation-tab runner
  baselines/                   # Phase 6 baseline modes A-E, the BrainOS-free
                               #   lexical retriever Mode C/E use, and the
                               #   Phase 11 ablation profiles (D1-D4)
  brain/                       # BrainOS adapter, retrieval policy, context
                               #   builder, memory policy, tokenizers, traces
  providers/                   # LLM provider interfaces and adapters
  security/                    # Phase 13: the shared injection guard, the
                               #   findings vocabulary and per-session ledger,
                               #   and the artifact credential scanner (CLI)
  storage/                     # Conversation and evaluation persistence
  evaluation/                  # Benchmark runners, mode strategies, metrics,
                               #   reports, the Phase 12 error taxonomy, and the
                               #   Phase 17 automated pipeline (CLI)
  reproducibility/             # Phase 16: the §22 run manifest, version reads,
                               #   rerun command, and persistence seam
benchmarks/
  context_rot/                 # Phase 7 generator, spec, scored dataset, manifest
  fixtures/                    # Deterministic fixtures (Phase 12 scripted answers)
docs/                          # Architecture, integration, evaluation, threat
                               #   model, security controls, deployment, and
                               #   per-phase logs
tests/                         # Unit, integration, evaluation, security, and UI
                               #   tests; the §24 traceability ledger; the
                               #   optional-dependency skip policy (conftest.py)
results/                       # Generated results (not committed by default):
                               #   raw/ aggregated/ plots/ report/, plus ui/ for
                               #   the Evaluation tab's session-scoped runs
```

## Local setup

Python 3.10 or newer is required. The base package intentionally has no mandatory third-party dependency while the upstream BrainOS API is being validated.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ui,providers,dev]"
```

Start the UI with:

```bash
python app.py          # http://localhost:7860
```

Select a provider, paste a session-only key, pick a model, and press **Connect**
to list the models that key can reach. The key is held in server memory, the key
box is cleared immediately, and **Forget key** drops it at any time. Without a
key the app still observes, retrieves, and builds context — the panels show the
exact prompt, but no model is called.

The tabs are Chat, Memory (stored / in-prompt / filtered-out / conflicts),
Context (summary, statistics, final prompt), Cognitive Trace, Security, Usage,
and — below the chat row, full width — **Evaluation**: pick a preset, preview
what it would cost, and run the benchmark. Retrieval-only by default (free, no
key, no prompts sent); tick **Call my model** to generate answers with the key
from the sidebar, billed by your provider and bounded by the preset's ceilings.

The upstream BrainOS revision is pinned and the Phase 2 adapter mapping is
wired. Install the optional integration extra to use the live runtime:

```bash
pip install -e ".[integration]"
```

### Test suite

```bash
pip install -e ".[dev,ui,providers,evaluation,integration]"   # everything
pytest -q                                                     # 1349 tests
pytest -q --cov --cov-report=term-missing                     # 94%, floor 90
ruff check .
```

The same two commands are the two jobs in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) (`verify` and
`without-extras`), and they run on every pull request.

Two optional dependencies change what the suite can do, and neither is required
to run it:

- **the pinned BrainOS runtime** (`[integration]`) — tests that exercise the
  application's real composition (the evaluation pipeline, the Evaluation-tab
  runner, the reproducibility artifacts) carry `@pytest.mark.requires_runtime`
  and skip with an install hint when it is absent. A base install
  (`pip install -e ".[dev]"`) runs `1149 passed, 104 skipped, 0 failed`.
- **matplotlib** (`[evaluation]`) — figure tests carry
  `@pytest.mark.requires_figures`.

The suite is organized as `tests/unit`, `tests/integration`, `tests/evaluation`,
`tests/security`, and `tests/ui`. `tests/unit/test_plan_traceability.py` is the
ledger for the implementation plan's §24: it names the test that carries each
requirement, checks that the plan still states that requirement, and runs the
suite's own collection so a rename or a delete fails the build instead of
silently dropping a guarantee.

### Provider adapter smoke test

The provider layer is usable without a live request by injecting a client in
unit tests. In an application service, construct a session-only configuration
and use the factory:

```python
from providers import ProviderConfig, create_provider

config = ProviderConfig(
    provider="openai",
    model="gpt-4o-mini",
    api_key="provided-for-this-session",
)
provider = create_provider(config)
provider.validate_credentials()
reply = provider.generate([{"role": "user", "content": "Hello"}])
print(reply.text)
```

Do not put a real key in source code, logs, benchmark records, or committed
examples. Generic OpenAI-compatible endpoints use
`provider="openai-compatible"` and a session-local `base_url`.

## Evaluation commands

```bash
# regenerate the committed benchmark (byte-identical, SHA-256 pinned)
python benchmarks/context_rot/generation.py

# a research-scale tier, written outside the tracked tree
python benchmarks/context_rot/generation.py --tier standard --variants 2 \
  --output benchmarks/context_rot/generated/standard.jsonl

# replay a dataset through one mode; retrieval metrics need no API key
python -m evaluation.run --mode brainos --output results/run.json

# grade supplied answers (JSONL: {task_id, mode?, answer}) and aggregate
python -m evaluation.run --mode brainos --answers results/model_answers.jsonl \
  --output results/brainos.json

# compare runs
python -m evaluation.compare results/full_context.json results/brainos.json

# Phase 11: ablate Mode D, pairing each ablation against the full system
python -m evaluation.experiment --preset quick --modes ablations --dry-run \
  --baseline-mode brainos --output results/phase11-dry.json
python -m evaluation.compare results/phase11-dry.json --stats \
  --baseline-mode brainos --output results/phase11-stat-report.json

# Phase 12: classify failures and aggregate them by mode, category, and length.
# Accepts run files and experiment artifacts; --dataset enriches the records
# with the fact ledger, --records writes the plan's per-failure JSON Lines.
python -m evaluation.errors results/phase11-dry.json \
  --output results/report/error-report.json \
  --records results/raw/error-records.jsonl

# Phase 12 answer side without a key: the committed scripted-answer fixture
python -m evaluation.run --mode brainos \
  --answers benchmarks/fixtures/scripted_answers.jsonl --output results/run.json
python -m evaluation.errors results/run.json --examples 1

# Phase 13: is there a credential in anything a run wrote?
# --secrets-env names an environment variable, so the value is never an argument.
python -m security.scan results/ data/brainos_lab.sqlite3 --secrets-env OPENAI_API_KEY

# Phase 16: inspect a run's reproducibility manifest and exact re-run command
python -m evaluation.run --mode brainos --limit 1 --output results/run.json
python -c "import json; print(json.load(open('results/run.json'))['repro']['rerun_command'])"

# Phase 17: the whole §23 layout in one command. A dry run needs no key — it
# builds every prompt, measures tokens, runs retrieval, compares modes, runs the
# statistics and the failure taxonomy, renders six figures, and writes the report.
python -m evaluation.pipeline --dry-run --limit 2 --output-dir results/phase17-dry
#   → results/phase17-dry/{raw,aggregated,plots,report}/  (exit 0, scan clean)

# A generated run reads the credential from the environment variable it names.
python -m evaluation.pipeline --preset research --modes all --provider openai \
  --model gpt-4o-mini --api-key-env OPENAI_API_KEY --baseline-mode full_context \
  --output-dir results/phase17-research

# A stage subset pulls in the stages it is made of, and the report then says
# which stages ran ("Partial pipeline — Only these stages ran: …").
python -m evaluation.pipeline --dry-run --stages report --output-dir results/p17

# Exit codes: 0 complete · 2 uncontrolled comparison · 3 stopped at a cost
# ceiling · 4 a stage failed or a credential reached an artifact (quarantined).
```

`--session-isolation` replays each transcript session separately, which is how
the cross-session category measures the product's "memory never crosses a
session" invariant instead of assuming it. `--limit N` caps the number of tasks
as a first cost control; the full Phase 15 run budgets (`quick` / `standard` /
`research` presets, per-request and per-run ceilings) apply on top of it once
`--generate` sends prompts to a model. Every run artifact also carries a
`repro` manifest — versions, task ids, dataset digest, and a `rerun_command`.
`python -m evaluation.pipeline` composes all of the above and writes the same
manifest on every artifact it produces, plus a `pipeline_rerun_command` that
reproduces the whole run.

## Deployment

The repository is shaped like a Hugging Face Space: `README.md` opens with the
manifest (`sdk: gradio`, `app_file: app.py`) and the build reads
`requirements.txt` and `packages.txt`. None of that is folklore — the manifest's
keys, the fact that `app_file` exposes the entry point, the manifest's SDK version
against the requirement the build installs, and every deployment variable below
are pinned by `tests/unit/test_deployment_config.py`.

A deployment narrows the Evaluation tab with environment variables instead of
editing source, and the variables can only tighten — the preset's own ceilings
are still applied underneath:

```bash
BRAINOS_LAB_DB=":memory:"             # stateless: no visitor transcript is kept
BRAINOS_LAB_EVAL_PRESETS="quick"      # a retrieval-only public posture
BRAINOS_LAB_EVAL_MAX_TASKS="20"
BRAINOS_LAB_EVAL_MAX_REQUESTS="60"
BRAINOS_LAB_EVAL_ALLOW_GENERATION="0"
BRAINOS_LAB_EVAL_RETENTION_DAYS="0"   # no automatic sweep; End session still deletes
```

**No Space is published yet.** The MVP checklist's first item ("open the HF
Space") is the one item this repository cannot satisfy by itself; when a Space is
opened, its URL belongs here and in `CONTEXT.md`. See
[`docs/deployment.md`](docs/deployment.md) for the full variable table, the two
recommended postures (public Space vs. research deployment), and the operator
checklist.

## Security principles

- Provider API keys are session-only secrets and must never be persisted or logged.
- The API key box is cleared on every connect attempt; the key is never rendered
  back to the browser, not even masked, and every panel value is redacted
  against it.
- Conversation and memory data are isolated by session, persisted only with
  credentials redacted at the write site, and deleted from disk by the clear
  and end-session controls — where "deleted" means the bytes are zeroed
  (`secure_delete`) and the freed pages are rewritten away (`VACUUM`), not merely
  unlinked.
- Retrieved memory and retrieved transcript chunks are data, not higher-priority
  instructions: both render inside their own delimiters, introduced as untrusted,
  with structural breakouts rewritten and intent-bearing patterns flagged (and
  quarantined under `drop_suspicious_memories`).
- The active session key is redacted by exact match from browser-visible
  diagnostics, from recalled memory text, from retrieved chunks, from replayed
  conversation history, and from the current message before any of them can reach
  a prompt.
- A replayed transcript cannot claim a role the application owns: history
  arriving as `system`, `tool`, `developer`, or `function` is sent as `user`, and
  the coercion is counted.
- Every guard action is reported in one vocabulary (category / action / route /
  stage / families) in the **Security** tab, in the session export, and in
  evaluation artifacts — as counts and masked previews, never as the payload or
  the credential it caught.
- Evaluation artifacts never contain provider credentials, and
  `python -m security.scan results/ data/brainos_lab.sqlite3` checks that
  claim over whatever a run actually produced (exit `0` clean, `1` findings,
  `2` nothing could be scanned). Naming a database scans the whole database:
  SQLite runs in WAL mode, so the newest rows can be in `lab.sqlite3-wal`
  rather than in the main file, and the scanner expands a named file's
  `-wal` / `-shm` / `-journal` sidecars instead of reporting one file clean.
- The Phase 17 pipeline scans its own output in two passes — everything stages
  1–6 wrote, then the report and manifest written afterwards — and a finding is
  not merely reported: the implicated files are deleted from that run's own
  output directory, the run exits `4`, and `report/QUARANTINED.txt` plus a
  `## Quarantine` section record what was removed and why. A credential that
  reached a model's answer should be treated as compromised and rotated.
- A benchmark run started from the browser uses the visitor's session key as a
  value passed to the provider factory. It is never exported to the environment,
  never written to an artifact (the manifest records the *route*, e.g.
  `the active browser session (openai bring-your-own-key)`), and it is deleted
  with the session's rows and artifacts on **End session**.
- Detection is pattern-based and English-only, and the reports say so: a clean
  report means nothing matched, not that the content was safe. The structural
  controls — delimiters, the untrusted-data preamble, and the rewrites — are what
  do not depend on recognising an attack.
- Users remain responsible for usage and costs charged by their provider account.

See [`docs/threat-model.md`](docs/threat-model.md) for the assets, actors, routes,
and residual risk, [`docs/security.md`](docs/security.md) for the maintained
controls, [`docs/phase-13-security.md`](docs/phase-13-security.md) for how they
were built and measured, [`docs/phase-16-reproducibility.md`](docs/phase-16-reproducibility.md)
for the reproducibility manifest,
[`docs/phase-17-automated-pipeline.md`](docs/phase-17-automated-pipeline.md) for
the automated pipeline and its Evaluation tab, and
[`docs/architecture.md`](docs/architecture.md) for the system boundaries.

## Research positioning

The project evaluates the following question rather than assuming a result:

> Does BrainOS-style external cognitive memory reduce historical context while maintaining task performance in long-running LLM interactions?

The initial release will compare Full Context, Sliding Window, Conventional RAG, and BrainOS under controlled benchmark conditions. Results should report both quality and efficiency, including retrieval quality, answer accuracy, context tokens, latency, and robustness as conversation length increases.
