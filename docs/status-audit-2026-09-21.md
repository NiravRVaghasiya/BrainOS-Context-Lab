# Status audit — 2026-09-21

**Subject:** `NiravRVaghasiya/BrainOS-Context-Lab` at `187384d` (the PR #19 merge,
which is `origin/main`), audited on branch `arena/01a0c313-brainos-context-lab`.
**Scope:** what is implemented, what is verified, what the implementation plan
asks for that does not exist yet, and whether the plan's exit criteria,
milestones, and success criteria have been reached.

**Where its gaps stand now** (added after Phases 19 and 20; the audit body below
is the snapshot it was written as): gap 1 *no model in the loop* is still open —
it needs a key and a budget, and nothing else in this repository can substitute
for it; gap 2 *no research-scale dataset* is closed (the tiers are generated with
committed manifests); gap 3 *Phase 20 deliverables* is open with the model run;
gap 4 *Milestone 6 is not a release* is closed on the repository side — the Space
manifest, the deployment document and the deployment-narrowed Evaluation tab now
exist — with the Space URL itself still missing; gap 5 *Phase 19 not started* is
closed; gap 6 *the three missing tests and retention* is closed; gap 7
*abstention* is open and is a finding a real run will quantify; gap 8 *CWD-relative
roots* is fixed; gap 9 *no tag or release* is open (an operator decision, and the
last step of the deployment checklist).

## Verdict

**Phases 0–18 of the plan's 20 are complete, and the claims they make reproduce
in a clean checkout.** The instrument the plan describes — provider abstraction,
BrainOS adapter, context engine, BYOK UI, five baseline modes, a generated
context-rot benchmark, the metric suite, controlled experiments, statistics,
ablations, the failure taxonomy, the security controls, reproducibility
manifests, the one-command pipeline, and a §24 test ledger — is built, green,
and honest about what it has *not* measured.

**The experiment it exists to run has not been run.** Every number in the
repository comes from a dry run or a deterministic fake. No prompt has ever
been graded from a real model, the committed dataset is a 7-task smoke tier, and
the absence of a model in the loop is exactly why the plan §30 success criteria
(efficiency · quality · robustness) and most of the §29 evaluation matrix are
still unanswered.

Two phases remain: **Phase 19 — MVP Definition** (§25) and **Phase 20 —
Research Release** (§26). Phase 19 is a verification phase the repository is
one small artifact away from; Phase 20 cannot start honestly until a real
generated run exists.

## 1. Verification performed for this audit

Nothing below is quoted from the repository's own documentation; each line was
re-run in this checkout.

| Check | Command | Result |
| --- | --- | --- |
| Dependency install | `python3 -m venv .venv && .venv/bin/pip install -e ".[dev,ui,providers,evaluation,integration]"` | OK — pinned BrainOS runtime, Gradio, OpenAI SDK, matplotlib, pandas all import |
| Full suite + coverage | `.venv/bin/python -m pytest -q --cov --cov-report=term-missing:skip-covered` | **1248 passed**, 0 failed, 232 s; **93.99%** coverage (9067 stmts / 545 missed) against a 90% floor |
| Lint | `.venv/bin/ruff check .` | clean |
| §24 ledger alone | `pytest tests/unit/test_plan_traceability.py -q` | 12 passed |
| End-to-end pipeline | `python -m evaluation.pipeline --dry-run --limit 2 --output-dir /tmp/p17` | exit 0; 7/7 stages `ok`; 6 figures; report 15 314 chars; `files=16 findings=0 clean=True` |
| App boots | `GRADIO_STRICT_CORS=0 BRAINOS_LAB_DB=:memory: python app.py` | listening on `0.0.0.0:7860`, `HTTP 200`, 121 Gradio components |
| Credential scan, shipped surfaces | `python -m security.scan src app.py requirements.txt packages.txt pyproject.toml benchmarks docs` | 95 files, **0 findings**, `clean=True` |
| Credential scan, `tests/` | `python -m security.scan tests` | 39 findings — fixture-shaped sentinels, expected; the repository-clean assertion covers shipped surfaces only |
| MVP walkthrough (§25 items 2–14) | purpose-written script driving `UIController` against the **live pinned BrainOS runtime** with a stub provider | all 14 items pass: connect → chat → fact stored → retrieved later → panels → mode comparison → usage → benchmark run (7/7 stages) → export (7 152 chars, no key) → end session |
| CI on `main` | `gh run view 35577905245` | both jobs (`lint, test, coverage floor`, `suite runs without the optional extras`) green |

Repository shape: 126 Python files, 26.1k source LOC, 20.4k test LOC. Tests by
directory: unit 531, security 355, evaluation 259, integration 86, UI 17.

## 2. What is done — phase ledger

| Phase | Plan § | Status | Independent evidence |
| --- | --- | --- | --- |
| 0 — Research baseline | §4 | Complete | BrainOS pinned at `1d9eb7a0…`; upstream tests/eval/benchmark validated; `docs/phase-0-research-baseline.md` |
| 1 — Provider abstraction | §5 | Complete (adapter level) | `src/providers/`; **no live network call has ever been made** (`docs/phase-1-provider-abstraction.md` says so explicitly) |
| 2 — BrainOS adapter | §6 | Complete | `src/brain/adapter.py`; live tests against the pinned runtime pass |
| 3 — Context construction | §7 | Complete | dedupe → relevance → conflict → recency → budget, 26-field accounting, per-memory audit |
| 4 — Chat web UI | §8 | Complete | Chat, Memory, Context, Cognitive Trace, Security, Usage, Evaluation tabs; server boots and serves |
| 5 — Persistence | §9 | Complete | SQLite behind store protocols; session-key redaction at the write site; row-level isolation |
| 6 — Baseline modes A–E | §10 | Complete | full context, sliding window, lexical RAG, BrainOS, BrainOS+RAG, each with its own window |
| 7 — Context-rot benchmark | §11 | Complete | deterministic generator, 7 categories, fact ledger + evidence contract, SHA-256-pinned dataset |
| 8 — Metrics | §12–14 | Complete | recall/precision@K, accuracy, faithfulness, conflict **and abstention** accuracy, reduction, savings, QAE, degradation, AUC |
| 9 — Controlled experiments | §15 | Complete | `python -m evaluation.experiment`; post-run constants check; `quick`/`standard`/`research` budgets |
| 10 — Statistical evaluation | §16 | Complete | paired t-tests with exact p-values, CIs, Cohen's d / Hedges' g, sign test, all 6 required plots |
| 11 — Ablation study | §17 | Complete | D1–D4; working memory/consolidation excluded with documented reasons |
| 12 — Error analysis | §18 | Complete | nine-label taxonomy, stage attribution, per-failure JSON records |
| 13 — Security | §19 | Complete | shared injection guard, per-session findings ledger, quarantine, secure deletion, artifact scanner |
| 14 — HF deployment readiness | §20 | Complete as *readiness* | Space files present, WAL SQLite, bounded queue, honest persistence disclosure — but **no Space is published** |
| 15 — Cost controls | §21 | Complete | 7 chat ceilings, session budget, timeouts, usage tab, refusals as user-visible status |
| 16 — Reproducibility | §22 | Complete | `repro` manifest on every artifact — but the recorded `application_version` is `0.1.0` by fallback (no tag exists) |
| 17 — Automated pipeline | §23 | Complete | seven stages, `results/{raw,aggregated,plots,report}`, exit codes 0/2/3/4, Evaluation tab |
| 18 — Test suite | §24 | Complete | ledger maps every §24 item to named tests; log-hygiene and adapter-edge gaps closed; two-job CI |
| 19 — MVP definition | §25 | **Not started** | no §25 row in the ledger; no single test walks the 14 items |
| 20 — Research release | §26 | **Not started** | no results, no ablation results, no release artifacts |

Merged history: PRs #1–#19, one per phase, all merged; no open issues, no
milestones configured on GitHub, no tags, no releases.

## 3. Plan criteria — met or not

### 3.1 Exit criteria (the plan states them only for §4, §5, §6)

- **Phase 0 (§4):** met — upstream tests (316), upstream eval and long-run
  benchmark, and an `observe → recall` smoke test all passed against the pinned
  revision.
- **Phase 1 (§5):** **met at adapter/test level, not end-to-end.** "Connect an
  OpenAI-compatible model, send `Hello`, receive a response" is demonstrated
  against a deterministic fake client and, later, the real OpenAI SDK talking to
  a throwaway `127.0.0.1` stub. No request has ever reached a real provider.
- **Phase 2 (§6):** met — observe, retrieve later, build a relevant-memory set,
  and show the retrieved items, validated live against the pinned runtime.

### 3.2 Milestones (§27)

| Milestone | Content | Verdict |
| --- | --- | --- |
| 1 — Integration | BrainOS + one model | **Met** (with a stub/fake model; no live provider) |
| 2 — Web UI | BYOK + chat + memory inspection | **Met** |
| 3 — Baselines | Full context, sliding window, RAG, BrainOS | **Met** |
| 4 — Benchmark | long-context synthetic benchmark | **Met as an instrument**; only the 800-token smoke tier is committed and the 5k–120k ladder has never been generated |
| 5 — Evaluation | metrics + plots + ablations | **Met** |
| 6 — HF Release | polished UI, documentation, security, reproducibility | **Partially met** — everything except the release itself: no published Space, no results, no tag |

### 3.3 MVP checklist (§25, 14 items)

All 14 behaviours work in the running application today; twelve are covered by
tests spread across `tests/unit`, `tests/ui`, and `tests/integration`, and items
1–2 ("open the HF Space", "select OpenAI") are deployment facts rather than
behaviours. What is missing is exactly what `CONTEXT.md`'s "Next safe step"
names: **one artifact that walks the checklist in one place, registered in the
§24 ledger so it cannot rot.**

### 3.4 Success criteria (§30)

| Criterion | Evidence available today | Verdict |
| --- | --- | --- |
| Efficiency — BrainOS substantially reduces historical context | 64–84% reduction across dry runs, driven mainly by the history window | **Measured, but not yet research-grade** |
| Quality — performance comparable to full-context | accuracy, faithfulness, and QAE are **unset (`—`)**: no answer has ever been graded from a model | **Not measurable yet** |
| Robustness — slower degradation with length | `area_under_degradation_curve` is `null` on one length by design | **Not measurable yet** |

### 3.5 Evaluation matrix (§29)

Rows with implementation *and* a dry-run number: Recall@K, Precision@K, conflict
accuracy, context reduction, quality/tokens. Rows implemented but **blank until
a model is in the loop**: answer accuracy, temporal accuracy, abstention
accuracy, accuracy-vs-length curve. Row with no artifacts at all: **cross-model
evaluation** (§15's small/medium, ideally one API model plus one open-weight
model) — every run so far has used one model identifier.

## 4. What is missing, ranked

**Blocking any research claim**

1. **No model in the loop.** No generated answer has been graded, so quality
   metrics are unset and §29's headline comparisons are incomplete. The
   provider path has never touched a real API either, which means the first
   `--preset standard` run also exercises the provider adapter's
   live-network path for the first time.
2. **No research-scale dataset.** The committed dataset is `smoke`: 7 tasks, one
   length (≈800 tokens), one seed, one variant. The plan's ladder
   (5k/10k/20k/40k/80k/120k) and multiple variants/trials have never been
   generated, so no degradation curve exists.
3. **Phase 20 deliverables absent.** §26 requires Results and Ablation results
   in the release; neither can exist until 1 and 2 do. README, architecture,
   installation, methodology, dataset description, baseline definitions,
   limitations, security model, and reproducibility instructions are all
   already present.
4. **Milestone 6 "HF Release" is not a release.** `app.py`,
   `requirements.txt`, `packages.txt` are Space-ready, but no Space URL appears
   anywhere in the repository, and `README.md` carries **no Hugging Face YAML
   front matter** (`sdk:` / `app_file:` / title metadata) — a Space created from
   this repo needs that block before it will build as a Gradio Space.
5. **Phase 19 not started.** No §25 row in `tests/unit/test_plan_traceability.py`
   (13 `Requirement(...)` rows today, all §24), and no single test walks the
   checklist end to end.

**Known gaps the project already tracks** (`CONTEXT.md`, Phases 16–18)

6. No `SlowProvider` timeout test; no multi-threaded stress test; no retention
   policy for `results/ui/` — an abandoned session is ended by nobody, which
   matters before the Evaluation tab is enabled publicly with generation on.
7. **Abstention does not work yet**, documented honestly in
   `docs/limitations.md`: for an absent-answer question the relevance floor
   still admits memories (best relevance 0.344 against a 0.12 floor). §29's
   "does it know when it doesn't know" is currently a known failure, and
   abstention accuracy is one of the metrics a real run would expose.

**Found during this audit** (not previously recorded)

8. **The Evaluation tab's paths are CWD-relative.**
   `DEFAULT_DATASET = benchmarks/context_rot/dataset.jsonl`,
   `DATASET_ROOT = benchmarks`, and `UI_RESULTS_ROOT = results` in
   `src/app/evaluation.py` resolve against the process working directory. Running
   the controller from any other directory reproduces
   `Dataset not found: benchmarks/context_rot/dataset.jsonl` and fails the
   benchmark run (verified). The documented launch (`python app.py` from the
   repository root, or a Space, whose CWD is the repo root) is unaffected, but
   the installed console script `brainos-context-lab` would be broken from
   elsewhere. The roots are injectable constructor parameters, so the fix is
   small: resolve them relative to the package/repository root.
9. **No tag and no release.** `application_version` falls back to `0.1.0`
   (`src/reproducibility/run_provenance.py`), so a result can name the pinned
   BrainOS revision but not the application revision that produced it. §22 asks
   for the application version in the manifest; a `v0.x` tag at research-release
   time is what would make that field meaningful.
10. **Not a defect, but worth knowing:** `python -m security.scan tests` reports
    39 findings (fixture-shaped sentinels). That is by design — the repository
    assertion covers shipped surfaces only — but anyone running the scanner over
    the whole tree should expect a non-clean exit and read the report rather than
    panic.

## 5. Recommended next steps

**Phase 19 (verification — small, mechanical, high value)**

1. Add the §25 rows to `tests/unit/test_plan_traceability.py` and one
   end-to-end MVP test that walks all 14 items in a single session with the
   fakes, so a README claim again has a test that fails when it stops being true.
2. Add the two missing tests (`SlowProvider` timeout, threaded stress) and give
   `results/ui/` a retention rule, or set `EvaluationPolicy` to refuse
   generation by default until it has one.
3. Fix finding 8 (repo-root-relative dataset and results roots) — it is the only
   functional bug this audit found.

**Phase 20 (research release — needs a provider key and budget)**

4. Generate `standard` and then `research` tiers with multiple variants; run the
   controlled experiment across the five modes with a real model, then repeat
   with a second model for §29's cross-model row.
5. Publish the numbers that come out — including the ones that go against the
   project (the multi-hop relevance-filter gap, abstention, the smoke-tier
   caveat). The repository's habit of reporting its own negative results is its
   strongest asset; the release should not lose it.
6. Tag a release, add the HF YAML front matter, and deploy the Space with
   `EvaluationPolicy` tightened (generation off or hard ceilings) and retention
   decided, then record the Space URL in the README.

**Bottom line.** This is a well-built, unusually honest evaluation harness with
a verified green suite and a documented negative-result culture — and it has so
far measured only its own plumbing. The remaining work is not more
infrastructure; it is one real run, the numbers that come out of it, and the
release that publishes them.
