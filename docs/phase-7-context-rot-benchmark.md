# Phase 7 log — Context-rot benchmark

**Status:** complete (branch `arena/01a09c46-brainos-context-lab`)
**Plan section:** Phase 7 — Context-Rot Benchmark (§11), with the dataset-side of
§12–§14 prepared
**Depends on:** Phase 6 (the five baseline modes and the replay seam) and every
phase before it.

## Objective

From the plan: build long conversations with facts distributed across them —
recent, old, contradictory, repeated, irrelevant, temporally dependent — and ask
questions about those facts at the end. Seven task categories: single-hop,
multi-hop, temporal reasoning, conflict resolution, distractor resistance,
cross-session memory, and correct abstention.

Phase 6 left the benchmark as a three-task single-hop fixture with no fact
ledger, no scoring, and no way to say what "correct" meant. Phase 7 makes the
benchmark a real instrument: generated long conversations, a scorable fact
ledger, and rules that turn a replay into Recall@K, evidence-availability, and
answer verdicts — without needing a provider key to measure most of it.

## Starting state

- `benchmarks/context_rot/generation.py` produced three 2-message tasks; its own
  docstring said it was not a research dataset.
- `src/evaluation/datasets.py` carried `BenchmarkTask` with
  `required_memory_ids` and nothing else: no fact ledger, no evidence contract,
  no notion of a stale or absent fact.
- `EvaluationRunner.run()` accepted an evaluator and returned
  `aggregate_metrics={}`; nothing computed a metric from a run.
- `evaluation/metrics.py` already had `recall_at_k`, `precision_at_k`,
  `context_reduction`, and `summarize` — implemented and tested, but not called
  by any runner.
- Scoring, per the Phase 6 hand-off note, "needs a model" and was therefore
  deferred; the note also listed four constraints (evidence allowance is a
  contract, Mode C cannot abstain by design, use an exact counter for research
  runs, chunk drop reasons stay namespaced).

## Work completed

### New / changed modules

| File | Purpose |
| --- | --- |
| [`benchmarks/context_rot/spec.py`](../benchmarks/context_rot/spec.py) (new) | The declarative half of the benchmark: generator version, categories and their plan letters, the length tiers and the plan's 5k–120k ladder, the fact/distractor vocabulary, filler corpora, and every sentence template. A new category is a template plus one plan function, not a rewrite. |
| [`benchmarks/context_rot/generation.py`](../benchmarks/context_rot/generation.py) (rewritten) | The structural half: per-category fact plans, deterministic packing of facts and filler to a target length, session boundaries, the fact ledger, and the manifest. `python benchmarks/context_rot/generation.py --tier standard` produces a research-scale dataset; the committed file is the `smoke` tier. |
| [`benchmarks/context_rot/dataset.jsonl`](../benchmarks/context_rot/dataset.jsonl) (regenerated) | 7 tasks (one per category, 800-token tier, 66–80 messages each), 53 KB, SHA-256 pinned by the manifest. |
| [`benchmarks/context_rot/MANIFEST.json`](../benchmarks/context_rot/MANIFEST.json) (new) | Generator version, tier, lengths, categories, variants, seed, task count, mean achieved tokens, dataset hash, and the command that regenerates it. |
| [`src/evaluation/datasets.py`](../src/evaluation/datasets.py) (rewritten) | The data contract: `BenchmarkFact` (markers, type, introduced turn, subject/value, supersession chain, session index), `EvidenceContract` (required/supporting/forbidden/abstention), `BenchmarkTask` (ledger + contract + acceptable answers + session count), `validate_task`/`dataset_issues`, and JSONL load/write. Legacy `required_memory_ids` records still load. |
| [`src/evaluation/scoring.py`](../src/evaluation/scoring.py) (new) | Normalization, answer matching, abstention detection, marker-based fact detection, `RetrievalScore`, `AnswerScore`, `score_record`, `aggregate_scores`, and `AnswerSet` for `--answers` files. |
| [`src/evaluation/modes.py`](../src/evaluation/modes.py) (extended) | Replays now record the retrieved *text* (not just runtime ids), the stored-memory count, and the sessions used; `session_isolation=True` starts a fresh session at each transcript session boundary. |
| [`src/evaluation/runner.py`](../src/evaluation/runner.py) (extended) | Accepts an optional scorer and an answer map, attaches `scores` to each result, fills `aggregate_metrics`, validates the dataset, and records provenance (`benchmark_version`, `dataset_sha256`, `session_isolation`, `dataset_notes`). |
| [`src/evaluation/run.py`](../src/evaluation/run.py) (extended) | `--answers`, `--session-isolation`, `--limit`; hashes the dataset; prints a one-line summary; reports dataset problems on stderr instead of averaging them into a metric. |
| [`benchmarks/context_rot/README.md`](../benchmarks/context_rot/README.md) (rewritten) | What a task is, the marker convention, the categories, the tiers, how to generate, how to score, and what the benchmark is not. |
| Tests | `tests/unit/test_dataset.py` rewritten (15), `tests/evaluation/test_context_rot_dataset.py` (49), `tests/evaluation/test_scoring.py` (42), `tests/integration/test_context_rot_live.py` (10): **379 → 494**, `ruff check .` clean. |

### Key design decisions

- **A ledger, not just a transcript.** A task declares every fact it plants, the
  markers that identify it, and its supersession chain. Without that, "did the
  mode retrieve the right thing" is unanswerable after the fact — the runtime's
  memory ids are minted during the replay and differ from run to run.
- **Markers, all-of-them, co-occurring in one item.** A fact's markers pair its
  subject with its value, and detection requires them in the *same* memory, chunk,
  or prompt message. Joining evidence into one blob first counts "Project Cobalt"
  in one memory and "eu-west-1" in another as evidence for a fact that asserts
  both — a false positive that would silently inflate Recall@K.
- **Two validity properties, enforced by tests.** Every planted fact must be
  storable by the application's own memory policy (a fact that is never stored
  cannot be retrieved by any mode), and no filler turn may be storable (filler is
  length, not evidence; if it entered memory the comparison would be about
  noise). Both caught real generator bugs — see below.
- **Facts are never in reach of a window.** A fact is planted at most 78% of the
  way in and always at least four messages before the question, so the sliding
  window must lose it. Anything else would measure the window's luck.
- **Distractors scale with length but stay bounded** (≤13 facts, also ≤ the
  project vocabulary) so a long conversation stresses selection without making
  recall impossible — and so no two ledger facts can share a marker set.
- **Determinism is a feature with a hash.** Everything derives from
  `(seed, category, length, variant)`; the manifest pins the dataset hash, and a
  test regenerates from the manifest parameters and asserts the same hash.
- **Scoring is split.** Retrieval scoring is model-free, so a run without
  credentials still produces Recall@K, precision, and evidence availability.
  Answer scoring takes a supplied answer string, so a real model run, a re-grade
  under different rules, and a deterministic mock all go through one code path.
- **Abstention inverts the scoring order.** Where abstention is expected — the
  abstention category, or a cross-session task replayed with `--session-isolation`
  — declining is the only correct behaviour and asserting a value is a
  hallucination. Under isolation, a correct-looking answer can only be a guess or
  a session leak, and neither may score as success.
- **Verbatim replay vs session isolation are separate runs.** By default the
  whole transcript is one session (the long-range memory case). With
  `--session-isolation` a new session starts at each boundary, which measures the
  product invariant instead of assuming it.
- **The committed dataset stays reviewable.** The smoke tier is committed; every
  larger tier is generated on demand into a Git-ignored directory. A 120k-token
  conversation is a build artifact, not a source file.

### The committed dataset

| Task | Category | Messages | Achieved tokens | Required facts | Question distance |
| --- | --- | --- | --- | --- | --- |
| `cr-single-hop-800-00` | A | 70 | 804 | 1 (+1 repeated) | 55 turns |
| `cr-multi-hop-800-00` | B | 80 | 913 | 2 | 69 turns |
| `cr-temporal-800-00` | C | 80 | 877 | 1 (older forbidden) | 21 turns |
| `cr-conflict-800-00` | D | 68 | 744 | 1 (older forbidden) | 23 turns |
| `cr-distractor-800-00` | E | 70 | 803 | 1 (13 irrelevant) | 63 turns |
| `cr-cross-session-800-00` | F | 68 | 732 | 1 (session 0) | 57 turns |
| `cr-abstention-800-00` | G | 66 | 725 | none | — |

## Measured behaviour

### Mode comparison over the whole smoke dataset (live pinned BrainOS)

Seven tasks, five modes, no provider, dependency-free token counter. "In prompt"
is the share of *answerable* tasks whose required evidence reached the prompt.

| Mode | mean tokens sent | full-context reference | reduction | Recall@K | Precision@K | evidence in prompt |
| --- | --- | --- | --- | --- | --- | --- |
| A full context | 1152.7 | 1152.7 | 0.0% | 0.00 | 0.14 | 100% |
| B sliding window | 189.4 | 1152.7 | 83.5% | 0.00 | 0.14 | **0%** |
| C lexical RAG | 270.6 | 1152.7 | 76.5% | 1.00 | 0.39 | 100% |
| D BrainOS | 193.6 | 1152.7 | **83.2%** | 1.00 | 0.39 | 83.3% |
| E BrainOS + RAG | 367.1 | 1152.7 | 68.1% | 1.00 | 0.39 | 100% |

**Integration-validation observations, not research results** — one seed, seven
tasks, one length, no model in the loop, estimated token counter.

Mode D cost 4.2 tokens more than Mode B on average and carried evidence for 5 of
6 answerable tasks where Mode B carried evidence for 0. That is the comparison
the project exists to make, now measured on a generated benchmark rather than
asserted.

Per-category evidence availability:

| Category | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| single_hop | yes | **no** | yes | yes | yes |
| multi_hop | yes | **no** | yes | **no** | yes |
| temporal | yes | **no** | yes | yes | yes |
| conflict | yes | **no** | yes | yes | yes |
| distractor | yes | **no** | yes | yes | yes |
| cross_session | yes | **no** | yes | yes | yes |
| abstention | no | no | no | no | no |

### Findings this phase produced

1. **Mode D fails multi-hop in the current configuration.** Recall@K is 1.0 —
   BrainOS recalled both required facts — but only one reached the prompt
   (`selected_memory_count=1`, `memory_block_tokens=77` of a 1024-token
   allowance). The binding constraint is therefore the Phase 3 retrieval
   policy's relevance filtering and ranking, *not* the token budget: the second
   hop falls below the relative tail trim against a question that matches the
   first hop's vocabulary better. Modes C and E carry both hops. This is a
   concrete, reproducible target for the Phase 11 ablation and a reason Phase 8
   must report Recall@K and evidence-in-prompt separately — they disagree here.
2. **A stale value reaches every mode's prompt** on the temporal and conflict
   tasks (`prompt_contains_forbidden` is true for A, C, D, and E). In Mode A that
   is inevitable; in Mode D it means the superseded memory was retrieved
   alongside the correction and the policy did not drop it. Conflict resolution
   therefore has to be scored on the *answer* (`stale_answer`), which is exactly
   why the benchmark scores the two halves separately.
3. **Abstention is still not achieved by BrainOS mode** — the abstention task
   selects two irrelevant memories in Mode D (and six chunks in Mode C). The
   Phase 3/4 finding is unchanged; the category now measures it instead of
   relying on a one-off probe.
4. **Mode E earns its cost on this dataset** — it carried both hops for 1.9× Mode
   D's tokens. One seed is not evidence; it is a hypothesis for Phase 9.
5. **The harness is cheap enough to run in CI**: the seven-task, five-mode sweep
   plus the scored run takes ~4 seconds against the live pinned runtime.

### End-to-end scored run (no provider)

`EvaluationRunner` + `task_evaluator()` + `score_record` with answers scripted
from the contract ("I don't know" for the abstention task):

```text
task_count=7  graded=7  retrieval_recall=1.00  retrieval_precision=0.39
evidence_in_prompt_rate=0.83  answer_accuracy=1.00  abstention_accuracy=1.00
stale_answer_rate=0.00  mean_context_reduction=0.832
```

Accuracy is 1.0 only because the answers were scripted from the same contract the
scorer grades against — it validates the plumbing, not the system. The CLI path
(`python -m evaluation.run --mode brainos --answers … --output …`) was validated
the same way: with two answers supplied it reports `graded=2 accuracy=1.000`,
records `dataset_sha256`, and writes per-task `scores` blocks.

## Bugs found and fixed while building it

1. **Filler was being stored as memory.** `Nothing new since the last check,
   pass N.` matched the memory policy's statement-verb set through `\blast\b`
   ("last" is a verb there: *the release lasts 90 days*). Every mode's memory
   filled with chatter. Reworded to "previous"; the "filler never enters memory"
   test now guards the whole corpus.
2. **A distractor template was never storable.** `Project X archives release
   notes for N days` — "archives" is not in the statement set, so that
   distractor produced no memory and was invisible to the mode being tested.
   Reworded to "retains".
3. **`session_boundaries` metadata was overwritten by the plan's notes**, so the
   cross-session lookup received an integer instead of the boundary list.
4. **`score_record` dropped retrieved memory texts** through an operator
   precedence bug (`a or [] + b`): when memory evidence existed, the concatenation
   never ran, so retrieval scoring silently saw only chunks.
5. **No abstention pattern was reachable.** Normalization strips apostrophes, so
   `i don't know` could never match normalized text — every abstention answer,
   including correct ones, graded `incorrect`. Patterns are now normalized the
   same way the answers are.
6. **Marker detection across a joined blob produced false positives** (markers
   from two different memories counting as one fact). Detection is now per
   evidence item and per prompt message.
7. **`wrong_abstention` was documented but never produced**, and the first fix
   updated the dataclass without updating the returned payload. `score_record`
   now rewrites both.
8. **Ungraded records leaked into the error taxonomy** — a task with no answer
   supplied was labelled `missed_memory`. Missing data is not an observed
   failure: `ungraded` stays `ungraded`, and retrieval availability is reported
   as its own field.
9. **`distractor_count` could exceed the project vocabulary**, letting two ledger
   facts share a marker set. Capped at the available projects, which keeps
   detection unambiguous.
10. **`EvidenceContract.from_dict` crashed on an already-built contract** — the
    schema now accepts either form, which is what tests and hand-written tasks
    want.

## Validation

```bash
.venv/bin/pytest -q          # 494 passed
.venv/bin/ruff check .       # All checks passed!  (whole repository)

# regenerate the committed dataset (byte-identical)
.venv/bin/python benchmarks/context_rot/generation.py

# a scored run: no provider key, answers supplied as data
PYTHONPATH=src .venv/bin/python -m evaluation.run \
  --mode brainos --dataset benchmarks/context_rot/dataset.jsonl \
  --answers /tmp/answers.jsonl --output results/run.json
```

The live tests in `tests/integration/test_context_rot_live.py` run the committed
benchmark against the pinned BrainOS revision and skip when `brainos_runtime` is
not installed. No provider API key was used anywhere in this phase; generation is
exercised through `FakeProvider`-free paths and scoring through scripted answers.

## Constraints carried into later phases

1. **Retrieval-recall and evidence-in-prompt disagree** (the multi-hop case
   above), so Phase 8 must report both and Phase 11 should ablate the relative
   tail trim and the `max_memories` cap.
2. **The smoke tier is one length**, so no accuracy-vs-length curve exists yet.
   Phase 8/9 experiments should use `--tier standard` (or `research`) and
   multiple seeds/variants, writing into the Git-ignored generated directory.
3. **Grading answers needs the Phase 15 cost controls** before it can be exposed
   in the UI Evaluation tab (Phase 17): `--limit` exists as a first control, but
   token and run budgets do not.
4. **Use an exact token counter for research runs.** Every number here is
   `estimate_tokens`; `tiktoken` is not installed in this environment and
   `stats.token_counter` names the counter that produced each figure.
5. **The evidence allowance is still the contract**: Modes C/D/E share 1024
   tokens, so a new retrieval mode must fit inside it or the comparison stops
   being about selection.
6. **Mode C cannot abstain and that is a result.** The abstention category is
   where the two separate; do not "fix" the retriever with a floor without
   recording that the baseline changed.
7. **Do not tune thresholds on the smoke tier.** One seed at one length is a
   plumbing fixture; calibration belongs to Phase 8 with the standard tiers.
8. **Session isolation is measured, not assumed.** Keep the isolated replay in
   the suite; it is the only test that would catch a memory leak across sessions
   introduced by a future change to the adapter or storage layer.
