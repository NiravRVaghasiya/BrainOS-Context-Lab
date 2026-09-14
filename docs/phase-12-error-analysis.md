# Phase 12 log — Error analysis

**Status:** complete (branch `arena/01a09f9e-brainos-context-lab`)
**Plan section:** Phase 12 — Error Analysis (§18).
**Depends on:** Phase 3 (retrieval audit / drop reasons), Phase 7 (scorer,
verdicts, ledger), Phase 8 (metric split), Phase 9 (controlled comparison),
Phase 10 (paired statistics), Phase 11 (ablations).

## Objective

From the plan:

> Create a structured error taxonomy: `missed_memory`, `wrong_memory`,
> `stale_memory`, `conflicting_memory`, `irrelevant_memory`, `hallucination`,
> `over_compression`, `under_compression`, `wrong_abstention` … For every failure
> record: task id, mode, conversation length, expected memory, retrieved
> memories, answer, failure type. This is more valuable than simply collecting
> aggregate accuracy.

Phase 11 left four constraints, and they are the specification this phase was
built to:

1. build on the existing seams — `score_record` already emits verdicts and the
   Phase 12 labels, and `RetrievalReport.dropped` already carries audit reasons
   with a documented reason → taxonomy mapping: **aggregate, do not re-grade**;
2. keep `over_compression` (evidence dropped on the way to the prompt) and
   `missed_memory` (never recalled) as different findings with different owners;
3. carry all four metric contexts on every failure record — verdict, retrieval,
   prompt evidence, grounding — so a failure can be attributed to a stage;
4. aggregate by mode (baselines **and** ablations), category, and length, and
   show where each failure type concentrates.

## Starting state

- The scorer exported six verdicts and six error labels. Four of the plan's nine
  labels were never emitted by anything: `over_compression`,
  `under_compression`, `irrelevant_memory`, `wrong_memory`.
- The Phase 3 retrieval audit — every dropped memory with its reason — reached
  the UI turn but stopped there. `ModeReplay.stats` carried the *cost* of the
  prompt, not the *reasons* behind its evidence.
- `RetrievalScore` reported `evidence_in_prompt` as a boolean: it said evidence
  was lost, not which fact was lost where.
- Nothing aggregated failures: `aggregate_metrics` reported `error_counts` and
  `verdict_counts`, which is a count per label, not a distribution over modes,
  categories, and lengths.

## Work completed

### New / changed modules

| File | Purpose |
| --- | --- |
| [`src/evaluation/errors.py`](../src/evaluation/errors.py) (new) | The taxonomy (labels, stages, definitions, attribution order), the drop-reason → label table, `classify_failure` (per-record attribution), `failure_record` (the plan's §18 shape plus the four metric contexts), `collect_failures`, `error_report` (aggregation, concentration, scorer-agreement check), `write_failure_records`, and the `python -m evaluation.errors` CLI. |
| [`src/evaluation/scoring.py`](../src/evaluation/scoring.py) | `RetrievalScore` gains `supporting_fact_ids`, `prompt_fact_ids`, `prompt_forbidden_fact_ids` and the derived `absent_from_prompt_fact_ids` / `lost_after_selection_fact_ids` / `unneeded_prompt_fact_ids`; `score_record` passes the Phase 3 audit through as `retrieval_audit` (`retrieval_audit()`). No verdict or metric changed. |
| [`src/evaluation/modes.py`](../src/evaluation/modes.py) | `ModeReplay` carries the question turn's `retrieval_report` (drop reasons + conflict resolutions), so a replay record explains its own prompt. |
| [`src/evaluation/analysis.py`](../src/evaluation/analysis.py) | `extract_mode_result_items` is public: Phase 12 normalizes live runs, exported experiments, and run files through the same function the statistical analysis uses. |
| [`benchmarks/fixtures/scripted_answers.jsonl`](../benchmarks/fixtures/scripted_answers.jsonl) (new) | A committed, deliberately imperfect answer set: correct by contract, except where a mode should fail (multi-hop guess, superseded values on the temporal/conflict ablations, an asserted fact on a session-isolated replay, a decline with the evidence present). It makes the Phase 12 validation reproducible without a provider key. |
| [`benchmarks/context_rot/generation.py`](../benchmarks/context_rot/generation.py) | `--manifest` now defaults to `MANIFEST.json` beside `--output` instead of the committed `benchmarks/context_rot/MANIFEST.json`. Regenerating the smoke tier is unchanged (same default paths, same bytes); regenerating any other tier can no longer overwrite the hash the smoke results are pinned to. |
| Tests | `tests/evaluation/test_error_taxonomy.py` (47), `tests/integration/test_error_analysis_live.py` (12, pinned runtime), `tests/security/test_error_records.py` (4), plus the manifest-default regression in `test_context_rot_dataset.py`. **632 → 696 tests**; `ruff check .` clean. |

### The taxonomy

Nine labels, each owned by exactly one pipeline stage. The stage is the point: a
recall failure is fixed in retrieval, a selection failure in the policy, a prompt
failure in the budget or the delimiters, a generation failure in the model or the
instructions.

| Label | Stage | Fires when | Definition |
| --- | --- | --- | --- |
| `missed_memory` | recall | a required fact is absent from the prompt and was never selected, in a mode that has a retriever or a history window | the mode's selection stage did not surface the fact, so no prompt could have carried it |
| `irrelevant_memory` | selection | a required fact was selected and then dropped as `low_relevance` / `weak_relevance` | the relevance judgement removed evidence the task needed |
| `over_compression` | selection | a required fact was selected and dropped by the item cap or a token budget (`cap`, `budget`, `memory_budget`, `chunk_*`), **or** was absent from a prompt that replays the whole transcript | required evidence was compressed out of the prompt |
| `under_compression` | prompt | the prompt carried a fact the task's contract forbids (a superseded value or a declared distractor) | harmful retention: an answer can be right beside the value it was meant to prefer over |
| `conflicting_memory` | generation | `stale_answer` on a **conflict** task | an explicit correction was not applied |
| `stale_memory` | generation | `stale_answer` anywhere else (the temporal category) | information replaced over time was not superseded |
| `hallucination` | generation | a value was asserted although abstention was expected, **or** a wrong value was asserted over a clean, complete prompt | a value the prompt does not support |
| `wrong_memory` | generation | the answer is wrong, the prompt carried every required fact, and it *also* carried evidence the task does not accept | the memory the model used was wrong for the task, not the model inventing |
| `wrong_abstention` | generation | the answer declined although every required fact was in the prompt | the information was there and the model did not use it |

Primary label = the earliest stage that can explain the failure, in
`ATTRIBUTION_ORDER`. Everything else detected stays on the record as
`contributors` and is counted by the aggregate, so a record is never reduced to a
single cause.

### Drop reasons → labels

The mapping published in [`docs/evaluation.md`](evaluation.md) is implemented
literally in `DROP_REASON_LABELS`, and unknown reasons map to **no label** — a
reason whose failure meaning is unknown must not produce a fabricated finding.

| Drop reason | Label |
| --- | --- |
| `low_relevance`, `weak_relevance` | `irrelevant_memory` |
| `stale`, `expired` | `stale_memory` |
| `superseded` | `conflicting_memory` |
| `cap`, `budget`, `memory_budget`, `chunk_budget`, `chunk_ceiling` | `over_compression` |
| `duplicate`, `duplicate_history`, `empty` | noise removed — never a failure |
| `suspicious` | quarantined by the injection guard (Phase 13), not a failure |

When the audit's truncated preview no longer carries a fact's markers, the drop
cannot be attributed to one fact; attribution then falls back to the record-level
reason groups and the note says so.

### Two decisions the reason table alone does not settle

**`over_compression` is a contributor on every record whose prompt lost required
evidence**, whichever stage lost it. The primary label says *why* the evidence
went missing; the contributor says *that* it did. Phase 11 asked for exactly this
separation — "the filter dropped needed evidence" and "never recalled" must not
collapse — and reporting only the primary would have hidden it, because on the
smoke tier the relevance filter always owns the multi-hop loss. The invariant is
asserted in `tests/integration/test_error_analysis_live.py`: for every failure
record, `absent_from_prompt_fact_ids` is non-empty **iff** `over_compression` is
among its labels.

**`under_compression` means harmful retention only.** A prompt that merely kept
*unneeded* ledger facts is reported as a field
(`unneeded_prompt_fact_ids` / `unneeded_in_prompt_count`), not as a failure:
on a record with a failing answer the excess cannot be separated from the model's
own error, and labelling it would blame the context for every wrong answer. A
prompt that kept a *forbidden* fact is a defect whatever the answer did — an
answer that is right while the superseded value sits in the prompt is not a
conflict-resolution success.

### Invariants the implementation asserts

- **The taxonomy never contradicts the scorer.** It reproduces `stale_answer`
  (conflict vs everything else), `wrong_abstention`, and the answer-side
  consequences of an `incorrect` verdict; it refines only `missed_memory` (was
  the fact ever selected? was it judged irrelevant?) and `hallucination` (was the
  prompt clean?). `labels_vs_scorer.unexpected` counts anything else and must
  stay zero; the report prints `agree` / `refined` / `unexpected` on every run.
- **Latent defects are records too.** A correct answer over a prompt that lost or
  kept the wrong evidence is reported with `latent: true`, never dropped: that is
  the D4-style "six correct guesses" case Phase 11 measured, and it is not a
  success.
- **Every record names a stage** from `{recall, selection, prompt, generation}`.

### CLI

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.errors results/run.json \
  --dataset benchmarks/context_rot/dataset.jsonl \
  --output results/report/error-report.json \
  --records results/raw/error-records.jsonl \
  --examples 1 --top 10
```

The positional arguments accept run files (`evaluation.run`) and experiment
artifacts (`evaluation.experiment`), in any combination. `--dataset` enriches
records with the fact ledger (expected memory text, per-fact blame); when the file
is absent the analysis still runs, with fact ids instead of fact text.

### Bugs found and fixed while building it

1. **The dry-run taxonomy labelled every ungraded record
   `under_compression`.** The first `under_compression` rule ("the prompt kept
   facts the task did not need") is true of every long-context prompt, so all 30
   records of the first dry run failed on a property of the benchmark rather than
   a property of the answer. A label that fires on every ungraded record is not a
   finding: the rule is now harmful retention only (a *forbidden* value in the
   prompt), and unneeded retention is reported as a field.
2. **`under_compression`'s wasteful-retention note disappeared for
   `wrong_memory` records.** The note guarding the two cases was written as an
   `elif` chain, so a record whose primary label was `wrong_memory` lost both
   notes. The conditions are now independent: every record whose prompt carried
   an unneeded fact gets the note, and records whose primary label already names
   the facts (`wrong_memory`) are the one exception.
3. **The taxonomy initially contradicted the scorer twice.** A temporal
   `stale_answer` was labelled `conflicting_memory` (the scorer reserves that for
   the conflict category), and a wrong value over a clean, complete prompt was
   labelled `wrong_memory` (reserved for prompts that carried evidence the task
   rejects). Both were fixed in the taxonomy, not in the scorer; the report's
   `labels_vs_scorer` check exists precisely to catch this class of drift, and it
   reports `unexpected=0` on both dry-run and scripted-answer runs.
4. **A path argument to the library read as "no failures".**
   `collect_failures("results/run.json")` (and `error_report([...path])`) returned
   an empty report — a path is neither a mapping nor a sequence of artifacts —
   which is indistinguishable from a clean run. Reading files is the CLI's job,
   so the normalizer now refuses a bare `str`/`Path` with a message that says so.
5. **Regenerating the quick tier overwrote the committed benchmark manifest.**
   `--manifest` defaulted to the committed smoke manifest, so a run that wrote
   `generated/phase12-quick.jsonl` silently replaced the `dataset_sha256` that
   the smoke results are pinned to. The default now derives from `--output`
   (`test_generating_another_tier_keeps_its_manifest_beside_its_dataset`).

## Measured behaviour

Live pinned BrainOS, committed smoke dataset, estimated counter, no provider key.
**Integration-validation observations, not research results.**

### Retrieval side (dry run, no answers graded)

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick \
  --modes all --dry-run --output results/phase12-dry-all.json
PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick \
  --modes ablations --dry-run --output results/phase12-dry-ablations.json \
  --baseline-mode brainos
PYTHONPATH=src .venv/bin/python -m evaluation.errors results/phase12-dry-all.json
PYTHONPATH=src .venv/bin/python -m evaluation.errors results/phase12-dry-ablations.json
```

```text
scored=35 graded=0 defects=15 observed=0 latent=15          (five baselines)
mode            records  latent  evidence_lost  top failures
brainos         3        3       1              under_compression=2, irrelevant_memory=1
brainos_rag     2        2       0              under_compression=2
full_context    2        2       0              under_compression=2
rag             2        2       0              under_compression=2
sliding_window  6        6       6              missed_memory=6

scored=35 graded=0 defects=17 observed=0 latent=17          (ablations)
mode                  records  latent  evidence_lost  top failures
brainos               3        3       1              under_compression=2, irrelevant_memory=1
brainos_no_conflict   3        3       1              under_compression=2, irrelevant_memory=1
brainos_no_memory     6        6       6              missed_memory=6
brainos_no_relevance  2        2       0              under_compression=2
brainos_no_temporal   3        3       1              under_compression=2, irrelevant_memory=1
```

Three things are visible that the aggregate metrics did not show:

- the multi-hop failure is a **selection** failure: `fact-2` was recalled, then
  dropped with `low_relevance` (`relevance=0.1167` below `floor=0.1200`), so the
  prompt carried one of two required facts;
- removing the relevance filter (D2) removes the failure entirely — `evidence_lost`
  1 → 0, no `irrelevant_memory` — which is Phase 11's D2 repair expressed as an
  error distribution instead of a token delta;
- every mode that keeps a superseded value in its prompt is flagged
  `under_compression` on the temporal and conflict tasks, including Mode A:
  forbidden-in-prompt is 2 of 2 there.

### Length dimension (dry run, three tiers)

```bash
PYTHONPATH=src .venv/bin/python benchmarks/context_rot/generation.py --tier quick \
  --variants 1 --output benchmarks/context_rot/generated/phase12-quick.jsonl
PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick --modes all \
  --dry-run --dataset benchmarks/context_rot/generated/phase12-quick.jsonl --limit 14 \
  --output results/phase12-dry-quick.json
PYTHONPATH=src .venv/bin/python -m evaluation.errors \
  results/phase12-dry-all.json results/phase12-dry-ablations.json \
  results/phase12-dry-quick.json --output results/report/phase12-3tier-report.json
```

```text
scored=140 defects=64 latent=64
by_length   defects  evidence_lost  primary labels
800         32       16             missed=12, under=16, irrelevant=4
2000        15       7              missed=6,  under=8,  irrelevant=1
4000        17       10             missed=6,  under=7,  irrelevant=1, over=3
```

`over_compression` first appears as a *primary* label at 4 000 tokens, and only
for Mode A: the full transcript no longer fits, so required evidence is truncated
out of the prompt of the one mode that has no retriever to blame. That is the
context-rot effect the benchmark exists to measure, arriving in the taxonomy
before any model was called.

### Answer side (scripted answers, all nine modes, key-free)

```bash
for mode in full_context sliding_window rag brainos brainos_rag \
            brainos_no_temporal brainos_no_relevance brainos_no_conflict \
            brainos_no_memory; do
  PYTHONPATH=src .venv/bin/python -m evaluation.run --mode "$mode" \
    --answers benchmarks/fixtures/scripted_answers.jsonl \
    --output "results/phase12/run-$mode.json"
done
PYTHONPATH=src .venv/bin/python -m evaluation.run --mode brainos --session-isolation \
  --answers benchmarks/fixtures/scripted_answers.jsonl \
  --output results/phase12/run-brainos-isolated.json
PYTHONPATH=src .venv/bin/python -m evaluation.errors results/phase12/*.json \
  --output results/report/phase12-graded-report.json \
  --records results/raw/phase12-graded-records.jsonl
```

```text
scored=70 graded=70 defects=37 observed=10 latent=27
primary labels      missed_memory 12, under_compression 14, irrelevant_memory 4,
                    hallucination 3, conflicting_memory 1, stale_memory 1,
                    wrong_abstention 1, wrong_memory 1
contributors        over_compression 17, under_compression 2, missed_memory 1
stages              prompt 14, recall 12, generation 7, selection 4
drop reasons seen   low_relevance 8, weak_relevance 5  → all irrelevant_memory
labels vs scorer    agree=7 refined=3 unexpected=0
```

The eight observed failures the fixture drives all land on the intended label:
the multi-hop guess under `brainos` (`irrelevant_memory` + `over_compression`,
where the model was never given the second hop), the same question under D4
(`missed_memory`, nothing selected at all), the superseded value on the temporal
ablation (`stale_memory`), the corrected value on the conflict ablation
(`conflicting_memory`), the asserted fact on the isolated replay
(`hallucination`), the asserted value on the abstention task (`hallucination`),
the decline with the evidence present (`wrong_abstention`), and Mode A's wrong
value over an unneeded-fact prompt (`wrong_memory`).

`labels_vs_scorer` is the honest half of that table: 7 labels the scorer already
decided were reproduced, 3 were the documented refinements
(`missed_memory → irrelevant_memory` twice — the filter judged the fact
irrelevant — and `hallucination → wrong_memory` once, where the prompt was not
clean), and **no** label contradicted the scorer.

One example record (fields trimmed), which is the Phase 7 finding written down as
a failure:

```json
{
  "task_id": "cr-multi-hop-800-00",
  "mode": "brainos",
  "conversation_length": 80,
  "failure_type": "irrelevant_memory",
  "stage": "selection",
  "contributors": ["over_compression"],
  "observed_failure": true,
  "expected_memory": "telemetry service of Project Granite = MySQL 8; Project Granite = ca-central-1",
  "expected_memory_detail": [
    {"fact_id": "fact-1", "in_prompt": true,  "dropped_reason": "",              "blame": ""},
    {"fact_id": "fact-2", "in_prompt": false, "dropped_reason": "low_relevance", "blame": "irrelevant_memory"}
  ],
  "retrieved_memories": ["…MySQL 8…", "…MongoDB 7…", "…ca-central-1…"],
  "answer": "eu-west-1",
  "verdict": "incorrect",
  "scorer_error_type": "missed_memory",
  "retrieval": {"recall": 1.0, "precision": 0.667, "evidence_in_prompt": false},
  "grounding": {"faithfulness": 0.0},
  "drop_reason_counts": {"low_relevance": 2},
  "notes": ["required fact fact-2 was dropped from the selection (reason=low_relevance → irrelevant_memory)"]
}
```

Recall@K is 1.0 and the prompt still did not carry the fact: the record shows
*which* stage failed, which is the whole point of the phase.

## Findings this phase produced (carry into Phase 13)

1. **The four missing labels were missing for a reason: nobody had the reasons.**
   Every taxonomy label that the scorer could not emit needed data the application
   *had* and threw away — the per-fact prompt/selection split and the drop-reason
   audit. Exposing them changed no verdict and no metric, and made four labels
   computable.
2. **The relevance filter is now measurable as an error distribution.** `brainos`
   loses required evidence on 1 of 3 answerable smoke tasks (`irrelevant_memory`);
   D2 loses none. A report that only counted `evidence_in_prompt` would have
   hidden which task and which hop.
3. **Full context fails *its own* way.** Mode A is never blamed on a retriever —
   there is none — so its failures are `under_compression` (it keeps the
   superseded value: forbidden-in-prompt 2/2 on temporal and conflict) and, from
   4 000 tokens up, `over_compression` (the transcript no longer fits). Both are
   plain statements about the baseline the project is measured against.
4. **Scripted answers cannot support a claim about the model.** They validate
   that each label is reachable and correctly owned; only real generations
   (`--generate`, Phase 9's path) can say whether a model actually fails where the
   fixture pretends it does. Phase 13/17 must run the same report over generated
   answers and compare the distributions, not the accuracies.
5. **Two labels are dataset-limited, not code-limited.** `stale_memory` needs a
   `stale_answer` outside the conflict category — the temporal category supplies
   it, and the fixture drives it via D1. `wrong_memory` needs a wrong answer over
   a prompt that kept evidence the task rejects; on the smoke tier that is Mode A
   over an irrelevant fact. Neither is a general-purpose stress test at one seed
   and one length.

## Constraints carried into later phases

1. **Do not re-grade.** `verdict`, `error_type`, and every retrieval number come
   from `evaluation.scoring`. If a future phase needs a new label, it must be
   computable from what a run recorded — otherwise record more, and keep
   `labels_vs_scorer.unexpected` at zero.
2. **Keep the two compression labels distinct.** `over_compression` is about
   volume (cap, budget, truncation); `under_compression` is about harmful
   retention (a forbidden value in the prompt); wasteful retention is a field,
   not a label.
3. **Keep the audit with the replay.** `ModeReplay.retrieval_report` is what makes
   fact-level blame possible; a future replay path (Phase 17's pipeline) must
   carry it or the taxonomy degrades to reason-group attribution.
4. **Never put credentials in these artifacts.** `tests/security/test_error_records.py`
   pins that reports and failure records stay key-free even when a provider error
   carried a key, and that provenance is limited to paths and digests.
5. **The smoke tier still cannot support a claim** (Phase 8's constraint, still in
   force). Phase 12's numbers are integration-validation observations from one
   seed, one variant, and no model in the loop.

## Validation

```bash
.venv/bin/pytest -q          # 696 passed
.venv/bin/ruff check .       # All checks passed!

# retrieval side, no provider key
PYTHONPATH=src .venv/bin/python -m evaluation.experiment --preset quick \
  --modes all --dry-run --output results/phase12-dry-all.json
PYTHONPATH=src .venv/bin/python -m evaluation.errors results/phase12-dry-all.json

# answer side, no provider key: the committed scripted-answer fixture
PYTHONPATH=src .venv/bin/python -m evaluation.run --mode brainos \
  --answers benchmarks/fixtures/scripted_answers.jsonl \
  --output results/phase12/run-brainos.json
PYTHONPATH=src .venv/bin/python -m evaluation.errors results/phase12/run-brainos.json \
  --output results/report/phase12-graded-report.json \
  --records results/raw/phase12-graded-records.jsonl
```

No provider API key was used anywhere. The live tests skip when
`brainos_runtime` is not installed; the taxonomy tests need neither the runtime
nor a provider.
