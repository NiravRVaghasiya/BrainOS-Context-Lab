# Context-rot benchmark

The Phase 7 benchmark for the project's research question: does performance
degrade more slowly when BrainOS, rather than a raw transcript window, decides
what history the model sees?

## What a task is

Each JSONL record is one conversation plus one question asked at the end:

```text
conversation      ~40–12,000 messages of filler + planted facts
facts             the ledger: every fact, where it was planted, what replaced it
evidence          required / supporting / forbidden facts (+ abstention flag)
question          asked after the fact is out of any small window's reach
expected_answer   what a correct answer contains
acceptable_answers  variants a correct answer may use instead
metadata          seed, target/achieved tokens, distances, session boundaries
```

Facts are found in a prompt, in retrieved evidence, or in an answer by their
**markers** — distinctive substrings such as `Project Cobalt` and `eu-west-1`.
All of a fact's markers must appear, and they must co-occur in one evidence item,
so "the project" and "the region" in two different memories never count as
evidence for a fact that asserts them together. The runtime mints its own memory
ids during a replay, so markers are the only stable way to score retrieval.

## Categories (plan §11)

| Key | Plan | What it probes |
| --- | --- | --- |
| `single_hop` | A | one stored fact answers the question directly; the fact is also repeated once, to test deduplication |
| `multi_hop` | B | the answer combines two facts stated far apart |
| `temporal` | C | an older value was replaced; the newer one is correct and the older is forbidden |
| `conflict` | D | the user explicitly corrects an earlier statement (`Correction: …`) |
| `distractor` | E | the fact is buried among 3× as many irrelevant durable facts |
| `cross_session` | F | a fact from an earlier session is needed later |
| `abstention` | G | nothing in the conversation answers the question; declining is correct |

Two properties are enforced by tests rather than assumed, because a benchmark
that violates them produces numbers that look meaningful and are not:

1. **every planted fact is storable by the application's own memory policy** — a
   fact the policy rejects could never be retrieved by any mode;
2. **filler is never storable** — filler turns are questions, greetings, and
   acknowledgements, so memory fills with facts instead of noise.

## Length ladder

| Tier | Target lengths (estimated tokens) | Typical use |
| --- | --- | --- |
| `smoke` | 800 | the committed dataset: reviewable in a diff, runs in seconds |
| `quick` | 2,000 / 4,000 | sanity runs before a real experiment |
| `standard` | 5,000 / 10,000 / 20,000 / 40,000 | the comparison a paper would report |
| `research` | 5,000 … 120,000 (plan §14) | the full ladder; expensive to replay, cheap to generate |

Lengths are estimated with the repository's dependency-free counter
(`ceil(chars / 4)`). Every run records `token_counter`, so runs measured with
different counters are never compared silently.

## Generating

```bash
# the committed tier, byte-for-byte reproducible
python benchmarks/context_rot/generation.py

# a real experiment tier, written outside the repository by default
python benchmarks/context_rot/generation.py \
  --tier standard --variants 2 --seed 20260913 \
  --output benchmarks/context_rot/generated/standard.jsonl \
  --manifest benchmarks/context_rot/generated/standard.manifest.json
```

`dataset.jsonl` is generated from `spec.py` (content) and `generation.py`
(structure) and pinned by `MANIFEST.json`, which records the generator version,
tier, seed, task count, and the SHA-256 of the dataset. Tier manifests written
into `generated/` are committed for the same reason — a run's `dataset_sha256`
can then be tied to the parameters that produced it — while the datasets
themselves stay local (a `research` tier is ~15 MB, a `top-rung` slice ~7 MB). Regenerating with the
manifest's parameters reproduces the file exactly; `tests/evaluation/
test_context_rot_dataset.py` asserts both.

Do not edit `dataset.jsonl` by hand. Change `spec.py`, bump
`GENERATOR_VERSION`, regenerate, and expect the manifest hash to change.

## Scoring

Retrieval scoring is model-free: given the evidence a mode selected and the
prompt it produced, it reports Recall@K, Precision@K, whether every required
fact reached the prompt, and whether a forbidden (stale) fact was present.

Answer scoring needs an answer string but never a model call — supply one with
`--answers` (a JSONL of `{task_id, mode?, answer}`), which is how a real model
run, or a deterministic mock, is graded:

```bash
python -m evaluation.run --mode brainos \
  --dataset benchmarks/context_rot/dataset.jsonl \
  --answers results/model_answers.jsonl \
  --output results/run.json

python -m evaluation.compare results/full_context.json results/brainos.json
```

Verdicts: `correct`, `abstained`, `wrong_abstention`, `stale_answer`,
`incorrect`, `ungraded`. Failures carry a label from the plan's error taxonomy
(`missed_memory`, `stale_memory`, `conflicting_memory`, `hallucination`,
`wrong_abstention`, `wrong_answer`) so Phase 12's error analysis starts from the
same record the aggregate metrics do.

Phase 8 adds faithfulness (was the answer grounded in the prompt?),
conflict-resolution accuracy (temporal and conflict categories only), token
savings, quality-adjusted efficiency, per-length curves, and a degradation/AUC
block. A single length — this smoke tier — cannot support a curve; the AUC
fields are `null` rather than a silent zero. Recall@K, evidence-in-prompt,
accuracy, and faithfulness are four numbers and must all be reported: they
disagree on multi-hop.

## What this benchmark is not

- It is not a natural-language dataset: the filler is template-generated, so it
  measures context management, not conversational realism.
- It is not a leaderboard. The committed tier is a smoke dataset (7 tasks, one
  length); a claim needs `standard` or `research`, multiple trials, and a model
  in the loop.
- It is not fair to models that answer from world knowledge: project names and
  values are synthetic, so an answer can only come from the supplied context.
