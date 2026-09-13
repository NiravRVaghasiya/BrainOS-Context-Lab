# Phase 6 log — Baseline modes

**Status:** complete (branch `arena/01a09c0c-brainos-context-lab`)
**Plan section:** Phase 6 — Baseline modes
**Depends on:** Phase 5 (conversation persistence) and every phase before it.

## Objective

From the plan: implement exactly five ways of turning a long conversation into a
prompt — Full Context, Sliding Window, Conventional RAG, BrainOS, and BrainOS +
RAG — so that "does BrainOS help?" is answered against real alternatives instead
of a straw man. Only the context-management strategy may change between modes;
model, prompt, task wording, and accounting stay identical.

## Starting state

- `src/app/state.py` exposed a two-value seam: `MEMORY_MODES = ("brainos",
  "no_memory")` and `ContextSettings.uses_memory()`. `ConversationService._recall()`
  already skipped recall for memory-less modes, and the UI had a two-option
  radio.
- `ContextBudget` had no notion of retrieved evidence other than BrainOS memory.
- `evaluation/runner.py` was a shell that raised `NotImplementedError`, and
  `evaluation/run.py` exited with *"Implement the baseline mode adapters before
  running evaluations."* Its `--mode` choices already listed exactly the five
  plan modes.
- The Phase 3 finding was still an open constraint: with a history window larger
  than the conversation, BrainOS mode degenerates into full context *plus*
  overhead (0.0% reduction, 95 tokens worse than the baseline).

## Work completed

### 1. Mode registry (`src/baselines/modes.py`, new)

One frozen `BaselineMode` per strategy, carrying everything that defines it:
which evidence sources it uses, its history window, and its per-section budgets.

| Mode | Selector | Memory | RAG | Recent turns | History budget | Memory budget | Chunk budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | `full_context` | — | — | unlimited | whole ceiling | 0 | 0 |
| B | `sliding_window` | — | — | 8 | whole ceiling | 0 | 0 |
| C | `rag` | — | yes | 2 | 256 | 0 | 1024 |
| D | `brainos` | yes | — | 2 | 256 | 1024 | 0 |
| E | `brainos_rag` | yes | yes | 2 | 256 | 512 | 512 |

Two design rules, both derived from the Phase 3 measurement rather than from
taste:

- **Every mode fixes its own history window.** `recent_turn_budget` is `None`
  for A and B, meaning "derived from the session's `max_tokens` ceiling" —
  "replay everything" must never be silently clipped by a window the user set
  for a different mode.
- **Modes C, D, and E share one evidence allowance** (`EVIDENCE_BUDGET = 1024`).
  They differ in *what selects* the evidence, so they get the same volume; Mode
  E splits the allowance between its two sources instead of doubling it. Any
  accuracy difference is then attributable to selection, not to budget.

BrainOS still **observes** in every mode. Observation is a side process, not
context construction; keeping the runtime warm means a user can switch modes
mid-conversation without losing memory the new mode would have used. What a
mode controls is what reaches the prompt.

`no_memory`, the Phase 4/5 selector value, is accepted as an alias for Mode B
rather than rejected, so a saved configuration keeps working. Any *other*
unknown value raises: a typo in an evaluation configuration must not silently
produce results labelled with a strategy that never ran.

### 2. Lexical retrieval baseline (`src/baselines/rag.py`, new)

The "conventional RAG" side of the comparison: `chunk_history()` splits the
transcript at sentence boundaries into ~400-character chunks, and
`retrieve_chunks()` ranks them by IDF-weighted lexical overlap, reusing the
Phase 3 machinery (`content_tokens`, `document_frequencies`, `lexical_score`) so
Modes C and D share one notion of lexical relevance and differ in *what* is
scored.

Three deliberate properties:

- **No BrainOS and no vector store.** Importing the runtime here would make the
  baseline a re-skin of the system under test; embeddings are ruled out by the
  plan's "what not to build" list and would make runs irreproducible.
- **Deterministic.** Ties break on transcript position then chunk id, so the same
  conversation and question always produce the same top-k.
- **No abstention.** The retriever returns `top_k` chunks whenever the
  conversation has content, even for a question nothing answers. That is what
  ordinary RAG does, and it is one of the effects the benchmark exists to
  expose rather than hide.

### 3. Context builder: a second evidence source (`src/brain/context_builder.py`)

`build_context()` gained `retrieved_chunks`, rendered into their own
`<retrieved_history>` block under a new `ContextBudget.chunk_budget` (default
`0`, so every pre-Phase-6 caller builds exactly the prompt it built before).

- Each chunk is labelled with its provenance — `[user #12]` / `[assistant #13]`.
  A chunk the *assistant* said is the model's own earlier claim; presenting it
  as user-asserted evidence would be a quiet correctness bug.
- A chunk whose text duplicates a message the history window already carries is
  dropped (`duplicate_history`) rather than paid for twice.
- Eviction order under `max_tokens` is now **question (never) → oldest history →
  lowest-ranked chunks → lowest-ranked memories → system prompt**. Chunks go
  before memories because memory lines are the denser evidence and BrainOS's
  ranking is the variable under test.
- `ContextStats` gained seven fields (`selected_chunk_count`,
  `retrieved_chunk_tokens`, `chunk_block_tokens`, `candidate_chunk_count`,
  `candidate_chunk_tokens`, `dropped_chunks_for_budget`, `evidence_tokens`).
  Every mode reports the same field set; a mode that uses no chunks reports
  zeros, so a cross-mode table needs no special cases.
- `full_context_reference_tokens` deliberately still prices *only* system +
  full history + question: Mode A has no evidence blocks, so including them
  would make the reduction metric compare a mode against something Mode A never
  was.

The builder duck-types chunks through a `RetrievedChunk` `Protocol` rather than
importing `baselines`, so the dependency stays one-directional and a
BrainOS-free evaluation can import either side alone.

### 4. Application wiring

| File | Change |
| --- | --- |
| `src/app/state.py` | Five-mode vocabulary; new `chunk_budget` / `rag_top_k` fields; `mode_profile()`, `uses_rag()`, `with_mode_defaults()`. **The defaults are now Mode D's canonical profile**, not neutral values — a fresh session must not start in the degenerate configuration Phase 3 measured. |
| `src/app/service.py` | `_retrieve_chunks()` (session-key guarded before the builder sees it), chunks passed to `build_context`, `ConversationTurn.retrieved_chunks`, chunk counts in diagnostics, and a new `record_assistant_message()` so a benchmark's scripted turns enter the transcript and memory the way generated ones do. |
| `src/app/controller.py` | `apply_mode()`; `update_context()` resolves and validates the mode and, **when the mode changes, the mode's budgets win over anything else submitted in the same call** — otherwise the sidebar (which submits every slider with the mode) would carry the old window into the new mode. `context_payload()` now reports the mode profile; `TurnView` gained `chunk_rows`. |
| `src/app/panels.py` | `CHUNK_COLUMNS` / `chunk_rows()`, a chunk line and an evidence-total line in the context summary, and `chunks` in the token breakdown. |
| `src/app/ui.py` | Five-mode dropdown with a live description, a `mode.change` handler that pushes the applied budgets back into the sliders, plus "recent turns kept" and "retrieved-chunk budget" controls so the window the experiment runs with is visible and editable. |

### 5. Evaluation seam (`src/evaluation/modes.py`, new)

`replay_task()`, `compare_modes()`, and `task_evaluator()` run a benchmark task
through any mode using the application's own service and builder. Each
(task, mode) pair gets a fresh session and a fresh runtime, no provider, and no
storage backend. `evaluation/run.py` now passes `task_evaluator()` to the
runner, so the CLI works instead of exiting with the scaffold message:

```bash
python -m evaluation.run --mode brainos --output results/run.json
```

Two things the seam deliberately does not do: **no model calls** (answer
accuracy belongs to Phase 8) and **no scoring**. The `expected_answer_in_prompt`
flag says the evidence was *available* to the model, not that the model used it.

## Measured behaviour (live pinned BrainOS, 28-message transcript)

Two facts (turn 0 and turn 26) separated by 24 filler turns, asked at turn 28.
Dependency-free token estimator; no model in the loop; no provider API key.

| Mode | Sent | Full-context ref | Reduction | Memories | Chunks | History msgs | Evidence tokens | Fact in prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A `full_context` | 510 | 510 | 0.0% | 0 | 0 | 28 | 0 | yes |
| B `sliding_window` | 189 | 510 | 62.9% | 0 | 0 | 8 | 0 | **no** |
| C `rag` | 256 | 510 | 49.8% | 0 | 6 | 2 | 159 | yes |
| D `brainos` | 182 | 510 | **64.3%** | 2 | 0 | 2 | 85 | yes |
| E `brainos_rag` | 345 | 510 | 32.4% | 2 | 6 | 2 | 244 | yes |

**These are integration-validation observations, not research results.** One
synthetic conversation, an estimated token counter, no model, single trial.

What the table shows, and what it does not:

- Mode D cost **fewer tokens than Mode B and still carried the fact B lost**.
  That is the comparison the project exists to make, and Phase 6 makes it
  expressible for the first time.
- Mode C recovered the fact by reaching back past the window, at 49.8%
  reduction — cheaper than A, dearer than D.
- Mode E is *not* free: combining the sources cost 345 tokens, 1.9× Mode D,
  because both blocks fill their budgets. The hybrid's value has to be earned
  on quality, not assumed.
- All three retrieval modes stayed inside the shared 1024-token evidence
  allowance; how much of it each used (159 / 85 / 244) is the finding, not a
  budget violation.

## Bugs found and fixed while testing

1. **`_apply_mode` returned the wrong session.** The callback resolved the
   session id from its *input* after acting, so when the browser's state was
   empty it minted a second session and handed the browser an id whose mode had
   never changed — the switch would have been silently lost on the next turn.
   Fixed by resolving the session before acting, matching `_update_context`.
2. **The builder trusted the retriever to guard chunk text.** A hand-built chunk
   containing `</retrieved_history>` broke out of the block. The builder now
   guards chunk text itself (idempotently, in `_as_chunk` and again in
   `_render_chunk`), so no caller has to remember.
3. **`_DELIMITER_RE` did not cover `retrieved_history`.** The Phase 3 guard
   stripped `<retrieved_memory>` breakouts but not the new delimiter. Extended.

## Live HTTP validation (running Gradio server + pinned BrainOS + real SQLite)

Verified over HTTP against `python app.py`:

- `/apply_mode` writes exactly the budgets above and returns them to the
  sliders — Mode A came back with `recent turns = None` (blank = whole
  conversation) and `window = 4096` (the ceiling), Mode E with `512 / 512`.
- `/start_session` returns the active mode description; the chat status and the
  Context summary name the mode that ran (`mode \`Mode C — Lexical RAG\``).
- The "Retrieved transcript chunks (Modes C/E)" panel is registered with its
  headers, and `context_statistics` carries the chunk fields.
- `/export_session` recorded `mode=rag`, `letter=C`, `uses_rag=True`,
  `chunk_budget=1024`, `evidence_budget=1024`, with no `api_key` field and no
  credential anywhere in the file.
- The session key appeared in no panel, no response, and no raw database byte
  (4 message rows and 4 memory rows written; both checked as bytes).

**Limitation found, not introduced here:** Gradio does not expose `gr.State` to
API clients, so `/chat` accepts only `message` and an HTTP client gets a fresh
session per request. The browser path is unaffected (state rides in
`gr.State`), and `UIController.ensure_session` documents this self-heal. The
consequence for validation is that the *multi-turn* cross-mode comparison cannot
be driven over HTTP; it is covered instead by
`tests/integration/test_baseline_modes_live.py` against the pinned runtime.
Generation was also not exercised over HTTP — the `providers` extra was not
installed and no real API key was used; generation is covered by the
fake-provider tests.

## Decisions carried forward

- **Audit reasons are namespaced by source.** Chunk drops use
  `duplicate_history`, `chunk_budget`, and `chunk_ceiling`, so a Precision@K
  computation over `report.dropped` can still isolate memory. Update the
  Phase 12 taxonomy mapping accordingly.
- **The evidence allowance is a contract.** Any new retrieval mode must fit
  inside `EVIDENCE_BUDGET` or the comparison stops being about selection.
- **Mode C's lack of abstention is a difference to measure, not a gap to fix.**
  BrainOS mode can return nothing; the lexical baseline cannot. Phase 8 scores
  abstention accuracy, and this is where the two will separate.
- **Raw history is replayed verbatim.** The credential guard covers recalled
  memory, retrieved chunks, diagnostics, and persisted rows — not the
  transcript inside the recent window, which Mode A must send as written. A key
  a user pasted into a conversation therefore still reaches their own provider
  in Mode A/B history. That is pre-existing Phase 3 behaviour, restated here
  because Phase 6 added a second route (retrieval) that *is* guarded.

## Next

Phase 7 — Context-rot benchmark. The mode harness, the per-mode budgets, and
the runner seam are in place; what is missing is the dataset (long
conversations with distributed, contradictory, and temporally replaced facts)
and the scoring that needs a model.
