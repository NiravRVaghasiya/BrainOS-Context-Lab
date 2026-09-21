# Limitations and non-goals

This project is an experimental evaluation platform, not a claim that BrainOS provides human-like cognition or eliminates context rot.

## Initial limitations

- The first release supports OpenAI and generic OpenAI-compatible endpoints only.
- BrainOS remains an upstream dependency whose exact revision and runtime API must be validated before integration claims are made.
- Initial memory extraction is conservative and may miss useful information.
  Phase 3 repaired the worst of it (declarative verbs such as "happen" and
  "requires" were rejected while small talk containing "are" was stored), but
  extraction is still pattern-based, English-only, and will mislabel content.
- Relevance, conflict, and recency handling use lightweight, dependency-free
  heuristics: a small suffix stemmer, IDF-weighted lexical overlap, and
  `subject → value` claim patterns. The stemmer normalizes proper nouns ending
  in "s" ("Atlas" → "atla"); this is consistent on both sides of every
  comparison so matching is unaffected, but debug output shows stems.
- **Abstention does not work yet.** For a question whose answer is absent, the
  relevance floor still admits memories (measured: best relevance `0.344`
  against a `0.12` floor, 4 memories selected instead of none). The threshold
  was deliberately not tuned on a single synthetic conversation; calibration
  belongs to the benchmark phases where abstention accuracy is scored.
- Contradiction detection depends on BrainOS `contradictions()`, which the
  pinned runtime populates only for subject-versioned memories
  (`remember(subject=...)`). Conversations observed through `observe()` rely on
  the application heuristic, which resolves a pair only when the newer claim
  carries an explicit correction marker and otherwise keeps both, flagged
  `contested`.
- Measured token reduction depends mostly on the recent-history window, not on
  memory selection: with a window larger than the conversation, BrainOS mode
  costs more than full context. Any reduction figure is meaningless without the
  budget configuration that produced it.
- Token estimates may begin as approximate counts until provider/model-specific
  tokenizers are integrated. `ContextStats.token_counter` records which counter
  produced a figure; runs using different counters are not comparable.
- Synthetic context-rot tasks may not represent real user conversations. Phase 7's
  filler is template-generated, so the benchmark measures context management, not
  conversational realism; the fact vocabulary is synthetic, which is what keeps
  answers grounded in the supplied context but also means the benchmark says
  nothing about world knowledge.
- The benchmark is only as valid as its two enforced properties: every planted
  fact is storable by the application's memory policy, and no filler turn is. The
  tests assert both, but a change to the memory policy can invalidate a generated
  dataset without failing anything else — regenerate and re-run the validity
  tests when the policy changes.
- Retrieval and answer scoring are lexical, not semantic: marker co-occurrence and
  string-normalized answer matching. A correct answer phrased with an unlisted
  synonym is graded incorrect, and a fact restated with different words is not
  detected as evidence. Every number the benchmark reports inherits that limit.
- The committed dataset is a smoke tier (7 tasks, one length, one seed). It
  validates plumbing; it cannot support a claim. Robustness curves need the
  `standard`/`research` tiers, multiple variants, and a model in the loop.
  Phase 8 records this in the artifact: `degradation.area_under_degradation_curve`
  is JSON `null` with an explicit note when only one length is present.
- Faithfulness, Recall@K, evidence-in-prompt, and answer accuracy are four
  different numbers. On the smoke run they disagree on multi-hop (recall 1.0,
  evidence-in-prompt 0.0, scripted accuracy 1.0, faithfulness 0.0 for that
  task). Collapsing them hides the failure the benchmark exists to expose.
- Quality-adjusted efficiency is undefined as a cross-mode comparison while
  answers are scripted. Latency is omitted until a generation path exists;
  replay wall-clock is not a generation metric.
- Measured benchmark numbers came from the dependency-free token estimator
  (`estimate_tokens`); runs using a different counter are not comparable, and
  `stats.token_counter` is what records which one produced a figure.
- Provider behavior, model changes, and stochastic generation can affect results.
- Retention of browser artifacts runs on **use**, not on a timer: the sweep
  happens when a run starts and when an operator runs `python -m app.retention`.
  A Space with no traffic and no scheduled job keeps every abandoned session's
  files, and a `--dry-run` sweep is the honest way to see what would go.
- Path anchoring fixes the working-directory bug for a source checkout (the
  documented deployment, including an editable install on a Space), but an
  installed distribution that does not contain `benchmarks/` still has no
  committed dataset: the anchored path is only used when it exists, and the error
  then names the plan's relative path. The SQLite location remains
  operator-controlled through `BRAINOS_LAB_DB`.
- Session-local persistence is not a multi-user account system.
- Evaluation results are only meaningful when baselines use the same models, prompts, and scoring rules.

## Explicitly out of scope for v1

- user accounts and billing,
- fine-tuning in the Space,
- multi-agent systems,
- complex vector databases,
- custom model hosting,
- mobile applications and browser extensions,
- permanent storage of user API keys, and
- broad claims about solving context rot.
