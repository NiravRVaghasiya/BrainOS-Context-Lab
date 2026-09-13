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
- Synthetic context-rot tasks may not represent real user conversations.
- Provider behavior, model changes, and stochastic generation can affect results.
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
