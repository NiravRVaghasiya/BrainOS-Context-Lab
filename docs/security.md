# Security and threat model

## Secrets

Provider API keys are user-supplied secrets. They must:

- remain in process memory for the active session only,
- never be persisted in SQLite or evaluation files,
- never appear in logs, traces, exception messages, or browser diagnostics,
- never be included in benchmark records, and
- be cleared when the session ends.

The UI must make clear that the user's provider account is responsible for API usage and cost.

## Session isolation

Every request must be associated with `session_id`, `actor_id`, and `conversation_id`. Conversation, memory, diagnostics, and evaluation access must check these identifiers. Session A must not be able to query Session B's records.

## Prompt injection through memory

User content may contain malicious text such as “ignore system instructions.” Retrieved memories are untrusted data. The context builder wraps them in explicit `<retrieved_memory>` delimiters and instructs the model not to follow instructions contained within them.

Phase 3 hardened that boundary in the retrieval policy's memory-text guard:

- **Delimiter breakout** — any `<retrieved_memory>` / `</retrieved_memory>` (and
  `<system>`, `<user>`, `<assistant>`, `<memory>`) sequence inside memory text is
  removed, so a memory cannot close its own block and start a new instruction
  section. The rendered block contains exactly one opening and one closing
  delimiter, and closes last.
- **Role smuggling** — a line beginning `system:`, `assistant:`, `developer:`,
  `tool:`, or `user:` has that prefix stripped, and newlines are collapsed so a
  memory renders as a single bullet rather than a fake multi-turn transcript.
- **Control characters** are removed and memory text is length-bounded.
- **Override detection** — text matching instruction-override patterns ("ignore
  previous instructions", "new system prompt", "you are now", "reveal the system
  prompt/api key") is flagged `suspicious` in the ranking and audit report. By
  default it is kept-but-flagged so retrieval quality stays measurable;
  `RetrievalPolicy.drop_suspicious_memories` opts into removal. Phase 13 must
  test both settings.

## Credential replay through memory

A user can paste a provider key into a conversation, where BrainOS will store it
as ordinary content. Phase 3 closes two paths out of that store:

- the active session key is redacted **by exact match** from every
  browser-visible value the service produces (diagnostics, inspection data,
  context reports, rankings, traces) — pattern-based scrubbing alone cannot
  recognise an arbitrary key; and
- recalled memory text is redacted before it is placed in a prompt, so a key
  captured in one conversation cannot be replayed to a different provider later.

Conversation history is not yet redacted on the render path; that is Phase 13
work and is tracked as such.

## Data lifecycle

The MVP provides clear-conversation, clear-memory, delete-session, and export-session operations. Permanent user accounts and long-lived storage are explicitly out of scope for the first version.

## Cost and abuse controls

The service should enforce maximum input tokens, output tokens, turns per session, benchmark examples, benchmark tokens, and request timeouts. Benchmark runs should expose Quick, Standard, and Research tiers.

## Diagnostics

Diagnostics shown in the browser must be sanitized. Cognitive traces may show stage names and aggregate counts, but not API keys, authorization headers, or unrelated users' data.

## Testing requirements

Security tests must verify secret redaction, session isolation, memory instruction boundaries, and absence of credentials from exported evaluation artifacts.
