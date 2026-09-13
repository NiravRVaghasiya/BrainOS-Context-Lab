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

User content may contain malicious text such as “ignore system instructions.” Retrieved memories are untrusted data. The context builder must wrap them in explicit `<retrieved_memory>` delimiters and instruct the model not to follow instructions contained within them.

## Data lifecycle

The MVP provides clear-conversation, clear-memory, delete-session, and export-session operations. Permanent user accounts and long-lived storage are explicitly out of scope for the first version.

## Cost and abuse controls

The service should enforce maximum input tokens, output tokens, turns per session, benchmark examples, benchmark tokens, and request timeouts. Benchmark runs should expose Quick, Standard, and Research tiers.

## Diagnostics

Diagnostics shown in the browser must be sanitized. Cognitive traces may show stage names and aggregate counts, but not API keys, authorization headers, or unrelated users' data.

## Testing requirements

Security tests must verify secret redaction, session isolation, memory instruction boundaries, and absence of credentials from exported evaluation artifacts.
