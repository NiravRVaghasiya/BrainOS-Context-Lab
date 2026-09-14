# Security and threat model

The reasoning behind these controls — assets, actors, routes, residual risk — is
in [`threat-model.md`](threat-model.md). The Phase 13 implementation log is
[`phase-13-security.md`](phase-13-security.md).

## Secrets

Provider API keys are user-supplied secrets. They must:

- remain in process memory for the active session only,
- never be persisted in SQLite or evaluation files,
- never appear in logs, traces, exception messages, or browser diagnostics,
- never be included in benchmark records, and
- be cleared when the session ends.

The UI must make clear that the user's provider account is responsible for API
usage and cost.

Enforcement is exact-match redaction against the active session credential on
every route that can carry text out of the process (memory, chunks, replayed
history, the current message, persistence, diagnostics), plus field-name
stripping (`api_key`, `authorization`, `token`, …) on anything serialized.
Pattern scrubbing alone cannot recognise an arbitrary key, so it is a second
layer on written artifacts, not the primary control.

## Session isolation

Every request must be associated with `session_id`, `actor_id`, and
`conversation_id`. Conversation, memory, diagnostics, and evaluation access must
check these identifiers. Session A must not be able to query Session B's
records. One BrainOS runtime is constructed per session and never shared; the
browser holds only the non-secret session identifier.

## Prompt injection through retrieved content

User content may contain malicious text such as "ignore system instructions."
Retrieved memories and retrieved transcript chunks are untrusted data. The
context builder wraps them in explicit `<retrieved_memory>` /
`<retrieved_history>` delimiters and instructs the model not to follow
instructions contained within them.

Phase 3 hardened that boundary; Phase 13 moved the rules into one guard
(`security.guard`) shared by every route and split its output in two, because
the two halves mean different things:

- **Structural neutralization — the primary control.** It does not depend on
  recognising an attack:
  - *Delimiter / template breakout*: any `<retrieved_memory>`,
    `<retrieved_history>`, `<system>`, `<user>`, `<assistant>`, `<memory>`
    sequence, any foreign chat-template marker (`<|im_start|>`, `<<SYS>>`,
    `<|endoftext|>`, …), and any Markdown instruction-block header
    (`### System`) are removed from retrieved text, so a memory cannot close its
    own block and open a new instruction section. The rendered block contains
    exactly one opening and one closing delimiter, and closes last.
  - *Role smuggling*: a line beginning `system:`, `assistant:`, `developer:`,
    `tool:`, or `function:` has that prefix stripped, and newlines are collapsed
    so a memory renders as a single bullet rather than a fake multi-turn
    transcript.
  - *Invisible characters* (zero-width, bidi overrides, Unicode tag characters)
    and *control characters* are stripped — they exist to hide a payload from
    readers and from regexes alike.
  - Text is length-bounded on every route.
- **Injection detection — a tripwire and a measurement instrument.** Seven
  families are matched: `instruction_override`, `role_assumption`,
  `prompt_exfiltration`, `credential_exfiltration`, and `policy_bypass` carry
  *intent*; `delimiter_breakout` and `role_smuggling` are *structural*. Only the
  intent families make a candidate `suspicious`, so a memory that merely
  contains a stray delimiter is cleaned and kept rather than quarantined.
  Detection is English-only and pattern-based: a clean report means nothing
  matched, not that the content was safe.

By default a suspicious memory is **kept and flagged**, so retrieval quality
stays measurable and the user can see what was caught;
`RetrievalPolicy.drop_suspicious_memories` opts into **quarantine**, which
removes it and records the drop reason `suspicious`. Both settings are tested,
including against the pinned runtime
(`tests/integration/test_security_live.py`).

## Replayed history

Two controls, both counted in the security report:

- **Credentials.** The active session key is removed by exact match from every
  history message *and* from the current message before either reaches a prompt.
  This closed the last open route: a key pasted into turn 1 used to be replayed
  to the provider on every later turn, including after the user pointed the
  session at a different endpoint. Only the live credential is matched, so
  transcript fidelity is otherwise untouched and a keyless evaluation replay is
  byte-identical to before.
- **Roles.** A history message may only be `user` or `assistant`. Anything else
  — `system`, `tool`, `developer`, `function` — is sent as `user`, because only
  the application writes the system prompt. Those coercions are reported as
  `history_role_downgraded`; an unrecognised role is coerced without being
  counted as an escalation attempt.

History text is deliberately **not** injection-rewritten: these are real chat
turns with their own role, not text embedded inside a delimited block, and
rewriting a user's own words would corrupt the conversation without adding a
control the message boundaries do not already provide. See the residual-risk
section of the threat model.

## Findings and the Security panel

Every guard action is recorded in one vocabulary
(`category` / `action` / `route` / `stage` / `families`), accumulated per
session in a thread-safe ledger, and rendered in the UI's **Security** tab as a
markdown summary, a bounded findings table, and the raw count block.

Three rules the reporting layer keeps:

1. **A finding never carries the payload it caught, unbounded.** Previews are
   clipped to 160 characters, invisible characters are stripped, and
   credential-shaped substrings are masked (`sk-…[30 chars]`) — a finding raised
   against an injected key must not republish that key in a panel or an
   artifact. Credentials are described by shape and length, never reproduced.
2. **Counts are cumulative, detail is bounded.** Category counters always
   increase; the recorded list is capped at 50 entries, so a long session cannot
   grow an unbounded report and cannot lose a count either.
3. **A quarantine is a security finding, not an answer failure.** The `suspicious`
   drop reason maps to no label in the nine-type error taxonomy
   (`DROP_REASON_LABELS["suspicious"] == ""`), which keeps
   `labels_vs_scorer.unexpected` at 0.

Run, mode, and experiment artifacts each carry a `security` block, so a reader
can see whether anything was quarantined or redacted while a published number
was produced.

## Artifact scanning

`python -m security.scan` is the operational half of "no credential in any
artifact": point it at whatever a run produced and it answers.

```bash
python -m security.scan results/ data/brainos_lab.sqlite3
python -m security.scan results/run.json --secrets-env OPENAI_API_KEY
```

Two detection layers, because they fail differently:

- **structure** — any JSON field whose *name* is a secret name and whose value
  is non-empty and not already `[redacted]`. A field name survives
  serialization even when a pattern does not, so this layer catches a design
  mistake rather than a pasted string.
- **content** — credential-shaped strings (`sk-…`, `sk-ant-…`, `ghp_…`, `AKIA…`,
  `AIza…`, `hf_…`, `xoxb-…`, `Bearer …`, `name=value`, PEM private-key blocks)
  in text or raw bytes, plus an exact match against any secret supplied through
  `--secrets-env` (an environment variable name, never a command-line literal).

The `name=value` pattern is the noisy one, so its value is filtered: a
placeholder word, a function call, or an all-lowercase identifier is not
reported. Precision matters more than recall on that single pattern, because a
report full of source-code false positives gets ignored — and the exact-match
layer covers any real credential whose shape is unremarkable.

Reports always state their scope (`files_scanned`, `bytes_scanned`,
`files_skipped`), so "clean" cannot be vacuous. Exit codes: `0` clean,
`1` findings, `2` nothing could be scanned. `.git` and `__pycache__` are skipped
as copies of content scanned elsewhere.

At Phase 13 the shipped surfaces — `src`, `docs`, `README.md`, `CONTEXT.md`,
`benchmarks`, and the implementation plan — scan clean: 77 files, 1,190,325
bytes, 0 findings. `tests/` is excluded from that assertion on purpose: it holds
deliberate fake credentials and injection payloads as fixtures.

## Data lifecycle

The MVP provides clear-conversation, clear-memory, delete-session, and
export-session operations. Permanent user accounts and long-lived storage are
explicitly out of scope for the first version.

Deletion removes bytes, not just rows: every store connection sets
`PRAGMA secure_delete=ON` (freed content is zeroed rather than left readable in
the file), and the controller runs `VACUUM` after each user-data deletion so
freed pages are returned and the file shrinks. Vacuum is best-effort by
contract — a failure is logged by exception type and never turns a successful
delete into an error the user sees. The security ledger survives
"clear conversation" deliberately: it is an audit trail of what the guards
caught, not conversation content. "End session" drops it with the session.

## Cost and abuse controls

The service should enforce maximum input tokens, output tokens, turns per
session, benchmark examples, benchmark tokens, and request timeouts. Benchmark
runs should expose Quick, Standard, and Research tiers. (Phase 15.) For a public
demo no shared provider key is placed in source or Space configuration; visitors
bring their own, held for the session only.

## Diagnostics

Diagnostics shown in the browser must be sanitized. Cognitive traces may show
stage names and aggregate counts, but not API keys, authorization headers, or
unrelated users' data.

## Testing requirements

Security tests must verify secret redaction, session isolation, memory
instruction boundaries, and absence of credentials from exported evaluation
artifacts. Phase 13 adds four more standing requirements:

- the injection probe corpus (14 attack probes, 9 benign probes that must stay
  untouched) is exercised against **every** route, not just the guard function;
- the guard must not damage the research: over the 522 messages and facts in
  `benchmarks/context_rot/dataset.jsonl` it flags nothing and rewrites nothing —
  the whole committed dataset, asserted by a test rather than quoted from a
  one-off measurement;
- deletion is asserted at the byte level against the database file, not at the
  row level;
- the controls are proven against the pinned `brainos_runtime`, not only against
  fakes (`tests/integration/test_security_live.py`).
