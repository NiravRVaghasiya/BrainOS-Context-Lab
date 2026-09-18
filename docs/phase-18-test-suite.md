# Phase 18 — Test Suite and Runtime Hardening

**Plan reference:** §24 (Phase 18 — Test Suite)
**Status:** complete in this turn
**Validation:** **1183 passed** · `ruff check .` clean · pinned BrainOS live suites run

Phase 18 closes the two runtime-boundary gaps the Phase 17 hand-off left open:
a real provider timeout test and a multi-threaded persistence stress test. It
also turns a documented cost-control promise into an enforced boundary. Before
this phase, the chat controller built a prompt, called the provider, charged
the request, and only then recorded that the estimated prompt exceeded
`max_input_tokens`. That contradicted the plan's "refused before the provider is
called" rule and could spend on a request the user had explicitly capped.

## Goals

The implementation plan's test-suite phase requires four layers:

* exhaustive unit coverage for adapters, providers, context construction,
  token budgeting, filtering, isolation, and conflicts;
* an integration path from user input through observation, recall, context,
  provider, response, and memory update;
* deterministic mock-model evaluation checks; and
* security tests for credential freedom, session isolation, prompt-injection
  containment, and diagnostics.

Those layers already existed from Phases 0–17. This phase keeps them intact and
adds the missing operational proofs rather than replacing the deterministic
harness with a network-dependent test suite.

## What changed

### Chat cost enforcement

`ConversationService.handle_user_message()` now accepts the chat-side input and
output ceilings. Once the final prompt has been built:

1. an enabled `max_input_tokens` ceiling is checked before `_generate()`;
2. an oversized prompt returns a credential-blind error and
   `usage.input_limit_exceeded = true`;
3. the controller records a refusal but does not charge a provider request or
   input tokens; and
4. the effective output ceiling is the tightest positive value from the
   provider connection and the session's `ChatLimits`, sent as the provider's
   `max_tokens` request parameter.

A provider that reports more completion tokens than that requested ceiling is
still represented honestly in the turn usage. The application cannot retract a
response a provider has already produced, but it records the overage rather
than silently treating it as compliant.

### Deterministic timeout fake

`tests/fakes.py` adds `SlowProvider`, a normal provider double that exposes start
and finish events and sleeps for a finite interval. The new test in
`tests/integration/test_phase18_hardening.py` runs the real `UIController` and
`ConversationService` path with a 30 ms timeout and a 200 ms provider delay. It
asserts that:

* the callback returns before the provider's sleep completes;
* the turn is marked `timed_out` and `failed`;
* the effective provider `max_tokens` is forwarded;
* usage records one request and one timeout without exposing the session key;
* the late provider result is not appended as an assistant message.

A companion deterministic provider ignores `max_tokens` and reports a 99-token
completion under a three-token ceiling. The response remains visible, but the
turn records `output_limit_exceeded` and the budget charges the reported usage
rather than hiding the gateway's overage.

This is deliberately not a mock assertion that a number was forwarded. It
exercises `_generate()`'s future boundary and the controller's usage accounting.

### SQLite worker stress test

The same new integration module starts eight workers against one temporary
WAL-backed database. Every worker concurrently:

* appends 20 private transcript messages;
* upserts a session-local memory mirror;
* upserts an evaluation run; and
* reads its own rows while other workers are writing.

The final assertions check exact message counts, one memory per session, final
evaluation metrics, exact session/conversation filtering, WAL mode, and
`secure_delete`. No worker may see a neighbouring session's transcript or
memory. The test is intentionally storage-level: the controller's process-local
`SessionManager` is not thread-safe by design, while SQLite's per-operation
connections and row filters are the deployment guarantee Phase 14 promised.

## Contracts carried forward

1. **Input-limit refusal is pre-provider.** A refusal must not create a lazy
   provider instance, increment `requests`, or add prompt tokens to the budget.
   It remains a user-visible status and is not a security finding.
2. **Output caps are defense in depth.** The provider request carries the
   effective `max_tokens`; reported usage remains authoritative if a gateway
   ignores it.
3. **Timeouts wrap provider work only.** BrainOS observation, recall, and local
   context construction stay outside the wall-clock provider timeout.
4. **A late timed-out response is discarded.** The worker may finish because
   Python cannot safely kill an arbitrary provider thread, but its response must
   not mutate the conversation after the service has returned.
5. **Storage isolation is row-level and exact.** Every operation keeps its own
   SQLite connection and filters by both session and conversation identifiers.
   The stress test must remain green if Gradio's queue permits concurrent
   callbacks.
6. **Do not weaken Phase 17 pins.** Credential scans, quarantine, ungraded
   `—` metrics, the Evaluation tab's callback contract, and the deterministic
   pipeline remain part of the full-suite gate.

## Validation

With the optional runtime dependencies installed from `requirements.txt`:

```bash
.venv/bin/pytest -q
# 1183 passed

.venv/bin/ruff check .
# All checks passed!
```

The suite includes the pinned BrainOS integration tests, the deterministic
pipeline/evaluation/security tests, the Gradio UI contract tests, the new
slow-provider timeout proof, and the concurrent SQLite stress test. No external
provider API key is used.

## Findings and constraints for later phases

* The product now has a tested timeout boundary, but an abandoned browser
  session can still leave `results/ui/<session>/` artifacts until Phase 19 adds
  retention cleanup or moves ephemeral runs to a temporary root.
* The SQLite stress test proves a single process and one database file. A
  multi-process deployment still needs the deployment's worker/process model and
  filesystem semantics reviewed before increasing queue or worker counts.
* Phase 19 should polish the MVP and choose the public `EvaluationPolicy`; the
  defensible default remains `quick`, generation disabled, `max_tasks=20`, and
  `max_requests=60` until a deployment operator deliberately changes it.
* The benchmark numbers remain deterministic plumbing results, not evidence for
  the research claim. A real provider run at the research tier is still a
  Phase 20/research-release responsibility.
