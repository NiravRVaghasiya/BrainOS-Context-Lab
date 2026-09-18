# Phase 15 — Cost Controls

> Plan reference: §15 ("Cost Controls"), §21 ("Cost Controls"), and the
> Phase 14 "next safe step" which named this phase as the remaining abuse
> mitigation before a public BYOK Space can ship.

Phase 15 layers per-request and per-session cost ceilings on top of the
chat path. The evaluation runner already had `RunLimits` / `RunBudget`
(Phase 9); the chat side had only `UILimits.max_turns = 200` and
`UILimits.max_message_chars = 8000`. A public Gradio Space without
per-turn token ceilings is the residual risk the Phase 14 log recorded;
this phase closes it.

## Goals

From the plan §15 / §21:

1. **max input tokens** per request — a prompt larger than the ceiling is
   refused *before* the provider is called, so it costs nothing.
2. **max output tokens** per request — the maximum completion length the
   session will accept.
3. **max turns/session** — already existed as `UILimits.max_turns`; now
   tracked through the same budget object as the token ceilings, with a
   user-visible "limit reached" state.
4. **max session tokens** — input + output tokens across the whole
   session; a cumulative spend cap.
5. **max session requests** — total provider calls per session.
6. **request timeout** — wall-clock ceiling on one provider call, so a
   slow model cannot hold a Gradio worker indefinitely.
7. **Show estimated usage where possible** — a Usage tab in the UI, a
   usage block in the session export, and per-turn usage on the turn view.
8. **Quick / Standard / Research presets** — already in
   `evaluation.limits.PRESETS`; the Evaluation tab now references them.

Constraints from the Phase 14 log:

- **Cost controls apply to both chat and evaluation.** The runner's
  budgets exist; chat now has the same hard ceilings, exposed through
  the same `UILimits` surface.
- **Concurrency is bounded, not serialized.** The Phase 14 queue is a
  traffic shaper, not a cost control; token and turn caps apply per
  session, not globally.
- **Guard and ledger stay unchanged.** A token-cap refusal is a
  user-visible status, not a security finding.
- **Re-prove live after Phase 15.** A capped request must still redact
  the key from the error, must still clear on End session, and must
  still leave the byte-level secure-deletion guarantee intact.

## What was built

### New module: `src/app/limits.py`

The chat-path cost controls, parallel to `evaluation.limits`:

| Type | Purpose |
| --- | --- |
| `ChatLimits` | Frozen set of ceilings: `max_input_tokens`, `max_output_tokens`, `max_turns`, `max_message_chars`, `max_session_tokens`, `max_session_requests`, `request_timeout_seconds`. Validated in `__post_init__`. |
| `ChatBudget` | Mutable session-cumulative accounting. `check_turn()` raises `ChatLimitExceeded`; `charge()` records one request from provider-reported usage or the estimator; `record_turn()` / `record_refusal()` count turns and refusals; `snapshot()` returns the JSON-ready cost report. |
| `ChatLimitExceeded` | A session ceiling was reached. The message is safe to render: it names the limit and the usage, never a prompt or a credential. |

The split mirrors `evaluation.limits`: `ChatLimits` is immutable (a limit
that changes mid-session is not a limit), `ChatBudget` is the mutable
accounting the controller updates after every turn.

### Extended: `src/app/controller.py`

`UILimits` grew from two fields to seven — the plan's full chat-side
cost surface:

```python
@dataclass(frozen=True)
class UILimits:
    max_turns: int = 200
    max_message_chars: int = 8000
    max_input_tokens: int = 32_000
    max_output_tokens: int = 4_096
    max_session_tokens: int = 500_000
    max_session_requests: int = 500
    request_timeout_seconds: float = 120.0
```

`chat()` now:

1. creates the session's `ChatBudget` on first use (`_ensure_budget`),
2. calls `budget.check_turn()` before the provider is reached,
3. counts the turn only after the service is successfully created (a
   failed service creation does not consume a turn),
4. passes `request_timeout` to `service.handle_user_message`,
5. charges the budget from the turn's reported usage,
6. records a refusal if the prompt exceeded the per-request ceiling.

`update_costs()` replaces the controller's `UILimits` with a new frozen
object carrying the validated values. New ceilings apply from the next
turn; existing session budgets are not retroactively rewritten.

`TurnView` gained three fields: `usage_summary` (markdown), `usage_report`
(JSON), and `turn_usage` (per-turn dict). The export payload gained a
`usage` block.

### Extended: `src/app/service.py`

`handle_user_message()` accepts `request_timeout: float | None` and
wraps the provider call with `concurrent.futures.ThreadPoolExecutor` so
a slow or stalled provider cannot hold a Gradio worker indefinitely. A
`_TimeoutError` (a `ProviderError` subclass) is raised on timeout and
rendered as a redacted status, exactly like any other provider failure.

`ConversationTurn` gained a `usage` dict: `prompt_tokens`,
`completion_tokens`, `timed_out`, `failed`, `model`. Provider-reported
usage is preferred; the estimator fills in when the provider reports
nothing.

### Extended: `src/app/state.py`

`SessionState` gained `usage: Any` — the session's `ChatBudget`,
created lazily by the controller. `clear_conversation()` resets it
alongside the transcript, because the ceilings protect one
conversation's worth of spend.

### Extended: `src/app/panels.py`

`usage_summary()` renders the session's cost accounting as markdown:
headline (within limits or limit reached), usage table (turns, requests,
input/output/total tokens, refused, timed out), remaining table, and
limits table.

### Extended: `src/app/ui.py`

The sidebar gained a **Cost Controls** section with four widgets:

- Max input tokens (per request)
- Max output tokens (per request)
- Request timeout (seconds)
- Max session tokens (total)

Each widget's `.change` event calls `_update_costs`, which updates the
controller's `UILimits` and renders a confirmation.

The inspection tabs gained a **Usage** tab: a markdown summary and a
JSON viewer. The `PanelComponents` order grew from 13 to 15 fields.

The Evaluation tab's markdown was updated to reference the presets and
explain that benchmark runs are driven from the CLI.

### New tests

| File | Count |
| --- | --- |
| `tests/unit/test_chat_limits.py` | 20 — `ChatLimits` validation, `ChatBudget` gating, charging, snapshot, refusal, credential-blind |
| `tests/unit/test_chat_limits_controller.py` | 13 — budget creation, usage tracking, turn view, session limits enforced, `update_costs`, export, clear-conversation reset, timeout |

### Updated tests

| File | Change |
| --- | --- |
| `tests/unit/test_controller.py` | Turn-limit assertion accepts the new `ChatLimitExceeded` message format |
| `tests/ui/test_ui.py` | Callback count updated (12 → 16); panel component count updated (13 → 15); security-tab position adjusted for the new usage components |

## Decisions later phases must not undo

1. **A refusal is a status, not a security finding.** `ChatLimitExceeded`
   renders a user-visible message; it does not enter the security
   ledger. A token-cap refusal and an injection quarantine are different
   kinds of event and must stay in different vocabularies.
2. **The budget is created lazily.** A session that never sends a turn
   costs nothing to track. `_ensure_budget()` creates the `ChatBudget`
   on first use; `state.usage` is `None` until then.
3. **Turn counting happens after service creation.** A failed BrainOS
   adapter creation must not consume a turn from the budget — the
   visitor did not get a turn, they got an error. `budget.record_turn()`
   is called after `self.service()` succeeds.
4. **Limits are replaced, not mutated.** `update_costs()` builds a new
   `UILimits` and assigns it. Existing session budgets keep their
   original limits until the next turn creates a new budget (after a
   clear-conversation) or the session ends.
5. **The timeout wraps the provider call, not the whole turn.** BrainOS
   observe/recall/context-construction are local and fast; the
   wall-clock timeout applies only to the network-bound provider call.
   A `_TimeoutError` is a `ProviderError` subclass so the existing error
   rendering handles it.
6. **`clear_conversation` resets the budget.** The ceilings protect one
   conversation's worth of spend; a cleared conversation starts a fresh
   budget. `end_session` already drops the whole session (including the
   budget) via `SessionManager.end()`.
7. **Reported usage is preferred over estimated.** When the provider
   reports `prompt_tokens` and `completion_tokens`, those are what get
   charged. The estimator fills in only when the provider reports
   nothing, and the `token_counter` field in the snapshot names which
   was used.
8. **The usage block travels into the export.** A session's cost
   accounting is part of its record; `export_session()` includes it,
   and the scanner already covers exports.

## Measured behaviour

```bash
.venv/bin/pytest -q                                          # 1048 passed
.venv/bin/ruff check .                                       # All checks passed!
.venv/bin/python -m security.scan src docs README.md CONTEXT.md benchmarks \
    app.py requirements.txt packages.txt pyproject.toml \
    BrainOS_Context_Lab_Implementation_Plan.md               # 90 files, 0 findings
```

No provider API key was used anywhere: generation is exercised through
`FakeLLMProvider` and the localhost stub.

### Chat limits enforced (live, no provider key)

```python
limited = UIController(
    provider_factory=FakeLLMProvider,
    adapter_factory=adapter_factory,
    limits=UILimits(max_session_tokens=1),
)
sid = limited.connect(None, provider="openai", model="m", api_key=KEY).session_id
first = limited.chat(sid, "first message")   # succeeds
second = limited.chat(sid, "second message") # refused: "max_session_tokens"
```

### Usage tracking (live, no provider key)

```python
controller.chat(sid, "hello")
payload = controller.usage_payload(sid)
# → {"requests": 1, "user_turns": 1, "input_tokens": ~200,
#     "output_tokens": ~20, "total_tokens": ~220,
#     "remaining_turns": 199, "remaining_session_tokens": ~499780,
#     "within_limits": True, ...}
```

## Bugs and gaps found while building it

1. **Turn counting before service creation consumed budget on failure.**
   `budget.record_turn()` was called before `self.service()`, so a
   BrainOS-not-installed error consumed a turn from the budget. Fixed
   by moving `record_turn()` after service creation.
2. **`UILimits` did not validate `request_timeout_seconds`.** The
   `ChatLimits.__post_init__` validates it, but `UILimits` accepted a
   negative timeout silently. Fixed by validating through
   `ChatLimits` in `update_costs()` before assigning.
3. **`PanelComponents.order` grew but the UI test pinned the old count.**
   The test asserted `== 13` and the security tab position; both
   updated for the new 15-field order.

## Constraints carried into Phase 16+

1. **Phase 16 (reproducibility) must record the chat limits.** A
   reproducible session export should include the `ChatLimits` that
   governed it, so a reader can tell whether a low token count is "the
   session was new" or "the operator set a tight ceiling".
2. **Phase 17 (evaluation pipeline in the UI) should surface the
   presets.** The Evaluation tab already references them in markdown;
   Phase 17 should let the visitor pick a preset and see the
   `estimate_ceiling` before running.
3. **Phase 18 (test suite completion) should add a timeout test.**
   A `SlowProvider` fake that sleeps longer than the configured timeout
   would prove the `_TimeoutError` path end-to-end.
4. **The `-wal` and `-shm` files are still part of the database.**
   Phase 14's secure-deletion contract holds; Phase 15 adds no new
   storage paths.
5. **Cost controls are per-session, not per-user.** A multi-user
   deployment (Phase 20+) would need per-user budgets on top of these
   per-session ones.
