"""Cost controls for the chat path (plan §15 / §21, Phase 15).

Users bring their own API key, so the application's job on the chat side is
the same as on the evaluation side: make a runaway session impossible rather
than report the damage afterwards. The evaluation runner already has
:class:`evaluation.limits.RunLimits` and :class:`evaluation.limits.RunBudget`;
this module provides the chat-path counterpart.

Two objects, deliberately parallel to the evaluation pair:

``ChatLimits``
    A frozen set of ceilings for one chat session. Immutable because a limit
    that can change mid-session is not a limit: a visitor who raises their
    own ceiling partway through is a visitor whose spend nobody can audit.

``ChatBudget``
    The mutable accounting for one session. Created once at session start
    and updated after every turn. When a ceiling is hit the controller
    refuses the next turn and renders a user-visible status; the refusal
    is a **status**, not a security finding — the guard vocabulary stays
    untouched.

Enforcement rules:

* a turn whose *estimated* prompt tokens exceed ``max_input_tokens`` is
  **refused before the provider is called**, so a runaway context window
  costs nothing;
* a provider response whose reported output tokens exceed
  ``max_output_tokens`` is **counted and flagged** but still rendered —
  the model already produced the reply and discarding it would confuse
  the user;
* exhausting ``max_turns``, ``max_session_tokens``, or
  ``max_session_requests`` **refuses subsequent turns** until the user
  clears the conversation or starts a new session;
* ``request_timeout_seconds`` wraps the provider call so a slow model
  cannot hold a Gradio worker indefinitely.

Defaults follow the plan's spirit: generous enough for a real conversation,
tight enough that an adversarial visitor on a public Space cannot burn a
meaningful budget before the queue drains them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from brain.tokenizers import TokenCounter, estimate_tokens

#: Identifies the cost-control semantics recorded on a session.
CHAT_LIMITS_VERSION = "chat-limits-v1"

#: Hard-cap field names, used to build readable messages and to serialize.
CHAT_LIMIT_FIELDS: tuple[str, ...] = (
    "max_input_tokens",
    "max_output_tokens",
    "max_turns",
    "max_message_chars",
    "max_session_tokens",
    "max_session_requests",
    "request_timeout_seconds",
)


class ChatLimitExceeded(RuntimeError):
    """A session-level chat cost limit was reached and the turn must be refused.

    The message is safe to render in the UI: it names the limit and the usage
    that hit it, never a request body, a prompt, or a credential.
    """

    def __init__(self, limit: str, *, allowed: int, used: int) -> None:
        super().__init__(
            f"Session limit reached: {limit} ceiling {allowed} (used {used}). "
            "Clear the conversation or start a new session to continue."
        )
        self.limit = limit
        self.allowed = allowed
        self.used = used


@dataclass(frozen=True)
class ChatLimits:
    """The ceilings one chat session may not cross.

    The fields are the plan's Phase 15 list plus the two that already existed
    on :class:`app.controller.UILimits` (``max_turns``, ``max_message_chars``),
    now in one place so a deployment configures chat cost from one object.
    """

    #: Tokens a single prompt may cost. Larger prompts are refused, not sent.
    max_input_tokens: int = 32_000
    #: Tokens a single completion may produce.
    max_output_tokens: int = 4_096
    #: User turns per session.
    max_turns: int = 200
    #: Characters in a single user message.
    max_message_chars: int = 8_000
    #: Input + output tokens across the whole session.
    max_session_tokens: int = 500_000
    #: Provider requests across the whole session.
    max_session_requests: int = 500
    #: Per-request wall-clock timeout (seconds).
    request_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in CHAT_LIMIT_FIELDS:
            value = getattr(self, name)
            if name == "request_timeout_seconds":
                if (
                    not isinstance(value, (int, float))
                    or not isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "request_timeout_seconds must be greater than zero."
                    )
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_limits_version": CHAT_LIMITS_VERSION,
            **{name: getattr(self, name) for name in CHAT_LIMIT_FIELDS},
        }


@dataclass
class ChatBudget:
    """Mutable accounting for one session, updated after every turn.

    Every method is credential-blind: it counts tokens and requests, and is
    never handed a prompt body or a key.
    """

    limits: ChatLimits
    counter: TokenCounter = estimate_tokens
    counter_name: str = "estimate_tokens"
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    user_turns: int = 0
    refused: int = 0
    timeouts: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def fits_prompt(self, prompt_tokens: int) -> tuple[bool, str]:
        """Return whether one prompt may be sent under the per-request ceiling."""

        ceiling = self.limits.max_input_tokens
        if ceiling and prompt_tokens > ceiling:
            return False, (
                f"prompt_over_input_limit: {prompt_tokens} tokens exceeds the "
                f"session's max_input_tokens ceiling of {ceiling}"
            )
        return True, ""

    def check_turn(self) -> None:
        """Raise :class:`ChatLimitExceeded` when the session may not continue."""

        limits = self.limits
        if limits.max_turns and self.user_turns >= limits.max_turns:
            raise ChatLimitExceeded(
                "max_turns", allowed=limits.max_turns, used=self.user_turns
            )
        if limits.max_session_requests and self.requests >= limits.max_session_requests:
            raise ChatLimitExceeded(
                "max_session_requests",
                allowed=limits.max_session_requests,
                used=self.requests,
            )
        if limits.max_session_tokens and self.total_tokens >= limits.max_session_tokens:
            raise ChatLimitExceeded(
                "max_session_tokens",
                allowed=limits.max_session_tokens,
                used=self.total_tokens,
            )

    def charge(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_text: str = "",
        usage: Mapping[str, int] | None = None,
        failed: bool = False,
        timed_out: bool = False,
    ) -> dict[str, int]:
        """Record one request and return how many tokens it was charged."""

        reported_input = _int_field(usage, "prompt_tokens")
        reported_output = _int_field(usage, "completion_tokens")
        input_tokens = (
            reported_input if reported_input is not None else int(prompt_tokens or 0)
        )
        output_tokens = (
            reported_output
            if reported_output is not None
            else self.counter(str(completion_text or ""))
        )
        self.requests += 1
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        if timed_out:
            self.timeouts += 1
        record = {
            "request": self.requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "failed": failed,
            "timed_out": timed_out,
        }
        self.history.append(record)
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}

    def record_turn(self) -> None:
        """Count one user turn (called before the provider is invoked)."""

        self.user_turns += 1

    def record_refusal(self, reason: str) -> None:
        """Count a turn refused by a session ceiling."""

        self.refused += 1
        self.history.append(
            {"refused": True, "reason": reason, "turn": self.user_turns}
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the JSON-ready cost report for the session."""

        limits = self.limits
        return {
            "chat_limits_version": CHAT_LIMITS_VERSION,
            "limits": limits.to_dict(),
            "token_counter": self.counter_name,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "user_turns": self.user_turns,
            "refused": self.refused,
            "timeouts": self.timeouts,
            "remaining_turns": _remaining(limits.max_turns, self.user_turns),
            "remaining_requests": _remaining(
                limits.max_session_requests, self.requests
            ),
            "remaining_session_tokens": _remaining(
                limits.max_session_tokens, self.total_tokens
            ),
            "within_limits": (
                _within(limits.max_turns, self.user_turns)
                and _within(limits.max_session_requests, self.requests)
                and _within(limits.max_session_tokens, self.total_tokens)
            ),
        }


def _int_field(usage: Mapping[str, int] | None, name: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _remaining(limit: int, used: int) -> int | None:
    if not limit:
        return None
    return max(0, int(limit) - int(used))


def _within(limit: int, used: int) -> bool:
    if not limit:
        return True
    return int(used) <= int(limit)


__all__ = [
    "CHAT_LIMITS_VERSION",
    "CHAT_LIMIT_FIELDS",
    "ChatBudget",
    "ChatLimitExceeded",
    "ChatLimits",
]
