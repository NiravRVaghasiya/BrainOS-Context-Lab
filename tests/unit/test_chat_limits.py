"""Tests for the chat-path cost controls (Phase 15, plan §15 / §21).

These tests pin the behavioural contract of :mod:`app.limits`: the ceilings
are enforced, the accounting is exact, the refusal is a status (not a
security finding), and the budget never publishes a credential.
"""

from __future__ import annotations

import pytest

from app.limits import (
    CHAT_LIMIT_FIELDS,
    ChatBudget,
    ChatLimitExceeded,
    ChatLimits,
)

# --------------------------------------------------------------------------- #
# ChatLimits
# --------------------------------------------------------------------------- #


def test_chat_limits_defaults_are_generous_but_bounded() -> None:
    limits = ChatLimits()
    assert limits.max_input_tokens == 32_000
    assert limits.max_output_tokens == 4_096
    assert limits.max_turns == 200
    assert limits.max_message_chars == 8_000
    assert limits.max_session_tokens == 500_000
    assert limits.max_session_requests == 500
    assert limits.request_timeout_seconds == 120.0


def test_chat_limits_rejects_negative_integers() -> None:
    with pytest.raises(ValueError, match="max_input_tokens"):
        ChatLimits(max_input_tokens=-1)


def test_chat_limits_rejects_zero_timeout() -> None:
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        ChatLimits(request_timeout_seconds=0)


def test_chat_limits_rejects_non_integer_fields() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        ChatLimits(max_turns=1.5)  # type: ignore[arg-type]


def test_chat_limits_to_dict_carries_version_and_every_field() -> None:
    limits = ChatLimits()
    payload = limits.to_dict()
    assert payload["chat_limits_version"] == "chat-limits-v1"
    for field_name in CHAT_LIMIT_FIELDS:
        assert field_name in payload


def test_chat_limits_is_frozen() -> None:
    limits = ChatLimits()
    with pytest.raises(AttributeError):
        limits.max_turns = 10  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ChatBudget — turn and request gating
# --------------------------------------------------------------------------- #


def test_budget_check_turn_allows_until_max_turns() -> None:
    budget = ChatBudget(limits=ChatLimits(max_turns=2))
    budget.record_turn()
    budget.check_turn()  # 1 < 2, allowed
    budget.record_turn()
    with pytest.raises(ChatLimitExceeded) as exc_info:
        budget.check_turn()
    assert exc_info.value.limit == "max_turns"
    assert exc_info.value.allowed == 2
    assert exc_info.value.used == 2


def test_budget_check_turn_refuses_on_session_requests() -> None:
    budget = ChatBudget(limits=ChatLimits(max_session_requests=1))
    budget.charge(prompt_tokens=10, completion_text="ok")
    with pytest.raises(ChatLimitExceeded) as exc_info:
        budget.check_turn()
    assert exc_info.value.limit == "max_session_requests"


def test_budget_check_turn_refuses_on_session_tokens() -> None:
    budget = ChatBudget(limits=ChatLimits(max_session_tokens=100))
    budget.charge(prompt_tokens=60, completion_text="x" * 200)
    # After this charge, total_tokens >= 100.
    with pytest.raises(ChatLimitExceeded) as exc_info:
        budget.check_turn()
    assert exc_info.value.limit == "max_session_tokens"


def test_budget_fits_prompt_refuses_oversized_prompts() -> None:
    budget = ChatBudget(limits=ChatLimits(max_input_tokens=100))
    ok, reason = budget.fits_prompt(50)
    assert ok is True and reason == ""
    ok, reason = budget.fits_prompt(200)
    assert ok is False
    assert "max_input_tokens" in reason


def test_budget_charge_counts_reported_usage_when_available() -> None:
    budget = ChatBudget(limits=ChatLimits())
    charged = budget.charge(
        prompt_tokens=100,
        completion_text="hello",
        usage={"prompt_tokens": 42, "completion_tokens": 7},
    )
    assert charged == {"input_tokens": 42, "output_tokens": 7}
    assert budget.input_tokens == 42
    assert budget.output_tokens == 7
    assert budget.requests == 1


def test_budget_charge_falls_back_to_estimator_when_provider_reports_nothing() -> None:
    budget = ChatBudget(limits=ChatLimits())
    charged = budget.charge(prompt_tokens=100, completion_text="hello world")
    assert charged["input_tokens"] == 100
    assert charged["output_tokens"] > 0  # the estimator counted "hello world"
    assert budget.requests == 1


def test_budget_charge_records_timeouts_and_failures() -> None:
    budget = ChatBudget(limits=ChatLimits())
    budget.charge(prompt_tokens=10, completion_text="", failed=True)
    budget.charge(prompt_tokens=10, completion_text="", timed_out=True)
    assert budget.timeouts == 1
    assert len(budget.history) == 2
    assert budget.history[0]["failed"] is True
    assert budget.history[1]["timed_out"] is True


def test_budget_record_refusal_counts_and_records_reason() -> None:
    budget = ChatBudget(limits=ChatLimits())
    budget.record_refusal("max_input_tokens")
    assert budget.refused == 1
    assert budget.history[-1]["refused"] is True
    assert budget.history[-1]["reason"] == "max_input_tokens"


# --------------------------------------------------------------------------- #
# ChatBudget — snapshot
# --------------------------------------------------------------------------- #


def test_budget_snapshot_reports_remaining_and_within_limits() -> None:
    budget = ChatBudget(limits=ChatLimits(max_turns=5, max_session_tokens=1000))
    budget.record_turn()
    budget.record_turn()
    budget.charge(prompt_tokens=200, completion_text="ok")
    snap = budget.snapshot()

    assert snap["user_turns"] == 2
    assert snap["remaining_turns"] == 3
    assert snap["remaining_session_tokens"] == 800 - snap["output_tokens"]
    assert snap["within_limits"] is True
    assert snap["limits"]["max_turns"] == 5
    assert snap["chat_limits_version"] == "chat-limits-v1"


def test_budget_snapshot_within_limits_false_when_exceeded() -> None:
    budget = ChatBudget(limits=ChatLimits(max_turns=1))
    budget.record_turn()
    # At the limit: within_limits is still True (used <= limit), but
    # remaining_turns is 0 and the next check_turn() will refuse.
    snap = budget.snapshot()
    assert snap["within_limits"] is True
    assert snap["remaining_turns"] == 0
    # Exceeding the limit (e.g. by counting past the ceiling) flips it:
    budget.record_turn()
    snap = budget.snapshot()
    assert snap["within_limits"] is False


def test_budget_snapshot_carries_no_credential() -> None:
    budget = ChatBudget(limits=ChatLimits())
    budget.charge(prompt_tokens=10, completion_text="sk-secret-KEY-0001")
    snap = budget.snapshot()
    blob = str(snap)
    assert "sk-secret-KEY-0001" not in blob


def test_budget_total_tokens_is_sum_of_input_and_output() -> None:
    budget = ChatBudget(limits=ChatLimits())
    budget.charge(
        usage={"prompt_tokens": 100, "completion_tokens": 50}
    )
    assert budget.total_tokens == 150


# --------------------------------------------------------------------------- #
# ChatLimitExceeded
# --------------------------------------------------------------------------- #


def test_chat_limit_exceeded_message_is_credential_blind() -> None:
    exc = ChatLimitExceeded("max_turns", allowed=10, used=10)
    message = str(exc)
    assert "max_turns" in message
    assert "10" in message
    assert "api_key" not in message.lower()


def test_chat_limit_exceeded_carries_limit_and_usage() -> None:
    exc = ChatLimitExceeded("max_session_tokens", allowed=1000, used=1000)
    assert exc.limit == "max_session_tokens"
    assert exc.allowed == 1000
    assert exc.used == 1000
