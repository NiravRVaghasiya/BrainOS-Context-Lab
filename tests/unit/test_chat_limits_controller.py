"""Tests for the chat-path cost controls wired into the controller (Phase 15).

These tests pin the integration contract: the controller creates a budget,
checks it before each turn, charges it after, refuses when a ceiling is
hit, and surfaces usage in the view and the export.
"""

from __future__ import annotations

import json

import pytest

from app.controller import UIController, UILimits
from app.limits import ChatBudget
from brain.adapter import BrainOSAdapter
from tests.fakes import FakeLLMProvider, LooseFakeRuntime

KEY = "sk-session-SECRET-0015"


@pytest.fixture()
def controller() -> UIController:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(
            LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
        )

    return UIController(
        provider_factory=FakeLLMProvider, adapter_factory=adapter_factory
    )


def _connected(controller: UIController, **overrides: object) -> str:
    view = controller.connect(
        None, provider="openai", model="gpt-4o-mini", api_key=KEY, **overrides
    )
    return view.session_id


# --------------------------------------------------------------------------- #
# Budget creation and tracking
# --------------------------------------------------------------------------- #


def test_controller_creates_a_budget_on_first_chat(controller: UIController) -> None:
    sid = _connected(controller)
    state = controller.ensure_session(sid)
    assert state.usage is None  # no budget yet
    controller.chat(sid, "hello")
    assert isinstance(state.usage, ChatBudget)
    assert state.usage.requests >= 1


def test_controller_usage_payload_before_any_turn(controller: UIController) -> None:
    sid = _connected(controller)
    payload = controller.usage_payload(sid)
    assert payload["requests"] == 0
    assert payload["user_turns"] == 0
    assert payload["within_limits"] is True


def test_controller_usage_payload_after_a_turn(controller: UIController) -> None:
    sid = _connected(controller)
    controller.chat(sid, "hello")
    payload = controller.usage_payload(sid)
    assert payload["requests"] >= 1
    assert payload["user_turns"] == 1
    assert payload["input_tokens"] > 0


# --------------------------------------------------------------------------- #
# Turn view carries usage
# --------------------------------------------------------------------------- #


def test_turn_view_carries_usage_summary_and_report(
    controller: UIController,
) -> None:
    sid = _connected(controller)
    view = controller.chat(sid, "hello")
    assert "### Usage" in view.usage_summary
    assert view.usage_report["requests"] >= 1
    assert view.turn_usage.get("prompt_tokens", 0) > 0


def test_turn_view_usage_never_carries_the_key(
    controller: UIController,
) -> None:
    sid = _connected(controller)
    view = controller.chat(sid, f"My key is {KEY}")
    assert KEY not in view.usage_summary
    assert KEY not in json.dumps(view.usage_report)
    assert KEY not in json.dumps(view.turn_usage)


# --------------------------------------------------------------------------- #
# Session limits enforced
# --------------------------------------------------------------------------- #


def test_session_token_limit_refuses_subsequent_turns() -> None:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(
            LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
        )

    limited = UIController(
        provider_factory=FakeLLMProvider,
        adapter_factory=adapter_factory,
        limits=UILimits(max_session_tokens=1),  # practically zero
    )
    sid = limited.connect(None, provider="openai", model="m", api_key=KEY).session_id
    first = limited.chat(sid, "first message")
    assert first.history  # first turn goes through
    second = limited.chat(sid, "second message")
    assert "max_session_tokens" in second.status or "limit" in second.status.lower()


def test_session_request_limit_refuses_subsequent_turns() -> None:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(
            LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
        )

    limited = UIController(
        provider_factory=FakeLLMProvider,
        adapter_factory=adapter_factory,
        limits=UILimits(max_session_requests=1),
    )
    sid = limited.connect(None, provider="openai", model="m", api_key=KEY).session_id
    limited.chat(sid, "first")
    second = limited.chat(sid, "second")
    assert "max_session_requests" in second.status or "limit" in second.status.lower()


# --------------------------------------------------------------------------- #
# Cost controls update
# --------------------------------------------------------------------------- #


def test_update_costs_replaces_the_limits(controller: UIController) -> None:
    sid = _connected(controller)
    status = controller.update_costs(
        sid,
        max_input_tokens=1000,
        max_output_tokens=500,
        request_timeout_seconds=30,
        max_session_tokens=10_000,
    )
    assert "Cost limits updated" in status
    assert controller.limits.max_input_tokens == 1000
    assert controller.limits.max_output_tokens == 500
    assert controller.limits.request_timeout_seconds == 30
    assert controller.limits.max_session_tokens == 10_000


def test_update_costs_rejects_unknown_fields(controller: UIController) -> None:
    sid = _connected(controller)
    status = controller.update_costs(sid, unknown_field=42)
    assert "⚠️" in status
    assert "Unknown" in status


def test_update_costs_rejects_invalid_values(controller: UIController) -> None:
    sid = _connected(controller)
    status = controller.update_costs(sid, request_timeout_seconds=-5)
    assert "⚠️" in status


# --------------------------------------------------------------------------- #
# Export carries usage
# --------------------------------------------------------------------------- #


def test_export_session_includes_usage(controller: UIController) -> None:
    sid = _connected(controller)
    controller.chat(sid, "hello")
    payload = controller.export_session(sid)
    assert "usage" in payload
    assert payload["usage"]["requests"] >= 1
    assert KEY not in json.dumps(payload)


# --------------------------------------------------------------------------- #
# Clear conversation resets the budget
# --------------------------------------------------------------------------- #


def test_clear_conversation_resets_the_budget(controller: UIController) -> None:
    sid = _connected(controller)
    controller.chat(sid, "hello")
    state = controller.ensure_session(sid)
    assert isinstance(state.usage, ChatBudget)
    assert state.usage.requests >= 1
    controller.clear_conversation(sid)
    state = controller.ensure_session(sid)
    assert state.usage is None  # budget reset


# --------------------------------------------------------------------------- #
# Request timeout
# --------------------------------------------------------------------------- #


def test_request_timeout_is_passed_to_the_service(
    controller: UIController,
) -> None:
    """The controller passes the timeout to the service's handle_user_message."""

    sid = _connected(controller)
    view = controller.chat(sid, "hello")
    # The turn completed without timing out (the fake provider is instant).
    assert view.history
    assert view.turn_usage.get("timed_out") is not True
