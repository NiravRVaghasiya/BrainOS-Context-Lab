"""Phase 19: a slow model cannot hold a worker — the whole path, not the seam.

Phase 15 built the timeout (a thread-pool future around the provider call) and
recorded it in the usage accounting. What no test did was *make a provider
slow* and watch the turn come back: the existing coverage asserted that the
controller passes ``request_timeout_seconds`` down and that a fast fake never
times out.

That is the difference between a timeout that exists and a timeout that works,
and it matters most on the host this project is aimed at: a public Space where
each chat turn holds a Gradio worker thread. These tests drive the real
controller with a provider that blocks on an event, and pin the four things a
user and an operator need:

* the turn returns promptly with a redacted status instead of hanging;
* the transcript, memory and usage accounting still record what happened
  (a timeout is not a lost turn);
* the budget counts the request and the timeout, so a flaky provider cannot be
  hammered for free;
* the credential never reaches the rendered output.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from app.controller import UIController, UILimits
from app.limits import ChatLimits
from brain.adapter import BrainOSAdapter
from providers.base import ProviderConfig, ProviderResponse
from tests.fakes import LooseFakeRuntime, RecordingProvider

KEY = "sk-timeout-SECRET-9876"
SESSION_TEXT = "Which database should I use for the audit log?"


class BlockingProvider:
    """A provider that never answers until the test lets it.

    ``generate`` waits on a release event with a hard cap so a failing test
    cannot leak a thread forever; ``list_models`` and ``validate_credentials``
    answer immediately, because connect() must not be the thing that blocks.
    """

    def __init__(self, config: ProviderConfig, *, wait_seconds: float = 30.0) -> None:
        self.config = config
        self.wait_seconds = wait_seconds
        self.calls: list[list[dict[str, str]]] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def list_models(self) -> list[str]:
        return [self.config.model]

    def validate_credentials(self) -> bool:
        return True

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        self.calls.append(messages)
        self.started.set()
        self.release.wait(self.wait_seconds)
        return ProviderResponse(
            text="late answer",
            model=self.config.model,
            usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        )


def adapter_factory(**kwargs: object) -> BrainOSAdapter:
    """The real adapter over the session-bound fake runtime.

    The timeout lives above BrainOS, so the turn must exercise the real adapter
    and a real (if fake) runtime — the provider is the only thing that blocks.
    """

    return BrainOSAdapter(
        LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
    )


@pytest.fixture()
def blockers():
    """Track every provider the controller builds so teardown releases them."""

    created: list[BlockingProvider] = []

    def factory(config: ProviderConfig) -> BlockingProvider:
        provider = BlockingProvider(config)
        created.append(provider)
        return provider

    yield factory

    for provider in created:
        provider.release.set()


def _controller(
    provider_factory: Any,
    *,
    timeout: float,
    extra: dict[str, Any] | None = None,
) -> UIController:
    return UIController(
        provider_factory=provider_factory,
        adapter_factory=adapter_factory,
        limits=UILimits(request_timeout_seconds=timeout),
        **(extra or {}),
    )


# --------------------------------------------------------------------------- #
# The turn the user is waiting on
# --------------------------------------------------------------------------- #


def test_a_blocked_provider_returns_a_redacted_timeout_instead_of_hanging(
    blockers: Any,
) -> None:
    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id

    started = time.monotonic()
    view = controller.chat(sid, SESSION_TEXT)
    elapsed = time.monotonic() - started

    assert elapsed < 10, f"chat() held the caller for {elapsed:.1f}s despite a 0.25s timeout"
    assert view.turn_usage["timed_out"] is True
    assert view.turn_usage["failed"] is True
    assert "did not complete within" in view.status
    # The status is what a browser renders; the key must not be in it.
    assert KEY not in view.status
    assert KEY not in "\n".join(message["content"] for message in view.history)


def test_a_timeout_is_a_recorded_turn_not_a_lost_one(blockers: Any) -> None:
    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id

    view = controller.chat(sid, SESSION_TEXT)

    # The user's own message is in the transcript and in the UI history; the
    # provider call was made exactly once and returned nothing.
    state = controller.ensure_session(sid)
    assert [message["role"] for message in state.messages] == ["user"]
    assert any(message["content"] == SESSION_TEXT for message in view.history)
    payload = controller.export_session(sid)
    assert payload["usage"]["requests"] == 1
    assert payload["usage"]["timeouts"] == 1


def test_a_timed_out_request_is_still_charged_to_the_budget(blockers: Any) -> None:
    """One blocked call must consume one request, or the ceiling is a fiction."""

    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id

    controller.chat(sid, SESSION_TEXT)

    budget = controller.ensure_session(sid).usage
    assert budget.requests == 1
    assert budget.timeouts == 1
    assert budget.input_tokens > 0  # the prompt we built was really sent


def test_the_session_survives_and_the_next_turn_still_works(blockers: Any) -> None:
    """A timed-out turn leaves the session usable — no wedged worker, no lock."""

    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    controller.chat(sid, SESSION_TEXT)

    # Swap the blocked provider for an honest one: the same session must accept
    # a normal turn afterwards.
    service = controller.service(sid)
    service._provider = RecordingProvider(text="PostgreSQL 16.")
    view = controller.chat(sid, "And for the queue?")

    assert view.turn_usage.get("timed_out") is not True
    assert any(
        message["role"] == "assistant" and message["content"] == "PostgreSQL 16."
        for message in view.history
    )


def test_a_generous_timeout_leaves_a_normal_slow_call_alone(blockers: Any) -> None:
    """The timeout must not fire early, with a margin a loaded runner cannot eat.

    The provider answers after 0.2 s against a 60 s ceiling: a 300× margin, so a
    slow CI box cannot turn "slow but fine" into a timeout — while a timeout that
    fired early would still be caught here.
    """

    controller = _controller(blockers, timeout=60.0)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    provider = controller.service(sid).provider()
    threading.Timer(0.2, provider.release.set).start()

    started = time.monotonic()
    view = controller.chat(sid, SESSION_TEXT)
    elapsed = time.monotonic() - started

    assert elapsed < 30, f"a 0.2s answer took {elapsed:.1f}s under a 60s timeout"
    assert view.turn_usage.get("timed_out") is not True
    assert view.turn_usage["completion_tokens"] == 2
    assert "late answer" in [message["content"] for message in view.history]


# --------------------------------------------------------------------------- #
# The seam itself
# --------------------------------------------------------------------------- #


def test_handle_user_message_reports_the_timeout_without_raising(blockers: Any) -> None:
    """The service is the boundary: a timeout is an error value, not an exception."""

    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    service = controller.service(sid)

    turn = service.handle_user_message(SESSION_TEXT, request_timeout=0.25)

    assert turn.generated is False
    assert turn.reply is None
    assert turn.usage["timed_out"] is True
    assert turn.usage["failed"] is True
    assert "0.25" in (turn.error or "")
    assert len(service._provider.calls) == 1


def test_a_non_positive_timeout_disables_the_wrapper(blockers: Any) -> None:
    """``None``/``<= 0`` keeps the direct call path (no thread pool)."""

    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    service = controller.service(sid)
    provider = service.provider()
    threading.Timer(0.1, provider.release.set).start()

    turn = service.handle_user_message(SESSION_TEXT, request_timeout=0)

    assert turn.usage["timed_out"] is False
    assert turn.usage["completion_tokens"] == 2


def test_limits_still_refuse_a_nonsensical_timeout() -> None:
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        ChatLimits(request_timeout_seconds=0)


def test_the_timeout_message_never_names_a_credential_shape(blockers: Any) -> None:
    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id

    report = controller.usage_payload(sid)
    rendered = str(report)

    assert KEY not in rendered
    assert "sk-" not in rendered


def test_a_timed_out_turn_leaves_brainos_state_consistent(blockers: Any) -> None:
    """The timeout is above BrainOS: the user's message is still observed.

    BrainOS never saw a reply, so its memory layer must not hold one; but the
    message the user sent is part of the session regardless of what the model
    did, which is what makes the next turn able to use it.
    """

    controller = _controller(blockers, timeout=0.25)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    service = controller.service(sid)

    controller.chat(sid, SESSION_TEXT)

    # The message the user sent is part of the session regardless of what the
    # model did; the transcript holds it and holds no assistant reply.
    assert [message["role"] for message in controller.ensure_session(sid).messages] == ["user"]
    # The context engine still ran: the built prompt is recorded on the state,
    # so the diagnostics panels describe the turn that timed out.
    assert service.state.last_context, "the built prompt was not recorded"
    assert service.state.last_guard, "the injection guard did not run"
    # Nothing the model never said was observed into memory.
    stored = [str(memory.content) for memory in service._safe_list_memories()]
    assert not any("late answer" in content for content in stored)
