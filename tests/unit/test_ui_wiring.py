"""Structural tests for the Gradio wiring (skipped when Gradio is absent).

These verify the Phase 4 layout and event wiring without launching a server:
the app builds, the planned panels exist, and the callbacks are registered.
Behaviour itself is covered headlessly by ``test_ui_controller.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.controller import ChatController
from app.service import ConversationService
from app.state import SessionState
from brain.adapter import BrainOSAdapter
from tests.fakes import FakeProvider, FakeRuntime


def build_demo() -> tuple[Any, Any]:
    gr = pytest.importorskip("gradio")
    from app.ui import create_app

    def factory() -> ChatController:
        return ChatController(
            SessionState(),
            service_factory=lambda state: ConversationService(
                state,
                adapter=BrainOSAdapter(FakeRuntime()),
                provider=FakeProvider(),
            ),
        )

    return gr, create_app(controller_factory=factory)


def test_create_app_builds_blocks_and_wires_events() -> None:
    gr, demo = build_demo()
    assert isinstance(demo, gr.Blocks)
    # load, send click, send submit, validate, list models, clear,
    # clear memory, export, end, mode change
    assert len(demo.fns) >= 10


def test_create_app_declares_planned_panels() -> None:
    _gr, demo = build_demo()
    # Components carry their visible text in `label` (inputs) or `value`
    # (buttons); collect both.
    labels = set()
    for block in demo.blocks.values():
        for attribute in ("label", "value"):
            text = getattr(block, attribute, None)
            if isinstance(text, str) and text:
                labels.add(text)
    for expected in (
        # Sidebar (plan §8)
        "Provider",
        "Model",
        "API key (session-only)",
        "Endpoint (optional)",
        "Temperature",
        "Context budget (max tokens)",
        "Memory mode",
        # Main
        "Conversation",
        "Message",
        # Inspection tabs
        "Stored memories",
        "Retrieved memories",
        "Context accounting (all fields)",
        "Final prompt sent to the model",
        "Cognitive trace",
        "Dropped memories",
        "Conflict resolutions",
        # Phase 5 data controls
        "Clear memory",
        "Export session",
        "Session export",
    ):
        assert expected in labels, f"missing panel: {expected}"


def test_chatbot_helper_tolerates_gradio_without_type_param() -> None:
    from app.ui import _make_chatbot

    class _GR:
        class Chatbot:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

    bot = _make_chatbot(_GR, label="c")
    assert "type" not in bot.kwargs

    class _GRTyped:
        class Chatbot:
            def __init__(self, type: str = "tuples", **kwargs: Any) -> None:  # noqa: A002
                self.kwargs = {"type": type, **kwargs}

    bot = _make_chatbot(_GRTyped, label="c")
    assert bot.kwargs["type"] == "messages"
