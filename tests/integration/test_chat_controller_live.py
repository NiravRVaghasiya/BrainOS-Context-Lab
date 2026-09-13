"""Live Phase 4 validation: the chat controller against the pinned BrainOS.

The unit tests cover the controller with fakes. This file runs the same
controller against the real runtime, with a deterministic provider double, so
that the panels are validated against the memory the runtime actually stores
and retrieves. No provider API key is used.
"""

from __future__ import annotations

import json

import pytest

from app.controller import UIController
from tests.fakes import FakeLLMProvider

pytest.importorskip("brainos_runtime")


def _provider_factory(config: object) -> FakeLLMProvider:
    # A neutral reply keeps the assistant turns out of memory, so the panels
    # show what the *user* said rather than what the fake model answered.
    return FakeLLMProvider(config, text="Noted.")  # type: ignore[arg-type]


def _controller() -> UIController:
    controller = UIController(provider_factory=_provider_factory)
    return controller


def test_live_panels_show_a_fact_recalled_many_turns_later() -> None:
    controller = _controller()
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=""
    ).session_id
    controller.update_context(
        session_id, recent_turn_budget=256, memory_budget=512, max_memories=6
    )

    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    for index in range(8):
        controller.chat(session_id, f"How is the rollout looking today? ({index})")
    view = controller.chat(session_id, "What database does Project Atlas use?")

    assert any("PostgreSQL" in str(row[1]) for row in view.retrieved_rows), view.retrieved_rows
    assert "<retrieved_memory>" in view.prompt
    assert view.stats["final_context_tokens"] > 0
    assert "Recall" in view.trace


def test_live_no_memory_control_retrieves_nothing() -> None:
    controller = _controller()
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=""
    ).session_id
    controller.update_context(session_id, mode="no_memory")
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")

    view = controller.chat(session_id, "What database does Project Atlas use?")

    assert view.retrieved_rows == []
    assert "<retrieved_memory>" not in view.prompt


def test_live_clear_memory_forgets_but_keeps_the_transcript() -> None:
    controller = _controller()
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=""
    ).session_id
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    view = controller.clear_memory(session_id)

    assert view.stored_rows == []
    # No API key, so the transcript is the single user turn — and it survives.
    assert [message["role"] for message in view.history] == ["user"]
    assert "PostgreSQL 16" in view.history[0]["content"]


def test_live_export_contains_the_conversation() -> None:
    controller = _controller()
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=""
    ).session_id
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")

    payload = json.loads(controller.export_text(session_id))

    assert any("PostgreSQL" in str(message.get("content", "")) for message in payload["messages"])
    assert payload["memories"]
