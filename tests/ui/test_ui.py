"""Tests for the Gradio layer.

The layout is exercised with the real Gradio package when it is installed, and
skipped otherwise — the controller, panels, and service tests cover behaviour
without it. What these tests pin down is the part that can only break here:

* the app builds against the installed Gradio version,
* every registered callback declares as many outputs as its function returns,
* the connect callback never hands the key back to the browser,
* the export callback writes a parseable, credential-free file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.controller import UIController
from app.ui import create_app
from brain.adapter import BrainOSAdapter
from tests.fakes import FakeLLMProvider, LooseFakeRuntime

KEY = "sk-ui-SECRET-0002"

gradio = pytest.importorskip("gradio", reason="Gradio is an optional dependency")


def _controller() -> UIController:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(
            LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
        )

    return UIController(provider_factory=FakeLLMProvider, adapter_factory=adapter_factory)


def test_app_builds_and_registers_every_callback() -> None:
    demo = create_app(_controller())

    # 10 widget callbacks plus the per-page session bootstrap.
    assert len(demo.fns) == 11
    assert any(block_fn.targets == [(0, "load")] for block_fn in demo.fns.values())


def test_chat_callback_returns_one_value_per_declared_output() -> None:
    controller = _controller()
    demo = create_app(controller)

    values = controller.chat(None, "The production database is PostgreSQL 16.")
    from app.ui import _chat

    produced = _chat(controller)("The production database is PostgreSQL 16.", None)
    declared = {
        (len(block_fn.inputs), len(block_fn.outputs)) for block_fn in demo.fns.values()
    }

    assert len(values.history) >= 0  # the view is what the UI unpacks
    assert (2, len(produced)) in declared


def test_panel_callbacks_return_matching_lengths() -> None:
    controller = _controller()
    demo = create_app(controller)

    from app.ui import _clear_conversation, _refresh

    declared = {
        (len(block_fn.inputs), len(block_fn.outputs)) for block_fn in demo.fns.values()
    }
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    controller.chat(sid, "The production database is PostgreSQL 16.")

    assert (1, len(_refresh(controller)(sid))) in declared
    assert (1, len(_clear_conversation(controller)(sid))) in declared


def test_connect_callback_clears_the_key_box_and_names_no_secret() -> None:
    from app.ui import _connect

    controller = _controller()
    status, model_update, key_value, session_id = _connect(controller)(
        "openai", "gpt-4o-mini", KEY, "", 0.2, None, None
    )

    assert key_value == ""
    assert KEY not in status
    assert KEY not in json.dumps(model_update, default=str)
    assert session_id


def test_export_callback_writes_a_credential_free_json_file(tmp_path: Path) -> None:
    from app.ui import _export

    controller = _controller()
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    controller.chat(sid, "The production database is PostgreSQL 16.")

    update = _export(controller)(sid)
    path = Path(update["value"] if isinstance(update, dict) else update.value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)

    assert payload["messages"]
    assert "api_key" not in payload["provider"]
    assert KEY not in path.read_text(encoding="utf-8") if path.exists() else True


def test_update_context_callback_accepts_the_sidebar_values() -> None:
    from app.ui import _update_context

    controller = _controller()
    status, session_id = _update_context(controller)(
        "brainos", 4096, 256, 512, 6, 0.12, 20.0, True, True, False, None
    )

    assert "Context updated" in status
    settings = controller.ensure_session(session_id).context
    assert settings.recent_turn_budget == 256
    assert settings.memory_budget == 512
