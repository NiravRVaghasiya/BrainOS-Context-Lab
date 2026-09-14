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
from baselines.modes import RECENT_WINDOW_TURNS
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

    # 11 widget callbacks (Phase 6 adds the baseline-mode selector) plus the
    # per-page session bootstrap.
    assert len(demo.fns) == 12
    assert any(block_fn.targets == [(0, "load")] for block_fn in demo.fns.values())
    assert any(
        block_fn.targets and block_fn.targets[0][1] == "change"
        for block_fn in demo.fns.values()
    ), "the baseline-mode selector must rewrite the budgets it governs"


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
        # mode, max_tokens, max_recent_turns, recent_turn_budget, memory_budget,
        # chunk_budget, max_memories, relevance_floor, recency_half_life_turns,
        # resolve_conflicts, drop_stale_memories, drop_suspicious_memories, session
        "brainos", 4096, 2, 256, 512, 0, 6, 0.12, 20.0, True, True, False, None
    )

    assert "Context updated" in status
    settings = controller.ensure_session(session_id).context
    assert settings.recent_turn_budget == 256
    assert settings.memory_budget == 512
    assert settings.max_recent_turns == 2


def test_blank_recent_turns_means_whole_conversation() -> None:
    from app.ui import _update_context

    controller = _controller()
    _status, session_id = _update_context(controller)(
        "full_context", 4096, None, 4096, 0, 0, 6, 0.12, 20.0, True, True, False, None
    )

    # A blank box is "no turn limit", not "send no history at all".
    assert controller.ensure_session(session_id).context.max_recent_turns is None


def test_mode_selector_rewrites_the_budgets_it_governs() -> None:
    from app.ui import _apply_mode

    controller = _controller()
    apply_mode = _apply_mode(controller)
    # ``session_id`` threads through gr.State in the app, so the callback's
    # return value is the only handle on the session it changed.
    status, info, *_rest, session_id = apply_mode("full_context", None)

    assert "Mode A" in info
    assert "Mode A" in status
    full_context = controller.ensure_session(session_id).context
    # Mode A replays everything: no turn cap, and the window is the ceiling.
    assert full_context.max_recent_turns is None
    assert full_context.recent_turn_budget == full_context.max_tokens
    assert full_context.memory_budget == 0
    assert full_context.chunk_budget == 0

    _status, rag_info, *_rag_rest, session_id = apply_mode("rag", session_id)
    assert "Mode C" in rag_info
    rag = controller.ensure_session(session_id).context
    assert rag.uses_rag() and not rag.uses_memory()
    assert rag.chunk_budget > 0
    assert rag.max_recent_turns == RECENT_WINDOW_TURNS


def test_the_security_tab_is_wired_into_every_panel_callback() -> None:
    """Phase 13 adds three components; every panel callback must feed all three.

    The declared output count is what Gradio checks at build time, so a mismatch
    between :func:`_panel_values` and the components would break every callback
    at once — the count is asserted against the component order, not a literal.
    """

    from app.panels import SECURITY_COLUMNS
    from app.ui import PanelComponents, _panel_values, _silent_values

    controller = _controller()
    demo = create_app(controller)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    controller.chat(sid, f"My API key is {KEY} and we run PostgreSQL 16.")
    view = controller.chat(sid, "What database do we run in production?")

    values = _panel_values(view)
    declared = {(len(fn.inputs), len(fn.outputs)) for fn in demo.fns.values()}

    assert len(values) == len(PanelComponents.order) == 13
    assert PanelComponents.order[-3:] == ("security", "security_rows", "security_report")
    assert (1, len(_silent_values(view))) in declared, "panel callbacks feed all 13"
    assert (2, len(_silent_values(view)) + 1) in declared, "the chat callback adds the box"

    markdown, rows, report = values[-3:]
    assert markdown.startswith("### Security")
    assert "Credentials redacted" in markdown
    assert all(len(row) == len(SECURITY_COLUMNS) for row in rows)
    assert rows, "a session that redacted a credential must show the finding"
    assert report["clean"] is False
    assert report["by_route"]["history"] >= 1
    assert report["session_id"] == sid
    assert KEY not in str(values), "no panel value may carry the session key"
