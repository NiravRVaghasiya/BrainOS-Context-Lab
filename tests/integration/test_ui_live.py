"""Live Phase 4 check: the controller on the real pinned BrainOS runtime.

Skipped unless the integration extra is installed. This exercises the
production wiring — default service factory, real ``create_brain_adapter``,
no injected provider — the same path the Gradio app takes, without a browser
and without an API key.
"""

from __future__ import annotations

import pytest

pytest.importorskip("brainos_runtime")

from app.controller import ChatController  # noqa: E402
from app.session import SessionManager  # noqa: E402


def test_live_controller_roundtrip() -> None:
    manager = SessionManager()
    controller = ChatController(manager=manager)
    try:
        first = controller.send_message(
            "For Project Atlas, the production database is PostgreSQL 16."
        )
        assert "not installed" not in first["status"]
        assert first["stored_rows"], "the durable fact should be stored by the live runtime"

        second = controller.send_message("What database does Project Atlas use?")
        assert "No provider credentials" in second["status"]
        assert second["context_stats"]["selected_memory_count"] >= 1
        assert "<retrieved_memory>" in second["final_prompt"]
        assert any("PostgreSQL" in str(row[4]) for row in second["retrieved_rows"])
        assert second["trace_rows"]

        cleared = controller.clear_conversation()
        assert cleared["history"] == []
        assert cleared["stored_rows"] == []
    finally:
        manager.end(controller.state.session_id)


def test_live_sessions_are_isolated() -> None:
    manager = SessionManager()
    first = ChatController(manager=manager)
    second = ChatController(manager=manager)
    try:
        first.send_message("The retention policy requires 400 days of audit logs.")
        view = second.send_message("How long must audit logs be retained?")
        assert all("400 days" not in str(row) for row in view["retrieved_rows"])
        assert all("400 days" not in str(row) for row in view["stored_rows"])
    finally:
        manager.end(first.state.session_id)
        manager.end(second.state.session_id)


def test_live_controller_persists_to_sqlite(tmp_path, monkeypatch) -> None:
    """Production-shaped persistence: live runtime + SQLite store on disk."""

    monkeypatch.setenv("BRAINOS_LAB_DB", str(tmp_path / "live.sqlite3"))
    from storage.sqlite import SqliteConversationStore, SqliteMemoryStore

    conversation_store = SqliteConversationStore()
    memory_store = SqliteMemoryStore()
    manager = SessionManager()
    controller = ChatController(
        manager=manager,
        conversation_store=conversation_store,
        memory_store=memory_store,
    )
    try:
        controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")
        controller.send_message("What database does Project Atlas use?")

        session_id = controller.state.session_id
        assert (tmp_path / "live.sqlite3").exists()
        rows = conversation_store.list_messages(session_id, controller.state.conversation_id)
        assert [row.role for row in rows] == ["user", "user"]  # no provider → no replies
        mirrored = memory_store.list_memories(session_id)
        assert any("PostgreSQL" in record["text"] for record in mirrored)

        payload = controller.export_session()
        assert len(payload["transcript"]) == 2
        assert any("PostgreSQL" in record["text"] for record in payload["memories"])

        controller.end_session()
        assert conversation_store.list_conversations(session_id) == []
        assert memory_store.list_memories(session_id) == []
    finally:
        manager.end(controller.state.session_id)
