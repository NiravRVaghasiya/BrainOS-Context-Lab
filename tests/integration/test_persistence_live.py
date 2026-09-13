"""Live Phase 5 validation: SQLite persistence against the pinned BrainOS.

The unit tests cover the persistence seam with fakes and in-memory stores.
This file runs the production shape — the real runtime, the real SQLite
backends on disk (path via ``BRAINOS_LAB_DB``), and a deterministic provider
double — to prove that writes, mirrors, exports, and user-data deletes work
against the same runtime the shipped app uses. No provider API key is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.controller import UIController
from tests.fakes import FakeLLMProvider

pytest.importorskip("brainos_runtime")

FACT = "For Project Atlas, the production database is PostgreSQL 16."


def _provider_factory(config: object) -> FakeLLMProvider:
    # A neutral reply keeps assistant turns out of memory, so the mirrors show
    # what the *user* said rather than what the fake model answered.
    return FakeLLMProvider(config, text="Noted.")  # type: ignore[arg-type]


def _live_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Controller on real SQLite stores rooted in a temp database."""

    monkeypatch.setenv("BRAINOS_LAB_DB", str(tmp_path / "live.sqlite3"))
    from storage.sqlite import SqliteConversationStore, SqliteMemoryStore

    controller = UIController(
        provider_factory=_provider_factory,
        conversation_store=SqliteConversationStore(),
        memory_store=SqliteMemoryStore(),
    )
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=""
    ).session_id
    return controller, session_id, tmp_path / "live.sqlite3"


def test_live_controller_persists_to_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, session_id, db_path = _live_controller(tmp_path, monkeypatch)
    controller.chat(session_id, FACT)
    controller.chat(session_id, "What database does Project Atlas use?")

    assert db_path.exists()
    from storage.sqlite import SqliteConversationStore, SqliteMemoryStore

    conversation_store = SqliteConversationStore(db_path)
    memory_store = SqliteMemoryStore(db_path)

    state = controller.sessions.get(session_id)
    rows = conversation_store.list_messages(session_id, state.conversation_id)
    assert [row.role for row in rows] == ["user", "user"]  # no key → no replies
    mirrored = memory_store.list_memories(session_id)
    assert any("PostgreSQL" in record["text"] for record in mirrored)

    payload = json.loads(controller.export_text(session_id))
    assert len(payload["messages"]) == 2
    assert payload["persistence"]["enabled"] is True
    conversations = payload["persistence"]["conversations"]
    assert len(conversations) == 1
    assert len(conversations[0]["messages"]) == 2
    assert any(
        "PostgreSQL" in record["text"]
        for record in payload["persistence"]["memories"]
    )

    ended = controller.end_session(session_id)
    assert "Session ended" in ended.status
    assert conversation_store.list_conversations(session_id) == []
    assert memory_store.list_memories(session_id) == []


def test_live_clear_memory_empties_the_mirror_but_keeps_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, session_id, db_path = _live_controller(tmp_path, monkeypatch)
    controller.chat(session_id, FACT)

    view = controller.clear_memory(session_id)
    assert view.stored_rows == []
    assert [message["role"] for message in view.history] == ["user"]

    from storage.sqlite import SqliteConversationStore, SqliteMemoryStore

    assert SqliteMemoryStore(db_path).list_memories(session_id) == []
    state = controller.sessions.get(session_id)
    rows = SqliteConversationStore(db_path).list_messages(
        session_id, state.conversation_id
    )
    assert [row.role for row in rows] == ["user"], "the transcript rows survive"
