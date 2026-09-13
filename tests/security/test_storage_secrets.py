"""Phase 5 security tests: what may reach disk.

The plan's storage rule is absolute — never store keys in the database. These
tests assert it at the strongest level available: the raw bytes of the SQLite
file, the mirrored rows, and the session export, after a session that actively
tried to persist its credential (key typed into chat, key smuggled into
metadata).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.controller import ChatController
from app.service import ConversationService
from app.state import SessionState
from brain.adapter import BrainOSAdapter
from providers.base import ProviderConfig
from storage.conversations import ConversationMessage
from storage.sqlite import SqliteConversationStore, SqliteMemoryStore
from tests.fakes import FakeProvider, FakeRuntime


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "lab.sqlite3"


def make_sqlite_controller(db: Path, api_key: str) -> ChatController:
    state = SessionState(provider=ProviderConfig(model="fake-model", api_key=api_key))

    def factory(session_state: SessionState) -> ConversationService:
        return ConversationService(
            session_state,
            adapter=BrainOSAdapter(FakeRuntime()),
            provider=FakeProvider(),
        )

    return ChatController(
        state,
        service_factory=factory,
        conversation_store=SqliteConversationStore(db),
        memory_store=SqliteMemoryStore(db),
    )


def test_raw_database_bytes_never_contain_the_session_key(db: Path) -> None:
    key = "sk-session-do-not-persist-777"
    controller = make_sqlite_controller(db, key)
    # The worst case: the user types their live credential into the chat, and
    # BrainOS stores the sentence as a memory.
    controller.send_message(f"My provider credential is {key} for this account.")
    controller.send_message("What is my provider credential?")

    raw = db.read_bytes()
    assert key.encode() not in raw
    assert b"[redacted]" in raw  # the transcript stays, the key does not

    memories = SqliteMemoryStore(db).list_memories(controller.state.session_id)
    assert memories, "the sentence itself is still mirrored"
    assert all(key not in json.dumps(record) for record in memories)


def test_export_and_views_never_contain_the_session_key(db: Path) -> None:
    key = "export-secret-key-555"
    controller = make_sqlite_controller(db, key)
    controller.apply_provider(api_key=key, model="fake-model")
    controller.send_message(f"Remember that the deploy credential is {key}.")
    controller.validate_connection()

    # Disk and downloads are absolutely key-free…
    payload = controller.export_session()
    assert key not in json.dumps(payload, default=str)
    path, _ = controller.export_session_file()
    assert key not in Path(path).read_text(encoding="utf-8")

    # …and so is every diagnostic surface. The chat transcript and the final
    # prompt are the documented faithful surfaces: they show the user what
    # they typed and what was sent to their own provider (Phase 4 decision),
    # while the key is redacted from everything derived *from* memory.
    view = controller.views()
    diagnostic_surfaces = {
        surface: view[surface]
        for surface in (
            "status",
            "stored_rows",
            "retrieved_rows",
            "ranking_json",
            "context_summary",
            "trace_rows",
            "decision_md",
            "dropped_rows",
            "conflict_rows",
            "connection_status",
            "provider_summary",
        )
    }
    assert key not in json.dumps(diagnostic_surfaces, default=str)
    assert any("[redacted]" in str(row) for row in view["stored_rows"])


def test_metadata_secret_fields_cannot_be_smuggled_onto_disk(db: Path) -> None:
    store = SqliteConversationStore(db)
    store.append(
        ConversationMessage(
            session_id="s-1",
            conversation_id="c-1",
            role="user",
            content="payload",
            metadata={"api_key": "smuggled", "note": {"authorization": "Bearer x"}},
        )
    )
    row = store.list_messages("s-1", "c-1")[0]
    dumped = json.dumps(row.metadata)
    assert "smuggled" not in dumped
    assert "Bearer x" not in dumped
    assert row.metadata == {"note": {}}


def test_deleted_session_data_is_gone_from_the_store(db: Path) -> None:
    controller = make_sqlite_controller(db, "temporary-key-321")
    controller.send_message("The production database is PostgreSQL 16.")
    session_id = controller.state.session_id
    assert SqliteConversationStore(db).list_conversations(session_id)
    assert SqliteMemoryStore(db).list_memories(session_id)

    controller.end_session()
    # The isolation contract is API-level: no query can reach the deleted
    # session's rows. (SQLite may keep freed pages until VACUUM; physical
    # scrubbing is a documented Phase 13 hardening candidate.)
    assert SqliteConversationStore(db).list_conversations(session_id) == []
    assert SqliteMemoryStore(db).list_memories(session_id) == []
    # Other sessions are untouched by the delete.
    other = SqliteConversationStore(db)
    other.append(
        ConversationMessage(
            session_id="other", conversation_id="c", role="user", content="kept"
        )
    )
    controller.end_session()
    assert [row.content for row in other.list_messages("other", "c")] == ["kept"]
