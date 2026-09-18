"""Phase 5 security tests: what may reach disk.

The plan's storage rule is absolute — never store keys in the database. These
tests assert it at the strongest level available: the raw bytes of the SQLite
file, the mirrored rows, and the session export, after a session that actively
tried to persist its credential (key typed into chat, key smuggled into
metadata). The harness is the upstream ``UIController`` with the real SQLite
backends and fake provider/runtime seams.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.controller import UIController
from brain.adapter import BrainOSAdapter
from security.scan import SQLITE_SIDECAR_SUFFIXES
from storage.conversations import ConversationMessage
from storage.sqlite import SqliteConversationStore, SqliteMemoryStore
from tests.fakes import FakeLLMProvider, FakeRuntime


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "lab.sqlite3"


def _adapter_factory(**kwargs: object) -> BrainOSAdapter:
    return BrainOSAdapter(
        FakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
    )


def make_sqlite_controller(db: Path, api_key: str) -> tuple[UIController, str]:
    """Controller on the real SQLite backends with one connected session."""

    controller = UIController(
        provider_factory=FakeLLMProvider,
        adapter_factory=_adapter_factory,
        conversation_store=SqliteConversationStore(db),
        memory_store=SqliteMemoryStore(db),
    )
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=api_key
    ).session_id
    return controller, session_id


def database_bytes(db: Path) -> bytes:
    """Every byte this database owns: the file plus its WAL/journal sidecars.

    Phase 14 runs SQLite in WAL mode, so committed rows can still be sitting in
    ``<db>-wal`` and the main file can be one page long. Reading only the main
    file made this test depend on *when the last connection happened to close*
    (an auto-checkpoint merges the WAL then): it passed alone and failed after
    unrelated tests, while the redaction it checks was correct in both cases.
    Checkpointing first and reading the sidecars too removes the ordering
    dependency and makes the claim stronger — no credential in any file this
    database writes, which is also what ``security.scan`` now covers.
    """

    connection = sqlite3.connect(db)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    files = [db, *(Path(f"{db}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES)]
    return b"".join(path.read_bytes() for path in files if path.exists())


def test_raw_database_bytes_never_contain_the_session_key(db: Path) -> None:
    key = "sk-session-do-not-persist-777"
    controller, session_id = make_sqlite_controller(db, key)
    # The worst case: the user types their live credential into the chat, and
    # BrainOS stores the sentence as a memory.
    controller.chat(session_id, f"My provider credential is {key} for this account.")
    controller.chat(session_id, "What is my provider credential?")

    raw = database_bytes(db)
    assert key.encode() not in raw
    assert b"[redacted]" in raw  # the transcript stays, the key does not

    memories = SqliteMemoryStore(db).list_memories(session_id)
    assert memories, "the sentence itself is still mirrored"
    assert all(key not in json.dumps(record) for record in memories)


def test_export_and_every_view_are_key_free(db: Path) -> None:
    key = "export-secret-key-555"
    controller, session_id = make_sqlite_controller(db, key)
    controller.chat(session_id, f"Remember that the deploy credential is {key}.")
    view = controller.chat(session_id, "What was the deploy credential again?")

    # Disk and downloads are absolutely key-free…
    payload = controller.export_session(session_id)
    assert key not in json.dumps(payload, default=str)
    assert key not in controller.export_text(session_id)
    text_payload = json.loads(controller.export_text(session_id))
    assert text_payload["persistence"]["enabled"] is True
    assert key not in json.dumps(text_payload["persistence"], default=str)

    # …and so is every rendered surface. The upstream controller sanitizes
    # each TurnView field against the session key, so unlike earlier designs
    # even the transcript view is scrubbed.
    surfaces = {
        "history": view.history,
        "status": view.status,
        "stored_rows": view.stored_rows,
        "retrieved_rows": view.retrieved_rows,
        "dropped_rows": view.dropped_rows,
        "conflict_rows": view.conflict_rows,
        "stats": view.stats,
        "summary": view.summary,
        "prompt": view.prompt,
        "trace": view.trace,
        "trace_events": view.trace_events,
    }
    assert key not in json.dumps(surfaces, default=str)
    assert any("[redacted]" in str(row) for row in view.stored_rows)


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
    controller, session_id = make_sqlite_controller(db, "temporary-key-321")
    controller.chat(session_id, "The production database is PostgreSQL 16.")
    other = SqliteConversationStore(db)
    other.append(
        ConversationMessage(
            session_id="other", conversation_id="c", role="user", content="kept"
        )
    )
    assert SqliteConversationStore(db).list_conversations(session_id)
    assert SqliteMemoryStore(db).list_memories(session_id)

    controller.end_session(session_id)
    # The isolation contract is API-level: no query can reach the deleted
    # session's rows. (SQLite may keep freed pages until VACUUM; physical
    # scrubbing is a documented Phase 13 hardening candidate.)
    assert SqliteConversationStore(db).list_conversations(session_id) == []
    assert SqliteMemoryStore(db).list_memories(session_id) == []
    # Another session's rows are untouched by the delete.
    assert [row.content for row in other.list_messages("other", "c")] == ["kept"]
