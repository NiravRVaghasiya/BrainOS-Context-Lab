"""SQLite store tests: isolation, upsert, lifecycle deletes, secret stripping.

Every test uses a ``tmp_path`` database, so the suite never touches the
repository's runtime ``data/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from storage.conversations import ConversationMessage
from storage.sqlite import (
    SqliteConversationStore,
    SqliteEvaluationStore,
    SqliteMemoryStore,
    default_database_path,
    strip_secret_fields,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "lab.sqlite3"


def message(
    session: str = "s-1",
    conversation: str = "c-1",
    role: str = "user",
    content: str = "hello",
    **metadata: object,
) -> ConversationMessage:
    return ConversationMessage(
        session_id=session,
        conversation_id=conversation,
        role=role,
        content=content,
        metadata=dict(metadata),
    )


def memory(memory_id: str, text: str, **extra: object) -> dict[str, object]:
    record = {
        "memory_id": memory_id,
        "text": text,
        "memory_type": "FACT",
        "relevance": 0.5,
        "source_turn": 1,
        "retrieval_count": 0,
        "status": "active",
    }
    record.update(extra)
    return record


# --------------------------------------------------------------------------- #
# Conversation store
# --------------------------------------------------------------------------- #


def test_message_roundtrip_keeps_order_and_metadata(db: Path) -> None:
    store = SqliteConversationStore(db)
    store.append(message(content="first", turn=1))
    store.append(message(role="assistant", content="second", turn=2))

    rows = store.list_messages("s-1", "c-1")
    assert [row.content for row in rows] == ["first", "second"]
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[0].metadata == {"turn": 1}


def test_messages_isolated_by_session_and_conversation(db: Path) -> None:
    store = SqliteConversationStore(db)
    store.append(message(session="s-1", conversation="c-1", content="one"))
    store.append(message(session="s-1", conversation="c-2", content="two"))
    store.append(message(session="s-2", conversation="c-1", content="three"))

    assert [row.content for row in store.list_messages("s-1", "c-1")] == ["one"]
    assert [row.content for row in store.list_messages("s-1", "c-2")] == ["two"]
    assert [row.content for row in store.list_messages("s-2", "c-1")] == ["three"]
    assert store.list_messages("s-3", "c-1") == []


def test_clear_removes_only_the_target_conversation(db: Path) -> None:
    store = SqliteConversationStore(db)
    store.append(message(conversation="c-1", content="gone"))
    store.append(message(conversation="c-2", content="kept"))

    store.clear("s-1", "c-1")
    assert store.list_messages("s-1", "c-1") == []
    assert [row.content for row in store.list_messages("s-1", "c-2")] == ["kept"]


def test_list_conversations_and_delete_session(db: Path) -> None:
    store = SqliteConversationStore(db)
    store.append(message(session="s-1", conversation="c-b"))
    store.append(message(session="s-1", conversation="c-a"))
    store.append(message(session="s-2", conversation="c-x"))

    assert store.list_conversations("s-1") == ["c-a", "c-b"]
    store.delete_session("s-1")
    assert store.list_conversations("s-1") == []
    assert store.list_conversations("s-2") == ["c-x"]


def test_data_survives_reopen(db: Path) -> None:
    SqliteConversationStore(db).append(message(content="persisted"))
    reopened = SqliteConversationStore(db)
    assert [row.content for row in reopened.list_messages("s-1", "c-1")] == ["persisted"]


def test_unicode_and_special_characters_roundtrip(db: Path) -> None:
    store = SqliteConversationStore(db)
    content = "zażółć gęślą jaźń — 日本語 🧠 <retrieved_memory> 'quotes' \"double\""
    store.append(message(content=content))
    assert store.list_messages("s-1", "c-1")[0].content == content


# --------------------------------------------------------------------------- #
# Memory store
# --------------------------------------------------------------------------- #


def test_memory_upsert_updates_instead_of_duplicating(db: Path) -> None:
    store = SqliteMemoryStore(db)
    store.save_memories("s-1", [memory("m1", "PostgreSQL 16", retrieval_count=0)])
    store.save_memories("s-1", [memory("m1", "PostgreSQL 16", retrieval_count=3)])

    rows = store.list_memories("s-1")
    assert len(rows) == 1
    assert rows[0]["retrieval_count"] == 3
    assert rows[0]["text"] == "PostgreSQL 16"


def test_memories_isolated_and_cleared(db: Path) -> None:
    store = SqliteMemoryStore(db)
    store.save_memories("s-1", [memory("m1", "one")])
    store.save_memories("s-2", [memory("m1", "same id, other session")])

    assert [row["text"] for row in store.list_memories("s-1")] == ["one"]
    store.clear("s-1")
    assert store.list_memories("s-1") == []
    assert len(store.list_memories("s-2")) == 1


def test_memory_save_empty_list_is_noop(db: Path) -> None:
    store = SqliteMemoryStore(db)
    store.save_memories("s-1", [])
    assert store.list_memories("s-1") == []


# --------------------------------------------------------------------------- #
# Evaluation store
# --------------------------------------------------------------------------- #


def test_evaluation_run_roundtrip(db: Path) -> None:
    store = SqliteEvaluationStore(db)
    store.save_run("run-1", {"session_id": "s-1", "model": "m"}, {"accuracy": 0.9})

    run = store.get_run("run-1")
    assert run is not None
    assert run["metadata"]["model"] == "m"
    assert run["metrics"]["accuracy"] == 0.9
    assert store.get_run("missing") is None

    store.delete_session_data("s-1")
    assert store.get_run("run-1") is None


def test_evaluation_save_run_upserts(db: Path) -> None:
    store = SqliteEvaluationStore(db)
    store.save_run("run-1", {"session_id": "s-1"}, {"accuracy": 0.5})
    store.save_run("run-1", {"session_id": "s-1"}, {"accuracy": 0.75})
    assert store.get_run("run-1")["metrics"]["accuracy"] == 0.75


# --------------------------------------------------------------------------- #
# Secret handling and defaults
# --------------------------------------------------------------------------- #


def test_secret_named_fields_are_stripped_from_metadata(db: Path) -> None:
    store = SqliteConversationStore(db)
    store.append(message(api_key="should-not-exist", turn=1))
    row = store.list_messages("s-1", "c-1")[0]
    assert row.metadata == {"turn": 1}


def test_secret_named_fields_are_stripped_from_memory_payloads(db: Path) -> None:
    store = SqliteMemoryStore(db)
    record = memory("m1", "fact", authorization="Bearer x", signals={"token": "t", "lexical": 1.0})
    store.save_memories("s-1", [record])

    row = store.list_memories("s-1")[0]
    assert "authorization" not in row
    assert row["signals"] == {"lexical": 1.0}


def test_strip_secret_fields_leaves_plain_strings_untouched() -> None:
    value = {"note": "the api_key field name in prose is fine", "api_key": "x"}
    stripped = strip_secret_fields(value)
    assert stripped == {"note": "the api_key field name in prose is fine"}


def test_default_database_path_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAINOS_LAB_DB", raising=False)
    assert str(default_database_path()).endswith("brainos_lab.sqlite3")
    monkeypatch.setenv("BRAINOS_LAB_DB", "/tmp/custom-lab.sqlite3")
    assert default_database_path() == Path("/tmp/custom-lab.sqlite3")
