"""Secure deletion: a delete must remove the bytes, not just the rows.

The plan's user-data controls ("clear conversation", "clear memory", "end
session") are only real if the database file stops containing the removed
content. SQLite's default behaviour is to unlink rows and leave the freed page
in the file, where the text is still readable with a hex editor. Two mechanisms
close that, and both are tested here:

* ``PRAGMA secure_delete=ON`` zeroes freed content. The pragma is a property of
  a *connection*, so :meth:`storage.sqlite.SqliteStore._connect` sets it on every
  one — a connection that forgot it would silently leave recoverable bytes.
* ``VACUUM`` rewrites the file without its freed pages, so the removed content is
  not merely zeroed but gone, and the file shrinks.

The application wiring matters as much as the primitives: the controller vacuums
after every user-data deletion, and a vacuum failure must never turn a successful
delete into an error the user sees.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app.controller import UIController
from brain.adapter import BrainOSAdapter
from storage.conversations import ConversationMessage
from storage.sqlite import (
    SqliteConversationStore,
    SqliteEvaluationStore,
    SqliteMemoryStore,
    SqliteStore,
)
from tests.fakes import FakeLLMProvider, FakeRuntime, InMemoryConversationStore, InMemoryMemoryStore

#: Unique, long enough to survive nowhere by accident.
MARKER = "MARKER-atlas-production-database-PostgreSQL-16-9f3c2b"
OTHER_MARKER = "MARKER-other-session-rollout-check-4d8e1a"
SESSION = "session-a"
OTHER_SESSION = "session-b"


def message(text: str, *, session_id: str = SESSION) -> ConversationMessage:
    return ConversationMessage(
        session_id=session_id,
        conversation_id=f"{session_id}-main",
        role="user",
        content=text,
        metadata={"turn": 1},
    )


def _footprint(database: Path) -> int:
    """Total bytes in SQLite's footprint for ``database``.

    Under WAL journal mode (Phase 14) pages accumulate in the ``-wal`` sidecar
    until a checkpoint moves them into the main file, so measuring only the
    main file lies in both directions: the main file can be 4 KB while the WAL
    holds the data, and after a TRUNCATE checkpoint the WAL is zero bytes
    while the main file holds the records. Summing the main file, WAL, and
    SHM gives a journal-mode-independent size for secure-deletion assertions.
    """

    total = database.stat().st_size if database.exists() else 0
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.exists():
            total += sidecar.stat().st_size
    return total


def raw(database: Path) -> bytes:
    """Return every byte SQLite can use to hold a page.

    Phase 14 switched the stores to WAL journal mode so concurrent Gradio
    worker threads don't lock the file. Under WAL the database's `-wal` and
    `-shm` sidecar files hold recent writes until a checkpoint moves them
    into the main file — a raw-read that ignored them would conclude "gone"
    while the bytes were still sitting in the WAL, or "present" while the
    main file was still empty. Reading the main file, the WAL, and the shm
    together makes the secure-deletion assertion honest across both journal
    modes.
    """

    pieces = [database.read_bytes() if database.exists() else b""]
    for suffix in ("-wal", "-shm"):
        sidecar = database.with_name(database.name + suffix)
        if sidecar.exists():
            pieces.append(sidecar.read_bytes())
    return b"".join(pieces)


class VacuumSpy(SqliteConversationStore):
    """A real store that also counts how often it was asked to vacuum."""

    def __init__(self, database_path: Path | str | None = None) -> None:
        super().__init__(database_path)
        self.vacuum_calls = 0

    def vacuum(self) -> bool:
        self.vacuum_calls += 1
        return super().vacuum()


class ExplodingVacuumStore(SqliteConversationStore):
    """A store whose vacuum fails, to prove the delete still succeeds."""

    def vacuum(self) -> bool:
        raise RuntimeError(f"vacuum unavailable for {MARKER}")


def controller_with(
    tmp_path: Path,
    *,
    conversation_store: Any,
    memory_store: Any = None,
    evaluation_store: Any = None,
) -> tuple[UIController, str]:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(FakeRuntime(str(kwargs["session_id"])))

    controller = UIController(
        provider_factory=FakeLLMProvider,
        adapter_factory=adapter_factory,
        conversation_store=conversation_store,
        memory_store=memory_store,
        evaluation_store=evaluation_store,
    )
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=""
    ).session_id
    return controller, session_id


# --------------------------------------------------------------------------- #
# The pragma
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "store_class", [SqliteStore, SqliteConversationStore, SqliteMemoryStore, SqliteEvaluationStore]
)
def test_every_store_opens_with_secure_delete_on(
    tmp_path: Path, store_class: type[SqliteStore]
) -> None:
    store = store_class(tmp_path / "lab.sqlite3")

    assert store.secure_delete_enabled() is True


def test_the_pragma_is_a_connection_property_the_store_sets_every_time(
    tmp_path: Path,
) -> None:
    """The setting lives on the connection, so the store must assert it per open.

    Some SQLite builds compile ``SECURE_DELETE`` in (this one does), which makes
    the default ``ON`` and hides the mistake. Turning the pragma off on a plain
    connection shows the real semantics — and that the store's own connections do
    not inherit that off state.
    """

    database = tmp_path / "lab.sqlite3"
    store = SqliteConversationStore(database)
    store.append(message(MARKER))

    plain = sqlite3.connect(str(database))
    try:
        plain.execute("PRAGMA secure_delete=OFF")
        assert plain.execute("PRAGMA secure_delete").fetchone()[0] == 0
    finally:
        plain.close()

    assert store.secure_delete_enabled() is True, "the store sets it on every connection"
    # A second store object on the same file agrees, so nothing depends on which
    # instance happened to open the database first.
    assert SqliteConversationStore(database).secure_delete_enabled() is True


def test_secure_delete_is_not_just_a_reported_flag(tmp_path: Path) -> None:
    """The flag is worth nothing unless deleted content is actually zeroed.

    This is the behavioural assertion the pragma exists for, and it holds on a
    build without ``SECURE_DELETE`` compiled in too, because the store sets the
    pragma itself.
    """

    database = tmp_path / "lab.sqlite3"
    store = SqliteConversationStore(database)
    store.append(message(MARKER))
    assert MARKER.encode() in raw(database), "the fixture must be on disk to prove anything"

    store.delete_session(SESSION)

    assert MARKER.encode() not in raw(database)
    assert store.list_messages(SESSION, f"{SESSION}-main") == []


# --------------------------------------------------------------------------- #
# Vacuum
# --------------------------------------------------------------------------- #


def test_vacuum_reclaims_the_pages_a_deletion_freed(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    store = SqliteConversationStore(database)
    for index in range(200):
        store.append(message(f"{MARKER} line {index} " + "padding " * 12))
    # Force a checkpoint so all of the inserted data is in the main file
    # before we measure; otherwise the WAL holds it and ``st_size`` undercounts.
    with store._connect() as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size_before = _footprint(database)

    store.delete_session(SESSION)
    assert store.vacuum() is True

    size_after = _footprint(database)
    assert size_after < size_before, "freed pages are returned to the filesystem"
    assert MARKER.encode() not in raw(database)


def test_vacuum_is_safe_to_repeat_and_on_an_empty_database(tmp_path: Path) -> None:
    store = SqliteConversationStore(tmp_path / "lab.sqlite3")
    store.append(message(MARKER))

    assert store.vacuum() is True
    assert store.vacuum() is True

    empty = SqliteConversationStore(tmp_path / "empty.sqlite3")
    assert empty.vacuum() is False, "no file, nothing to rewrite"
    empty.append(message("creates the file"))
    assert empty.vacuum() is True


def test_vacuum_reports_failure_instead_of_raising_on_a_corrupt_file(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corrupt.sqlite3"
    database.write_bytes(b"this is not a SQLite database at all" * 40)

    assert SqliteConversationStore(database).vacuum() is False


def test_vacuum_never_changes_what_a_store_can_still_read(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    store = SqliteConversationStore(database)
    store.append(message(MARKER))
    store.append(message(OTHER_MARKER, session_id=OTHER_SESSION))

    store.delete_session(SESSION)
    assert store.vacuum() is True

    kept = store.list_messages(OTHER_SESSION, f"{OTHER_SESSION}-main")
    assert [item.content for item in kept] == [OTHER_MARKER]
    assert OTHER_MARKER.encode() in raw(database), "other sessions are untouched"
    assert MARKER.encode() not in raw(database)


def test_cleared_memory_bytes_do_not_survive_in_the_file(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    store = SqliteMemoryStore(database)
    store.save_memories(
        SESSION, [{"memory_id": "m1", "text": MARKER, "memory_type": "fact"}]
    )
    assert MARKER.encode() in raw(database)

    store.clear(SESSION)
    assert store.vacuum() is True

    assert store.list_memories(SESSION) == []
    assert MARKER.encode() not in raw(database)


def test_deleted_evaluation_rows_do_not_survive_in_the_file(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    store = SqliteEvaluationStore(database)
    store.save_run(
        "run-1",
        {"session_id": SESSION, "note": MARKER},
        {"answer_accuracy": 1.0},
    )
    store.save_run("run-2", {"session_id": OTHER_SESSION, "note": OTHER_MARKER}, {})
    assert MARKER.encode() in raw(database)

    store.delete_session_data(SESSION)
    assert store.vacuum() is True

    assert store.get_run("run-1") is None
    assert store.get_run("run-2") is not None
    assert MARKER.encode() not in raw(database)
    assert OTHER_MARKER.encode() in raw(database)


# --------------------------------------------------------------------------- #
# Application wiring: the controller vacuums after every user-data deletion
# --------------------------------------------------------------------------- #


def test_the_controller_vacuums_after_each_user_data_control(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    conversations = VacuumSpy(database)
    memories = SqliteMemoryStore(database)
    controller, session_id = controller_with(
        tmp_path, conversation_store=conversations, memory_store=memories
    )

    controller.chat(session_id, f"We run {MARKER} in production.")
    assert MARKER.encode() in raw(database)

    controller.clear_memory(session_id)
    after_memory = conversations.vacuum_calls

    controller.clear_conversation(session_id)
    after_conversation = conversations.vacuum_calls

    controller.end_session(session_id)
    after_session = conversations.vacuum_calls

    assert after_memory >= 1, "clear memory vacuums"
    assert after_conversation > after_memory, "clear conversation vacuums"
    assert after_session > after_conversation, "end session vacuums"
    assert MARKER.encode() not in raw(database)


def test_the_evaluation_store_is_vacuumed_too(tmp_path: Path) -> None:
    database = tmp_path / "lab.sqlite3"
    evaluations = SqliteEvaluationStore(database)
    evaluations.save_run("run-1", {"session_id": SESSION, "note": MARKER}, {})
    controller, _session_id = controller_with(
        tmp_path,
        conversation_store=SqliteConversationStore(database),
        memory_store=SqliteMemoryStore(database),
        evaluation_store=evaluations,
    )

    controller._services.clear()
    controller.end_session(SESSION)

    assert evaluations.get_run("run-1") is None
    assert MARKER.encode() not in raw(database)


def test_a_failing_vacuum_never_turns_a_delete_into_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hardening is best-effort; the user's delete still has to work."""

    database = tmp_path / "lab.sqlite3"
    controller, session_id = controller_with(
        tmp_path, conversation_store=ExplodingVacuumStore(database)
    )
    controller.chat(session_id, f"We run {MARKER} in production.")

    view = controller.end_session(session_id)

    assert "Session ended" in view.status
    assert SqliteConversationStore(database).list_messages(session_id, f"{session_id}-main") == []
    logged = capsys.readouterr().err
    assert "vacuum failed" in logged
    assert "RuntimeError" in logged
    assert MARKER not in logged, "a storage warning never echoes stored content"


def test_stores_without_a_vacuum_are_skipped_silently(tmp_path: Path) -> None:
    """The in-memory doubles the rest of the suite runs on have no database file."""

    assert not hasattr(InMemoryConversationStore(), "vacuum")
    controller, session_id = controller_with(
        tmp_path,
        conversation_store=InMemoryConversationStore(),
        memory_store=InMemoryMemoryStore(),
    )
    controller.chat(session_id, f"We run {MARKER} in production.")

    view = controller.clear_conversation(session_id)

    assert "cleared" in view.status.lower()
    assert list(tmp_path.iterdir()) == [], "no database file was created at all"
