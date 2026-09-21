"""Phase 19: the multi-visitor claim, exercised with real threads.

Phase 14 put the SQLite stores into WAL mode with a 30-second busy timeout and
argued that a public Space's concurrent Gradio workers would therefore
serialize instead of raising ``database is locked``. Nothing in the suite ever
ran two writers at once — the claim was in the deployment tests as a comment and
in this phase's hand-off as an open gap ("a multi-threaded stress test").

These tests close it. They are deliberately blunt: many threads, one database,
every store, and an assertion at the end that no row was lost, no row crossed a
session boundary, and the WAL was checkpointed back into the main file (which is
what makes Phase 13's byte-level secure deletion hold under load).

They need no BrainOS runtime, no Gradio, and no network — the point is the
storage layer's behaviour under concurrency, not a provider's.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from storage.conversations import ConversationMessage
from storage.sqlite import (
    SqliteConversationStore,
    SqliteEvaluationStore,
    SqliteMemoryStore,
)

THREADS = 8
WRITES_PER_THREAD = 25
OPENS = 16


def _run_threads(worker: Callable[[int], None], count: int = THREADS) -> list[BaseException]:
    """Run ``worker(index)`` in ``count`` threads at once; return any failures."""

    barrier = threading.Barrier(count)
    failures: list[BaseException] = []
    lock = threading.Lock()

    def target(index: int) -> None:
        try:
            barrier.wait(timeout=30)  # maximize overlap; nothing starts early
            worker(index)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            with lock:
                failures.append(exc)

    threads = [
        threading.Thread(target=target, args=(index,), name=f"writer-{index}")
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "a writer thread never finished"
    return failures


def test_concurrent_appends_lose_nothing_and_stay_in_their_session(tmp_path: Path) -> None:
    """Eight threads writing their own session's transcript, at the same instant."""

    store = SqliteConversationStore(tmp_path / "load.sqlite3")

    def worker(index: int) -> None:
        session = f"session-{index}"
        for turn in range(WRITES_PER_THREAD):
            store.append(
                ConversationMessage(
                    session_id=session,
                    conversation_id="c1",
                    role="user",
                    content=f"thread {index} turn {turn}",
                )
            )

    failures = _run_threads(worker)

    assert failures == [], failures
    for index in range(THREADS):
        messages = store.list_messages(f"session-{index}", "c1")
        assert len(messages) == WRITES_PER_THREAD
        assert {message.content for message in messages} == {
            f"thread {index} turn {turn}" for turn in range(WRITES_PER_THREAD)
        }
        assert all(message.session_id == f"session-{index}" for message in messages)


def test_concurrent_memory_and_evaluation_writes_share_one_database(tmp_path: Path) -> None:
    """The three stores, on one file, from three groups of threads.

    This is the shape a busy Space has: a chat turn writing a transcript and a
    memory mirror while another visitor's benchmark run records itself.
    """

    database = tmp_path / "shared.sqlite3"
    conversations = SqliteConversationStore(database)
    memories = SqliteMemoryStore(database)
    evaluations = SqliteEvaluationStore(database)

    def worker(index: int) -> None:
        session = f"session-{index}"
        conversations.append(
            ConversationMessage(session_id=session, conversation_id="c1", role="user", content="hi")
        )
        memories.save_memories(
            session,
            [
                {
                    "memory_id": f"{session}-m{turn}",
                    "content": f"fact {turn} for {session}",
                    "type": "semantic",
                }
                for turn in range(WRITES_PER_THREAD)
            ],
        )
        evaluations.save_run(
            f"run-{index}",
            {"session_id": session},
            {"tokens": WRITES_PER_THREAD},
        )

    failures = _run_threads(worker)

    assert failures == [], failures
    assert len(conversations.list_conversations("session-3")) == 1
    assert len(memories.list_memories("session-3")) == WRITES_PER_THREAD
    assert evaluations.get_run("run-3") is not None
    assert evaluations.get_run("run-3")["session_id"] == "session-3"
    assert evaluations.get_run("run-3")["metrics"] == {"tokens": WRITES_PER_THREAD}


def test_deleting_one_session_under_load_does_not_touch_the_others(tmp_path: Path) -> None:
    """A deletion racing a writer must not take rows from a third session."""

    database = tmp_path / "delete-under-load.sqlite3"
    store = SqliteConversationStore(database)
    for index in range(THREADS):
        store.append(
            ConversationMessage(
                session_id=f"session-{index}", conversation_id="c1", role="user", content="seed"
            )
        )

    def worker(index: int) -> None:
        session = f"session-{index}"
        for _ in range(WRITES_PER_THREAD):
            store.append(
                ConversationMessage(
                    session_id=session, conversation_id="c1", role="user", content="more"
                )
            )
        store.clear(session, "c1")
        store.append(
            ConversationMessage(
                session_id=session, conversation_id="c1", role="user", content="after-clear"
            )
        )

    failures = _run_threads(worker)

    assert failures == [], failures
    for index in range(THREADS):
        messages = store.list_messages(f"session-{index}", "c1")
        assert [message.content for message in messages] == ["after-clear"]


def test_a_delete_under_load_completes_when_the_writers_stop(tmp_path: Path) -> None:
    """Phase 13's secure deletion, racing Phase 14's concurrency.

    ``secure_delete`` plus a truncating checkpoint after a destructive write is
    what makes "deleted" mean "gone from the file". Racing writers make that
    checkpoint *best effort* — SQLite reports a busy checkpoint instead of
    performing it — so the honest invariants are the two asserted here: the rows
    are gone while the load is running, and once the store is quiescent the
    deleted bytes are gone from the database file and every sidecar. A test that
    demanded both at once would be asserting the outcome of a race.
    """

    database = tmp_path / "secure.sqlite3"
    store = SqliteConversationStore(database)
    marker = "session-a-DELETE-ME-42-e7f3a1c9"

    def worker(index: int) -> None:
        if index == 0:
            for _ in range(WRITES_PER_THREAD):
                store.append(
                    ConversationMessage(
                        session_id="session-a",
                        conversation_id="c1",
                        role="user",
                        content=marker * 40,
                    )
                )
            store.clear("session-a", "c1")
            store.vacuum()
            return
        for turn in range(WRITES_PER_THREAD):
            store.append(
                ConversationMessage(
                    session_id=f"session-{index}",
                    conversation_id="c1",
                    role="user",
                    content=f"noise {index}-{turn}",
                )
            )

    failures = _run_threads(worker)

    assert failures == [], failures
    with sqlite3.connect(str(database)) as connection:
        rows = connection.execute(
            "SELECT content FROM messages WHERE session_id = 'session-a'"
        ).fetchall()
    assert rows == [], "the deleted session's rows survived the racing writers"

    # Quiescent: a checkpoint can now complete, and the bytes must be gone.
    assert store.vacuum() is True
    payload = marker.encode("utf-8")
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(f"{database}{suffix}")
        if candidate.exists():
            assert payload not in candidate.read_bytes(), (
                f"deleted content survived in {candidate.name}"
            )


def test_concurrent_first_opens_do_not_race_the_journal_mode_switch(tmp_path: Path) -> None:
    """The WAL switch is a file-level fact, not a per-connection duty.

    ``PRAGMA journal_mode=WAL`` needs a schema lock, and SQLite does *not* apply
    ``busy_timeout`` to it — so issuing it from every connection (which the store
    did) let two workers opening a fresh database raise ``database is locked``
    before doing any work. The store now reads the mode first and only switches
    when the file is not already WAL. Each round throws a fresh database at a
    barrier-synchronised crowd of openers, which is the case that used to fail.
    """

    failures: list[BaseException] = []
    lock = threading.Lock()

    for round_index in range(3):
        database = tmp_path / f"fresh-{round_index}.sqlite3"
        store = SqliteConversationStore(database)
        barrier = threading.Barrier(OPENS)

        def opener(index: int) -> None:
            try:
                barrier.wait(timeout=30)
                connection = store._connect()  # the seam under test
                try:
                    connection.execute("SELECT 1").fetchone()
                finally:
                    connection.close()
                # A write is the operation a worker actually performs first.
                store.append(
                    ConversationMessage(
                        session_id=f"session-{index}",
                        conversation_id="c1",
                        role="user",
                        content="opened",
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                with lock:
                    failures.append(exc)

        threads = [
            threading.Thread(target=opener, args=(index,), name=f"opener-{index}")
            for index in range(OPENS)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert failures == [], failures
        assert store.journal_mode() == "wal"
        for index in range(OPENS):
            assert len(store.list_messages(f"session-{index}", "c1")) == 1


class _ScriptedConnection:
    """A connection stand-in that records pragmas and can fail on demand.

    ``_ensure_wal`` only needs ``execute``/``fetchone``, so the failure branches
    can be driven exactly instead of hoping a thread race hits them.
    """

    def __init__(self, *, journal_mode: str, failures: int = 0) -> None:
        self.journal_mode = journal_mode
        self.failures = failures
        self.statements: list[str] = []

    def execute(self, statement: str) -> _Cursor:  # type: ignore[override]
        text = " ".join(statement.strip().lower().split())
        self.statements.append(text)
        if text == "pragma journal_mode=wal":
            if self.failures:
                self.failures -= 1
                raise sqlite3.OperationalError("database is locked")
            self.journal_mode = "wal"
            return _Cursor(None)
        if text == "pragma journal_mode":
            return _Cursor((self.journal_mode,))
        raise AssertionError(f"unexpected statement: {statement}")

    def switches(self) -> int:
        return sum(1 for text in self.statements if text.startswith("pragma journal_mode="))


class _Cursor:
    def __init__(self, row: tuple[object, ...] | None) -> None:
        self._row = row

    def fetchone(self) -> tuple[object, ...] | None:
        return self._row


def test_an_already_wal_database_is_never_switched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug's shape: every connection re-issued the schema-locking switch."""

    monkeypatch.setattr("storage.sqlite.time.sleep", lambda _seconds: None)
    store = SqliteConversationStore(tmp_path / "already.sqlite3")
    connection = _ScriptedConnection(journal_mode="wal")

    store._ensure_wal(connection)  # type: ignore[arg-type]

    assert connection.statements == ["pragma journal_mode"]
    assert connection.switches() == 0


def test_the_journal_mode_switch_is_retried_until_it_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A losing racer retries the switch instead of failing the caller's write."""

    monkeypatch.setattr("storage.sqlite.time.sleep", lambda _seconds: None)
    store = SqliteConversationStore(tmp_path / "fresh.sqlite3")
    connection = _ScriptedConnection(journal_mode="delete", failures=2)

    store._ensure_wal(connection)  # type: ignore[arg-type]

    assert connection.switches() == 3
    assert connection.journal_mode == "wal"


def test_a_switch_that_keeps_losing_does_not_fail_the_opening_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Giving up is deliberate: the write path still works in ``delete`` mode."""

    monkeypatch.setattr("storage.sqlite.time.sleep", lambda _seconds: None)
    store = SqliteConversationStore(tmp_path / "contended.sqlite3")
    connection = _ScriptedConnection(journal_mode="delete", failures=99)

    store._ensure_wal(connection)  # type: ignore[arg-type]

    assert connection.switches() == 5
