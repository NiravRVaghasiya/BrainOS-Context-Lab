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

from storage.conversations import ConversationMessage
from storage.sqlite import (
    SqliteConversationStore,
    SqliteEvaluationStore,
    SqliteMemoryStore,
)

THREADS = 8
WRITES_PER_THREAD = 25


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


def test_a_sweep_of_the_store_under_load_leaves_no_deleted_bytes_in_the_file(
    tmp_path: Path,
) -> None:
    """Phase 13's secure deletion, under Phase 14's concurrency.

    ``secure_delete`` plus a truncating checkpoint after every destructive write
    is what makes "deleted" mean "gone from the file". A concurrent writer must
    not push deleted content back into the WAL where a byte scan could read it.
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
    payload = marker.encode("utf-8")
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(f"{database}{suffix}")
        if candidate.exists():
            assert payload not in candidate.read_bytes(), (
                f"deleted content survived in {candidate.name}"
            )
    with sqlite3.connect(str(database)) as connection:
        rows = connection.execute(
            "SELECT content FROM messages WHERE session_id = 'session-a'"
        ).fetchall()
    assert rows == []
