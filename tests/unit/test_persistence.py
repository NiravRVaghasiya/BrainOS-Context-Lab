"""Phase 5 wiring tests: persistence attaches without changing contracts.

The stores here are the in-memory doubles from ``tests/fakes.py``; the SQLite
backend itself is covered by ``test_storage_sqlite.py``. What is asserted is
the *seam*: the service writes turns best-effort through the upstream
``UIController``/``ConversationService`` factories, the controller owns
lifecycle deletes and export, a failing store never breaks a turn, and a
controller without stores stays purely in-memory.
"""

from __future__ import annotations

import json

from app.controller import UIController
from app.service import ConversationService
from app.state import SessionState
from brain.adapter import BrainOSAdapter
from tests.fakes import (
    FakeLLMProvider,
    FakeProvider,
    FakeRuntime,
    InMemoryConversationStore,
    InMemoryEvaluationStore,
    InMemoryMemoryStore,
    RaisingStore,
)

KEY = "sk-persistence-test-0001"
FACT = "For Project Atlas, the production database is PostgreSQL 16."


def _adapter_factory(**kwargs: object) -> BrainOSAdapter:
    """Fresh fake runtime per creation, so clear-memory really forgets."""

    return BrainOSAdapter(
        FakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
    )


def make_persistent_controller(**stores_override: object) -> tuple[UIController, dict]:
    """Controller on fakes with injected in-memory stores."""

    stores: dict[str, object] = {
        "conversation_store": InMemoryConversationStore(),
        "memory_store": InMemoryMemoryStore(),
        "evaluation_store": InMemoryEvaluationStore(),
    }
    stores.update(stores_override)
    controller = UIController(
        provider_factory=FakeLLMProvider,
        adapter_factory=_adapter_factory,
        **stores,
    )
    return controller, stores


def connected(controller: UIController, *, api_key: str = KEY) -> str:
    """Connect one session and return its id."""

    return controller.connect(
        None, provider="openai", model="gpt-4o-mini", api_key=api_key
    ).session_id


# --------------------------------------------------------------------------- #
# Turn persistence
# --------------------------------------------------------------------------- #


def test_turn_persists_user_and_assistant_messages() -> None:
    controller, stores = make_persistent_controller()
    session_id = connected(controller)
    controller.chat(session_id, FACT)

    state = controller.sessions.get(session_id)
    rows = stores["conversation_store"].list_messages(
        session_id, state.conversation_id
    )
    assert [row.role for row in rows] == ["user", "assistant"]
    assert "PostgreSQL" in rows[0].content
    assert rows[0].metadata["turn"] == 1


def test_memories_are_mirrored_with_types() -> None:
    controller, stores = make_persistent_controller()
    session_id = connected(controller)
    controller.chat(session_id, FACT)

    mirrored = stores["memory_store"].list_memories(session_id)
    assert mirrored, "the durable fact should be mirrored to the store"
    assert any("PostgreSQL" in record["text"] for record in mirrored)
    assert any(record.get("memory_type") for record in mirrored)


def test_mirror_upserts_instead_of_growing_per_turn() -> None:
    # No key: memory-only turns that store nothing must not grow the mirror,
    # which is what proves the per-turn upsert (FakeRuntime, unlike the real
    # runtime, would happily store a duplicate reply memory under a new id
    # every turn if generation were on).
    controller, stores = make_persistent_controller()
    session_id = connected(controller, api_key="")
    controller.chat(session_id, FACT)
    first_count = len(stores["memory_store"].list_memories(session_id))
    assert first_count == 1

    controller.chat(session_id, "What database does Project Atlas use?")
    controller.chat(session_id, "And what about the staging environment?")
    second_count = len(stores["memory_store"].list_memories(session_id))
    assert second_count == first_count


def test_persistence_failure_never_breaks_a_turn() -> None:
    controller, _ = make_persistent_controller(
        conversation_store=RaisingStore(), memory_store=RaisingStore()
    )
    session_id = connected(controller)
    view = controller.chat(session_id, FACT)

    assert len(view.history) == 2
    assert view.stored_rows, "the turn itself must succeed despite failing storage"


def test_controller_without_stores_stays_in_memory() -> None:
    controller = UIController(
        provider_factory=FakeLLMProvider, adapter_factory=_adapter_factory
    )
    session_id = connected(controller)
    controller.chat(session_id, FACT)

    payload = controller.export_session(session_id)
    assert payload["persistence"]["enabled"] is False
    assert payload["persistence"]["conversations"] == []
    assert payload["messages"], "the in-memory transcript still exports"


def test_service_accepts_stores_directly() -> None:
    """The service seam works without a controller (evaluation runner path)."""

    state = SessionState()
    conversation_store = InMemoryConversationStore()
    service = ConversationService(
        state,
        adapter=BrainOSAdapter(FakeRuntime()),
        provider=FakeProvider(),
        conversation_store=conversation_store,
    )
    service.handle_user_message(FACT)
    assert conversation_store.append_calls == 2


# --------------------------------------------------------------------------- #
# Lifecycle deletes
# --------------------------------------------------------------------------- #


def test_clear_conversation_deletes_persisted_rows() -> None:
    controller, stores = make_persistent_controller()
    session_id = connected(controller)
    controller.chat(session_id, FACT)
    old_conversation = controller.sessions.get(session_id).conversation_id

    view = controller.clear_conversation(session_id)
    assert view.history == []
    assert "persisted rows deleted" in view.status
    assert (
        stores["conversation_store"].list_messages(session_id, old_conversation)
        == []
    )
    assert stores["memory_store"].list_memories(session_id) == []

    controller.chat(session_id, "Deployments happen every Friday at 17:00 UTC.")
    new_conversation = controller.sessions.get(session_id).conversation_id
    assert new_conversation != old_conversation
    assert stores["conversation_store"].list_conversations(session_id) == [
        new_conversation
    ]


def test_clear_memory_keeps_transcript_and_drops_mirror() -> None:
    controller, stores = make_persistent_controller()
    session_id = connected(controller)
    controller.chat(session_id, FACT)

    view = controller.clear_memory(session_id)
    assert view.history, "the transcript survives a memory clear"
    assert view.stored_rows == []
    assert "memory cleared" in view.status.lower()
    assert stores["memory_store"].list_memories(session_id) == []
    assert stores["conversation_store"].list_conversations(session_id)

    # The fresh runtime no longer knows the old fact.
    after = controller.chat(session_id, "What database do we use?")
    assert all("PostgreSQL" not in str(row) for row in after.retrieved_rows)


def test_end_session_deletes_everything_persisted() -> None:
    controller, stores = make_persistent_controller()
    session_id = connected(controller)
    stores["evaluation_store"].save_run(
        "run-1", {"session_id": session_id}, {"accuracy": 1.0}
    )
    controller.chat(session_id, FACT)

    ended = controller.end_session(session_id)
    assert "Session ended" in ended.status
    assert "persisted" in ended.status
    assert stores["conversation_store"].list_conversations(session_id) == []
    assert stores["memory_store"].list_memories(session_id) == []
    assert stores["evaluation_store"].get_run("run-1") is None
    assert controller.sessions.get(session_id) is None
    assert ended.session_id and ended.session_id != session_id


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_contains_live_and_persisted_views_without_the_key() -> None:
    controller, _ = make_persistent_controller()
    session_id = connected(controller)
    controller.chat(session_id, FACT)
    controller.chat(session_id, "What database does Project Atlas use?")

    payload = controller.export_session(session_id)
    assert payload["session_id"] == session_id
    assert [row["role"] for row in payload["messages"]][:2] == ["user", "assistant"]
    assert any("PostgreSQL" in record["text"] for record in payload["memories"])
    assert payload["provider"]["model"] == "gpt-4o-mini"
    assert "api_key" not in payload["provider"]
    assert payload["context"]["budget"]["max_tokens"] == 4096
    assert KEY not in json.dumps(payload)

    persistence = payload["persistence"]
    assert persistence["enabled"] is True
    assert len(persistence["conversations"]) == 1
    assert [row["role"] for row in persistence["conversations"][0]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert any(
        "PostgreSQL" in record["text"] for record in persistence["memories"]
    )


def test_export_text_is_valid_json_and_redacts_a_pasted_key() -> None:
    controller, _ = make_persistent_controller()
    session_id = connected(controller)
    controller.chat(session_id, f"My provider credential is {KEY} — remember it.")

    payload = json.loads(controller.export_text(session_id))
    dumped = json.dumps(payload)
    assert KEY not in dumped
    assert payload["messages"], "the export must still carry the transcript"
    assert "[REDACTED" in dumped or "redacted" in dumped.lower()


def test_export_persistence_block_is_empty_without_stores() -> None:
    controller = UIController(
        provider_factory=FakeLLMProvider, adapter_factory=_adapter_factory
    )
    session_id = connected(controller)
    controller.chat(session_id, FACT)

    payload = controller.export_session(session_id)
    assert payload["persistence"] == {
        "enabled": False,
        "conversations": [],
        "memories": [],
    }
    assert any("PostgreSQL" in record["text"] for record in payload["memories"])
