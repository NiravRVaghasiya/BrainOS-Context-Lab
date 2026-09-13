"""Phase 5 wiring tests: persistence attaches without changing contracts.

The stores here are the in-memory doubles from ``tests/fakes.py``; the SQLite
backend itself is covered by ``test_storage_sqlite.py``. What is asserted is
the *seam*: the service writes turns best-effort, the controller owns
lifecycle deletes and export, a failing store never breaks a turn, and a bare
controller stays purely in-memory.
"""

from __future__ import annotations

from app.controller import ChatController
from app.service import ConversationService
from app.state import SessionState
from brain.adapter import BrainOSAdapter
from tests.fakes import (
    FakeProvider,
    FakeRuntime,
    InMemoryConversationStore,
    InMemoryEvaluationStore,
    InMemoryMemoryStore,
    RaisingStore,
)

_SENTINEL = object()


def make_persistent_controller(*, provider: object = _SENTINEL, **kwargs: object):
    """Controller on fakes with injected in-memory stores."""

    stores = {
        "conversation_store": InMemoryConversationStore(),
        "memory_store": InMemoryMemoryStore(),
        "evaluation_store": InMemoryEvaluationStore(),
    }
    stores.update(kwargs.pop("stores", {}))
    injected_provider = FakeProvider() if provider is _SENTINEL else provider

    def factory(state: SessionState) -> ConversationService:
        return ConversationService(
            state,
            adapter=BrainOSAdapter(FakeRuntime()),
            provider=injected_provider,
        )

    controller = ChatController(SessionState(), service_factory=factory, **stores, **kwargs)
    return controller, stores


# --------------------------------------------------------------------------- #
# Turn persistence
# --------------------------------------------------------------------------- #


def test_turn_persists_user_and_assistant_messages() -> None:
    controller, stores = make_persistent_controller()
    controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")

    rows = stores["conversation_store"].list_messages(
        controller.state.session_id, controller.state.conversation_id
    )
    assert [row.role for row in rows] == ["user", "assistant"]
    assert "PostgreSQL" in rows[0].content
    assert rows[0].metadata["turn"] == 1


def test_memories_are_mirrored_with_types() -> None:
    controller, stores = make_persistent_controller()
    controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")

    mirrored = stores["memory_store"].list_memories(controller.state.session_id)
    assert mirrored, "the durable fact should be mirrored to the store"
    assert any("PostgreSQL" in record["text"] for record in mirrored)
    assert any(record.get("memory_type") for record in mirrored)


def test_mirror_upserts_instead_of_growing_per_turn() -> None:
    # provider=None: turns that store nothing must not grow the mirror, which
    # is what proves the per-turn upsert (FakeRuntime, unlike the real runtime,
    # would happily store a duplicate reply memory under a new id every turn).
    controller, stores = make_persistent_controller(provider=None)
    controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")
    first_count = len(stores["memory_store"].list_memories(controller.state.session_id))
    assert first_count == 1

    controller.send_message("What database does Project Atlas use?")
    controller.send_message("And what about the staging environment?")
    second_count = len(stores["memory_store"].list_memories(controller.state.session_id))
    assert second_count == first_count


def test_persistence_failure_never_breaks_a_turn() -> None:
    controller, _ = make_persistent_controller(
        stores={"conversation_store": RaisingStore(), "memory_store": RaisingStore()}
    )
    view = controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")

    assert "Reply generated" in view["status"]
    assert view["stored_rows"], "the turn itself must succeed despite failing storage"


def test_bare_controller_stays_in_memory() -> None:
    def factory(state: SessionState) -> ConversationService:
        return ConversationService(
            state, adapter=BrainOSAdapter(FakeRuntime()), provider=FakeProvider()
        )

    controller = ChatController(SessionState(), service_factory=factory)
    controller.send_message("The production database is PostgreSQL 16.")
    assert controller.conversation_store is None
    assert controller.memory_store is None


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
    service.handle_user_message("The production database is PostgreSQL 16.")
    assert conversation_store.append_calls == 2


# --------------------------------------------------------------------------- #
# Lifecycle deletes
# --------------------------------------------------------------------------- #


def test_clear_conversation_deletes_persisted_rows() -> None:
    controller, stores = make_persistent_controller()
    controller.send_message("The production database is PostgreSQL 16.")
    old_conversation = controller.state.conversation_id

    controller.clear_conversation()
    assert stores["conversation_store"].list_messages(
        controller.state.session_id, old_conversation
    ) == []
    assert stores["memory_store"].list_memories(controller.state.session_id) == []

    controller.send_message("Deployments happen every Friday at 17:00 UTC.")
    assert controller.state.conversation_id != old_conversation
    assert stores["conversation_store"].list_conversations(controller.state.session_id) == [
        controller.state.conversation_id
    ]


def test_clear_memory_keeps_transcript_and_drops_mirror() -> None:
    controller, stores = make_persistent_controller()
    controller.send_message("The production database is PostgreSQL 16.")

    view = controller.clear_memory()
    assert view["history"], "the transcript survives a memory clear"
    assert view["stored_rows"] == []
    assert "memory cleared" in view["status"].lower()
    assert stores["memory_store"].list_memories(controller.state.session_id) == []
    assert stores["conversation_store"].list_conversations(controller.state.session_id)

    # The fresh runtime no longer knows the old fact.
    after = controller.send_message("What database do we use?")
    assert all("PostgreSQL" not in str(row) for row in after["retrieved_rows"])


def test_end_session_deletes_everything_persisted() -> None:
    controller, stores = make_persistent_controller()
    stores["evaluation_store"].save_run(
        "run-1", {"session_id": controller.state.session_id}, {"accuracy": 1.0}
    )
    controller.send_message("The production database is PostgreSQL 16.")
    session_id = controller.state.session_id

    fresh, view = controller.end_session()
    assert stores["conversation_store"].list_conversations(session_id) == []
    assert stores["memory_store"].list_memories(session_id) == []
    assert stores["evaluation_store"].get_run("run-1") is None
    assert "persisted session data deleted" in view["status"]
    # The fresh controller shares the same store instances.
    assert fresh.conversation_store is stores["conversation_store"]


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_contains_transcript_memories_and_last_turn() -> None:
    controller, _ = make_persistent_controller()
    controller.apply_provider(model="fake-model", api_key="export-secret")
    controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")
    controller.send_message("What database does Project Atlas use?")

    payload = controller.export_session()
    assert payload["session_id"] == controller.state.session_id
    assert [row["role"] for row in payload["transcript"]][:2] == ["user", "assistant"]
    assert any("PostgreSQL" in record["text"] for record in payload["memories"])
    assert payload["provider"]["model"] == "fake-model"
    assert "api_key" not in payload["provider"]
    assert payload["context_settings"]["max_tokens"] == 4096
    assert payload["last_turn"]["context_stats"]["final_context_tokens"] > 0
    assert payload["turns_used"] == 2


def test_export_file_is_valid_json_without_the_key() -> None:
    import json

    controller, _ = make_persistent_controller()
    controller.apply_provider(model="fake-model", api_key="file-secret-key")
    controller.send_message("The production database is PostgreSQL 16.")

    path, summary = controller.export_session_file()
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    assert "Session exported" in summary
    assert payload["transcript"]
    assert "file-secret-key" not in json.dumps(payload)


def test_export_falls_back_to_process_state_without_stores() -> None:
    def factory(state: SessionState) -> ConversationService:
        return ConversationService(
            state, adapter=BrainOSAdapter(FakeRuntime()), provider=FakeProvider()
        )

    controller = ChatController(SessionState(), service_factory=factory)
    controller.send_message("The production database is PostgreSQL 16.")

    payload = controller.export_session()
    assert [row["role"] for row in payload["transcript"]][:2] == ["user", "assistant"]
    assert any("PostgreSQL" in record["text"] for record in payload["memories"])
