from app.service import ConversationService
from app.session import SessionManager
from app.state import SessionState
from brain.adapter import BrainOSAdapter
from providers.base import ProviderConfig
from tests.fakes import FakeProvider, FakeRuntime


def test_service_observes_fact_and_retrieves_it_later() -> None:
    state = SessionState(provider=ProviderConfig(model="fake-model", api_key="session-secret"))
    runtime = FakeRuntime()
    service = ConversationService(
        state,
        adapter=BrainOSAdapter(runtime),
        provider=FakeProvider("You use PostgreSQL 16."),
    )

    first = service.handle_user_message(
        "For Project Atlas, the production database is PostgreSQL 16."
    )
    second = service.handle_user_message("What database does Project Atlas use?")

    assert first.stored_memories
    assert any("PostgreSQL" in record.text for record in second.retrieved_memories)
    assert second.generated is True
    assert second.reply == "You use PostgreSQL 16."
    assert "<retrieved_memory>" in second.context_messages[1]["content"]
    assert "session-secret" not in str(state.diagnostics)
    assert "api_key" not in state.diagnostics["provider"]
    inspect = service.inspect()
    assert "session-secret" not in str(inspect)
    assert inspect["provider"]["model"] == "fake-model"


def test_service_skips_generation_without_credentials() -> None:
    state = SessionState(provider=ProviderConfig(model="fake-model"))
    service = ConversationService(state, adapter=BrainOSAdapter(FakeRuntime()))
    turn = service.handle_user_message(
        "For Project Atlas, the production database is PostgreSQL 16."
    )
    assert turn.generated is False
    assert turn.reply is None
    assert turn.stored_memories


def test_session_isolation_across_services() -> None:
    manager = SessionManager()
    first_state = manager.start()
    second_state = manager.start()
    first = ConversationService(first_state, adapter=BrainOSAdapter(FakeRuntime("s-a", "a-a")))
    second = ConversationService(second_state, adapter=BrainOSAdapter(FakeRuntime("s-b", "a-b")))

    first.handle_user_message("The production database is PostgreSQL 16.")
    leaked = second.handle_user_message("What database do we use?")

    assert all("PostgreSQL" not in record.text for record in leaked.retrieved_memories)
    assert manager.end(first_state.session_id) is True
    assert first_state.brain is None


def test_clear_conversation_drops_runtime_for_owned_adapter() -> None:
    state = SessionState()
    original = BrainOSAdapter(FakeRuntime())
    ConversationService(state, adapter=original)
    state.brain = original
    # Injected adapters are retained; owned adapters are replaced. Exercise the
    # session-state drop used by session cleanup.
    state.clear_conversation()
    assert state.brain is None
    assert state.messages == []
