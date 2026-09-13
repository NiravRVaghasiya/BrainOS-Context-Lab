from app.service import ConversationService
from app.session import SessionManager
from app.state import ProviderConfig
from brain.adapter import BrainOSAdapter
from brain.trace import sanitize_trace
from tests.fakes import FakeProvider, FakeRuntime


def test_session_end_clears_provider_key() -> None:
    manager = SessionManager()
    session = manager.start()
    session.provider = ProviderConfig(api_key="secret-key", model="test")

    assert manager.end(session.session_id) is True
    assert manager.get(session.session_id) is None
    assert session.provider.api_key == ""
    assert session.brain is None


def test_trace_redacts_secret_bearing_mapping() -> None:
    result = sanitize_trace(
        [{"name": "provider", "detail": "safe", "api_key": "secret-key"}]
    )

    assert result[0]["detail"] == "[redacted]"
    assert "secret-key" not in str(result)


def test_service_diagnostics_never_include_provider_key() -> None:
    manager = SessionManager()
    session = manager.start()
    session.provider = ProviderConfig(api_key="secret-key", model="test")
    service = ConversationService(
        session,
        adapter=BrainOSAdapter(FakeRuntime()),
        provider=FakeProvider(),
    )
    service.handle_user_message("For Project Atlas, the production database is PostgreSQL 16.")
    inspect = service.inspect()
    assert "secret-key" not in str(inspect)
    assert "secret-key" not in str(session.diagnostics)
    assert "api_key" not in inspect["provider"]
