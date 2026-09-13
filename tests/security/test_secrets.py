from app.session import SessionManager
from app.state import ProviderConfig
from brain.trace import sanitize_trace


def test_session_end_clears_provider_key() -> None:
    manager = SessionManager()
    session = manager.start()
    session.provider = ProviderConfig(api_key="secret-key", model="test")

    assert manager.end(session.session_id) is True
    assert manager.get(session.session_id) is None
    assert session.provider.api_key == ""


def test_trace_redacts_secret_bearing_mapping() -> None:
    result = sanitize_trace(
        [{"name": "provider", "detail": "safe", "api_key": "secret-key"}]
    )

    assert result[0]["detail"] == "[redacted]"
    assert "secret-key" not in str(result)
