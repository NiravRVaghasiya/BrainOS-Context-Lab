from brain.trace import map_runtime_trace, sanitize_trace, sanitize_value


def test_map_runtime_trace_summarizes_cycles_without_memory_text() -> None:
    events = map_runtime_trace(
        {
            "session_id": "s-1",
            "cycles": 1,
            "records": [
                {
                    "event_type": "user_message",
                    "retrieved": ["The database is PostgreSQL."],
                    "candidates_considered": 4,
                    "decision": "act",
                    "steps": ["OBSERVE", "RETRIEVE"],
                    "api_key": "secret-key",
                }
            ],
        }
    )
    names = [event.name for event in events]
    assert names[:4] == ["session", "observe", "recall", "decide"]
    blob = " ".join(event.detail for event in events)
    assert "PostgreSQL" not in blob
    assert "secret-key" not in blob
    assert "selected=1" in blob
    assert "candidates=4" in blob


def test_sanitize_value_drops_secret_fields() -> None:
    result = sanitize_value(
        {
            "query": "Which database?",
            "api_key": "session-secret",
            "token": "abc",
            "selected": [{"content": "PostgreSQL", "authorization": "Bearer abc"}],
        }
    )
    assert "api_key" not in result
    assert "token" not in result
    assert result["selected"][0]["content"] == "PostgreSQL"
    assert "authorization" not in result["selected"][0]


def test_sanitize_trace_redacts_secret_bearing_mapping() -> None:
    result = sanitize_trace(
        [{"name": "provider", "detail": "safe", "api_key": "secret-key"}]
    )
    assert result[0]["detail"] == "[redacted]"
    assert "secret-key" not in str(result)
