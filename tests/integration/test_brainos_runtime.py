"""Live mapping tests against the pinned BrainOS revision.

These tests are skipped unless the optional integration extra is installed.
They exist to prove the adapter talks to the real ``observe(source=...)``,
``recall(top_k=...)``, decision-string, ``why()``, and structured ``trace()``
APIs rather than a guessed wrapper.
"""

from __future__ import annotations

import pytest

pytest.importorskip("brainos_runtime")

from brain.adapter import create_brain_adapter  # noqa: E402


def test_pinned_runtime_observe_recall_round_trip() -> None:
    adapter = create_brain_adapter(
        session_id="phase2-session",
        actor_id="phase2-actor",
        policy=None,
    )
    adapter.observe(
        "For Project Atlas, the production database is PostgreSQL 16.",
        metadata={"source": "user", "event_type": "user_message"},
    )
    memories = adapter.recall("What database does Project Atlas use?", limit=8)
    decision = adapter.decide("What database does Project Atlas use?")
    explanation = adapter.explain("What database does Project Atlas use?")
    events = adapter.trace()

    assert any("PostgreSQL" in record.text for record in memories)
    assert all(record.memory_id != "unknown" for record in memories)
    assert decision.action in {"act", "retrieve", "search", "ask", "clarify"}
    assert explanation.get("query") == "What database does Project Atlas use?"
    assert "api_key" not in explanation
    assert any(event.name in {"observe", "recall", "session"} for event in events)


def test_pinned_runtimes_are_isolated_by_session() -> None:
    first = create_brain_adapter(session_id="iso-a", actor_id="actor-a")
    second = create_brain_adapter(session_id="iso-b", actor_id="actor-b")
    first.observe("Session A uses the vault passphrase sunflower.")
    second.observe("Session B prefers SQLite for local storage.")

    leaked = second.recall("sunflower passphrase")
    kept = first.recall("sunflower passphrase")
    assert all("sunflower" not in record.text.lower() for record in leaked)
    assert any("sunflower" in record.text.lower() for record in kept)
