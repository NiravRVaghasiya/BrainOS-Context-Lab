from brain.adapter import BrainOSAdapter, Decision
from tests.fakes import FakeRuntime


def test_adapter_maps_real_runtime_signatures() -> None:
    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime, session_id="session-a", actor_id="actor-a")

    adapter.observe(
        "The database is PostgreSQL.",
        metadata={"role": "user", "memory_type": "FACT"},
    )
    memories = adapter.recall("Which database?", limit=5)
    decision = adapter.decide("Which database?")
    explanation = adapter.explain("Which database?")

    assert runtime.observe_calls[0]["source"] == "user"
    assert runtime.observe_calls[0]["event_type"] == "user_message"
    assert "metadata" not in runtime.observe_calls[0]
    assert runtime.recall_calls[0] == {"query": "Which database?", "top_k": 5}
    assert memories[0].memory_id == "session-a-m1"
    assert memories[0].text == "The database is PostgreSQL."
    assert decision.sufficient is True
    assert decision.action == "act"
    assert explanation["decision"] == "act"
    assert "api_key" not in explanation
    assert "should-never-leak" not in str(explanation)
    names = [event.name for event in adapter.trace()]
    assert "observe" in names
    assert "recall" in names
    assert "decide" in names


def test_adapter_prefers_assess_and_maps_ask_decisions() -> None:
    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime)
    decision = adapter.decide("unknown topic")
    assert decision == Decision(
        sufficient=False,
        reason="no relevant memories retrieved",
        confidence=0.1,
        action="ask",
    )
