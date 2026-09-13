import pytest

from brain.adapter import (
    BrainOSAdapter,
    BrainOSNotConfiguredError,
    Decision,
    create_brain_adapter,
)
from brain.memory_policy import MemoryPolicy, MemoryType
from tests.fakes import FakeRuntime


def test_observe_skips_empty_and_policy_rejected_text() -> None:
    runtime = FakeRuntime()
    policy = MemoryPolicy(minimum_confidence=0.75)
    adapter = BrainOSAdapter(runtime, policy=policy)

    adapter.observe("   ")
    adapter.observe("hi", metadata={"confidence": 0.2, "memory_type": MemoryType.FACT})
    adapter.observe(
        "For Project Atlas, the production database is PostgreSQL 16.",
        metadata={"confidence": 0.9, "memory_type": MemoryType.PROJECT_STATE, "role": "user"},
    )

    assert runtime.observe_calls == [
        {
            "event": "For Project Atlas, the production database is PostgreSQL 16.",
            "source": "user",
            "event_type": "user_message",
        }
    ]
    names = [event.name for event in adapter.trace()]
    assert names.count("observe_skipped") >= 2


def test_observe_maps_assistant_role_to_event_type() -> None:
    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime)
    adapter.observe("Noted the database choice.", metadata={"role": "assistant"})
    assert runtime.observe_calls[0]["source"] == "assistant"
    assert runtime.observe_calls[0]["event_type"] == "assistant_message"


def test_recall_enriches_string_results_from_active_memories() -> None:
    runtime = FakeRuntime()
    adapter = BrainOSAdapter(runtime)
    adapter.observe("Deployments occur Friday at 17:00.")
    records = adapter.recall("When are deployments?", top_k=3)
    assert records[0].memory_id.startswith("session-a-m")
    assert records[0].text == "Deployments occur Friday at 17:00."
    assert records[0].memory_type == "FACT"
    assert runtime.recall_calls[0]["top_k"] == 3


def test_decide_maps_decision_strings_without_assess() -> None:
    class StringOnlyRuntime(FakeRuntime):
        assess = None  # type: ignore[assignment]

        def decide(self, query: str) -> str:
            return "clarify"

    adapter = BrainOSAdapter(StringOnlyRuntime())
    decision = adapter.decide("what changed?")
    assert decision == Decision(sufficient=False, reason="clarify", action="clarify")


def test_unconfigured_adapter_raises_explicit_error() -> None:
    adapter = BrainOSAdapter()
    with pytest.raises(BrainOSNotConfiguredError):
        adapter.observe("anything")


def test_create_brain_adapter_uses_injected_runtime_without_importing_brainos() -> None:
    runtime = FakeRuntime(session_id="s-1", actor_id="a-1")
    adapter = create_brain_adapter(session_id="s-1", actor_id="a-1", runtime=runtime)
    adapter.observe("The API timeout is 30 seconds.")
    assert adapter.session_id == "s-1"
    assert runtime.session_id == "s-1"


def test_two_injected_runtimes_do_not_share_memory() -> None:
    first = BrainOSAdapter(FakeRuntime(session_id="s-a", actor_id="actor-a"))
    second = BrainOSAdapter(FakeRuntime(session_id="s-b", actor_id="actor-b"))
    first.observe("Session A secret: the vault PIN is 1234.")
    second.observe("Session B uses MySQL.")

    first_hits = first.recall("vault PIN")
    second_hits = second.recall("vault PIN")
    assert any("1234" in record.text for record in first_hits)
    assert all("1234" not in record.text for record in second_hits)
