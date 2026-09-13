from brain.adapter import BrainOSAdapter, Decision


class FakeRuntime:
    def observe(self, text, metadata):
        self.observed = (text, metadata)

    def recall(self, query, limit):
        return [{"id": "fact-1", "text": "The database is PostgreSQL."}]

    def decide(self, query):
        return True

    def why(self, query):
        return {"reason": "matched fact"}


def test_adapter_normalizes_injected_runtime() -> None:
    adapter = BrainOSAdapter(FakeRuntime())
    adapter.observe("The database is PostgreSQL.")
    memories = adapter.recall("Which database?")
    decision = adapter.decide("Which database?")

    assert memories[0].memory_id == "fact-1"
    assert memories[0].text == "The database is PostgreSQL."
    assert decision == Decision(sufficient=True)
    assert [event.name for event in adapter.trace()] == ["observe", "recall", "decide"]
