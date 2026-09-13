from brain.memory_policy import MemoryCandidate, MemoryPolicy, MemoryType


def test_policy_filters_low_confidence_candidates() -> None:
    policy = MemoryPolicy(minimum_confidence=0.8)
    candidates = [
        MemoryCandidate("stable fact", MemoryType.FACT, confidence=0.9),
        MemoryCandidate("uncertain fact", MemoryType.FACT, confidence=0.2),
    ]

    selected = policy.select(candidates)

    assert [item.text for item in selected] == ["stable fact"]
