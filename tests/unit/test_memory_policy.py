from brain.memory_policy import MemoryCandidate, MemoryPolicy, MemoryType, extract_candidates


def test_policy_filters_low_confidence_candidates() -> None:
    policy = MemoryPolicy(minimum_confidence=0.8)
    candidates = [
        MemoryCandidate("stable fact", MemoryType.FACT, confidence=0.9),
        MemoryCandidate("uncertain fact", MemoryType.FACT, confidence=0.2),
    ]

    selected = policy.select(candidates)

    assert [item.text for item in selected] == ["stable fact"]


def test_extract_candidates_keeps_project_facts_and_skips_chatter() -> None:
    facts = extract_candidates(
        "For Project Atlas, the production database is PostgreSQL 16."
    )
    assert facts[0].memory_type is MemoryType.PROJECT_STATE
    assert facts[0].confidence >= 0.75

    assert extract_candidates("Hello") == []
    assert extract_candidates("What database do we use?") == []
