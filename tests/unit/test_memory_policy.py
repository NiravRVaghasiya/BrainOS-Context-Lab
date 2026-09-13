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


def test_declarative_facts_without_copula_verbs_are_stored() -> None:
    """Regression: 'happen'/'requires' facts were silently rejected in Phase 2."""

    for text, expected in [
        (
            "Deployments for Project Atlas happen every Friday at 17:00 UTC.",
            MemoryType.TEMPORAL_EVENT,
        ),
        (
            "The Project Atlas retention policy requires 400 days of audit logs.",
            MemoryType.PROJECT_STATE,
        ),
        (
            "The Project Atlas cache layer runs Redis 7 with a 30 minute TTL.",
            MemoryType.PROJECT_STATE,
        ),
    ]:
        candidates = extract_candidates(text)
        assert candidates, text
        assert candidates[0].memory_type is expected
        assert MemoryPolicy().accepts(candidates[0])


def test_questions_and_requests_are_never_stored() -> None:
    for text in [
        "What is the weather like where you are? Just making conversation here.",
        "Can you summarise the standup notes from this morning for the team?",
        "Remind me what the on-call rotation looks like next month for the team?",
        "Let us talk about the naming convention we use for feature branches.",
        "How do we rotate the Project Atlas database credentials?",
    ]:
        assert extract_candidates(text) == [], text


def test_conversational_filler_is_not_stored() -> None:
    assert extract_candidates("I am thinking about the quarterly roadmap review.") == []
    assert extract_candidates("Just making conversation here about the weather today.") == []


def test_mixed_turn_contributes_only_its_declarative_sentence() -> None:
    candidates = extract_candidates(
        "What do we use for caching? Our cache layer uses Redis 7 with a 30 minute TTL."
    )
    assert [item.text for item in candidates] == [
        "Our cache layer uses Redis 7 with a 30 minute TTL."
    ]
    # Classified from the surviving sentence alone: without "Project Atlas" in it
    # there is no project evidence, so FACT is the honest label.
    assert candidates[0].memory_type is MemoryType.FACT
    assert MemoryPolicy().accepts(candidates[0])


def test_multi_fact_turn_yields_one_candidate_per_statement() -> None:
    candidates = extract_candidates(
        "The production database is PostgreSQL 16. The staging database is MySQL 8."
    )
    assert len(candidates) == 2
    assert MemoryPolicy().select(candidates) == candidates


def test_policy_caps_candidates_per_turn() -> None:
    text = ". ".join(f"The shard {index} database is PostgreSQL {index}" for index in range(9))
    candidates = extract_candidates(text + ".")
    assert len(MemoryPolicy(max_items_per_turn=3).select(candidates)) == 3
