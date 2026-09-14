"""Tests for the Phase 11 ablation profiles.

The ablation study is only interpretable if every ablation is Mode D with
exactly one component removed, so these tests pin the properties the
comparison depends on: shared window and evidence budget, one removal each,
rejected unknown selectors, and knob restoration when a session leaves an
ablation for a baseline.
"""

from __future__ import annotations

import pytest

from app.state import ContextSettings
from baselines.ablations import (
    ABLATIONS,
    EXCLUDED_ABLATIONS,
    POLICY_KNOB_FIELDS,
    AblationProfile,
    ablation_pairs,
    ablation_profile,
    is_ablation,
    prepare_records,
    strip_lifecycle,
    strip_temporal_signals,
)
from baselines.modes import (
    ABLATION_NO_CONFLICT,
    ABLATION_NO_MEMORY,
    ABLATION_NO_RELEVANCE,
    ABLATION_NO_TEMPORAL,
    ABLATION_ORDER,
    EVIDENCE_BUDGET,
    MODE_BRAINOS,
    MODE_ORDER,
    MODES,
    ablation_choices,
    mode_choices,
    resolve_mode,
)
from brain.adapter import MemoryRecord
from brain.retrieval_policy import RetrievalPolicy


def test_the_four_ablations_are_registered_in_plan_order() -> None:
    assert ABLATION_ORDER == (
        ABLATION_NO_TEMPORAL,
        ABLATION_NO_RELEVANCE,
        ABLATION_NO_CONFLICT,
        ABLATION_NO_MEMORY,
    )
    assert set(ABLATIONS) == set(ABLATION_ORDER)
    for ablation in ABLATION_ORDER:
        assert ablation in MODES
        assert ablation not in MODE_ORDER, "ablations must not join the UI vocabulary"


def test_ablation_labels_name_the_removed_component() -> None:
    assert [MODES[a].letter for a in ABLATION_ORDER] == ["D1", "D2", "D3", "D4"]
    for ablation in ABLATION_ORDER:
        assert MODES[ablation].label.startswith("Ablation D")
        assert isinstance(ABLATIONS[ablation], AblationProfile)
        assert ABLATIONS[ablation].removes
        assert ABLATIONS[ablation].hypothesis
    assert [value for _label, value in ablation_choices()] == list(ABLATION_ORDER)
    assert [value for _label, value in mode_choices()] == list(MODE_ORDER)


def test_every_ablation_shares_mode_d_window_and_evidence_budget() -> None:
    full = MODES[MODE_BRAINOS]
    for ablation in ABLATION_ORDER:
        profile = MODES[ablation]
        assert profile.max_recent_turns == full.max_recent_turns, ablation
        assert profile.recent_turn_budget == full.recent_turn_budget, ablation
        assert not profile.uses_rag, ablation
    for ablation in (
        ABLATION_NO_TEMPORAL,
        ABLATION_NO_RELEVANCE,
        ABLATION_NO_CONFLICT,
    ):
        assert MODES[ablation].uses_memory, ablation
        assert MODES[ablation].memory_budget == EVIDENCE_BUDGET, ablation
        assert MODES[ablation].evidence_budget == EVIDENCE_BUDGET, ablation


def test_no_memory_is_the_window_without_the_memory() -> None:
    profile = MODES[ABLATION_NO_MEMORY]

    assert not profile.uses_memory
    assert profile.memory_budget == 0
    assert profile.evidence_budget == 0
    # The point of D4 over Mode B: the same small window Mode D runs, so the
    # D-vs-D4 gap isolates the memory itself rather than the window change.
    assert profile.max_recent_turns == MODES[MODE_BRAINOS].max_recent_turns
    assert profile.max_recent_turns < MODES["sliding_window"].max_recent_turns


def test_resolve_mode_accepts_ablations_and_still_rejects_unknown() -> None:
    for ablation in ABLATION_ORDER:
        assert resolve_mode(ablation) == ablation
    with pytest.raises(ValueError, match="Unknown baseline mode"):
        resolve_mode("brainos_no_telepathy")


def test_ablation_profile_round_trip_and_rejection() -> None:
    assert ablation_profile(ABLATION_NO_CONFLICT).ablation == ABLATION_NO_CONFLICT
    with pytest.raises(ValueError, match="Unknown ablation"):
        ablation_profile("brainos")
    assert is_ablation(ABLATION_NO_TEMPORAL)
    assert not is_ablation(MODE_BRAINOS)
    assert not is_ablation("telepathy")


def test_excluded_components_are_documented_not_silent() -> None:
    assert "no working memory" in EXCLUDED_ABLATIONS
    assert "no consolidation" in EXCLUDED_ABLATIONS
    for reason in EXCLUDED_ABLATIONS.values():
        assert len(reason) > 40


def test_policy_knob_fields_cover_every_override() -> None:
    assert set(POLICY_KNOB_FIELDS) == {
        key for profile in ABLATIONS.values() for key in profile.policy_overrides
    }
    fields = ContextSettings().__dataclass_fields__
    for name in POLICY_KNOB_FIELDS:
        assert name in fields, f"{name} must be a ContextSettings field"


@pytest.mark.parametrize(
    ("ablation", "expected"),
    [
        (ABLATION_NO_TEMPORAL, {"weight_recency": 0.0}),
        (
            ABLATION_NO_RELEVANCE,
            {"relevance_floor": 0.0, "relative_relevance_ratio": 0.0},
        ),
        (
            ABLATION_NO_CONFLICT,
            {
                "resolve_conflicts": False,
                "drop_stale_memories": False,
                "heuristic_conflict_detection": False,
            },
        ),
        (ABLATION_NO_MEMORY, {}),
    ],
)
def test_switching_to_an_ablation_applies_its_removal(
    ablation: str, expected: dict
) -> None:
    switched = ContextSettings().with_mode_defaults(ablation)

    assert switched.mode == ablation
    for name, value in expected.items():
        assert getattr(switched, name) == value, (ablation, name)
    # ... while the window and evidence budgets are Mode D's.
    full = ContextSettings().with_mode_defaults(MODE_BRAINOS)
    assert switched.recent_turn_budget == full.recent_turn_budget
    assert switched.max_recent_turns == full.max_recent_turns
    if ablation != ABLATION_NO_MEMORY:
        assert switched.memory_budget == full.memory_budget
    else:
        assert switched.memory_budget == 0


def test_leaving_an_ablation_restores_the_full_system() -> None:
    ablated = ContextSettings().with_mode_defaults(ABLATION_NO_RELEVANCE)
    assert ablated.relevance_floor == 0.0

    restored = ablated.with_mode_defaults(MODE_BRAINOS)

    defaults = ContextSettings()
    for name in POLICY_KNOB_FIELDS:
        assert getattr(restored, name) == getattr(defaults, name), name
    assert restored.mode == MODE_BRAINOS


def test_switching_between_ablations_applies_the_new_removal() -> None:
    switched = ContextSettings().with_mode_defaults(ABLATION_NO_RELEVANCE)
    switched = switched.with_mode_defaults(ABLATION_NO_TEMPORAL)

    assert switched.weight_recency == 0.0
    # The previous ablation's knobs are not restored mid-study: each ablation
    # is a defined condition, and mixing removals would be a fifth condition.
    assert switched.relevance_floor == 0.0


def test_new_policy_knobs_reach_the_retrieval_policy() -> None:
    settings = ContextSettings(
        relative_relevance_ratio=0.9, heuristic_conflict_detection=False
    )
    policy = settings.retrieval_policy()

    assert policy.relative_relevance_ratio == 0.9
    assert policy.heuristic_conflict_detection is False
    # Defaults preserve the pre-Phase-11 behaviour exactly.
    assert ContextSettings().retrieval_policy() == RetrievalPolicy(max_memories=6)


def _record(**overrides) -> MemoryRecord:  # type: ignore[no-untyped-def]
    base = {
        "memory_id": "m1",
        "text": "The production database is PostgreSQL 16.",
        "signals": {"lexical": 0.8, "recency": 0.9, "temporal_relevance": 1.0},
        "status": "active",
        "valid_until": "2030-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return MemoryRecord(**base)


def test_strip_temporal_signals_removes_only_temporal_keys() -> None:
    (stripped,) = strip_temporal_signals([_record()])

    assert stripped.signals == {"lexical": 0.8}
    assert stripped.text.startswith("The production database")
    assert stripped.status == "active"


def test_strip_lifecycle_neutralizes_status_and_validity() -> None:
    (stripped,) = strip_lifecycle(
        [_record(status="superseded", valid_until="2020-01-01T00:00:00+00:00")]
    )

    assert stripped.status is None
    assert stripped.valid_until is None
    assert stripped.text.startswith("The production database")
    assert stripped.signals["lexical"] == 0.8


def test_prepare_records_applies_each_ablation_exactly_once() -> None:
    records = [_record()]

    assert prepare_records(None, records) == records
    assert prepare_records(MODE_BRAINOS, records) == records
    assert prepare_records("telepathy", records) == records

    temporal = prepare_records(ABLATION_NO_TEMPORAL, records)
    assert temporal[0].signals == {"lexical": 0.8}
    assert temporal[0].status == "active"

    conflict = prepare_records(ABLATION_NO_CONFLICT, records)
    assert conflict[0].status is None
    assert conflict[0].valid_until is None
    assert conflict[0].signals["recency"] == 0.9

    relevance = prepare_records(ABLATION_NO_RELEVANCE, records)
    assert relevance[0].signals["recency"] == 0.9
    assert relevance[0].status == "active"

    # The caller's records are never mutated in place.
    assert records[0].signals["recency"] == 0.9
    assert records[0].status == "active"


def test_ablation_pairs_compare_against_the_full_system() -> None:
    assert ablation_pairs([MODE_BRAINOS, *ABLATION_ORDER]) == [
        (ablation, MODE_BRAINOS) for ablation in ABLATION_ORDER
    ]
    # No full reference, no pairs: an ablation without brainos is not
    # interpretable, so the analysis refuses to invent a comparison.
    assert ablation_pairs(list(ABLATION_ORDER)) == []
    assert ablation_pairs([MODE_BRAINOS]) == []
