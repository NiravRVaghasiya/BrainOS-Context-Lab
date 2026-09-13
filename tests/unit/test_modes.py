"""Tests for the Phase 6 baseline-mode registry.

The registry is what makes the controlled comparison possible, so these tests
pin the properties the experiment depends on rather than the literal numbers:
the plan's five modes exist, each one fixes its own history window, the three
retrieval modes share one evidence budget, and an unknown selector is an error
instead of a silent fallback.
"""

from __future__ import annotations

import pytest

from app.state import ContextSettings
from baselines.modes import (
    DEFAULT_MODE,
    EVIDENCE_BUDGET,
    MODE_BRAINOS,
    MODE_BRAINOS_RAG,
    MODE_FULL_CONTEXT,
    MODE_ORDER,
    MODE_RAG,
    MODE_SLIDING_WINDOW,
    MODES,
    mode_choices,
    mode_label,
    mode_profile,
    resolve_mode,
)


def test_the_plans_five_modes_are_the_vocabulary() -> None:
    assert MODE_ORDER == (
        MODE_FULL_CONTEXT,
        MODE_SLIDING_WINDOW,
        MODE_RAG,
        MODE_BRAINOS,
        MODE_BRAINOS_RAG,
    )
    assert [profile.letter for profile in (MODES[m] for m in MODE_ORDER)] == list("ABCDE")
    assert DEFAULT_MODE == MODE_BRAINOS


def test_memory_free_modes_are_the_controls() -> None:
    assert not MODES[MODE_FULL_CONTEXT].uses_memory
    assert not MODES[MODE_SLIDING_WINDOW].uses_memory
    assert not MODES[MODE_RAG].uses_memory
    assert MODES[MODE_BRAINOS].uses_memory
    assert MODES[MODE_BRAINOS_RAG].uses_memory


def test_only_the_retrieval_modes_use_the_lexical_baseline() -> None:
    assert not MODES[MODE_FULL_CONTEXT].uses_rag
    assert not MODES[MODE_BRAINOS].uses_rag
    assert MODES[MODE_RAG].uses_rag
    assert MODES[MODE_BRAINOS_RAG].uses_rag


@pytest.mark.parametrize("mode", MODE_ORDER)
def test_every_mode_fixes_its_own_history_window(mode: str) -> None:
    """The Phase 3/4 finding: a mode without its own window is not a mode.

    Two modes sharing a window larger than the conversation are
    indistinguishable, so the plan's Mode A must be defined by "no limit" and
    every other mode by a bounded one.
    """

    profile = MODES[mode]
    if mode == MODE_FULL_CONTEXT:
        assert profile.max_recent_turns is None
        assert profile.recent_turn_budget is None, "derived from the session ceiling"
    else:
        assert profile.max_recent_turns is not None
        assert profile.max_recent_turns > 0


def test_full_context_is_the_only_unbounded_window() -> None:
    limited = [MODES[m].max_recent_turns for m in MODE_ORDER if m != MODE_FULL_CONTEXT]
    assert all(turns is not None and turns > 0 for turns in limited)
    assert MODES[MODE_SLIDING_WINDOW].max_recent_turns > MODES[MODE_RAG].max_recent_turns


def test_the_three_retrieval_modes_share_one_evidence_budget() -> None:
    """Modes C/D/E differ in *what selects* evidence, so volume must be equal."""

    budgets = {
        mode: MODES[mode].evidence_budget
        for mode in (MODE_RAG, MODE_BRAINOS, MODE_BRAINOS_RAG)
    }
    assert budgets == {mode: EVIDENCE_BUDGET for mode in budgets}, budgets
    # Mode E splits the allowance between its two sources rather than doubling it.
    hybrid = MODES[MODE_BRAINOS_RAG]
    assert hybrid.memory_budget > 0
    assert hybrid.chunk_budget > 0
    assert hybrid.memory_budget + hybrid.chunk_budget == EVIDENCE_BUDGET


def test_memory_free_modes_allocate_no_evidence_budget() -> None:
    for mode in (MODE_FULL_CONTEXT, MODE_SLIDING_WINDOW):
        assert MODES[mode].evidence_budget == 0


def test_budget_overrides_derive_the_full_context_window_from_the_ceiling() -> None:
    overrides = MODES[MODE_FULL_CONTEXT].budget_overrides(8192)

    assert overrides["recent_turn_budget"] == 8192
    assert overrides["max_recent_turns"] is None
    assert overrides["memory_budget"] == 0
    assert overrides["chunk_budget"] == 0


def test_budget_overrides_use_the_declared_window_for_bounded_modes() -> None:
    profile = MODES[MODE_BRAINOS]
    overrides = profile.budget_overrides(4096)

    assert overrides["recent_turn_budget"] == profile.recent_turn_budget
    assert overrides["memory_budget"] == profile.memory_budget


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown baseline mode"):
        resolve_mode("telepathy")


def test_the_phase_4_no_memory_value_is_an_alias_for_mode_b() -> None:
    assert resolve_mode("no_memory") == MODE_SLIDING_WINDOW
    assert resolve_mode(MODE_SLIDING_WINDOW) == MODE_SLIDING_WINDOW


def test_mode_profile_and_label_round_trip() -> None:
    assert mode_profile("brainos").mode == MODE_BRAINOS
    assert mode_label(MODE_RAG).startswith("Mode C")
    # An unknown label degrades to the raw value instead of raising, because it
    # is used for display.
    assert mode_label("telepathy") == "telepathy"


def test_mode_choices_are_labelled_pairs_in_plan_order() -> None:
    choices = mode_choices()

    assert [value for _label, value in choices] == list(MODE_ORDER)
    assert all(label.startswith("Mode ") for label, _value in choices)


def test_default_settings_are_the_default_modes_profile() -> None:
    """A fresh session must not start in the degenerate configuration.

    Phase 3 measured 0.0% reduction for a BrainOS session whose history window
    was larger than the conversation, so ``ContextSettings()`` carries Mode D's
    budgets rather than neutral ones.
    """

    settings = ContextSettings()
    profile = settings.mode_profile()

    assert settings.mode == DEFAULT_MODE
    assert settings.recent_turn_budget == profile.recent_turn_budget
    assert settings.memory_budget == profile.memory_budget
    assert settings.max_recent_turns == profile.max_recent_turns
    assert settings.chunk_budget == profile.chunk_budget
    assert settings.uses_memory() and not settings.uses_rag()


@pytest.mark.parametrize("mode", MODE_ORDER)
def test_with_mode_defaults_rewrites_the_budgets_the_mode_governs(mode: str) -> None:
    settings = ContextSettings().with_mode_defaults(mode)
    profile = MODES[mode]

    assert settings.mode == mode
    assert settings.memory_budget == profile.memory_budget
    assert settings.chunk_budget == profile.chunk_budget
    assert settings.max_recent_turns == profile.max_recent_turns
    expected_window = (
        settings.max_tokens if profile.recent_turn_budget is None
        else profile.recent_turn_budget
    )
    assert settings.recent_turn_budget == expected_window
    # The budget objects the builder receives agree with the selector.
    assert settings.context_budget().chunk_budget == profile.chunk_budget
    assert settings.context_budget().max_recent_turns == profile.max_recent_turns


def test_with_mode_defaults_keeps_the_retrieval_knobs() -> None:
    """Switching strategy is not a reason to lose a user's policy tuning."""

    settings = ContextSettings(relevance_floor=0.4, max_memories=3)

    tuned = settings.with_mode_defaults(MODE_RAG)

    assert tuned.relevance_floor == 0.4
    assert tuned.max_memories == 3


def test_with_mode_defaults_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown baseline mode"):
        ContextSettings().with_mode_defaults("telepathy")
