"""Phase 6 baseline modes through the application service and controller.

The builder tests cover chunk mechanics in isolation; these tests cover the
thing the plan actually asks for — that selecting a mode changes what reaches
the model, that every mode reports the same accounting, and that the modes are
separable on a conversation long enough for the window to matter.
"""

from __future__ import annotations

import pytest

from app.controller import UIController
from app.service import ConversationService
from app.state import ContextSettings, SessionState
from baselines.modes import (
    MODE_BRAINOS,
    MODE_BRAINOS_RAG,
    MODE_FULL_CONTEXT,
    MODE_ORDER,
    MODE_RAG,
    MODE_SLIDING_WINDOW,
    MODES,
)
from brain.adapter import BrainOSAdapter
from providers.base import ProviderConfig
from tests.fakes import FakeLLMProvider, FakeProvider, FakeRuntime

FACT = "For Project Atlas, the production database is PostgreSQL 16."
DEPLOY = "We deploy on Fridays at 17:00 UTC."
QUESTION = "What database does Project Atlas use in production?"


def _long_transcript(turns: int = 10) -> list[str]:
    """Filler turns that push the opening facts out of a small window."""

    filler = []
    for index in range(turns):
        filler.append(f"How is the rollout looking today? Check number {index}.")
        filler.append(f"Rollout check {index} is steady; nothing new to report.")
    return filler


def _service(mode: str, provider: object | None = None) -> ConversationService:
    state = SessionState(
        provider=ProviderConfig(model="fake-model", api_key="session-secret"),
        context=ContextSettings().with_mode_defaults(mode),
    )
    return ConversationService(
        state,
        adapter=BrainOSAdapter(FakeRuntime()),
        # The canned reply must not contain the fact under test, or the
        # sliding-window mode would "remember" it through the fake answers.
        provider=provider
        if provider is not None
        else FakeProvider("Understood; I will keep that in mind."),
    )


def _converse(service: ConversationService) -> object:
    """Establish two facts, bury them, then ask about the first."""

    service.handle_user_message(FACT)
    service.record_assistant_message("Noted: Project Atlas runs PostgreSQL 16.")
    for line in _long_transcript():
        service.handle_user_message(line)
    service.handle_user_message(DEPLOY)
    service.record_assistant_message("Understood: the window is Friday 17:00 UTC.")
    return service.handle_user_message(QUESTION)


@pytest.mark.parametrize("mode", MODE_ORDER)
def test_every_mode_runs_a_turn_and_reports_the_same_accounting(mode: str) -> None:
    turn = _converse(_service(mode))

    assert turn.context_messages
    assert turn.context_messages[-1]["content"] == QUESTION
    for key in (
        "raw_history_tokens",
        "recent_history_tokens",
        "retrieved_memory_tokens",
        "system_tokens",
        "final_context_tokens",
        "selected_chunk_count",
        "chunk_block_tokens",
        "evidence_tokens",
        "full_context_reference_tokens",
    ):
        assert key in turn.context_stats, (mode, key)
    assert turn.generated is True


@pytest.mark.parametrize("mode", MODE_ORDER)
def test_the_system_prompt_is_identical_across_modes(mode: str) -> None:
    """Only the context-management strategy may differ (plan §9)."""

    turn = _converse(_service(mode))

    assert turn.context_messages[0]["role"] == "system"
    assert turn.context_messages[0]["content"].startswith(
        "You are an assistant with an external cognitive-memory layer."
    )


def test_memory_free_modes_send_no_memory_block() -> None:
    for mode in (MODE_FULL_CONTEXT, MODE_SLIDING_WINDOW, MODE_RAG):
        turn = _converse(_service(mode))
        prompt = "\n".join(m["content"] for m in turn.context_messages)

        assert "<retrieved_memory>" not in prompt, mode
        assert turn.retrieved_memories == (), mode


def test_brainos_modes_inject_memory() -> None:
    for mode in (MODE_BRAINOS, MODE_BRAINOS_RAG):
        turn = _converse(_service(mode))
        prompt = "\n".join(m["content"] for m in turn.context_messages)

        assert "<retrieved_memory>" in prompt, mode
        assert turn.retrieved_memories, mode


def test_only_the_rag_modes_inject_transcript_chunks() -> None:
    for mode in MODE_ORDER:
        turn = _converse(_service(mode))
        prompt = "\n".join(m["content"] for m in turn.context_messages)
        expected = MODES[mode].uses_rag

        assert bool(turn.retrieved_chunks) is expected, mode
        assert ("<retrieved_history>" in prompt) is expected, mode


def test_full_context_replays_the_whole_conversation() -> None:
    turn = _converse(_service(MODE_FULL_CONTEXT))
    stats = turn.context_stats

    assert stats["history_messages_selected"] == stats["history_messages_considered"]
    assert stats["dropped_history_for_budget"] == 0
    # Mode A *is* the reference the other modes are measured against.
    assert stats["final_context_tokens"] == stats["full_context_reference_tokens"]
    assert stats["context_reduction_vs_full_context"] == 0.0


def test_the_sliding_window_forgets_what_full_context_keeps() -> None:
    windowed = _converse(_service(MODE_SLIDING_WINDOW))
    full = _converse(_service(MODE_FULL_CONTEXT))

    assert windowed.context_stats["history_messages_selected"] == (
        MODES[MODE_SLIDING_WINDOW].max_recent_turns
    )
    assert windowed.context_stats["history_messages_selected"] < (
        full.context_stats["history_messages_selected"]
    )
    assert windowed.context_stats["final_context_tokens"] < (
        full.context_stats["final_context_tokens"]
    )
    # The fact is now behind the window and was not retrieved either.
    prompt = "\n".join(m["content"] for m in windowed.context_messages)
    assert "PostgreSQL" not in prompt


def test_brainos_keeps_the_fact_the_sliding_window_lost() -> None:
    """The comparison the project exists to make, at single-conversation scale."""

    windowed = _converse(_service(MODE_SLIDING_WINDOW))
    managed = _converse(_service(MODE_BRAINOS))
    window_prompt = "\n".join(m["content"] for m in windowed.context_messages)
    managed_prompt = "\n".join(m["content"] for m in managed.context_messages)

    assert "PostgreSQL" not in window_prompt
    assert "PostgreSQL" in managed_prompt
    assert managed.context_stats["final_context_tokens"] <= (
        windowed.context_stats["final_context_tokens"]
    )
    assert managed.context_stats["context_reduction_vs_full_context"] > 0.0


def test_rag_reaches_back_past_the_window_without_brainos() -> None:
    turn = _converse(_service(MODE_RAG))
    prompt = "\n".join(m["content"] for m in turn.context_messages)

    assert turn.retrieved_memories == ()
    assert "PostgreSQL" in prompt
    assert turn.context_stats["history_messages_selected"] < (
        turn.context_stats["history_messages_considered"]
    )
    assert turn.context_stats["selected_chunk_count"] > 0


def test_the_hybrid_mode_carries_both_evidence_sources() -> None:
    turn = _converse(_service(MODE_BRAINOS_RAG))
    stats = turn.context_stats

    assert turn.retrieved_memories
    assert turn.retrieved_chunks
    assert stats["memory_block_tokens"] > 0
    assert stats["chunk_block_tokens"] > 0
    assert stats["evidence_tokens"] == (
        stats["memory_block_tokens"] + stats["chunk_block_tokens"]
    )


def test_the_retrieval_modes_are_given_the_same_evidence_allowance() -> None:
    """Volume is held constant so the comparison isolates selection."""

    allowances = {
        mode: _service(mode).state.context.memory_budget
        + _service(mode).state.context.chunk_budget
        for mode in (MODE_RAG, MODE_BRAINOS, MODE_BRAINOS_RAG)
    }

    assert len(set(allowances.values())) == 1, allowances


def test_record_assistant_message_extends_the_transcript_without_a_provider() -> None:
    service = _service(MODE_BRAINOS, provider=None)

    service.handle_user_message(FACT)
    stored = service.record_assistant_message("Noted: Project Atlas runs PostgreSQL 16.")

    assert service.state.messages[-1]["role"] == "assistant"
    assert stored  # the scripted turn is observed like a generated one
    assert service.record_assistant_message("   ") == []


def test_record_assistant_message_is_persisted_when_a_store_is_present() -> None:
    from tests.fakes import InMemoryConversationStore

    store = InMemoryConversationStore()
    state = SessionState(context=ContextSettings().with_mode_defaults(MODE_BRAINOS))
    service = ConversationService(
        state, adapter=BrainOSAdapter(FakeRuntime()), conversation_store=store
    )

    service.record_assistant_message("Noted: Project Atlas runs PostgreSQL 16.")

    assert [row.role for row in store.rows] == ["assistant"]


# --------------------------------------------------------------------------- #
# Controller surface
# --------------------------------------------------------------------------- #


def _controller() -> UIController:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(FakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"])))

    return UIController(provider_factory=FakeLLMProvider, adapter_factory=adapter_factory)


def _connected(controller: UIController) -> str:
    return controller.connect(
        None, provider="openai", model="gpt-4o-mini", api_key="sk-session-SECRET-9"
    ).session_id


def test_mode_change_rewrites_the_window_and_the_evidence_budgets() -> None:
    controller = _controller()
    session_id = _connected(controller)

    controller.apply_mode(session_id, MODE_FULL_CONTEXT)
    full = controller.ensure_session(session_id).context
    assert full.max_recent_turns is None
    assert full.recent_turn_budget == full.max_tokens
    assert full.chunk_budget == 0

    controller.apply_mode(session_id, MODE_RAG)
    rag = controller.ensure_session(session_id).context
    assert rag.max_recent_turns == MODES[MODE_RAG].max_recent_turns
    assert rag.chunk_budget == MODES[MODE_RAG].chunk_budget
    assert rag.memory_budget == 0


def test_budgets_submitted_with_an_unchanged_mode_are_the_users() -> None:
    """Tuning inside a mode must still work after the mode rule was added."""

    controller = _controller()
    session_id = _connected(controller)
    controller.apply_mode(session_id, MODE_BRAINOS)

    controller.update_context(
        session_id, mode=MODE_BRAINOS, recent_turn_budget=999, memory_budget=77
    )
    settings = controller.ensure_session(session_id).context

    assert settings.recent_turn_budget == 999
    assert settings.memory_budget == 77


def test_budgets_submitted_with_a_mode_change_lose_to_the_mode() -> None:
    """Otherwise the sidebar would carry the old window into the new mode."""

    controller = _controller()
    session_id = _connected(controller)
    controller.apply_mode(session_id, MODE_BRAINOS)

    controller.update_context(
        session_id,
        mode=MODE_FULL_CONTEXT,
        recent_turn_budget=MODES[MODE_BRAINOS].recent_turn_budget,
    )
    settings = controller.ensure_session(session_id).context

    assert settings.mode == MODE_FULL_CONTEXT
    assert settings.recent_turn_budget == settings.max_tokens


def test_context_payload_describes_the_active_mode() -> None:
    controller = _controller()
    session_id = _connected(controller)
    controller.apply_mode(session_id, MODE_BRAINOS_RAG)

    payload = controller.context_payload(session_id)

    assert payload["mode"] == MODE_BRAINOS_RAG
    assert payload["mode_label"].startswith("Mode E")
    assert payload["uses_memory"] is True
    assert payload["uses_rag"] is True
    assert payload["mode_profile"]["evidence_budget"] > 0
    assert payload["budget"]["chunk_budget"] == MODES[MODE_BRAINOS_RAG].chunk_budget


def test_the_chunk_panel_follows_the_mode() -> None:
    controller = _controller()
    session_id = _connected(controller)

    controller.apply_mode(session_id, MODE_FULL_CONTEXT)
    controller.chat(session_id, FACT)
    plain = controller.chat(session_id, QUESTION)
    assert plain.chunk_rows == []

    controller.apply_mode(session_id, MODE_RAG)
    ragged = controller.chat(session_id, QUESTION)

    assert ragged.chunk_rows
    assert ragged.chunk_rows[0][2] in {"user", "assistant"}
    assert "Mode C" in ragged.summary
    assert "retrieved chunks" in ragged.summary


def test_mode_state_survives_a_panel_refresh() -> None:
    controller = _controller()
    session_id = _connected(controller)
    controller.apply_mode(session_id, MODE_RAG)
    controller.chat(session_id, FACT)
    # Enough filler that the fact falls outside the mode's recent window;
    # otherwise the retriever would only find a duplicate of the window.
    for line in _long_transcript(4):
        controller.chat(session_id, line)
    controller.chat(session_id, QUESTION)

    refreshed = controller.refresh(session_id)

    assert refreshed.chunk_rows
    assert "Mode C" in refreshed.summary


def test_export_records_the_mode_that_produced_the_transcript() -> None:
    controller = _controller()
    session_id = _connected(controller)
    controller.apply_mode(session_id, MODE_RAG)
    controller.chat(session_id, FACT)

    payload = controller.export_session(session_id)

    assert payload["context"]["mode"] == MODE_RAG
    assert payload["context"]["mode_profile"]["letter"] == "C"
