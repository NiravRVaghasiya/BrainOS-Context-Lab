"""Phase 4 controller tests: every value the browser renders is asserted here.

The controller is the only place UI callbacks touch application state, so
these tests cover the wiring contract headlessly: panel population, settings
application and validation, credential discipline, session lifecycle, turn
limits, and the audit surfaces (drops, conflicts, ranking, trace).
"""

from __future__ import annotations

import json
from typing import Any

from app.controller import DEFAULT_MAX_TURNS, ChatController
from app.service import ConversationService
from app.session import SessionManager
from app.state import SessionState
from brain.adapter import BrainOSAdapter, BrainOSNotConfiguredError
from tests.fakes import FakeProvider, FakeRuntime, LooseFakeRuntime

_SENTINEL = object()


def make_controller(
    *,
    runtime: Any = None,
    provider: Any = _SENTINEL,
    manager: SessionManager | None = None,
    **kwargs: Any,
) -> ChatController:
    """Build a controller on fakes; ``provider=None`` injects no provider."""

    state = manager.start() if manager is not None else SessionState()
    injected_runtime = runtime if runtime is not None else FakeRuntime()
    injected_provider = FakeProvider() if provider is _SENTINEL else provider

    def factory(session_state: SessionState) -> ConversationService:
        return ConversationService(
            session_state,
            adapter=BrainOSAdapter(injected_runtime),
            provider=injected_provider,
        )

    return ChatController(state, manager=manager, service_factory=factory, **kwargs)


def dump(view: dict[str, Any]) -> str:
    return json.dumps(view, default=str)


# --------------------------------------------------------------------------- #
# Panel population
# --------------------------------------------------------------------------- #


def test_views_are_safe_before_any_turn() -> None:
    controller = make_controller()
    view = controller.views()
    assert view["history"] == []
    assert view["stored_rows"] == []
    assert view["retrieved_rows"] == []
    assert view["context_stats"] == {}
    assert view["final_prompt"] == ""
    assert view["status"]
    assert view["max_turns"] == DEFAULT_MAX_TURNS


def test_send_message_populates_every_panel() -> None:
    controller = make_controller()
    controller.apply_provider(model="fake-model", api_key="session-key")
    controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")
    view = controller.send_message("What database does Project Atlas use?")

    assert [message["role"] for message in view["history"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert view["turns_used"] == 2
    assert "Reply generated" in view["status"]

    assert view["stored_rows"], "the durable fact should appear in stored memories"
    assert any(row[0] == "FACT" for row in view["stored_rows"])
    assert "PostgreSQL" in " ".join(str(cell) for row in view["stored_rows"] for cell in row)

    assert view["retrieved_rows"], "expected the fact to be retrieved and ranked"
    assert view["retrieved_rows"][0][0] == 1
    assert "PostgreSQL" in view["retrieved_rows"][0][4]
    assert view["ranking_json"] and "score" in view["ranking_json"][0]

    for field in (
        "raw_history_tokens",
        "recent_history_tokens",
        "retrieved_memory_tokens",
        "system_tokens",
        "final_context_tokens",
    ):
        assert field in view["context_stats"]
    assert "tokens" in view["context_summary"]
    assert "<retrieved_memory>" in view["final_prompt"]
    assert view["final_prompt"].splitlines()[0] == "SYSTEM"
    assert view["trace_rows"]
    assert view["decision_md"]


def test_irrelevant_memory_surfaces_as_dropped_row() -> None:
    controller = make_controller(runtime=LooseFakeRuntime(), provider=None)
    controller.send_message("The production database is PostgreSQL 16.")
    controller.send_message("Deployments happen every Friday at 17:00 UTC.")
    view = controller.send_message("What database is used in production?")

    reasons = {row[0] for row in view["dropped_rows"]}
    assert reasons & {"low_relevance", "weak_relevance"}, reasons
    joined = " ".join(str(cell) for row in view["retrieved_rows"] for cell in row)
    assert "PostgreSQL" in joined


def test_runtime_stale_memory_is_dropped_and_shown() -> None:
    runtime = FakeRuntime(stale=["session-a-m1"])
    controller = make_controller(runtime=runtime, provider=None)
    controller.send_message("The production database is PostgreSQL 16.")
    controller.send_message("Correction: the production database is MySQL 8.")
    view = controller.send_message("What is the production database?")

    reasons = {row[0] for row in view["dropped_rows"]}
    assert reasons & {"stale", "superseded"}, reasons
    joined = " ".join(str(cell) for row in view["retrieved_rows"] for cell in row)
    assert "MySQL" in joined
    assert "PostgreSQL" not in joined


# --------------------------------------------------------------------------- #
# Provider settings
# --------------------------------------------------------------------------- #


def test_apply_provider_updates_state_and_invalidates_cache_only_on_change() -> None:
    controller = make_controller()
    controller.apply_provider(model="m1", api_key="k", temperature=0.4)
    service = controller.service
    controller.apply_provider(model="m1", api_key="k", temperature=0.4)
    assert controller.service is service
    controller.apply_provider(model="m2")
    assert controller.state.provider.model == "m2"
    assert controller.state.provider.temperature == 0.4
    assert controller.state.provider.api_key == "k"
    assert controller.service is not service


def test_blank_api_key_keeps_existing_credential() -> None:
    controller = make_controller()
    controller.apply_provider(api_key="kept-secret", model="m")
    controller.apply_provider(api_key="", model="m")
    assert controller.state.provider.api_key == "kept-secret"


def test_invalid_provider_settings_are_rejected_safely() -> None:
    controller = make_controller()
    message = controller.apply_provider(temperature=-1.0)
    assert "not applied" in message
    assert controller.state.provider.temperature == 0.2


def test_validate_connection_reports_success_without_key() -> None:
    controller = make_controller()
    controller.apply_provider(api_key="session-key", model="fake-model")
    status = controller.validate_connection()
    assert "Connected" in status
    assert "session-key" not in status


def test_validate_connection_requires_credentials() -> None:
    controller = make_controller(provider=None)
    status = controller.validate_connection()
    assert "required" in status.lower()


def test_validate_connection_failure_redacts_secret() -> None:
    failing = FakeProvider(error="401 incorrect api key sk-live-secret-123")
    controller = make_controller(provider=failing)
    controller.apply_provider(api_key="sk-live-secret-123", model="m")
    status = controller.validate_connection()
    assert "sk-live-secret-123" not in status
    assert "[redacted]" in status
    assert "failed" in status.lower()


def test_list_models_populates_choices() -> None:
    controller = make_controller(provider=FakeProvider(models=("alpha", "beta")))
    controller.apply_provider(api_key="k", model="")
    controller.list_models()
    view = controller.views()
    assert view["model_choices"] == ["alpha", "beta"]


def test_list_models_requires_key_and_redacts_errors() -> None:
    controller = make_controller(provider=None)
    assert "without an API key" in controller.list_models()

    failing = FakeProvider(error="boom token=abc123")
    controller = make_controller(provider=failing)
    controller.apply_provider(api_key="abc123", model="m")
    status = controller.list_models()
    assert "abc123" not in status


# --------------------------------------------------------------------------- #
# Generation gating and errors
# --------------------------------------------------------------------------- #


def test_turn_without_credentials_still_runs_brainos() -> None:
    controller = make_controller(provider=None)
    view = controller.send_message("The staging server runs Ubuntu 24.04.")
    assert "No provider credentials" in view["status"]
    assert [message["role"] for message in view["history"]] == ["user"]
    assert view["stored_rows"], "BrainOS still observes without a provider"


def test_provider_error_surfaces_redacted_in_status() -> None:
    failing = FakeProvider(error="upstream 500; authorization: Bearer super-secret-key")
    controller = make_controller(provider=failing)
    controller.apply_provider(api_key="super-secret-key", model="m")
    view = controller.send_message("The production database is PostgreSQL 16.")
    assert "Provider error" in view["status"]
    assert "super-secret-key" not in dump(view)


def test_missing_brainos_returns_install_hint() -> None:
    def factory(state: SessionState) -> ConversationService:
        raise BrainOSNotConfiguredError("BrainOS is not installed.")

    controller = ChatController(SessionState(), service_factory=factory)
    view = controller.send_message("hello")
    assert "not installed" in view["status"]
    assert "integration" in view["status"]
    assert view["history"] == []
    assert view["stored_rows"] == []
    cleared = controller.clear_conversation()
    assert cleared["history"] == []


def test_turn_limit_stops_runaway_sessions() -> None:
    controller = make_controller(max_turns=2)
    controller.send_message("The production database is PostgreSQL 16.")
    controller.send_message("Deployments happen every Friday at 17:00 UTC.")
    view = controller.send_message("What database do we use?")
    assert "limit" in view["status"].lower()
    assert view["turns_used"] == 2
    assert len(view["history"]) == 4


# --------------------------------------------------------------------------- #
# Context settings and modes
# --------------------------------------------------------------------------- #


def test_context_settings_apply_and_validate() -> None:
    controller = make_controller()
    message = controller.apply_context_settings(max_tokens=2048, max_memories=4)
    assert "applied" in message
    assert controller.state.context.max_tokens == 2048
    assert controller.state.context.max_memories == 4

    rejected = controller.apply_context_settings(relevance_floor=5.0)
    assert "not applied" in rejected
    assert controller.state.context.relevance_floor == 0.12

    assert "unchanged" in controller.apply_context_settings(max_tokens=2048)
    assert "unchanged" in controller.apply_context_settings(unknown_knob=1)


def test_baseline_modes_announce_phase_six() -> None:
    controller = make_controller()
    notice = controller.set_memory_mode("rag")
    assert "Phase 6" in notice
    assert controller.state.context.mode == "rag"
    assert controller.views()["mode_notice"] == notice
    assert controller.set_memory_mode("brainos") == ""
    assert controller.state.context.mode == "brainos"


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #


def test_clear_conversation_resets_panels_and_memory() -> None:
    controller = make_controller()
    controller.apply_provider(api_key="k", model="m")
    controller.send_message("The production database is PostgreSQL 16.")
    first_conversation = controller.state.conversation_id

    view = controller.clear_conversation()
    assert view["history"] == []
    assert view["stored_rows"] == []
    assert view["final_prompt"] == ""
    assert view["turns_used"] == 0
    assert controller.state.conversation_id != first_conversation
    assert controller.state.provider.api_key == "k"


def test_end_session_clears_credentials_and_starts_fresh() -> None:
    manager = SessionManager()
    controller = make_controller(manager=manager)
    controller.apply_provider(api_key="end-secret", model="m")
    controller.send_message("The production database is PostgreSQL 16.")

    fresh, view = controller.end_session()
    assert controller.state.provider.api_key == ""
    assert manager.get(controller.state.session_id) is None
    assert fresh.state.session_id != controller.state.session_id
    assert fresh.state.provider.api_key == ""
    assert view["history"] == []
    assert "Session ended" in view["status"]
    assert "end-secret" not in dump(view)


def test_sessions_do_not_share_memory() -> None:
    manager = SessionManager()
    first = make_controller(manager=manager, runtime=FakeRuntime("s-1", "a-1"))
    # No provider in the second session: its canned assistant reply would
    # otherwise be observed as the session's *own* memory and the assertion
    # below could not distinguish self-content from a cross-session leak.
    second = make_controller(manager=manager, runtime=FakeRuntime("s-2", "a-2"), provider=None)
    first.send_message("The production database is PostgreSQL 16.")
    view = second.send_message("What database do we use?")

    assert first.state.session_id != second.state.session_id
    assert all("PostgreSQL" not in str(row) for row in view["retrieved_rows"])
    assert all("PostgreSQL" not in str(row) for row in view["stored_rows"])
    assert view["stored_rows"] == []
