"""Tests for the Gradio-free controller behind the chat UI.

These tests are the behavioural contract of Phase 4: a browser action maps onto
exactly one controller call, and every value the controller returns is safe to
render. The fakes stand in for the provider and BrainOS, so no credential and
no network is involved.
"""

import json

import pytest

from app.controller import UIController, UILimits
from app.session import SessionManager
from brain.adapter import BrainOSAdapter, BrainOSNotConfiguredError
from providers import ProviderError
from tests.fakes import FakeLLMProvider, LooseFakeRuntime

KEY = "sk-session-SECRET-0001"


@pytest.fixture()
def controller() -> UIController:
    """A controller wired to fakes, with one session already started."""

    created: list[str] = []

    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        created.append(str(kwargs["session_id"]))
        return BrainOSAdapter(LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"])))

    instance = UIController(
        provider_factory=FakeLLMProvider, adapter_factory=adapter_factory
    )
    instance.adapters_created = created  # type: ignore[attr-defined]
    return instance


def _connected(controller: UIController) -> str:
    view = controller.connect(
        None,
        provider="openai",
        model="gpt-4o-mini",
        api_key=KEY,
    )
    return view.session_id


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #


def test_connect_lists_models_and_clears_the_key_box(controller: UIController) -> None:
    view = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY)

    assert view.connected is True
    assert view.models == ("gpt-4o-mini", "gpt-4o")
    assert view.key_value == ""
    assert view.session_id
    # The status names the provider but never the credential.
    assert "openai" in view.status
    assert KEY not in view.status


def test_connect_keeps_the_key_in_server_memory_only(controller: UIController) -> None:
    session_id = _connected(controller)
    state = controller.ensure_session(session_id)

    assert state.provider.api_key == KEY
    assert "api_key" not in state.provider.safe_dict()
    blob = json.dumps(controller.export_session(session_id))
    assert KEY not in blob


def test_connect_without_a_key_runs_memory_only(controller: UIController) -> None:
    view = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key="")

    assert view.connected is False
    assert "no model will be called" in view.status


def test_connect_reports_redacted_provider_errors(controller: UIController) -> None:
    def failing(config: object) -> FakeLLMProvider:
        return FakeLLMProvider(config, fail=True)  # type: ignore[arg-type]

    instance = UIController(provider_factory=failing)
    view = instance.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY)

    assert view.connected is False
    assert KEY not in view.status
    assert "[redacted]" in view.status
    assert view.key_value == ""


def test_connect_rejects_an_invalid_configuration(controller: UIController) -> None:
    view = controller.connect(None, provider="openai", model="m", api_key=KEY, temperature=-3)
    assert view.connected is False
    assert "⚠️" in view.status


def test_connect_does_not_create_the_brainos_runtime(controller: UIController) -> None:
    controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY)
    assert controller.adapters_created == []  # type: ignore[attr-defined]


def test_disconnect_clears_the_key_and_stops_generation(controller: UIController) -> None:
    session_id = _connected(controller)
    view = controller.disconnect(session_id)

    assert "cleared" in view.status
    assert controller.ensure_session(session_id).provider.api_key == ""
    turn = controller.chat(session_id, "What database do we use?")
    assert "no model called" in turn.status


def test_refresh_models_reuses_the_stored_key(controller: UIController) -> None:
    session_id = _connected(controller)
    view = controller.refresh_models(session_id)

    assert view.connected is True
    assert view.session_id == session_id


# --------------------------------------------------------------------------- #
# Chat turns
# --------------------------------------------------------------------------- #


def test_chat_turn_fills_every_panel(controller: UIController) -> None:
    session_id = _connected(controller)
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    view = controller.chat(session_id, "What database does Project Atlas use?")

    assert [message["role"] for message in view.history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert view.stored_rows
    assert view.retrieved_rows
    assert "<retrieved_memory>" in view.prompt
    assert "Sent to model" in view.summary
    assert "Recall" in view.trace
    assert view.trace_events
    assert view.stats["final_context_tokens"] > 0


def test_chat_records_the_fact_in_the_memory_panel(controller: UIController) -> None:
    session_id = _connected(controller)
    view = controller.chat(
        session_id, "For Project Atlas, the production database is PostgreSQL 16."
    )

    assert any("PostgreSQL 16" in str(row[1]) for row in view.stored_rows)


def test_chat_without_credentials_still_observes_and_retrieves(
    controller: UIController,
) -> None:
    session_id = controller.connect(
        None, provider="openai", model="gpt-4o-mini", api_key=""
    ).session_id
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    view = controller.chat(session_id, "What database does Project Atlas use?")

    assert view.retrieved_rows
    assert "no model called" in view.status
    # The transcript records the question even when no model answers.
    assert view.history[-1]["role"] == "user"


def test_no_memory_mode_skips_recall(controller: UIController) -> None:
    session_id = _connected(controller)
    controller.update_context(session_id, mode="no_memory")
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    view = controller.chat(session_id, "What database does Project Atlas use?")

    assert view.retrieved_rows == []
    assert "<retrieved_memory>" not in view.prompt
    assert "mode `no_memory`" in view.status


def test_empty_message_does_not_create_a_turn(controller: UIController) -> None:
    session_id = _connected(controller)
    view = controller.chat(session_id, "   ")

    assert view.history == []
    assert "Type a message" in view.status


def test_turn_and_message_limits_are_enforced(controller: UIController) -> None:
    session_id = _connected(controller)
    long_view = controller.chat(session_id, "x" * 9000)
    assert "characters" in long_view.status

    limited = UIController(
        provider_factory=FakeLLMProvider,
        limits=UILimits(max_turns=1, max_message_chars=100),
    )
    sid = limited.connect(None, provider="openai", model="m", api_key=KEY).session_id
    assert limited.chat(sid, "first").history
    blocked = limited.chat(sid, "second")
    assert "reached 1 turns" in blocked.status


def test_a_stale_session_id_gets_a_fresh_isolated_session(controller: UIController) -> None:
    view = controller.chat("session-that-no-longer-exists", "hello")

    assert view.session_id
    assert controller.sessions.get(view.session_id) is not None
    assert view.session_id != "session-that-no-longer-exists"


def test_sessions_do_not_share_memory(controller: UIController) -> None:
    first = _connected(controller)
    second = controller.ensure_session(None).session_id
    controller.connect(second, provider="openai", model="gpt-4o-mini", api_key=KEY)
    controller.chat(first, "For Project Atlas, the production database is PostgreSQL 16.")

    other = controller.chat(second, "What database does Project Atlas use?")

    # Nothing from session one reached session two: no retrieved evidence, and
    # no stored memory that was never said in this session.
    assert other.retrieved_rows == []
    assert not any("For Project Atlas" in str(row[1]) for row in other.stored_rows)


# --------------------------------------------------------------------------- #
# Secret hygiene
# --------------------------------------------------------------------------- #


def test_the_session_key_never_reaches_a_panel(controller: UIController) -> None:
    session_id = _connected(controller)
    # A user pastes their key into the conversation itself: BrainOS may store it
    # verbatim, and no panel may hand it back to the browser.
    first = controller.chat(session_id, f"My api key is {KEY} remember it")
    second = controller.chat(session_id, "What is my api key?")

    for view in (first, second):
        blob = json.dumps(
            {
                "history": view.history,
                "status": view.status,
                "stored": view.stored_rows,
                "retrieved": view.retrieved_rows,
                "dropped": view.dropped_rows,
                "summary": view.summary,
                "prompt": view.prompt,
                "trace": view.trace,
                "events": view.trace_events,
                "stats": view.stats,
            },
            default=str,
        )
        assert KEY not in blob


def test_provider_errors_are_redacted_in_the_status(controller: UIController) -> None:
    def failing(config: object) -> object:
        class Boom(FakeLLMProvider):
            def generate(self, messages: object, **kwargs: object) -> object:
                raise ProviderError(f"rejected key {config.api_key}")  # type: ignore[attr-defined]

        return Boom(config)

    instance = UIController(provider_factory=failing, adapter_factory=controller._adapter_factory)
    sid = instance.connect(None, provider="openai", model="m", api_key=KEY).session_id
    view = instance.chat(sid, "hello")

    assert KEY not in view.status
    assert "[redacted]" in view.status


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #


def test_clear_conversation_empties_transcript_and_memory(controller: UIController) -> None:
    session_id = _connected(controller)
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    view = controller.clear_conversation(session_id)

    assert view.history == []
    assert view.stored_rows == []
    assert "cleared" in view.status


def test_clear_memory_keeps_the_transcript(controller: UIController) -> None:
    session_id = _connected(controller)
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    view = controller.clear_memory(session_id)

    assert len(view.history) == 2
    assert view.stored_rows == []


def test_end_session_drops_state_and_starts_a_new_one(controller: UIController) -> None:
    session_id = _connected(controller)
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    ended = controller.end_session(session_id)

    assert ended.session_id != session_id
    assert controller.sessions.get(session_id) is None
    assert controller.sessions.get(ended.session_id) is not None
    assert "Session ended" in ended.status


def test_refresh_renders_the_last_turn_without_sending_one(controller: UIController) -> None:
    session_id = _connected(controller)
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    before = controller.ensure_session(session_id).messages.copy()
    view = controller.refresh(session_id)

    assert controller.ensure_session(session_id).messages == before
    assert view.stored_rows


def test_one_runtime_is_created_per_session(controller: UIController) -> None:
    session_id = _connected(controller)
    controller.chat(session_id, "one")
    controller.chat(session_id, "two")

    assert controller.adapters_created.count(session_id) == 1  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Context configuration
# --------------------------------------------------------------------------- #


def test_update_context_applies_budget_and_policy(controller: UIController) -> None:
    session_id = _connected(controller)
    status = controller.update_context(
        session_id,
        recent_turn_budget=256,
        memory_budget=512,
        max_memories=6,
        relevance_floor=0.2,
        drop_suspicious_memories=True,
    )

    settings = controller.ensure_session(session_id).context
    assert settings.recent_turn_budget == 256
    assert settings.memory_budget == 512
    assert settings.max_memories == 6
    assert settings.relevance_floor == 0.2
    assert settings.drop_suspicious_memories is True
    assert "Context updated" in status


def test_update_context_rejects_unknown_and_invalid_values(controller: UIController) -> None:
    session_id = _connected(controller)
    assert "Unknown context setting" in controller.update_context(session_id, nope=1)

    invalid = controller.update_context(session_id, max_tokens=-5)
    assert "⚠️" in invalid
    assert controller.ensure_session(session_id).context.max_tokens == 4096


def test_context_payload_is_exportable_and_secret_free(controller: UIController) -> None:
    session_id = _connected(controller)
    payload = controller.context_payload(session_id)

    assert payload["mode"] == "brainos"
    assert payload["uses_memory"] is True
    assert payload["budget"]["max_tokens"] == 4096
    assert KEY not in json.dumps(payload)


# --------------------------------------------------------------------------- #
# Export and degraded runtimes
# --------------------------------------------------------------------------- #


def test_export_session_contains_transcript_memory_and_settings(
    controller: UIController,
) -> None:
    session_id = _connected(controller)
    controller.chat(session_id, "For Project Atlas, the production database is PostgreSQL 16.")
    payload = json.loads(controller.export_text(session_id))

    assert payload["session_id"] == session_id
    assert payload["memories"]
    assert payload["messages"]
    assert "api_key" not in payload["provider"]
    assert payload["context"]["uses_memory"] is True
    assert KEY not in json.dumps(payload)


def test_missing_brainos_gives_an_install_hint() -> None:
    def missing(**kwargs: object) -> BrainOSAdapter:
        raise BrainOSNotConfiguredError("BrainOS is not installed.")

    instance = UIController(provider_factory=FakeLLMProvider, adapter_factory=missing)
    sid = instance.connect(None, provider="openai", model="m", api_key=KEY).session_id
    view = instance.chat(sid, "hello")

    assert "BrainOS runtime is not installed" in view.status


def test_controller_shares_a_session_manager_when_injected() -> None:
    manager = SessionManager()
    instance = UIController(sessions=manager, provider_factory=FakeLLMProvider)
    state = instance.ensure_session(None)

    assert manager.get(state.session_id) is state
