"""Phase 4 security tests: what the browser is allowed to see.

The controller is the single producer of browser-visible values, so the
credential rules are asserted against its complete view dict: the session key
never appears anywhere, credential-shaped strings are scrubbed from diagnostic
tables, and the chat transcript itself stays a faithful record of what the
user typed.
"""

from __future__ import annotations

import json

from app.controller import ChatController
from app.service import ConversationService
from app.state import SessionState
from brain.adapter import BrainOSAdapter
from providers.base import ProviderConfig
from tests.fakes import FakeProvider, LooseFakeRuntime


def make_view_controller(api_key: str = "") -> ChatController:
    state = SessionState(provider=ProviderConfig(model="m", api_key=api_key))
    runtime = LooseFakeRuntime()
    return ChatController(
        state,
        service_factory=lambda session_state: ConversationService(
            session_state,
            adapter=BrainOSAdapter(runtime),
            provider=FakeProvider(),
        ),
    )


def test_session_key_never_appears_in_any_view_value() -> None:
    controller = make_view_controller()
    controller.apply_provider(api_key="sk-session-supervisor-42", model="fake-model")
    controller.send_message("For Project Atlas, the production database is PostgreSQL 16.")
    controller.validate_connection()
    controller.list_models()
    view = controller.views()

    dumped = json.dumps(view, default=str)
    assert "sk-session-supervisor-42" not in dumped
    assert "api_key" not in view
    assert "key set" in view["provider_summary"]
    assert "sk-session-supervisor-42" not in view["connection_status"]


def test_panels_redact_the_session_key_from_memory_text() -> None:
    controller = make_view_controller(api_key="vault-master-key")
    controller.send_message("The vault credential is vault-master-key for the backup system.")
    controller.send_message("What is the vault credential?")
    view = controller.views()

    dumped = json.dumps(
        [view["stored_rows"], view["retrieved_rows"], view["dropped_rows"], view["ranking_json"]],
        default=str,
    )
    assert "vault-master-key" not in dumped


def test_panels_scrub_credential_shapes_but_transcript_stays_faithful() -> None:
    controller = make_view_controller()
    controller.send_message("The deploy token is sk-proj-ABCDEFG123456789 for production.")
    view = controller.views()

    tables = json.dumps(
        [view["stored_rows"], view["retrieved_rows"], view["dropped_rows"]], default=str
    )
    assert "sk-proj-ABCDEFG123456789" not in tables
    # The chat itself shows the user what they typed; scrubbing the transcript
    # would misrepresent the conversation.
    assert any("sk-proj-" in message["content"] for message in view["history"])


def test_end_session_removes_key_from_state_and_views() -> None:
    controller = make_view_controller()
    controller.apply_provider(api_key="temporary-key", model="fake-model")
    controller.send_message("The production database is PostgreSQL 16.")
    fresh, view = controller.end_session()

    assert controller.state.provider.api_key == ""
    assert fresh.state.provider.api_key == ""
    assert "temporary-key" not in json.dumps(view, default=str)
    assert "key not set" in view["provider_summary"]
