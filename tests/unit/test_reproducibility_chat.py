"""Phase 16 — reproducibility additions to the chat path.

The session export is the one human-readable artifact a chat produces, and
Phase 16 makes it re-inspectable: the cost ceilings that governed the session,
the application/BrainOS/benchmark revisions, and a transcript digest that ties
the conversation to the result. These tests pin that block and the one
credential boundary the phase also closed — a provider that echoes the pasted
key inside a model id must not be able to put that model id back in the UI.
"""

from __future__ import annotations

import json

from app.controller import UIController
from brain.adapter import BrainOSAdapter
from reproducibility.run_provenance import app_version, brainos_version
from tests.fakes import FakeLLMProvider, LooseFakeRuntime

KEY = "sk-session-SECRET-0001"
FACT = "For Project Atlas, the production database is PostgreSQL 16."


def _controller() -> UIController:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"])))

    return UIController(provider_factory=FakeLLMProvider, adapter_factory=adapter_factory)


def _connected(controller: UIController) -> str:
    return controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id


def test_export_carries_limits_versions_and_a_transcript_digest() -> None:
    controller = _controller()
    session_id = _connected(controller)
    controller.chat(session_id, FACT)

    payload = controller.export_session(session_id)

    assert payload["limits"]["max_input_tokens"] > 0
    assert payload["limits"]["max_session_tokens"] > 0
    assert payload["versions"]["application"] == app_version()
    assert payload["versions"]["brainos"] == brainos_version()
    assert payload["versions"]["benchmark"] == "context-rot-v1"
    assert payload["chat_history_sha256"]
    # The digest is replayable: the exported transcript hashes to it.
    import hashlib

    digest = hashlib.sha256()
    for message in payload["messages"]:
        digest.update(str(message["role"]).encode("utf-8"))
        digest.update(b"\n")
        digest.update(str(message["content"]).encode("utf-8"))
        digest.update(b"\n")
    assert payload["chat_history_sha256"] == digest.hexdigest()

    assert KEY not in json.dumps(payload)


def test_transcript_digest_changes_when_the_transcript_changes() -> None:
    controller = _controller()
    session_id = _connected(controller)
    controller.chat(session_id, FACT)
    first = controller.export_session(session_id)["chat_history_sha256"]

    controller.chat(session_id, "What database does Project Atlas use?")
    second = controller.export_session(session_id)["chat_history_sha256"]

    assert first != second


def test_export_is_credential_free_even_after_many_turns() -> None:
    controller = _controller()
    session_id = _connected(controller)
    for message in (FACT, "What database does Project Atlas use?", "Thanks."):
        controller.chat(session_id, message)

    payload = controller.export_session(session_id)
    rendered = json.dumps(payload)

    assert KEY not in rendered
    assert '"api_key"' not in rendered


def test_connect_drops_model_ids_that_echo_the_key() -> None:
    """A provider echoing the pasted key in a model id cannot reach the UI."""

    echo = f"gw-{KEY}"

    def listing(config):
        provider = FakeLLMProvider(config)
        provider._models = (echo, "gpt-4o-mini", "gpt-4o")
        return provider

    controller = UIController(provider_factory=listing)

    view = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY)

    assert echo not in view.models
    assert "gpt-4o-mini" in view.models
    assert view.connected is True
