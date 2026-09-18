"""Phase 18 hardening tests for the two remaining runtime-risk boundaries.

The earlier phases had broad unit and integration coverage, but two claims still
needed an end-to-end pin:

* a real slow provider must return through the chat timeout instead of merely
  receiving a numeric argument; and
* the SQLite stores must preserve row-level session isolation when several
  Gradio-style worker threads read and write the same database concurrently.

These tests use deterministic fakes and a temporary database. They do not call
an external provider and do not depend on the upstream BrainOS installation.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.controller import UIController, UILimits
from brain.adapter import BrainOSAdapter
from providers.base import ProviderResponse
from storage.conversations import ConversationMessage
from storage.sqlite import (
    SqliteConversationStore,
    SqliteEvaluationStore,
    SqliteMemoryStore,
)
from tests.fakes import FakeLLMProvider, LooseFakeRuntime, SlowProvider

KEY = "sk-phase18-test-key"


def _adapter_factory(**kwargs: Any) -> BrainOSAdapter:
    return BrainOSAdapter(
        LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
    )


class _OverproducingProvider(FakeLLMProvider):
    """Provider that ignores ``max_tokens`` to pin honest overage reporting."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.request_kwargs: dict[str, Any] = {}

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        self.requests.append(messages)
        self.request_kwargs = dict(kwargs)
        return ProviderResponse(
            text="A deliberately long fake completion.",
            model=self.config.model,
            usage={"prompt_tokens": 8, "completion_tokens": 99, "total_tokens": 107},
        )


def test_slow_provider_is_timed_out_end_to_end() -> None:
    """A slow provider cannot hold the chat callback past its configured limit."""

    providers: list[SlowProvider] = []

    def provider_factory(config: Any) -> SlowProvider:
        provider = SlowProvider(config, delay=0.20)
        providers.append(provider)
        return provider

    controller = UIController(
        provider_factory=provider_factory,
        adapter_factory=_adapter_factory,
        limits=UILimits(request_timeout_seconds=0.03),
    )
    session_id = controller.connect(
        None,
        provider="openai",
        model="slow-model",
        api_key=KEY,
        max_output_tokens=7,
    ).session_id

    started_at = time.monotonic()
    view = controller.chat(session_id, "Please answer after the delay.")
    elapsed = time.monotonic() - started_at

    # One provider instance is used for connect validation and another is lazy
    # constructed by the service. The latter is the request-bearing instance.
    request_provider = providers[-1]
    assert request_provider.started.wait(0.5)
    assert request_provider.request_kwargs["max_tokens"] == 7
    assert elapsed < 0.15, "the controller should return at the timeout, not after sleep"
    assert view.turn_usage["timed_out"] is True
    assert view.turn_usage["failed"] is True
    assert "within 0.03 seconds" in view.status
    assert view.usage_report["requests"] == 1
    assert view.usage_report["timeouts"] == 1
    assert view.usage_report["within_limits"] is True
    assert KEY not in view.status

    # Let the finite fake finish before the test releases the provider object;
    # the service has already returned and does not append its late reply.
    assert request_provider.finished.wait(1.0)
    assert view.history[-1]["role"] == "user"
    assert all(item["role"] != "assistant" for item in view.history)


def test_output_ceiling_is_forwarded_and_provider_overage_stays_visible() -> None:
    """A gateway that ignores ``max_tokens`` cannot hide its reported overage."""

    providers: list[_OverproducingProvider] = []

    def provider_factory(config: Any) -> _OverproducingProvider:
        provider = _OverproducingProvider(config)
        providers.append(provider)
        return provider

    controller = UIController(
        provider_factory=provider_factory,
        adapter_factory=_adapter_factory,
        limits=UILimits(max_output_tokens=3),
    )
    session_id = controller.connect(
        None,
        provider="openai",
        model="overproducing-model",
        api_key=KEY,
        max_output_tokens=9,
    ).session_id

    view = controller.chat(session_id, "Return a completion.")
    request_provider = providers[-1]

    assert request_provider.request_kwargs["max_tokens"] == 3
    assert view.history[-1] == {
        "role": "assistant",
        "content": "A deliberately long fake completion.",
    }
    assert view.turn_usage["completion_tokens"] == 99
    assert view.turn_usage["output_limit_exceeded"] is True
    assert view.usage_report["output_tokens"] == 99
    assert view.usage_report["requests"] == 1


def test_input_ceiling_refuses_before_constructing_a_provider_request() -> None:
    """An oversized prompt is not charged or handed to the provider."""

    providers: list[FakeLLMProvider] = []

    def provider_factory(config: Any) -> FakeLLMProvider:
        provider = FakeLLMProvider(config)
        providers.append(provider)
        return provider

    controller = UIController(
        provider_factory=provider_factory,
        adapter_factory=_adapter_factory,
        limits=UILimits(max_input_tokens=1),
    )
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=KEY
    ).session_id

    view = controller.chat(session_id, "This prompt is larger than one token.")

    # The first instance serviced connect/list-model validation. No second
    # instance should be needed because the service rejects before _generate().
    assert len(providers) == 1
    assert providers[0].requests == []
    assert view.turn_usage["input_limit_exceeded"] is True
    assert view.turn_usage["failed"] is True
    assert "max_input_tokens" in view.status
    assert view.usage_report["requests"] == 0
    assert view.usage_report["input_tokens"] == 0
    assert view.usage_report["refused"] == 1
    assert view.usage_report["user_turns"] == 1
    assert KEY not in str(view)


def test_sqlite_stores_preserve_session_isolation_under_worker_load(tmp_path: Path) -> None:
    """Concurrent Gradio-style workers do not lose rows or cross sessions."""

    database = tmp_path / "stress" / "brainos.sqlite3"
    conversations = SqliteConversationStore(database)
    memories = SqliteMemoryStore(database)
    evaluations = SqliteEvaluationStore(database)

    worker_count = 8
    turns_per_worker = 20
    barrier = threading.Barrier(worker_count)

    def write_session(index: int) -> tuple[str, int]:
        session_id = f"phase18-session-{index}"
        conversation_id = f"conversation-{index}"
        barrier.wait(timeout=5)
        for turn in range(turns_per_worker):
            conversations.append(
                ConversationMessage(
                    session_id=session_id,
                    conversation_id=conversation_id,
                    role="user" if turn % 2 == 0 else "assistant",
                    content=f"session {index} private turn {turn}",
                    metadata={"turn": turn},
                )
            )
            memories.save_memories(
                session_id,
                [
                    {
                        "memory_id": f"memory-{index}",
                        "text": f"private fact for session {index}",
                        "memory_type": "FACT",
                        "source_turn": turn,
                        "retrieval_count": turn,
                    }
                ],
            )
            evaluations.save_run(
                f"run-{index}",
                {"session_id": session_id, "worker": index},
                {"last_turn": turn},
            )
            # Reads share the same connection-opening path as the production
            # stores and make the test exercise WAL readers during writes.
            if turn % 4 == 0:
                assert len(conversations.list_messages(session_id, conversation_id)) == turn + 1
                assert len(memories.list_memories(session_id)) == 1
                assert evaluations.get_run(f"run-{index}") is not None
        return session_id, index

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        completed = list(pool.map(write_session, range(worker_count)))

    assert len(completed) == worker_count
    for session_id, index in completed:
        conversation_id = f"conversation-{index}"
        rows = conversations.list_messages(session_id, conversation_id)
        assert len(rows) == turns_per_worker
        assert {row.session_id for row in rows} == {session_id}
        assert all(f"session {index} private" in row.content for row in rows)

        session_memories = memories.list_memories(session_id)
        assert len(session_memories) == 1
        assert session_memories[0]["text"] == f"private fact for session {index}"

        run = evaluations.get_run(f"run-{index}")
        assert run is not None
        assert run["session_id"] == session_id
        assert run["metadata"]["worker"] == index
        assert run["metrics"]["last_turn"] == turns_per_worker - 1

        # Exact filtering is part of the security boundary, not just a count
        # assertion: a neighbouring session's identifier must return nothing.
        other = (index + 1) % worker_count
        assert conversations.list_messages(
            session_id, f"conversation-{other}"
        ) == []
        assert memories.list_memories(f"phase18-session-{other}")[0]["text"] != (
            f"private fact for session {index}"
        )

    assert conversations.journal_mode() == "wal"
    assert conversations.secure_delete_enabled() is True
