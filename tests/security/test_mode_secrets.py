"""Phase 6 security: the new evidence path must not widen the credential surface.

Phase 5's rule was that persistence must not widen what can leak. Phase 6 adds a
second route from conversation content into a prompt — the lexical retriever —
and a second route out of it, the chunk panel. Both are covered here, plus the
architectural rule that keeps the baseline modes runnable without BrainOS: the
``baselines`` package must never import the runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.controller import UIController
from app.service import ConversationService
from app.state import ContextSettings, ProviderConfig, SessionState
from baselines.modes import MODE_BRAINOS_RAG, MODE_RAG
from brain.adapter import BrainOSAdapter
from evaluation.datasets import BenchmarkTask
from evaluation.modes import replay_task
from tests.fakes import FakeLLMProvider, FakeRuntime

KEY = "sk-modes-SECRET-7777"
LEAKY_TURN = f"My API key is {KEY} and the production database is PostgreSQL 16."
QUESTION = "What database is in production?"


def _controller(mode: str) -> tuple[UIController, str]:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(FakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"])))

    controller = UIController(
        provider_factory=FakeLLMProvider, adapter_factory=adapter_factory
    )
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=KEY
    ).session_id
    controller.apply_mode(session_id, mode)
    return controller, session_id


def _bury(controller: UIController, session_id: str) -> None:
    """Push the leaky turn outside the recent window so only retrieval finds it."""

    for index in range(10):
        controller.chat(session_id, f"How is the rollout looking today? Check {index}.")


@pytest.mark.parametrize("mode", [MODE_RAG, MODE_BRAINOS_RAG])
def test_a_key_pasted_into_the_transcript_never_reaches_the_prompt(mode: str) -> None:
    controller, session_id = _controller(mode)
    controller.chat(session_id, LEAKY_TURN)
    _bury(controller, session_id)

    view = controller.chat(session_id, QUESTION)

    assert view.chunk_rows, "the retriever must have found the turn for this to mean anything"
    assert KEY not in view.prompt
    assert KEY not in str(view.chunk_rows)
    assert KEY not in str(view.stats)
    assert KEY not in view.summary
    assert KEY not in view.trace


@pytest.mark.parametrize("mode", [MODE_RAG, MODE_BRAINOS_RAG])
def test_a_key_pasted_into_the_transcript_never_reaches_diagnostics(mode: str) -> None:
    controller, session_id = _controller(mode)
    controller.chat(session_id, LEAKY_TURN)
    _bury(controller, session_id)
    controller.chat(session_id, QUESTION)

    assert KEY not in str(controller.service(session_id).inspect())
    assert KEY not in str(controller.export_session(session_id))
    assert KEY not in controller.export_text(session_id)


def test_the_service_guards_chunk_text_before_the_builder_sees_it() -> None:
    state = SessionState(
        provider=ProviderConfig(model="fake-model", api_key=KEY),
        context=ContextSettings().with_mode_defaults(MODE_RAG),
    )
    service = ConversationService(state, adapter=BrainOSAdapter(FakeRuntime()))
    service.handle_user_message(LEAKY_TURN)
    for index in range(10):
        service.handle_user_message(f"Rollout check {index} is steady.")

    turn = service.handle_user_message(QUESTION)

    assert turn.retrieved_chunks
    assert all(KEY not in chunk.text for chunk in turn.retrieved_chunks)
    assert all(KEY not in str(chunk.to_dict()) for chunk in turn.retrieved_chunks)


def test_memory_free_modes_do_not_retrieve_either() -> None:
    """The credential guard is not the only protection: no chunks, no leak."""

    controller, session_id = _controller("full_context")
    controller.chat(session_id, LEAKY_TURN)

    view = controller.chat(session_id, QUESTION)

    assert view.chunk_rows == []
    assert KEY not in view.prompt


def _secret_task(turns: int = 10) -> BenchmarkTask:
    conversation: list[dict[str, str]] = [{"role": "user", "content": LEAKY_TURN}]
    for index in range(turns):
        conversation.append({"role": "user", "content": f"Rollout check {index} is steady."})
    return BenchmarkTask(
        task_id="secret-task",
        category="single_hop",
        conversation=conversation,
        question=QUESTION,
        expected_answer="PostgreSQL",
        conversation_length=len(conversation),
    )


def test_a_default_replay_holds_no_provider_credential_at_all() -> None:
    """The replay path configures no provider, so there is nothing to leak."""

    record = replay_task(_secret_task(), MODE_RAG).to_dict()

    assert "api_key" not in str(record).lower()
    assert record["stats"]["token_counter"]


def _keyed_service(mode: str) -> ConversationService:
    state = SessionState(
        provider=ProviderConfig(model="fake-model", api_key=KEY),
        context=ContextSettings().with_mode_defaults(mode),
    )
    return ConversationService(state, adapter=BrainOSAdapter(FakeRuntime()))


def _prompt_text(replay) -> str:
    return "\n".join(message["content"] for message in replay.prompt_messages)


def test_a_replay_with_a_session_key_redacts_it_from_the_prompt() -> None:
    """A replay that *does* hold a key must guard the new evidence path.

    Raw history inside the recent window is replayed verbatim by design — Mode A
    has to send the real conversation. What must not happen is a credential
    being *re-published* as retrieved evidence, which is the route Phase 6 added.
    """

    replay = replay_task(_secret_task(), MODE_RAG, service_factory=_keyed_service)

    assert replay.selected_chunk_count > 0, "the retriever must have run"
    assert KEY not in _prompt_text(replay)
    assert KEY not in str(replay.to_dict()["retrieved_chunk_ids"])


def test_a_keyless_replay_has_no_credential_to_redact() -> None:
    """Documents the boundary rather than asserting a defect.

    The replay path configures no provider, so the session holds no credential
    and the guard has nothing to match. A key that appears in the *dataset's*
    conversation is that dataset's content, and it is replayed faithfully —
    which is what makes the keyed case above a real test of the guard rather
    than a test of an empty secret list.
    """

    replay = replay_task(_secret_task(), MODE_RAG)

    assert replay.selected_chunk_count > 0
    assert KEY in _prompt_text(replay)
    assert "api_key" not in str(replay.to_dict()).lower()


def test_the_baselines_package_never_imports_the_brainos_runtime() -> None:
    """Plan rule: BrainOS stays behind the adapter.

    Mode C exists to be the BrainOS-free baseline, so importing the runtime here
    would quietly invalidate the comparison — and would make the modes
    uninstallable without the integration extra.
    """

    package = Path(__file__).resolve().parents[2] / "src" / "baselines"
    offenders: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders.extend(
                f"{path.name}: {name}"
                for name in names
                if name.split(".")[0] in {"brainos_runtime", "brainos_eval", "brainos"}
            )

    assert offenders == []


def test_the_evaluation_mode_seam_does_not_import_a_provider() -> None:
    """A replay must be runnable without any provider installed or configured."""

    path = Path(__file__).resolve().parents[2] / "src" / "evaluation" / "modes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert not any(name.startswith("providers") for name in imported), imported
