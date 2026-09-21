"""Phase 18: the plan's log rule, enforced.

Plan §24 asks for a test that "API key never appears in logs", and §13's first
security requirement is "Never write API keys to logs". This repository has no
logger at all: the single diagnostic a user-data failure produces is a
``print(..., file=sys.stderr)`` that carries an exception *type*, never its
message (``app.service._warn_storage``). Credential-freedom of the artifacts a
run writes is already pinned by ``tests/security/test_artifact_scan.py`` and
``tests/security/test_pipeline_secrets.py``; what was not pinned is the *log
surface* — logging records, stdout, stderr, and warnings.

These tests pin it from both sides:

* dynamically — a session that holds a credential and then fails a connect,
  fails a chat, fails a storage write, and pastes a key into chat is run while
  every logging record, stdout write, stderr write, and warning is captured, and
  the sentinel key is searched for in all of them;
* statically — no shipped module may configure logging, attach a handler, or
  emit a record, and importing every shipped package must not install a sink.

The point is not that logging is forbidden. The point is that adding one here is
a security change: this module fails when it happens, so the redaction review
happens before a key is written to a file an operator can read.
"""

from __future__ import annotations

import ast
import importlib
import io
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.controller import UIController
from app.service import ConversationService
from app.state import ProviderConfig, SessionState
from brain.adapter import BrainOSAdapter
from providers import OpenAIProvider
from storage.sqlite import SqliteConversationStore, SqliteMemoryStore
from tests.fakes import FakeLLMProvider, FakeRuntime, RecordingProvider

KEY = "sk-log-SENTINEL-9876543210"

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

#: Logger namespaces this repository owns. A record from one of these is a
#: record this project caused; a record from a dependency is still scanned for
#: the credential but is not itself a policy violation.
OWNED_LOGGERS = (
    "app",
    "baselines",
    "brain",
    "evaluation",
    "providers",
    "reproducibility",
    "security",
    "storage",
)


class _CollectingHandler(logging.Handler):
    """Collect every record the root logger sees, whatever its level."""

    def __init__(self, sink: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.DEBUG)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record)


@dataclass
class CapturedIO:
    """Everything an operator, a log aggregator, or a file could observe."""

    records: list[logging.LogRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    def render(self) -> str:
        """Return the captured surface as one searchable string."""

        lines = [record.getMessage() for record in self.records]
        lines.extend(self.warnings)
        lines.extend([self.stdout, self.stderr])
        return "\n".join(lines)

    def owned_records(self) -> list[logging.LogRecord]:
        """Return records emitted from this repository's own namespaces."""

        return [
            record
            for record in self.records
            if record.name == OWNED_LOGGERS
            or record.name.startswith(tuple(f"{name}." for name in OWNED_LOGGERS))
        ]


@contextmanager
def captured_io() -> Iterator[CapturedIO]:
    """Capture logging, warnings, stdout, and stderr for the enclosed block."""

    captured = CapturedIO()
    handler = _CollectingHandler(captured.records)
    root = logging.getLogger()
    previous_level = root.level
    out, err = io.StringIO(), io.StringIO()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with redirect_stdout(out), redirect_stderr(err):
                yield captured
            captured.warnings = [str(item.message) for item in caught]
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        captured.stdout = out.getvalue()
        captured.stderr = err.getvalue()


def _shipped_packages() -> list[str]:
    """Return the importable top-level packages under ``src/``."""

    return sorted(
        path.name
        for path in SRC.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )


def _shipped_files() -> list[Path]:
    """Return every shipped module and the Space entry point."""

    return sorted([*SRC.rglob("*.py"), ROOT / "app.py"])


def _adapter_factory(**kwargs: object) -> BrainOSAdapter:
    return BrainOSAdapter(
        FakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
    )


# --------------------------------------------------------------------------- #
# The static invariant: there is no log route to leak through
# --------------------------------------------------------------------------- #


def test_importing_every_shipped_package_installs_no_log_sink() -> None:
    """The test runner installs its own handler; the application must not."""

    root = logging.getLogger()
    before = list(root.handlers)

    with captured_io() as captured:
        for name in _shipped_packages():
            importlib.import_module(name)

    assert root.handlers == before
    assert captured.owned_records() == []


def test_no_shipped_module_configures_logging_or_emits_a_record() -> None:
    """A logger added to this application must arrive with a redaction review."""

    offenders: list[str] = []
    for path in _shipped_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "logging"
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} logging.{node.attr}")
            elif isinstance(node, ast.ImportFrom) and node.module == "logging":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} from logging import")

    assert offenders == []


# --------------------------------------------------------------------------- #
# The dynamic invariant: a credentialed failure leaks nothing
# --------------------------------------------------------------------------- #


def test_a_refused_connect_and_a_failed_chat_never_write_the_key() -> None:
    def failing(config: Any) -> FakeLLMProvider:
        return FakeLLMProvider(config, fail=True)

    controller = UIController(provider_factory=failing, adapter_factory=_adapter_factory)

    with captured_io() as captured:
        view = controller.connect(
            None, provider="openai", model="gpt-4o-mini", api_key=KEY
        )
        controller.chat(view.session_id, "Hello")

    assert view.connected is False
    assert "[redacted]" in view.status
    assert KEY not in captured.render()
    assert captured.owned_records() == []


def test_the_storage_diagnostic_prints_the_exception_type_and_nothing_else() -> None:
    """Persistence is best-effort, and its failure notice may not quote content."""

    class FailingStore(SqliteConversationStore):
        def append(self, message: Any) -> None:
            raise RuntimeError(f"disk full while storing {KEY} and the user's message")

    state = SessionState()
    service = ConversationService(
        state,
        adapter=BrainOSAdapter(FakeRuntime(state.session_id, state.actor_id)),
        provider=FakeLLMProvider(ProviderConfig(model="fake-model", api_key=KEY)),
        conversation_store=FailingStore(Path(":memory:")),
    )

    with captured_io() as captured:
        service.handle_user_message("For Project Atlas, the database is PostgreSQL 16.")

    assert "[storage]" in captured.stderr
    assert "RuntimeError" in captured.stderr
    assert "disk full" not in captured.stderr
    assert KEY not in captured.render()


def test_a_full_session_leaves_no_application_log_record(tmp_path: Path) -> None:
    db = tmp_path / "lab.sqlite3"
    controller = UIController(
        provider_factory=FakeLLMProvider,
        adapter_factory=_adapter_factory,
        conversation_store=SqliteConversationStore(db),
        memory_store=SqliteMemoryStore(db),
    )

    with captured_io() as captured:
        session_id = controller.connect(
            None, provider="openai", model="gpt-4o-mini", api_key=KEY
        ).session_id
        controller.chat(session_id, "For Project Atlas, the database is PostgreSQL 16.")
        controller.chat(session_id, "Which database does Project Atlas use?")
        controller.clear_conversation(session_id)
        controller.end_session(session_id)

    assert KEY not in captured.render()
    assert captured.owned_records() == []


def test_a_key_pasted_into_chat_never_reaches_a_stream() -> None:
    """The user is the most likely source of a leaked key, so chat is a route too."""

    controller = UIController(
        provider_factory=FakeLLMProvider, adapter_factory=_adapter_factory
    )
    session_id = controller.connect(
        None, provider="openai", model="gpt-4o-mini", api_key=""
    ).session_id

    with captured_io() as captured:
        controller.chat(session_id, f"My key is {KEY}, please remember it.")

    assert KEY not in captured.render()
    assert captured.owned_records() == []


# --------------------------------------------------------------------------- #
# The CLI surfaces a researcher actually watches
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_the_single_mode_cli_writes_no_credential_to_stdout_or_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.run as run_module

    monkeypatch.setenv("LAB_LOG_KEY", KEY)
    monkeypatch.setattr(
        run_module,
        "build_provider",
        lambda spec, *, api_key, **kwargs: RecordingProvider(
            fail=f"401 unauthorized for key {api_key}"
        ),
    )

    with captured_io() as captured:
        code = run_module.main(
            [
                "--mode",
                "brainos",
                "--dataset",
                "benchmarks/context_rot/dataset.jsonl",
                "--limit",
                "1",
                "--generate",
                "--model",
                "fake-model",
                "--api-key-env",
                "LAB_LOG_KEY",
                "--output",
                str(tmp_path / "run.json"),
            ]
        )

    assert code == 0
    assert "mode=brainos" in captured.stdout, "the summary must still print"
    assert KEY not in captured.render()


def test_the_pipeline_cli_writes_no_credential_to_stdout_or_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.pipeline as pipeline_module

    monkeypatch.setenv("LAB_LOG_PIPELINE_KEY", KEY)
    monkeypatch.setattr(
        pipeline_module,
        "build_provider",
        lambda spec, *, api_key, **kwargs: RecordingProvider(
            text=f"Sure, the key is {api_key}."
        ),
    )

    with captured_io() as captured:
        code = pipeline_module.main(
            [
                "--model",
                "fake-model",
                "--api-key-env",
                "LAB_LOG_PIPELINE_KEY",
                "--limit",
                "1",
                "--modes",
                "full_context",
                "--no-plots",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert "pipeline=" in captured.stdout, "the stage summary must still print"
    assert KEY not in captured.render()
    assert code != 0, "a credential echoed into an answer is quarantined, not ignored"


def test_a_provider_that_fails_mid_session_stays_redacted_on_every_surface() -> None:
    """The turn's error string reaches the UI, so it is a log surface too.

    The failing exception is raised *inside* the real OpenAI adapter, with the
    key in its message — exactly what an ``openai`` SDK authentication error
    looks like — so this test covers the whole route: SDK exception, adapter
    redaction, service turn record, diagnostics, stdout, stderr, and records.
    """

    class ExplodingCompletions:
        def create(self, **request: Any) -> Any:
            raise RuntimeError(f"401 Unauthorized: invalid api key {KEY}")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=ExplodingCompletions()),
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[])),
    )
    state = SessionState()
    config = ProviderConfig(provider="openai", model="gpt-4o-mini", api_key=KEY)
    state.provider = config
    service = ConversationService(
        state,
        adapter=BrainOSAdapter(FakeRuntime(state.session_id, state.actor_id)),
        provider=OpenAIProvider(config, client=client),
    )

    with captured_io() as captured:
        turn = service.handle_user_message("Which database do we use?")
        diagnostics = service.inspect()

    assert turn.generated is False
    assert turn.error is not None
    assert "[redacted]" in turn.error
    assert KEY not in captured.render()
    assert KEY not in str(diagnostics)
    assert captured.owned_records() == []
