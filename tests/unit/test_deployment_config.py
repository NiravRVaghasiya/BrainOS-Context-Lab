"""Phase 14 — deployment configuration is an enforced contract, not folklore.

These tests pin down the things an HF Space (or any other public deployment)
relies on that aren't visible from inside a running chat turn:

* ``app.py`` resolves ``src/`` on ``sys.path`` and exposes ``main``,
* ``requirements.txt`` lists the UI, provider, BrainOS, and evaluation deps
  and contains **no** shared API key and no hard-coded endpoint,
* ``packages.txt`` exists (the Space SDK refuses a missing one),
* the SQLite backend reads ``BRAINOS_LAB_DB=:memory:`` as a signal to run in
  memory (ephemeral/stateless deployments), and opens disk-backed stores in
  WAL mode with a busy timeout so concurrent Gradio workers don't lock,
* the Gradio entry point calls ``.queue()`` with bounded concurrency so a
  public Space cannot be driven into unbounded parallel provider calls,
* the UI header honestly discloses whether persistence is enabled.

These tests do not require Gradio or BrainOS to be installed; they read files
and exercise the storage module with dependency-free assertions.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _reload_sqlite(monkeypatch: pytest.MonkeyPatch, db_env: str | None):
    """Reload ``storage.sqlite`` with ``BRAINOS_LAB_DB`` pinned to ``db_env``."""

    if db_env is None:
        monkeypatch.delenv("BRAINOS_LAB_DB", raising=False)
    else:
        monkeypatch.setenv("BRAINOS_LAB_DB", db_env)
    import storage.sqlite as sqlite_mod

    return importlib.reload(sqlite_mod)


def _reload_ui(monkeypatch: pytest.MonkeyPatch, db_env: str | None):
    """Reload ``app.ui`` so ``_header_markdown()`` picks up the new env."""

    sqlite_mod = _reload_sqlite(monkeypatch, db_env)
    import app.ui as ui_mod

    importlib.reload(ui_mod)
    return ui_mod, sqlite_mod


# --------------------------------------------------------------------------- #
# Space files exist and are well-formed
# --------------------------------------------------------------------------- #


def test_app_py_exists_and_bootstraps_the_source_path() -> None:
    text = _read("app.py")
    assert "src_dir" in text
    assert 'sys.path.insert(0, str(src_dir))' in text
    assert "from app.ui import main" in text


def test_requirements_txt_lists_the_runtime_dependencies() -> None:
    text = _read("requirements.txt")
    for required in ("gradio", "openai", "brainos-cli", "matplotlib", "pandas"):
        assert required in text, f"requirements.txt must list {required}"
    assert "-e ." in text
    # No shared key / demo token / placeholder endpoint.
    assert "sk-" not in text
    assert "hf_" not in text
    assert "api_key" not in text.lower()
    # Pinned BrainOS commit hash from Phase 0 is present (not just "main").
    assert "1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc" in text


def test_packages_txt_exists_and_is_comment_or_apt_packages_only() -> None:
    """Gradio Spaces accept an empty packages.txt but not a missing one."""

    path = REPO_ROOT / "packages.txt"
    assert path.exists()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert re.match(r"^[a-z0-9][a-z0-9.+-]*$", stripped), (
            f"packages.txt line does not look like an apt package: {stripped!r}"
        )


# --------------------------------------------------------------------------- #
# SQLite backend deployment behaviour
# --------------------------------------------------------------------------- #


def test_memory_sentinel_switches_to_shared_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_mod = _reload_sqlite(monkeypatch, ":memory:")

    assert sqlite_mod.default_database_path() == ":memory:"
    assert sqlite_mod.persistence_enabled() is False

    store = sqlite_mod.SqliteConversationStore()
    assert store.database_path == ":memory:"
    with store._connect() as conn:
        jm = conn.execute("PRAGMA journal_mode").fetchone()
        assert jm[0] == "memory", "in-memory stores must not try WAL"


def test_disk_backed_store_uses_wal_busy_timeout_secure_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sqlite_mod = _reload_sqlite(monkeypatch, str(tmp_path / "lab.sqlite3"))

    store = sqlite_mod.SqliteConversationStore(tmp_path / "lab.sqlite3")
    with store._connect() as conn:
        jm = conn.execute("PRAGMA journal_mode").fetchone()
        busy = conn.execute("PRAGMA busy_timeout").fetchone()
        sync = conn.execute("PRAGMA synchronous").fetchone()
        sd = conn.execute("PRAGMA secure_delete").fetchone()
    assert jm[0] == "wal", "Phase 14 requires WAL for concurrent readers"
    assert busy[0] >= 30000, "busy timeout must be at least 30s"
    assert sync[0] in (1, 2), "synchronous must be NORMAL or FULL"
    assert sd[0] == 1, "secure_delete stays on in Phase 14"


def test_two_memory_stores_share_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared-cache URI is what makes per-op connections see data."""

    sqlite_mod = _reload_sqlite(monkeypatch, ":memory:")
    store_a = sqlite_mod.SqliteConversationStore()
    store_b = sqlite_mod.SqliteConversationStore()
    from storage.conversations import ConversationMessage

    store_a.append(
        ConversationMessage(
            session_id="s",
            conversation_id="s-main",
            role="user",
            content="hello",
            created_at="2026-01-01T00:00:00+00:00",
            metadata={},
        )
    )
    rows = store_b.list_messages("s", "s-main")
    assert len(rows) == 1
    assert rows[0].content == "hello"


def test_deleting_a_session_checkpoints_the_wal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After a session delete, the WAL must not retain a plaintext marker."""

    sqlite_mod = _reload_sqlite(monkeypatch, str(tmp_path / "lab.sqlite3"))
    MARKER = "DEPLOY-TEST-MARKER-4a9f2c"
    store = sqlite_mod.SqliteConversationStore(tmp_path / "lab.sqlite3")
    from storage.conversations import ConversationMessage

    store.append(
        ConversationMessage(
            session_id="s",
            conversation_id="s-main",
            role="user",
            content=f"the secret is {MARKER}",
            created_at="2026-01-01T00:00:00+00:00",
            metadata={},
        )
    )
    store.delete_session("s")
    wal = tmp_path / "lab.sqlite3-wal"
    if wal.exists():
        assert MARKER.encode() not in wal.read_bytes(), (
            "delete_session must TRUNCATE the WAL so deleted content is gone"
        )


# --------------------------------------------------------------------------- #
# UI entry point
# --------------------------------------------------------------------------- #


def test_main_enables_a_bounded_queue_for_public_traffic() -> None:
    """Phase 14: a public surface must bound concurrency via Gradio's queue."""

    src = _read("src/app/ui.py")
    assert "demo.queue(" in src, "main() must call demo.queue() before launch"
    assert "default_concurrency_limit" in src
    assert "max_size" in src
    assert "api_open=False" in src, "internal API endpoints must not be exposed"
    queue_pos = src.index("demo.queue(")
    launch_pos = src.index("demo.launch(")
    assert queue_pos < launch_pos


def test_header_discloses_in_memory_when_persistence_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui_mod, _sqlite_mod = _reload_ui(monkeypatch, ":memory:")
    header = ui_mod._header_markdown()
    assert "fully in memory" in header
    assert "server-side SQLite" not in header


def test_header_discloses_persistence_when_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ui_mod, _sqlite_mod = _reload_ui(monkeypatch, str(tmp_path / "lab.sqlite3"))
    header = ui_mod._header_markdown()
    assert "server-side" in header and "SQLite" in header
    assert "fully in memory" not in header


def test_default_database_path_is_repo_local_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_mod = _reload_sqlite(monkeypatch, None)
    assert sqlite_mod.persistence_enabled() is True
    import pathlib

    assert sqlite_mod.default_database_path() == pathlib.Path("data") / "brainos_lab.sqlite3"
