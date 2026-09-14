"""SQLite persistence backend for the Phase 5 storage protocols.

Design rules:

* **Thread safety by construction.** Gradio runs callbacks in worker threads,
  so every operation opens and closes its own connection instead of sharing
  one. The database file is server-wide; isolation happens at the row level
  through exact ``session_id`` / ``conversation_id`` filters.
* **Secrets never reach disk.** Secret-named fields are stripped recursively
  from every metadata/payload blob before it is written, as defence in depth
  over the service-layer sanitization. Callers additionally redact the active
  session key from message/memory text before persisting.
* **A deletion removes the bytes, not just the rows.** Every connection sets
  ``PRAGMA secure_delete=ON``, so freed content is zeroed rather than left in
  the file for a later reader, and the session-level deletes are followed by a
  ``VACUUM`` that rewrites the file without the freed pages. Phase 5 documented
  this as a hardening candidate; Phase 13 closes it, because "delete session"
  that leaves the transcript recoverable from the file is not the control the
  UI promises.
* **Runtime data stays out of Git.** The default path lives under ``data/``
  and the ``*.sqlite3`` ignore rule covers it; ``BRAINOS_LAB_DB`` overrides
  the location (HF Spaces deployments point it at a writable directory).
* The memory table is a **mirror** for inspection and export; BrainOS remains
  authoritative for recall.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .conversations import ConversationMessage

DEFAULT_DATABASE_ENV = "BRAINOS_LAB_DB"
DEFAULT_DATABASE_PATH = Path("data") / "brainos_lab.sqlite3"

_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "client_secret",
        "refresh_token",
    }
)

_MESSAGES_DDL = """
CREATE TABLE IF NOT EXISTS messages (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}'
)
"""
_MESSAGES_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS messages_session_conversation "
    "ON messages (session_id, conversation_id, seq)"
)

_MEMORIES_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    session_id      TEXT NOT NULL,
    memory_id       TEXT NOT NULL,
    text            TEXT NOT NULL DEFAULT '',
    memory_type     TEXT NOT NULL DEFAULT '',
    relevance       REAL,
    source_turn     INTEGER,
    created_at      TEXT NOT NULL DEFAULT '',
    observed_at     TEXT NOT NULL DEFAULT '',
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active',
    payload         TEXT NOT NULL DEFAULT '{}',
    updated_at      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (session_id, memory_id)
)
"""

_EVAL_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id     TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    metrics    TEXT NOT NULL DEFAULT '{}'
)
"""
_EVAL_SESSION_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS evaluation_runs_session ON evaluation_runs (session_id)"
)


def default_database_path() -> Path:
    """Return the configured database location, or the repository default."""

    configured = os.getenv(DEFAULT_DATABASE_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_DATABASE_PATH


def strip_secret_fields(value: Any) -> Any:
    """Recursively drop secret-named mapping fields from a JSON-ready value.

    Strings are left untouched: transcript and memory text fidelity is a
    deliberate decision (callers redact the session key itself), and scrubbing
    arbitrary text here would corrupt legitimate content.
    """

    if isinstance(value, Mapping):
        return {
            str(key): strip_secret_fields(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in _SECRET_FIELD_NAMES
        }
    if isinstance(value, (list, tuple)):
        return [strip_secret_fields(item) for item in value]
    return value


def _dumps(value: Any) -> str:
    return json.dumps(strip_secret_fields(value), default=str, ensure_ascii=False)


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {}


class SqliteStore:
    """Shared connection handling for the SQLite-backed stores."""

    def __init__(self, database_path: Path | str | None = None) -> None:
        self.database_path = Path(database_path) if database_path else default_database_path()
        self._ddl: tuple[str, ...] = ()

    def _connect(self) -> sqlite3.Connection:
        parent = self.database_path.parent
        if str(parent) not in {"", "."}:
            parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        # Phase 13: zero deleted content instead of merely unlinking the rows.
        # ``secure_delete`` is per-connection, so it has to be set on every one;
        # a connection that forgets it would leave recoverable bytes behind.
        connection.execute("PRAGMA secure_delete=ON")
        for statement in self._ddl:
            connection.execute(statement)
        return connection

    def secure_delete_enabled(self) -> bool:
        """Whether this database file reports ``secure_delete`` as on.

        Exposed for tests and for the security report: the pragma is a property
        of a connection, so the honest check is to open one and ask.
        """

        with self._connect() as connection:
            row = connection.execute("PRAGMA secure_delete").fetchone()
        return bool(row[0]) if row is not None else False

    def vacuum(self) -> bool:
        """Rewrite the database file without its freed pages.

        Called after a user-data deletion so the file on disk stops containing
        the removed content, not just the rows that pointed at it. Best-effort
        by contract: it cannot run inside an open transaction, and a failure
        here must never break the delete that prompted it, so the result is a
        boolean the caller may log rather than an exception.
        """

        if not self.database_path.exists():
            return False
        connection = None
        try:
            connection = sqlite3.connect(str(self.database_path), timeout=30.0)
            connection.isolation_level = None
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("VACUUM")
            return True
        except sqlite3.Error:
            return False
        finally:
            if connection is not None:
                connection.close()


class SqliteConversationStore(SqliteStore):
    """Append-only transcript store with session/conversation isolation."""

    def __init__(self, database_path: Path | str | None = None) -> None:
        super().__init__(database_path)
        self._ddl = (_MESSAGES_DDL, _MESSAGES_INDEX_DDL)

    def append(self, message: ConversationMessage) -> None:
        """Persist one message; secret-named metadata fields are stripped."""

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages "
                "(session_id, conversation_id, role, content, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(message.session_id),
                    str(message.conversation_id),
                    str(message.role),
                    str(message.content),
                    str(message.created_at),
                    _dumps(message.metadata or {}),
                ),
            )

    def list_messages(self, session_id: str, conversation_id: str) -> list[ConversationMessage]:
        """Return one conversation's messages in insertion order, exactly filtered."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id, conversation_id, role, content, created_at, metadata "
                "FROM messages WHERE session_id = ? AND conversation_id = ? ORDER BY seq",
                (str(session_id), str(conversation_id)),
            ).fetchall()
        return [
            ConversationMessage(
                session_id=row["session_id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                metadata=_loads(row["metadata"]),
            )
            for row in rows
        ]

    def clear(self, session_id: str, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE session_id = ? AND conversation_id = ?",
                (str(session_id), str(conversation_id)),
            )

    def list_conversations(self, session_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT conversation_id FROM messages "
                "WHERE session_id = ? ORDER BY conversation_id",
                (str(session_id),),
            ).fetchall()
        return [row["conversation_id"] for row in rows]

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (str(session_id),)
            )


class SqliteMemoryStore(SqliteStore):
    """Mirror of sanitized memory records for inspection and export."""

    def __init__(self, database_path: Path | str | None = None) -> None:
        super().__init__(database_path)
        self._ddl = (_MEMORIES_DDL,)

    def save_memories(self, session_id: str, memories: list[dict[str, Any]]) -> None:
        """Upsert records by ``memory_id``; counters update, rows never duplicate."""

        if not memories:
            return
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            for record in memories:
                if not isinstance(record, Mapping):
                    continue
                safe = strip_secret_fields(dict(record))
                relevance = safe.get("relevance")
                source_turn = safe.get("source_turn")
                retrieval_count = safe.get("retrieval_count")
                connection.execute(
                    "INSERT INTO memories (session_id, memory_id, text, memory_type, "
                    "relevance, source_turn, created_at, observed_at, retrieval_count, "
                    "status, payload, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id, memory_id) DO UPDATE SET "
                    "text=excluded.text, memory_type=excluded.memory_type, "
                    "relevance=excluded.relevance, source_turn=excluded.source_turn, "
                    "retrieval_count=excluded.retrieval_count, status=excluded.status, "
                    "payload=excluded.payload, updated_at=excluded.updated_at",
                    (
                        str(session_id),
                        str(safe.get("memory_id", "")),
                        str(safe.get("text", "")),
                        str(safe.get("memory_type", "")),
                        float(relevance) if isinstance(relevance, (int, float)) else None,
                        int(source_turn) if isinstance(source_turn, int) else None,
                        str(safe.get("created_at", "")),
                        str(safe.get("observed_at", "")),
                        int(retrieval_count) if isinstance(retrieval_count, int) else 0,
                        str(safe.get("status") or "active"),
                        _dumps(safe),
                        now,
                    ),
                )

    def list_memories(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM memories WHERE session_id = ? ORDER BY updated_at, memory_id",
                (str(session_id),),
            ).fetchall()
        payloads = [_loads(row["payload"]) for row in rows]
        return [payload for payload in payloads if isinstance(payload, dict)]

    def clear(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM memories WHERE session_id = ?", (str(session_id),)
            )


class SqliteEvaluationStore(SqliteStore):
    """Run metadata and metrics persistence for the evaluation phases."""

    def __init__(self, database_path: Path | str | None = None) -> None:
        super().__init__(database_path)
        self._ddl = (_EVAL_RUNS_DDL, _EVAL_SESSION_INDEX_DDL)

    def save_run(self, run_id: str, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
        from datetime import datetime, timezone

        session_id = str((metadata or {}).get("session_id", ""))
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_runs (run_id, session_id, created_at, metadata, metrics) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET session_id=excluded.session_id, "
                "metadata=excluded.metadata, metrics=excluded.metrics",
                (
                    str(run_id),
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                    _dumps(metadata or {}),
                    _dumps(metrics or {}),
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id, session_id, created_at, metadata, metrics "
                "FROM evaluation_runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "metadata": _loads(row["metadata"]),
            "metrics": _loads(row["metrics"]),
        }

    def delete_session_data(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM evaluation_runs WHERE session_id = ?", (str(session_id),)
            )


__all__ = [
    "DEFAULT_DATABASE_ENV",
    "DEFAULT_DATABASE_PATH",
    "SqliteConversationStore",
    "SqliteEvaluationStore",
    "SqliteMemoryStore",
    "SqliteStore",
    "default_database_path",
    "strip_secret_fields",
]
