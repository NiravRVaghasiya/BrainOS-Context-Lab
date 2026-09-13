"""Evaluation result persistence abstraction."""

from __future__ import annotations

from typing import Any, Protocol


class EvaluationStore(Protocol):
    """Store run metadata and metrics without secrets."""

    def save_run(self, run_id: str, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Persist one reproducible evaluation result."""

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a run by identifier."""

    def delete_session_data(self, session_id: str) -> None:
        """Remove session-associated evaluation data."""
