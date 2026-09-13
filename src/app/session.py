"""Session lifecycle and secret handling.

The initial implementation is intentionally process-local. It provides the
boundary that can later be backed by a more formal session store without
changing the UI or provider interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from .state import SessionState


@dataclass
class SessionManager:
    """Manage isolated, in-memory application sessions.

    This class must not be replaced with persistent storage for API keys. A
    production deployment should also configure an expiry/eviction policy.
    """

    _sessions: dict[str, SessionState] = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def start(self) -> SessionState:
        """Create and register a new isolated session."""

        state = SessionState()
        with self._lock:
            self._sessions[state.session_id] = state
        return state

    def get(self, session_id: str) -> SessionState | None:
        """Return a session only when its exact identifier is supplied."""

        with self._lock:
            return self._sessions.get(session_id)

    def end(self, session_id: str) -> bool:
        """Clear secrets and remove a session from process memory."""

        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return False
        state.clear_credentials()
        state.clear_conversation()
        return True

    def clear_all(self) -> None:
        """Emergency cleanup for shutdown hooks and tests."""

        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for state in sessions:
            state.clear_credentials()
            state.clear_conversation()
