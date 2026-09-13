"""Gradio-free controller behind the chat UI.

The web layer is deliberately thin. Every behaviour the plan asks the UI to
expose lives here, where it can be tested without a browser or a web
framework:

* session lifecycle (start / clear conversation / clear memory / end),
* provider connection with credential validation and model listing,
* one chat turn through :class:`ConversationService`,
* rendering of the Memory, Context, and Cognitive Trace panels,
* session export.

Two constraints shape the module:

1. **No credential leaves the process.** The controller accepts a key from the
   form, stores it on the session's :class:`ProviderConfig`, and returns an
   empty string for the key box. Every value it returns passes through
   :func:`brain.trace.sanitize_value` with the active key as a known secret, so
   a key pasted into a conversation cannot be echoed back by a panel either.
2. **BrainOS stays behind the adapter.** The controller talks to
   :class:`ConversationService` only; it never imports ``brainos_runtime``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from brain.adapter import BrainOSAdapter, BrainOSNotConfiguredError, MemoryRecord
from brain.trace import sanitize_value
from providers import ProviderError, create_provider
from providers.base import ProviderConfig, ProviderConfigurationError

from . import panels
from .service import ConversationService
from .session import SessionManager
from .state import ContextSettings, SessionState

PROVIDER_LABELS: tuple[tuple[str, str], ...] = (
    ("OpenAI", "openai"),
    ("OpenAI-compatible", "openai-compatible"),
)
DEFAULT_MODEL_PLACEHOLDER = "gpt-4o-mini"

_BRAINOS_INSTALL_HINT = (
    "BrainOS runtime is not installed, so no memory can be stored or retrieved. "
    "Install the pinned upstream revision with "
    "`pip install -e '.[integration]'`."
)


@dataclass(frozen=True)
class UILimits:
    """Guard rails that protect a bring-your-own-key user from runaway usage.

    These are the two limits the plan's Phase 15 list that matter before any
    network-facing surface exists: a session cannot grow without bound, and a
    single message cannot be large enough to be a paste accident rather than a
    chat turn. Benchmark controls (the rest of Phase 15) are not exposed yet.
    """

    max_turns: int = 200
    max_message_chars: int = 8000


@dataclass(frozen=True)
class ConnectionView:
    """Result of a connect / disconnect / validate action."""

    status: str
    session_id: str = ""
    connected: bool = False
    models: tuple[str, ...] = ()
    model_value: str = ""
    key_value: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnView:
    """Everything the UI panels need after one turn (or after a reset)."""

    #: Session identifier, so the UI can persist the (non-secret) id it needs
    #: even when the controller had to start a fresh session.
    session_id: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    status: str = ""
    stored_rows: list[list[Any]] = field(default_factory=list)
    retrieved_rows: list[list[Any]] = field(default_factory=list)
    dropped_rows: list[list[Any]] = field(default_factory=list)
    conflict_rows: list[list[Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    prompt: str = ""
    trace: str = ""
    trace_events: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SessionView:
    """Status of a session-level action (clear / end / export)."""

    status: str = ""
    session_id: str = ""


class UIController:
    """Own the mapping between UI actions and the application service layer."""

    def __init__(
        self,
        sessions: SessionManager | None = None,
        *,
        provider_factory: Callable[[ProviderConfig], Any] | None = None,
        adapter_factory: Callable[..., BrainOSAdapter] | None = None,
        limits: UILimits | None = None,
    ) -> None:
        self.sessions = sessions or SessionManager()
        self.limits = limits or UILimits()
        self._provider_factory = provider_factory or create_provider
        self._adapter_factory = adapter_factory
        self._services: dict[str, ConversationService] = {}
        self._last_turn: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #

    def ensure_session(self, session_id: str | None) -> SessionState:
        """Return the session for ``session_id``, starting a fresh one if needed.

        A browser may present an identifier the server no longer holds (process
        restart, ended session). The fallback starts a new isolated session
        instead of failing, because an expired id must never restore another
        user's memory.
        """

        if session_id:
            state = self.sessions.get(session_id)
            if state is not None:
                return state
        return self.sessions.start()

    def service(self, session_id: str | None) -> ConversationService:
        """Return the cached service for a session, creating it on first use."""

        state = self.ensure_session(session_id)
        service = self._services.get(state.session_id)
        if service is None:
            service = ConversationService(
                state,
                adapter_factory=self._adapter_factory,
                provider_factory=self._provider_factory,
            )
            self._services[state.session_id] = service
        return service

    # ------------------------------------------------------------------ #
    # Provider connection
    # ------------------------------------------------------------------ #

    def connect(
        self,
        session_id: str | None,
        *,
        provider: str,
        model: str,
        api_key: str,
        endpoint: str = "",
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> ConnectionView:
        """Validate and store a session-local provider configuration.

        A blank key keeps the key already configured for this session, so
        changing the model (or the endpoint) does not require retyping it. The
        key box is always cleared in the response — the value stays in server
        memory and is never rendered back into the browser.
        """

        state = self.ensure_session(session_id)
        secrets = self._secrets(state)
        existing = state.provider
        key = (api_key or "").strip() or existing.api_key
        try:
            config = ProviderConfig(
                provider=(provider or existing.provider).strip() or existing.provider,
                model=(model or "").strip() or existing.model,
                api_key=key,
                base_url=(endpoint or "").strip() or None,
                temperature=float(temperature),
                max_tokens=max_output_tokens,
            )
        except (ProviderConfigurationError, TypeError, ValueError) as exc:
            return ConnectionView(
                status=self._redact(f"⚠️ {exc}", secrets),
                session_id=state.session_id,
                key_value="",
            )

        state.provider = config
        # Only reset an *existing* service. Connecting a provider must not
        # create the BrainOS runtime as a side effect, otherwise a missing
        # installation would break the connect button instead of the chat.
        service = self._services.get(state.session_id)
        if service is not None:
            service.reset_provider()

        if not key:
            return ConnectionView(
                status=(
                    "No API key set — BrainOS will still observe and retrieve "
                    "memory, but no model will be called."
                ),
                connected=False,
                models=(),
                model_value=config.model,
                session_id=state.session_id,
                key_value="",
            )

        try:
            client = self._provider_factory(config)
            client.validate_credentials()
            models = tuple(self._clean_models(client.list_models()))
        except ProviderError as exc:
            return ConnectionView(
                status=self._redact(f"⚠️ {exc}", self._secrets(state)),
                session_id=state.session_id,
                connected=False,
                models=(),
                model_value=config.model,
                key_value="",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, redacted
            return ConnectionView(
                status=self._redact(
                    f"⚠️ {type(exc).__name__}: {exc}", self._secrets(state)
                ),
                session_id=state.session_id,
                connected=False,
                models=(),
                model_value=config.model,
                key_value="",
            )

        if config.model and config.model not in models:
            models = (config.model, *models)
        return ConnectionView(
            status=(
                f"Connected to **{config.provider}** · model `{config.model or '—'}` · "
                f"{len(models)} model(s) available. The key is held in server "
                "memory for this session only."
            ),
            session_id=state.session_id,
            connected=True,
            models=models,
            model_value=config.model,
            key_value="",
            diagnostics=config.safe_dict(),
        )

    def disconnect(self, session_id: str | None) -> ConnectionView:
        """Forget the session key and drop the cached provider client."""

        state = self.ensure_session(session_id)
        service = self._services.get(state.session_id)
        if service is not None:
            service.reset_provider()
        state.clear_credentials()
        return ConnectionView(
            status="API key cleared from server memory. Generation is disabled.",
            session_id=state.session_id,
            connected=False,
            models=(),
            model_value=state.provider.model,
            key_value="",
        )

    def refresh_models(self, session_id: str | None) -> ConnectionView:
        """Re-list models for the configured key without changing it."""

        state = self.ensure_session(session_id)
        if not state.provider.api_key:
            return ConnectionView(
                status="Set an API key to list models.",
                session_id=state.session_id,
                connected=False,
                model_value=state.provider.model,
                key_value="",
            )
        return self.connect(
            state.session_id,
            provider=state.provider.provider,
            model=state.provider.model,
            api_key="",
            endpoint=state.provider.base_url or "",
            temperature=state.provider.temperature,
            max_output_tokens=state.provider.max_tokens,
        )

    # ------------------------------------------------------------------ #
    # Context configuration
    # ------------------------------------------------------------------ #

    def update_context(self, session_id: str | None, **fields: Any) -> str:
        """Apply context-budget and retrieval settings from the sidebar.

        Unknown fields are rejected rather than ignored: a typo in a slider name
        would otherwise leave the user looking at settings that do nothing.
        """

        state = self.ensure_session(session_id)
        known = set(ContextSettings().__dataclass_fields__)
        unknown = sorted(set(fields) - known)
        if unknown:
            return f"⚠️ Unknown context setting(s): {', '.join(unknown)}"
        settings = replace(state.context, **fields)
        try:
            settings.context_budget()
            settings.retrieval_policy()
        except (TypeError, ValueError) as exc:
            return f"⚠️ {exc}"
        state.context = settings
        return (
            f"Context updated · max {settings.max_tokens} tokens · "
            f"history window {settings.recent_turn_budget} · "
            f"memory budget {settings.memory_budget} · "
            f"max {settings.max_memories} memories · mode `{settings.mode}`"
        )

    def context_payload(self, session_id: str | None) -> dict[str, Any]:
        """Return the current context settings for diagnostics and export."""

        state = self.ensure_session(session_id)
        settings = state.context
        payload = {
            "mode": settings.mode,
            "uses_memory": settings.uses_memory(),
            "budget": settings.context_budget().__dict__,
            "policy": settings.retrieval_policy().__dict__,
        }
        return self._redact(payload, self._secrets(state))

    # ------------------------------------------------------------------ #
    # Conversation
    # ------------------------------------------------------------------ #

    def chat(self, session_id: str | None, message: str) -> TurnView:
        """Run one user turn and render every inspection panel."""

        state = self.ensure_session(session_id)
        text = (message or "").strip()
        if not text:
            return self._view(state, notice="Type a message to start.")
        if len(text) > self.limits.max_message_chars:
            return self._view(
                state,
                notice=(
                    f"⚠️ Message is {len(text)} characters; the limit is "
                    f"{self.limits.max_message_chars}. Shorten it or raise the limit."
                ),
            )
        if self._user_turns(state) >= self.limits.max_turns:
            return self._view(
                state,
                notice=(
                    f"⚠️ This session reached {self.limits.max_turns} turns. "
                    "Start a new session to continue."
                ),
            )

        try:
            service = self.service(state.session_id)
        except BrainOSNotConfiguredError:
            return self._view(state, notice=f"⚠️ {_BRAINOS_INSTALL_HINT}")

        try:
            turn = service.handle_user_message(text)
        except BrainOSNotConfiguredError:
            return self._view(state, notice=f"⚠️ {_BRAINOS_INSTALL_HINT}")
        except Exception as exc:  # noqa: BLE001 - a web turn must not crash the app
            return self._view(
                state,
                notice=self._redact(
                    f"⚠️ {type(exc).__name__}: {exc}", self._secrets(state)
                ),
            )

        self._last_turn[state.session_id] = turn
        return self._view(
            state,
            turn=turn,
            error=turn.error or "",
            notice=(
                "The provider request failed; the error below is redacted."
                if turn.error
                else ""
            ),
        )

    def refresh(self, session_id: str | None) -> TurnView:
        """Re-render the panels from the last turn without sending anything."""

        state = self.ensure_session(session_id)
        return self._view(state, turn=self._last_turn.get(state.session_id))

    def clear_conversation(self, session_id: str | None) -> TurnView:
        """Clear the transcript and the session's BrainOS runtime."""

        state = self.ensure_session(session_id)
        service = self._services.get(state.session_id)
        if service is None:
            state.clear_conversation()
        else:
            service.clear_conversation()
        self._last_turn.pop(state.session_id, None)
        return self._view(state, notice="Conversation and memory cleared.")

    def clear_memory(self, session_id: str | None) -> TurnView:
        """Drop BrainOS memory while keeping the visible transcript."""

        state = self.ensure_session(session_id)
        service = self._services.get(state.session_id)
        if service is None:
            state.brain = None
        else:
            service.reset_memory()
        self._last_turn.pop(state.session_id, None)
        return self._view(state, notice="BrainOS memory cleared; transcript kept.")

    def end_session(self, session_id: str | None) -> SessionView:
        """End a session: drop memory, transcript, and the provider key."""

        if not session_id:
            return SessionView(status="No active session.", session_id="")
        self._services.pop(session_id, None)
        self._last_turn.pop(session_id, None)
        ended = self.sessions.end(session_id)
        state = self.sessions.start()
        return SessionView(
            status=(
                "Session ended — key, transcript, and memory dropped."
                if ended
                else "No active session."
            ),
            session_id=state.session_id,
        )

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def export_session(self, session_id: str | None) -> dict[str, Any]:
        """Return a sanitized, JSON-serializable snapshot of a session."""

        state = self.ensure_session(session_id)
        service = self._services.get(state.session_id)
        memories: Sequence[MemoryRecord] = (
            service.stored_memories() if service is not None else []
        )
        payload = {
            "session_id": state.session_id,
            "conversation_id": state.conversation_id,
            "provider": state.provider.safe_dict(),
            "context": self.context_payload(state.session_id),
            "messages": [dict(message) for message in state.messages],
            "memories": [
                {
                    "memory_id": record.memory_id,
                    "text": record.text,
                    "memory_type": record.memory_type,
                    "source_turn": record.source_turn,
                    "observed_at": record.observed_at,
                    "retrieval_count": record.retrieval_count,
                    "confidence": record.confidence,
                    "status": record.status,
                }
                for record in memories
            ],
        }
        return self._redact(payload, self._secrets(state))

    def export_text(self, session_id: str | None) -> str:
        """Return the export snapshot as indented JSON text."""

        return panels.export_payload(self.export_session(session_id))

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _view(
        self,
        state: SessionState,
        *,
        turn: Any | None = None,
        error: str = "",
        notice: str = "",
    ) -> TurnView:
        """Render every panel from session state and the last turn."""

        secrets = self._secrets(state)
        service = self._services.get(state.session_id)
        memories = self._stored_memories_safe(service)

        retrieved: list[MemoryRecord] = []
        ranking: list[Mapping[str, Any]] = []
        stats: Mapping[str, Any] = {}
        report: Mapping[str, Any] = {}
        events: list[Mapping[str, str]] = []
        generated = False
        if turn is not None:
            retrieved = list(turn.retrieved_memories)
            ranking = [dict(item) for item in turn.memory_ranking]
            stats = dict(turn.context_stats)
            report = dict(turn.context_report)
            events = list(turn.trace)
            generated = bool(turn.generated)

        # Redaction happens on the *rendered* values, not on the records:
        # sanitize_value would turn a MemoryRecord into a dict and the panel
        # formatting would then read the wrong shape.
        history = [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            for message in state.messages
            if isinstance(message, Mapping)
        ]
        query = next(
            (
                str(message.get("content", ""))
                for message in reversed(state.messages)
                if isinstance(message, Mapping) and message.get("role") == "user"
            ),
            "",
        )
        status = panels.status_line(
            turn=len(state.messages),
            mode=state.context.mode,
            model=state.provider.model,
            generated=generated and not error,
            stats=stats,
            error=error,
            notice=notice,
            configured=bool(state.provider.api_key),
            has_turn=turn is not None,
        )
        return TurnView(
            session_id=state.session_id,
            history=self._redact(history, secrets),
            status=self._redact(status, secrets),
            stored_rows=self._redact(panels.memory_rows(memories), secrets),
            retrieved_rows=self._redact(
                panels.retrieved_rows(retrieved, ranking), secrets
            ),
            dropped_rows=panels.dropped_rows(self._redact(report, secrets)),
            conflict_rows=panels.conflict_rows(self._redact(report, secrets)),
            stats=self._redact(dict(stats), secrets),
            summary=self._redact(
                panels.context_summary(
                    self._redact(stats, secrets),
                    report=self._redact(report, secrets),
                    mode=state.context.mode,
                    turn=len(state.messages),
                ),
                secrets,
            ),
            prompt=self._redact(
                panels.render_prompt(
                    turn.context_messages if turn is not None else state.last_context
                ),
                secrets,
            ),
            trace=self._redact(
                panels.trace_markdown(
                    query=str(query),
                    stats=self._redact(stats, secrets),
                    report=self._redact(report, secrets),
                    events=self._redact(events, secrets),
                    stored=list(turn.stored_memories)
                    if turn is not None
                    else memories,
                    generated=generated,
                    error=error,
                    model=state.provider.model,
                ),
                secrets,
            ),
            trace_events=self._redact(
                [dict(event) for event in events if isinstance(event, Mapping)], secrets
            ),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stored_memories_safe(service: ConversationService | None) -> list[MemoryRecord]:
        """List stored memories, tolerating an unavailable runtime.

        The memory panel is a best-effort surface: if BrainOS is not installed
        or the runtime has no memory yet, the chat still works and the panel is
        simply empty.
        """

        if service is None:
            return []
        try:
            return service.stored_memories()
        except Exception:  # noqa: BLE001 - panel is best-effort
            return []

    def _secrets(self, state: SessionState) -> tuple[str, ...]:
        key = state.provider.api_key
        return (key,) if key else ()

    def _redact(self, value: Any, secrets: tuple[str, ...]) -> Any:
        return sanitize_value(value, secrets=secrets)

    def _user_turns(self, state: SessionState) -> int:
        return sum(
            1
            for message in state.messages
            if isinstance(message, Mapping) and message.get("role") == "user"
        )

    @staticmethod
    def _clean_models(models: Any) -> list[str]:
        if not isinstance(models, (list, tuple)):
            return []
        cleaned: list[str] = []
        for item in models:
            text = str(getattr(item, "id", item)).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned


__all__ = [
    "DEFAULT_MODEL_PLACEHOLDER",
    "PROVIDER_LABELS",
    "ConnectionView",
    "SessionView",
    "TurnView",
    "UIController",
    "UILimits",
]
