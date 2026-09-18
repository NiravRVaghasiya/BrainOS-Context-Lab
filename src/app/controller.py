"""Gradio-free controller behind the chat UI.

The web layer is deliberately thin. Every behaviour the plan asks the UI to
expose lives here, where it can be tested without a browser or a web
framework:

* session lifecycle (start / clear conversation / clear memory / end),
* provider connection with credential validation and model listing,
* one chat turn through :class:`ConversationService`,
* rendering of the Memory, Context, and Cognitive Trace panels,
* session export.

Phase 5 makes these actions durable: the controller owns optional storage
backends (``create_app`` injects the SQLite ones), forwards them to each
session's service, deletes every persisted row when a session ends, and the
export snapshot includes what the stores hold.

Two constraints shape the module:

1. **No credential leaves the process.** The controller accepts a key from the
   form, stores it on the session's :class:`ProviderConfig`, and returns an
   empty string for the key box. Every value it returns passes through
   :func:`brain.trace.sanitize_value` with the active key as a known secret, so
   a key pasted into a conversation cannot be echoed back by a panel either.
2. **BrainOS stays behind the adapter.** The controller talks to
   :class:`ConversationService` only; it never imports ``brainos_runtime``.

Phase 13 adds a third: **every guard is visible**. The controller renders the
session's security ledger into its own tab, includes it in the export, and
follows each user-data deletion with a database ``VACUUM`` so "delete session"
removes the bytes and not only the rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from baselines.modes import mode_choices, mode_label, resolve_mode
from baselines.rag import HistoryChunk
from brain.adapter import BrainOSAdapter, BrainOSNotConfiguredError, MemoryRecord
from brain.trace import sanitize_value
from providers import ProviderError, create_provider
from providers.base import ProviderConfig, ProviderConfigurationError

from . import panels
from .limits import ChatBudget, ChatLimitExceeded, ChatLimits
from .service import ConversationService, _warn_storage
from .session import SessionManager
from .state import ContextSettings, SessionState

PROVIDER_LABELS: tuple[tuple[str, str], ...] = (
    ("OpenAI", "openai"),
    ("OpenAI-compatible", "openai-compatible"),
)
#: ``(label, value)`` pairs for the baseline-mode selector, in plan order.
MODE_CHOICES: tuple[tuple[str, str], ...] = mode_choices()
DEFAULT_MODEL_PLACEHOLDER = "gpt-4o-mini"

_BRAINOS_INSTALL_HINT = (
    "BrainOS runtime is not installed, so no memory can be stored or retrieved. "
    "Install the pinned upstream revision with "
    "`pip install -e '.[integration]'`."
)


@dataclass(frozen=True)
class UILimits:
    """Guard rails that protect a bring-your-own-key user from runaway usage.

    Phase 15 (plan §15 / §21): the full cost-control surface for the chat path.
    The two fields that existed before Phase 15 (``max_turns``,
    ``max_message_chars``) are retained as the user-facing knobs; the remaining
    ceilings are configured through :class:`ChatLimits`, which this object
    wraps. A :class:`UILimits` is still what a caller constructs, and its
    :meth:`chat_limits` method produces the frozen limits object the budget
    tracks against.

    The split is intentional: ``UILimits`` carries the defaults an operator
    changes at deployment time; ``ChatLimits`` is the immutable per-session
    contract the accounting honours.
    """

    max_turns: int = 200
    max_message_chars: int = 8000
    max_input_tokens: int = 32_000
    max_output_tokens: int = 4_096
    max_session_tokens: int = 500_000
    max_session_requests: int = 500
    request_timeout_seconds: float = 120.0

    def chat_limits(self) -> ChatLimits:
        """Return the immutable limits contract for one session's budget."""

        return ChatLimits(
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            max_turns=self.max_turns,
            max_message_chars=self.max_message_chars,
            max_session_tokens=self.max_session_tokens,
            max_session_requests=self.max_session_requests,
            request_timeout_seconds=self.request_timeout_seconds,
        )


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
    chunk_rows: list[list[Any]] = field(default_factory=list)
    dropped_rows: list[list[Any]] = field(default_factory=list)
    conflict_rows: list[list[Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    prompt: str = ""
    trace: str = ""
    trace_events: list[dict[str, str]] = field(default_factory=list)
    #: Phase 13: the session's guard audit — a markdown summary, the bounded
    #: findings table, and the raw count block for the JSON viewer.
    security: str = ""
    security_rows: list[list[Any]] = field(default_factory=list)
    security_report: dict[str, Any] = field(default_factory=dict)
    #: Phase 15: session cost accounting — the markdown summary, the JSON
    #: snapshot, and the per-turn usage dict for the current turn.
    usage_summary: str = ""
    usage_report: dict[str, Any] = field(default_factory=dict)
    turn_usage: dict[str, Any] = field(default_factory=dict)


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
        conversation_store: Any | None = None,
        memory_store: Any | None = None,
        evaluation_store: Any | None = None,
    ) -> None:
        """Create a controller.

        The three ``*_store`` parameters are the Phase 5 persistence seam
        (structural protocols from :mod:`storage.conversations` and
        :mod:`storage.evaluations`). They default to ``None`` — a purely
        in-memory controller — and ``create_app`` injects the SQLite backends.
        The controller forwards them to each session's
        :class:`ConversationService` and uses them for session-level deletion
        and export; it never lets a storage failure break a UI action.
        """

        self.sessions = sessions or SessionManager()
        self.limits = limits or UILimits()
        self._provider_factory = provider_factory or create_provider
        self._adapter_factory = adapter_factory
        self._conversation_store = conversation_store
        self._memory_store = memory_store
        self._evaluation_store = evaluation_store
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
                conversation_store=self._conversation_store,
                memory_store=self._memory_store,
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
    # Cost controls (Phase 15)
    # ------------------------------------------------------------------ #

    def update_costs(self, session_id: str | None, **fields: Any) -> str:
        """Apply per-request and per-session cost ceilings from the sidebar.

        The controller's :class:`UILimits` is replaced with a new frozen
        object carrying the validated values. Existing session budgets are
        not retroactively rewritten — the new ceilings apply from the next
        turn — because a limit change mid-conversation would confuse the
        visitor about what they are actually allowed to spend.
        """

        known = {
            "max_input_tokens",
            "max_output_tokens",
            "request_timeout_seconds",
            "max_session_tokens",
        }
        unknown = sorted(set(fields) - known)
        if unknown:
            return f"⚠️ Unknown cost setting(s): {', '.join(unknown)}"
        current = self.limits
        updated = {
            "max_turns": current.max_turns,
            "max_message_chars": current.max_message_chars,
            "max_input_tokens": current.max_input_tokens,
            "max_output_tokens": current.max_output_tokens,
            "max_session_tokens": current.max_session_tokens,
            "max_session_requests": current.max_session_requests,
            "request_timeout_seconds": current.request_timeout_seconds,
        }
        for key, value in fields.items():
            if key in ("max_input_tokens", "max_output_tokens", "max_session_tokens"):
                try:
                    updated[key] = int(value)
                except (TypeError, ValueError):
                    return f"⚠️ {key} must be a non-negative integer."
            elif key == "request_timeout_seconds":
                try:
                    updated[key] = float(value)
                except (TypeError, ValueError):
                    return f"⚠️ {key} must be a positive number."
        # Validate the timeout separately since UILimits does not enforce it.
        if updated["request_timeout_seconds"] <= 0:
            return "⚠️ request_timeout_seconds must be greater than zero."
        try:
            new_limits = UILimits(**updated)
            # Validate through ChatLimits which enforces the full contract.
            new_limits.chat_limits()
        except (TypeError, ValueError) as exc:
            return f"⚠️ {exc}"
        self.limits = new_limits
        return (
            f"Cost limits updated · max input {self.limits.max_input_tokens} · "
            f"max output {self.limits.max_output_tokens} · "
            f"timeout {self.limits.request_timeout_seconds:g}s · "
            f"session budget {self.limits.max_session_tokens} tokens"
        )

    # ------------------------------------------------------------------ #
    # Context configuration
    # ------------------------------------------------------------------ #

    def update_context(self, session_id: str | None, **fields: Any) -> str:
        """Apply context-budget and retrieval settings from the sidebar.

        Unknown fields are rejected rather than ignored: a typo in a slider name
        would otherwise leave the user looking at settings that do nothing.

        A mode change (Phase 6) rewrites the budgets that mode defines, and the
        mode's values win over any budget submitted in the same call. That
        ordering is what keeps the baseline comparison honest: the sidebar
        submits every slider together with the mode, so letting the sliders win
        would silently carry the previous mode's history window into the new one
        — and Phase 3 measured that two modes sharing a window larger than the
        conversation are indistinguishable.
        """

        state = self.ensure_session(session_id)
        known = set(ContextSettings().__dataclass_fields__)
        unknown = sorted(set(fields) - known)
        if unknown:
            return f"⚠️ Unknown context setting(s): {', '.join(unknown)}"
        settings = replace(state.context, **fields)
        requested = fields.get("mode")
        if requested is not None:
            try:
                resolved = resolve_mode(requested)
            except ValueError as exc:
                return f"⚠️ {exc}"
            if resolved != state.context.mode:
                settings = settings.with_mode_defaults(resolved)
        try:
            settings.mode_profile()
            settings.context_budget()
            settings.retrieval_policy()
        except (TypeError, ValueError) as exc:
            return f"⚠️ {exc}"
        state.context = settings
        profile = settings.mode_profile()
        window = (
            f"last {settings.max_recent_turns} turns"
            if settings.max_recent_turns is not None
            else "whole conversation"
        )
        return (
            f"Context updated · {profile.label} · max {settings.max_tokens} tokens · "
            f"history {window} / {settings.recent_turn_budget} tokens · "
            f"memory budget {settings.memory_budget} · "
            f"chunk budget {settings.chunk_budget} · "
            f"max {settings.max_memories} memories"
        )

    def apply_mode(self, session_id: str | None, mode: Any) -> tuple[str, ContextSettings]:
        """Switch baseline mode and return the settings that mode applied.

        The UI calls this from the mode selector so the sidebar can show the
        budgets that will actually run. Returning the resolved settings (rather
        than just a status string) is what makes the two impossible to desync.
        """

        state = self.ensure_session(session_id)
        status = self.update_context(state.session_id, mode=mode)
        return status, state.context

    def context_payload(self, session_id: str | None) -> dict[str, Any]:
        """Return the current context settings for diagnostics and export."""

        state = self.ensure_session(session_id)
        settings = state.context
        payload = {
            "mode": settings.mode,
            "mode_label": settings.mode_profile().label,
            "mode_profile": settings.mode_profile().to_dict(),
            "uses_memory": settings.uses_memory(),
            "uses_rag": settings.uses_rag(),
            "budget": settings.context_budget().__dict__,
            "policy": settings.retrieval_policy().__dict__,
            "rag_top_k": settings.rag_top_k,
        }
        return self._redact(payload, self._secrets(state))

    # ------------------------------------------------------------------ #
    # Conversation
    # ------------------------------------------------------------------ #

    def chat(self, session_id: str | None, message: str) -> TurnView:
        """Run one user turn and render every inspection panel.

        Phase 15: the session's :class:`ChatBudget` is checked before the
        provider is called and charged after. A ceiling hit renders a
        user-visible refusal — the turn is not sent, the budget is not
        charged, and the visitor is told how to continue (clear the
        conversation or start a new session).
        """

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
        budget = self._ensure_budget(state)
        try:
            budget.check_turn()
        except ChatLimitExceeded as exc:
            budget.record_refusal(exc.limit)
            return self._view(
                state,
                notice=f"⚠️ {exc}",
            )

        try:
            service = self.service(state.session_id)
        except BrainOSNotConfiguredError:
            return self._view(state, notice=f"⚠️ {_BRAINOS_INSTALL_HINT}")

        # Count the turn only after the service is successfully created —
        # a failed service creation must not consume a turn from the budget.
        budget.record_turn()

        try:
            turn = service.handle_user_message(
                text,
                request_timeout=self.limits.request_timeout_seconds,
            )
        except BrainOSNotConfiguredError:
            return self._view(state, notice=f"⚠️ {_BRAINOS_INSTALL_HINT}")
        except Exception as exc:  # noqa: BLE001 - a web turn must not crash the app
            return self._view(
                state,
                notice=self._redact(
                    f"⚠️ {type(exc).__name__}: {exc}", self._secrets(state)
                ),
            )

        # Phase 15: charge the budget from the turn's reported usage.
        budget.charge(
            prompt_tokens=turn.usage.get("prompt_tokens"),
            completion_text=turn.reply or "",
            usage={
                "prompt_tokens": turn.usage.get("prompt_tokens"),
                "completion_tokens": turn.usage.get("completion_tokens"),
            },
            failed=bool(turn.error) or bool(turn.usage.get("failed")),
            timed_out=bool(turn.usage.get("timed_out")),
        )

        # Check per-request input ceiling post-hoc for the prompt we built
        # (the pre-check above is on turn/token/session, not on prompt size).
        prompt_tokens = turn.usage.get("prompt_tokens", 0)
        if (
            self.limits.max_input_tokens
            and prompt_tokens > self.limits.max_input_tokens
        ):
            budget.record_refusal("max_input_tokens")

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
        self._vacuum_stores()
        self._last_turn.pop(state.session_id, None)
        return self._view(
            state,
            notice=(
                "Conversation and memory cleared — persisted rows deleted too."
            ),
        )

    def clear_memory(self, session_id: str | None) -> TurnView:
        """Drop BrainOS memory while keeping the visible transcript."""

        state = self.ensure_session(session_id)
        service = self._services.get(state.session_id)
        if service is None:
            state.brain = None
        else:
            service.reset_memory()
        self._vacuum_stores()
        self._last_turn.pop(state.session_id, None)
        return self._view(
            state,
            notice=(
                "BrainOS memory cleared (persisted mirror too); transcript kept."
            ),
        )

    def end_session(self, session_id: str | None) -> SessionView:
        """End a session: drop memory, transcript, and the provider key."""

        if not session_id:
            return SessionView(status="No active session.", session_id="")
        self._services.pop(session_id, None)
        self._last_turn.pop(session_id, None)
        self._delete_persisted_session(session_id)
        self._vacuum_stores()
        ended = self.sessions.end(session_id)
        state = self.sessions.start()
        return SessionView(
            status=(
                "Session ended — key, transcript, memory, and all persisted "
                "session rows dropped."
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
            "persistence": self._persistence_payload(state.session_id),
            "security": self._security_report_safe(service, state),
            "usage": self.usage_payload(state.session_id),
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
        security = self._security_report_safe(service, state)

        retrieved: list[MemoryRecord] = []
        chunks: list[HistoryChunk] = []
        ranking: list[Mapping[str, Any]] = []
        stats: Mapping[str, Any] = {}
        report: Mapping[str, Any] = {}
        events: list[Mapping[str, str]] = []
        generated = False
        if turn is not None:
            retrieved = list(turn.retrieved_memories)
            chunks = list(turn.retrieved_chunks)
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
            chunk_rows=self._redact(panels.chunk_rows(chunks), secrets),
            dropped_rows=panels.dropped_rows(self._redact(report, secrets)),
            conflict_rows=panels.conflict_rows(self._redact(report, secrets)),
            stats=self._redact(dict(stats), secrets),
            summary=self._redact(
                panels.context_summary(
                    self._redact(stats, secrets),
                    report=self._redact(report, secrets),
                    mode=mode_label(state.context.mode),
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
            security=self._redact(panels.security_markdown(security), secrets),
            security_rows=self._redact(panels.security_rows(security), secrets),
            security_report=security,
            usage_summary=self._redact(
                panels.usage_summary(self.usage_payload(state.session_id)),
                secrets,
            ),
            usage_report=self.usage_payload(state.session_id),
            turn_usage=dict(turn.usage) if turn is not None else {},
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

    def _persistence_payload(self, session_id: str) -> dict[str, Any]:
        """The persisted view of a session for the export snapshot.

        ``messages`` (top level) is the live transcript; ``persistence`` shows
        what the durable stores hold for this session — every conversation
        with rows still in the database plus the memory mirror. Conversations
        cleared through the UI are absent because clearing deletes their rows.
        Reads are best-effort: a failing backend downgrades the export instead
        of failing it.
        """

        payload: dict[str, Any] = {
            "enabled": self._conversation_store is not None
            or self._memory_store is not None,
            "conversations": [],
            "memories": [],
        }
        if self._conversation_store is not None:
            try:
                for conversation_id in self._conversation_store.list_conversations(
                    session_id
                ):
                    messages = self._conversation_store.list_messages(
                        session_id, conversation_id
                    )
                    payload["conversations"].append(
                        {
                            "conversation_id": conversation_id,
                            "messages": [
                                {
                                    "role": message.role,
                                    "content": message.content,
                                    "created_at": message.created_at,
                                }
                                for message in messages
                            ],
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - export is best-effort
                _warn_storage("export transcript read failed", exc)
        if self._memory_store is not None:
            try:
                payload["memories"] = self._memory_store.list_memories(session_id)
            except Exception as exc:  # noqa: BLE001 - export is best-effort
                _warn_storage("export memory read failed", exc)
        return payload

    def _delete_persisted_session(self, session_id: str) -> None:
        """Forget everything a session persisted (plan §19 user-data deletion).

        Best-effort per backend: a failing store logs the exception type and
        the remaining stores are still cleared, so one broken table cannot
        retain rows the user asked to delete.
        """

        if self._conversation_store is not None:
            try:
                self._conversation_store.delete_session(session_id)
            except Exception as exc:  # noqa: BLE001 - best-effort deletion
                _warn_storage("conversation deletion failed", exc)
        if self._memory_store is not None:
            try:
                self._memory_store.clear(session_id)
            except Exception as exc:  # noqa: BLE001 - best-effort deletion
                _warn_storage("memory deletion failed", exc)
        if self._evaluation_store is not None:
            try:
                self._evaluation_store.delete_session_data(session_id)
            except Exception as exc:  # noqa: BLE001 - best-effort deletion
                _warn_storage("evaluation deletion failed", exc)

    def _security_report_safe(self, service: Any, state: SessionState) -> dict[str, Any]:
        """The session's guard audit, or an empty block when there is no service.

        Best-effort like every other panel: a report that cannot be built must
        not break the turn that asked for it.
        """

        if service is None:
            return {
                "security_version": "security-v1",
                "record_count": 0,
                "clean": True,
                "totals": {},
                "by_action": {},
                "by_route": {},
                "by_stage": {},
                "by_family": {},
                "recent": [],
                "last_prompt": dict(getattr(state, "last_guard", {}) or {}),
            }
        try:
            return service.security_report()
        except Exception:  # noqa: BLE001 - the panel is best-effort
            return {}

    def _vacuum_stores(self) -> None:
        """Rewrite the database files after a user-data deletion.

        ``secure_delete`` already zeroes the removed content; ``VACUUM`` also
        reclaims the pages, so the file shrinks and no freed page is left
        holding a deleted transcript. Skipped silently for stores that do not
        implement it (the in-memory test doubles), and never fatal: a vacuum
        failure must not turn a successful delete into an error.
        """

        for store in (
            self._conversation_store,
            self._memory_store,
            self._evaluation_store,
        ):
            vacuum = getattr(store, "vacuum", None)
            if callable(vacuum):
                try:
                    vacuum()
                except Exception as exc:  # noqa: BLE001 - best-effort hardening
                    _warn_storage("vacuum failed", exc)

    def _secrets(self, state: SessionState) -> tuple[str, ...]:
        key = state.provider.api_key
        return (key,) if key else ()

    def _ensure_budget(self, state: SessionState) -> ChatBudget:
        """Return the session's cost budget, creating it on first use.

        Phase 15: the budget is created lazily from the controller's
        :class:`UILimits` so a session that never sends a turn costs nothing
        to track. The budget is stored on the session state so
        ``clear_conversation`` can reset it alongside the transcript.
        """

        budget = state.usage
        if isinstance(budget, ChatBudget):
            return budget
        budget = ChatBudget(limits=self.limits.chat_limits())
        state.usage = budget
        return budget

    def usage_payload(self, session_id: str | None) -> dict[str, Any]:
        """Return the current session's cost accounting for diagnostics and export."""

        state = self.ensure_session(session_id)
        budget = state.usage
        if isinstance(budget, ChatBudget):
            return budget.snapshot()
        # No turn has been sent yet — report the limits and zero usage.
        return ChatBudget(limits=self.limits.chat_limits()).snapshot()

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
    "MODE_CHOICES",
    "PROVIDER_LABELS",
    "ChatLimits",
    "ConnectionView",
    "SessionView",
    "TurnView",
    "UIController",
    "UILimits",
]
