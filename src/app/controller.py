"""Session-scoped controller between the UI callbacks and the service layer.

Phase 4 wires the Gradio interface to :class:`app.service.ConversationService`.
Callbacks stay thin: this controller owns everything a callback would otherwise
improvise — lazy service creation, settings application, view construction, and
credential discipline. It never imports Gradio, so the whole UI behaviour is
unit-testable headlessly.

Rules carried forward from the Phase 3 hand-off contract:

* Context is configured through :class:`~app.state.ContextSettings` only, so
  the token budget and the retrieval policy can never desynchronize.
* Panels render the sanitized values the service already produces
  (``inspect()``, ``context_stats``, ``context_report``, ``memory_ranking``,
  ``trace``); nothing is re-serialized by hand.
* The API key travels browser → server only. No view produced here ever
  contains it, and runtime-derived strings are additionally scrubbed against
  the active key before they are returned.
* BrainOS stays behind ``BrainMemoryAdapter``; this module never imports
  ``brainos_runtime``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from brain.adapter import BrainOSNotConfiguredError
from brain.trace import sanitize_value
from providers import ProviderConfigurationError, ProviderError, create_provider
from providers.base import safe_error_message

from .service import ConversationService, ConversationTurn
from .session import SessionManager
from .state import ContextSettings, SessionState

BRAINOS_INSTALL_HINT = (
    "BrainOS runtime is not installed on this server. Cognitive memory needs "
    "the integration extra: `pip install -e '.[integration]'`."
)

#: Baseline modes planned for Phase 6. Selecting one today keeps the BrainOS
#: pipeline active and says so in the UI rather than silently mislabeling runs.
BASELINE_MODES_PENDING = ("full_context", "sliding_window", "rag")

#: Early guard against runaway sessions; full cost controls arrive in Phase 15.
DEFAULT_MAX_TURNS = 400

PROVIDER_CHOICES = ("OpenAI", "OpenAI-compatible")
MEMORY_MODE_CHOICES = ("brainos", *BASELINE_MODES_PENDING)

_TABLE_TEXT_LIMIT = 240

_CONTEXT_FIELDS = frozenset(
    field.name for field in dataclasses.fields(ContextSettings) if field.name != "mode"
)
_INT_CONTEXT_FIELDS = frozenset(
    {
        "max_tokens",
        "recent_turn_budget",
        "memory_budget",
        "system_budget",
        "max_recent_turns",
        "per_message_overhead",
        "max_memories",
    }
)


def _provider_name(label: Any) -> str:
    """Normalize a UI provider label onto the factory's provider name."""

    return str(label or "").strip().lower().replace(" ", "-")


def _optional_int(value: Any) -> int | None:
    """Coerce an optional numeric UI value; zero and empty mean 'unset'."""

    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _clip(text: str, limit: int = _TABLE_TEXT_LIMIT) -> str:
    """Bound table cell length so panels stay readable."""

    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _score(value: Any) -> Any:
    """Render an optional float for a table cell."""

    if value is None:
        return ""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return ""


def _render_prompt(messages: Any) -> str:
    """Render a message list exactly as ``BuiltContext.final_prompt()`` does."""

    blocks = [
        f"{str(message.get('role', '')).upper()}\n{message.get('content', '')}"
        for message in messages
        if isinstance(message, dict)
    ]
    return "\n\n".join(blocks)


def _empty_turn_view() -> dict[str, Any]:
    return {
        "retrieved_rows": [],
        "ranking_json": [],
        "context_summary": "",
        "context_stats": {},
        "final_prompt": "",
        "trace_rows": [],
        "decision_md": "",
        "dropped_rows": [],
        "conflict_rows": [],
    }


class ChatController:
    """One browser session: state, lazily created service, and view models.

    The controller is stored in ``gr.State``. Every public method returns plain
    data (strings, row lists, dicts) that the UI maps onto components; no
    method returns the API key or any value derived from it unredacted.
    """

    def __init__(
        self,
        state: SessionState | None = None,
        *,
        manager: SessionManager | None = None,
        service_factory: Callable[[SessionState], ConversationService] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        conversation_store: Any | None = None,
        memory_store: Any | None = None,
        evaluation_store: Any | None = None,
    ) -> None:
        self.manager = manager
        if state is not None:
            self.state = state
        elif manager is not None:
            self.state = manager.start()
        else:
            self.state = SessionState()
        self._service_factory = service_factory or (
            lambda session_state: ConversationService(session_state)
        )
        self._service: ConversationService | None = None
        # Phase 5 persistence. Stores are controller-owned and attached to the
        # service at creation time. The controller knows only the storage
        # *protocols*; choosing the SQLite backend is deployment wiring
        # (``app.ui.create_app``), so a bare controller — scripts, unit tests —
        # stays purely in-memory and never touches disk.
        self._conversation_store = conversation_store
        self._memory_store = memory_store
        self._evaluation_store = evaluation_store
        self._turns_used = 0
        self.max_turns = max_turns
        self._connection_status = (
            "Not connected. Enter an API key and model, then validate the connection."
        )
        self._model_choices: list[str] = []
        self._mode_notice = ""
        self._last_status = (
            "Session ready. Configure a provider to generate replies; "
            "BrainOS observes and retrieves either way."
        )
        self._turn_view: dict[str, Any] = _empty_turn_view()

    # ------------------------------------------------------------------ #
    # Service lifecycle
    # ------------------------------------------------------------------ #

    @property
    def service(self) -> ConversationService:
        """Return the session's service, creating it on first use.

        Creation imports the BrainOS runtime and may raise
        :class:`BrainOSNotConfiguredError`; browser-facing callers convert that
        into the install hint instead of a traceback. The controller's stores
        are attached after creation when the factory did not supply them, so
        persistence works through any service seam.
        """

        if self._service is None:
            self._service = self._service_factory(self.state)
            if self._conversation_store is not None and getattr(
                self._service, "conversation_store", None
            ) is None:
                self._service.conversation_store = self._conversation_store
            if self._memory_store is not None and getattr(
                self._service, "memory_store", None
            ) is None:
                self._service.memory_store = self._memory_store
        return self._service

    @property
    def conversation_store(self) -> Any:
        """The injected transcript store, or ``None`` for in-memory sessions."""

        return self._conversation_store

    @property
    def memory_store(self) -> Any:
        """The injected memory mirror, or ``None`` for in-memory sessions."""

        return self._memory_store

    @property
    def evaluation_store(self) -> Any:
        """The injected evaluation-run store, or ``None``."""

        return self._evaluation_store

    def _secrets(self) -> tuple[str, ...]:
        key = self.state.provider.api_key
        return (key,) if key else ()

    def _sanitize(self, value: Any) -> Any:
        return sanitize_value(value, secrets=self._secrets())

    def _active_provider(self) -> Any:
        """Return the session's provider, preferring the service's seam.

        Falling back to the factory directly when the BrainOS runtime is
        unavailable keeps provider credentials validatable even on a server
        without the integration extra installed.
        """

        try:
            return self.service.provider()
        except BrainOSNotConfiguredError:
            return create_provider(self.state.provider)

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #

    def apply_provider(
        self,
        *,
        provider: Any = None,
        model: Any = None,
        api_key: Any = None,
        base_url: Any = None,
        temperature: Any = None,
        max_output_tokens: Any = None,
    ) -> str:
        """Store sidebar provider settings on the session state.

        An empty ``api_key`` keeps the credential already held in server
        memory, so the browser never has to echo the key back. Any real change
        drops the cached service (and with it the cached provider client); the
        session's BrainOS runtime lives on ``state.brain`` and survives.
        """

        current = self.state.provider
        name = _provider_name(current.provider if provider is None else provider)
        key = current.api_key
        if api_key is not None and str(api_key).strip():
            key = str(api_key).strip()
        url: str | None = current.base_url if base_url is None else str(base_url).strip()
        if url is not None and not url:
            url = None
        try:
            temperature_value = (
                current.temperature if temperature is None else float(temperature)
            )
            model_text = (current.model if model is None else str(model)).strip()
            output_tokens = (
                current.max_tokens
                if max_output_tokens is None
                else _optional_int(max_output_tokens)
            )
            candidate = replace(
                current,
                provider=name,
                model=model_text,
                api_key=key,
                base_url=url,
                temperature=temperature_value,
                max_tokens=output_tokens,
            )
        except (ProviderConfigurationError, TypeError, ValueError) as exc:
            return f"Provider settings not applied: {exc}"
        if candidate == current:
            return "Provider settings unchanged."
        self.state.provider = candidate
        self._service = None
        return "Provider settings applied to this session."

    def apply_context_settings(self, **changes: Any) -> str:
        """Validate and store context knobs through ``ContextSettings`` only.

        The candidate settings must build both a ``ContextBudget`` and a
        ``RetrievalPolicy`` before they are accepted, so an invalid slider
        value can never reach context construction mid-turn.
        """

        updates: dict[str, Any] = {}
        for key, value in changes.items():
            if key not in _CONTEXT_FIELDS or value is None:
                continue
            if key in _INT_CONTEXT_FIELDS:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    return f"Context settings not applied: {key} must be a whole number."
            updates[key] = value
        if not updates:
            return "Context settings unchanged."
        candidate = replace(self.state.context, **updates)
        if candidate == self.state.context:
            return "Context settings unchanged."
        try:
            candidate.context_budget()
            candidate.retrieval_policy()
        except Exception as exc:
            return f"Context settings not applied: {exc}"
        self.state.context = candidate
        return "Context settings applied."

    def set_memory_mode(self, mode: Any) -> str:
        """Record the baseline-mode selection and return its UI notice."""

        name = str(mode or "brainos").strip().lower() or "brainos"
        self.state.context.mode = name
        if name in BASELINE_MODES_PENDING:
            self._mode_notice = (
                f"Baseline mode `{name}` arrives with Phase 6; until then every "
                "turn runs the BrainOS pipeline. The selection is recorded so "
                "runs are never silently mislabeled."
            )
        else:
            self._mode_notice = ""
        return self._mode_notice

    # ------------------------------------------------------------------ #
    # Provider actions
    # ------------------------------------------------------------------ #

    def validate_connection(self) -> str:
        """Check the active credentials without ever echoing them back."""

        config = self.state.provider
        if not config.api_key or not config.model:
            self._connection_status = (
                "Connection not validated: an API key and a model identifier are required."
            )
            return self._connection_status
        try:
            provider = self._active_provider()
            provider.validate_credentials()
        except (ProviderError, BrainOSNotConfiguredError) as exc:
            self._connection_status = (
                f"Connection failed: {safe_error_message(exc, secrets=self._secrets())}"
            )
        except Exception as exc:  # never send a traceback to the browser
            self._connection_status = (
                f"Connection failed: {safe_error_message(exc, secrets=self._secrets())}"
            )
        else:
            self._connection_status = (
                f"Connected: credentials accepted for model `{config.model}` "
                f"on `{config.provider}`."
            )
        return self._connection_status

    def list_models(self) -> str:
        """Fetch model identifiers for the sidebar dropdown."""

        config = self.state.provider
        if not config.api_key:
            self._connection_status = "Cannot list models without an API key."
            return self._connection_status
        try:
            provider = self._active_provider()
            models = provider.list_models()
        except (ProviderError, BrainOSNotConfiguredError) as exc:
            self._connection_status = (
                f"Model listing failed: {safe_error_message(exc, secrets=self._secrets())}"
            )
        except Exception as exc:
            self._connection_status = (
                f"Model listing failed: {safe_error_message(exc, secrets=self._secrets())}"
            )
        else:
            self._model_choices = [str(model).strip() for model in models if str(model).strip()]
            self._connection_status = (
                f"{len(self._model_choices)} models available for `{config.provider}`."
            )
        return self._connection_status

    # ------------------------------------------------------------------ #
    # Conversation
    # ------------------------------------------------------------------ #

    def send_message(self, text: str) -> dict[str, Any]:
        """Run one turn through the service and return the full panel view."""

        message = str(text or "").strip()
        if not message:
            return self.views(status="Type a message to send.")
        if self._turns_used >= self.max_turns:
            return self.views(
                status=(
                    f"Session turn limit ({self.max_turns}) reached. End the session "
                    "to start a fresh conversation."
                )
            )
        try:
            turn = self.service.handle_user_message(message)
        except BrainOSNotConfiguredError:
            return self.views(status=BRAINOS_INSTALL_HINT)
        except ProviderError as exc:
            return self.views(
                status=(
                    "Provider configuration error: "
                    f"{safe_error_message(exc, secrets=self._secrets())}"
                )
            )
        except Exception as exc:  # never send a traceback to the browser
            return self.views(
                status=f"Turn failed: {safe_error_message(exc, secrets=self._secrets())}"
            )
        self._turns_used += 1
        self._turn_view = self._turn_view_from(turn)
        return self.views(status=self._compose_status(turn))

    def clear_conversation(self) -> dict[str, Any]:
        """Clear the transcript and drop the session's BrainOS memory."""

        try:
            self.service.clear_conversation()
        except Exception:
            # Without a usable runtime the local transcript is still cleared.
            self.state.clear_conversation()
        self._service = None
        self._turns_used = 0
        self._turn_view = _empty_turn_view()
        return self.views(
            status=(
                "Conversation cleared and BrainOS memory dropped for this session. "
                "Provider settings were kept; the API key stays in server memory "
                "until the session ends."
            )
        )

    def clear_memory(self) -> dict[str, Any]:
        """Drop BrainOS memory while keeping the transcript (plan §19).

        Distinct from :meth:`clear_conversation`: the conversation continues,
        but the runtime — and its persisted mirror — start empty. The next
        turn lazily creates a fresh runtime through the usual seam.
        """

        self.state.brain = None
        self.state.diagnostics.clear()
        self.state.last_context.clear()
        self._service = None
        self._turn_view = _empty_turn_view()
        if self._memory_store is not None:
            try:
                self._memory_store.clear(self.state.session_id)
            except Exception:
                pass
        return self.views(
            status=(
                "BrainOS memory cleared for this session; the transcript was "
                "kept. Facts observed from here on are stored in a fresh "
                "runtime."
            )
        )

    def end_session(self) -> tuple[ChatController, dict[str, Any]]:
        """Clear credentials from server memory and start a fresh session.

        *Delete session* in the plan's data-control list: everything persisted
        for the session — transcript rows, memory mirror, evaluation runs — is
        removed before the session itself is dropped. Deletion never creates a
        store: a session that persisted nothing has nothing to delete.
        """

        self._delete_persisted_session(self.state.session_id)
        if self.manager is not None:
            self.manager.end(self.state.session_id)
        else:
            self.state.clear_credentials()
            self.state.clear_conversation()
        self._service = None
        fresh = ChatController(
            manager=self.manager,
            service_factory=self._service_factory,
            max_turns=self.max_turns,
            conversation_store=self._conversation_store,
            memory_store=self._memory_store,
            evaluation_store=self._evaluation_store,
        )
        return fresh, fresh.views(
            status=(
                "Session ended: credentials cleared from server memory, the "
                "BrainOS runtime dropped, and all persisted session data "
                "deleted. A fresh session is ready."
            )
        )

    def _delete_persisted_session(self, session_id: str) -> None:
        if self._conversation_store is not None:
            try:
                self._conversation_store.delete_session(session_id)
            except Exception:
                pass
        if self._memory_store is not None:
            try:
                self._memory_store.clear(session_id)
            except Exception:
                pass
        if self._evaluation_store is not None:
            try:
                self._evaluation_store.delete_session_data(session_id)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def export_session(self) -> dict[str, Any]:
        """Return the session export: config, transcript, memories, last turn.

        Reads the persisted mirrors when they exist and falls back to process
        state otherwise, so an export is complete even for a session that was
        never persisted. The payload passes the same sanitizer as every other
        browser-visible value.
        """

        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "application": "BrainOS Context Lab",
            "session_id": self.state.session_id,
            "conversation_id": self.state.conversation_id,
            "provider": self.state.provider.safe_dict(),
            "context_settings": dataclasses.asdict(self.state.context),
            "turns_used": self._turns_used,
            "max_turns": self.max_turns,
            "transcript": self._export_transcript(),
            "memories": self._export_memories(),
            "last_turn": {
                "context_stats": self._turn_view.get("context_stats", {}),
                "ranking": self._turn_view.get("ranking_json", []),
                "dropped": self._turn_view.get("dropped_rows", []),
                "conflicts": self._turn_view.get("conflict_rows", []),
            },
        }
        return self._sanitize(payload)

    def export_session_file(self) -> tuple[str, str]:
        """Write :meth:`export_session` to a temporary JSON file for download."""

        payload = self.export_session()
        handle, path = tempfile.mkstemp(prefix="brainos-session-", suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
        summary = (
            f"Session exported: {len(payload['transcript'])} messages, "
            f"{len(payload['memories'])} memories."
        )
        return path, summary

    def _export_transcript(self) -> list[dict[str, Any]]:
        store = self._conversation_store
        if store is not None:
            try:
                rows: list[dict[str, Any]] = []
                for conversation_id in store.list_conversations(self.state.session_id):
                    for message in store.list_messages(self.state.session_id, conversation_id):
                        rows.append(
                            {
                                "conversation_id": conversation_id,
                                "role": message.role,
                                "content": message.content,
                                "created_at": message.created_at,
                            }
                        )
                return rows
            except Exception:
                pass
        return [dict(message) for message in self.state.messages]

    def _export_memories(self) -> list[dict[str, Any]]:
        store = self._memory_store
        if store is not None:
            try:
                return list(store.list_memories(self.state.session_id))
            except Exception:
                pass
        if self._service is not None or self.state.brain is not None:
            try:
                return list(self.service.inspect()["memories"])
            except Exception:
                return []
        return []

    # ------------------------------------------------------------------ #
    # View models
    # ------------------------------------------------------------------ #

    def views(self, *, status: str | None = None) -> dict[str, Any]:
        """Return every browser-visible value for the current session state."""

        if status is not None:
            self._last_status = status
        stored_rows = self._stored_rows()
        provider = self.state.provider
        view: dict[str, Any] = {
            "history": [dict(message) for message in self.state.messages],
            "status": self._last_status,
            "stored_rows": stored_rows,
            "stored_count": len(stored_rows),
            "provider_summary": (
                f"{provider.provider} · {provider.model or 'no model'} · "
                f"key {'set' if provider.api_key else 'not set'}"
            ),
            "connection_status": self._connection_status,
            "model_choices": list(self._model_choices),
            "model_value": provider.model,
            "mode": self.state.context.mode,
            "mode_notice": self._mode_notice,
            "turns_used": self._turns_used,
            "max_turns": self.max_turns,
        }
        view.update(self._turn_view)
        return view

    def _stored_rows(self) -> list[list[Any]]:
        """Render stored memories from the service's sanitized inspection."""

        if self._service is None and self.state.brain is None:
            # Nothing can have been stored yet; do not create a runtime just
            # to render an empty table (page load must stay BrainOS-optional).
            return []
        try:
            memories = self.service.inspect()["memories"]
        except Exception:
            return []
        rows: list[list[Any]] = []
        for record in memories:
            observed = str(record.get("observed_at") or record.get("created_at") or "")
            source_turn = record.get("source_turn")
            retrieval_count = record.get("retrieval_count")
            rows.append(
                [
                    str(record.get("memory_type", "")),
                    _clip(str(record.get("text", ""))),
                    observed[:19].replace("T", " "),
                    int(retrieval_count) if isinstance(retrieval_count, int) else 0,
                    "" if source_turn is None else int(source_turn),
                    str(record.get("status") or "active"),
                ]
            )
        return rows

    def _turn_view_from(self, turn: ConversationTurn) -> dict[str, Any]:
        text_by_id = {record.memory_id: record.text for record in turn.retrieved_memories}
        retrieved_rows: list[list[Any]] = []
        for item in turn.memory_ranking:
            flags = [name for name in ("contested", "suspicious") if item.get(name)]
            memory_text = text_by_id.get(str(item.get("memory_id", "")), "")
            retrieved_rows.append(
                [
                    item.get("rank", ""),
                    _score(item.get("score")),
                    _score(item.get("relevance")),
                    str(item.get("memory_type", "")),
                    _clip(str(self._sanitize(memory_text))),
                    ", ".join(flags),
                ]
            )
        report = turn.context_report or {}
        dropped_rows = [
            [
                str(item.get("reason", "")),
                _clip(str(self._sanitize(item.get("text", "")))),
                str(self._sanitize(item.get("detail", ""))),
                _score(item.get("score")),
            ]
            for item in report.get("dropped", [])
            if isinstance(item, dict)
        ]
        conflict_rows = [
            [
                str(self._sanitize(item.get("subject", ""))),
                str(item.get("kept_memory_id", "")),
                str(item.get("dropped_memory_id", "")) or "(contested — both kept)",
                str(item.get("source", "")),
                str(self._sanitize(item.get("reason", ""))),
            ]
            for item in report.get("conflicts", [])
            if isinstance(item, dict)
        ]
        trace_rows = [
            [
                str(event.get("name", "")),
                str(event.get("detail", "")),
                str(event.get("timestamp", ""))[:19].replace("T", " "),
            ]
            for event in turn.trace
        ]
        return {
            "retrieved_rows": retrieved_rows,
            "ranking_json": [dict(item) for item in turn.memory_ranking],
            "context_summary": self._context_summary(turn.context_stats),
            "context_stats": dict(turn.context_stats),
            "final_prompt": _render_prompt(turn.context_messages),
            "trace_rows": trace_rows,
            "decision_md": self._decision_markdown(turn),
            "dropped_rows": dropped_rows,
            "conflict_rows": conflict_rows,
        }

    def _context_summary(self, stats: dict[str, Any]) -> str:
        """Markdown digest of the plan's required accounting fields."""

        if not stats:
            return ""

        def number(name: str) -> Any:
            return stats.get(name, 0)

        lines = [
            (
                f"**Final context:** {number('final_context_tokens')} tokens "
                f"(budget {number('max_tokens')}, "
                f"{100.0 * float(stats.get('budget_utilization', 0.0)):.1f}% used, "
                f"counter `{stats.get('token_counter', 'estimate_tokens')}`)"
            ),
            (
                f"**Raw history:** {number('raw_history_tokens')} tokens → recent window "
                f"{number('recent_history_tokens')} tokens "
                f"({number('history_messages_selected')}/{number('history_messages_considered')} "
                "messages kept)"
            ),
            (
                f"**Memory:** {number('retrieved_memory_tokens')} tokens across "
                f"{number('selected_memory_count')} selected of "
                f"{number('candidate_memory_count')} candidates"
            ),
            (
                f"**System:** {number('system_tokens')} tokens"
                f"{' (truncated)' if stats.get('system_truncated') else ''} · "
                f"**current message:** {number('current_message_tokens')} tokens"
            ),
        ]
        reference = int(number("full_context_reference_tokens") or 0)
        if reference > 0:
            reduction = 100.0 * float(stats.get("context_reduction_vs_full_context", 0.0))
            lines.append(
                f"**Full-context reference:** {reference} tokens → saved "
                f"{number('token_savings_vs_full_context')} ({reduction:.1f}% reduction)"
            )
        if stats.get("exceeds_max_tokens"):
            lines.append("**Warning:** the final context exceeds the configured ceiling.")
        return "\n\n".join(lines)

    def _decision_markdown(self, turn: ConversationTurn) -> str:
        decision = turn.decision
        if decision is None:
            return "No decision signal was produced for this turn."
        verdict = "sufficient" if decision.sufficient else "insufficient"
        parts = [f"**Decision:** context {verdict}"]
        if decision.action:
            parts.append(f"action `{decision.action}`")
        if decision.confidence is not None:
            parts.append(f"confidence {float(decision.confidence):.2f}")
        text = " · ".join(parts)
        if decision.reason:
            text += f"\n\n{decision.reason}"
        return str(self._sanitize(text))

    def _compose_status(self, turn: ConversationTurn) -> str:
        stats = turn.context_stats or {}
        if turn.error:
            # Real provider adapters redact before raising, but the browser
            # boundary must not trust that: scrub the session key and common
            # credential shapes from any error text before it is rendered.
            headline = f"Provider error: {self._sanitize(turn.error)}"
        elif turn.generated:
            headline = "Reply generated."
        else:
            headline = (
                "No provider credentials — BrainOS observed and retrieved, but no "
                "reply was generated. Set an API key and model in the sidebar."
            )
        summary = (
            f" {len(turn.stored_memories)} new memories stored; "
            f"{stats.get('selected_memory_count', 0)} selected for context "
            f"({stats.get('final_context_tokens', 0)} tokens"
        )
        reference = int(stats.get("full_context_reference_tokens", 0) or 0)
        if reference > 0:
            reduction = 100.0 * float(stats.get("context_reduction_vs_full_context", 0.0))
            summary += f" vs {reference} full-context, {reduction:.1f}% reduction"
        summary += f"). Turn {self._turns_used + 1} of {self.max_turns}."
        return headline + summary


__all__ = [
    "BASELINE_MODES_PENDING",
    "BRAINOS_INSTALL_HINT",
    "DEFAULT_MAX_TURNS",
    "MEMORY_MODE_CHOICES",
    "PROVIDER_CHOICES",
    "ChatController",
]
