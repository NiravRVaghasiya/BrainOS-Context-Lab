"""Gradio chat UI wired to the session-scoped ConversationService (Phase 4).

Layout follows the plan: a configuration sidebar (provider, model, API key,
endpoint, temperature, context budget, memory mode), the chat in the middle,
and an inspection column with Memory / Context / Cognitive Trace / Evaluation
tabs.

Callbacks are deliberately thin. Each one forwards to
:class:`app.controller.ChatController`, which returns plain view data; this
module only maps that data onto Gradio components. No BrainOS import, no
provider SDK import, and the API key never travels back to the browser.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any

from .controller import MEMORY_MODE_CHOICES, PROVIDER_CHOICES, ChatController
from .session import SessionManager

TITLE = "BrainOS Context Lab"

STORED_HEADERS = ["Type", "Memory", "Observed (UTC)", "Retrievals", "Source turn", "Status"]
RETRIEVED_HEADERS = ["Rank", "Score", "Relevance", "Type", "Memory", "Flags"]
TRACE_HEADERS = ["Stage", "Detail", "Time (UTC)"]
DROPPED_HEADERS = ["Drop reason", "Memory", "Detail", "Score"]
CONFLICT_HEADERS = ["Subject", "Kept", "Dropped", "Source", "Reason"]

PIPELINE_MARKDOWN = (
    "```text\n"
    "Query → Observe → Recall → Relevance filtering\n"
    "      → Context construction → LLM → Observation\n"
    "```\n\n"
    "The trace below is sanitized (counts and stages, never credentials), and "
    "the tables under it show every memory dropped at each retrieval stage "
    "with its reason."
)

EVALUATION_MARKDOWN = (
    "### Evaluation\n\n"
    "Benchmark controls arrive with the automated evaluation pipeline "
    "(Phase 17) and are gated by the Phase 15 cost controls before they are "
    "exposed here.\n\n"
    "Until then this session enforces a conservative turn limit, and every "
    "turn already records the accounting (tokens, reductions, drops) the "
    "benchmark runner will consume."
)

SECURITY_MARKDOWN = (
    "**Your provider account pays for API usage.** Keys live in server memory "
    "for this session only — never logged, never stored, never sent back to "
    "the browser. *End session* clears them."
)


def _make_chatbot(gr: Any, **kwargs: Any) -> Any:
    """Construct a messages-format chatbot across Gradio 4/5/6.

    Gradio 5 requires ``type="messages"`` for role/content dicts; Gradio 6
    removed the parameter because the messages format is the only format.
    """

    parameters = inspect.signature(gr.Chatbot.__init__).parameters
    if "type" in parameters:
        kwargs.setdefault("type", "messages")
    return gr.Chatbot(**kwargs)


def create_app(controller_factory: Callable[[], ChatController] | None = None) -> Any:
    """Build and return the Gradio application.

    Gradio is an optional dependency so the non-UI package and unit tests can
    be used without installing a web framework. ``controller_factory`` is a
    test seam: production uses one :class:`SessionManager` for every browser
    session it creates.
    """

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The UI requires Gradio. Install it with `pip install -e \".[ui]\"`."
        ) from exc

    if controller_factory is None:
        manager = SessionManager()

        def controller_factory() -> ChatController:
            return ChatController(manager=manager)

    with gr.Blocks(title=TITLE) as demo:
        session = gr.State(None)

        gr.Markdown(
            "# BrainOS Context Lab\n"
            "> Bring your model. Give it memory. Measure context efficiency.\n\n"
            "**Status:** Phase 4 — chat, memory inspection, context accounting, "
            "and the cognitive trace are wired to the session-scoped "
            "`ConversationService`. Baseline modes (Phase 6) and benchmarks "
            "(Phase 17) are not active yet."
        )

        with gr.Row():
            # ---------------------------------------------------------- #
            # Sidebar: provider, context, and session controls
            # ---------------------------------------------------------- #
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Provider")
                provider_dd = gr.Dropdown(
                    list(PROVIDER_CHOICES), value=PROVIDER_CHOICES[0], label="Provider"
                )
                model_dd = gr.Dropdown(
                    choices=[],
                    value=None,
                    allow_custom_value=True,
                    label="Model",
                    info="Type a model id or fetch choices with “List models”.",
                )
                api_key_tb = gr.Textbox(
                    label="API key (session-only)",
                    type="password",
                    placeholder="Held in server memory; never echoed back",
                )
                endpoint_tb = gr.Textbox(
                    label="Endpoint (optional)",
                    placeholder="https://… for OpenAI-compatible servers",
                )
                temperature_sl = gr.Slider(0.0, 2.0, value=0.2, step=0.1, label="Temperature")
                max_output_nb = gr.Number(
                    value=0, label="Max output tokens (0 = provider default)", precision=0
                )
                with gr.Row():
                    validate_btn = gr.Button("Validate connection", variant="secondary")
                    list_models_btn = gr.Button("List models", variant="secondary")
                connection_md = gr.Markdown(
                    "Not connected. Enter an API key and model, then validate the connection."
                )

                gr.Markdown("### Context")
                context_budget_sl = gr.Slider(
                    512, 32768, value=4096, step=256, label="Context budget (max tokens)"
                )
                memory_mode_dd = gr.Dropdown(
                    list(MEMORY_MODE_CHOICES), value="brainos", label="Memory mode"
                )
                mode_notice_md = gr.Markdown("")
                with gr.Accordion("Advanced context settings", open=False):
                    recent_budget_sl = gr.Slider(
                        0, 8192, value=2048, step=64, label="Recent-history budget (tokens)"
                    )
                    memory_budget_sl = gr.Slider(
                        0, 4096, value=1536, step=64, label="Memory budget (tokens)"
                    )
                    max_memories_sl = gr.Slider(1, 24, value=12, step=1, label="Max memories")

                gr.Markdown("### Session")
                with gr.Row():
                    clear_btn = gr.Button("Clear conversation")
                    end_btn = gr.Button("End session", variant="stop")
                gr.Markdown(SECURITY_MARKDOWN)

            # ---------------------------------------------------------- #
            # Main: chat
            # ---------------------------------------------------------- #
            with gr.Column(scale=2):
                gr.Markdown("### Chat")
                chatbot = _make_chatbot(gr, label="Conversation", height=480)
                status_md = gr.Markdown(
                    "Session ready. Configure a provider to generate replies; "
                    "BrainOS observes and retrieves either way."
                )
                with gr.Row():
                    message_tb = gr.Textbox(
                        label="Message",
                        placeholder="Tell the assistant something worth remembering…",
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # ---------------------------------------------------------- #
            # Right: inspection tabs
            # ---------------------------------------------------------- #
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Inspection")
                with gr.Tabs():
                    with gr.Tab("Memory"):
                        gr.Markdown("**Stored memories** (this session)")
                        stored_df = gr.Dataframe(
                            headers=STORED_HEADERS,
                            value=[],
                            interactive=False,
                            wrap=True,
                            label="Stored memories",
                        )
                        gr.Markdown("**Retrieved for the last turn** (rank order)")
                        retrieved_df = gr.Dataframe(
                            headers=RETRIEVED_HEADERS,
                            value=[],
                            interactive=False,
                            wrap=True,
                            label="Retrieved memories",
                        )
                        ranking_json = gr.JSON(label="Ranking components", value=[])
                    with gr.Tab("Context"):
                        context_md = gr.Markdown(
                            "No turn yet — send a message to build context."
                        )
                        stats_json = gr.JSON(label="Context accounting (all fields)", value={})
                        prompt_tb = gr.Textbox(
                            label="Final prompt sent to the model",
                            lines=14,
                            interactive=False,
                        )
                    with gr.Tab("Cognitive Trace"):
                        gr.Markdown(PIPELINE_MARKDOWN)
                        decision_md = gr.Markdown("No decision yet.")
                        trace_df = gr.Dataframe(
                            headers=TRACE_HEADERS,
                            value=[],
                            interactive=False,
                            wrap=True,
                            label="Cognitive trace",
                        )
                        gr.Markdown("**Dropped at each retrieval stage**")
                        dropped_df = gr.Dataframe(
                            headers=DROPPED_HEADERS,
                            value=[],
                            interactive=False,
                            wrap=True,
                            label="Dropped memories",
                        )
                        gr.Markdown("**Conflict resolutions**")
                        conflicts_df = gr.Dataframe(
                            headers=CONFLICT_HEADERS,
                            value=[],
                            interactive=False,
                            wrap=True,
                            label="Conflict resolutions",
                        )
                    with gr.Tab("Evaluation"):
                        gr.Markdown(EVALUATION_MARKDOWN)

        panel_outputs = [
            chatbot,
            status_md,
            stored_df,
            retrieved_df,
            ranking_json,
            context_md,
            stats_json,
            prompt_tb,
            decision_md,
            trace_df,
            dropped_df,
            conflicts_df,
            mode_notice_md,
            connection_md,
        ]
        settings_inputs = [
            provider_dd,
            model_dd,
            api_key_tb,
            endpoint_tb,
            temperature_sl,
            max_output_nb,
            memory_mode_dd,
            context_budget_sl,
            recent_budget_sl,
            memory_budget_sl,
            max_memories_sl,
        ]

        def panel_values(view: dict[str, Any]) -> list[Any]:
            """Map a controller view onto the inspection panels, in order."""

            return [
                view["history"],
                view["status"],
                view["stored_rows"],
                view["retrieved_rows"],
                view["ranking_json"],
                view["context_summary"] or "No turn yet — send a message to build context.",
                view["context_stats"],
                view["final_prompt"],
                view["decision_md"] or "No decision yet.",
                view["trace_rows"],
                view["dropped_rows"],
                view["conflict_rows"],
                view["mode_notice"],
                view["connection_status"],
            ]

        def ensure(controller: ChatController | None) -> ChatController:
            return controller if controller is not None else controller_factory()

        def apply_settings(controller: ChatController, settings: tuple[Any, ...]) -> str:
            """Push the sidebar into the session state; return any rejection."""

            (
                provider,
                model,
                api_key,
                endpoint,
                temperature,
                max_output,
                mode,
                max_tokens,
                recent_budget,
                memory_budget,
                max_memories,
            ) = settings
            notes = [
                controller.apply_provider(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=endpoint,
                    temperature=temperature,
                    max_output_tokens=max_output,
                ),
                controller.set_memory_mode(mode),
                controller.apply_context_settings(
                    max_tokens=max_tokens,
                    recent_turn_budget=recent_budget,
                    memory_budget=memory_budget,
                    max_memories=max_memories,
                ),
            ]
            return " ".join(note for note in notes if "not applied" in note)

        def on_load(controller: ChatController | None) -> list[Any]:
            controller = ensure(controller)
            return [controller, *panel_values(controller.views())]

        def on_send(
            controller: ChatController | None, message: str, *settings: Any
        ) -> list[Any]:
            controller = ensure(controller)
            warning = apply_settings(controller, settings)
            view = controller.send_message(message)
            if warning:
                view["status"] = f"{warning} — {view['status']}"
            return [controller, *panel_values(view), ""]

        def on_validate(controller: ChatController | None, *settings: Any) -> list[Any]:
            controller = ensure(controller)
            apply_settings(controller, settings)
            controller.validate_connection()
            return [controller, controller.views()["connection_status"]]

        def on_list_models(controller: ChatController | None, *settings: Any) -> list[Any]:
            controller = ensure(controller)
            apply_settings(controller, settings)
            controller.list_models()
            view = controller.views()
            return [
                controller,
                gr.update(choices=view["model_choices"], value=view["model_value"] or None),
                view["connection_status"],
            ]

        def on_clear(controller: ChatController | None) -> list[Any]:
            controller = ensure(controller)
            return [controller, *panel_values(controller.clear_conversation())]

        def on_end(controller: ChatController | None) -> list[Any]:
            controller = ensure(controller)
            fresh, view = controller.end_session()
            return [fresh, *panel_values(view)]

        def on_mode_change(controller: ChatController | None, mode: str) -> list[Any]:
            controller = ensure(controller)
            return [controller, controller.set_memory_mode(mode)]

        demo.load(on_load, inputs=[session], outputs=[session, *panel_outputs])
        send_btn.click(
            on_send,
            inputs=[session, message_tb, *settings_inputs],
            outputs=[session, *panel_outputs, message_tb],
        )
        message_tb.submit(
            on_send,
            inputs=[session, message_tb, *settings_inputs],
            outputs=[session, *panel_outputs, message_tb],
        )
        validate_btn.click(
            on_validate,
            inputs=[session, *settings_inputs],
            outputs=[session, connection_md],
        )
        list_models_btn.click(
            on_list_models,
            inputs=[session, *settings_inputs],
            outputs=[session, model_dd, connection_md],
        )
        clear_btn.click(on_clear, inputs=[session], outputs=[session, *panel_outputs])
        end_btn.click(on_end, inputs=[session], outputs=[session, *panel_outputs])
        memory_mode_dd.change(
            on_mode_change, inputs=[session, memory_mode_dd], outputs=[session, mode_notice_md]
        )

    return demo


def main() -> None:
    """Launch the UI using settings compatible with local and HF deployment."""

    demo = create_app()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
