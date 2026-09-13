"""Gradio UI scaffold.

The callbacks for provider calls, BrainOS observation, and evaluation are
intentionally not wired yet. Keeping the UI shell working makes the planned
surfaces visible while Phase 0 validates the upstream runtime API.
"""

from __future__ import annotations

import os

TITLE = "BrainOS Context Lab"


def create_app():
    """Build and return the Gradio application.

    Gradio is an optional dependency so the non-UI package and unit tests can
    be used without installing a web framework.
    """

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The UI requires Gradio. Install it with `pip install -e \".[ui]\"`."
        ) from exc

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(
            "# BrainOS Context Lab\n"
            "> Bring your model. Give it memory. Measure context efficiency.\n\n"
            "**Status:** Phase 2 BrainOS adapter mapping is in place. Chat "
            "callbacks are not wired yet."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Configuration")
                gr.Dropdown(["OpenAI", "OpenAI-compatible"], label="Provider", value="OpenAI")
                gr.Textbox(label="Model", placeholder="Model identifier")
                gr.Textbox(label="API key", type="password", placeholder="Session-only key")
                gr.Textbox(label="Endpoint", placeholder="Optional compatible endpoint")
                gr.Slider(0, 2, value=0.2, step=0.1, label="Temperature")
                gr.Slider(512, 32768, value=4096, step=256, label="Context budget")
                gr.Dropdown(
                    ["brainos", "full_context", "sliding_window", "rag"],
                    value="brainos",
                    label="Memory mode",
                )
                gr.Markdown(
                    "Your provider account is responsible for API usage and cost. "
                    "Keys are intended to remain session-local."
                )

            with gr.Column(scale=2):
                gr.Markdown("### Chat")
                gr.Chatbot(label="Conversation", type="messages", height=500)
                gr.Textbox(label="Message", placeholder="Chat wiring will be added after Phase 0")

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Inspection")
                with gr.Tab("Memory"):
                    gr.JSON(label="Stored memories", value=[])
                    gr.JSON(label="Retrieved memories", value=[])
                with gr.Tab("Context"):
                    gr.JSON(label="Context statistics", value={})
                    gr.Textbox(label="Final prompt", lines=12)
                with gr.Tab("Cognitive Trace"):
                    gr.Markdown(
                        "Query → Observe → Recall → Relevance filtering → "
                        "Context construction → LLM → Observation"
                    )
                with gr.Tab("Evaluation"):
                    gr.Markdown(
                        "Benchmark controls and exports will be added in the evaluation phases."
                    )

    return demo


def main() -> None:
    """Launch the UI using settings compatible with local and HF deployment."""

    demo = create_app()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
