"""Gradio web UI for BrainOS Context Lab.

The layout follows the plan's Phase 4 sketch — configuration sidebar, chat in
the middle, inspection panels on the right — but every callback is a thin
adapter over :class:`app.controller.UIController`. The UI owns no state of its
own beyond the session identifier: the server-side session is authoritative for
the transcript, the memories, and the provider key.

Security properties this file must preserve:

* the API key box is cleared on every connect attempt; the key is never echoed
  back into the browser, not even masked,
* all panel values come from the controller, which sanitizes them against the
  active session key,
* ``gr.State`` only ever holds the non-secret session identifier.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baselines.modes import (
    DEFAULT_MODE,
    EVIDENCE_ITEMS,
    RECENT_WINDOW_BUDGET,
    RECENT_WINDOW_TURNS,
)

from .controller import (
    DEFAULT_MODEL_PLACEHOLDER,
    MODE_CHOICES,
    PROVIDER_LABELS,
    UIController,
)
from .evaluation import public_evaluation_policy
from .panels import (
    CHUNK_COLUMNS,
    CONFLICT_COLUMNS,
    DROPPED_MEMORY_COLUMNS,
    EVALUATION_HEADLINE_COLUMNS,
    EVALUATION_HISTORY_COLUMNS,
    EVALUATION_STAGE_COLUMNS,
    PRESET_COLUMNS,
    RETRIEVED_MEMORY_COLUMNS,
    SECURITY_COLUMNS,
    STORED_MEMORY_COLUMNS,
    evaluation_cost_markdown,
)

TITLE = "BrainOS Context Lab"


def _header_markdown() -> str:
    """Build the header copy, reflecting whether persistence is enabled.

    Phase 14: the same code runs locally (where ``data/brainos_lab.sqlite3`` is
    the default) and on an ephemeral deployment (where an operator may set
    ``BRAINOS_LAB_DB=:memory:`` to keep everything in RAM). The header must
    not claim data is persisted when that would be a lie, and must disclose
    persistence when it does happen.
    """

    try:
        from storage.sqlite import persistence_enabled
    except Exception:  # noqa: BLE001 - header copy must not break the build
        persistence_enabled_result = True
    else:
        persistence_enabled_result = persistence_enabled()

    persistence_line = (
        "Transcript messages and BrainOS memories are persisted to a server-side "
        "SQLite database, with credentials redacted before writing; **Clear "
        "conversation**, **Clear memory**, and **End session** delete the matching "
        "rows."
        if persistence_enabled_result
        else "This deployment runs **fully in memory**: closing the tab, ending the "
        "session, or restarting the Space discards every transcript and every memory."
    )

    return f"""# {TITLE}
> Bring your model. Give it memory. Measure context efficiency.

BrainOS observes the conversation, retrieves what matters, and builds the
prompt. The right-hand panels show exactly what was remembered, what was sent,
and what it would have cost to send everything.

**Your provider account is responsible for API usage and cost.** Keys are held
in server memory for the active session only and are never written to disk.
{persistence_line}
"""

EVALUATION_MARKDOWN = """### Evaluation pipeline

Run the Context Rot benchmark from the browser. This tab drives the same
automated pipeline as the command line (`python -m evaluation.pipeline`), so a
run here and a run in a terminal write the same artifacts and the same
reproducibility manifest.

**Retrieval-only by default.** With **Call my model** off, the pipeline builds
every mode's prompt, measures its tokens, runs retrieval, and compares the
context assembly — without sending anything anywhere. That is free, and it is
the honest way to see what BrainOS does to a prompt. Turn generation on and the
run uses **the API key you entered in the sidebar**: your provider account is
billed directly, and the ceilings below are enforced, not advisory.

**Preview before you spend.** Every number in the cost table is a ceiling the
runner enforces. A prompt larger than the per-request ceiling is *skipped*,
never truncated and never sent; a run that hits a ceiling stops and says so.

**Unset is not zero.** A retrieval-only run grades no answers, so accuracy,
faithfulness, and quality-adjusted efficiency show `—` — never `0.000`, which
would read as "every answer was wrong".

Artifacts land in `results/raw/`, `results/aggregated/`, `results/plots/`, and
`results/report/` for a CLI run, and in a session-scoped `results/ui/<session>/`
directory here. **End session** deletes this session's rows *and* its artifacts.
"""

USAGE_INTRO_MARKDOWN = """### Usage

Session cost accounting: what this conversation has cost so far, how much
budget remains, and which ceilings (if any) have been reached.

**Your provider account is billed directly.** These limits protect against
runaway usage on this demo server but cannot cap what your provider charges.
Token counts use the session's configured counter (estimated by default).
"""


SECURITY_INTRO_MARKDOWN = """### Security

What the guards did while building this session's prompts: instruction-like
retrieved text, structural rewrites (delimiter breakouts, role prefixes,
invisible characters), credentials removed from history or memory, and transcript
messages that arrived claiming a role the application owns.

Findings are **counts and clipped previews**, never the payload or the
credential. The table is bounded; the totals are exact for the session.
"""

FOOTER_MARKDOWN = (
    "Retrieved evidence is data, not instructions: BrainOS memory renders inside "
    "`<retrieved_memory>` delimiters and retrieved transcript chunks inside "
    "`<retrieved_history>` delimiters, each introduced as untrusted text the "
    "model must not treat as instructions."
)


@dataclass
class SidebarComponents:
    """Configuration widgets, in the order :meth:`connect` feeds them."""

    provider: Any
    model: Any
    api_key: Any
    endpoint: Any
    temperature: Any
    max_output_tokens: Any
    connect_btn: Any
    disconnect_btn: Any
    connection_status: Any
    mode: Any
    mode_info: Any
    max_tokens: Any
    max_recent_turns: Any
    recent_turn_budget: Any
    memory_budget: Any
    chunk_budget: Any
    max_memories: Any
    relevance_floor: Any
    recency_half_life_turns: Any
    resolve_conflicts: Any
    drop_stale_memories: Any
    drop_suspicious_memories: Any
    context_btn: Any
    context_status: Any
    #: Phase 15: cost controls for the chat path.
    chat_max_input_tokens: Any
    chat_max_output_tokens: Any
    chat_request_timeout: Any
    chat_max_session_tokens: Any
    cost_status: Any


@dataclass
class ChatComponents:
    """Transcript and session controls."""

    chatbot: Any
    message: Any
    send_btn: Any
    chat_status: Any
    clear_chat_btn: Any
    clear_memory_btn: Any
    refresh_btn: Any
    end_session_btn: Any
    export_btn: Any


@dataclass
class PanelComponents:
    """Inspection panels, in the order :func:`_panel_values` emits them."""

    stored: Any
    retrieved: Any
    dropped: Any
    conflicts: Any
    summary: Any
    stats: Any
    chunks: Any
    prompt: Any
    trace: Any
    trace_events: Any
    security: Any
    security_rows: Any
    security_report: Any
    #: Phase 15: the session's cost accounting — a markdown summary, the JSON
    #: snapshot, and the per-turn usage dict.
    usage_summary: Any
    usage_report: Any
    order: tuple[str, ...] = field(
        default=(
            "stored",
            "retrieved",
            "dropped",
            "conflicts",
            "summary",
            "stats",
            "chunks",
            "prompt",
            "trace",
            "trace_events",
            "security",
            "security_rows",
            "security_report",
            "usage_summary",
            "usage_report",
        )
    )

    def as_outputs(self) -> list[Any]:
        return [getattr(self, name) for name in self.order]


@dataclass
class EvaluationComponents:
    """Evaluation tab widgets, in the order :func:`_evaluation_values` emits them.

    The run form and the preview form share one input list. A visitor who has
    filled in a baseline mode and a token counter and then presses **Preview
    cost** should not have to wonder why two of their controls were ignored —
    so both callbacks read the whole form, and the preview simply does not act
    on the parts that only matter once a run starts.
    """

    preset: Any
    modes: Any
    limit: Any
    dataset: Any
    baseline: Any
    counter: Any
    generate: Any
    render_plots: Any
    preview_btn: Any
    run_btn: Any
    history_btn: Any
    status: Any
    cost: Any
    presets_table: Any
    headline: Any
    stages: Any
    gallery: Any
    report: Any
    artifacts: Any
    repro: Any
    history_table: Any
    downloads: Any
    payload: Any
    order: tuple[str, ...] = field(
        default=(
            "status",
            "cost",
            "presets_table",
            "headline",
            "stages",
            "gallery",
            "report",
            "artifacts",
            "repro",
            "history_table",
            "downloads",
            "payload",
        )
    )

    def as_outputs(self) -> list[Any]:
        return [getattr(self, name) for name in self.order]

    def as_inputs(self) -> list[Any]:
        """The run form, in the order the callbacks read it."""

        return [
            self.preset,
            self.modes,
            self.limit,
            self.dataset,
            self.generate,
            self.render_plots,
            self.baseline,
            self.counter,
        ]


def _build_evaluation(gr: Any, controller: UIController) -> EvaluationComponents:
    """The Evaluation tab: the benchmark workbench (Phase 17).

    Full width rather than squeezed into the inspection column, because the
    pipeline's output is a matrix — one row per mode and trial, one column per
    metric — and a table that scrolls sideways in a 340px column is a table
    nobody reads.
    """

    runner = controller.evaluation
    catalogue = controller.evaluation_presets()
    preset_choices = list(runner.preset_choices())
    dataset_choices = list(runner.dataset_choices())
    with gr.Tab("Evaluation"):
        gr.Markdown(EVALUATION_MARKDOWN)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("#### Run configuration")
                preset = gr.Dropdown(
                    choices=preset_choices,
                    value=str(preset_choices[0][1]) if preset_choices else "quick",
                    label="Preset (budget ceilings)",
                )
                modes = gr.Dropdown(
                    choices=list(runner.mode_choices()),
                    value=[],
                    multiselect=True,
                    label="Modes (empty = the preset's modes)",
                )
                limit = gr.Number(
                    value=0, precision=0, label="Task limit (0 = the preset's ceiling)"
                )
                dataset = gr.Dropdown(
                    choices=dataset_choices,
                    value=str(dataset_choices[0]),
                    allow_custom_value=True,
                    label="Dataset (inside `benchmarks/` only)",
                )
                baseline = gr.Dropdown(
                    choices=list(runner.baseline_choices()),
                    value="full_context",
                    label="Paired-comparison baseline",
                )
                counter = gr.Radio(
                    choices=[
                        ("Estimate (no network)", "estimate"),
                        ("tiktoken (exact, slower)", "tiktoken"),
                    ],
                    value="estimate",
                    label="Token counter",
                )
                generate = gr.Checkbox(
                    value=False,
                    label="Call my model (spends my sidebar API key)",
                )
                render_plots = gr.Checkbox(value=True, label="Render figures")
                with gr.Row():
                    preview_btn = gr.Button("Preview cost")
                    run_btn = gr.Button("Run benchmark", variant="primary")
                    history_btn = gr.Button("History")
            with gr.Column(scale=2, min_width=420):
                status = gr.Markdown(catalogue.status)
                cost = gr.Markdown(evaluation_cost_markdown({}))
                presets_table = gr.Dataframe(
                    headers=list(PRESET_COLUMNS),
                    label="Preset ceilings",
                    value=catalogue.preset_rows,
                    interactive=False,
                    wrap=True,
                )
        gr.Markdown("#### Results")
        headline = gr.Dataframe(
            headers=list(EVALUATION_HEADLINE_COLUMNS),
            label="Evaluation matrix (one row per mode and trial)",
            value=[],
            interactive=False,
            wrap=True,
        )
        with gr.Row():
            with gr.Column(scale=2, min_width=380):
                stages = gr.Dataframe(
                    headers=list(EVALUATION_STAGE_COLUMNS),
                    label="Pipeline stages",
                    value=[],
                    interactive=False,
                    wrap=True,
                )
                history_table = gr.Dataframe(
                    headers=list(EVALUATION_HISTORY_COLUMNS),
                    label="This session's runs",
                    value=[],
                    interactive=False,
                    wrap=True,
                )
            with gr.Column(scale=1, min_width=280):
                gallery = gr.Gallery(
                    label="Figures (`results/plots/`)",
                    columns=2,
                    height=260,
                    interactive=False,
                )
                downloads = gr.File(label="Report, manifest, and figures")
        report = gr.Markdown("")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                artifacts = gr.Markdown("")
            with gr.Column(scale=1, min_width=300):
                repro = gr.Markdown("")
        payload = gr.JSON(label="Run summary", value={})

    return EvaluationComponents(
        preset=preset,
        modes=modes,
        limit=limit,
        dataset=dataset,
        baseline=baseline,
        counter=counter,
        generate=generate,
        render_plots=render_plots,
        preview_btn=preview_btn,
        run_btn=run_btn,
        history_btn=history_btn,
        status=status,
        cost=cost,
        presets_table=presets_table,
        headline=headline,
        stages=stages,
        gallery=gallery,
        report=report,
        artifacts=artifacts,
        repro=repro,
        history_table=history_table,
        downloads=downloads,
        payload=payload,
    )


def create_app(controller: UIController | None = None) -> Any:
    """Build and return the Gradio application.

    Gradio is an optional dependency so the non-UI package, the CLI, and the
    unit tests can be used without installing a web framework.

    Phase 5: when no controller is injected, the default one is constructed
    with the server-wide SQLite backends (path overridable via
    ``BRAINOS_LAB_DB``), so the shipped app persists conversations and memory
    mirrors. An injected controller keeps exactly the stores its caller gave
    it — tests pass fakes and never touch the real database.
    """

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The UI requires Gradio. Install it with `pip install -e \".[ui]\"`."
        ) from exc

    if controller is None:
        from storage.sqlite import (
            SqliteConversationStore,
            SqliteEvaluationStore,
            SqliteMemoryStore,
        )

        controller = UIController(
            conversation_store=SqliteConversationStore(),
            memory_store=SqliteMemoryStore(),
            evaluation_store=SqliteEvaluationStore(),
            # The public browser surface is retrieval-only and bounded by
            # default. Tests, local callers, and explicitly injected
            # controllers keep their chosen policy instead of silently
            # inheriting deployment defaults.
            evaluation_policy=public_evaluation_policy(),
        )

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(_header_markdown())
        session_state = gr.State(None)  # non-secret session identifier only

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                sidebar = _build_sidebar(gr)
            with gr.Column(scale=2, min_width=420):
                chat = _build_chat(gr)
            with gr.Column(scale=1, min_width=340):
                panels = _build_inspection(gr)

        # Phase 17: the benchmark workbench is its own full-width tab, below the
        # chat/inspection row it reports on.
        evaluation = _build_evaluation(gr, controller)

        gr.Markdown(FOOTER_MARKDOWN)

        _wire(gr, controller, session_state, sidebar, chat, panels, evaluation)
        # A fresh browser tab gets a fresh session. The controller also
        # self-heals if this never fires (API clients, restored pages).
        demo.load(
            _start_session(controller),
            inputs=None,
            outputs=[session_state, sidebar.mode_info],
        )

    return demo


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def _build_sidebar(gr: Any) -> SidebarComponents:
    """Provider and context controls."""

    gr.Markdown("### Provider")
    provider = gr.Dropdown(
        choices=[list(pair) for pair in PROVIDER_LABELS],
        value=PROVIDER_LABELS[0][1],
        label="Provider",
    )
    model = gr.Dropdown(
        choices=[],
        value=None,
        label="Model",
        allow_custom_value=True,
        filterable=True,
        info=f"Type your own, e.g. {DEFAULT_MODEL_PLACEHOLDER}",
    )
    api_key = gr.Textbox(
        label="API key",
        type="password",
        placeholder="Session-only key — never stored",
    )
    endpoint = gr.Textbox(
        label="Endpoint",
        placeholder="Optional base URL for OpenAI-compatible servers",
    )
    temperature = gr.Slider(0, 2, value=0.2, step=0.1, label="Temperature")
    max_output_tokens = gr.Number(
        value=None, label="Max output tokens (blank = provider default)", precision=0
    )
    with gr.Row():
        connect_btn = gr.Button("Connect", variant="primary")
        disconnect_btn = gr.Button("Forget key")
    connection_status = gr.Markdown("Not connected.")

    gr.Markdown("### Context")
    mode = gr.Dropdown(
        choices=[list(pair) for pair in MODE_CHOICES],
        value=DEFAULT_MODE,
        label="Baseline mode",
        info="Switching modes rewrites the budgets below to that mode's defaults.",
    )
    mode_info = gr.Markdown("")
    max_tokens = gr.Slider(512, 32768, value=4096, step=256, label="Max context tokens")
    max_recent_turns = gr.Number(
        value=RECENT_WINDOW_TURNS,
        label="Recent turns kept (blank = whole conversation)",
        precision=0,
        info="Mode A leaves this blank; Mode B sets it to a window.",
    )
    recent_turn_budget = gr.Slider(
        0,
        8192,
        value=RECENT_WINDOW_BUDGET,
        step=64,
        label="Recent-history budget (tokens)",
        info="A smaller window forces the prompt to lean on retrieved evidence.",
    )
    memory_budget = gr.Slider(0, 4096, value=1024, step=64, label="Memory budget (tokens)")
    chunk_budget = gr.Slider(
        0,
        4096,
        value=0,
        step=64,
        label="Retrieved-chunk budget (tokens)",
        info="Only Modes C and E spend this; it is the lexical-RAG evidence slot.",
    )
    max_memories = gr.Slider(
        1, 20, value=EVIDENCE_ITEMS, step=1, label="Max memories in prompt"
    )
    with gr.Accordion("Retrieval policy", open=False):
        relevance_floor = gr.Slider(0.0, 1.0, value=0.12, step=0.01, label="Relevance floor")
        recency_half_life_turns = gr.Slider(
            1.0, 200.0, value=20.0, step=1.0, label="Recency half-life (turns)"
        )
        resolve_conflicts = gr.Checkbox(value=True, label="Resolve conflicts")
        drop_stale_memories = gr.Checkbox(value=True, label="Drop stale memories")
        drop_suspicious_memories = gr.Checkbox(
            value=False, label="Drop instruction-like memories"
        )
    context_btn = gr.Button("Apply context settings")
    context_status = gr.Markdown("")

    gr.Markdown("### Cost Controls")
    gr.Markdown(
        "Per-request and per-session ceilings protect against runaway usage. "
        "Your provider account is billed directly — these limits cap what this "
        "demo server will send, not what your provider charges."
    )
    chat_max_input_tokens = gr.Number(
        value=32_000,
        label="Max input tokens (per request)",
        precision=0,
        info="Prompts larger than this are refused before the provider is called.",
    )
    chat_max_output_tokens = gr.Number(
        value=4_096,
        label="Max output tokens (per request)",
        precision=0,
        info="The maximum completion length per request.",
    )
    chat_request_timeout = gr.Number(
        value=120,
        label="Request timeout (seconds)",
        precision=0,
        info="A provider call that takes longer is refused and recorded.",
    )
    chat_max_session_tokens = gr.Number(
        value=500_000,
        label="Max session tokens (total)",
        precision=0,
        info="Input + output tokens across the whole session.",
    )
    cost_status = gr.Markdown("")

    return SidebarComponents(
        provider=provider,
        model=model,
        api_key=api_key,
        endpoint=endpoint,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        connect_btn=connect_btn,
        disconnect_btn=disconnect_btn,
        connection_status=connection_status,
        mode=mode,
        mode_info=mode_info,
        max_tokens=max_tokens,
        max_recent_turns=max_recent_turns,
        recent_turn_budget=recent_turn_budget,
        memory_budget=memory_budget,
        chunk_budget=chunk_budget,
        max_memories=max_memories,
        relevance_floor=relevance_floor,
        recency_half_life_turns=recency_half_life_turns,
        resolve_conflicts=resolve_conflicts,
        drop_stale_memories=drop_stale_memories,
        drop_suspicious_memories=drop_suspicious_memories,
        context_btn=context_btn,
        context_status=context_status,
        chat_max_input_tokens=chat_max_input_tokens,
        chat_max_output_tokens=chat_max_output_tokens,
        chat_request_timeout=chat_request_timeout,
        chat_max_session_tokens=chat_max_session_tokens,
        cost_status=cost_status,
    )


def _build_chat(gr: Any) -> ChatComponents:
    """Transcript plus the session controls under it."""

    # Gradio 6 renders the messages format ({"role", "content"}) natively,
    # which is also the provider-neutral shape the context builder produces.
    chatbot = gr.Chatbot(label="Conversation", height=520)
    with gr.Row():
        message = gr.Textbox(
            label="Message",
            placeholder="Chat with BrainOS memory in the loop",
            scale=4,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
    chat_status = gr.Markdown("")

    with gr.Row():
        clear_chat_btn = gr.Button("Clear conversation")
        clear_memory_btn = gr.Button("Clear memory")
        refresh_btn = gr.Button("Refresh panels")
        end_session_btn = gr.Button("End session", variant="stop")
    export_btn = gr.DownloadButton("Export session JSON")

    return ChatComponents(
        chatbot=chatbot,
        message=message,
        send_btn=send_btn,
        chat_status=chat_status,
        clear_chat_btn=clear_chat_btn,
        clear_memory_btn=clear_memory_btn,
        refresh_btn=refresh_btn,
        end_session_btn=end_session_btn,
        export_btn=export_btn,
    )


def _build_inspection(gr: Any) -> PanelComponents:
    """Right-hand inspection tabs."""

    gr.Markdown("### Inspection")
    with gr.Tab("Memory"):
        stored = gr.Dataframe(
            headers=list(STORED_MEMORY_COLUMNS),
            label="Stored memories",
            value=[],
            interactive=False,
            wrap=True,
        )
        retrieved = gr.Dataframe(
            headers=list(RETRIEVED_MEMORY_COLUMNS),
            label="In this prompt (ranked)",
            value=[],
            interactive=False,
            wrap=True,
        )
        dropped = gr.Dataframe(
            headers=list(DROPPED_MEMORY_COLUMNS),
            label="Recalled but filtered out (audit)",
            value=[],
            interactive=False,
            wrap=True,
        )
        conflicts = gr.Dataframe(
            headers=list(CONFLICT_COLUMNS),
            label="Conflicts",
            value=[],
            interactive=False,
            wrap=True,
        )
    with gr.Tab("Context"):
        summary = gr.Markdown("")
        stats = gr.JSON(label="Context statistics", value={})
        chunks = gr.Dataframe(
            headers=list(CHUNK_COLUMNS),
            label="Retrieved transcript chunks (Modes C/E)",
            value=[],
            interactive=False,
            wrap=True,
        )
        prompt = gr.Textbox(
            label="Final prompt sent to the model",
            lines=16,
            max_lines=40,
        )
    with gr.Tab("Cognitive Trace"):
        trace = gr.Markdown("")
        trace_events = gr.JSON(label="Runtime trace events", value=[])
    with gr.Tab("Security"):
        security = gr.Markdown(SECURITY_INTRO_MARKDOWN)
        security_rows = gr.Dataframe(
            headers=list(SECURITY_COLUMNS),
            label="Guard findings (most recent last)",
            value=[],
            interactive=False,
            wrap=True,
        )
        security_report = gr.JSON(label="Security report", value={})
    with gr.Tab("Usage"):
        usage_summary = gr.Markdown(USAGE_INTRO_MARKDOWN)
        usage_report = gr.JSON(label="Session usage", value={})

    return PanelComponents(
        stored=stored,
        retrieved=retrieved,
        dropped=dropped,
        conflicts=conflicts,
        summary=summary,
        stats=stats,
        chunks=chunks,
        prompt=prompt,
        trace=trace,
        trace_events=trace_events,
        security=security,
        security_rows=security_rows,
        security_report=security_report,
        usage_summary=usage_summary,
        usage_report=usage_report,
    )


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def _wire(
    gr: Any,
    controller: UIController,
    session_state: Any,
    sidebar: SidebarComponents,
    chat: ChatComponents,
    panels: PanelComponents,
    evaluation: EvaluationComponents,
) -> None:
    """Attach the controller to the widgets.

    Kept separate from the layout so the components can be built (and the
    callback ordering asserted) without launching a server.
    """

    panel_outputs = panels.as_outputs()
    chat_outputs = [chat.chatbot, chat.message, chat.chat_status, *panel_outputs]
    silent_outputs = [chat.chatbot, chat.chat_status, *panel_outputs]

    sidebar.connect_btn.click(
        _connect(controller),
        inputs=[
            sidebar.provider,
            sidebar.model,
            sidebar.api_key,
            sidebar.endpoint,
            sidebar.temperature,
            sidebar.max_output_tokens,
            session_state,
        ],
        outputs=[
            sidebar.connection_status,
            sidebar.model,
            sidebar.api_key,
            session_state,
        ],
    )
    sidebar.disconnect_btn.click(
        _disconnect(controller),
        inputs=[session_state],
        outputs=[
            sidebar.connection_status,
            sidebar.model,
            sidebar.api_key,
            session_state,
        ],
    )
    sidebar.context_btn.click(
        _update_context(controller),
        inputs=[
            sidebar.mode,
            sidebar.max_tokens,
            sidebar.max_recent_turns,
            sidebar.recent_turn_budget,
            sidebar.memory_budget,
            sidebar.chunk_budget,
            sidebar.max_memories,
            sidebar.relevance_floor,
            sidebar.recency_half_life_turns,
            sidebar.resolve_conflicts,
            sidebar.drop_stale_memories,
            sidebar.drop_suspicious_memories,
            session_state,
        ],
        outputs=[sidebar.context_status, session_state],
    )
    # Phase 6: switching baseline mode is its own action, not a field of the
    # "apply" form. It writes the mode's budgets server-side and pushes them
    # back into the sliders, so what the sidebar displays is always what the
    # next turn will actually run.
    sidebar.mode.change(
        _apply_mode(controller),
        inputs=[sidebar.mode, session_state],
        outputs=[
            sidebar.context_status,
            sidebar.mode_info,
            sidebar.max_tokens,
            sidebar.max_recent_turns,
            sidebar.recent_turn_budget,
            sidebar.memory_budget,
            sidebar.chunk_budget,
            sidebar.max_memories,
            session_state,
        ],
    )

    chat.send_btn.click(
        _chat(controller),
        inputs=[chat.message, session_state],
        outputs=[*chat_outputs, session_state],
    )
    # Enter-to-send is the same callback; it does not need its own API endpoint.
    chat.message.submit(
        _chat(controller),
        inputs=[chat.message, session_state],
        outputs=[*chat_outputs, session_state],
        api_name=False,
    )
    chat.refresh_btn.click(
        _refresh(controller), inputs=[session_state], outputs=[*silent_outputs, session_state]
    )
    chat.clear_chat_btn.click(
        _clear_conversation(controller),
        inputs=[session_state],
        outputs=[*silent_outputs, session_state],
    )
    chat.clear_memory_btn.click(
        _clear_memory(controller),
        inputs=[session_state],
        outputs=[*silent_outputs, session_state],
    )
    chat.end_session_btn.click(
        _end_session(controller),
        inputs=[session_state],
        outputs=[*silent_outputs, session_state],
    )
    chat.export_btn.click(
        _export(controller), inputs=[session_state], outputs=[chat.export_btn]
    )
    # Phase 17: the Evaluation tab. Preview and run share one input list so a
    # visitor's form is read whole either way; history re-reads the session's
    # runs without touching the form.
    evaluation_outputs = [*evaluation.as_outputs(), session_state]
    evaluation.preview_btn.click(
        _evaluation_preview(controller),
        inputs=[*evaluation.as_inputs(), session_state],
        outputs=evaluation_outputs,
    )
    evaluation.run_btn.click(
        _evaluation_run(controller),
        inputs=[*evaluation.as_inputs(), session_state],
        outputs=evaluation_outputs,
    )
    evaluation.history_btn.click(
        _evaluation_history(controller),
        inputs=[session_state],
        outputs=evaluation_outputs,
    )

    # Phase 15: cost control widgets update the controller's limits on change.
    for widget in (
        sidebar.chat_max_input_tokens,
        sidebar.chat_max_output_tokens,
        sidebar.chat_request_timeout,
        sidebar.chat_max_session_tokens,
    ):
        widget.change(
            _update_costs(controller),
            inputs=[
                sidebar.chat_max_input_tokens,
                sidebar.chat_max_output_tokens,
                sidebar.chat_request_timeout,
                sidebar.chat_max_session_tokens,
                session_state,
            ],
            outputs=[sidebar.cost_status, session_state],
        )


# --------------------------------------------------------------------------- #
# Callback factories
# --------------------------------------------------------------------------- #


def _start_session(controller: UIController) -> Any:
    def start_session() -> tuple[str, str]:
        """Start a session and describe the baseline mode it begins in."""

        state = controller.ensure_session(None)
        profile = state.context.mode_profile()
        return state.session_id, f"**{profile.label}** — {profile.description}"

    return start_session


def _connect(controller: UIController) -> Any:
    def connect(
        provider: str,
        model: str,
        api_key: str,
        endpoint: str,
        temperature: float,
        max_output_tokens: Any,
        session_id: str | None,
    ) -> tuple[Any, ...]:
        import gradio as gr

        session_id = controller.ensure_session(session_id).session_id
        view = controller.connect(
            session_id,
            provider=provider,
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            temperature=temperature,
            max_output_tokens=_optional_int(max_output_tokens),
        )
        choices = list(view.models) or ([model] if model else [])
        return (
            view.status,
            gr.update(choices=choices, value=view.model_value or model),
            view.key_value,
            view.session_id,
        )

    return connect


def _disconnect(controller: UIController) -> Any:
    def disconnect(session_id: str | None) -> tuple[Any, ...]:
        import gradio as gr

        session_id = controller.ensure_session(session_id).session_id
        view = controller.disconnect(session_id)
        return (
            view.status,
            gr.update(choices=[], value=view.model_value),
            view.key_value,
            view.session_id,
        )

    return disconnect


def _update_context(controller: UIController) -> Any:
    def update_context(
        mode: str,
        max_tokens: float,
        max_recent_turns: Any,
        recent_turn_budget: float,
        memory_budget: float,
        chunk_budget: float,
        max_memories: float,
        relevance_floor: float,
        recency_half_life_turns: float,
        resolve_conflicts: bool,
        drop_stale_memories: bool,
        drop_suspicious_memories: bool,
        session_id: str | None,
    ) -> tuple[str, str]:
        # Resolve the session *before* acting, so the id returned to the
        # browser is the session whose settings were changed.
        session_id = controller.ensure_session(session_id).session_id
        status = controller.update_context(
            session_id,
            mode=mode,
            max_tokens=int(max_tokens),
            max_recent_turns=_optional_turns(max_recent_turns),
            recent_turn_budget=int(recent_turn_budget),
            memory_budget=int(memory_budget),
            chunk_budget=int(chunk_budget),
            max_memories=int(max_memories),
            relevance_floor=float(relevance_floor),
            recency_half_life_turns=float(recency_half_life_turns),
            resolve_conflicts=bool(resolve_conflicts),
            drop_stale_memories=bool(drop_stale_memories),
            drop_suspicious_memories=bool(drop_suspicious_memories),
        )
        return status, controller.ensure_session(session_id).session_id

    return update_context


def _update_costs(controller: UIController) -> Any:
    """Callback for the cost control widgets (Phase 15)."""

    def update_costs(
        max_input_tokens: Any,
        max_output_tokens: Any,
        request_timeout: Any,
        max_session_tokens: Any,
        session_id: str | None,
    ) -> tuple[str, str]:
        session_id = controller.ensure_session(session_id).session_id
        status = controller.update_costs(
            session_id,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            request_timeout_seconds=request_timeout,
            max_session_tokens=max_session_tokens,
        )
        return status, session_id

    return update_costs


def _apply_mode(controller: UIController) -> Any:
    """Callback for the baseline-mode selector (Phase 6)."""

    def apply_mode(mode: str, session_id: str | None) -> tuple[Any, ...]:
        import gradio as gr

        # Resolve the session *before* acting. Passing the raw input back into
        # ``ensure_session`` afterwards would mint a second session and hand the
        # browser an id whose mode was never changed.
        session_id = controller.ensure_session(session_id).session_id
        status, settings = controller.apply_mode(session_id, mode)
        profile = settings.mode_profile()
        return (
            status,
            f"**{profile.label}** — {profile.description}",
            gr.update(value=settings.max_tokens),
            gr.update(value=settings.max_recent_turns),
            gr.update(value=settings.recent_turn_budget),
            gr.update(value=settings.memory_budget),
            gr.update(value=settings.chunk_budget),
            gr.update(value=settings.max_memories),
            session_id,
        )

    return apply_mode


def _optional_turns(value: Any) -> int | None:
    """Map the "recent turns kept" box onto ``max_recent_turns``.

    A blank box means "no turn limit" (Mode A), which is ``None`` rather than
    ``0``: zero turns would send no raw history at all, a different and much
    more aggressive setting that belongs in a mode, not in an empty field.
    """

    if value is None or value == "":
        return None
    try:
        turns = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, turns)


def _chat(controller: UIController) -> Any:
    def chat(message: str, session_id: str | None) -> tuple[Any, ...]:
        """Run one turn; the returned tuple is the declared output order."""

        view = controller.chat(session_id, message)
        return (
            view.history,
            "",
            view.status,
            *_panel_values(view),
            view.session_id,
        )

    return chat


def _refresh(controller: UIController) -> Any:
    def refresh(session_id: str | None) -> tuple[Any, ...]:
        return _silent_values(controller.refresh(session_id))

    return refresh


def _clear_conversation(controller: UIController) -> Any:
    def clear_conversation(session_id: str | None) -> tuple[Any, ...]:
        return _silent_values(controller.clear_conversation(session_id))

    return clear_conversation


def _clear_memory(controller: UIController) -> Any:
    def clear_memory(session_id: str | None) -> tuple[Any, ...]:
        return _silent_values(controller.clear_memory(session_id))

    return clear_memory


def _end_session(controller: UIController) -> Any:
    def end_session(session_id: str | None) -> tuple[Any, ...]:
        ended = controller.end_session(session_id)
        return _silent_values(controller.refresh(ended.session_id))

    return end_session


def _export(controller: UIController) -> Any:
    def export_session(session_id: str | None) -> Any:
        import gradio as gr

        text = controller.export_text(session_id)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="brainos-context-lab-session-",
            delete=False,
        ) as handle:
            handle.write(text)
            path = handle.name
        return gr.update(value=path, visible=True)

    return export_session


def _evaluation_preview(controller: UIController) -> Any:
    def preview_cost(
        preset: Any,
        modes: Any,
        limit: Any,
        dataset: Any,
        generate: Any,
        # Read whole, acted on only by a run: figures, paired baselines, and the
        # token counter have nothing to preview.
        render_plots: Any,
        baseline: Any,
        counter: Any,
        session_id: str | None,
    ) -> tuple[Any, ...]:
        view = controller.evaluation_preview(
            session_id,
            preset=str(preset or "quick"),
            modes=_selected_modes(modes),
            limit=_optional_int(limit) or 0,
            dataset=_optional_text(dataset),
            generate=bool(generate),
        )
        return (*_evaluation_values(view), view.session_id)

    return preview_cost


def _evaluation_run(controller: UIController) -> Any:
    def run_benchmark(
        preset: Any,
        modes: Any,
        limit: Any,
        dataset: Any,
        generate: Any,
        render_plots: Any,
        baseline: Any,
        counter: Any,
        session_id: str | None,
    ) -> tuple[Any, ...]:
        view = controller.run_evaluation(
            session_id,
            preset=str(preset or "quick"),
            modes=_selected_modes(modes),
            limit=_optional_int(limit) or 0,
            dataset=_optional_text(dataset),
            generate=bool(generate),
            render_plots=bool(render_plots),
            baseline_mode=str(baseline or "full_context"),
            token_counter=str(counter or "estimate"),
        )
        return (*_evaluation_values(view), view.session_id)

    return run_benchmark


def _evaluation_history(controller: UIController) -> Any:
    def evaluation_history(session_id: str | None) -> tuple[Any, ...]:
        view = controller.evaluation_history(session_id)
        return (*_evaluation_values(view), view.session_id)

    return evaluation_history


def _selected_modes(value: Any) -> list[str] | None:
    """The multiselect's modes, or ``None`` for "the preset's own modes"."""

    if value is None:
        return None
    selected = [str(mode) for mode in value if str(mode).strip()]
    return selected or None


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _evaluation_values(view: Any) -> tuple[Any, ...]:
    """Evaluation tab values, in :class:`EvaluationComponents` order.

    Downloads are filtered to files that exist: a refused or aborted run has no
    report, and handing Gradio a path that is not there turns a readable refusal
    into a broken widget.
    """

    downloads = [
        path
        for path in (view.report_path, view.artifact_path, *view.plots)
        if path and Path(str(path)).is_file()
    ]
    return (
        view.status,
        view.cost_markdown,
        view.preset_rows,
        view.headline_rows,
        view.stage_rows,
        [path for path in view.plots if Path(str(path)).is_file()],
        view.report_markdown,
        view.artifacts_markdown,
        view.repro_markdown,
        view.history_rows,
        downloads,
        view.summary or view.cost_payload or {},
    )


def _panel_values(view: Any) -> tuple[Any, ...]:
    """Inspection panel values, in :class:`PanelComponents` order."""

    return (
        view.stored_rows,
        view.retrieved_rows,
        view.dropped_rows,
        view.conflict_rows,
        view.summary,
        view.stats,
        view.chunk_rows,
        view.prompt,
        view.trace,
        view.trace_events,
        view.security,
        view.security_rows,
        view.security_report,
        view.usage_summary,
        view.usage_report,
    )


def _silent_values(view: Any) -> tuple[Any, ...]:
    """Values for actions that do not return a new message box."""

    return (view.history, view.status, *_panel_values(view), view.session_id)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


#: Maximum concurrent callbacks Gradio will run for this Space. Above this,
#: visitors wait in a short queue. The number is deliberately modest: each turn
#: runs BrainOS recall plus a provider call, both of which are CPU/network
#: bound, and a BYOK public demo must protect one visitor's budget from being
#: starved by traffic spikes. Phase 15 will layer per-turn token/turn ceilings
#: on top of this concurrency gate.
DEFAULT_CONCURRENCY_LIMIT = 3
#: Maximum queue size before Gradio returns "queue full" to new visitors rather
#: than letting wait times grow without bound.
DEFAULT_MAX_QUEUE_SIZE = 32


def main() -> None:
    """Launch the UI using settings compatible with local and HF deployment.

    Phase 14: every option here has an environment-variable override so a
    deployment (HF Space, Docker, a local dev server) can tune the server
    without editing source. Gradio's queue is enabled unconditionally — a
    public surface must serialize callbacks to a bounded pool instead of
    letting every request hit the BrainOS runtime and SQLite concurrently.
    """

    demo = create_app()

    concurrency_limit = int(os.getenv("BRAINOS_LAB_CONCURRENCY", str(DEFAULT_CONCURRENCY_LIMIT)))
    max_queue_size = int(os.getenv("BRAINOS_LAB_MAX_QUEUE", str(DEFAULT_MAX_QUEUE_SIZE)))
    demo.queue(
        default_concurrency_limit=concurrency_limit,
        max_size=max_queue_size,
        api_open=False,
    )

    launch_options: dict[str, Any] = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        "show_error": True,
        "share": False,
    }
    # ``strict_cors`` defaults to True (Gradio refuses cross-origin requests to
    # a local server). Deployments that must be embedded — the sandbox preview,
    # an HF Space iframe — opt out explicitly rather than the app defaulting to
    # a permissive posture.
    strict_cors = os.getenv("GRADIO_STRICT_CORS")
    if strict_cors is not None:
        launch_options["strict_cors"] = strict_cors.strip().lower() not in {
            "0",
            "false",
            "no",
        }
    demo.launch(**launch_options)


__all__ = ["TITLE", "create_app", "main"]
