"""FastAPI-native adapter for Vercel deployment.

This module provides a Vercel-friendly ASGI app that does NOT rely on
Gradio's persistent server. It exposes the same ConversationService logic
via REST endpoints, suitable for a Next.js frontend or direct API use.

Environment variables respected:
- BRAINOS_LAB_DB=:memory: (default for Vercel)
- BRAINOS_LAB_EVAL_ALLOW_GENERATION=0 (disables generation in eval tab)
- RESULTS_ROOT=/tmp/results (override to avoid read-only FS)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on path when imported as api/index.py on Vercel
_repo_root = Path(__file__).resolve().parents[2]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Vercel defaults - must be set before importing storage
os.environ.setdefault("BRAINOS_LAB_DB", ":memory:")
if os.getenv("VERCEL") == "1" and "RESULTS_ROOT" not in os.environ:
    os.environ["RESULTS_ROOT"] = "/tmp/results"

# Now import app internals - noqa: E402 (must happen after path/env bootstrap)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError as exc:
    raise RuntimeError(
        "FastAPI is required for Vercel deployment. "
        "Install with `pip install fastapi uvicorn` or add to requirements.txt"
    ) from exc

from app.controller import UIController, UILimits  # noqa: E402
from app.evaluation import EvaluationPolicy  # noqa: E402
from checkout import REPO_ROOT  # noqa: E402
from storage.sqlite import (  # noqa: E402
    SqliteConversationStore,
    SqliteEvaluationStore,
    SqliteMemoryStore,
    persistence_enabled,
)

# ---------------------------------------------------------------------------
# Pydantic models for REST API
# ---------------------------------------------------------------------------


class ConnectRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    endpoint: str = ""
    temperature: float = 0.2
    max_output_tokens: int | None = None
    session_id: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    mode: str | None = None


class ContextUpdateRequest(BaseModel):
    session_id: str | None = None
    mode: str | None = None
    max_tokens: int | None = None
    max_recent_turns: int | None = None
    recent_turn_budget: int | None = None
    memory_budget: int | None = None
    chunk_budget: int | None = None
    max_memories: int | None = None


# ---------------------------------------------------------------------------
# Controller factory - uses :memory: by default for Vercel
# ---------------------------------------------------------------------------


def _make_controller() -> UIController:
    """Create a controller with Vercel-safe defaults."""

    is_vercel = os.getenv("VERCEL") == "1"
    db_path = os.getenv("BRAINOS_LAB_DB", ":memory:")

    if db_path != ":memory:":
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            db_path = ":memory:"

    try:
        policy = EvaluationPolicy.from_environment()
    except ValueError:
        policy = EvaluationPolicy(
            allowed_presets=("quick",),
            max_tasks=5,
            max_requests=15,
            allow_generation=False,
            results_retention_seconds=None,
        )

    if is_vercel and os.getenv("BRAINOS_LAB_EVAL_ALLOW_GENERATION", "0") in {
        "0",
        "false",
        "no",
        "off",
    }:
        policy = EvaluationPolicy(
            allowed_presets=policy.allowed_presets,
            max_tasks=min(policy.max_tasks, 10),
            max_requests=min(policy.max_requests, 20),
            allow_generation=False,
            max_runs_per_session=policy.max_runs_per_session,
            results_retention_seconds=None,
        )

    limits = UILimits(
        max_turns=100,
        max_message_chars=4000,
        max_input_tokens=16000,
        max_output_tokens=2048,
        max_session_tokens=100000,
        max_session_requests=100,
        request_timeout_seconds=30.0,
    )

    return UIController(
        conversation_store=SqliteConversationStore(db_path),
        memory_store=SqliteMemoryStore(db_path),
        evaluation_store=SqliteEvaluationStore(db_path),
        evaluation_policy=policy,
        limits=limits,
    )


_controller = _make_controller()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(
    title="BrainOS Context Lab - Vercel API",
    description=(
        "BYOK chat with BrainOS memory. "
        "Ephemeral deployment: sessions live in memory only, "
        "evaluation disabled or retrieval-only."
    ),
    version="0.1.0-vercel",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "persistence_enabled": persistence_enabled(),
        "vercel": os.getenv("VERCEL") == "1",
        "db": os.getenv("BRAINOS_LAB_DB", ":memory:"),
        "repo_root": str(REPO_ROOT),
        "eval_allow_generation": os.getenv(
            "BRAINOS_LAB_EVAL_ALLOW_GENERATION", "0"
        ),
        "controller_sessions": len(_controller.sessions._sessions),
    }


@fastapi_app.post("/api/session")
def create_session() -> dict[str, Any]:
    state = _controller.ensure_session(None)
    return {
        "session_id": state.session_id,
        "conversation_id": state.conversation_id,
        "mode": state.context.mode,
    }


@fastapi_app.post("/api/connect")
def connect(req: ConnectRequest) -> dict[str, Any]:
    view = _controller.connect(
        req.session_id,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        endpoint=req.endpoint,
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
    )
    return {
        "status": view.status,
        "session_id": view.session_id,
        "connected": view.connected,
        "models": list(view.models),
        "model_value": view.model_value,
    }


@fastapi_app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")

    if req.mode:
        try:
            _controller.apply_mode(req.session_id, req.mode)
        except Exception:
            pass

    turn = _controller.chat(req.session_id, req.message)

    return {
        "session_id": turn.session_id,
        "history": turn.history,
        "status": turn.status,
        "prompt": turn.prompt,
        "stats": turn.stats,
        "stored_rows": turn.stored_rows,
        "retrieved_rows": turn.retrieved_rows,
        "dropped_rows": turn.dropped_rows,
        "conflict_rows": turn.conflict_rows,
        "chunk_rows": turn.chunk_rows,
        "trace": turn.trace,
        "security": turn.security,
        "usage": turn.usage_report,
        "turn_usage": turn.turn_usage,
    }


@fastapi_app.get("/api/memory/{session_id}")
def get_memory(session_id: str) -> dict[str, Any]:
    _controller.ensure_session(session_id)
    service = _controller.service(session_id)
    try:
        memories = service.stored_memories()
        rows = [
            {
                "memory_id": m.memory_id,
                "text": m.text,
                "memory_type": m.memory_type,
                "relevance": m.relevance,
                "source_turn": m.source_turn,
                "observed_at": m.observed_at,
            }
            for m in memories
        ]
    except Exception:
        rows = []

    return {"session_id": session_id, "memories": rows}


@fastapi_app.get("/api/context/{session_id}")
def get_context(session_id: str) -> dict[str, Any]:
    payload = _controller.context_payload(session_id)
    usage = _controller.usage_payload(session_id)
    return {"context": payload, "usage": usage}


@fastapi_app.post("/api/clear/conversation")
def clear_conversation(session_id: str | None = None) -> dict[str, Any]:
    view = _controller.clear_conversation(session_id)
    return {
        "session_id": view.session_id,
        "history": view.history,
        "status": view.status,
    }


@fastapi_app.post("/api/clear/memory")
def clear_memory(session_id: str | None = None) -> dict[str, Any]:
    view = _controller.clear_memory(session_id)
    return {"session_id": view.session_id, "status": view.status}


@fastapi_app.post("/api/session/end")
def end_session(session_id: str | None = None) -> dict[str, Any]:
    view = _controller.end_session(session_id)
    return {"status": view.status, "new_session_id": view.session_id}


@fastapi_app.get("/api/evaluation/presets")
def eval_presets() -> dict[str, Any]:
    view = _controller.evaluation_presets()
    return {
        "status": view.status,
        "preset_rows": view.preset_rows,
        "ok": view.ok,
    }


@fastapi_app.get("/api/evaluation/preview")
def eval_preview(
    session_id: str | None = None,
    preset: str = "quick",
    dataset: str | None = None,
) -> dict[str, Any]:
    view = _controller.evaluation_preview(
        session_id, preset=preset, dataset=dataset, generate=False
    )
    return {
        "status": view.status,
        "cost_markdown": view.cost_markdown,
        "cost_payload": view.cost_payload,
        "ok": view.ok,
    }


# ---------------------------------------------------------------------------
# Optional: mount Gradio UI at root for minimal lift
# ---------------------------------------------------------------------------


def _try_mount_gradio() -> Any:
    """Attempt to mount Gradio UI onto FastAPI for / route."""

    try:
        import gradio as gr

        from app.ui import create_app as create_gradio_app  # noqa: E402

        demo = create_gradio_app(controller=_controller)

        try:
            demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
        except Exception:
            pass

        mounted = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)
        return mounted
    except Exception as exc:
        print(
            f"[vercel_app] Gradio mount failed, serving API only: {exc}",
            file=sys.stderr,
        )
        return fastapi_app


app_with_gradio = _try_mount_gradio()
app = app_with_gradio

__all__ = ["fastapi_app", "app", "app_with_gradio", "_controller"]
