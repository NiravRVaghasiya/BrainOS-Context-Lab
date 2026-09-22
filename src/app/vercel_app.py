"""FastAPI-native adapter for Vercel deployment.

This module provides a Vercel-friendly ASGI app that does NOT rely on
Gradio's persistent server. It exposes the same ConversationService logic
via REST endpoints, suitable for a Next.js frontend or direct API use.

Why this exists:
- Gradio's `demo.launch()` + `demo.queue()` assumes a long-lived process.
  Vercel Functions are short-lived, ephemeral, and per-request.
- SQLite file persistence doesn't work on Vercel (read-only FS except /tmp).
- Evaluation pipeline writes to `results/` and can run 70+ mins - impossible
  in 10-60s function timeout.

This adapter:
- Uses :memory: stores by default (honest about ephemerality)
- Writes artifacts to /tmp if needed
- Exposes /api/chat, /api/session, /api/memory, /api/context, /api/health
- Optionally mounts Gradio UI at / if gradio is available (for minimal lift)

Environment variables respected:
- BRAINOS_LAB_DB=:memory: (default for Vercel)
- BRAINOS_LAB_EVAL_ALLOW_GENERATION=0 (disables generation in eval tab)
- RESULTS_ROOT=/tmp/results (override to avoid read-only FS)

Usage (local):
    BRAINOS_LAB_DB=:memory: uvicorn app.vercel_app:fastapi_app --host 0.0.0.0 --port 7860

Vercel will auto-detect `app` variable from api/index.py which imports this.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on path when imported as api/index.py on Vercel
# Vercel's cwd is repo root, but we defend anyway.
_repo_root = Path(__file__).resolve().parents[2]
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Vercel defaults - must be set before importing storage
os.environ.setdefault("BRAINOS_LAB_DB", ":memory:")
# On Vercel, results should go to /tmp, not repo root (read-only)
if os.getenv("VERCEL") == "1" and "RESULTS_ROOT" not in os.environ:
    os.environ["RESULTS_ROOT"] = "/tmp/results"

# Now import app internals
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise RuntimeError(
        "FastAPI is required for Vercel deployment. "
        "Install with `pip install fastapi uvicorn` or add to requirements.txt"
    ) from exc

from app.controller import UIController, UILimits
from app.evaluation import EvaluationPolicy
from storage.sqlite import (
    SqliteConversationStore,
    SqliteEvaluationStore,
    SqliteMemoryStore,
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
    mode: str | None = None  # optional mode override per turn


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
    """Create a controller with Vercel-safe defaults.

    - :memory: SQLite stores (shared-cache URI handled by SqliteStore)
    - EvaluationPolicy tightened for serverless (quick only, no generation)
    - UILimits conservative (32k input, 4k output, 30s timeout)
    """
    # Detect Vercel env
    is_vercel = os.getenv("VERCEL") == "1"
    db_path = os.getenv("BRAINOS_LAB_DB", ":memory:")

    # For :memory:, SqliteStore uses shared-cache URI internally
    # For /tmp path, ensure parent exists
    if db_path != ":memory:":
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to :memory: if /tmp not writable for some reason
            db_path = ":memory:"

    # Evaluation policy from env (same as HF deployment.md)
    try:
        policy = EvaluationPolicy.from_environment()
    except ValueError:
        # If env vars malformed, fall back to safe restrictive policy
        policy = EvaluationPolicy(
            allowed_presets=("quick",),
            max_tasks=5,
            max_requests=15,
            allow_generation=False,
            results_retention_seconds=None,
        )

    # If on Vercel and generation not explicitly allowed, force off
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


# Singleton controller for serverless - note: not shared across Lambda instances
# This is a known limitation; for production multi-instance, replace with Redis
_controller = _make_controller()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(
    title="BrainOS Context Lab - Vercel API",
    description=(
        "BYOK chat with BrainOS memory. "
        "Ephemeral deployment: sessions live in memory only, "
        "evaluation disabled or retrieval-only. "
        "See /docs for OpenAPI."
    ),
    version="0.1.0-vercel",
)

# CORS for Next.js frontend on same Vercel project or external
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production via env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/api/health")
def health() -> dict[str, Any]:
    from storage.sqlite import persistence_enabled
    from checkout import REPO_ROOT

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

    # Optional mode switch per request (useful for A/B)
    if req.mode:
        try:
            _controller.apply_mode(req.session_id, req.mode)
        except Exception as exc:
            # Mode switch failure shouldn't break chat, just report
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
    state = _controller.ensure_session(session_id)
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
    return {"session_id": view.session_id, "history": view.history, "status": view.status}


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
    # Force retrieval-only on Vercel
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
# This is what api/index.py will use. If gradio is not installed or mounting
# fails, we still have the REST API above.

def _try_mount_gradio() -> Any:
    """Attempt to mount Gradio UI onto FastAPI for / route.

    Returns the FastAPI app (with Gradio mounted) or the bare FastAPI app.
    """
    try:
        import gradio as gr
        from app.ui import create_app as create_gradio_app

        # Reuse same controller so REST and UI share sessions (within same instance)
        demo = create_gradio_app(controller=_controller)

        # Gradio 6: mount_gradio_app
        # For serverless, disable queue threading that assumes long-lived process
        # We still call queue but with concurrency 1
        try:
            demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
        except Exception:
            pass

        # Mount at root - FastAPI routes under /api/* will still work because
        # Gradio mount is at / and FastAPI router precedence matters.
        # We mount Gradio at "/" but FastAPI's /api/* takes precedence if defined first.
        mounted = gr.mount_gradio_app(fastapi_app, demo, path="/")
        return mounted
    except Exception as exc:
        # Gradio not available or mount failed - return FastAPI only
        # Log to stderr so Vercel logs show it
        print(f"[vercel_app] Gradio mount failed, serving API only: {exc}", file=sys.stderr)
        return fastapi_app


# The app that Vercel will serve - try Gradio, fallback to API-only
app_with_gradio = _try_mount_gradio()

# For api/index.py to import
app = app_with_gradio

__all__ = ["fastapi_app", "app", "app_with_gradio", "_controller"]
