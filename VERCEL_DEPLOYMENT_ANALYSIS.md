# Vercel Deployment Analysis for BrainOS Context Lab

**Date:** 2026-09-22
**Repo:** NiravRVaghasiya/BrainOS-Context-Lab
**Current Deployment:** Hugging Face Spaces (Gradio SDK, `app.py`, `requirements.txt`, `packages.txt`)
**Target:** Vercel

## Executive Summary

**Can it be deployed on Vercel? Yes, but not as a drop-in replacement.** The current app is designed for a long-lived Python server with local SQLite persistence and filesystem writes. Vercel's serverless model (ephemeral filesystem, 10-60s function timeout, no WebSocket persistence) conflicts with 4 core assumptions. A **minimal lift** is possible for the chat-only path, but the full evaluation pipeline requires either external storage or a hybrid architecture.

**Recommendation:** 
- **Short term:** Keep HF Space for full evaluation (benchmark) workloads; deploy a **chat-only** variant to Vercel using `:memory:` DB and `/tmp` for artifacts, with Evaluation tab disabled.
- **Long term:** Split into **Next.js frontend (Vercel) + FastAPI backend (Vercel Functions with Postgres/KV/Blob)** if you want full parity.

---

## 1. Current Architecture (Hugging Face)

```
app.py
  -> src/app/ui.py :: create_app()
     -> UIController (session manager, provider factory, budget)
        -> ConversationService (BrainOS adapter + context builder + provider)
           -> SqliteConversationStore / SqliteMemoryStore / SqliteEvaluationStore
              -> data/brainos_lab.sqlite3 (WAL, secure_delete, vacuum)
     -> UIEvaluationRunner -> evaluation.pipeline.run_pipeline
        -> results/raw/, aggregated/, plots/, report/, results/ui/<session>/<run>/
```

Key HF-specific files:
- `app.py`: bootstraps `src/` onto sys.path, calls `app.ui.main()` which does `demo.queue(concurrency=3, max_size=32).launch(server_name=0.0.0.0, port=7860)`
- `requirements.txt`: flat list including `gradio>=6.0`, `openai>=1.0`, `brainos-cli @ git+...`, `matplotlib`, `pandas`, `-e .`
- `packages.txt`: empty placeholder (pure Python)
- `src/storage/sqlite.py`: WAL + busy_timeout 30s, `MEMORY_DATABASE_SENTINEL=":memory:"` for stateless mode, `persistence_enabled()` header disclosure
- `src/app/evaluation.py`: `EvaluationPolicy.from_environment()` reads 5 `BRAINOS_LAB_EVAL_*` vars that can only tighten ceilings

Deployment posture docs (`docs/deployment.md`):
- Public Space: `BRAINOS_LAB_DB=":memory:"`, concurrency 2, queue 16, `EVAL_PRESETS=quick`, `MAX_TASKS=20`, `ALLOW_GENERATION=0`, `RETENTION_DAYS=0`
- Research: defaults, all presets, generation on, 7-day retention

## 2. Vercel Platform Constraints

| Concern | Hugging Face Spaces | Vercel |
|---------|---------------------|--------|
| **Compute** | Long-lived Docker container, 0.0.0.0:7860 persistent | Serverless Functions (AWS Lambda), per-request, 10s (Hobby) / 60s (Pro) / 300s (Enterprise) |
| **Filesystem** | Persistent volume, `data/*.sqlite3` + `results/` survive restarts | Read-only except `/tmp` (512MB ephemeral, cleared per invocation) |
| **WebSockets / Queue** | Gradio queue with 3 workers, WebSocket streaming | No native WebSocket persistence across invocations; Gradio queue will be recreated per cold start |
| **Concurrency** | Thread-based, 8 threads tested in `test_sqlite_concurrency.py` | Each function instance isolated; in-memory `SessionManager` not shared across instances |
| **Git deps** | `brainos-cli @ git+https://...` installed at build time | Supported but increases build time & cold start; private repos need token |
| **Size limit** | No strict limit (HF CPU basic) | 250MB unzipped function size, 50MB compressed |
| **Background jobs** | `evaluation.pipeline` can run 70+ mins (standard tier 56 tasks x 5 modes) | Impossible in single function invocation; needs external queue |
| **DB** | SQLite file with WAL, vacuum, secure_delete | SQLite won't persist; need Vercel Postgres / Neon / Upstash Redis / KV |

## 3. Compatibility Matrix

| Feature | Works on Vercel as-is? | Fix |
|---------|------------------------|-----|
| **Chat UI (Gradio)** | ⚠️ Partial - Gradio 6 can mount as FastAPI ASGI app, but queue & websockets flaky | Use `gr.mount_gradio_app` inside FastAPI, disable queue or set `BRAINOS_LAB_DB=:memory:` |
| **BYOK provider** | ✅ Yes - key held in server memory per session, same model works | Need sticky sessions or external session store (Redis) because memory not shared |
| **SQLite persistence** | ❌ No - file lost | Set `BRAINOS_LAB_DB=:memory:` for ephemeral demo, or migrate to Vercel Postgres via new `PostgresConversationStore` |
| **Evaluation pipeline** | ❌ No - writes to `results/`, long runtime, matplotlib | Disable via `BRAINOS_LAB_EVAL_ALLOW_GENERATION=0` + `PRESETS=quick` or move to external worker; use Vercel Blob for artifacts |
| **Security ledger / panels** | ✅ Yes - in-memory, rendered same | Works if session stays in same instance |
| **Cost controls** | ✅ Yes | Same |
| **BrainOS runtime** | ⚠️ Risky - `brainos-cli` from git, may exceed size, needs compilation | Pre-build wheel or vendor; check if BrainOS has native deps |
| **Concurrency tests** | ❌ Fail - WAL mode irrelevant on serverless | Replace with distributed lock or Postgres |

## 4. What Breaks Specifically

1. **SessionManager is in-memory dict** (`src/app/session.py`). On Vercel, two requests may hit different Lambda instances → session lost. HF's single container keeps dict.
2. **SQLite WAL + vacuum** (`storage/sqlite.py`): `vacuum()` tries to open file path, checkpoint WAL. On Vercel `/tmp` only, and shared-cache `:memory:` URI (`file:brainos_lab_shared?mode=memory&cache=shared`) won't be shared across Lambdas.
3. **Results artifacts**: `UIEvaluationRunner` writes to `results/ui/<session>/<run>/` resolved via `checkout.repo_anchored`. On Vercel, `REPO_ROOT` is read-only; must write to `/tmp/results/ui/...`. Also `retention.py` sweeps based on file age - needs cron, not request-triggered.
4. **Gradio `demo.launch()`**: Binds to 0.0.0.0:7860, starts queue. Vercel expects ASGI `app` object, not a blocking launch. `main()` in `ui.py` must be bypassed.
5. **Long evaluation**: `run_pipeline` does experiment → raw → comparison → statistics → errors → plots → report. Standard preset = 56 tasks × 3 modes × 1 trial = 168 LLM calls minimum; research = 500 tasks × 7.5k requests ceiling. Even with 300s timeout, will hit ceiling.
6. **Matplotlib**: Needs writable `MPLCONFIGDIR`, font cache. Works but adds size.

## 5. Minimal Lift Path (Chat-Only Vercel Deploy)

Goal: Get chat working on Vercel in <30 mins, no evaluation.

**Steps:**

1. Create `api/index.py` ASGI entry:
```python
import os, sys
from pathlib import Path
os.environ.setdefault("BRAINOS_LAB_DB", ":memory:")
os.environ.setdefault("BRAINOS_LAB_EVAL_ALLOW_GENERATION", "0")
os.environ.setdefault("BRAINOS_LAB_EVAL_PRESETS", "quick")
os.environ.setdefault("BRAINOS_LAB_EVAL_MAX_TASKS", "5")

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi import FastAPI
import gradio as gr
from app.ui import create_app
from storage.sqlite import SqliteConversationStore, SqliteMemoryStore, SqliteEvaluationStore
from app.controller import UIController
from app.evaluation import EvaluationPolicy

fastapi_app = FastAPI()

# Use :memory: stores
controller = UIController(
  conversation_store=SqliteConversationStore(":memory:"),
  memory_store=SqliteMemoryStore(":memory:"),
  evaluation_store=SqliteEvaluationStore(":memory:"),
  evaluation_policy=EvaluationPolicy(allowed_presets=("quick",), max_tasks=5, max_requests=15, allow_generation=False)
)
demo = create_app(controller=controller)
# Gradio 6 mount
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
```

2. `vercel.json`:
```json
{
  "rewrites": [{"source": "/(.*)", "destination": "/api/index"}],
  "functions": {
    "api/index.py": {
      "maxDuration": 60
    }
  }
}
```

3. Env vars in Vercel dashboard:
```
BRAINOS_LAB_DB=:memory:
BRAINOS_LAB_EVAL_ALLOW_GENERATION=0
BRAINOS_LAB_CONCURRENCY=1
GRADIO_SERVER_NAME=0.0.0.0
PYTHONPATH=src
```

4. `requirements.txt` for Vercel: same as HF but add `fastapi`, `uvicorn`. Ensure `gradio>=6.0` compatible with `fastapi`.

**Limitations of minimal lift:**
- Sessions lost on cold start / different instance (no sticky)
- No persistence (header will say "fully in memory" - honest)
- Evaluation tab shows preview only, run button will fail after 60s
- Concurrency 1 to avoid SQLite :memory: sharing issues
- Cold start 5-10s due to BrainOS git install

**Tested?** Not in this repo yet, but pattern is documented by Gradio: https://www.gradio.app/guides/fastapi-app-with-the-gradio-client

## 6. Proper Migration Path (Full Parity on Vercel)

If you want full features on Vercel, you need to refactor storage and UI:

### 6a. Storage Abstraction
Current: `SqliteConversationStore`, `SqliteMemoryStore`, `SqliteEvaluationStore` implement protocols from `storage/conversations.py`, `storage/evaluations.py`.
New: Implement `PostgresConversationStore` using `psycopg2` + Vercel Postgres (Neon), or `VercelKVStore` using `@vercel/kv` REST.

The protocols are already abstract, so you can swap:
```python
# src/storage/postgres.py
class PostgresConversationStore:
    def append(self, message): ...
    def list_messages(self, session_id, conversation_id): ...
```

Env: `BRAINOS_LAB_DB` becomes `POSTGRES_URL` or `KV_REST_API_URL`.

### 6b. Session Management
Replace in-memory `SessionManager` with Redis-backed:
- Store `SessionState` as JSON in Upstash Redis with TTL 1h
- Or use Vercel's built-in session via cookies + JWT

### 6c. Evaluation Artifacts
- Replace `results/` filesystem writes with Vercel Blob (`@vercel/blob`): `blob.put(path, data)`
- Plots: generate matplotlib to BytesIO, upload to Blob, return URL
- Report: markdown stored in Blob, rendered via Next.js
- Long runs: offload to Vercel Cron + Queue or external worker (e.g., Modal, Fly.io) that posts back results

### 6d. UI: Gradio → Next.js
Gradio is not idiomatic for Vercel. Recommended:
- Frontend: Next.js 14 App Router (TypeScript, Tailwind, shadcn/ui) - matches Vercel strengths
- Backend: FastAPI routes `/api/chat`, `/api/memory`, `/api/context`, `/api/evaluation/preview`, `/api/evaluation/run`
- Keep `UIController` and `ConversationService` as backend logic, but expose via REST
- Chat: Use Vercel AI SDK (`ai` npm package) for streaming
- Panels: React components fetching `/api/inspection?session_id=...`

This is what `clipcoinagency/BrainOS` template does (Next.js + Supabase + TanStack Query).

### 6e. BrainOS Dependency
- Check BrainOS size: `pip show brainos-cli` dependencies. If it pulls torch/transformers, it will exceed 250MB.
- Solution: Vendor as separate microservice on Fly.io / Modal / HF Inference Endpoint, call via HTTP.
- Or keep BrainOS in Vercel function but use `vercel --prod` with `includeFiles` to prune.

## 7. Cost & Performance Comparison

| Metric | HF Spaces (CPU basic) | Vercel Hobby | Vercel Pro |
|--------|----------------------|--------------|------------|
| **Cost** | Free, sleeps after 48h inactivity | Free 100GB-hours, $20/mo Pro | $20/seat + usage |
| **Cold start** | ~30s (git install) | 5-15s (Python) | Same but more memory |
| **Persistence** | File persists until Space rebuild | None (needs external DB $) |
| **Evaluation** | Can run 70min+ jobs | Must be external |
| **Custom domain** | Yes (paid) | Yes (free) |
| **BYOK safety** | Server memory only, honest header | Same, but need Redis encryption |
| **Scaling** | Single container, queue depth 32 | Auto-scale to many Lambdas (session issue) |

## 8. Recommendation Decision Tree

```
Do you need Evaluation pipeline (benchmark) on Vercel?
├─ No, only chat demo → Minimal lift (api/index.py + :memory: + no eval) ✅ 1 day
├─ Yes, but retrieval-only (no LLM calls) → Minimal lift + Vercel Blob for artifacts, maxDuration 60s, quick preset only ⚠️ 2-3 days
└─ Yes, full with generation → Full migration (Next.js + Postgres + external worker) ❌ 2-3 weeks

Do you need persistence across restarts?
├─ No → :memory: is fine, header already handles it
└─ Yes → Must implement Postgres/KV store, cannot use SQLite

Do you care about session isolation across instances?
├─ No (single user demo) → In-memory SessionManager ok
└─ Yes (multi-user public) → Need Redis-backed sessions
```

## 9. Provided Artifacts in This Branch

This branch adds:

- `api/index.py`: Vercel serverless entry that mounts Gradio as FastAPI with :memory: stores and evaluation disabled
- `src/app/vercel_app.py`: FastAPI-native alternative exposing REST endpoints (chat, memory, context, health) without Gradio, suitable for Next.js frontend
- `vercel.json`: Rewrite + function config (maxDuration 60, memory 1024)
- `docs/vercel-deployment.md`: Operator guide for Vercel (env vars, limitations)

To test locally:
```bash
pip install fastapi uvicorn
BRAINOS_LAB_DB=:memory: BRAINOS_LAB_EVAL_ALLOW_GENERATION=0 python -m uvicorn api.index:app --host 0.0.0.0 --port 7860
```

To deploy:
```bash
vercel --prod --env BRAINOS_LAB_DB=:memory: --env BRAINOS_LAB_EVAL_ALLOW_GENERATION=0
# Or via dashboard: import repo, set env vars, deploy
```

## 10. Final Verdict

**Can we deploy on Vercel instead of Hugging Face?**

- **Technically yes** for chat-only, ephemeral demo.
- **Practically no** for full research platform as-is, because Vercel is serverless and the app is stateful/long-running.

**Best hybrid:** Keep HF Space as canonical research deployment (it already has `BRAINOS_LAB_DB=:memory:` support, honest header, retention). Deploy lightweight chat-only mirror to Vercel for marketing / landing page, with link back to HF for full evaluation.

If you want Vercel to be primary, budget 2-3 weeks for storage + UI rewrite.

---
*Generated by analysis of app.py, src/app/ui.py, controller.py, service.py, storage/sqlite.py, evaluation.py, deployment.md, pyproject.toml, requirements.txt, and Vercel docs.*
