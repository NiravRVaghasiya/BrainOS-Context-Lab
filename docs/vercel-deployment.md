# Vercel Deployment Guide (Alternative to Hugging Face Spaces)

This guide shows how to deploy BrainOS Context Lab to Vercel instead of (or alongside) Hugging Face Spaces.

> **TL;DR:** Chat works on Vercel with `:memory:` DB and evaluation disabled. Full benchmark pipeline does NOT work on Vercel due to serverless timeouts and filesystem limits. See `VERCEL_DEPLOYMENT_ANALYSIS.md` for full trade-off.

## Why Vercel is Different

| Aspect | HF Spaces | Vercel |
|--------|-----------|--------|
| Compute | Long-lived container, `0.0.0.0:7860` | Serverless Lambda per request, 10-60s timeout |
| FS | Persistent `data/*.sqlite3`, `results/` | Read-only except `/tmp` (512MB ephemeral) |
| Sessions | In-memory dict survives across requests (single container) | Each Lambda isolated, dict not shared |
| Evaluation | Can run 70+ min jobs | Impossible in one function call |

## Option A: Minimal Lift (Chat-Only, 15 mins)

Best for: demo, landing page, portfolio.

### 1. Files added in this branch

- `api/index.py` - Vercel entry, sets `BRAINOS_LAB_DB=:memory:` and mounts Gradio via FastAPI
- `src/app/vercel_app.py` - FastAPI-native REST API (`/api/chat`, `/api/health`, etc.) + optional Gradio mount
- `vercel.json` - rewrites + function config
- `pyproject.toml` - `[tool.vercel] entrypoint = "api.index:app"` names the entrypoint, so the builder is told where the app lives instead of guessing between the two candidate files it finds

> **`app` has to be assigned at module scope.** Vercel finds the application by a
> *static* read of the entrypoint: it recognises a statement in the module body
> and nothing else, so an assignment inside `try:` / `except:` is invisible even
> though it is the first thing Python executes. That is why `api/index.py` ends
> with `app = _build_app()` and the error handling lives inside `_build_app()`.
> The object must also *be* a FastAPI application — the runtime half of the check
> imports the module and asks — which is why the degraded fallback is a real
> `FastAPI()` instance and not a bare ASGI function.

### 2. Deploy via CLI

```bash
npm i -g vercel
vercel --prod
# When prompted:
# - Set up and deploy? Y
# - Which scope? your-team
# - Link to existing project? N
# - Project name? brainos-context-lab
# - In which directory is your code located? ./
```

### 3. Env vars in Vercel dashboard (Project > Settings > Environment Variables)

```
BRAINOS_LAB_DB=:memory:
BRAINOS_LAB_EVAL_ALLOW_GENERATION=0
BRAINOS_LAB_EVAL_PRESETS=quick
BRAINOS_LAB_EVAL_MAX_TASKS=5
BRAINOS_LAB_EVAL_MAX_REQUESTS=15
BRAINOS_LAB_CONCURRENCY=1
BRAINOS_LAB_MAX_QUEUE=8
```

**Do not set `PYTHONPATH` in Vercel.** If you copied an earlier version of
this guide, delete that variable from Project → Settings → Environment
Variables (Production and Preview), then redeploy. Vercel manages the import
path for its bundled dependencies; `api/index.py` already prepends the absolute
`src/` directory without replacing that path.

Optional but recommended:

```
GRADIO_SERVER_NAME=0.0.0.0
RESULTS_ROOT=/tmp/results
```

### 4. Explicit, size-bounded Vercel dependencies

**Do not rely on manifest auto-detection.** The base `pyproject.toml` intentionally
has `dependencies = []`. Selecting it without the deployment requirements leaves
FastAPI out of the function. Conversely, installing the full HF requirements
produced a 295.30 MB bundle against this deployment's 225 MB function limit.

`vercel.json` explicitly configures:

```json
"installCommand": "uv pip install -r requirements-vercel.txt",
"buildCommand": "python scripts/prepare_vercel_bundle.py && python scripts/check_vercel_runtime.py"
```

The Vercel profile preserves FastAPI, Gradio, OpenAI, the pinned BrainOS runtime,
and Pandas/NumPy (required by Gradio). It omits optional matplotlib and its font
and rendering dependencies. **Chat and evaluation tables remain available;
figure rendering is disabled in this profile.** Full local/HF installs still use
`requirements.txt` and retain chart rendering.

The preparation script removes only three unused Gradio asset directories from
the disposable build virtual environment: the browser FFmpeg transcoder, Node
SSR bundle, and sample media. The app has no audio/video inputs and explicitly
disables SSR. Python modules, native libraries, and frontend JS/CSS are kept.
The script also updates Gradio's wheel RECORD so Vercel does not count or try
to copy deleted files. Gradio is pinned to the asset layout validated by the
script; version upgrades require revalidation.

The preparation step enforces a **210 MB dependency budget**, leaving room below
the reported 225 MB limit for app code and platform files. This is not a measure
of the final Vercel artifact. A clean Python 3.11 validation measured 200.58 MB
of dependencies after removing 48.08 MB of unused assets. Vercel's final bundle
check remains authoritative, and different Python/platform wheels can vary.

The post-trim smoke check requires dependency imports, ASGI startup, API health,
Gradio configuration, and the HTML page's JS/CSS assets to succeed. It does not
call a model or require credentials.

### 5. Test the deployment app locally

```bash
pip install -r requirements-vercel.txt
python scripts/check_vercel_runtime.py
BRAINOS_LAB_DB=:memory: BRAINOS_LAB_EVAL_ALLOW_GENERATION=0 python -m uvicorn api.index:app --host 0.0.0.0 --port 7860 --reload
# Open http://localhost:7860
# API docs at http://localhost:7860/docs
# Health at http://localhost:7860/api/health
```

### 6. What works / what doesn't in Option A

✅ Chat with BYOK key (held in server memory per session)
✅ Memory panels (stored, retrieved, dropped, conflicts)
✅ Context inspection (prompt, stats, trace)
✅ Security ledger
✅ Cost controls (per-request timeout 30s)
✅ Header says "fully in memory" (honest, via `persistence_enabled()`)
❌ Persistence across cold starts (session lost if Lambda recycles)
❌ Evaluation run (preview works, run times out after 60s)
❌ `results/ui/` artifacts not persisted (written to `/tmp`)
❌ Multi-user concurrency beyond 1 may see SQLite shared-cache issues

## Option B: Proper Migration (Full Parity, 2-3 weeks)

For full features on Vercel, you need:

### B1. Replace SQLite with Vercel Postgres / KV

The storage layer already uses protocols (`storage/conversations.py`). Implement:

```python
# src/storage/postgres.py
import psycopg2
class PostgresConversationStore:
    def __init__(self, dsn=os.getenv("POSTGRES_URL")): ...
    def append(self, message): ...
```

Then in `vercel_app.py`:

```python
if os.getenv("POSTGRES_URL"):
    from storage.postgres import PostgresConversationStore
    conv_store = PostgresConversationStore()
else:
    conv_store = SqliteConversationStore(":memory:")
```

### B2. Replace SessionManager with Redis

```python
# src/app/session_redis.py
import upstash_redis
class RedisSessionManager:
    def get(self, session_id): 
        data = redis.get(f"session:{session_id}")
        return SessionState.from_dict(json.loads(data)) if data else None
```

### B3. Evaluation artifacts to Vercel Blob

```python
from vercel_blob import put
# instead of Path(...).write_text(...)
url = put(f"results/ui/{session_id}/{run_id}/report.md", report_markdown).url
```

Long runs: offload to external worker (Modal, Fly.io, HF Space) that posts back.

### B4. UI: Gradio → Next.js

- Create `frontend/` Next.js app (App Router, Tailwind)
- Use Vercel AI SDK for streaming chat
- Call `/api/chat` REST endpoints
- Deploy frontend to Vercel, backend as Python functions (this repo)

Example structure:

```
frontend/
  app/
    page.tsx -> Chat + panels
    api/chat/route.ts -> proxies to Python backend
  components/
    MemoryPanel.tsx
    ContextPanel.tsx
```

This matches the architecture of `clipcoinagency/BrainOS` (Next.js + Supabase).

## Option C: Hybrid (Recommended)

Keep HF Space as canonical research deployment (it already supports `:memory:` mode, honest header, retention). Deploy Vercel as marketing frontend that links to HF for full eval.

```
your-domain.com (Vercel Next.js) -> chat demo (Vercel Python)
  "Run full benchmark" button -> huggingface.co/spaces/<you>/BrainOS-Context-Lab
```

## Troubleshooting Vercel

### `ModuleNotFoundError: No module named 'fastapi'`

If this appears both in `app.vercel_app` and in `_degraded_app`, the function
cannot import FastAPI at all. The fallback needs FastAPI too, so it cannot rescue
an incomplete dependency installation. This is not a Gradio queue or timeout
failure.

Deploy the revision containing the explicit install/build commands above, with
Vercel's Root Directory set to the repository root. Clear conflicting dashboard
Install/Build Command overrides and redeploy **without the existing build cache**.
The build logs must show `uv pip install -r requirements-vercel.txt` followed by
`Vercel runtime smoke check passed`. Redeploying the old commit alone does not
apply the fix. Then check `/api/health` and `/` in the new deployment.

### Build succeeds, but “This page is unavailable” / function temporarily failed

A successful build does not prove the function starts or serves a request.
Open the failing deployment's **Logs**, request `/api/health`, and inspect the
first Python exception (not just the final `FUNCTION_INVOCATION_FAILED` line).

1. Confirm the dependency install and smoke check above ran successfully.
   Remove any custom `PYTHONPATH` variable from Vercel project/team settings.
   The old configuration set it to `src`, which can replace the runtime's
   dependency search path. The checked-in config no longer overrides it.
   A failure before `api/index.py` loads cannot be handled by its fallback.
2. Deploy the updated revision. Environment-variable changes apply to new
   deployments, not the deployment already serving traffic.
3. Check **both** `/api/health` and `/`. Health should return JSON with
   `"status": "ok"`; `/` should return the Gradio HTML page. HTTP 200 alone is
   insufficient: the import fallback reports `"status": "degraded"`, and the
   API can start even if mounting Gradio fails.
4. If it still fails, use the runtime traceback to distinguish a missing module,
   a read-only filesystem write, a startup/lifespan exception, or a timeout.
   Do not increase memory or change dependencies blindly. Share the traceback
   with credentials redacted.

Local smoke check with the deployment dependencies installed:

```bash
VERCEL=1 GRADIO_ANALYTICS_ENABLED=False python - <<'PYTHON'
from fastapi.testclient import TestClient
from api.index import app

with TestClient(app) as client:
    health = client.get("/api/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok", health.text
    page = client.get("/")
    assert page.status_code == 200, page.text
    assert "text/html" in page.headers["content-type"]
PYTHON
```

This exercises ASGI startup and page serving locally, not Vercel's production
runtime. A passing smoke check does not replace checking deployment logs.

### Build fails: `Found app.py, api/index.py but none define a top-level "app" FastAPI instance`

The builder detected FastAPI (it is in `requirements.txt`) and then looked for the
application. It scans its default entrypoint locations — `app.py`, `index.py`,
`server.py`, `main.py`, `wsgi.py`, `asgi.py` at the project root or under `src/`,
`app/`, `api/` — for a *module-level* assignment to `app`, and this repository has
two of those files:

* `app.py` is the Hugging Face Space launcher. It defines no ASGI app, and it
  cannot: importing the UI at module scope closes an import cycle through
  `src/evaluation` (see the comment in that file).
* `api/index.py` is the Vercel entrypoint — the file that must define `app`.

Fix, all three parts of which are now in the repository:

1. `api/index.py` ends with a plain `app = _build_app()`. The `try` / `except`
   around the real import lives inside `_build_app()`, not in the module body.
2. `pyproject.toml` declares `[tool.vercel] entrypoint = "api.index:app"`, so the
   builder is pointed at that file rather than left to choose between candidates
   it cannot tell apart.
3. `tests/unit/test_vercel_entrypoint.py` reproduces the check — a static AST
   read plus an import that asserts `app` is a `FastAPI` instance — so a future
   edit that re-nests the assignment fails CI instead of the deployment.

If it still fails, confirm the value Vercel resolved: `vercel build` prints the
entrypoint, and `python -c "import api.index; print(type(api.index.app))"` says
what an import produces.

### Build fails on `brainos-cli @ git+...`

Vercel's build log shows `Failed to build`. Fix:

- Check the explicit install command ran and the build can reach the pinned Git revision. `packages.txt` is for HF Spaces, not Vercel.
- Or vendor: `pip wheel brainos-cli @ git+... -w wheels/` and commit wheel, then `requirements.txt` points to wheel file.

### Function bundle exceeds the reported 225 MB limit

Deploy the lean profile and preparation step in section 4, without the existing
build cache. Do not remove Pandas/NumPy or shared libraries: Gradio needs them.
Do not increase `memory` or `maxDuration`: they do not change the size cap.
`includeFiles: "src/**"` adds source files; it is not a dependency whitelist.

Expect build logs showing the removed asset bytes, the remaining dependency
size, and `Vercel runtime smoke check passed`. If the dependency budget fails,
inspect the installed package sizes and revalidate dependency versions. If only
the final Vercel check fails, inspect platform-added files and bundled repository
artifacts as well; the dependency budget intentionally is not a final-bundle claim.

### Gradio UI blank / WebSocket error

Vercel Functions don't support WebSockets well. Gradio's queue uses WebSockets.

Fixes:
- In `vercel_app.py`, we set `concurrency=1, max_size=8` and mount via FastAPI. If still blank, use API-only mode (FastAPI REST) and build Next.js frontend.
- Set `GRADIO_STRICT_CORS=0` env var to allow iframe embedding (Vercel preview).

### Sessions lost

Expected on serverless. Mitigations:
- Sticky sessions via Vercel's `x-vercel-id`? Not reliable.
- Implement Redis session store (Option B).
- For demo, warn user: "This deployment is ephemeral, sessions may reset on cold start" - header already says "fully in memory".

### Evaluation run 504 timeout

Expected. Vercel Hobby timeout 10s, Pro 60s. Evaluation needs minutes.

Fix: Disable generation (`ALLOW_GENERATION=0`), limit to 5 tasks, or offload to external worker.

## Cost Estimate

- Vercel Hobby: free, 100GB-hours, 10s timeout → chat-only ok, eval no
- Vercel Pro: $20/mo per seat, 60s timeout, 1024MB memory → chat ok, small eval preview ok
- Vercel + Postgres (Neon free tier) + KV (Upstash free) + Blob (free 100GB) → ~$0-20/mo for full parity
- HF Spaces CPU basic: free, sleeps after 48h, persistent disk → best for long eval jobs

## Checklist for Vercel Deploy

- [ ] `api/index.py` exists and exports `app` **at module scope** (`app = _build_app()`)
- [ ] `pyproject.toml` declares `[tool.vercel] entrypoint = "api.index:app"`
- [ ] `vercel.json` rewrites to `/api/index`
- [ ] `requirements-vercel.txt` is explicitly installed
- [ ] Bundle preparation and post-trim startup/asset checks pass
- [ ] Env vars set: `BRAINOS_LAB_DB=:memory:`, `EVAL_ALLOW_GENERATION=0`
- [ ] Test locally: `uvicorn api.index:app --port 7860`
- [ ] `vercel --prod` succeeds
- [ ] `/api/health` returns 200
- [ ] Chat works with BYOK key
- [ ] Header says "fully in memory" (honest)
- [ ] Evaluation tab shows preview but run button disabled or times out gracefully
- [ ] No secrets in logs (check `python -m security.scan` still clean)

## References

- Vercel Python runtime: https://vercel.com/docs/functions/runtimes/python
- Gradio FastAPI mount: https://www.gradio.app/guides/fastapi-app-with-the-gradio-client
- Vercel Postgres: https://vercel.com/docs/storage/vercel-postgres
- Vercel KV: https://vercel.com/docs/storage/vercel-kv
- Vercel Blob: https://vercel.com/docs/storage/vercel-blob
- This repo's HF deployment: `docs/deployment.md`
- Full analysis: `VERCEL_DEPLOYMENT_ANALYSIS.md`
