"""FastAPI application setup with Gradio UI, WebSocket, and TTS streaming."""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))  # Load .env file for API keys

# Setup structured JSON logging FIRST
from app.logging_config import setup_json_logging

logger = setup_json_logging()
logger.info("Starting Voice AI Pipeline")

# Now import the rest
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import ws_asr, tts_stream, standalone_ui, recordings, recordings_ui, training, training_ui, personas, listeners
from app.api import _system, corpus
from app.api._errors import register_error_handlers

# Jinja2 + StaticFiles wiring (RFC_M6 Phase 0-pre — dev-UI refactor).
# `auto_reload=True` so HTML edits show up without a server restart in dev.
# Jinja2's bytecode cache handles invalidation by mtime when auto_reload
# is on — don't replace `env.cache`, that breaks the cache's `.get(key)`
# protocol.
_app_root = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_app_root / "templates"))
templates.env.auto_reload = True
# gradio_ui imported lazily below to handle missing gradio

# Import telemetry collector
from telemetry import TelemetryCollector

# Start Prometheus metrics server on port 9090
telemetry_collector = TelemetryCollector(port=9090, enable_at_start=True)

# Create FastAPI app
app = FastAPI(
    title="Voice AI Pipeline",
    version="1.0.0",
    description="Personal Legacy AI — Voice streaming with ASR + LLM + TTS",
)

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Force browsers to revalidate /static/* on every request.

    StaticFiles emits Last-Modified + ETag but no Cache-Control, so
    browsers heuristically cache by mtime — which means after every JS
    deploy, users see stale code until they hard-refresh manually.
    Bit user 2026-05-20 (stuck FSM in OLD CONNECTING path despite
    server having the new code).

    Demo-safe fix: send `Cache-Control: no-cache` for /static/* so the
    browser always sends If-Modified-Since and we serve fresh bytes.
    Long-term we should switch to content-hashed asset filenames +
    immutable cache (see 00df3a5 review #5).
    """
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# Mount static assets (RFC_M6 Phase 0-pre). All dev-UI CSS + JS lives at
# `app/static/`; access via `/static/...`.
app.mount(
    "/static",
    StaticFiles(directory=str(_app_root / "static")),
    name="static",
)

# Expose templates so the *_ui.py modules can resolve them via app.state.
app.state.templates = templates

# Include routers
app.include_router(ws_asr.router)
app.include_router(tts_stream.router)
app.include_router(standalone_ui.router)
app.include_router(recordings.router)
app.include_router(recordings_ui.router)
app.include_router(training.router)
app.include_router(training_ui.router)
app.include_router(personas.router)
app.include_router(listeners.router)
app.include_router(_system.router)
app.include_router(corpus.router)

# Wire DomainError → HTTP handlers (single source of truth for error responses).
register_error_handlers(app)

print(f"[DEBUG] Routers included. Total routes: {len(app.routes)}")
for r in app.routes:
    if hasattr(r, 'path'):
        print(f"  {r.path}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "asr": os.getenv("USE_QWEN_ASR", "true").lower() == "true",
            "llm": os.getenv("USE_MOCK_LLM", "false").lower() != "true",
            "tts": "available",
        }
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Voice AI Pipeline API",
        "docs": "/docs",
        "ui": "/ui",
        "metrics": ":9090/metrics",
    }


def create_app() -> FastAPI:
    """Factory for creating the app (used by uvicorn and Gradio)."""
    return app


# ============================================================================
# Standalone UI is mounted via standalone_ui router (at /ui)
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event — preload ASR model synchronously before accepting connections."""
    import asyncio

    async def preload_asr():
        # Get the ASR engine from state_manager (created at module import)
        from app.api.ws_asr import state_manager
        asr_engine = state_manager._default_asr

        # Only preload if it's Qwen3ASR (not MockASR)
        if asr_engine.__class__.__name__ == "Qwen3ASR" and asr_engine._model is None:
            logger.info("Preloading Qwen3-ASR model (first load ~30s)...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, asr_engine.load_model)
            logger.info("Qwen3-ASR model preloaded successfully")
        else:
            logger.info(f"ASR engine already loaded or using {asr_engine.__class__.__name__}")

    async def preload_tts():
        from app.services.tts import get_tts_engine
        engine = get_tts_engine()

        # Auto-activate latest merged model BEFORE warmup to avoid loading base model twice
        try:
            from app.services.training import get_version_manager
            vm = get_version_manager()
            ready = [v for v in vm.list_versions() if v.status == "ready"]
            if ready:
                # Sort by created_at timestamp (newest first), not lexicographic version_id
                ready.sort(key=lambda v: v.created_at or "", reverse=True)
                latest = ready[0]
                logger.info(f"Auto-activating latest merged model: {latest.version_id}")
                engine.activate_version(latest.version_id)
            else:
                logger.info("No ready merged models found, using base VoiceDesign")
        except Exception as e:
            logger.warning(f"Could not auto-activate merged model: {e}")

        # Now warmup — will load whichever model is active (merged or base)
        logger.info("Warming up TTS model (CUDA graphs)...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.warmup)
        logger.info("TTS model warmed up")

    # Wait for ASR preload to complete before accepting connections
    # This prevents race condition where request comes in before model is loaded
    await preload_asr()
    # Also preload and warmup TTS
    await preload_tts()

    # Reset any corpus items left in `ingesting` state from a previous
    # crash to `failed` so they don't stay stuck (RFC_M6 Phase 0 task 62C
    # / review #8 of c7ee1f4).
    try:
        from app.api._dependencies import _get_or_create_ingestion_service
        from app import config as _cfg
        svc = _get_or_create_ingestion_service(app.state)
        n = svc.sweep_stranded_all(_cfg.personas_dir())
        if n:
            logger.warning(f"Reset {n} stranded ingesting items on startup")
    except Exception as e:
        logger.exception(f"Corpus sweep_stranded_all failed: {e}")

    # Reset any recordings left in `processing` state from a previous
    # crash to `failed` so the "處理中" UI doesn't show ghost rows.
    # Mirrors the corpus sweep above — recordings processing is a
    # BackgroundTask, so a server kill mid-job orphans the metadata.
    try:
        from app.api._dependencies import _get_or_create_recordings_service
        rec_svc = _get_or_create_recordings_service(app.state)
        n = rec_svc.sweep_stranded()
        if n:
            logger.warning(f"Reset {n} stranded processing recordings on startup")
    except Exception as e:
        logger.exception(f"Recordings sweep_stranded failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
