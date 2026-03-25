"""FastAPI application setup with Gradio UI, WebSocket, and TTS streaming."""
import os

# Setup structured JSON logging FIRST
from app.logging_config import setup_json_logging

logger = setup_json_logging()
logger.info("Starting Voice AI Pipeline")

# Now import the rest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ws_asr, tts_stream
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

# Include routers
app.include_router(ws_asr.router)
app.include_router(tts_stream.router)


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
# Gradio UI Mounting
# ============================================================================

def mount_gradio_app(app: FastAPI):
    """
    Mount Gradio UI at /ui.

    This is called after app creation to avoid circular imports.
    """
    try:
        from gradio_ui import build_ui

        ui = build_ui()
        app = ui.mount(app, path="/ui")
        logger.info("Gradio UI mounted at /ui")
        return app
    except ImportError as e:
        logger.warning(f"Gradio not available: {e}")
        return app


# Mount Gradio on import (after all routes are registered)
# This is done lazily to avoid import issues during testing
_gradio_mounted = False


@app.on_event("startup")
async def startup_event():
    """Mount Gradio on startup."""
    global _gradio_mounted
    if not _gradio_mounted:
        try:
            import gradio
            # Lazy import to avoid top-level gradio import
            from app.api.gradio_ui import build_ui
            ui = build_ui()
            app.mount("/ui", ui)
            _gradio_mounted = True
            logger.info("Gradio UI mounted at /ui")
        except ImportError as e:
            logger.warning(f"Gradio not installed: {e}")
            logger.warning("Install with: pip install gradio")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
