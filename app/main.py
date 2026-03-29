"""FastAPI application setup with Gradio UI, WebSocket, and TTS streaming."""
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))  # Load .env file for API keys

# Setup structured JSON logging FIRST
from app.logging_config import setup_json_logging

logger = setup_json_logging()
logger.info("Starting Voice AI Pipeline")

# Now import the rest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ws_asr, tts_stream, standalone_ui, recordings, reference_data, recordings_ui, training
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
app.include_router(standalone_ui.router)
app.include_router(recordings.router)
app.include_router(reference_data.router)
app.include_router(recordings_ui.router)
app.include_router(training.router)

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
        logger.info("Warming up TTS model (CUDA graphs)...")
        loop = asyncio.get_event_loop()
        # warmup is sync, run in executor to not block
        await loop.run_in_executor(None, engine.warmup)
        logger.info("TTS model warmed up")

    # Wait for ASR preload to complete before accepting connections
    # This prevents race condition where request comes in before model is loaded
    await preload_asr()
    # Also preload and warmup TTS
    await preload_tts()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
