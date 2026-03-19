"""FastAPI application setup."""
from fastapi import FastAPI
from app.api import ws_asr

# Import telemetry - start metrics server
from telemetry import TelemetryCollector

# Start Prometheus metrics server on port 9090
telemetry_collector = TelemetryCollector(port=9090, enable_at_start=True)

app = FastAPI(title="Voice AI Pipeline ASR Service", version="1.0.0")

# Include WebSocket router
app.include_router(ws_asr.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
