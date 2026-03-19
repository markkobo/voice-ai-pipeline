# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice AI pipeline with WebSocket-based ASR (Automatic Speech Recognition) service. Modular architecture with VAD (Voice Activity Detection) and Qwen3-ASR integration.

## Setup & Running

```bash
# Clone the repo
git clone https://github.com/markkobo/voice-ai-pipeline.git
cd voice-ai-pipeline

# Install dependencies
pip install -r requirements.txt

# Run with Qwen3-ASR (default, requires model download)
python -m app.main

# Or run with MockASR for testing
USE_QWEN_ASR=false python -m app.main
```

Server runs on `http://0.0.0.0:8000` with hot reload.

## Architecture

```
app/
├── main.py                 # FastAPI app setup
├── api/
│   └── ws_asr.py          # WebSocket /ws/asr endpoint
├── core/
│   └── state_manager.py   # Session state, audio buffers, utterance tracking
└── services/
    ├── vad_engine.py       # BaseVAD + EnergyVAD (RMS-based speech detection)
    └── asr_engine.py      # BaseASR + Qwen3ASR + MockASR
tests/
├── conftest.py            # pytest fixtures
└── test_ws_asr.py         # Unit and integration tests
```

## Protocol

**Client sends (Text frames - JSON):**
- `{"type": "config", "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"}}`
- `{"type": "control", "action": "commit_utterance"}`

**Client sends (Binary frames):** PCM 16-bit audio chunks

**Server returns:**
- `{"type": "asr_result", "utterance_id": "...", "is_final": false, "text": "...", ...}` (partial)
- `{"type": "asr_result", "utterance_id": "...", "is_final": true, "text": "...", "extensions": {"emotion": {...}}, "telemetry": {...}}` (final)

## Testing

```bash
# Run tests (uses MockASR)
USE_QWEN_ASR=false pytest tests/ -v

# Run specific test
USE_QWEN_ASR=false pytest tests/test_ws_asr.py::TestWebSocketIntegration -v
```
