# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a voice AI pipeline project with a WebSocket-based ASR (Automatic Speech Recognition) service. Currently a minimal viable product (MVP) with a mock ASR engine.

## Running the Server

```bash
python asr_server.py
```

This starts the FastAPI server on `http://0.0.0.0:8000` with hot reload enabled.

## Architecture

- **asr_server.py**: FastAPI WebSocket server exposing `/ws/asr` endpoint
  - Accepts WebSocket connections with JSON config messages and binary audio frames
  - Returns partial and final ASR results via WebSocket
  - Includes telemetry (latency tracking) in responses
  - Currently uses a mock ASR engine (placeholder for Qwen3-ASR)

- **litellm_config.yaml**: LiteLLM configuration for unified LLM API routing

## WebSocket Protocol

Clients must send:
1. A `config` message first (JSON with `type: "config"`)
2. Binary audio frames after config

Server returns:
- `asr_result` messages with `is_final: false` (partial results)
- `asr_result` messages with `is_final: true` (final results)

## Future Integrations (Not Yet Implemented)

- Qwen3-ASR inference engine (replacing mock)
- VAD module (Voice Activity Detection - WebRTC or Silero)
- SER module (Speech Emotion Recognition)
