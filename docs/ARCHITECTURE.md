# Voice AI Pipeline — Architecture Documentation

## Overview

A real-time voice AI pipeline that enables natural conversation with a persona-aware AI assistant. The system processes voice input through a series of streaming services, generates responses using an LLM with emotion tagging, and synthesizes speech using a TTS engine.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Browser (Client)                                │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  MediaRecorder   │───►│  WebSocket      │───►│  HTTP Audio Playback    │  │
│  │  (WebM/Opus)    │    │  (text/ctrl)    │    │  (AudioContext +       │  │
│  │  mic capture     │    │                 │    │   <audio> element)      │  │
│  └─────────────────┘    └────────┬────────┘    └─────────────────────────┘  │
│                                  │                                           │
│                                  │ JSON: asr_result, llm_token,            │
│                                  │      tts_ready, etc.                     │
│                                  │                                           │
│                                  ▼                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                                     │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                         WebSocket /ws/asr                              │    │
│  │  1. Receives WebM binary audio from client                           │    │
│  │  2. Decodes WebM → PCM (pydub)                                       │    │
│  │  3. Passes to VAD for speech detection                               │    │
│  │  4. On VAD commit: sends to ASR for transcription                  │    │
│  │  5. Passes text to LLM streaming                                    │    │
│  │  6. Parses emotion tags from LLM output                             │    │
│  │  7. Sends tts_ready with HTTP stream URL to client                  │    │
│  └───────────────────────────────┬──────────────────────────────────────┘    │
│                                  │                                            │
│                                  ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                      Pipeline Services                                │    │
│  │                                                                       │    │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │    │
│  │   │    VAD     │───►│    ASR      │───►│         LLM            │  │    │
│  │   │ (Energy)   │    │ (Qwen3-ASR │    │  (OpenAI streaming)    │  │    │
│  │   │            │    │  or Mock)   │    │  + emotion parsing     │  │    │
│  │   └─────────────┘    └─────────────┘    └───────────┬─────────────┘  │    │
│  │                                                       │               │    │
│  │                                                       ▼               │    │
│  │                                          ┌─────────────────────────┐  │    │
│  │                                          │   EmotionMapper         │  │    │
│  │                                          │  [情感: 撒嬌] →        │  │    │
│  │                                          │  instruct string       │  │    │
│  │                                          └───────────┬─────────────┘  │    │
│  │                                                      │               │    │
│  └──────────────────────────────────────────────────────┼───────────────┘    │
│                                                          │                    │
│                                                          ▼                    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     TTS Service (HTTP /api/tts/stream)              │    │
│  │                                                                       │    │
│  │   ┌─────────────────────────┐    ┌─────────────────────────────────┐ │    │
│  │   │   Faster-Qwen3-TTS     │───►│  PCM 24kHz mono streaming       │ │    │
│  │   │   (VoiceDesign mode)   │    │  (HTTP chunked transfer)       │ │    │
│  │   │   + emotion instruct   │    │                                 │ │    │
│  │   └─────────────────────────┘    └─────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     Supporting Services                              │    │
│  │                                                                       │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌────────────────────────────┐  │    │
│  │   │StateManager │  │PromptManager│  │  TelemetryCollector       │  │    │
│  │   │ (sessions,  │  │ (persona + │  │  (Prometheus metrics +    │  │    │
│  │   │  tasks)     │  │  listener) │  │   Grafana dashboard)     │  │    │
│  │   └─────────────┘  └─────────────┘  └────────────────────────────┘  │    │
│  │                                                                       │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐│    │
│  │   │  Logging: JSON structured logs → /logs/app.log                 ││    │
│  │   └─────────────────────────────────────────────────────────────────┘│    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Protocol Specification

### WebSocket Messages (Client → Server)

#### Config Message
```json
{
  "type": "config",
  "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
  "persona_id": "xiao_s",
  "listener_id": "child",
  "model": "gpt-4o-mini"
}
```

#### Control Message
```json
{"type": "control", "action": "commit_utterance"}
{"type": "control", "action": "cancel"}
```

#### Binary Message
- Raw WebM/Opus audio bytes from MediaRecorder
- Default chunk interval: 100ms

---

### WebSocket Messages (Server → Client)

| Message Type | Fields | Description |
|-------------|--------|-------------|
| `asr_result` | `utterance_id`, `is_final`, `text`, `telemetry` | ASR transcription result |
| `vad_commit` | `utterance_id`, `energy`, `telemetry` | VAD detected end of speech |
| `llm_start` | `utterance_id` | LLM stream started |
| `llm_token` | `content`, `emotion` | LLM token (emotion when first detected) |
| `tts_ready` | `text`, `emotion`, `instruct`, `stream_url` | TTS stream URL for client to fetch |
| `llm_done` | `text`, `total_tokens`, `telemetry` | LLM stream complete |
| `llm_cancelled` | `partial_text` | LLM interrupted by new speech |
| `llm_error` | `error` | LLM error |

---

### HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/tts/stream` | Stream TTS audio (query params: `text`, `emotion`, `model`) |
| `POST` | `/api/tts/session` | Create named TTS session |
| `GET` | `/api/tts/stream/{session_id}` | Stream TTS for named session |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

---

## Data Flow — Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Step 1: Audio Capture                             │
│                                                                             │
│  User presses mic button                                                    │
│       │                                                                     │
│       ▼                                                                     │
│  navigator.mediaDevices.getUserMedia({audio: {sampleRate: 24000}})          │
│       │                                                                     │
│       ▼                                                                     │
│  MediaRecorder records in WebM/Opus format, 100ms chunks                   │
│       │                                                                     │
│       ▼                                                                     │
│  WebSocket.send(binary WebM bytes)                                          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 2: Server Processing                             │
│                                                                             │
│  WebSocket receives binary                                                   │
│       │                                                                     │
│       ▼                                                                     │
│  decode_webm_to_pcm(webm_bytes, 24000)  [pydub]                           │
│       │                                                                     │
│       ▼                                                                     │
│  VAD.detect(pcm_bytes)  — Energy-based detection                           │
│       │                                                                     │
│       ├─── Speech detected ──► accumulate audio, continue recording          │
│       │                                                                     │
│       └─── Silence after speech ──► VAD commits                            │
│                                      │                                      │
│                                      ▼                                      │
│  ASR.recognize(pcm_bytes)  [Qwen3-ASR or MockASR]                         │
│       │                                                                     │
│       ▼                                                                     │
│  Returns: {"text": "...", "asr_inference_ms": 120}                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 3: LLM Streaming                                │
│                                                                             │
│  PromptManager.get_prompt(persona_id, listener_id)                          │
│       │                                                                     │
│       ▼                                                                     │
│  System prompt assembled:                                                   │
│   "你是小S... 與對方說話時：對小孩說話要非常溫柔寵溺..."                     │
│   "[情感: 類型] 必須在回覆最開頭"                                           │
│       │                                                                     │
│       ▼                                                                     │
│  OpenAI.stream(prompt, system_prompt)                                       │
│       │                                                                     │
│       ▼                                                                     │
│  Tokens stream back:                                                        │
│   Token 1: "「"                                                            │
│   Token 2: "[情感: 寵溺]"  ──► EmotionMapper.update()                    │
│                                        │                                    │
│                                        ├── emotion = "寵溺"                 │
│                                        ├── instruct =                        │
│                                        │   "(gentle, high-pitched...)"       │
│                                        └── cleaned = "「"                    │
│       │                                                                     │
│       ▼                                                                     │
│  tts_ready message sent to client:                                         │
│  {                                                                           │
│    "type": "tts_ready",                                                    │
│    "text": "「好啦～",                                                     │
│    "emotion": "寵溺",                                                      │
│    "instruct": "(gentle, high-pitched, warm and loving tone)",             │
│    "stream_url": "/api/tts/stream?text=「好啦～&emotion=寵溺&model=0.6B"  │
│  }                                                                          │
│       │                                                                     │
│       ▼                                                                     │
│  Subsequent tokens continue streaming via llm_token messages               │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Step 4: TTS Streaming                                │
│                                                                             │
│  Client receives tts_ready                                                  │
│       │                                                                     │
│       ▼                                                                     │
│  fetch(/api/tts/stream?text=...&emotion=...&model=...)                     │
│       │                                                                     │
│       ▼                                                                     │
│  FasterQwenTTSEngine.generate_streaming(                                   │
│    text="「好啦～",                                                         │
│    instruct="(gentle, high-pitched, warm and loving tone)",                 │
│    language="Chinese"                                                       │
│  )                                                                          │
│       │                                                                     │
│       ▼                                                                     │
│  Yields PCM chunks (24kHz mono 16-bit)                                     │
│       │                                                                     │
│       ▼                                                                     │
│  HTTP chunked transfer encoding                                             │
│       │                                                                     │
│       ▼                                                                     │
│  Client receives chunks via ReadableStream                                   │
│       │                                                                     │
│       ▼                                                                     │
│  AudioContext.play(PCM chunks) — progressive playback                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Emotion Tag Flow

```
LLM Output: "「[情感: 撒嬌]好啦～那我們來玩遊戲！"

                    │
                    ▼ Regex: ^\[情感[:：]\s*(.*?)\]\s*
                    │
         ┌──────────┴──────────┐
         │                     │
    Emotion Found          No Match
         │                     │
         ▼                     ▼
   emotion = "撒嬚"      emotion = null
   cleaned = "「好啦～..."   cleaned = original

         │
         ▼
get_tts_instruct("撒嬌")
         │
         ▼
"(coquettish, soft, slightly slower pace, endearing inflection)"

         │
         ▼
TTS generate_streaming(
  text="「好啦～那我們來玩遊戲！",
  instruct="(coquettish, soft, slightly slower pace, endearing inflection)",
  language="Chinese"
)
```

---

## Emotion → Instruct Mapping

| Emotion Tag | TTS Instruct String |
|-------------|---------------------|
| 寵溺 | "(gentle, high-pitched, warm and loving tone, soft delivery)" |
| 撒嬌 | "(coquettish, soft, slightly slower pace, endearing inflection)" |
| 毒舌 | "(witty, fast-paced, sarcastic but playful tone, confident delivery)" |
| 幽默 | "(playful, light-hearted, occasional laughs, casual and funny)" |
| 認真 | "(serious, thoughtful, measured pace, clear and deliberate)" |
| 溫和 | "(calm, gentle, warm, relaxed and reassuring tone)" |
| 調皮 | "(mischievous, playful, slightly teasing, energetic)" |
| 感動 | "(emotional, sincere, heartfelt, slower and softer)" |
| 生氣 | "(annoyed, frustrated, slightly elevated pitch, impatient)" |
| 開心 | "(happy, bright, enthusiastic, faster pace with positive energy)" |
| 默認 | "(natural, conversational tone, warm and engaging)" |

---

## Persona & Listener System

### Persona JSON Structure (`app/resources/personas/{persona_id}.json`)

```json
{
  "persona_id": "xiao_s",
  "base_personality": "你是小S，毒舌但有愛心，說話俏皮機智...",
  "emotion_instruction": "在回覆最開頭必須包含 [情感: 類型]...",
  "relationships": {
    "child": "對小孩說話要非常溫柔寵溺，用疊字和鼓勵性話語...",
    "mom": "對媽媽撒嬌貼心，報喜不報憂...",
    "reporter": "面對記者要快速反應，毒舌機智...",
    "friend": "輕鬆自然，像跟好朋友聊天...",
    "default": "標準小S風格..."
  },
  "default_relationship": "default"
}
```

### Prompt Composition

```
System Prompt = base_personality
              + relationships[listener_id]
              + emotion_instruction
```

---

## VAD Sensitivity Presets

| Preset | Energy Threshold | Silence to Commit | Min Speech |
|--------|-----------------|-------------------|------------|
| low | 0.005 | 2.0s | 0.5s |
| medium | 0.02 | 1.5s | 0.3s |
| high | 0.05 | 1.0s | 0.2s |

---

## Directory Structure

```
voice-ai-pipeline/
├── app/
│   ├── main.py                          # FastAPI app + Gradio mount
│   ├── logging_config.py                # JSON structured logging
│   ├── api/
│   │   ├── ws_asr.py                   # WebSocket endpoint
│   │   ├── tts_stream.py                # TTS HTTP streaming
│   │   └── gradio_ui.py                 # Gradio UI
│   ├── core/
│   │   └── state_manager.py             # Session + task management
│   ├── services/
│   │   ├── asr/
│   │   │   ├── engine.py               # Qwen3ASR / MockASR
│   │   │   └── vad_engine.py           # EnergyVAD with presets
│   │   ├── llm/
│   │   │   ├── openai_client.py        # OpenAI streaming client
│   │   │   └── prompt_manager.py      # PersonaManager (JSON-based)
│   │   └── tts/
│   │       ├── qwen_tts_engine.py     # Faster-Qwen3-TTS wrapper
│   │       └── emotion_mapper.py      # Emotion → instruct
│   └── resources/
│       ├── personas/                    # JSON persona definitions
│       │   └── xiao_s.json
│       └── voice_profiles/              # Reference audio (future)
│           └── xiao_s/
├── logs/                               # JSON log output
│   └── app.log
├── telemetry/                           # Prometheus + Grafana
│   ├── metrics.py
│   ├── collector.py
│   └── grafana/
└── tests/
```

---

## Telemetry & Observability

### Metrics (Prometheus)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vad_latency_seconds` | Histogram | component, model | VAD detection latency |
| `asr_latency_seconds` | Histogram | component, model | ASR inference latency |
| `llm_ttft_seconds` | Histogram | component, model | LLM time to first token |
| `e2e_latency_seconds` | Histogram | component | End-to-end latency |
| `audio_chunks_total` | Counter | component, session_id | Audio chunks received |
| `utterances_total` | Counter | component, session_id | Utterances processed |
| `llm_tokens_total` | Counter | component, model | Tokens generated |
| `errors_total` | Counter | component, error_type | Errors |
| `ws_connections_total` | Counter | component, status | WS connections |
| `active_sessions` | Gauge | component | Active sessions |

### Logs

All logs written to `/logs/app.log` in JSON format:

```json
{
  "timestamp": "2026-03-24T12:00:00Z",
  "level": "INFO",
  "logger": "app.api.ws_asr",
  "component": "ws",
  "message": "Session started",
  "session_id": "abc-123"
}
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `websockets` | WebSocket support |
| `pydub` | Audio format conversion (WebM → PCM) |
| `gradio` | Web UI |
| `faster-qwen3-tts` | TTS engine |
| `qwen-asr` | ASR engine |
| `openai` | LLM client |
| `prometheus-client` | Metrics |
