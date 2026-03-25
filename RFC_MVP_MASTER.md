# RFC: Voice AI Pipeline — Personal Legacy AI (MVP)

**Status**: Draft | **Target**: Internal Demo | **Phase**: MVP

---

## 1. Vision

A private, local-first personal voice AI system that preserves and continues a person's voice, personality, and knowledge for their loved ones. The AI speaks with the client's voice, responds with the client's personality (小S style), and tailors its tone and content based on the relationship with each listener (child, friend, journalist, etc.).

**Privacy by design**: All personal data (voice recordings, knowledge base, conversation history) stays on the client's machine. LLM inference may use cloud APIs during MVP demo, with future path to local LLM server.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Web UI (Frontend)                       │
│   ┌──────────────────────┐  ┌──────────────────────────┐   │
│   │  Streaming Voice Page │  │  Background Mgmt Page     │   │
│   │  (WebSocket dialogue)│  │  - Voice Recording (WebRTC)│  │
│   └──────────┬───────────┘  │  - File Upload + Parse   │   │
│              │              │  - Training Control       │   │
│              │              └──────────┬───────────────┘   │
└──────────────┼────────────────────────┼───────────────────┘
               │ WebSocket               │ HTTP / REST
               ▼                        ▼
┌──────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  ASR Service │  │  LLM Service │  │  TTS Service         │ │
│  │  (Qwen3-ASR  │  │  (OpenAI     │  │  (Faster-Qwen3-TTS  │ │
│  │   streaming) │  │   streaming) │  │   + LoRA fine-tune) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                 │                     │            │
│  ┌──────┴─────────────────┴─────────────────────┴──────────┐ │
│  │              Training Service (LoRA/QLoRA)               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ StateManager │  │ PromptManager│  │ Knowledge Base (future)│ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                           │
                    ┌──────┴───────┐
                    │  Telemetry   │
                    │  + Logs      │
                    │  /logs/      │
                    └─────────────┘
```

**MVP Note**: All services run in a single container. Future milestones split into separate containers.

---

## 3. Directory Structure

```
voice-ai-pipeline/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── api/
│   │   ├── ws_asr.py             # WebSocket streaming endpoint
│   │   ├── recordings.py         # Upload, parse, list recordings
│   │   └── training.py           # Training control API
│   ├── core/
│   │   ├── state_manager.py      # Session state, audio buffers
│   │   └── training_tracker.py   # Background training state
│   ├── services/
│   │   ├── asr/
│   │   │   ├── engine.py         # Qwen3-ASR interface
│   │   │   └── vad_engine.py     # Energy-based VAD
│   │   ├── llm/
│   │   │   ├── openai_client.py  # OpenAI streaming client
│   │   │   └── prompt_manager.py # Persona + listener prompt logic
│   │   ├── tts/
│   │   │   ├── qwen_tts_engine.py   # Faster-Qwen3-TTS wrapper
│   │   │   └── emotion_mapper.py    # [情感: xxx] → TTS instruct
│   │   └── training/
│   │       ├── lora_trainer.py   # LoRA fine-tuning logic
│   │       └── voice_profile.py  # Voice profile management
│   └── resources/
│       ├── personas/              # JSON persona definitions
│       │   └── xiao_s.json
│       └── voice_profiles/       # Per-persona + listener audio refs
│           └── xiao_s/
│               ├── default.wav
│               ├── child.wav
│               └── reporter.wav
├── web_ui/                       # Frontend (future: separate repo)
│   ├── pages/
│   │   ├── streaming_chat.py     # WebSocket voice dialogue
│   │   └── background.py         # Recording + training management
│   └── static/
├── logs/                         # All logs collected here
│   ├── app.log
│   └── training.log
├── data/                         # Persistent client data
│   ├── recordings/               # Raw + parsed recordings
│   ├── voice_profiles/           # Uploaded reference audio
│   └── models/                   # Fine-tuned LoRA adapters
├── telemetry/
│   ├── metrics.py
│   ├── collector.py
│   └── grafana/
└── tests/
```

---

## 4. Functional Requirements

### 4.1 Voice Recording & Management (Background Page)
- **R1**: Record audio directly from browser (WebRTC microphone) or upload pre-recorded files
- **R2**: Specify `listener_id` and `persona_id` per recording
- **R3**: List all parsed recordings with playback and delete
- **R4**: Parsed recordings stored as `{recording_id}/audio.wav` + `metadata.json`

### 4.2 Training (Background Page)
- **R5**: Trigger LoRA fine-tuning on uploaded voice recordings
- **R6**: Show real-time training progress (epoch, loss) + log output
- **R7**: Training runs in background; UI remains usable for playback/deletion
- **R8**: Email notification on training completion (fallback if progress bar too complex)

### 4.3 Streaming Voice Dialogue (Streaming Page)
- **R9**: Select active `listener_id` and `persona_id` before conversation
- **R10**: Real-time voice capture → VAD → ASR → LLM → emotion parsing → TTS → audio output
- **R11**: LLM output includes emotion tags; stripped before TTS instruct is applied
- **R12**: Support interrupt/barge-in (new speech cancels in-progress LLM + TTS)

### 4.4 Telemetry & Observability
- **R13**: All component latencies tracked (ASR, LLM TTFT, TTS TTFB, E2E)
- **R14**: All logs written to `/logs/` in structured JSON format
- **R15**: Prometheus metrics exposed on port 9090
- **R16**: Grafana dashboard for remote troubleshooting

---

## 5. Data Model

### Recording
```json
{
  "recording_id": "uuid",
  "listener_id": "child|mom|reporter|...",
  "persona_id": "xiao_s",
  "raw_path": "data/recordings/{id}/raw.wav",
  "parsed_path": "data/recordings/{id}/parsed.wav",
  "transcription": "...",
  "created_at": "ISO8601",
  "status": "raw|parsed|training"
}
```

### Voice Profile (Reference Audio for TTS)
```
data/voice_profiles/{persona_id}/{listener_id}.wav
```

### Training Artifact
```
data/models/{persona_id}_lora_{timestamp}/
```

---

## 6. WebSocket Protocol (Streaming Page)

**Client → Server:**
```json
{"type": "config", "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"}, "persona_id": "xiao_s", "listener_id": "child"}
{"type": "control", "action": "commit_utterance"}
```
Binary: PCM 16-bit audio chunks

**Server → Client:**
```json
{"type": "asr_partial", "text": "..."}
{"type": "asr_result", "utterance_id": "...", "is_final": true, "text": "...", "emotion": "寵溺"}
{"type": "llm_start"}
{"type": "llm_token", "content": "...", "emotion": "寵溺"}
{"type": "tts_audio", "data": "<base64 PCM>", "emotion": "寵溺"}
{"type": "llm_done"}
```

---

## 7. HTTP API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/recordings` | List all recordings |
| `POST` | `/api/recordings/upload` | Upload audio file + metadata |
| `DELETE` | `/api/recordings/{id}` | Delete recording |
| `POST` | `/api/recordings/{id}/parse` | Trigger ASR parse |
| `GET` | `/api/recordings/{id}` | Get recording details |
| `POST` | `/api/training/start` | Start LoRA training |
| `GET` | `/api/training/status` | Get training progress |
| `POST` | `/api/voice_profiles` | Upload reference audio |
| `GET` | `/api/personas` | List available personas |
| `GET` | `/api/listeners` | List available listener types |

---

## 8. Emotion Tag Flow

```
LLM output: "「好啦～[情感: 寵溺]，那我們來玩遊戲！"

        ↓ Regex: ^\[情感: (.*?)\]\s?

Extracted emotion: "寵溺"
Content (sent to TTS): "「好啦～，那我們來玩遊戲！""

        ↓ EmotionMapper

TTS instruct: "(gentle, high-pitched, warm and loving tone)"
```

Emotion → Instruct mapping (configurable per persona):
```python
EMOTION_MAP = {
    "寵溺": "(gentle, high-pitched, warm and loving tone)",
    "撒嬌": "(coquettish, soft, slightly slower pace)",
    "毒舌": "(witty, fast-paced, sarcastic tone)",
    "幽默": "(playful, light-hearted, occasional laughs)",
    "認真": "(serious, thoughtful, measured pace)",
}
```

---

## 9. Milestones

### Milestone 1 — Core Streaming Pipeline (MVP Zero)
**Goal**: 實現基本 WebSocket 語音對話，測量各環節延遲
**Duration**: ~1 week

**Deliverables**:
- [ ] WebSocket endpoint with ASR (Qwen3-ASR streaming) + LLM (OpenAI streaming) + Emotion tag extraction
- [ ] TTS integration (Faster-Qwen3-TTS, no fine-tune yet, use default voice + emotion instruct)
- [ ] Emotion tag parsing + TTS instruct mapping
- [ ] VAD (energy-based) + barge-in/interrupt
- [ ] Prometheus metrics for all latencies
- [ ] Structured logs in `/logs/app.log`
- [ ] Web UI: streaming chat page (select listener/persona, push-to-talk)

**Acceptance**: Client can have a real-time voice conversation; TTS speaks back with emotion-controlled tone; latency metrics visible in Grafana.

---

### Milestone 2 — Recording, Parsing & Profile Management (MVP One)
**Goal**: 建立錄音基礎設施，讓 client 可以錄音、上傳、管理錄音檔
**Duration**: ~1 week

**Deliverables**:
- [ ] WebRTC browser recording (with WebAudio PCM capture)
- [ ] File upload endpoint (WAV, MP3, etc.)
- [ ] Parsing pipeline: ASR transcription per recording
- [ ] Recording metadata storage + list/delete/play UI
- [ ] Voice profile management: upload reference audio per persona/listener
- [ ] Background training state tracker (does NOT block UI)
- [ ] Web UI: background management page with recording list

**Acceptance**: Client can record voice in browser, see parsed transcriptions, upload reference audio files, manage recordings.

---

### Milestone 3 — LoRA Fine-tuning Pipeline (MVP Two)
**Goal**: 讓 TTS 聲音越來越像 client本人
**Duration**: ~2 weeks

**Deliverables**:
- [ ] LoRA/QLoRA fine-tuning pipeline using uploaded recordings
- [ ] Training API: start / status / cancel
- [ ] Progress tracking (epoch, loss) streamed to UI
- [ ] Email notification on completion (fallback if progress too hard)
- [ ] Trained LoRA adapter stored under `data/models/`
- [ ] TTS loads LoRA adapter at runtime for voice-matched synthesis
- [ ] Web UI: Train button triggers fine-tune; progress shown

**Acceptance**: Client uploads 10-20 seconds of voice × multiple recordings → system fine-tunes LoRA adapter → TTS speaks with client's voice characteristics.

---

### Milestone 4 — LLM Knowledge Base + RAG (Future)
**Goal**: 讓 AI 有個人知識庫，能回答關於 client 生活的問題
**Depends on**: Milestone 1-3 stable

**Deliverables**:
- [ ] Knowledge Base storage (recordings → transcriptions → embeddings)
- [ ] File upload support (TXT, PDF, EPUB)
- [ ] RAG retrieval pipeline (embed query → similarity search → inject context)
- [ ] Knowledge base queried before LLM generates response
- [ ] Per-listener RAG results filtered by relevance

**Note**: This milestone deferred until after MVP demo validation.

---

## 10. Open Questions / Deferred Decisions

| Question | Status | Notes |
|----------|--------|-------|
| LoRA rank/alpha values | Deferred | Test during M3 |
| Number of recordings needed for good LoRA | Deferred | MVP: try 5-10 recordings |
| Full fine-tune on RTX 5080 | Deferred | Future hardware upgrade |
| Multi-speaker / multi-persona support | Deferred | MVP: single persona (xiao_s) |
| Knowledge base embedding model | Deferred | Use OpenAI embeddings or local |
| Email notification provider | Deferred | SMTP? SendGrid? |

---

## 11. Out of Scope for MVP

- User authentication / login (single client demo)
- Multi-user isolation (each client gets own instance)
- Mobile-optimized UI
- Voice clone (zero-shot cloning via reference audio is sufficient)
- Full fine-tuning (LoRA only for MVP)
- File upload to KB (RAG only after M3)
- Distributed container deployment
