# RFC: Voice AI Pipeline — Personal Legacy AI (MVP)

**Status**: Draft | **Target**: Internal Demo | **Phase**: MVP

---

## 1. Vision

A private, local-first personal voice AI system that preserves and continues a person's voice, personality, and knowledge for their loved ones. The AI speaks with the client's voice, responds with the client's personality (小S style), and tailors its tone and content based on the relationship with each listener (child, friend, journalist, etc.).

**Privacy by design**: All personal data (voice recordings, knowledge base, conversation history) stays on the client's machine. LLM inference may use cloud APIs during MVP demo, with future path to local LLM server.

**Performance target**: End-to-end latency < 2s for natural conversation feel. Key metric: "speech_to_response_start" — from user stops speaking to hearing first TTS audio.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Web UI (Frontend)                         │
│   ┌──────────────────────┐  ┌──────────────────────────┐   │
│   │  Streaming Voice Page │  │  Background Mgmt Page     │   │
│   │  - AudioWorklet/     │  │  - WebRTC Recording       │   │
│   │    ScriptProcessor   │  │  - File Upload + Parse    │   │
│   │  - WebSocket (PCM+JSON)│  │  - Training Control       │   │
│   └──────────┬───────────┘  └──────────┬───────────────┘   │
└──────────────┼─────────────────────────┼───────────────────┘
               │ WebSocket                │ HTTP / REST
               ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  ASR Service │  │  LLM Service │  │  TTS Service        │ │
│  │  - Silero VAD│  │  (OpenAI     │  │  (Faster-Qwen3-TTS  │ │
│  │  - Qwen3-ASR │  │   streaming) │  │   streaming chunks)  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                 │                     │              │
│  ┌──────┴────────────────┴─────────────────────┴────────────┐ │
│  │              StateManager (WebSocket Session)           │ │
│  │  - Audio buffers, utterance tracking                    │ │
│  │  - LLM/TTS task cancellation (barge-in)                │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                           │
                    ┌──────┴───────┐
                    │  Telemetry   │
                    │  + Logs      │
                    │  /logs/      │
                    └──────────────┘
```

**Key latency targets** (P1 improvements):
- VAD: < 500ms detection
- ASR: < 1s inference
- LLM TTFT: < 1s (API dependent)
- TTS first chunk: < 500ms
- **speech_to_response_start: < 2s**

---

## 3. Directory Structure

```
voice-ai-pipeline/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── api/
│   │   ├── ws_asr.py             # WebSocket endpoint (ASR + LLM + TTS)
│   │   ├── tts_stream.py        # HTTP TTS streaming endpoint
│   │   ├── recordings.py          # Recording management API
│   │   ├── training.py           # Training control API
│   │   ├── standalone_ui.py       # Vanilla JS/HTML UI (primary)
│   │   └── gradio_ui.py         # Gradio UI (fallback, keep for testing)
│   ├── core/
│   │   ├── state_manager.py      # Session state, audio buffers, task management
│   │   └── training_tracker.py   # Background training state
│   ├── services/
│   │   ├── asr/
│   │   │   ├── engine.py         # Qwen3-ASR interface
│   │   │   └── silero_vad.py    # Silero VAD (P1 upgrade from EnergyVAD)
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
│       ├── personas/
│       │   └── xiao_s.json
│       └── voice_profiles/
│           └── xiao_s/
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
│   ├── voice_profiles/          # Uploaded reference audio
│   └── models/                  # Fine-tuned LoRA adapters
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
- **R13**: All component latencies tracked (VAD, ASR, LLM TTFT, TTS TTFB, E2E)
- **R14**: **New P1**: Track `speech_to_response_start` — from user stops speaking to first TTS audio playing
- **R15**: All logs written to `/logs/` in structured JSON format
- **R16**: Prometheus metrics exposed on port 9090
- **R17**: Grafana dashboard for latency analysis

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

### Milestone 1 — Core Streaming Pipeline (MVP Zero) ✅ COMPLETED
**Goal**: 實現基本 WebSocket 語音對話，測量各環節延遲

**Completed**:
- [x] WebSocket endpoint with ASR (Qwen3-ASR streaming) + LLM (OpenAI streaming) + Emotion tag extraction
- [x] TTS integration (Faster-Qwen3-TTS, no fine-tune yet, use default voice + emotion instruct)
- [x] Emotion tag parsing + TTS instruct mapping
- [x] VAD (energy-based) + barge-in/interrupt
- [x] Prometheus metrics for all latencies
- [x] Structured logs in `/logs/app.log`
- [x] Standalone Web UI: streaming chat page (push-to-talk)

**Post-M1 Fixes (2026-03-25~29)**:
- [x] TTS fallback: FasterQwenTTSEngine falls back to Qwen3TTSModel when CUDA graph capture fails
- [x] Torchaudio CUDA mismatch fixed
- [x] Wrong TTS method fixed: `generate_voice_clone_streaming` → `generate_voice_design_streaming`
- [x] Emotion tag in display: EmotionMapper strips tag before display; `ttsText` used for TTS URL
- [x] tts_ready re-fetch storm: Server sends `tts_ready` only once per utterance
- [x] TTS model size: Default changed from 0.6B → 1.7B VoiceDesign
- [x] UI emotion tag display: Fragmented emotion tags handled correctly (skip `[, 情, 感, :, 默, 幽` fragments)
- [x] UI message box: One box per response (not one per token)
- [x] TTS flow control: MAX_BUFFER_SEC=8 prevents ring buffer overflow
- [x] AudioWorklet basic ring buffer with no spike detection (simplified to avoid artifacts)

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

### Milestone 3 — Latency Optimization (P1 Focus)
**Goal**: 降低感知延遲，讓對話更像真人

**P1 Priorities** (按優先順序):
1. [ ] **Silero VAD** 替換 Energy VAD
   - 更精準的語音檢測
   - 較少的 false positive/negative
   - 預期: VAD latency < 500ms

2. [ ] **TTS streaming chunks 立即播放**
   - 目前 TTS chunks 生成後等待完整 fetch 才播放
   - 改: TTS chunks 生成時就發給 client，client 立即播放
   - 預期: speech_to_response_start 降低 500ms+

3. [ ] **新增 Telemetry Metrics**
   ```python
   speech_to_response_start_seconds = Histogram(
       "speech_to_response_start_seconds",
       "Time from user stops speaking to first TTS audio playing",
       buckets=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
   )
   tts_first_chunk_seconds = Histogram(
       "tts_first_chunk_seconds",
       "Time from LLM first token to TTS first chunk",
   )
   ```

4. [ ] **Prompt 優化**
   - 短回覆適合即時對話
   - 避免過長的回覆

---

### Milestone 4 — LoRA Fine-tuning Pipeline (MVP Two)
**Goal**: 讓 TTS 聲音越來越像 client本人
**Duration**: ~2 weeks

**Deliverables**:
- [ ] LoRA/QLoRA fine-tuning pipeline using uploaded recordings
- [ ] Training API: start / status / cancel
- [ ] Progress tracking (epoch, loss) streamed to UI
- [ ] Email notification on completion
- [ ] Trained LoRA adapter stored under `data/models/`
- [ ] TTS loads LoRA adapter at runtime for voice-matched synthesis

**Acceptance**: Client uploads 10-20 seconds of voice × multiple recordings → system fine-tunes LoRA adapter → TTS speaks with client's voice characteristics.

---

### Milestone 5 — LLM Knowledge Base + RAG (Future)
**Goal**: 讓 AI 有個人知識庫，能回答關於 client 生活的問題
**Depends on**: Milestones 1-4 stable

**Deliverables**:
- [ ] Knowledge Base storage (recordings → transcriptions → embeddings)
- [ ] File upload support (TXT, PDF, EPUB)
- [ ] RAG retrieval pipeline
- [ ] Knowledge base queried before LLM generates response
- [ ] Per-listener RAG results filtered by relevance

**Note**: This milestone deferred until after MVP demo validation.

---

## 10. Open Questions / Deferred Decisions

| Question | Status | Notes |
|----------|--------|-------|
| LoRA rank/alpha values | Deferred | Test during M4 |
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
- File upload to KB (RAG only after M4)
- Distributed container deployment
