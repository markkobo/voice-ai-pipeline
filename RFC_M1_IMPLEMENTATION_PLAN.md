# M1 Implementation Plan — Core Streaming Pipeline

**Milestone**: M1 | **Status**: ✅ Implementation Complete | **Commit**: `631ca4f`
**Duration**: ~1 week

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Browser (Client)                                │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  MediaRecorder   │──►│  WebSocket      │──►│  HTTP Audio Playback    │  │
│  │  (WebM/Opus)     │    │  (text/ctrl)   │    │  (AudioContext +       │  │
│  │  mic capture     │    │                 │    │   <audio> element)     │  │
│  └─────────────────┘    └────────┬────────┘    └─────────────────────────┘  │
│                                  │                                           │
│                                  │ JSON: asr_result, llm_token,            │
│                                  │      tts_ready, etc.                     │
│                                  ▼                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                                     │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                         WebSocket /ws/asr                              │    │
│  │  1. Receives WebM binary audio from client                           │    │
│  │  2. Decodes WebM → PCM (pydub)                                       │    │
│  │  3. Passes to VAD for speech detection                                │    │
│  │  4. On VAD commit: sends to ASR for transcription                    │    │
│  │  5. Passes text to LLM streaming                                     │    │
│  │  6. Parses emotion tags from LLM output                              │    │
│  │  7. Sends tts_ready with HTTP stream URL to client                   │    │
│  └───────────────────────────────┬──────────────────────────────────────┘    │
│                                  │                                            │
│                                  ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                      Pipeline Services                                │    │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │    │
│  │   │    VAD     │───►│    ASR      │───►│         LLM         │  │    │
│  │   │ (Energy)   │    │ (Qwen3-ASR)│    │  (OpenAI streaming) │  │    │
│  │   └─────────────┘    └─────────────┘    └──────────┬──────────┘  │    │
│  │                                                      │               │    │
│  │                                          ┌───────────▼───────────┐  │    │
│  │                                          │   EmotionMapper       │  │    │
│  │                                          │  [E:撒嬌] →        │  │    │
│  │                                          │  instruct string     │  │    │
│  │                                          └───────────┬───────────┘  │    │
│  └──────────────────────────────────────────────────────┼───────────────┘    │
│                                                          │                    │
│  ┌──────────────────────────────────────────────────────▼──────────────────┐ │
│  │                     TTS Service (HTTP /api/tts/stream)                 │ │
│  │   ┌─────────────────────────┐    ┌─────────────────────────────────────┐│ │
│  │   │   Faster-Qwen3-TTS   │───►│  PCM 24kHz mono streaming         ││ │
│  │   │   (VoiceDesign mode)  │    │  (HTTP chunked transfer)           ││ │
│  │   │   + emotion instruct   │    │                                     ││ │
│  │   └─────────────────────────┘    └─────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Architecture Decisions

### TTS Audio vs WebSocket — Decided: WebSocket Binary Only

**Final Architecture** (2026-04-01 update):
- **WebSocket**: text/control channel (asr_result, llm_token, tts_start, tts_done) + binary PCM chunks
- **No HTTP streaming**: TTS PCM chunks sent directly via WS binary frames

**Rationale**:
- Lower latency (no HTTP round-trip for chunk delivery)
- Simpler client (no AudioWorklet + fetch() duality)
- AudioWorklet receives PCM chunks directly via `ws.send_bytes()`
- `tts_start`/`tts_done` control AudioWorklet state

### Browser Audio Format — Decided: WebM → Server Decode

**Final Architecture**:
- Browser records via `MediaRecorder` in WebM/Opus format
- Server decodes WebM → PCM 24kHz mono using `pydub`
- PCM fed to VAD/ASR

**Rationale**:
- Browser-side JS decode/resample is CPU-intensive and unpredictable across devices
- Server-side decode is ~5-10ms/sec, negligible overhead
- MediaRecorder WebM is browser-native, simple implementation

---

## 3. Components Built

### 3.1 New Files Created

| File | Status | Purpose |
|------|--------|---------|
| `app/api/tts_stream.py` | ✅ | HTTP streaming endpoint for TTS audio |
| `app/services/tts/qwen_tts_engine.py` | ✅ | Faster-Qwen3-TTS VoiceDesign wrapper |
| `app/services/tts/emotion_mapper.py` | ✅ | `[E:情緒]內容` → TTS instruct |
| `app/api/gradio_ui.py` | ✅ | Gradio UI with WS + audio playback JS |
| `app/logging_config.py` | ✅ | Structured JSON logging → `/logs/app.log` |
| `app/resources/personas/xiao_s.json` | ✅ | Persona JSON definition |
| `app/resources/voice_profiles/xiao_s/default.wav` | ✅ | Placeholder ref audio |
| `app/services/tts/__init__.py` | ✅ | TTS module init |

### 3.2 Modified Files

| File | Status | Changes |
|------|--------|---------|
| `app/api/ws_asr.py` | ✅ | Emotion parsing, WS binary TTS, WebM decode |
| `app/core/state_manager.py` | ✅ | TTS session tracking, listener_id |
| `app/services/llm/prompt_manager.py` | ✅ | JSON-based persona loading (PersonaManager) |
| `app/services/asr/vad_engine.py` | ✅ | Sensitivity presets (low/medium/high) |
| `app/main.py` | ✅ | Gradio mount, logging setup, route registration |
| `app/services/llm/__init__.py` | ✅ | Export PersonaManager |

---

## 4. Protocol — Final Version

### WebSocket Messages (Server → Client)

| Message Type | Fields | Description |
|-------------|--------|-------------|
| `asr_result` | `utterance_id`, `is_final`, `text`, `telemetry` | ASR transcription |
| `vad_commit` | `utterance_id`, `energy`, `telemetry` | VAD detected end of speech |
| `llm_start` | `utterance_id` | LLM stream started |
| `llm_token` | `content`, `emotion` | LLM token (emotion on first detection) |
| `tts_start` | `sentence_idx` | Client prepares AudioWorklet for PCM chunks |
| `llm_done` | `text`, `total_tokens`, `telemetry` | LLM stream complete |
| `tts_done` | `sentence_idx` | Sentence streaming complete |
| `llm_cancelled` | `partial_text` | LLM interrupted |
| `llm_error` | `error` | LLM error |

### Binary Frames
- **Server → Client**: Raw Int16 PCM chunks (24kHz mono) sent via `ws.send_bytes()`

---

## 5. Bugs Fixed During Implementation

### Bug 1: EmotionMapper Incremental Return Duplication

**Problem**: `update()` returned entire buffer on each call, causing text to accumulate incorrectly.

**Fix**: Added `_buffer_returned_len` tracking to return only new text since last call.

```python
# Before (wrong)
return None, self._buffer  # Returns entire buffer every time

# After (correct)
new_text = self._buffer[self._buffer_returned_len:]
self._buffer_returned_len = len(self._buffer)
return None, new_text  # Returns only new text
```

### Bug 2: TTS Accumulation Before Emotion Detection (SUPERSEDED)

**Problem**: `tts_accumulated` started accumulating before emotion tag was fully detected, including partial tag characters like `'[E:撒嬌'`.

**Fix (OLD)**: Use `tts_text_parts: list[str]` and only start accumulating AFTER emotion tag is consumed.

**Current (2026-04-01)**: EmotionParser state machine handles this correctly with `[E:情緒]內容` format. The `]` delimiter provides clear boundary.

---

## 6. Test Results

### Comprehensive Test Suite — `test_m1_comprehensive.py`

**Run with mock services (no GPU needed):**
```bash
USE_QWEN_ASR=false USE_MOCK_LLM=true python test_m1_comprehensive.py
```

**Run with real services:**
```bash
python test_m1_comprehensive.py
```

### Unit Tests

| Component | Test | Result |
|-----------|------|--------|
| EmotionMapper | `'[E:撒嬌]好啦～'` → emotion=`撒嬌`, instruct | ✅ |
| EmotionMapper | Streaming tokens character-by-character | ✅ |
| EmotionMapper | No emotion tag → `emotion=None` | ✅ |
| VAD | `silence_frames_to_commit=25` (medium preset) | ✅ |
| VAD | Commit after 8 speech + 25 silence chunks | ✅ |
| VAD | No commit if speech < `min_speech_frames` | ✅ |
| PersonaManager | Load `xiao_s.json` | ✅ |
| PersonaManager | `get_prompt("xiao_s", "child")` includes relationship | ✅ |
| Logging | Structured JSON logs → `/logs/app.log` | ✅ |

### Integration Tests

| Test | Description | Result |
|------|-------------|--------|
| Health Check | `GET /health` returns status | ✅ |
| Prometheus Metrics | `GET /metrics` has all Vad/LLM/TTS metrics | ✅ |
| VAD Detection | WebSocket → audio chunks → VAD commit → ASR result | ✅ |
| LLM Streaming | WebSocket → ASR → LLM streaming → `llm_done` | ✅ |
| TTS WS Binary | WS binary chunks streamed to AudioWorklet | ✅ |
| Barge-in | New speech → cancels active LLM | ✅ |

**Total: 10/10 tests passing**

### Integration Flow Test (Mock)

```
LLM: '[E:撒嬌]好啦～那我們來玩遊戲！'
     ↓ EmotionParser.update() (char by char)
Emotion detected: 撒嬌 @ token ']'
Instruct: (coquettish, soft, slightly slower pace, endearing inflection)
Final TTS text: '好啦～那我們來玩遊戲！' (emotion tag correctly removed)
```

---

## 7. Open Issues / Pending

- [x] Browser testing completed (Standalone UI with WebSocket + AudioWorklet)
- [x] VAD sensitivity slider in UI updates VAD config in real-time

---

## 8. Known Issues / Post-M1 Fixes Applied

### TTS WAV Format Fix (P1) — SUPERSEDED
**Issue**: Browser `decodeAudioData` fails with raw PCM — needs WAV header

**Fix Applied (OLD)**: TTS endpoint now returns WAV format with proper header.
**Current (2026-04-01)**: TTS uses WS binary only. Raw Int16 PCM sent directly to AudioWorklet — no HTTP fetch, no WAV header needed.

### PyTorch Version for FasterQwen3TTS (DONE)
**Issue**: PyTorch 2.4.1 `torch.multinomial` cannot be captured in CUDA graphs

**Fix**: Upgraded to PyTorch 2.6.0+cu124
Status: Fixed, `requirements.txt` updated

### Uvicorn StatReload Crash (DONE)
**Issue**: `reload=True` caused StatReload when HuggingFace cache updated, crashing server mid-request

**Fix**: Set `reload=False` in `app/main.py`
Status: Fixed

---

## 9. Next: Milestone P1 — Latency Optimization

**Focus**: Reduce `speech_to_response_start` to < 2s for natural conversation feel

### P1.1: Silero VAD
- Replace EnergyVAD with Silero VAD (ONNX-based)
- Expected: Better accuracy, fewer false positives
- Files: `app/services/asr/silero_vad.py` (new)

### P1.2: TTS Chunks Immediate Playback — ✅ IMPLEMENTED
- TTS generates streaming chunks via WS binary (`send_bytes()`)
- Client AudioWorklet receives PCM chunks directly
- `tts_start`/`tts_done` control AudioWorklet state
- Status: Implemented (2026-04-01)

### P1.3: New Telemetry Metrics
```python
# app/telemetry/metrics.py additions
speech_to_response_start_seconds = Histogram(
    "speech_to_response_start_seconds",
    "Time from user stops speaking to first TTS audio playing",
    buckets=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
)
tts_first_chunk_seconds = Histogram(
    "tts_first_chunk_seconds",
    "Time from LLM first token to TTS first chunk",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0],
)
```

---

## 8. Dependencies Added

```
gradio>=4.0.0
faster-qwen3-tts
pydub>=0.25.0
```

---

## 9. Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| Continuous streaming conversation works end-to-end | ✅ Verified by integration tests |
| VAD sensitivity slider changes behavior | ✅ Implemented |
| Listener/persona selection changes LLM output tone | ✅ Implemented |
| Emotion tags parsed from LLM output and applied to TTS | ✅ Tested (mock) |
| TTS audio streams back via WS binary | ✅ Verified by integration test |
| Barge-in interrupts both LLM and TTS | ✅ Verified by integration test |
| Debug mode shows all intermediate data | ✅ Implemented |
| Structured JSON logs → `/logs/app.log` | ✅ Verified by unit test |
| Gradio UI with log viewer | ✅ Implemented |
| Telemetry (Prometheus metrics) | ✅ Verified by integration test |

---

## 10. File Structure

```
app/
├── main.py                          ✅ Modified
├── logging_config.py                ✅ New
├── api/
│   ├── ws_asr.py                  ✅ Modified
│   ├── tts_stream.py              ✅ New
│   ├── gradio_ui.py               ✅ New
│   └── __init__.py
├── core/
│   └── state_manager.py             ✅ Modified
├── services/
│   ├── asr/
│   │   ├── engine.py              (existing)
│   │   └── vad_engine.py          ✅ Modified
│   ├── llm/
│   │   ├── openai_client.py      (existing)
│   │   ├── prompt_manager.py      ✅ Modified
│   │   └── __init__.py            ✅ Modified
│   └── tts/
│       ├── qwen_tts_engine.py     ✅ New
│       ├── emotion_mapper.py      ✅ New
│       └── __init__.py            ✅ New
└── resources/
    ├── personas/
    │   └── xiao_s.json            ✅ New
    └── voice_profiles/
        └── xiao_s/
            └── default.wav         ✅ New (placeholder)
```

---

## Implementation Status — 2026-05-14 (post Phase 2 streaming hardening)

The M1 streaming pipeline is fully built and the chronic baseline test
failures it spawned are resolved. See `tests/_phase2_acceptance.md` for the
full delta.

**Streaming pipeline as built (`app/api/ws_asr.py`):**
- Browser PCM → WebSocket → VAD → ASR → LLM stream → emotion parser →
  per-sentence TTS → PCM chunks back via WS binary.
- VAD: still `EnergyVAD` (RFC marked Silero as planned but not done —
  energy-based works well enough; switch when the latency budget tightens).
- ASR: Qwen3-ASR streaming via `app/services/asr/engine.py:Qwen3ASR`.
- LLM: OpenAI streaming client at `app/services/llm/openai_client.py`,
  per-listener prompts from `app/services/llm/prompt_manager.py`.
- Emotion: state-machine parser at `app/services/tts/emotion_mapper.py`,
  `[E:情緒]內容` format.
- TTS: `FasterQwen3TTS` with CUDA-graph capture; falls back to non-streaming
  Qwen3TTSModel on capture failure.

**Phase 2 bug fixes (chronic baseline failures resolved):**
1. **`EmotionParser.is_ready` permanently False under Path B** — `Path B`
   replaced TTS instruct strings with text prosody enhancement, so
   `get_tts_instruct` always returns None. The `is_ready` property
   required `current_instruct is not None`, so it never went True.
   Fixed: `is_ready` now just checks `is_emotion_locked`.
2. **Cancel-before-LLM-start race** — `cancel_llm_task` was a no-op when
   the LLM task hadn't called `set_llm_task` yet. Added
   `llm_pending_cancel` sticky flag to `StateManager.SessionState`;
   `set_llm_task` honors it immediately. Proven by
   `tests/test_ws_asr.py:TestStickyCancel`.
3. **3 duplicated drain loops** at lines 395/473/518 — consolidated into
   `drain_emotion_parser(parser)` with `DRAIN_MAX_ITERATIONS=256`
   termination cap so a future parser bug can't infinite-loop the WS
   handler.
4. **6 silent `except Exception: pass` swallows** around TTS task awaits
   — replaced with `await_prior_tts_task(task, "context")` which logs
   the exception via `log.exception` and sends a `tts_error` frame so
   the client knows audio went sideways.
5. **`websocket.send_*` not guarded** — every send now goes through
   `safe_send_text` / `safe_send_bytes` which catches only client-
   disconnect errors and re-raises everything else.
6. **TTS engine double-load race** — `threading.RLock` around
   `_ensure_loaded()` and `activate_version()`. Without it, two
   concurrent `generate_streaming()` calls both observed
   `is_loaded=False` and both triggered model load, leaking VRAM.
   Proven safe under 50-thread contention by
   `tests/unit/test_tts_engine_lock.py`.
7. **Dead code removed:** `EMOTION_TAG_RE` (line 284) and the legacy
   `tts_ready` HTTP-fetch frame (lines 569-600). CLAUDE.md confirmed
   the client doesn't read `tts_ready`.

**Helpers extracted:** `app/api/_ws_helpers.py` with `drain_emotion_parser`,
`safe_send_text`, `safe_send_bytes`, `await_prior_tts_task`,
`send_tts_error_frame`, all unit-tested.

**Property testing:** `tests/unit/test_emotion_parser_property.py` runs
~450 random chunk-split scenarios through the parser via Hypothesis.

**Latency:** WS round-trip + TTFT + TTS-first-chunk still well under the
2s budget on the A10G.

**Deferred:** Silero VAD upgrade; httpx-ws-based async integration tests
for cancel timing (the TestClient sync model races at the cancel point).

### Follow-up — 2026-05-15

Live-server testing surfaced 5 streaming bugs (full detail in
`tests/_phase2_followups.md`):

1. `start_speech` cancelled in-flight LLM/TTS unconditionally — combined
   with the sticky-cancel flag, re-tapping the mic killed responses before
   they emitted a token. Fix: `start_speech` only resets buffer + VAD;
   the existing VAD-barge-in path handles the actual cancel when speech
   is detected.
2. Qwen3-ASR cold-start ~13 s on first inference despite startup
   preload. Fix: `load_model()` ends with a 0.5 s silence transcribe
   pass to trigger graph capture during startup.
3. `SessionState.tts_model` declared as a class type hint but never
   initialized in `__init__` — first read raised `AttributeError`, generic
   `except` closed the WS. Initialize `tts_model` and `llm_model`
   explicitly.
4. `min_silence_duration` 300 ms cut speech short on natural inhales;
   bumped default to 700 ms in both `SileroVADConfig` and `SileroVAD`.
   `sensitivity="high"` preset still drops to 200 ms.
5. `get_tts_generation_lock()` was referenced by the preview endpoint but
   missing from `qwen_tts_engine.py`. Added a module-level lazy-init
   `asyncio.Lock`.

Streaming-state UI: new persistent top bar on all three UI pages polls
`GET /api/system/status` every 5 s, shows VRAM bar, voice badge,
training spinner. Selective gating disables GPU-contending controls
when training is active (chat mic, recordings parse/process, training
start/activate/preview).
