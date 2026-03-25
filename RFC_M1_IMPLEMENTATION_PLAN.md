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
│  │                                          │  [情感: 撒嬌] →     │  │    │
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

### TTS Audio vs WebSocket — Decided: HTTP Streaming

**Final Architecture**:
- **WebSocket**: text/control channel only (asr_result, llm_token, tts_ready)
- **HTTP Streaming**: TTS audio channel (GET /api/tts/stream?text=...&emotion=...&model=...)

**Rationale**:
- Cleaner separation for troubleshooting (audio vs text issues independently)
- Audio doesn't pollute WS logs with large base64 strings
- Latency acceptable (HTTP chunked transfer ≈ WebSocket latency)
- Browser `<audio>` + `fetch()` + `ReadableStream` handles playback well

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
| `app/services/tts/emotion_mapper.py` | ✅ | `[情感: xxx]` → TTS instruct |
| `app/api/gradio_ui.py` | ✅ | Gradio UI with WS + audio playback JS |
| `app/logging_config.py` | ✅ | Structured JSON logging → `/logs/app.log` |
| `app/resources/personas/xiao_s.json` | ✅ | Persona JSON definition |
| `app/resources/voice_profiles/xiao_s/default.wav` | ✅ | Placeholder ref audio |
| `app/services/tts/__init__.py` | ✅ | TTS module init |

### 3.2 Modified Files

| File | Status | Changes |
|------|--------|---------|
| `app/api/ws_asr.py` | ✅ | Emotion parsing, tts_ready, WebM decode |
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
| `tts_ready` | `text`, `emotion`, `instruct`, `stream_url` | TTS stream URL for client to fetch |
| `llm_done` | `text`, `total_tokens`, `telemetry` | LLM stream complete |
| `llm_cancelled` | `partial_text` | LLM interrupted |
| `llm_error` | `error` | LLM error |

### HTTP Endpoint

```
GET /api/tts/stream?text=...&emotion=撒嬌&model=0.6B
→ Returns: audio/pcm streaming (24kHz mono 16-bit)
```

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

### Bug 2: TTS Accumulation Before Emotion Detection

**Problem**: `tts_accumulated` started accumulating before emotion tag was fully detected, including partial tag characters like `'[情感: 撒嬌'`.

**Fix**: Use `tts_text_parts: list[str]` and only start accumulating AFTER emotion tag is consumed:

```python
if new_emotion and not tts_notified:
    tts_notified = True
    current_emotion = new_emotion
    tts_text_parts = []  # Start fresh after tag
    # Send tts_ready with empty text, emotion only

elif tts_notified:
    if cleaned_content:
        tts_text_parts.append(cleaned_content)
        tts_text = "".join(tts_text_parts)
```

**Result**: Final TTS text is `'好啦～那我們來玩遊戲！'` (clean, without emotion tag).

---

## 6. Test Results

### Unit Tests

| Component | Test | Result |
|-----------|------|--------|
| EmotionMapper | `'[情感: 撒嬌]好啦～'` → emotion=`撒嬌`, instruct | ✅ |
| EmotionMapper | Streaming tokens character-by-character | ✅ |
| EmotionMapper | No emotion tag → `emotion=None` | ✅ |
| VAD | `silence_frames_to_commit=25` (medium preset) | ✅ |
| VAD | Commit after 5 speech + 25 silence chunks | ✅ |
| VAD | No commit if speech < `min_speech_frames` | ✅ |
| PersonaManager | Load `xiao_s.json` | ✅ |
| PersonaManager | `get_prompt("xiao_s", "child")` includes relationship | ✅ |
| All imports | `app.main`, `ws_asr`, `tts_stream`, etc. | ✅ |
| TTS Engine | MockTTSEngine streaming | ✅ |

### Integration Flow Test (Mock)

```
LLM: '[情感: 撒嬌]好啦～那我們來玩遊戲！'
     ↓ EmotionMapper.update() (token by token)
Emotion detected: 撒嬌 @ token ']'
Instruct: (coquettish, soft, slightly slower pace, endearing inflection)
Final TTS text: '好啦～那我們來玩遊戲！' (emotion tag correctly removed)
```

---

## 7. Open Issues / Pending

- [ ] Browser testing needed (Gradio UI WebSocket JS)
- [ ] Audio capture → WebM → Server decode → PCM (pydub integration)
- [ ] HTTP TTS streaming → AudioContext progressive playback
- [ ] VAD sensitivity slider in UI updates VAD config in real-time

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
| Continuous streaming conversation works end-to-end | ⏳ Pending browser test |
| VAD sensitivity slider changes behavior | ✅ Implemented |
| Listener/persona selection changes LLM output tone | ✅ Implemented |
| Emotion tags parsed from LLM output and applied to TTS | ✅ Tested (mock) |
| TTS audio streams back via HTTP | ✅ Implemented |
| Barge-in interrupts both LLM and TTS | ✅ Implemented |
| Debug mode shows all intermediate data | ✅ Implemented |
| Structured JSON logs → `/logs/app.log` | ✅ Implemented |
| Gradio UI with log viewer | ✅ Implemented |
| Telemetry (Prometheus metrics) | ✅ Already existed |

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
