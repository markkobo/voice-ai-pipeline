# M1 Implementation Plan — Core Streaming Pipeline

**Milestone**: M1 | **Status**: Ready for Implementation
**Duration**: ~1 week

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser                                   │
│   ┌─────────────────┐  ┌──────────────────────────────────┐   │
│   │  Gradio UI      │  │  WebRTC Audio Capture             │   │
│   │  - mic button   │  │  (MediaRecorder → chunks)         │   │
│   │  - VAD slider   │  │                                   │   │
│   │  - listener sel │  │  Audio Playback                    │   │
│   │  - persona sel  │  │  (<audio> element ← HTTP stream)  │   │
│   │  - debug toggle │  │                                   │   │
│   │  - model select  │  │                                   │   │
│   │  - log viewer   │  │                                   │   │
│   └────────┬────────┘  └──────────────────────────────────┘   │
│            │ WebSocket                           ▲               │
│            │ text/control              audio chunks │               │
│            ▼                                     │               │
│  ┌─────────────────────────────────────────────────┐             │
│  │              FastAPI Backend                      │             │
│  │                                                  │             │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │             │
│  │  │ WS Endpoint│→│ VAD     │→│ ASR (Qwen3)  │  │             │
│  │  │ /ws/asr   │  │EnergyVAD│  │ streaming    │  │             │
│  │  └──────────┘  └──────────┘  └──────┬───────┘  │             │
│  │                                       │          │             │
│  │                          ┌────────────▼────────┐ │             │
│  │                          │ LLM (OpenAI)        │ │             │
│  │                          │ streaming          │ │             │
│  │                          │ + emotion parse    │ │             │
│  │                          └────────────┬────────┘ │             │
│  │                                       │            │             │
│  │                          ┌────────────▼────────┐ │             │
│  │                          │ EmotionMapper        │ │             │
│  │                          │ [情感: xxx] → instruct│ │             │
│  │                          └────────────┬────────┘ │             │
│  │                                       │            │             │
│  │  ┌────────────────────────────────────▼──────────┐│             │
│  │  │ TTS Service (Faster-Qwen3-TTS, VoiceDesign)   ││             │
│  │  │ + early trigger on first emotion detection   ││             │
│  │  └─────────────────────┬───────────────────────┘│             │
│  │                        │                          │             │
│  └────────────────────────┼──────────────────────────┘             │
│                           │                                       │
│            ┌──────────────┴───────────────┐                       │
│            │ HTTP Streaming Endpoint      │                       │
│            │ GET /api/tts/stream/{session}│                       │
│            └──────────────┬───────────────┘                       │
└───────────────────────────┼─────────────────────────────────────────┘
                            │ binary audio chunks
                            ▼
                    <audio> playback
```

---

## 2. Components to Build / Modify

### 2.1 New Files to Create

| File | Purpose |
|------|---------|
| `app/api/tts_stream.py` | HTTP SSE endpoint for TTS audio streaming |
| `app/services/tts/qwen_tts_engine.py` | Faster-Qwen3-TTS wrapper with VoiceDesign + emotion instruct |
| `app/services/tts/emotion_mapper.py` | `[情感: xxx]` → TTS natural language instruct |
| `app/api/gradio_ui.py` | Gradio UI page (streaming chat page) |
| `app/logging_config.py` | Structured JSON logging setup |
| `app/resources/personas/xiao_s.json` | Persona JSON definition |
| `app/resources/voice_profiles/xiao_s/default.wav` | Placeholder ref audio |

### 2.2 Files to Modify

| File | Changes |
|------|---------|
| `app/api/ws_asr.py` | Add emotion parsing from LLM stream; early TTS trigger; add `listener_id`/`persona_id` to config |
| `app/core/state_manager.py` | Add `tts_session_id`; add TTS cancellation support; rename `speaker_id` → `listener_id` |
| `app/services/llm/prompt_manager.py` | Support `persona_id` + `listener_id` → dynamic prompt from JSON |
| `app/main.py` | Mount Gradio UI; add `/api/tts/stream/{session_id}` route; configure JSON logging |

---

## 3. Detailed Implementation Steps

### Step 1: JSON Logging Setup
**File**: `app/logging_config.py` (new)

- Use Python `logging` with `JSONFormatter`
- All logs go to `/logs/app.log`
- Add `component`, `session_id`, `level`, `timestamp` fields
- Log levels: DEBUG, INFO, WARNING, ERROR

**UI Integration**:
- Gradio UI has an expandable log viewer (`gr.JSON()` or custom HTML component)
- Shows parsed human-readable log entries

---

### Step 2: Emotion Mapper + TTS Engine
**Files**: `app/services/tts/emotion_mapper.py`, `app/services/tts/qwen_tts_engine.py` (new)

**EmotionMapper**:
```python
# emotion_mapper.py
EMOTION_INSTRUCT_MAP = {
    "寵溺": "(gentle, high-pitched, warm and loving tone)",
    "撒嬌": "(coquettish, soft, slightly slower pace)",
    "毒舌": "(witty, fast-paced, sarcastic tone)",
    "幽默": "(playful, light-hearted, occasional laughs)",
    "認真": "(serious, thoughtful, measured pace)",
    # Fallback
    "默認": "(natural, conversational tone)",
}
```

**Faster-Qwen3-TTS Integration**:
- Use `VoiceDesign` mode: `model.generate_voice_clone_streaming(text, language, instruct=instruct_string)`
- Accept `instruct` parameter from emotion mapper
- Support `chunk_size=8` (~667ms per chunk) for streaming
- Implement `generate_streaming()` async generator yielding audio chunks
- Support model selection: `Qwen3-TTS-12Hz-0.6B-VoiceDesign` vs `Qwen3-TTS-12Hz-1.7B-VoiceDesign`

---

### Step 3: Persona JSON + PromptManager Update
**Files**: `app/resources/personas/xiao_s.json` (new), `app/services/llm/prompt_manager.py`

**xiao_s.json**:
```json
{
  "persona_id": "xiao_s",
  "base_personality": "你是小S，毒舌但有愛心，說話俏皮機智，語氣像在跟朋友聊天。",
  "emotion_instruction": "在回覆最開頭必須包含 [情感: 類型]，例如：[情感: 幽默] 或 [情感: 寵溺]。",
  "relationships": {
    "child": "對小孩說話要溫柔寵溺，用疊字，充滿鼓勵，時常說我愛你。",
    "mom": "對媽媽撒嬌，報喜不報憂，分享生活趣事，語氣貼心。",
    "reporter": "毒舌機智，面對記者要快速反應，語氣有防禦性但不失幽默。",
    "friend": "輕鬆自然，像跟好朋友聊天，可以開玩笑。",
    "default": "標準小S風格，中英文夾雜，適當使用台灣語助詞。"
  },
  "default_relationship": "default"
}
```

**PromptManager changes**:
- Load persona JSON from `app/resources/personas/{persona_id}.json`
- `get_prompt(persona_id, listener_id)`:
  - Load JSON for `persona_id`
  - Look up `relationships[listener_id]` (fallback to `default_relationship`)
  - Compose: `base_personality + relationship_description + emotion_instruction`
- Cache loaded JSON in memory dict

---

### Step 4: WebSocket Protocol Upgrade
**File**: `app/api/ws_asr.py`

**Updated Client → Server config frame**:
```json
{
  "type": "config",
  "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
  "persona_id": "xiao_s",
  "listener_id": "child",
  "model": "gpt-4o-mini"
}
```

**New Server → Client frame types**:
```json
{"type": "llm_token", "content": "「", "emotion": null}
{"type": "llm_token", "content": "好", "emotion": "寵溺"}  // first emotion detected
{"type": "tts_start", "session_id": "abc"}
{"type": "tts_audio", "data": "<base64 PCM>", "emotion": "寵溺"}
{"type": "tts_done"}
```

**Emotion parsing logic** (in `run_llm_stream`):
```python
EMOTION_PATTERN = re.compile(r'^\[情感:\s*(.*?)\]\s*')

def parse_emotion(text: str) -> tuple[Optional[str], str]:
    """Extract emotion tag and return (emotion, cleaned_text)."""
    match = EMOTION_PATTERN.match(text)
    if match:
        return match.group(1), EMOTION_PATTERN.sub('', text, count=1)
    return None, text
```

**Early TTS trigger flow**:
1. LLM starts streaming
2. Each `content_delta` → check for emotion tag at start
3. When first emotion detected → immediately start TTS stream
4. Subsequent tokens → append to TTS input
5. TTS yields audio chunks via `tts_audio` frames

**Barge-in handling**:
- New speech detected by VAD → cancel LLM task (existing)
- **NEW**: Also cancel TTS stream (new TTS session is created for new turn)
- Send `{"type": "tts_stop"}` to client

---

### Step 5: TTS Streaming Endpoint
**File**: `app/api/tts_stream.py` (new)

**Architecture**: HTTP Server-Sent Events (SSE)

**POST /api/tts/session** — Create TTS session
```json
Request: {"persona_id": "xiao_s", "emotion": "寵溺", "model": "0.6B"}
Response: {"session_id": "abc", "stream_url": "/api/tts/stream/abc"}
```

**GET /api/tts/stream/{session_id}** — Stream audio chunks
- Returns `Content-Type: audio/pcm` stream
- Yields binary PCM chunks as they are generated
- Client uses `<audio src="/api/tts/stream/{session_id}">` with `fetch()` + `ReadableStream`

**Alternative (simpler for MVP)**:
- Skip session management
- Client opens audio stream, server streams chunks as they come
- Interrupt = client simply stops fetching (server continues but client ignores)

**Recommended for M1**: Use simpler approach — single `GET /api/tts/stream?persona_id=...&emotion=...&model=...` that returns streaming audio. No session state needed. Browser handles playback natively.

---

### Step 6: VAD — Add Sensitivity Presets
**File**: `app/services/asr/vad_engine.py`

**Changes to EnergyVAD**:
```python
class EnergyVAD(BaseVAD):
    PRESETS = {
        "low":    {"energy_threshold": 0.005, "silence_duration_to_commit": 2.0, "min_speech_duration": 0.5},
        "medium": {"energy_threshold": 0.02,  "silence_duration_to_commit": 1.5, "min_speech_duration": 0.3},
        "high":   {"energy_threshold": 0.05,  "silence_duration_to_commit": 1.0, "min_speech_duration": 0.2},
    }

    def __init__(self, sample_rate: int = 24000, sensitivity: str = "medium"):
        preset = self.PRESETS.get(sensitivity, self.PRESETS["medium"])
        self.energy_threshold = preset["energy_threshold"]
        self.silence_duration_to_commit = preset["silence_duration_to_commit"]
        self.min_speech_duration = preset["min_speech_duration"]
        # ... rest of init
```

**State tracking**:
- Add `silence_frames` counter
- `detect()` returns True when speech ends (energy < threshold for `silence_duration_to_commit`)
- This drives the "VAD auto-commit" behavior

---

### Step 7: Gradio UI
**File**: `app/api/gradio_ui.py` (new)

**Layout**:
```
┌─────────────────────────────────────────────────────┐
│  Voice AI — Streaming Chat                          │
├─────────────────────────────────────────────────────┤
│  [Listener ▼]  [Persona ▼]  [VAD Sens ▼]  [Debug ☑]│
│  [LLM Model ▼]  [TTS Model ▼]                      │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │            Conversation Display               │   │
│  │  You: 你今天過得怎樣？                         │   │
│  │  AI: [emotion: 寵溺] 好啦～我很想你呢！       │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [🎤 Start] [⏹ Force Stop]   VAD: ● Speaking      │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Debug Panel (collapsible)                   │   │
│  │  - ASR Input: "你今天過得怎樣？"             │   │
│  │  - LLM Prompt: "你是小S... child..."         │   │
│  │  - Emotion Tag: "寵溺"                      │   │
│  │  - TTS Instruct: "(gentle, warm...)"        │   │
│  │  - Telemetry: ASR=120ms, LLM_TTFT=300ms...  │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Log Viewer (collapsible, JSON parsed)      │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Components**:
- `gr.Dropdown` for listener_id (child, mom, reporter, friend, default)
- `gr.Dropdown` for persona_id (xiao_s, ...)
- `gr.Dropdown` for VAD sensitivity (low, medium, high)
- `gr.Dropdown` for LLM model (gpt-4o-mini, gpt-4o)
- `gr.Dropdown` for TTS model (0.6B, 1.7B)
- `gr.Checkbox` for debug mode toggle
- `gr.Audio` for microphone input (`type="microphone"`)
- `gr.Audio` for TTS playback — use `<audio>` element pointing to `/api/tts/stream`
- `gr.Textbox` for conversation history
- `gr.JSON` for debug panel
- `gr.JSON` for log viewer

**WebRTC Audio Capture**:
- Gradio's `gr.Audio(type="microphone")` handles browser mic capture
- Output as a `gr.Audio(type="numpy")` for playback
- Need to pipe browser audio chunks → WebSocket

**Alternative (better streaming UX)**:
- Use custom HTML/JS in Gradio via `gr.HTML()` + `gr.Textbox()` for interaction
- Custom audio recording via Web Audio API + `MediaRecorder`
- Custom audio playback via `<audio>` + fetch streaming

**Decision for M1**: Use `gr.Audio(type="microphone")` for recording; use JavaScript fetch + `ReadableStream` to POST audio chunks via WebSocket. Audio playback via separate `<audio>` element polling `/api/tts/stream`.

---

### Step 8: StateManager — Add TTS Session Tracking
**File**: `app/core/state_manager.py`

**Add to SessionState**:
```python
class SessionState:
    # ... existing fields ...

    # TTS state (M1 new)
    tts_session_id: Optional[str]
    tts_task: Optional[asyncio.Task]
    tts_cancellation_event: Optional[asyncio.Event]
    current_emotion: Optional[str]
    current_instruct: Optional[str]
```

**Add methods**:
```python
def set_tts_task(self, session_id, task, event): ...
def cancel_tts_task(self, session_id) -> bool: ...
def clear_tts_task(self, session_id): ...
```

**Update `cancel_llm_task`** to also cancel TTS on barge-in.

---

### Step 9: Main.py Integration
**File**: `app/main.py`

```python
from fastapi import FastAPI
from app.api import ws_asr, tts_stream, gradio_ui

app = FastAPI()

# Mount Gradio UI at /ui
app = gr.mount_gradio_app(app, gradio_ui.build_ui(), path="/ui")

# WebSocket
app.include_router(ws_asr.router)

# TTS streaming (separate from WS)
app.include_router(tts_stream.router)
```

**JSON Logging Setup**:
```python
import logging
from app.logging_config import setup_json_logging

setup_json_logging()
```

---

## 4. Data Flow — Full Pipeline (M1)

```
1. User presses mic button → Gradio starts WebRTC recording
2. Audio chunks → POST to WebSocket /ws/asr
3. Server receives config: {listener_id: "child", persona_id: "xiao_s", ...}
4. Server stores in SessionState
5. Audio chunk → VAD.detect()
   - VAD sensitivity from config
   - Energy-based detection
6. Silence detected → commit_utterance()
7. ASR recognize(audio_bytes) → text
8. PromptManager.get_prompt("xiao_s", "child") → system_prompt
9. LLM.stream(prompt, system_prompt)
   - Tokens stream back
   - First token with [情感: xxx] → extract emotion
   - Start TTS streaming immediately (early trigger)
   - Subsequent tokens → append to TTS input buffer
10. TTS yields audio chunks
11. TTS chunks sent via HTTP streaming to client audio element
12. TTS done
13. New speech detected → cancel TTS + LLM tasks (barge-in)
```

---

## 5. File Structure After M1

```
app/
├── main.py
├── logging_config.py              # NEW: JSON logging setup
├── api/
│   ├── ws_asr.py                 # MODIFIED: emotion parsing, early TTS
│   ├── tts_stream.py             # NEW: TTS HTTP streaming endpoint
│   ├── gradio_ui.py             # NEW: Gradio UI page
│   └── __init__.py
├── core/
│   ├── state_manager.py         # MODIFIED: TTS session tracking
│   └── __init__.py
├── services/
│   ├── asr/
│   │   ├── engine.py            # Existing
│   │   ├── vad_engine.py       # MODIFIED: sensitivity presets
│   │   └── __init__.py
│   ├── llm/
│   │   ├── openai_client.py    # Existing
│   │   ├── mock_client.py      # Existing
│   │   ├── prompt_manager.py   # MODIFIED: JSON-based persona
│   │   └── __init__.py
│   ├── tts/
│   │   ├── qwen_tts_engine.py  # NEW: Faster-Qwen3-TTS wrapper
│   │   ├── emotion_mapper.py   # NEW: emotion → instruct
│   │   └── __init__.py
│   └── __init__.py
└── resources/
    ├── personas/
    │   └── xiao_s.json         # NEW
    └── voice_profiles/
        └── xiao_s/
            └── default.wav      # NEW (placeholder)
```

---

## 6. Testing Strategy

### Unit Tests
- `emotion_mapper.py` — test all emotion → instruct mappings
- `vad_engine.py` — test sensitivity presets
- `prompt_manager.py` — test JSON loading + persona/listener combination
- `ws_asr.py` — test emotion regex parsing

### Integration Tests
- Full WebSocket flow with MockASR + MockLLM
- VAD auto-commit timing tests
- Barge-in (interrupt) tests

### Manual Testing Checklist
- [ ] Browser mic recording → VAD auto-commit → ASR text
- [ ] LLM streaming → emotion tag extracted at first token
- [ ] TTS streaming audio plays back with correct emotion tone
- [ ] Barge-in cancels both LLM and TTS
- [ ] VAD sensitivity slider changes behavior
- [ ] Debug panel shows all intermediate data
- [ ] Log viewer shows parsed human-readable entries
- [ ] Listener/persona switch → different LLM tone + emotion
- [ ] LLM model switch → works correctly
- [ ] TTS model switch (0.6B vs 1.7B) → works

---

## 7. Open Questions / Deferred Decisions

| Item | Decision Needed | Recommendation |
|------|-----------------|----------------|
| Gradio Audio → WebSocket bridge | How to pipe browser audio to WS? | Use JS `MediaRecorder` + `WebSocket.send()` manually in Gradio HTML |
| TTS audio playback on browser | How to play streamed audio? | `<audio src="/api/tts/stream?..." autoplay>` + JS fetch with `ReadableStream` |
| Emotion regex fallback | What if no emotion tag found? | Use "default" emotion; log warning |
| First-turn behavior | TTS waits for emotion or uses default? | Default to "default" emotion until LLM emits one |
| Session cleanup | TTS sessions need timeout? | 5 minute idle timeout |

---

## 8. Dependencies to Add

```bash
pip install faster-qwen3-tts
pip install gradio
pip install python-multipart  # for file uploads (future M2)
```

---

## 9. Milestone M1 Acceptance Criteria

- [ ] Continuous streaming conversation works end-to-end
- [ ] VAD sensitivity slider changes detection behavior
- [ ] Listener/persona selection changes LLM output tone
- [ ] Emotion tags are parsed from LLM output and applied to TTS
- [ ] TTS audio streams back to browser with correct emotion tone
- [ ] Barge-in interrupts both LLM and TTS
- [ ] Debug mode shows all intermediate data (ASR, LLM prompt, emotion, TTS instruct, latencies)
- [ ] Structured JSON logs written to `/logs/app.log`
- [ ] Gradio UI log viewer shows human-readable parsed logs
- [ ] All component latencies visible in Grafana dashboard
