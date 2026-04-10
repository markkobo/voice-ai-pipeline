# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal Voice AI system (小S personality) — streaming pipeline:
**VAD → ASR → LLM streaming → Emotion parsing → TTS synthesis**

Target users: family members (child, mom, elder) with persona-aware responses.

## Setup & Running

```bash
cd /workspace/voice-ai-pipeline

# Install dependencies
pip install -r requirements.txt

# Copy/create .env with API keys
cp .env.example .env  # then fill in OPENAI_API_KEY, HF_TOKEN if needed

# Run server (port 8080, Qwen3-ASR + Qwen3-TTS)
bash scripts/restart.sh

# Watch mode — auto-reload on code changes (for development)
bash scripts/restart.sh --watch

# Force full restart (after env var changes, TTS reload needed)
bash scripts/restart.sh --force

# Just verify server is up
bash scripts/restart.sh --ui

# Run with mock services for testing
USE_QWEN_ASR=false USE_MOCK_TTS=true bash scripts/restart.sh
```

**Ports**: Server on `8080`, Prometheus metrics on `9090`.

## Architecture

```
app/
├── main.py                    # FastAPI app entry point, startup events
├── api/
│   ├── ws_asr.py             # WebSocket /ws/asr — ASR+LLM+TTS pipeline
│   ├── tts_stream.py         # HTTP TTS streaming endpoint /api/tts/stream
│   ├── standalone_ui.py      # Standalone HTML/JS UI (no Gradio)
│   └── gradio_ui.py          # Gradio-based UI (optional)
├── core/
│   └── state_manager.py      # Session state, audio buffers, utterance tracking
└── services/
    ├── asr/
    │   ├── vad_engine.py     # EnergyVAD (RMS-based speech detection)
    │   └── engine.py         # BaseASR + Qwen3ASR + MockASR
    ├── llm/
    │   ├── openai_client.py  # OpenAI streaming client
    │   └── prompt_manager.py # Persona-aware prompt templates
    └── tts/
        ├── qwen_tts_engine.py # TTS engine (FasterQwen3TTS + Qwen3TTSModel fallback)
        └── emotion_mapper.py  # [E:情緒] tag → TTS instruct string
telemetry/                      # Prometheus metrics + collector
tests/
```

## Voice AI Pipeline Flow

```
Browser microphone (onaudioprocess → Int16 PCM)
    ↓ WebSocket binary
/ws/asr — accumulate PCM in buffer
    ↓ commit_utterance control
VAD + ASR (Qwen3-ASR) → transcription
    ↓
LLM streaming (OpenAI gpt-4o-mini)
    ↓ LLM token with [E:情緒] tag
EmotionMapper — parse tag, return instruct + cleaned text
    ↓ first emotion detected
tts_start → server streams PCM chunks over WebSocket binary
    ↓
TTS (Qwen3-TTS 1.7B VoiceDesign) → PCM audio → AudioWorklet plays
```

## WebSocket Protocol

**Client → Server (JSON text frames):**
```json
{"type": "config", "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"}, "persona_id": "xiao_s", "listener_id": "child", "model": "gpt-4o-mini"}
{"type": "control", "action": "commit_utterance"}
{"type": "control", "action": "cancel"}
{"type": "control", "action": "start_speech"}  // barge-in, cancel LLM, clear buffer
```

**Client → Server (Binary frames):** Int16 PCM audio data

**Server → Client (JSON text frames):**
```json
{"type": "asr_result", "is_final": true, "text": "..."}
{"type": "llm_start", "utterance_id": "..."}
{"type": "llm_token", "content": "「", "emotion": null}
{"type": "llm_token", "content": "好啦", "emotion": "寵溺"}  // emotion appears on FIRST token after tag
{"type": "tts_start", "sentence_idx": 0}  // → client prepares AudioWorklet
{"type": "llm_done", "text": "「寵溺好啦～", "total_tokens": 12}
{"type": "tts_done", "sentence_idx": 0}    // sentence streaming complete
{"type": "llm_cancelled"}
{"type": "llm_error", "error": "..."}
```

**Server → Client (Binary frames):** Raw Int16 PCM chunks streamed directly via WebSocket binary

## Emotion System

**LLM output format:** Text prefixed with `[E:情緒]內容` emotion tag, e.g.:
```
[E:寵溺]好啦～不要生氣嘛
```
- `[E:` marks start of emotion tag
- `]` is the clear delimiter after emotion value
- Content follows after `]`

**EmotionParser** (`app/services/tts/emotion_mapper.py`):
- State machine parser for `[E:情緒]內容` streaming format
- Returns `(emotion, cleaned_text)` — strips tag from text
- Maps emotion → TTS natural language instruct string
- Character-by-character streaming emit for low latency

**Emotion → TTS instruct mapping:**
| Emotion | TTS Instruct |
|---------|-------------|
| 寵溺 | (gentle, high-pitched, warm and loving tone, soft delivery) |
| 撒嬌 | (coquettish, soft, slightly slower pace, endearing inflection) |
| 幽默 | (playful, light-hearted, occasional laughs, casual and funny) |
| 毒舌 | (witty, fast-paced, sarcastic but playful tone, confident delivery) |
| 溫和 | (calm, gentle, warm, relaxed and reassuring tone) |
| 開心 | (happy, bright, enthusiastic, faster pace with positive energy) |
| ... | ... |
| 默認 | (natural, conversational tone, warm and engaging) |

## TTS Engine

**Models:** `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (default, 1.7B) or `Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign`

**FasterQwen3TTS vs Qwen3TTSModel:**
- `FasterQwen3TTS` uses CUDA graphs for 6-10x speedup — but requires working CUDA graph capture
- If CUDA graph capture fails (common in some environments), falls back to `Qwen3TTSModel` (non-streaming, batches audio)
- Detection is automatic — no config change needed

**If TTS doesn't work:**
```bash
# Check torch/torchaudio CUDA match
python3 -c "import torch; import torchaudio; print(torch.__version__, torchaudio.__version__)"

# Install matching torchaudio if needed
pip install torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Use mock TTS for testing
USE_MOCK_TTS=true bash scripts/restart.sh --force
```

## UI

**Standalone UI** (`/ui`): Pure HTML/JS, no Gradio dependency. Connects directly to WebSocket and plays audio.

- Hold-to-record button (onaudioprocess captures raw PCM)
- Real-time streaming text display
- Emotion display per AI message
- Debug panel with protocol log

**Gradio UI** (`/gradio`): Optional, requires Gradio package.

## Persona System

Personas defined in `app/services/llm/prompt_manager.py`:
- `xiao_s` — 小S personality (default)
- `caregiver` — 照護者
- `elder_gentle` — 長輩-溫柔
- `elder_playful` — 長輩-活潑

Listener context affects system prompt (`child`, `mom`, `friend`, `default`).

## Testing

```bash
# Run tests (uses MockASR)
USE_QWEN_ASR=false pytest tests/ -v

# Run specific test
USE_QWEN_ASR=false pytest tests/test_ws_asr.py::TestWebSocketIntegration -v
```

## Development Philosophy

### Write Tests BEFORE Implementing Parsers/Streaming Code

String format parsing (emotion tags, delimiters, state machines) is error-prone. **Write unit tests first** to define expected behavior:

```python
# BEFORE implementing EmotionParser, write tests:
def test_full_format_streaming():
    """Full format in one chunk - emits one char at a time."""
    parser = EmotionParser()
    result = parser.update('[E:調皮]哈哈哈')
    assert result == ('調皮', '哈')  # First emit has emotion

def test_partial_tag_waits():
    """Partial [E: without closing ] waits for more."""
    parser = EmotionParser()
    result = parser.update('[E:')
    assert result is None  # Need more data
```

This would have caught the drain loop bug and emotion locking issues **before** integration.

### Do NOT Trade Streaming for Simplicity

The requirement is **real-time conversation < 2s latency**. Trading streaming for simpler batch processing violates this:

- ✅ Do: Stream TTS PCM chunks via WS binary as they generate
- ❌ Don't: Buffer entire audio, then send via HTTP fetch (adds latency)

If streaming is hard to implement correctly, **write more tests** rather than falling back to batch.

### Do NOT Trade Fine-tuning Quality for Convenience

LoRA fine-tuning is in the Milestone 4 roadmap for a reason — it significantly improves voice cloning quality. Don't skip it or use generic base models to save time:

- ✅ Do: Train LoRA adapters per persona voice
- ❌ Don't: Use base Qwen3-TTS without fine-tuning (sounds generic)

### State Machine / Streaming Code = High Test Priority

Code with complex state (parsers, streaming handlers, protocol buffers) is 10x more likely to have subtle bugs. Prioritize tests for:

1. **Emotion parsers** — all edge cases of partial tags, streaming chunks
2. **WebSocket drain loops** — ensure no infinite loops, all buffered data flushed
3. **VAD state transitions** — speech → silence → commit timing

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI base URL |
| `USE_QWEN_ASR` | `true` | Use Qwen3-ASR (or MockASR if false) |
| `USE_MOCK_LLM` | `false` | Use mock LLM for testing |
| `USE_MOCK_TTS` | `false` | Use mock TTS (no GPU) |
| `USE_QWEN_ASR` | `true` | Use Qwen3-ASR vs MockASR |
| `CUDA_VISIBLE_DEVICES` | (auto) | CUDA device |
| `HF_TOKEN` | (optional) | HuggingFace token for gated models |

## Important Implementation Notes

1. **onaudioprocess → raw PCM**: Browser captures audio via ScriptProcessorNode's onaudioprocess, converts Float32 → Int16, accumulates chunks, sends combined PCM blob on mouseup.

2. **VAD runs only on commit_utterance**: Audio is accumulated without VAD processing. VAD only fires on explicit commit, not on every chunk.

3. **TTS streaming via WS binary only**: Server streams PCM chunks as WebSocket binary frames. `tts_start`/`tts_done` control the AudioWorklet. HTTP fetch path (`tts_ready`) removed — client ignores `tts_ready` message.

4. **Emotion tag stripped at display**: Client's `llm_token` handler filters emotion tags before displaying. `ttsText` variable holds filtered text.

5. **server --port flag**: `uvicorn.run` in `main.py` hardcodes port 8080. Change `app/main.py:112` if needed.

## Recent Fixes (2026-04-05)

- **EmotionParser new format**: `[E:情緒]內容` format with state machine (was `[情感: xxx]` with regex)
- **Drain loop fix**: ws_asr.py properly drains all buffered chars when emotion is locked
- **CUDA OOM prevention**: Global `_cuda_lock` + `torch.cuda.empty_cache()` serializes GPU operations
- **VAD auto-send**: `process_audio()` correctly returns `vad_commit` when silence detected after speech
- **VAD barge-in**: New speech cancels LLM+TTS via `_vad_had_speech` + `_vad_committed` tracking
- **WS binary only TTS**: Removed HTTP fetch path. TTS streams PCM via WS binary frames
- **TTS fallback**: Falls back to non-streaming when CUDA graph errors occur
- **Auto-merge after training**: Training pipeline calls `merge_lora()` after success
- **Smart restart script**: Detects uncommitted changes for proper reload categorization
