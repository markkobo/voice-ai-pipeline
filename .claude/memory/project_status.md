---
name: project status
description: Voice AI Pipeline current state, architecture, and recent fixes
type: project
---

## Current State (2026-04-05)

Voice AI pipeline is WORKING end-to-end:
- ASR: Qwen3-ASR 1.7B ✅
- LLM: OpenAI gpt-4o-mini streaming ✅
- Emotion parsing: `[E:情緒]內容` format → TTS instruct ✅ (updated 2026-04-01)
- TTS: Qwen3-TTS 1.7B VoiceDesign with WS binary streaming ✅

## Architecture
- Browser mic → onaudioprocess (Int16 PCM) → WebSocket → VAD → ASR → LLM → EmotionParser → tts_start → WS binary PCM chunks → AudioWorklet plays
- Standalone UI at /ui (no Gradio dependency)
- Server on port 8080, metrics on 9090

## Emotion Format (updated 2026-04-01)
- **OLD**: `[情感: 撒嬌]好啦～` (regex-based parsing)
- **CURRENT**: `[E:情緒]內容` 例：`[E:寵溺]好啦～不要生氣嘛`
- Delimiter: `]` clearly marks end of emotion value

## TTS Streaming (updated 2026-04-01)
- **OLD**: HTTP fetch to `/api/tts/stream`
- **CURRENT**: WS binary only - PCM chunks sent via `ws.send_bytes()` directly to AudioWorklet
- Control messages: `tts_start`/`tts_done` per sentence

## Recent Fixes (2026-04-01)
- EmotionParser: new `[E:情緒]內容` format with state machine
- ws_asr.py: drain loop properly flushes buffered characters
- pipeline.py: global `_cuda_lock` + `torch.cuda.empty_cache()` prevents OOM
- restart.sh: detects uncommitted changes for proper reload

## For New Sessions
1. Read /workspace/voice-ai-pipeline/CLAUDE.md for full technical docs
2. Read /workspace/voice-ai-pipeline/manual.md for operational notes
3. HF_TOKEN and HF_HOME are in .env (gitignored)
4. Models cached at /workspace/voice-ai-pipeline/.cache/huggingface/hub/ (8.7GB total)
