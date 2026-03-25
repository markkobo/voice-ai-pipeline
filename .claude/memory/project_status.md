---
name: project status
description: Voice AI Pipeline current state, architecture, and recent fixes
type: project
---

## Current State (2026-03-25)

Voice AI pipeline is WORKING end-to-end:
- ASR: Qwen3-ASR 1.7B ✅
- LLM: OpenAI gpt-4o-mini streaming ✅
- Emotion parsing: [情感: xxx] tags → TTS instruct ✅
- TTS: Qwen3-TTS 1.7B VoiceDesign (fallback mode) ✅

## Architecture
- Browser mic → onaudioprocess (Int16 PCM) → WebSocket → VAD → ASR → LLM → EmotionMapper → tts_ready → HTTP TTS fetch → browser plays
- Standalone UI at /ui (no Gradio dependency)
- Server on port 8080, metrics on 9090

## Recent Fixes (already committed)
- TTS: FasterQwenTTSEngine falls back to Qwen3TTSModel on CUDA graph failure
- torchaudio CUDA mismatch fixed: install torchaudio==2.4.1+cu121
- Wrong TTS method fixed: generate_voice_design_streaming (not generate_voice_clone_streaming)
- Emotion tts_ready sent only ONCE per utterance (tts_url_sent flag)
- Default TTS model: 1.7B VoiceDesign

## For New Sessions
1. Read /workspace/voice-ai-pipeline-1/CLAUDE.md for full technical docs
2. Read /workspace/voice-ai-pipeline-1/manual.md for operational notes
3. HF_TOKEN and HF_HOME are in .env (gitignored)
4. Models cached at /workspace/voice-ai-pipeline-1/.cache/huggingface/hub/ (8.7GB total)
