身為一個有 C++ 與高頻交易背景的工程師，你本來就不該容忍 Pydub 這種透過 subprocess 呼叫 ffmpeg 的草率做法，更不該依賴 Gradio 這種玩具框架來測試低延遲串流。

我們現在要把這些毒瘤從你的架構文件中徹底切除。

以下我為你準備了一份可以直接餵給 Claude Code 的 Prompt (附帶完整的 Markdown 覆寫內容)。這份 Prompt 會指示 Claude Code 讀取並直接覆寫你的 RFC_MVP_master.md，把架構對齊到「真正的低延遲全雙工 WebSocket」標準。

執行步驟
確保你已經在 RunPod 的終端機內啟動了 Claude Code (claude)。

直接將下方虛線框內的所有內容（包含英文指示與 XML 標籤）複製並貼給 Claude。

Plaintext
Act as a strict Staff Software Engineer. I need you to completely overwrite the existing `RFC_MVP_master.md` file in the current workspace with the updated architecture below. 

The previous architecture had severe latency and state-management flaws (using HTTP streaming for TTS, Pydub for WebM decoding, Gradio for UI, and Energy VAD). This new version enforces a pure full-duplex WebSocket architecture, Client-side PCM resampling via AudioWorklet, Silero VAD, and Vanilla JS/HTML frontend.

Please execute a file write to replace the contents of `RFC_MVP_master.md` with the following:

<document>
# RFC: Voice AI Pipeline — Personal Legacy AI (MVP)

**Status**: Draft (Revised for Strict Low-Latency) | **Target**: Internal Demo | **Phase**: MVP

---

## 1. Vision

A private, local-first personal voice AI system that preserves and continues a person's voice, personality, and knowledge for their loved ones. The AI speaks with the client's voice, responds with the client's personality, and tailors its tone and content based on the relationship with each listener.

**Privacy by design**: All personal data stays on the client's machine. LLM inference uses cloud APIs during MVP demo, with a future path to a local LLM server.
**Performance standard**: End-to-End Latency < 1.5s. Full-duplex audio over a single WebSocket connection.

---

## 2. Architecture Overview

┌─────────────────────────────────────────────────────────────┐
│                     Vanilla Web UI (Frontend)               │
│   ┌──────────────────────┐  ┌──────────────────────────┐    │
│   │  Streaming Voice Page│  │  Background Mgmt Page    │    │
│   │  - AudioWorklet      │  │  - WebRTC Recording      │    │
│   │    (24kHz PCM capture│  │  - File Upload + Parse   │    │
│   │  - AudioContext queue│  │  - Training Control      │    │
│   └──────────┬───────────┘  └──────────┬───────────────┘    │
└──────────────┼─────────────────────────┼────────────────────┘
│ WebSocket               │ HTTP / REST
│ (Full-Duplex PCM+JSON)  │
▼                         ▼
┌──────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ ASR Service │  │ LLM Service │  │ TTS Service         │   │
│  │ - Silero VAD│  │ (OpenAI     │  │ (Faster-Qwen3-TTS   │   │
│  │ - Qwen3-ASR │  │  streaming) │  │  streaming bytes)   │   │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘   │
│         │                │                    │              │
│  ┌──────┴────────────────┴────────────────────┴──────────┐   │
│  │            StateManager (WebSocket Session)           │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘


---

## 3. Directory Structure Updates (Changes from Legacy)

voice-ai-pipeline/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── api/
│   │   ├── ws_asr.py              # Single WebSocket endpoint (ASR + TTS)
│   │   ├── recordings.py

│   │   └── training.py

│   ├── core/
│   │   └── state_manager.py       # Manages WS sessions, audio buffers, barge-in state
│   ├── services/
│   │   ├── asr/
│   │   │   ├── engine.py

│   │   │   └── silero_vad.py      # Replaced Energy VAD with ONNX Silero VAD
│   │   ├── tts/
│   │   │   └── emotion_mapper.py

├── web_ui/                        # Replaced Gradio with Vanilla HTML/JS
│   ├── index.html                 # Main streaming UI
│   ├── js/
│   │   ├── app.js                 # WebSocket & State logic
│   │   └── audio_processor.js     # AudioWorklet for downsampling to 24kHz PCM


---

## 4. Functional Requirements (Revised for Low Latency)

### Streaming Voice Dialogue (Streaming Page)
- **R10**: Browser captures audio, `AudioWorklet` resamples to 24kHz 16-bit mono PCM, and streams pure binary chunks via WebSocket to Server. Server bypasses `pydub`/`ffmpeg` entirely.
- **R11**: Server routes PCM to Silero VAD. Upon voice commit, triggers Qwen3-ASR -> LLM -> TTS. 
- **R12**: TTS generates audio and sends base64/binary PCM chunks back through the *same* WebSocket. Browser uses Web Audio API for immediate queued playback.
- **R13**: Support interrupt/barge-in. When VAD detects new user speech while TTS is playing, Server aborts LLM/TTS, and sends `{"type": "clear_buffer"}` to Client. Client instantly flushes its AudioContext queue.

---

## 5. WebSocket Protocol (Single Connection)

**Client → Server (Upstream):**
- Text: `{"type": "config", "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm_s16le"}, "persona_id": "xiao_s", "listener_id": "child"}`
- Text: `{"type": "control", "action": "commit_utterance"}`
- Binary: Raw PCM 16-bit chunks (e.g., every 100ms)

**Server → Client (Downstream):**
- Text: `{"type": "asr_result", "utterance_id": "...", "is_final": true, "text": "..."}`
- Text: `{"type": "llm_token", "content": "..."}`
- Text: `{"type": "tts_audio", "audio_b64": "<base64 encoded PCM chunk>"}` (Replaces HTTP streaming)
- Text: `{"type": "clear_buffer"}` (Barge-in command: tells client to stop playing current audio)

---

## 6. Milestones 

### Milestone 1 — Core Streaming Pipeline (Architecture Overhaul)
**Goal**: Implement strict low-latency full-duplex WebSocket streaming.
- [ ] Implement `AudioWorklet` in browser to stream raw 24kHz PCM.
- [ ] Replace Energy VAD with Silero VAD (ONNX).
- [ ] Refactor `/ws/asr` to handle both incoming ASR PCM and outgoing TTS base64 chunks.
- [ ] Implement Barge-in state machine (Server sends `clear_buffer` and cancels async TTS tasks).
- [ ] Remove all Gradio code and `pydub` subprocess dependencies.

</document>

Once the file is written, confirm the changes and explain the immediate next coding task to implement the `AudioWorklet` for frontend PCM capture.
為什麼這份改動至關重要？
消滅了 HTTP TTS 流：現在 ASR 輸入跟 TTS 輸出在同一個 WebSocket 生命週期裡。這代表你的 state_manager 終於有能力做全域中斷。當 Silero VAD 偵測到使用者講話，你可以一鍵取消 LLM 協程、停止 TTS 生成，並送出 clear_buffer 讓前端閉嘴。

切斷了 Ffmpeg 依賴：要求前端 AudioWorklet 直接傳 24kHz PCM。後端收到就是乾淨的 Bytes，直接餵進記憶體裡的 VAD 模型，0 I/O 延遲。

拔除了 Gradio：強迫你面對 Web Audio API，這是掌握串流播放的唯一正途。