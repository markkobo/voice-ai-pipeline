# RFC: Milestone 2.3 - Adaptive Voice Cloning & Demo Integration

**Status**: Active | **Target**: Claude Code Implementation | **Milestone**: 2.3

## 1. 核心目標 (Objective)
實現「聲學個性化」與「關係導向語音克隆」。系統需根據當前選定的 `persona_id` 與 `listener_id` 組合，自動加載對應的參考音訊（Reference Audio）進行 Zero-shot 克隆，並將 LLM 的情緒標籤轉化為 TTS 的情感控制指令。

## 2. 資源目錄結構 (Directory Structure)

建立專屬的語音 Profile 存放路徑，確保音色與關係對位：
- **Path**: `app/resources/voice_profiles/`
- **結構**:
    ```
    voice_profiles/
    ├── xiao_s/
    │   ├── default.wav       # 預設小 S 音色 (Fallback)
    │   ├── child.wav         # 對小孩的溫柔音色
    │   └── reporter.wav      # 對記者的專業/機智音色
    └── grandpa/
        └── default.wav
    ```

## 3. 功能需求 (Functional Requirements)

### 3.1 關係感知語音克隆 (Adaptive Cloning)
- **Zero-shot Engine**: 整合 `Qwen3-TTS`。
- **Profile Matching**: 
    - 優先讀取 `voice_profiles/{persona_id}/{listener_id}.wav`。
    - 若不存在，則 Fallback 至 `{persona_id}/default.wav`。
- **克隆參數**: 參考音訊建議長度為 10-20 秒，以確保韻律感穩定。

### 3.2 情感標籤轉化 (Natural Language Control)
- **邏輯**: 將解析出的 `emotion` 標籤對應至 Qwen3-TTS 的自然語言描述。
- **映射範例**:
    - `[E:毒舌]` -> `"(witty, fast-paced, sarcastic but playful tone, confident delivery)"`
    - `[E:寵溺]` -> `"(gentle, high-pitched, warm and loving tone, soft delivery)"`
    - `[E:撒嬌]` -> `"(coquettish, soft, slightly slower pace, endearing inflection)"`

### 3.3 新增 API 與協議
- **POST `/api/voice/clone`**: 接收 `file`, `persona_id`, `listener_id` 並存入對應目錄。
- **WebSocket 擴展**: `llm_token` 幀需包含提取後的 `emotion`，並傳遞給 TTS 引擎。

## 4. 實作路徑 (Implementation Steps)

### Step 1: 建立 TTS 服務層
- 實作 `app/services/tts/qwen_tts_engine.py`。
- 支援 `infer(text, ref_audio_path, emotion_prompt)` 接口。

### Step 2: 整合與狀態管理
- 更新 `state_manager.py`，新增追蹤當前 Session 的 `voice_ref_path`。
- 在 `ws_asr.py` 中，將 LLM 產出的第一個 Token（含情感標籤）立即觸發 TTS 預加載。

### Step 3: 打斷機制優化 (Audio Stop)
- 當偵測到新語音（Barge-in）時，除了 Cancel LLM，需發送 `{"type": "audio_stop"}` 給前端清空音訊緩衝區。

## 5. Claude Code 執行指令 (Action Prompt)

> 「請讀取本 RFC 2.3 並執行 Milestone 2.3：
> 1. 建立 `app/resources/voice_profiles/` 目錄結構，並更新 `state_manager.py` 支援根據 persona/listener 檢索音訊路徑。
> 2. 實作 `services/tts/qwen_tts_engine.py`，串接 Qwen3-TTS 的語音克隆 API，並實作情感標籤到自然語言指令的映射。
> 3. 修改 `api/ws_asr.py`：在 commit_utterance 後，將 LLM 產出的內容流式餵給 TTS，並確保情感標籤能引導 TTS 的語氣。
> 4. 建立 `/api/voice/clone` 端點，支援針對特定 Listener 上傳參考音訊。
> 5. 撰寫單元測試，驗證切換 `listener_id` 時，系統是否能正確加載對應的 `.wav` 參考音並產出帶有正確情感指令的請求。」

---

## Implementation Status — 2026-05-14

RFC 2.3 is ~60% built. Emotion routing and the `voice_profiles/`
directory layout are in place. The zero-shot voice-clone path
specified here was superseded in practice by the LoRA approach (see
RFC_M4_LORA_TRAINING.md) — the live system fine-tunes per persona
instead of doing reference-audio-based cloning at inference time.

**As built:**
- `app/resources/voice_profiles/{persona_id}/default.wav` — placeholder
  reference audio per persona. Path resolution in
  `app/services/tts/qwen_tts_engine.py:find_reference_audio()`.
- `app/services/tts/emotion_mapper.py:EMOTION_TEXT_ENHANCEMENT` — maps
  emotion tags to text-prosody markers (Path B) instead of TTS instruct
  strings. Implemented enhancers: 撒嬌 / 生氣 / 開心 / 溫和 / 幽默 /
  寵溺 / 毒舌 / 調皮 / 感動 / 認真 / 默認.
- Listener-aware emotion: `Listener.default_emotion` (Phase 1.3) drives
  the prompt; LLM emits `[E:情緒]` accordingly; emotion_mapper enhances
  the content before feeding TTS.
- `app/api/training.py:POST /api/training/voice-clone/activate` —
  activates voice-clone mode on the TTS engine (x-vector from reference
  audio); Pydantic body `VoiceCloneActivateRequest`.

**Path A vs Path B:** RFC 2.3 specified TTS instruct strings ("Path A"
in code comments). The codebase migrated to Path B (text-prosody
enhancement via `enhance_text(content, emotion)`) because instruct
strings on Qwen3-TTS gave inconsistent results. `get_tts_instruct`
remains as a no-op stub returning None for backwards compat. See
`emotion_mapper.py` header comment and Phase 2 acceptance doc.

**Not built:**
- `POST /api/voice/clone` for uploading per-listener reference audio —
  the user-facing endpoint doesn't exist. Voice profiles are baked in
  at `app/resources/voice_profiles/`.
- Per-listener `.wav` files at
  `voice_profiles/{persona_id}/{listener_id}.wav` — only `default.wav`
  per persona; listener routing to a per-listener reference is the
  RFC_M5 multi-adapter work, also deferred.
- "Listener changes voice acoustically" — explicitly NOT done.
  Listener changes the LLM prompt + the emotion enhancer, not the
  voice model. Per RFC_2_2 §3: "Listener affects emotion routing only,
  not voice model." That's the design.