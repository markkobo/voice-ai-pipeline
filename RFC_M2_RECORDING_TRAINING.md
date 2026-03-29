# RFC: Milestone 2 — Recording, Parsing & Voice Profile Management

**Status**: Draft | **Target**: MVP Demo | **Phase**: M2

---

## 1. Overview

**Goal**: 建立錄音基礎設施，讓 client 可以錄音、上傳、管理錄音檔，並為未來的 LoRA training 做準備。

**Duration**: ~1 week

---

## 2. Functional Requirements

### 2.1 Recording Methods

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-1 | WebRTC Browser Recording | 瀏覽器內直接錄音（麥克風），使用 MediaRecorder API |
| FR-2 | File Upload | 上傳已錄好的音檔（WAV, MP3, M4A, WebM） |
| FR-3 | Batch Upload | 同時上傳多個檔案 |
| FR-4 | Recording Duration Limit | 錄音時長限制：最短 3 秒，最長 5 分鐘 |
| FR-5 | Audio Quality Indicator | 錄音時顯示即時 dB meter，確保音量適中 |

### 2.2 Audio Quality Check

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-6 | Real-time Quality Check (Recording) | 錄音時即時分析 SNR / 音量，顯示是否可用於 training |
| FR-7 | Post-Upload Quality Check | 上傳後自動分析音頻品質 |
| FR-8 | Quality Threshold | 設定最低品質閾值，低於閾值自動標記或刪除 |
| FR-9 | Quality Metrics | 計算並儲存: SNR, RMS volume, silence ratio, clarity score |
| FR-10 | Training Readiness | 顯示錄音是否可用於 training（quality_ok: true/false） |
| FR-11 | Quality Warning | 品質不佳時在 UI 顯示警告，但不阻擋上傳 |

### 2.3 Audio Processing Pipeline

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-12 | Noise Reduction | 使用 rnnoise 去除背景雜音 |
| FR-13 | Voice Enhancement | 使用 speechbrain/demucs 強化人聲 |
| FR-14 | Speaker Diarization | 使用 pyannote.audio 分離/識別不同說話者並分段 |
| FR-15 | Progress Tracking | 每個 processing step 有獨立的 progress bar |
| FR-16 | Processing Status | 顯示: pending → denoising → enhancing → diarizing → done/failed |
| FR-17 | Concurrent Processing | 最多 1-2 個錄音同時處理（避免 GPU OOM） |
| FR-18 | Auto-Retry | 失敗時自動重試最多 2 次，之後手動確認 |
| FR-19 | Resume from Failure | 從失敗的 step 可選擇重試或從頭開始 |
| FR-20 | Processed File Auto-Cleanup | 處理後的檔案 3 天後自動刪除，需要可重新 parse |

### 2.4 Transcription (ASR)

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-21 | ASR Transcription | 使用 Whisper 將音頻轉為文字 |
| FR-22 | Real-time Transcription Display | 錄音完成後立即顯示逐字稿 |
| FR-23 | Transcription in UI | 逐字稿顯示在錄音卡片上 |
| FR-24 | Editable Transcription | 可手動編輯 transcription |
| FR-25 | Transcription Failure Handling | 失敗時記錄 log，可用 C (Continue) 跳過繼續 |
| FR-26 | Transcription Quality | 顯示 transcription confidence score |

### 2.5 Recording Management

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-27 | List Recordings | 列表顯示所有錄音（含狀態、時長、日期、品質） |
| FR-28 | Playback | 播放處理後的音頻 |
| FR-29 | Delete Recording | 刪除錄音（含所有處理過的版本） |
| FR-30 | Edit Metadata | 可選擇/修改 listener_id 和 persona_id |
| FR-31 | Recording Metadata | 包含: recording_id, listener_id, persona_id, title, duration, created_at, status, transcription, quality metrics |
| FR-32 | Raw File Retention | 原音永久保留，處理後的檔案 3 天後刪除 |

### 2.6 Voice Profile Management

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-33 | Upload Reference Audio | 上傳 TTS 參考音檔（WAV 24kHz/16bit） |
| FR-34 | Voice Profile Storage | 按 persona_id + listener_id 儲存 |
| FR-35 | List Voice Profiles | 列表顯示所有 voice profiles |
| FR-36 | Delete Voice Profile | 刪除 voice profile |

### 2.7 Training (Background)

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-37 | Start Training | 觸發 LoRA fine-tuning（使用已處理的錄音） |
| FR-38 | Training Progress | 顯示 epoch, loss 等即時進度 |
| FR-39 | Background Training | Training 在背景執行，不 blocking UI |
| FR-40 | Gray Out During Training | Training 期間禁用 UI 相關 buttons |
| FR-41 | Training Completion | 完成後通知（UI notification） |
| FR-42 | Training Failure Handling | 失敗時顯示錯誤訊息，可重試 |

### 2.8 Version Management

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-43 | Version Numbering | 格式: `v{increment}_{YYYYMMDD_HHMMSS}` (例: v1_20260329_143022) |
| FR-44 | List Training Versions | 列表顯示所有版本 + 狀態（training/ready/failed） |
| FR-45 | Version Status | tracking: training, ready, failed |
| FR-46 | Select Active Version | 在對話 UI 選擇使用哪個 training 版本 |
| FR-47 | Current Version Highlight | 顯示目前正在使用的版本 |
| FR-48 | Delete Version | 刪除不需要的 training 版本 |

### 2.9 Debug Panel & Logging

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-49 | Debug Log Panel | Recording/Training 頁面有即時 log 面板 |
| FR-50 | Log Levels | 區分: INFO, WARNING, ERROR, DEBUG |
| FR-51 | Processing Logs | 每個 processing step 的詳細 log |
| FR-52 | Training Logs | Training epoch/loss/錯誤 log |
| FR-53 | ASR Logs | Transcription 詳細 log（成功/失敗） |
| FR-54 | Audio Quality Logs | 品質分析的詳細 metrics log |
| FR-55 | Log Timestamps | 每條 log 帶 timestamp |
| FR-56 | Error Context | 失敗時記錄完整的 error context 和 stack trace |
| FR-57 | Export Logs | 可下載完整 log 檔案 |

### 2.10 Export/Backup

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-58 | Download Processed Audio | 下載處理後的音頻檔案 |
| FR-59 | Download Transcription | 下載逐字稿 (TXT/JSON) |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Recording Start Latency | < 100ms from button press to recording start |
| NFR-2 | File Upload Size Limit | Max 50MB per file |
| NFR-3 | Processing Speed | 處理速度 ≥ 1x realtime (60s audio ≤ 60s processing) |
| NFR-4 | UI Responsiveness | All UI interactions < 100ms response |

### 3.2 Metrics & Telemetry

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-5 | Disk Space Metrics | 追蹤: raw_size, processed_size, total_recordings |
| NFR-6 | Processing Latency | 追蹤每個 step: denoise_ms, enhance_ms, diarize_ms, asr_ms |
| NFR-7 | Training Latency | 追蹤: training_total_ms, per_epoch_ms, final_loss |
| NFR-8 | Quality Metrics | 追蹤: SNR_db, RMS_volume, silence_ratio, clarity_score |
| NFR-9 | Prometheus Export | 所有 metrics 可透过 /metrics endpoint 取得 |

### 3.3 Storage

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-10 | Raw Audio Format | WAV, 48kHz, 16-bit |
| NFR-11 | Processed Audio Format | FLAC, 48kHz, 16-bit (lossless compression) |
| NFR-12 | TTS Reference Format | WAV, 24kHz, 16-bit (Qwen3-TTS native) |
| NFR-13 | Metadata Storage | JSON files in same folder as WAV |
| NFR-14 | Storage Naming | Folder name includes listener_id, persona_id, timestamp |

### 3.4 Browser Compatibility

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-15 | Supported Browsers | Chrome/Edge (primary), Firefox (secondary) |
| NFR-16 | Mobile/Tablet Support | 針對 tablet UI 優化 |

### 3.5 Security/Privacy

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-17 | Local Storage Only | 所有資料存在本地，不上傳雲端（除 LLM API） |
| NFR-18 | No Audio to LLM API | 只有文字 transcription 會送到 LLM |

---

## 4. Folder/File Naming Convention

### 4.1 Recording Folder Structure

```
data/recordings/
└── {listener_id}_{persona_id}_{timestamp}/
    ├── audio.wav              # 原始錄音
    ├── audio_denoised.wav     # 降噪後 (3天後刪除)
    ├── audio_enhanced.wav     # 強化後 (3天後刪除)
    ├── metadata.json          # 完整 metadata
    └── transcription.txt      # 逐字稿 (純文字，方便移動)
```

**範例**: `child_xiao_s_20260329_143022/`

### 4.2 Voice Profile Structure

```
data/voice_profiles/
└── {persona_id}_{listener_id}.wav   # 例: xiao_s_child.wav
```

### 4.3 Training Version Structure

```
data/models/
└── {persona_id}_{version_id}/
    ├── adapter.pt             # LoRA weights
    ├── config.json            # Training config
    └── metadata.json          # Version metadata
```

---

## 5. Metadata Schema

### 5.1 Recording Metadata (metadata.json)

```json
{
  "recording_id": "uuid",
  "title": "string (optional)",

  "listener_id": "child | mom | dad | friend | reporter | elder | default",
  "persona_id": "xiao_s | caregiver | elder_gentle | elder_playful",

  "folder_name": "{listener_id}_{persona_id}_{timestamp}",
  "raw_filename": "audio.wav",
  "denoised_filename": "audio_denoised.wav",
  "enhanced_filename": "audio_enhanced.wav",

  "transcription": {
    "text": "string",
    "confidence": 0.95,
    "language": "zh",
    "segments": [
      {
        "start": 0.0,
        "end": 5.5,
        "text": "string"
      }
    ]
  },

  "duration_seconds": 45.5,
  "file_size_bytes": 4500000,

  "quality_metrics": {
    "snr_db": 25.5,
    "rms_volume": -12.3,
    "silence_ratio": 0.1,
    "clarity_score": 0.85,
    "training_ready": true
  },

  "status": "raw | processing | processed | failed",
  "processing_steps": {
    "denoise": {
      "status": "pending | in_progress | done | failed",
      "progress": 100,
      "error_message": null,
      "started_at": "ISO8601",
      "completed_at": "ISO8601"
    },
    "enhance": {
      "status": "pending | in_progress | done | failed",
      "progress": 100,
      "error_message": null,
      "started_at": "ISO8601",
      "completed_at": "ISO8601"
    },
    "diarize": {
      "status": "pending | in_progress | done | failed",
      "progress": 100,
      "error_message": null,
      "started_at": "ISO8601",
      "completed_at": "ISO8601"
    },
    "transcribe": {
      "status": "pending | in_progress | done | failed",
      "progress": 100,
      "error_message": null,
      "started_at": "ISO8601",
      "completed_at": "ISO8601"
    }
  },

  "speaker_segments": [
    {
      "speaker_id": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 12.5
    }
  ],

  "pipeline_metrics": {
    "denoise_ms": 1200,
    "enhance_ms": 3500,
    "diarize_ms": 8000,
    "transcribe_ms": 2200,
    "total_ms": 14900
  },

  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "processed_at": "ISO8601 (when status=processed)",
  "processed_expires_at": "ISO8601 (processed_at + 3 days, for auto-cleanup)"
}
```

### 5.2 Recording Index (index.json)

```json
{
  "recordings": [
    {
      "recording_id": "uuid",
      "folder_name": "child_xiao_s_20260329_143022",
      "listener_id": "child",
      "persona_id": "xiao_s",
      "duration_seconds": 45.5,
      "quality_training_ready": true,
      "status": "processed",
      "created_at": "ISO8601"
    }
  ]
}
```

### 5.3 Voice Profile Metadata

```json
{
  "profile_id": "uuid",
  "persona_id": "string",
  "listener_id": "string",
  "filename": "{persona_id}_{listener_id}.wav",
  "duration_seconds": 10.0,
  "sample_rate": 24000,
  "bit_depth": 16,
  "created_at": "ISO8601"
}
```

### 5.4 Training Version Metadata

```json
{
  "versions": [
    {
      "version_id": "v1_20260329_143022",
      "persona_id": "string",
      "status": "training | ready | failed",

      "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",

      "lora_path": "data/models/{persona_id}_{version_id}/",

      "training_config": {
        "rank": 16,
        "alpha": 32,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "batch_size": 4,
        "warmup_steps": 100
      },

      "metrics": {
        "initial_loss": 2.45,
        "final_loss": 0.85,
        "training_time_seconds": 3600,
        "per_epoch_ms": [320000, 310000, ...]
      },

      "recordings_used": ["uuid1", "uuid2", "uuid3"],
      "num_recordings_used": 3,

      "created_at": "ISO8601",
      "completed_at": "ISO8601"
    }
  ],

  "active_version": {
    "persona_id": "string",
    "version_id": "string"
  }
}
```

---

## 6. Audio Processing Pipeline

### 6.1 Step 1: Quality Check

```
Input:  audio.wav (raw)
Output: quality_metrics{}

Checks:
- SNR (Signal-to-Noise Ratio) > 15 dB
- RMS Volume > -40 dB
- Silence Ratio < 80%
- Clarity Score > 0.6

If quality check fails:
- Mark quality_metrics.training_ready = false
- Show warning in UI
- Log: "Audio quality below threshold: SNR={}, RMS={}"
- Allow user to proceed or delete
```

### 6.2 Step 2: Noise Reduction (rnnoise)

```
Input:  raw/audio.wav (48kHz/16bit)
Output: denoised/audio.wav (48kHz/16bit)

Tool: rnnoise (C library + Python binding)
Processing time target: < 2s for 60s audio
```

### 6.3 Step 3: Voice Enhancement (speechbrain)

```
Input:  denoised/audio.wav
Output: enhanced/audio.wav

Tool: speechbrain.separator.HookExtractor
Model: speechbrain/sepformer-wham16k
Purpose: 進一步分離人聲、抑制殘留噪音
Processing time target: < 10s for 60s audio
```

### 6.4 Step 4: Speaker Diarization (pyannote.audio)

```
Input:  enhanced/audio.wav
Output: speaker_segments[]

Tool: pyannote.audio
Model: pyannote/segmentation-3.0
Output: Array of {speaker_id, start_time, end_time}
Processing time target: < 15s for 60s audio
```

### 6.5 Step 5: Transcription (Whisper)

```
Input:  enhanced/audio.wav
Output: transcription{}

Tool: faster-whisper or openai-whisper
Model: openai/whisper-large-v3 (or distil-whisper for speed)
Language: zh (Chinese)

If transcription fails:
- Log: "ASR failed: {error}"
- Set status.transcribe = failed
- Allow user to use C (Continue) to skip
- Set transcription.text = ""
Processing time target: < 10s for 60s audio
```

---

## 7. API Endpoints

### 7.1 Recording Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/recordings` | List all recordings |
| `POST` | `/api/recordings/upload` | Upload audio file(s) |
| `GET` | `/api/recordings/{id}` | Get recording details + metadata |
| `DELETE` | `/api/recordings/{id}` | Delete recording + all processed files |
| `PATCH` | `/api/recordings/{id}` | Update metadata (listener_id, persona_id, title, transcription) |
| `POST` | `/api/recordings/{id}/process` | Trigger processing pipeline |
| `GET` | `/api/recordings/{id}/stream` | Stream audio for playback |
| `GET` | `/api/recordings/{id}/download` | Download processed audio |
| `GET` | `/api/recordings/{id}/transcription` | Get/edit transcription |

### 7.2 Voice Profile Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/voice_profiles` | Upload reference audio |
| `GET` | `/api/voice_profiles` | List all voice profiles |
| `DELETE` | `/api/voice_profiles/{id}` | Delete voice profile |

### 7.3 Training Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/training/start` | Start LoRA training |
| `GET` | `/api/training/status` | Get current training status |
| `GET` | `/api/training/versions` | List all training versions |
| `POST` | `/api/training/versions/{id}/activate` | Set active version |
| `DELETE` | `/api/training/versions/{id}` | Delete training version |

### 7.4 Reference Data Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/personas` | List available personas |
| `GET` | `/api/listeners` | List available listener types |

### 7.5 Metrics Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/metrics/recordings` | Get recording metrics (disk space, counts) |
| `GET` | `/api/metrics/processing` | Get processing latency metrics |
| `GET` | `/api/metrics/training` | Get training latency metrics |

---

## 8. UI Design

### 8.1 Page Structure

```
/ui                     → Streaming Voice Chat (existing)
/ui/recordings          → Recording Management Page (new)
```

### 8.2 Recording Management Page Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ [返回對話]                                          [版本切換 ▼]    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 🎤 錄音 (WebRTC)                                            │   │
│  │                                                              │   │
│  │  ▂▃▅▆▇▆▅▃▂▃▅▆▇  dB meter (即時)                          │   │
│  │  [音量: -12dB / 品質: ✓ OK]                                │   │
│  │                                                              │   │
│  │  👤 [child ▼]  🎭 [xiao_s ▼]                               │   │
│  │                                                              │   │
│  │  [●REC]  00:00:15 / 05:00                      [■STOP]     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  或 上傳檔案                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  拖放檔案到此處 或 [選擇檔案]                                 │   │
│  │  支援: WAV, MP3, M4A, WebM (最大 50MB)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Debug Panel                                       [折疊/展開]    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 14:30:22 [INFO] Recording started                          │   │
│  │ 14:30:25 [INFO] Quality check: SNR=22dB, RMS=-15dB ✓     │   │
│  │ 14:30:28 [INFO] Recording stopped, duration=28s            │   │
│  │ 14:30:28 [INFO] Saving to data/recordings/child_xiao_s... │   │
│  │ 14:30:28 [INFO] Processing pipeline started                │   │
│  │ 14:30:29 [INFO] Denoise: 100% done (1200ms)               │   │
│  │ 14:30:33 [INFO] Enhance: 100% done (3500ms)               │   │
│  │ 14:30:41 [INFO] Diarize: 100% done (8000ms)               │   │
│  │ 14:30:43 [INFO] Transcribe: 100% done (2200ms)            │   │
│  │ 14:30:43 [INFO] Pipeline complete. Total: 14900ms        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  錄音列表                                          [全部刪除]       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 🎤 錄音 1        👤 child  🎭 xiao_s   0:45  [▶][✕]       │   │
│  │    品質: ✓ OK | SNR=25dB | 清晰度=0.85                     │   │
│  │    狀態: 已處理 | 2026-03-29                                │   │
│  │    逐字稿: "這是錄音內容的文字..."              [展開/編輯]  │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ 🎤 錄音 2        👤 mom    🎭 xiao_s   1:23  [▶][✕]       │   │
│  │    品質: ⚠ 警告 | SNR=12dB 低於閾值                       │   │
│  │    狀態: 處理中... [████░░░░] 60%                         │   │
│  │    Denoise → Enhance → Diarize → Transcribe                 │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ 🎤 錄音 3        👤 friend 🎭 caregiver  2:01  [▶][✕]       │   │
│  │    狀態: 處理失敗                                          │   │
│  │    [ERROR] Transcribe failed: ASR timeout      [重試][刪除]│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  語音 Profile                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ xiao_s + child   10s   [▶] [上傳新] [刪除]                 │   │
│  │ xiao_s + mom     12s   [▶] [上傳新] [刪除]                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Training                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [開始 Training] (需要至少 3 個可用於 training 的錄音)         │   │
│  │                                                              │   │
│  │  版本           狀態       日期          操作                 │   │
│  │  v1_20260328   ready     2026-03-28   [使用中✓] [刪除]    │   │
│  │  v2_20260329   training   2026-03-29   [.......] [取消]    │   │
│  │                 [████████░░░░] 70% (epoch 7/10)            │   │
│  │                 loss: 0.85 | 預估剩餘: 15分鐘               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Storage: Raw: 450MB | Processed: 120MB | Versions: 2              │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Recording Card States

```
狀態: raw
[▶][編輯metadata][刪除] [處理] [品質: ✓ OK]

狀態: processing
[░░░░░░░░░] 30% (Denoise)
[████████░░] 60% (Enhance)
[██████░░░░] 90% (Diarize + Transcribe)

狀態: processed
[▶][✕][編輯][下載][重新處理]
逐字稿: [展開] "這是逐字稿內容..."

狀態: failed
[✕] 錯誤: {error_message}
[重試] [跳過並繼續 (C)] [刪除]
```

### 8.4 Version Selector (in Streaming Chat UI)

```
┌─────────────────────────────────────────────────────────────┐
│  版本: [v1_20260328 ✓ ▼]                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Debug Panel Specification

### 9.1 Log Format

```
{timestamp} [{level}] [{component}] {message} [{extra_data}]

範例:
14:30:22.123 [INFO] [RECORDING] Recording started {duration_limit: "5min"}
14:30:25.456 [DEBUG] [QUALITY] SNR=22.5dB, RMS=-15.3dB, silence_ratio=0.08
14:30:28.789 [INFO] [PIPELINE] Processing started {recording_id: "uuid", steps: ["denoise", "enhance", "diarize", "transcribe"]}
14:30:29.012 [DEBUG] [DENOISE] Input: audio.wav (4.5MB), Output: audio_denoised.wav (4.4MB)
14:30:29.234 [INFO] [DENOISE] Complete {duration_ms: 1200}
14:30:33.456 [ERROR] [TRANSCRIBE] ASR failed {error: "timeout after 30s", code: "ASR_TIMEOUT"}
```

### 9.2 Log Components

| Component | Description |
|-----------|-------------|
| `RECORDING` | WebRTC recording events |
| `QUALITY` | Audio quality analysis |
| `PIPELINE` | Processing pipeline events |
| `DENOISE` | Noise reduction step |
| `ENHANCE` | Voice enhancement step |
| `DIARIZE` | Speaker diarization step |
| `TRANSCRIBE` | ASR transcription |
| `TRAINING` | LoRA training events |
| `UPLOAD` | File upload events |
| `PLAYBACK` | Audio playback events |

### 9.3 Log Levels

| Level | Usage |
|-------|-------|
| `DEBUG` | Detailed debug info ( SNR values, timing, etc.) |
| `INFO` | Normal operation events |
| `WARNING` | Non-fatal issues (low quality, retry, etc.) |
| `ERROR` | Failures that need attention |

---

## 10. Testing Specification

### 10.1 Test Structure

```
tests/
├── unit/
│   ├── test_recording_api.py
│   ├── test_processing_pipeline.py
│   ├── test_file_storage.py
│   ├── test_validators.py
│   ├── test_audio_utils.py
│   ├── test_quality_checker.py
│   └── test_transcription.py
├── integration/
│   ├── test_upload_process_flow.py
│   ├── test_webrtc_recording_flow.py
│   ├── test_processing_e2e.py
│   ├── test_training_e2e.py
│   ├── test_version_management.py
│   └── test_api_endpoints.py
├── conftest.py
└── test_data/
    ├── sample.wav              # 10s clean audio
    ├── sample_noisy.wav        # 10s with background noise
    ├── sample_quiet.wav        # 10s low volume
    ├── sample_silent.wav       # 10s mostly silence
    ├── sample_multi_speaker.wav # 10s with 2 speakers
    └── sample_corrupt.wav      # invalid audio file
```

### 10.2 Unit Tests

| Component | Test Cases |
|-----------|------------|
| **Quality Checker** | SNR calculation, RMS calculation, silence ratio, clarity score, threshold detection |
| **File Storage** | folder naming, path generation, file existence, cleanup |
| **Validators** | listener_id enum, persona_id enum, file format validation, duration limits |
| **Audio Utils** | format conversion, duration extraction, sample rate detection |
| **Transcription** | text extraction, confidence parsing, segment parsing |
| **Pipeline Steps** | rnnoise interface, speechbrain interface, pyannote interface (mocked) |

### 10.3 Integration Tests

| Test | Description |
|------|-------------|
| **Upload Flow** | Upload WAV → Save to correct folder → Create metadata.json → Add to index |
| **Quality Check Flow** | Upload noisy audio → Quality check runs → Warning displayed |
| **Processing Pipeline E2E** | raw → denoise → enhance → diarize → transcribe → verify all outputs |
| **Processing Failure Recovery** | Denoise succeeds, enhance fails → Resume from enhance → Complete |
| **Transcription Failure** | ASR fails → Log error → User clicks "C" → Status = processed with empty transcription |
| **Auto-Cleanup** | Processed file older than 3 days → On startup check → Delete expired files |
| **Playback** | Upload → Process → Get stream URL → Playback works |
| **Delete Recording** | Create recording → Delete → All files removed, index updated |
| **Version Management** | Train → Complete → Activate → Chat uses new version |

### 10.4 Test Fixtures

```python
# conftest.py
@pytest.fixture
def temp_data_dir(tmp_path):
    """Temporary data directory for tests"""
    pass

@pytest.fixture
def sample_clean_audio(temp_data_dir):
    """10s clean WAV file"""
    pass

@pytest.fixture
def sample_noisy_audio(temp_data_dir):
    """10s noisy WAV file"""
    pass

@pytest.fixture
def sample_recording_metadata():
    """Sample recording metadata"""
    pass

@pytest.fixture
def mock_quality_checker():
    """Mock quality checker for fast tests"""
    pass

@pytest.fixture
def mock_whisper():
    """Mock Whisper for fast tests"""
    pass
```

---

## 11. Metrics & Telemetry

### 11.1 Prometheus Metrics

```python
# Recording metrics
recordings_total = Gauge("recordings_total", "Total number of recordings")
recordings_raw_size_bytes = Gauge("recordings_raw_size_bytes", "Total raw audio size")
recordings_processed_size_bytes = Gauge("recordings_processed_size_bytes", "Total processed audio size")
recordings_quality_ok_count = Gauge("recordings_quality_ok_count", "Recordings meeting quality threshold")

# Processing latency (histogram)
processing_denoise_seconds = Histogram("processing_denoise_seconds", "Denoise step latency", buckets=[0.5, 1, 2, 5, 10])
processing_enhance_seconds = Histogram("processing_enhance_seconds", "Enhance step latency", buckets=[1, 5, 10, 20, 30])
processing_diarize_seconds = Histogram("processing_diarize_seconds", "Diarize step latency", buckets=[5, 10, 20, 30, 60])
processing_transcribe_seconds = Histogram("processing_transcribe_seconds", "Transcribe step latency", buckets=[2, 5, 10, 20, 30])
processing_total_seconds = Histogram("processing_total_seconds", "Total processing latency", buckets=[10, 30, 60, 120, 300])

# Training metrics
training_current = Gauge("training_current", "Currently running training (0 or 1)")
training_epochs_total = Gauge("training_epochs_total", "Total epochs in current training")
training_epoch_current = Gauge("training_epoch_current", "Current epoch")
training_loss_current = Gauge("training_loss_current", "Current loss")
training_elapsed_seconds = Gauge("training_elapsed_seconds", "Training elapsed time")

# Quality metrics
quality_snr_db = Histogram("quality_snr_db", "Signal-to-Noise Ratio", buckets=[5, 10, 15, 20, 25, 30])
quality_clarity_score = Histogram("quality_clarity_score", "Clarity Score", buckets=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
```

### 11.2 Metrics Endpoint

```
GET /metrics (Prometheus format)
GET /api/metrics/summary (JSON format)
```

---

## 12. Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-1 | User can record audio in browser | Record 10s audio, see in list |
| AC-2 | User can upload WAV/MP3 file | Upload 5MB file, see in list |
| AC-3 | Quality check runs automatically | Upload noisy audio, see warning |
| AC-4 | Processing pipeline runs automatically | After upload, see progress bars |
| AC-5 | Each processing step shows progress | Denoise → Enhance → Diarize → Transcribe |
| AC-6 | Transcription is displayed | See ASR text under recording |
| AC-7 | Transcription is editable | Edit text, save, verify persisted |
| AC-8 | Processed audio plays correctly | Click play on processed recording |
| AC-9 | User can delete recording | Delete recording, removed from list |
| AC-10 | Metadata (listener/persona) is editable | Change listener_id, see update |
| AC-11 | Training starts in background | Start training, UI remains responsive |
| AC-12 | Training progress is visible | See epoch/loss updating |
| AC-13 | Training can be selected in chat | Select version, chat uses it |
| AC-14 | UI is gray-out during training | Buttons disabled during training |
| AC-15 | Voice profile can be uploaded | Upload ref audio, see in profiles |
| AC-16 | Debug panel shows logs | See all processing steps in log panel |
| AC-17 | Failed steps show error logs | Transcribe fails, see ERROR in debug panel |
| AC-18 | Processed files auto-delete after 3 days | Wait 3 days, verify files deleted |
| AC-19 | Raw files are never auto-deleted | Verify raw audio persists |
| AC-20 | All metrics are tracked | Disk space, latencies visible |

---

## 13. Open Questions / Deferred Decisions

| Question | Status | Notes |
|----------|--------|-------|
| Email notification | Deferred | SMTP? SendGrid? |
| LoRA rank/alpha values | Deferred | Test during M4 |
| Number of recordings for good LoRA | Deferred | MVP: try 5-10 recordings |
| Full fine-tune on RTX 5080 | Deferred | Future hardware upgrade |
| Multi-speaker support | Deferred | MVP: single persona |
| Knowledge base embedding | Deferred | Use OpenAI embeddings or local |
| pyannote.audio torch conflict | Known Issue | pyannote upgrades torch, breaks CUDA graphs |

---

## 14. Dependencies & Installation Notes

### PyTorch Version Conflict

**IMPORTANT**: `pyannote.audio` installs latest torch by default, which breaks CUDA graphs (TTS acceleration).

**Correct install order:**
```bash
# 1. Install torch FIRST (exact version required for CUDA graphs)
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 2. Install pipeline packages
pip install faster-whisper speechbrain pyannote.audio

# 3. Restore torch if pyannote.audio broke it
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### qwen-asr vs faster-whisper

- **qwen-asr**: Real-time streaming ASR for WebSocket voice chat (low latency)
- **faster-whisper**: Batch transcription for pipeline processing (high accuracy)

They serve different purposes — not redundant.

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt install ffmpeg    # Audio format conversion
sudo apt install rnnoise   # Noise reduction
```

### Python Packages

```txt
# Core
torch==2.6.0+cu124           # MUST be exact version
torchaudio==2.6.0+cu124
fastapi>=0.100.0
uvicorn>=0.23.0
pydub>=0.25.0
python-dotenv>=1.0.0

# ASR - Real-time streaming
qwen-asr>=0.0.6

# TTS - Voice synthesis
faster-qwen3-tts
qwen-tts>=0.1.1

# Pipeline - Batch processing
faster-whisper>=1.0.0
speechbrain>=1.0.0
pyannote.audio>=3.0.0

# Metrics
prometheus-client>=0.19.0

# Background Tasks
fastapi.BackgroundTasks (included with FastAPI)
```

---

## 16. Post-Implementation Issues & Improvements

> Documented during M2 implementation review (2026-03-29). These issues were found after Phase 1-5 completion.

### 16.1 Architecture Issues

#### ISSUE-1: `list_all_recordings()` does full filesystem traversal on every call
**Severity**: High | **Component**: `app/services/recordings/file_storage.py`

**Problem**: Every call to `list_all_recordings()` iterates over all listener/persona directories and parses every folder name. At scale (hundreds of recordings), this causes:
- Slow API responses (1-5 seconds)
- Increased disk I/O
- No caching of frequently-accessed data

**Current code**:
```python
def list_all_recordings():
    for listener_id in VALID_LISTENER_IDS:
        for persona_id in VALID_PERSONA_IDS:
            base = DATA_DIR / listener_id / persona_id
            if base.exists():
                for folder in base.iterdir():
                    # parse every folder every call
```

**Recommended fix**:
- Add recording index cache (`recordings_index.json`) updated on create/delete
- Use `functools.lru_cache` with TTL for `list_all_recordings()`
- Or migrate to SQLite for metadata storage

**Priority**: P0

---

#### ISSUE-2: Metadata JSON in same folder as audio — coupling risk
**Severity**: Medium | **Component**: `app/services/recordings/metadata.py`

**Problem**: Recording metadata is stored in the raw folder alongside audio files. Deleting/moving folders can lose metadata if not handled carefully.

**Recommended fix**:
- Consider separate metadata store (SQLite)
- Or ensure delete operations always go through API (not filesystem)

**Priority**: P2

---

### 16.2 Error Handling Issues

#### ISSUE-3: Pipeline has no retry logic
**Severity**: High | **Component**: `app/services/recordings/pipeline.py`

**Problem**: If denoise fails mid-processing (network blip, OOM), the entire pipeline fails and user must manually retry from start.

**Current**:
```python
def run(self) -> ProcessingResult:
    try:
        self._run_quality_check()
        self._run_denoise()  # no retry
        self._run_enhance()  # no retry
        ...
```

**Recommended fix**:
```python
def _run_with_retry(step_fn, step_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            return step_fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"[PIPELINE] {step_name} failed, retrying ({attempt+1}/{max_retries})")
            time.sleep(2 ** attempt)  # exponential backoff
```

**Priority**: P0

---

#### ISSUE-4: Quality check failure marked as "done" instead of "skipped"
**Severity**: Medium | **Component**: `app/services/recordings/pipeline.py:110-138`

**Problem**: When quality check throws an exception, it catches it and marks the step as "done" with 100% progress, even though no quality check actually ran.

**Current**:
```python
except Exception as e:
    self._log(f"Quality check failed: {e}", "ERROR")
    # Quality check failure is not fatal - continue with pipeline
    self.metadata.update_processing_step("denoise", "done", progress=100)  # misleading!
```

**Recommended fix**:
- Add "skipped" or "warning" status to processing_steps
- Or log as "done_with_warning"

**Priority**: P1

---

### 16.3 API Design Issues

#### ISSUE-5: `GET /api/recordings` has no pagination
**Severity**: High | **Component**: `app/api/recordings.py`

**Problem**: Returns all recordings at once. With hundreds of recordings, this causes slow responses and large JSON payloads.

**Recommended fix**:
```python
@router.get("/")
async def list_recordings(page: int = 1, limit: int = 20):
    # return paginated results with total count
    return {
        "recordings": [...],
        "total": total_count,
        "page": page,
        "limit": limit,
    }
```

**Priority**: P0

---

#### ISSUE-6: `POST /api/training/versions` is a stub — doesn't trigger actual training
**Severity**: High | **Component**: `app/api/training.py`

**Problem**: Creates a version record with status="training" but nothing actually trains. The actual LoRA training pipeline is not connected.

**Recommended fix**:
- Document clearly that this is a stub for MVP
- Integrate with actual training job queue (Celery/Redis worker, or BackgroundTasks)
- Add training completion callback that updates version status

**Priority**: P0

---

#### ISSUE-7: No cross-reference between recordings and training versions
**Severity**: Medium | **Component**: `app/services/recordings/`, `app/services/training/`

**Problem**: A recording can be deleted even if it's referenced by a training version. Training versions track `num_recordings_used` but don't store actual recording IDs.

**Current**:
```python
# training.py - TrainingVersion
num_recordings_used: int = 0  # just a count, not IDs
```

**Recommended fix**:
```python
@dataclass
class TrainingVersion:
    ...
    recording_ids_used: list[str] = []  # actual recording IDs
```

**Priority**: P1

---

#### ISSUE-8: `processed_expires_at` is set but no cleanup job runs
**Severity**: High | **Component**: `app/services/recordings/metadata.py`

**Problem**: `processed_expires_at` is calculated when status becomes "processed", but nothing actually deletes expired recordings.

**Recommended fix**:
```python
# Add endpoint or startup task
@router.post("/api/maintenance/cleanup-expired")
async def cleanup_expired_recordings():
    """Delete recordings where processed_expires_at < now"""
    expired = get_expired_recordings()
    for paths in expired:
        paths.delete_all()
    return {"deleted": len(expired)}
```

**Priority**: P0

---

### 16.4 Data Integrity Issues

#### ISSUE-9: Deleting recording doesn't check training dependencies
**Severity**: High | **Component**: `app/api/recordings.py:delete_recording`

**Problem**: Recording can be deleted while being used by an active training version.

**Recommended fix**:
```python
@router.delete("/{recording_id}")
async def delete_recording(recording_id: str):
    # Check if any training version uses this recording
    manager = get_version_manager()
    for version in manager.list_versions():
        if recording_id in version.recording_ids_used:
            if version.status == "training":
                raise HTTPException(400, "Cannot delete recording used in active training")
```

**Priority**: P0

---

### 16.5 Performance Issues

#### ISSUE-10: Upload reads entire file into memory
**Severity**: Medium | **Component**: `app/api/recordings.py:upload_recording`

**Problem**:
```python
content = await file.read()  # entire file in memory
if len(content) > MAX_FILE_SIZE:
    ...
f.write(content)
```

**Recommended fix**: Use streaming upload with chunked processing.

**Priority**: P1

---

#### ISSUE-11: Pipeline steps run sequentially — denoise and enhance could parallelize
**Severity**: Low | **Component**: `app/services/recordings/pipeline.py`

**Problem**: `denoise` and `enhance` are independent but run sequentially.

**Recommended fix**: Run as `asyncio.gather(denoise_task, enhance_task)` if using async pipeline.

**Priority**: P2

---

### 16.6 Testing Gaps

#### ISSUE-12: No integration tests with actual file processing
**Severity**: High | **Component**: `tests/integration/`

**Problem**: Integration tests mock everything. No real file I/O or pipeline execution tests.

**Recommended fix**:
- Add integration tests with `tmp_path` fixtures creating actual audio files
- Test full pipeline with real (but small) audio

**Priority**: P1

---

#### ISSUE-13: Training tests mock filesystem
**Severity**: Low | **Component**: `tests/unit/test_training.py`

**Problem**: All training unit tests patch `MODELS_DIR` and `VERSION_INDEX_FILE`. No test verifies actual file creation.

**Recommended fix**:
- Add integration test with real temp directory

**Priority**: P2

---

### 16.7 Small Improvements

| ID | Component | Issue | Fix |
|----|-----------|-------|-----|
| IMPROVE-1 | `app/services/training.py` | Hardcoded `MODELS_DIR` | Use env var `MODELS_DIR` |
| IMPROVE-2 | `app/api/training.py` | No request validation | Validate `num_recordings > 0` |
| IMPROVE-3 | `app/api/training.py` | No logging in `activate_version`, `delete_version` | Add `logger.info` |
| IMPROVE-4 | `app/services/recordings/file_storage.py` | `xiao_s` contains underscore, parsing edge case | Add unit test for underscore in persona_id |
| IMPROVE-5 | `app/services/recordings/pipeline.py` | Processing step status only has "done"/"failed" | Add "skipped" status |

---

### 16.8 Priority Summary

> All issues resolved as of 2026-03-29 (commits 57cfc5c, 2ab187d, 08bd30c, 44db83f)

| Priority | Issues | Status | Fix |
|----------|--------|--------|-----|
| **P0** | ISSUE-1, ISSUE-3, ISSUE-5, ISSUE-6, ISSUE-8, ISSUE-9 | ✅ Fixed | Caching, retry, pagination, stub docs, cleanup endpoint, training check |
| **P1** | ISSUE-4, ISSUE-7, ISSUE-10, ISSUE-12 | ✅ Fixed | Skipped status, recording_ids list, streaming upload, integration tests |
| **P2** | ISSUE-2, ISSUE-11, ISSUE-13, IMPROVE-1~5 | ✅ Fixed | Design doc, parallel TODO, filesystem tests, env var, logging |
| **Known** | pyannote.audio torch conflict | ⚠️ Documented | Must restore torch after pyannote install |

---

## 17. Implementation Phases

### Phase 1: Foundation (2 days)
1. File storage layer (folder naming, JSON metadata)
2. Basic upload API + UI
3. List/delete UI
4. Basic playback
5. Unit tests for storage layer

### Phase 2: WebRTC Recording (1 day)
1. Browser MediaRecorder integration
2. dB meter component
3. Recording state management
4. Integration tests for recording

### Phase 3: Quality Check (1 day)
1. Audio quality analysis module
2. Real-time quality display during recording
3. Post-upload quality check
4. Quality warning UI
5. Unit + integration tests

### Phase 4: Processing Pipeline (2 days)
1. rnnoise integration
2. speechbrain integration
3. pyannote integration
4. Whisper transcription
5. Progress tracking + debug panel
6. Full integration tests

### Phase 5: Training + Versioning (1 day)
1. Training trigger API
2. Background task handling
3. Version management
4. UI integration
5. Metrics tracking
