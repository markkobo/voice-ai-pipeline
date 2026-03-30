# RFC: Milestone 4 — LoRA Fine-tuning Pipeline

**Status**: Draft | **Target**: MVP Demo | **Milestone**: M4

---

## 1. Overview

**Goal**: 實現 LoRA fine-tuning pipeline，讓 TTS 聲音越來越像 client 本人。

**Duration**: ~2 weeks

---

## 2. Training Strategy

### 2.1 Fresh Training (Recommended)

每次 training 都是獨立的完整版本，不使用 incremental training。

```
V1: train [rec_A, rec_B, rec_C] → LoRA v1
V2: train [rec_A, rec_B, rec_C, rec_D] → LoRA v2 (全新)
V3: train [rec_A, rec_B, rec_C, rec_D, rec_E] → LoRA v3 (全新)
```

**為什麼不用 Incremental:**
- 語音 identity 需要從頭到尾一致
- Incremental 可能導致 voice drift
- 刪除某個錄音 = train 新版本排除它
- 實作簡單，沒有 checkpoint/merge 管理

### 2.2 Storage Structure

```
data/models/
├── index.json                    # VersionManager 的 index
├── v1_20260329_143022/          # Version 目錄
│   ├── manifest.json            # 訓練用的錄音清單
│   ├── adapter.pt               # LoRA weights
│   ├── config.json              # Training config
│   └── training.log             # 訓練日誌
├── v2_20260330_090123/
│   ├── manifest.json
│   ├── adapter.pt
│   └── ...
```

### 2.3 manifest.json Structure

```json
{
  "version_id": "v1_20260329_143022",
  "persona_id": "xiao_s",
  "recordings": [
    {
      "recording_id": "uuid-1",
      "folder_name": "default_xiao_s_20260329_231356",
      "speaker_used": "SPEAKER_00",
      "audio_path": "data/recordings/raw/default_xiao_s_20260329_231356/speakers/SPEAKER_00.wav",
      "duration_seconds": 45.2,
      "transcription": "好啦～不要生氣嘛"
    },
    {
      "recording_id": "uuid-2",
      "folder_name": "default_xiao_s_20260329_232148",
      "speaker_used": null,
      "audio_path": "data/recordings/enhanced/default_xiao_s_20260329_232148/audio.wav",
      "duration_seconds": 30.5,
      "transcription": "..."
    }
  ],
  "total_duration_seconds": 75.7,
  "training_config": {
    "rank": 16,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "batch_size": 4
  },
  "created_at": "2026-03-29T14:30:22",
  "completed_at": "2026-03-29T15:30:00",
  "final_loss": 0.045
}
```

---

## 3. Data Model

### 3.1 Recording Metadata (已擴展)

```python
# app/services/recordings/metadata.py
{
    "recording_id": "uuid",
    "listener_id": "child|mom|dad|friend|reporter|elder|default",
    "persona_id": "xiao_s|caregiver|elder_gentle|elder_playful",
    "speaker_segments": [
        {"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 5.2},
        {"speaker_id": "SPEAKER_01", "start_time": 5.2, "end_time": 10.5},
    ],
    "speaker_labels": {
        "SPEAKER_00": "xiao_s",
        "SPEAKER_01": "mom"
    }
}
```

### 3.2 TrainingVersion (已存在)

```python
@dataclass
class TrainingVersion:
    version_id: str
    persona_id: str
    status: str  # "training", "ready", "failed"
    lora_path: Optional[str]
    rank: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 4
    final_loss: Optional[float]
    training_time_seconds: Optional[int]
    recording_ids_used: list[str]
    created_at: Optional[str]
    completed_at: Optional[str]
```

### 3.3 TrainingProgress (New)

```python
@dataclass
class TrainingProgress:
    version_id: str
    status: str  # training/ready/failed
    current_epoch: int
    total_epochs: int
    current_loss: float
    best_loss: float
    elapsed_seconds: int
    eta_seconds: int
    last_updated: datetime
```

---

## 4. Functional Requirements

### 4.1 Core Training

| ID | Requirement | Description |
|----|-------------|-------------|
| FR-T1 | Select Persona for Training | 選擇要訓練哪個人格的聲音 |
| FR-T2 | Multi-Recording Selection | 選擇多個錄音一起訓練 |
| FR-T3 | Per-Speaker Selection | 多人錄音中選擇使用哪個 speaker |
| FR-T4 | Training Settings | Client 可選 epochs, rank, batch_size |
| FR-T5 | Minimum Audio Check | 至少 10s 音頻才能訓練 |
| FR-T6 | Training Progress | 即時顯示 epoch, loss |
| FR-T7 | Background Training | Training 在背景執行，不 blocking UI |
| FR-T8 |断线重连 | 断线後可查詢 training status |
| FR-T9 | Version Management | List/Activate/Delete versions |
| FR-T10 | TTS Integration | Training 完成後 TTS 使用 LoRA |

### 4.2 Speaker Labeling Flow

```
Recording (多人對話)
  → Processing Pipeline (denoise/enhance/diarize)
  → Speaker Extraction (每個 speaker 保存為 SPEAKER_XX.wav)
  → User 手動標記: SPEAKER_00 = xiao_s, SPEAKER_01 = mom
  → Training 時: 只取標記為 target persona 的 speaker audio
```

---

## 5. API Endpoints

### 5.1 Training Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/training/versions` | Create & start training |
| `GET` | `/api/training/versions` | List all versions |
| `GET` | `/api/training/versions/{id}` | Get version details |
| `GET` | `/api/training/versions/{id}/progress` | SSE stream for progress |
| `POST` | `/api/training/versions/{id}/activate` | Activate version |
| `DELETE` | `/api/training/versions/{id}` | Delete version |
| `GET` | `/api/training/versions/{id}/manifest` | Get training manifest |
| `GET` | `/api/training/status` | Get current training status |
| `GET` | `/api/training/active?persona_id=xiao_s` | Get active version |

### 5.2 Recording Endpoints (已擴展)

| Method | Path | Description |
|--------|------|-------------|
| `PATCH` | `/api/recordings/{id}/speakers` | Update speaker labels |
| `GET` | `/api/recordings/{id}/speakers` | Get speaker info |

### 5.3 Progress SSE Endpoint

```
GET /api/training/versions/{id}/progress

SSE Stream:
data: {"event": "progress", "epoch": 3, "loss": 0.12, "step": 150, "best_loss": 0.10}
data: {"event": "progress", "epoch": 3, "loss": 0.11, "step": 160, "best_loss": 0.10}
data: {"event": "complete", "final_loss": 0.05, "training_time": 1800}
data: {"event": "error", "error": "CUDA OOM"}
```

Client 斷線後可 GET `/api/training/versions/{id}` 取得當前狀態。

---

## 6. Training Selection UI

### 6.1 Main Flow

```
[選擇 Training Target]
  Persona: [xiao_s ▼]

[可用於 xiao_s 訓練的錄音]
  Filter: [全部 ▼] [已處理 ✓]

  ┌─ Recording Card ─────────────────────────────────┐
  │ default_xiao_s_20260329_231356        [2 speakers] │
  │ 45s | processed | quality: ✓                       │
  │                                                   │
  │ Speakers:                                         │
  │   SPEAKER_00: xiao_s    [☑ Include]  [▶] [▶]    │
  │   SPEAKER_01: mom       [☐ Exclude]  [▶]        │
  │                                                   │
  │ Transcription: 好啦～不要生氣嘛...                 │
  │                                        [☑ Select] │
  └───────────────────────────────────────────────────┘

  ┌─ Recording Card ─────────────────────────────────┐
  │ child_xiao_s_20260330_110000          [1 speaker]│
  │ 30s | processed | quality: ✓                       │
  │                                                   │
  │   Full audio (no speaker split)                   │
  │                                        [☑ Select] │
  └───────────────────────────────────────────────────┘

[Training Settings]
  Epochs: [10 ▼]   LoRA Rank: [16 ▼]   Batch Size: [4 ▼]

[Training Preview]
  將訓練 xiao_s:
  • default_xiao_s_20260329_231356 (SPEAKER_00) - 45s
  • child_xiao_s_20260330_110000 (full) - 30s
  ─────────────────────────────
  總計: ~75s, 2 個錄音

  ⚠️ 建議至少 10s 音頻用於訓練

  [開始訓練]
```

### 6.2 Per-Speaker Selection Modal

```
Recording: default_xiao_s_20260329_231356
Speakers found:

  SPEAKER_00 [xiao_s ▼]  [☑ Include in training]
    30s audio | 5 segments
    [▶ 播放這段] [查看文字稿]

  SPEAKER_01 [mom ▼]    [☐ Exclude]
    25s audio | 4 segments
    [▶ 播放這段]

  [全部使用於 xiao_s 訓練] [全部排除]
```

### 6.3 Version/Model Summary Page

```
[Models > xiao_s]

Versions:
┌─────────────────────────────────────────────────────────┐
│ v3 (current active)                          [Activate]│
│ Trained: 2026-03-30 15:30                               │
│ Duration: 120s (5 recordings)                          │
│ Loss: 0.042 | Epochs: 10 | Rank: 16                    │
│ Recordings: default_xiao_s..., child_xiao_s..., ...     │
│                                          [▶ Preview]   │
│                                          [✕ Delete]    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ v2                                        [Activate]    │
│ Trained: 2026-03-29 10:30                               │
│ Duration: 75s (3 recordings)                           │
│ Loss: 0.055 | Epochs: 10 | Rank: 16                    │
└─────────────────────────────────────────────────────────┘

Recording → Version Mapping:
  default_xiao_s_20260329_231356
    └── SPEAKER_00 → v1, v2, v3 (included)

  child_xiao_s_20260330_110000
    └── full audio → v3 only
```

---

## 7. Training Settings

### 7.1 Client-Selectable Options

| Setting | Options | Default |
|---------|---------|---------|
| Epochs | 1, 5, 10, 20, 30, 50 | 10 |
| LoRA Rank | 4, 8, 16, 32 | 16 |
| Batch Size | 1, 2, 4, 8 | 4 |

### 7.2 Fixed Settings

```python
learning_rate = 1e-4
warmup_steps = 100
gradient_accumulation_steps = 4
base_model = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
```

---

## 8. Implementation Components

### 8.1 Files to Create/Modify

```
app/
├── services/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── version_manager.py    # Modify: add manifest support
│   │   ├── lora_trainer.py       # NEW: LoRA training logic
│   │   ├── training_job.py       # NEW: Background job runner
│   │   └── progress_tracker.py   # NEW: Progress tracking
│   └── recordings/
│       └── metadata.py           # Modify: speaker_labels already done
├── api/
│   ├── training.py               # Modify: add SSE, manifest endpoints
│   └── recordings.py             # Modify: speaker endpoints already done
└── services/tts/
    └── qwen_tts_engine.py        # Modify: load LoRA adapter
```

### 8.2 LoRA Trainer Interface

```python
class LoraTrainer:
    def __init__(
        self,
        version_id: str,
        persona_id: str,
        audio_paths: list[Path],
        config: TrainingConfig,
    ):
        ...

    def train(self) -> TrainingResult:
        """Run training, return result."""
        ...

    def get_progress(self) -> TrainingProgress:
        """Get current progress."""
        ...

    def cancel(self):
        """Cancel ongoing training."""
        ...
```

### 8.3 Background Job Runner

```python
class TrainingJob:
    """Background training job runner."""

    def __init__(self, version_id: str):
        self.version_id = version_id
        self._process = None

    def start(self):
        """Start training in background process."""
        ...

    def poll(self) -> Optional[TrainingProgress]:
        """Poll for progress."""
        ...

    def cancel(self):
        """Cancel training."""
        ...
```

---

## 9. Training Flow

```
1. User 選擇 persona_id (e.g., "xiao_s")
2. System filter 出所有 persona_id=xiao_s 的 recordings
3. User 選擇要訓練的 recordings (多選)
4. 對於每個 selected recording:
   - 如果有 speaker_labels → 顯示 speaker 列表，user 選擇
   - 如果沒有 → 直接用整個音頻
5. User 選擇 Training Settings (epochs, rank, batch_size)
6. User 點 "開始訓練"
7. Backend:
   a. 創建 TrainingVersion (status=training)
   b. 寫入 manifest.json
   c. 啟動 background training job
   d. Job 定期寫入 progress.json
   e. SSE 推送 progress 到 client
8. 訓練完成 → status=ready, final_loss 更新
9. User activate version → TTS 使用 LoRA adapter
```

---

## 10.断线重连 Strategy

```
Training 開始
  → Client 連接 SSE /api/training/versions/{id}/progress
  → 接收 progress updates

Client 斷線
  → Training 繼續在 server 執行
  → Progress 寫入 progress.json

Client 重連
  → GET /api/training/versions/{id}
    → 如果 training: 返回當前 progress, status=training
    → 如果 ready: 返回 final_loss, status=ready
  → Client 可重新連接 SSE 取得後續 updates
```

---

## 11. TTS Integration

### 11.1 Loading LoRA at Runtime

```python
# app/services/tts/qwen_tts_engine.py

class QwenTTSEngine:
    def __init__(self):
        self.current_lora_path = None
        self.model = None

    def activate_version(self, version_id: str):
        """Load LoRA adapter for version."""
        version = get_version_manager().get_version(version_id)
        if not version or version.status != "ready":
            raise ValueError(f"Version {version_id} not ready")

        self.current_lora_path = version.lora_path
        # Reload model with LoRA
        self._load_model_with_lora()

    def _load_model_with_lora(self):
        """Load base model + LoRA adapter."""
        # ...
```

### 11.2 Inference with LoRA

```python
def synthesize(self, text, emotion_instruct):
    if self.current_lora_path:
        # Use LoRA adapter
        audio = self.model.generate(
            text,
            adapter_path=self.current_lora_path,
            emotion=emotion_instruct,
        )
    else:
        # Use base model
        audio = self.model.generate(text, emotion=emotion_instruct)
    return audio
```

---

## 12. Error Handling

| Scenario | Handling |
|----------|----------|
| CUDA OOM | Training fails → status=failed, log error |
| Recording file missing | Skip file, log warning, continue if still enough audio |
| Training interrupted | Clean up partial files, status=failed |
| No valid audio found | Reject training at start with clear error |
| LoRA load failure | Fall back to base model, log warning |

---

## 13. Testing Strategy

### 13.1 Unit Tests

- `test_version_manager`: CRUD operations
- `test_training_selection`: persona/recording filtering
- `test_speaker_label_filtering`: correct audio paths selected
- `test_progress_tracking`: progress.json read/write

### 13.2 Integration Tests

- `test_training_flow`: full training with mock audio
- `test断线重连`: SSE disconnect/reconnect
- `test_tts_integration`: LoRA activation + synthesis

---

## 14. Open Questions

| Question | Decision | Notes |
|----------|----------|-------|
| Epochs 範圍 | Client 可選 1-50 | 成熟後可隱藏進階選項 |
| LoRA Rank 範圍 | Client 可選 4, 8, 16, 32 | Qwen3-TTS 官方建議 |
| Loss curve display | 暫時不做 | 以後再加 |
| 刪除 Recording warning | 要 | Warn 如果存在於 version 中 |

---

## 15. Milestones

- [x] M4.1: RFC 規劃 (本文檔)
- [ ] M4.2: Training API endpoints + SSE
- [ ] M4.3: Version Manager 擴展 (manifest)
- [ ] M4.4: LoRA Trainer implementation
- [ ] M4.5: Background job runner
- [ ] M4.6: Training Selection UI
- [ ] M4.7: Model Summary UI
- [ ] M4.8: TTS LoRA integration
- [ ] M4.9: Integration tests
