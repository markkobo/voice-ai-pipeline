# RFC: MVP Redesign — Recording/Persona/Training UI + Data Model

**Status**: Draft | **Target**: MVP Demo | **Date**: 2026-03-31

---

## 1. Overview

This RFC redesigns the MVP UI pages and core data model to support:

1. **Multi-persona household**: Multiple family members, each trainable as a separate voice model
2. **Multi-listener**: Recordings tagged with who is being spoken to, affects emotion tag routing at inference
3. **Per-speaker training**: One recording → multiple speaker segments → train individual voices independently
4. **Mobile/tablet-friendly**: All three pages work on touch devices

**Key design decisions already confirmed with client:**
- Each speaker segment = one persona (one person's voice in multiple recordings → aggregate for training)
- Listener affects emotion routing only (not voice model)
- Persona = fixed household set; Listener = family + dynamic (students, audience, reporter)
- Recording page: read-write recordings + persona/listener editing; read-only models
- Training page: model file management (nickname, delete, activate); read-only recordings/segments
- Conversation page: persona + model version selection with nicknames; read-only (all settings done before conversation)

---

## 2. Data Model

### 2.1 Persona

A **Persona** = one fixed household voice identity. Each persona can have multiple trained versions.

```json
{
  "persona_id": "xiao_s",
  "name": "小S",
  "type": "fixed",
  "is_family": true,
  "created_at": "2026-03-01T00:00:00Z"
}
```

- `type`: `"fixed"` (family member, cannot be deleted if has trained versions) | `"dynamic"` (guest, temporary)
- `is_family`: true for household members, false for others
- Fixed personas: xiao_s, caregiver, elder_gentle, elder_playful
- Dynamic personas can be created by client for guests/temporary users

### 2.2 Listener

A **Listener** = who is being spoken to in a recording / during conversation. Affects emotion tag selection at inference.

```json
{
  "listener_id": "child",
  "name": "小孩",
  "is_family": true,
  "default_emotion": "撒嬌",
  "created_at": "2026-03-01T00:00:00Z"
}
```

- `is_family`: true for family members, false for general public (student, reporter, audience)
- `default_emotion`: default emotion tag when this listener is selected (e.g., child → 撒嬌)
- Listeners are dynamically managed (add/remove) by client
- Pre-seeded with: child, mom, dad, friend, elder, reporter, default

### 2.3 Raw Recording

A **Raw Recording** = original uploaded/recorded audio file.

```json
{
  "recording_id": "uuid",
  "persona_id": "xiao_s",
  "listener_id": "child",
  "title": "20260331_趣味問答",
  "folder_name": "default_xiao_s_20260329_231356",
  "raw_audio_path": "data/recordings/raw/{folder_name}/audio.wav",
  "duration_seconds": 45.2,
  "status": "raw | processing | processed | failed",
  "error_message": null,
  "created_at": "2026-03-29T23:13:56Z"
}
```

### 2.4 Parsed Speaker Segment

A **Parsed Speaker Segment** = one speaker's turn within a recording, extracted by the diarization pipeline.

```json
{
  "segment_id": "uuid",
  "recording_id": "parent-uuid",
  "speaker_index": "SPEAKER_00",
  "persona_id": "xiao_s",
  "listener_id": "child",
  "audio_path": "data/recordings/raw/{folder}/speakers/SPEAKER_00.wav",
  "start_time": 0.0,
  "end_time": 12.5,
  "duration_seconds": 12.5,
  "transcription": "好啦～不要生氣嘛",
  "transcription_confidence": 0.92,
  "quality_score": 0.85,
  "training_ready": true
}
```

- `persona_id`: the identity of this speaker (must match a Persona)
- `listener_id`: who was being spoken to
- `training_ready`: true if quality meets minimum threshold
- Multiple segments from the same recording can belong to different personas

### 2.5 Training Version

A **Training Version** = one trained voice model for one persona, with a client-editable nickname.

```json
{
  "version_id": "v3_20260331_143022",
  "persona_id": "xiao_s",
  "nickname": "最新測試版",
  "status": "training | ready | failed",
  "lora_path": "data/models/xiao_s_v3_20260331_143022",
  "merged_path": "data/models/merged_qwen3_tts_xiao_s_v3",
  "rank": 16,
  "num_epochs": 10,
  "batch_size": 4,
  "final_loss": 0.042,
  "total_audio_seconds": 120.5,
  "num_segments": 8,
  "recording_ids_used": ["uuid-1", "uuid-2"],
  "created_at": "2026-03-31T14:30:22Z",
  "completed_at": "2026-03-31T15:30:00Z"
}
```

- `nickname`: client-editable display name (e.g., "第一次訓練", "高質量版")
- Default nickname: `v{n}`

---

## 3. Page Responsibilities

### 3.1 Recording Page (`/ui/recordings`)

**Role**: Recording management + persona/listener editing (read-write for recordings, read-only for models)

**Functions:**
- [x] WebRTC recording with persona/listener selection
- [x] File upload with persona/listener selection
- [x] **NEW**: Playback bar with progress, play/pause/stop controls
- [x] **NEW**: Parse / Re-parse button (re-triggers pipeline)
- [x] **NEW**: Tree view of parsed speaker segments
- [x] **NEW**: Inline editing of speaker's persona_id + listener_id
- [x] **NEW**: Jump to Training page button
- [ ] Listener CRUD (add/remove from dropdown)
- [ ] Persona CRUD (add/remove dynamic personas)

**Read-only**: Model versions, training settings

### 3.2 Training Page (`/ui/training`)

**Role**: Model file management (read-only for recordings, read-write for model metadata)

**Functions:**
- [x] Select persona target
- [x] **NEW**: Listener filter for recordings
- [x] **NEW**: Tree view of recordings → speaker segments (read-only display)
- [x] **NEW**: Multi-select segments for training
- [x] **NEW**: Edit model nickname (inline)
- [x] **NEW**: Delete model version
- [x] **NEW**: Preview model (synthesize test phrase)
- [x] **NEW**: Jump to Conversation page button
- [ ] Version comparison (future)
- [ ] Aggregate same-persona segments across recordings (future)

**Read-only**: Speaker segments, recording metadata

### 3.3 Conversation Page (`/ui`)

**Role**: Voice dialogue (read-only settings)

**Functions:**
- [x] **NEW**: Model version selector with nicknames (e.g., "v3: 最新測試版")
- [x] Persona selector (already existed)
- [x] Listener selector (already existed)
- [x] **NEW**: VAD sensitivity selector (already wired)
- [x] **NEW**: "按住對話" → "開始/停止對話" (start/stop toggle)
- [ ] Active model version indicator
- [ ] Emotion display per response

**Read-only**: All settings finalized before conversation

### 3.4 Settings (inline in Recording page)

**Persona Management:**
- Fixed personas (family): shown but cannot be deleted
- Dynamic personas: add new / delete
- Inline add button next to persona selector

**Listener Management:**
- Add new listener (name, is_family, default_emotion)
- Delete unused listener
- Inline add button next to listener selector

---

## 4. UI Design

### 4.1 Tree View — Recording Page

Hierarchical folder structure: **Recording Folder → Speaker Segments**

```
📁 20260331_趣味問答 [xiao_s → 對 child說]  45s  ⏵ ⏸ ⏹  [Parse]  [Delete]
  📂 Speakers (2 found)
  ├─ 👤 SPEAKER_00 [小S ▼] [對 小孩 ▼]  12.5s  ⏵  [品質: ✓ 0.85]  [轉寫: 好啦～不要生氣嘛...]
  │   波形 ████████████░░░░░░░░░ 0:05 / 0:12
  ├─ 👤 SPEAKER_01 [媽媽 ▼] [對 小孩 ▼]   8.3s  ⏵  [品質: ✓ 0.78]  [轉寫: 那你問我啊...]
  │   波形 ████████░░░░░░░░░░░░░░ 0:03 / 0:08
  └─ 👤 SPEAKER_02 [小S ▼] [對 小孩 ▼]  24.2s  ⏵  [品質: ✓ 0.91]  [轉寫: 好～我來出題目...]
      波形 ████████████████████░░ 0:15 / 0:24

📁 20260330_睡前故事 [xiao_s → 對 elder說]  30s  ⏵  [Re-parse]  [Delete]
  📂 Speakers (1 found)
  └─ 👤 SPEAKER_00 [小S ▼] [對 長輩 ▼]  30.0s  ⏵  [品質: ✓ 0.88]  [轉寫: 今天要講什麼故事呢...]

[+ 跳轉到 Training 頁面]
```

**Interaction:**
- Click folder row → expand/collapse speaker list
- Play button → play segment audio (with stop/pause)
- Dropdowns → inline edit persona + listener for this segment
- Parse button (raw/processing/failed only) → re-triggers pipeline
- Quality badge: green ✓ if training_ready, yellow ⚠ if quality < threshold
- Transcription preview: truncated text, click to expand

**Mobile/Tablet:** Tree view collapses by default on small screens, one tap to expand recording and see speakers.

### 4.2 Playback Bar (per segment)

```
[⏵] ──[████████░░░░░░░░░]── 0:05 / 0:12 ──[⏸]──[🔊 ████░░]
```

- Play / Pause toggle
- Progress seekbar (click to seek)
- Stop button (reset to beginning)
- Volume control

### 4.3 Training Page — Segment Selection

```
[選擇訓練目標]
Persona: [小S (xiao_s) ▼]  [+ 新增 persona]

[過濾錄音]
聆聽者: [全部 ▼]  [只顯示可用於訓練 ✓]

📁 20260331_趣味問答  [3 segments, 全部展開]
  ├─ ☑ SPEAKER_00 (小S, 對小孩)  12.5s  品質 0.85  轉寫: 好啦～不要生氣嘛...
  ├─ ☑ SPEAKER_01 (媽媽, 對小孩)   8.3s  品質 0.78  轉寫: 那你問我啊...
  └─ ☑ SPEAKER_02 (小S, 對小孩)  24.2s  品質 0.91  轉寫: 好～我來出題目...

📁 20260330_睡前故事  [1 segment]
  └─ ☑ SPEAKER_00 (小S, 對長輩)  30.0s  品質 0.88  轉寫: 今天要講什麼故事呢...

[Training Settings]
Epochs: [10 ▼]   LoRA Rank: [16 ▼]   Batch Size: [4 ▼]

[Training Preview]
將訓練「小S」:
• 20260331_趣味問答 - SPEAKER_00 + SPEAKER_02 (36.7s, 品質 0.85+)
• 20260330_睡前故事 - SPEAKER_00 (30.0s, 品質 0.88)
────────────────────────────────────────
總計: ~66.7s, 2 個錄音, 3 個片段
⚠️ 建議至少 10s

[開始訓練]
```

**Segment cards (read-only):**
- Checkbox to select/deselect
- Persona + listener shown (not editable)
- Quality score + transcription preview
- Play button for audio preview

### 4.4 Model Version Management (Training Page — Versions Tab)

```
[xiao_s 的版本]

┌──────────────────────────────────────────────────────┐
│ ✏️ v3: 第一次訓練                              [啟用]│
│ 訓練時間: 2026-03-31 15:30                          │
│ 音頻: 66.7s | Loss: 0.042 | Epochs: 10 | Rank: 16  │
│ 片段: 3 個錄音, 3 個片段                            │
│                                              [▶ 預覽]│
│                                              [✕ 刪除]│
└──────────────────────────────────────────────────────┘

[+ 跳轉到 Conversation 頁面]
```

- Click nickname to edit inline
- Activate button: sets this version as active for inference
- Preview: synthesize test phrase and stream audio
- Delete: removes version (warns if active)

### 4.5 Conversation Page — Start/Stop Toggle

```
┌─────────────────────────────────────────────┐
│  🎙️ 聲音版本: [v3: 第一次訓練 ▼]  [ⓘ]        │
│  👤 聆聽者: [小孩 ▼]                         │
│  🎭 人格: [小S ▼]                            │
│  🔇 VAD: [中 ▼]                              │
├─────────────────────────────────────────────┤
│                                             │
│         [ 🎤  開始對話 ]                     │
│                                             │
│  (or when active: 對話中... [停止])         │
└─────────────────────────────────────────────┘
```

- "開始對話" button → connects WebSocket, starts audio capture
- Button changes to "停止對話" during active conversation
- Model version dropdown shows nickname (e.g., "v3: 第一次訓練")
- Hover on version info icon (ⓘ) → shows metadata (loss, duration, date)

---

## 5. API Changes

### 5.1 New/Modified Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/personas` | List all personas |
| `POST` | `/api/personas` | Create dynamic persona |
| `DELETE` | `/api/personas/{id}` | Delete dynamic persona |
| `GET` | `/api/listeners` | List all listeners |
| `POST` | `/api/listeners` | Create listener |
| `PATCH` | `/api/listeners/{id}` | Update listener |
| `DELETE` | `/api/listeners/{id}` | Delete listener |
| `GET` | `/api/recordings/{id}/segments` | Get all speaker segments for recording |
| `PATCH` | `/api/recordings/segments/{segment_id}` | Update segment persona_id/listener_id |
| `GET` | `/api/training/versions/{id}` | (existing + nickname) |
| `PATCH` | `/api/training/versions/{id}` | Update nickname |
| `POST` | `/api/recordings/{id}/parse` | Re-parse recording |

### 5.2 Segment Selection for Training

Training now selects by **segment_id** (not recording_id):

```python
POST /api/training/versions
{
  "persona_id": "xiao_s",
  "segment_ids": ["seg-uuid-1", "seg-uuid-2", "seg-uuid-3"],
  "rank": 16,
  "num_epochs": 10,
  "batch_size": 4
}
```

Backend aggregates segments by persona_id, validates all belong to same persona, checks minimum duration.

---

## 6. Emotion Routing (No Voice Change for Different Listeners)

**Industry consensus**: Voice cloning =音色 clone, NOT "voice changes based on listener."

The architecture for listener-aware responses:

```
Input: "今天考試考怎麼樣？"
                    ↓
        Persona (who is speaking) → Voice LoRA
        Listener (who is listening) → Emotion tag routing
                    ↓
        LLM receives: "[情感: 撒嬌] 今天考試考怎麼樣？"
        (emotion tag determined by listener_id + context)
                    ↓
        TTS: 小S voice LoRA + "(coquettish, soft, slightly slower pace)"
```

**Listener does NOT change the voice model.** It only influences which emotion tag the LLM is prompted to use. This is the industry standard approach.

---

## 7. Implementation Phases

### Phase 1: Core Data Model + Recording Tree View
1. Add Persona/Listener API endpoints
2. Extend RecordingMetadata with segment support
3. Add `/api/recordings/{id}/segments` endpoint
4. Rebuild Recording page with Tree View UI
5. Add playback bar (play/pause/stop/seek) for segments
6. Inline persona/listener editing for segments
7. Add Parse / Re-parse button
8. Add listener CRUD (add inline in recordings page)

### Phase 2: Training Read-Only View + Model Management
1. Training page: show read-only tree of recordings + segments
2. Add listener filter to training recordings
3. Multi-select segments (not recording-level)
4. Add model nickname editing (PATCH endpoint)
5. Delete version endpoint + confirmation
6. Add "Jump to Conversation" button

### Phase 3: Conversation Page Refinements
1. Model version dropdown with nicknames
2. "按住對話" → "開始/停止對話" toggle button
3. Version info tooltip (ⓘ)
4. Active model version indicator

### Phase 4: Persona/Listener Settings
1. Persona CRUD (fixed ones non-deletable)
2. Listener CRUD (inline in Recording page)
3. Default emotion per listener

---

## 8. Storage Structure

```
data/
├── recordings/
│   ├── raw/
│   │   └── {folder_name}/
│   │       ├── audio.wav            # Original upload
│   │       └── speakers/
│   │           ├── SPEAKER_00.wav  # Extracted segment
│   │           ├── SPEAKER_01.wav
│   │           └── metadata.json   # Per-segment metadata
│   └── metadata/
│       └── {recording_id}.json      # RecordingMetadata
├── personas/
│   └── personas.json                # Persona definitions
├── listeners/
│   └── listeners.json               # Listener definitions
└── models/
    ├── index.json                   # TrainingVersion index
    └── {persona_id}_v{n}_{timestamp}/
        ├── adapter.pt               # LoRA weights
        ├── manifest.json            # segment_ids used
        └── training.log
```

---

## 9. Open Questions

| Question | Decision |
|----------|----------|
| Delete recording with trained versions | Warn but allow; version keeps audio path reference |
| Aggregate same persona segments automatically | No — manual multi-select per training run |
| Minimum segments per training | 1 segment (minimum 10s total audio) |
| Version nickname uniqueness | Not required; just for display |
| Dynamic persona cleanup when deleted | Remove from any recordings/segments; warn if version exists |

---

## 10. Milestones

- [x] M2.1: Persona/Listener API + CRUD
- [x] M2.2: Recording segment API + Tree View UI
- [x] M2.3: Playback bar with seek + Parse/Re-parse
- [x] M2.4: Inline persona/listener editing for segments
- [x] M3.1: Training page read-only tree view + listener filter
- [x] M3.2: Segment multi-select for training
- [x] M3.3: Model nickname edit + delete
- [x] M3.4: "Jump to Conversation" button
- [x] M4.1: Conversation model version dropdown with nicknames
- [x] M4.2: Start/Stop toggle button
- [x] M4.3: Version info tooltip
