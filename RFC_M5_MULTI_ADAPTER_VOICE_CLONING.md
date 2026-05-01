# RFC: Milestone 5 — Multi-Adapter Voice Cloning (Personal Legacy AI)

**Status**: Draft | **Target**: Personal Voice Legacy | **Date**: 2026-04-16

---

## 1. Vision

**Personal Legacy AI**: Preserve and recreate a person's voice and speaking mannerisms for different audiences — a father reading bedtime stories to his child vs. talking to his wife vs. professional presentations.

The system learns **separate voice adapters per listener category**, so each AI response sounds like the real person speaking naturally to that specific listener — not a flat "average" voice, not a generic character.

---

## 2. Core Problem (The "Average Voice" Curse)

If you mix all audio into one SFT/LoRA training:
- Bedtime stories (gentle) + arguments (intense) + meetings (neutral) → model produces a flat "somewhere in between" voice
- Child hears: "a monotone dad who sounds like he's in a business meeting while reading stories"
- Wife hears: "a weirdly animated version of dad who doesn't sound intimate"

**Solution**: Separate voice adapters per listener, trained on context-pure audio only.

---

## 3. Immediate Next Step: One Voice First

Before building multi-adapter infrastructure, we need **one working voice** to:
1. Verify SFT vs LoRA quality on this specific voice
2. Establish baseline for comparison
3. Have a reference implementation to extend

**Action**: Run SFT training on current recordings. Compare output. Then decide on architecture.

---

## 4. Data Model: Listener-Categorized Audio

### 4.1 Existing Model (Already Implemented)

The system already supports `listener_id` on recordings and segments:

```json
// Segment model (already exists in RFC_MVP_REDESIGN)
{
  "segment_id": "uuid",
  "recording_id": "parent-uuid",
  "persona_id": "xiao_s",
  "listener_id": "child",  // ← key field
  "audio_path": "...",
  "duration_seconds": 30.0,
  "transcription": "今天要講什麼故事呢...",
  "training_ready": true
}
```

### 4.2 Listener Categories

Pre-seeded listeners (from RFC_MVP_REDESIGN):
- `child` — talking to children (bedtime stories, games)
- `wife` / `mom` — intimate/conversational
- `professional` — meetings, presentations
- `friend` — casual conversation
- `elder` — respectful, slower pace
- `default` — fallback

### 4.3 Storage Structure Per Adapter

```
data/models/
├── index.json                              # TrainingVersion index (already exists)
├── xiao_s_v1_20260416_000000/             # Version directory
│   ├── manifest.json                       # Training config + segment_ids
│   ├── adapter.pt                          # LoRA weights OR full model
│   └── training.log
├── merged_qwen3_tts_xiao_s_v1/            # Merged model (already exists)
│   └── ...
```

### 4.4 New: Adapter Registry

```json
// data/models/adapter_registry.json
{
  "adapters": [
    {
      "adapter_id": "xiao_s__child",
      "persona_id": "xiao_s",
      "listener_id": "child",
      "model_path": "data/models/merged_qwen3_tts_xiao_s__child",
      "version_id": "v1_20260416_000000",
      "status": "active",
      "total_audio_seconds": 1200,
      "num_segments": 15,
      "created_at": "2026-04-16T00:00:00Z"
    }
  ]
}
```

---

## 5. Architecture: Two Strategic Paths

### 5.1 Path A: Conditioned SFT (Single Model + Labels)

Train one SFT model on all audio, but prepend emotion/context labels to text:

```json
{"text": "[對小孩]今天要講什麼故事呢～", "audio": "kid_story.wav"}
{"text": "[會議]請問各位有什麼問題？", "audio": "meeting.wav"}
```

**Inference**: LLM output includes listener tag → TTS sees tag → switches acoustic style.

| Pros | Cons |
|------|------|
| Single model in memory | Label quality critical — bad labels = confused model |
| Easy to manage | "Regression to mean" still happens within each label |
| VRAM efficient | Requires very clean, consistent labeling |

**VRAM**: ~3.5GB model + activation

### 5.2 Path B: Multi-Adapter Routing (Multiple LoRA/SFT Adapters)

Train separate LoRA or SFT adapter per listener:

```
Listener: child      → LoRA adapter: xiao_s__child (gentle, warm)
Listener: wife      → LoRA adapter: xiao_s__wife (intimate, playful)
Listener: professional → LoRA adapter: xiao_s__professional (clear, neutral)
```

**Inference**: Route to correct adapter based on `listener_id`.

| Pros | Cons |
|------|------|
| 100% voice isolation per listener | Higher VRAM per adapter |
| No cross-contamination of prosody | Training cost multiplied |
| Architecture clean — no reliance on LLM labels | More model files to manage |

**VRAM for LoRA per adapter**: ~3.5GB base + ~0.5GB LoRA + activation ≈ 8-10GB
**VRAM for SFT per adapter**: ~3.5GB base + activation ≈ 12-14GB

With 24GB RTX 4090 + gradient checkpointing + batch_size=1:
- **2 concurrent SFT adapters** feasible
- **4+ concurrent LoRA adapters** feasible

### 5.3 Decision Recommendation

**Recommended**: Path B (Multi-Adapter Routing with LoRA) for initial implementation:
- Fast iteration — train each adapter in ~5-10 min
- Easy to A/B test — swap adapters, compare voice quality
- More robust — no reliance on label consistency
- Upgrade path to SFT per adapter when quality needs it

**Upgrade to SFT per adapter** when: LoRA quality is noticeably "not quite right" compared to real voice.

---

## 6. Training Pipeline Changes

### 6.1 New Training Flow: Listener-Targeted Training

```
User selects:
  - Persona: xiao_s
  - Listener category: child
  - Segments: [seg-uuid-1, seg-uuid-2, ...] (all with listener_id=child)

Backend:
  1. Filter segments by (persona_id, listener_id)
  2. Validate: all segments must have same listener_id
  3. Create TrainingVersion with adapter_id = "{persona_id}__{listener_id}"
  4. Train LoRA/SFT adapter
  5. Register in adapter_registry.json
  6. Activate for inference
```

### 6.2 API Changes

**New endpoint for listener-targeted training:**
```
POST /api/training/versions
{
  "persona_id": "xiao_s",
  "listener_id": "child",           // NEW: which listener this adapter is for
  "segment_ids": ["uuid-1", "uuid-2"],
  "training_type": "lora",          // or "sft"
  "num_epochs": 10,
  "rank": 16
}
```

**Modified GET active version:**
```
GET /api/training/active?persona_id=xiao_s&listener_id=child
→ Returns adapter for specific listener
```

**New endpoint to list adapters:**
```
GET /api/training/adapters?persona_id=xiao_s
→ Returns all adapters for xiao_s, grouped by listener_id
```

### 6.3 UI Changes: Training Page

```
[選擇訓練目標]
Persona: [小S (xiao_s) ▼]
Listener: [對小孩 ▼]          // NEW dropdown

[可用的錄音 — 對小孩說話]
  ☑ seg-uuid-1 (SPEAKER_00)  30s  品質 0.91  轉寫: 今天要講什麼...
  ☑ seg-uuid-2 (SPEAKER_00)  25s  品質 0.88  轉寫: 床邊故事時間...
  ☑ seg-uuid-3 (SPEAKER_00)  20s  品質 0.85  轉寫: 乖, 睡覺...

[Training Settings]
  Training Type: [LoRA ▼]    // NEW: LoRA or SFT selector
  Epochs: [10 ▼]
  Rank: [16 ▼]

[Training Preview]
  將訓練「小S」對「小孩」的聲音:
  • 3 個片段, 共 75s
  • Adapter ID: xiao_s__child

  [開始訓練]
```

### 6.4 UI Changes: Version/Adapter Management

```
[xiao_s 的聲音適配器]

┌──────────────────────────────────────────────────────────┐
│ 🎯 對小孩 (child) — 目前啟用                    [啟用]  │
│ Adapter: xiao_s__child                                  │
│ 訓練時間: 2026-04-16 15:30                              │
│ 音頻: 75s | Loss: 0.042 | Epochs: 10 | Rank: 16       │
│                                               [▶ 預覽]  │
│                                               [✕ 刪除]  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 🎯 對老婆 (wife) — 未訓練                      [訓練]   │
│ 尚未有此 listener 的錄音                                    │
│                                               [▶ 預覽]  │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Inference: Dynamic Adapter Routing

### 7.1 State Manager Changes

```python
# app/core/state_manager.py

class StateManager:
    def __init__(self):
        # Existing: self.active_model_path (single model)
        # New: per-listener adapter routing
        self._adapters: dict[str, str] = {}  # listener_id → adapter_path
        self._active_listener: str = "default"

    def activate_adapter(self, persona_id: str, listener_id: str, version_id: str):
        """Load and activate a listener-specific adapter."""
        adapter_path = self._get_adapter_path(persona_id, listener_id, version_id)
        # Load adapter into TTS engine
        self.tts_engine.load_adapter(adapter_path)
        self._adapters[listener_id] = adapter_path

    def get_adapter_for_listener(self, listener_id: str) -> Optional[str]:
        """Get the active adapter path for a listener."""
        return self._adapters.get(listener_id)
```

### 7.2 TTS Engine Changes

```python
# app/services/tts/qwen_tts_engine.py

class FasterQwenTTSEngine:
    def load_adapter(self, adapter_path: str):
        """Load a specific LoRA/SFT adapter for voice cloning."""
        # If using LoRA: load base model + apply LoRA adapter
        # If using SFT merged: load merged model directly
        ...

    def generate_for_listener(self, text: str, listener_id: str, **kwargs) -> bytes:
        """Generate audio using the appropriate adapter for listener."""
        adapter_path = state_manager.get_adapter_for_listener(listener_id)
        if adapter_path:
            self.load_adapter(adapter_path)
        return self.generate(text, **kwargs)
```

### 7.3 WebSocket Routing

```python
# app/api/ws_asr.py

# On WebSocket connect, client sends listener_id in config:
# {"type": "config", "persona_id": "xiao_s", "listener_id": "child", ...}

# When generating TTS:
if listener_id in state_manager._adapters:
    audio = tts_engine.generate_for_listener(text, listener_id, ...)
else:
    # Fallback to default/base model
    audio = tts_engine.generate(text, ...)
```

---

## 8. Recording/Upload UI for Listener Assignment

### 8.1 Recording Page: Listener Assignment

Current recording UI already has persona/listener dropdowns. No major changes needed.

**Optional enhancement**: Show "training-ready" segments grouped by listener:

```
[對小孩說話] (3 segments, 75s total)
  📁 20260415_睡前故事
    ├─ SPEAKER_00 (小S)  30s  [▶] [Training Ready ✓]

[對老婆說話] (1 segment, 20s total)
  📁 20260414_晚餐對話
    └─ SPEAKER_00 (小S)  20s  [▶] [Training Ready ✓]
```

### 8.2 Bulk Re-tagging

For existing recordings without listener_id:
- Add "Assign Listener" bulk action in Recording page
- Select multiple segments → assign listener category

---

## 9. Work Items by Role

### 9.1 Backend Engineer

| Item | Description | Priority |
|------|-------------|----------|
| **B-1**: Fix and verify SFT training | SFT training is broken (USE_LORA bug fixed, needs testing) | P0 |
| **B-2**: Train one voice (baseline) | Run SFT training on current recordings, produce v1 model | P0 |
| **B-3**: Implement adapter registry | New `adapter_registry.json` + `GET /api/training/adapters` endpoint | P1 |
| **B-4**: Listener-filtered training | Filter segments by `listener_id` in training request | P1 |
| **B-5**: Multi-adapter loading | TTS engine loads specific adapter based on `listener_id` at runtime | P1 |
| **B-6**: State manager routing | Route TTS generation to correct adapter based on `listener_id` | P1 |
| **B-7**: Fallback to default | If no adapter for listener, use default persona adapter | P2 |

### 9.2 Frontend Engineer

| Item | Description | Priority |
|------|-------------|----------|
| **F-1**: Listener dropdown on Training page | Add `listener_id` filter to training segment selection | P1 |
| **F-2**: Adapter management UI | Show all adapters per persona, grouped by listener | P1 |
| **F-3**: Adapter activation | Click to activate specific listener adapter | P1 |
| **F-4**: "Train New Adapter" flow | Guided flow: select persona → select listener → select segments → train | P1 |
| **F-5**: Recording page grouping | Group segments by listener_id in recording tree view | P2 |
| **F-6**: Bulk re-tag UI | Select multiple segments → assign listener | P2 |

### 9.3 UX Designer

| Item | Description | Priority |
|------|-------------|----------|
| **U-1**: Adapter comparison UX | How to preview/compare two adapters for same persona | P2 |
| **U-2**: Training wizard flow | Step-by-step flow for first-time listener adapter creation | P1 |
| **U-3**: Empty state design | What to show when no adapter exists for a listener | P2 |

### 9.4 Project Manager

| Item | Description | Priority |
|------|-------------|----------|
| **P-1**: Data audit | Inventory: how many minutes per listener_id do we have? | P0 |
| **P-2**: Recording campaign | Plan recording sessions to fill gaps per listener | P1 |
| **P-3**: Quality validation | Listen to LoRA vs SFT outputs, decide on path | P0 |
| **P-4**: Milestone tracking | Track B-1 through B-7 completion | Ongoing |

---

## 10. Milestones

- [ ] **M5.0**: SFT training verified working (baseline voice exists)
- [ ] **M5.1**: A/B test LoRA vs SFT — decide on voice quality
- [ ] **M5.2**: Adapter registry + API (B-3)
- [ ] **M5.3**: Listener-filtered training UI (F-1, F-4)
- [ ] **M5.4**: Multi-adapter inference routing (B-5, B-6)
- [ ] **M5.5**: Adapter management UI (F-2, F-3)
- [ ] **M5.6**: Recording bulk re-tag (F-6)
- [ ] **M5.7**: Full listener-specific voice system operational

---

## 11. Open Questions

| Question | Decision Needed | Status |
|----------|-----------------|--------|
| SFT vs LoRA quality | Need to listen to outputs first | Pending |
| Min audio per adapter | 15min? 30min? | TBD by data audit |
| Adapter naming convention | `xiao_s__child` or `xiao_s_child`? | Underscore preferred |
| Fallback behavior | If child adapter missing, use default or reject? | Reject + prompt to train |
| Multiple active adapters | Can only one be "active" per listener? | Yes, one active per (persona, listener) |

---

## 12. Dependencies

- **M5.2-M5.7** depend on **M5.0** (working baseline voice)
- **M5.3** depends on **M5.2** (adapter registry must exist first)
- **M5.4** depends on **M5.2** and **M5.3**
- **F-4** depends on **B-4** (training API must support listener_id first)

---

## 13. Related RFCs

- `RFC_M4_LORA_TRAINING.md` — Current training pipeline (LoRA + SFT, single model)
- `RFC_MVP_REDESIGN.md` — Recording/persona/listener data model
- `RFC_2_3_Adaptive_Voice_Cloning.md` — Emotion-aware voice cloning (superseded by this RFC)
