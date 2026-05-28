# EverHome — Consolidated Roadmap (2026 Q3)

**Status:** Proposed (not contractual)
**Owner:** Mark (product), Claude (architecture / code)
**Date drafted:** 2026-05-28
**Supersedes:** the listener-routing scope of `RFC_M5_MULTI_ADAPTER_VOICE_CLONING.md` and the persona/RAG scope of `RFC_M6_PERSONA_LLM_LEGACY.md` (both kept for history; this doc is the operational plan)

> All milestones in this document are **proposed**, not committed. Effort
> estimates are rough. Sequencing is current best understanding; expect
> reshuffling as the 06/02 demo lands and as OSS releases move.

---

## 1. Executive Summary

### Product

**EverHome — "A Time Capsule You Can Talk To"**

A privacy-first family-memory appliance. Family members upload a person's
voice, text, photos, chat exports; EverHome trains per-person voice and
persona, and lets relatives later talk to that person — in their voice,
with their cadence, knowing their stories. The box runs on hardware the
family owns; no personal data leaves it.

The 小S famous-figure pipeline is the **validation path** for this same
appliance, not a separate product (per `RFC_M6_PERSONA_LLM_LEGACY.md` §1
and `project_legacy_product_vision`).

### Markets

| Tier | Buyer | Notes |
|---|---|---|
| B2C primary | Diaspora Chinese families preserving an elder | Direct-to-consumer; box-as-product unit economics |
| B2B2C adjacent | Eldercare facilities, hospice, memorial-service providers | Same tech, higher budget, longer sales cycle; voice cloning is the moat |

### Immediate priority

**06/02 NY Tech Week demo.** Cannot risk demo flow for any architectural
pivot. M-Demo (§3) is polish-only.

### Six-month direction

Hybrid pipeline (Qwen Omni 7B handling ASR+LLM, Qwen3-TTS retained for
cloning) + OpenCharacter-trained persona LoRA + dual-index memory RAG +
multi-listener voice routing.

### Twelve-month direction

Migrate to an OSS Chinese end-to-end speech-to-speech model **only when**
one ships with voice cloning. Abstracted `BaseConversationEngine`
interface (§4) makes this a swap, not a rewrite. Watch list: Chroma
Chinese forks, future Qwen3.5-Omni iterations, community attempts to add
a cloning head to Qwen Omni.

### Single-line stance

> **Don't trade the moat (voice cloning) for the trend (E2E S2S). Build
> the swap-ready scaffolding now; swap when an OSS model with cloning
> actually exists.**

---

## 2. Current State (as of 2026-05-28)

### What works

| Capability | Where | Status |
|---|---|---|
| Streaming voice pipeline (VAD → ASR → LLM → TTS) | `app/api/ws_asr.py` | Shipped, ~1-2s end-to-end |
| Persona/listener-aware prompts | `app/services/llm/prompt_manager.py` + `app/resources/personas/*.json` | Shipped (RFC 2.2, RFC 2.2 Addendum) |
| Emotion tag streaming parser | `app/services/tts/emotion_mapper.py` | Shipped + property tests |
| Recording → diarization → segment → ASR | `app/services/training_service/` | Shipped (RFC M2) |
| Qwen3-TTS LoRA / SFT training | `app/services/training_service/training_job.py` | Shipped (RFC M4); LoRA path = `code_predictor` only (see `tts_voice_cloning_mechanics`) |
| Voice cloning via `custom_voice` config + baked speaker embedding | `merge_lora()` path | Shipped |
| Standalone HTML/JS chat UI | `app/api/standalone_ui.py` | Shipped |
| Demo mode (`?demo=1`) + English version | UI layer | Shipped |
| Persona corpus storage scaffolding | `data/personas/<persona_id>/corpus/` | Shipped (`RFC_M6` Phase 0 slice 1) |
| Multi-model LLM dispatch | `OpenAIClient.stream(model=...)` | Shipped (RFC M5 step (a)) |
| TTS `language="auto"` default (no Beijing accent leak) | `qwen_tts_engine.py` | Shipped (commit `3cd1547`; see `tts_inference_language_gotcha`) |
| DeepFilterNet denoise + LUFS normalize | recording pipeline | Shipped |

### Known limitations

| Gap | Impact | Owner doc |
|---|---|---|
| Source audio bandwidth (telephone-grade on best v7 model) | Demo voice sounds muffled | `docs/EverHome_demo_storyboard.md` final section |
| Text / ebook / PDF / EPUB / DOCX upload not implemented | Can't ingest grandma's letters or books | `RFC_M6` Phase 0 slice 2 (pending) |
| OCR for letter photos not implemented | Handwritten correspondence locked out | `RFC_M6` Phase 4 |
| Memory RAG (dual-index, BGE-M3, LanceDB) not implemented | LLM has no persona-specific facts; pure prompt steering | `RFC_M6` Phase 1-2 |
| Persona LoRA not implemented | LLM can mimic 小S only via prompt (works for famous figures, fails for elders per `persona_llm_architecture`) | `RFC_M6` Phase 3 |
| Multi-listener TTS routing not wired in chat | One adapter active per persona; listener changes prompt only | `RFC_M5` §15 |
| LLM is cloud (OpenAI gpt-4o-mini) | Violates "100% local" privacy moat | `RFC_M6` Phase 2 |
| No Constitutional-AI persona definition surface | Persona is a JSON of paragraphs, not a trained character | New (this roadmap) |
| `talker.model` LoRA only trains code_predictor (5 layers) | Voice cloning quality ceiling capped at SFT-equivalent | `training_pipeline_deferred` item 1 |

### Stack summary

```
Audio in →  Browser (raw PCM via onaudioprocess)
            ↓ WebSocket binary
ASR      →  Qwen3-ASR  (or MockASR for tests)
LLM      →  OpenAI gpt-4o-mini (PROVISIONAL — RFC_M6 Phase 2 wants Qwen2.5-7B-Q4 local)
Emotion  →  EmotionParser state machine ([E:情緒]內容)
TTS      →  Qwen3-TTS 1.7B VoiceDesign / FasterQwen3TTS fallback to Qwen3TTSModel
Audio out → AudioWorklet (Int16 PCM frames)

Hardware: AWS g5.4xlarge (A10G 23 GB) — dev
Hardware target: DGX Spark 128 GB unified (per hardware_targets memory)
```

---

## 3. Milestone Roadmap

### M-Demo — 06/02 demo polish (NOW → 2026-06-02)

**Goal.** Land the NY Tech Week demo. No architectural changes.

**Dependencies.** None.

**Deliverables.**

- Re-record one clean 5-min source sample with USB-mic-grade audio
  (browser mic, 48 kHz, close to mouth, quiet room) — per
  `docs/EverHome_demo_storyboard.md` "The one thing to fix" section.
- Retrain SFT (not LoRA) on the clean sample, validate preview.
- Verify demo prompts on the sticky-note list still land emotionally.
- Pre-stage fallback screen recording of the full demo.
- Pre-test the vibe-coding prompt and confirm Claude Code produces the
  intended diff.
- Confirm cloudflared tunnel + backup tunnel both healthy 30 min before
  doors open.

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| TTS fails live | Pre-recorded fallback in Quicktime |
| Tunnel 524 on long preview | Active model = SFT v7-class, not LoRA; verified <60s for short prompts |
| GPU NVML drift | Reboot box 24h before demo (see `training_pipeline_deferred` item 2) |
| Source audio still telephone-grade | New recording before 06/01 cutoff |

**Effort estimate.** 1-3 days. Most work is already done.

---

### M7 — Text / Ebook / Image ingestion (2026-06)

**Goal.** Implement the ingestion pipeline that `RFC_M6_PERSONA_LLM_LEGACY.md`
Phase 0 slice 2 describes: turn anything the user has in flat-file form
into corpus chunks that feed RAG (and later persona-LoRA training).

**Dependencies.** M-Demo done.

**Deliverables.**

| Input | Pipeline |
|---|---|
| `.txt`, `.md` | Encoding normalize → segment to chunks |
| `.pdf` | pdfplumber / PyMuPDF extract → fallback to Tesseract `chi_tra` OCR for image-only PDFs |
| `.epub` | ebooklib → chapter splits |
| `.docx` | python-docx |
| Photos of letters (JPG/PNG/HEIC) | Tesseract `chi_tra` → corpus chunk with `source_kind=letter_photo` |
| Family photos (JPG/PNG/HEIC) | EXIF metadata + optional LLM-generated caption; stored as RAG side-context |
| WhatsApp `.txt` / Line `.txt` / WeChat CSV+HTML | Platform-specific parser → per-message rows |

**New endpoints** (extend `RFC_M6` Phase 0):
- `POST /api/corpus/upload?persona_id=&kind=` — multipart upload
- `POST /api/corpus/ingest?persona_id=&item_id=` — background ingestion
- `GET /api/corpus/{persona_id}` — list items + status
- `DELETE /api/corpus/{persona_id}/items/{item_id}`

**UI.** New "Corpus" tab on `/ui/recordings` (per `RFC_M6` Phase 0). Drag-drop
upload, per-kind size + last-ingested view, "ready_for_rag" / "ready_for_lora"
threshold badges.

**Fail-loud requirement** (per `feedback_fail_loud` memory). OCR failures,
unsupported formats, and partial parses must surface in the UI — never
`status=ready` with empty extracted text.

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| Tesseract `chi_tra` poor on handwritten | Document fallback to commercial OCR (ABBYY-class) but ship printed-OCR first; flag handwritten as `quality=low` in manifest |
| Large PDFs / EPUBs blow memory | Stream-chunk extraction; per-file size limit (default 100 MB) with override |
| Mixed Traditional / Simplified | Both supported by Qwen tokenizer; no conversion at ingest time |

**Cross-references.** Subsumes `RFC_M6` Phase 0 slice 2. Cross-cuts
`RFC_M6` Phase 4 (photo→letter OCR moves earlier).

**Effort estimate.** 2-3 weeks.

---

### M8 — Memory RAG (2026-06 → 2026-07)

**Goal.** Stand up the dual-index RAG described in
`RFC_M6_PERSONA_LLM_LEGACY.md` §2. Retrieval results stream into the LLM
prompt at conversation time.

**Dependencies.** M7 (corpus must exist to index).

**Architecture (per `persona_llm_architecture` memory + `RFC_M6` §3 Phase 1).**

```
Style index    ← verbatim quips, retrieved as few-shot examples
Stance index   ← (topic, stance, evidence, source) tuples extracted by LLM

Embedding:     BGE-M3 (Chinese-native, hybrid dense+sparse)
Vector store:  LanceDB (Apache 2.0, embedded, Parquet-backed)
Chunking:      Anthropic contextual prefix (~49% fewer top-20 misses)
```

**Deliverables.**

- `app/services/rag/` package: `indexer.py`, `retriever.py`,
  `contextual_chunker.py`
- LanceDB schema per persona: `data/personas/<persona_id>/lancedb/`
- `POST /api/rag/reindex?persona_id=` — rebuild from corpus
- Retrieval call inserted into `ws_asr.py` LLM-streaming path (replaces
  the stub `rag_retrieval_seconds` metric)
- Sidebar in chat UI: "Pulled from: <doc title>, <date>" (transparency
  per `feedback_fail_loud`)

**Validation path.** Use 小S corpus (IG + podcasts + books once ingested
via M7) to A/B test. Compare prompt-only vs prompt+RAG on 20 held-out
questions.

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| BGE-M3 weights size on Spark VRAM budget | 568 MB at FP16; fits |
| Retrieval latency adds to E2E pipeline | Pre-warm + parallelize with LLM TTFT; target ≤80ms top-5 retrieval |
| LLM hallucinates outside retrieved context | Persona-drift mitigation (anchor reinjection per `RFC_M6` §9 ref) deferred to M9 |

**Cross-references.** Subsumes `RFC_M6` Phase 1-2.

**Effort estimate.** 3-4 weeks.

---

### M9 — OpenCharacter persona LLM (2026-07 → 2026-08)

**Goal.** Train per-person LoRA on a Chinese-capable base, using the
**Open Character Training** recipe (arXiv 2511.01689, Nov 2025). Replaces
ad-hoc prompt-manager personas with a trained character that's robust
against adversarial prompts.

**Why this technique (not full SFT, not prompt-only).** Per
`persona_llm_architecture` memory:

- Prompt-only fails for elders (base LLM has no prior on a regular
  person).
- Full SFT on small corpora destroys instruction following ("dual
  catastrophic forgetting" — ICCV 2025 SMoLoRA, OPLoRA 2510.13003).
- LoRA + general-instruction-mix preserves base.
- For low-data personas (<500 organic turns) Constitutional-AI
  synthetic expansion is the recipe — per the OpenCharacter paper.

**Paper reference.**

- **OpenCharacter / Open Character Training** — arXiv **2511.01689**
  (Nov 2025). First open implementation of character training via
  Constitutional AI + synthetic introspective dialogue. 3-stage process
  validated on Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B. More robust to
  adversarial prompting than system-prompt-only.
- Repo: <https://anonymous.4open.science/r/OpenCharacterTraining>
- HF: <https://huggingface.co/papers/2511.01689>
- Methodology source: Anthropic Constitutional AI (Bai et al. 2022,
  arXiv 2212.08073).

**Dependencies.** M7 (corpus). M8 helpful but not strictly required —
RAG provides facts; OpenCharacter provides voice/stance.

**Deliverables.**

- New UI surface: **"Constitution" page** per persona. Lets the user
  (or a family member acting as steward) author the constitution
  document — high-level principles like "warm, never cynical with
  children", "values: education, family loyalty", "language: Mandarin
  with Taiwanese particles 啦 / 咧 / 蛤".
- 3-stage training pipeline in `app/services/training_service/persona_llm/`:
  1. Generate introspective dialogue from constitution + corpus
     seeds (Constitutional AI step)
  2. SFT LoRA on Qwen 2.5 7B base, mixed with 30-50% general
     instruction data
  3. DPO on (in-character, off-character) preference pairs from a
     self-critic
- OPLoRA orthogonality (arXiv 2510.13003) applied to protect base
  capabilities.
- Per-persona LoRA file at `data/personas/<persona_id>/persona_lora/`.
- Switch from OpenAI gpt-4o-mini → local Qwen 2.5 7B (vLLM or llama.cpp)
  on the same call. Mirrors `RFC_M6` Phase 2.

**Choice of base.** Qwen 2.5 7B Instruct (per `chinese_support_stack`
memory: CMMLU-leading, Chinese-native, AWQ-Q4 fits in 6 GB VRAM).

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| LoRA training cost on A10G (24 GB tight for 7B + rank 16) | Use rank 8 + gradient checkpointing during dev; full quality run on Spark when available |
| Constitution authoring UX is unbounded | Ship template-with-fill-in-blanks first; free-form mode as power-user toggle |
| Per-person LoRA proliferation | LRU loader: keep N most-recent in VRAM, swap on demand; document N-family-members ceiling |
| Synthetic data leaks frontier-model style into the persona | Privacy ratchet (per `RFC_M6` §4.1) — soft default, opt-in per persona; show generated samples in UI before training |

**Cross-references.** Subsumes `RFC_M6` Phase 3. Independent of M10 but
shares the per-persona artifact directory.

**Effort estimate.** 6-8 weeks. (Heaviest milestone in this plan.)

---

### M10 — Multi-listener TTS routing (2026-08 → 2026-09)

**Goal.** Finish the listener-routing path in `RFC_M5_MULTI_ADAPTER_VOICE_CLONING.md`
that's currently 40% built (per `RFC_M5` §15).

**Dependencies.** None hard; can interleave with M9.

**What's already done (per `RFC_M5` §15).**
- `audio_resolver.py` filters segments by `(persona, listener)` pair
- `TrainingVersion.recording_ids_used` / `segment_ids_used` provenance
- `Listener.default_emotion` for prompt-tier conditioning
- `OpenAIClient.stream(model=...)` — step (a)

**What's missing.**

| Item (from `RFC_M5` §15.1) | Description |
|---|---|
| B-3 | `adapter_registry.json` as a separate JSON file at `data/models/` |
| B-4 | `GET /api/training/adapters?persona_id=...` grouping endpoint |
| B-5 | `StateManager.get_adapter_for_listener()` + dynamic load |
| B-6 | `qwen_tts_engine.activate_version()` per-listener routing |
| F-2, F-3 | Adapter management UI (group versions by listener) |
| F-4 | "Train new adapter" guided flow |

**Plus per (persona, listener) repository key.** Per the recent
conversation: persona/listener repository organization is currently
flat; needs key `(persona_id, listener_id)` at the data layer to make
per-listener LoRA storage clean.

**Deliverables.**

- `adapter_registry.json` with schema from `RFC_M5` §4.4.
- `ws_asr.py` reads `listener_id` from config frame → routes to correct
  adapter (LoRA file swap at TTS engine level).
- Adapter LRU: pin most-recent N adapters in VRAM, evict LRU on swap.
- Training UI listener dropdown surfaces only segments matching the
  chosen `(persona, listener)` pair.

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| TTS engine reload cost on listener swap | Pre-warm next-likely adapter when conversation starts; LRU N=2 |
| Listener taxonomy growth (per `RFC_M6` §4.4 open item) | Allow free-form listener IDs; preseed `child/mom/friend/default/elder` per `RFC_MVP_REDESIGN.md` §2.2 |
| User trains an adapter on too-few seconds | Min 30s gate per listener; warning at <2 min |

**Cross-references.** Completes `RFC_M5`. Touches `RFC_M6` Phase 0
(corpus listener tagging).

**Effort estimate.** 2-3 weeks.

---

### M11 — TTS engine abstraction (LOW PRIORITY, ANY TIME)

**Goal.** Carve out a `BaseTTSEngine` interface so swapping TTS models
(CosyVoice 2, future Qwen iterations, hypothetical OSS clones) is
plug-and-play.

**Dependencies.** None.

**Deliverables.**

```python
# app/services/tts/base.py (new)
class BaseTTSEngine(Protocol):
    def load_model(self, model_path: str) -> None: ...
    def load_adapter(self, adapter_path: str) -> None: ...
    async def generate_streaming(
        self,
        text: str,
        *,
        emotion: str | None = None,
        language: str = "auto",  # per tts_inference_language_gotcha
    ) -> AsyncIterator[bytes]: ...
    def unload(self) -> None: ...  # VRAM hygiene per recent ws_asr fix
```

`FasterQwen3TTS` and `Qwen3TTSModel` both implement `BaseTTSEngine`.
Future `CosyVoiceEngine`, `ChromaEngine` etc. fit the same protocol.

**Risk.** None — refactor only, behavior unchanged.

**Effort estimate.** 1-2 hours.

**Why we're not blocking on it.** No swap on the horizon for 1-2
months. Doing the refactor now costs little; deferring costs little.
Plumb it in as a coding-hygiene pass when convenient.

---

### M12 — Hybrid pipeline with Qwen 2.5 Omni 7B (2026-09 → 2026-10)

**Goal.** Eliminate the ASR→LLM hop by collapsing ASR and LLM into
**Qwen 2.5 Omni 7B**. Retain Qwen3-TTS for voice cloning. Target
end-to-end sub-1s latency.

**Why this hybrid (per `qwen_omni_evaluation` memory).**

- Qwen 2.5/3 Omni ships with **only 3 fixed voices** (Ethan, Chelsie,
  Aiden). Full migration gives up voice cloning — the moat.
- The 30B variant needs 78-145 GB VRAM (A100×2). Doesn't fit on A10G
  or DGX Spark.
- The 7B variant fits A10G (~16-24 GB VRAM) and inherits the
  no-cloning limitation, so we keep Qwen3-TTS for the output side.

**Dependencies.**
- M9 (persona LoRA) should be operational on Qwen 2.5 7B Instruct;
  Omni-7B is a sibling base, so the persona LoRA *may* transfer with
  some fine-tuning. Validate before committing.
- M11 (TTS abstraction) helpful for clean engine swap if needed.

**Deliverables.**

- `BaseConversationEngine` interface (§4) with two implementations:
  - `PipelineConversationEngine` (current ASR + LLM + TTS chain)
  - `HybridOmniConversationEngine` (Omni for input+reasoning, TTS sidecar)
- Audio input directly to Omni (no separate Qwen3-ASR pass).
- Latency telemetry: confirm sub-1s `audio_in → first_tts_chunk`.
- Quality regression suite: 50 held-out prompts run against both
  engines, side-by-side rating.

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| Omni-7B persona LoRA transfer doesn't work | Keep current LoRA-on-Qwen-2.5-7B-Instruct path as fallback; don't deprecate |
| VRAM budget tight on A10G with Omni-7B + Qwen3-TTS + RAG | Profile early; if needed, defer to Spark availability |
| Omni quality on Traditional Chinese understanding regresses vs Qwen3-ASR + Qwen2.5-LLM stack | A/B regression suite gates the swap |

**Cross-references.** New (not in any current RFC). Documented here.

**Effort estimate.** 4-6 weeks.

---

### M13 — OSS end-to-end speech-to-speech with cloning (2026-Q4 +, OPPORTUNISTIC)

**Goal.** When an OSS Chinese E2E S2S model with voice cloning lands,
swap the entire pipeline behind the `BaseConversationEngine` interface.

**Dependencies.** M11 + M12 (so the swap is genuinely plug-and-play).

**Watch list (per recent conversation + `qwen_omni_evaluation` Phase 3).**

| Candidate | Signal | Status |
|---|---|---|
| Qwen3.5-Omni (future) | Iteration of Omni line; cloning request open upstream | TBD |
| Chroma Chinese fork | FlashLabs FlashAI 2.0 is built on Chroma but no cloning; community fork rumored | TBD |
| OSS cloning head on Qwen Omni (community or ours) | Architecture-level patch | Speculative |

**Triggers to actually migrate.**

1. Model ships with verifiable voice cloning (≥10s reference, recognizable output).
2. License is permissive (Apache 2.0 / MIT preferred).
3. Quality regression against M12 hybrid passes (50-prompt suite).
4. VRAM fits Spark 128 GB unified.

**Until triggers fire:** stay on M12 hybrid. Don't migrate
speculatively.

**Effort estimate.** 2-4 weeks **once a viable model exists** — most of
the cost is regression-testing, not integration (interface already
abstracted).

---

## 4. Architecture Prep for Future Swaps

Three interfaces + one watch script. None costs more than half a day.

### 4.1 `BaseTTSEngine` (sketched in §M11)

Lets us swap Qwen3-TTS for CosyVoice 2 / future iterations without
touching `ws_asr.py`. Minimum surface:

```python
class BaseTTSEngine(Protocol):
    def load_model(self, model_path: str) -> None: ...
    def load_adapter(self, adapter_path: str) -> None: ...
    async def generate_streaming(self, text, *, emotion, language="auto") -> AsyncIterator[bytes]: ...
    def unload(self) -> None: ...
```

### 4.2 `BaseConversationEngine`

The bigger abstraction. Lets us swap the entire ASR→LLM→TTS chain for an
E2E S2S model without rewriting `ws_asr.py`.

```python
class BaseConversationEngine(Protocol):
    async def handle_utterance(
        self,
        pcm_audio: bytes,
        *,
        persona_id: str,
        listener_id: str,
        session_state: SessionState,
    ) -> AsyncIterator[ConversationEvent]:
        """
        Yields events: ASRResult | LLMToken | EmotionDetected | TTSChunk | Done.
        Pipeline impl orchestrates ASR + LLM + TTS internally.
        E2E impl just calls a single model.
        """
```

Implementations:
- `PipelineConversationEngine` — current code, wrapped.
- `HybridOmniConversationEngine` — Omni for ASR+LLM, TTS sidecar (M12).
- `E2ES2SConversationEngine` — single-model future (M13).

### 4.3 Training-data layer model-agnosticism

The corpus + segment + speaker schemas in `RFC_MVP_REDESIGN.md` §2 are
already model-agnostic (PCM + JSON metadata). Verify before each
milestone that no Qwen3-TTS-specific assumption has leaked into:
- `app/services/training_service/audio_resolver.py`
- `app/services/training_service/models.py`
- `data/personas/<persona_id>/voice/` layout

### 4.4 OSS voice-model watch script (concept)

```bash
# scripts/_oss_voice_watch.sh — runs weekly, posts to Telegram
# Checks:
#   - HF trending tag "tts" filtered by language=zh
#   - arXiv cs.SD weekly new submissions matching {tts,clone,zero-shot}
#   - GitHub trending: "voice cloning", "speech synthesis"
#   - Specific repos: Qwen, alibaba/cosyvoice, FlashLabs/*, etc.
# Posts a one-line summary per new candidate via existing Telegram channel
# (chat_id 6143208798, per reference_user_comms memory).
```

Implementation deferred but the concept lives here so we don't miss the
M13 trigger.

---

## 5. Hardware Decision Tree

Per `hardware_targets` memory:

```
Current:       AWS g5.4xlarge (A10G 23 GB)
               ↓
M-Demo:        Stay on A10G — sufficient
M7-M8:         Stay on A10G — RAG + ingestion is CPU/disk-heavy not VRAM-heavy
M9:            Stay on A10G for dev; full-quality runs may want Spark
M10:           Stay on A10G — adapter swap is engineering, not bigger model
M11:           No hardware change
M12 hybrid:    Profile VRAM with Omni-7B (~16-24 GB) + TTS (~3.4 GB) + RAG (~3 GB)
               + LoRA adapter (~0.5 GB) + KV cache. Likely 24 GB tight.
               TRIGGER: upgrade to DGX Spark or RTX 5090-class if A10G overflows
M13 E2E:       Spark almost certainly required (unified 128 GB for cloning head + S2S)
```

**Spark / RTX 5090 trigger conditions:**

1. M12 ablation proves Omni hybrid is clearly better (latency + quality)
   AND A10G can't hold the full stack.
2. M9 persona-LoRA training on Qwen 2.5 7B with full instruction-data
   mix needs >24 GB during gradient accumulation.
3. Multiple concurrent family members (M10 + production) need >2
   adapters resident.

**Don't buy speculatively.** $4k Spark expense waits for ablation.

**Mac Mini M4 Max fallback.** Per `hardware_targets`: inference-only;
training stays cloud. Only relevant if we ship a non-training appliance
SKU. Not on near-term roadmap.

---

## 6. Research References

| Paper / Project | arXiv / URL | Use in roadmap |
|---|---|---|
| **Open Character Training** | arXiv 2511.01689 (Nov 2025) | M9 — persona LoRA training recipe |
| Anthropic Constitutional AI | arXiv 2212.08073 (Bai et al. 2022) | M9 — methodology foundation |
| OPLoRA | arXiv 2510.13003 | M9 — orthogonal LoRA, preserves base |
| BGE-M3 | HuggingFace `BAAI/bge-m3` | M8 — Chinese-native hybrid embedding |
| Anthropic Contextual Retrieval | anthropic.com/news/contextual-retrieval | M8 — chunk-prefix-with-context |
| Persona Drift | arXiv 2402.10962 | M9 — drift measurement + anchor reinjection |
| Qwen 2.5 Omni 7B | huggingface.co/Qwen | M12 — hybrid ASR+LLM |
| Qwen 3 Omni 30B-A3B | huggingface.co/Qwen | NOT used — doesn't fit hardware budget |
| Chroma (FlashLabs) | flashlabs.ai | M13 — watch list, no cloning today |
| LanceDB | lancedb.com (Apache 2.0) | M8 — vector store |
| Qwen 2.5 7B Instruct | huggingface.co/Qwen | M9 — persona LoRA base |
| Qwen3-TTS 1.7B VoiceDesign | huggingface.co/Qwen | All milestones — TTS retained for cloning |

---

## 7. Markets & Business Hooks

### 7.1 B2C — EverHome direct-to-consumer

**Positioning.** "A Time Capsule You Can Talk To" — privacy-first family
memory appliance.

**ICP (ideal customer profile).** Diaspora Chinese families (Taiwan / HK /
overseas) with an aging parent. Cultural fit: filial obligation + practical
distance + Mandarin-native UX nobody else serves well (per
`chinese_support_stack` memory).

**Pricing wedge.** Hardware-as-product unit economics. Spark-class box
(~$3-4k) + recurring storage / ingestion-service tier.

**Distribution open question (per `RFC_M6` §7).** Funeral-home / hospice
partnerships? Diaspora-community marketing? TBD.

### 7.2 B2B2C — eldercare / hospice / memorial SaaS

**Positioning.** "Preserve a resident's voice and stories — for their
family, on your hardware, no data leakage."

**ICP.** Eldercare facilities, hospice providers, memorial-service
companies (US, Taiwan, HK). Same tech as B2C; higher budget, longer
sales cycle, lower churn.

**Why now.** FlashLabs going B2B with FlashAI 2.0 signals the enterprise
voice-assistant market is real money (per `qwen_omni_evaluation`
strategic side-note). They have no cloning — we do.

**Moat structure (per `RFC_M6` §7).**

| Moat | Defensibility |
|---|---|
| Voice cloning per-individual fidelity | HIGH — competitors (FlashLabs, generic Omni) ship fixed voices |
| Privacy-by-architecture | HIGH — cloud competitors structurally can't match; needs audit/attestation story |
| Chinese-native at every layer | HIGH for Chinese-speaking markets; competitors English-first |
| Hardware-as-product | MEDIUM — reproducible if competitor commits capex |
| Per-person persona depth | MEDIUM — OpenCharacter is published; we're early implementers |

---

## 8. Open Questions / Risks

| # | Open question | Impact |
|---|---|---|
| 1 | Source audio bandwidth — best-sounding v7 model trained on telephone-grade source | Demo voice sounds muffled (per `EverHome_demo_storyboard.md`); requires per-family-member good-mic recording at intake |
| 2 | OSS E2E S2S model with cloning timeline uncertain | M13 may slip to 2027 |
| 3 | Per-person LoRA scalability — how many family members per Spark box? | Unknown ceiling; LRU swap with N=2-4 likely OK for ~10 family members per box |
| 4 | Latency targets — current ~1-2s → sub-1s with M12 → sub-500ms with M13 | Each milestone needs measured baseline; no claim without telemetry |
| 5 | Privacy ratchet for synthetic data (M9 Constitutional AI step needs frontier model?) | Open #1 in `RFC_M6` §4. Default soft (one-shot allowed during persona setup); hard mode forces ≥10k organic turns |
| 6 | Listener taxonomy: fixed vs growing | Open #4 in `RFC_M6` §4. Default growing per memory; needs UI design for M10 |
| 7 | Real `talker.model` LoRA training (vs current `code_predictor`-only) | Voice cloning quality ceiling capped at SFT-equivalent until this lands (per `training_pipeline_deferred` item 1; ~1-2 day investigation, out of demo path) |
| 8 | GPU NVML driver mismatch on long-uptime hosts | Dev-loop friction (per `training_pipeline_deferred` item 2). Add startup check; reboot before demo |
| 9 | Family UI surface (per `RFC_M6` Phase 4) — separate from dev UI, Claude Design tool | Not in any M7-M13 milestone; needs its own milestone after M10 |
| 10 | Constitution authoring UX — free-form vs template | M9 design open; ship template first |

---

## 9. Cross-Reference Table

| Old RFC | Original scope | Status as of 2026-05-28 | Superseded by |
|---|---|---|---|
| `RFC_MVP_MASTER.md` | Vision + early architecture | Mostly shipped (M1) | M-Demo (polish only) |
| `RFC_MVP_REDESIGN.md` | Data model (persona / listener / segment / training version) | Shipped — data model stable | — (referenced, not superseded) |
| `RFC_M1_IMPLEMENTATION_PLAN.md` | Core streaming pipeline | Shipped | — |
| `RFC_M2_RECORDING_TRAINING.md` | Recording → diarization → segment → ASR | Shipped + iterating | M-Demo polish (source audio quality) |
| `RFC_M4_LORA_TRAINING.md` | LoRA + SFT TTS training | Shipped (experimental); LoRA = code_predictor only | — (real talker LoRA = `training_pipeline_deferred`) |
| `RFC_M5_MULTI_ADAPTER_VOICE_CLONING.md` | Multi-listener TTS routing | Step (a) done; listener-tier in prompt only | **M10 (this roadmap)** |
| `RFC_M6_PERSONA_LLM_LEGACY.md` | Persona LLM + RAG + family appliance | Phase 0 slice 1 done; rest design-only | **M7 + M8 + M9 (this roadmap)**; Family UI deferred |
| `RFC_2_2.md` | Relationship-aware persona | Shipped | Folded into M9 (Constitution-driven persona supersedes JSON prompts) |
| `RFC_2_2_Addendum_Persona_Factory.md` | Persona-as-data JSON loader | Shipped | Folded into M9 (UI authoring for personas + constitution) |
| `RFC_2_3_Adaptive_Voice_Cloning.md` | Per-listener reference-audio cloning | Superseded in practice by LoRA path (`RFC_M4`) | **M10** completes listener routing via adapter, not reference audio |

---

## 10. Sequencing at a glance

```
2026
│
├─ MAY 28  ─┐  (now)
│           ├─ M-Demo polish (USB mic, retrain SFT, fallback recording)
├─ JUN 02  ─┘  ← NY Tech Week demo
│
├─ JUN     ─┐
│           ├─ M7 — Text / ebook / image ingestion
├─ JUL     ─┘
│
├─ JUL     ─┐
│           ├─ M8 — Memory RAG (BGE-M3 + LanceDB + dual index)
├─ AUG     ─┘
│
├─ JUL     ─┐
│           ├─ M9 — OpenCharacter persona LoRA + Qwen 2.5 7B local
├─ AUG     ─┘
│
├─ AUG     ─┐
│           ├─ M10 — Multi-listener TTS routing (adapter_registry.json)
├─ SEP     ─┘
│
│           M11 — TTS engine abstraction (1-2 hr, slot in any time)
│
├─ SEP     ─┐
│           ├─ M12 — Hybrid pipeline (Qwen 2.5 Omni 7B + Qwen3-TTS)
├─ OCT     ─┘
│
└─ Q4 +     M13 — OSS E2E S2S migration (opportunistic, when model ships)
```

Milestones M8 and M9 run in parallel (different code paths, share corpus
from M7).

---

## 11. Document conventions

- **English only** — master branch is English.
- **Proposed, not committed** — sequencing reshuffles as reality lands.
- **Don't delete old RFCs** — they're history. Refer to them by file name.
- **Fail loud** (per `feedback_fail_loud` memory) — every milestone's
  deliverables must surface failures in HTTP/UI, never silent success.
- **No prompt-only or full-SFT alternatives for persona** (per
  `persona_llm_architecture` memory) — base + LoRA + RAG is the stack.
- **TTS timbre = baked speaker embedding** (per `tts_voice_cloning_mechanics`
  memory) — not the LoRA delta. Don't reverse this anywhere.
- **TTS `language="auto"` default** (per `tts_inference_language_gotcha`
  memory) — don't regress to `"Chinese"`.
- **Voice cloning is the moat** — don't trade it for E2E S2S unless and
  until an OSS model ships with cloning.
