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
interface (§4) makes this a swap, not a rewrite. **Step-Audio 2 mini
(Apache 2.0, native CN, zero-shot cloning via text-audio token
interleaving) is the leading open-weight bet today** and the primary
M13 target. Watch list also: Kimi-Audio, LLaMA-Omni 2, Chroma Chinese
forks, future Qwen3.5-Omni iterations.

The **highest-leverage single action** is a 1-day Qwen3.5-Omni-Light
cloning eval spike — Light has open weights on HF as of 2026-03-30. If
cloning quality matches the paper benchmarks, M12 (hybrid) and M13
(E2E S2S) collapse into one week of deployment instead of a 6-12 month
wait. See M12a below.

A new **M-Consent** milestone (§M-Consent, inserted before M7) covers
consent capture, revocation, audio watermarking, and audit trail —
NO FAKES Act / EU AI Act / CA AB 1836 are now a commercial blocker for
both B2C and B2B sale.

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

### M-Consent — Voice Cloning Consent, Revocation, and Compliance (2026-06, PARALLEL TO M7)

**Goal.** Build consent capture + revocation flow + audio watermarking
+ audit trail. Required for B2B sales (eldercare / hospice / memorial
SaaS) and exposed B2C in regulated markets. **Currently absent from
the roadmap entirely** — implied by `project_legacy_product_vision`
memory but never written up in any prior RFC.

**Dependencies.** M-Demo done. Can run parallel to M7 (different code
surfaces).

**Regulatory drivers.**

- **NO FAKES Act (US, 2025)** — prohibits unauthorized digital
  replicas of voice/likeness; lifetime + 70 years post-mortem.
- **EU AI Act** (in effect 2024-2026) — voice cloning classified
  high-risk; transparency / watermarking obligations for
  AI-generated content.
- **California AB 1836 (2026)** — post-mortem voice and likeness
  rights, 70-year duration. **California AB 2602 (Jan 2025)** —
  strengthens informed-consent requirements for digital replicas.
- Diaspora ICP includes US (California is the largest single market).

**Reference paper.** **The Making of Digital Ghosts: Designing Ethical
AI Afterlives** ([arXiv:2511.20094](https://arxiv.org/pdf/2511.20094),
Nov 2025) — ethical framework for deceased-person modeling. Covers
consent, psychological harm (esp. to children), identity distortion.
EverHome IS this category; we need a written ethical position before
customer onboarding.

**Deliverables.**

- **Consent form UI** in `/ui/recordings` — per-recording opt-in,
  listener-scope selector (who is allowed to hear this voice?),
  expiration date, jurisdiction tag at intake.
- **Revocation API** — `DELETE /api/consent/{recording_id}` with
  persistent tombstone. Recordings deleted but **trained model
  retraining required** (LoRA derived from revoked data is itself
  revoked — surface this honestly in the UI per `feedback_fail_loud`).
- **Audio watermarking** on synthesized output —
  - Audible disclosure option for B2C (configurable per persona)
  - Inaudible audit-trail watermark for B2B (compliance trail)
- **Audit log** — who generated what voice for what listener when;
  exportable per persona for legal discovery.
- **Pre-mortem vs post-mortem path** — explicit UI distinction. For
  living personas, consent is revocable by the person themselves. For
  deceased, consent stewardship transfers to the designated family
  steward per the consent record.

**Cross-references.** Touches `RFC_M6` Phase 0 (corpus consent
metadata). Informs M10 listener routing (listener-scope is a consent
property, not just a routing key).

**Risks.**

| Risk | Mitigation |
|---|---|
| Without this milestone, B2B (eldercare / hospice / memorial SaaS) won't buy | Compliance is medical-grade required; ship before B2B sales motion |
| Watermark library quality varies wildly | Use SilentCipher or AudioSeal class libraries; document choice + key rotation |
| Revocation requires LoRA retraining → costly | Document expected retraining latency; queue revocation-triggered retrains; surface "model rebuilding" status in UI |
| Pre-mortem clone of a still-living person — consent revocability over time | Per [Digital Doppelgangers, arXiv:2502.21248](https://arxiv.org/html/2502.21248v1) — revocability flow must be one-click + reversible-during-grace-period |

**Effort estimate.** 2-3 weeks (UI + backend + watermark library
integration). Can be parallel to M7.

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
| `.pdf` | **MinerU 2.5-Pro** (VLM-based, handles layout / tables / charts / cross-page merge; Chinese-native, image-in-table OCR) → markdown chunks |
| `.epub` | ebooklib → chapter splits |
| `.docx` | python-docx |
| Photos of letters (JPG/PNG/HEIC) | **PaddleOCR-VL 1.5** (94.5% OmniDocBench; supports simplified + traditional Chinese, pinyin, handwriting, vertical text) → corpus chunk with `source_kind=letter_photo` |
| Family photos (JPG/PNG/HEIC) | EXIF metadata + optional LLM-generated caption; stored as RAG side-context |
| WhatsApp `.txt` / Line `.txt` / WeChat CSV+HTML | Platform-specific parser → per-message rows |

**OCR/PDF stack rationale.** The 2026 SOTA for Chinese + handwriting +
complex layout has moved to VLM-based unified parsers. PaddleOCR-VL 1.5
+ MinerU 2.5-Pro together replace pdfplumber + PyMuPDF + Tesseract
`chi_tra` in a single tool surface. Handwritten elder letters — the M7
known weakness — are now in-scope rather than requiring `quality=low`
admission of defeat.

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
| PaddleOCR-VL VRAM cost on A10G alongside other models | VLM is ~1-2 GB; profile during M7 build; can run on CPU at lower throughput if needed |
| Large PDFs / EPUBs blow memory | Stream-chunk extraction; per-file size limit (default 100 MB) with override |
| Mixed Traditional / Simplified | Both supported by Qwen tokenizer + PaddleOCR-VL natively; no conversion at ingest time |

**Cross-references.** Subsumes `RFC_M6` Phase 0 slice 2. Cross-cuts
`RFC_M6` Phase 4 (photo→letter OCR moves earlier).

**Effort estimate.** 2-3 weeks.

---

### M8 — Memory RAG (2026-06 → 2026-07)

**Goal.** Stand up the dual-index RAG described in
`RFC_M6_PERSONA_LLM_LEGACY.md` §2. Retrieval results stream into the LLM
prompt at conversation time.

**Dependencies.** M7 (corpus must exist to index).

**Architecture (per `persona_llm_architecture` memory + `RFC_M6` §3 Phase 1 + 2026 research).**

Primary architecture is now **ID-RAG** (Identity Retrieval-Augmented
Generation, [arXiv:2509.25299](https://arxiv.org/abs/2509.25299), Sept
2025) — grounds the agent persona in a dynamic identity knowledge graph
(beliefs, traits, values) explicitly retrieved each turn. This replaces
the plain dual-index BGE-M3 design.

```
Identity graph (ID-RAG)   ← beliefs / traits / values, retrieved each turn
Semantic facts (HippoRAG 2) ← multi-hop knowledge graph + Personalized PageRank
Style index               ← verbatim quips, retrieved as few-shot examples
Conversation memory       ← A-MEM or Mem0 layer, consolidates per-turn

Embedding:     BGE-M3 (Chinese-native, hybrid dense+sparse) — retained as primitive
Vector store:  LanceDB (Apache 2.0, embedded, Parquet-backed)
Chunking:      Anthropic contextual prefix (~49% fewer top-20 misses)
```

- **ID-RAG** — identity knowledge graph per persona; queried every
  utterance to anchor "who is this person" before retrieval over facts.
- **HippoRAG 2** ([arXiv:2502.14802](https://arxiv.org/html/2502.14802v1))
  retained as the semantic-facts retriever — knowledge graph +
  Personalized PageRank, higher multi-hop F1/Recall@5 than vanilla
  top-k, lower indexing cost than GraphRAG / RAPTOR / LightRAG.
- **A-MEM** ([arXiv:2502.12110](https://arxiv.org/pdf/2502.12110)) or
  **Mem0** ([arXiv:2504.19413](https://arxiv.org/pdf/2504.19413)) for
  conversation-time memory consolidation — "every chat with grandma
  adds to her memory of you," not just static-corpus retrieval. Pick
  one after a small bake-off; Mem0 is more turnkey, A-MEM is more
  surgical.

**Deliverables.**

- `app/services/rag/` package: `id_graph.py`, `hippo_retriever.py`,
  `conversation_memory.py`, `contextual_chunker.py`
- LanceDB schema per persona: `data/personas/<persona_id>/lancedb/`
- ID-graph store per persona: `data/personas/<persona_id>/id_graph/`
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
| **Persona fidelity degrades over 100+ rounds** ([arXiv:2512.12775](https://arxiv.org/pdf/2512.12775)) — especially goal-oriented dialogue; elder users will accumulate 1000s of turns over months | Explicit identity anchor reinjection per turn (matches `RFC_M6` §9 already, validated by this paper); ID-RAG architecture above is the structural mitigation |

**Cross-references.** Subsumes `RFC_M6` Phase 1-2.

**Effort estimate.** 3-4 weeks.

---

### M8.5 — Instruction-conditioned TTS fine-tuning (2026-06 → 2026-07)

**Why this comes before M9.** Industry best practice (per
[EMORL-TTS arXiv:2510.05758](https://arxiv.org/abs/2510.05758),
[Characteristic-Specific Partial Fine-Tuning arXiv:2501.14273](https://arxiv.org/abs/2501.14273))
is to **define emotion vocabulary and label training data BEFORE the
persona LLM fine-tune** — so the persona LLM in M9 can learn to emit
only trained-vocabulary emotion tags, avoiding the "LLM emits emotion X,
TTS doesn't recognize it" mismatch. M8.5 produces the emotion-conditioned
TTS + a fixed vocabulary; M9 trains the persona LLM to emit only those
tags.

**Goal.** Enable audible emotion / listener tone shift in the cloned
voice **without per-sentence voice drift**. Currently passing a varying
`instruct` per sentence to a SFT-trained `custom_voice` model
destabilizes prosody — each sentence sounds like a different person
(observed during 06/02 demo prep; see §11 demo learnings). The SFT
model was trained on `(audio, text)` pairs without instruct conditioning,
so the `instruct` field at inference becomes an out-of-distribution
perturbation rather than a controllable style axis.

**Why this milestone exists.** Today's emotion routing path
(`emotion_mapper.py` → `instruct=<style string>`) had to be reverted to
`instruct=None` + `language="Chinese"` for demo stability. Listener
tone-shift now relies on LLM word choice + tone-chip visual only — no
audible voice differentiation. M9 (persona LLM) and M10 (multi-listener
voice routing) both depend on audible differentiation; without M8.5, M9
learns an undefined emotion vocabulary and M10 only ships visual +
prompt-level tone changes.

**Approach** (4 phases, ~1 month):

- **Phase A (week 1, 2026-06) — Auto-emotion labeling of existing recordings.**
  Run **emotion2vec** ([FunASR](https://github.com/modelscope/FunASR) /
  modelscope `iic/emotion2vec_plus_large`) over current 小S corpus
  segments. Produce `(segment_id, emotion_label, confidence)` rows in
  `data/personas/<persona_id>/labels/emotion.jsonl`. Spot-check 50
  segments by hand; expect ~70-80% classifier accuracy on Chinese.
  Fallback / cross-check tools: **wav2vec2-emotion**
  (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`),
  **openSMILE** (`eGeMAPSv02` feature set for emotion analysis).
- **Phase B (week 2, 2026-06) — Gap-fill recording for under-represented
  emotions.** From Phase A distribution, identify under-represented
  emotions among the 4-6 core targets (`gentle`, `warm`, `focused`,
  `reflective`, `upbeat`, `+1 reserved`). Record gap-filling content;
  balance dataset across the 4-6 core emotions. Output: ~30-40 minutes
  of labeled supplementary audio.
- **Phase C (week 3, 2026-07) — Retrain SFT with mixed instruct conditioning.**
  20-30% of training pairs carry an emotion-matched `instruct` string;
  70-80% are left empty (`instruct=None`). Goal: preserve speaker
  generalization (the 70-80% empty path) while teaching the model to
  follow `instruct` when present (the 20-30% conditioned path). Dual-
  loss training is the likely recipe per the CosyVoice 2 paper
  ([arXiv:2412.10117](https://arxiv.org/abs/2412.10117)) — speaker-
  identity loss alongside text-audio loss to prevent the cloned voice
  from drifting when conditioned.
- **Phase D (week 4, 2026-07) — LLM-side instruct vocabulary alignment + A/B
  ablation.** Lock the final emotion vocabulary the retrained TTS
  understands. This vocabulary is the **input contract for M9** — the
  persona LLM SFT corpus must emit `[E:情緒]` tags drawn only from this
  set. Run blind A/B on 8 utterances per (persona, listener, emotion)
  with a family member — confirm (1) speaker still recognizable (no
  drift); (2) emotion audibly distinguishable.

**Industry references.**

- **EMORL-TTS** — [arXiv:2510.05758](https://arxiv.org/abs/2510.05758)
  — emotion-vocabulary-first methodology; train TTS on a closed set of
  emotion labels before any downstream LLM/persona work.
- **Characteristic-Specific Partial Fine-Tuning** —
  [arXiv:2501.14273](https://arxiv.org/abs/2501.14273) — argues for
  staged fine-tuning where downstream behaviors are conditioned on
  upstream vocabularies, not co-trained.
- **CosyVoice 2 (Alibaba)** — [arXiv:2412.10117](https://arxiv.org/abs/2412.10117)
  — flagship paradigm for instruction-conditioned TTS: `(text,
  style_prompt, audio)` tuples; instruct-following without speaker
  drift via dual-loss training.
- **Step-Audio 2 / 2.5 Realtime (StepFun)** — roleplay-specific RLHF +
  10k+ persona-matrix training; closest-in-spirit gold standard for
  emotion-controllable cloned voice (closed API only).
- **StyleTTS 2** — [arXiv:2306.07691](https://arxiv.org/abs/2306.07691)
  — adaptive style transfer via diffusion-based style sampling;
  reference for "preserve speaker, vary style" architectural pattern.
- **Bark / Suno-AI** — style-token approach to controllable TTS;
  reference for vocabulary design (discrete style tokens vs free-form
  instruct).
- **emotion2vec** — speech emotion representation; primary auto-labeler
  for Phase A.
- **wav2vec2-emotion** — cross-check labeler.
- **openSMILE** — paralinguistic feature extraction; manual review
  support tool.

**Dependencies.** Depends on M7 (corpus exists for emotion labeling).
M11 (TTS engine abstraction) helpful so that retrained SFT model can be
served behind `BaseTTSEngine`. Real `talker.model` LoRA (companion
deliverable in M11) would compose naturally with instruct conditioning
— train the talker LoRA on the same Phase C mixed-instruct corpus.

**Blocks.** M8.5 **blocks M9** — the persona LLM in M9 must be trained
to emit `[E:情緒]` tags drawn from the vocabulary this milestone locks
at Phase D. Co-training risks the LLM emitting emotions the TTS does
not recognize.

**Deliverables.**

- `app/services/training_service/emotion_labeling/` package wrapping
  emotion2vec + wav2vec2-emotion + openSMILE behind a unified label-
  writer interface.
- `data/personas/<persona_id>/labels/emotion.jsonl` per persona.
- Retrained SFT custom_voice model with mixed-instruct conditioning;
  versioned as `talker_v<N>_instruct_conditioned`.
- Updated `emotion_mapper.py` vocabulary aligned with trained
  instruct strings (not the current free-form natural-language
  strings).
- **Locked emotion vocabulary spec** — `docs/EMOTION_VOCAB_v1.md` — the
  input contract for M9 persona LLM SFT.
- Blind A/B eval harness — `scripts/ab_eval_emotion.py` — records
  family-member listener responses per (persona, listener, emotion).

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| Mixed-instruct training destabilizes speaker identity | Dual-loss training (CosyVoice 2 recipe); validate after every 500 steps with speaker-similarity metric |
| Auto-classifier accuracy ~70-80% on Chinese | Manual review pass on the ~30-40 min labeled dataset; spot-check before training |
| Instruct vocabulary scope mismatch (LLM emits N emotions, training covers M) | Lock vocabulary at Phase D **before M9 SFT data generation begins**; out-of-distribution emotions fall back to `instruct=None` with logged warning per `feedback_fail_loud` |
| 30-40 min supplementary recording requires scheduled studio time | Schedule before Phase A wraps; gap-fill list known by end of week 1 |

**Acceptance gate.** Per (persona, listener, emotion) blind A/B with a
family member — speaker recognized AND emotion audibly distinguishable
on ≥6/8 utterances. Same gate doubles as the M8.5 deliverable for the
close-relative-recognizable A/B from `RESEARCH_SFT_S2S.md` §"Top 3
recommendations" item 3.

**Cross-references.** Sits between M8 (Memory RAG) and M9 (persona LLM)
because M9 SFT data generation must consume the vocabulary locked here.
Composes with the real `talker.model` LoRA promoted in M11.

**Effort estimate.** ~1 month (4 phases × 1 week each), with studio
recording in week 2 as the calendar gate.

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

**Paper reference — TWO complementary OpenCharacter papers.**

There are two distinct "OpenCharacter" papers, easy to confuse, that we
combine:

- **OpenCharacter (Salesforce, Jan 2025)** — [arXiv:2501.15427](https://arxiv.org/abs/2501.15427).
  Synthesizes 326K character-aligned instruction-response pairs from
  PersonaHub via response-rewrite (`-R`) and response-generate (`-G`)
  strategies; SFT on LLaMA-3 8B matches GPT-4o on role-play. HF
  dataset: `xywang1/OpenCharacter` (326K rows, 1.19 GB). Use this
  dataset as the persona-instruction mix.
- **Open Character Training (Constitutional AI, Nov 2025)** —
  [arXiv:2511.01689](https://arxiv.org/abs/2511.01689). First open
  implementation of character training via Constitutional AI +
  synthetic introspective dialogue. 3-stage process validated on
  Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B. More robust to adversarial
  prompting than system-prompt-only. Repo:
  <https://anonymous.4open.science/r/OpenCharacterTraining>; HF:
  <https://huggingface.co/papers/2511.01689>.
- **Methodology source:** Anthropic Constitutional AI (Bai et al. 2022,
  arXiv 2212.08073).

**Combined recipe:** Salesforce 326K dataset as the general persona
instruction mix, blended at 30-50% to combat catastrophic forgetting,
then the Nov 2025 Constitutional AI + introspective-dialogue → SFT →
DPO pipeline on top.

**PersonaHub** (Tencent, [GitHub](https://github.com/tencent-ailab/persona-hub))
— 1B synthetic personas; 370M elite subset released 2025. Solves the
"elder won't have 326K examples" problem via data augmentation: sample-
condition on demographically-similar PersonaHub personas to augment an
elder's 50 organic turns into a synthetic 5K-turn corpus before LoRA
training.

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
     seeds (Constitutional AI step). For low-data personas, also
     augment via PersonaHub demographically-similar sampling.
  2. SFT LoRA on **Qwen 3 8B** base, mixed with 30-50% general
     instruction data (Salesforce OpenCharacter 326K dataset).
  3. DPO on (in-character, off-character) preference pairs from a
     self-critic.
- OPLoRA orthogonality (arXiv 2510.13003) applied to protect base
  capabilities.
- Per-persona LoRA file at `data/personas/<persona_id>/persona_lora/`.
- Switch from OpenAI gpt-4o-mini → local Qwen 3 8B (vLLM or llama.cpp)
  on the same call. Mirrors `RFC_M6` Phase 2.

**Choice of base.** **Qwen 3 8B** (updated from Qwen 2.5 7B). Same
VRAM budget, current generation, better Chinese, better
instruction-following. Per `chinese_support_stack` memory: Qwen line
remains CMMLU-leading and Chinese-native; the persona-LoRA recipe
transfers cleanly. Apache 2.0; supports vLLM / SGLang / llama.cpp /
Ollama / MLX / KTransformers.

**Acceptance gate.** **CharacterEval** ([arXiv:2401.01275](https://arxiv.org/abs/2401.01275))
— Chinese-native persona benchmark built on Chinese novels/scripts;
suits EverHome's domain better than English-only role-play evals. Run
the trained persona LoRA against CharacterEval as M9 acceptance.

**Evaluation methodology.** **Memory-Driven Role-Playing**
([arXiv:2603.19313](https://arxiv.org/pdf/2603.19313)) decomposes
role-playing memory use into 4 stages (cue → retrieve → integrate →
respond) with per-stage diagnostic metrics. Use this to diagnose
*where* a persona fails when CharacterEval scores drop, not just
*that* it failed.

**Risks & mitigations.**

| Risk | Mitigation |
|---|---|
| LoRA training cost on A10G (24 GB tight for 7B + rank 16) | Use rank 8 + gradient checkpointing during dev; full quality run on Spark when available |
| Constitution authoring UX is unbounded | Ship template-with-fill-in-blanks first; free-form mode as power-user toggle |
| Per-person LoRA proliferation | LRU loader: keep N most-recent in VRAM, swap on demand; document N-family-members ceiling |
| Synthetic data leaks frontier-model style into the persona | Privacy ratchet (per `RFC_M6` §4.1) — soft default, opt-in per persona; show generated samples in UI before training |

**Optional advanced add-on — M9.5: FinePE (Mixture of LoRA Experts).**

[FinePE](https://www.sciencedirect.com/science/article/abs/pii/S1568494626003911)
proposes MoLE-style routing across personality-subtrait LoRAs —
separate adapters for e.g. "writing style", "factual memory", "speech
mannerisms", gated at inference. Directly addresses persona-LoRA
composition for richer characters.

- ⚠ **Unverified for our domain.** Promising on paper, no public
  validation on Chinese-elder-persona use case yet.
- Mark as M9.5 (post-M9 enhancement). Don't block M9 on it; revisit
  after M9 ships and CharacterEval baselines exist.

**Cross-references.** Subsumes `RFC_M6` Phase 3. Independent of M10 but
shares the per-persona artifact directory.

**Effort estimate.** 6-8 weeks. (Heaviest milestone in this plan.)
M9.5 FinePE add-on +2-3 weeks if pursued.

---

### M10 — Multi-listener voice routing (2026-08 → 2026-09)

**Goal.** Finish the listener-routing path in `RFC_M5_MULTI_ADAPTER_VOICE_CLONING.md`
that's currently 40% built (per `RFC_M5` §15).

**Scope split — what survives an S2S pivot vs what doesn't.** The first
60% of the routing work (config schema, listener taxonomy, persona ×
listener matrix at the API surface) **survives any S2S pivot** — S2S
Talker models still need to know "which persona, addressing which
listener." The actual `set_active_adapter()` LoRA-swap mechanics
(`RFC_M5` §15 line 5) may become **obsolete** if we move to an S2S
model with system-prompt-style speaker selection (Qwen3.5-Omni-style,
see M13).

**Recommendation.** Build the persona/listener selection logic now (it
pays off regardless). **Defer the deep LoRA-swap engineering until after
M11 (TTS abstraction) ships and the M12a Step-Audio 2 mini eval
clarifies the long-term architecture.** A `set_speaker_prompt()` call
to an S2S Talker is structurally simpler than `set_active_adapter()`
LoRA swap — don't over-engineer the swap if it's about to be replaced.

**Dependencies.** None hard; can interleave with M9. Sequence M11
ahead.

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

### M11 — TTS engine abstraction (MEDIUM-HIGH PRIORITY — SHIP BEFORE M10/M12)

**Priority change (2026-05-28).** Bumped from LOW → **MEDIUM-HIGH**.
`BaseTTSEngine` is the **migration insurance policy** for the entire
TTS swap roadmap (Qwen3-TTS → IndexTTS-2 → Step-Audio 2 → Qwen3.5-Omni
when open). Without the abstraction, every swap requires invasive
surgery in `app/services/tts/qwen_tts_engine.py` and
`app/api/ws_asr.py`. Ship M11 **before or alongside M10/M12**, not
after.

**Goal.** Carve out a `BaseTTSEngine` interface so swapping TTS models
(IndexTTS-2, Step-Audio 2 mini, CosyVoice 2, future Qwen iterations,
hypothetical OSS clones) is plug-and-play.

**Immediate TTS upgrade candidate.** **IndexTTS-2** (Bilibili,
[GitHub](https://github.com/index-tts/index-tts)) — 3-10s reference
audio, emotion control, duration control, native CN/EN/JP. Drop-in
replacement for Qwen3-TTS via the new abstraction; independent of any
S2S migration. Worth a parallel eval track once `BaseTTSEngine` lands.

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

**Effort estimate.** 1-2 hours for the refactor itself; +1-3 days for
a first parallel implementation (IndexTTS-2 wrapper) to validate the
interface.

**Why this is now priority.** Multiple swap candidates are live
(IndexTTS-2 today; Step-Audio 2 mini for M12a; Qwen3.5-Omni when open).
Without the abstraction, each evaluation requires invasive `ws_asr.py`
surgery and risks breaking the shipped pipeline. Ship M11 first.

**Companion deliverable — real `talker.model` LoRA (promoted from
deferred).** Per `RESEARCH_SFT_S2S.md`, Qwen3-TTS-LoRA is the only
architecture in our budget where speaker identity can be deepened via
fine-tuning today. The `training_pipeline_deferred` item 1 (real
`talker.model` LoRA vs current `code_predictor`-only ~5-layer LoRA)
should land alongside M11 — not after M12a, not after M13. Expected
investigation budget: ~1-2 days to scope, ~1-2 weeks to land if the
gradient flow through `talker.model` is clean. This is the only
remaining lever for close-relative-recognizable cloning quality.

---

### M12 — Hybrid pipeline with Qwen 2.5 Omni 7B (2026-09 → 2026-10)

**Sequencing change (2026-05-28).** Before committing to M12, run two
spikes:

- **M12a: Step-Audio 2 mini evaluation track** (Apache 2.0, native CN,
  zero-shot voice cloning via text-audio token interleaving; CER 3.11%,
  URO-Bench CN 83.3% win-rate; [Repo](https://github.com/stepfun-ai/Step-Audio2)
  / [Tech Report arXiv:2507.16632](https://arxiv.org/pdf/2507.16632)).
  Run before or in parallel to M12. **Scope correction (per
  `RESEARCH_SFT_S2S.md`): this is a latency / prosody / streaming eval,
  NOT a cloning-depth eval.** Step-Audio 2 mini's cloning is zero-shot
  only — no SFT/fine-tune recipe is exposed ([Issue #67](https://github.com/stepfun-ai/Step-Audio2/issues/67)
  unanswered since 2025-10-08). If zero-shot quality isn't close-
  relative-recognizable (likely outcome for elder personas), there is
  no fine-tune escape hatch.
- **Qwen3.5-Omni-Light cloning spike (1-day eval)** — Light has open
  weights on HF as of 2026-03-30 ([Tech Report arXiv:2604.15804](https://arxiv.org/abs/2604.15804)).
  Spend one day testing zero-shot cloning quality on existing test/v11
  source audio. **Scope correction: latency / prosody eval, NOT
  cloning-depth eval.** Even if cloning quality lands at paper-claimed
  level, no SFT recipe is documented — Qwen-Omni line has a closed
  codec tokenizer (per [HF discussion #40](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/discussions/40)).
  M12 / M13 cannot collapse for cloning-depth purposes. They may
  collapse for **latency + emotion expressivity** purposes.

Update M12 dependencies: **spike first → decide architecture → execute.**

**Goal (if both spikes negative / inconclusive).** Eliminate the
ASR→LLM hop by collapsing ASR and LLM into **Qwen 2.5 Omni 7B**.
Retain Qwen3-TTS (or IndexTTS-2 via M11) for voice cloning. Target
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

**Status correction (2026-05-30).** Qwen3.5-Omni-**Light** is open-
weight on HF as of 2026-03-30 (`Qwen/Qwen3.5-Omni-Light-*`); the
**Plus** and **Flash** tiers remain API-only. Earlier drafts of this
roadmap said "weights not open today" — corrected here.

**Watch list (re-ranked 2026-05-30 with SFT-on-S2S findings).**

| # | Candidate | Signal | SFT for new speaker? | Status |
|---|---|---|---|---|
| 1 | **Step-Audio 2 / Step-Audio 2 mini (StepFun)** | Apache 2.0, native CN, zero-shot cloning via text-audio token interleaving; CER 3.11%, URO-Bench CN 83.3% | **NO** — no SFT recipe; [issue #67](https://github.com/stepfun-ai/Step-Audio2/issues/67) unanswered | **Latency/prosody eval only — M12a target** |
| 2 | **Qwen3.5-Omni-Light** | Open-weight Light tier (2026-03-30); Plus/Flash API-only; system-prompt-style cloning; zero-shot from 10-30s ref audio | **NO** — closed codec tokenizer (per [HF discussion #40](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/discussions/40)); SFT path undocumented | **Latency/prosody eval — 1-day spike** |
| 3 | **Kimi-Audio-7B (Moonshot)** | Built on Qwen 2.5 base — M9 persona LoRA may transfer with adapter surgery; ~14 GB; partial cloning | **NO** — single-speaker detokenizer (per [arXiv:2504.18425](https://arxiv.org/abs/2504.18425)); [issue #139](https://github.com/MoonshotAI/Kimi-Audio/issues/139) unanswered | Evaluable for ASR/understanding only |
| 4 | **LLaMA-Omni 2 (ICT-CAS)** | Built on Qwen 2.5 + CosyVoice2; 226-583ms latency; cloning inherits from CosyVoice2 | Inherits from CosyVoice2 (modular) | Evaluable |
| 5 | **GLM-4-Voice-9B (Zhipu/Tsinghua)** | Native CN+EN; instruction-controlled prosody (not ref-audio clone); good for prosody control | n/a — no ref-audio clone path | Evaluable |
| 6 | **Step-Audio 2.5 Realtime (StepFun)** | **Closed (API only)** — roleplay-specific RLHF + 10k+ persona-matrix training; closest-in-spirit to EverHome | n/a (closed) | **Reference / quality-ceiling benchmark** |
| 7 | Chroma 1.0 (FlashLabs) | English-trained but **architecture is language-agnostic** (correction from earlier roadmap claim — was "English-only"); Chinese fine-tune may be feasible | Unknown | Watch |
| 8 | Future Qwen3.5-Omni Plus/Flash open release | If/when QwenLM open-sources Plus/Flash, latency-side of M12+M13 collapses; cloning depth still depends on SFT recipe | Unknown — depends on tokenizer access | Watch |
| 9 | OSS cloning head on Qwen Omni (community or ours) | Architecture-level patch | Speculative | Speculative |

**Triggers to actually migrate.**

1. Model ships with verifiable voice cloning **AND a documented
   SFT/fine-tune recipe for new speakers** (paper benchmarks aren't
   enough — close-relative recognition requires deepenable cloning).
2. License is permissive (Apache 2.0 / MIT preferred).
3. Quality regression against M12 hybrid + current Qwen3-TTS-LoRA
   passes (50-prompt suite + per-(persona, listener) blind A/B with
   family member, per M8.5).
4. VRAM fits Spark 128 GB unified.

**Until triggers fire:** stay on M12 hybrid for latency; stay on
Qwen3-TTS-LoRA for cloning. Per `RESEARCH_SFT_S2S.md`, cloning-depth
migration is 12+ months out, not 6.

**Effort estimate.** 2-4 weeks **once a viable model exists** — most of
the cost is regression-testing, not integration (interface already
abstracted).

---

## 4. Architecture Prep for Future Swaps

Three interfaces + one watch script. None costs more than half a day.

### 4.0 Migration risk table — what survives an S2S pivot

Per the migration analysis in `RESEARCH_VERIFY_S2S_PERSONA.md` §4, here
is the milestone-by-milestone exposure to an S2S pivot in 6 months
(Step-Audio 2 mini or Qwen3.5-Omni-Light if cloning eval passes):

| Milestone | Survives S2S? | Investment recommendation |
|---|---|---|
| **M-Consent** | ✓✓ Survives. Compliance is model-agnostic. | Build now, full speed. |
| **M7** Text/ebook/image ingestion | ✓✓ Untouched by S2S. Corpus is upstream of any model. | Build now, full speed. |
| **M8** Memory RAG | ✓✓ Survives. ID-RAG architecture maps cleanly onto S2S. | Build now, full speed. |
| **M9** OpenCharacter persona LLM | ✓ Data + technique survive. LoRA target model may change (Qwen 3 8B → Step-Audio 2 base → Qwen3.5-Omni base). | Build the data pipeline and eval harness now. Train LoRA against current backbone but design for backbone-swap. |
| **M10** Multi-listener voice routing | ⚠ First 60% (config, routing, persona × listener matrix) survives. LoRA-swap mechanics may be replaced by S2S `set_speaker_prompt()`. | Build routing logic now, defer deep LoRA-swap until after M12a. |
| **M11** TTS engine abstraction | ✓✓ **Critical.** This *is* the migration insurance policy. | **Bumped to MEDIUM-HIGH priority. Ship before M10/M12.** |
| **M12** Qwen2.5-Omni hybrid | ⚠ At risk — Step-Audio 2 mini may be a better target. | Run M12a Step-Audio 2 mini + Qwen3.5-Omni-Light spike before committing M12. **Re-scoped: latency / prosody eval only, NOT cloning-depth eval** (per `RESEARCH_SFT_S2S.md`). |
| **M12a** Step-Audio 2 mini + Qwen3.5-Omni-Light spike | ⚠ Partial — latency wins survive, cloning-depth path does not. | **Scope correction: this is a latency / prosody / streaming eval. It cannot collapse the cloning-depth problem** — no OSS Chinese S2S exposes a working SFT recipe for new speakers. Cloning stays on Qwen3-TTS-LoRA. |
| **M13** OSS E2E S2S w/ cloning | The whole point. Was opportunistic; **stays opportunistic** — no open S2S can be SFT'd today, so cloning-depth migration is 12+ months out, not 6. | Watch Step-Audio 2 / Qwen3.5-Omni for an SFT recipe to drop. Until then, Qwen3-TTS-LoRA remains the cloning path. |

**TTS LoRA stance (updated 2026-05-30 per `RESEARCH_SFT_S2S.md`).**
Earlier drafts of this doc recommended freezing TTS LoRA feature work.
**Softened.** The SFT-on-S2S audit found that **no open-weight Chinese
S2S model supports SFT for new speakers today** (Qwen-Omni codec
tokenizer is closed; Kimi-Audio detokenizer is single-speaker; Step-
Audio 2 mini is zero-shot only with no SFT recipe; Qwen3.5-Omni Plus/
Flash variants are API-only and Light has no documented SFT path).
Cloning depth via S2S is therefore **12+ months out**, not 6.

Consequence: the shipped Qwen3-TTS LoRA path is **the moat**, not sunk
cost. **Real `talker.model` LoRA** (per `training_pipeline_deferred`
item 1) is **upgraded in priority** — it was deferred work; it should
now be folded into M8.5 / M11 as a first-class deliverable. Zero-shot
speaker prompts capture gross timbre + average prosody; SFT/LoRA on a
dedicated TTS is the only architecture in our budget that captures
close-relative-recognizable micro-features.

### 4.1 `BaseTTSEngine` (sketched in §M11) — CRITICAL, SHIP BEFORE ANY S2S EVAL

Lets us swap Qwen3-TTS for IndexTTS-2 / Step-Audio 2 / Qwen3.5-Omni
(when open) / CosyVoice 2 / future iterations without touching
`ws_asr.py`. Minimum surface:

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
| **OpenCharacter (Salesforce)** | [arXiv 2501.15427](https://arxiv.org/abs/2501.15427) (Jan 2025) | M9 — 326K-row persona-instruction dataset for SFT mix |
| **Open Character Training (Constitutional AI)** | [arXiv 2511.01689](https://arxiv.org/abs/2511.01689) (Nov 2025) | M9 — Constitutional AI methodology, 3-stage SFT→DPO recipe |
| Anthropic Constitutional AI | arXiv 2212.08073 (Bai et al. 2022) | M9 — methodology foundation |
| OPLoRA | arXiv 2510.13003 | M9 — orthogonal LoRA, preserves base |
| **Tencent PersonaHub** | [GitHub](https://github.com/tencent-ailab/persona-hub) | M9 — 1B synthetic personas, 370M elite subset, data augmentation for low-data elders |
| **CharacterEval (CN benchmark)** | [arXiv 2401.01275](https://arxiv.org/abs/2401.01275) | M9 — Chinese-native persona benchmark, acceptance gate |
| **Memory-Driven Role-Playing** | [arXiv 2603.19313](https://arxiv.org/pdf/2603.19313) | M9 — 4-stage evaluation methodology (cue → retrieve → integrate → respond) |
| **AMADEUS / CharacterRAG dataset** | [arXiv 2508.02016](https://arxiv.org/pdf/2508.02016) | M9 — training-free RP eval harness, 15-character/450-QA dataset |
| **Persistent Personas? (extended interactions)** | [arXiv 2512.12775](https://arxiv.org/pdf/2512.12775) | M8/M9 — persona fidelity degrades over 100+ rounds; mitigation = identity anchor reinjection |
| **FinePE (MoLE persona editing)** | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494626003911) | M9.5 — Mixture of LoRA Experts for persona-subtrait composition (⚠ unverified) |
| **ID-RAG (Identity RAG)** | [arXiv 2509.25299](https://arxiv.org/abs/2509.25299) (Sept 2025) | M8 — primary architecture; identity knowledge graph retrieved each turn |
| **HippoRAG 2 (From RAG to Memory)** | [arXiv 2502.14802](https://arxiv.org/html/2502.14802v1) | M8 — semantic-facts retriever (KG + Personalized PageRank) |
| **A-MEM (Agentic Memory)** | [arXiv 2502.12110](https://arxiv.org/pdf/2502.12110) | M8 — conversation-time memory consolidation candidate |
| **Mem0 (Scalable Long-Term Memory)** | [arXiv 2504.19413](https://arxiv.org/pdf/2504.19413) | M8 — conversation memory candidate (alt to A-MEM) |
| BGE-M3 | HuggingFace `BAAI/bge-m3` | M8 — Chinese-native hybrid embedding (retained as primitive) |
| Anthropic Contextual Retrieval | anthropic.com/news/contextual-retrieval | M8 — chunk-prefix-with-context |
| Persona Drift | arXiv 2402.10962 | M9 — drift measurement + anchor reinjection |
| **PaddleOCR-VL 1.5** | [HF PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) ([tech report arXiv 2507.05595](https://arxiv.org/html/2507.05595v1)) | M7 — OCR (handwriting + CN simplified/traditional + pinyin + vertical text) |
| **MinerU 2.5-Pro** | [GitHub opendatalab/MinerU](https://github.com/opendatalab/MinerU) | M7 — PDF/document parser (Chinese-native, VLM-based) |
| **IndexTTS-2** | [GitHub index-tts/index-tts](https://github.com/index-tts/index-tts) | M11 — immediate TTS upgrade candidate (3-10s ref audio, emotion + duration control) |
| **Step-Audio 2 mini** | [arXiv 2507.16632](https://arxiv.org/pdf/2507.16632) / [GitHub stepfun-ai/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) | M12a / M13 — primary open-weight S2S candidate, Apache 2.0 |
| **Step-Audio 2.5 Realtime** | [MarkTechPost](https://www.marktechpost.com/2026/05/24/stepfun-releases-stepaudio-2-5-realtime-an-end-to-end-voice-model-with-roleplay-specific-rlhf-and-paralinguistic-comprehension/) | M13 — closed API; benchmark / quality-ceiling reference (10k+ persona RLHF) |
| **Qwen3.5-Omni Technical Report** | [arXiv 2604.15804](https://arxiv.org/html/2604.15804v1) | M12a / M13 — Light variant open-weight on HF as of 2026-03-30; cloning spike target |
| Qwen 2.5 Omni 7B | huggingface.co/Qwen | M12 — hybrid ASR+LLM (fallback if M12a spikes inconclusive) |
| Qwen 3 Omni 30B-A3B | huggingface.co/Qwen | NOT used — doesn't fit hardware budget |
| Chroma (FlashLabs) | flashlabs.ai | M13 — watch list (English-trained, architecture language-agnostic — correction from earlier roadmap) |
| LanceDB | lancedb.com (Apache 2.0) | M8 — vector store |
| **Qwen 3 8B** | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | M9 — persona LoRA base (replaces Qwen 2.5 7B) |
| Qwen3-TTS 1.7B VoiceDesign | huggingface.co/Qwen | All milestones — TTS retained for cloning until M11 swap |
| **The Making of Digital Ghosts** | [arXiv 2511.20094](https://arxiv.org/pdf/2511.20094) (Nov 2025) | M-Consent — ethical framework for deceased-person modeling |
| **Digital Doppelgangers (pre-mortem clones)** | [arXiv 2502.21248](https://arxiv.org/html/2502.21248v1) | M-Consent — revocability flow for still-living personas |
| **EMORL-TTS** | [arXiv 2510.05758](https://arxiv.org/abs/2510.05758) | M8.5 — emotion-vocabulary-first methodology; train TTS on closed emotion set before downstream LLM/persona work |
| **Characteristic-Specific Partial Fine-Tuning** | [arXiv 2501.14273](https://arxiv.org/abs/2501.14273) | M8.5 — staged fine-tuning rationale: downstream behaviors conditioned on upstream vocabularies, not co-trained |
| **CosyVoice 2** | [arXiv 2412.10117](https://arxiv.org/abs/2412.10117) (Dec 2024) | M8.5 — flagship instruction-conditioned TTS paradigm; dual-loss training recipe to prevent speaker drift under instruct |
| **StyleTTS 2** | [arXiv 2306.07691](https://arxiv.org/abs/2306.07691) | M8.5 — adaptive style transfer via diffusion; preserve-speaker-vary-style pattern |
| **Bark (Suno-AI)** | [GitHub suno-ai/bark](https://github.com/suno-ai/bark) | M8.5 — style-token approach reference for controllable TTS vocabulary design |
| **emotion2vec (FunASR)** | [GitHub modelscope/FunASR](https://github.com/modelscope/FunASR) / `iic/emotion2vec_plus_large` | M8.5 — primary auto-labeler for Phase A emotion labeling of recorded corpus |
| **wav2vec2-emotion** | [HF audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) | M8.5 — cross-check emotion classifier |
| **openSMILE (eGeMAPSv02)** | [audEERING openSMILE](https://github.com/audeering/opensmile) | M8.5 — paralinguistic feature extraction for manual review |
| **SFT-on-S2S audit (internal)** | `docs/RESEARCH_SFT_S2S.md` (2026-05-28) | M11 / M12a / M13 — no open Chinese S2S supports SFT for new speakers; cloning depth stays on Qwen3-TTS-LoRA for 12+ months |

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

**Priority changes 2026-05-28:**

- **M11 (TTS engine abstraction) bumped from LOW → MEDIUM-HIGH.**
  Ship before M10/M12. Without it, every TTS swap evaluation (IndexTTS-2,
  Step-Audio 2 mini, Qwen3.5-Omni-Light) requires invasive surgery in
  `ws_asr.py`.
- **M-Consent inserted before M7** (parallel track). Compliance is a
  commercial blocker for B2B sales.
- **M12a (Step-Audio 2 mini + Qwen3.5-Omni-Light spikes) added** as a
  prerequisite gate to M12.

---

## 10. Sequencing at a glance

> **Updated 2026-06-03 per GPT-5 + Gemini 2.5 Pro reviews** (transcripts
> at `docs/REVIEW_GPT5_2026-06-03.md` + `docs/REVIEW_GEMINI25PRO_2026-06-03.md`,
> action items in `RFC_M6_PERSONA_LLM_LEGACY.md` §11). Changes from
> prior version:
>
> From GPT-5:
> - **D-Retro** inserted right after M-Demo to harden 06/02 fixes.
> - **License audit** ships as a blocker before M7 ingest code lands.
>   DONE — see `docs/THIRD_PARTY_LICENSE_AUDIT.md`. Hard blockers:
>   PersonaHub (cc-by-nc-sa) and LLaMA-Omni 2 weights (research-only).
> - **AudioWorklet migration** — added to D-Retro then DEFERRED per
>   user (upload-mode is current path).
> - **M-Consent UI gates M7 ingest UI** (was parallel; now serial-at-UI).
> - **M11 BaseTTSEngine** broadens scope to include VoxCPM 2 +
>   CosyVoice 2 as first-class backends.
>
> From Gemini 2.5 Pro (this iteration):
> - **M12a pulled to top of queue** (1-day spike, after D-Retro) — its
>   outcome can reshape M11/M12/M13.
> - **M9 split** into M9.1 (local Qwen 3 8B + simple LoRA, 2 weeks,
>   privacy moat closes here) → M9.2 (rich LoRA with M8.5 vocabulary,
>   3-4 weeks, hot-swap on running infra).
> - **M7 split** into M7.1 (txt/md/pdf minimal, 3-5d, unblocks M8) →
>   M7.2 (epub/docx/photos/chat exports, ~2w, after M8 wired).
> - **Strategic reframe:** moat = cloning depth × curated corpus +
>   memory graph. M7+M8+M9 are co-equal moat investments. Per-family
>   data + memory is the durable hard-to-copy asset; cloning depth is
>   necessary but not sufficient. See PROJECT_BRIEF §1.
> - **M10 is the publication opportunity.** Per-listener LoRA
>   composition from the same speaker is rare in TTS literature;
>   EverHome has structurally unusual training data (per-listener
>   labeled recordings). Build M10 publication-ready from day one
>   (controlled ablations, documented method, held-out family eval).
>   See RFC §11.13.
> - **M-Onboarding** added as a pre-launch milestone — guided
>   multi-session data-collection app; the "data curator" UX is
>   currently the largest under-addressed gap (Gemini D-1 finding).

```
2026
│
├─ MAY 28  ─┐  (now)
│           ├─ M-Demo polish (USB mic, retrain SFT, fallback recording)
├─ JUN 02  ─┘  ← NY Tech Week demo
│
├─ JUN 03  ─┐
│           ├─ D-Retro (DONE 2026-06-03)
│           │   · Contract tests for ASR hallucination filter, listen-only,
│           │     empty-asr handling, language directive, cache-buster ✓
│           │   · [DEFERRED] ScriptProcessorNode → AudioWorkletProcessor —
│           │     upload-mode is current path
│           │   · §12 retrospective writeup ✓
│           │   · License audit ✓ (THIRD_PARTY_LICENSE_AUDIT.md;
│           │     blockers: PersonaHub, LLaMA-Omni 2 weights)
│           │
│           ├─ M12a — Cloning eval spike (1 day, NEXT after D-Retro)
│           │   · Qwen3.5-Omni-Light zero-shot cloning on 30s Mark sample
│           │   · Step-Audio 2 mini zero-shot cloning
│           │   · A/B against current Qwen3-TTS-LoRA via human eval rubric
│           │   · Decision tree: collapse M11/M12/M13 if cloning matches;
│           │     confirm current path if clearly below
│           │
│           ├─ Unlearning design spike (1 day, M-Consent prework)
│           │   · SISA-shard granularity decision (USENIX Security 2019)
│           │   · Per-shard A10G retrain cost estimate
│           │   · Revocation SLA draft
│           │
│           └─ Release tracking automation (small)
│               · Watch script: HF/arXiv/GitHub poll for VoxCPM, CosyVoice,
│                 GLM-4-Voice, Kimi-Audio, Qwen3.5-Omni
│
├─ JUN     ─┐
│           ├─ M-Consent UI ships FIRST (gates M7 ingest UI server-side)
│           ├─ M7.1 — Minimal ingest (3-5d): .txt / .md / .pdf only
│           │      · Unblocks M8; 80% of typical first-family corpus
│           │      · POST /api/corpus/ingest checks consent record exists
├─ JUL     ─┘
│
│           ★ M11 — TTS engine abstraction (BUMPED priority — ship here, before M10/M12)
│              · BaseTTSEngine + Qwen3TTSEngine + VoxCPM2Engine +
│                CosyVoice2Engine (all first-class, A/B-able)
│
├─ JUN-JUL ─┐
│           ├─ M8 — Memory RAG (incremental, 3 commits inside one milestone)
│           │      · M8.1 (3-5d): embeddings + LanceDB + retrieve + ws_asr injection
│           │           ⇒ AI cites corpus ("根據你的 email...")
│           │      · M8.2 (~7d): identity graph + HippoRAG multi-hop
│           │           ⇒ AI connects facts (mom + medication + appointment)
│           │      · M8.3 (3-5d): conversation memory + consolidation (A-MEM / Mem0)
├─ AUG     ─┘           ⇒ Cross-session memory ("上次你問了...")
│
├─ JUN-JUL ─┐
│           ├─ M8.5 — Instruction-conditioned TTS fine-tuning (emotion2vec labels → mixed-instruct SFT)
├─ JUL     ─┘    ← locks emotion vocabulary; BLOCKS M9
│              · Real talker.model LoRA pulled forward as Week-0 subtask
│
├─ JUL     ─┐
│           ├─ M9.1 — Local Qwen 3 8B + simple persona LoRA (~2 weeks)
│           │      · Ships local-LLM deployment infra (vLLM, serving,
│           │        latency, OpenAI swap-out) — **privacy moat closes here**
│           │      · Uses current emotion vocab (pre-M8.5); may re-train later
│           │
├─ AUG     ─┘
│
├─ AUG     ─┐
│           ├─ M9.2 — Rich persona LoRA with M8.5 vocabulary (~3-4 weeks)
│           │      · Re-train with emotion2vec-anchored vocab
│           │      · Hot-swap on running local LLM (no infra change)
│           │
├─ SEP     ─┘
│
├─ AUG     ─┐
│           ├─ M10 — Multi-listener voice routing (PUBLICATION-CANDIDATE)
│           │      · Per-(speaker,listener) LoRA composition — rare in TTS lit
│           │      · Build publication-ready from day one (ablations,
│           │        held-out family eval, documented method).
│           │      · Target venue: ICASSP 2027 or Interspeech 2027.
├─ SEP     ─┘
│
├─ AUG     ─┐
│           ├─ M12a — Step-Audio 2 mini eval + Qwen3.5-Omni-Light cloning spike
├─ SEP     ─┘    (highest-leverage single action; may collapse M12+M13)
│              · Also bake-off Kimi-Audio-7B + LLaMA-Omni 2 as M12 base candidates
│
├─ SEP     ─┐
│           ├─ M12 — Hybrid pipeline (winner from M12a bake-off)
├─ OCT     ─┘   — IF M12a spikes inconclusive
│
├─ Q3 +     M-Onboarding — guided multi-session data-collection app
│              (the "data curator" UX; pre-launch blocker)
│              Per Gemini D-1 finding — currently the largest
│              under-addressed product gap.
│
├─ Q3 +     M7.2 — Full ingest (epub, docx, photos, chat exports)
│              Ships after M8.1 + M8.2 wired and proven on M7.1 inputs
│
└─ Q4 +     M13 — OSS E2E S2S migration (Step-Audio 2 primary; Chroma watch)
```

**Hard dependency edges:**
- M-Consent UI **gates** M7.1 (server-side consent check before any ingest)
- License audit **blocked** M7 + M9 code — DONE 2026-06-03
- D-Retro contract tests **gate** M7.1 (locks 06/02 fixes from regression)
- M7.1 **precedes** M8 (minimal corpus to retrieve from). M7.2 ships after M8 proven.
- M8.1 / M8.2 / M8.3 are sequential within M8 (each commit additive on prior)
- M8.5 **blocks** M9.2 (rich LoRA needs locked vocabulary). M8.5 does NOT block M9.1.
- M9.1 **precedes** M9.2 (hot-swap on same infra)
- M11 abstraction **precedes** M10 + M12 (swap-ready scaffolding first)

**Parallelizable edges:**
- M-Consent backend + M7 engine work (only the UI gating is serial)
- M8 + M8.5 (different code paths, share M7 corpus)
- M12a may run alongside M9/M10 (different person if team grows)

---

## 11. Demo learnings (2026-05-30)

Surfaced during 06/02 NY Tech Week demo prep. Captured here so the
roadmap reflects observed behavior, not aspirational design.

- **CustomVoice + per-sentence `instruct` mutual interference.** Passing
  a varying `instruct` per sentence (Path A — drive emotion via TTS
  instruct) made the SFT-trained custom_voice drift — each sentence
  sounded like a different person. Reverted to `instruct=None` +
  `language="Chinese"`. Listener tone-shift now relies on LLM word
  choice + tone-chip visual only. **This is the motivating evidence
  for M8.5** (instruction-conditioned TTS fine-tuning).
- **Codec prefill matters: `"Chinese"` vs `"auto"` is not equivalent
  for SFT-trained models.** The SFT model was trained against the
  `"Chinese"` prefill (4 tokens with `codec_language_id=2055`);
  switching to `"auto"` prefill (3 tokens, no-think mode) shifted v15's
  voice to faster / higher pitch / distortion. Beijing-accent fix is
  actually via `spk_is_dialect=false`, **NOT** via language. This
  refines the `tts_inference_language_gotcha` memory: `language="auto"`
  is correct for zero-shot models; **for SFT-trained custom_voice
  models, match the language prefill used during training**.
- **Listen-only mode.** New control where ASR fires but LLM doesn't
  respond — enables a "voice Turing test" demo flow where the audience
  hears the cloned voice repeat back without conversational pressure.
  Production code; not yet documented in the WS protocol section of
  `CLAUDE.md`.
- **Hybrid disclosure rule.** Persona JSON now has a `disclosure_rule`
  field; the LLM must disclose AI identity if directly asked but
  otherwise speaks in first person. This sits between "always
  disclose" (kills the demo) and "never disclose" (ethical hazard +
  M-Consent compliance risk). Should be folded into the M-Consent
  ethical-position deliverable.
- **EverHome demo persona.** A rich English system prompt with founder
  bio (ex-Bloomberg, ex-Google), product moat, competitive analysis,
  and roadmap context was built for the 06/02 audience. This persona
  is currently dev-only; productizing it (per-deployment "company
  spokesperson" persona kind) is out of M7-M13 scope but worth
  tracking.

---

## 12. Demo retrospective (2026-06-02)

Demo shipped. NY Tech Week audience ~80-90 vibe-coding-curious. 8 minutes,
2nd of 2 demoers, persona = "EverHome Demo (Mark)". Recording produced
post-demo via Loom Chrome-tab capture (Share-tab-audio path).

### What worked

- **Voice clone reveal landed.** Audience reacted at the "let AI continue"
  moment when the AI continued the founder's spoken intro in his voice.
- **Persona / listener switch** showed observable tone difference between
  Child / Reporter / Friend with same voice.
- **Disclosure moment** — "Are you a real person?" → AI confirmed clone
  identity. Got nods from audience; sets up M-Consent narrative naturally.
- **Cloudflare named tunnel** (`everhome.mkk.dev`) held the whole demo;
  no 524 timeouts despite the longer TTS preview calls.
- **Single-page chat UI** without Gradio was fast enough; no first-paint
  hiccups on the conference WiFi.

### What broke (and was fixed live or pre-demo)

| Symptom | Root cause | Fix |
|---|---|---|
| ASR fired "The first was the first to be built." on silence | Qwen3-ASR (Whisper-family) hallucinates on near-silent buffers | Hallucination-text set + peak-amplitude<0.12 silence guard in `engine.py recognize()` |
| Listen-only mode stuck in "AI thinking…" forever after first turn | Server skipped LLM but never sent any signal → client FSM had no transition trigger | Server always sends `asr_result` even when empty; client re-arms mic on listen-only / empty-asr |
| Language dropdown changes didn't reach server | Browser cached `standalone.js` for 4 hours (Cloudflare edge `max-age=14400` overrode our `no-cache` middleware) | Added `?v={mtime}` query string to script tag; bumps on every deploy |
| English replies sound fast / flat (not founder's flow) | TTS LoRA trained on Chinese audio only — prosody is Chinese, mapped onto English tokens | Deferred fix; Mark to record English audio when time allows. Workaround: UI Language dropdown lets the user pin to Chinese for natural prosody |
| Persona dropdown defaulted to xiao_s on demo machine | HTML hardcoded default + JS fallback both prefer xiao_s | Both changed to default to `test` (EverHome Demo / Mark) |
| ASR hallucinations bled into LLM call | Server forwarded any non-empty ASR text to LLM | Empty-text drop in `commit_utterance` handler + hallucination filter |

### What the audience asked about

- "Is the voice really running locally?" — yes; on the GPU box, not a
  cloud API. (Currently true for TTS only; LLM is still OpenAI until M9.)
- "Privacy?" — confirms B2B compliance angle is what they zero in on.
  Reinforces M-Consent priority.
- "How much?" — no price asked-and-told (consumer-appliance range, no
  subscription, no fees). Validates appliance positioning.
- "Why you?" — short ex-Bloomberg / ex-Google + personal motivation.
  Don't oversell.

### Roadmap deltas (this section's writeup)

Captured in detail in §10 and `RFC_M6_PERSONA_LLM_LEGACY.md` §11:

- **D-Retro bucket** (this section's fixes) → contract tests + AudioWorklet
  migration. 1-2 days. Locked in before M7 starts.
- **License audit** is a blocker before M7 / M9. ~1 day.
- **M-Consent UI gates M7** (was parallel, now serial-at-UI-layer).
- **M8a thin-slice memory** (3-5 days) inserted before full M8.
- **Unlearning design spike** (SISA training pattern) as M-Consent prework.
- **M11 BaseTTSEngine** broadens to include VoxCPM 2 + CosyVoice 2 as
  first-class backends (not stubs).

### Anti-pattern observed

I (Claude) shipped a server-side `RMS < 0.005` silence guard in
`state_manager.commit_utterance` mid-demo that dropped real phone-mic
speech (low AGC output) → empty ASR → client stuck in THINKING. Reverted
within minutes. Lesson: **buffer-averaged RMS is the wrong discriminator
for silence vs speech with AGC-amplified mobile audio.** Peak amplitude
in the ASR engine + known-text filter on the output is the working
discriminator (validated against `《大明宫词》` peak=0.086 and `Hi, I'm Mark...`
peak=0.517 in the live log).

---

## 13. Document conventions

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
