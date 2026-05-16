# RFC M6 — Persona LLM + Legacy-Preservation Product

**Status:** Draft, in active discussion (2026-05-15 → )
**Owners:** Mark (product), Claude (architecture/code)
**Supersedes for LLM scope:** ad-hoc prompt-manager + RAG stub in `ws_asr.py:683`
**Cross-refs:** RFC_2_2 (persona factory), RFC_M4 (LoRA training infra), RFC_M5 (multi-adapter routing)

## 1. Product framing (the real goal)

The system is **not** a celebrity-chatbot toy. The end product is a privacy-first
"preserve a person" appliance:

- An elder (or any individual) uploads their own data: text, books, social
  posts, chat-app exports, photos of letters, recorded voice, video.
- The system trains a per-person voice model (already built — Qwen3-TTS LoRA
  pipeline, RFC_M4) and a per-person LLM persona-LoRA (this RFC).
- Family members later interact via a separate family-facing UI; the
  appliance responds in the person's voice + cadence + factual memories.
- **Everything runs on a box the family owns.** No data leaves.

Famous-figure (小S) demo is the **validation path** for the same pipeline,
not a separate product. Public figures simply happen to have a large public
corpus available, so the ingestion pipeline can be exercised end-to-end
before we have organic per-elder data.

## 2. Three-tier architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Base instruction-tuned LLM (frozen)                            │
│   - Qwen2.5-7B-Instruct AWQ-Q4 (~6 GB VRAM)                    │
│   - Optionally 14B-Q4 (~9 GB) for headroom on Spark            │
│   - Provides: reasoning, safety, instruction following         │
└────────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌────────────────────────────────────────────────────────────────┐
│ Persona-LoRA (per individual, swappable)                       │
│   - Trained on harvested in-character text                     │
│   - Encodes HOW they talk (cadence, idioms, opinion)           │
│   - Mirror of the TTS-LoRA path we already understand          │
└────────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌────────────────────────────────────────────────────────────────┐
│ RAG over their corpus (per individual, dual-index)             │
│   - Style index: verbatim quips, retrieved as few-shot         │
│   - Stance index: extracted opinion graph, retrieved as fact   │
│   - Embedding: BGE-M3 (Traditional Chinese-native, hybrid)     │
│   - Chunking: Anthropic-contextual (chunk + LLM-generated ctx) │
└────────────────────────────────────────────────────────────────┘
```

### Why this stack

| Layer | What it does | Alt considered | Why rejected |
|---|---|---|---|
| Base | Reasoning, safety, following | Train from scratch | Cost-prohibitive, no edge |
| Persona-LoRA | Cadence, idioms, "voice" | Prompt-only | Base model has zero prior on a regular elder; prompt-only is generic |
| Persona-LoRA | Cadence, idioms, "voice" | Full SFT | Destroys instruction-following at small corpus sizes (≤10k turns) |
| RAG | Specific stories, opinions | Bake into LoRA | LoRA can't memorize specifics; RAG is editable post-train |
| BGE-M3 | Embedding | OpenAI text-embedding-3 | Worse Chinese; can't run local |
| Anthropic chunking | Chunk strategy | Sliding window | ~49% fewer top-20 misses per Anthropic's published benchmark |

## 3. Roadmap

### Phase 0 — Corpus ingestion (blocks everything)

**New surface area in the repo:**

```
data/personas/<persona_id>/
├── corpus/
│   ├── text/          ← .txt .md .pdf .epub uploads
│   ├── transcripts/   ← podcast/interview/video → ASR'd text
│   ├── conversations/ ← Line/WeChat/WhatsApp exports
│   └── metadata.json  ← per-file source, date, listener tag, ingest_status
├── manifest.json      ← rolled-up stats, ready_for_training flag
└── …  (existing: voice/, lora/)
```

**New endpoints (extend `app/api/recordings.py` lineage):**

- `POST /api/corpus/upload?persona_id=&kind=` — text/pdf/epub/audio/video
- `POST /api/corpus/ingest?persona_id=&item_id=` — kick off processing (background)
- `GET  /api/corpus/{persona_id}` — list items + states
- `GET  /api/corpus/{persona_id}/manifest` — rolled-up stats
- `DELETE /api/corpus/{persona_id}/items/{item_id}`

**Processing pipeline (background jobs):**

| Input | Steps |
|---|---|
| `.txt`, `.md` | normalize encoding, segment to chunks |
| `.pdf` | pdfplumber/PyMuPDF text extract → fall back to OCR (Tesseract chi_tra) |
| `.epub` | ebooklib → chapter splits |
| `.docx` | python-docx |
| audio | reuse existing pipeline.py: denoise → diarize → ASR → speaker-filter (keep persona-side only via ECAPA-TDNN similarity to seed voice clips) |
| video | ffmpeg → audio path; OCR on text overlays if present |
| chat exports | parser per platform (Line export ZIP, WeChat backup, WhatsApp .txt) |

**UI (extend dev UI, not the family UI):**

- New "Corpus" tab on `/ui/recordings` (or its own page)
- Per-persona view: size by kind, last ingested, "ready_for_training" status
- Drag-drop upload

**Storage thresholds (manifest flags):**

- `ready_for_rag`: ≥ 50 chunks indexed
- `ready_for_lora_synthetic_expansion`: ≥ 500 organic in-character turns
- `ready_for_lora_organic`: ≥ 5000 organic in-character turns

### Phase 1 — 小S famous-figure RAG demo (validates pipeline)

- Ingest 小S corpus via Phase 0 infra:
  - 13 IG + 15 podcast recordings → ASR + speaker-filter → ~10k chars
  - Her books (5+ titles) — to be sourced as EPUB or OCR
  - Podcast/TV transcripts — fan-made subs where legal
  - Interview articles
- Build dual-index RAG:
  - Style index: verbatim quips/zingers from transcripts/books
  - Stance index: LLM-extracted (topic, stance, evidence, source) tuples
  - BGE-M3 hybrid, FAISS-flat in-RAM
  - Anthropic contextual prefix per chunk
- **LLM: OpenAI API** (gpt-4o-mini, acceptable since 小S is public)
- Demoable end-to-end: family member talks to "小S", responses pull from corpus.

### Phase 2 — Local LLM swap-in

- Qwen2.5-7B-Instruct AWQ-Q4 served via vLLM or llama.cpp on the box
- Same prompt + RAG; only the inference endpoint changes
- Validates "100% local" privacy property
- Drops OpenAI cost / dependency

### Phase 3 — Persona-LoRA training

- Harvest in-character text from the corpus
- For <500 organic turns: **Constitutional-AI-driven synthetic expansion**
  per Open Character Training (2511.01689, Nov 2025) — generate
  introspective dialogue from the seed corpus + constitution document
- SFT LoRA on Qwen2.5-7B base, mix 30-50% general instruction data to
  preserve following
- Apply OPLoRA orthogonality (2510.13003) to protect base capabilities
- DPO on (in-character, off-character) pairs harvested from self-critic
- Per-persona LoRA file, activated alongside TTS LoRA at session start

### Phase 4 — Legacy product polish

- Family-facing UI (separate from dev UI) — mobile-friendly, no debug
- Auto-importers: Line/WeChat/WhatsApp export parsers, photo→letter OCR
- Listener memory (per-family-member conversation persistence)
- Optional: birthday/holiday speak-in-their-voice reminder

## 4. Open decisions

1. **Privacy ratchet for training-data prep**
   - **Soft** (preferred): inference 100% local; training-data synthesis may
     hit a frontier API once per persona during initial training
   - **Hard**: nothing ever leaves the box. Forces ≥10k organic turns per
     persona; small-corpus elders can't be served well.
   - **Decision:** *pending*

2. **Hardware target for shipped appliance**
   - **DGX Spark 128 GB GB10** (~$3-4k) — fits training + inference,
     best-in-class for this workload
   - **PC w/ RTX 5090 32 GB + 64 GB sys-RAM** — close second, requires
     more setup
   - **Mac Mini M4 Max 64 GB** — only inference, training must be cloud
   - **Decision:** Spark for production demo box; current AWS g5.4xlarge for dev

3. **Demo path**
   - Famous-figure (小S) first, then pivot to elder
   - **Decision: locked in 2026-05-16** — same pipeline serves both;
     famous-figure is validation, not a fork

4. **Listener taxonomy**
   - Fixed (child/mom/friend/default) or growing (named family members)
   - **Decision:** *pending; default to growing*

5. **Family UI scope**
   - **Decision: locked 2026-05-16** — separate from dev UI; mobile-first

## 5. Hardware feasibility numbers

| Component | Model | VRAM (FP16) | VRAM (Q4) |
|---|---|---|---|
| Base LLM (inference) | Qwen2.5-7B-Instruct | 14 GB | 6 GB |
| Base LLM (inference) | Qwen2.5-14B-Instruct | 28 GB | 9 GB |
| TTS | Qwen3-TTS 1.7B VoiceDesign | 3.4 GB | n/a |
| ASR | Qwen3-ASR | ~3 GB | ~2 GB |
| VAD / embedding / KV-cache | various | 3-5 GB | 3-5 GB |
| **Total inference** | **7B path** | **23 GB** | **~17 GB** |
| **Total inference** | **14B path** | **40 GB** | **~20 GB** |
| Persona-LoRA training | Qwen2.5-7B + rank 16 | 24 GB min, 40 GB ok | n/a |

A10G (23 GB) fits 7B-Q4 inference comfortably. DGX Spark (128 GB unified)
fits 14B training + inference simultaneously with headroom.

## 6. Data scale rule-of-thumb (persona-LoRA SFT)

| Organic in-character turns | Result |
|---|---|
| < 500 | Synthetic expansion mandatory; generic-leaning |
| 500 – 3 000 | Usable, recognizable, occasional drift |
| 5 000 – 10 000 | Distinctive cadence, low drift, ship-quality |
| 10 000 + | Confidently in-character |

For reference: 小S's 65 min of speaker-filtered audio yields ~300 turns —
**below the floor**. Her books will push us well past 10k once ingested.

## 7. Startup-edge analysis

The **technology** is not a moat — the stack is open-source and replicable
in a quarter by a competent team. The moats are:

1. **Privacy-by-architecture (auditable).** Families uploading mom's letters
   need to *believe* the data won't ship out. Hardware-attestable claim
   is the strongest version. No cloud-only competitor can match this.
2. **Voice × LLM fidelity together.** Most legacy products are text-only or
   ship low-quality voice. We already have the high-fidelity voice path.
3. **Chinese (Traditional) first-class.** Replika / Character.AI optimize
   for English. Qwen2.5 + Qwen3-TTS + BGE-M3 are Chinese-native at every
   layer. Real edge for Taiwanese / HK / overseas Chinese families.
4. **Per-individual fidelity, not generic NPC.** Character.AI vibes are
   "fictional character"; this is "your actual mother". Different emotional
   contract — and different willingness to pay.
5. **Hardware as the product.** Sell the Spark/Mini-PC pre-configured.
   Recurring revenue from corpus-ingestion service contracts or storage.
   Per-unit unit economics, not per-token.
6. **Open-weights stack.** Not subject to API price hikes or model
   deprecation. Predictable cost per unit shipped.

### Where competitors are weak

- **HereAfter AI, StoryFile** — text or low-fidelity-voice; cloud.
- **Replika, Character.AI** — generic NPCs; cloud; English-first.
- **Personal AI, Pi.ai** — assistant-with-personality, not preserve-a-person.
- **Chinese-market virtual-idol startups** — cloud; targeted at fans, not
  family.

### Where this project could lose

- **Distribution** — how do bereaved families find you? Funeral-home /
  hospice partnerships? Diaspora-community marketing? Unclear yet.
- **Trust** — privacy claims have to survive audit. May need a third-party
  attestation story.
- **Cold-start corpus** — most families don't have well-organized digital
  text from grandma. Need killer importers (Line/WeChat backup,
  letter-photo OCR) and probably a guided interview flow to *generate*
  corpus from the elder while they're still around.

## 8. Chinese support audit

| Layer | Status |
|---|---|
| Base LLM (Qwen2.5) | ✅ Native, CMMLU-leading |
| TTS (Qwen3-TTS) | ✅ Native, already proven on 小S |
| ASR (Qwen3-ASR / faster-whisper) | ✅ Both Traditional + Simplified |
| Embedding (BGE-M3 / Qwen3-Embed) | ✅ Chinese-native, hybrid retrieval |
| Tokenizer | ✅ Qwen tokenizer handles Chinese natively |
| OCR (Tesseract chi_tra) | ✅ Decent for printed; commercial for handwritten |
| Diarization (pyannote 3.4) | ✅ Language-agnostic |
| Anthropic contextual chunking | ✅ LLM-agnostic, works in Chinese |

**Net: full pipeline is Chinese-native at every step.** This is rare and
defensible.

## 9. Reference papers / projects

- **Open Character Training** (arXiv 2511.01689, Nov 2025) — Constitutional
  AI for persona; introspective synthetic data; preserves base capabilities.
  Direct inspiration for Phase 3 synthetic expansion.
- **OpenCharacter** (arXiv 2501.15427, Jan 2025) — synthetic persona SFT
  recipe, LLaMA-3-8B base. Different paper, complementary.
- **Anthropic Contextual Retrieval** — chunk-prefix-with-context. Phase 2.
- **BGE-M3** (HuggingFace) — hybrid embedding. Phase 2.
- **Qwen3-Embedding** — alternative embedding. Phase 2.
- **OPLoRA** (arXiv 2510.13003) — orthogonal LoRA, preserves base.
  Phase 3.
- **Persona Drift** (arXiv 2402.10962) — measure + mitigate drift.
  Phase 1 (anchor reinjection).
- **SillyTavern character-card format** — adopt for persona JSON schema.
  Phase 1.

## 10. Status as of 2026-05-16

- Phase 0 spec written (this doc); implementation not started.
- 小S TTS LoRA `v2_20260514_152118_456516` is the active voice model.
- Stub `rag_retrieval_seconds` metric exists in `ws_asr.py:683` but no
  retriever wired.
- `data/personas/personas.json` exists with minimal shape; needs to grow
  into the layout in §3 Phase 0.
- Awaiting decisions on privacy ratchet, listener taxonomy.

