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
| LanceDB | Vector store | FAISS / Mongo Atlas / Qdrant | FAISS lacks durability + filtering; Mongo Atlas needs cloud or mongot-licensing footgun; Qdrant adds a daemon (we'll graduate to it if multi-client); LanceDB is embedded, durable, Apache 2.0 |
| Anthropic chunking | Chunk strategy | Sliding window | ~49% fewer top-20 misses per Anthropic's published benchmark |

## 3. Roadmap

### Phase 0-pre — Dev-UI refactor to Jinja2 + static JS (one-time cost)

Status as of 2026-05-16: `standalone_ui.py` / `recordings_ui.py` /
`training_ui.py` carry 1000-2000-line Python triple-quoted strings of
HTML+JS+CSS. No IDE highlighting on the inner content, escape footguns
(SyntaxWarnings on `\s` and `\[` show up on every test run), no
find-in-file across the boundary.

Decision (locked 2026-05-16): migrate to Jinja2 templates + static JS
files. Each `*_ui.py` shrinks to a ~20-line route returning
`TemplateResponse(...)`. HTML/JS/CSS get real IDE support.

Slices, each its own commit:
- Pilot: `standalone_ui.py` (smallest, most-edited)
- Then: `recordings_ui.py`
- Then: `training_ui.py`

Out of scope: build step (no TypeScript/Vite), framework migration (no
React/Svelte/HTMX). The dev UI's interactivity (AudioWorklet, WS binary
streaming, polling) maps poorly to server-rendered partials.

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
| chat exports | parser per platform: WhatsApp .txt, Line .txt (incl. ZIP w/ media), WeChat CSV+HTML (no native export — third-party tools required, documented in UI) |

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
  - BGE-M3 hybrid (dense + sparse via the model's native multi-vector
    output) for Traditional Chinese-native retrieval
  - **LanceDB as the vector store** (locked 2026-05-16) — embedded
    (no separate process), Parquet-backed durability, HNSW + IVF,
    Apache 2.0. Replaces the earlier FAISS-flat plan because FAISS has
    no durable storage and no metadata filtering. LanceDB hits both
    while staying single-binary. Qdrant is the upgrade path if we
    later need a multi-client query server.
  - Anthropic contextual prefix per chunk (~49% fewer top-20 misses
    per Anthropic's published benchmark)
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

### Phase 4 — Legacy product (family UI + Memory Sessions)

The shipped product has **two distinct UIs** — the dev UI we have today
keeps growing; the family-facing UI is a separate surface designed for
non-technical use. Built via **Claude Design** (claude.ai/design,
Anthropic Labs, included in Pro/Max/Team plans):

1. User opens claude.ai/design, points it at the voice-ai-pipeline repo
2. Claude Design reads the codebase + existing UI files, builds a design
   system that auto-applies colors/typography/components for consistency
3. Iterative design on the family UI (mobile-first, conversation-focused)
4. Hand-off via Claude Design's "send to Claude Code" → Claude Code
   session integrates the export as `app/api/family_ui.py`, wires it to
   existing WebSocket endpoints

The family UI has two modes per user role:

**Mode A — Elder self-recording / self-uploading (driven by Memory Sessions):**

This is the killer feature. Instead of asking the elder to dump random
files, the appliance actively prompts them against detected **manifest
coverage gaps**. "We have 23k chars about Alice but only 1.8k about Bob.
Tell me a story about Bob — anything that comes to mind." Mic on,
transcript captured + tagged listener_tag=bob, status=ingested.

Schema work:
- Extend `CorpusManifest` with `by_listener_tag: dict[str, int]` and
  later `by_topic: dict[str, int]` once a topic extractor is built.
- New `GuidedPrompt` model: `{prompt_id, template, target_listener,
  target_topic, language, sensitivity_tags}`.
- New `MemorySession` storing per-prompt response audio + transcript.

Prompt design rules (locked 2026-05-16 discussion):
- Story-shaped ("Tell me about a time when..."), not introspective
  ("How did you feel when..."). Latter pattern feels intrusive in
  Taiwanese / HK family contexts.
- Open-ended; never put words in the elder's mouth.
- Weekly digest cadence (not daily nag).
- Voice-with-auto-ASR is the default; typing is fallback.
- Family members can write custom prompts ("Mom, tell me about my 8th
  birthday") that are queued for the elder.

**Mode B — Family member conversation interface:**

Mobile-friendly chat with the elder's voice/persona. Shows manifest
status as a passive sidebar ("Alice coverage 87%, Bob 32%") so family
can encourage Mom to do a Memory Session on the weaker areas.

**Other Phase 4 work:**

- Auto-importers: WhatsApp .txt / Line .txt / WeChat CSV+HTML parsers
  surfaced through the corpus upload flow.
- Photo→letter OCR for handwritten correspondence (Tesseract chi_tra
  baseline; ABBYY-class commercial OCR fallback).
- Listener memory: per-family-member conversation persistence so the
  appliance can callback ("remember when little Mei was 5?").
- Optional: birthday/holiday "speak-in-their-voice" reminder bot.

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
- **LanceDB** (lancedb.com, Apache 2.0) — embedded vector store, Parquet-
  backed durability, HNSW + IVF. Phase 2 vector backend.
- **Claude Design** (anthropic.com/news/claude-design-anthropic-labs,
  research preview Nov 2025) — codebase-aware UI generation, exports to
  standalone HTML or hands off to Claude Code. Phase 4 family UI.
- **HereAfter AI / StoryFile** — prior art for guided-interview
  ingestion. Their failure mode (interview is the *only* mode) informs
  our "guided prompts as supplement to free-form upload" design.

## 10. Status as of 2026-05-16

- **Phase 0 slice 1 — shipped** (commit `a0e7b8e`). Per-persona corpus
  storage + `/api/corpus/*` endpoints + 14 contract tests; 295/1/0 suite.
- **Phase 0-pre — in progress**: dev-UI Jinja2 migration. Pilot on
  `standalone_ui.py` first.
- **Phase 0 slice 2 — next**: ingestion engine (PDF/EPUB/.txt/.md +
  audio→ASR→speaker-filter + chat exports). On hold until dev-UI
  migration lands so we don't compound complexity.
- 小S TTS LoRA `v2_20260514_152118_456516` is the active voice model.
- Stub `rag_retrieval_seconds` metric exists in `ws_asr.py:683` but no
  retriever wired.
- `data/personas/personas.json` exists with minimal shape; coexists
  with the new `data/personas/<persona_id>/corpus/` layout.
- Awaiting decisions on privacy ratchet (open #1), listener taxonomy
  (open #4).

### Decisions made in 2026-05-16 conversations (delta from §4)

| Topic | Decision |
|---|---|
| Vector store | LanceDB (was FAISS-flat) |
| Chat exports | WhatsApp .txt + Line .txt + WeChat CSV/HTML in slice 2 |
| Family UI tool | Claude Design (claude.ai/design) with Claude Code hand-off |
| Dev UI ergonomics | Migrate to Jinja2 + static JS (one-time refactor) |
| Memory Sessions | First-class Phase 4 feature, not nice-to-have; manifest gap-driven prompts; story-shaped not introspective |
| Family UI scope | Two modes: Mode A elder self-recording driven by Memory Sessions; Mode B family-member conversation |

---

## 11. GPT-5 review action items (2026-06-03)

External review by GPT-5 (full transcript at `docs/REVIEW_GPT5_2026-06-03.md`)
surfaced five high-confidence corrections to the M6/M7/M8/M9 plan. Captured
here so the RFC stays the source of truth.

### 11.1 — License audit BEFORE M7/M9 starts (DONE 2026-06-03)

Audit complete: `docs/THIRD_PARTY_LICENSE_AUDIT.md`. Results: 14 of 20
items clean (Apache-2.0 / MIT), 4 MAYBE need second-pass, **2 hard
blockers** confirmed.

#### Hard blocker #1 — PersonaHub (M9 SFT data path)

**Status:** Confirmed `cc-by-nc-sa-4.0` on
https://huggingface.co/datasets/proj-persona/PersonaHub. Dataset card
explicitly: "intended for research purposes only." **Cannot be used to
train any LoRA / model that ships in the commercial product.**

**Fallback plan — M9 SFT data generation (in priority order):**

1. **Primary: synthesize our own persona seed corpus.** Use Qwen 3 8B
   (Apache-2.0, confirmed item #11) as the generator, conditioned on
   demographic templates we author in-house. Each persona seed = a
   structured template (age, occupation, region, family role, speaking
   register, sample memories from M7-ingested corpus) → Qwen 3 8B
   produces ~50-200 turns of in-character dialogue per seed. Then apply
   the OpenCharacter recipe (Constitutional AI + introspective dialogue
   → SFT → DPO; method only, no Salesforce data) on top.

2. **Do not use the Salesforce `xywang1/OpenCharacter` 326K dataset
   directly** — it is derived from PersonaHub and therefore inherits
   the NC restriction (verify the HF card from a logged-in session
   before assuming otherwise; web-fetch returned 401 during audit).

3. **Treat PersonaHub itself as research-time inspiration only** —
   never ship weights derived from it. Acceptable to read the dataset
   to design our own template schema; not acceptable to use rows as
   training pairs.

4. **Data-provenance hygiene:** every training run logs
   `(seed_template_hash, generator_model_id, generator_model_license)`
   into the training metadata so we can prove all SFT pairs trace
   back to commercially-cleared inputs. Add to M9 training-job
   metadata schema as a hard requirement.

#### Hard blocker #2 — LLaMA-Omni 2 model weights (M12 candidate)

**Status:** Code Apache-2.0, **weights research-only**:
"Our model is intended for academic research purposes only and may NOT
be used for commercial purposes." Commercial license via
`fengyang@ict.ac.cn`.

**Fallback plan — M12 hybrid pipeline:**

- LLaMA-Omni 2 is structurally just a wrapper around CosyVoice 2
  (Apache-2.0, item #4) on the TTS side and Qwen 2.5 on the LLM side.
  **Build the same hybrid directly from those Apache-2.0 components
  without LLaMA-Omni 2 weights** — same architecture, same quality
  ceiling, clean license.
- Drop LLaMA-Omni 2 from the M12 / M13 commercial candidate list.
  Keep as a research-time eval if useful, but do not ship.

#### MAYBE items requiring second-pass before milestone gates

| Item | Action | Gate it blocks |
|---|---|---|
| Qwen3-TTS 12Hz 0.6B VoiceDesign | Open HF card, confirm `apache-2.0` | M11 ship |
| IndexTTS-2 | Read repo `LICENSE` + `LICENSE_ZH.txt` verbatim; assume restricted until written confirmation from indexspeech@bilibili.com | M11 ship |
| Kimi-Audio-7B weights | Logged-in HF check of `moonshotai/Kimi-Audio-7B` + `-Instruct` cards | M12 ship |
| emotion2vec weights | Fetch ModelScope `iic/emotion2vec_plus_large` page; check for non-commercial clause | M8.5 kickoff |
| MinerU 2.5 custom license | Read `LICENSE.md` from repo verbatim | M7 ship |
| Salesforce `xywang1/OpenCharacter` 326K | Logged-in HF check; almost certainly NC by inheritance from PersonaHub | M9 SFT data planning |

#### Ongoing compliance hygiene

- Snapshot every shipped dependency's verbatim license file into
  `data/compliance/LICENSE_<item>_<date>.txt` at the moment of model
  swap, so a future audit can prove what was in the box at any
  point in time.
- Re-run the audit when adding any new model / dataset to the stack.
  This audit is point-in-time (2026-06-03); upstream licenses CAN
  change.

### 11.2 — Mic capture: ScriptProcessorNode → AudioWorkletProcessor (DEFERRED 2026-06-03)

Current `app/static/js/standalone.js` captures via `ScriptProcessorNode`
+ `onaudioprocess`. MDN marks this API deprecated since 2020. Safari/iOS
and WeChat in-app browser will break it at any release. The intermittent
"low RMS on phone" symptoms from 2026-05-20 onward are partly attributable
to this deprecated path interacting badly with mobile AGC.

**Status: DEFERRED (2026-06-03 user direction).** Current path uses
upload-mode for the demo workflow, not live browser capture in the
critical path. Risk of break-during-migration outweighs benefit while
the API is still working in Chrome desktop. Revisit when (a) we see a
real Safari/iOS user case, (b) we add live browser-capture personas
back into the primary product flow, or (c) MDN/Chrome deprecates the
API to "removed".

Tracked here; not actioned now.

### 11.3 — M-Consent gates M7 (sequencing change)

Original plan ran M-Consent parallel to M7. GPT-5 review (correctly)
flags that this lets users upload a corpus before a consent record
exists — which under NO FAKES Act / EU AI Act / CA AB 1836 creates
retroactive unlearning obligations we can't satisfy with monolithic
LoRAs.

**Action.** Re-sequence: **M-Consent UI ships in front of M7 corpus
ingest UI.** Server-side check: `POST /api/corpus/ingest` fails if
the persona has no current consent record for that source_kind.
Engineering can proceed in parallel; only the user-facing flow is
gated.

### 11.4 — M8 incremental delivery (revised 2026-06-03 per user direction)

GPT-5 originally recommended an M8a thin slice before full M8.
**User pushback was right:** the thin slice is not throwaway code —
BGE-M3 + LanceDB schema + retrieve API + prompt injection
infrastructure is required regardless of single-index vs ID-RAG.
M8 full = M8a + additive layers, not parallel work.

**Revised plan:** ship M8 as **three incremental commits inside one
milestone**, each independently reviewable + reversible + user-visible.
Same total effort (2-3 weeks), same end state, lower review risk per
commit.

| Commit | Days | User-visible behavior | Locks |
|---|---|---|---|
| M8.1 — embeddings + retrieve + injection | 3-5 | AI cites corpus ("根據你 2024-03 的 email...") | LanceDB schema, retrieval-injection shape in ws_asr.py, latency budget |
| M8.2 — identity graph + HippoRAG multi-hop | ~7 | AI connects facts (mom + medication + appointment) | Graph schema, multi-hop retriever |
| M8.3 — conversation memory + consolidation (A-MEM / Mem0 layer) | 3-5 | Cross-session memory ("上次你問了...") | Decay / promotion / consolidation rules |

**Why this beats the original "M8a thin-slice"** framing:
- Each commit ships user-visible value (not just plumbing).
- No throwaway code — M8.2 and M8.3 are additive on M8.1.
- Reviewable surface area per commit stays small.
- User can A/B turn off later stages if they regress quality.

Update ROADMAP §10 accordingly — replace "M8a → M8" with the three
M8.x commits.

#### MongoDB AI memory taxonomy — what we borrow

Inspiration from MongoDB's *Bringing Attention to Memory in AI Agents
and Agentic Systems* (https://www.mongodb.com/resources/basics/artificial-intelligence/agent-memory),
researched 2026-06-03. MongoDB's writeup is conceptual (no schema), but
the **memory taxonomy is worth borrowing as a naming + organization
convention** even though our stack stays LanceDB + BGE-M3 (not Atlas
Vector Search + Voyage).

MongoDB distinguishes:
- **Short-term:** Working Memory (active scratchpad), Semantic Cache
  (query/response reuse), Shared Memory (cross-agent workspace)
- **Long-term:** Episodic (specific events), Semantic (facts/concepts),
  Procedural (workflows/skills), Associative (links/connections)

**Three M8 deltas this drives:**

1. **Split into separate LanceDB tables by memory type, not one blob.**
   - `memory_episodic` — per-conversation-turn records (timestamp,
     speaker, raw transcript). Source of the "AI remembers what was
     said" experience.
   - `memory_semantic` — extracted facts about the persona ("mom's
     birthday is March 12", "father loved jazz"). M7 ingest produces
     these.
   - `memory_procedural` — learned behavioral patterns ("mom prefers
     Hakka greetings on weekends", "father always starts stories with
     a metaphor"). Emerges from M8.3 consolidation.
   - Per-type retention + decay policies become explicit, not implicit
     in one global "memory" column.

2. **Promote "Associative Memory" to a first-class layer.** This maps
   onto HippoRAG 2's neighborhood graph (planned for M8.2) but framed
   as a memory *type*, not just a retrieval trick. Better mental model
   for explaining to family users why the box "remembers connections."

3. **Semantic Cache as tier-0 before LLM call.** Cheap latency win that
   *also* reduces cloud-LLM exposure during the Phase-1 OpenAI window
   (privacy-adjacent benefit). Cache key = (persona_id, listener_id,
   normalized ASR text); cache hit returns prior LLM reply directly,
   skip OpenAI round-trip. Add to M8.1 if cheap; defer to M8.3 if not.

**What we do NOT borrow:**
- Atlas Vector Search (cloud-managed, violates on-box rule). LanceDB
  stays.
- Voyage embeddings (non-Chinese-native). BGE-M3 stays per
  [[chinese-support-stack]] memory.
- "Single document, all memory inside" pattern — LanceDB scan
  performance would degrade and per-type decay policies become hard.

**Schema implication for M8.1:** the embedding/retrieve API should take
a `memory_type` parameter from day one even if only `episodic` is wired
up. Adding the parameter later forces a callsite migration; baking it
in now is free.

### 11.5 — Unlearning design spike for M-Consent (1 day)

Current LoRA training is monolithic — revocation = full retrain.
GPT-5 cites SISA training (Bourtoule et al., USENIX Security 2019) as
the pattern: shard training data, train per-shard adapters, drop the
shard whose data was revoked + retrain only that shard.

**Action.** 1-day design spike attached to M-Consent kickoff:
- Decide shard granularity (per-recording-session? per-week?)
- Estimate per-shard retrain cost on A10G
- Publish revocation SLA (e.g., "revoked segment removed from
  synthesized output within 7 days")
- Surface honestly in UI per `feedback_fail_loud`

### 11.6 — Broaden M11 BaseTTSEngine to first-class VoxCPM 2 + CosyVoice 2

§3 Phase 3 listed Qwen3-TTS only with IndexTTS-2 as fallback. GPT-5
review notes that **VoxCPM 2 (OpenBMB/Tsinghua, late-2025/2026) and
CosyVoice 2/3 are advancing fast**. If either ships a robust per-speaker
LoRA recipe matching our cloning depth, the Qwen3-TTS moat is at risk
within 6-12 months.

**Action.** When M11 BaseTTSEngine ships, include `VoxCPM2Engine` and
`CosyVoice2Engine` as warm-backup implementations (not just stubs). A/B
against current Qwen3-TTS-LoRA path on the close-relative-recognition
human eval (M8 §4.0 §5). Promote whichever wins.

### 11.7 — Track / re-examine items (no immediate action, monitor)

| Item | Watch trigger |
|---|---|
| Qwen3.5-Omni Talker SFT recipe | HF release page, paper updates to arXiv:2604.15804 |
| Step-Audio 2 mini fine-tune recipe | GitHub stepfun-ai/Step-Audio2 issue #67 close |
| Big-platform "legacy voice" feature (Apple/Google) | Apple Personal Voice + on-device LLM news |
| EU AI Act audible-disclosure mandates | Track delegated acts, NY/TX state bills |
| Audio watermarking robustness | Meta AudioSeal (2024) field performance after MP3 64kbps + room re-record |
| Browser audio API breakage | Safari/Chrome release notes for ScriptProcessorNode removal |

### 11.9 — M12a pulled to top of queue (Gemini review 2026-06-03)

Gemini 2.5 Pro review (transcript at
`docs/REVIEW_GEMINI25PRO_2026-06-03.md`) flagged this more aggressively
than GPT-5: M12a (Qwen3.5-Omni-Light + Step-Audio 2 mini cloning eval
spike) is a 1-day investigation that has the potential to **invalidate
or reshape M11, M12, and M13**. Pushing it to August means weeks of
TTS abstraction and hybrid-pipeline work might be superseded by a
single discovery.

**Action.** Move M12a to **first task after D-Retro** (target: this
week). One-day spike, measurable outcomes:
- Run zero-shot cloning on each candidate with a 30-second Mark sample.
- Score against the same human-eval rubric M9 will use (close-relative
  recognition A/B).
- Decision tree: if cloning depth matches Qwen3-TTS-LoRA → fast-track
  M13 evaluation, defer M8.5/M11 polish. If clearly below → confirm
  current TTS-LoRA path, no roadmap change.

### 11.10 — M9 split: ship local LLM faster (Gemini review 2026-06-03)

Gemini observed: M8.5 is a 3-4 week TTS research milestone that
**blocks M9** under the current plan. That delays the fundamental
privacy promise (local LLM, no OpenAI in the request path) by 2-3
months.

**Action.** Split M9 into two:
- **M9.1 — Local Qwen 3 8B + simple persona LoRA** (~2 weeks). Ships
  the local-LLM value. Persona LoRA trained without M8.5's locked
  emotion vocabulary — uses our current emotion tag set as-is. May
  need to re-train when M8.5 ships, but the *deployment infrastructure*
  (vLLM, model serving, latency tuning, OpenAI swap-out) is the long
  pole and ships independently. **Privacy moat closes here, not at
  M9.2.**
- **M9.2 — Rich persona LoRA with locked vocabulary** (~3-4 weeks,
  after M8.5). Re-train with emotion2vec-anchored vocabulary. Hot-swap
  the adapter on the live local LLM. No infra change.

**Why this matters:** the local-LLM moat is what makes the "100%
on-box" claim true. Shipping it 2-3 months earlier de-risks the demo
narrative and the B2B sales motion.

### 11.11 — M7 split: minimal ingest unblocks M8 (Gemini review 2026-06-03)

Gemini observed: M7 is 2-3 weeks because it supports 7 input formats
(txt, md, pdf, epub, docx, photos, chat exports). M8 only needs *some*
content to be useful.

**Action.** Split M7:
- **M7.1 — Minimal ingest** (~3-5 days): `.txt`, `.md`, `.pdf` only.
  Unblocks M8.
- **M7.2 — Full ingest** (~2 weeks): epub, docx, photos (PaddleOCR-VL),
  chat exports. Ships after M8 is wired and proven.

**Why this matters:** M8 is the value-delivery milestone (cross-session
memory). Waiting for the full M7 just to start M8 wastes ~1.5 weeks
when 80% of corpus value (text, PDFs of letters, emails saved as PDF)
already covers the typical first family's input.

### 11.12 — Strategic reframe: corpus is the moat, not just the clone

Gemini's most-quoted finding: **"The Real Moat is the Corpus, Not the
Clone."** A competitor could cede voice-cloning depth, use a "good
enough" zero-shot model, and focus entirely on making it easy for a
family to build a complete authentic memory graph. If they get the
M7/M8/M9 pipeline 10x better than us, families may not care that the
voice is slightly off.

**Implication:** voice cloning depth is a NECESSARY condition (otherwise
the product fails the recognition test) but it is NOT the SUFFICIENT
moat. The sufficient moat is the **per-family curated corpus + memory
graph + persona model** — the asset that takes years of family
participation to build and cannot be copied between vendors.

**Action.**
- Update PROJECT_BRIEF.md §1 to frame the moat as
  "voice cloning depth × curated family corpus + memory graph", not
  voice alone.
- Treat M7 + M8 + M9 as **co-equal moat investments** with M10/M11,
  not as plumbing.
- The "data curator" role / onboarding UX (the part most likely to
  bottleneck adoption) gets a dedicated milestone before any commercial
  launch. Tentative name: **M-Onboarding** — guided multi-session
  data-collection app, family-member-friendly UX, not just
  `POST /api/corpus/upload`.

### 11.13 — M10 is the research-publication opportunity

User observation (msg 1971): most of the roadmap is implementing
existing papers. M10 (multi-listener voice routing) might be where the
project actually has **novel research contribution potential**.

**Why M10 is unusually publishable:**

- The wider TTS literature treats voice cloning as **per-speaker** and
  emotion as the secondary axis. **Per-LISTENER adaptation from the
  same speaker** — i.e., the same person adapting voice when speaking
  to a child vs a colleague vs an elder — is rarer in published work
  (Interspeech / ICASSP). Closest related: persona-aware TTS,
  code-switching, dialog-adaptive prosody. None directly study
  speaker × listener × LoRA composition.
- EverHome has structurally rare training data: the recording pipeline
  naturally captures **the same speaker speaking to multiple known
  relationships** (per-listener prompts in `/ui/recordings`). Most
  academic TTS datasets do not have this axis labeled.
- The (speaker, listener) → LoRA composition is novel as engineering
  — could be presented as a method paper.

**Candidate paper framing:**
- Title (working): "Listener-Adaptive Voice Cloning via Per-Relationship
  LoRA Composition on Family-Scale Audio"
- Venue: Interspeech 2027 (deadline ~March 2027) or ICASSP 2027
  (deadline ~Sept 2026). ICASSP if M10 ships by August 2026; Interspeech
  if later.
- Contributions: (1) the per-listener LoRA composition method;
  (2) a release-the-method-not-the-data approach with a synthetic
  evaluation set (real family data stays on the box per privacy moat);
  (3) the close-relative-recognition human-eval rubric itself (no
  published benchmark exists per `RESEARCH_SFT_S2S.md`).

**Implication for M10 design.** Build M10 with the publication in
mind from day one:
- Document the LoRA-composition algorithm precisely (not just "we
  train per-listener").
- Run controlled ablations: single-LoRA-per-listener vs
  composition-at-inference vs interpolation. Pick whichever produces
  the publishable result.
- Capture training metadata (per-listener data hours, A/B win rates)
  that a reviewer would ask for.
- Reserve a small held-out family-recognition test set with consenting
  participants for the human eval section.

**Risk:** if M10 is also the moat, publishing the method tells
competitors how to do it. **Mitigation:** the data + per-family corpus
is the durable moat (per §11.12). The method becoming public IS the
academic contribution; the moat doesn't depend on method obscurity.

### 11.8 — D-Retro bucket (2026-06-02 demo learnings)

Captured fixes from demo prep + day-of that need to be locked in with
tests and documented in roadmap §12 retrospective:

1. ASR hallucination filter (Qwen3-ASR emits "The first was the first
   to be built.", "《大明宫词》", short stock phrases on silence).
   Filter at `app/services/asr/engine.py` recognize() — known-text
   set + peak<0.12 silence guard.
2. Listen-only mode: server skips LLM but client was stuck in THINKING
   because no llm_start/llm_done arrived. Server now always sends
   asr_result; client transitions to READY (or re-arms via
   auto-continue) on empty/listen-only.
3. UI Language dropdown (auto/中文/English) — server injects LANGUAGE
   directive into system prompt. Persona JSON no longer hard-codes
   English-only.
4. Cache-busting `?v={mtime}` on `/static/js/*.js` — Cloudflare edge
   was caching JS for 4 hours, browsers kept old bundle.
5. Persona dropdown defaults to "test" (EverHome Demo / Mark) so
   demo path doesn't need clicking.

**Action.** Add contract tests for each (silence guard, hallucination
filter, empty-asr handling, language directive). Capture in roadmap
§12 retrospective. 1-2 days work.

