# EverHome — Research Review for ROADMAP_2026Q3

**Status:** Reference scan (not contractual)
**Date drafted:** 2026-05-28
**Companion to:** `docs/ROADMAP_2026Q3.md`
**Scope:** 2025-mid through 2026-Q2 publications, plus foundational pre-2024 where load-bearing.

> Purpose: surface work the roadmap doesn't yet cite, so 6-month
> architecture decisions (M7-M13) are made against the current frontier
> rather than the Jan-2026 frontier.

---

## 0. Major clarification — two distinct OpenCharacter papers

The roadmap §6 cites **arXiv 2511.01689** ("Open Character Training",
Nov 2025, Constitutional AI). The earlier number **arXiv 2501.15427**
(Jan 2025) is a **separate** paper, also titled "OpenCharacter" — from
Salesforce AI Research. The two are easy to confuse and complement each
other:

| Paper | Authors / lab | Method | Base model | Code/data |
|---|---|---|---|---|
| **2501.15427** OpenCharacter (Jan 2025) | Salesforce AI | Synthesize 20K personas from Persona Hub → response-rewriting + response-generation → SFT | LLaMA-3 8B Instruct | HF dataset `xywang1/OpenCharacter` (326K rows, 1.19 GB) |
| **2511.01689** Open Character Training (Nov 2025) | Independent / anonymous (4open.science) | Constitutional AI + synthetic introspective dialogue → SFT → DPO | Llama 3.1 8B / Qwen 2.5 7B / Gemma 3 4B | anonymous.4open.science repo |

**Implication for M9.** The 2501 paper gives us a **ready-made 326K-row
character SFT dataset** as a general-instruction mix that can sit
alongside our 30-50% instruction-preservation slice — it's worth a
read even though our planned recipe is the 2511 one. Different problem
(scaling to many personas vs. depth on one), compatible techniques.

---

## 1. LLM Persona / Character Training

**Our current plan (M9).** Constitutional AI + synthetic introspective
dialogue (OpenCharacter 2511.01689) → LoRA SFT on Qwen 2.5 7B Instruct →
DPO. OPLoRA orthogonality to protect base capabilities.

**Gap / what we might be missing.** No persona-drift measurement plan,
no benchmark for the trained character's stability under adversarial
prompts (the "jailbreak grandma" failure mode is exactly our threat
model). Also missing: a survey of persona-design taxonomies that could
shape the Constitution-authoring UX.

**Top candidates:**

1. **OpenCharacter (Salesforce)** (Jan 2025, paper + dataset) — [arXiv 2501.15427](https://arxiv.org/abs/2501.15427)
   - Synthesizes 326K character-aligned instruction-response pairs from Persona Hub via response rewriting and response generation; SFT on LLaMA-3 8B reaches GPT-4o-comparable role-play quality.
   - Relevance: the dataset itself is a high-quality **persona-instruction mix** we can blend into M9 training to combat catastrophic forgetting while still steering persona — distinct from the Constitutional AI step, complementary to 2511.
   - Recommendation: ✓ integrate (use the dataset as part of the 30-50% mix)

2. **Open Character Training (Constitutional AI)** (Nov 2025) — [arXiv 2511.01689](https://arxiv.org/abs/2511.01689) — already in roadmap §6. No change.

3. **PERSONA — Dynamic and Compositional Personality Modeling** (ICLR 2026) — [arXiv 2602.15669](https://arxiv.org/pdf/2602.15669)
   - Four-component framework: PERSONA-BASE extracts orthogonal OCEAN trait vectors; PERSONA-ALGEBRA composes traits via vector arithmetic; PERSONA-FLOW does inference-time steering; PERSONA-EVOLVE benchmarks.
   - Relevance: inference-time trait steering could be a **cheaper** alternative for low-data personas (most elder corpora) before we commit to a LoRA. Worth a 1-week ablation in M9 against the LoRA baseline.
   - Recommendation: ⚠ evaluate

4. **Nautilus Compass — Black-box Persona Drift Detection** (2026) — [arXiv 2605.09863](https://arxiv.org/html/2605.09863)
   - Production-oriented persona-drift detector that works without retraining; identifies linear activation directions corresponding to sycophancy, hallucination, etc.
   - Relevance: M9 needs a **drift telemetry hook** in `ws_asr.py` to catch persona breakdown over long sessions (esp. for elders, where one drift = lost trust). Currently roadmap §8 lists this as Open Question #4 without owner.
   - Recommendation: ✓ integrate as M9 sub-deliverable

5. **Persona-Consistent Dialogue Generation via Pseudo Preference Tuning** (COLING 2025) — [ACL Anthology](https://aclanthology.org/2025.coling-main.369/)
   - DPO on pseudo-preference pairs for persona consistency; beats SFT and standard RL.
   - Relevance: our M9 already plans DPO. This paper is the **pseudo-pair construction recipe** we'd otherwise have to invent; cite it directly.
   - Recommendation: ✓ integrate (recipe reference for the DPO step)

6. **Systematizing LLM Persona Design — Four-Quadrant Taxonomy** (Nov 2025) — [arXiv 2511.02979](https://arxiv.org/html/2511.02979v1)
   - Taxonomy for AI-companion persona design across four axes (data-bound vs free, narrow vs broad, etc.).
   - Relevance: informs M9's Constitution-authoring UX. Not load-bearing for code, but useful to cite when defending the template-with-fill-in-blanks default.
   - Recommendation: ⚠ evaluate (UX reference only)

**Not found (searched, no strong hits):**
- No 2025-2026 work on persona LoRA specifically for **Traditional Chinese
  elder voice / Taiwanese particle preservation**. Our corpus is uniquely
  small here. Note for product positioning.

---

## 2. Long-term Memory for Personal AI

**Our current plan (M8).** Dual-index RAG (style + stance), BGE-M3
embeddings, LanceDB store, Anthropic-style contextual chunk prefix.

**Gap / what we might be missing.** Roadmap treats memory as static
indexed corpus + retrieval. Missing: **memory consolidation over time
across conversations** (the family member talks to grandma weekly — does
the system get smarter about her, or just retrieve the same chunks?),
**episodic vs semantic split** (currently both indices are semantic),
and explicit benchmark choice (LoCoMo).

**Top candidates:**

1. **HippoRAG 2 — From RAG to Memory** (Feb 2025) — [arXiv 2502.14802](https://arxiv.org/html/2502.14802v1)
   - Neurobiologically inspired: knowledge graph + Personalized PageRank, used for retrieval routing not corpus expansion. Significantly higher multi-hop F1 + Recall@5 vs pure-embedding RAG, lower offline indexing cost than GraphRAG / RAPTOR / LightRAG.
   - Relevance: our **stance index** is exactly a multi-hop retrieval problem ("what did dad believe about X, and how did that evolve?"). HippoRAG 2 should be the default architecture for the stance index, not vanilla BGE-M3+top-k.
   - Recommendation: ✓ integrate (replace stance-index plan)

2. **A-MEM — Agentic Memory for LLM Agents** (Feb 2025) — [arXiv 2502.12110](https://arxiv.org/pdf/2502.12110)
   - LLM-native memory admission via Zettelkasten-inspired linking; the LLM decides what to store and how to connect new memories to existing ones.
   - Relevance: this is the **conversation-time memory consolidation** layer we don't currently have. M8 indexes the static corpus; A-MEM would handle "every chat with grandma adds to her memory of you."
   - Recommendation: ✓ integrate as M8.5 or M14 (small add, big UX win)

3. **Mem0 — Scalable Long-Term Memory** (ECAI 2025) — [arXiv 2504.19413](https://arxiv.org/pdf/2504.19413) — [repo: 48K stars](https://github.com/mem0ai/mem0)
   - Production-tested semantic memory layer with LoCoMo benchmark numbers (67% J-score, 200 ms p95, ~1.8K tokens/conv vs 26K full-context). $24M funded Oct 2025.
   - Relevance: candidate **off-the-shelf** for our conversation-memory layer if A-MEM is too DIY. Apache-licensed, drop-in. But: cloud-default; we'd need to self-host.
   - Recommendation: ⚠ evaluate (vs A-MEM, vs build-our-own)

4. **TiMem — Temporal-Hierarchical Memory Consolidation** (2026) — [arXiv 2601.02845](https://arxiv.org/pdf/2601.02845)
   - Five-level temporal memory tree consolidating from factual segments → persona profiles, with query-complexity-adaptive recall.
   - Relevance: directly addresses our stated goal of "the system learns the persona over time." More principled than A-MEM but heavier.
   - Recommendation: ⚠ evaluate

5. **Beyond Fact Retrieval — Generative Semantic Workspaces** (Nov 2025) — [arXiv 2511.07587](https://arxiv.org/html/2511.07587v1)
   - Argues GraphRAG/LightRAG/HippoRAG/RAPTOR all fail at **episodic** memory (temporally-anchored, single-event recall) — proposes a generative workspace.
   - Relevance: validates our roadmap's split decision (we do plan dual-index, just with weaker primitives). Worth reading before locking M8 architecture.
   - Recommendation: ⚠ evaluate (architectural read; may change M8 split)

6. **LoCoMo benchmark** (referenced in Mem0 paper)
   - Long-conversation memory benchmark — the de-facto standard.
   - Relevance: M8 acceptance criteria currently says "20 held-out questions on 小S corpus." We should also score on LoCoMo for credibility / publication readiness.
   - Recommendation: ✓ integrate as M8 secondary validation

**Not found:**
- No Chinese-native conversational-memory benchmark. LoCoMo is English.
  Acceptable since memory is mostly architecture-level, but worth flagging.

---

## 3. Voice Cloning Quality

**Our current plan.** Qwen3-TTS-12Hz-1.7B VoiceDesign + `custom_voice`
(speaker_embedding bake) + LoRA on `code_predictor` (5 layers). Real
`talker.model` LoRA deferred (`training_pipeline_deferred` item 1).

**Gap / what we might be missing.** Several **newer open Chinese voice
cloning models** matched or surpassed Qwen3-TTS on SEED-TTS-EVAL during
the 4 months we've been heads-down on the demo. Roadmap §M11 (TTS
abstraction) is correctly low-priority but the candidate list needs a
refresh.

**Top candidates:**

1. **VoxCPM (OpenBMB + Tsinghua, 0.5B / 2B)** (Sep 2025) — [arXiv 2509.24650](https://arxiv.org/abs/2509.24650) — [repo](https://github.com/OpenBMB/VoxCPM)
   - Tokenizer-free TTS, 1.8M hours bilingual ZH-EN corpus. SEED-TTS-EVAL: English WER 1.85%, Chinese CER 0.93% (SOTA among open). Zero-shot cloning from single reference. VoxCPM2 (2B) supports 30 languages + 48 kHz output.
   - Relevance: **leading candidate to replace Qwen3-TTS** when M11 abstraction lands. Tokenizer-free = no language-leak gotcha like ours (per `tts_inference_language_gotcha` memory). 48 kHz fixes our "telephone-grade source" problem (roadmap Known Limitation #1).
   - Recommendation: ✓ integrate (high priority — directly addresses two known roadmap gaps)

2. **IndexTTS-2 (bilibili)** (Feb 2025) — [arXiv 2502.05512](https://arxiv.org/abs/2502.05512)
   - XTTS-derivative, Chinese-character + pinyin hybrid modeling (controllable pronunciation of polyphonic chars). Emotional + duration control. Listed in 2026 industry "top 3 open voice cloning."
   - Relevance: the **pinyin-hybrid** trick is exactly what 小S Taiwanese particles need (蛤 / 啦 / 咧 are pronunciation-disambiguation cases). Worth evaluating just for this.
   - Recommendation: ⚠ evaluate

3. **CosyVoice 2 / Fun-CosyVoice 3.0 (Alibaba FunAudioLLM)** (Dec 2024 / Dec 2025)
   - 9 major languages + 18+ Chinese dialects, 150 ms latency bi-streaming, zero-shot cloning. RL-optimized v3 (Dec 2025).
   - Relevance: **most mature Chinese open TTS**. The 18-dialect support is unique — relevant for diaspora ICP (Hakka, Min, Cantonese). Apache 2.0.
   - Recommendation: ⚠ evaluate (v3 vs VoxCPM is the real bake-off)

4. **Qwen3-TTS (Jan 2026)** — [Qwen3-TTS repo](https://github.com/QwenLM/Qwen3-TTS) — [tech report 2601.15621](https://arxiv.org/html/2601.15621v1)
   - We're already on the predecessor. Released as a series; 3-second cloning, Beijing/Sichuan dialect support, cross-lingual cloning.
   - Relevance: **upgrade path** — verify our current 1.7B VoiceDesign is the latest line. If a newer Qwen3-TTS release exists post-Jan-2026, our M-Demo SFT may be on a deprecated base.
   - Recommendation: ✓ integrate (verify version is current; trivial)

5. **Step-Audio-EditX (StepFun)** — [repo](https://github.com/stepfun-ai/Step-Audio-EditX)
   - 3B LLM-based RL audio-edit model — edits emotion, speaking style, paralinguistics in **existing** audio + robust zero-shot TTS.
   - Relevance: not a primary TTS for us, but useful for the M7 ingestion pipeline — can we **edit** recorded corpus audio to expand training samples? Speculative.
   - Recommendation: ⚠ evaluate (only after M-Demo)

**Roadmap correction.** §1 calls Qwen2.5/3-Omni "no cloning". That was
true at roadmap drafting but **Qwen3.5-Omni** (per its tech report
[arXiv 2604.15804](https://arxiv.org/html/2604.15804v1)) is being
described upstream as having zero-shot voice cloning. Needs verification
before M12 design lock — could collapse M12 and M13 into one milestone.

---

## 4. End-to-end Speech-to-Speech (S2S)

**Our current plan.** Track for M13 (opportunistic). Aware of Chroma
1.0 (English), Qwen-Omni (no cloning), Moshi, GPT-4o Realtime.

**Gap / what we might be missing.** Two material 2025-2026 Chinese-native
S2S systems are missing from roadmap §M13 watch list. Also: Chroma may
be more transferable than we credit it.

**Top candidates:**

1. **Step-Audio 2 (StepFun)** (Jul 2025) — [arXiv 2507.16632](https://arxiv.org/pdf/2507.16632)
   - End-to-end multi-modal LLM, interleaved discrete text-audio tokens. **Best-in-class on Chinese S2S conversation** per their report — beats GPT-4o Audio, Kimi-Audio, Qwen-Omni. Chinese CER 3.08%, English WER 3.14% on ASR.
   - Relevance: **this is the M13 candidate we should be tracking, not Chroma**. Chinese-native by design, open. Cloning status unclear — needs verification.
   - Recommendation: ✓ integrate (add to M13 watch list as primary candidate)

2. **Kimi-Audio (Moonshot AI)** (Apr 2025) — [arXiv 2504.18425](https://arxiv.org/html/2504.18425v1) — [HF 7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)
   - Audio foundation model on Qwen 2.5 7B base, 13M hours of audio + text. Unified ASR + AQA + AAC + SER + end-to-end speech conversation.
   - Relevance: directly built on Qwen 2.5 7B — **our M9 persona LoRA might transfer** with adapter surgery. This is a more interesting M12 hybrid than vanilla Qwen Omni.
   - Recommendation: ✓ integrate (re-evaluate M12 base choice: Kimi-Audio vs Qwen 2.5 Omni 7B)

3. **LLaMA-Omni 2 (ICT-CAS)** (ACL 2025 main) — [paper](https://aclanthology.org/2025.acl-long.912.pdf)
   - 0.5B-32B speech LMs built on Qwen 2.5 Instruct, 583 ms real-time speech synthesis. Modular.
   - Relevance: same Qwen 2.5 base as our M9 LoRA — **strongest direct LoRA-transfer candidate**. Sub-1 s latency already achieved.
   - Recommendation: ✓ integrate (M12 alternative; ablate against Qwen Omni 7B)

4. **GLM-4-Voice (ZhipuAI)** (Dec 2024 / iterating) — [arXiv 2412.02612](https://arxiv.org/pdf/2412.02612)
   - Open-source Chinese-first end-to-end spoken chatbot from GLM family.
   - Relevance: Chinese-native peer. Not clearly ahead of Step-Audio 2 on benchmarks, but worth tracking. Cloning unclear.
   - Recommendation: ⚠ evaluate

5. **Chroma 1.0 (FlashLabs)** — already in roadmap. **Correction:** roadmap §1 says "English-only" — the tech report [arXiv 2601.11141](https://arxiv.org/abs/2601.11141) suggests it's English-trained but the architecture itself is language-agnostic. Worth confirming via repo whether a Chinese fine-tune is feasible.

6. **Stream-Omni (CAS, Jun 2025)** — single-paper, cross-modal real-time. Less mature than Step-Audio 2. Recommendation: ✗ not relevant for now.

**Major roadmap update needed.** §M13 watch list currently has Chroma
as the leading candidate. **Step-Audio 2, Kimi-Audio, and LLaMA-Omni 2
all surpass it for Chinese.** Re-rank the watch list.

---

## 5. Multi-modal Document Ingestion

**Our current plan (M7).** pdfplumber/PyMuPDF + Tesseract `chi_tra` for
OCR, ebooklib for EPUB, python-docx for DOCX. Per-file size limit 100 MB.

**Gap / what we might be missing.** The 2026 SOTA for PDF + handwriting
+ Chinese has moved to **VLM-based unified parsers** that handle layout,
OCR, tables, and chart parsing in one pass. Tesseract `chi_tra` is
**clearly outclassed in 2026** — recommend a fundamental swap before M7
implementation starts.

**Top candidates:**

1. **MinerU 2.5-Pro (OpenDataLab)** (2026) — [repo](https://github.com/opendatalab/MinerU) — [pypi mineru 2.0](https://pypi.org/project/mineru/2.0.0/)
   - VLM-based (1.2B), specialized in Chinese + scientific + financial docs. Handles layout, OCR, tables, charts, cross-page table merging, image-in-table OCR. Top tool in 2026 open PDF-to-markdown comparisons.
   - Relevance: **replaces pdfplumber + PyMuPDF + Tesseract chi_tra in one tool**. Designed for Chinese. This is the clear M7 winner.
   - Recommendation: ✓ integrate (replace the entire planned M7 PDF/OCR stack)

2. **PaddleOCR-VL 1.5 (Baidu)** (Jan 2026) — [HF model](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) — [tech report arXiv 2507.05595](https://arxiv.org/html/2507.05595v1)
   - 94.5% on OmniDocBench, surpasses top general LLMs and specialized parsers. Supports 5 text types incl. simplified + traditional Chinese, pinyin, handwriting, vertical text. 13% accuracy improvement over PaddleOCR 3.0.
   - Relevance: **direct replacement for Tesseract `chi_tra`**, especially for handwritten elder letters (M7 known weakness — `quality=low` flag is admission of defeat against handwritten).
   - Recommendation: ✓ integrate (primary OCR for letter photos)

3. **Docling (IBM Research, LF AI & Data Foundation)** — [Docling on chatforest](https://chatforest.com/reviews/ocr-document-intelligence-mcp-servers/)
   - DocLayNet + TableFormer; comprehensive multi-format parser with MCP server available.
   - Relevance: weaker on Chinese than MinerU but stronger on Western academic / business docs. Useful fallback if MinerU underperforms on EN.
   - Recommendation: ⚠ evaluate (secondary; mix-of-experts setup)

4. **Vision-Guided Chunking** (Jun 2025) — [arXiv 2506.16035](https://arxiv.org/pdf/2506.16035)
   - VLM-guided chunk boundaries that respect visual document structure.
   - Relevance: improves M7→M8 handoff. Chunks at the visual layout level not text-flow level — relevant for ebooks with chapter breaks and letters with paragraph structure.
   - Recommendation: ⚠ evaluate

5. **MarkItDown (Microsoft)** — lightweight PDF→MD; **no OCR, no advanced layout**. Not relevant for our use case.
   - Recommendation: ✗ not relevant

**Critical roadmap update.** M7 plans to ship Tesseract `chi_tra` and
"document fallback to commercial OCR (ABBYY-class)" for handwritten.
The right move in 2026 is **start with PaddleOCR-VL 1.5** — no
commercial OCR needed.

---

## 6. Privacy / On-device LLM

**Our current plan.** Qwen 2.5 7B Instruct AWQ-Q4 local (M9). DGX Spark
128 GB unified for production. Cloud OpenAI gpt-4o-mini as provisional.

**Gap / what we might be missing.** The 2026 edge-LLM landscape has
shifted to **multi-size model families with speculative decoding** —
some recent Chinese-capable options weren't on our radar. Also: Qwen 2.5
is now a generation behind.

**Top candidates:**

1. **Qwen 3 family (Qwen3-0.6B / 1.7B / 4B / 8B / 14B / 32B)** — [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) — [blog](https://qwenlm.github.io/blog/qwen3/)
   - Apache 2.0. Successor to Qwen 2.5 — recommended for vLLM / SGLang / llama.cpp / Ollama / MLX / KTransformers deployment.
   - Relevance: **M9 should target Qwen 3 8B**, not Qwen 2.5 7B. Same VRAM budget, better Chinese, better instruction-following. The persona LoRA recipe transfers.
   - Recommendation: ✓ integrate (M9 base model swap)

2. **THUDM GLM-4-9B-0414** — listed among top 2026 edge deployment models
   - Relevance: Chinese-native peer to Qwen 3 8B. Worth an A/B for persona-LoRA training quality.
   - Recommendation: ⚠ evaluate

3. **vLLM with NVFP4 + PagedAttention** (per Qwen3.6-27B bake-off article)
   - 3-4× throughput over llama.cpp at high concurrency.
   - Relevance: when M9 goes multi-family member (>2 concurrent sessions per box), vLLM not llama.cpp. Plan for this.
   - Recommendation: ✓ integrate (production runtime decision, not now)

4. **Speculative decoding with small draft models (Qwen3-0.6B drafting for Qwen3-8B)**
   - Up to 2.8× faster inference on edge per 2026 edge-AI surveys.
   - Relevance: tightens sub-1s latency goal (roadmap Open Question #4). The 0.6B draft model is free with Qwen 3 family.
   - Recommendation: ⚠ evaluate (M12 latency budget)

5. **Sustainable LLM Inference for Edge AI** (Apr 2025) — [arXiv 2504.03360](https://arxiv.org/pdf/2504.03360)
   - Energy/accuracy/latency benchmark for quantized edge LLMs.
   - Relevance: useful when sizing the Spark BOM power budget. Cite, don't integrate.
   - Recommendation: ⚠ evaluate (BOM exercise)

**Not found:**
- No 2025-2026 work specifically on **per-person LoRA hot-swap memory
  pressure** on consumer GPUs. Roadmap §M10 plans LRU N=2-4; this remains
  a design exercise without research backing.

---

## 7. Voice as Identity / Ethics

**Our current plan.** None explicit. `feedback_fail_loud` memory governs
UX honesty; nothing on legal / consent / post-mortem.

**Gap / what we might be missing.** EverHome is **explicitly a
post-mortem voice product** for many users. The 2025-2026 legal landscape
has moved fast and we have no documented consent flow.

**Top candidates:**

1. **NO FAKES Act (US, 2025)** + **EU AI Act voice-cloning "high-risk" classification**
   - US: lifetime + 70 years post-mortem voice rights.
   - EU: voice cloning = high-risk, watermarking required.
   - Relevance: **EverHome must implement a consent record and provenance trail before B2C launch**. Watermarking the output may be required for EU sale.
   - Recommendation: ✓ integrate (compliance milestone, currently absent from roadmap)

2. **California AB 2602 (Jan 2025) + AB 1836 (2026, deceased performers)**
   - Strengthens informed-consent requirements for digital replicas, covers deceased.
   - Relevance: same compliance requirement, California specifics. Diaspora ICP includes US.
   - Recommendation: ✓ integrate (compliance reference)

3. **Griefbots, Deadbots, Postmortem Avatars** (Philosophy & Technology 2024) — [Springer](https://link.springer.com/article/10.1007/s13347-024-00744-w)
   - Academic ethics framework: consent of the deceased, exploitation risks, grief impact.
   - Relevance: cite in our product positioning. EverHome's "preserve, not impersonate" stance differentiates from exploitative competitors.
   - Recommendation: ✓ integrate (product-doc citation)

4. **Digital Doppelgangers — Pre-Mortem AI Clones** (Feb 2025) — [arXiv 2502.21248](https://arxiv.org/html/2502.21248v1)
   - Ethics of cloning a still-living person; consent revocability, evolving identity.
   - Relevance: EverHome use case where elder is still alive — directly applicable. Defines a revocation UX we need.
   - Recommendation: ✓ integrate (informs the M10 / family-UI consent surface)

5. **AI deadbots and pathological grief** (The Conversation, 2025) — popular-press synthesis
   - Risk: emotional addiction to the clone, complicated mourning.
   - Relevance: not load-bearing for arch but should shape **default product behavior** (e.g., session-frequency telemetry, "are you okay?" check-ins). Out of scope for roadmap but a known product risk.
   - Recommendation: ⚠ evaluate (product-design read)

**New milestone candidate.** Roadmap has no consent / provenance /
watermark milestone. Recommend **M-Consent** as a slot between M9 and
M10, covering: (a) consent record per `(speaker_id, listener_id)`,
(b) revocability flow, (c) optional watermark on synthesized audio,
(d) jurisdiction tag at intake.

---

## Recommendations — top 5 things that should change in the roadmap

Ranked by ROI (engineering effort × strategic impact):

1. **M7 OCR/PDF stack: swap Tesseract `chi_tra` + pdfplumber for PaddleOCR-VL 1.5 + MinerU 2.5-Pro.** Highest ROI: solves the handwritten-letter known weakness, no commercial OCR fallback needed, single-tool simplification. Effort: <1 day decision; same code surface.

2. **M9 base model: Qwen 2.5 7B → Qwen 3 8B.** Same VRAM, current generation, better Chinese. Effort: model swap, validate persona-LoRA recipe transfers. Risk: low.

3. **M13 watch list update: Step-Audio 2 (Chinese SOTA), Kimi-Audio (Qwen 2.5 base = LoRA-transferable), LLaMA-Omni 2 (sub-1s already).** Chroma 1.0 is no longer the leader for Chinese. Also: **verify Qwen3.5-Omni cloning claim** — could collapse M12+M13.

4. **Add M-Consent milestone** between M9 and M10. Consent record, revocation flow, optional watermark, jurisdiction tag. NO FAKES Act + EU AI Act compliance is now a **commercial blocker** for EverHome B2C/B2B2C.

5. **M8 architecture: stance index → HippoRAG 2; add A-MEM (or Mem0) for conversation-time memory consolidation.** Plain BGE-M3 + LanceDB top-k is one generation behind. HippoRAG 2 is open and lower-indexing-cost than alternatives. A-MEM closes the "system gets smarter about grandma over time" loop that the roadmap implicitly promises but doesn't deliver.

**Honorable mentions** (worth a 1-day evaluation, not roadmap-changing):
- VoxCPM as M11 TTS swap target (48 kHz output addresses telephone-grade source limitation).
- Nautilus Compass persona-drift telemetry as M9 sub-deliverable.
- OpenCharacter (Salesforce 2501.15427) dataset for M9 instruction mix.

---

## Things we searched for and found nothing material

- **Chinese-native LoCoMo-equivalent conversational-memory benchmark.** Doesn't exist. M8 validation will need an English benchmark or a hand-rolled Chinese eval.
- **Persona LoRA recipes for Traditional Chinese / Taiwanese particle preservation.** None. Our small-corpus elder use case is novel.
- **Per-person LoRA hot-swap memory pressure on consumer GPUs.** No research; our LRU N=2-4 plan is engineering judgment, not literature-backed.
- **OSS speech-to-speech with verifiable Chinese voice cloning.** Step-Audio 2 / Kimi-Audio / LLaMA-Omni 2 don't clearly document cloning quality. Chroma's cloning is English-trained. The roadmap stance ("don't migrate until cloning ships") remains correct — no model meets all of {Chinese-native, OSS, cloning verified}.
- **VRAM budget studies for Spark-class unified-memory deployments.** No public work specific to GB10 / DGX Spark for our stack shape (LLM + TTS + multiple LoRA + RAG).

---

## Things the roadmap got wrong / outdated since drafting

1. **Roadmap §1 calls Chroma 1.0 "English-only".** The tech report
   (arXiv 2601.11141) shows English training data, but the architecture
   is language-agnostic. Re-classify as "English-trained, Chinese fine-tune
   feasible."

2. **Roadmap §1 says Qwen Omni line has "no cloning".** True for
   Qwen2.5-Omni and Qwen3-Omni-30B-A3B, but **Qwen3.5-Omni**
   (arXiv 2604.15804) is being described as having zero-shot cloning.
   Verify before locking M12. If true, M12 and M13 can collapse.

3. **Roadmap §6 cites only OpenCharacter 2511.01689.** The earlier
   2501.15427 (Salesforce) is a different paper with a usable 326K-row
   dataset; should be added as a separate reference.

4. **Roadmap §M7 OCR stack (Tesseract `chi_tra`) is 2-generation
   outdated** by 2026 standards. PaddleOCR-VL 1.5 is the new floor.

5. **Roadmap §M9 base model (Qwen 2.5 7B Instruct) is one generation
   behind.** Qwen 3 8B at the same VRAM is the current choice.

6. **Roadmap has no compliance / consent milestone.** This is a
   commercial blocker, not an engineering nice-to-have, given the
   2025-2026 regulatory shift (NO FAKES Act, EU AI Act, CA AB 2602/1836).
