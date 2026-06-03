# EverHome — Project Brief

> Self-contained context for external LLMs (OpenAI / Gemini) doing plan or
> code review. Read this first; everything else is a pointer.
>
> Generated 2026-06-03 from MEMORY.md, CLAUDE.md, ROADMAP_2026Q3.md.
> Keep this doc under ~5K tokens — it gets prepended to every external
> review prompt.

---

## 1. Product (one paragraph)

**EverHome — "A Time Capsule You Can Talk To."** A privacy-first family-
memory appliance. Family members upload a person's voice, text, photos,
chat exports; EverHome trains per-person voice + persona, and lets
relatives later talk to that person — in their voice, with their cadence,
knowing their stories. The box runs on hardware the family owns; no
personal data leaves it. The 小S (and now Mark) famous-figure pipeline is
the **validation path** for this same appliance, not a separate product.

Buyers: (1) B2C diaspora Chinese families preserving an elder. (2) B2B2C
eldercare / hospice / memorial-service providers.

**The moat = voice cloning depth × curated family corpus + memory graph.**
Voice cloning at close-relative recognition is NECESSARY (the product
fails its recognition test otherwise) but not SUFFICIENT. The
durable, hard-to-copy asset is the per-family corpus + memory graph +
persona model that takes years of family participation to build. A
competitor who ships better corpus/memory tooling with worse cloning
can outcompete a competitor who ships better cloning with worse
corpus/memory tooling. Treat M7 (ingest) + M8 (memory RAG) + M9
(persona LLM) as co-equal moat investments with M10/M11 — not as
plumbing for the voice layer. Cloning approach: SFT/LoRA on dedicated
TTS, not omni S2S models that can't fine-tune new speakers (per
`docs/RESEARCH_SFT_S2S.md`).

---

## 2. Non-negotiable design rules

1. **Privacy first.** Every feature: "does this leak data off the box?"
   is the first design question. LLM is currently OpenAI (cloud) only
   as a Phase 1 stopgap — M9 swaps it to local Qwen 3 8B + LoRA.
2. **Voice cloning depth over latency.** SFT/LoRA on Qwen3-TTS per
   person. Don't replace it with zero-shot omni models. See
   `docs/RESEARCH_SFT_S2S.md` — no current OSS Chinese S2S supports
   SFT for new speakers; that moat holds for 12+ months.
3. **Three-tier persona stack** (frozen base + per-person LoRA + RAG).
   Don't propose prompt-only (only works for famous figures, not
   elders) or full-SFT (catastrophic forgetting on small corpora).
4. **Chinese-native at every layer** (LLM Qwen, TTS Qwen3-TTS, ASR
   Qwen3-ASR, embedding BGE-M3). Deliberate choice — diaspora Chinese
   families are the primary market.
5. **Fail loud.** Never `status=ready` with empty result, never empty
   HTTP body + 200. UI must surface failures.
6. **Disclosure rule.** When the persona is asked directly "are you
   human?" it must disclose AI identity. Coded into persona prompts.

---

## 3. Current stack (2026-06-03)

```
Audio in →  Browser raw PCM (onaudioprocess) via WebSocket binary
ASR      →  Qwen3-ASR (with hallucination filter — drops "The first
            was the first to be built." stock phrases + peak<0.12 silence)
LLM      →  OpenAI gpt-4o-mini (PROVISIONAL — M9 swaps to local Qwen 3 8B)
            + per-utterance LANGUAGE directive (UI dropdown: auto/中文/English)
Emotion  →  EmotionParser state machine ([E:emotion]content streaming)
TTS      →  Qwen3-TTS 1.7B SFT (per-persona merged model)
            + DeepFilterNet denoise + LUFS normalize at training time
Audio out → AudioWorklet (Int16 PCM)
Memory   →  In-session conversation_history (cap 20 turns). No cross-
            session memory yet (M8 fixes this).
```

**Files:** `app/api/ws_asr.py` (WS pipeline), `app/core/state_manager.py`
(session), `app/services/{asr,llm,tts}/` (engines),
`app/resources/personas/*.json` (personas).

**Hardware:**
- Dev: AWS g5.4xlarge (A10G 23 GB)
- Production target: NVIDIA DGX Spark 128 GB (unified memory)
- Mac Mini = inference-only fallback (no training)

**Public:** `https://everhome.mkk.dev` (Cloudflare named tunnel). Landing
at `/`, chat UI at `/ui`, demo mode `/ui?demo=1`.

---

## 4. Milestones — sequencing (Q3 2026)

Past: **M-Demo** (06/02 NY Tech Week) — done.

Up next (this conversation's working order):

| ID | Name | Effort | Why now |
|---|---|---|---|
| **D-Retro** | Demo retrospective + harden fixes from 06/02 (cache buster, hallucination filter, listen-only, language toggle) | 1-2 days | Lock in what worked; small tests |
| **M7** | Text / ebook / image ingestion (PDF via MinerU 2.5, photos via PaddleOCR-VL 1.5) | 2-3 weeks | Required input for M8 |
| **M8** | Memory RAG — ID-RAG (arXiv 2509.25299) + HippoRAG 2 + BGE-M3 + LanceDB | 2-3 weeks | The product. AI without cross-session memory is not legacy preservation |
| **M-Consent** | Consent capture + revocation + watermark + audit. NO FAKES Act / EU AI Act / CA AB 1836 driven | 1-2 weeks | Parallel to M7; B2B blocker |
| **M11** | `BaseTTSEngine` abstraction — swap-ready scaffolding | 3-5 days | Should ship before M10/M12 |
| **M8.5** | Instruction-conditioned TTS fine-tuning (emotion2vec auto-label + Constitutional AI, CosyVoice 2 paradigm) | 3-4 weeks | Industry survey confirmed ordering: do BEFORE M9 |
| **M9** | OpenCharacter persona LLM (Qwen 3 8B + per-person LoRA per arXiv 2511.01689) | 4-6 weeks | Local-LLM privacy moat |
| **M10** | Multi-listener voice routing (per-listener LoRA adapters) | 2 weeks | After M11 abstraction |
| **M12** | Hybrid pipeline with Qwen 2.5 Omni 7B (ASR+LLM, TTS stays Qwen3) | 2-3 weeks | Latency win |
| **M12a** | 1-day Qwen3.5-Omni-Light cloning eval spike | 1 day | If cloning works, M12+M13 collapse |
| **M13** | OSS E2E speech-to-speech with cloning | Opportunistic — only when Step-Audio 2 / Kimi-Audio / similar ship SFT recipe |

**Single-line stance:** Don't trade the moat (voice cloning) for the
trend (E2E S2S). Build swap-ready scaffolding now; swap when an OSS
model with cloning actually exists.

---

## 5. Known gaps / things people get wrong

| Gap | Why it matters |
|---|---|
| LLM is cloud (OpenAI) | Violates "100% local" privacy moat. M9 fixes |
| No cross-session memory | AI doesn't actually preserve anything. M8 fixes |
| No text/photo/PDF ingest | Can only train from voice. M7 fixes |
| Persona LoRA not built | LLM mimics 小S/Mark via prompt only. Works for famous figures, fails for elders. M9 fixes |
| `talker.model` LoRA only trains `code_predictor` (5 layers) — cloning depth ceiling | Bigger SFT/LoRA scope is in `training_pipeline_deferred` memory |
| English prosody mismatch on Chinese-trained voice | Voice timbre preserved, but pace/flow wrong. Fix: add English audio to training corpus, OR M8.5 instruction-conditioning |
| No consent / watermark / audit | Commercial blocker for B2B. M-Consent fixes |

---

## 6. Things external reviewers should NOT propose

- **Zero-shot 10-30s voice cloning** (ElevenLabs / OpenAI Voice / closed APIs). Cannot match close-relative recognition; cloud-only violates privacy moat.
- **Prompt-only persona steering** for the legacy product. Only works for famous figures.
- **Full-parameter SFT of base LLM** on small corpora. Catastrophic forgetting (ICCV 2025 SMoLoRA, OPLoRA 2510.13003).
- **OpenAI text-embedding-3** as default. BGE-M3 is the standard here (Chinese-native, hybrid, local).
- **Qwen3-Omni** as TTS path. Closed tokenizer, no new-speaker SFT, only 3 preset speakers. Useful for ASR+LLM only.
- **Replacing TTS-LoRA with omni S2S** before an OSS model ships SFT recipe. Conditional NO per `RESEARCH_SFT_S2S.md`.

---

## 7. Naming + conventions

- "Persona" = a person we model (small S, Mark, an elder). One JSON
  under `app/resources/personas/<id>.json` + optional trained voice
  version at `data/models/merged_qwen3_tts_<id>_v*`.
- "Listener" = the relationship the persona is speaking to
  (child/mom/friend/reporter/elder/default). Affects prompt tone, not
  voice (yet — M10 adds per-listener adapter).
- "Version" = a TTS LoRA training run. `v12`, `v13` etc. Repo at
  `data/training/<persona>/versions/`.
- Tests use `MockASR` / `USE_MOCK_TTS=true`. Production uses real
  Qwen models on GPU.

---

## 8. Where to look for more depth

| If question is about | Read |
|---|---|
| Why this product exists | `~/.claude/projects/-home-rding-voice-ai-pipeline/memory/project_legacy_product_vision.md` |
| LLM stack rationale | `~/.claude/projects/.../memory/persona_llm_architecture.md`, `RFC_M6_PERSONA_LLM_LEGACY.md` |
| Why we don't use omni S2S | `docs/RESEARCH_SFT_S2S.md`, `docs/RESEARCH_VERIFY_S2S_PERSONA.md` |
| TTS voice cloning internals | `~/.claude/projects/.../memory/tts_voice_cloning_mechanics.md`, `tts_inference_language_gotcha.md` |
| Hardware choices | `~/.claude/projects/.../memory/hardware_targets.md` |
| Full roadmap with effort + deps | `docs/ROADMAP_2026Q3.md` (1271 lines) |
| Demo speaker note / slides | `docs/speaker_note.md`, `docs/slide_skeleton.md` |
| Architecture & WebSocket protocol | `CLAUDE.md` |

---

## 9. Open decisions / risks

- **English prosody fix path** (deferred): inference-time instruct hack vs add English audio to training corpus vs M8.5 instruction-conditioning. User leaning toward "record English when have time" (the right fix).
- **Step-Audio 2 mini** as the leading open-weight S2S bet — zero-shot only, no SFT exposed yet. Track GitHub issue #67.
- **Qwen3.5-Omni-Light** open-weights since 2026-03-30 — M12a spike will tell us if E2E S2S timeline collapses.
- **DGX Spark availability** is the production-hardware gating item.
