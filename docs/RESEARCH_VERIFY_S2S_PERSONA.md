# Research Verification + S2S + Persona Followup

**Date:** 2026-05-28
**Author:** Mark Ko (research pass for EverHome roadmap)
**Scope:** Verify Qwen3.5-Omni cloning claim, horizontal-compare Chinese S2S models, deepen persona research, plan pipeline → S2S migration.
**Out of scope:** Editing `ROADMAP_2026Q3.md` or `RESEARCH_REVIEW_2026Q3.md` (separate update pass).

---

## 1. Verdict on Qwen3.5-Omni cloning

**Verdict: TRUE in the technical report. Qwen3.5-Omni-Light open-weight as of 2026-03-30 (Plus/Flash remain API-only).** The arXiv paper (Qwen3.5-Omni Technical Report, [arXiv:2604.15804](https://arxiv.org/abs/2604.15804), 2026-04-22) does claim zero-shot voice cloning. **Correction applied 2026-05-28 follow-up:** the Light variant has open weights on HuggingFace, contradicting an earlier draft of this section that said "weights not open today." [CLAIM] Light retains zero-shot cloning at the same quality as Plus/Flash [/CLAIM] — needs hands-on eval before relying on it. If yes, M12+M13 collapse is real and one-week deployable, not 6-12 months out — recommend a Light-variant cloning eval spike before committing M12. From the paper itself, retrieved via the arXiv HTML render:

> "Beyond preset voices, the model enables zero-shot voice cloning from user-provided samples."

> "We introduce a dedicated system prompt for Talker that specifies target voice characteristics, thereby enabling both zero-shot voice cloning and controllable speech generation. Compared with conventional speaker embeddings, this prompt can encode richer multimodal cues, including textual descriptions and codec sequences."

So the **architecture is described** — Talker takes a textual/codec speaker-prompt rather than a fixed speaker_embedding bake (which contrasts with our current Qwen3-TTS custom_voice mechanism — see `tts_voice_cloning_mechanics` memory). The paper additionally claims preserved speaker identity across 29 languages.

**But three caveats break the "collapses M12+M13" claim:**

1. **The 3-Omni open repo (Instruct / Thinking on github.com/QwenLM/Qwen3-Omni) is separate from 3.5-Omni.** The 3-Omni Instruct README is explicit: only three preset speakers (`Ethan`, `Chelsie`, `Aiden`). No `reference_audio` parameter. Source: [HF model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct). For 3.5-Omni: the **Light** variant was open-sourced on HuggingFace 2026-03-30 (see the correction note in the verdict above); Plus/Flash remain API-only.
2. **3.5-Omni is API-only.** Per [WaveSpeed analysis](https://wavespeed.ai/blog/posts/what-is-qwen3-5-omni/): cloning is "available on both Plus and Flash via API — you send a 10–30 second voice sample, and the model clones it for output." [CLAIM] open-weights for 3.5-Omni [/CLAIM] — couldn't independently confirm; aggregator sources caveat with "verify on QwenLM GitHub before depending on it."
3. **Aggregator sites conflate models.** `qwen3lm.com/voice-cloning/` shows a `reference_audio` Python snippet and the-decoder.com talks about "3-second cloning," but the latter is in fact describing **Qwen3-TTS-VC-Flash** (released 2025-12-23 — separate dedicated TTS model, not Omni). Be very careful: third-party blog posts merge Qwen3-TTS-VC + Qwen3.5-Omni + Qwen3-Omni-Flash into one capability paragraph.

**Bottom line for the roadmap.** Yes the *paper* shows the architecture EverHome will eventually want (system-prompt-style speaker control beats speaker_embedding bake for elder personas with thin recording data). No the *weights* aren't usable today. M12 (Qwen2.5-Omni-7B hybrid) and M13 (E2E S2S with cloning) remain *separate* milestones because the open-weights Omni family does not yet inherit 3.5's cloning capability. If/when QwenLM open-sources 3.5-Omni or its successor, M12+M13 do collapse — but plan as if it's 6-12 months away, not imminent.

---

## 2. Chinese S2S horizontal comparison

| Model | OSS license | Chinese ASR/TTS | Voice cloning | Cloning quality | VRAM (est.) | Latency claim | Notes |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-Omni-7B** | Apache 2.0 (open) | Strong CN | NO (fixed speakers only) | n/a | ~16-24 GB | ~250 ms | Reference for EverHome M12 hybrid. ASR/LLM/TTS in one, but cloning still goes through Qwen3-TTS sidecar. |
| **Qwen3-Omni-30B-A3B-Instruct** | Apache 2.0 (open) | Strong CN | NO — 3 preset speakers (Ethan / Chelsie / Aiden) | n/a | A3B active params; ~24-40 GB hosted | streaming | MoE 30B-A3B. Thinker+Talker split. Open weights confirmed [HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct). |
| **Qwen3-Omni-30B-A3B-Thinking** | Apache 2.0 | Strong CN | n/a — text out only | n/a | ~24-40 GB | n/a | Thinker only, no Talker. Not useful for EverHome end-user path. |
| **Qwen3.5-Omni-Light** | Apache 2.0 open weights on HF since 2026-03-30 | 10-language CN/EN | **YES — 10-30 s reference audio, zero-shot** | High per paper benchmarks; cross-lingual identity preserved | TBD (Light tier) | TBD | Open-weight Light variant deployable on-prem. arXiv 2604.15804. Plus/Flash remain API-only. Hands-on eval needed before committing M12. |
| **Qwen3-Omni-Flash-2025-12-01** | Closed (in-house) | Strong CN | NO (confirmed by HN users) | n/a | n/a | 234 ms | Predecessor to 3.5-Omni. Closed weights. Per [HN discussion](https://news.ycombinator.com/item?id=46219538) "It is an in-house closed weight model for their own chat platform." |
| **Step-Audio 2 mini (Base/Think)** | Apache 2.0 (open) | CER 3.11% on CN test sets | **YES — zero-shot, by prefilling text-audio interleaved tokens** | URO-Bench CN 83.3% win-rate | ~14-24 GB (7B-ish) | streaming | Per HF discussion: cloning by "prefilling the prompt text-audio interleaving tokens and completing new tokens based on given text." [Repo](https://github.com/stepfun-ai/Step-Audio2). |
| **Step-Audio 2.5 Realtime (StepFun)** | Closed (API only, WSS) | Native CN | YES — built on 10k+ personas + 1M-scale persona matrix | High; 82.18 on paralinguistic comprehension | n/a (API) | realtime WS | Released 2026-05; *roleplay-specific RLHF*, paralinguistic-aware. Closest-in-spirit to EverHome but closed. |
| **Kimi-Audio-7B-Instruct (Moonshot)** | Open (HF weights) | Strong CN | Partial — emotion/style transfer; *not formally zero-shot from one sample* | Good audio understanding; weaker generation | ~14 GB | ~300 ms vocoder | 13M-hour training. End-to-end speech chat but cloning isn't an advertised feature. |
| **LLaMA-Omni 2 (ICTNLP)** | Open | Bilingual EN-CN (0.5B-32B Bilingual) | NO — uses CosyVoice2 for synthesis | Inherits CosyVoice2 cloning quality | 1-32B variants; A10G fits 7B easily | 226-583 ms | Built on Qwen2.5 backbone + Whisper-large-v3 + CosyVoice2. Modular, not truly unified. |
| **GLM-4-Voice-9B (Zhipu/Tsinghua)** | Open (HF zai-org/glm-4-voice-9b) | Native CN+EN | Partial — emotion/intonation/dialect controlled by instruction, not by ref audio | Good prosody control | ~18-22 GB | streaming (10-token min before audio) | Speech tokenizer (12.5 tok/s) + CosyVoice decoder. Strong CN dialogue. |
| **Mini-Omni 2 (gpt-omni)** | Open | EN-leaning | NO — uses CosyVoice + SNAC | n/a | ~7 GB | streaming | Vision+speech+text. Qwen2 backbone. Cute project, but not production. |
| **IndexTTS-2 (Bilibili)** | Open (Sept 2025) | Native CN, EN, JP | **YES — 3-10 s ref audio** + emotion audio + emotion text | "Stronger than 11Labs" per third-party claims; emotion + duration control | ~8-12 GB | non-streaming AR | TTS-only (not S2S). Strong candidate for replacing Qwen3-TTS in our current pipeline. [Repo](https://github.com/index-tts/index-tts). |
| **MiniMax Speech-02 / abab-voice** | Commercial API only | Native CN (multiple dialects) | YES — 10 s clone | Excellent (industry-leading per HD demos) | n/a | API | Pro/Enterprise plans grant commercial license. *Not OSS — violates EverHome privacy moat*. |
| **OpenAI Realtime / Anthropic voice** | Closed | EN-leaning | NO custom cloning (preset voices) | n/a | n/a | <500 ms | Context only. Violates privacy moat. |

### Verdicts (2-3 sentences each)

- **Qwen2.5-Omni-7B** — Today's pragmatic M12 target. Fits A10G with room for sidecar TTS. No cloning, so still needs Qwen3-TTS or IndexTTS-2 for elder voices. ✓ for hybrid pipeline.
- **Qwen3-Omni-30B-A3B-Instruct** — Bigger and prettier than 2.5, but 3 fixed speakers is a dealbreaker for legacy use. A3B (active params) is manageable, but VRAM for full model + RAG + TTS sidecar pushes off A10G onto Spark. ⚠ premature.
- **Qwen3.5-Omni** — The architectural endgame *if* weights open. Until then, irrelevant to a privacy-first on-device appliance. ✗ for now.
- **Step-Audio 2 mini (open)** — **Strongest current open-weight S2S candidate with cloning.** Apache 2.0, zero-shot via token-interleaving, CN-native (CER 3.11%). Worth a serious eval pass *before M12* commits to Qwen-Omni. ✓✓ investigate first.
- **Step-Audio 2.5 Realtime (API)** — Closed, but roleplay-RLHF and persona-matrix training is exactly EverHome's domain. Watch as a benchmark / quality ceiling reference. ⚠ closed.
- **Kimi-Audio-7B** — Excellent audio understanding, weaker generation. Useful as an ASR/understanding fallback, not as a TTS replacement. ⚠ partial.
- **LLaMA-Omni 2** — Built on Qwen2.5 + CosyVoice2. The cloning is really CosyVoice2's; no novel cloning IP. Could be a fast prototype. ⚠ adds nothing over running Qwen2.5 + CosyVoice2 ourselves.
- **GLM-4-Voice-9B** — Solid CN dialogue model, but instruction-controlled prosody ≠ ref-audio cloning. Falls short of legacy-preservation need. ⚠.
- **IndexTTS-2** — Not S2S, but a *very* compelling drop-in replacement for Qwen3-TTS today. Voice + emotion + duration control from 3-10 s ref audio. Independent of any S2S migration. ✓ worth a parallel eval track.
- **MiniMax Speech-02** — Quality is excellent but closed-API + commercial-license = violates `project_legacy_product_vision` privacy moat. ✗.

### Ranked: which to migrate to first if cloning verified

1. **Step-Audio 2 mini (open)** — only currently-shippable open-weight option with native zero-shot cloning + Chinese-native. First eval target.
2. **Qwen3.5-Omni** when/if open-sourced — long-term architectural fit, especially the system-prompt speaker mechanism that scales to thin-data elder personas.
3. **Qwen2.5-Omni-7B + IndexTTS-2 sidecar (hybrid)** — safe fallback. Beats current Qwen3-TTS sidecar without giving up the modular pipeline. This is essentially "M12 with a better TTS."

---

## 3. Persona research deep-dive

Recent (2025–2026) work directly relevant to personal-legacy AI:

1. **OpenCharacter** — [arXiv:2501.15427](https://arxiv.org/abs/2501.15427), Salesforce Jan 2025.
   - *What:* SFT on 20K+ synthetic personas (PersonaHub-derived). Two strategies — response-rewrite (`-R`) and response-generate (`-G`). 8B LLaMA-3 matches GPT-4o on role-play.
   - *Relevance:* M9 already centered on this paper. Lesson reinforced — character generalization improves with *scale of synthetic personas* not depth per persona.
   - ✓ Use. Already canonical for EverHome.

2. **Persistent Personas? Role-Playing, Instruction Following, and Safety in Extended Interactions** — [arXiv:2512.12775](https://arxiv.org/pdf/2512.12775), Dec 2025.
   - *What:* Benchmarks persona fidelity over 100+ rounds across 7 SOTA LLMs. Finding: **persona fidelity degrades with dialogue length, especially goal-oriented dialogue.**
   - *Relevance:* Hard finding for EverHome. Elder users will accumulate 1000s of turns over months. Plain LoRA + RAG insufficient — need explicit identity reinjection (paper proposes a state-management protocol).
   - ✓ Use. Cite as risk for M9/M10. Anchor reinjection from `RFC_M6` §9 is essentially this paper's recommendation.

3. **ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence** — [arXiv:2509.25299](https://arxiv.org/abs/2509.25299), Sept 2025.
   - *What:* Grounds agent persona in a dynamic identity knowledge graph (beliefs, traits, values) explicitly retrieved each turn.
   - *Relevance:* Directly informs M8 (Memory RAG) design. Dual-index (semantic facts + identity graph) is exactly what `persona_llm_architecture` memory describes. ID-RAG provides retrieval scoring and consistency metrics we can adopt.
   - ✓ Use. Strongest candidate to redesign M8 around.

4. **AMADEUS + CharacterRAG dataset** — discussed in [Dynamic Context Adaptation for Consistent Role-Playing Agents with RAG, arXiv:2508.02016](https://arxiv.org/pdf/2508.02016), Aug 2025.
   - *What:* Training-free framework + 15-character / 450-QA dataset. Tests whether RAG retains persona consistency when answering out-of-knowledge questions.
   - *Relevance:* Directly answers Task 3's "character-grounded RAG" question. Their training-free architecture is a cheap test we can run on EverHome before committing to M9 LoRA.
   - ✓ Use. Eval harness candidate.

5. **Memory-Driven Role-Playing: Evaluation and Enhancement of Persona Knowledge Utilization** — [arXiv:2603.19313](https://arxiv.org/pdf/2603.19313), early 2026.
   - *What:* Decomposes role-playing memory use into 4 stages (cue → retrieve → integrate → respond) with per-stage diagnostic metrics. Built on CharacterEval (CN benchmark).
   - *Relevance:* Gives us *evaluation methodology* for "is the trained persona actually right?" — Task 3's evaluation question. CharacterEval is Chinese-native (Chinese novels/scripts) which suits EverHome.
   - ✓ Use. Run CharacterEval on our persona stack as M9 acceptance gate.

6. **FinePE: Fine-grained personality editing via Mixture of LoRA Experts** — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494626003911), 2026.
   - *What:* MoLE-style routing across personality-subtrait LoRAs. Psychological-trait-decomposed adapters.
   - *Relevance:* Directly addresses Task 3's "persona-LoRA composition" — separate "writing style" + "factual memory" + "speech mannerisms" adapters via gating.
   - ⚠ Promising but unverified for our domain. Would be a big M9.5 / M11 add-on, not core path. Watch.

7. **The Making of Digital Ghosts: Designing Ethical AI Afterlives** — [arXiv:2511.20094](https://arxiv.org/pdf/2511.20094), Nov 2025.
   - *What:* Ethical framework for deceased-person modeling. Covers consent, psychological harm to children, identity distortion.
   - *Relevance:* EverHome IS this category. We need a written ethical position before customer onboarding for the "preserve grandma" use case. Suggests strong disclosure + consent + safety standards (not bans).
   - ✓ Use. Cite in the legacy-product onboarding flow. Not technical, but blocking for product.

8. **Tencent PersonaHub** — [GitHub](https://github.com/tencent-ailab/persona-hub).
   - *What:* 1B synthetic personas; 370M elite subset released 2025.
   - *Relevance:* OpenCharacter trains on this. For low-data elders, we can sample-condition on demographically-similar PersonaHub personas to *augment* the elder's 50 turns into a synthetic 5K-turn corpus before LoRA-tuning. Cheap solution to the "elder won't have 326K examples" problem.
   - ✓ Use. Data-augmentation strategy for M9.

### "Soul prompts" vs fine-tuning — when each is right

Synthesis from recent surveys:
- **Prompt-only works** when persona is famous and well-represented in pretraining (e.g. 小S — current EverHome state). Drift over long dialogue but cheap.
- **Fine-tuning needed** when persona is private (elder, deceased relative) — pretraining has no signal. Even then, *fine-grained* persona prompts add diminishing returns past coarse summaries (per recent prompt-engineering surveys).
- **Hybrid (what EverHome should do):** small LoRA for persona-specific style/mannerisms + persona prompt for runtime control + RAG for facts. This matches `persona_llm_architecture` memory.

---

## 4. Pipeline-to-S2S migration analysis

EverHome's current persona-data pipeline (5 stages):

| # | Stage | File(s) | Survives S2S? | Notes |
|---|---|---|---|---|
| 1 | Recordings → speaker embedding (timbre) | `app/services/training_service/` (RFC M2 + M4); `tts_voice_cloning_mechanics` memory says timbre is baked via `custom_voice` config | **REPLACED** if S2S has zero-shot system-prompt cloning (Qwen3.5-Omni style). The ref-audio + Talker-prompt obsoletes the speaker_embedding bake. The *recording capture pipeline itself* survives (diarization, segmentation, ASR for verification). | The bake step in `training_job.py` becomes a no-op or moves to "store ref audio + transcript pairs." Worth keeping recording pipeline running through M9/M10 — it's reusable. |
| 2 | SFT/LoRA on Qwen3-TTS (prosody / speaker style) | `app/services/training_service/training_job.py`, `app/services/tts/qwen_tts_engine.py` | **OBSOLETE** if S2S Talker takes ref audio directly. LoRA on a separate TTS becomes pointless. | Big sunk cost if M12/M13 trigger. Investing in M4-style TTS LoRA polish during M9/M10 is risky. *Recommend: freeze TTS LoRA work after current shipping state. Don't add features.* |
| 3 | Persona prompts (text behavior) | `app/services/llm/prompt_manager.py` | **SURVIVES UNCHANGED.** S2S models still take system prompts. Format is identical. | Highest-leverage code in the project right now. Every dollar spent on M9 persona-LLM LoRA is also dollars spent improving these prompts. |
| 4 | RAG memory (planned M8) | `RFC_M6_PERSONA_LLM_LEGACY.md` Phase 1-2; BGE-M3 + LanceDB | **SURVIVES.** RAG sits *above* the model — S2S models can consume retrieved context exactly like a text LLM does. Maybe even more important in S2S (no separate ASR/LLM/TTS stages to inject facts at). | Build M8 with confidence. It pays off whether we stay hybrid or migrate. ID-RAG (§3 #3) is the architectural target. |
| 5 | Emotion mapping `[E:寵溺]` → instruct string | `app/services/tts/emotion_mapper.py` | **PARTIALLY OBSOLETE.** Today's emotion tag → TTS-instruct mapping is a workaround because Qwen3-TTS takes a free-form instruct. End-to-end S2S models (Qwen3.5-Omni, Step-Audio 2.5) handle emotion natively in speech output — no tag parsing. | The *state machine parser* survives as long as we route to Qwen3-TTS. After S2S cutover the [E:...] tag itself becomes a hint to the S2S system prompt, not a separate processing stage. emotion_mapper.py shrinks by ~80%. |

### Roadmap milestone risk assessment (re: S2S pivot in 6 months)

| Milestone | Survives S2S? | Investment recommendation |
|---|---|---|
| **M7** Text/ebook/image ingestion | ✓✓ Untouched by S2S. Corpus is upstream of any model. | **Build now, full speed.** |
| **M8** Memory RAG | ✓✓ Survives. ID-RAG architecture maps cleanly onto S2S. | **Build now, full speed.** Use ID-RAG (arXiv 2509.25299) as architectural reference. |
| **M9** OpenCharacter persona LLM | ✓ Survives as data + technique. LoRA target model may change (Qwen2.5-7B → Qwen2.5-Omni base → Step-Audio 2 base). | **Build the data pipeline and eval harness now.** Train LoRA against current backbone but design for backbone-swap. Add CharacterEval gate (§3 #5). |
| **M10** Multi-listener TTS routing | ⚠ Becomes obsolete if S2S Talker accepts per-utterance speaker prompt directly. Adapter registry still useful for *speaker prompt selection*. | **Build the persona/listener routing logic, NOT the LoRA-swap mechanics.** The first 60% of `RFC_M5` §15 (config, routing, persona × listener matrix) survives. The actual `set_active_adapter()` LoRA swap (§15 line 5) becomes a "set speaker_prompt" call. |
| **M11** TTS engine abstraction | ✓✓ **Critical.** This *is* the migration insurance policy. | **Bump priority from LOW to MEDIUM-HIGH.** Without `BaseTTSEngine`, swapping Qwen3-TTS for Step-Audio 2 or IndexTTS-2 is invasive. |
| **M12** Qwen2.5-Omni hybrid | At risk — Step-Audio 2 mini may be a better target | **Add Step-Audio 2 mini eval as a parallel sub-milestone (M12a) before committing.** Run on the M9 CharacterEval gate. |
| **M13** OSS E2E S2S w/ cloning | The whole point of this analysis. Was opportunistic; **upgrade to definite if Step-Audio 2 cloning works.** | **Run Step-Audio 2 mini Apache-2.0 pilot in parallel with M12.** Don't wait. |

---

## 5. Roadmap impact summary (top 5 ranked changes)

1. **Add M12a: Step-Audio 2 mini evaluation track.** Apache 2.0, native CN, zero-shot voice cloning. This is the only currently-shippable open-weight S2S with cloning. Run before — or in parallel with — M12 Qwen2.5-Omni hybrid. If it passes CharacterEval + voice-quality gate, it leapfrogs both M12 and M13.

2. **Promote M11 (TTS engine abstraction) from LOW to MEDIUM-HIGH.** The `BaseTTSEngine` abstraction is the *migration insurance policy* — without it, swapping TTS backends (Qwen3-TTS → IndexTTS-2 → Step-Audio 2 → Qwen3.5-Omni when open) requires deep surgery in `app/services/tts/qwen_tts_engine.py` and `app/api/ws_asr.py`. Move it ahead of M10.

3. **Freeze TTS LoRA feature work after current state.** Continue using shipped Qwen3-TTS LoRA, but don't add features (per `RFC_M4`). Per Task 4 analysis, this is the stage most likely to be obsoleted. Reroute that team's bandwidth to M8 RAG or M9 persona LLM where the work compounds.

4. **Redesign M8 around ID-RAG** ([arXiv:2509.25299](https://arxiv.org/abs/2509.25299)). Dual-index isn't enough — need explicit identity knowledge graph that survives long dialogue (per [arXiv:2512.12775](https://arxiv.org/pdf/2512.12775) showing persona fidelity degrades over 100+ rounds). This is the highest-leverage change for the elder use case where dialogues will run for months.

5. **Run a Qwen3.5-Omni-Light cloning spike NOW (highest-leverage single action).** Light is open-weight on HF since 2026-03-30 (confirmed). Spend one day testing zero-shot cloning quality on the existing test/v11 source audio. If quality lands at the paper-claimed level, M12 (Qwen2.5-Omni hybrid) and M13 (E2E S2S w/ cloning) **collapse into a one-week deployment milestone** instead of a 6-12 month wait. Step-Audio 2 mini remains the parallel open-weight fallback if Light's cloning underperforms on Traditional Chinese / Taiwan accent.

---

## Sources

- [arXiv:2604.15804 — Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)
- [HuggingFace: Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [HuggingFace: Qwen/Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [GitHub: QwenLM/Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)
- [WaveSpeed: What Is Qwen3.5-Omni](https://wavespeed.ai/blog/posts/what-is-qwen3-5-omni/)
- [Hacker News: Qwen3-Omni-Flash-2025-12-01](https://news.ycombinator.com/item?id=46219538)
- [the-decoder: Qwen voice cloning from three seconds](https://the-decoder.com/alibabas-new-qwen-models-can-clone-voices-from-three-seconds-of-audio/)
- [GitHub: stepfun-ai/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2)
- [Step-Audio 2 Technical Report (arXiv:2507.16632)](https://arxiv.org/pdf/2507.16632)
- [MarkTechPost: StepFun StepAudio 2.5 Realtime](https://www.marktechpost.com/2026/05/24/stepfun-releases-stepaudio-2-5-realtime-an-end-to-end-voice-model-with-roleplay-specific-rlhf-and-paralinguistic-comprehension/)
- [HuggingFace: moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)
- [GitHub: MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio)
- [GitHub: ictnlp/LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2)
- [arXiv:2505.02625 — LLaMA-Omni 2](https://arxiv.org/pdf/2505.02625)
- [HuggingFace: zai-org/glm-4-voice-9b](https://huggingface.co/zai-org/glm-4-voice-9b)
- [arXiv:2412.02612 — GLM-4-Voice](https://arxiv.org/pdf/2412.02612)
- [GitHub: gpt-omni/mini-omni2](https://github.com/gpt-omni/mini-omni2)
- [GitHub: index-tts/index-tts (IndexTTS-2)](https://github.com/index-tts/index-tts)
- [MiniMax Speech-02 docs](https://platform.minimax.io/docs/guides/speech-voice-clone)
- [arXiv:2501.15427 — OpenCharacter (Salesforce)](https://arxiv.org/abs/2501.15427)
- [arXiv:2512.12775 — Persistent Personas? (extended interactions)](https://arxiv.org/pdf/2512.12775)
- [arXiv:2509.25299 — ID-RAG (Identity RAG)](https://arxiv.org/abs/2509.25299)
- [arXiv:2508.02016 — Dynamic Context Adaptation for RP Agents (AMADEUS/CharacterRAG)](https://arxiv.org/pdf/2508.02016)
- [arXiv:2603.19313 — Memory-Driven Role-Playing](https://arxiv.org/pdf/2603.19313)
- [arXiv:2401.01275 — CharacterEval CN benchmark](https://arxiv.org/abs/2401.01275)
- [arXiv:2511.20094 — The Making of Digital Ghosts](https://arxiv.org/pdf/2511.20094)
- [GitHub: tencent-ailab/persona-hub](https://github.com/tencent-ailab/persona-hub)
- [ScienceDirect — FinePE Mixture of LoRA Experts](https://www.sciencedirect.com/science/article/abs/pii/S1568494626003911)
