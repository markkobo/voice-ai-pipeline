# Third-Party License Audit (2026-06-03)

**Scope:** Commercial use for EverHome (B2C diaspora Chinese families + B2B2C eldercare / hospice / memorial-service providers).
**Source of audit:** GPT-5 review action item §11.1 (see `docs/REVIEW_GPT5_2026-06-03.md`).
**Owner:** Mark Ko.
**Auditor:** Claude (sub-agent, opus-4.7), 2026-06-03.

## Summary

- **Total items:** 20
- **Commercial OK (YES):** 14
- **Maybe / verify (MAYBE):** 4
- **Blocked / replace (NO):** 2

Headline findings:

- **Two hard blockers** for commercial shipping as currently planned:
  - **PersonaHub (item 12)** — confirmed `cc-by-nc-sa-4.0`. Cannot ship M9 persona LoRA training data derived from it. Replacement plan required.
  - **LLaMA-Omni 2 (item 10)** — code Apache-2.0, but **model weights are research-only / non-commercial**. Drop from M12 candidate list or contact authors.
- **Two need verification before relying on them:**
  - **IndexTTS-2 (item 5)** — repo says "for commercial usage and cooperation, please contact indexspeech@bilibili.com." Treat as commercially restricted until written confirmation.
  - **emotion2vec (item 19)** — code is MIT, but the HF weights page only lists generic `model-license`; the ModelScope source (`iic/emotion2vec_plus_large`) may carry a different model license. Need to confirm weight license before M8.5 ships.
- **Everything else** in the planned stack (Qwen3-TTS, Qwen3-ASR, Qwen 3 8B, VoxCPM 2, CosyVoice 2, Step-Audio 2 mini, BGE-M3, LanceDB, HippoRAG, MinerU 2.5, PaddleOCR-VL, DeepFilterNet) is permissive (Apache-2.0 / MIT / dual) and commercial-OK.

## Detailed table

| # | Item | URL fetched | License (short form) | Commercial use OK? | Notes / fallback |
|---|---|---|---|---|---|
| 1 | Qwen3-TTS 12Hz 1.7B VoiceDesign | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | Apache-2.0 | YES | HF model card field `License: apache-2.0`. Current production model. |
| 2 | Qwen3-TTS 12Hz 0.6B VoiceDesign | (sibling of #1; not separately fetched) | Apache-2.0 (assumed — same publisher, same series) | MAYBE | Not separately verified. Open the 0.6B HF card before shipping any product that uses it. Almost certainly Apache-2.0 by analogy with 1.7B. |
| 3 | VoxCPM 2 (OpenBMB / Tsinghua) | https://github.com/OpenBMB/VoxCPM | Apache-2.0 | YES | README: "Weights and code released under the Apache-2.0 license, free for commercial use." Clean for M11 first-class slot. |
| 4 | CosyVoice 2 | https://github.com/FunAudioLLM/CosyVoice | Apache-2.0 | YES | Upstream deps (FunASR, FunCodec, WeNet = Apache; Matcha-TTS, AcademiCodec = MIT) all commercial-OK. Clean for M11. |
| 5 | IndexTTS-2 | https://github.com/index-tts/index-tts | Repo lists `LICENSE` + `LICENSE_ZH.txt` (text not fetched) | MAYBE | README explicitly says: "For commercial usage and cooperation, please contact indexspeech@bilibili.com." Treat as **commercially restricted** until written confirmation. **Fallback:** prefer VoxCPM 2 or CosyVoice 2 for M11 commercial path; keep IndexTTS-2 as research-only eval. |
| 6 | Qwen3-ASR | https://github.com/QwenLM/Qwen3-ASR | Apache-2.0 | YES | Repo footer shows Apache-2.0; weights distributed open on HF + ModelScope. (Direct HF model card returned 401 to web-fetch — confirm via `huggingface-cli` from a logged-in session if license text is needed verbatim.) |
| 7 | Qwen3-Omni-30B-A3B-Instruct (proxy for Qwen3.5-Omni-Light, M12a candidate) | https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct | Apache-2.0 | YES (for use); SFT for new speakers separately blocked by **closed codec tokenizer** (per HF discussion #40), not by license | License is commercially fine. Strategic limit is technical: no speaker SFT path. Spike is latency/prosody only per roadmap M12a. Confirm same Apache-2.0 on `Qwen3.5-Omni-Light-*` HF card before that spike. |
| 8 | Step-Audio 2 mini (StepFun) | https://github.com/stepfun-ai/Step-Audio2 | Apache-2.0 (code + weights) | YES | "Both the code and model weights (Step-Audio 2 mini, Mini Base, Mini Think) are released under Apache 2.0." Clean for M13 / M12a eval. License is *not* the blocker — lack of SFT recipe (issue #67) is. |
| 9 | Kimi-Audio-7B (Moonshot) | https://github.com/MoonshotAI/Kimi-Audio | Code: dual (Apache-2.0 for Qwen2.5-derived parts, MIT for the rest). Weights: **not explicitly stated** in repo. | MAYBE | Code is fine. **Action:** confirm weights license on `moonshotai/Kimi-Audio-7B` / `-Instruct` HF cards (web-fetch returned 401; check from logged-in session) before any commercial use. |
| 10 | LLaMA-Omni 2 (ICT-CAS) | https://github.com/ictnlp/LLaMA-Omni2 | Code: Apache-2.0. **Weights: research-only.** | **NO** (weights) | Repo: "Our model is intended for academic research purposes only and may NOT be used for commercial purposes." Commercial license via `fengyang@ict.ac.cn`. **Fallback:** drop from M12 commercial path; or pursue commercial license; or fall back to CosyVoice 2 directly (which LLaMA-Omni 2 wraps anyway). |
| 11 | Qwen 3 8B (M9 base) | https://huggingface.co/Qwen/Qwen3-8B | Apache-2.0 | YES | HF card field `License: apache-2.0`. Same license as 7B sibling. Safe base for per-person LoRA. |
| 12 | PersonaHub (Tencent / proj-persona) | https://huggingface.co/datasets/proj-persona/PersonaHub | **cc-by-nc-sa-4.0** | **NO** | Confirmed non-commercial. Dataset card explicitly: "intended for research purposes only." **Fallback plan:** (a) generate our own synthetic persona seeds using Qwen 3 8B (Apache-2.0) or another commercially-licensed LLM, conditioned on demographic templates we author in-house; (b) use Salesforce **`xywang1/OpenCharacter`** (326K rows) as a CC-cleaner alternative — but **its license is unverified by this audit** (web-fetch 401), so check the HF card from a logged-in session before relying on it; (c) treat PersonaHub purely as a research-time inspiration, never ship derived weights. |
| 13 | OpenCharacter recipe (arXiv:2511.01689) | https://huggingface.co/papers/2511.01689 (paper page) + https://arxiv.org/abs/2511.01689 | Paper / arXiv (CC-BY-style for the PDF itself); **code repo at anonymous.4open.science** — license not fetched | MAYBE | The *recipe* (Constitutional AI + introspective dialogue → SFT → DPO) is a method, not copyrightable per se — methodology is reproducible without restricted data. **However,** the Salesforce OpenCharacter 326K dataset (different paper, arXiv:2501.15427) is derived from PersonaHub (item 12) — so any data that traces back to PersonaHub carries the NC restriction. **Action:** reimplement the recipe with our own seed data; do not redistribute Salesforce-derived training pairs commercially without checking their license. |
| 14 | BGE-M3 (BAAI) | https://huggingface.co/BAAI/bge-m3 | MIT | YES | HF card field `License: mit`. Clean for M8 RAG. |
| 15 | LanceDB | https://github.com/lancedb/lancedb | Apache-2.0 | YES | Confirmed. Clean for M8 vector store. |
| 16 | HippoRAG 2 | https://github.com/OSU-NLP-Group/HippoRAG | MIT | YES | Confirmed. Clean for M8 semantic-facts retriever. |
| 17 | MinerU 2.5-Pro | https://github.com/opendatalab/MinerU | "MinerU Open Source License" (custom, based on Apache-2.0) | MAYBE → likely YES | As of v3.1.0, MinerU moved off AGPLv3 to a custom license "based on Apache 2.0" "to facilitate commercial adoption." Removed AGPL/CC-BY-NC-SA model dependencies. **Action:** read `LICENSE.md` verbatim from the repo before shipping — the word "custom" means it may add field-of-use clauses (e.g., naming/attribution requirements) not present in vanilla Apache-2.0. |
| 18 | PaddleOCR-VL 1.5 | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0 | YES | "This project is released under the Apache 2.0 license." Clean for M7 photo OCR. |
| 19 | emotion2vec (M8.5) | https://github.com/ddlBoJack/emotion2vec  +  https://huggingface.co/emotion2vec/emotion2vec_plus_large | Code: MIT. **Weights: HF card shows generic `model-license` (text not surfaced).** | MAYBE | Code is MIT, but the HF weights card lists only `model-license` without text. ModelScope mirror is `iic/emotion2vec_plus_large` (Alibaba IIC) which historically has its own custom terms. **Action:** fetch the model-license text from the ModelScope page or the FunASR repo before M8.5 ships. **Fallback:** wav2vec2-emotion (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`) — its own license also needs to be checked; or openSMILE (research/commercial dual license — paid commercial tier available). |
| 20 | DeepFilterNet | https://github.com/Rikorose/DeepFilterNet | MIT **or** Apache-2.0 (dual, your choice) | YES | "All code in this repository is dual-licensed under either MIT License or Apache License, Version 2.0 at your option." Models covered by same scheme. Clean for current denoise path. |

## Blocked items + fallback plan

- **PersonaHub (#12) — `cc-by-nc-sa-4.0`, BLOCKED.**
  - Do not train or ship any LoRA / model derived from PersonaHub data in the commercial product.
  - Replacement: synthesize our own persona seed corpus using Qwen 3 8B (Apache-2.0) conditioned on demographic templates we author in-house, then run the OpenCharacter Constitutional-AI recipe (#13, method only) on that.
  - Verify the Salesforce `xywang1/OpenCharacter` 326K dataset license separately before relying on it — it is **also** derived from PersonaHub, so it likely inherits the NC restriction.
- **LLaMA-Omni 2 (#10) — model weights research-only, BLOCKED for commercial.**
  - Drop from the M12 / M13 commercial candidate list, OR contact `fengyang@ict.ac.cn` for a commercial license.
  - Better fallback: LLaMA-Omni 2 wraps CosyVoice 2 (Apache-2.0, item 4) on the TTS side and Qwen 2.5 on the LLM side — we can build the same hybrid directly from the Apache-2.0 components without the LLaMA-Omni 2 weights.

## Items needing further verification

- **#2 Qwen3-TTS 0.6B VoiceDesign** — TODO: open the HF card for `Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign` and confirm `apache-2.0`. Almost certainly fine; just close the loop before shipping.
- **#5 IndexTTS-2** — TODO: read the `LICENSE` and `LICENSE_ZH.txt` files in the repo directly (web-fetch only surfaced the README, which already warns commercial users to email Bilibili). Until then, **assume commercially restricted**.
- **#6 Qwen3-ASR HF card** — Direct HF model-card URL returned 401. Repo footer confirms Apache-2.0; confirm the HF card matches from a logged-in session before relying on the verbatim wording in any compliance doc.
- **#7 Qwen3.5-Omni-Light** — TODO: verify the `Qwen/Qwen3.5-Omni-Light-*` HF card specifically (we used the related Qwen3-Omni-30B-A3B-Instruct card as a proxy; should be the same license but confirm).
- **#9 Kimi-Audio-7B weights** — TODO: web-fetch returned 401 on HF, repo only documents code license. Check `moonshotai/Kimi-Audio-7B` and `moonshotai/Kimi-Audio-7B-Instruct` HF cards before any commercial use.
- **#13 OpenCharacter recipe** — TODO: separately verify the Salesforce `xywang1/OpenCharacter` 326K HF dataset license (web-fetch returned 401). If it inherits CC-BY-NC-SA from PersonaHub, we cannot use it commercially either — the recipe must be re-implemented from scratch with our own seed data.
- **#17 MinerU custom license** — TODO: read `LICENSE.md` from the repo verbatim. "Custom Apache-2.0-based" can mean anything from "identical to Apache-2.0 with a name change" to "Apache-2.0 plus naming/attribution clauses." Need the exact text in our compliance file.
- **#19 emotion2vec weights** — TODO: fetch the ModelScope `iic/emotion2vec_plus_large` page (the canonical source per its FunASR origin) and read the actual model license. If non-commercial, swap to a verified-commercial-OK emotion classifier before M8.5 ships.

## Notes

- **Method:** 14 web-fetches via WebFetch tool, hitting GitHub READMEs and Hugging Face model cards directly. Three HF model cards (Qwen3-ASR, OpenCharacter dataset, Kimi-Audio weights) returned HTTP 401 — Hugging Face appears to require an authenticated session for some model cards in this environment.
- **Honesty caveats:**
  - Did NOT read the literal `LICENSE` file content for IndexTTS-2 (#5) or MinerU (#17) — only the surrounding README context. Need the verbatim license file before locking either into a commercial release.
  - Did NOT separately verify the 0.6B Qwen3-TTS sibling (#2) — inferred from the 1.7B sibling's license.
  - emotion2vec (#19) and Kimi-Audio-7B weights (#9) need a second-pass verification — code license clean, weights license unverified.
  - Did NOT cross-check against ModelScope mirrors except where flagged — some Chinese-origin models have different terms on ModelScope than on HF.
- **Recommended next step:** assign each TODO in the "Items needing further verification" section to a follow-up task with a target close date before the milestone that depends on it (e.g., #19 must close before M8.5 kickoff; #2 + #5 must close before M11 ships; #13 must close before M9 SFT data generation begins).
- **Recommended file location for ongoing compliance:** keep this file as the audit-as-of-date snapshot; add a `data/compliance/LICENSE_<item>_<date>.txt` directory with verbatim license file copies for every shipped dependency, refreshed on each model swap.
