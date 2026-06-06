# Resume Material — EverHome (for Indeed Senior MLE application)

**Written 2026-06-06.** Indeed uses an AI screener; this doc is structured
so both the ATS and the human reviewer get what they need.

Three layers:
- **§1 Resume bullets** — 5-7 lines that fit on a CV under "Independent
  Project" or "Founder, EverHome".
- **§2 Cover-letter narrative** — 2 paragraphs the human reads after the
  ATS pass.
- **§3 ATS keyword surface** — raw keyword list to make sure the
  parsing layer catches all relevant signals.
- **§4 Interview talking points** — 10 specific stories with quantified
  outcomes.

---

## §1 Resume bullets

> **EverHome — Founder & Sole Engineer** · 2026-03-present · Solo project
> Privacy-first family voice AI legacy preservation system. Streaming ASR
> → LLM → emotion-conditioned TTS, fine-tuned for close-relative voice
> recognition. Live at everhome.mkk.dev.

Bullet points (pick 5-7 depending on resume length):

- **Designed and shipped a real-time multilingual voice AI pipeline**
  (VAD → ASR → LLM streaming → emotion-conditioned TTS) achieving
  end-to-end latency of 1-2 seconds; ~33K lines of Python across
  75 modules, 660 tests (contract + unit + integration + property tests).

- **Fine-tuned Qwen3-TTS 1.7B per-person voice cloning via SFT/LoRA**
  on 30-60 minutes of speaker audio; designed and operationalized the
  evaluation rubric for close-relative voice recognition (5-dimension
  × 5-scale blind A/B), since no published benchmark existed for this
  family-recognition task.

- **Shipped at NY Tech Week 2026** to a live audience of ~80 (8-min demo
  including persona/listener routing reveal and ethical-disclosure
  moment); debugged and fixed three live regressions including a
  Qwen3-ASR silent-buffer hallucination ("The first was the first to be
  built." on near-silent audio) using a peak-amplitude discriminator
  pipeline (peak < 0.12 → drop), backed by 31 unit tests.

- **Built end-to-end training infrastructure**: speaker diarization,
  DeepFilterNet denoise + LUFS normalization, SFT/LoRA training jobs
  with progress streaming, merged-model packaging, per-persona
  versioning, hot-swap inference activation — all wired into a FastAPI
  + WebSocket + Jinja2/JS frontend on a single A10G GPU box.

- **Designed a regulatory-compliance gating layer** ahead of B2B sale:
  consent records with per-purpose scope + jurisdiction tagging (NO FAKES
  Act, EU AI Act, CA AB 1836/2602), revocation with audit-trail
  tombstones (no hard-delete), and server-side ingest gates with
  TOCTTOU-narrowed double-check (25 contract tests).

- **Drove the project's architectural direction via dual-LLM code
  review**: ran every major commit through both GPT-5 (sharp tactical
  review) and Gemini 2.5 Pro (strategic + 1M-context plan review),
  caught real bugs (audio-peak gating regression, TOCTTOU race) before
  shipping, integrated five strategic deltas into a published roadmap.

- **Identified per-(speaker, listener) LoRA composition as a
  publication-grade research contribution** (rare in TTS literature
  since most academic datasets lack listener relationship labels);
  designed an experiment plan targeting ICASSP 2027 or Interspeech 2027
  with synthetic eval set release + reproduction kit, while keeping
  per-family training data on-box.

---

## §2 Cover-letter narrative

Two paragraphs. The first sells the work; the second connects to Indeed.

**Paragraph 1 — what was built:**

> Over the last quarter I designed, built, and shipped EverHome — a
> privacy-first family voice AI that preserves a person's voice,
> speech patterns, and stories so loved ones can keep talking with
> them. The full pipeline (VAD → Qwen3-ASR → OpenAI streaming LLM →
> emotion-conditioned Qwen3-TTS fine-tuned via SFT/LoRA) runs at ~1-2s
> end-to-end latency on a single A10G GPU, with a Cloudflare-named-
> tunnel public surface. I built it as a solo founder across ~33K lines
> of Python, 660 tests, and 234 commits, and shipped it live at NY Tech
> Week 2026 to ~80 attendees including the persona/listener routing
> reveal and the ethical-disclosure moment. After the demo I ran the
> full project plan through both GPT-5 and Gemini 2.5 Pro as external
> code/strategy reviewers; the dual-LLM review surfaced architectural
> deltas (M9 split for faster privacy moat closure, M7 split for
> earlier M8 unblock, M12a evaluation pulled forward) that I integrated
> into a published roadmap, and caught two real bugs (audio-peak gating
> regression and TOCTTOU race) before they shipped.

**Paragraph 2 — why Indeed:**

> Indeed's scale of multilingual job-seeker conversation — voice search,
> resume-to-job matching, AI-driven applicant screening — is exactly
> where this kind of streaming-multilingual-with-fine-tune-depth
> infrastructure matters. The work I'm proudest of on EverHome maps
> directly onto Indeed's hard problems: building fine-tune pipelines
> where evaluation is messy (no published benchmark for
> close-relative recognition; I had to design the rubric); shipping
> regulatory-compliance infrastructure (consent records, revocation
> tombstones, jurisdiction-aware purpose scope under NO FAKES Act / EU
> AI Act / CA AB 1836) that I expect any responsible ML product at
> Indeed's scale to need; and treating LLM hallucination as a real
> production failure mode that needs mechanical defenses (the
> peak-amplitude silence guard caught Qwen3-ASR hallucinations on
> demo day in under an hour because the right test rubric was already
> in place). I'd be excited to bring that ship-to-production cadence
> and that defensive ML engineering posture to Indeed's senior MLE team.

---

## §3 ATS keyword surface (for the auto-screener)

Make sure these terms appear naturally in the bullets above. Indeed's
ATS likely pattern-matches on:

**ML / model:** fine-tuning, LoRA, SFT, supervised fine-tuning, transfer
learning, model deployment, model evaluation, A/B testing, blind eval,
human evaluation, multilingual, Chinese, English, code-switching,
speaker adaptation, voice cloning, TTS, ASR, speech recognition,
text-to-speech, voice activity detection (VAD), LLM, large language
model, persona, prompt engineering, RAG, retrieval-augmented
generation, vector store, embeddings, BGE-M3, instruction-conditioning,
streaming inference, low-latency inference

**Models / libraries:** Qwen3-TTS, Qwen3-ASR, Qwen 3 8B, OpenAI GPT-5,
GPT-4o, Gemini 2.5 Pro, Whisper, BGE-M3, LanceDB, MinerU, PaddleOCR-VL,
CosyVoice, VoxCPM, Step-Audio, HippoRAG, OpenCharacter

**Infra:** FastAPI, WebSocket, async Python, asyncio, CUDA, PyTorch,
transformers, vLLM, Cloudflare tunnel, AWS A10G g5.4xlarge, NVIDIA DGX
Spark, Linux, Docker (planned), Hugging Face, ModelScope, Prometheus,
contract tests, integration tests, property tests, pytest

**Concepts:** real-time, streaming pipeline, end-to-end latency,
hallucination detection, regulatory compliance, NO FAKES Act, EU AI
Act, CA AB 1836, consent management, revocation, audit trail, TOCTTOU,
race conditions, atomic writes, POSIX flock, JSON-backed repository,
file-system locking, on-device inference, on-prem deployment, privacy
by design, data sovereignty, federated learning (planned),
constitutional AI, RLHF, persona LoRA

**Architecture:** FSM (finite state machine), per-utterance state,
session management, emotion tag parsing, instruction-conditioned TTS,
listener-conditioned voice routing, dual-index RAG, identity knowledge
graph, conversation memory, SISA training (for unlearning)

**Process / soft:** founder, sole engineer, shipped to production, live
demo, multi-LLM code review, GPT-5 review, Gemini review, dual-LLM
critique loop, A/B human evaluation methodology, regulatory
compliance design, ethical AI, voice ethics, family preservation

---

## §4 Interview talking points (10 stories with metrics)

When asked "tell me about a hard problem you solved", pick from these:

### 1. "The hallucination filter that caught itself"
Qwen3-ASR returned a deterministic English sentence ("The first was the
first to be built.") on silent audio. First attempt to fix was a
buffer-mean RMS threshold; that dropped real speech on phones (mobile
AGC pushes mean RMS below the threshold). Pivoted to a peak-amplitude
discriminator (peak < 0.12 = silence/noise) which cleanly separated
real speech (peak 0.5+) from noise (peak < 0.1) regardless of buffer
duration or AGC. Then GPT-5 review caught a SECOND bug: my filter
also dropped loud "Okay" / "Thank you" because the known-phrase set
ran regardless of peak. Tightened the gate. 31 unit tests now lock the
behavior including the "loud-Okay" case.

### 2. "Two LLM reviews are worth one human reviewer"
After implementing M-Consent, ran the commit through both GPT-5 and
Gemini 2.5 Pro. GPT-5 caught the TOCTTOU race window between consent
check and file write (could allow data ingest seconds after consent
revocation). Gemini caught implicit dependency on repository sort order
+ silent skip on missing-record files (fail-silent vs fail-loud
philosophy). Neither model alone caught both. Fixed all three before
shipping. The dual-LLM review process is itself the methodological
contribution.

### 3. "When the audio worklet API breaks tomorrow"
ScriptProcessorNode (the browser's PCM-capture API) was deprecated by
MDN in 2020. Mobile Safari and WeChat in-app browser are most likely
to drop it first. The migration to AudioWorkletProcessor +
SharedArrayBuffer is 1-2 days of work BUT introduces real demo risk.
Captured the deferral decision in RFC §11.2 with explicit revisit
triggers — when the trade-off changes (real Safari user case, real
live-capture path, MDN moves to "removed"), pull the lever. The
discipline of declaring "we're not doing this now and here are the
specific conditions under which we WILL" is the engineering judgment
worth talking about.

### 4. "The corpus IS the moat"
Original framing was "voice cloning depth is the moat." After Gemini
review, reframed: voice cloning depth is NECESSARY but not SUFFICIENT.
The durable hard-to-copy asset is the per-family corpus + memory graph
+ persona model. A competitor with worse cloning but better
ingest/memory tooling wins on family adoption. This re-positioning
changed the sequencing of M7/M8/M9 — they're now co-equal moat
investments, not plumbing for the voice layer. Stake out your
positioning early, but be willing to be wrong loudly.

### 5. "Designing the evaluation BEFORE the method"
For per-listener voice cloning (M10), the planned ICASSP paper
contribution is not just the method (per-(speaker, listener) LoRA
composition) but the EVALUATION rubric itself. No published benchmark
exists for close-relative voice recognition. Operationalizing the
rubric — 5 sentences × 5 dimensions × 5 scale × family-blind A/B —
is its own contribution. The lesson: when there's no benchmark in your
problem domain, designing one IS the work.

### 6. "License audit as an existential blocker"
The roadmap had M9 (persona LoRA via OpenCharacter recipe) using
PersonaHub training data. Discovered the dataset is `cc-by-nc-sa-4.0`
— commercially blocked. Fallback: synthesize seed data in-house using
Qwen 3 8B (Apache-2.0) + apply OpenCharacter recipe as method-only,
log generator-model lineage so future audit can prove every SFT pair
traces to a commercial-clean input. Same exercise caught LLaMA-Omni 2
(research-only weights). The license audit shipped BEFORE the M7/M9
code; finding this after building would have meant a corpus + model
rebuild. Lesson: legal posture is part of model architecture, not a
post-build sweep.

### 7. "Why I treat tests like a UI"
Built 660 tests across contract / unit / integration / property layers.
Most useful pattern: contract tests as the API surface lock — the
test file describes what the endpoint must guarantee and breaks any
refactor that violates that guarantee. The emotion parser shipped
WITH property tests before integration (Hypothesis-style — partial
tags, streaming chunks, idempotence, no-spurious-output invariants).
For Indeed-scale ML: contract tests become the integration boundary
between ML team and product team.

### 8. "The persona dropdown that wouldn't load Mark"
Demo day, ~3 hours before stage time, persona dropdown defaulted to
xiao_s instead of the EverHome Demo persona. Diagnosed: HTML default +
JS fallback both preferred xiao_s. Three-line fix. But the SUBSEQUENT
test: Cloudflare's edge cache held standalone.js for 4 hours
(max-age=14400) — my no-cache middleware was overridden at the edge.
Browser still ran old JS. Fixed with `?v={mtime}` cache buster on
script tag. GPT-5 review later flagged that the buster only watched
JS, not CSS — extended to all 4 served assets with MAX-mtime
discriminator. Iterating under demo-day pressure builds the muscle.

### 9. "Building for revocation, not just for capture"
M-Consent.1 stores consent records as JSON files with per-persona
indexes, POSIX flock concurrency control, atomic-rename writes. But
the key design decision was: DELETE doesn't delete. It revokes with a
tombstone. The original consent block stays so the audit trail shows
what was authorized + when + by whom + why it was revoked. This is the
shape regulators (NO FAKES Act, EU AI Act, CA AB 1836) want — provable
chain of custody, not just clean state. Built the data model around
the regulatory contract, not the happy path.

### 10. "Multilingual is data architecture, not just tokenizer choice"
TTS LoRA trained on Chinese audio produces English with Chinese
prosody (fast pace, flat intonation). Voice timbre is preserved,
"flow" is not. The fix is NOT a fancier model — it's adding English
audio to the training corpus. But the deeper lesson: the choice to
go Chinese-native at every layer (Qwen LLM, BGE-M3 embeddings,
Qwen3-ASR/TTS) was deliberate at the architecture level, not the
component level. Indeed's multilingual job-search has the same
geometry: the data + model + UX choices need to align top-to-bottom,
not just at the model layer.

---

## §5 What NOT to claim

Stay honest. The product is:

- ✅ Real, shipped, live at everhome.mkk.dev
- ✅ Used by ~80 demo attendees + ongoing dev usage
- ❌ NOT used by paying customers yet (don't claim user count beyond demo)
- ❌ NOT yet 100% local (LLM is currently cloud OpenAI; M9 fixes this)
- ❌ NOT yet shipping cross-session memory (M8 designed only)
- ❌ NOT yet published research (M10 paper plan exists, no submission yet)

The story is "I built and shipped the core; the rest of the roadmap is
designed and committed to." That's a stronger story than overclaiming.
