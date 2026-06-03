# M10 — Research Plan: Listener-Adaptive Voice Cloning

**Status.** Plan only (2026-06-03). M10 implementation has not started.

**Premise.** Most of the EverHome roadmap implements published methods
(ID-RAG, HippoRAG 2, OpenCharacter, MinerU). **M10 is the milestone
where EverHome has structurally unusual training data + a novel method
worth publishing.** Build it publication-ready from day one.

---

## 1. The research question

In TTS literature, voice identity is the dominant axis (per-speaker
LoRA / SFT) and emotion is the secondary axis (instruct-conditioned
prosody, emotion2vec). **"Who is the speaker talking TO" — the listener
relationship — is rarely a first-class axis.**

> Can a single speaker's voice be cloned such that the SAME persona,
> speaking through a learned per-relationship adapter, sounds
> recognizably different (and appropriately so) when addressing a
> child vs a colleague vs an elder — while preserving speaker
> identity beneath the relationship-conditioned surface?

If yes, **how do you compose per-relationship LoRA adapters at inference
time** so a family-voice AI can switch relationships per conversational
turn?

---

## 2. The novel contribution claim

Three claims, in priority order:

1. **Method:** per-(speaker, listener) LoRA composition for TTS, with
   inference-time relationship switching that preserves speaker
   identity. Either via independent LoRA-per-listener or via
   compositional LoRA fusion (e.g., LoRA-Hub-style summation,
   orthogonal LoRA composition per OPLoRA 2510.13003).
2. **Evaluation methodology:** the close-relative-recognition human
   eval rubric, formalized. No published benchmark currently exists
   for "does it sound like Mark to Mark's mother", per
   `RESEARCH_SFT_S2S.md`. Operationalizing this is itself a
   contribution.
3. **Data position paper:** describe the data structure (recordings
   labeled by listener relationship) without releasing the data
   itself — release synthetic counterparts + the labeling protocol.

---

## 3. Why we are uniquely positioned to publish this

- **Most academic TTS datasets lack listener labels.** LibriTTS,
  VCTK, AISHELL, etc. record speakers reading scripts in isolation.
  EverHome's recording pipeline structurally captures the same speaker
  addressing different listeners (per `app/api/recordings_ui.py`
  prompts and `app/api/listeners.py` taxonomy).
- **Family-scale data, not laboratory data.** Studio TTS data has
  controlled prosody. Real family data has the genuine variation
  ("how grandma actually talks to her grandchild" vs "how grandma
  talks to her doctor") that this work studies.
- **Solo-founder advantage:** we can ship this in production and
  evaluate against real users without IRB friction (the families are
  the customers, evaluating their own family member). Academic groups
  cannot get this data; corporate groups cannot share it.

---

## 4. Related work survey (must read before drafting)

- **Persona-aware TTS:** various, including Persona-TTS (~2024).
  Closest related but treats persona as global, not as per-listener
  adaptation.
- **Code-switching TTS:** for bilingual speakers. Method overlaps with
  listener-conditioned adapter composition. Cite as related.
- **Dialog-adaptive prosody:** prior work on conversational turn-taking
  and prosody adaptation. Listener-relationship is a stronger signal
  than turn position.
- **LoRA composition / fusion:** LoRA-Hub (NeurIPS 2023 workshop),
  OPLoRA (arXiv:2510.13003), MoE-LoRA. Provide the composition
  primitives.
- **Multi-style / multi-emotion TTS via LoRA:** CosyVoice 2's
  instruction-conditioning, Style2Talker. These study STYLE per
  speaker; listener is a different conditioning axis.
- **Voice-cloning ethics + family contexts:** arXiv:2511.20094
  "Digital Ghosts" — frame our work as ethically scoped, not
  surveillance / impersonation.

**Action.** Survey ≥20 papers from Interspeech / ICASSP 2024-2026 on
"speaker adaptation + conditioning" before any method experiments. The
positioning depends on this.

---

## 5. Method (sketch)

Three variants to ablate; pick whichever wins on the human eval.

### Variant 5.1 — Stacked LoRA per (speaker, listener) pair

Train a separate LoRA per listener WITHIN each speaker's training
data. Per persona = 1 base TTS + N LoRAs (one per listener type).
Switch adapter at inference. **Pro:** simple. **Con:** O(N) adapters
per persona; cold-start problem for new listener types.

### Variant 5.2 — Compositional LoRA: speaker × listener as two heads

Train ONE speaker-LoRA + N listener-LoRAs (shared across speakers).
At inference, compose them — additive fusion, gated mixture, or
orthogonal merge per OPLoRA. **Pro:** new listener types transfer
across personas; fewer adapters total. **Con:** trickier training;
risk of cross-interference.

### Variant 5.3 — Listener as an instruction-conditioning signal (no per-listener LoRA)

Pass listener-relationship as a text instruct to an
instruction-conditioned TTS (per M8.5). **Pro:** zero new training
machinery on top of M8.5. **Con:** likely weakest fidelity — instruct
control is coarser than learned adapters. **Use as the baseline that
LoRA variants must beat.**

---

## 6. Dataset (the experiment plan, not the data)

### Per persona, per listener, target:
- ≥ 30 minutes of recordings, listener-labeled at intake
- Same persona speaks all 5 listener types: child, mom, friend,
  reporter, elder, default
- Total per persona: ~3 hours

### Released artifacts (not the family data itself)
- **Listener-labeled prompt scheme** — the exact prompts we use in
  `app/api/recordings_ui.py` to elicit per-listener speech (so other
  groups can reproduce the data structure on their own speakers).
- **Synthetic eval set** — 100-200 recordings of consenting volunteers
  speaking to 5 listener types, released under CC-BY for benchmark
  reuse. Curated to mirror the real distribution.
- **Public reproduction kit** — eval scripts + scoring rubric + a
  small held-out set.

### Privacy + ethics ratchet
- Real family data NEVER leaves the EverHome appliance.
- Paper claims are validated against the synthetic eval set.
- Anonymized aggregate statistics from real-family human eval may be
  reported only with per-family written consent.

---

## 7. Evaluation methodology (the contribution-as-protocol)

For each (speaker, listener) pair:

| Test | Metric | Who scores |
|---|---|---|
| Close-relative recognition A/B | Win-rate vs ablations | Family members of the speaker, blind |
| Listener-appropriate-tone | Likert 1-5 | Family members, blind to method |
| Speaker identity preservation across listeners | Speaker-verification model embedding similarity (SECS, FSD) — should be HIGH across listeners (same speaker) | Automated |
| Listener-conditioned prosody differentiation | Prosody-embedding distance — should be MEASURABLE between listeners | Automated |

The combination of (a) close-relative blind A/B with real family,
plus (b) automated speaker-identity + prosody-differentiation checks,
is itself the proposed evaluation framework.

---

## 8. Ablations table to fill at build time

| Variant | Dataset hours/listener | Avg. close-rel A/B win-rate vs baseline | SECS speaker-similarity | Listener-prosody distinguishability |
|---|---|---|---|---|
| 5.3 instruct-only (baseline) | — | 50% | ? | ? |
| 5.1 stacked-LoRA | 30 min | ? | ? | ? |
| 5.1 stacked-LoRA | 60 min | ? | ? | ? |
| 5.2 compositional | 30 min × N | ? | ? | ? |
| 5.2 compositional | 60 min × N | ? | ? | ? |

Win condition for publication: variant 5.1 or 5.2 must (a) beat 5.3
by ≥ +15% on close-rel A/B win-rate (b) preserve SECS ≥ 0.85
(speaker still recognizable), (c) listener-prosody distinguishability
≥ 0.6 (the listener axis is real).

---

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Real family data can't be shared → reviewer can't reproduce | Released synthetic eval set + reproduction kit. Real-family results are an *upper bound* claim, not the headline. |
| Per-listener prosody distinguishability is too small to measure | Pre-experiment on synthetic data; if effect is < 0.3, drop the per-listener LoRA claim and reposition as persona-aware TTS. |
| Solo founder / no institutional affiliation → reviewer bias | Submit with `Mark Ko, Independent Researcher` byline. Mention an industry contributor if appropriate. Cite EverHome as the production deployment, not as commercial promotion. |
| Method becomes public → competitor copies | Per RFC §11.12 the durable moat is the per-family corpus, not the method. The paper IS the contribution. |

---

## 10. Authorship + venues

- **Byline.** `Mark Ko, Independent Researcher` (or `Mark Ko, EverHome`).
- **Target venues** (ranked):
  1. **ICASSP 2027** — deadline ~Sept 2026. Best fit if M10 ships by
     August 2026.
  2. **Interspeech 2027** — deadline ~March 2027. Backup; more
     speech-community-friendly but lower visibility outside.
  3. **NeurIPS 2027 workshop** (Personalization, Adaptation, Speech) — fallback if the main-track deadline misses.
  4. **arXiv-only preprint + Hugging Face demo** — always do this in
     parallel for visibility regardless of conference acceptance.

---

## 11. Implementation order (when M10 starts)

1. **Week 1:** related-work survey (≥ 20 papers); finalize the
   evaluation rubric; pre-register the experiment design.
2. **Week 2-3:** dataset curation + listener labeling on existing
   EverHome recordings. Spec out the synthetic eval set release.
3. **Week 4-5:** implement variant 5.1 (stacked-LoRA, simplest);
   train + run automated evals.
4. **Week 6-7:** implement variant 5.2 (compositional); A/B against
   5.1 on automated metrics.
5. **Week 8:** human eval (family-recognition A/B); finalize ablation
   table.
6. **Week 9:** writeup, anonymize, submit.
7. **Always:** ship the working version into the EverHome product so
   real-family data continuously validates the method.

---

## 12. Open questions to revisit before kickoff

- Is the listener taxonomy in `app/api/listeners.py` the right shape
  for the paper, or should we use a more academic categorization
  (e.g., social-distance axis: intimate / personal / social /
  public)?
- Do we need IRB / ethics review even as a solo researcher? Probably
  not for production-deployment data + voluntary synthetic eval, but
  cite the SPSP-style consent framework anyway.
- Should the paper be co-authored with a clinical psychologist (for
  the family-recognition methodology section)? Probably yes — get an
  industry contributor for credibility. Cost: 1 co-author email.

---

## 13. Linked docs

- `docs/PROJECT_BRIEF.md` §1 (moat framing)
- `RFC_M6_PERSONA_LLM_LEGACY.md` §11.13 (publication-opportunity decision)
- `docs/REVIEW_GEMINI25PRO_2026-06-03.md` (the strategic framing that
  prompted treating M10 as research)
- `docs/RESEARCH_SFT_S2S.md` (close-relative-recognition gap in the
  literature)
