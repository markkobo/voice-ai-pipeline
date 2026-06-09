# Why V12 doesn't sound like you — and what to do about it

**Date:** 2026-06-09. **Trigger:** User asked to compare our SFT recipe
against [Qwen3-TTS-EasyFinetuning](https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning)
because V12 has lingering 大陸腔 and doesn't fully match the speaker's voice.

**Bottom line:** **our training loop only touches `talker.code_predictor`
(the codec quantizer head). It NEVER trains the full talker LM that
produces prosody and accent.** The 大陸腔 is the frozen talker's default
Beijing prior leaking through, because no gradient ever reached the
weights that would teach it your accent. The external repo trains the
full talker on `(text, audio_codes)` pairs with ASR transcripts — that's
the missing piece.

The cheap "patch spk_is_dialect" fix does NOT apply — V12 already has
`spk_is_dialect.test = False`. The problem is in HOW we train, not WHAT
we set at inference.

---

## Side-by-side comparison

| Aspect | EverHome (V12) | Qwen3-TTS-EasyFinetuning | Material? |
|---|---|---|---|
| Base model | Qwen3-TTS-12Hz-1.7B | Same (also 0.6B option) | No |
| Method | SFT or LoRA on `talker.code_predictor` only | Full SFT on full talker LM | **YES** |
| Training signal | **Audio-only.** `forward_sub_talker_finetune(codec_ids, talker_hidden=cached_emb)` runs ONE frame at a time through code_predictor only | **Text + audio.** `talker(inputs_embeds=text+codec, labels=codec_0_labels)` + `0.3 × sub_talker_loss`. Full LM forward pass per step | **YES — primary cause** |
| Per-sample input | One codec frame + same static speaker embedding for every chunk | Real `(text, audio_codes)` pairs from JSONL with ASR transcripts | YES |
| Data pipeline | Whole WAV → 24kHz resample → peak normalize → chunk to 300-frame / 150-hop overlapping windows of codec tokens | `pydub.silence.split_on_silence` → 1-15s chunks → **Qwen3-ASR-1.7B transcribes each chunk** → JSONL `{audio, text, ref_audio}` | YES |
| Speaker embedding | Baked at merge into `talker.model.codec_embedding.weight[spk_id]`. Training never saw that slot. | Bake at position `3000 + idx*20` PLUS inject at `input_codec_embedding[:, 6, :]` every forward step → talker learns to read that channel | YES |
| `spk_is_dialect` | UI-selectable. **V12 has `False` for `test`** (verified 2026-06-09) | Hardcoded `False` | Same when ours is correct |
| Inference `language=` | `"auto"` (engine default at qwen_tts_engine.py:406) | CLI default `"English"`, passes through | Ours is correct |
| Hyperparameters | SFT epoch=30 default, batch=1, LR=1e-5 | SFT epoch=2, batch=2, grad_accum=4, **LR=1e-7**, FA2 | Theirs more conservative |

---

## Why this explains both symptoms

**Timbre is partially captured** — the baked `speaker_embedding` in the
merged custom_voice config adds a row to `codec_embedding.weight` that
gives the codec quantizer a target color for your voice. That's why V12
sounds *somewhat* like you.

**Prosody / accent is NOT captured** — accent lives in the talker LM (a
language model that produces codec sequences conditioned on text). We
never train that. The talker carries the model's pre-training prior,
which is Mandarin-heavy with Mainland defaults. When the talker
produces a codec stream for "你看", the prosody and 兒化/捲舌 patterns
come from THAT prior, not from your training data.

**Codec_predictor learns nothing useful in isolation** — the
`forward_sub_talker_finetune` path feeds the SAME static speaker
embedding for every chunk plus one codec frame at a time. With only a
codec-quantizer-level loss and no text context, the only thing it can
learn is a frame-level reconstruction nudge — which doesn't transfer to
inference where the talker generates whole sequences autoregressively.

---

## What does NOT help

- **Patching `spk_is_dialect`** — already `False` for `test` in V12.
- **More epochs of the current recipe** — wrong loss surface; more steps
  won't generate gradient signal for the talker.
- **Lower learning rate** — same reason.
- **Adding `language="auto"` at inference** — already in place.
- **Different reference audio at speaker_embedding bake** — marginal at
  best; the bake adds one row, accent is dominated by the LM weights
  the bake didn't touch.

---

## What WILL help — the V13 recipe

Match the external repo's training shape. This is ~3-5 days of
implementation work, not a config flip.

### Step 1: Build an ASR-transcribed dataset

For each persona's existing recordings under
`data/recordings/denoised/`:

1. Split each WAV into 1-15s chunks via
   `pydub.silence.split_on_silence(min_silence_len=500,
   silence_thresh=dBFS-16, keep_silence=200)`. This is what their
   `step1_audio_split.py` does.
2. Run **Qwen3-ASR-1.7B** (we already have this loaded) on each chunk
   to produce a Traditional Chinese transcript.
3. Write JSONL:
   ```json
   {"audio": "rel/path/chunk_001.wav", "text": "你看", "speaker_id": "test"}
   ```
4. Persist at `data/training/test/v13_dataset/train.jsonl`.

### Step 2: Implement text-conditioned training loop

New training type, e.g. `training_type="sft_text"`, in
`app/services/training_service/training_job.py`:

- Load `(text, audio_codes)` pairs from the JSONL.
- Forward: `unwrap_model.talker(inputs_embeds=text_emb + codec_emb,
  labels=codec_0_labels[:, 1:])` — full talker autoregressive loss.
- Add the auxiliary loss: `+ 0.3 * forward_sub_talker_finetune(...)` —
  keeps the codec-predictor signal we already have.
- Inject speaker embedding at `input_codec_embedding[:, 6, :]` on every
  forward (the external repo's load-bearing detail — talker learns to
  condition on the embedding during training so the bake at inference
  isn't a fresh signal it's never seen).
- AdamW, **LR=1e-7** (not 1e-5), epochs=3, batch=2, grad_accum=4, Flash
  Attention 2 enabled. Their hyperparameters are conservative because
  the gradients on full-talker SFT are larger than on code_predictor-
  only.

### Step 3: Train V13 on identical audio as V12

Same source recordings, same DeepFilterNet denoise + LUFS normalize.
The only delta vs V12 is the training loop. This makes V13 the cleanest
A/B against V12.

Expected training time on A10G: ~20-40 min for 3 epochs (vs V12's
~30-60 min for 30 epochs).

### Step 4: Blind A/B V13 vs V12

10 prompts (5 with Taiwan-specific vocabulary, 5 generic). Blind triad:
"which sounds more like Mark?" Score over the 10 prompts.

**Success criterion:** V13 wins ≥ 7/10 over V12, AND specifically wins
on the Taiwan-vocab prompts ≥ 4/5. If V13 wins on Taiwan-vocab but not
overall, accent fix worked but identity is still weak — investigate
speaker_encoder next. If V13 doesn't beat V12 at all, the talker-loss
hypothesis is wrong and we'd look at the speaker-embedding injection
position.

---

## Risks of moving to the V13 recipe

| Risk | Mitigation |
|---|---|
| Full-talker SFT can degrade base model on prompts outside the persona's data (catastrophic forgetting) | LR=1e-7 + only 3 epochs is the external repo's choice for exactly this reason. Keep V12 as fallback. |
| ASR transcripts contain errors that mislead training | Spot-check 20 random chunks; remove any with WER > 30% |
| Per-chunk speaker_embedding injection at position 6 is undocumented behavior — could change in future Qwen3-TTS releases | Pin the Qwen3-TTS model version we use; document this position as a known dependency |
| Training time + compute | ~30-40 min per persona. Acceptable; the bottleneck is dataset prep + ASR pass which is one-time |

---

## Confidence

**Diagnosis confidence: HIGH** — the difference between training
code_predictor-only vs training the full talker is precisely the
mechanism that controls accent and prosody. The symptom (timbre
partially OK, accent stuck at Beijing default) maps directly onto
"speaker_embedding bake works, talker LM unchanged."

**Fix confidence: MEDIUM-HIGH** — the external repo's recipe is the
direct counterfactual; it should work. The only uncertainty is the
specific embedding-injection position (their position 6 might be model-
version-specific). If V13 doesn't beat V12, the talker-loss hypothesis
is still correct but our implementation of the injection needs adjustment.

---

## What to do TODAY

1. Read this doc. Decide whether to invest 3-5 days in the V13 recipe.
2. If yes, allocate the audio-recording session — the existing V12
   training data is sufficient; we just need it segmented + ASR'd.
3. If no, document the limitation in `RFC_M6_PERSONA_LLM_LEGACY.md` and
   accept that V12-class fine-tunes will sound somewhat like the
   speaker but with a residual Mandarin default accent.

---

## Gemini 2.5 Pro review revisions (2026-06-09)

Full review at `docs/REVIEW_V13_PLAN_GEMINI_2026-06-09.md`. Diagnosis
confirmed correct (full talker is where accent lives, frozen talker
leaks Mainland prior). Five plan changes BEFORE we touch implementation:

### Change 1 — try LoRA on talker FIRST, full SFT only as fallback

Full-talker SFT is "the sledgehammer" — most likely path to catastrophic
forgetting. Try LoRA on the talker's attention + FFN layers first
(mirrors our successful M9 LoRA approach for persona LLMs). Only escalate
to full SFT if LoRA's prosody/accent capacity is insufficient.

### Change 2 — DO NOT use Qwen3-ASR for transcripts

Circular bias: Qwen3-ASR has its own Mainland accent priors. Transcribing
our Taiwan-accented speaker with Qwen3-ASR risks getting transcripts
where Taiwan vocabulary has been "normalized" to Mainland equivalents.
Training on those transcripts would reinforce the exact problem we're
trying to fix.

**Use human-verified transcripts.** The cheapest path: hand-transcribe a
small batch (50-100 chunks, ~30 min of work) to bootstrap the dataset,
then expand later. Or use Whisper (different bias geometry) and
spot-check.

### Change 3 — verify the LR=1e-7 hyperparameter against actual script

Their `1e-7` is 100x below the conventional default. Could be a typo in
their config that they never noticed, OR genuinely necessary because
full-talker gradients explode at higher rates. **Open their actual
training script (not just the config) and confirm the LR is what runs.**
If it's actually `1e-5`, our analysis was wrong; if it's actually `1e-7`,
note the instability that drove them there.

### Change 4 — speaker-embedding position 6 is brittle reverse-engineering

The `input_codec_embedding[:, 6, :]` injection point is not documented in
the official Qwen3-TTS code; it's something the external repo found by
inspection. Could break silently on the next Qwen3-TTS release. Before
relying on this:
- Pin the exact Qwen3-TTS version we use
- Document the position-6 dependency as a known fragility
- Plan a regression test: train V13.1 on Qwen3-TTS-snapshot-N, then
  retrain V13.2 on Qwen3-TTS-snapshot-N+1 and compare. If outputs
  diverge, the injection point needs adjustment.

### Change 5 — n=10 A/B test is statistically weak

n=10 with 7/10 win = p≈0.17 (not significant). Bump to **n ≥ 30**
prompts for the blind A/B (split: 15 Taiwan-vocab, 15 generic). With
n=30 and the same 70% win-rate, p≈0.02 → defensible call.

### Change 6 — additional failure modes to watch for

Beyond "talker forgets base model on out-of-domain prompts" already in
the risk table:

- **Loss of linguistic coherence** on complex grammar / long sentences
  (small SFT corpus often = simple spoken sentences only)
- **Acoustic quality collapse** — artifacts, higher noise floor on
  in-domain prompts
- **Prosodic monotony** — model overfits to narrow emotional range in
  the training set
- **Punctuation blindness** — model forgets how to interpret commas,
  question marks, etc., produces run-on sentences

Add explicit smoke tests for each: 5 long sentences, 5 with varied
punctuation, 5 with explicit emotion tags. Compare V13 vs V12 + base
Qwen3-TTS on each.

### Change 7 — DO THIS FIRST: 1-hour gradient check (kills or confirms V13)

Before any data pipeline work, run a minimal script:

1. One hand-transcribed `(text, audio)` pair
2. Implement the proposed `sft_text` forward + backward pass
3. Verify:
   - `talker.model.layers[i].weight.grad` is not None and non-zero for
     multiple layers
   - Speaker-embedding injection at position 6 doesn't cause NaNs
   - Loss decreases monotonically over ~10 training steps
4. If gradients don't flow OR model is unstable, V13 is **dead on
   arrival** — no point building the ASR / dataset / hyperparameter
   tuning around a training loop that doesn't work.

This is the cheapest possible test of the core hypothesis. ~1 hour of
work to know whether to invest the next 3-5 days.

---

## Revised "what to do TODAY" (after Gemini review)

1. **(1 hour) Gradient check.** Implement the V13 forward+backward on
   one hand-transcribed sample. Verify talker layer gradients + injection
   point stability. **If this fails, stop — V13 is dead.**
2. **(1 hour) Read external repo's actual training script** (not just
   config). Confirm whether LR=1e-7 is real and whether they apply LoRA
   somewhere we missed.
3. **(if step 1 + 2 pass) Hand-transcribe 50-100 chunks** of existing
   Mark recordings. ~30-60 min of human work; dramatically lower data-
   bias risk vs ASR-transcribed.
4. **(if all of above) Implement LoRA-on-talker first** (not full SFT).
   Train V13-lora vs V13-sft head-to-head on a small subset before
   committing to full runs.
5. **n=30 blind A/B**, not n=10, with Taiwan-vocab / generic / complex-
   grammar / varied-punctuation buckets.

Total revised effort: 5-7 days (was 3-5), but ~80% lower risk of
silently wasting the time.
