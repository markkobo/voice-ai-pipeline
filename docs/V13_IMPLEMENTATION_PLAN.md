# V13 Implementation Plan — Train the Talker, Fix the Accent

**Status:** Action-ready. Consolidates `TRAINING_RECIPE_COMPARISON.md` +
Gemini review at `REVIEW_V13_PLAN_GEMINI_2026-06-09.md` + verified facts
from the external repo's actual `master` branch (2026-06-09).

**Goal:** A V13 fine-tune that captures speaker accent + prosody (not
just timbre), beating V12 in blind A/B by ≥7/10 on Taiwan-vocab prompts
at n=30.

---

## 1. What we verified about the external repo (so plans aren't guesses)

Source: `mozi1924/Qwen3-TTS-EasyFinetuning` master branch, files
`sft_12hz.py`, `cli.py`, `dataset.py`, `prepare_data.py`. Read by
subagent 2026-06-09; verbatim quotes in the audit transcript.

| Fact | Value | Source |
|---|---|---|
| `learning_rate` | **1e-7** (NOT a typo — this is the actual CLI default) | `cli.py:504` |
| `epochs` | 2 | `cli.py:505` |
| `batch_size` | 2 | `cli.py:503` |
| `gradient_accumulation_steps` | 4 | `cli.py:506` |
| Optimizer | AdamW, weight_decay=0.01 | `sft_12hz.py:533` |
| LR scheduler | **None** — DummyLRScheduler (no-op, no warmup) | `sft_12hz.py:535-545` |
| LoRA used anywhere? | **No.** Zero hits for lora/peft/LoraConfig across all files. Full-parameter SFT. | `sft_12hz.py:533` (`AdamW(model.parameters(), ...)`) |
| Loss | `outputs.loss + 0.3 * sub_talker_loss` | `sft_12hz.py:733-736` |
| Speaker emb injection | `input_codec_embedding[:, 6, :] = speaker_embedding` (verbatim) | `sft_12hz.py:712` |
| Speaker emb storage | Also writes into `state_dict["talker.model.codec_embedding.weight"][3000]` (reserved slot) | `sft_12hz.py:309` |
| Dataset format | JSONL with `{audio, text}` required (plus optional `speaker_id`, `language`, `ref_audio`); `prepare_data.py` adds `audio_codes` | `dataset.py:125-129` |

---

## 2. Decisions locked for V13 (from this conversation)

| Decision | Choice | Rationale |
|---|---|---|
| Method | **LoRA on talker FIRST**, full SFT as fallback | Gemini review §2: full SFT is "the sledgehammer." LoRA mirrors our successful M9 approach. User confirmed (msg 2050 #1). |
| Transcripts | **OpenAI Whisper-3 via API** (NOT Qwen3-ASR) | Qwen3-ASR has Mainland-accent priors → circular bias on Taiwan-accented audio. Whisper has different geometry. User specified (msg 2050 #2). |
| LR verification | **Done.** 1e-7 is real, not a typo. | External repo `cli.py:504`. User asked (msg 2050 #3). |
| Position-6 injection | **Brief note + version pin**, no full regression framework | User specified "簡短記錄" (msg 2050 #4). Pin Qwen3-TTS version + comment in code referencing `sft_12hz.py:712` as upstream truth. |
| A/B test | **n=30, blind, 4 buckets** (see §5) | Gemini review: n=10 is p≈0.17. User asked for guidance (msg 2050 #5). |

**Where we deviate from the external recipe** and accept the risk:

- We use LoRA, they use full SFT. **If LoRA fails to capture accent
  after 1 well-tuned run, escalate to full SFT.** No third attempt; if
  full SFT also doesn't beat V12 in blind A/B, the talker-loss
  hypothesis is wrong and we re-diagnose.
- For LoRA we'll need a different LR than 1e-7 (LoRA is usually 1e-4
  to 1e-5 for transformer attention/FFN). Start at **1e-4** and tune
  down if loss explodes.

---

## 3. Implementation order (the actual plan)

### Step 0 — Kill switch: 1-hour gradient check (BEFORE anything else)

Per Gemini review. The cheapest possible test of the core hypothesis.

```python
# scripts/v13_gradient_check.py — write this as the FIRST thing.
# 1. Load Qwen3-TTS-12Hz-1.7B-Base.
# 2. Hand-transcribe ONE 5-second Mark utterance: ("我每次想起阿嬤", "data/recordings/.../chunk_001.wav")
# 3. Build input_text_embedding + input_codec_embedding per sft_12hz.py:707-714.
# 4. Inject: input_codec_embedding[:, 6, :] = speaker_embedding
# 5. Forward: outputs = talker(inputs_embeds=..., labels=codec_0_labels[:, 1:])
# 6. loss = outputs.loss + 0.3 * sub_talker_loss
# 7. loss.backward()
# 8. Check:
#    - talker.model.layers[0].self_attn.q_proj.weight.grad is not None  ✅
#    - talker.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0  ✅
#    - No NaN in any layer's grad  ✅
#    - Re-run 10 times → loss decreases monotonically (or at least 8/10 down)  ✅
```

**If gradient check passes →** proceed to Step 1.
**If gradient check fails →** V13 is dead on arrival. Stop and re-diagnose.

Time budget: 1 hour. If it stretches past 2, also stop — something is
wrong with the assumption.

---

### Step 1 — Build the dataset (~2-3 hours)

```bash
# 1a. Silence-split existing recordings
python scripts/v13_split_audio.py \
    --in data/recordings/denoised/child_test_*/audio.wav \
    --out data/training/test_v13/chunks/ \
    --min-len 1.0 --max-len 15.0 --silence-thresh -16

# Expected: ~50-100 chunks per WAV, 1-15s each.

# 1b. Transcribe with Whisper-3 via OpenAI API
python scripts/v13_transcribe.py \
    --in data/training/test_v13/chunks/ \
    --out data/training/test_v13/train.jsonl \
    --model whisper-1 --language zh

# Each line: {"audio": "chunks/001.wav", "text": "...", "speaker_id": "test"}

# 1c. Spot-check: pick 10 random chunks, manually verify transcript.
#     If WER > 30% on any → remove that chunk.

# 1d. Run prepare_data.py equivalent to add audio_codes (tokenize audio
#     into codec_ids using the base model's tokenizer). Mirror
#     external repo's prepare_data.py.
python scripts/v13_prepare.py \
    --jsonl data/training/test_v13/train.jsonl \
    --out data/training/test_v13/train_prepared.jsonl
```

**Whisper-3 cost estimate**: ~$0.006/min × ~60 min audio ≈ **$0.36 total**.
Negligible.

---

### Step 2 — Implement training (LoRA path first)

New training type in `app/services/training_service/training_job.py`:
`training_type="sft_text_lora"`.

Mirror `sft_12hz.py:700-740` for the forward pass:

```python
# Per-batch training step
text_emb = talker.get_text_embeddings()(input_text_ids) * text_mask
codec_emb = talker.model.codec_embedding(input_codec_ids) * codec_mask
codec_emb[:, 6, :] = speaker_embedding  # NOTE: matches sft_12hz.py:712.
                                        # Brittle — see §6.
input_embeds = text_emb + codec_emb

outputs = talker(inputs_embeds=input_embeds,
                 labels=codec_0_labels[:, 1:])

# Aux loss on the codec_predictor head (matches external repo).
_, sub_talker_loss = talker.forward_sub_talker_finetune(
    talker_codec_ids, talker_hidden_states
)
loss = outputs.loss + 0.3 * sub_talker_loss
```

LoRA setup (this is OUR choice; external repo doesn't do this):

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=16,  # match our existing M9 LoRA rank
    lora_alpha=32,
    target_modules=[
        # Attention projections in talker LM layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        # FFN projections (Qwen-family naming)
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
talker = get_peft_model(talker, lora_cfg)
```

Hyperparameters (LoRA-adjusted from external repo's full-SFT settings):

| Param | Value | Why |
|---|---|---|
| `learning_rate` | **1e-4** | LoRA conventional; external repo's 1e-7 is for full-SFT and would barely move LoRA weights |
| `epochs` | 3 | One more than external's 2 — LoRA needs slightly more passes |
| `batch_size` | 2 | Match external |
| `grad_accum` | 4 | Match external |
| Optimizer | AdamW, wd=0.01 | Match external |
| Scheduler | None (constant LR) | Match external (DummyLRScheduler) |
| Flash Attention 2 | Enabled | Match external |

Save the LoRA adapter (not a merged model) so we can A/B against V12
by hot-swapping the adapter in/out.

---

### Step 3 — Train V13-lora (1-2 hours wall-clock)

```bash
bash scripts/restart.sh --stop  # Free GPU
python -m app.services.training_service.training_job \
    --persona test \
    --training_type sft_text_lora \
    --train_jsonl data/training/test_v13/train_prepared.jsonl \
    --version v13-lora \
    --lr 1e-4 --epochs 3 --batch_size 2 --grad_accum 4
bash scripts/restart.sh  # Restore EverHome server
```

Save final adapter at `data/training/test/versions/v13-lora/adapter/`.

---

### Step 4 — Smoke tests (BEFORE the blind A/B)

Catch the catastrophic-forgetting failure modes Gemini flagged. Each
test runs V12 + V13-lora head-to-head and you listen for obvious
breakage, not just preference.

| Smoke test | Prompts | Failure signal |
|---|---|---|
| Linguistic coherence | 5 long sentences (50+ tokens, complex grammar) | V13 garbled / nonsensical where V12 was fine |
| Acoustic quality | 5 short generic prompts | V13 has artifacts / hiss / dropped phonemes where V12 was clean |
| Prosodic range | 5 prompts with explicit `[E:emotion]` tags spanning warm/upbeat/reflective/wry/gentle | V13 sounds flat regardless of emotion |
| Punctuation interpretation | 5 sentences with mid-sentence pauses, questions, exclamations | V13 ignores punctuation, runs sentences together |

**If any smoke-test bucket clearly regresses, stop the A/B** — fix the
hyperparameters or fall back to full SFT. Don't bury the regression
under bucket-averaged scores.

---

### Step 5 — Blind A/B test (n=30, the real evaluation)

30 prompts in 4 buckets. Each prompt generated twice (V12 + V13-lora),
order shuffled per prompt, presented to you blind.

| Bucket | n | Prompts |
|---|---|---|
| **Taiwan-vocab** | 10 | Sentences with Taiwan-specific vocabulary, tones, particles (e.g. 「啦」「齁」「醬子」), Taiwanese loanwords if natural |
| **Generic ZH** | 10 | Neutral Mandarin sentences with no regional markers |
| **Complex grammar / long** | 5 | Sentences 50+ tokens, multi-clause, embedded quotes |
| **Varied punctuation** | 5 | Includes commas, periods, question marks, exclamations, ellipses |

For each prompt, score:

| Dimension | 1-5 scale (5 = excellent) |
|---|---|
| Speaker identity ("does it sound like Mark?") | 1=generic / 5=indistinguishable from real recording |
| Prosody / accent ("does it sound like Mark's accent?") | 1=Mainland default / 5=Mark's native accent |
| Articulation ("clean output, no glitches?") | 1=mumbled, dropped / 5=crystal clear |
| Emotion fit ("does the emotion match the line?") | 1=flat / 5=natural fit |

Plus binary: **"which of these two sounds more like you?"** Track A vs B,
then unblind at the end.

### Success criteria

V13-lora SHIPS if ALL of:
- Binary win rate ≥ 70% (≥21/30). At n=30 + 70%, binomial p≈0.02.
- Taiwan-vocab win rate ≥ 70% (≥7/10). This is the actual accent test.
- No smoke-test bucket regresses (all V13 smoke scores ≥ V12 by ≥-0.5
  on the 1-5 scale).
- Speaker-identity dimension averages ≥ V12 average across all 30
  prompts.

V13-lora FAILS (escalate to full SFT) if:
- Binary win rate < 50% (V12 actually better).
- Smoke tests regress (catastrophic forgetting detected).

V13-lora is INCONCLUSIVE (retry with different LoRA hyperparameters) if:
- Binary win rate 50-69%.

---

## 4. Tooling — a tiny A/B harness

To avoid bias, build a minimal HTML page that:
1. Reads `data/training/test_v13/eval/pairs.jsonl` (each line: `{prompt, v12_wav, v13_wav}`).
2. For each pair, randomly swaps A/B labels.
3. Plays A then B (you can replay).
4. Form: 1-5 sliders × 4 dimensions, plus "A or B more like you?"
5. Submits to `/api/v13_eval/score` which appends to a CSV.
6. After all 30 prompts: unblinds, computes win rate by bucket + binomial p.

Estimated build time: ~2 hours. Worth it because doing this by ear with
no UI invites confirmation bias.

---

## 5. Brief note on position-6 injection fragility (per user msg 2050 #4)

The line `input_codec_embedding[:, 6, :] = speaker_embedding` (verified
at `sft_12hz.py:712`) is undocumented in official Qwen3-TTS. Position 6
of the codec embedding sequence is the reserved speaker-conditioning
slot (paired with row 3000 of `codec_embedding.weight` per
`sft_12hz.py:309`).

**Mitigation:**
- Pin Qwen3-TTS to the exact version we use for V13.
- Inline code comment at the injection site quoting the upstream
  source (`# matches mozi1924/Qwen3-TTS-EasyFinetuning sft_12hz.py:712 — fragile`).
- When upgrading Qwen3-TTS later, verify the position-6 contract still
  holds by re-running Step 0's gradient check on the new version
  before any production V14 training.

Not building a regression test framework. Just the version pin + code
comment + a one-line entry in the V13 README about the upgrade gate.

---

## 6. Total estimated effort

| Phase | Time |
|---|---|
| Step 0: gradient check | 1 hour (kill switch) |
| Step 1: dataset | 2-3 hours (incl Whisper-3 API + spot-check) |
| Step 2: training code | 4-6 hours (forward pass + LoRA wiring + training_job integration) |
| Step 3: train V13-lora | 1-2 hours wall-clock |
| Step 4: smoke tests | 1 hour |
| Step 5: A/B harness + eval | 3-4 hours (incl 2hr harness build + 1hr listening) |
| **Total** | **12-17 hours** (~2-3 working days) |

This is the lower-risk plan. If V13-lora fails and we escalate to
V13-sft (full SFT with LR=1e-7), add ~3 more hours for the training
loop variant + retest.

---

## 7. What ships when V13-lora succeeds

- New training type `sft_text_lora` lives in
  `app/services/training_service/`.
- New persona LoRA adapter stored at
  `data/training/<persona>/versions/v<N>-lora/`.
- Training UI gains a "Type" option for `sft_text_lora` (default stays
  `sft` until V13 has 30-day stability data).
- Documentation update to `HOW_TO_TRAIN_YOUR_VOICE.md` once stable.

---

## 8. Linked docs

- `docs/TRAINING_RECIPE_COMPARISON.md` — full diagnosis (the WHY)
- `docs/REVIEW_V13_PLAN_GEMINI_2026-06-09.md` — Gemini review (the WHAT-WENT-WRONG)
- External: https://github.com/mozi1924/Qwen3-TTS-EasyFinetuning (master branch as of 2026-06-09)
