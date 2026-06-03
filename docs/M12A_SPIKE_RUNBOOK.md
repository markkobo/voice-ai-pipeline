# M12a — Zero-Shot Cloning Eval Spike (Runbook)

**Goal.** In one day, determine whether **Qwen3.5-Omni-Light** or
**Step-Audio 2 mini** zero-shot cloning matches the depth of our
current Qwen3-TTS-LoRA path. If yes, collapse M11/M12/M13 by a few
months. If no, confirm current path and re-bury M12a.

**Why now.** Gemini 2.5 Pro review (2026-06-03) flagged this as the
single highest-leverage action in the roadmap. One-day investigation
that can reshape M11/M12/M13 sequencing.

**Owner.** Mark Ko (runs inference + listens). Claude helps prepare
the harness and analyze artifacts.

**Status.** Scaffolding ready (this doc). Inference + listening
pending. Hardware: A10G with 14 GB free while server is up; will
need to stop the EverHome server temporarily.

---

## 1. Inputs (already on disk)

### Reference audio (Mark, ~30 seconds)
```
/home/rding/voice-ai-pipeline/data/recordings/denoised/child_test_20260530_034842_915324/audio.wav
duration: 29.5s
```
Denoised, LUFS-normalized; same source family as the v12 LoRA training set.

### Production baseline (already trained, ready to compare)
```
TTS engine: Qwen3-TTS 12Hz 1.7B SFT
Merged model: /home/rding/voice-ai-pipeline/data/models/merged_qwen3_tts_test_v12_20260530_034627_235835
```
This is the EverHome-demo voice. Treat its output as the "ceiling we
must match or beat" for the spike to be a win.

### Test sentences (the things each model must say)

Five sentences, mixing register + length + language. Same five for
every candidate so blind A/B is fair.

```
1. EN short:  "Hey, you're back. How was your day?"
2. EN medium: "I started building this because I watched my grandmother lose the ability to share her stories — and I never recorded her voice."
3. EN technical: "We fine-tune Qwen3-TTS on about thirty minutes of audio per person, and the whole thing runs locally on an appliance."
4. ZH short:  "好啦，不要生氣嘛。"
5. ZH medium: "我每次想起阿嬤，就想起她炒菜的時候哼的那首歌。"
```

These cover the cases EverHome ships in production — short conversational,
emotional reflection, technical pitch, casual ZH, narrative ZH.

---

## 2. Candidates

### A. Qwen3.5-Omni-Light
- **HF page.** Look up the exact card under `Qwen/Qwen3.5-Omni-Light-*` (the audit used `Qwen3-Omni-30B-A3B-Instruct` as license proxy; confirm Apache-2.0 on the actual Light variant before downloading).
- **Size estimate.** ~3-7B (Light tier).
- **Inference path.** Talker module per the paper (arXiv:2604.15804) accepts a reference speaker prompt. Use the 29.5s Mark reference as that prompt.
- **Known limitation per `RESEARCH_SFT_S2S.md`.** Closed codec tokenizer — no SFT path. We evaluate zero-shot ONLY.

### B. Step-Audio 2 mini
- **HF page.** `stepfun-ai/Step-Audio-2-mini-Base` and `stepfun-ai/Step-Audio-2-mini-Instruct`.
- **License.** Apache-2.0 (code + weights, confirmed in license audit).
- **Size.** ~7B.
- **Inference path.** Text-audio token interleaving; zero-shot cloning via the same audio-token route, per StepFun maintainer's HF discussion #1.
- **Known limitation.** No SFT recipe (issue #67 open). Evaluate zero-shot only.

### C. (Control) Qwen3-TTS-LoRA v12 (already running)
- Just hit the running EverHome server: `curl -X POST http://localhost:8080/api/tts/stream ...` with each test sentence.

---

## 3. Setup steps (≈ 1-2 hours, mostly downloads)

```bash
# 0. Free GPU memory
bash scripts/restart.sh --stop      # or: pkill -f "python -m app.main"
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# 1. Create spike workspace
mkdir -p data/m12a_spike/{ref,outputs/{qwen35_omni_light,step_audio_2_mini,qwen3_tts_lora_v12}}
cp data/recordings/denoised/child_test_20260530_034842_915324/audio.wav \
   data/m12a_spike/ref/mark_30s_reference.wav

# 2. Download Qwen3.5-Omni-Light (verify exact model name first via HF web)
source .venv/bin/activate
huggingface-cli download Qwen/Qwen3.5-Omni-Light-<variant> \
  --local-dir data/m12a_spike/models/qwen35_omni_light

# 3. Download Step-Audio 2 mini
huggingface-cli download stepfun-ai/Step-Audio-2-mini-Base \
  --local-dir data/m12a_spike/models/step_audio_2_mini

# 4. (Spike harness lives at scripts/m12a_spike.py — see §4)
python scripts/m12a_spike.py --model qwen35_omni_light \
       --ref data/m12a_spike/ref/mark_30s_reference.wav \
       --out data/m12a_spike/outputs/qwen35_omni_light/

python scripts/m12a_spike.py --model step_audio_2_mini \
       --ref data/m12a_spike/ref/mark_30s_reference.wav \
       --out data/m12a_spike/outputs/step_audio_2_mini/

# 5. Production baseline (server must be UP — restart if you stopped it)
bash scripts/restart.sh
python scripts/m12a_spike.py --model qwen3_tts_lora_v12 \
       --out data/m12a_spike/outputs/qwen3_tts_lora_v12/
```

Restart the EverHome server when done so the public demo at
`everhome.mkk.dev` is back up.

---

## 4. Human eval rubric

For each of the 5 test sentences, Mark listens to all 3 outputs in
RANDOM order (don't reveal which is which until after scoring).

Score each output 1-5 on each dimension:

| Dimension | 1 = bad | 5 = excellent |
|---|---|---|
| **Speaker identity** (does it sound like Mark to Mark?) | Generic voice with Mark's name | Indistinguishable from a real recording |
| **Prosody / cadence** (pace, pauses, intonation match Mark's natural speech) | Robotic / unnatural pacing | Flows like Mark talking |
| **Emotion appropriateness** (does the emotion fit the sentence?) | Flat or wrong emotion | Right tone for the content |
| **Articulation** (clarity, mumbling, glitches) | Slurred, glitchy, dropped phonemes | Clean, intelligible |
| **CN/EN code-switching** (sentences 4-5) | Heavy accent leak, mispronunciation | Native-level Chinese phonemes |

Total possible per sentence: 25. Per model across 5 sentences: 125.

**Decision criterion.** A candidate "matches" Qwen3-TTS-LoRA v12 if
its TOTAL score is within **5 points** of the baseline (~4%). A
candidate is "clearly above" if it beats baseline by **10+** AND
wins on speaker identity for at least 4/5 sentences.

---

## 5. Decision tree

```
                            ┌──────────────────────────────────────┐
                            │ Does any candidate "clearly above"   │
                            │ Qwen3-TTS-LoRA v12 across all 5      │
                            │ sentences (≥10 points + 4/5 ident.)? │
                            └──────────────────────────────────────┘
                                          │
                ┌─────────────────────────┴──────────────────────────┐
                YES                                                   NO
                │                                                      │
    ┌───────────▼─────────────┐                          ┌─────────────▼────────────┐
    │ FAST-TRACK M13          │                          │ Does any candidate       │
    │ - Defer M8.5 polish     │                          │ "match" (within 5 pts)?  │
    │ - Defer M11 abstraction │                          └──────────────────────────┘
    │ - Plan migration to     │                                       │
    │   winning model         │                  ┌────────────────────┴────────────┐
    │ - Save 2-3 months       │                  YES                                NO
    └─────────────────────────┘                  │                                  │
                                     ┌───────────▼──────────┐         ┌─────────────▼────────┐
                                     │ KEEP M11 abstraction │         │ CONFIRM current path │
                                     │ - Add winner as 1st- │         │ - TTS-LoRA stays     │
                                     │   class engine in M11│         │ - Bury M12a          │
                                     │ - Eval again post-M11│         │ - Press on M8.5 + M9 │
                                     └──────────────────────┘         └──────────────────────┘
```

---

## 6. What this spike will NOT do

- **Train.** Both candidates lack working SFT recipes for new speakers
  (per `RESEARCH_SFT_S2S.md`). This spike is zero-shot only.
- **Test cross-lingual quality on bilingual code-switching at depth.**
  Sentences 4-5 give a signal; a real eval would need ~20+ ZH
  sentences. If a candidate clearly wins on EN but fails on ZH, that's
  a partial result — note it explicitly.
- **Test latency.** Engineering question for M11/M12, not for M12a.

---

## 7. Outputs to retain

- `data/m12a_spike/outputs/<model>/{sent_1,sent_2,sent_3,sent_4,sent_5}.wav` — five outputs per model, three models.
- `data/m12a_spike/score_card.csv` — Mark's blind scores, one row per (model, sentence, dimension).
- `data/m12a_spike/summary.md` — verdict + reasoning + which path in the §5 decision tree was taken.

Commit `data/m12a_spike/summary.md` to the repo (not the audio — too
heavy + privacy). Reference the verdict in the next ROADMAP update.

---

## 8. Estimated wall-clock (single sitting)

| Step | Time |
|---|---|
| Stop server + free GPU | 2 min |
| Download Qwen3.5-Omni-Light weights | 20-40 min |
| Download Step-Audio 2 mini weights | 20-30 min |
| Get Qwen3.5-Omni-Light inference working (likely tooling stumbles) | 30-60 min |
| Get Step-Audio 2 mini inference working | 30-60 min |
| Run 5 sentences × 2 candidates + 5 sentences from baseline | 15-30 min |
| Mark listens + scores blind | 30 min |
| Write up `summary.md` + decide path | 15 min |
| Restart server | 2 min |
| **Total** | **2-4 hours** |

Run this on a quiet weekend morning, not mid-week.
