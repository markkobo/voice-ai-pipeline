# EverHome — How to Train Your Voice (Step-by-Step)

A practical guide for anyone (technical or not) to record audio, train
an SFT voice model on their own voice, and A/B-test whether the model
actually sounds like them.

**Time needed (first time):** ~45 minutes recording + ~30-90 minutes
unattended training + 10 minutes A/B test. After that, retrains are ~30 min.

**Prerequisites:** EverHome server running at `https://everhome.mkk.dev/`
or your local equivalent. An active consent record exists for your
persona (M-Consent.1 gates corpus + recording — Mark's existing personas
already have this).

---

## Part 1 — Recording (≈30-60 minutes of audio)

### What "good for training" sounds like

The model learns from what it hears. If your audio has hiss, distant
mic noise, background music, or you're shouting across a room, the
model will learn THOSE features instead of your actual voice. The
target is **clean, close-mic, natural conversational speech**.

| ✅ Good | ❌ Bad |
|---|---|
| Phone held 4-8" from mouth | Phone on the desk 3 feet away |
| Quiet room, no fan, no AC | TV/podcast playing in background |
| Conversational tone, natural pace | Reading robotically, stilted |
| 30-60 minutes total across multiple short takes | One single 60-min monologue |
| Mix of emotions / contexts (story, chat, technical) | All one register |

### Step-by-step

1. **Open** `https://everhome.mkk.dev/ui/recordings`.
2. **Select your persona** from the **🎭 Persona** dropdown. If you don't
   exist yet, click the **+** next to the dropdown to add a new persona
   (display name + lowercase ID, e.g. `mom`).
3. **Select a listener** from the **👤 Listener** dropdown. The listener
   tells the model "who am I speaking TO" — this matters more than you'd
   think for natural prosody. Available options:
   - `default` — generic, safe choice
   - `child` — gentler, simpler register
   - `mom` / `friend` / `reporter` / `elder` — relationship-specific
   - Create your own with the **+** button
4. **Hold-to-record.** Press and hold the big **🎤** button. The button
   pulses red while recording. Release to stop. Your recording auto-saves.
5. **Speak naturally**, not from a script. Tell stories. Talk about your
   day. Read something aloud as if to a friend. Each take should be
   30 seconds to 5 minutes.
6. **Aim for ≥ 30 minutes of TOTAL audio per (persona, listener) pair.**
   The training UI shows a "ready for training" badge once you cross the
   threshold (visible per-persona in the list view).
7. **Vary the content.** Five takes of "hello hello hello" gives a worse
   model than five takes of "I was telling my daughter about that time
   I…", "let me explain how this works…", etc.
8. **Check each take.** Click the row to expand → play it back. Delete
   anything that:
   - Has background voices (other people talking)
   - Has obvious clipping (audio sounds crackly / "fuzzy peak")
   - Is too quiet to hear without straining
   - Has long silent gaps (>5 sec) — split into separate takes instead
9. **Repeat for each listener** you want the model to handle. If you're
   only training one combined voice, just stay on `default`.

### Stop recording when you have:

- ≥ 30 min total per listener (60 min is much better)
- A mix of registers (story + casual + emotional content)
- All takes pass the "I'd listen to this without wincing" test

---

## Part 2 — Training an SFT model

SFT (Supervised Fine-Tuning) makes the base Qwen3-TTS model learn YOUR
voice specifically. After ~30-90 minutes of training, a "merged model"
is saved that produces speech in your voice when prompted with text.

### Step-by-step

1. **Open** `https://everhome.mkk.dev/ui/training`.
2. **Select your persona** from the dropdown. The page shows how many
   minutes of audio you've recorded per listener.
3. **Listener filter** — leave as "All Listeners" for your first run.
   For multi-listener routing (M10 territory), train one model per
   listener.
4. **Training settings** — defaults are tuned for the typical case:
   - **Training Type:** `SFT (recommended, stable)` — **leave this**.
     LoRA is experimental and currently produces noisy output on some
     personas.
   - **Epochs:** `30` — good for most personas. Increase to 50 if you
     have very little data (<20 min) or if v1 sounds undertrained.
     Decrease to 10-20 if you have a lot (>60 min) and SFT keeps
     diverging.
   - **LoRA Rank:** `32` — only matters if Training Type is LoRA.
     Ignore for SFT.
   - **Batch Size:** `1` — the only value that fits on an A10G. Higher
     batch sizes OOM the GPU.
   - **Learning Rate:** `1e-5` — the safer default. Lower (1e-6) means
     SFT barely moves; higher (1e-4) with LoRA caused runaway training
     on v10. Stick with `1e-5` unless you know why you're changing it.
   - **Language:** `Auto (recommended — preserves accent)` — **leave
     this on Auto.** Selecting "Chinese" forces the Beijing-codec
     codec_language_id which adds a Beijing accent to Taiwan-accented
     sources. The Auto setting writes `spk_is_dialect=false` so
     inference uses the speaker's actual accent.
5. **Click 🚀 Start Training.** The page shows live progress:
   - Epoch counter (e.g. `12 / 30`)
   - Loss (should decrease over time; if it plateaus at >5 or spikes
     wildly, something is wrong)
   - ETA (rough estimate, updates as training runs)
6. **Wait.** Training a 30-epoch SFT on 30 minutes of audio takes
   roughly 30-60 minutes on an A10G. You can close the browser and
   come back — the training runs server-side and survives reconnect.
7. **When done**, the page shows ✅ + a new version under the
   **Versions** tab (e.g. `v13_20260605_021500_xxxxx`). Click it
   to:
   - **Preview** it speaking a test phrase (top-right input box).
   - **Activate** it as the live model for this persona. Once active,
     the chat UI's TTS uses this voice.

### What to do if training fails

| Symptom | Cause | Fix |
|---|---|---|
| Loss spikes to NaN / Infinity | Audio with clipping or learning rate too high | Lower LR to 1e-6; remove the clipped takes |
| ETA keeps growing | Server-side GPU contention | Check the chat UI isn't actively running TTS during training |
| "OOM" error | Batch size >1 | Set batch size back to 1 |
| v1 sounds like generic Qwen voice with your name | Undertrained — not enough data or too few epochs | Record more audio (target 60 min); retrain with 50 epochs |
| v1 sounds fast / flat / wrong accent | Language set to Chinese on a non-Beijing speaker | Retrain with Language=Auto |

---

## Part 3 — A/B testing your voice vs the model

The honest test: can YOUR family tell the difference between a clip of
you and a clip of the model? Done right, this is the only meaningful
quality gate (per `docs/RESEARCH_SFT_S2S.md` — no published benchmark
exists for close-relative recognition).

### Quick-and-dirty test (just for yourself)

1. **Open** `https://everhome.mkk.dev/ui` (the chat UI).
2. **Switch persona** to the one you just trained (the dropdown).
3. **Pick a Reply Language** that matches your training data
   (`中文` if you trained on Chinese audio, `English` if English). Mixing
   produces the "fast English on a Chinese-trained voice" symptom we
   shipped a workaround for.
4. **Type a sentence** in the chat (or speak it). Listen to how the AI
   replies. Does it sound like you?
5. **Compare to a real take.** Open `/ui/recordings`, find a clip you
   recorded, play it back side-by-side.

### Blind A/B test (the honest test)

This is the close-relative-recognition rubric we plan to use as the M10
acceptance gate. Anyone can run it on themselves today.

**Setup (5 min):**

1. Pick **5 short sentences** the model can plausibly say (~10-30 words
   each). Mix EN + ZH if you trained bilingual. Examples:
   - Conversational: "Hey, you're back. How was your day?"
   - Reflective: "我每次想起阿嬤, 就想起她炒菜的時候哼的那首歌."
   - Technical: "We fine-tune the model on about thirty minutes of audio."
2. For each sentence, generate **2 audio clips**:
   - **A — Real you:** record yourself saying it. Use the recordings UI.
   - **B — The model:** open the chat UI → toggle 🎧 Listen-only ON →
     speak the sentence → click 🎤 Let AI continue → AI continues in your
     voice. Save the audio (browser dev tools → Network → find the WS
     binary frames, or use a Loom recording on the tab with "Share tab
     audio" checked, per the speaker note).
3. **Randomize order.** Don't tell yourself which is which.

**Test (15 min):**

For each of the 5 pairs, score each clip 1-5 on each dimension:

| Dimension | 1 (bad) | 5 (excellent) |
|---|---|---|
| **Speaker identity** | Generic voice with your name | Indistinguishable from a real recording |
| **Prosody / cadence** | Robotic / unnatural pacing | Flows like you talking |
| **Emotion appropriateness** | Flat or wrong emotion for the line | Right tone for the content |
| **Articulation** | Slurred, glitchy, dropped phonemes | Clean, intelligible |
| **CN/EN code-switching** (if mixed) | Heavy accent leak, mispronunciation | Native-level on both languages |

Score each clip out of 25 (5 dims × 5). A model that scores within
**5 points** of your real recording across 5 sentences is "matching"
quality. Within 10 points = "usable but needs more training". Below
that = retrain with more data or higher epochs.

**Family-recognition test (the real gold standard):**

Ask a family member who knows your voice well to listen WITHOUT being
told which clip is which. Ask one question: **"Does this sound like
[your name]?"** Score Yes / Sort-of / No per clip. If they can't
reliably tell the model apart from you — that's the bar EverHome is
trying to hit.

### When to retrain

Retrain (with more data and/or more epochs) if:

- Family-recognition score is below 60% (they can usually tell it's the
  model)
- Speaker-identity dimension averages below 3.5 across the 5 sentences
- Any sentence has an articulation score below 3 (glitchy output)

---

## Troubleshooting cheat sheet

| Symptom | Likely cause | Where to fix |
|---|---|---|
| Recording row has 0:00 duration or won't play | Mic permission denied / browser issue | Reload page, allow mic, re-record |
| "No active consent record" 403 when uploading | M-Consent.1 gate fired | Create consent via `POST /api/consent/` or in the consent UI (M-Consent.2 — not shipped yet, use API for now) |
| Training won't start, dropdown empty | Persona has no recordings, or no listener variants | Record at least one take first |
| Voice sounds like Beijing accent but I'm from Taiwan | Trained with Language=Chinese | Retrain with Language=Auto |
| English replies sound fast / flat | TTS LoRA trained on Chinese only | Record English audio + retrain, OR pin Reply Language to 中文 in chat UI |
| Different sentences sound like different people | Per-emotion instruct is too varied | Currently mitigated by instruct=None per persona |
| ASR transcribes "The first was the first to be built." when silent | Qwen3-ASR hallucination on silence | Already filtered server-side (commit `db1e49d`); refresh page to get latest |

---

## What's in the box vs not yet shipped

| Capability | Status |
|---|---|
| Record per-persona, per-listener audio in the browser | ✅ shipped (`/ui/recordings`) |
| Denoise + LUFS-normalize at recording time | ✅ shipped (DeepFilterNet) |
| SFT training on Qwen3-TTS 1.7B | ✅ shipped (`/ui/training`) |
| Per-persona active voice version + chat preview | ✅ shipped |
| Listen-only + "Let AI continue" demo gimmick | ✅ shipped (chat UI) |
| Multi-listener voice routing (per-listener LoRA) | 🟡 M10 — design only |
| Consent capture UI | 🟡 M-Consent.2 — API exists, UI pending |
| Cross-session memory ("AI remembers what you told it") | 🟡 M8 — design only |
| Local LLM (no OpenAI dependency) | 🟡 M9 — design only |

---

## Linked docs

- `docs/PROJECT_BRIEF.md` — why this product exists + design rules
- `docs/M12A_SPIKE_RUNBOOK.md` — for evaluating alternative cloning models
- `docs/M10_RESEARCH_PLAN.md` — per-listener voice routing (research path)
- `docs/RESEARCH_SFT_S2S.md` — why SFT depth is the moat
- `CLAUDE.md` — code architecture for contributors
