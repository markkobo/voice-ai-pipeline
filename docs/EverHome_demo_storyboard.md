# EverHome — 15-min demo storyboard

**A Time Capsule You Can Talk To**

Voice. Memory. Local.

Cloned voices of the people who matter, running on a box in your home — never the cloud.

---

## Timing

| Min | Block |
|---|---|
| 00:00 – 01:30 | The hook |
| 01:30 – 03:30 | What is EverHome (one slide) |
| 03:30 – 07:30 | Live demo |
| 07:30 – 10:00 | How it works (3 visuals, no code) |
| 10:00 – 12:00 | Where it's going |
| 12:00 – 14:00 | Vibe coding moment |
| 14:00 – 15:00 | Close |

---

## 00:00 — 01:30 · The hook

**Say**:
> Last week I cloned my child's voice and put him on a box in my apartment. I can talk to him from anywhere in the house. He's still 3 years old in that box — and he always will be.

*[Pause. Press play on a pre-recorded clip of the trained child voice.]*

> This is EverHome. It runs entirely on my hardware. Nothing leaves my house.

> 🔒 **Safety**: Opener uses a saved WAV, not live TTS. The room hears the strongest version of the voice on first impression.

---

## 01:30 — 03:30 · What is it

One slide. Don't switch for 90 seconds.

> # **EverHome**
> A time capsule you can talk to.
> Voice. Memory. Local.

- 🎙 Clones the voices of the people who matter
- 🧠 Knows your family — answers your kid differently than your wife
- 🏠 Runs on your hardware, never the cloud

---

## 03:30 — 07:30 · Live demo

*Chat UI projected. Already logged in. Already on the correct persona/version. Press mic.*

**Say** while doing:
> I'm going to ask my kid what he wants for dinner.

Speak naturally. Let everyone hear your voice → the wait → the cloned voice respond. **The waiting silence is fine.** Don't fill it.

Three short exchanges (write these on a sticky note):

1. "What did we do last Saturday?"
2. "Sing me your favorite song"
3. "Tell me you love me" *(this lands hard — be ready for it)*

Then switch the persona/listener dropdown to a different family member, ask the SAME question.

**Say**:
> Watch — same prompt, different relationship.

### Safety
- Pre-staged 3 prompts on a sticky note
- Demo voice = SFT v7 (working). NEVER use the LoRA v1 (runaway garbage)
- Test 30 min before doors open
- Fallback: a screen-recording of the same exchange queued in Quicktime
- If TTS fails: hit play on the fallback, say "wifi gremlins today, this is what it normally sounds like" — audience won't mind

---

## 07:30 — 10:00 · How it works

Three slides. 50 seconds each. No code.

### Slide A — The Corpus
```
Voice memos.
Old texts.
Chat logs.
WhatsApp exports.
              ↓
   EverHome reads it all
   and remembers.
```

### Slide B — The Person
```
   ┌──────────┐       ┌─────────────┐
   │  Voice   │       │ Personality │
   │  cloned  │   +   │   learned   │
   └──────────┘       └─────────────┘
              ↓
       They become one person.
```

### Slide C — The House
*Draw a house. A small server inside. No wires going outside.*

**Say** over Slide C:
> This is the part I care about most. The people you love are not data points for someone else's training run.

---

## 10:00 — 12:00 · Where it's going

> I'm 7 days into this. Here's what's coming.

Four bullets max:

- **Now** — Voice + chat working
- **Next month** — Each family member trains in 10 minutes from their phone
- **Q3** — Memory: EverHome remembers conversations across years
- **Q4** — A real appliance you plug in and forget about

---

## 12:00 — 14:00 · Vibe coding moment

**Say**:
> This event is about building with AI. Let me show you something I added at the airport this morning.

*Open Claude Code. Type one prompt LIVE.*

Pick a tiny visible polish. Pre-write the prompt on a sticky note:

> "When EverHome is talking, animate the avatar with a soft glow."

Watch the diff happen. Reload the chat UI. Show it works.

### Safety
- Pre-test the change at home — know it works
- Have the exact prompt written down
- If Claude misfires live: "this happens, you iterate" — that's literally the event theme

---

## 14:00 — 15:00 · Close

**Say**:
> EverHome is a 15-minute idea I've been chasing for 7 days. I want it to be the next 6 months of my life. If you have a parent whose voice you wish you'd recorded, or a kid you want to keep at this exact age — talk to me after. Thank you.

Final slide:

> # EverHome
> everhome.dev | @yourhandle

---

## Pre-flight checklist (30 min before)

- [ ] Server running, GPU healthy (`nvidia-smi`)
- [ ] cloudflared tunnel up with verified URL (have a backup tunnel ready)
- [ ] Active version = best SFT model, **NOT** a LoRA model
- [ ] Pre-recorded opener WAV ready in Quicktime queue
- [ ] Sticky note: 3 demo prompts + 1 vibe-code prompt
- [ ] Backup screen-recording of full demo flow
- [ ] Laptop on AC, not battery
- [ ] HDMI dongle + test projector resolution
- [ ] Phone on airplane mode (no notifications on screen)
- [ ] Water nearby

---

## The one thing to fix before 06/02

Your best-sounding model (v7) was trained on telephone-grade source audio (eff_bw 443 Hz). It works but sounds muffled.

**Re-record one clean 5-minute sample**:
- Browser mic (not phone call)
- Quiet room
- 48 kHz
- Phone held close to mouth

Then retrain SFT before 06/02. Highest-leverage single change to make the demo land emotionally.
