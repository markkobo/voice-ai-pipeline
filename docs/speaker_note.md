# EverHome — NY Tech Week Demo Speaker Note

**Date:** 2026-06-02 6pm · **Format:** 8 min · 2nd of 2 demoers · ~80-90 ppl
**Audience:** Vibe-coding curious, young, not deeply technical
**URL on slide:** everhome.mkk.dev · QR code (docs/everhome_qr.png)

> Print this or read off phone. Each section = one slide. Timings include
> ad-lib buffer. Demo eats ~4 min. **You can cut the "How" slide if running
> short** — the demo + ethics make the case.

---

## Slide 1 — Hi (60-90s)

**Title:** EverHome — A Time Capsule You Can Talk To
**Subtitle:** *Local voice AI that sounds like the person your family actually remembers.*

> "Hi, I'm Mark. I built EverHome.
>
> It's a voice AI you keep at home — like a small box on your shelf — that
> sounds like a specific person in your family. Not a generic warm voice
> with their name on it. **The actual person.** The way they laugh, the
> phrases they use, what they think about.
>
> The whole thing runs locally. Nothing leaves your house."

*Pause. Let the room imagine it.*

---

## Slide 2 — Why now (90s)

**Title:** Why this didn't exist 6 months ago

> "There are two reasons families don't have this yet.
>
> **One.** Today's voice AI — ElevenLabs, OpenAI Voice, the big names —
> is built for call centers and ad-reads. To use them, you upload your
> grandmother's voice to a cloud server. And the ones that sound best
> won't let you fine-tune them on a specific person, so you end up with
> a generic warm voice with their name printed on it.
>
> **Two.** Open-source voice cloning crossed the close-relative-recognition
> threshold in the last six months. You can now fine-tune on 30 minutes
> of audio and get something that actually sounds like that person — on
> hardware you can buy at Best Buy.
>
> So the components exist. They just haven't been put together for
> families. That's what EverHome is."

---

## Slide 3 — Demo (3-4 min) 🎬 LIVE

**Title:** Let me show you.

*Open `everhome.mkk.dev/ui?demo=1` — splash. Click through.*

**Demo flow (3 acts):**

**Act 1 — Voice clone reveal (60s)**
> "This is my AI clone. About 30 minutes of recorded conversation, trained
> on a home GPU. Some of what you're about to hear is me. Some of it is
> the AI. See if you can tell."

*Toggle 🎧 Listen-only. Speak one sentence (intro to a story).*
*Click "Let AI continue" → AI continues in your voice for ~20s.*
*Audience reaction beat.*

**Act 2 — Memory + persona (60s)**
> "Same voice, but it knows different things and adjusts its tone for who
> it's talking to. Watch."

*Switch listener dropdown: Child → Reporter → Friend. Say a short prompt
each time. Audience hears the same voice but different tone & word choice.*

**Act 3 — Ethics moment (30s)**
> "And there's a guardrail. If you ask it directly..."

*To the AI:* "Are you a real person?"
*AI responds:* "I'm Mark's AI voice clone, built for the EverHome demo..."

> "It tells you. Always. We designed for disclosure, not deception."

---

## Slide 4 — Behind the curtain (60-90s)

**Title:** How a one-person team built this

> "Stack is open-source end to end. Qwen voice cloning, Claude for the
> persona reasoning, a Cloudflare tunnel for access. I wrote most of it
> with Claude Code over weekends.
>
> The hard part wasn't the AI. It was the small stuff:
> getting the voice to sound right when it switches between Chinese and
> English. Stopping the AI from saying *star-star* out loud when the
> language model uses markdown bold. Making the recording stay private —
> which sounds easy until you realize every default in the modern stack
> is 'upload it to my cloud.'
>
> If you've ever vibe-coded a side project that grew teeth, this is what
> it feels like at 35 minutes of training audio."

---

## Slide 5 — The ask (30s)

**Title:** EverHome — everhome.mkk.dev
**Big QR code center-of-slide → everhome.mkk.dev**

> "This is Phase 1. Phase 2 puts the appliance in your home. Phase 3 is
> real-time speech-to-speech, fully local.
>
> If you want to try it, scan the QR. If you want to build with me, the
> same QR has my email. Thanks."

*Off stage. Smile.*

---

# Backup / ad-lib

**If running short (skip Slide 4):**
After Demo Act 3, go straight to QR. "If you want to know how I built it, the
README is on the site. Scan, take a look, talk to me after."

**If running long (compress Slide 2):**
Cut Reason 2 paragraph. Open-source angle can land in Q&A.

**Q&A primers:**
- *"Is this real?"* → "Yes. The voice you heard is from a model trained on
  my recordings on the GPU sitting in my office. Live, no pre-render."
- *"Privacy?"* → "On-device. Nothing uploaded. We're designing for medical-
  grade compliance for the B2B path — eldercare, hospice — where
  uploading isn't even legal."
- *"Why local?"* → "Because the people who most need this — families,
  eldercare, hospice — are exactly the ones who can't upload."
- *"How much?"* → "We haven't priced. The hardware will be in the
  consumer-appliance range. No subscription, no usage fees."
- *"Why you?"* → Brief: ex-Bloomberg / ex-Google infra; personal motivation.
  Don't oversell. The product is the pitch.

---

# Stage notes

- **Demo URL** is the live one at `everhome.mkk.dev/ui?demo=1` — pre-load
  it before the talk so the splash doesn't take a beat to load.
- **Failover**: if WiFi flakes, have a 30-second pre-recorded voice clip
  saved on the phone — switch to "let me play what it sounds like"
  without breaking flow.
- **QR slide** has `everhome.mkk.dev` typed in big text under the QR — so
  someone at the back of the room can read it even if scan fails.
- **First-person AI** — keep saying "my clone" or "the AI clone" instead
  of "AI" or "it" — humanizes without overclaiming.
