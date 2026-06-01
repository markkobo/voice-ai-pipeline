# EverHome — Slide Skeleton

**For Google Slides shared deck · NY Tech Week 2026-06-02 · 8 min · Mark Ko (2nd of 2)**

> Bullet points only. No long sentences. Speaker note carries the prose
> (see `docs/speaker_note.md`). Each slide is the **visual anchor** while
> Mark talks; if a slide could be replaced with a photo of Mark and the
> demo still landed, **less is more**.

---

## Slide 1 — Title (60-90s)

```
EVERHOME
A Time Capsule You Can Talk To

Local voice AI that sounds like the person your family
actually remembers.

— Mark Ko
```

**Visual:** clean dark background (matches landing's deep purple → navy
gradient). Logo wordmark only. No images yet.

---

## Slide 2 — Why now (90s)

```
Why this didn't exist 6 months ago

· Cloud voice AI is built for call centers — your grandmother's
  voice goes to a server you don't own.

· Open-source voice cloning crossed the close-relative threshold
  in the last six months.

· The components exist. They haven't been put together for families.
```

**Visual:** 3 lines, generous spacing, no icons. Lean into typography.

---

## Slide 3 — Demo (3-4 min) 🎬

```
Live demo
```

**Visual:** Single word "Demo" or no slide at all (cut to browser).

**Stage flow** (don't put this on slide, just for Mark):
1. Open `everhome.mkk.dev/ui?demo=1` — EverHome splash appears
2. Toggle 🎧 Listen-only ON
3. Speak 1 sentence story-opener
4. Click "🎤 Let AI continue" → AI continues in Mark's voice ~20s
5. Switch listener: Child → Reporter → Friend (say same prompt 3x)
6. Ask: "Are you a real person?" → AI discloses

---

## Slide 4 — Behind the curtain (60-90s)

```
How a one-person team built this

· Open-source voice + Claude Code + Cloudflare tunnel.
  No proprietary models.

· The hard part wasn't the AI. It was the small stuff:
  Chinese ↔ English code-switching.
  Every default in the modern stack is "upload it to my cloud."

· If you've ever vibe-coded a side project that grew teeth —
  this is what it feels like at 35 minutes of training audio.
```

**Visual:** 3 bullets. Optional: tiny logos in the corner — Qwen, Claude
Code icon, Cloudflare logo — but they shouldn't dominate.

---

## Slide 5 — The ask (30s)

```
EVERHOME

everhome.mkk.dev

[ BIG QR CODE — center ]

Phase 1: today (chat MVP)
Phase 2: appliance in your home
Phase 3: real-time speech-to-speech, fully local

Try it. Build with me.
```

**Visual:** QR code dominant, takes 50% of slide. URL underneath in big
type (24pt+) so anyone in the back can read it even if their phone can't
scan from that distance.

**QR file:** `docs/everhome_qr.png` (already generated, 396×396, encodes
`https://everhome.mkk.dev/`)

---

# Notes for the deck-builder

- **Total slides: 5.** Slide 3 (demo) might literally be a blank slide
  or "DEMO" word art — Mark is presenting from a browser tab, not the
  deck.
- **Color palette** (matches landing):
  - Background: `#0f1828` (deep navy)
  - Heading: `#f0e6d2` (cream)
  - Accent: `#c89efa` (purple) — for key words / underlines
  - Body: `#e6e6e6`
- **Font**: system sans-serif. No webfonts (faster).
- **No transitions / animations** — slow demo down. Hard cuts only.
- **Time budget**: S1 90s, S2 90s, S3 3-4min, S4 90s, S5 30s ≈ 8 min flat.
- **If a slide could be a one-liner with no bullets, prefer that.** The
  prose is in the speaker note; the slide is the anchor.

---

# Things NOT to put on slides

- Long paragraphs (slide is not the speaker note)
- Architecture diagrams (audience is vibe-coders, not enterprise architects)
- Founder bio (he's standing right there)
- Roadmap timeline as a Gantt chart (Phase 1/2/3 names are enough)
- Logos of competitors (FlashLabs, ElevenLabs etc. — irrelevant on a slide)
- "Thank you" final slide — the QR + ask IS the close
