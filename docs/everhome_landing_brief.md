# EverHome — Landing Page Design Brief

Companion to `docs/everhome_landing.html`. This file is the editorial /
design spec; the HTML is the implementation. Keep them in sync — if you
change one, update the other.

---

## 1. Audience & goals

**Primary audiences (in priority order):**

1. **Investors** — seed / pre-seed VCs and angels in consumer AI,
   eldercare-tech, and privacy-first software. Decision: should we take
   a meeting? Time on page: ~90 seconds.
2. **Tech press** — journalists at TechCrunch / The Information /
   Bloomberg covering voice AI, generative consumer tech, and
   eldercare. Decision: is there a story here? Time on page: ~3 minutes
   if they bother to scroll.
3. **B2B2C partners** — eldercare facilities, hospice networks, and
   memorial-service SaaS companies looking for a voice-AI tier. Decision:
   is this someone we can pilot with? Time on page: ~2 minutes.
4. **Curious individuals** — family members who heard about the demo
   from social, word-of-mouth, or a journalist. Decision: can I try
   this? Time on page: ~30 seconds before tapping "Talk to it".

**Page goal:** convert each cohort to the right next step. For
investors and press, the CTA is "Request a meeting" (email). For
partners, the CTA is also email but with a different subject line. For
individuals, the CTA is the live demo at `everhome.mkk.dev/ui?demo=1`.

We do **not** ask anyone to "sign up" or "join waitlist" — there is no
product to ship them yet. We ask for a conversation.

## 2. Voice & tone

Mark Ko — ex-Bloomberg, ex-Google. **Technical but warm.** Confident
on the moat, honest on the roadmap. No hype words ("revolutionary",
"AI-powered", "the future of"). No emoji. No exclamation points in body
copy. Headlines may use one well-placed em dash.

Style references: Stripe's early landing pages, Anthropic's "Core
Views" essay, Cloudflare's product launch posts. Long-form sentences
when the idea earns them. Short sentences when stating a moat.

Avoid: marketing copy that could describe any AI company. The page
should be unmistakably about **legacy preservation**, not "voice
agents" or "conversational AI". If a section could be on OpenAI's site
unchanged, rewrite it.

## 3. Section-by-section content brief

### Hero
- **Title:** EverHome
- **Tagline:** "A Time Capsule You Can Talk To."
- **Subhead:** "Privacy-first voice AI that preserves the people who
  raised you. Built for families. Runs on your hardware."
- **CTAs:** Primary — "Talk to it" (links to live demo). Secondary —
  "Request a meeting" (mailto).
- **Background:** deep purple → navy gradient (matches chat UI demo
  mode for visual continuity).

### The problem
- Families lose access to grandparents' voices, stories, and ways of
  speaking. Three failure modes: dementia, distance, death.
- One paragraph each. No statistics yet (would need to source them
  carefully); use evocative language instead.
- End with a single sentence acknowledging cloud-AI alternatives exist
  but require sending your grandmother's voice to a third party — the
  reason we exist.

### The solution
- One sentence: "EverHome turns 30–60 minutes of recorded conversation
  into a voice AI clone that your family can talk to forever, on
  hardware you own."
- Three supporting bullets:
  - Cloned voice that close relatives recognize (not a generic TTS).
  - Memory of who they were — stories, opinions, ways of speaking.
  - Runs on a small box in your home. Nothing leaves the house.

### How it works (3 steps)
1. **Record** — 30–60 minutes of conversation, guided by a prompt list.
2. **Train** — a few hours on a small home appliance (DGX Spark / Mac
   Mini class). Fully local.
3. **Talk** — speak to them anytime. The voice clones their timbre;
   the memory captures their stories.
- Visual: 3 column cards on desktop, stack on mobile. Subtle iconography
  if any (Unicode symbols only — no external icon fonts).

### Why now
- Open-source voice cloning crossed the close-relative-recognition
  threshold in late 2025 (CosyVoice 2, Qwen3-TTS, Step-Audio 2).
- Consumer hardware caught up — NVIDIA DGX Spark, Mac Mini M4, and
  Strix Halo all ship 64–128 GB unified memory in the $2–4K range.
- Cloud TTS giants (ElevenLabs, OpenAI Voice Mode) optimize for scale,
  not privacy. The legacy-preservation use case requires the opposite.

### What makes EverHome different
Three differentiators, each one paragraph:
1. **Privacy moat** — local-first appliance. No accounts, no upload,
   no recurring fees. Your grandmother's voice stays in your house.
2. **Chinese-native end-to-end** — every layer (ASR, LLM, TTS,
   embedding) is Chinese-native by deliberate choice. English-only
   competitors miss our highest-value market.
3. **Close-relative-recognition quality** — supervised fine-tuning per
   speaker, not zero-shot voice prompting. The difference is whether
   the call sounds like *them* or like a generic warm voice.

### Markets
Two-by-two block:
- **B2C direct** — families who want to preserve a grandparent, parent,
  or partner. One-time hardware + ongoing cloud-free.
- **B2B2C** — eldercare facilities, hospice partners, memorial services,
  and Chinese-diaspora community organizations. White-label or
  co-branded deployments. Each gets a per-deployment persona spec.

### The founder
Brief bio (Mark Ko, ex-Bloomberg engineering, ex-Google product). One
sentence on why he's building this — a personal connection to legacy
preservation, transitioning from infrastructure work into a
mission-driven consumer product.
- Photo placeholder (`[TODO: founder photo]`).
- LinkedIn link: `[TODO: link]`.

### Roadmap (transparency)
- **Phase 1 — today.** Chat MVP. Cloned voice, persona prompt, browser
  UI. Shipping demo at NY Tech Week 2026-06-02.
- **Phase 2 — Q3 2026.** Hybrid pipeline: lower latency, deeper memory
  via RAG, per-listener voice routing.
- **Phase 3 — 2027.** End-to-end speech-to-speech with cloning when an
  open-source model lets us SFT a new speaker without leaving local
  hardware (12+ months out per our 2026Q2 audit).

### Ethical guardrails
Single block, four bullets:
- Consent capture before training (the recorded person must opt in on
  camera).
- Disclosure rule — the AI must reveal it is AI if directly asked.
- Audit trail — every training event logged, every inference logged,
  exportable on demand.
- Watermarking — generated audio carries an inaudible mark so a clone
  cannot be passed off as the real person in evidentiary contexts.

This section is non-negotiable. Even investors who don't ask about
ethics will read it as a signal that we have thought about the cliff.

### Footer
- Contact: `[TODO: email]`
- GitHub: `[TODO: link]` (will be public after security review)
- Twitter / X: `[TODO: link]`
- Copyright Mark Ko 2026

## 4. Brand & visual system

**Color palette** (extracted from `app/static/css/standalone.css`
`body.demo-mode`):

| Token | Hex | Usage |
|---|---|---|
| `--bg-from` | `#1a1424` | Top of gradient background |
| `--bg-to` | `#0f1828` | Bottom of gradient background |
| `--surface` | `rgba(15, 24, 40, 0.7)` | Card backgrounds |
| `--text` | `#e6e6e6` | Body copy |
| `--heading` | `#f0e6d2` | Headings (cream) |
| `--accent` | `#c89efa` | Links, CTAs, accent rules (purple) |
| `--accent-cyan` | `#7ee7ff` | Secondary accent for stats / pull-quotes |
| `--muted` | `#aab2c5` | Captions, labels |
| `--border` | `rgba(200,158,250,0.15)` | Card edges |

**Typography:**
- System font stack: `-apple-system, BlinkMacSystemFont, "Segoe UI",
  Roboto, sans-serif`. No webfonts (privacy + load speed).
- Heading scale: hero 48px desktop / 32px mobile, section H2 32px / 24px,
  H3 20px / 18px.
- Body: 17px desktop, 16px mobile, line-height 1.6.

**Layout:**
- Max content width: 1080px centered.
- Section padding: 96px top/bottom desktop, 48px mobile.
- Mobile-first responsive. Primary breakpoint at 600px (per spec).
  Below 600px: single column, larger touch targets, stacked CTAs.

**Imagery:**
- No stock photos. Placeholder block (`[TODO: hero image]`) where a
  custom illustration or photograph should sit.
- Pipeline diagram (How it works) is pure CSS / Unicode — no external
  image asset.

## 5. Technical constraints

- **Single file.** Entire page is `docs/everhome_landing.html` with
  inline `<style>`. No external CSS, no JS, no build step.
- **Self-contained.** Drop into Cloudflare Pages, S3, or any static
  host. Should also render acceptably with `file://`.
- **Accessibility:**
  - Semantic HTML (`<header>`, `<main>`, `<section>`, `<footer>`).
  - Skip link for screen readers.
  - Alt text on every image (placeholders too).
  - Color contrast ≥ AA on body text.
- **Performance:** No webfonts, no JS, no third-party scripts. Should
  score 100 on Lighthouse for unused-CSS / unused-JS.
- **SEO + social:**
  - `<title>`, `<meta description>`.
  - Open Graph: `og:title`, `og:description`, `og:image`, `og:url`.
  - Twitter card: `summary_large_image`.
  - Placeholder image paths only (user will swap in real assets).

## 6. Hosting & deployment notes

**Recommended path** (per user's `reference_user_comms` memory): host on
Cloudflare Pages, point a subdomain of `mkk.dev` at it. The chat app
already lives behind a CF tunnel at `everhome.mkk.dev`; the landing can
sit at `mkk.dev/everhome` or `everhome.mkk.dev/` (root) with the chat
moved to `everhome.mkk.dev/app`.

**Alternative:** feed this brief + the HTML into Lovable or v0.dev for
fancier iteration with imagery and motion. The HTML here is the
authoritative content; visual polish can be layered on.

**Not recommended:** hosting on a separate platform (Vercel, Netlify) —
the user already has CF set up and DM-paired with their phone via the
existing tunnel. Adding a new vendor adds operational surface for no
gain.

## 7. Open editorial questions

- **Real founder bio length.** Current placeholder is two sentences;
  the real version should be ~5 sentences once Mark writes it.
- **Founder photo.** Need a portrait — preferably warm, not corporate.
- **Hero image / illustration.** Options: an abstract waveform; a
  family-style photograph (consent-cleared); a stylized "time capsule"
  illustration. Defer until visual direction lands.
- **Quote from a family tester.** Once any non-Mark family member has
  used the system for a week, capture one sentence verbatim for a
  pull-quote in The Solution or Markets section.
- **Press logos.** If any tech press covers the 06/02 demo, add an "As
  seen in" row above the footer. Do not fabricate logos.

## 8. Update cadence

- Re-read after every major milestone ships (M-Consent, M7, M8, M9).
- Update the Roadmap section when ROADMAP_2026Q3.md phases shift.
- Update the Ethical guardrails section when M-Consent lands real
  consent capture + watermarking.
