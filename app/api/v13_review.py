"""V13 — minimal web tool to listen + verify Whisper transcripts.

Loads `data/training/<persona>_v13/train_prepared_filtered.jsonl`,
shows random chunks with audio player + transcript, lets the user
mark each ✓ / ✗ / skip. Verdicts append to `verdicts.csv` so we can
later filter out the ✗ ones before training.

Routes:
    GET  /ui/v13_review?persona=test         single-page reviewer
    GET  /api/v13_review/data?persona=test&n=20      JSONL records sampled
    POST /api/v13_review/score               append a verdict row
    GET  /api/v13_review/audio/{persona}/{rel}      serve the chunk WAV

No auth — relies on the box's existing cloudflared exposure being the
user's own surface. Same posture as the recordings UI.
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["v13_review"])

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data/training"


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
def _persona_root(persona: str) -> Path:
    return _DATA_ROOT / f"{persona}_v13"


def _jsonl_path(persona: str) -> Path:
    # Prefer the filtered file; fall back to unfiltered if filter hasn't been run.
    p = _persona_root(persona) / "train_prepared_filtered.jsonl"
    if p.exists():
        return p
    return _persona_root(persona) / "train_prepared.jsonl"


def _verdicts_path(persona: str) -> Path:
    return _persona_root(persona) / "verdicts.csv"


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona: str = Field(..., min_length=1)
    audio: str = Field(..., min_length=1)
    text: str
    verdict: str  # "ok" / "bad" / "partial"
    note: Optional[str] = None


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
@router.get("/api/v13_review/data")
async def api_v13_data(
    persona: str = Query(...),
    n: int = Query(20, ge=1, le=200),
    seed: Optional[int] = Query(None),
):
    src = _jsonl_path(persona)
    if not src.exists():
        raise HTTPException(404, f"no V13 dataset for persona {persona!r} ({src})")
    records = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # Don't ship audio_codes to the browser — too big.
                rec.pop("audio_codes", None)
                rec.pop("audio_codes_shape", None)
                records.append(rec)
    rng = random.Random(seed if seed is not None else None)
    sample = rng.sample(records, k=min(n, len(records)))
    return {"persona": persona, "total": len(records), "sampled": len(sample), "records": sample}


@router.get("/api/v13_review/audio/{persona}/{rel_path:path}")
async def api_v13_audio(persona: str, rel_path: str):
    # rel_path comes from the JSONL "audio" field which is relative to
    # data/training/<persona>_v13/.
    full = _persona_root(persona) / rel_path
    # Path traversal guard.
    try:
        full.resolve().relative_to(_persona_root(persona).resolve())
    except ValueError:
        raise HTTPException(400, "invalid path")
    if not full.exists():
        raise HTTPException(404, f"audio not found: {rel_path}")
    return FileResponse(str(full), media_type="audio/wav")


@router.post("/api/v13_review/score")
async def api_v13_score(verdict: Verdict):
    path = _verdicts_path(verdict.persona)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["ts", "audio", "text", "verdict", "note"])
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            verdict.audio,
            verdict.text,
            verdict.verdict,
            verdict.note or "",
        ])
    return {"status": "ok", "saved_to": str(path.relative_to(_persona_root(verdict.persona).parent))}


@router.get("/api/v13_review/verdicts")
async def api_v13_verdicts(persona: str = Query(...)):
    """Running tally of verdicts for this persona."""
    path = _verdicts_path(persona)
    counts = {"ok": 0, "bad": 0, "partial": 0}
    bad_rows = []
    if path.exists():
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("verdict", "")
                if v in counts:
                    counts[v] += 1
                if v == "bad":
                    bad_rows.append({"audio": row["audio"], "text": row["text"]})
    return {"persona": persona, "counts": counts, "bad": bad_rows[-20:]}  # last 20 bad


@router.get("/ui/v13_review", response_class=HTMLResponse)
async def ui_v13_review(persona: str = Query("test")):
    """Single-page reviewer. No build step — vanilla HTML + JS."""
    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>V13 ASR Review · __PERSONA__</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#0f1828;color:#e6e6e6;margin:0;padding:24px;max-width:780px;margin:auto}
h1{color:#c89efa;font-size:1.4rem;margin:0 0 12px}
.meta{color:#888;font-size:.85rem;margin-bottom:24px}
.card{background:#1d2a3a;border-radius:12px;padding:20px;margin-bottom:16px}
.audiopath{font-family:monospace;font-size:.75rem;color:#6a8;word-break:break-all;margin-bottom:8px}
.text{font-size:1.15rem;line-height:1.5;color:#fff;margin-bottom:16px;padding:12px;background:#0f1828;border-radius:8px}
audio{width:100%;margin-bottom:12px}
.buttons{display:flex;gap:8px;flex-wrap:wrap}
button{padding:10px 18px;border:none;border-radius:8px;font-size:.95rem;cursor:pointer;font-weight:600}
.ok{background:#2a7;color:#fff}
.partial{background:#c89efa;color:#0f1828}
.bad{background:#c54;color:#fff}
.skip{background:#444;color:#bbb}
.tally{position:fixed;top:12px;right:12px;background:#1d2a3a;padding:10px 16px;border-radius:8px;font-size:.85rem}
.done{text-align:center;padding:40px;color:#6a8;font-size:1.2rem}
.kbd{display:inline-block;border:1px solid #555;border-bottom-width:2px;border-radius:4px;padding:1px 6px;font-family:monospace;font-size:.75rem;color:#aaa;margin-left:4px}
</style></head>
<body>
<div class="tally" id="tally">✓ 0 · ⚠ 0 · ✗ 0</div>
<h1>V13 ASR Review — persona=<span id="persona">__PERSONA__</span></h1>
<div class="meta">Listen to each chunk, compare to the transcript, mark <strong style="color:#2a7">OK</strong>, <strong style="color:#c89efa">Partial</strong>, or <strong style="color:#c54">Bad</strong>.
Shortcuts: <span class="kbd">1</span> OK · <span class="kbd">2</span> Partial · <span class="kbd">3</span> Bad · <span class="kbd">0</span> Skip · <span class="kbd">Space</span> replay</div>
<div id="content">Loading...</div>

<script>
const persona = new URLSearchParams(location.search).get("persona") || "__PERSONA__";
document.getElementById("persona").textContent = persona;
let records = [];
let idx = 0;
let counts = {ok:0, partial:0, bad:0};

async function init() {
  const res = await fetch(`/api/v13_review/data?persona=${encodeURIComponent(persona)}&n=30`);
  if (!res.ok) { document.getElementById("content").innerHTML = `<div class="done">No V13 data: ${res.status}</div>`; return; }
  const data = await res.json();
  records = data.records;
  document.querySelector(".meta").textContent += `  ·  ${data.total} total records, ${data.sampled} sampled`;
  show();
}

function show() {
  if (idx >= records.length) {
    document.getElementById("content").innerHTML = `<div class="done">Done — ${counts.ok+counts.partial+counts.bad} reviewed.<br>Verdicts saved to verdicts.csv on the box.</div>`;
    return;
  }
  const rec = records[idx];
  document.getElementById("content").innerHTML = `
    <div class="card">
      <div class="audiopath">${rec.audio}  ·  #${idx+1} of ${records.length}</div>
      <audio controls autoplay src="/api/v13_review/audio/${encodeURIComponent(persona)}/${encodeURIComponent(rec.audio)}"></audio>
      <div class="text">${rec.text}</div>
      <div class="buttons">
        <button class="ok"      onclick="score('ok')">✓ OK <span class="kbd">1</span></button>
        <button class="partial" onclick="score('partial')">⚠ Partial <span class="kbd">2</span></button>
        <button class="bad"     onclick="score('bad')">✗ Bad <span class="kbd">3</span></button>
        <button class="skip"    onclick="skip()">Skip <span class="kbd">0</span></button>
      </div>
    </div>`;
}

async function score(verdict) {
  const rec = records[idx];
  counts[verdict] = (counts[verdict] || 0) + 1;
  document.getElementById("tally").textContent = `✓ ${counts.ok} · ⚠ ${counts.partial} · ✗ ${counts.bad}`;
  await fetch("/api/v13_review/score", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({persona, audio: rec.audio, text: rec.text, verdict}),
  });
  idx++;
  show();
}

function skip() { idx++; show(); }

document.addEventListener("keydown", e => {
  if (e.key === "1") score("ok");
  else if (e.key === "2") score("partial");
  else if (e.key === "3") score("bad");
  else if (e.key === "0") skip();
  else if (e.key === " ") {
    const a = document.querySelector("audio");
    if (a) { a.currentTime = 0; a.play(); }
    e.preventDefault();
  }
});

init();
</script>
</body></html>"""
    return html.replace("__PERSONA__", persona)
