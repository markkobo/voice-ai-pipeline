"""V13 Step 1 — build the dataset.

Implements V13_IMPLEMENTATION_PLAN §Step 1:

  1. Silence-split existing recordings into 1-15s chunks
     (pydub.split_on_silence, dBFS-16 threshold, keep_silence=200ms).
  2. Transcribe each chunk via OpenAI Whisper-3 API (whisper-1 model,
     language=zh). NOT Qwen3-ASR — Qwen has Mainland accent priors
     that would create circular bias on Taiwan-accented audio
     (Gemini review §2, 2026-06-09).
  3. Write JSONL: {audio: rel_path, text: transcript, speaker_id: persona}.
  4. Spot-check: print 10 random chunks for manual review.

Idempotent: re-running skips chunks + transcripts that already exist.

Output:
    data/training/<persona>_v13/chunks/      *.wav (24kHz mono)
    data/training/<persona>_v13/train.jsonl  one record per chunk
    data/training/<persona>_v13/spot_check.txt  10 random chunks to review

Run:
    .venv/bin/python scripts/v13_build_dataset.py --persona test

Approx cost: ~$0.006/min × total_audio_minutes via Whisper API.
For test persona with ~119 min recorded → ~$0.71.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v13_dataset")

ROOT = Path(__file__).resolve().parent.parent

# Silence-split parameters — mirror mozi1924/Qwen3-TTS-EasyFinetuning
# step1_audio_split.py defaults (verified during V13 plan research).
MIN_CHUNK_MS = 1_000          # 1.0s
MAX_CHUNK_MS = 15_000         # 15s
MIN_SILENCE_MS = 500          # collapse if 500ms+ silence detected
SILENCE_THRESH_OFFSET_DB = -16  # dBFS - 16 = dynamic threshold
KEEP_SILENCE_MS = 200         # keep 200ms padding at chunk edges
TARGET_SR = 24000             # Qwen3-TTS native rate

WHISPER_MODEL = "whisper-1"   # OpenAI Whisper-3 (released as whisper-1 in API)
WHISPER_LANGUAGE = "zh"       # Mandarin Chinese (Whisper supports both
                              # Traditional and Simplified naturally — output
                              # tends to match input register)


def find_persona_recordings(persona: str) -> list[Path]:
    """Locate denoised audio files for a given persona. Persona "test"
    has folders named `child_test_*`, `default_test_*`, etc. We match
    on `_<persona>_` to avoid catching `test1` when looking for `test`."""
    base = ROOT / "data/recordings/denoised"
    wavs: list[Path] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        # Folder name shape: <listener>_<persona>_<timestamp>_<rand>
        parts = d.name.split("_")
        if len(parts) < 2:
            continue
        # listener_persona means parts[1] is the persona (or parts[1]_parts[2])
        # Conservative match: persona is exactly the substring after the
        # first underscore and before the next underscore that starts a
        # date (8 digits).
        if f"_{persona}_" in d.name or d.name.endswith(f"_{persona}"):
            audio_path = d / "audio.wav"
            if audio_path.exists():
                wavs.append(audio_path)
    return sorted(wavs)


def silence_split_one(wav_path: Path, out_dir: Path) -> list[Path]:
    """Split one WAV into 1-15s chunks, saved as 24kHz mono. Idempotent."""
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    rec_name = wav_path.parent.name
    rec_out_dir = out_dir / rec_name
    rec_out_dir.mkdir(parents=True, exist_ok=True)

    # If chunks already exist for this recording, skip the slow split.
    existing = sorted(rec_out_dir.glob("chunk_*.wav"))
    if existing:
        log.info("  [skip] %d existing chunks for %s", len(existing), rec_name)
        return existing

    log.info("  splitting %s ...", rec_name)
    audio = AudioSegment.from_file(str(wav_path))
    # Ensure 24kHz mono before split (so the saved chunks are training-ready)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)

    silence_thresh = audio.dBFS + SILENCE_THRESH_OFFSET_DB
    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=silence_thresh,
        keep_silence=KEEP_SILENCE_MS,
    )

    out_paths: list[Path] = []
    for i, chunk in enumerate(chunks):
        # Reject chunks that are too short. Long chunks (> MAX_CHUNK_MS)
        # are kept but flagged — drop them in case Whisper times out.
        if len(chunk) < MIN_CHUNK_MS:
            continue
        if len(chunk) > MAX_CHUNK_MS:
            log.warning("    chunk %d is %.1fs > %.1fs cap; trimming",
                        i, len(chunk) / 1000, MAX_CHUNK_MS / 1000)
            chunk = chunk[:MAX_CHUNK_MS]
        out_path = rec_out_dir / f"chunk_{i:03d}.wav"
        chunk.export(str(out_path), format="wav")
        out_paths.append(out_path)
    log.info("  → %d chunks", len(out_paths))
    return out_paths


def transcribe_one(client, chunk_path: Path, max_retries: int = 4) -> Optional[str]:
    """Whisper-3 transcribe a single chunk with exponential backoff
    on transient errors (rate limit, network, 5xx). Returns None only
    after retries exhausted. Gemini review f28120b §latent-1."""
    import time
    for attempt in range(max_retries + 1):
        try:
            with open(chunk_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    language=WHISPER_LANGUAGE,
                    # Per OpenAI docs: small `prompt` helps Whisper bias
                    # toward Traditional Chinese / Taiwanese vocabulary.
                    # The actual training audio is Mark speaking Mandarin
                    # with Taiwan accent, mixing in English technical
                    # terms. Bias prompt covers that.
                    prompt="這是一段台灣口音的中英文混合對話，包含技術名詞。",
                )
            text = (resp.text or "").strip()
            return text if text else None
        except Exception as e:
            err_str = str(e).lower()
            # Transient signals: rate-limit (429), 5xx, timeout/connection.
            transient = (
                "rate" in err_str or "429" in err_str
                or "500" in err_str or "502" in err_str or "503" in err_str or "504" in err_str
                or "timeout" in err_str or "connection" in err_str
            )
            if attempt < max_retries and transient:
                wait = 2 ** attempt
                log.warning("    transient error for %s (attempt %d): %s — retrying in %ds",
                            chunk_path.name, attempt + 1, e, wait)
                time.sleep(wait)
                continue
            log.error("    transcribe failed for %s: %s", chunk_path.name, e)
            return None


def build_dataset(persona: str, max_chunks: Optional[int] = None) -> int:
    out_root = ROOT / "data/training" / f"{persona}_v13"
    chunks_dir = out_root / "chunks"
    jsonl_path = out_root / "train.jsonl"
    spot_check_path = out_root / "spot_check.txt"

    out_root.mkdir(parents=True, exist_ok=True)

    wavs = find_persona_recordings(persona)
    log.info("Persona %r: %d source recordings", persona, len(wavs))
    if not wavs:
        log.error("No recordings found for persona %r", persona)
        return 1

    # Step 1a — silence-split
    log.info("=" * 70)
    log.info("Step 1a — silence split")
    log.info("=" * 70)
    all_chunks: list[Path] = []
    for wav in wavs:
        chunks = silence_split_one(wav, chunks_dir)
        all_chunks.extend(chunks)
    log.info("Total chunks: %d", len(all_chunks))

    if max_chunks is not None and len(all_chunks) > max_chunks:
        log.info("Limiting to first %d chunks (--max-chunks)", max_chunks)
        all_chunks = all_chunks[:max_chunks]

    # Step 1b — Whisper transcribe
    log.info("=" * 70)
    log.info("Step 1b — Whisper-3 transcribe (OpenAI API)")
    log.info("=" * 70)

    # Load existing transcripts so we don't pay to re-transcribe.
    existing_records: dict[str, dict] = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing_records[rec["audio"]] = rec
        log.info("Loaded %d existing transcripts; will skip those",
                 len(existing_records))

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    # Fail-fast on missing API key. Without this check, every chunk
    # transcription silently fails with a delayed authentication
    # error and the script reports "failed=N" but no clear root cause.
    # Gemini review f28120b §bugs-1.
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        log.error(
            "OPENAI_API_KEY not set. Put it in .env at the project root "
            "or export it in the shell before running this script."
        )
        return 2
    from openai import OpenAI
    client = OpenAI()

    new_records: list[dict] = []
    skipped = 0
    failed = 0

    for i, chunk_path in enumerate(all_chunks):
        rel_path = str(chunk_path.relative_to(out_root))
        if rel_path in existing_records:
            skipped += 1
            continue

        if i % 25 == 0 or i + 1 == len(all_chunks):
            log.info("  [%d/%d] transcribing... (skipped=%d, failed=%d, new=%d)",
                     i + 1, len(all_chunks), skipped, failed, len(new_records))

        text = transcribe_one(client, chunk_path)
        if text is None:
            failed += 1
            continue
        new_records.append({
            "audio": rel_path,
            "text": text,
            "speaker_id": persona,
            "language": "Auto",
        })

        # Persist after every 10 new records so we don't lose progress
        # if the run is killed mid-way.
        if len(new_records) % 10 == 0:
            _write_jsonl_atomic(jsonl_path, existing_records, new_records)
            log.info("    ↳ progress saved (%d existing + %d new)",
                     len(existing_records), len(new_records))

    # Final persist
    _write_jsonl_atomic(jsonl_path, existing_records, new_records)

    total_records = len(existing_records) + len(new_records)
    log.info("=" * 70)
    log.info("Transcription done: %d total records (skipped=%d, failed=%d, new=%d)",
             total_records, skipped, failed, len(new_records))
    log.info("JSONL: %s", jsonl_path)

    # Step 1c — spot check: print 10 random chunks for manual WER review
    _emit_spot_check(jsonl_path, spot_check_path, out_root)

    return 0


def _write_jsonl_atomic(path: Path, existing: dict, new: list[dict]) -> None:
    """Write all (existing + new) records to a temp file then rename."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        # Sort by audio path for deterministic ordering
        all_records = list(existing.values()) + new
        all_records.sort(key=lambda r: r["audio"])
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _emit_spot_check(jsonl_path: Path, spot_check_path: Path, out_root: Path) -> None:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        log.warning("No records to spot-check")
        return

    n = min(10, len(records))
    rng = random.Random(42)
    sample = rng.sample(records, n)

    lines = []
    lines.append(f"V13 spot-check — {n} random chunks for manual WER review")
    lines.append(f"JSONL: {jsonl_path}")
    lines.append("")
    lines.append("Listen to each chunk and confirm the transcript is accurate.")
    lines.append("If WER > 30% on any chunk, remove that row from train.jsonl.")
    lines.append("Watch especially for:")
    lines.append("  - Taiwan-specific vocab 'normalized' to Mainland equivalent")
    lines.append("  - English technical terms misheard as Chinese")
    lines.append("  - Truncated phrases at chunk boundaries")
    lines.append("")
    lines.append("-" * 70)
    for i, rec in enumerate(sample):
        wav_full = out_root / rec["audio"]
        lines.append(f"#{i+1}  {rec['audio']}")
        lines.append(f"      text: {rec['text']}")
        lines.append(f"      play: ffplay -nodisp -autoexit {wav_full}")
        lines.append("")

    spot_check_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Spot-check written to: %s (review %d random chunks)", spot_check_path, n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True, help="persona_id (e.g. test, test1)")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="cap on number of chunks to transcribe (for cost-safe trial run)")
    args = p.parse_args()
    return build_dataset(args.persona, args.max_chunks)


if __name__ == "__main__":
    sys.exit(main())
