"""V13 Step 1d — tokenize each chunk's audio into codec_ids.

Implements the prepare_data.py equivalent from
mozi1924/Qwen3-TTS-EasyFinetuning. The external repo's training loop
expects each JSONL record to carry pre-computed `audio_codes`
alongside `audio` + `text`. This script reads the train.jsonl produced
by `v13_build_dataset.py` and writes `train_prepared.jsonl` with
`audio_codes` added per record.

Why separate from v13_build_dataset.py:
- The build step makes thousands of slow OpenAI API calls — we don't
  want to re-run that if codec tokenization breaks.
- Codec tokenization is GPU-bound; running it as a separate phase
  lets us free the GPU between Whisper (CPU/API) and tokenization.

Idempotent:
- Skips records that already have audio_codes in train_prepared.jsonl.
- --max-chunks N for cost-safe trial runs.

Run:
    bash scripts/restart.sh --stop      # free GPU
    .venv/bin/python scripts/v13_prepare_codecs.py --persona test
    bash scripts/restart.sh             # restore server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v13_prepare")

ROOT = Path(__file__).resolve().parent.parent
TARGET_SR = 24000


def load_audio_for_tokenizer(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV → mono float32 normalized to [-1, 1] at 24kHz."""
    sr, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        from scipy import signal as sps  # noqa: PLC0415
        num_samples = int(len(audio) * TARGET_SR / sr)
        audio = sps.resample(audio, num_samples).astype(np.float32)
        sr = TARGET_SR
    return audio, sr


def prepare(persona: str, max_chunks: Optional[int] = None) -> int:
    out_root = ROOT / "data/training" / f"{persona}_v13"
    train_jsonl = out_root / "train.jsonl"
    prepared_jsonl = out_root / "train_prepared.jsonl"

    if not train_jsonl.exists():
        log.error("train.jsonl not found at %s — run v13_build_dataset.py first", train_jsonl)
        return 1

    # Load all source records
    records = []
    with open(train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Source records: %d (from %s)", len(records), train_jsonl)

    # Load any existing prepared output for idempotency
    prepared_by_audio: dict[str, dict] = {}
    if prepared_jsonl.exists():
        with open(prepared_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    prepared_by_audio[rec["audio"]] = rec
        log.info("Already-prepared records: %d (will skip)", len(prepared_by_audio))

    if max_chunks is not None:
        records = records[:max_chunks]
        log.info("Limiting to first %d records (--max-chunks)", max_chunks)

    # Filter to records that still need preparing
    todo = [r for r in records if r["audio"] not in prepared_by_audio]
    if not todo:
        log.info("Nothing to do — every record is already prepared.")
        return 0
    log.info("To prepare: %d records", len(todo))

    # Load Qwen3-TTS tokenizer
    log.info("Loading Qwen3-TTS speech tokenizer...")
    from qwen_tts import Qwen3TTSTokenizer  # noqa: PLC0415
    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")
    log.info("Tokenizer loaded; encoding audio_codes...")

    new_count = 0
    skipped = 0
    failed = 0

    for i, rec in enumerate(todo):
        audio_path = out_root / rec["audio"]
        if not audio_path.exists():
            log.warning("  [skip] missing audio file: %s", audio_path)
            skipped += 1
            continue

        try:
            audio, sr = load_audio_for_tokenizer(audio_path)
            enc = speech_tokenizer.encode(audio, sr=sr)
            codes = np.asarray(enc["audio_codes"][0])  # (seq_len, num_code_groups)
        except Exception as e:
            log.error("  [fail] %s: %s", rec["audio"], e)
            failed += 1
            continue

        out_rec = dict(rec)
        out_rec["audio_codes"] = codes.astype(np.int64).tolist()
        out_rec["audio_codes_shape"] = list(codes.shape)  # for diagnostics
        prepared_by_audio[rec["audio"]] = out_rec
        new_count += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            log.info("  [%d/%d] prepared (new=%d, skipped=%d, failed=%d)",
                     i + 1, len(todo), new_count, skipped, failed)
            _write_jsonl_atomic(prepared_jsonl, prepared_by_audio)

    _write_jsonl_atomic(prepared_jsonl, prepared_by_audio)
    log.info("=" * 70)
    log.info("Prepared %d records: new=%d skipped=%d failed=%d",
             len(prepared_by_audio), new_count, skipped, failed)
    log.info("Output: %s", prepared_jsonl)
    if prepared_by_audio:
        sample_shape = next(iter(prepared_by_audio.values()))["audio_codes_shape"]
        log.info("Sample audio_codes shape: %s (T_codec, num_code_groups)", sample_shape)
    return 0


def _write_jsonl_atomic(path: Path, records: dict) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        # Sort by audio path for deterministic ordering
        sorted_recs = sorted(records.values(), key=lambda r: r["audio"])
        for rec in sorted_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="cap on number of records to prepare (for trial run)")
    args = p.parse_args()
    return prepare(args.persona, args.max_chunks)


if __name__ == "__main__":
    sys.exit(main())
