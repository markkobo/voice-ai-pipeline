"""V13 — filter Whisper hallucinations from train_prepared.jsonl.

Whisper hallucinates known stock phrases on silent / near-silent audio
chunks. Spotted in the 2026-06-09 spot-check:
  - Exact echo of the bias prompt ("這是一段台灣口音的中英文混合對話...")
  - YouTube-style subscription requests ("請不吝點贊訂閱轉發打賞...")

Both are clear pollution — they aren't anything Mark said. Training on
them would teach the TTS model to produce these phrases.

This is the same problem class as Qwen3-ASR hallucinations (engine.py),
just from a different model. We can't share the filter because Whisper's
hallucination set is different.

Run:
    .venv/bin/python scripts/v13_filter_hallucinations.py --persona test

Writes filtered files alongside the originals with `_filtered` suffix.
Does NOT modify the source jsonls — caller can diff to verify.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v13_filter")

ROOT = Path(__file__).resolve().parent.parent


# Exact bias prompt we sent to Whisper — when echoed verbatim it's an
# unambiguous "I had nothing to say about this audio" signal.
BIAS_PROMPT = "這是一段台灣口音的中英文混合對話，包含技術名詞。"

# Substrings that indicate Whisper fell into its YouTube-training-data
# attractor. The presence of ANY of these means the transcript is
# almost certainly hallucinated (Mark isn't reading YouTube boilerplate).
YOUTUBE_HALLUCINATION_SUBSTRINGS = [
    "請不吝點贊",
    "請不吝點贊訂閱",
    "點贊訂閱",
    "打賞支持",
    "明鏡與點點欄目",
    "明鏡電視",
    "點點欄目",
    "歡迎訂閱",
    "感謝您的觀看",
    "字幕由Amara.org社區提供",
    # User-flagged Bads from /ui/v13_review on 2026-06-09:
    "以上言論不代表本台立場",
    "請保存在你設計的網頁",
]

# Optional second-pass filter: trivially short transcripts. NOT enabled
# by default — Mark may genuinely say "好", "對啊", etc.
SHORT_THRESHOLD_CHARS = 0  # 0 = disabled. Set to e.g. 2 to drop singletons.


def is_hallucination(text: str) -> tuple[bool, str]:
    """Returns (is_halluc, reason)."""
    if not text or not text.strip():
        return True, "empty"
    stripped = text.strip()
    if stripped == BIAS_PROMPT.strip():
        return True, "bias_prompt_echo"
    for sub in YOUTUBE_HALLUCINATION_SUBSTRINGS:
        if sub in stripped:
            return True, f"youtube_pattern:{sub}"
    if SHORT_THRESHOLD_CHARS > 0 and len(stripped) < SHORT_THRESHOLD_CHARS:
        return True, "too_short"
    return False, ""


def filter_jsonl(src: Path, dst: Path) -> tuple[int, int, dict[str, int]]:
    """Copy src to dst, dropping hallucinated records. Returns
    (kept, dropped, reason_counts)."""
    kept = 0
    dropped = 0
    reasons: dict[str, int] = {}
    with open(src) as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            halluc, reason = is_hallucination(rec.get("text", ""))
            if halluc:
                dropped += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped, reasons


def load_user_bad_audios(persona: str) -> set[str]:
    """Read /ui/v13_review verdicts.csv and return the set of audio rel-paths
    the user marked 'bad'. These are dropped on top of the pattern filter."""
    csv_path = ROOT / "data/training" / f"{persona}_v13" / "verdicts.csv"
    if not csv_path.exists():
        return set()
    import csv as _csv
    bads: set[str] = set()
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            if row.get("verdict") == "bad":
                bads.add(row["audio"])
    if bads:
        log.info("Loaded %d user-flagged Bad audios from verdicts.csv", len(bads))
    return bads


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True)
    args = p.parse_args()

    out_root = ROOT / "data/training" / f"{args.persona}_v13"
    sources = [
        ("train.jsonl", "train_filtered.jsonl"),
        ("train_prepared.jsonl", "train_prepared_filtered.jsonl"),
    ]

    user_bads = load_user_bad_audios(args.persona)

    for src_name, dst_name in sources:
        src = out_root / src_name
        dst = out_root / dst_name
        if not src.exists():
            log.warning("source not found, skipping: %s", src)
            continue
        kept, dropped, reasons = filter_jsonl(src, dst)
        # Second pass: drop user-flagged bads.
        if user_bads:
            import json as _json
            keep_lines = []
            ubad_dropped = 0
            with open(dst) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = _json.loads(line)
                    if rec["audio"] in user_bads:
                        ubad_dropped += 1
                        continue
                    keep_lines.append(line)
            with open(dst, "w", encoding="utf-8") as f:
                for line in keep_lines:
                    f.write(line + "\n")
            if ubad_dropped:
                reasons[f"user_verdict_bad"] = ubad_dropped
                kept -= ubad_dropped
                dropped += ubad_dropped
        log.info("%s → %s: kept=%d, dropped=%d", src_name, dst_name, kept, dropped)
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            log.info("    %4d  %s", c, r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
