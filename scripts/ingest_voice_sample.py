#!/usr/bin/env python3
"""
Ingest voice_sample files (podcast + IG audio) as recordings.

Usage:
    python scripts/ingest_voice_sample.py [--process]

Without --process: only ingests files
With --process: ingests AND runs processing pipeline on all files (via API)
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Install deps if missing
try:
    import numpy as np
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "-q"])
    import numpy as np

import soundfile as sf

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data dirs
DATA_DIR = PROJECT_ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
RAW_DIR = RECORDINGS_DIR / "raw"
DENOISED_DIR = RECORDINGS_DIR / "denoised"
ENHANCED_DIR = RECORDINGS_DIR / "enhanced"
INDEX_FILE = RECORDINGS_DIR / "index.json"
VOICE_SAMPLE_DIR = PROJECT_ROOT / "voice_sample"

# Valid IDs
VALID_LISTENER_IDS = {"child", "mom", "dad", "friend", "reporter", "elder", "default"}
VALID_PERSONA_IDS = {"xiao_s", "caregiver", "elder_gentle", "elder_playful"}


def get_audio_duration(path: Path) -> float:
    """Get duration of audio file in seconds."""
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception as e:
        print(f"  Warning: could not get audio duration: {e}")
        return 0.0


def extract_title_from_filename(filename: str) -> str:
    """Extract readable title from filename.

    Filename: "10_老娘的老娘 Vol 03 女性独立 S妈:真正的独立是好坏都能自己承担 [pxYqouCt58E]_(Vocals).wav"
    """
    name = filename
    for ext in [".wav", ".mp4", ".mp3", ".m4a"]:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break

    # Remove (Vocals) suffix
    name = re.sub(r"\s*_\(?Vocals\)?\s*$", "", name, flags=re.IGNORECASE).strip()

    # Remove YouTube/video ID in brackets at end
    name = re.sub(r'\s*\[[^\]]+\]\s*$', '', name)

    # Remove leading number prefix like "10_" or "1_"
    name = re.sub(r'^\d+_', '', name)

    name = name.strip()

    if not name:
        return "Voice sample"

    # Truncate if too long
    if len(name) > 80:
        name = name[:80] + "..."

    return name


def load_index() -> dict:
    """Load recordings index."""
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"recordings": []}


def save_index(index: dict):
    """Save recordings index."""
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def is_already_ingested(folder_name: str) -> bool:
    """Check if recording already exists in index."""
    index = load_index()
    return any(r.get("folder_name") == folder_name for r in index["recordings"])


def register_recording(folder_name: str, listener_id: str, persona_id: str, timestamp: str, recording_id: str):
    """Register recording in index."""
    index = load_index()
    index["recordings"].append({
        "recording_id": recording_id,
        "listener_id": listener_id,
        "persona_id": persona_id,
        "timestamp": timestamp,
        "folder_name": folder_name,
    })
    save_index(index)


def create_recording(source_type: str, wav_file: Path) -> dict:
    """Ingest a single voice_sample file as a recording.

    Returns dict with status.
    """
    # Get timestamp from file mtime
    stat = wav_file.stat()
    file_mtime = datetime.fromtimestamp(stat.st_mtime)
    timestamp = file_mtime.strftime("%Y%m%d_%H%M%S")

    listener_id = "default"
    persona_id = "xiao_s"

    # Folder: {listener_id}_{persona_id}_{source_type}_{timestamp}
    folder_name = f"{listener_id}_{persona_id}_{source_type}_{timestamp}"
    recording_id = folder_name

    # Check if already exists
    if is_already_ingested(folder_name):
        return {"status": "skipped", "folder_name": folder_name}

    # Create folder structure
    raw_folder = RAW_DIR / folder_name
    raw_folder.mkdir(parents=True, exist_ok=True)
    (raw_folder / "speakers").mkdir(parents=True, exist_ok=True)

    # Copy audio
    dest_audio = raw_folder / "audio.wav"
    shutil.copy2(str(wav_file), str(dest_audio))
    file_size = wav_file.stat().st_size

    # Get duration
    duration = get_audio_duration(dest_audio)

    # Extract title
    title = extract_title_from_filename(wav_file.name)

    # Create metadata.json
    metadata = {
        "recording_id": recording_id,
        "folder_name": folder_name,
        "listener_id": listener_id,
        "persona_id": persona_id,
        "title": title,
        "duration_seconds": duration,
        "file_size_bytes": file_size,
        "transcription": {"text": None, "confidence": None, "language": "zh", "segments": []},
        "quality_metrics": {
            "snr_db": None, "rms_volume": None, "silence_ratio": None,
            "clarity_score": None, "training_ready": None,
        },
        "status": "raw",
        "processing_steps": {
            "denoise": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
            "enhance": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
            "diarize": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
            "transcribe": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
        },
        "speaker_segments": [],
        "speaker_labels": {},
        "pipeline_metrics": {
            "denoise_ms": None, "enhance_ms": None, "diarize_ms": None,
            "transcribe_ms": None, "total_ms": None,
        },
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "processed_at": None,
        "processed_expires_at": None,
        "error_message": None,
    }

    metadata_path = raw_folder / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Register in index
    register_recording(folder_name, listener_id, persona_id, f"{source_type}_{timestamp}", recording_id)

    return {
        "status": "ok",
        "folder_name": folder_name,
        "title": title,
        "duration": duration,
        "file_size": file_size,
    }


def process_via_api(recording_id: str, base_url: str = "http://localhost:8080") -> bool:
    """Trigger processing via API call."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            f"{base_url}/api/recordings/{recording_id}/process",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("status") == "processing_started"
    except Exception as e:
        print(f"    API error: {e}")
        return False


def ingest_all(dry_run: bool = False) -> list:
    """Ingest all voice_sample files."""
    results = []

    for source_type, source_dir in [("podcast", VOICE_SAMPLE_DIR / "podcast"), ("IG", VOICE_SAMPLE_DIR / "IG")]:
        if not source_dir.exists():
            print(f"[INGEST] {source_type}/ not found")
            continue

        wav_files = sorted(source_dir.glob("*.wav"))
        print(f"[INGEST] {source_type}/: found {len(wav_files)} files")

        for wav_file in wav_files:
            if dry_run:
                print(f"  DRYRUN: {wav_file.name}")
                continue

            result = create_recording(source_type, wav_file)
            if result["status"] == "ok":
                print(f"  OK: {result['folder_name']} | {result['title']} | {result['duration']:.0f}s")
            elif result["status"] == "skipped":
                print(f"  SKIP: {result['folder_name']}")
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Ingest voice_sample files as recordings")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    parser.add_argument("--process", action="store_true", help="Also trigger processing pipeline via API")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between processing API calls (seconds)")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL for --process")
    args = parser.parse_args()

    print(f"=== Voice Sample Ingestion ===")
    print(f"Source: {VOICE_SAMPLE_DIR}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Process after ingest: {'YES' if args.process else 'NO'}")
    print()

    # Ensure dirs exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DENOISED_DIR.mkdir(parents=True, exist_ok=True)
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    # Ingest
    results = ingest_all(dry_run=args.dry_run)

    if args.dry_run:
        return

    ingested = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") == "skipped"]

    print(f"\n=== Done ===")
    print(f"  Ingested: {len(ingested)}")
    print(f"  Skipped: {len(skipped)}")

    if not ingested:
        print("\nNo new files ingested.")
        return

    # Trigger processing if requested
    if args.process:
        # Import here so script still works when server is down (for ingestion)
        import urllib.request
        import urllib.error

        print(f"\n=== Triggering Processing ({len(ingested)} files) ===")
        print(f"Server: {args.server}")
        print("NOTE: GPU is serialized — only one file processes at a time.")
        print("This will take several hours for all files.\n")

        # Check server health
        try:
            with urllib.request.urlopen(f"{args.server}/health", timeout=5) as resp:
                if resp.status != 200:
                    print(f"Server not healthy (status={resp.status}). Skipping processing.")
                    return
        except Exception as e:
            print(f"Server not reachable: {e}. Run './scripts/restart.sh' first, then re-run with --process")
            return

        for i, r in enumerate(ingested, 1):
            folder_name = r["folder_name"]
            print(f"[{i}/{len(ingested)}] Triggering: {folder_name}")

            ok = process_via_api(folder_name, args.server)
            if ok:
                print(f"  -> Processing started")
            else:
                print(f"  -> FAILED to start processing")

            if i < len(ingested):
                print(f"  Waiting {args.delay}s...")
                time.sleep(args.delay)

        print(f"\n=== All Processing Triggered ===")
        print("Monitor progress at: /ui/recordings")


if __name__ == "__main__":
    main()
