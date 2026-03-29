"""
Recording file storage utilities.

Manages folder structure and file paths for recordings.
Folder naming: {listener_id}_{persona_id}_{timestamp}/
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid


# Base data directory
DATA_DIR = Path("/workspace/voice-ai-pipeline-1/data")
RECORDINGS_DIR = DATA_DIR / "recordings"
RAW_DIR = RECORDINGS_DIR / "raw"
DENOISED_DIR = RECORDINGS_DIR / "denoised"
ENHANCED_DIR = RECORDINGS_DIR / "enhanced"
VOICE_PROFILES_DIR = DATA_DIR / "voice_profiles"
MODELS_DIR = DATA_DIR / "models"

# Recording index cache for fast lookup
RECORDINGS_INDEX_FILE = RECORDINGS_DIR / "index.json"
_recordings_cache: Optional[dict] = None

# Valid IDs (also defined as class attributes in RecordingPaths)
VALID_LISTENER_IDS = {"child", "mom", "dad", "friend", "reporter", "elder", "default"}
VALID_PERSONA_IDS = {"xiao_s", "caregiver", "elder_gentle", "elder_playful"}


class RecordingPaths:
    """Manages file paths for a single recording."""

    VALID_LISTENER_IDS = {"child", "mom", "dad", "friend", "reporter", "elder", "default"}
    VALID_PERSONA_IDS = {"xiao_s", "caregiver", "elder_gentle", "elder_playful"}

    def __init__(
        self,
        listener_id: str,
        persona_id: str,
        timestamp: Optional[str] = None,
        recording_id: Optional[str] = None,
    ):
        if listener_id not in self.VALID_LISTENER_IDS:
            raise ValueError(f"Invalid listener_id: {listener_id}")
        if persona_id not in self.VALID_PERSONA_IDS:
            raise ValueError(f"Invalid persona_id: {persona_id}")

        self.listener_id = listener_id
        self.persona_id = persona_id
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_id = recording_id or str(uuid.uuid4())
        self.folder_name = f"{listener_id}_{persona_id}_{self.timestamp}"

    @property
    def raw_folder(self) -> Path:
        return RAW_DIR / self.folder_name

    @property
    def denoised_folder(self) -> Path:
        return DENOISED_DIR / self.folder_name

    @property
    def enhanced_folder(self) -> Path:
        return ENHANCED_DIR / self.folder_name

    @property
    def raw_audio_path(self) -> Path:
        return self.raw_folder / "audio.wav"

    @property
    def denoised_audio_path(self) -> Path:
        return self.denoised_folder / "audio.wav"

    @property
    def enhanced_audio_path(self) -> Path:
        return self.enhanced_folder / "audio.wav"

    @property
    def metadata_path(self) -> Path:
        """Metadata stored in raw folder (same as WAV)."""
        return self.raw_folder / "metadata.json"

    @property
    def transcription_path(self) -> Path:
        """Transcription as plain text (easy to move/copy)."""
        return self.raw_folder / "transcription.txt"

    def create_folders(self) -> None:
        """Create all recording folders."""
        for folder in [self.raw_folder, self.denoised_folder, self.enhanced_folder]:
            folder.mkdir(parents=True, exist_ok=True)

    def save_audio(self, source_path: Path, stage: str = "raw") -> Path:
        """
        Copy audio file to appropriate folder.

        Args:
            source_path: Path to source audio file
            stage: "raw", "denoised", or "enhanced"
        """
        if stage == "raw":
            dest = self.raw_audio_path
        elif stage == "denoised":
            dest = self.denoised_audio_path
        elif stage == "enhanced":
            dest = self.enhanced_audio_path
        else:
            raise ValueError(f"Invalid stage: {stage}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        return dest

    def delete_all(self) -> None:
        """Delete all folders and files for this recording."""
        for folder in [self.raw_folder, self.denoised_folder, self.enhanced_folder]:
            if folder.exists():
                shutil.rmtree(folder)


def get_recording_by_folder(folder_name: str) -> Optional[RecordingPaths]:
    """
    Parse folder name to get RecordingPaths object.

    Folder format: {listener_id}_{persona_id}_{timestamp}
    Note: persona_id can contain underscores (e.g., "xiao_s")
    """
    parts = folder_name.split("_")
    if len(parts) < 3:
        return None

    # listener_id is always the first part (no underscores)
    listener_id = parts[0]
    if listener_id not in RecordingPaths.VALID_LISTENER_IDS:
        return None

    # Find persona_id after listener_id (can contain underscores)
    # persona_id must be in VALID_PERSONA_IDS
    # Try combining remaining parts until we find a valid persona_id
    for i in range(1, len(parts)):
        candidate_parts = parts[1:i + 1]
        persona_candidate = "_".join(candidate_parts)
        if persona_candidate in RecordingPaths.VALID_PERSONA_IDS:
            # Found persona_id at parts[1:i+1]
            timestamp_parts = parts[i + 1:]
            if len(timestamp_parts) == 0:
                return None
            # Timestamp is everything after persona_id
            # Format: YYYYMMDD_HHMMSS or YYYYMMDD
            timestamp = "_".join(timestamp_parts)

            return RecordingPaths(
                listener_id=listener_id,
                persona_id=persona_candidate,
                timestamp=timestamp,
            )

    return None


def _load_recordings_cache() -> dict:
    """Load recordings index from file."""
    global _recordings_cache
    if _recordings_cache is not None:
        return _recordings_cache

    if RECORDINGS_INDEX_FILE.exists():
        with open(RECORDINGS_INDEX_FILE, "r", encoding="utf-8") as f:
            _recordings_cache = json.load(f)
    else:
        _recordings_cache = {"recordings": []}

    return _recordings_cache


def _save_recordings_cache():
    """Save recordings index to file."""
    global _recordings_cache
    if _recordings_cache is None:
        return
    with open(RECORDINGS_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(_recordings_cache, f, ensure_ascii=False, indent=2)


def invalidate_recordings_cache():
    """Invalidate the recordings cache. Call after create/delete."""
    global _recordings_cache
    _recordings_cache = None


def list_all_recordings() -> list[RecordingPaths]:
    """
    List all recordings in the raw folder directory.

    Uses an index cache for fast repeated lookups.
    Cache is invalidated on recording create/delete.
    """
    # Rebuild from cache
    cache = _load_recordings_cache()

    # If cache has recordings, use it
    if cache.get("recordings"):
        recordings = []
        for rec in cache["recordings"]:
            try:
                rp = RecordingPaths(
                    listener_id=rec["listener_id"],
                    persona_id=rec["persona_id"],
                    timestamp=rec["timestamp"],
                    recording_id=rec["recording_id"],
                )
                recordings.append(rp)
            except (ValueError, KeyError):
                # Skip invalid entries, trigger cache rebuild
                invalidate_recordings_cache()
                return list_all_recordings()
        recordings.sort(key=lambda r: r.timestamp, reverse=True)
        return recordings

    # Cache miss or empty - rebuild from filesystem
    recordings = []
    if not RAW_DIR.exists():
        return recordings

    for folder in RAW_DIR.iterdir():
        if folder.is_dir():
            rp = get_recording_by_folder(folder.name)
            if rp:
                recordings.append(rp)

    # Sort by timestamp descending (newest first)
    recordings.sort(key=lambda r: r.timestamp, reverse=True)

    # Update cache
    _recordings_cache = {
        "recordings": [
            {
                "recording_id": r.recording_id,
                "listener_id": r.listener_id,
                "persona_id": r.persona_id,
                "timestamp": r.timestamp,
                "folder_name": r.folder_name,
            }
            for r in recordings
        ]
    }
    _save_recordings_cache()

    return recordings


def register_recording_in_cache(paths: RecordingPaths):
    """Register a new recording in the cache index."""
    cache = _load_recordings_cache()
    cache["recordings"].append({
        "recording_id": paths.recording_id,
        "listener_id": paths.listener_id,
        "persona_id": paths.persona_id,
        "timestamp": paths.timestamp,
        "folder_name": paths.folder_name,
    })
    _save_recordings_cache()


def unregister_recording_from_cache(recording_id: str):
    """Unregister a recording from the cache index."""
    cache = _load_recordings_cache()
    cache["recordings"] = [
        r for r in cache["recordings"] if r.get("recording_id") != recording_id
    ]
    _save_recordings_cache()


def get_storage_stats() -> dict:
    """Get storage statistics for recordings."""
    stats = {
        "raw_size_bytes": 0,
        "denoised_size_bytes": 0,
        "enhanced_size_bytes": 0,
        "total_recordings": 0,
    }

    if RAW_DIR.exists():
        for folder in RAW_DIR.iterdir():
            if folder.is_dir():
                stats["total_recordings"] += 1
                for f in folder.rglob("*"):
                    if f.is_file():
                        stats["raw_size_bytes"] += f.stat().st_size

    if DENOISED_DIR.exists():
        for folder in DENOISED_DIR.iterdir():
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.is_file():
                        stats["denoised_size_bytes"] += f.stat().st_size

    if ENHANCED_DIR.exists():
        for folder in ENHANCED_DIR.iterdir():
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.is_file():
                        stats["enhanced_size_bytes"] += f.stat().st_size

    return stats
