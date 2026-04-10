"""
Training version management.

Manages LoRA training versions for TTS voice cloning.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

# Base directory for models (configurable via env var)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/voice-ai-pipeline/data/models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Version index file
VERSION_INDEX_FILE = MODELS_DIR / "index.json"


@dataclass
class TrainingVersion:
    """Represents a training version."""
    version_id: str  # e.g., "v1_20260329_143022"
    persona_id: str
    status: str  # "training", "ready", "failed"
    nickname: Optional[str] = None
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    lora_path: Optional[str] = None
    model_type: Optional[str] = None  # "custom_voice" for SFT, None for LoRA
    rank: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 4
    final_loss: Optional[float] = None
    training_time_seconds: Optional[int] = None
    recording_ids_used: list[str] = field(default_factory=list)
    segment_ids_used: list[str] = field(default_factory=list)  # ["recId_speakerId", ...]
    num_recordings_used: int = 0  # Deprecated: use len(recording_ids_used) instead
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_recordings_used"] = len(self.recording_ids_used)
        d["display_name"] = self.nickname or self.version_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingVersion":
        """Create TrainingVersion from dict, filtering extra fields from to_dict()."""
        # Fields that TrainingVersion actually has
        fields = {
            "version_id", "persona_id", "status", "nickname", "base_model",
            "lora_path", "model_type", "rank", "learning_rate", "num_epochs", "batch_size",
            "final_loss", "training_time_seconds", "recording_ids_used",
            "segment_ids_used", "num_recordings_used", "created_at", "completed_at"
        }
        # Filter to only known fields
        filtered = {k: v for k, v in d.items() if k in fields}
        return cls(**filtered)


@dataclass
class ActiveVersion:
    """Represents the currently active version for a persona."""
    persona_id: str
    version_id: str


class VersionManager:
    """Manages training versions."""

    def __init__(self):
        self._versions: list[TrainingVersion] = []
        self._active_version: Optional[ActiveVersion] = None
        self._load_index()

    def _load_index(self):
        """Load version index from file."""
        if VERSION_INDEX_FILE.exists():
            with open(VERSION_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._versions = [TrainingVersion.from_dict(v) for v in data.get("versions", [])]

            active = data.get("active_version")
            if active:
                self._active_version = ActiveVersion(**active)
        else:
            self._versions = []
            self._active_version = None

    def _save_index(self):
        """Save version index to file."""
        data = {
            "versions": [v.to_dict() for v in self._versions],
            "active_version": asdict(self._active_version) if self._active_version else None
        }
        with open(VERSION_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_version(
        self,
        persona_id: str,
        recording_ids: list[str],
        rank: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        batch_size: int = 4,
        segment_ids: list[str] = None,
    ) -> TrainingVersion:
        """Create a new training version."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_count = sum(1 for v in self._versions if v.persona_id == persona_id)
        version_id = f"v{existing_count + 1}_{timestamp}"

        version = TrainingVersion(
            version_id=version_id,
            persona_id=persona_id,
            status="training",
            recording_ids_used=recording_ids,
            segment_ids_used=segment_ids or [],
            rank=rank,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            created_at=datetime.now().isoformat(),
        )

        # Create model directory
        lora_dir = MODELS_DIR / f"{persona_id}_{version_id}"
        lora_dir.mkdir(parents=True, exist_ok=True)
        version.lora_path = str(lora_dir)

        self._versions.append(version)
        self._save_index()

        logger.info(f"[TRAINING] Created new version: {version_id} for {persona_id}")
        return version

    def get_version(self, version_id: str) -> Optional[TrainingVersion]:
        """Get a version by ID."""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None

    def list_versions(self, persona_id: Optional[str] = None) -> list[TrainingVersion]:
        """List all versions, optionally filtered by persona."""
        if persona_id:
            return [v for v in self._versions if v.persona_id == persona_id]
        return self._versions

    def get_active_version(self, persona_id: str) -> Optional[TrainingVersion]:
        """Get the active version for a persona."""
        if self._active_version and self._active_version.persona_id == persona_id:
            return self.get_version(self._active_version.version_id)
        return None

    def set_active_version(self, version_id: str) -> bool:
        """Set a version as active."""
        version = self.get_version(version_id)
        if not version:
            return False

        if version.status != "ready":
            logger.warning(f"[TRAINING] Cannot activate version {version_id}: status is {version.status}")
            return False

        self._active_version = ActiveVersion(
            persona_id=version.persona_id,
            version_id=version.version_id
        )
        self._save_index()

        logger.info(f"[TRAINING] Activated version: {version_id}")
        return True

    def update_version(self, version_id: str, nickname: Optional[str] = None) -> bool:
        """Update version metadata (e.g., nickname)."""
        version = self.get_version(version_id)
        if not version:
            return False
        if nickname is not None:
            version.nickname = nickname
        self._save_index()
        return True

    def update_version_status(
        self,
        version_id: str,
        status: str,
        final_loss: Optional[float] = None,
        training_time_seconds: Optional[int] = None,
    ):
        """Update version status after training."""
        version = self.get_version(version_id)
        if not version:
            logger.error(f"[TRAINING] Version not found: {version_id}")
            return

        version.status = status
        if final_loss is not None:
            version.final_loss = final_loss
        if training_time_seconds is not None:
            version.training_time_seconds = training_time_seconds
        if status == "ready":
            version.completed_at = datetime.now().isoformat()

        self._save_index()
        logger.info(f"[TRAINING] Version {version_id} status updated to {status}")

    def delete_version(self, version_id: str) -> bool:
        """Delete a version."""
        version = self.get_version(version_id)
        if not version:
            return False

        if self._active_version and self._active_version.version_id == version_id:
            logger.warning(f"[TRAINING] Cannot delete active version: {version_id}")
            return False

        self._versions = [v for v in self._versions if v.version_id != version_id]

        if version.lora_path:
            lora_path = Path(version.lora_path)
            if lora_path.exists():
                shutil.rmtree(lora_path)

        self._save_index()
        logger.info(f"[TRAINING] Deleted version: {version_id}")
        return True

    def get_training_status(self) -> dict:
        """Get current training status."""
        training_versions = [v for v in self._versions if v.status == "training"]
        if training_versions:
            v = training_versions[0]
            return {
                "is_training": True,
                "version_id": v.version_id,
                "persona_id": v.persona_id,
                "status": v.status,
            }
        return {"is_training": False}

    def get_version_dir(self, version_id: str) -> Optional[Path]:
        """Get version directory path."""
        version = self.get_version(version_id)
        if version and version.lora_path:
            return Path(version.lora_path)
        return None

    def save_manifest(self, version_id: str, manifest: dict):
        """Save training manifest."""
        version_dir = self.get_version_dir(version_id)
        if not version_dir:
            return
        manifest_path = version_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def get_manifest(self, version_id: str) -> Optional[dict]:
        """Load training manifest."""
        version_dir = self.get_version_dir(version_id)
        if not version_dir:
            return None
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)


# Global version manager instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager


def get_training_audio_for_persona(
    persona_id: str,
    selected_recordings: list[dict],
    segment_ids: list[str] = None,
) -> list[tuple[Path, float, str]]:
    """
    Get training audio paths for a persona using segment_ids.

    Args:
        persona_id: Target persona to train
        selected_recordings: List of recording metadata dicts (for fallback path lookup)
        segment_ids: List of segment identifiers in format "{recording_id}_{speaker_id}"
                     e.g., "uuid123_SPEAKER_00"

    Returns:
        List of (audio_path, duration_seconds, recording_id)
    """
    from app.services.recordings import get_recording_by_folder, RecordingMetadata

    # Build recording lookup by rec_id
    rec_by_id = {rec["recording_id"]: rec for rec in selected_recordings}

    # Build segment lookup: {recId_speakerId: {rec, paths, metadata}}
    audio_files = []

    for seg_id in (segment_ids or []):
        # Parse segment_id = "{recording_id}_{speaker_id}"
        # recording_id is UUID (36 chars: 8-4-4-4-12 with dashes)
        # speaker_id is like SPEAKER_00
        if len(seg_id) < 37:
            logger.warning(f"[TRAINING] Invalid segment_id format (too short): {seg_id}")
            continue

        rec_id = seg_id[:36]  # UUID is first 36 chars
        speaker_id = seg_id[37:]  # After underscore at position 36

        # Find recording
        rec = rec_by_id.get(rec_id)
        if not rec:
            logger.warning(f"[TRAINING] Recording not found for segment: {seg_id}")
            continue

        folder_name = rec.get("folder_name", "")
        paths = get_recording_by_folder(folder_name)
        if not paths:
            logger.warning(f"[TRAINING] Could not find recording folder: {folder_name}")
            continue

        # Get speaker audio path
        speaker_audio = paths.speakers_folder / f"{speaker_id}.wav"
        if not speaker_audio.exists():
            logger.warning(f"[TRAINING] Speaker audio not found: {speaker_audio}")
            continue

        # Get duration from metadata (enriched segments have duration_seconds)
        metadata = RecordingMetadata(paths)
        metadata.reload()
        duration = 30.0  # default fallback
        for seg in metadata.data.get("speaker_segments", []):
            if seg.get("speaker_id") == speaker_id and seg.get("duration_seconds"):
                duration = seg["duration_seconds"]
                break

        audio_files.append((speaker_audio, duration, rec_id))
        logger.info(f"[TRAINING] Added segment: {speaker_id} from {rec_id[:8]} ({duration:.1f}s)")

    return audio_files