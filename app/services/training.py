"""
Training version management.

Manages LoRA training versions for TTS voice cloning.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

# Base directory for models
MODELS_DIR = Path("/workspace/voice-ai-pipeline-1/data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Version index file
VERSION_INDEX_FILE = MODELS_DIR / "index.json"


@dataclass
class TrainingVersion:
    """Represents a training version."""
    version_id: str  # e.g., "v1_20260329_143022"
    persona_id: str
    status: str  # "training", "ready", "failed"
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    lora_path: Optional[str] = None
    rank: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 4
    final_loss: Optional[float] = None
    training_time_seconds: Optional[int] = None
    recording_ids_used: list[str] = field(default_factory=list)
    num_recordings_used: int = 0  # Deprecated: use len(recording_ids_used) instead
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_recordings_used"] = len(self.recording_ids_used)  # Always derive from list
        return d


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

            self._versions = [TrainingVersion(**v) for v in data.get("versions", [])]

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

    def create_version(self, persona_id: str, recording_ids: list[str]) -> TrainingVersion:
        """Create a new training version."""
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_count = sum(1 for v in self._versions if v.persona_id == persona_id)
        version_id = f"v{existing_count + 1}_{timestamp}"

        version = TrainingVersion(
            version_id=version_id,
            persona_id=persona_id,
            status="training",
            recording_ids_used=recording_ids,
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

        # Don't delete if it's the active version
        if self._active_version and self._active_version.version_id == version_id:
            logger.warning(f"[TRAINING] Cannot delete active version: {version_id}")
            return False

        # Remove from list
        self._versions = [v for v in self._versions if v.version_id != version_id]

        # Delete files
        if version.lora_path:
            import shutil
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


# Global version manager instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager
