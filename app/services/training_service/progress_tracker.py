"""
Training progress tracking.

Writes progress to disk (progress.json) for SSE streaming and断线重连.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Time estimation constants
TIME_PER_AUDIO_SECOND = 0.5  # seconds per audio second per epoch (RTX 4090 baseline)
OVERHEAD_FACTOR = 1.3


@dataclass
class TrainingProgress:
    """Training progress state."""
    version_id: str
    status: str = "training"  # training/ready/failed
    current_epoch: int = 0
    total_epochs: int = 10
    current_loss: float = 0.0
    best_loss: float = float('inf')
    epoch_times: list[float] = field(default_factory=list)
    elapsed_seconds: int = 0
    eta_seconds: int = 0
    progress_pct: int = 0
    last_updated: Optional[str] = None
    error_message: Optional[str] = None
    total_audio_duration: float = 0.0  # seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["progress_pct"] = self.progress_pct
        # Replace inf/-inf with None for JSON serialization
        if d.get("best_loss") == float('inf'):
            d["best_loss"] = None
        return d


class ProgressTracker:
    """Tracks and persists training progress."""

    def __init__(self, version_id: str, version_dir: Path, total_epochs: int, total_audio_duration: float):
        self.version_id = version_id
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.version_dir / "progress.json"
        self.total_epochs = total_epochs
        self.total_audio_duration = total_audio_duration
        self._epoch_start_time: Optional[float] = None
        self._start_time: float = time.time()

        # Initialize progress
        self._progress = TrainingProgress(
            version_id=version_id,
            total_epochs=total_epochs,
            total_audio_duration=total_audio_duration,
            last_updated=datetime.now().isoformat(),
        )
        self._save()

    def _save(self):
        """Save progress to disk."""
        self._progress.last_updated = datetime.now().isoformat()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self._progress.to_dict(), f, ensure_ascii=False, indent=2)

    def _calculate_eta(self) -> int:
        """Calculate ETA based on actual epoch times or estimation."""
        if self._progress.current_epoch >= 2 and self._progress.epoch_times:
            avg_time = sum(self._progress.epoch_times) / len(self._progress.epoch_times)
            remaining_epochs = self.total_epochs - self._progress.current_epoch
            eta = avg_time * remaining_epochs
        else:
            # Fallback: estimate based on audio duration
            estimated_per_epoch = (
                self.total_audio_duration * TIME_PER_AUDIO_SECOND * OVERHEAD_FACTOR
            )
            eta = estimated_per_epoch * (self.total_epochs - self._progress.current_epoch)
        return max(0, int(eta))

    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self._epoch_start_time = time.time()
        self._progress.current_epoch = epoch
        self._progress.progress_pct = int((epoch / self.total_epochs) * 100)
        self._progress.elapsed_seconds = int(time.time() - self._start_time)
        self._progress.eta_seconds = self._calculate_eta()
        self._save()
        logger.info(f"[TRAINING:{self.version_id[:8]}] Epoch {epoch}/{self.total_epochs} started")

    def update_loss(self, loss: float):
        """Update current loss."""
        self._progress.current_loss = loss
        if loss < self._progress.best_loss:
            self._progress.best_loss = loss
        self._save()

    def complete_epoch(self, epoch: int, loss: float):
        """Mark an epoch as complete."""
        if self._epoch_start_time is not None:
            epoch_time = time.time() - self._epoch_start_time
            self._progress.epoch_times.append(epoch_time)
            self._epoch_start_time = None

        self._progress.current_loss = loss
        if loss < self._progress.best_loss:
            self._progress.best_loss = loss

        self._progress.current_epoch = epoch
        self._progress.progress_pct = int((epoch / self.total_epochs) * 100)
        self._progress.elapsed_seconds = int(time.time() - self._start_time)
        self._progress.eta_seconds = self._calculate_eta()
        self._save()

        logger.info(
            f"[TRAINING:{self.version_id[:8]}] Epoch {epoch}/{self.total_epochs} "
            f"complete, loss={loss:.4f}, eta={self._progress.eta_seconds}s"
        )

    def complete(self, final_loss: float, training_time_seconds: int):
        """Mark training as complete."""
        self._progress.status = "ready"
        self._progress.current_loss = final_loss
        self._progress.best_loss = min(final_loss, self._progress.best_loss)
        self._progress.progress_pct = 100
        self._progress.elapsed_seconds = training_time_seconds
        self._progress.eta_seconds = 0
        self._save()
        logger.info(
            f"[TRAINING:{self.version_id[:8]}] Training complete, "
            f"final_loss={final_loss:.4f}, time={training_time_seconds}s"
        )

    def fail(self, error_message: str):
        """Mark training as failed."""
        self._progress.status = "failed"
        self._progress.error_message = error_message
        self._progress.eta_seconds = 0
        self._save()
        logger.error(f"[TRAINING:{self.version_id[:8]}] Training failed: {error_message}")

    def get_progress(self) -> TrainingProgress:
        """Get current progress."""
        return self._progress

    @staticmethod
    def load(version_id: str, version_dir: Path) -> Optional[TrainingProgress]:
        """Load progress from disk."""
        progress_file = Path(version_dir) / "progress.json"
        if not progress_file.exists():
            return None
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TrainingProgress(**data)
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return None

    @staticmethod
    def estimate_training_time(audio_duration: float, num_epochs: int) -> int:
        """Estimate total training time in seconds."""
        return int(audio_duration * num_epochs * TIME_PER_AUDIO_SECOND * OVERHEAD_FACTOR)
