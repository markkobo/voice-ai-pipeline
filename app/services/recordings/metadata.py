"""
Recording metadata management.

Metadata stored as JSON in same folder as WAV file.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .file_storage import RecordingPaths


# Default processing expires in 3 days
PROCESSING_EXPIRY_DAYS = 3


class RecordingMetadata:
    """Metadata for a single recording."""

    def __init__(self, paths: RecordingPaths):
        self.paths = paths
        self._data: dict = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load existing metadata or create new structure."""
        if self.paths.metadata_path.exists():
            with open(self.paths.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "recording_id": self.paths.recording_id,
            "folder_name": self.paths.folder_name,
            "listener_id": self.paths.listener_id,
            "persona_id": self.paths.persona_id,
            "title": None,
            "duration_seconds": None,
            "file_size_bytes": None,
            "transcription": {
                "text": None,
                "confidence": None,
                "language": "zh",
                "segments": [],
            },
            "quality_metrics": {
                "snr_db": None,
                "rms_volume": None,
                "silence_ratio": None,
                "clarity_score": None,
                "training_ready": None,
            },
            "status": "raw",
            "processing_steps": {
                "denoise": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
                "enhance": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
                "diarize": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
                "transcribe": {"status": "pending", "progress": 0, "error_message": None, "started_at": None, "completed_at": None},
            },
            "speaker_segments": [],
            "pipeline_metrics": {
                "denoise_ms": None,
                "enhance_ms": None,
                "diarize_ms": None,
                "transcribe_ms": None,
                "total_ms": None,
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "processed_at": None,
            "processed_expires_at": None,
            "error_message": None,
        }

    def save(self) -> None:
        """Save metadata to JSON file."""
        self._data["updated_at"] = datetime.now().isoformat()
        self.paths.raw_folder.mkdir(parents=True, exist_ok=True)
        with open(self.paths.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @property
    def data(self) -> dict:
        return self._data

    def update_status(self, status: str) -> None:
        """Update recording status."""
        self._data["status"] = status
        self._data["updated_at"] = datetime.now().isoformat()

        if status == "processed":
            self._data["processed_at"] = datetime.now().isoformat()
            self._data["processed_expires_at"] = (
                datetime.now() + timedelta(days=PROCESSING_EXPIRY_DAYS)
            ).isoformat()

        self.save()

    def update_processing_step(
        self,
        step: str,
        status: str,
        progress: int = 100,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Update a specific processing step status."""
        if step not in self._data["processing_steps"]:
            raise ValueError(f"Invalid step: {step}")

        step_data = self._data["processing_steps"][step]
        step_data["status"] = status
        step_data["progress"] = progress

        if status == "in_progress":
            step_data["started_at"] = datetime.now().isoformat()
        elif status == "done":
            step_data["completed_at"] = datetime.now().isoformat()
        elif status == "skipped":
            # Skipped means step didn't run but pipeline continued
            step_data["error_message"] = error_message or "Step skipped"
            step_data["completed_at"] = datetime.now().isoformat()
        elif status == "failed":
            step_data["error_message"] = error_message
            step_data["completed_at"] = datetime.now().isoformat()

        if duration_ms is not None:
            self._data["pipeline_metrics"][f"{step}_ms"] = duration_ms
            # Calculate total
            ms_values = [v for v in self._data["pipeline_metrics"].values() if v is not None]
            self._data["pipeline_metrics"]["total_ms"] = sum(ms_values)

        self.save()

    def update_quality_metrics(self, metrics: dict) -> None:
        """Update quality metrics."""
        for key, value in metrics.items():
            if key in self._data["quality_metrics"]:
                self._data["quality_metrics"][key] = value

        # Auto-determine training_ready based on thresholds
        snr = self._data["quality_metrics"].get("snr_db")
        clarity = self._data["quality_metrics"].get("clarity_score")

        if snr is not None and clarity is not None:
            # Thresholds: SNR > 15dB, clarity > 0.6
            self._data["quality_metrics"]["training_ready"] = snr > 15 and clarity > 0.6

        self.save()

    def update_transcription(
        self,
        text: str,
        confidence: Optional[float] = None,
        segments: Optional[list] = None,
    ) -> None:
        """Update transcription."""
        self._data["transcription"]["text"] = text
        if confidence is not None:
            self._data["transcription"]["confidence"] = confidence
        if segments is not None:
            self._data["transcription"]["segments"] = segments
        self.save()

    def update_audio_info(self, duration_seconds: float, file_size_bytes: int) -> None:
        """Update audio file info."""
        self._data["duration_seconds"] = duration_seconds
        self._data["file_size_bytes"] = file_size_bytes
        self.save()

    def add_error(self, error_message: str) -> None:
        """Record an error."""
        self._data["status"] = "failed"
        self._data["error_message"] = error_message
        self._data["updated_at"] = datetime.now().isoformat()
        self.save()

    def save_transcription_text(self, text: str) -> None:
        """Save transcription as plain text file."""
        with open(self.paths.transcription_path, "w", encoding="utf-8") as f:
            f.write(text)

    def is_expired(self) -> bool:
        """Check if processed files should be auto-deleted."""
        if self._data["processed_expires_at"] is None:
            return False
        expires = datetime.fromisoformat(self._data["processed_expires_at"])
        return datetime.now() > expires


def load_recording_metadata(folder_name: str) -> Optional[RecordingMetadata]:
    """Load metadata from a recording folder."""
    # Parse folder name to get RecordingPaths
    from .file_storage import get_recording_by_folder

    paths = get_recording_by_folder(folder_name)
    if paths is None:
        return None

    if not paths.metadata_path.exists():
        return None

    return RecordingMetadata(paths)


def list_recordings_metadata() -> list[dict]:
    """List metadata for all recordings."""
    recordings = []
    for paths in list_all_recordings():
        metadata = RecordingMetadata(paths)
        recordings.append(metadata.data)

    # Sort by created_at descending
    recordings.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return recordings
