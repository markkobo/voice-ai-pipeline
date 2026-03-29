"""Unit tests for recording metadata."""

import pytest
import json
import tempfile
from pathlib import Path

from app.services.recordings.file_storage import RecordingPaths
from app.services.recordings.metadata import (
    RecordingMetadata,
    PROCESSING_EXPIRY_DAYS,
)


class TestRecordingMetadata:
    """Test RecordingMetadata class."""

    @pytest.fixture
    def temp_recording(self, tmp_path):
        """Create a temporary recording folder for testing."""
        # Override data directory to temp path
        import app.services.recordings.file_storage as fs
        original_raw = fs.RAW_DIR
        fs.RAW_DIR = tmp_path / "raw"
        fs.RAW_DIR.mkdir(parents=True, exist_ok=True)

        rp = RecordingPaths(listener_id="child", persona_id="xiao_s")
        rp.create_folders()

        yield rp

        # Restore original
        fs.RAW_DIR = original_raw

    def test_create_new_metadata(self, temp_recording):
        """Test creating new metadata."""
        metadata = RecordingMetadata(temp_recording)

        assert metadata.data["recording_id"] == temp_recording.recording_id
        assert metadata.data["listener_id"] == "child"
        assert metadata.data["persona_id"] == "xiao_s"
        assert metadata.data["status"] == "raw"
        assert metadata.data["transcription"]["text"] is None

    def test_save_metadata(self, temp_recording):
        """Test saving metadata to file."""
        metadata = RecordingMetadata(temp_recording)
        metadata.save()

        assert temp_recording.metadata_path.exists()

        # Load and verify
        with open(temp_recording.metadata_path) as f:
            loaded = json.load(f)
        assert loaded["recording_id"] == metadata.data["recording_id"]

    def test_update_status(self, temp_recording):
        """Test updating recording status."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_status("processing")

        assert metadata.data["status"] == "processing"

    def test_update_status_to_processed_sets_expiry(self, temp_recording):
        """Test that status=processed sets expiry date."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_status("processed")

        assert metadata.data["status"] == "processed"
        assert metadata.data["processed_at"] is not None
        assert metadata.data["processed_expires_at"] is not None

    def test_update_processing_step(self, temp_recording):
        """Test updating a processing step."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_processing_step("denoise", "in_progress")

        step = metadata.data["processing_steps"]["denoise"]
        assert step["status"] == "in_progress"
        assert step["started_at"] is not None

    def test_update_processing_step_done(self, temp_recording):
        """Test updating processing step to done."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_processing_step("enhance", "done", duration_ms=3500)

        step = metadata.data["processing_steps"]["enhance"]
        assert step["status"] == "done"
        assert step["completed_at"] is not None
        assert metadata.data["pipeline_metrics"]["enhance_ms"] == 3500

    def test_update_processing_step_failed(self, temp_recording):
        """Test updating processing step to failed."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_processing_step("transcribe", "failed", error_message="ASR timeout")

        step = metadata.data["processing_steps"]["transcribe"]
        assert step["status"] == "failed"
        assert step["error_message"] == "ASR timeout"

    def test_update_quality_metrics(self, temp_recording):
        """Test updating quality metrics."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_quality_metrics({
            "snr_db": 25.5,
            "rms_volume": -12.3,
            "silence_ratio": 0.1,
            "clarity_score": 0.85,
        })

        qm = metadata.data["quality_metrics"]
        assert qm["snr_db"] == 25.5
        assert qm["rms_volume"] == -12.3
        assert qm["clarity_score"] == 0.85
        # SNR > 15 and clarity > 0.6 should set training_ready = True
        assert qm["training_ready"] is True

    def test_update_quality_metrics_below_threshold(self, temp_recording):
        """Test that low quality sets training_ready = False."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_quality_metrics({
            "snr_db": 10.0,  # Below 15
            "clarity_score": 0.5,  # Below 0.6
        })

        assert metadata.data["quality_metrics"]["training_ready"] is False

    def test_update_transcription(self, temp_recording):
        """Test updating transcription."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_transcription(
            text="這是測試文字稿",
            confidence=0.95,
            segments=[{"start": 0.0, "end": 5.0, "text": "這是測試"}],
        )

        assert metadata.data["transcription"]["text"] == "這是測試文字稿"
        assert metadata.data["transcription"]["confidence"] == 0.95
        assert len(metadata.data["transcription"]["segments"]) == 1

    def test_save_transcription_text(self, temp_recording):
        """Test saving transcription as plain text."""
        metadata = RecordingMetadata(temp_recording)
        metadata.save_transcription_text("這是測試文字稿")

        assert temp_recording.transcription_path.exists()
        with open(temp_recording.transcription_path) as f:
            assert f.read() == "這是測試文字稿"

    def test_update_audio_info(self, temp_recording):
        """Test updating audio info."""
        metadata = RecordingMetadata(temp_recording)
        metadata.update_audio_info(duration_seconds=45.5, file_size_bytes=4500000)

        assert metadata.data["duration_seconds"] == 45.5
        assert metadata.data["file_size_bytes"] == 4500000

    def test_add_error(self, temp_recording):
        """Test recording an error."""
        metadata = RecordingMetadata(temp_recording)
        metadata.add_error("Test error message")

        assert metadata.data["status"] == "failed"
        assert metadata.data["error_message"] == "Test error message"

    def test_is_expired_false_when_not_processed(self, temp_recording):
        """Test is_expired returns False when not processed."""
        metadata = RecordingMetadata(temp_recording)
        assert metadata.is_expired() is False

    def test_processing_expiry_days(self):
        """Test that PROCESSING_EXPIRY_DAYS is 3."""
        assert PROCESSING_EXPIRY_DAYS == 3
