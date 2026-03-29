"""Unit tests for recording file storage."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from app.services.recordings.file_storage import (
    RecordingPaths,
    get_recording_by_folder,
    list_all_recordings,
    get_storage_stats,
)


class TestRecordingPaths:
    """Test RecordingPaths class."""

    def test_create_with_valid_ids(self):
        """Test creating RecordingPaths with valid IDs."""
        rp = RecordingPaths(listener_id="child", persona_id="xiao_s")
        assert rp.listener_id == "child"
        assert rp.persona_id == "xiao_s"
        assert rp.timestamp is not None
        assert rp.folder_name.startswith("child_xiao_s_")

    def test_create_with_custom_timestamp(self):
        """Test creating RecordingPaths with custom timestamp."""
        rp = RecordingPaths(
            listener_id="mom",
            persona_id="caregiver",
            timestamp="20260329_120000",
        )
        assert rp.timestamp == "20260329_120000"
        assert rp.folder_name == "mom_caregiver_20260329_120000"

    def test_invalid_listener_id(self):
        """Test that invalid listener_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid listener_id"):
            RecordingPaths(listener_id="invalid", persona_id="xiao_s")

    def test_invalid_persona_id(self):
        """Test that invalid persona_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid persona_id"):
            RecordingPaths(listener_id="child", persona_id="invalid")

    def test_folder_paths(self):
        """Test that all folder paths are correct."""
        rp = RecordingPaths(listener_id="child", persona_id="xiao_s", timestamp="20260329")
        assert "raw" in str(rp.raw_folder)
        assert "denoised" in str(rp.denoised_folder)
        assert "enhanced" in str(rp.enhanced_folder)

    def test_audio_paths(self):
        """Test that audio file paths are correct."""
        rp = RecordingPaths(listener_id="child", persona_id="xiao_s", timestamp="20260329")
        assert rp.raw_audio_path.name == "audio.wav"
        assert rp.denoised_audio_path.name == "audio.wav"
        assert rp.enhanced_audio_path.name == "audio.wav"

    def test_metadata_path_in_raw_folder(self):
        """Test that metadata is stored in raw folder."""
        rp = RecordingPaths(listener_id="child", persona_id="xiao_s", timestamp="20260329")
        assert rp.metadata_path.parent == rp.raw_folder
        assert rp.metadata_path.name == "metadata.json"

    def test_transcription_path_in_raw_folder(self):
        """Test that transcription is stored in raw folder."""
        rp = RecordingPaths(listener_id="child", persona_id="xiao_s", timestamp="20260329")
        assert rp.transcription_path.parent == rp.raw_folder
        assert rp.transcription_path.name == "transcription.txt"


class TestGetRecordingByFolder:
    """Test get_recording_by_folder function."""

    def test_parse_valid_folder_name(self):
        """Test parsing valid folder name."""
        rp = get_recording_by_folder("child_xiao_s_20260329_120000")
        assert rp is not None
        assert rp.listener_id == "child"
        assert rp.persona_id == "xiao_s"
        assert rp.timestamp == "20260329_120000"

    def test_parse_mom_recording(self):
        """Test parsing mom recording."""
        rp = get_recording_by_folder("mom_xiao_s_20260328")
        assert rp is not None
        assert rp.listener_id == "mom"
        assert rp.persona_id == "xiao_s"

    def test_parse_invalid_folder(self):
        """Test parsing invalid folder name returns None."""
        assert get_recording_by_folder("invalid_folder") is None

    def test_parse_missing_timestamp(self):
        """Test parsing folder with missing timestamp."""
        assert get_recording_by_folder("child_xiao_s") is None


class TestStorageStats:
    """Test get_storage_stats function."""

    def test_empty_stats(self):
        """Test stats when no recordings exist."""
        stats = get_storage_stats()
        assert "raw_size_bytes" in stats
        assert "denoised_size_bytes" in stats
        assert "enhanced_size_bytes" in stats
        assert "total_recordings" in stats
        assert stats["total_recordings"] >= 0
