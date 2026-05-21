"""Unit tests for recording file storage path helpers.

Note: the legacy index/cache (``list_all_recordings``,
``get_recording_by_folder``, ``register_recording_in_cache`` ...) has been
removed in favor of :class:`JsonRecordingsRepository`. Tests for those
behaviors now live in ``tests/unit/test_recordings_repository.py``. This
file only covers the pure path-math + folder-conventions parts of
``RecordingPaths``.
"""

import pytest

from app.services.recordings.file_storage import (
    RecordingPaths,
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


class TestRecordingPathsFolderNameAPI:
    """Test the preferred ``folder_name=`` construction form.

    This is the path production code should use. It does NOT assign a fresh
    random UUID — ``recording_id`` defaults to ``None`` unless the caller
    passes it explicitly. The historical landmine (UUID drift in
    ``list_all_recordings``) is now structurally impossible from this API.
    """

    def test_construct_from_folder_name_parses_parts(self):
        rp = RecordingPaths(folder_name="child_xiao_s_20260329_120000")
        assert rp.listener_id == "child"
        assert rp.persona_id == "xiao_s"
        assert rp.timestamp == "20260329_120000"
        assert rp.folder_name == "child_xiao_s_20260329_120000"

    def test_folder_name_api_does_not_assign_uuid(self):
        """The new API does NOT generate a UUID — that's the whole point."""
        rp = RecordingPaths(folder_name="child_xiao_s_20260329_120000")
        assert rp.recording_id is None

    def test_folder_name_api_accepts_explicit_recording_id(self):
        rp = RecordingPaths(
            folder_name="child_xiao_s_20260329_120000",
            recording_id="explicit-id-from-repo",
        )
        assert rp.recording_id == "explicit-id-from-repo"

    def test_opaque_folder_name_is_tolerated(self):
        """Folders that don't parse should not raise — pure path math."""
        rp = RecordingPaths(folder_name="totally-opaque-folder")
        assert rp.folder_name == "totally-opaque-folder"
        assert rp.listener_id is None
        assert rp.persona_id is None


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
