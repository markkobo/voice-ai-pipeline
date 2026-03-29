"""Unit tests for training version management."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.training import (
    TrainingVersion,
    ActiveVersion,
    VersionManager,
    get_version_manager,
)


class TestTrainingVersion:
    """Test TrainingVersion dataclass."""

    def test_training_version_creation(self):
        """Test creating a TrainingVersion."""
        version = TrainingVersion(
            version_id="v1_20260329_143022",
            persona_id="xiao_s",
            status="training",
            num_recordings_used=5,
        )
        assert version.version_id == "v1_20260329_143022"
        assert version.persona_id == "xiao_s"
        assert version.status == "training"
        assert version.num_recordings_used == 5
        assert version.base_model == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        assert version.rank == 16
        assert version.learning_rate == 1e-4
        assert version.num_epochs == 10

    def test_training_version_to_dict(self):
        """Test converting to dict."""
        version = TrainingVersion(
            version_id="v1_20260329_143022",
            persona_id="xiao_s",
            status="ready",
        )
        d = version.to_dict()
        assert d["version_id"] == "v1_20260329_143022"
        assert d["persona_id"] == "xiao_s"
        assert d["status"] == "ready"


class TestActiveVersion:
    """Test ActiveVersion dataclass."""

    def test_active_version_creation(self):
        """Test creating an ActiveVersion."""
        active = ActiveVersion(persona_id="xiao_s", version_id="v1_20260329_143022")
        assert active.persona_id == "xiao_s"
        assert active.version_id == "v1_20260329_143022"


class TestVersionManager:
    """Test VersionManager class."""

    def setup_method(self):
        """Set up test fixtures with temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_file = Path(self.temp_dir) / "index.json"

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_manager(self) -> VersionManager:
        """Create a VersionManager with temp index file."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                return manager

    def test_create_version(self):
        """Test creating a new version."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                version = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])

                assert version.persona_id == "xiao_s"
                assert version.status == "training"
                assert len(version.recording_ids_used) == 5
                assert version.recording_ids_used == ["r1", "r2", "r3", "r4", "r5"]
                assert version.created_at is not None
                assert "v1_" in version.version_id

    def test_get_version(self):
        """Test getting a version by ID."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                created = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])

                retrieved = manager.get_version(created.version_id)
                assert retrieved is not None
                assert retrieved.version_id == created.version_id

    def test_get_version_not_found(self):
        """Test getting a non-existent version."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                result = manager.get_version("nonexistent")
                assert result is None

    def test_list_versions(self):
        """Test listing versions."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                v1 = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])
                v2 = manager.create_version("caregiver", recording_ids=["r1", "r2", "r3"])

                all_versions = manager.list_versions()
                assert len(all_versions) == 2

                xiao_s_versions = manager.list_versions("xiao_s")
                assert len(xiao_s_versions) == 1
                assert xiao_s_versions[0].persona_id == "xiao_s"

    def test_set_active_version(self):
        """Test setting active version."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                version = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])

                # Cannot activate before ready
                result = manager.set_active_version(version.version_id)
                assert result is False

                # Mark as ready
                manager.update_version_status(version.version_id, "ready")

                # Now can activate
                result = manager.set_active_version(version.version_id)
                assert result is True

                active = manager.get_active_version("xiao_s")
                assert active is not None
                assert active.version_id == version.version_id

    def test_update_version_status(self):
        """Test updating version status."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                version = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])

                manager.update_version_status(
                    version.version_id,
                    status="ready",
                    final_loss=0.123,
                    training_time_seconds=3600,
                )

                updated = manager.get_version(version.version_id)
                assert updated.status == "ready"
                assert updated.final_loss == 0.123
                assert updated.training_time_seconds == 3600
                assert updated.completed_at is not None

    def test_delete_version(self):
        """Test deleting a version."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                version = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])

                result = manager.delete_version(version.version_id)
                assert result is True

                deleted = manager.get_version(version.version_id)
                assert deleted is None

    def test_delete_active_version_fails(self):
        """Test that deleting active version fails."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()
                version = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])
                manager.update_version_status(version.version_id, "ready")
                manager.set_active_version(version.version_id)

                result = manager.delete_version(version.version_id)
                assert result is False

                # Version should still exist
                retrieved = manager.get_version(version.version_id)
                assert retrieved is not None

    def test_get_training_status(self):
        """Test getting training status."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                manager = VersionManager()

                # No training
                status = manager.get_training_status()
                assert status["is_training"] is False

                # With training
                version = manager.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])
                status = manager.get_training_status()
                assert status["is_training"] is True
                assert status["version_id"] == version.version_id
                assert status["persona_id"] == "xiao_s"

    def test_persistence(self):
        """Test that versions persist across manager instances."""
        with patch("app.services.training.MODELS_DIR", Path(self.temp_dir)):
            with patch("app.services.training.VERSION_INDEX_FILE", self.index_file):
                # Create and save a version
                manager1 = VersionManager()
                v1 = manager1.create_version("xiao_s", recording_ids=["r1", "r2", "r3", "r4", "r5"])
                manager1.update_version_status(v1.version_id, "ready")

                # Load in new manager
                manager2 = VersionManager()
                assert len(manager2.list_versions()) == 1

                retrieved = manager2.get_version(v1.version_id)
                assert retrieved is not None
                assert retrieved.status == "ready"
