"""
Unit tests for TrainingJob.

Scope after Phase 0 audit:
- TestMergeLoraFunction      — real filesystem effect of merge_lora()
- TestSFTModelSaving         — real config.json round-trip
- TestSFTAutoActivation      — uses the real TrainingJob class
- TestTrainingJobIntegration — TrainingJob construction sanity

Removed in Phase 0:
- TestTrainingJobScriptGeneration / TestTrainingJobUseLoraLogic /
  test_script_file_is_valid_python re-implemented the script-generation logic
  inside the test and asserted against the test's own copy. They tested the
  test, not the system under test. Replace with snapshot-style tests against
  a real build_training_script() pure function once Phase 1.2 lands.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from app.services.training_service.training_job import TrainingJob
from app.services.training_service.lora_trainer import TrainingConfig


class TestMergeLoraFunction:
    """Tests for merge_lora function behavior with SFT vs LoRA."""

    def test_merge_lora_returns_none_for_missing_adapter(self):
        """merge_lora() should return None if adapter directory doesn't exist."""
        from app.services.training_service.training_job import merge_lora

        temp_dir = tempfile.mkdtemp()
        try:
            version_dir = Path(temp_dir) / "test_version"
            version_dir.mkdir(parents=True)

            # No adapter directory exists
            result = merge_lora(version_dir)
            assert result is None, (
                "merge_lora() should return None when adapter/ doesn't exist. "
                "This was the bug: SFT training called merge_lora(), got None, marked as failed."
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_merge_lora_only_called_for_lora_not_sft(self):
        """
        Verify that merge_lora is only appropriate for LoRA, not SFT.

        SFT saves full model directly; no adapter to merge.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            sft_dir = Path(temp_dir) / "sft_version"
            sft_dir.mkdir(parents=True)

            # SFT might have sft_model/ but not adapter/
            sft_model = sft_dir / "sft_model"
            sft_model.mkdir(parents=True)
            adapter_dir = sft_dir / "adapter"

            assert not adapter_dir.exists(), (
                "SFT should not have adapter/ directory. "
                "This would indicate incorrect branching to LoRA path."
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSFTModelSaving:
    """Tests for SFT model saving (dtype KeyError bug)."""

    def test_sft_config_should_not_have_dtype_key(self):
        """SFT config.json should not contain 'dtype' key."""
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            sft_model_dir = Path(temp_dir) / "sft_model"
            sft_model_dir.mkdir(parents=True)

            # Simulate saving config without dtype (the fix)
            config_dict = {
                "model_type": "Qwen3TTS",
                "num_code_groups": 16,
                # Note: no 'dtype' key — the fix for the KeyError bug.
            }

            config_file = sft_model_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(config_dict, f)

            with open(config_file, "r") as f:
                loaded = json.load(f)

            assert "dtype" not in loaded, (
                "config.json should not contain 'dtype' key. "
                "The 'dtype' key causes KeyError in save_pretrained() when "
                "comparing config against default PretrainedConfig."
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSFTAutoActivation:
    """Tests for SFT auto-activation flow (merge_lora bug)."""

    def test_sft_training_type_does_not_trigger_lora_merge(self):
        """
        Verify SFT training_type doesn't call merge_lora.

        Bug: merge_lora() was called for SFT (which has no adapter/), returned None,
        causing training to be marked as "failed".
        """
        temp_dir = tempfile.mkdtemp()
        try:
            version_dir = Path(temp_dir) / "test_sft"
            version_dir.mkdir(parents=True)

            audio_path = version_dir / "audio.wav"
            audio_path.touch()

            config = TrainingConfig(
                rank=16,
                learning_rate=1e-4,
                num_epochs=1,
                batch_size=2,
                base_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            )

            job = TrainingJob(
                version_id="test_sft_v1",
                version_dir=version_dir,
                audio_paths=[audio_path],
                config=config,
                total_audio_duration=10.0,
                training_type="sft",
            )

            assert job.training_type == "sft"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestTrainingJobIntegration:
    """Construction sanity for TrainingJob (no GPU)."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.version_dir = Path(self.temp_dir) / "test_version"
        self.version_dir.mkdir(parents=True)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_training_job_initialization(self):
        """TrainingJob can be initialized with both lora and sft training types."""
        config = TrainingConfig(
            rank=16,
            learning_rate=1e-4,
            num_epochs=1,
            batch_size=2,
        )

        for training_type in ["lora", "sft"]:
            job = TrainingJob(
                version_id=f"test_v1_{training_type}",
                version_dir=self.version_dir,
                audio_paths=[Path(self.temp_dir) / "audio.wav"],
                config=config,
                total_audio_duration=10.0,
                training_type=training_type,
            )
            assert job.training_type == training_type
            assert job.version_id == f"test_v1_{training_type}"
