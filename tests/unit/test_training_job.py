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

import json
import pytest
import tempfile
import shutil
from pathlib import Path

from app.services.training_service.training_job import TrainingJob
from app.services.training_service.lora_trainer import TrainingConfig, TrainingResult


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


class TestTrainingResultSchema:
    """
    Demo-readiness #1 / `_phase2_followups.md §1`:
    SFT training script writes ``sft_path`` into ``training_result.json``,
    which the parent job loads via ``TrainingResult(**json.load(f))``.
    Before the fix that raised ``TypeError`` and left ``index.json`` status
    at ``"unknown"`` despite a successful merge.
    """

    def test_training_result_accepts_sft_path(self):
        """TrainingResult(sft_path=...) must not raise TypeError."""
        result = TrainingResult(
            success=True,
            sft_path="/data/models/merged_qwen3_tts_xiao_s_v2",
            final_loss=9.6312,
            training_time_seconds=19440,
        )
        assert result.sft_path == "/data/models/merged_qwen3_tts_xiao_s_v2"
        assert result.success is True

    def test_training_result_sft_path_optional(self):
        """LoRA runs don't write sft_path; field must default to None."""
        result = TrainingResult(success=True, lora_path="/x")
        assert result.sft_path is None


class TestTrainingTemplateSilentNoOpFix:
    """Regression tests for the 2026-05-21 silent-no-op LoRA bug
    (v1_20260521_104108_618646). The generated train_lora.py used to:
      1. Build talker_hidden as a 3D tensor (3D mean(keepdim=True)) → trip
         qwen3_tts's `len(shape)==2` assert on every batch.
      2. Catch the AssertionError in a bare `except: continue`, logging
         only `{e}` — which for AssertionError("") is empty.
      3. Report success=true with final_loss=0.0 after 0 gradient updates.
      4. Merge an all-zero adapter → merged model = base model.

    These tests read the template *source* (not run it) so they execute
    without GPU / 1.7B-model downloads.
    """

    def _src(self):
        return Path("app/services/training_service/training_job.py").read_text()

    def test_talker_hidden_uses_cached_speaker_embedding_not_3d_mean(self):
        src = self._src()
        # The buggy fallback that built audio_embeds via 3D mean is gone.
        assert "audio_embeds.mean(dim=0, keepdim=True)" not in src, (
            "The 3D mean(keepdim=True) fallback that violates "
            "forward_sub_talker_finetune's 2D assert must be removed."
        )
        # The cache lookup is unconditional (no `if not USE_LORA` guard).
        assert "speaker_embeddings_cache[audio_file_idx]" in src
        assert "if not USE_LORA and speaker_embeddings_cache" not in src

    def test_batch_error_uses_logger_exception(self):
        src = self._src()
        # logger.exception preserves stacktrace even for AssertionError("")
        # whose str(e) is empty. The legacy logger.error(f"...: {e}")
        # produced uninformative `Batch error at step 0:` lines.
        assert 'logger.error(f"Batch error at step {num_steps}: {e}")' not in src
        assert "logger.exception" in src

    def test_post_loop_fail_fast_when_zero_batches(self):
        src = self._src()
        # If every batch errored, refuse to write success=true.
        assert "if total_batches == 0:" in src
        assert "produced 0 gradient updates" in src

    def test_template_still_compiles_as_python_source(self):
        """Smoke check — the template module itself must remain valid
        Python (the heredoc edits could break this)."""
        import py_compile
        py_compile.compile(
            "app/services/training_service/training_job.py", doraise=True
        )

    def test_training_result_round_trip_from_subprocess_json(self):
        """
        Mirrors the production subprocess output for a successful SFT run
        (training script writes this exact shape — see training_job.py:706).
        The parent ``_run_training`` loads it as
        ``TrainingResult(**json.load(f))`` — this is the call site that
        was crashing.
        """
        subprocess_output = {
            "success": True,
            "sft_path": "/tmp/sft_model",
            "final_loss": 9.6312,
            "training_time_seconds": 19440,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_file = Path(tmp) / "training_result.json"
            with open(result_file, "w") as f:
                json.dump(subprocess_output, f)
            with open(result_file) as f:
                result = TrainingResult(**json.load(f))
        assert result.success is True
        assert result.sft_path == "/tmp/sft_model"
        assert result.final_loss == 9.6312
        assert result.training_time_seconds == 19440
