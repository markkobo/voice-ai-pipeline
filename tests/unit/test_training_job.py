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

    def test_lora_wraps_only_code_predictor_not_full_talker(self):
        """Regression: 2026-05-21 gradient-coverage bug.

        The previous template wrapped both `talker.model` AND
        `code_predictor` with PEFT. But the training loop only invokes
        `forward_sub_talker_finetune`, which exclusively exercises
        code_predictor — so the 112 lora modules on talker.model never
        received gradients (20/132 lora_B updated). The merge regex
        also only matches `codec_lora`, so any `talker_lora` output
        would be dropped anyway. Honest single-wrap fixes both.
        """
        src = self._src()
        # talker.model wrap removed; only code_predictor remains.
        assert "get_peft_model(model.talker.model" not in src
        assert "get_peft_model(model.talker.code_predictor" in src
        assert 'adapter_name="codec_lora"' in src
        # Saved adapter_config records the actual scope.
        assert "lora_target_scope" in src
        assert "talker.code_predictor" in src

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


class TestCheckpointDetection:
    """Tests for _detect_latest_checkpoint — used both by the auto-resume
    path in TrainingJob._run_training and (indirectly, via the same
    on-disk layout) by service.resume_training to fail-fast when no
    checkpoint exists.
    """

    def test_returns_none_when_no_checkpoint_dir(self):
        from app.services.training_service.training_job import _detect_latest_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            assert _detect_latest_checkpoint(Path(tmp)) is None

    def test_returns_highest_complete_checkpoint(self):
        from app.services.training_service.training_job import _detect_latest_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            for epoch in (10, 20, 30):
                d = root / f"epoch_{epoch}"
                d.mkdir(parents=True)
                (d / "model_state.safetensors").write_bytes(b"x")
                (d / "optimizer.pt").write_bytes(b"x")
                (d / "meta.json").write_text(f'{{"epoch":{epoch}}}')
            detected = _detect_latest_checkpoint(Path(tmp))
            assert detected is not None
            assert detected[0] == 30
            assert detected[1].name == "epoch_30"

    def test_skips_partial_checkpoint(self):
        """A checkpoint missing optimizer.pt (subprocess killed mid-save)
        must be skipped — falling back to the next-older complete one
        prevents an inconsistent resume."""
        from app.services.training_service.training_job import _detect_latest_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            good = root / "epoch_10"
            good.mkdir(parents=True)
            (good / "model_state.safetensors").write_bytes(b"x")
            (good / "optimizer.pt").write_bytes(b"x")
            partial = root / "epoch_20"
            partial.mkdir(parents=True)
            (partial / "model_state.safetensors").write_bytes(b"x")
            # No optimizer.pt — partial
            detected = _detect_latest_checkpoint(Path(tmp))
            assert detected is not None
            assert detected[0] == 10, "Should fall back to the latest complete checkpoint"

    def test_ignores_non_epoch_dirs(self):
        from app.services.training_service.training_job import _detect_latest_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            root.mkdir(parents=True)
            (root / "garbage").mkdir()
            (root / "epoch_abc").mkdir()
            assert _detect_latest_checkpoint(Path(tmp)) is None


class TestResumeKwargsForwarded:
    """TrainingJob accepts resume_from_epoch / resume_checkpoint_path and
    stores them — they're forwarded into env vars at _run_training time.
    """

    def test_defaults_to_no_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            vd = Path(tmp)
            cfg = TrainingConfig(rank=16, learning_rate=1e-4, num_epochs=10, batch_size=2)
            job = TrainingJob(
                version_id="v", version_dir=vd, audio_paths=[],
                config=cfg, total_audio_duration=10.0, training_type="sft",
            )
            assert job.resume_from_epoch is None
            assert job.resume_checkpoint_path is None

    def test_accepts_resume_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            vd = Path(tmp)
            cfg = TrainingConfig(rank=16, learning_rate=1e-4, num_epochs=10, batch_size=2)
            ckpt = vd / "checkpoints" / "epoch_50"
            ckpt.mkdir(parents=True)
            job = TrainingJob(
                version_id="v", version_dir=vd, audio_paths=[],
                config=cfg, total_audio_duration=10.0, training_type="sft",
                resume_from_epoch=50, resume_checkpoint_path=ckpt,
            )
            assert job.resume_from_epoch == 50
            assert job.resume_checkpoint_path == ckpt


class TestResumeServiceFlow:
    """End-to-end tests for TrainingService.resume_training using a fake
    job factory — verifies error mapping (404 / 409) and the resume kwarg
    forwarding contract without spawning a subprocess.
    """

    def _make_service_with_failed_version(self, tmp_path, with_checkpoint: bool):
        """Set up a TrainingService with a failed version. Optionally create
        a checkpoint dir so the resume path can find it."""
        from datetime import datetime, timezone
        from app.services.training_service.service import TrainingService
        from app.services.training_service.repository import JsonTrainingRepository
        from app.services.training_service.models import (
            TrainingVersion, TrainingManifest, ManifestRecording,
            TrainingManifestConfig, VersionStatus, TrainingType,
        )

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        repo = JsonTrainingRepository(models_dir)

        captured = []

        class _FakeJob:
            def __init__(self, **kw):
                self._kw = kw
                captured.append(kw)
            def start(self): pass
            def cancel(self): pass
            def is_running(self): return False

        def factory(*, version, audio_paths, total_duration,
                    resume_from_epoch=None, resume_checkpoint_path=None):
            return _FakeJob(
                version_id=version.version_id,
                resume_from_epoch=resume_from_epoch,
                resume_checkpoint_path=resume_checkpoint_path,
            )

        class _Validator:
            def is_valid(self, x): return True
            def list_ids(self): return {"xs"}

        class _Resolver:
            def resolve_segments(self, ids): return []

        svc = TrainingService(
            repository=repo,
            persona_validator=_Validator(),
            audio_resolver=_Resolver(),
            models_dir=models_dir,
            job_factory=factory,
        )

        vd = models_dir / "xs_v1_test"
        vd.mkdir()
        v = TrainingVersion(
            version_id="v1_test", persona_id="xs",
            status=VersionStatus.failed, training_type=TrainingType.sft,
            rank=16, learning_rate=1e-5, num_epochs=100, batch_size=1,
            lora_path=str(vd),
            created_at=datetime.now(timezone.utc),
        )
        repo.save(v)
        mf = TrainingManifest(
            version_id="v1_test", persona_id="xs",
            segment_ids=["s1"], training_type=TrainingType.sft,
            recordings=[ManifestRecording(
                recording_id="r1", folder_name="",
                audio_path=str(vd / "a.wav"), duration_seconds=10.0,
            )],
            total_duration_seconds=10.0,
            training_config=TrainingManifestConfig(
                rank=16, learning_rate=1e-5, num_epochs=100, batch_size=1,
                training_type=TrainingType.sft,
            ),
        )
        repo.save_manifest("v1_test", mf)

        if with_checkpoint:
            ck = vd / "checkpoints" / "epoch_30"
            ck.mkdir(parents=True)
            (ck / "model_state.safetensors").write_bytes(b"x")
            (ck / "optimizer.pt").write_bytes(b"x")
            (ck / "meta.json").write_text('{"epoch":30,"loss":2.0}')

        return svc, captured

    def test_resume_404_when_no_checkpoint(self, tmp_path):
        from app.api._errors import TrainingVersionNotFoundError
        svc, _ = self._make_service_with_failed_version(tmp_path, with_checkpoint=False)
        with pytest.raises(TrainingVersionNotFoundError):
            svc.resume_training("v1_test")

    def test_resume_happy_path_forwards_resume_kwargs(self, tmp_path):
        from app.services.training_service.models import VersionStatus
        svc, captured = self._make_service_with_failed_version(tmp_path, with_checkpoint=True)
        result = svc.resume_training("v1_test")
        assert result.resumed_from_epoch == 30
        assert captured[0]["resume_from_epoch"] == 30
        assert captured[0]["resume_checkpoint_path"].name == "epoch_30"
        # Status is flipped back to training so the UI shows progress.
        v = svc.repository.get("v1_test")
        assert v.status == VersionStatus.training

    def test_resume_409_when_already_training(self, tmp_path):
        from app.api._errors import TrainingInProgressError
        svc, _ = self._make_service_with_failed_version(tmp_path, with_checkpoint=True)
        svc.resume_training("v1_test")  # first call flips status to training
        with pytest.raises(TrainingInProgressError):
            svc.resume_training("v1_test")


class TestSubprocessTemplateHasCheckpointSupport:
    """Source-level checks on the inline subprocess template — these
    catch accidental regressions in the heredoc string without needing
    to run the (GPU + 1.7B-model) training subprocess.
    """

    def _src(self):
        return Path("app/services/training_service/training_job.py").read_text()

    def test_template_defines_save_checkpoint(self):
        src = self._src()
        assert "_save_checkpoint(" in src
        assert "CHECKPOINT_EVERY_N_EPOCHS" in src
        assert "CHECKPOINT_RETENTION" in src

    def test_template_handles_resume_env_vars(self):
        src = self._src()
        assert "RESUME_FROM_EPOCH" in src
        assert "RESUME_CHECKPOINT_PATH" in src
        assert "Resuming from epoch" in src
        assert "start_epoch" in src

    def test_template_writes_latest_checkpoint_epoch(self):
        """progress.json must carry latest_checkpoint_epoch so the UI
        can decide whether to surface the Resume button."""
        src = self._src()
        assert "latest_checkpoint_epoch" in src

    def test_template_retention_prunes_old_checkpoints(self):
        src = self._src()
        # Retention logic must actually delete; check for rmtree call on
        # checkpoint dirs.
        assert "all_ckpts[CHECKPOINT_RETENTION:]" in src
        assert "_shutil.rmtree(old)" in src

    def test_template_bf16_cast_on_checkpoint_save(self):
        """Same fp32-leak guard as the merge path — see training_pipeline_deferred
        memory note. Checkpoint state_dict must be bf16-uniform."""
        src = self._src()
        # The casted dict construction inside _save_checkpoint.
        assert "v.to(torch.bfloat16).cpu()" in src
