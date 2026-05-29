"""
TrainingService — orchestrator for training lifecycle.

Pure business logic; no FastAPI imports. Raises DomainError subclasses so the
API layer can map them to HTTP status codes.

Collaborators (all injectable):
- TrainingRepository — load/save TrainingVersion + manifest + progress with locking
- AudioResolver — segment_ids → resolved audio paths + durations
- IdValidator — persona_id validation (same protocol as recordings)
- JobFactory — produces a startable training job (default: real TrainingJob;
  tests inject a fake that doesn't spawn a subprocess)
- VramCoordinator — best-effort TTS/ASR unload for SFT mode (no-op in tests)

The service holds a `_jobs` registry as an instance attribute, not a module
global — moving this off the module level fixes the test-isolation bug
documented in audit #3.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

from app.api._errors import (
    ActiveVersionLockedError,
    InvalidPersonaIdError,
    InvalidTrainingParamsError,
    NoTrainingAudioError,
    TrainingInProgressError,
    TrainingVersionNotFoundError,
    VersionNotReadyError,
)

from .audio_resolver import AudioResolver, ResolvedSegment
from .models import (
    MIN_TRAINING_AUDIO_SECONDS,
    VALID_LANGUAGE_TOKENS,
    ActiveVersion,
    ManifestRecording,
    TrainingManifest,
    TrainingManifestConfig,
    TrainingProgressSnapshot,
    TrainingType,
    TrainingVersion,
    VersionStatus,
    validate_batch_size,
    validate_epochs,
    validate_rank,
)
from .progress_tracker import ProgressTracker
from .repository import TrainingRepository, TrainingVersionNotFound

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collaborator protocols.
# ---------------------------------------------------------------------------
class IdValidator(Protocol):
    def is_valid(self, id_: str) -> bool: ...
    def list_ids(self) -> set[str]: ...


class StartableJob(Protocol):
    """The bit of TrainingJob the service cares about."""

    def start(self) -> None: ...
    def cancel(self) -> None: ...
    def is_running(self) -> bool: ...


# JobFactory signature — kept loose so the service stays decoupled from
# TrainingConfig / SFTConfig specifics. The default factory in
# `app/api/_dependencies.py` knows how to build a real TrainingJob.
JobFactory = Callable[..., StartableJob]


class VramCoordinator(Protocol):
    """Optional integration that unloads TTS/ASR during SFT training."""

    def unload_for_training(self) -> None: ...
    def release_for_training(self) -> None: ...


class _NullVramCoordinator:
    """No-op VRAM coordinator used in tests / environments without GPU."""

    def unload_for_training(self) -> None:
        return

    def release_for_training(self) -> None:
        return


# ---------------------------------------------------------------------------
# Constraints exposed to the API layer for use in Pydantic models.
# ---------------------------------------------------------------------------
# LoRA's 1e-4 was the original default but caused unconverged loss
# (~7.0 plateau) and runaway max-token generation in every 2026-05-25+
# LoRA run. Dropped to 1e-5 for less-catastrophic moves; combined with
# the higher default rank (32 in the UI) it improves chances of
# convergence but does NOT fully fix the underlying forward-path issue.
# LoRA remains experimental until forward_talker_finetune lands.
DEFAULT_LORA_LEARNING_RATE = 1e-5
# SFT was 1e-6 — empirically too low. v10 (30 epochs, 422s audio) ended
# at loss 9.44; converged SFT runs should plateau in the 2-4 range, so
# the model barely moved. Bumping to 1e-5 (same as LoRA) for meaningful
# steps while still well below the 5e-5 catastrophic-forgetting band
# for code_predictor. Re-evaluate if next training also stalls.
DEFAULT_SFT_LEARNING_RATE = 1e-5
TIME_PER_AUDIO_SECOND = 0.5
TRAINING_OVERHEAD_FACTOR = 1.3
SFT_TIME_MULTIPLIER = 5.0

# Checkpoint settings — used by the subprocess template to save a
# resumable mid-training snapshot every N epochs. Each checkpoint is
# multi-GB (model_state.safetensors + optimizer.pt for a 1.7B SFT model
# is roughly 3-6 GB in bf16), so retention is mandatory. Not exposed as
# a user-facing parameter — the constants are the policy.
#
# The 28800s wall-clock-cap on v12 truncated the run at epoch 66/100
# (commit 1eb5e05); without checkpoints the user lost 8h of progress.
# At every-10-epoch save, the equivalent loss would have been at most
# the work of the last 10 epochs.
DEFAULT_CHECKPOINT_EVERY_N_EPOCHS = 10
DEFAULT_CHECKPOINT_RETENTION = 2  # keep latest N to bound disk usage


@dataclass(frozen=True)
class CreatedVersion:
    """Result of a successful create_version call."""

    version: TrainingVersion
    total_duration_seconds: float
    estimated_time_seconds: int


@dataclass(frozen=True)
class ResumedVersion:
    """Result of a successful resume_training call.

    Carries `resumed_from_epoch` so the API layer can echo "we picked up
    at epoch N" back to the caller — useful for UI confirmation toasts
    and for log forensics.
    """

    version: TrainingVersion
    resumed_from_epoch: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class TrainingService:
    def __init__(
        self,
        repository: TrainingRepository,
        persona_validator: IdValidator,
        audio_resolver: AudioResolver,
        models_dir: Path,
        job_factory: Optional[JobFactory] = None,
        vram_coordinator: Optional[VramCoordinator] = None,
    ) -> None:
        self.repository = repository
        self.persona_validator = persona_validator
        self.audio_resolver = audio_resolver
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._job_factory = job_factory or _default_job_factory
        self._vram = vram_coordinator or _NullVramCoordinator()
        # Per-service job registry — replaces the module-level _training_jobs.
        self._jobs: dict[str, StartableJob] = {}
        self._jobs_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _require_valid_persona(self, persona_id: str) -> None:
        if not self.persona_validator.is_valid(persona_id):
            raise InvalidPersonaIdError(
                f"Unknown persona_id: {persona_id!r}",
                details={
                    "persona_id": persona_id,
                    "valid": sorted(self.persona_validator.list_ids()),
                },
            )

    # ------------------------------------------------------------------
    # Read-side methods
    # ------------------------------------------------------------------
    def list_versions(self, persona_id: Optional[str] = None) -> list[TrainingVersion]:
        return self.repository.list(persona_id)

    def get_version(self, version_id: str) -> TrainingVersion:
        try:
            return self.repository.get(version_id)
        except TrainingVersionNotFound as e:
            raise TrainingVersionNotFoundError(
                f"Training version not found: {version_id}",
                details={"version_id": version_id},
            ) from e

    def get_active(self, persona_id: str) -> Optional[TrainingVersion]:
        active = self.repository.get_active(persona_id)
        if active is None:
            return None
        return self.repository.get_or_none(active.version_id)

    def get_manifest(self, version_id: str) -> TrainingManifest:
        manifest = self.repository.get_manifest(version_id)
        if manifest is None:
            raise TrainingVersionNotFoundError(
                f"Manifest not found for version {version_id}",
                details={"version_id": version_id},
            )
        return manifest

    def read_progress(self, version_id: str) -> Optional[TrainingProgressSnapshot]:
        # Surface the not-found case so SSE can return a clean error.
        if not self.repository.exists(version_id):
            raise TrainingVersionNotFoundError(
                f"Training version not found: {version_id}",
                details={"version_id": version_id},
            )
        return self.repository.read_progress(version_id)

    def refresh_status_from_progress(self, version_id: str) -> TrainingVersion:
        """
        If progress.json says training is done, sync the version's status to
        the repo. Returns the (possibly-updated) version.

        Split out of list/get to make reads idempotent — the legacy code did
        this implicitly inside list_versions and surprised callers.
        """
        progress = self.repository.read_progress(version_id)
        if progress is None:
            return self.get_version(version_id)
        if progress.status.value not in ("ready", "failed"):
            return self.get_version(version_id)

        def mutate(v: TrainingVersion) -> None:
            v.status = (
                VersionStatus.ready
                if progress.status.value == "ready"
                else VersionStatus.failed
            )
            if progress.status.value == "ready":
                v.final_loss = progress.current_loss
                v.training_time_seconds = progress.elapsed_seconds
                v.completed_at = datetime.now(timezone.utc)
            elif progress.status.value == "failed":
                # Surface the subprocess's reason in the index so the UI
                # can show "失敗 (NVML driver mismatch)" instead of a
                # bare "✕ 失敗" with no clue. Falls back to a
                # training_result.json read for cases where the
                # subprocess died before writing to progress.json (e.g.
                # python crashed pre-init) but did write a result file.
                err = progress.error_message
                if not err and v.lora_path:
                    try:
                        result_file = Path(v.lora_path) / "training_result.json"
                        if result_file.exists():
                            with open(result_file) as f:
                                err = json.load(f).get("error")
                    except Exception:
                        pass
                v.error_message = err or "training failed (no error message captured)"
                v.completed_at = datetime.now(timezone.utc)

        try:
            return self.repository.update(version_id, mutate)
        except TrainingVersionNotFound as e:
            raise TrainingVersionNotFoundError(
                f"Training version not found: {version_id}",
                details={"version_id": version_id},
            ) from e

    def get_training_status(self) -> dict:
        """Match the legacy `{is_training, version_id?, persona_id?}` shape."""
        for v in self.repository.list():
            if v.status == VersionStatus.training:
                return {
                    "is_training": True,
                    "version_id": v.version_id,
                    "persona_id": v.persona_id,
                    "status": v.status.value,
                }
        return {"is_training": False}

    # ------------------------------------------------------------------
    # Startup sweep — reconcile stranded `training` versions.
    # ------------------------------------------------------------------
    def sweep_stranded(self) -> int:
        """Reconcile any version stuck in `status=training` with on-disk truth.

        Run at server startup so a training job that already finished (or
        died) on a previous run doesn't stay stuck in the "訓練中" UI state.
        Mirrors `RecordingsService.sweep_stranded` and the corpus
        ingestion sweep.

        Two cases per stranded version:
        1. progress.json says `ready` / `failed` → call
           refresh_status_from_progress to sync index.json.
        2. progress.json is absent or still says `training` → the
           subprocess died without writing a terminal state; flip to
           failed with an `interrupted` message.

        Returns the number of versions reconciled.
        """
        # Check for live training subprocesses BEFORE flipping anything.
        # The subagent-driven server restart on 2026-05-25 flipped a
        # running training to "failed (interrupted)" because the sweep
        # couldn't tell the difference between "subprocess died" and
        # "subprocess still alive, parent just restarted". The
        # subprocess kept writing progress.json the whole time; the
        # sweep just broke the UI by claiming it was dead.
        #
        # subprocess command line contains the lora_path:
        # `.venv/bin/python {version_dir}/train_lora.py`. If any
        # currently-running process matches that path, the subprocess
        # is still alive — leave it alone, the next refresh on
        # progress.json will eventually fire.
        import subprocess as _sp
        live_pids_by_dir: dict[str, int] = {}
        try:
            _out = _sp.run(
                ["ps", "axwo", "pid,args"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for line in _out.splitlines():
                if "train_lora.py" in line and "/data/models/" in line:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        pid_s, args = parts
                        # Pull the version_dir name out of the path.
                        # e.g. /...data/models/test_v9_2026.../train_lora.py
                        for tok in args.split():
                            if "/data/models/" in tok and tok.endswith("train_lora.py"):
                                vdir = tok.rsplit("/", 1)[0].rsplit("/", 1)[1]
                                try:
                                    live_pids_by_dir[vdir] = int(pid_s)
                                except ValueError:
                                    pass
        except Exception as e:
            log.warning("sweep_stranded: ps scan failed (%s); will fall back to assuming all are dead", e)

        count = 0
        for v in self.repository.list():
            if v.status != VersionStatus.training:
                continue

            # If the subprocess is still alive, this isn't a stranded
            # version — it's a healthy in-flight run that the parent
            # restart didn't kill (subprocesses don't share fate with
            # the FastAPI server). Leave it; next progress update will
            # naturally refresh status when the subprocess finishes.
            if v.lora_path:
                vdir_name = v.lora_path.rstrip("/").rsplit("/", 1)[-1]
                if vdir_name in live_pids_by_dir:
                    log.info(
                        "sweep_stranded: skipping %s (live subprocess pid=%d)",
                        v.version_id, live_pids_by_dir[vdir_name],
                    )
                    continue

            progress = self.repository.read_progress(v.version_id)
            terminal = progress is not None and progress.status.value in (
                "ready",
                "failed",
            )
            try:
                if terminal:
                    self.refresh_status_from_progress(v.version_id)
                    # Backfill merged_path if the merged model directory is
                    # already on disk but the index never recorded it
                    # (parent died after subprocess finished merging but
                    # before index update). Without this, activation can't
                    # locate the model. Naming convention matches
                    # training_job.merge_lora() — see line ~1022.
                    self._backfill_merged_path(v.version_id)
                else:
                    def _flip(ver: TrainingVersion) -> None:
                        ver.status = VersionStatus.failed
                        ver.error_message = (
                            "interrupted (server restarted mid-training)"
                        )
                        ver.completed_at = datetime.now(timezone.utc)
                    self.repository.update(v.version_id, _flip)
                count += 1
                log.warning(
                    "Reconciled stranded training version: id=%s persona=%s terminal=%s",
                    v.version_id, v.persona_id, terminal,
                )
            except Exception as e:
                log.exception(
                    "Failed to reconcile stranded training version %s: %s",
                    v.version_id, e,
                )
        return count

    def _backfill_merged_path(self, version_id: str) -> None:
        """If a merged model directory exists on disk but the index entry
        has no `merged_path`, record it. Idempotent — no-op when the path
        is already set or the directory is absent.
        """
        v = self.repository.get_or_none(version_id)
        if v is None or v.merged_path or not v.lora_path:
            return
        lora_dir = Path(v.lora_path)
        # New naming: full lora_dir.name. Falls back to legacy parts[:3]
        # form for backfilling older trainings that landed at the
        # collision-prone short name (e.g. xiao_s_v2 → merged_qwen3_tts_xiao_s_v2).
        merged_dir = lora_dir.parent / f"merged_qwen3_tts_{lora_dir.name}"
        if not merged_dir.exists():
            parts = lora_dir.name.split("_")
            if len(parts) >= 3:
                legacy = lora_dir.parent / f"merged_qwen3_tts_{'_'.join(parts[:3])}"
                if legacy.exists():
                    merged_dir = legacy
                else:
                    return
            else:
                return

        def _set(ver: TrainingVersion) -> None:
            ver.merged_path = str(merged_dir.resolve())

        try:
            self.repository.update(version_id, _set)
            log.info("Backfilled merged_path for %s → %s", version_id, merged_dir)
        except TrainingVersionNotFound:
            pass

    # ------------------------------------------------------------------
    # Mutate — small/safe ops
    # ------------------------------------------------------------------
    def set_nickname(self, version_id: str, nickname: Optional[str]) -> TrainingVersion:
        def mutate(v: TrainingVersion) -> None:
            v.nickname = nickname

        try:
            return self.repository.update(version_id, mutate)
        except TrainingVersionNotFound as e:
            raise TrainingVersionNotFoundError(
                f"Training version not found: {version_id}",
                details={"version_id": version_id},
            ) from e

    def activate_version(self, version_id: str) -> TrainingVersion:
        version = self.get_version(version_id)
        if version.status != VersionStatus.ready:
            raise VersionNotReadyError(
                f"Cannot activate version with status {version.status.value!r}",
                details={"version_id": version_id, "current_status": version.status.value},
            )
        self.repository.set_active(
            ActiveVersion(persona_id=version.persona_id, version_id=version_id)
        )
        log.info("[TRAINING] Activated %s for persona %s", version_id, version.persona_id)
        return version

    def resume_training(self, version_id: str) -> "ResumedVersion":
        """Resume a paused/failed/cancelled training run from its latest checkpoint.

        Behavior:
          - 404 if version doesn't exist OR if no usable checkpoint exists
            under <version_dir>/checkpoints/.
          - 409 if any training is already running (any version, any persona —
            same conflict policy as create_version: one subprocess at a time).
          - Otherwise spawn a new subprocess for the SAME version_id (manifest
            stays the same, no v(N+1)) with RESUME_FROM_EPOCH /
            RESUME_CHECKPOINT_PATH set so the subprocess template loads
            the model + optimizer state and starts at start_epoch = N+1.

        Returns the version, the manifest's resolved audio paths, and the
        epoch number being resumed from — the API layer needs the epoch
        to put in the response.
        """
        from .training_job import _detect_latest_checkpoint  # local to avoid cycle

        version = self.get_version(version_id)
        if not version.lora_path:
            raise TrainingVersionNotFoundError(
                f"Version {version_id} has no on-disk training directory",
                details={"version_id": version_id},
            )
        version_dir = Path(version.lora_path)
        detected = _detect_latest_checkpoint(version_dir)
        if detected is None:
            # 404 matches the spec: "no checkpoint dir" → not-found, not
            # not-ready. The UI button is only shown when
            # latest_checkpoint_epoch is non-null, so this branch mostly
            # catches racing API callers / stale UI state.
            raise TrainingVersionNotFoundError(
                f"No usable checkpoint for version {version_id}",
                details={"version_id": version_id, "checkpoints_dir": str(version_dir / "checkpoints")},
            )
        resume_epoch, ckpt_path = detected

        # Refuse if a training is already in flight — single-subprocess
        # discipline (same as create_version step 3). Otherwise two
        # subprocesses would fight for VRAM and both would OOM.
        for existing in self.repository.list():
            if existing.status == VersionStatus.training:
                raise TrainingInProgressError(
                    f"Training already in progress for version {existing.version_id}",
                    details={
                        "version_id": existing.version_id,
                        "persona_id": existing.persona_id,
                    },
                )

        # Re-resolve audio from the saved manifest. The manifest is
        # immutable per version, so this gives us back the exact same
        # segment set the original run used.
        manifest = self.get_manifest(version_id)
        audio_paths = [Path(r.audio_path) for r in manifest.recordings]
        if not audio_paths:
            raise NoTrainingAudioError(
                f"Manifest for {version_id} has no audio recordings",
                details={"version_id": version_id},
            )
        total_duration = manifest.total_duration_seconds

        # Flip status back to `training` so the UI shows the right state
        # and so a second /resume call hits the in-progress guard above.
        def _mark_training(v: TrainingVersion) -> None:
            v.status = VersionStatus.training
            v.error_message = None
            v.completed_at = None

        self.repository.update(version_id, _mark_training)

        # SFT-only VRAM unload — same as create_version.
        if version.training_type == TrainingType.sft:
            try:
                self._vram.unload_for_training()
            except Exception as e:
                log.warning("[TRAINING] VRAM unload failed during resume: %s", e)

        job = self._job_factory(
            version=version,
            audio_paths=audio_paths,
            total_duration=total_duration,
            resume_from_epoch=resume_epoch,
            resume_checkpoint_path=ckpt_path,
        )
        with self._jobs_lock:
            self._jobs[version_id] = job
        try:
            job.start()
        except Exception:
            with self._jobs_lock:
                self._jobs.pop(version_id, None)
            raise

        log.info(
            "[TRAINING] Resumed %s from epoch %d (ckpt=%s)",
            version_id, resume_epoch, ckpt_path,
        )
        return ResumedVersion(
            version=self.repository.get(version_id),
            resumed_from_epoch=resume_epoch,
        )

    def cancel_version(self, version_id: str) -> TrainingVersion:
        """Stop a running job (if any) and mark the version as cancelled."""
        # First touch the job registry so the subprocess gets the cancel
        # signal even if the repository write below fails.
        with self._jobs_lock:
            job = self._jobs.pop(version_id, None)
        if job is not None:
            try:
                job.cancel()
            except Exception as e:
                log.warning("Cancel call on job for %s raised: %s", version_id, e)

        def mutate(v: TrainingVersion) -> None:
            # Only transition non-terminal versions — preserve completed/failed state.
            if v.status == VersionStatus.training:
                v.status = VersionStatus.cancelled

        try:
            return self.repository.update(version_id, mutate)
        except TrainingVersionNotFound as e:
            raise TrainingVersionNotFoundError(
                f"Training version not found: {version_id}",
                details={"version_id": version_id},
            ) from e

    def delete_version(self, version_id: str) -> None:
        active = self.repository.get_active(self.get_version(version_id).persona_id)
        if active and active.version_id == version_id:
            raise ActiveVersionLockedError(
                f"Cannot delete active version {version_id}; deactivate first",
                details={"version_id": version_id},
            )

        # Cancel any running job to free its thread + locks.
        with self._jobs_lock:
            job = self._jobs.pop(version_id, None)
        if job is not None and job.is_running():
            try:
                job.cancel()
            except Exception as e:
                log.warning("Cancel during delete for %s raised: %s", version_id, e)

        try:
            self.repository.delete(version_id)
        except TrainingVersionNotFound as e:
            raise TrainingVersionNotFoundError(
                f"Training version not found: {version_id}",
                details={"version_id": version_id},
            ) from e

    # ------------------------------------------------------------------
    # Create + start training
    # ------------------------------------------------------------------
    def create_version(
        self,
        *,
        persona_id: str,
        segment_ids: list[str],
        rank: int = 16,
        num_epochs: int = 10,
        batch_size: int = 4,
        training_type: TrainingType = TrainingType.lora,
        learning_rate: Optional[float] = None,
        language_token: Optional[str] = None,
    ) -> CreatedVersion:
        """
        Validate inputs → resolve audio → create version → start job.

        Raises:
            InvalidPersonaIdError, InvalidTrainingParamsError, NoTrainingAudioError,
            TrainingInProgressError if another training is already running.
        """
        # 1. Persona must exist.
        self._require_valid_persona(persona_id)

        # 2. Param validation (single source of truth in models.py).
        try:
            validate_rank(rank)
            validate_epochs(num_epochs)
            validate_batch_size(batch_size)
        except ValueError as e:
            raise InvalidTrainingParamsError(str(e), details={"rank": rank, "num_epochs": num_epochs, "batch_size": batch_size}) from e
        if not isinstance(training_type, TrainingType):
            raise InvalidTrainingParamsError(
                f"training_type must be one of {[t.value for t in TrainingType]}; got {training_type!r}",
                details={"training_type": str(training_type)},
            )
        if learning_rate is not None and not (1e-8 <= learning_rate <= 1e-1):
            raise InvalidTrainingParamsError(
                f"learning_rate out of plausible range [1e-8, 1e-1]: {learning_rate}",
                details={"learning_rate": learning_rate},
            )
        # `language_token` accepts None (UI "不指定" → bake as False) or
        # one of the codec_language_id keys in the base model. Anything
        # else would crash the engine's `codec_language_id[dialect]`
        # lookup at inference time — reject here with a clear error.
        if language_token is not None and language_token not in VALID_LANGUAGE_TOKENS:
            raise InvalidTrainingParamsError(
                f"language_token must be None or one of {sorted(VALID_LANGUAGE_TOKENS)}; "
                f"got {language_token!r}",
                details={"language_token": language_token},
            )

        # 3. Refuse if another training is already running — prevents the
        #    concurrent-subprocess VRAM thrash the audit flagged.
        for existing in self.repository.list():
            if existing.status == VersionStatus.training:
                raise TrainingInProgressError(
                    f"Training already in progress for version {existing.version_id}",
                    details={"version_id": existing.version_id, "persona_id": existing.persona_id},
                )

        # 4. Resolve segments — duration-unknown segments raise here.
        if not segment_ids:
            raise NoTrainingAudioError(
                "No segment_ids provided",
                details={"segment_ids": []},
            )
        resolved = self.audio_resolver.resolve_segments(segment_ids)
        if not resolved:
            raise NoTrainingAudioError(
                "No resolvable segments",
                details={"segment_ids": segment_ids},
            )
        total_duration = sum(seg.duration_seconds for seg in resolved)
        if total_duration < MIN_TRAINING_AUDIO_SECONDS:
            raise NoTrainingAudioError(
                f"Total audio duration too short: {total_duration:.1f}s "
                f"(minimum {MIN_TRAINING_AUDIO_SECONDS}s)",
                details={
                    "total_duration_seconds": total_duration,
                    "min_required": MIN_TRAINING_AUDIO_SECONDS,
                },
            )

        # 5. Create version + dir.
        lr = learning_rate
        if lr is None:
            lr = (
                DEFAULT_LORA_LEARNING_RATE
                if training_type == TrainingType.lora
                else DEFAULT_SFT_LEARNING_RATE
            )
        version_id = self._allocate_version_id(persona_id)
        lora_dir = self.models_dir / f"{persona_id}_{version_id}"
        lora_dir.mkdir(parents=True, exist_ok=True)
        version = TrainingVersion(
            version_id=version_id,
            persona_id=persona_id,
            status=VersionStatus.training,
            training_type=training_type,
            model_type="custom_voice" if training_type == TrainingType.sft else None,
            rank=rank,
            learning_rate=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            recording_ids_used=list({seg.recording_id for seg in resolved}),
            segment_ids_used=list(segment_ids),
            lora_path=str(lora_dir),
            created_at=datetime.now(timezone.utc),
            language_token=language_token,
        )
        self.repository.save(version)

        # 6. Save manifest.
        manifest = TrainingManifest(
            version_id=version_id,
            persona_id=persona_id,
            segment_ids=list(segment_ids),
            training_type=training_type,
            recordings=[
                ManifestRecording(
                    recording_id=seg.recording_id,
                    folder_name="",  # unknown at this layer — kept blank for backwards compat
                    audio_path=str(seg.audio_path),
                    duration_seconds=seg.duration_seconds,
                )
                for seg in resolved
            ],
            total_duration_seconds=total_duration,
            training_config=TrainingManifestConfig(
                rank=rank,
                learning_rate=lr,
                num_epochs=num_epochs,
                batch_size=batch_size,
                training_type=training_type,
            ),
        )
        self.repository.save_manifest(version_id, manifest)

        # 7. SFT-only: ask the coordinator to release VRAM. Failures are
        #    logged but the create still proceeds — the subprocess will retry
        #    in its own startup.
        if training_type == TrainingType.sft:
            try:
                self._vram.unload_for_training()
            except Exception as e:
                log.warning("[TRAINING] VRAM unload failed: %s", e)

        # 8. Spawn the background job. Failures here bubble up — version is
        #    already in the repo, so the user can cancel it.
        job = self._job_factory(
            version=version,
            audio_paths=[seg.audio_path for seg in resolved],
            total_duration=total_duration,
        )
        with self._jobs_lock:
            self._jobs[version_id] = job
        try:
            job.start()
        except Exception:
            # Roll back the job registration but leave the version (in
            # training state) so it shows up as failable.
            with self._jobs_lock:
                self._jobs.pop(version_id, None)
            raise

        # 9. Time estimate (LoRA baseline × SFT multiplier).
        per_epoch = total_duration * TIME_PER_AUDIO_SECOND * TRAINING_OVERHEAD_FACTOR
        base = int(per_epoch * num_epochs)
        if training_type == TrainingType.sft:
            estimated = int(base * SFT_TIME_MULTIPLIER)
        else:
            estimated = base

        return CreatedVersion(
            version=version,
            total_duration_seconds=total_duration,
            estimated_time_seconds=estimated,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _allocate_version_id(self, persona_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        existing_count = sum(
            1 for v in self.repository.list() if v.persona_id == persona_id
        )
        return f"v{existing_count + 1}_{timestamp}"

    def estimate_training_time(self, total_duration: float, num_epochs: int) -> int:
        return ProgressTracker.estimate_training_time(total_duration, num_epochs)


# ---------------------------------------------------------------------------
# Default job factory — wraps the existing TrainingJob class.
# Tests inject a fake factory that returns a job which never spawns a thread.
# ---------------------------------------------------------------------------
def _default_job_factory(
    *,
    version: TrainingVersion,
    audio_paths: list[Path],
    total_duration: float,
    resume_from_epoch: Optional[int] = None,
    resume_checkpoint_path: Optional[Path] = None,
) -> StartableJob:
    """Build a real TrainingJob from the saved version + resolved audio.

    `resume_from_epoch` / `resume_checkpoint_path` are forwarded to the
    subprocess template as env vars so a resumed run picks up at
    `start_epoch = resume_from_epoch + 1` with the saved optimizer state.
    Both None = fresh run (create_version path).
    """
    from .lora_trainer import TrainingConfig
    from .sft_trainer import SFTConfig
    from .training_job import TrainingJob

    if version.training_type == TrainingType.sft:
        config = SFTConfig(
            learning_rate=version.learning_rate,
            num_epochs=version.num_epochs,
            batch_size=1,  # full-model SFT — small batch
            gradient_accumulation_steps=8,
        )
    else:
        config = TrainingConfig(
            rank=version.rank,
            learning_rate=version.learning_rate,
            num_epochs=version.num_epochs,
            batch_size=version.batch_size,
        )

    return TrainingJob(
        version_id=version.version_id,
        version_dir=Path(version.lora_path or ""),
        audio_paths=audio_paths,
        config=config,
        total_audio_duration=total_duration,
        training_type=(version.training_type.value if version.training_type else "lora"),
        persona_id=version.persona_id,
        language_token=version.language_token,
        resume_from_epoch=resume_from_epoch,
        resume_checkpoint_path=resume_checkpoint_path,
    )
