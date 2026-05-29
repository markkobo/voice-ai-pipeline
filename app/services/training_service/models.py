"""
Pydantic domain models for training.

Mirrors the on-disk JSON shape that the legacy
`app/services/training.py:TrainingVersion.to_dict()` produces so existing
index.json files keep loading. New writes use these models directly.

Pure data — no IO, no business logic. The repository (Task #18) serializes,
the service (Task #19) drives state transitions.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class VersionStatus(str, Enum):
    training = "training"
    ready = "ready"
    failed = "failed"
    cancelled = "cancelled"


class TrainingType(str, Enum):
    lora = "lora"
    sft = "sft"


class ProgressStatus(str, Enum):
    training = "training"
    # Intermediate phase: epochs finished, LoRA adapter saved, now merging
    # the adapter into the base model. For a 1.7B model this can take
    # several minutes on g5.4xlarge — without this state the status bar
    # would show "training 100% 10/10" for the entire merge window
    # (validator used to reject status="merging" with only the three
    # legal values below).
    merging = "merging"
    ready = "ready"
    failed = "failed"


# ---------------------------------------------------------------------------
# Constraints (single source of truth — referenced by validators below and
# by the API request models).
# ---------------------------------------------------------------------------
VALID_LORA_RANKS = {4, 8, 16, 32}
# Valid values for `language_token` — must match
# `talker_config.codec_language_id` keys in the Qwen3-TTS-12Hz-1.7B-Base
# config (10 languages). `None` (UI: "不指定") is also accepted and
# translates to Python `False` at bake time → no dialect override.
VALID_LANGUAGE_TOKENS = {
    "chinese",
    "english",
    "japanese",
    "korean",
    "french",
    "spanish",
    "german",
    "italian",
    "portuguese",
    "russian",
}
MIN_EPOCHS = 1
# Upper bound bumped from 50 → 200 to support SFT runs, which routinely
# need 100+ epochs for full-model fine-tuning to converge on a small
# corpus. LoRA still typically uses 10-50; this cap doesn't force higher
# values, it just allows them.
MAX_EPOCHS = 200
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 32
MIN_TRAINING_AUDIO_SECONDS = 10.0


# ---------------------------------------------------------------------------
# TrainingVersion — the aggregate root persisted in index.json.
# ---------------------------------------------------------------------------
class TrainingVersion(BaseModel):
    """
    A single training run.

    `extra="ignore"` lets us tolerate legacy fields like `num_recordings_used`
    or `display_name` that the old code wrote into to_dict() but aren't real
    model state. New writes never include those.
    """

    model_config = ConfigDict(extra="ignore")

    version_id: str
    persona_id: str
    status: VersionStatus = VersionStatus.training
    nickname: Optional[str] = None
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    lora_path: Optional[str] = None
    # Demo-readiness #4: explicit merged-model directory. Persisted by
    # the training job on completion so TTS activation reads it directly
    # instead of re-deriving from persona_id underscore-counts. Optional
    # so legacy index.json (no `merged_path` key) still loads with
    # `extra="ignore"`.
    merged_path: Optional[str] = None
    model_type: Optional[str] = None  # "custom_voice" for SFT, None for LoRA
    training_type: Optional[TrainingType] = None
    rank: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 4
    final_loss: Optional[float] = None
    training_time_seconds: Optional[int] = None
    recording_ids_used: list[str] = Field(default_factory=list)
    segment_ids_used: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # Populated when the version transitions to `failed` — either by the
    # subprocess via progress.json["error_message"], or by sweep_stranded
    # when reconciling a version whose subprocess died without writing a
    # terminal state.
    error_message: Optional[str] = None
    # Forces Qwen3-TTS codec language tokens for this persona via
    # talker_config.spk_is_dialect[persona] in the baked custom_voice
    # config. `None` (UI: "不指定") writes Python `False` at bake time —
    # i.e. no dialect override; the engine uses whatever language is
    # passed at inference. Non-None values must be one of
    # codec_language_id keys (chinese, english, japanese, korean,
    # french, spanish, german, italian, portuguese, russian). For
    # Taiwan-accented sources, `None` is recommended — setting
    # "chinese" forces the Beijing-accented codec path which destroys
    # the cloned accent (root cause of 2026-05-27 hand-patch incident).
    language_token: Optional[str] = None

    def to_legacy_dict(self) -> dict:
        """Match the legacy `to_dict()` shape that UI clients consume.

        Adds derived fields `num_recordings_used` and `display_name` for
        backwards compat with the old API.
        """
        d = self.model_dump(mode="json")
        d["num_recordings_used"] = len(self.recording_ids_used)
        d["display_name"] = self.nickname or self.version_id
        return d


# ---------------------------------------------------------------------------
# ActiveVersion — which version is currently in use for which persona.
# ---------------------------------------------------------------------------
class ActiveVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    version_id: str


# ---------------------------------------------------------------------------
# TrainingProgress — written by the subprocess to progress.json.
# Mirrors `training_service.progress_tracker.TrainingProgress` 1:1.
# ---------------------------------------------------------------------------
class TrainingProgressSnapshot(BaseModel):
    """A read-only snapshot of progress.json on disk.

    The progress writer (subprocess) controls this file; the API only reads
    it. Schema matches what the inline training script in training_job.py
    produces.
    """

    model_config = ConfigDict(extra="ignore")

    version_id: str
    status: ProgressStatus = ProgressStatus.training
    current_epoch: int = 0
    total_epochs: int = 10
    current_loss: float = 0.0
    best_loss: Optional[float] = None
    epoch_times: list[float] = Field(default_factory=list)
    elapsed_seconds: int = 0
    eta_seconds: int = 0
    progress_pct: int = 0
    last_updated: Optional[str] = None
    error_message: Optional[str] = None
    total_audio_duration: float = 0.0
    # Subprocess-only fields not part of the core schema but tolerated:
    persona_id: Optional[str] = None
    training_type: Optional[TrainingType] = None
    final_loss: Optional[float] = None
    training_time_seconds: Optional[int] = None
    # Highest epoch index whose checkpoint exists on disk under
    # `<version_dir>/checkpoints/epoch_{N}/`. Lets the UI show "Last
    # checkpoint: epoch N" and decide whether to surface the Resume
    # button on a failed/cancelled version. `None` means no checkpoint
    # has been written yet (training hasn't reached the first
    # CHECKPOINT_EVERY_N_EPOCHS boundary, or this is a pre-checkpoint
    # legacy run).
    latest_checkpoint_epoch: Optional[int] = None


# ---------------------------------------------------------------------------
# Training manifest — one per version, written next to the LoRA weights.
# ---------------------------------------------------------------------------
class ManifestRecording(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recording_id: str
    folder_name: str
    audio_path: str
    duration_seconds: float


class TrainingManifestConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    training_type: TrainingType


class TrainingManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version_id: str
    persona_id: str
    segment_ids: list[str]
    training_type: TrainingType
    recordings: list[ManifestRecording]
    total_duration_seconds: float
    training_config: TrainingManifestConfig


# ---------------------------------------------------------------------------
# Validation helpers — used by both the API request models (Task #22) and
# the service layer (Task #19). Single source of truth.
# ---------------------------------------------------------------------------
def validate_rank(rank: int) -> int:
    if rank not in VALID_LORA_RANKS:
        raise ValueError(
            f"rank must be one of {sorted(VALID_LORA_RANKS)}, got {rank}"
        )
    return rank


def validate_epochs(num_epochs: int) -> int:
    if not (MIN_EPOCHS <= num_epochs <= MAX_EPOCHS):
        raise ValueError(
            f"num_epochs must be in [{MIN_EPOCHS}, {MAX_EPOCHS}], got {num_epochs}"
        )
    return num_epochs


def validate_batch_size(batch_size: int) -> int:
    if not (MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE):
        raise ValueError(
            f"batch_size must be in [{MIN_BATCH_SIZE}, {MAX_BATCH_SIZE}], got {batch_size}"
        )
    return batch_size


# ---------------------------------------------------------------------------
# segment_id helper — parses "{recording_id}_SPEAKER_NN" into its two parts.
# Single source of truth (was duplicated in api/training.py and
# services/training.py per audit).
# ---------------------------------------------------------------------------
SEGMENT_SPEAKER_MARKER = "_SPEAKER_"


def parse_segment_id(segment_id: str) -> tuple[str, str]:
    """
    Parse `{recording_id}_SPEAKER_NN` into (recording_id, speaker_id).

    `speaker_id` includes the `SPEAKER_` prefix (e.g. `"SPEAKER_00"`) — that
    matches what the diarization layer uses.

    Raises:
        ValueError if the marker is not present.
    """
    marker_pos = segment_id.rfind(SEGMENT_SPEAKER_MARKER)
    if marker_pos == -1:
        raise ValueError(
            f"Invalid segment_id format (no {SEGMENT_SPEAKER_MARKER}): {segment_id!r}"
        )
    recording_id = segment_id[:marker_pos]
    speaker_id = segment_id[marker_pos + 1:]  # keep the SPEAKER_ prefix
    if not recording_id:
        raise ValueError(f"Empty recording_id in segment_id {segment_id!r}")
    return recording_id, speaker_id
