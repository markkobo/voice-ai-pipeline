"""
Pydantic domain models for recordings.

These models replace the dict-based metadata that was previously serialized
ad-hoc inside RecordingMetadata. Fields follow RFC_M2 (recording metadata,
quality metrics) and RFC_MVP_REDESIGN (speaker segments with persona/listener).

Pure data classes — no IO, no business logic. The repository (JSON-backed)
serializes/deserializes these; the service layer operates on them.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RecordingStatus(str, Enum):
    raw = "raw"
    processing = "processing"
    processed = "processed"
    failed = "failed"


class ProcessingStepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    skipped = "skipped"
    failed = "failed"


# ---------------------------------------------------------------------------
# Nested value objects
# ---------------------------------------------------------------------------
class ProcessingStep(BaseModel):
    """State of a single pipeline step (denoise / enhance / diarize / transcribe)."""

    model_config = ConfigDict(extra="forbid")

    status: ProcessingStepStatus = ProcessingStepStatus.pending
    progress: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ProcessingSteps(BaseModel):
    """All four pipeline steps in one place."""

    model_config = ConfigDict(extra="forbid")

    denoise: ProcessingStep = Field(default_factory=ProcessingStep)
    enhance: ProcessingStep = Field(default_factory=ProcessingStep)
    diarize: ProcessingStep = Field(default_factory=ProcessingStep)
    transcribe: ProcessingStep = Field(default_factory=ProcessingStep)


class QualityMetrics(BaseModel):
    """
    Audio quality metrics (RFC_M2 §quality_metrics).

    `training_ready` is derived from snr_db / clarity_score thresholds — the
    service is responsible for setting it, not the model.
    """

    model_config = ConfigDict(extra="forbid")

    snr_db: Optional[float] = None
    rms_volume: Optional[float] = None
    silence_ratio: Optional[float] = None
    clarity_score: Optional[float] = None
    training_ready: Optional[bool] = None


class TranscriptionSegment(BaseModel):
    """A timestamped chunk of transcription text."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str


class Transcription(BaseModel):
    """Full transcription state for a recording."""

    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = None
    confidence: Optional[float] = None
    language: str = "zh"
    segments: list[TranscriptionSegment] = Field(default_factory=list)


class PipelineMetrics(BaseModel):
    """Per-step + total runtime for the processing pipeline."""

    model_config = ConfigDict(extra="forbid")

    denoise_ms: Optional[int] = None
    enhance_ms: Optional[int] = None
    diarize_ms: Optional[int] = None
    transcribe_ms: Optional[int] = None
    total_ms: Optional[int] = None


class SegmentQualityFlags(BaseModel):
    """Nested quality flags emitted by `pipeline.py` per speaker segment."""

    model_config = ConfigDict(extra="ignore")

    has_overlap: Optional[bool] = None
    low_energy: Optional[bool] = None
    high_noise: Optional[bool] = None
    too_short: Optional[bool] = None


class SpeakerSegment(BaseModel):
    """
    A single speaker segment from diarization (RFC_M2 §speaker_segments +
    RFC_MVP_REDESIGN §parsed segments).

    persona_id/listener_id default to the recording-level value but can be
    overridden per segment when a recording contains multiple speakers.

    `extra="ignore"` so legacy pipeline.py output that emits additional
    keys (transcription metadata variants, etc.) doesn't break loading.
    Known fields are still validated.
    """

    model_config = ConfigDict(extra="ignore")

    speaker_id: str
    start_time: float
    end_time: float
    duration_seconds: Optional[float] = None
    audio_path: Optional[str] = None
    transcription: Optional[str] = None
    transcription_confidence: Optional[float] = None

    # Quality scoring — both the legacy nested-dict form (`quality_flags`)
    # and the flat per-segment metrics emitted by pipeline.py are accepted.
    quality_score: Optional[float] = None
    quality_flags: Optional[SegmentQualityFlags] = None
    has_overlap: Optional[bool] = None
    low_energy: Optional[bool] = None
    high_noise: Optional[bool] = None
    too_short: Optional[bool] = None
    snr_db: Optional[float] = None
    clarity_score: Optional[float] = None
    training_ready: Optional[bool] = None

    # Voice-cloning training audit (effective bandwidth, clipping, etc.) —
    # written once at end of diarize by audit_voice_training_quality().
    # Shape: {"level": "good"|"marginal"|"bad", "warnings": [...], "metrics": {...}}.
    # `dict` not a sub-model so the Chinese warning strings + raw metrics
    # round-trip without forced schema rigidity.
    voice_audit: Optional[dict] = None

    # Routing identity (set explicitly or inherited from the recording).
    persona_id: Optional[str] = None
    listener_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Aggregate root
# ---------------------------------------------------------------------------
class Recording(BaseModel):
    """
    A single recording — the aggregate root persisted to metadata.json.

    Fields kept in dict-compatible shape so the JSON serialization matches what
    RecordingMetadata wrote previously. The repository (Task #8) handles
    load/save; the service (Task #9) drives state transitions.
    """

    model_config = ConfigDict(
        # Reject unknown keys when loading from disk — surfaces drift instead
        # of silently accepting fields we don't know about.
        extra="forbid",
        # Allow ISO 8601 strings to be parsed into datetime objects.
        json_schema_serialization_defaults_required=True,
    )

    recording_id: str
    folder_name: str
    listener_id: str
    persona_id: str
    title: Optional[str] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    transcription: Transcription = Field(default_factory=Transcription)
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    status: RecordingStatus = RecordingStatus.raw
    processing_steps: ProcessingSteps = Field(default_factory=ProcessingSteps)
    speaker_segments: list[SpeakerSegment] = Field(default_factory=list)
    speaker_labels: dict[str, str] = Field(default_factory=dict)
    pipeline_metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    processed_expires_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # -- Convenience methods (pure; do not touch IO) --

    @classmethod
    def new(
        cls,
        recording_id: str,
        folder_name: str,
        listener_id: str,
        persona_id: str,
    ) -> "Recording":
        """Construct a fresh `raw` recording with sensible defaults."""
        now = datetime.now(timezone.utc)
        return cls(
            recording_id=recording_id,
            folder_name=folder_name,
            listener_id=listener_id,
            persona_id=persona_id,
            created_at=now,
            updated_at=now,
        )

    def is_expired(self, expiry_days: int = 3) -> bool:
        """True if processed_expires_at is in the past (RFC_M2 §expiry)."""
        if self.processed_expires_at is None:
            return False
        # Compare using timezone-aware now.
        return datetime.now(timezone.utc) > self.processed_expires_at

    def mark_processed(self, expiry_days: int = 3) -> None:
        """Transition to `processed` and set the auto-cleanup deadline."""
        now = datetime.now(timezone.utc)
        self.status = RecordingStatus.processed
        self.processed_at = now
        self.processed_expires_at = now + timedelta(days=expiry_days)
        self.updated_at = now

    def mark_failed(self, error_message: str) -> None:
        """Record a terminal failure."""
        self.status = RecordingStatus.failed
        self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc)

    def apply_quality_metrics(
        self,
        snr_db: Optional[float] = None,
        rms_volume: Optional[float] = None,
        silence_ratio: Optional[float] = None,
        clarity_score: Optional[float] = None,
        snr_threshold_db: float = 15.0,
        clarity_threshold: float = 0.6,
    ) -> None:
        """
        Update quality metrics and derive training_ready from thresholds.

        Mirrors the RFC_M2 thresholds: SNR > 15dB AND clarity > 0.6 ⇒ training_ready.
        """
        if snr_db is not None:
            self.quality_metrics.snr_db = snr_db
        if rms_volume is not None:
            self.quality_metrics.rms_volume = rms_volume
        if silence_ratio is not None:
            self.quality_metrics.silence_ratio = silence_ratio
        if clarity_score is not None:
            self.quality_metrics.clarity_score = clarity_score

        snr = self.quality_metrics.snr_db
        clarity = self.quality_metrics.clarity_score
        if snr is not None and clarity is not None:
            self.quality_metrics.training_ready = (
                snr > snr_threshold_db and clarity > clarity_threshold
            )
        self.updated_at = datetime.now(timezone.utc)
