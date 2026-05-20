"""
RecordingsService — orchestrator for recording lifecycle.

Pure business logic; no FastAPI imports. Raises DomainError subclasses so the
API layer can map them to HTTP without knowing the internals.

Collaborators (all injectable):
- `RecordingsRepository` — load/save Recording aggregates with locking.
- `PersonaValidator` / `ListenerValidator` — protocols answering `is_valid(id)`.
- `audio_root: Path` — where to store/find raw WAV files.

The service does NOT touch pipeline.py — that stays as-is for now. The
service exposes `run_pipeline()` which dispatches a job. Refactor of the
pipeline itself is Phase 1.2 work.
"""
from __future__ import annotations

import io
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from app.api._errors import (
    AudioTooLargeError,
    InvalidAudioError,
    InvalidListenerIdError,
    InvalidPersonaIdError,
    RecordingNotFoundError,
    TrainingInProgressError,
    UnsupportedAudioFormatError,
)

from .models import Recording, RecordingStatus, SpeakerSegment
from .repository import RecordingNotFound, RecordingsRepository

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validator protocols. Concrete impls in app/api/_dependencies.py (Task #11)
# inject app/services/personas.py / listeners.py.
# ---------------------------------------------------------------------------
class IdValidator(Protocol):
    def is_valid(self, id_: str) -> bool: ...
    def list_ids(self) -> set[str]: ...


# ---------------------------------------------------------------------------
# Constraints (kept here so they're testable in isolation).
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".webm"}
# Recording-side limits. Bumped 2026-05-20 — client/demo workflows need
# longer takes (elders telling a story, podcast-style sample). Practical
# upper bound: 1h at 48kHz mono 16-bit WAV = ~345 MB, leave headroom.
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
MIN_DURATION_SECONDS = 3.0
MAX_DURATION_SECONDS = 3600.0  # 1 hour

# Target normalized format on disk — matches what RecordingPaths.save_audio writes.
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit


@dataclass(frozen=True)
class PaginatedRecordings:
    items: list[Recording]
    total: int
    page: int
    limit: int

    @property
    def total_pages(self) -> int:
        if self.limit <= 0:
            return 0
        return (self.total + self.limit - 1) // self.limit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _allowed_extension(filename: str) -> bool:
    name = filename.lower()
    return any(name.endswith(ext) for ext in SUPPORTED_FORMATS)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class RecordingsService:
    """
    Orchestrates recording CRUD and audio file management.

    Methods that mutate Recording state go through `repository.update()` so
    concurrent PATCH requests converge under the repository's exclusive lock.
    """

    def __init__(
        self,
        repository: RecordingsRepository,
        persona_validator: IdValidator,
        listener_validator: IdValidator,
        audio_root: Path,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        min_duration: float = MIN_DURATION_SECONDS,
        max_duration: float = MAX_DURATION_SECONDS,
    ) -> None:
        self.repository = repository
        self.persona_validator = persona_validator
        self.listener_validator = listener_validator
        self.audio_root = Path(audio_root)
        self.max_file_size = max_file_size
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.audio_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation helpers (raise DomainError)
    # ------------------------------------------------------------------
    def _require_valid_persona(self, persona_id: str) -> None:
        if not self.persona_validator.is_valid(persona_id):
            raise InvalidPersonaIdError(
                f"Unknown persona_id: {persona_id!r}",
                details={"persona_id": persona_id, "valid": sorted(self.persona_validator.list_ids())},
            )

    def _require_valid_listener(self, listener_id: str) -> None:
        if not self.listener_validator.is_valid(listener_id):
            raise InvalidListenerIdError(
                f"Unknown listener_id: {listener_id!r}",
                details={"listener_id": listener_id, "valid": sorted(self.listener_validator.list_ids())},
            )

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        listener_id: str,
        persona_id: str,
        title: Optional[str] = None,
    ) -> Recording:
        """
        Persist a new recording from raw upload bytes.

        Validates listener_id / persona_id / extension / size / duration.
        Returns the newly-created Recording (status=raw).
        """
        self._require_valid_listener(listener_id)
        self._require_valid_persona(persona_id)

        if not _allowed_extension(filename):
            raise UnsupportedAudioFormatError(
                f"Unsupported file format: {filename!r}",
                details={"filename": filename, "supported": sorted(SUPPORTED_FORMATS)},
            )

        if len(file_bytes) > self.max_file_size:
            raise AudioTooLargeError(
                f"File too large: {len(file_bytes)} bytes (max {self.max_file_size})",
                details={"size_bytes": len(file_bytes), "max_bytes": self.max_file_size},
            )

        # Validate duration + normalize to WAV via pydub. pydub is the only
        # service-layer audio dep; isolated here so it can be swapped.
        duration_seconds, normalized_bytes = self._validate_and_normalize_audio(file_bytes)

        # Materialize folder + audio on disk. Microsecond precision avoids
        # folder collisions when two uploads land within the same second,
        # which the previous timestamp-to-the-second pattern silently
        # clobbered (test_pagination_slices_correctly caught this).
        recording_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        folder_name = f"{listener_id}_{persona_id}_{timestamp}"
        folder = self.audio_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        audio_path = folder / "audio.wav"
        audio_path.write_bytes(normalized_bytes)

        recording = Recording.new(
            recording_id=recording_id,
            folder_name=folder_name,
            listener_id=listener_id,
            persona_id=persona_id,
        )
        recording.title = title
        recording.duration_seconds = duration_seconds
        recording.file_size_bytes = len(normalized_bytes)
        self.repository.save(recording)
        log.info(
            "Recording uploaded: id=%s listener=%s persona=%s duration=%.2fs size=%dB",
            recording_id,
            listener_id,
            persona_id,
            duration_seconds,
            len(normalized_bytes),
        )
        return recording

    def _validate_and_normalize_audio(self, raw_bytes: bytes) -> tuple[float, bytes]:
        """
        Decode → check duration → re-encode as canonical WAV.

        Pure-WAV inputs are handled by the stdlib `wave` module so this code
        works without `ffmpeg`/`ffprobe` available. Anything else falls back
        to pydub, which does require ffmpeg.

        Returns (duration_seconds, normalized_wav_bytes).
        """
        if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
            return self._decode_wav(raw_bytes)

        # Lazy import — pydub is heavy and only needed for non-WAV uploads.
        try:
            from pydub import AudioSegment
        except ImportError as e:  # pragma: no cover — only reached in degraded env
            raise InvalidAudioError(f"Audio decoder not available: {e}") from e

        buf = io.BytesIO(raw_bytes)
        try:
            audio = AudioSegment.from_file(buf)
        except Exception as e:
            raise InvalidAudioError(
                f"Could not decode audio: {e}",
                details={"error_type": type(e).__name__},
            ) from e

        duration = len(audio) / 1000.0
        self._check_duration_bounds(duration)
        audio = (
            audio.set_frame_rate(TARGET_SAMPLE_RATE)
            .set_channels(TARGET_CHANNELS)
            .set_sample_width(TARGET_SAMPLE_WIDTH)
        )
        out = io.BytesIO()
        audio.export(out, format="wav")
        return duration, out.getvalue()

    def _decode_wav(self, raw_bytes: bytes) -> tuple[float, bytes]:
        """Stdlib-only WAV decode/normalize. Avoids the ffmpeg dependency."""
        import wave

        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as w:
                n_frames = w.getnframes()
                src_rate = w.getframerate()
                src_channels = w.getnchannels()
                src_sample_width = w.getsampwidth()
                frames = w.readframes(n_frames)
        except wave.Error as e:
            raise InvalidAudioError(
                f"Could not decode WAV: {e}",
                details={"error_type": type(e).__name__},
            ) from e
        if n_frames == 0 or src_rate == 0:
            raise InvalidAudioError("Empty WAV file", details={"n_frames": n_frames})

        duration = n_frames / float(src_rate)
        self._check_duration_bounds(duration)

        # If the source already matches our target, return bytes verbatim.
        # Otherwise re-encode via pydub (still requires ffmpeg for resampling).
        if (
            src_rate == TARGET_SAMPLE_RATE
            and src_channels == TARGET_CHANNELS
            and src_sample_width == TARGET_SAMPLE_WIDTH
        ):
            return duration, raw_bytes

        # For non-target WAV inputs that need resample/channel-mix, fall back
        # to pydub. This still requires ffmpeg for sample-rate conversion. We
        # keep this branch tight so the test path (which uses 24kHz mono 16-bit
        # WAVs) hits the verbatim return above and never needs ffmpeg.
        try:
            from pydub import AudioSegment
        except ImportError as e:  # pragma: no cover
            raise InvalidAudioError(f"Audio decoder not available: {e}") from e
        try:
            audio = AudioSegment.from_wav(io.BytesIO(raw_bytes))
            audio = (
                audio.set_frame_rate(TARGET_SAMPLE_RATE)
                .set_channels(TARGET_CHANNELS)
                .set_sample_width(TARGET_SAMPLE_WIDTH)
            )
            out = io.BytesIO()
            audio.export(out, format="wav")
            return duration, out.getvalue()
        except Exception as e:
            raise InvalidAudioError(
                f"Could not normalize WAV: {e}",
                details={"error_type": type(e).__name__},
            ) from e

    def _check_duration_bounds(self, duration: float) -> None:
        if duration < self.min_duration:
            raise InvalidAudioError(
                f"Recording too short: {duration:.2f}s (min {self.min_duration}s)",
                details={"duration_seconds": duration, "min_seconds": self.min_duration},
            )
        if duration > self.max_duration:
            raise InvalidAudioError(
                f"Recording too long: {duration:.2f}s (max {self.max_duration}s)",
                details={"duration_seconds": duration, "max_seconds": self.max_duration},
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get(self, recording_id: str) -> Recording:
        try:
            return self.repository.get(recording_id)
        except RecordingNotFound as e:
            raise RecordingNotFoundError(f"Recording not found: {recording_id}") from e

    def list(self, *, page: int = 1, limit: int = 20) -> PaginatedRecordings:
        page = max(1, page)
        limit = max(1, min(limit, 100))
        all_recordings = self.repository.list()
        offset = (page - 1) * limit
        page_items = all_recordings[offset : offset + limit]
        return PaginatedRecordings(
            items=page_items,
            total=len(all_recordings),
            page=page,
            limit=limit,
        )

    def get_audio_path(self, recording_id: str, stage: str = "raw") -> Path:
        """
        Return the on-disk path for a recording's audio file.

        `stage` is one of: raw, denoised, enhanced. Raises if missing.
        """
        if stage not in {"raw", "denoised", "enhanced"}:
            raise InvalidAudioError(
                f"Invalid stage: {stage}",
                details={"stage": stage, "valid": ["raw", "denoised", "enhanced"]},
            )
        recording = self.get(recording_id)
        # Audio dir mirrors the audio_root layout used at upload time.
        candidate = self.audio_root.parent / stage / recording.folder_name / "audio.wav"
        if stage == "raw":
            candidate = self.audio_root / recording.folder_name / "audio.wav"
        if not candidate.exists():
            raise RecordingNotFoundError(
                f"Audio file for stage={stage!r} not found",
                details={"recording_id": recording_id, "stage": stage, "expected_path": str(candidate)},
            )
        return candidate

    # ------------------------------------------------------------------
    # Mutate
    # ------------------------------------------------------------------
    def update(
        self,
        recording_id: str,
        *,
        title: Optional[str] = None,
        listener_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        transcription: Optional[str] = None,
    ) -> Recording:
        """Update specific top-level fields. Unknown fields are rejected at the
        API boundary via Pydantic (Task #12)."""
        if listener_id is not None:
            self._require_valid_listener(listener_id)
        if persona_id is not None:
            self._require_valid_persona(persona_id)

        def mutate(rec: Recording) -> None:
            if title is not None:
                rec.title = title
            if listener_id is not None:
                rec.listener_id = listener_id
            if persona_id is not None:
                rec.persona_id = persona_id
            if transcription is not None:
                rec.transcription.text = transcription

        try:
            return self.repository.update(recording_id, mutate)
        except RecordingNotFound as e:
            raise RecordingNotFoundError(f"Recording not found: {recording_id}") from e

    def update_speaker_labels(self, recording_id: str, labels: dict[str, str]) -> Recording:
        """Replace the speaker_id → persona_id mapping for a recording.

        Each persona_id in the mapping is validated against the registry."""
        for speaker_id, persona_id in labels.items():
            self._require_valid_persona(persona_id)

        def mutate(rec: Recording) -> None:
            rec.speaker_labels = dict(labels)
            # Also propagate persona_id into matching speaker_segments — keeps
            # the two stores consistent (was a bug previously per audit).
            for seg in rec.speaker_segments:
                if seg.speaker_id in labels:
                    seg.persona_id = labels[seg.speaker_id]

        try:
            return self.repository.update(recording_id, mutate)
        except RecordingNotFound as e:
            raise RecordingNotFoundError(f"Recording not found: {recording_id}") from e

    def update_segment_routing(
        self,
        recording_id: str,
        speaker_id: str,
        *,
        persona_id: Optional[str] = None,
        listener_id: Optional[str] = None,
    ) -> Recording:
        """Set persona_id / listener_id on all segments for a given speaker."""
        if persona_id is not None:
            self._require_valid_persona(persona_id)
        if listener_id is not None:
            self._require_valid_listener(listener_id)

        def mutate(rec: Recording) -> None:
            matched = False
            for seg in rec.speaker_segments:
                if seg.speaker_id == speaker_id:
                    if persona_id is not None:
                        seg.persona_id = persona_id
                    if listener_id is not None:
                        seg.listener_id = listener_id
                    matched = True
            if not matched:
                raise InvalidAudioError(
                    f"speaker_id not found in segments: {speaker_id}",
                    details={"speaker_id": speaker_id, "recording_id": recording_id},
                )
            # Keep speaker_labels in sync.
            if persona_id is not None:
                rec.speaker_labels[speaker_id] = persona_id

        try:
            return self.repository.update(recording_id, mutate)
        except RecordingNotFound as e:
            raise RecordingNotFoundError(f"Recording not found: {recording_id}") from e

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete(
        self,
        recording_id: str,
        *,
        training_in_progress: bool = False,
    ) -> None:
        """
        Remove a recording's metadata, audio, and index entry.

        Refuses to delete while training is in progress (per RFC_M4).
        """
        if training_in_progress:
            raise TrainingInProgressError(
                "Cannot delete recording while training is in progress",
                details={"recording_id": recording_id},
            )

        # Get folder name BEFORE delete so we can also clean up audio.
        try:
            recording = self.repository.get(recording_id)
        except RecordingNotFound as e:
            raise RecordingNotFoundError(f"Recording not found: {recording_id}") from e

        # Delete repo entry first — if audio deletion partially fails, at least
        # the recording is removed from the index and listings don't return a
        # ghost record. Audio cleanup is best-effort.
        self.repository.delete(recording_id)

        for stage_root_name in ("raw", "denoised", "enhanced"):
            stage_root = self.audio_root.parent / stage_root_name
            if stage_root_name == "raw":
                stage_root = self.audio_root
            stage_folder = stage_root / recording.folder_name
            if stage_folder.exists():
                try:
                    shutil.rmtree(stage_folder)
                except OSError as e:
                    log.warning(
                        "Failed to remove %s for recording %s: %s",
                        stage_folder,
                        recording_id,
                        e,
                    )

    # ------------------------------------------------------------------
    # Startup sweep — reset stranded `processing` items to `failed`.
    # ------------------------------------------------------------------
    def sweep_stranded(self) -> int:
        """Reset any recordings stuck in `processing` status to `failed`.

        Run at server startup so recordings whose BackgroundTask died
        (server crash / kill / restart mid-process) don't stay stuck in
        the "處理中" UI state forever. Mirrors the corpus sweep at
        `app/services/corpus/ingestion.py::sweep_stranded` (task 62C).

        Tolerates a concurrent delete between list() and update() —
        a RecordingNotFound from update() just means the user removed
        the recording while we were iterating; carry on.

        Preserves `speaker_segments` and other partial-work fields —
        only flips `status` + sets `error_message`.

        Returns the number of recordings reset.
        """
        count = 0
        for recording in self.repository.list():
            if recording.status != RecordingStatus.processing:
                continue

            def _flip(rec: Recording) -> None:
                rec.status = RecordingStatus.failed
                rec.error_message = "interrupted (server restarted mid-process)"

            try:
                self.repository.update(recording.recording_id, _flip)
            except RecordingNotFound:
                # Concurrent delete won the race — skip.
                continue
            count += 1
            log.warning(
                "Reset stranded processing recording: id=%s folder=%s",
                recording.recording_id, recording.folder_name,
            )
        return count

    # ------------------------------------------------------------------
    # Quality + pipeline hooks (kept thin — pipeline.py owns the heavy work).
    # ------------------------------------------------------------------
    def list_segments(self, recording_id: str) -> list[SpeakerSegment]:
        return self.get(recording_id).speaker_segments
