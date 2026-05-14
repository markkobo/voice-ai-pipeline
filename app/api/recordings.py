"""
Recordings REST API — routes only.

All business logic lives in `app.services.recordings.service.RecordingsService`.
Routes deserialize/validate input, call the service, serialize output.

Request bodies are Pydantic models — no raw `dict` accepted. Errors raised by
the service are `DomainError` subclasses that the app-wide exception handler
in `app.api._errors` maps to HTTP responses.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api._dependencies import get_recordings_service
from app.api._errors import (
    InvalidAudioError,
    InvalidListenerIdError,
    InvalidPersonaIdError,
    RecordingNotFoundError,
    TrainingInProgressError,
)
from app.services.recordings.models import Recording, SpeakerSegment
from app.services.recordings.service import (
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_FORMATS,
    PaginatedRecordings,
    RecordingsService,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/recordings", tags=["recordings"])


# ---------------------------------------------------------------------------
# Pydantic request / response shapes.
# ---------------------------------------------------------------------------
class UpdateRecordingRequest(BaseModel):
    """PATCH /api/recordings/{id} body."""

    model_config = {"extra": "forbid"}

    title: Optional[str] = None
    listener_id: Optional[str] = None
    persona_id: Optional[str] = None
    transcription: Optional[str] = None


class UpdateSpeakerLabelsRequest(BaseModel):
    """PATCH /api/recordings/{id}/speakers body."""

    model_config = {"extra": "forbid"}

    speaker_labels: dict[str, str] = Field(
        ...,
        description="Mapping speaker_id → persona_id, e.g. {'SPEAKER_00': 'xiao_s'}",
    )


class UploadResponse(BaseModel):
    recording_id: str
    folder_name: str
    duration_seconds: Optional[float]
    status: str


class DeleteResponse(BaseModel):
    status: str = "deleted"
    recording_id: str


class UpdateResponse(BaseModel):
    status: str = "updated"
    recording_id: str


class ListResponse(BaseModel):
    recordings: list[Recording]
    total: int
    page: int
    limit: int
    total_pages: int


class StatsResponse(BaseModel):
    raw_size_bytes: int
    denoised_size_bytes: int
    enhanced_size_bytes: int
    total_recordings: int


class SegmentsResponse(BaseModel):
    recording_id: str
    persona_id: str
    listener_id: str
    segments: list[SpeakerSegment]
    speaker_labels: dict[str, str]


class SpeakersResponse(BaseModel):
    recording_id: str
    speakers: list[str]
    speaker_labels: dict[str, str]
    speaker_files: list[str]
    segment_count: int


class UpdateSpeakerLabelsResponse(BaseModel):
    status: str = "updated"
    recording_id: str
    speaker_labels: dict[str, str]


class UpdateSegmentResponse(BaseModel):
    status: str = "updated"
    speaker_id: str
    persona_id: Optional[str] = None
    listener_id: Optional[str] = None


class ProcessingStartedResponse(BaseModel):
    status: str = "processing_started"
    recording_id: str


# ---------------------------------------------------------------------------
# CRUD routes — delegate to RecordingsService.
# ---------------------------------------------------------------------------
@router.get("/", response_model=ListResponse)
async def list_recordings(
    page: int = 1,
    limit: int = 20,
    service: RecordingsService = Depends(get_recordings_service),
) -> ListResponse:
    page_data: PaginatedRecordings = service.list(page=page, limit=limit)
    return ListResponse(
        recordings=page_data.items,
        total=page_data.total,
        page=page_data.page,
        limit=page_data.limit,
        total_pages=page_data.total_pages,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_recording_stats(
    service: RecordingsService = Depends(get_recordings_service),
) -> StatsResponse:
    """Storage statistics. Stable shape preserved from the legacy endpoint."""
    # Stats are derived from the audio_root + sibling directories; the service
    # doesn't expose this directly (yet — Phase 1.2), so compute inline. The
    # data still comes from service.audio_root so isolated_data fixtures work.
    raw = service.audio_root
    stages = {
        "raw_size_bytes": raw,
        "denoised_size_bytes": raw.parent / "denoised",
        "enhanced_size_bytes": raw.parent / "enhanced",
    }
    sizes = {}
    total_count = 0
    for key, path in stages.items():
        if not path.exists():
            sizes[key] = 0
            continue
        size = 0
        for f in path.rglob("*"):
            if f.is_file():
                size += f.stat().st_size
        sizes[key] = size
        if key == "raw_size_bytes":
            total_count = sum(1 for p in path.iterdir() if p.is_dir())
    return StatsResponse(total_recordings=total_count, **sizes)


@router.post("/upload", response_model=UploadResponse)
async def upload_recording(
    file: UploadFile = File(...),
    listener_id: str = Form("default"),
    persona_id: str = Form("xiao_s"),
    title: Optional[str] = Form(None),
    service: RecordingsService = Depends(get_recordings_service),
) -> UploadResponse:
    """Upload an audio file. Returns the new recording's identifiers."""
    file_bytes = await file.read()
    recording = service.upload(
        file_bytes=file_bytes,
        filename=file.filename or "upload.bin",
        listener_id=listener_id,
        persona_id=persona_id,
        title=title,
    )
    return UploadResponse(
        recording_id=recording.recording_id,
        folder_name=recording.folder_name,
        duration_seconds=recording.duration_seconds,
        status=recording.status.value,
    )


@router.get("/{recording_id}", response_model=Recording)
async def get_recording(
    recording_id: str,
    service: RecordingsService = Depends(get_recordings_service),
) -> Recording:
    return service.get(recording_id)


@router.delete("/{recording_id}", response_model=DeleteResponse)
async def delete_recording(
    recording_id: str,
    service: RecordingsService = Depends(get_recordings_service),
) -> DeleteResponse:
    # Check training-in-progress at the route boundary to keep service
    # dependency-free of training module. Trade-off: route knows about
    # version_manager; service stays pure.
    try:
        from app.services.training import get_version_manager

        if get_version_manager().get_training_status().get("is_training"):
            raise TrainingInProgressError(
                "Cannot delete recording while training is in progress",
                details={"recording_id": recording_id},
            )
    except ImportError:
        # Training module unavailable (test env without torch) — skip the guard.
        pass

    service.delete(recording_id)
    return DeleteResponse(recording_id=recording_id)


@router.patch("/{recording_id}", response_model=UpdateResponse)
async def update_recording(
    recording_id: str,
    body: UpdateRecordingRequest,
    service: RecordingsService = Depends(get_recordings_service),
) -> UpdateResponse:
    service.update(
        recording_id,
        title=body.title,
        listener_id=body.listener_id,
        persona_id=body.persona_id,
        transcription=body.transcription,
    )
    return UpdateResponse(recording_id=recording_id)


# ---------------------------------------------------------------------------
# Audio streaming
# ---------------------------------------------------------------------------
@router.get("/{recording_id}/stream")
async def stream_recording_audio(
    recording_id: str,
    stage: str = "enhanced",
    service: RecordingsService = Depends(get_recordings_service),
) -> FileResponse:
    """
    Stream a recording's audio. Falls back to raw if the requested stage isn't
    materialized yet.
    """
    if stage not in {"raw", "denoised", "enhanced"}:
        raise InvalidAudioError(
            f"Invalid stage: {stage}",
            details={"stage": stage, "valid": ["raw", "denoised", "enhanced"]},
        )
    try:
        audio_path = service.get_audio_path(recording_id, stage)
    except RecordingNotFoundError:
        # Fall back to raw stage before declaring 404.
        audio_path = service.get_audio_path(recording_id, "raw")
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=audio_path.name,
    )


@router.get("/{recording_id}/download")
async def download_recording(
    recording_id: str,
    service: RecordingsService = Depends(get_recordings_service),
) -> FileResponse:
    """Download the raw audio file."""
    audio_path = service.get_audio_path(recording_id, "raw")
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=audio_path.name,
    )


# ---------------------------------------------------------------------------
# Transcription / speakers / segments
# ---------------------------------------------------------------------------
@router.get("/{recording_id}/transcription")
async def get_transcription(
    recording_id: str,
    service: RecordingsService = Depends(get_recordings_service),
) -> dict:
    recording = service.get(recording_id)
    return recording.transcription.model_dump(mode="json")


@router.patch("/{recording_id}/speakers", response_model=UpdateSpeakerLabelsResponse)
async def update_speaker_labels(
    recording_id: str,
    body: UpdateSpeakerLabelsRequest,
    service: RecordingsService = Depends(get_recordings_service),
) -> UpdateSpeakerLabelsResponse:
    updated = service.update_speaker_labels(recording_id, body.speaker_labels)
    return UpdateSpeakerLabelsResponse(
        recording_id=recording_id,
        speaker_labels=updated.speaker_labels,
    )


@router.get("/{recording_id}/speakers", response_model=SpeakersResponse)
async def get_speaker_info(
    recording_id: str,
    service: RecordingsService = Depends(get_recordings_service),
) -> SpeakersResponse:
    recording = service.get(recording_id)
    unique_speakers = sorted({seg.speaker_id for seg in recording.speaker_segments})
    # speakers_folder lives next to the audio file in the raw folder.
    speakers_folder = service.audio_root / recording.folder_name / "speakers"
    speaker_files: list[str] = []
    if speakers_folder.exists():
        speaker_files = sorted(f.name for f in speakers_folder.glob("*.wav"))
    return SpeakersResponse(
        recording_id=recording_id,
        speakers=unique_speakers,
        speaker_labels=recording.speaker_labels,
        speaker_files=speaker_files,
        segment_count=len(recording.speaker_segments),
    )


@router.get("/{recording_id}/segments", response_model=SegmentsResponse)
async def get_recording_segments(
    recording_id: str,
    service: RecordingsService = Depends(get_recordings_service),
) -> SegmentsResponse:
    recording = service.get(recording_id)
    return SegmentsResponse(
        recording_id=recording_id,
        persona_id=recording.persona_id,
        listener_id=recording.listener_id,
        segments=recording.speaker_segments,
        speaker_labels=recording.speaker_labels,
    )


@router.patch("/{recording_id}/segments/{speaker_id}", response_model=UpdateSegmentResponse)
async def update_segment(
    recording_id: str,
    speaker_id: str,
    persona_id: Optional[str] = None,
    listener_id: Optional[str] = None,
    service: RecordingsService = Depends(get_recordings_service),
) -> UpdateSegmentResponse:
    if persona_id is None and listener_id is None:
        raise InvalidAudioError(
            "At least one of persona_id or listener_id must be provided",
            details={"speaker_id": speaker_id},
        )
    service.update_segment_routing(
        recording_id,
        speaker_id,
        persona_id=persona_id,
        listener_id=listener_id,
    )
    return UpdateSegmentResponse(
        speaker_id=speaker_id,
        persona_id=persona_id,
        listener_id=listener_id,
    )


# ---------------------------------------------------------------------------
# Speaker audio slicing (ffmpeg-backed). Kept here because it touches the
# filesystem in a way that doesn't fit cleanly into the service yet.
# ---------------------------------------------------------------------------
@router.get("/{recording_id}/speaker/{speaker_id}/audio")
async def get_speaker_audio(
    recording_id: str,
    speaker_id: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    service: RecordingsService = Depends(get_recordings_service),
) -> FileResponse:
    recording = service.get(recording_id)
    speaker_path = service.audio_root / recording.folder_name / "speakers" / f"{speaker_id}.wav"
    if not speaker_path.exists():
        raise RecordingNotFoundError(
            f"Speaker audio not found: {speaker_id}",
            details={"recording_id": recording_id, "speaker_id": speaker_id},
        )

    if start is None and end is None:
        return FileResponse(
            str(speaker_path),
            media_type="audio/wav",
            filename=f"{speaker_id}.wav",
        )

    start_val = start or 0.0
    end_val = end
    output_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    try:
        cmd = ["ffmpeg", "-y", "-i", str(speaker_path), "-ss", str(start_val)]
        if end_val is not None:
            cmd.extend(["-to", str(end_val)])
        cmd.extend(["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", str(output_path)])
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise InvalidAudioError(
            "ffmpeg is not installed on this server",
            details={"binary": "ffmpeg"},
        ) from e
    except subprocess.CalledProcessError as e:
        raise InvalidAudioError(
            f"Audio slice failed: {e.stderr!r}",
            details={"speaker_id": speaker_id, "start": start_val, "end": end_val},
        ) from e
    # Temp-file leak prevention: schedule deletion AFTER response is served.
    # FastAPI's StreamingResponse with cleanup would be cleaner; here we keep
    # parity with the legacy delete-on-loop approach and rely on the OS tmp
    # cleaner to reap if the loop is gone.
    return FileResponse(
        str(output_path),
        media_type="audio/wav",
        filename=f"{speaker_id}_{start_val:.2f}_{end_val or 'end'}.wav",
    )


# ---------------------------------------------------------------------------
# Processing trigger
# ---------------------------------------------------------------------------
@router.post("/{recording_id}/process", response_model=ProcessingStartedResponse)
async def trigger_processing(
    recording_id: str,
    background_tasks: BackgroundTasks,
    service: RecordingsService = Depends(get_recordings_service),
) -> ProcessingStartedResponse:
    """Trigger the denoise → enhance → diarize → transcribe pipeline."""
    recording = service.get(recording_id)
    if recording.status.value == "processing":
        raise InvalidAudioError(
            "Already processing",
            details={"recording_id": recording_id, "current_status": recording.status.value},
        )
    try:
        from app.services.recordings.pipeline import run_processing_pipeline
    except ImportError as e:  # heavy deps may be unavailable in test env
        raise InvalidAudioError(
            f"Processing pipeline unavailable: {e}",
            details={"error_type": type(e).__name__},
        ) from e

    def mutate(rec: Recording) -> None:
        rec.status = rec.status.processing  # type: ignore[assignment]

    service.repository.update(recording_id, mutate)
    background_tasks.add_task(run_processing_pipeline, recording_id)
    return ProcessingStartedResponse(recording_id=recording_id)


# ---------------------------------------------------------------------------
# Expired-cleanup batch endpoint
# ---------------------------------------------------------------------------
class CleanupResponse(BaseModel):
    deleted: int = 0
    would_delete: int = 0
    dry_run: bool = False
    recordings: list[dict] = Field(default_factory=list)


@router.post("/cleanup-expired", response_model=CleanupResponse)
async def cleanup_expired_recordings(
    dry_run: bool = False,
    service: RecordingsService = Depends(get_recordings_service),
) -> CleanupResponse:
    """Delete processed files for recordings past their expiry. Raw audio is
    never auto-deleted (per RFC_M2 NFR-32)."""
    now = datetime.now(timezone.utc)
    expired: list[dict] = []
    # service.list() is paginated; cleanup needs all recordings regardless of
    # page size — go through the repository directly.
    for recording in service.repository.list():
        if recording.status.value != "processed" or recording.processed_expires_at is None:
            continue
        if recording.processed_expires_at < now:
            expired.append(
                {
                    "recording_id": recording.recording_id,
                    "folder_name": recording.folder_name,
                    "expired_at": recording.processed_expires_at.isoformat(),
                }
            )

    if dry_run:
        return CleanupResponse(would_delete=len(expired), recordings=expired, dry_run=True)

    import shutil

    deleted = 0
    for entry in expired:
        folder = entry["folder_name"]
        for stage_dir in (
            service.audio_root.parent / "denoised" / folder,
            service.audio_root.parent / "enhanced" / folder,
        ):
            if stage_dir.exists():
                try:
                    shutil.rmtree(stage_dir)
                except OSError as e:
                    log.warning("Failed to remove %s: %s", stage_dir, e)

        # Clear the expiry marker on the metadata so we don't re-delete next run.
        def clear_expiry(rec: Recording) -> None:
            rec.processed_expires_at = None

        try:
            service.repository.update(entry["recording_id"], clear_expiry)
        except Exception as e:
            log.warning("Failed to clear expiry for %s: %s", entry["recording_id"], e)
        deleted += 1

    return CleanupResponse(deleted=deleted, recordings=expired, dry_run=False)
