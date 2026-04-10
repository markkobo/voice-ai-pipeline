"""
Recording API endpoints.

Handles:
- File upload
- Recording list/get/delete
- Processing trigger
- Audio streaming
"""

import json
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydub import AudioSegment

from app.services.recordings import (
    RecordingPaths,
    RecordingMetadata,
    list_all_recordings,
    list_recordings_metadata,
    load_recording_metadata,
    get_storage_stats,
    register_recording_in_cache,
    unregister_recording_from_cache,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/recordings", tags=["recordings"])

# Valid listener and persona IDs
VALID_LISTENER_IDS = {"child", "mom", "dad", "friend", "reporter", "elder", "default"}
VALID_PERSONA_IDS = {"xiao_s", "caregiver", "elder_gentle", "elder_playful"}

# Max file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

# Supported formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".webm"}


def validate_file(file_path: Path) -> tuple[float, int]:
    """
    Validate audio file and return duration and size.

    Returns:
        (duration_seconds, file_size_bytes)
    """
    size = file_path.stat().st_size
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE})")

    # Get duration using pydub
    audio = AudioSegment.from_file(str(file_path))
    duration = len(audio) / 1000.0  # ms to seconds

    if duration < 3:
        raise ValueError(f"Recording too short: {duration}s (min 3s)")
    if duration > 300:
        raise ValueError(f"Recording too long: {duration}s (max 300s)")

    return duration, size


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS)


@router.get("/")
async def list_recordings(page: int = 1, limit: int = 20):
    """
    List all recordings with metadata (paginated).

    Args:
        page: Page number (1-indexed)
        limit: Items per page (default 20, max 100)
    """
    limit = min(limit, 100)  # cap at 100
    offset = (page - 1) * limit

    all_recordings = list_recordings_metadata()
    total = len(all_recordings)
    paginated = all_recordings[offset:offset + limit]

    return {
        "recordings": paginated,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit,
    }


@router.get("/stats")
async def get_recording_stats():
    """Get storage statistics."""
    return get_storage_stats()


@router.post("/upload")
async def upload_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    listener_id: str = "default",
    persona_id: str = "xiao_s",
    title: Optional[str] = None,
):
    """
    Upload an audio file.

    Args:
        file: Audio file (WAV, MP3, M4A, WebM)
        listener_id: Who is speaking
        persona_id: Which persona this recording is for
        title: Optional title for the recording
    """
    logger.info(f"[UPLOAD] Starting upload: listener={listener_id}, persona={persona_id}, file={file.filename}")

    # Validate IDs
    if listener_id not in VALID_LISTENER_IDS:
        raise HTTPException(400, f"Invalid listener_id: {listener_id}")
    if persona_id not in VALID_PERSONA_IDS:
        raise HTTPException(400, f"Invalid persona_id: {persona_id}")

    # Validate file extension
    if not allowed_file(file.filename or ""):
        raise HTTPException(400, f"Unsupported file format. Supported: {SUPPORTED_FORMATS}")

    # Create recording paths
    paths = RecordingPaths(listener_id=listener_id, persona_id=persona_id)
    paths.create_folders()

    # Save uploaded file (streaming to avoid memory issues)
    temp_path = paths.raw_folder / "upload_temp"
    bytes_read = 0
    with open(temp_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            bytes_read += len(chunk)
            if bytes_read > MAX_FILE_SIZE:
                temp_path.unlink(missing_ok=True)
                raise HTTPException(400, f"File too large: {bytes_read} bytes (max {MAX_FILE_SIZE})")
            f.write(chunk)

    # Validate and convert to WAV
    try:
        duration, size = validate_file(temp_path)

        # Convert to WAV 48kHz 16bit if needed
        audio = AudioSegment.from_file(str(temp_path))
        audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)
        audio.export(str(paths.raw_audio_path), format="wav")

        # Clean up temp file
        temp_path.unlink()

    except ValueError as e:
        # Clean up and raise
        shutil.rmtree(paths.raw_folder)
        raise HTTPException(400, str(e))

    # Create metadata
    metadata = RecordingMetadata(paths)
    metadata._data["title"] = title
    metadata.update_audio_info(duration, size)
    metadata.save()

    # Register in cache index for fast lookup
    register_recording_in_cache(paths)

    logger.info(f"[UPLOAD] Complete: recording_id={metadata.data['recording_id']}, duration={duration}s, size={size}bytes")

    return {
        "recording_id": metadata.data["recording_id"],
        "folder_name": paths.folder_name,
        "duration_seconds": duration,
        "status": metadata.data["status"],
    }


@router.get("/{recording_id}")
async def get_recording(recording_id: str):
    """Get recording details and metadata."""
    # Find recording by ID
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            metadata = RecordingMetadata(paths)
            return metadata.data

    raise HTTPException(404, "Recording not found")


@router.delete("/{recording_id}")
async def delete_recording(recording_id: str):
    """
    Delete a recording and all its files.

    Note: Cannot delete if there's an active training session (ISSUE-9).
    Full dependency tracking requires ISSUE-7 fix (recording_ids in TrainingVersion).
    """
    logger.info(f"[DELETE] Deleting recording: {recording_id}")

    # Check if training is currently running
    from app.services.training import get_version_manager
    training_status = get_version_manager().get_training_status()
    if training_status.get("is_training"):
        raise HTTPException(400, "Cannot delete recording while training is in progress")

    # Find and delete
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            # Unregister from cache first
            unregister_recording_from_cache(recording_id)
            paths.delete_all()
            logger.info(f"[DELETE] Deleted: {paths.folder_name}")
            return {"status": "deleted", "recording_id": recording_id}

    raise HTTPException(404, "Recording not found")


@router.patch("/{recording_id}")
async def update_recording(recording_id: str, update: dict):
    """
    Update recording metadata.

    Supported fields: listener_id, persona_id, title, transcription
    """
    # Find recording
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            metadata = RecordingMetadata(paths)

            if "listener_id" in update:
                if update["listener_id"] not in VALID_LISTENER_IDS:
                    raise HTTPException(400, f"Invalid listener_id: {update['listener_id']}")
                metadata._data["listener_id"] = update["listener_id"]

            if "persona_id" in update:
                if update["persona_id"] not in VALID_PERSONA_IDS:
                    raise HTTPException(400, f"Invalid persona_id: {update['persona_id']}")
                metadata._data["persona_id"] = update["persona_id"]

            if "title" in update:
                metadata._data["title"] = update["title"]

            if "transcription" in update:
                text = update["transcription"]
                metadata.update_transcription(text)
                metadata.save_transcription_text(text)

            metadata.save()
            return {"status": "updated", "recording_id": recording_id}

    raise HTTPException(404, "Recording not found")


@router.get("/{recording_id}/stream")
async def stream_recording_audio(recording_id: str, stage: str = "enhanced"):
    """
    Stream recording audio for playback.

    Args:
        recording_id: Recording ID
        stage: "raw", "denoised", or "enhanced" (default: enhanced)
    """
    if stage not in {"raw", "denoised", "enhanced"}:
        raise HTTPException(400, "Invalid stage")

    # Find recording
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            if stage == "raw":
                audio_path = paths.raw_audio_path
            elif stage == "denoised":
                audio_path = paths.denoised_audio_path
            else:
                audio_path = paths.enhanced_audio_path

            if not audio_path.exists():
                # Fall back to raw if enhanced doesn't exist
                audio_path = paths.raw_audio_path

            if not audio_path.exists():
                raise HTTPException(404, "Audio file not found")

            return FileResponse(
                str(audio_path),
                media_type="audio/wav",
                filename=audio_path.name,
            )

    raise HTTPException(404, "Recording not found")


@router.get("/{recording_id}/download")
async def download_recording(recording_id: str):
    """Download recording audio file."""
    # First check recordings index
    recordings_index = Path("data/recordings/index.json")
    audio_path = None

    if recordings_index.exists():
        with open(recordings_index) as f:
            data = json.load(f)
        for rec in data.get("recordings", []):
            if rec.get("recording_id") == recording_id:
                folder = rec.get("folder_name", "")
                # Try to find audio file
                for pattern in ["audio.wav", "audio_processed.wav"]:
                    p = Path(f"data/recordings/raw/{folder}/{pattern}")
                    if p.exists():
                        audio_path = p
                        break
                    p = Path(f"data/recordings/enhanced/{folder}/{pattern}")
                    if p.exists():
                        audio_path = p
                        break
                break

    # If not found, try /tmp for test files
    if audio_path is None:
        test_path = Path(f"/tmp/{recording_id}.wav")
        if test_path.exists():
            audio_path = test_path

    if audio_path and audio_path.exists():
        return FileResponse(
            str(audio_path),
            media_type="audio/wav",
            filename=audio_path.name,
        )

    raise HTTPException(404, "Recording not found")


@router.get("/{recording_id}/speaker/{speaker_id}/audio")
async def get_speaker_audio(
    recording_id: str,
    speaker_id: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
):
    """
    Stream audio file for a specific speaker, optionally sliced by time range.

    Args:
        recording_id: Recording ID
        speaker_id: Speaker ID (e.g., "SPEAKER_00")
        start: Start time in seconds (optional)
        end: End time in seconds (optional)
    """
    import subprocess
    from pathlib import Path

    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            speaker_path = paths.speakers_folder / f"{speaker_id}.wav"
            if not speaker_path.exists():
                raise HTTPException(404, f"Speaker audio not found: {speaker_id}")

            # If time range is specified, slice the audio
            if start is not None or end is not None:
                import tempfile
                start_val = start or 0.0
                end_val = end
                output_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(speaker_path),
                        "-ss", str(start_val),
                    ]
                    if end_val is not None:
                        cmd.extend(["-to", str(end_val)])
                    cmd.extend(["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", str(output_path)])
                    subprocess.run(cmd, capture_output=True, check=True)
                    return FileResponse(
                        str(output_path),
                        media_type="audio/wav",
                        filename=f"{speaker_id}_{start_val:.2f}_{end_val or 'end'}.wav",
                    )
                except subprocess.CalledProcessError as e:
                    raise HTTPException(500, f"Audio slice failed: {e.stderr}")
                finally:
                    # Clean up temp file after response is sent
                    import asyncio
                    asyncio.get_event_loop().call_later(60, lambda: output_path.unlink(missing_ok=True))

            return FileResponse(
                str(speaker_path),
                media_type="audio/wav",
                filename=f"{speaker_id}.wav",
            )

    raise HTTPException(404, "Recording not found")


@router.get("/{recording_id}/transcription")
async def get_transcription(recording_id: str):
    """Get recording transcription."""
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            metadata = RecordingMetadata(paths)
            return metadata.data.get("transcription", {})

    raise HTTPException(404, "Recording not found")


@router.patch("/{recording_id}/speakers")
async def update_speaker_labels(recording_id: str, update: dict):
    """
    Update speaker labels for a recording.

    Args:
        update: {"speaker_labels": {"SPEAKER_00": "xiao_s", "SPEAKER_01": "mom"}}

    This maps extracted speaker files to personas for selective training.
    """
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            metadata = RecordingMetadata(paths)

            if "speaker_labels" not in update:
                raise HTTPException(400, "speaker_labels field required")

            labels = update["speaker_labels"]
            if not isinstance(labels, dict):
                raise HTTPException(400, "speaker_labels must be a dict")

            # Validate persona IDs
            for speaker_id, persona_id in labels.items():
                if persona_id not in VALID_PERSONA_IDS:
                    raise HTTPException(400, f"Invalid persona_id '{persona_id}' for speaker '{speaker_id}'")

            metadata.update_speaker_labels(labels)
            return {
                "status": "updated",
                "recording_id": recording_id,
                "speaker_labels": labels,
            }

    raise HTTPException(404, "Recording not found")


@router.get("/{recording_id}/speakers")
async def get_speaker_info(recording_id: str):
    """
    Get speaker information for a recording.

    Returns speaker segments and current labels.
    """
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            metadata = RecordingMetadata(paths)

            # Get existing labels
            labels = metadata.data.get("speaker_labels", {})

            # Get unique speakers from segments
            segments = metadata.data.get("speaker_segments", [])
            unique_speakers = sorted(set(seg["speaker_id"] for seg in segments))

            # Get available speaker audio files
            speaker_files = []
            if paths.speakers_folder.exists():
                for f in sorted(paths.speakers_folder.glob("*.wav")):
                    speaker_files.append(f.name)

            return {
                "recording_id": recording_id,
                "speakers": unique_speakers,
                "speaker_labels": labels,
                "speaker_files": speaker_files,
                "segment_count": len(segments),
            }

    raise HTTPException(404, "Recording not found")


@router.get("/{recording_id}/segments")
async def get_recording_segments(recording_id: str):
    """
    Get all speaker segments for a recording with enriched metadata.

    Returns list of segments with audio_path, duration, transcription, quality, etc.
    """
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            if not paths.metadata_path.exists():
                raise HTTPException(404, "Recording metadata not found")

            metadata = RecordingMetadata(paths)
            segments = metadata.data.get("speaker_segments", [])

            return {
                "recording_id": recording_id,
                "persona_id": metadata.data.get("persona_id"),
                "listener_id": metadata.data.get("listener_id"),
                "segments": segments,
                "speaker_labels": metadata.data.get("speaker_labels", {}),
            }

    raise HTTPException(404, "Recording not found")


@router.patch("/{recording_id}/segments/{speaker_id}")
async def update_segment(
    recording_id: str,
    speaker_id: str,
    persona_id: Optional[str] = None,
    listener_id: Optional[str] = None,
):
    """
    Update a speaker segment's persona_id or listener_id.

    This is used for labeling who is speaking in the recording.
    """
    from app.services.personas import get_persona
    from app.services.listeners import get_listener

    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            if not paths.metadata_path.exists():
                raise HTTPException(404, "Recording metadata not found")

            metadata = RecordingMetadata(paths)

            # Validate persona_id if provided
            if persona_id is not None:
                persona = get_persona(persona_id)
                if not persona:
                    raise HTTPException(400, f"Invalid persona_id: {persona_id}")

            # Validate listener_id if provided
            if listener_id is not None:
                listener = get_listener(listener_id)
                if not listener:
                    raise HTTPException(400, f"Invalid listener_id: {listener_id}")

            updated = metadata.update_segment(speaker_id, persona_id=persona_id, listener_id=listener_id)
            if not updated:
                raise HTTPException(404, f"Speaker segment not found: {speaker_id}")

            return {"status": "updated", "speaker_id": speaker_id, "persona_id": persona_id, "listener_id": listener_id}

    raise HTTPException(404, "Recording not found")


@router.post("/{recording_id}/process")
async def trigger_processing(recording_id: str, background_tasks: BackgroundTasks):
    """
    Trigger processing pipeline for a recording.

    Pipeline: denoise → enhance → diarize → transcribe
    """
    from app.services.recordings import run_processing_pipeline

    logger.info(f"[PIPELINE] Triggered for recording: {recording_id}")

    # Find recording
    recording = None
    for paths in list_all_recordings():
        if paths.recording_id == recording_id:
            recording = paths
            break

    if recording is None:
        raise HTTPException(404, "Recording not found")

    metadata = RecordingMetadata(recording)
    if metadata.data["status"] == "processing":
        raise HTTPException(400, "Already processing")

    # Start background processing
    metadata.update_status("processing")
    background_tasks.add_task(run_processing_pipeline, recording_id)

    return {"status": "processing_started", "recording_id": recording_id}


@router.post("/cleanup-expired")
async def cleanup_expired_recordings(dry_run: bool = False):
    """
    Delete recordings where processed_expires_at < now.

    Only deletes processed files (denoised/enhanced), keeps raw audio.
    Raw audio is never auto-deleted per NFR-32.

    Args:
        dry_run: If True, return list of recordings that would be deleted without deleting
    """
    from app.services.recordings import RecordingMetadata, unregister_recording_from_cache

    now = datetime.now()
    expired_recordings = []

    for paths in list_all_recordings():
        metadata = RecordingMetadata(paths)
        if metadata.data.get("status") != "processed":
            continue

        expires_at_str = metadata.data.get("processed_expires_at")
        if not expires_at_str:
            continue

        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        if expires_at < now:
            expired_recordings.append({
                "recording_id": paths.recording_id,
                "folder_name": paths.folder_name,
                "expired_at": expires_at_str,
            })

    if dry_run:
        return {
            "would_delete": len(expired_recordings),
            "recordings": expired_recordings,
            "dry_run": True,
        }

    # Actually delete processed files
    deleted_count = 0
    for rec in expired_recordings:
        for paths in list_all_recordings():
            if paths.recording_id == rec["recording_id"]:
                # Delete only processed folders (denoised, enhanced), keep raw
                if paths.denoised_folder.exists():
                    shutil.rmtree(paths.denoised_folder)
                if paths.enhanced_folder.exists():
                    shutil.rmtree(paths.enhanced_folder)

                # Update metadata to mark as expired
                metadata = RecordingMetadata(paths)
                metadata._data["processed_expires_at"] = None
                metadata.save()

                unregister_recording_from_cache(paths.recording_id)
                deleted_count += 1
                logger.info(f"[CLEANUP] Deleted processed files for: {paths.folder_name}")
                break

    return {
        "deleted": deleted_count,
        "recordings": expired_recordings,
        "dry_run": False,
    }
