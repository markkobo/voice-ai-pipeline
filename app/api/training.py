"""
Training API endpoints.

Handles:
- Training version management
- Training trigger/status
- Progress streaming (SSE)
- Model activation
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.training_service import (
    ProgressTracker,
    LoraTrainer,
    TrainingConfig,
    TrainingJob,
)
from app.services.training import (
    get_version_manager,
    TrainingVersion,
)
from app.services.recordings import list_recordings_metadata, RecordingPaths, RecordingMetadata

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])

# Training job registry
_training_jobs: dict[str, TrainingJob] = {}


# ============================================================================
# Training Progress SSE
# ============================================================================

async def sse_progress_generator(version_id: str):
    """Generate SSE events for training progress."""
    manager = get_version_manager()
    version_dir = manager.get_version_dir(version_id)

    if not version_dir:
        yield f"data: {json.dumps({'error': 'Version not found'})}\n\n"
        return

    progress_file = version_dir / "progress.json"
    last_data = None

    while True:
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    data = json.load(f)

                if data.get("status") in ("ready", "failed"):
                    if data["status"] == "ready":
                        yield f"data: {json.dumps({'event': 'complete', **data})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'error', 'error': data.get('error_message', 'Unknown error')})}\n\n"
                    break

                if data != last_data:
                    yield f"data: {json.dumps({'event': 'progress', **data})}\n\n"
                    last_data = data
            except Exception as e:
                logger.error(f"Error reading progress: {e}")

        await asyncio.sleep(2)


@router.get("/versions/{version_id}/progress")
async def stream_training_progress(version_id: str):
    """
    SSE stream for training progress.

    Client reconnects here after disconnect to get current status.
    """
    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(404, "Version not found")

    return StreamingResponse(
        sse_progress_generator(version_id),
        media_type="text/event-stream",
    )


# ============================================================================
# Version Management
# ============================================================================

@router.get("/versions")
async def list_versions(persona_id: Optional[str] = None):
    """List all training versions, optionally filtered by persona."""
    manager = get_version_manager()
    versions = manager.list_versions(persona_id)

    result = []
    for v in versions:
        v_dict = v.to_dict()

        # Include progress if training
        if v.status == "training":
            version_dir = manager.get_version_dir(v.version_id)
            if version_dir:
                progress = ProgressTracker.load(v.version_id, version_dir)
                if progress:
                    v_dict["progress"] = progress.to_dict()

        result.append(v_dict)

    return {
        "versions": result,
        "count": len(result),
    }


@router.get("/versions/{version_id}")
async def get_version(version_id: str):
    """Get a specific training version with current progress."""
    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(404, "Version not found")

    v_dict = version.to_dict()

    # Include progress if training
    if version.status == "training":
        version_dir = manager.get_version_dir(version_id)
        if version_dir:
            progress = ProgressTracker.load(version_id, version_dir)
            if progress:
                v_dict["progress"] = progress.to_dict()

    # Include manifest
    manifest = manager.get_manifest(version_id)
    if manifest:
        v_dict["manifest"] = manifest

    return v_dict


@router.get("/status")
async def get_training_status():
    """Get current training status."""
    manager = get_version_manager()
    return manager.get_training_status()


# ============================================================================
# Training Creation & Start
# ============================================================================

@router.post("/versions")
async def create_training(
    persona_id: str,
    recording_ids: list[str],
    rank: int = 16,
    num_epochs: int = 10,
    batch_size: int = 4,
    speaker_selections: Optional[dict[str, str]] = None,
):
    """
    Create and start a new training version.

    Args:
        persona_id: Target persona to train
        recording_ids: List of recording IDs to use
        rank: LoRA rank (4, 8, 16, 32)
        num_epochs: Number of training epochs
        batch_size: Batch size
        speaker_selections: {recording_id: speaker_id} for multi-speaker recordings
    """
    from app.services.training import get_training_audio_for_persona

    # Validate persona
    VALID_PERSONA_IDS = {"xiao_s", "caregiver", "elder_gentle", "elder_playful"}
    if persona_id not in VALID_PERSONA_IDS:
        raise HTTPException(400, f"Invalid persona_id: {persona_id}")

    # Validate recordings
    all_recordings = list_recordings_metadata()
    selected_recordings = []
    total_duration = 0.0

    for rec in all_recordings:
        if rec["recording_id"] in recording_ids:
            if rec.get("status") != "processed":
                raise HTTPException(400, f"Recording {rec['recording_id']} is not processed")
            duration = rec.get("duration_seconds", 0) or 0
            total_duration += duration
            selected_recordings.append(rec)

    if len(selected_recordings) != len(recording_ids):
        raise HTTPException(400, "Some recordings not found")

    if total_duration < 10:
        raise HTTPException(400, f"Total audio duration too short: {total_duration:.1f}s (minimum 10s)")

    # Get audio paths
    audio_files = get_training_audio_for_persona(persona_id, selected_recordings, speaker_selections)

    if not audio_files:
        raise HTTPException(400, "No valid audio files found for training")

    valid_audio_count = len([p for p, d, _ in audio_files if Path(p).exists()])
    if valid_audio_count == 0:
        raise HTTPException(400, "No audio files exist on disk")

    audio_paths = [p for p, _, _ in audio_files]
    actual_duration = sum(
        (Path(p).stat().st_size / 48000 / 2) for p in audio_paths if Path(p).exists()
    )

    # Create version
    manager = get_version_manager()
    version = manager.create_version(
        persona_id=persona_id,
        recording_ids=recording_ids,
        rank=rank,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Save manifest
    manifest = {
        "version_id": version.version_id,
        "persona_id": persona_id,
        "recordings": [
            {
                "recording_id": rec["recording_id"],
                "folder_name": rec.get("folder_name", ""),
                "speaker_used": speaker_selections.get(rec["recording_id"], "full") if speaker_selections else "full",
                "audio_path": str(path),
                "duration_seconds": duration,
            }
            for (path, duration, rec_id), rec in zip(audio_files, selected_recordings)
        ],
        "total_duration_seconds": actual_duration,
        "training_config": {
            "rank": rank,
            "learning_rate": 1e-4,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        },
    }
    manager.save_manifest(version.version_id, manifest)

    # Estimate training time
    estimated_time = ProgressTracker.estimate_training_time(actual_duration, num_epochs)

    # Start training job
    config = TrainingConfig(
        rank=rank,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    job = TrainingJob(
        version_id=version.version_id,
        version_dir=Path(version.lora_path),
        audio_paths=audio_paths,
        config=config,
        total_audio_duration=actual_duration,
    )

    _training_jobs[version.version_id] = job
    job.start()

    logger.info(f"[TRAINING] Started {version.version_id}: {len(audio_paths)} files, ~{estimated_time}s")

    return {
        "version_id": version.version_id,
        "persona_id": persona_id,
        "status": "training",
        "num_recordings": len(audio_paths),
        "total_duration_seconds": actual_duration,
        "estimated_time_seconds": estimated_time,
        "rank": rank,
        "num_epochs": num_epochs,
    }


@router.post("/versions/{version_id}/activate")
async def activate_version(version_id: str):
    """Activate a training version for use in TTS."""
    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(404, "Version not found")

    if version.status != "ready":
        raise HTTPException(400, f"Cannot activate version with status '{version.status}'. Must be 'ready'.")

    success = manager.set_active_version(version_id)
    if not success:
        raise HTTPException(500, "Failed to activate version")

    # Activate LoRA on TTS engine
    try:
        from app.services.tts.qwen_tts_engine import get_tts_engine
        engine = get_tts_engine()
        engine.activate_version(version_id)
    except Exception as e:
        logger.warning(f"[TRAINING] Failed to activate LoRA on TTS: {e}")

    logger.info(f"[TRAINING] Activated version: {version_id}")
    return {"status": "activated", "version_id": version_id}


@router.delete("/versions/{version_id}")
async def delete_version(version_id: str):
    """Delete a training version."""
    manager = get_version_manager()
    success = manager.delete_version(version_id)
    if not success:
        raise HTTPException(404, "Version not found or cannot delete active version")

    logger.info(f"[TRAINING] Deleted version: {version_id}")
    return {"status": "deleted", "version_id": version_id}


@router.get("/active")
async def get_active_version(persona_id: str):
    """Get the currently active version for a persona."""
    manager = get_version_manager()
    version = manager.get_active_version(persona_id)
    if not version:
        return {"active": False, "persona_id": persona_id}
    return {
        "active": True,
        "persona_id": persona_id,
        "version": version.to_dict(),
    }


@router.get("/versions/{version_id}/manifest")
async def get_version_manifest(version_id: str):
    """Get training manifest for a version."""
    manager = get_version_manager()
    manifest = manager.get_manifest(version_id)
    if not manifest:
        raise HTTPException(404, "Manifest not found")
    return manifest


@router.post("/versions/{version_id}/cancel")
async def cancel_training(version_id: str):
    """Cancel an ongoing training."""
    if version_id in _training_jobs:
        _training_jobs[version_id].cancel()
        del _training_jobs[version_id]
        return {"status": "cancelled", "version_id": version_id}

    manager = get_version_manager()
    version = manager.get_version(version_id)
    if version:
        version.status = "failed"
        manager._save_index()

    return {"status": "cancelled", "version_id": version_id}
