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

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.training_service import (
    ProgressTracker,
    LoraTrainer,
    TrainingConfig,
    TrainingJob,
    SFTConfig,
    SftTrainer,
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
# Request Models
# ============================================================================

class TrainingRequest(BaseModel):
    """Request body for creating a training version."""
    persona_id: str
    segment_ids: list[str]
    rank: int = 16
    num_epochs: int = 10
    batch_size: int = 4
    training_type: str = "lora"  # "lora" or "sft"


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
                    # Update version manager status
                    manager.update_version_status(
                        version_id,
                        data["status"],
                        final_loss=data.get("final_loss"),
                        training_time_seconds=data.get("training_time_seconds")
                    )
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

        # Include progress if training, and sync status if progress.json is complete
        if v.status == "training":
            version_dir = manager.get_version_dir(v.version_id)
            if version_dir:
                progress = ProgressTracker.load(v.version_id, version_dir)
                if progress:
                    v_dict["progress"] = progress.to_dict()
                    # Sync status from progress.json if training is done
                    if progress.status in ("ready", "failed"):
                        manager.update_version_status(
                            v.version_id,
                            progress.status,
                            final_loss=progress.current_loss if progress.status == "ready" else None,
                            training_time_seconds=progress.elapsed_seconds if progress.status == "ready" else None
                        )
                        v_dict["status"] = progress.status

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
async def create_training(request: TrainingRequest):
    """
    Create and start a new training version.

    Args:
        request: Training request with persona_id, segment_ids, rank, num_epochs, batch_size
    """
    persona_id = request.persona_id
    segment_ids = request.segment_ids
    rank = request.rank
    num_epochs = request.num_epochs
    batch_size = request.batch_size

    from app.services.training import get_training_audio_for_persona

    # Validate persona
    from app.services.personas import get_persona
    if not get_persona(persona_id):
        raise HTTPException(400, f"Invalid persona_id: {persona_id}")

    # Parse segment_ids to extract recording_ids
    # segment_id format: {recording_id}_{speaker_id}
    # speaker_id is like SPEAKER_00, use _SPEAKER_ pattern to split
    rec_ids_set = set()
    for seg_id in segment_ids:
        speaker_marker = '_SPEAKER_'
        marker_pos = seg_id.rfind(speaker_marker)
        if marker_pos == -1:
            raise HTTPException(400, f"Invalid segment_id format (no {speaker_marker}): {seg_id}")
        rec_id = seg_id[:marker_pos]
        speaker_id = seg_id[marker_pos + 1:]
        rec_ids_set.add(rec_id)

    recording_ids = list(rec_ids_set)

    # Validate recordings
    all_recordings = list_recordings_metadata()
    rec_by_id = {rec["recording_id"]: rec for rec in all_recordings}
    selected_recordings = []
    total_duration = 0.0

    for rec_id in recording_ids:
        rec = rec_by_id.get(rec_id)
        if not rec:
            raise HTTPException(400, f"Recording not found: {rec_id}")
        if rec.get("status") != "processed":
            raise HTTPException(400, f"Recording {rec_id} is not processed")
        selected_recordings.append(rec)

    if len(selected_recordings) != len(recording_ids):
        raise HTTPException(400, "Some recordings not found")

    # Get audio paths via segment_ids
    audio_files = get_training_audio_for_persona(persona_id, selected_recordings, segment_ids)

    if not audio_files:
        raise HTTPException(400, "No valid audio files found for training")

    actual_duration = sum(d for _, d, _ in audio_files)
    if actual_duration < 10:
        raise HTTPException(400, f"Total audio duration too short: {actual_duration:.1f}s (minimum 10s)")

    valid_audio_count = len([p for p, d, _ in audio_files if Path(p).exists()])
    if valid_audio_count == 0:
        raise HTTPException(400, "No audio files exist on disk")

    audio_paths = [p for p, _, _ in audio_files]

    # Create version
    manager = get_version_manager()
    version = manager.create_version(
        persona_id=persona_id,
        recording_ids=recording_ids,
        rank=rank,
        num_epochs=num_epochs,
        batch_size=batch_size,
        segment_ids=segment_ids,
    )

    # Save manifest
    manifest = {
        "version_id": version.version_id,
        "persona_id": persona_id,
        "segment_ids": segment_ids,
        "training_type": request.training_type,
        "recordings": [
            {
                "recording_id": rec["recording_id"],
                "folder_name": rec.get("folder_name", ""),
                "audio_path": str(path),
                "duration_seconds": duration,
            }
            for (path, duration, rec_id), rec in zip(audio_files, selected_recordings)
        ],
        "total_duration_seconds": actual_duration,
        "training_config": {
            "rank": rank,
            "learning_rate": 1e-4 if request.training_type == "lora" else 1e-6,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "training_type": request.training_type,
        },
    }
    manager.save_manifest(version.version_id, manifest)

    # Estimate training time (SFT is ~5x slower than LoRA)
    base_estimate = ProgressTracker.estimate_training_time(actual_duration, num_epochs)
    estimated_time = base_estimate * 5 if request.training_type == "sft" else base_estimate

    # Start training job
    if request.training_type == "sft":
        # SFT training - trains full model
        config = SFTConfig(
            learning_rate=1e-6,
            num_epochs=num_epochs,
            batch_size=1,  # Small batch for full model
            gradient_accumulation_steps=8,
        )
        sft_trainer = SftTrainer(
            version_id=version.version_id,
            persona_id=persona_id,
            audio_paths=audio_paths,
            output_dir=Path(version.lora_path),
            config=config,
        )
        job = TrainingJob(
            version_id=version.version_id,
            version_dir=Path(version.lora_path),
            audio_paths=audio_paths,
            config=config,  # Pass SFTConfig
            total_audio_duration=actual_duration,
            training_type="sft",
        )
    else:
        # LoRA training (default)
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
            training_type="lora",
        )

    _training_jobs[version.version_id] = job
    job.start()

    logger.info(f"[TRAINING] Started {version.version_id} ({request.training_type}): {len(audio_paths)} files, ~{estimated_time}s")

    return {
        "version_id": version.version_id,
        "persona_id": persona_id,
        "status": "training",
        "num_recordings": len(audio_paths),
        "total_duration_seconds": actual_duration,
        "estimated_time_seconds": estimated_time,
        "rank": rank,
        "num_epochs": num_epochs,
        "training_type": request.training_type,
    }


@router.patch("/versions/{version_id}")
async def update_version(version_id: str, nickname: Optional[str] = None):
    """
    Update version metadata.

    Args:
        version_id: Version ID to update
        nickname: New nickname for the version (display name)
    """
    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(404, "Version not found")

    success = manager.update_version(version_id, nickname=nickname)
    if not success:
        raise HTTPException(500, "Failed to update version")

    return {"status": "updated", "version_id": version_id, "nickname": nickname}


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


@router.post("/voice-clone/activate")
async def activate_voice_clone(
    persona_id: str = "xiao_s",
    ref_audio_path: Optional[str] = None,
):
    """
    Activate voice clone mode using x-vector extraction from reference audio.

    This approach extracts the speaker embedding at inference time from the reference audio,
    providing better voice matching than SFT with limited training data.

    Uses the Base model with generate_voice_clone_streaming and xvec_only=True.
    """
    try:
        from app.services.tts.qwen_tts_engine import get_tts_engine
        engine = get_tts_engine()
        engine.activate_voice_clone(persona_id, ref_audio_path)

        ref_used = getattr(engine, '_ref_audio_path', None)
        model_type = getattr(engine, '_model_type', None)

        logger.info(f"[TRAINING] Activated voice clone for {persona_id}, ref: {ref_used}")
        return {
            "status": "activated",
            "mode": "voice_clone",
            "persona_id": persona_id,
            "ref_audio_path": ref_used,
            "model_type": model_type,
        }
    except Exception as e:
        logger.error(f"[TRAINING] Failed to activate voice clone: {e}")
        raise HTTPException(500, str(e))


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


@router.post("/versions/{version_id}/preview")
async def preview_version(version_id: str, text: Optional[str] = None):
    """
    Generate preview audio for a training version.

    Activates the version and synthesizes a test phrase to hear the voice.
    Returns streaming audio/wav.

    Args:
        version_id: The training version to preview
        text: Optional custom preview text (default: "你好，這是我的聲音測試。")
    """
    from app.services.tts.qwen_tts_engine import get_tts_engine

    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(404, "Version not found")

    if version.status != "ready":
        raise HTTPException(400, f"Cannot preview version with status '{version.status}'. Must be 'ready'.")

    # Activate the version on TTS engine
    engine = get_tts_engine()
    engine.activate_version(version_id)

    # Preview text (use custom or default)
    preview_text = text.strip() if text and text.strip() else "你好，這是我的聲音測試。"

    async def audio_stream():
        """Generate audio chunks and yield them as WAV data."""
        import io
        import wave

        audio_chunks = []
        sample_rate = 24000

        async for event in engine.generate_streaming(
            text=preview_text,
            instruct=None,  # Path B: Use merged model's voice, no instruct
            language="Chinese",
        ):
            if event.event == "audio_chunk" and event.audio_data:
                audio_chunks.append(event.audio_data)

        if audio_chunks:
            full_audio = b"".join(audio_chunks)
            # Convert to WAV format
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(full_audio)
            wav_io.seek(0)
            yield wav_io.read()

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename=preview_{version_id}.wav"}
    )
