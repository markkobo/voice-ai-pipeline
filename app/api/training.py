"""
Training API endpoints.

Handles:
- Training version management
- Training trigger/status
- Model activation

NOTE: This is an MVP STUB implementation. Actual LoRA training is not yet
connected. The create_version endpoint creates a record, and simulate_training
can mark it as ready for testing purposes. Real training integration (M4) is
deferred until after MVP demo.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.services.training import get_version_manager, TrainingVersion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/versions")
async def list_versions(persona_id: Optional[str] = None):
    """List all training versions, optionally filtered by persona."""
    manager = get_version_manager()
    versions = manager.list_versions(persona_id)
    return {
        "versions": [v.to_dict() for v in versions],
        "count": len(versions),
    }


@router.get("/versions/{version_id}")
async def get_version(version_id: str):
    """Get a specific training version."""
    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(404, "Version not found")
    return version.to_dict()


@router.get("/status")
async def get_training_status():
    """Get current training status."""
    manager = get_version_manager()
    return manager.get_training_status()


@router.post("/versions")
async def create_version(persona_id: str, num_recordings: int):
    """
    Create a new training version.

    In production, this would trigger actual LoRA training.
    For now, it creates a version record and simulates training completion.
    """
    from app.services.recordings import list_recordings_metadata

    # Validate inputs
    if num_recordings < 1:
        raise HTTPException(400, "num_recordings must be at least 1")

    # Validate persona has enough recordings
    recordings = list_recordings_metadata()
    persona_recordings = [r for r in recordings if r.get("persona_id") == persona_id]

    # Filter to only processed recordings
    processed = [r for r in persona_recordings if r.get("status") == "processed"]
    if len(processed) < num_recordings:
        raise HTTPException(400, f"Not enough processed recordings. Need {num_recordings}, have {len(processed)}")

    # Get recording IDs for the recordings we'll use
    recording_ids = [r["recording_id"] for r in processed[:num_recordings]]

    manager = get_version_manager()
    version = manager.create_version(persona_id, recording_ids)

    logger.info(f"[TRAINING] Created version {version.version_id} for {persona_id}")

    return {
        "version_id": version.version_id,
        "persona_id": version.persona_id,
        "status": version.status,
        "num_recordings_used": version.num_recordings_used,
        "lora_path": version.lora_path,
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


@router.post("/simulate/{version_id}")
async def simulate_training_completion(version_id: str):
    """
    Simulate training completion for testing.

    Sets version to 'ready' status with dummy metrics.
    """
    manager = get_version_manager()
    version = manager.get_version(version_id)
    if not version:
        raise HTTPException(404, "Version not found")

    if version.status != "training":
        raise HTTPException(400, f"Version status is '{version.status}', expected 'training'")

    manager.update_version_status(
        version_id,
        status="ready",
        final_loss=0.05,
        training_time_seconds=1800,
    )

    logger.info(f"[TRAINING] Simulated completion for {version_id}")
    return {
        "status": "ready",
        "version_id": version_id,
        "final_loss": 0.05,
        "training_time_seconds": 1800,
    }
