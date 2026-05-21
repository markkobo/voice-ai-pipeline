"""
System status API.

`GET /api/system/status` returns a snapshot of:
- GPU VRAM (used / total / free in MiB) + utilisation %
- TTS engine state (which merged model is loaded + ready flag)
- ASR engine state (loaded flag)
- Training state (active + the current version's epoch / progress)
- Disk space on the data root

The UI polls this every 5s. Gating rules live in the frontend — this
endpoint is read-only and side-effect-free.

Designed to never raise: every probe is wrapped so a missing piece
(no GPU driver, no training in progress, etc.) returns sensible
defaults rather than a 500.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.api._dependencies import get_training_service
from app.services.training_service.service import TrainingService

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system"])


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------
class VramStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    used_mb: int = 0
    total_mb: int = 0
    free_mb: int = 0
    util_pct: int = 0


class TtsStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool = False
    active_version: Optional[str] = None
    model_type: Optional[str] = None


class TrainingStatusBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: bool = False
    version_id: Optional[str] = None
    persona_id: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress_pct: Optional[int] = None
    current_loss: Optional[float] = None
    # Subprocess-reported phase: "training" during the epoch loop,
    # "merging" after epochs finish while the parent merges LoRA into
    # the base model. The status bar uses this to show "merging…"
    # instead of leaving the pill stuck on "training 100% 10/10" for
    # the multi-minute merge window.
    phase: Optional[str] = None


class SystemStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vram: VramStatus
    tts: TtsStatus
    asr_ready: bool
    training: TrainingStatusBlock
    disk_free_gb: float


# ---------------------------------------------------------------------------
# Probes — each one returns sensible defaults on failure.
# ---------------------------------------------------------------------------
def _probe_vram() -> VramStatus:
    try:
        import torch

        if not torch.cuda.is_available():
            return VramStatus(available=False)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        used_bytes = total_bytes - free_bytes
        # Utilisation via nvml when available; falls back to 0 otherwise.
        util_pct = 0
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util_pct = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return VramStatus(
            available=True,
            used_mb=int(used_bytes / (1024 * 1024)),
            total_mb=int(total_bytes / (1024 * 1024)),
            free_mb=int(free_bytes / (1024 * 1024)),
            util_pct=util_pct,
        )
    except Exception as e:
        log.debug("VRAM probe failed: %s", e)
        return VramStatus(available=False)


def _probe_tts() -> TtsStatus:
    try:
        from app.services.tts.qwen_tts_engine import get_tts_engine

        engine = get_tts_engine()
        # `_current_lora_path` is set when activate_version is called;
        # `_merged_model_path` mirrors the merged-model dir.
        version_id = None
        merged_path = getattr(engine, "_merged_model_path", None)
        if merged_path:
            # path looks like ".../merged_qwen3_tts_xiao_s_v2" — last 3 underscore
            # components reconstruct the version_id is not straightforward;
            # use _current_lora_path which holds the full version dir if set.
            current = getattr(engine, "_current_lora_path", None)
            if current:
                version_id = Path(current).name
        return TtsStatus(
            ready=bool(getattr(engine, "_is_loaded", False)),
            active_version=version_id,
            model_type=getattr(engine, "_model_type", None),
        )
    except Exception as e:
        log.debug("TTS probe failed: %s", e)
        return TtsStatus()


def _probe_asr() -> bool:
    try:
        from app.api.ws_asr import state_manager

        engine = getattr(state_manager, "_default_asr", None)
        if engine is None:
            return False
        # MockASR is always "ready"; Qwen3ASR exposes _model once loaded.
        return getattr(engine, "_model", True) is not None
    except Exception as e:
        log.debug("ASR probe failed: %s", e)
        return False


def _probe_training(service: TrainingService) -> TrainingStatusBlock:
    try:
        status = service.get_training_status()
    except Exception as e:
        log.debug("Training status probe failed: %s", e)
        return TrainingStatusBlock(active=False)
    if not status.get("is_training"):
        return TrainingStatusBlock(active=False)
    version_id = status.get("version_id")
    block = TrainingStatusBlock(
        active=True,
        version_id=version_id,
        persona_id=status.get("persona_id"),
    )
    # Augment with epoch/progress from the version's progress.json if available.
    if version_id:
        try:
            snap = service.repository.read_progress(version_id)
        except Exception:
            snap = None
        if snap is not None:
            block.current_epoch = snap.current_epoch
            block.total_epochs = snap.total_epochs
            block.progress_pct = snap.progress_pct
            block.current_loss = snap.current_loss
            block.phase = snap.status.value
    return block


def _probe_disk() -> float:
    try:
        from app import config as _cfg

        usage = shutil.disk_usage(_cfg.data_root())
        return round(usage.free / (1024 ** 3), 2)
    except Exception as e:
        log.debug("Disk probe failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    training_service: TrainingService = Depends(get_training_service),
) -> SystemStatusResponse:
    return SystemStatusResponse(
        vram=_probe_vram(),
        tts=_probe_tts(),
        asr_ready=_probe_asr(),
        training=_probe_training(training_service),
        disk_free_gb=_probe_disk(),
    )
