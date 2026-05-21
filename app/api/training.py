"""
Training REST API — routes only.

Business logic lives in `app.services.training_service.service.TrainingService`.
Routes deserialize input, call the service, serialize output.

Notable Phase 1.2 changes:
- Every body is a Pydantic model with `extra="forbid"`.
- PATCH /versions/{id} and POST /voice-clone/activate now take bodies (used
  to be query params — inconsistent with REST conventions).
- The SSE progress stream replaces the unbounded `while True` poll with a
  bounded `asyncio.sleep` loop that emits a `timeout` event after
  `SSE_MAX_WAIT_SECONDS` of no activity.
- `except Exception:` swallows around TTS/ASR unload, LoRA activation, and
  voice-clone activation are replaced with explicit error mapping or logged
  raises — the API no longer claims success when an integration step failed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api._dependencies import get_training_service
from app.api._errors import (
    InvalidTrainingParamsError,
    TrainingVersionNotFoundError,
    VersionNotReadyError,
)
from app.services.training_service.models import (
    TrainingManifest,
    TrainingProgressSnapshot,
    TrainingType,
    TrainingVersion,
    VersionStatus,
)
from app.services.training_service.service import TrainingService

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
SSE_POLL_INTERVAL_SECONDS = 2
SSE_MAX_WAIT_SECONDS = 30 * 60  # 30 min before declaring the writer stuck


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class CreateTrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    segment_ids: list[str]
    rank: int = 16
    num_epochs: int = 10
    batch_size: int = 4
    training_type: TrainingType = TrainingType.lora
    learning_rate: Optional[float] = None


class UpdateVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nickname: Optional[str] = None


class VoiceCloneActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str = "xiao_s"
    ref_audio_path: Optional[str] = None


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = None
    ref_audio_path: Optional[str] = None


class CreateTrainingResponse(BaseModel):
    version_id: str
    persona_id: str
    status: VersionStatus
    num_recordings: int
    total_duration_seconds: float
    estimated_time_seconds: int
    rank: int
    num_epochs: int
    training_type: TrainingType


class ListVersionsResponse(BaseModel):
    versions: list[dict]
    count: int


class TrainingStatusResponse(BaseModel):
    is_training: bool
    version_id: Optional[str] = None
    persona_id: Optional[str] = None
    status: Optional[str] = None


class UpdateVersionResponse(BaseModel):
    status: str = "updated"
    version_id: str
    nickname: Optional[str] = None


class ActivateResponse(BaseModel):
    status: str = "activated"
    version_id: str


class VoiceCloneActivateResponse(BaseModel):
    status: str = "activated"
    mode: str = "voice_clone"
    persona_id: str
    ref_audio_path: Optional[str] = None
    model_type: Optional[str] = None


class DeleteVersionResponse(BaseModel):
    status: str = "deleted"
    version_id: str


class CancelResponse(BaseModel):
    status: str = "cancelled"
    version_id: str


class ActiveVersionResponse(BaseModel):
    active: bool
    persona_id: str
    version: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _version_with_progress(
    service: TrainingService, version: TrainingVersion
) -> dict:
    """Compose the legacy `{...version, progress?, manifest?}` response."""
    payload = version.to_legacy_dict()
    if version.status == VersionStatus.training:
        try:
            progress = service.read_progress(version.version_id)
        except TrainingVersionNotFoundError:
            progress = None
        if progress is not None:
            payload["progress"] = progress.model_dump(mode="json")
            # If the writer says we're done, sync the version status. This
            # was an implicit side-effect of list_versions in the legacy
            # code — now we do it explicitly here.
            if progress.status.value in ("ready", "failed"):
                refreshed = service.refresh_status_from_progress(version.version_id)
                payload = refreshed.to_legacy_dict()
                payload["progress"] = progress.model_dump(mode="json")
    return payload


# ---------------------------------------------------------------------------
# SSE: training progress stream
# ---------------------------------------------------------------------------
async def _sse_progress_generator(
    service: TrainingService,
    version_id: str,
):
    """
    Bounded poll of progress.json with explicit terminal/timeout events.

    Emits:
        data: {"event": "progress", ...snapshot}
        data: {"event": "complete", ...final}
        data: {"event": "error", "error": "..."}
        data: {"event": "timeout"}

    The legacy generator had no max-wait and no timeout event — if the
    subprocess writer hung, the SSE connection looped forever.
    """
    # Ensure version exists before streaming.
    try:
        service.get_version(version_id)
    except TrainingVersionNotFoundError:
        yield f"data: {json.dumps({'event': 'error', 'error': 'Version not found'})}\n\n"
        return

    last_snapshot: Optional[dict] = None
    waited = 0.0
    while waited < SSE_MAX_WAIT_SECONDS:
        try:
            progress = service.read_progress(version_id)
        except TrainingVersionNotFoundError:
            yield f"data: {json.dumps({'event': 'error', 'error': 'Version not found'})}\n\n"
            return
        except Exception as e:
            log.exception("SSE progress read failed for %s", version_id)
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
            return

        if progress is not None:
            snapshot = progress.model_dump(mode="json")
            if progress.status.value in ("ready", "failed"):
                service.refresh_status_from_progress(version_id)
                event_name = "complete" if progress.status.value == "ready" else "error"
                payload = {"event": event_name, **snapshot}
                if event_name == "error":
                    payload["error"] = (
                        progress.error_message or "Training failed"
                    )
                yield f"data: {json.dumps(payload)}\n\n"
                return
            if snapshot != last_snapshot:
                yield f"data: {json.dumps({'event': 'progress', **snapshot})}\n\n"
                last_snapshot = snapshot

        await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)
        waited += SSE_POLL_INTERVAL_SECONDS

    yield f"data: {json.dumps({'event': 'timeout', 'version_id': version_id})}\n\n"


@router.get("/versions/{version_id}/progress")
async def stream_training_progress(
    version_id: str,
    service: TrainingService = Depends(get_training_service),
) -> StreamingResponse:
    """SSE stream for training progress. Client reconnects here to resume."""
    # Eagerly check existence so 404 returns synchronously (better client UX
    # than a 200 followed by an error event).
    service.get_version(version_id)
    return StreamingResponse(
        _sse_progress_generator(service, version_id),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Version list / get / status
# ---------------------------------------------------------------------------
@router.get("/versions", response_model=ListVersionsResponse)
async def list_versions(
    persona_id: Optional[str] = None,
    service: TrainingService = Depends(get_training_service),
) -> ListVersionsResponse:
    versions = service.list_versions(persona_id)
    # Side-effect-free: if a version is in 'training' status, eagerly sync
    # from progress.json so the UI reflects completed runs immediately.
    payload: list[dict] = []
    for v in versions:
        if v.status == VersionStatus.training:
            try:
                v = service.refresh_status_from_progress(v.version_id)
            except TrainingVersionNotFoundError:
                continue
        payload.append(_version_with_progress(service, v))
    return ListVersionsResponse(versions=payload, count=len(payload))


@router.get("/versions/{version_id}")
async def get_version(
    version_id: str,
    service: TrainingService = Depends(get_training_service),
) -> dict:
    version = service.get_version(version_id)
    payload = _version_with_progress(service, version)
    manifest = service.repository.get_manifest(version_id)
    if manifest is not None:
        payload["manifest"] = manifest.model_dump(mode="json")
    return payload


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status(
    service: TrainingService = Depends(get_training_service),
) -> TrainingStatusResponse:
    return TrainingStatusResponse(**service.get_training_status())


# ---------------------------------------------------------------------------
# Create + start training
# ---------------------------------------------------------------------------
@router.post("/versions", response_model=CreateTrainingResponse)
async def create_training(
    body: CreateTrainingRequest,
    service: TrainingService = Depends(get_training_service),
) -> CreateTrainingResponse:
    """
    Validate, resolve audio, persist, and start a background training job.

    Validation surfaces explicit InvalidTrainingParamsError /
    NoTrainingAudioError responses — no more 400s with opaque strings.
    """
    result = service.create_version(
        persona_id=body.persona_id,
        segment_ids=body.segment_ids,
        rank=body.rank,
        num_epochs=body.num_epochs,
        batch_size=body.batch_size,
        training_type=body.training_type,
        learning_rate=body.learning_rate,
    )
    version = result.version
    return CreateTrainingResponse(
        version_id=version.version_id,
        persona_id=version.persona_id,
        status=version.status,
        num_recordings=len(version.recording_ids_used),
        total_duration_seconds=result.total_duration_seconds,
        estimated_time_seconds=result.estimated_time_seconds,
        rank=version.rank,
        num_epochs=version.num_epochs,
        training_type=version.training_type or body.training_type,
    )


# ---------------------------------------------------------------------------
# Update / activate / delete / cancel
# ---------------------------------------------------------------------------
@router.patch("/versions/{version_id}", response_model=UpdateVersionResponse)
async def update_version(
    version_id: str,
    body: UpdateVersionRequest,
    service: TrainingService = Depends(get_training_service),
) -> UpdateVersionResponse:
    updated = service.set_nickname(version_id, body.nickname)
    return UpdateVersionResponse(version_id=version_id, nickname=updated.nickname)


@router.post("/versions/{version_id}/activate", response_model=ActivateResponse)
async def activate_version(
    version_id: str,
    service: TrainingService = Depends(get_training_service),
) -> ActivateResponse:
    """
    Activate a ready version. Then attempts to load the merged LoRA into the
    live TTS engine — failures there are no longer silently swallowed.
    """
    service.activate_version(version_id)

    # Side-effect: load the LoRA into the live TTS engine. This is an
    # integration concern, kept out of the service. Failure here is a real
    # error and is logged + raised (used to be silently swallowed → caller
    # got a 200 but the engine was still on the old model).
    try:
        from app.services.tts.qwen_tts_engine import get_tts_engine

        engine = get_tts_engine()
        engine.activate_version(version_id)
    except ImportError as e:
        # TTS engine not importable in test env — that's fine.
        log.info("TTS engine unavailable; skipping live activation: %s", e)
    except Exception as e:
        log.exception("Failed to activate %s on TTS engine", version_id)
        # 502 conveys "downstream integration failed" better than 500.
        raise VersionNotReadyError(
            f"Activated in repository but TTS engine failed to load: {e}",
            details={"version_id": version_id, "stage": "tts_engine"},
        ) from e

    log.info("[TRAINING] Activated %s", version_id)
    return ActivateResponse(version_id=version_id)


@router.post("/voice-clone/activate", response_model=VoiceCloneActivateResponse)
async def activate_voice_clone(
    body: VoiceCloneActivateRequest,
) -> VoiceCloneActivateResponse:
    """
    Activate voice-clone mode (x-vector from a reference audio file).

    Pure integration with the TTS engine — no service state to mutate.
    Errors propagate as DomainError instead of HTTP 500 with raw `str(e)`.
    """
    try:
        from app.services.tts.qwen_tts_engine import get_tts_engine

        engine = get_tts_engine()
        engine.activate_voice_clone(body.persona_id, body.ref_audio_path)
        ref_used = getattr(engine, "_ref_audio_path", None)
        model_type = getattr(engine, "_model_type", None)
    except ImportError as e:
        log.info("TTS engine unavailable: %s", e)
        raise InvalidTrainingParamsError(
            "TTS engine not available in this environment",
            details={"persona_id": body.persona_id},
        ) from e
    except Exception as e:
        log.exception("Failed to activate voice clone for %s", body.persona_id)
        raise InvalidTrainingParamsError(
            f"Voice-clone activation failed: {e}",
            details={"persona_id": body.persona_id},
        ) from e

    return VoiceCloneActivateResponse(
        persona_id=body.persona_id,
        ref_audio_path=ref_used,
        model_type=model_type,
    )


@router.delete("/versions/{version_id}", response_model=DeleteVersionResponse)
async def delete_version(
    version_id: str,
    service: TrainingService = Depends(get_training_service),
) -> DeleteVersionResponse:
    service.delete_version(version_id)
    return DeleteVersionResponse(version_id=version_id)


@router.get("/active", response_model=ActiveVersionResponse)
async def get_active_version(
    persona_id: str,
    service: TrainingService = Depends(get_training_service),
) -> ActiveVersionResponse:
    version = service.get_active(persona_id)
    if version is None:
        return ActiveVersionResponse(active=False, persona_id=persona_id, version=None)
    return ActiveVersionResponse(
        active=True,
        persona_id=persona_id,
        version=version.to_legacy_dict(),
    )


@router.get("/versions/{version_id}/manifest", response_model=TrainingManifest)
async def get_version_manifest(
    version_id: str,
    service: TrainingService = Depends(get_training_service),
) -> TrainingManifest:
    return service.get_manifest(version_id)


@router.post("/versions/{version_id}/cancel", response_model=CancelResponse)
async def cancel_training(
    version_id: str,
    service: TrainingService = Depends(get_training_service),
) -> CancelResponse:
    service.cancel_version(version_id)
    return CancelResponse(version_id=version_id)


# ---------------------------------------------------------------------------
# Preview — generates a sample WAV from the activated voice.
# Kept simple: it's a thin integration wrapper. Heavy lifting is the TTS engine.
# ---------------------------------------------------------------------------
@router.post("/versions/{version_id}/preview")
async def preview_version(
    version_id: str,
    body: Optional[PreviewRequest] = None,
    service: TrainingService = Depends(get_training_service),
) -> StreamingResponse:
    version = service.get_version(version_id)
    if version.status != VersionStatus.ready:
        raise VersionNotReadyError(
            f"Cannot preview version with status {version.status.value!r}",
            details={"version_id": version_id, "current_status": version.status.value},
        )

    try:
        from app.services.tts.qwen_tts_engine import (
            get_tts_engine,
            get_tts_generation_lock,
        )
    except ImportError as e:
        raise InvalidTrainingParamsError(
            "TTS engine not available in this environment",
            details={"version_id": version_id},
        ) from e

    engine = get_tts_engine()
    engine.activate_version(version_id)

    is_sft_model = version.model_type in (
        "sft",
        "custom_voice",
        "custom_voice_compatible",
    )
    body = body or PreviewRequest()
    ref_audio = body.ref_audio_path if (body.ref_audio_path and not is_sft_model) else None

    if ref_audio:
        engine.activate_voice_clone(version.persona_id, ref_audio_path=ref_audio)

    preview_text = (body.text or "你好，這是我的聲音測試。").strip()

    # Generate the full WAV synchronously instead of streaming — this lets
    # us raise an explicit HTTP 500 with a real error body when the
    # engine produces zero audio chunks. The legacy StreamingResponse
    # variant returned HTTP 200 with a 0-byte body in that case, which
    # the browser silently played as nothing (root cause of the
    # 2026-05-21 "v2_20260521_144817_303063 preview silent" report —
    # merge_lora produced a Base-arch model that the engine couldn't
    # drive, so chunk_producer raised inside the generator and audio_chunks
    # stayed empty).
    import io
    import wave

    sample_rate = 24000
    lock = get_tts_generation_lock()
    async with lock:
        audio_chunks: list[bytes] = []
        last_error: Optional[str] = None
        try:
            async for event in engine.generate_streaming(
                text=preview_text,
                instruct=None,
                language="Chinese",
            ):
                if event.event == "audio_chunk" and event.audio_data:
                    audio_chunks.append(event.audio_data)
                elif event.event == "error" and getattr(event, "error", None):
                    last_error = str(event.error)
        except Exception as gen_err:
            last_error = f"{type(gen_err).__name__}: {gen_err}"

    if not audio_chunks:
        from fastapi import HTTPException
        detail = (
            f"TTS engine produced no audio for version {version_id!r}. "
            f"Likely cause: merged model is Base-architecture without "
            f"a baked speaker embedding (tts_model_type=base), or LoRA "
            f"merge failed. Check server logs for [TTS] / [MERGE] errors."
        )
        if last_error:
            detail += f" Engine reported: {last_error}"
        raise HTTPException(status_code=500, detail=detail)

    full_audio = b"".join(audio_chunks)
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(full_audio)
    wav_io.seek(0)
    wav_bytes = wav_io.read()

    async def audio_stream():
        yield wav_bytes

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename=preview_{version_id}.wav"},
    )
