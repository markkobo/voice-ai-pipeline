"""
TTS streaming endpoint — HTTP streaming of audio chunks to browser.

Browser connects to GET /api/tts/stream with query params:
  ?emotion=撒嬌&instruct=(coquettish...)&model=0.6B

Server streams PCM audio chunks via HTTP streaming response.
"""
import asyncio
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Query, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.logging_config import get_logger
from app.services.tts import (
    get_tts_engine,
    TTSStreamEvent,
    enhance_text,
)
from telemetry import metrics

log = get_logger(__name__, component="tts_stream")

router = APIRouter()

# Active TTS sessions: session_id -> queue of audio chunks
_tts_sessions: dict[str, asyncio.Queue] = {}


def _get_persona_reference_audio(persona_id: str) -> Optional[str]:
    """
    Get reference audio path for a persona.

    Returns path to reference audio file if available, None otherwise.
    Currently uses the latest processed recording for the persona.
    """
    try:
        from app.services.training import get_version_manager
        manager = get_version_manager()
        version = manager.get_active_version(persona_id)
        if version and version.lora_path:
            # Check for reference audio in version directory
            version_dir = Path(version.lora_path)
            # Look for reference audio (prefer enhanced, then denoised, then raw)
            for audio_name in ["enhanced_audio.wav", "reference_audio.wav"]:
                ref_audio = version_dir / audio_name
                if ref_audio.exists():
                    return str(ref_audio)
            # Fall back to first recording's enhanced audio
            manifest = manager.get_manifest(version.version_id)
            if manifest and manifest.get("recordings"):
                first_rec = manifest["recordings"][0]
                rec_path = Path(first_rec.get("audio_path", ""))
                if rec_path.exists():
                    return str(rec_path)
    except Exception as e:
        log.warning(f"Failed to get reference audio for {persona_id}: {e}")
    return None


def make_wav_header(num_samples: int, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Generate a WAV file header for PCM audio."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * bits_per_sample // 8
    file_size = 36 + data_size

    import struct
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        file_size,         # ChunkSize
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (PCM)
        1,                 # AudioFormat (PCM)
        num_channels,      # NumChannels
        sample_rate,       # SampleRate
        byte_rate,         # ByteRate
        block_align,       # BlockAlign
        bits_per_sample,   # BitsPerSample
        b'data',           # Subchunk2ID
        data_size,         # Subchunk2Size
    )
    return header


async def tts_audio_stream(
    text: str,
    emotion: str,
    model_size: str,
    session_id: str,
    persona_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """
    Generate TTS audio and yield chunks as they arrive.

    Args:
        text: Text to synthesize
        emotion: Emotion tag string
        model_size: "0.6B" or "1.7B"
        session_id: Session ID for cancellation tracking
        persona_id: Persona ID for voice clone (optional)

    Yields:
        Bytes chunks of WAV audio (header first, then PCM chunks)
    """
    engine = get_tts_engine(model_size=model_size)
    # Path B: Use text enhancement instead of instruct strings
    enhanced_text = enhance_text(text, emotion)

    # Look up reference audio for voice clone if persona_id is provided
    reference_audio = None
    if persona_id:
        # First try version-based reference audio (SFT/merged model path)
        reference_audio = _get_persona_reference_audio(persona_id)
        if reference_audio:
            log.info(f"Using voice clone for persona {persona_id}: {reference_audio}")
        else:
            # Fall back to TTS engine's auto-find for voice_clone mode
            from app.services.tts.qwen_tts_engine import FasterQwenTTSEngine
            reference_audio = FasterQwenTTSEngine.find_reference_audio(persona_id)
            if reference_audio:
                # Activate voice_clone mode on the engine
                engine.activate_voice_clone(persona_id, reference_audio)
                log.info(f"Activated voice clone mode for persona {persona_id}: {reference_audio}")

    log.info(
        f"TTS stream started: session={session_id}, emotion={emotion}, "
        f"voice_clone={reference_audio is not None}, text_len={len(text)}"
    )

    sample_rate = 24000
    header_sent = False
    total_samples = 0
    first_chunk_time = None
    stream_start_time = time.perf_counter()

    try:
        async for event in engine.generate_streaming(
            text=enhanced_text,
            instruct=None,  # Path B: Use text enhancement
            language="Chinese",
            reference_audio=reference_audio,
        ):
            if event.event == "audio_chunk" and event.audio_data:
                chunk = event.audio_data

                # Record time to first chunk
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    ttfc = first_chunk_time - stream_start_time
                    metrics.tts_first_chunk.labels(
                        component="tts_stream",
                        model=model_size,
                    ).observe(ttfc)
                    log.info(f"TTS first chunk: {ttfc:.3f}s")

                total_samples += len(chunk) // 2  # 16-bit samples

                # Stream chunks immediately as they arrive
                if not header_sent:
                    # Estimate based on text length: ~150-200 chars/sec, 24kHz sample rate
                    # For Chinese, roughly 1 char = 100-150 samples (0.1 second = 2400 samples for 2 chars)
                    chars_per_sec = 150
                    estimated_duration_sec = max(1.0, len(text) / chars_per_sec)
                    buffer_samples = int(estimated_duration_sec * sample_rate)
                    buffer_samples = max(buffer_samples, 24000)  # At least 1 second
                    wav_header = make_wav_header(buffer_samples, sample_rate)
                    yield wav_header
                    header_sent = True
                    log.info(f"WAV header: estimated {estimated_duration_sec:.1f}s = {buffer_samples} samples")

                yield chunk

            elif event.event == "error":
                log.error(f"TTS stream error: {event.error}")
                break

        log.info(f"TTS stream done: session={session_id}, total_samples={total_samples}")

    except asyncio.CancelledError:
        log.info(f"TTS stream cancelled: session={session_id}")
        raise


async def tts_raw_stream(
    text: str,
    emotion: str,
    model_size: str,
    session_id: str,
    persona_id: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """
    Generate TTS audio and yield raw PCM chunks (no WAV header).

    For use with WebSocket streaming or AudioWorklet playback.

    Yields:
        Raw PCM 16-bit mono audio chunks as they arrive (not buffered).
        Each yield sends a chunk immediately to enable true streaming playback.
    """
    engine = get_tts_engine(model_size=model_size)
    # Path B: Use text enhancement instead of instruct strings
    enhanced_text = enhance_text(text, emotion)

    # Look up reference audio for voice clone if persona_id is provided
    reference_audio = None
    if persona_id:
        # First try version-based reference audio (SFT/merged model path)
        reference_audio = _get_persona_reference_audio(persona_id)
        if reference_audio:
            log.info(f"Using voice clone for persona {persona_id}: {reference_audio}")
        else:
            # Fall back to TTS engine's auto-find for voice_clone mode
            from app.services.tts.qwen_tts_engine import FasterQwenTTSEngine
            reference_audio = FasterQwenTTSEngine.find_reference_audio(persona_id)
            if reference_audio:
                # Activate voice_clone mode on the engine
                engine.activate_voice_clone(persona_id, reference_audio)
                log.info(f"Activated voice clone mode for persona {persona_id}: {reference_audio}")

    log.info(
        f"TTS raw stream started: session={session_id}, emotion={emotion}, "
        f"voice_clone={reference_audio is not None}, text_len={len(text)}"
    )

    try:
        # Stream chunks as they arrive (not buffered) - yields immediately per chunk
        async for event in engine.generate_streaming(
            text=enhanced_text,
            instruct=None,  # Path B: Use text enhancement
            language="Chinese",
            reference_audio=reference_audio,
        ):
            if event.event == "audio_chunk" and event.audio_data:
                yield event.audio_data
            elif event.event == "error":
                log.error(f"TTS stream error: {event.error}")
                break

        log.info(f"TTS raw stream done: session={session_id}")

    except asyncio.CancelledError:
        log.info(f"TTS raw stream cancelled: session={session_id}")
        raise


@router.get("/api/tts/stream")
async def tts_stream(
    text: str = Query(..., description="Text to synthesize"),
    emotion: str = Query("默認", description="Emotion tag"),
    model: str = Query("0.6B", description="TTS model size (0.6B or 1.7B)"),
    persona_id: str = Query(None, description="Persona ID for voice clone"),
):
    """
    Stream TTS audio for the given text.

    Query params:
        text: Text to synthesize (URL encoded)
        emotion: Emotion tag (e.g., 撒嬌, 寵溺)
        model: TTS model size ("0.6B" or "1.7B")
        persona_id: Persona ID for voice clone (optional)

    Returns:
        StreamingResponse with audio/wav content
    """
    if model not in ("0.6B", "1.7B"):
        model = "0.6B"

    log.info(f"TTS stream request: emotion={emotion}, model={model}, persona={persona_id}, text={text[:50]}...")

    return StreamingResponse(
        tts_audio_stream(text=text, emotion=emotion, model_size=model, session_id="adhoc", persona_id=persona_id),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
        },
    )


@router.get("/api/tts/raw")
async def tts_raw(
    text: str = Query(..., description="Text to synthesize"),
    emotion: str = Query("默認", description="Emotion tag"),
    model: str = Query("0.6B", description="TTS model size (0.6B or 1.7B)"),
    persona_id: str = Query(None, description="Persona ID for voice clone"),
):
    """
    Stream TTS audio as raw PCM (16-bit, 24kHz, mono).

    For use with Web Audio API / AudioWorklet playback.

    Query params:
        text: Text to synthesize (URL encoded)
        emotion: Emotion tag
        model: TTS model size ("0.6B" or "1.7B")
        persona_id: Persona ID for voice clone (optional)

    Returns:
        StreamingResponse with audio/pcm content
    """
    if model not in ("0.6B", "1.7B"):
        model = "0.6B"

    log.info(f"TTS raw request: emotion={emotion}, model={model}, persona={persona_id}, text={text[:50]}...")

    return StreamingResponse(
        tts_raw_stream(text=text, emotion=emotion, model_size=model, session_id="adhoc", persona_id=persona_id),
        media_type="audio/pcm",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
            "X-Audio-SampleRate": "24000",
            "X-Audio-Channels": "1",
            "X-Audio-Bits": "16",
        },
    )


@router.post("/api/tts/session")
async def create_tts_session(
    text: str,
    emotion: str = "默認",
    model: str = "0.6B",
    session_id: Optional[str] = None,
):
    """
    Create a named TTS session for better tracking.

    POST body:
        text: Text to synthesize
        emotion: Emotion tag
        model: TTS model size
        session_id: Optional session ID

    Returns:
        {"session_id": "...", "stream_url": "/api/tts/stream/{session_id}"}
    """
    sid = session_id or str(uuid.uuid4())

    stream_url = f"/api/tts/stream/{sid}"

    log.info(f"TTS session created: {sid}, emotion={emotion}, model={model}")

    return {
        "session_id": sid,
        "stream_url": stream_url,
        "text": text,
        "emotion": emotion,
        "model": model,
    }


@router.get("/api/tts/stream/{session_id}")
async def tts_stream_named(
    session_id: str,
    text: str = Query(..., description="Text to synthesize"),
    emotion: str = Query("默認", description="Emotion tag"),
    model: str = Query("0.6B", description="TTS model size"),
):
    """
    Stream TTS audio for a named session.

    Named sessions allow client to track/manage multiple TTS streams.
    """
    if model not in ("0.6B", "1.7B"):
        model = "0.6B"

    log.info(f"TTS named stream: session={session_id}, emotion={emotion}, model={model}")

    return StreamingResponse(
        tts_audio_stream(
            text=text,
            emotion=emotion,
            model_size=model,
            session_id=session_id,
        ),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
        },
    )
