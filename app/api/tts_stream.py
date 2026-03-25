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
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Query, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.logging_config import get_logger
from app.services.tts import (
    get_tts_engine,
    TTSStreamEvent,
    get_tts_instruct,
)

log = get_logger(__name__, component="tts_stream")

router = APIRouter()

# Active TTS sessions: session_id -> queue of audio chunks
_tts_sessions: dict[str, asyncio.Queue] = {}


async def tts_audio_stream(
    text: str,
    emotion: str,
    model_size: str,
    session_id: str,
) -> AsyncIterator[bytes]:
    """
    Generate TTS audio and yield chunks as bytes.

    Args:
        text: Text to synthesize
        emotion: Emotion tag string
        model_size: "0.6B" or "1.7B"
        session_id: Session ID for cancellation tracking

    Yields:
        Bytes chunks of PCM audio
    """
    engine = get_tts_engine(model_size=model_size)
    instruct = get_tts_instruct(emotion)

    log.info(
        f"TTS stream started: session={session_id}, emotion={emotion}, "
        f"instruct={instruct}, text_len={len(text)}"
    )

    try:
        async for event in engine.generate_streaming(
            text=text,
            instruct=instruct,
            language="Chinese",
        ):
            if event.event == "audio_chunk" and event.audio_data:
                yield event.audio_data
            elif event.event == "error":
                log.error(f"TTS stream error: {event.error}")
                break

        log.info(f"TTS stream done: session={session_id}")

    except asyncio.CancelledError:
        log.info(f"TTS stream cancelled: session={session_id}")
        raise


@router.get("/api/tts/stream")
async def tts_stream(
    text: str = Query(..., description="Text to synthesize"),
    emotion: str = Query("默認", description="Emotion tag"),
    model: str = Query("0.6B", description="TTS model size (0.6B or 1.7B)"),
):
    """
    Stream TTS audio for the given text.

    Query params:
        text: Text to synthesize (URL encoded)
        emotion: Emotion tag (e.g., 撒嬌, 寵溺)
        model: TTS model size ("0.6B" or "1.7B")

    Returns:
        StreamingResponse with audio/pcm content
    """
    if model not in ("0.6B", "1.7B"):
        model = "0.6B"

    log.info(f"TTS stream request: emotion={emotion}, model={model}, text={text[:50]}...")

    return StreamingResponse(
        tts_audio_stream(text=text, emotion=emotion, model_size=model, session_id="adhoc"),
        media_type="audio/pcm",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
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
        media_type="audio/pcm",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )
