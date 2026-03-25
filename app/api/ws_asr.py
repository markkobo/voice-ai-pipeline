"""
WebSocket ASR endpoint with LLM streaming and emotion parsing.

Pipeline:
  1. Client sends config: {type: "config", audio, persona_id, listener_id, model}
  2. Client sends Binary WebM audio chunks
  3. Server decodes WebM → PCM
  4. VAD detects speech → commit_utterance → ASR → LLM streaming
  5. LLM emits tokens → emotion tag parsed → client fetches TTS audio via HTTP
  6. Client sends new audio = barge-in → cancel LLM task

Protocol:
  Client → Server:
    {"type": "config", "audio": {...}, "persona_id": "xiao_s", "listener_id": "child", "model": "gpt-4o-mini"}
    {"type": "control", "action": "commit_utterance"}
    Binary WebM audio chunks (from MediaRecorder)

  Server → Client:
    {"type": "asr_partial", "text": "..."}
    {"type": "asr_result", "is_final": true, "text": "...", "telemetry": {...}}
    {"type": "llm_start", "ttft_seconds": 0.3}
    {"type": "llm_token", "content": "「", "emotion": null}
    {"type": "llm_token", "content": "好", "emotion": "寵溺"}  <- first emotion detected
    {"type": "tts_ready", "text": "「好啦～", "emotion": "寵溺", "instruct": "(gentle...)", "stream_url": "/api/tts/stream?..."}
    {"type": "llm_done", "text": "「寵溺好啦～...", "total_tokens": 10}
    {"type": "llm_cancelled"}  <- barge-in
    {"type": "vad_commit"}  <- VAD detected end of speech
"""
import asyncio
import base64
import io
import json
import os
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.state_manager import StateManager
from app.services.llm import OpenAIClient, MockLLMClient, PersonaManager, PersonaType
from app.services.tts import EmotionMapper, get_tts_instruct
from app.logging_config import get_logger
from telemetry import metrics, rag_retrieval_seconds


log = get_logger(__name__, component="ws")

router = APIRouter()

use_qwen = os.getenv("USE_QWEN_ASR", "true").lower() == "true"
use_mock_llm = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
state_manager = StateManager(use_qwen=use_qwen)

# LLM clients
llm_client: Optional[OpenAIClient | MockLLMClient] = None
prompt_manager = PersonaManager()

# Audio decoder (lazy import to handle missing pydub)
_audio_decoder = None


def _get_audio_decoder():
    """Lazy-load audio decoder."""
    global _audio_decoder
    if _audio_decoder is None:
        try:
            from pydub import AudioSegment
            _audio_decoder = "pydub"
            log.info("Using pydub for audio decoding")
        except ImportError:
            log.warning("pydub not available, using fallback raw PCM")
            _audio_decoder = "none"
    return _audio_decoder


def decode_webm_to_pcm(webm_bytes: bytes, target_sample_rate: int = 24000) -> Optional[bytes]:
    """
    Decode WebM/Opus audio to PCM 16-bit mono.

    Args:
        webm_bytes: Raw WebM audio bytes from MediaRecorder
        target_sample_rate: Target sample rate (default 24kHz)

    Returns:
        PCM 16-bit mono bytes, or None if decode fails
    """
    decoder = _get_audio_decoder()

    if decoder == "none":
        # Fallback: assume raw PCM (no decode)
        return webm_bytes

    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(
            io.BytesIO(webm_bytes),
            format="webm"
        )

        # Convert to mono 24kHz PCM 16-bit
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sample_rate)
        audio = audio.set_sample_width(2)

        return audio.raw_data

    except Exception as e:
        log.warning(f"Audio decode failed: {e}, returning raw bytes")
        return webm_bytes


def get_llm_client() -> OpenAIClient | MockLLMClient:
    """Lazily initialize and return the LLM client."""
    global llm_client
    if llm_client is None:
        if use_mock_llm:
            llm_client = MockLLMClient()
            log.info("Using MockLLMClient")
        else:
            llm_client = OpenAIClient(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            log.info("Using OpenAIClient")
    return llm_client


# Emotion tag regex — matches [情感: 撒嬌] or [情感:撒嬌]
EMOTION_TAG_RE = re.compile(r'^\[情感[:：]\s*(.*?)\]\s*')


async def run_llm_stream(
    websocket: WebSocket,
    session_id: str,
    asr_text: str,
    persona_id: Optional[str],
    listener_id: Optional[str],
    llm_model: Optional[str],
) -> None:
    """
    Run LLM streaming and parse emotion tags.

    When first emotion is detected, send tts_ready with stream URL.
    This task is registered in StateManager so it can be cancelled on barge-in.
    """
    client = get_llm_client()
    cancellation_event = asyncio.Event()

    # Register task in state manager for cancellation
    task = asyncio.current_task()
    if task:
        state_manager.set_llm_task(session_id, task, cancellation_event)

    # Get persona-aware system prompt
    system_prompt = prompt_manager.get_prompt(
        persona_id=persona_id or "xiao_s",
        listener_id=listener_id,
    )

    e2e_start = time.perf_counter()
    accumulated_text = ""
    first_token_sent = False
    ttft_seconds: Optional[float] = None
    tts_notified = False

    # Emotion mapper for this turn
    emotion_mapper = EmotionMapper()

    # TTS state: accumulates text AFTER emotion tag is removed
    tts_text_parts: list[str] = []
    current_emotion: Optional[str] = None
    current_instruct: Optional[str] = None

    try:
        async for event in client.stream(
            prompt=asr_text,
            system_prompt=system_prompt,
            cancellation_event=cancellation_event,
        ):
            if event.event.value == "start":
                await websocket.send_text(json.dumps({
                    "type": "llm_start",
                    "utterance_id": session_id,
                }))

            elif event.event.value == "content_delta":
                accumulated_text += event.content

                # Parse emotion from this delta
                new_emotion, cleaned_content = emotion_mapper.update(event.content)

                if new_emotion and not tts_notified:
                    # First emotion detected — notify client to fetch TTS
                    tts_notified = True
                    current_emotion = new_emotion
                    current_instruct = get_tts_instruct(new_emotion)
                    # At this point, cleaned_content is empty (tag consumed buffer)
                    # Next tokens will have actual text
                    tts_text_parts = []  # Start fresh after tag

                    await websocket.send_text(json.dumps({
                        "type": "tts_ready",
                        "text": "",  # No text yet, emotion only
                        "emotion": current_emotion,
                        "instruct": current_instruct,
                        "stream_url": f"/api/tts/stream?emotion={current_emotion}&model=0.6B",
                    }))

                    await websocket.send_text(json.dumps({
                        "type": "llm_token",
                        "content": cleaned_content,
                        "emotion": current_emotion,
                    }))

                    # Record TTFT
                    if not first_token_sent and event.ttft_seconds is not None:
                        ttft_seconds = event.ttft_seconds
                        first_token_sent = True

                elif tts_notified:
                    # TTS already notified — accumulate text for TTS
                    if cleaned_content:
                        tts_text_parts.append(cleaned_content)
                        tts_text = "".join(tts_text_parts)

                        # Update TTS with new text (client can fetch updated text)
                        import urllib.parse
                        params = urllib.parse.urlencode({
                            "text": tts_text,
                            "emotion": current_emotion,
                            "model": "0.6B",
                        })
                        stream_url = f"/api/tts/stream?{params}"

                        await websocket.send_text(json.dumps({
                            "type": "tts_ready",
                            "text": tts_text,
                            "emotion": current_emotion,
                            "instruct": current_instruct,
                            "stream_url": stream_url,
                        }))

                    await websocket.send_text(json.dumps({
                        "type": "llm_token",
                        "content": cleaned_content,
                        "emotion": current_emotion,
                    }))

                else:
                    # No emotion yet — just send LLM token
                    await websocket.send_text(json.dumps({
                        "type": "llm_token",
                        "content": cleaned_content,
                        "emotion": None,
                    }))

                    if not first_token_sent and event.ttft_seconds is not None:
                        ttft_seconds = event.ttft_seconds
                        first_token_sent = True

            elif event.event.value == "content_done":
                e2e_latency = time.perf_counter() - e2e_start
                metrics.e2e_latency.labels(component="pipeline").observe(e2e_latency)

                await websocket.send_text(json.dumps({
                    "type": "llm_done",
                    "text": event.content,
                    "total_tokens": event.total_tokens,
                    "telemetry": {
                        "e2e_latency_seconds": e2e_latency,
                        "ttft_seconds": ttft_seconds,
                    },
                }))

            elif event.event.value == "cancelled":
                metrics.llm_tokens_total.labels(
                    component="llm",
                    model=getattr(client, "model", "mock"),
                    session_id=session_id,
                ).inc(len(accumulated_text))

                await websocket.send_text(json.dumps({
                    "type": "llm_cancelled",
                    "partial_text": event.content,
                }))

            elif event.event.value == "error":
                await websocket.send_text(json.dumps({
                    "type": "llm_error",
                    "error": event.error,
                }))

    finally:
        state_manager.clear_llm_task(session_id)


@router.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ASR + LLM streaming.

    Pipeline per utterance:
      1. Receive WebM audio chunks → decode to PCM
      2. VAD detection
      3. commit_utterance → ASR → LLM streaming + emotion parsing
      4. First emotion → send tts_ready with HTTP stream URL
      5. New speech (VAD) → cancel LLM (barge-in)
    """
    # Initialize audio decoder on first connection
    _get_audio_decoder()

    session_id = str(uuid.uuid4())
    await websocket.accept()

    metrics.ws_connections_total.labels(component="ws", status="connected").inc()
    metrics.active_sessions.labels(component="pipeline").inc()

    log.info(f"[{session_id}] WebSocket connected")

    try:
        while True:
            message = await websocket.receive()

            # Handle Text (JSON) messages
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    if msg_type == "config":
                        state_manager.create_session(session_id)
                        success = state_manager.update_config(session_id, payload)
                        log.info(
                            f"[{session_id}] Config: audio={payload.get('audio')}, "
                            f"persona={payload.get('persona_id')}, "
                            f"listener={payload.get('listener_id')}, "
                            f"model={payload.get('model')}"
                        )

                        if not success:
                            await websocket.close(code=1003, reason="Failed to apply config")
                            return

                    elif msg_type == "control" and payload.get("action") == "commit_utterance":
                        state = state_manager.get_session(session_id)
                        if not state or not state.is_configured:
                            await websocket.close(code=1003, reason="Config must be sent first")
                            return

                        # VAD commit — finalize audio and run ASR
                        asr_result = await state_manager.commit_utterance(session_id)
                        await websocket.send_text(json.dumps(asr_result))

                        # Skip LLM if ASR returned empty
                        if not asr_result.get("text"):
                            continue

                        # Get session config
                        persona_id = state.persona_id
                        listener_id = state.listener_id
                        llm_model = state.llm_model

                        # M1 stub for RAG retrieval time
                        rag_start = time.perf_counter()
                        retrieved_context = ""
                        rag_elapsed = time.perf_counter() - rag_start
                        rag_retrieval_seconds.labels(
                            component="rag",
                            index_name="default",
                        ).observe(rag_elapsed)

                        # Start LLM streaming in background task (cancellable on barge-in)
                        asyncio.create_task(
                            run_llm_stream(
                                websocket=websocket,
                                session_id=session_id,
                                asr_text=asr_result["text"],
                                persona_id=persona_id,
                                listener_id=listener_id,
                                llm_model=llm_model,
                            )
                        )

                    elif msg_type == "control" and payload.get("action") == "cancel":
                        # Explicit cancel from client
                        state_manager.cancel_llm_task(session_id)
                        log.info(f"[{session_id}] Explicit cancel")

                except json.JSONDecodeError:
                    log.warning(f"[{session_id}] Invalid JSON")

            # Handle Binary (WebM audio) messages
            elif "bytes" in message:
                state = state_manager.get_session(session_id)
                if not state or not state.is_configured:
                    await websocket.close(code=1003, reason="Config must be sent before audio data")
                    return

                webm_bytes = message["bytes"]

                # Decode WebM → PCM
                pcm_bytes = decode_webm_to_pcm(
                    webm_bytes,
                    target_sample_rate=state.audio_config.sample_rate
                )

                metrics.audio_chunks_total.labels(
                    component="ws",
                    session_id=session_id,
                ).inc()

                # Process through VAD
                vad_result = state_manager.process_audio(session_id, pcm_bytes)

                if vad_result:
                    # VAD committed — send signal to client
                    await websocket.send_text(json.dumps(vad_result))

                    # Cancel any active LLM (new speech started)
                    llm_cancelled = state_manager.cancel_llm_task(session_id)
                    if llm_cancelled:
                        await websocket.send_text(json.dumps({"type": "llm_cancelled"}))
                        log.info(f"[{session_id}] LLM cancelled due to new speech")

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        log.exception(f"[{session_id}] Error: {e}")
        metrics.errors_total.labels(
            component="ws",
            error_type="exception",
            model="",
        ).inc()
    finally:
        metrics.ws_connections_total.labels(component="ws", status="disconnected").inc()
        metrics.active_sessions.labels(component="pipeline").dec()
        state_manager.remove_session(session_id)
