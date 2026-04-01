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
    Binary PCM chunks (16-bit, 24kHz, mono) sent as WebSocket binary frames
    for each complete sentence as TTS generates them — plays immediately on client
    {"type": "tts_start", "sentence_idx": 0}  <- sent before first binary chunk
    {"type": "tts_ready", "text": "「好啦～", "emotion": "寵溺", "instruct": "(gentle...)", "stream_url": "/api/tts/stream?..."}
    (tts_ready still sent at llm_done for backward compat + final chunk)
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
import struct
import subprocess
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.state_manager import StateManager
from app.services.llm import OpenAIClient, MockLLMClient, PersonaManager, PersonaType
from app.services.tts import EmotionMapper, get_tts_instruct, parse_emotion_tag, get_tts_engine
from app.logging_config import get_logger
from telemetry import metrics, rag_retrieval_seconds


log = get_logger(__name__, component="ws")


async def _stream_tts_sentence(
    websocket: WebSocket,
    text: str,
    emotion: str,
    model_size: str,
    persona_id: Optional[str],
    sentence_idx: int,
    session_id: str = "",
) -> None:
    """
    Generate TTS for a complete sentence and stream raw PCM chunks over WebSocket.

    Chunks are sent as binary WebSocket frames as they arrive from the TTS engine,
    enabling immediate playback on the client (no waiting for full response).

    Args:
        websocket: Client WebSocket connection
        text: Sentence text to synthesize
        emotion: Emotion tag string
        model_size: "0.6B" or "1.7B"
        persona_id: Persona ID for voice clone (optional)
        sentence_idx: Index of this sentence in the utterance (for tracking)
    """
    if not text or not text.strip():
        return

    engine = get_tts_engine(model_size=model_size)
    instruct = get_tts_instruct(emotion)

    # Look up reference audio for voice clone (optional — pass None if unavailable)
    reference_audio = None
    if persona_id:
        try:
            # Try to get persona reference audio for voice clone
            # This is optional — streaming works without it
            from app.services.recordings import pipeline as rec_pipeline
            # Persona reference audio path would go here if implemented
        except Exception:
            pass

    stream_start = time.perf_counter()

    try:
        # Signal TTS start so client can prepare AudioWorklet
        await websocket.send_text(json.dumps({
            "type": "tts_start",
            "sentence_idx": sentence_idx,
            "emotion": emotion,
        }))

        first_chunk_sent = False
        async for event in engine.generate_streaming(
            text=text.strip(),
            instruct=instruct,
            language="Chinese",
            reference_audio=reference_audio,
        ):
            if event.event == "audio_chunk" and event.audio_data:
                chunk = event.audio_data

                # Record time to first chunk
                if not first_chunk_sent:
                    ttfc = time.perf_counter() - stream_start
                    metrics.tts_first_chunk.labels(
                        component="ws_stream",
                        model=model_size,
                    ).observe(ttfc)
                    log.info(f"TTS first chunk (ws): {ttfc:.3f}s, sentence={sentence_idx}")
                    first_chunk_sent = True

                # Stream chunk immediately — client plays as it arrives
                await websocket.send_bytes(chunk)

        # Signal sentence complete
        await websocket.send_text(json.dumps({
            "type": "tts_done",
            "sentence_idx": sentence_idx,
        }))
        log.info(f"TTS sentence done: idx={sentence_idx}, text='{text[:30]}...'")

    except Exception as e:
        log.error(f"TTS streaming error for sentence {sentence_idx}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "tts_error",
                "sentence_idx": sentence_idx,
                "error": str(e),
            }))
        except Exception:
            pass


router = APIRouter()

use_qwen = os.getenv("USE_QWEN_ASR", "true").lower() == "true"
use_mock_llm = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
state_manager = StateManager(use_qwen=use_qwen)

# OpenAI config
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

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
    Decode WebM/Opus or Ogg/Opus audio to PCM 16-bit mono using ffmpeg subprocess.

    Detects format from magic bytes:
      1a45dfa3 = WebM/Matroska
      4f676753 = Ogg
      52494646 = RIFF (WAV)

    Args:
        webm_bytes: Raw audio bytes from MediaRecorder
        target_sample_rate: Target sample rate (default 24kHz)

    Returns:
        PCM 16-bit mono bytes, or None if decode fails
    """
    if len(webm_bytes) < 100:
        log.debug(f"Audio chunk too small ({len(webm_bytes)} bytes), skipping")
        return None

    magic = webm_bytes[:4].hex()

    # Detect format from magic bytes
    if magic.startswith('1a45dfa3'):
        input_format = 'webm'
    elif magic.startswith('4f676753'):
        input_format = 'ogg'
    elif magic.startswith('52494646'):
        input_format = 'wav'
    else:
        log.debug(f"Unknown audio format magic: {magic}, skipping chunk")
        return None

    try:
        proc = subprocess.Popen(
            [
                '/usr/bin/ffmpeg', '-y',
                '-hide_banner', '-loglevel', 'quiet',
                '-f', input_format, '-i', 'pipe:0',
                '-f', 's16le', '-acodec', 'pcm_s16le',
                '-ac', '1', '-ar', str(target_sample_rate),
                'pipe:1'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pcm_out, ffmpeg_err = proc.communicate(input=webm_bytes, timeout=5)

        if proc.returncode != 0:
            log.debug(f"ffmpeg {input_format} decode failed (code {proc.returncode}) "
                      f"for {len(webm_bytes)}B chunk, magic={magic}")
            return None

        log.debug(f"Decoded {input_format} -> PCM: {len(webm_bytes)}B in, {len(pcm_out)}B out, "
                  f"{len(pcm_out)//2} samples, magic={magic}")
        return pcm_out

    except subprocess.TimeoutExpired:
        log.warning("ffmpeg decode timed out")
        return None
    except Exception as e:
        log.warning(f"Audio decode failed: {e}")
        return None


def get_llm_client() -> OpenAIClient | MockLLMClient:
    """Lazily initialize and return the LLM client."""
    global llm_client
    if llm_client is None:
        if use_mock_llm:
            llm_client = MockLLMClient()
            log.info("Using MockLLMClient")
        else:
            llm_client = OpenAIClient(
                model=OPENAI_MODEL,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=OPENAI_BASE_URL,
            )
            log.info(f"Using OpenAIClient: model={OPENAI_MODEL}, base={OPENAI_BASE_URL}")
    return llm_client


# Emotion tag regex — matches [情感: 撒嬌] or [情感:撒嬌]
EMOTION_TAG_RE = re.compile(r'^\[情感[:：]\s*(.*?)\]\s*')

# Sentence boundary pattern: Chinese (。！？；) + English (.!?) + closing quotes (」』)
# Split on these to get complete sentences for early TTS streaming
SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？；.!?」』])')


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
    tts_url_sent = False  # Only send tts_ready URL once per utterance

    # Emotion mapper for this turn
    emotion_mapper = EmotionMapper()

    # TTS state: accumulates text AFTER emotion tag is removed
    tts_text_parts: list[str] = []
    current_emotion: Optional[str] = None
    current_instruct: Optional[str] = None
    tts_sentence_idx = 0  # Index for tracking sentences streamed to client
    tts_streaming_tasks: list[asyncio.Task] = []  # Track background TTS tasks

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
                    # First emotion detected — mark state but DON'T send tts_ready yet
                    # Wait until we have actual text content before notifying client
                    tts_notified = True
                    current_emotion = new_emotion
                    current_instruct = get_tts_instruct(new_emotion)
                    # At this point, cleaned_content is empty (tag consumed buffer)
                    # Next tokens will have actual text
                    tts_text_parts = []  # Start fresh after tag

                    # Send llm_token (emotion detected) but don't notify TTS yet
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
                    # Emotion detected — accumulate text for TTS
                    if cleaned_content:
                        tts_text_parts.append(cleaned_content)
                        tts_text = "".join(tts_text_parts)

                        # Check for complete sentences and stream TTS as they finish
                        # This enables low-latency playback: audio starts before LLM is done
                        parts = SENTENCE_SPLIT_RE.split(tts_text)
                        if len(parts) > 1:
                            # At least one complete sentence found
                            # All parts except the last are complete sentences
                            for i, part in enumerate(parts[:-1]):
                                if part.strip():
                                    # Trigger TTS streaming for this complete sentence
                                    task = asyncio.create_task(_stream_tts_sentence(
                                        websocket=websocket,
                                        text=part,
                                        emotion=current_emotion or "默認",
                                        model_size="0.6B",  # Faster model for lower latency
                                        persona_id=persona_id,
                                        sentence_idx=tts_sentence_idx,
                                        session_id=session_id,
                                    ))
                                    tts_streaming_tasks.append(task)
                                    tts_sentence_idx += 1
                            # Keep only the last (incomplete) part for next round
                            tts_text_parts = [parts[-1]]

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

                # Stream any remaining incomplete sentence (no trailing sentence-ending punctuation)
                if tts_notified and tts_text_parts:
                    remaining = "".join(tts_text_parts).strip()
                    if remaining:
                        # Clean text for TTS
                        import re
                        remaining = re.sub(r'[「」『』【】〖〗《》〈〉〘〙〚〛‹›«»]', '', remaining).strip()
                        if remaining:
                            task = asyncio.create_task(_stream_tts_sentence(
                                websocket=websocket,
                                text=remaining,
                                emotion=current_emotion or "默認",
                                model_size="0.6B",
                                persona_id=persona_id,
                                sentence_idx=tts_sentence_idx,
                                session_id=session_id,
                            ))
                            tts_streaming_tasks.append(task)
                            tts_sentence_idx += 1

                # Wait for all sentence TTS streams to finish before sending tts_done
                if tts_streaming_tasks:
                    await asyncio.gather(*tts_streaming_tasks, return_exceptions=True)
                    tts_streaming_tasks.clear()

                # Send tts_ready with full text for backward compatibility
                # (client that doesn't support ws binary chunks will use HTTP stream)
                if tts_notified:
                    full_text = event.content or ""
                    _, clean_text = parse_emotion_tag(full_text)
                    clean_text = clean_text.strip()
                    if clean_text:
                        import re
                        clean_text = re.sub(r'[「」『』【】〖〗《》〈〉〘〙〚〛‹›«»]', '', clean_text).strip()
                    if clean_text:
                        import urllib.parse
                        params = urllib.parse.urlencode({
                            "text": clean_text,
                            "emotion": current_emotion,
                            "model": "0.6B",
                            "persona_id": persona_id or "xiao_s",
                        })
                        stream_url = f"/api/tts/stream?{params}"
                        await websocket.send_text(json.dumps({
                            "type": "tts_ready",
                            "text": clean_text,
                            "emotion": current_emotion,
                            "instruct": current_instruct,
                            "stream_url": stream_url,
                        }))

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

                # Cancel any in-flight TTS streaming tasks
                for task in tts_streaming_tasks:
                    task.cancel()
                tts_streaming_tasks.clear()

                await websocket.send_text(json.dumps({
                    "type": "llm_cancelled",
                    "partial_text": event.content,
                }))

            elif event.event.value == "error":
                # Cancel TTS tasks on error too
                for task in tts_streaming_tasks:
                    task.cancel()
                tts_streaming_tasks.clear()

                await websocket.send_text(json.dumps({
                    "type": "llm_error",
                    "error": event.error,
                }))

    finally:
        # Cancel any remaining TTS streaming tasks
        for task in tts_streaming_tasks:
            task.cancel()
        tts_streaming_tasks.clear()
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
                        log.info(f"[{session_id}] commit_utterance received: configured={state and state.is_configured}, "
                                 f"buffer_size={len(state.audio_buffer) if state else 0}")
                        if not state or not state.is_configured:
                            await websocket.close(code=1003, reason="Config must be sent first")
                            return

                        # VAD commit — finalize audio and run ASR
                        asr_result = await state_manager.commit_utterance(session_id)

                        # Skip if ASR returned empty (don't send to client)
                        if not asr_result.get("text"):
                            log.info(f"[{session_id}] ASR empty, skipping")
                            continue

                        await websocket.send_text(json.dumps(asr_result))
                        log.info(f"[{session_id}] ASR done: {asr_result.get('text', '')[:50]}")

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

                    elif msg_type == "control" and payload.get("action") == "start_speech":
                        # Client is starting to speak — cancel any ongoing LLM (barge-in)
                        # and reset audio buffer
                        state_manager.cancel_llm_task(session_id)
                        state_manager.cancel_tts_task(session_id)
                        state = state_manager.get_session(session_id)
                        if state:
                            state.audio_buffer.clear()
                            if hasattr(state.vad, 'reset'):
                                state.vad.reset()
                        log.info(f"[{session_id}] start_speech: cancelled LLM, cleared buffer")

                except json.JSONDecodeError:
                    log.warning(f"[{session_id}] Invalid JSON")

            # Handle Binary audio messages (accumulated Int16 PCM from client)
            elif "bytes" in message:
                state = state_manager.get_session(session_id)
                if not state or not state.is_configured:
                    await websocket.close(code=1003, reason="Config must be sent before audio data")
                    return

                audio_bytes = message["bytes"]
                audio_len = len(audio_bytes)
                log.debug(f"Received binary audio: {audio_len} bytes")

                metrics.audio_chunks_total.labels(
                    component="ws",
                    session_id=session_id,
                ).inc()

                # Validate: Int16 PCM, 2 bytes per sample
                if audio_len % 2 != 0:
                    log.debug(f"Skipping odd-length audio chunk: {audio_len}")
                    continue

                num_samples = audio_len // 2
                log.debug(f"Int16 PCM: {num_samples} samples, {audio_len} bytes")

                # Run VAD on each chunk for continuous monitoring + barge-in
                # process_audio() returns vad_commit dict when silence detected after speech
                vad_result = state_manager.process_audio(session_id, audio_bytes)
                if vad_result:
                    # VAD detected end of speech — notify client to stop recording
                    log.info(f"[{session_id}] VAD commit: user stopped speaking, sending vad_commit to client")
                    await websocket.send_text(json.dumps(vad_result))

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        log.exception(f"[{session_id}] Error: {e}")
        metrics.errors_total.labels(
            component="ws",
            error_type="exception",
            model="",
        ).inc()
        # Don't re-raise — exit cleanly
        return
    finally:
        metrics.ws_connections_total.labels(component="ws", status="disconnected").inc()
        metrics.active_sessions.labels(component="pipeline").dec()
        state_manager.remove_session(session_id)
