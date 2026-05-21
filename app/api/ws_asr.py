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

from app.api._ws_helpers import (
    await_prior_tts_task,
    drain_emotion_parser,
    safe_send_bytes,
    safe_send_text,
    send_tts_error_frame,
)
from app.core.state_manager import StateManager
from app.services.llm import OpenAIClient, MockLLMClient, PersonaManager, PersonaType
from app.services.tts import EmotionMapper, enhance_text, get_tts_engine
from app.logging_config import get_logger
from telemetry import metrics, rag_retrieval_seconds


log = get_logger(__name__, component="ws")

# Strong refs to fire-and-forget LLM tasks. Without this set, Python's GC
# can drop the task between create_task() and the task body's first
# `state_manager.set_llm_task()` call — the symptom is a mysterious
# CancelledError ~3s into the stream with no logs. Tasks remove themselves
# from the set on completion.
_llm_task_refs: set[asyncio.Task] = set()


async def _stream_tts_sentence(
    websocket: WebSocket,
    text: str,
    emotion: str,
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
        persona_id: Persona ID for voice clone (optional)
        sentence_idx: Index of this sentence in the utterance (for tracking)

    Note: the singleton TTS engine is initialized once at startup and may
    be re-loaded with a merged SFT model when a version is activated; the
    legacy 0.6B/1.7B picker was removed 2026-05-20.
    """
    if not text or not text.strip():
        return

    engine = get_tts_engine()
    # Read engine's actual loaded model size for the metric label.
    model_size = getattr(engine, "model_size", "1.7B")

    # Phase 1 Path B: Use text enhancement instead of instruct strings
    enhanced_text_content = enhance_text(text.strip(), emotion)

    log.info(f"[{session_id}] TTS _stream_tts_sentence: model_size={model_size}, emotion={emotion}, text='{text[:30]}...' enhanced='{enhanced_text_content[:30]}...' engine={type(engine).__name__ if engine else None}")

    # Resolve which TTS model the chat should use for this persona.
    #
    # Two distinct modes coexist:
    #
    #   (a) custom_voice mode — engine.activate_version(version_id) loads
    #       a merged Qwen3-TTS model with the persona's speaker embedding
    #       baked into talker.codec_embedding[spk_id]. This is the path
    #       the training-page Preview uses, and what the user expects
    #       when they "switch model" in the chat dropdown.
    #
    #   (b) voice_clone mode — engine.activate_voice_clone(persona_id,
    #       ref_audio) uses the BASE model + a reference wav at
    #       inference time. No training required, but quality is lower
    #       and the voice is decided by whatever ref_audio is found.
    #
    # The old code always took path (b) for chat, which silently
    # OVERWROTE whatever the user had activated via the training page
    # (e.g. v9 with the trained child's voice baked in). Result: preview
    # sounded like the trained voice, chat sounded like base + ref
    # audio — observed as "v9 sounds female in chat but male in preview"
    # on 2026-05-21.
    #
    # Now: if the persona has an active READY version with a merged_path
    # on disk, use path (a). Otherwise fall back to (b).
    reference_audio = None
    if persona_id:
        try:
            from app.services.training_service.repository import JsonTrainingRepository
            from app import config as _cfg
            repo = JsonTrainingRepository(_cfg.models_dir())
            active = repo.get_active(persona_id)
            chosen_version = None
            if active is not None:
                v = repo.get_or_none(active.version_id)
                if v and v.status.value == "ready" and v.merged_path:
                    chosen_version = v
            if chosen_version is not None:
                engine.activate_version(chosen_version.version_id)
                log.info(
                    f"[{session_id}] Activated custom_voice model for persona "
                    f"{persona_id}: {chosen_version.version_id} "
                    f"(merged_path={chosen_version.merged_path})"
                )
            else:
                # No active merged model — fall back to voice_clone with a
                # reference wav. Keeps personas that have NEVER been trained
                # working (uses the base model timbre as a starting point).
                from app.api.tts_stream import _get_persona_reference_audio
                from app.services.tts.qwen_tts_engine import FasterQwenTTSEngine
                reference_audio = (
                    _get_persona_reference_audio(persona_id)
                    or FasterQwenTTSEngine.find_reference_audio(persona_id)
                )
                if reference_audio:
                    engine.activate_voice_clone(persona_id, reference_audio)
                    log.info(
                        f"[{session_id}] No active version for {persona_id}; "
                        f"using voice_clone mode with reference: {reference_audio}"
                    )
                else:
                    log.info(
                        f"[{session_id}] No active version and no reference "
                        f"audio for persona {persona_id} — base model voice."
                    )
        except Exception as e:
            log.warning(f"[{session_id}] TTS model resolution failed for {persona_id}: {e}", exc_info=True)

    stream_start = time.perf_counter()

    try:
        # Signal TTS start so client can prepare AudioWorklet
        await safe_send_text(websocket, {
            "type": "tts_start",
            "sentence_idx": sentence_idx,
            "emotion": emotion,
        })

        first_chunk_sent = False
        async for event in engine.generate_streaming(
            text=enhanced_text_content,
            instruct=None,  # Path B: Use text enhancement, not instruct
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

                # Stream chunk immediately — client plays as it arrives.
                # safe_send_bytes returns False on disconnect; we stop early.
                if not await safe_send_bytes(websocket, chunk):
                    return

        # Signal sentence complete
        await safe_send_text(websocket, {
            "type": "tts_done",
            "sentence_idx": sentence_idx,
        })
        log.info(f"TTS sentence done: idx={sentence_idx}, text='{text[:30]}...'")

        # Add brief silence gap between sentences (500ms = 12000 samples at 24kHz)
        # This gives natural pause between sentences so they don't run into each other.
        silence_samples = int(24000 * 0.5)  # 500ms silence
        silence_bytes = b'\x00\x00' * silence_samples  # Int16 silence
        await safe_send_bytes(websocket, silence_bytes)

    except Exception as e:
        log.error(f"TTS streaming error for sentence {sentence_idx}: {e}")
        # safe_send_text swallows only client-disconnect errors. If the
        # client is gone we can't tell them about the failure — that's fine.
        await send_tts_error_frame(websocket, sentence_idx=sentence_idx, error=str(e))


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


# Sentence boundary pattern: Chinese (。！？；) + English (.!?) + closing quotes (」』)
# Split on these to get complete sentences for early TTS streaming.
SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？；.!?」』])')

# Legacy emotion-tag regexes — used at llm_done to strip any remaining
# inline tags before sending the final cleaned text to the client. Compiled
# once at module level so we don't re-compile on every message.
_LEGACY_EMOTION_TAG_RE = re.compile(r'[\[［](?:情感|感情)[:：][^\]＞]*[\]］]\s*')
_NEW_EMOTION_TAG_RE = re.compile(r'\[E:[^\]]+\]')
_QUOTE_STRIP_RE = re.compile(r'[「」『』【】〖〗《》〈〉〘〙〚〛‹›«»]')


async def run_llm_stream(
    websocket: WebSocket,
    session_id: str,
    asr_text: str,
    persona_id: Optional[str],
    listener_id: Optional[str],
    llm_model: Optional[str],
    utterance_seq: int = 0,
) -> None:
    """
    Run LLM streaming and parse emotion tags.

    When first emotion is detected, send tts_ready with stream URL.
    This task is registered in StateManager so it can be cancelled on barge-in.
    """
    client = get_llm_client()
    cancellation_event = asyncio.Event()

    # Register task in state manager for cancellation. Pass the seq so
    # the latched-cancel check honors only cancels stamped for this
    # utterance (review #21).
    task = asyncio.current_task()
    if task:
        state_manager.set_llm_task(
            session_id, task, cancellation_event,
            utterance_seq=utterance_seq,
        )

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
    tts_sentence_idx = 0  # Index for tracking sentences streamed to client
    current_tts_task: Optional[asyncio.Task] = None  # Single TTS task (sequential, not parallel)

    try:
        async for event in client.stream(
            prompt=asr_text,
            system_prompt=system_prompt,
            cancellation_event=cancellation_event,
        ):
            if event.event.value == "start":
                await safe_send_text(websocket, {
                    "type": "llm_start",
                    "utterance_id": session_id,
                })

            elif event.event.value == "content_delta":
                accumulated_text += event.content

                # Parse emotion from this delta
                new_emotion, cleaned_content = emotion_mapper.update(event.content)

                if new_emotion and not tts_notified:
                    # First emotion detected
                    tts_notified = True
                    current_emotion = new_emotion
                    # Path B: No instruct needed, using text enhancement
                    tts_text_parts = []  # Start fresh after tag

                    if cleaned_content:
                        # Emotion and first content char arrived together — start TTS immediately
                        tts_text_parts.append(cleaned_content)
                        tts_text = "".join(tts_text_parts)
                        # Start TTS for this first sentence
                        await await_prior_tts_task(current_tts_task, "first-emotion")
                        current_tts_task = asyncio.create_task(_stream_tts_sentence(
                            websocket=websocket,
                            text=tts_text,
                            emotion=current_emotion or "默認",
                            persona_id=persona_id,
                            sentence_idx=tts_sentence_idx,
                            session_id=session_id,
                        ))
                        tts_sentence_idx += 1

                    # Send llm_token only if there's content
                    if cleaned_content:
                        await safe_send_text(websocket, {
                            "type": "llm_token",
                            "content": cleaned_content,
                            "emotion": current_emotion,
                        })
                    # Record TTFT
                    if not first_token_sent and event.ttft_seconds is not None:
                        ttft_seconds = event.ttft_seconds
                        first_token_sent = True

                    # Drain remaining buffered characters from EmotionParser.
                    # Bounded via drain_emotion_parser — guarantees termination.
                    async for _drained_emotion, more_content in drain_emotion_parser(emotion_mapper):
                        tts_text_parts.append(more_content)
                        tts_text = "".join(tts_text_parts)
                        parts = SENTENCE_SPLIT_RE.split(tts_text)
                        if len(parts) > 1:
                            for part in parts[:-1]:
                                if part.strip():
                                    await await_prior_tts_task(current_tts_task, "drain-sentence-boundary")
                                    current_tts_task = asyncio.create_task(_stream_tts_sentence(
                                        websocket=websocket,
                                        text=part,
                                        emotion=current_emotion or "默認",
                                        persona_id=persona_id,
                                        sentence_idx=tts_sentence_idx,
                                        session_id=session_id,
                                    ))
                                    tts_sentence_idx += 1
                            tts_text_parts = [parts[-1]]
                        await safe_send_text(websocket, {
                            "type": "llm_token",
                            "content": more_content,
                            "emotion": current_emotion,
                        })

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
                                    await await_prior_tts_task(current_tts_task, "after-emotion-sentence-boundary")
                                    current_tts_task = asyncio.create_task(_stream_tts_sentence(
                                        websocket=websocket,
                                        text=part,
                                        emotion=current_emotion or "默認",
                                        persona_id=persona_id,
                                        sentence_idx=tts_sentence_idx,
                                        session_id=session_id,
                                    ))
                                    tts_sentence_idx += 1
                            # Keep only the last (incomplete) part for next round
                            tts_text_parts = [parts[-1]]

                    if cleaned_content:
                        await safe_send_text(websocket, {
                            "type": "llm_token",
                            "content": cleaned_content,
                            "emotion": current_emotion,
                        })

                    # Drain remaining buffered characters from EmotionParser.
                    async for _drained_emotion, more_content in drain_emotion_parser(emotion_mapper):
                        tts_text_parts.append(more_content)
                        tts_text = "".join(tts_text_parts)
                        parts = SENTENCE_SPLIT_RE.split(tts_text)
                        if len(parts) > 1:
                            for part in parts[:-1]:
                                if part.strip():
                                    await await_prior_tts_task(current_tts_task, "post-emotion-drain")
                                    current_tts_task = asyncio.create_task(_stream_tts_sentence(
                                        websocket=websocket,
                                        text=part,
                                        emotion=current_emotion or "默認",
                                        persona_id=persona_id,
                                        sentence_idx=tts_sentence_idx,
                                        session_id=session_id,
                                    ))
                                    tts_sentence_idx += 1
                            tts_text_parts = [parts[-1]]
                        await safe_send_text(websocket, {
                            "type": "llm_token",
                            "content": more_content,
                            "emotion": current_emotion,
                        })

                else:
                    # No emotion yet — just send LLM token (only if content is non-empty)
                    if cleaned_content:
                        await safe_send_text(websocket, {
                            "type": "llm_token",
                            "content": cleaned_content,
                            "emotion": None,
                        })

                    # Drain remaining buffered characters from EmotionParser.
                    async for _drained_emotion, more_content in drain_emotion_parser(emotion_mapper):
                        await safe_send_text(websocket, {
                            "type": "llm_token",
                            "content": more_content,
                            "emotion": None,
                        })

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
                            await await_prior_tts_task(current_tts_task, "incomplete-sentence")
                            current_tts_task = asyncio.create_task(_stream_tts_sentence(
                                websocket=websocket,
                                text=remaining,
                                emotion=current_emotion or "默認",
                                persona_id=persona_id,
                                sentence_idx=tts_sentence_idx,
                                session_id=session_id,
                            ))
                            tts_sentence_idx += 1

                # Wait for final TTS sentence to finish before sending tts_done
                if current_tts_task is not None:
                    await asyncio.gather(current_tts_task, return_exceptions=True)
                    current_tts_task = None

                # Send tts_ready with full text for backward compatibility
                # (client that doesn't support ws binary chunks will use HTTP stream)
                full_text = event.content or ""
                # Try to parse as JSON first (new format)
                clean_text = full_text.strip()
                try:
                    parsed = json.loads(full_text)
                    clean_text = parsed.get("content", parsed.get("text", ""))
                except (json.JSONDecodeError, TypeError):
                    # Strip both legacy [情感:xxx] / [感情:xxx] and new [E:xxx]
                    # tag formats. Regexes are compiled once at module level.
                    clean_text = _LEGACY_EMOTION_TAG_RE.sub("", clean_text).strip()
                    clean_text = _NEW_EMOTION_TAG_RE.sub("", clean_text).strip()
                if clean_text:
                    clean_text = _QUOTE_STRIP_RE.sub("", clean_text).strip()

                # NOTE — the legacy `tts_ready` frame (with stream_url) was
                # removed in Phase 2. CLAUDE.md confirms the client ignores
                # it; HTTP fetch path is gone, audio comes through WS binary.

                await safe_send_text(websocket, {
                    "type": "llm_done",
                    "text": clean_text or full_text,
                    "total_tokens": event.total_tokens,
                    "telemetry": {
                        "e2e_latency_seconds": e2e_latency,
                        "ttft_seconds": ttft_seconds,
                    },
                })

            elif event.event.value == "cancelled":
                metrics.llm_tokens_total.labels(
                    component="llm",
                    model=getattr(client, "model", "mock"),
                    session_id=session_id,
                ).inc(len(accumulated_text))

                # Cancel in-flight TTS task
                if current_tts_task:
                    current_tts_task.cancel()
                    current_tts_task = None

                await safe_send_text(websocket, {
                    "type": "llm_cancelled",
                    "partial_text": event.content,
                })

            elif event.event.value == "error":
                # Cancel TTS task on error too
                if current_tts_task:
                    current_tts_task.cancel()
                    current_tts_task = None

                await safe_send_text(websocket, {
                    "type": "llm_error",
                    "error": event.error,
                })

    finally:
        # Cancel any remaining TTS task
        if current_tts_task:
            current_tts_task.cancel()
            current_tts_task = None
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
                        # Legacy tts_model preference (0.6B/1.7B picker) was
                        # removed 2026-05-20 — server always uses the active
                        # SFT/LoRA merged model now. We accept the field for
                        # backward compatibility but log+ignore it.
                        if "tts_model" in payload:
                            log.info(
                                f"[{session_id}] ignoring legacy "
                                f"tts_model={payload.get('tts_model')!r} "
                                f"(server uses active version)"
                            )

                        # Activate the TTS model for this persona NOW, not
                        # on the first TTS-stream call. Without this, the
                        # status bar (which polls /api/system/status every
                        # 5s) keeps showing the previous persona's active
                        # version until the user speaks at least once and
                        # _stream_tts_sentence eagerly activates. User
                        # observation 2026-05-21: "switching persona in
                        # chat doesn't flash the status bar".
                        cfg_persona = payload.get("persona_id")
                        if cfg_persona:
                            try:
                                from app.services.tts import get_tts_engine
                                from app.services.training_service.repository import JsonTrainingRepository
                                from app import config as _cfg
                                _engine = get_tts_engine()
                                _repo = JsonTrainingRepository(_cfg.models_dir())
                                _active = _repo.get_active(cfg_persona)
                                if _active is not None:
                                    _v = _repo.get_or_none(_active.version_id)
                                    if _v and _v.status.value == "ready" and _v.merged_path:
                                        _engine.activate_version(_v.version_id)
                                        log.info(
                                            f"[{session_id}] Pre-activated v{_v.version_id} "
                                            f"for persona {cfg_persona} on config receipt"
                                        )
                            except Exception as _e:
                                log.warning(
                                    f"[{session_id}] Eager activation failed for {cfg_persona}: {_e}"
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

                        await safe_send_text(websocket, asr_result)
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

                        # Bump utterance seq BEFORE create_task so a cancel
                        # arriving immediately afterward stamps the right
                        # seq (review #21 stale-cancel-race fix).
                        utterance_seq = state_manager.begin_utterance(session_id)

                        # Start LLM streaming in background task (cancellable on barge-in).
                        # Hold a strong ref so the task isn't dropped before
                        # run_llm_stream registers it with state_manager.
                        _llm_bg_task = asyncio.create_task(
                            run_llm_stream(
                                websocket=websocket,
                                session_id=session_id,
                                asr_text=asr_result["text"],
                                persona_id=persona_id,
                                listener_id=listener_id,
                                llm_model=llm_model,
                                utterance_seq=utterance_seq,
                            )
                        )
                        _llm_task_refs.add(_llm_bg_task)
                        _llm_bg_task.add_done_callback(_llm_task_refs.discard)

                    elif msg_type == "control" and payload.get("action") == "cancel":
                        # Explicit cancel from client
                        state_manager.cancel_llm_task(session_id, origin="ws_explicit_cancel")
                        log.info(f"[{session_id}] Explicit cancel")

                    elif msg_type == "control" and payload.get("action") == "start_speech":
                        # Client clicked the mic — reset audio buffer + VAD for
                        # the new utterance. Do NOT pre-emptively cancel LLM/TTS
                        # here. The VAD barge-in path (state_manager.process_audio)
                        # cancels when actual speech is detected, so a too-eager
                        # re-tap of the mic doesn't kill the in-flight response
                        # before the user has actually started speaking.
                        state = state_manager.get_session(session_id)
                        if state:
                            state.audio_buffer.clear()
                            if hasattr(state.vad, 'reset'):
                                state.vad.reset()
                        log.info(f"[{session_id}] start_speech: cleared buffer (waiting for VAD to detect speech for barge-in)")

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
                    await safe_send_text(websocket, vad_result)

    except WebSocketDisconnect:
        # Cancel the in-flight LLM with a distinct origin so the diagnostic
        # log doesn't blame the disconnect on `remove_session` (review #22).
        log.info(f"[{session_id}] Client disconnected")
        state_manager.cancel_llm_task(session_id, origin="ws_disconnect")
    except RuntimeError as e:
        # Starlette raises RuntimeError('Cannot call "receive" once a disconnect
        # message has been received.') instead of WebSocketDisconnect in some
        # paths — treat it as a normal disconnect, not an error to log.
        if "disconnect" in str(e).lower():
            log.info(f"[{session_id}] Client disconnected (post-disconnect receive)")
            state_manager.cancel_llm_task(session_id, origin="ws_disconnect")
        else:
            raise
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
