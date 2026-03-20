"""
WebSocket ASR endpoint with LLM streaming and barge-in support.

Protocol:
- Client sends config: {"type": "config", "audio": {...}, "speaker_id": "optional"}
- Client sends Binary frames with PCM audio
- Client sends {"type": "control", "action": "commit_utterance"}
- Server returns {"type": "asr_result", ...} (partial/final ASR)
- Server returns {"type": "llm_start", ...} (LLM stream started, includes TTFT)
- Server returns {"type": "llm_token", "content": "...", ...} (streaming tokens)
- Server returns {"type": "llm_done", "text": "...", ...} (LLM stream complete)
- Server returns {"type": "llm_cancelled"} (stream cancelled by new speech)
"""
import asyncio
import json
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.state_manager import StateManager
from app.services.llm import OpenAIClient, MockLLMClient, PromptManager, PersonaType
from telemetry import metrics, rag_retrieval_seconds


router = APIRouter()

use_qwen = os.getenv("USE_QWEN_ASR", "true").lower() == "true"
use_mock_llm = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
state_manager = StateManager(use_qwen=use_qwen)

# LLM clients
llm_client: Optional[OpenAIClient | MockLLMClient] = None
prompt_manager = PromptManager()


def get_llm_client() -> OpenAIClient | MockLLMClient:
    """Lazily initialize and return the LLM client."""
    global llm_client
    if llm_client is None:
        if use_mock_llm:
            llm_client = MockLLMClient()
        else:
            llm_client = OpenAIClient(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
    return llm_client


async def run_llm_stream(
    websocket: WebSocket,
    session_id: str,
    asr_text: str,
    speaker_id: Optional[str],
) -> None:
    """
    Run LLM streaming after ASR completes and send tokens to client.

    This task is registered in StateManager so it can be cancelled on barge-in.
    """
    client = get_llm_client()
    cancellation_event = asyncio.Event()

    # Register task in state manager for cancellation
    task = asyncio.current_task()
    if task:
        state_manager.set_llm_task(session_id, task, cancellation_event)

    # Get speaker-aware system prompt
    system_prompt = prompt_manager.get_prompt(
        persona_type=PersonaType.CAREGIVER,
        speaker_id=speaker_id,
    )

    e2e_start = time.perf_counter()
    accumulated_text = ""
    first_token_sent = False
    ttft_seconds: Optional[float] = None

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
                payload = {
                    "type": "llm_token",
                    "content": event.content,
                }
                if not first_token_sent and event.ttft_seconds is not None:
                    ttft_seconds = event.ttft_seconds
                    payload["ttft_seconds"] = ttft_seconds
                    first_token_sent = True
                await websocket.send_text(json.dumps(payload))

            elif event.event.value == "content_done":
                e2e_latency = time.perf_counter() - e2e_start
                metrics.e2e_latency.labels(component="pipeline").observe(e2e_latency)

                await websocket.send_text(json.dumps({
                    "type": "llm_done",
                    "text": event.content,
                    "total_tokens": event.total_tokens,
                    "telemetry": {
                        "e2e_latency_seconds": e2e_latency,
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
      1. Receive audio chunks → VAD detection
      2. commit_utterance → ASR → LLM streaming
      3. New speech (VAD start) → cancel LLM (barge-in)
    """
    session_id = str(uuid.uuid4())
    await websocket.accept()

    metrics.ws_connections_total.labels(component="ws", status="connected").inc()
    metrics.active_sessions.labels(component="pipeline").inc()

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
                        print(f"[{session_id}] Config received: audio={payload.get('audio')}, "
                              f"speaker_id={payload.get('speaker_id')}")

                        if not success:
                            await websocket.close(code=1003, reason="Failed to apply config")
                            return

                    elif msg_type == "control" and payload.get("action") == "commit_utterance":
                        state = state_manager.get_session(session_id)
                        if not state or not state.is_configured:
                            await websocket.close(code=1003, reason="Config must be sent first")
                            return

                        metrics.utterances_total.labels(
                            component="pipeline",
                            session_id=session_id,
                        ).inc()

                        # Run ASR
                        asr_result = await state_manager.commit_utterance(session_id)
                        await websocket.send_text(json.dumps(asr_result))

                        # Skip LLM if ASR returned empty
                        if not asr_result.get("text"):
                            continue

                        # Milestone 2.1: RAG stub - record retrieval time (future: real retrieval)
                        rag_start = time.perf_counter()
                        # TODO: Replace with real RAG retrieval when services/rag/ is implemented
                        retrieved_context = ""  # placeholders
                        rag_elapsed = time.perf_counter() - rag_start
                        rag_retrieval_seconds.labels(
                            component="rag",
                            index_name="default",
                        ).observe(rag_elapsed)

                        # Build enriched prompt (future: inject retrieved_context)
                        enriched_prompt = asr_result["text"]

                        # Start LLM streaming in background task (cancellable on barge-in)
                        asyncio.create_task(
                            run_llm_stream(
                                websocket=websocket,
                                session_id=session_id,
                                asr_text=enriched_prompt,
                                speaker_id=state.speaker_id,
                            )
                        )

                except json.JSONDecodeError:
                    print(f"[{session_id}] Invalid JSON format")

            # Handle Binary (audio) messages
            elif "bytes" in message:
                state = state_manager.get_session(session_id)
                if not state or not state.is_configured:
                    await websocket.close(code=1003, reason="Config must be sent before audio data")
                    return

                audio_chunk = message["bytes"]

                metrics.audio_chunks_total.labels(
                    component="ws",
                    session_id=session_id,
                ).inc()

                partial_result = state_manager.process_audio(session_id, audio_chunk)

                if partial_result:
                    await websocket.send_text(json.dumps(partial_result))

                    # If VAD detected speech (barge-in), cancel any running LLM task
                    # This is now handled inside state_manager.process_audio()
                    # but we also send a signal to the client
                    if state.llm_task is not None:
                        was_cancelled = state_manager.cancel_llm_task(session_id)
                        if was_cancelled:
                            print(f"[{session_id}] LLM task cancelled due to new speech")

    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected")
    except Exception as e:
        print(f"[{session_id}] Error: {e}")
        metrics.errors_total.labels(
            component="ws",
            error_type="exception",
            model="",
        ).inc()
    finally:
        metrics.ws_connections_total.labels(component="ws", status="disconnected").inc()
        metrics.active_sessions.labels(component="pipeline").dec()
        state_manager.remove_session(session_id)
