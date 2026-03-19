"""WebSocket ASR endpoint."""
import json
import os
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.state_manager import StateManager

router = APIRouter()
# Use MockASR in test mode, Qwen3ASR otherwise
use_qwen = os.getenv("USE_QWEN_ASR", "true").lower() == "true"
state_manager = StateManager(use_qwen=use_qwen)


@router.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for ASR streaming.

    Protocol:
    - Client sends Text frame with config: {"type": "config", "audio": {...}}
    - Client sends Binary frames with PCM audio
    - Client sends Text frame to commit: {"type": "control", "action": "commit_utterance"}
    - Server returns Text frames with ASR results
    """
    session_id = str(uuid.uuid4())
    await websocket.accept()

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
                        print(f"[{session_id}] Config received: {payload.get('audio')}")

                        if not success:
                            await websocket.close(code=1003, reason="Failed to apply config")

                    elif msg_type == "control" and payload.get("action") == "commit_utterance":
                        # Commit current utterance and get final result
                        result = await state_manager.commit_utterance(session_id)
                        await websocket.send_text(json.dumps(result))

                except json.JSONDecodeError:
                    print(f"[{session_id}] Invalid JSON format")

            # Handle Binary (audio) messages
            elif "bytes" in message:
                state = state_manager.get_session(session_id)
                if not state or not state.is_configured:
                    await websocket.close(code=1003, reason="Config must be sent before audio data")
                    break

                audio_chunk = message["bytes"]
                partial_result = state_manager.process_audio(session_id, audio_chunk)

                if partial_result:
                    await websocket.send_text(json.dumps(partial_result))

    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected")
    except Exception as e:
        print(f"[{session_id}] Error: {e}")
    finally:
        state_manager.remove_session(session_id)
