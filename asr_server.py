import json
import time
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

app = FastAPI(title="MVP v1.0 ASR Service")

# 這裡未來會替換成真正的 Qwen3-ASR 推論引擎與 VAD 模組
async def mock_asr_engine(audio_chunk: bytes) -> str:
    # 模擬推論延遲
    await asyncio.sleep(0.05)
    return "模擬語音辨識結果..."

@app.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    current_utterance_id = str(uuid.uuid4())
    start_time = time.time()
    
    config_received = False
    audio_buffer = bytearray()

    try:
        while True:
            # 接收 WebSocket 訊息 (自動區分 Text 與 Binary)
            message = await websocket.receive()
            
            # 處理 Text 控制幀
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")
                    
                    if msg_type == "config":
                        config_received = True
                        print(f"[{session_id}] Config received: {payload['audio']}")
                        # 在此處初始化特定採樣率的 VAD/ASR 狀態
                        
                    elif msg_type == "control" and payload.get("action") == "commit_utterance":
                        # 客戶端主動結束這句話
                        if audio_buffer:
                            # 觸發最終識別
                            final_text = await mock_asr_engine(bytes(audio_buffer))
                            await send_final_result(websocket, current_utterance_id, final_text, start_time)
                            # 重置狀態準備下一句話
                            audio_buffer.clear()
                            current_utterance_id = str(uuid.uuid4())
                            start_time = time.time()
                            
                except json.JSONDecodeError:
                    print(f"[{session_id}] Invalid JSON format")
            
            # 處理 Binary 語音幀
            elif "bytes" in message:
                if not config_received:
                    await websocket.close(code=1003, reason="Config must be sent before audio data")
                    break
                
                audio_chunk = message["bytes"]
                audio_buffer.extend(audio_chunk)
                
                # 這裡應該要接入 VAD 邏輯 (WebRTC VAD 或 Silero VAD)
                # 假設這裡觸發了 Partial 輸出
                if len(audio_buffer) % 48000 == 0: # 粗略模擬：每累積一定量發送一次 Partial
                    partial_text = await mock_asr_engine(audio_chunk)
                    response = {
                        "type": "asr_result",
                        "utterance_id": current_utterance_id,
                        "is_final": False,
                        "text": partial_text
                    }
                    await websocket.send_text(json.dumps(response))
                    
    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected.")
    except Exception as e:
        print(f"[{session_id}] Error: {e}")

async def send_final_result(websocket: WebSocket, utterance_id: str, text: str, start_time: float):
    process_time_ms = int((time.time() - start_time) * 1000)
    
    response = {
        "type": "asr_result",
        "utterance_id": utterance_id,
        "is_final": True,
        "text": text,
        "extensions": {
            "emotion": {
                "primary": "neutral", # 未來整合 SER 模組
                "confidence": 0.99
            }
        },
        "telemetry": {
            "vad_latency_ms": 0, # 待 VAD 整合後計算
            "asr_inference_ms": process_time_ms,
            "total_turnaround_ms": process_time_ms
        }
    }
    await websocket.send_text(json.dumps(response))

if __name__ == "__main__":
    import uvicorn
    # 本地測試執行: python asr_server.py
    uvicorn.run("asr_server:app", host="0.0.0.0", port=8000, reload=True)