import asyncio
import json
import websockets

async def test_asr():
    uri = "ws://localhost:8000/ws/asr"

    async with websockets.connect(uri) as websocket:
        # Send config message first
        config = {
            "type": "config",
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "format": "pcm"
            }
        }
        await websocket.send(json.dumps(config))
        print("Sent config:", config)

        # Send multiple binary audio chunks to trigger partial results
        # Server sends partial every 48000 bytes
        for i in range(10):
            audio_chunk = b'\x00' * 4800  # 300ms of silence at 16kHz
            await websocket.send(audio_chunk)
            print(f"Sent audio chunk {i+1}: {len(audio_chunk)} bytes")

        # Send commit_utterance to get final result
        control = {
            "type": "control",
            "action": "commit_utterance"
        }
        await websocket.send(json.dumps(control))
        print("Sent commit_utterance control")

        # Receive all responses
        try:
            for _ in range(5):
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print("Received:", response)
        except asyncio.TimeoutError:
            print("No more responses (timeout)")

if __name__ == "__main__":
    asyncio.run(test_asr())
