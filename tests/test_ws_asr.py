"""Async WebSocket tests for ASR endpoint."""
import pytest
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

from app.core.state_manager import StateManager
from app.services.vad_engine import EnergyVAD
from app.services.asr_engine import MockASR


class TestStateManager:
    """Tests for StateManager."""

    def test_create_session(self):
        """Test session creation."""
        manager = StateManager()
        state = manager.create_session("test-session")

        assert state.session_id == "test-session"
        assert state.utterance_id is not None
        assert state.audio_config.sample_rate == 24000

    def test_get_session(self):
        """Test session retrieval."""
        manager = StateManager()
        manager.create_session("test-session")

        state = manager.get_session("test-session")
        assert state is not None

        missing = manager.get_session("nonexistent")
        assert missing is None

    def test_update_config(self):
        """Test config update."""
        manager = StateManager()
        manager.create_session("test-session")

        config = {
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "format": "pcm"
            }
        }

        success = manager.update_config("test-session", config)
        assert success is True

        state = manager.get_session("test-session")
        assert state.audio_config.sample_rate == 16000
        assert state.is_configured is True

    @pytest.mark.asyncio
    async def test_process_audio_empty_buffer(self):
        """Test audio processing with empty buffer returns empty on commit."""
        manager = StateManager()
        manager.create_session("test-session")
        manager.update_config("test-session", {"audio": {"sample_rate": 24000}})

        # Commit without audio
        result = await manager.commit_utterance("test-session")

        assert result["is_final"] is True
        assert result["text"] == ""
        assert "utterance_id" in result


class TestEnergyVAD:
    """Tests for EnergyVAD."""

    def test_silence_detection(self):
        """Test VAD with silence (zeros)."""
        vad = EnergyVAD(energy_threshold=0.01, adaptive=False)

        # Silent audio
        silence = b"\x00\x00" * 1000
        is_committing, energy = vad.detect(silence)

        # detect() returns (is_committing, energy) — is_committing is False for silence
        assert is_committing is False
        assert energy < 0.01

    def test_speech_detection(self):
        """Test VAD with simulated speech."""
        vad = EnergyVAD(energy_threshold=0.01, adaptive=False)

        # Generate audio with some energy (sine wave-like pattern)
        import struct
        samples = [int(16000 * (i % 10) / 10) for i in range(1000)]
        speech = struct.pack(f"{len(samples)}h", *samples)

        is_committing, energy = vad.detect(speech)

        # detect() returns (is_committing, energy). is_committing is False during
        # active speech (it only becomes True when silence is detected after enough
        # speech frames). The key check is that energy reflects the audio level.
        assert energy > 0
        assert energy > vad.energy_threshold  # speech should exceed threshold

    def test_adaptive_calibration(self):
        """Test adaptive VAD calibrates threshold from ambient noise floor."""
        import struct

        vad = EnergyVAD(sensitivity="medium", adaptive=True)
        # Before calibration: uses preset threshold
        assert vad._calibrated is False
        assert vad.energy_threshold == 0.02  # medium preset

        # Simulate 30 frames of ambient noise (quiet, below medium threshold)
        quiet_frames = 30
        for i in range(quiet_frames):
            rms = 0.005 + (i * 0.0001)  # gradually increasing quiet audio
            samples = [int(rms * 32768 * (j % 10) / 10) for j in range(1440)]
            chunk = struct.pack(f"{len(samples)}h", *samples)
            vad.detect(chunk)

        # After calibration: threshold should be ~2.5x noise floor
        assert vad._calibrated is True
        assert vad.energy_threshold < 0.02  # should be lower than preset
        assert vad.energy_threshold >= 0.015  # but not below absolute minimum

        # Reset should clear calibration
        vad.reset()
        assert vad._calibrated is False
        assert vad.energy_threshold == 0.02


class TestMockASR:
    """Tests for MockASR."""

    @pytest.mark.asyncio
    async def test_recognize_returns_result(self):
        """Test mock ASR returns expected structure."""
        asr = MockASR(latency_ms=10)
        result = await asr.recognize(b"dummy_audio")

        assert "text" in result
        assert "asr_inference_ms" in result
        assert result["text"] == "模擬語音辨識結果..."


class TestWebSocketIntegration:
    """Integration tests for WebSocket ASR."""

    def test_full_websocket_flow_with_test_client(self):
        """Test complete WebSocket flow: connect, config, audio, commit."""
        from app.main import app

        with TestClient(app) as client:
            with client.websocket_connect("/ws/asr") as ws:
                # Send config
                config_msg = {
                    "type": "config",
                    "audio": {
                        "sample_rate": 24000,
                        "channels": 1,
                        "format": "pcm"
                    }
                }
                ws.send_json(config_msg)

                # Send 5 binary chunks
                for i in range(5):
                    chunk = b"\x00\x00" * 480  # Small chunk
                    ws.send_bytes(chunk)

                # Send commit
                commit_msg = {"type": "control", "action": "commit_utterance"}
                ws.send_json(commit_msg)

                # Receive response
                response = ws.receive_json()

                # Assert final result has utterance_id
                assert response["type"] == "asr_result"
                assert response["is_final"] is True
                assert "utterance_id" in response
                assert response["utterance_id"] is not None
                assert "telemetry" in response

    def test_websocket_tts_streaming_flow(self):
        """Test WebSocket flow with TTS streaming: config → audio → ASR → LLM → TTS binary chunks."""
        from app.main import app

        received_messages = []
        binary_chunks = []

        with TestClient(app) as client:
            with client.websocket_connect("/ws/asr") as ws:
                # Send config with persona
                config_msg = {
                    "type": "config",
                    "audio": {
                        "sample_rate": 24000,
                        "channels": 1,
                        "format": "pcm"
                    },
                    "persona_id": "xiao_s",
                    "listener_id": "child",
                    "model": "gpt-4o-mini"
                }
                ws.send_json(config_msg)

                # Send enough audio chunks to simulate speech
                # Each chunk: 480 samples = 20ms at 24kHz
                # Send 100 chunks = ~2 seconds of audio
                for i in range(100):
                    chunk = b"\x00\x01" * 480  # Non-zero samples
                    ws.send_bytes(chunk)

                # Send commit
                commit_msg = {"type": "control", "action": "commit_utterance"}
                ws.send_json(commit_msg)

                # Collect all responses until LLM done or timeout
                import time
                start_time = time.time()
                timeout = 30  # seconds

                while time.time() - start_time < timeout:
                    try:
                        # Try to receive with timeout
                        ws._throttle = False  # Disable throttling for test
                        msg = ws.receive_json(timeout=1)
                        received_messages.append(msg)

                        # Check if we've received all expected messages
                        if msg.get("type") == "llm_done":
                            break
                    except Exception:
                        break

                # Verify we received ASR result
                asr_results = [m for m in received_messages if m.get("type") == "asr_result"]
                assert len(asr_results) >= 1, f"Expected at least 1 ASR result, got {len(asr_results)}"

                # Verify we received LLM start
                llm_starts = [m for m in received_messages if m.get("type") == "llm_start"]
                assert len(llm_starts) >= 1, f"Expected at least 1 LLM start, got {len(llm_starts)}"

                # Verify we received TTS start messages
                tts_starts = [m for m in received_messages if m.get("type") == "tts_start"]
                assert len(tts_starts) >= 1, f"Expected at least 1 TTS start, got {len(tts_starts)}"

                print(f"WS flow test: {len(received_messages)} messages, {len(tts_starts)} TTS sentences")


class TestWebSocketCancel:
    """Tests for WebSocket cancel/barge-in."""

    def test_websocket_cancel_stops_llm(self):
        """Test that cancel message stops ongoing LLM processing."""
        from app.main import app

        received_messages = []

        with TestClient(app) as client:
            with client.websocket_connect("/ws/asr") as ws:
                # Send config
                config_msg = {
                    "type": "config",
                    "audio": {
                        "sample_rate": 24000,
                        "channels": 1,
                        "format": "pcm"
                    }
                }
                ws.send_json(config_msg)

                # Send some audio
                for i in range(50):
                    chunk = b"\x00\x01" * 480
                    ws.send_bytes(chunk)

                # Send commit
                commit_msg = {"type": "control", "action": "commit_utterance"}
                ws.send_json(commit_msg)

                # Immediately send cancel
                cancel_msg = {"type": "control", "action": "cancel"}
                ws.send_json(cancel_msg)

                # Collect responses for a short time
                import time
                start_time = time.time()
                while time.time() - start_time < 5:
                    try:
                        msg = ws.receive_json(timeout=1)
                        received_messages.append(msg)
                        if msg.get("type") == "llm_cancelled":
                            break
                    except Exception:
                        break

                # Verify we received cancel confirmation
                cancelled = [m for m in received_messages if m.get("type") == "llm_cancelled"]
                assert len(cancelled) >= 1, "Expected llm_cancelled message"


@pytest.fixture
def health_check(test_client):
    """Test health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
