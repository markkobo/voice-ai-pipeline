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
        vad = EnergyVAD(energy_threshold=0.01)

        # Silent audio
        silence = b"\x00\x00" * 1000
        is_speaking, energy = vad.detect(silence)

        assert is_speaking is False
        assert energy < 0.01

    def test_speech_detection(self):
        """Test VAD with simulated speech."""
        vad = EnergyVAD(energy_threshold=0.01)

        # Generate audio with some energy (sine wave-like pattern)
        import struct
        samples = [int(16000 * (i % 10) / 10) for i in range(1000)]
        speech = struct.pack(f"{len(samples)}h", *samples)

        is_speaking, energy = vad.detect(speech)

        assert is_speaking is True
        assert energy > 0


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


@pytest.fixture
def health_check(test_client):
    """Test health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
