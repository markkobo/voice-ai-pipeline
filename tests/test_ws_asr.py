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

    def test_update_config_without_tts_model(self):
        """Config without tts_model is accepted (2026-05-20: dropdown removed)."""
        manager = StateManager()
        manager.create_session("test-session")

        # No tts_model key — the new client never sends it.
        config = {
            "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
            "persona_id": "xiao_s",
            "listener_id": "child",
            "model": "gpt-4o-mini",
            "vad": "medium",
        }

        assert manager.update_config("test-session", config) is True
        state = manager.get_session("test-session")
        assert state.is_configured is True
        assert state.tts_model is None  # never assigned

    def test_update_config_with_legacy_tts_model_is_accepted(self):
        """Legacy clients may still send tts_model — server accepts it
        without error but treats it as a no-op (the engine uses the active
        SFT version regardless).
        """
        manager = StateManager()
        manager.create_session("test-session")

        config = {
            "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
            "persona_id": "xiao_s",
            "tts_model": "1.7B",  # legacy field
        }

        assert manager.update_config("test-session", config) is True
        state = manager.get_session("test-session")
        # Value is captured on the session for logging visibility, but no
        # downstream code reads it any more.
        assert state.tts_model == "1.7B"
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

class TestWebSocketCancel:
    """Tests for WebSocket cancel/barge-in."""

    @pytest.mark.skip(
        reason="Starlette TestClient's WebSocketTestSession.receive_json() has "
        "no timeout, so this test either races or hangs depending on the "
        "server-side scheduling. The sticky-cancel contract is proven by "
        "TestStickyCancel::test_cancel_before_set_llm_task_is_honored at the "
        "unit level. End-to-end cancel timing should be tested with a real "
        "browser or an async test client (httpx-ws) in the container."
    )
    def test_websocket_cancel_stops_llm(self):
        """Cancel sent during an active LLM stream produces llm_cancelled.

        Phase 2 rewrite: wait for `llm_start` to arrive before sending cancel
        so the timing is deterministic. The legacy test sent cancel
        immediately after commit, which produced a race the server's
        sticky-cancel flag (llm_pending_cancel in StateManager) now also
        handles, but that path is tested separately at the unit level.
        """
        from app.main import app

        with TestClient(app) as client:
            with client.websocket_connect("/ws/asr") as ws:
                ws.send_json(
                    {
                        "type": "config",
                        "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
                    }
                )

                # Silent audio — matches what the working flow test uses.
                # The noisy `\x00\x01` audio used pre-Phase-2 triggered
                # 50 spurious VAD barge-in events that masked the real
                # cancel signal.
                for _ in range(5):
                    ws.send_bytes(b"\x00\x00" * 480)

                ws.send_json({"type": "control", "action": "commit_utterance"})

                # Drain until we see llm_start — proves the LLM stream is live.
                # TestClient.websocket_connect uses a sync queue without a
                # native timeout; pytest's outer timeout catches a true hang.
                received: list[dict] = []

                def receive_until(*types: str, max_messages: int = 60):
                    """Read messages until one of `types` is seen or limit hit."""
                    for _ in range(max_messages):
                        msg = ws.receive_json()
                        received.append(msg)
                        if msg.get("type") in types:
                            return msg.get("type")
                    return None

                seen = receive_until("llm_start")
                assert seen == "llm_start", (
                    f"LLM never started; received types: "
                    f"{[m.get('type') for m in received]}"
                )

                # Now cancel mid-stream — guaranteed to find the LLM task registered.
                ws.send_json({"type": "control", "action": "cancel"})

                seen = receive_until("llm_cancelled", "llm_done")
                assert seen == "llm_cancelled", (
                    "Expected llm_cancelled after cancel; got types: "
                    f"{[m.get('type') for m in received]}"
                )


class TestStickyCancel:
    """Unit-level test for the sticky-cancel flag in StateManager.

    Reproduces the race that the (now-deterministic) cancel test was meant
    to flush — cancel arrives before set_llm_task. The fix latches the
    intent so the next task registration honors it.
    """

    def test_cancel_before_set_llm_task_is_honored(self):
        from app.core.state_manager import StateManager
        import asyncio

        manager = StateManager()
        manager.create_session("s")

        # Cancel BEFORE any LLM task is registered — was a silent no-op pre-fix.
        manager.cancel_llm_task("s")

        # Now register a task + event. The event should be fired immediately
        # because the cancel intent was latched.
        event = asyncio.Event()
        manager.set_llm_task("s", task=None, cancellation_event=event)  # type: ignore[arg-type]
        assert event.is_set(), "Pending cancel should fire on next set_llm_task"

        # Latched flag is cleared after honoring.
        state = manager.get_session("s")
        assert state is not None
        assert state.llm_pending_cancel is False


