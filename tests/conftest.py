"""Pytest configuration and fixtures."""
import pytest
import asyncio
import tempfile
import shutil
from typing import Generator
from pathlib import Path

import httpx
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio_chunk() -> bytes:
    """Generate a sample PCM audio chunk (16-bit, 24kHz, 100ms)."""
    # 24000 Hz * 0.1s * 2 bytes/sample = 4800 bytes
    # Generate silence (zeros)
    return b"\x00\x00" * 2400


@pytest.fixture
def audio_config() -> dict:
    """Sample audio configuration."""
    return {
        "type": "config",
        "audio": {
            "sample_rate": 24000,
            "channels": 1,
            "format": "pcm"
        }
    }


@pytest.fixture
def commit_message() -> str:
    """Message to commit utterance."""
    return '{"type": "control", "action": "commit_utterance"}'


@pytest.fixture
def test_client():
    """Create test client for FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create subdirectories
    (data_dir / "recordings" / "raw").mkdir(parents=True)
    (data_dir / "recordings" / "denoised").mkdir(parents=True)
    (data_dir / "recordings" / "enhanced").mkdir(parents=True)
    (data_dir / "voice_profiles").mkdir()
    (data_dir / "models").mkdir()

    yield data_dir

    # Cleanup
    shutil.rmtree(data_dir)


@pytest.fixture
def sample_recording_metadata():
    """Sample recording metadata for testing."""
    return {
        "recording_id": "test-uuid-123",
        "folder_name": "child_xiao_s_20260329_120000",
        "listener_id": "child",
        "persona_id": "xiao_s",
        "title": "Test Recording",
        "duration_seconds": 45.5,
        "file_size_bytes": 4500000,
        "transcription": {
            "text": "這是測試文字稿",
            "confidence": 0.95,
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "這是測試"},
            ],
        },
        "quality_metrics": {
            "snr_db": 25.5,
            "rms_volume": -12.3,
            "silence_ratio": 0.1,
            "clarity_score": 0.85,
            "training_ready": True,
        },
        "status": "processed",
        "processing_steps": {
            "denoise": {"status": "done", "progress": 100, "started_at": None, "completed_at": None, "error_message": None},
            "enhance": {"status": "done", "progress": 100, "started_at": None, "completed_at": None, "error_message": None},
            "diarize": {"status": "done", "progress": 100, "started_at": None, "completed_at": None, "error_message": None},
            "transcribe": {"status": "done", "progress": 100, "started_at": None, "completed_at": None, "error_message": None},
        },
        "speaker_segments": [
            {"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 12.5},
        ],
        "pipeline_metrics": {
            "denoise_ms": 1200,
            "enhance_ms": 3500,
            "diarize_ms": 8000,
            "transcribe_ms": 2200,
            "total_ms": 14900,
        },
        "created_at": "2026-03-29T12:00:00",
        "updated_at": "2026-03-29T12:00:30",
        "processed_at": "2026-03-29T12:00:30",
        "processed_expires_at": "2026-04-01T12:00:30",
    }
