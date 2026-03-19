"""Pytest configuration and fixtures."""
import pytest
import asyncio
from typing import Generator

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
