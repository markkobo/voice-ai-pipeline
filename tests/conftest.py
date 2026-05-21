"""
Pytest configuration and shared fixtures.

Sets test-mode environment variables BEFORE importing the app and provides:
- Isolated data directories (recordings/training/models point to tmp_path)
- A FastAPI TestClient that doesn't trigger ASR/TTS preload
- Audio byte factories for upload tests
- Helpers for asserting Pydantic response schemas

Per-test isolation strategy: function-scope `isolated_data` monkeypatches the
module-level `Path` constants in app.services.recordings.file_storage and
related modules so each test sees a fresh data tree.
"""
from __future__ import annotations

import os
import struct
import wave
from pathlib import Path
from typing import Callable, Iterator

import pytest

# ---------------------------------------------------------------------------
# Environment must be set BEFORE any app.* import. pytest collects this file
# first, so these env vars are in place when fixtures import the app.
# ---------------------------------------------------------------------------
os.environ.setdefault("USE_QWEN_ASR", "false")
os.environ.setdefault("USE_MOCK_TTS", "true")
os.environ.setdefault("USE_MOCK_LLM", "true")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
# Disable the Prometheus HTTP server during tests so we don't bind a real port
# and don't conflict with parallel test workers. The collector itself still
# records metrics in-process.
os.environ.setdefault("TELEMETRY_HTTP_DISABLED", "true")

# Point app.config at tmp dirs via env vars. With the Phase 1.3 cleanup of
# /workspace hardcodes, this is now the canonical way to redirect — no more
# monkey-patching of setup_json_logging or module-level constants.
import tempfile as _tempfile  # noqa: E402

_session_tmp = Path(_tempfile.gettempdir())
os.environ.setdefault("MODELS_DIR", str(_session_tmp / "voice-ai-test-models"))
os.environ.setdefault("LOG_DIR", str(_session_tmp / "voice-ai-test-logs"))
# DATA_ROOT is set per-test by the isolated_data fixture so each test gets
# its own data tree; setting a default here would leak between tests.


# ---------------------------------------------------------------------------
# Telemetry: patch BEFORE app.main imports so the metrics server doesn't bind
# port 9090. We don't modify app/main.py here — that's Phase 1.1 work.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _disable_telemetry_http_server():
    """Prevent TelemetryCollector from binding port 9090 during tests."""
    from telemetry.collector import TelemetryCollector

    original_start = TelemetryCollector.start_server

    def noop_start_server(self):  # type: ignore[no-untyped-def]
        # Mark as started so the rest of the class behaves consistently.
        self._server_thread = object()

    TelemetryCollector.start_server = noop_start_server  # type: ignore[assignment]
    yield
    TelemetryCollector.start_server = original_start  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# App + TestClient.
#
# We deliberately do NOT use `with TestClient(app) as client:` — the
# `with`-block triggers FastAPI startup events, which would try to preload the
# real Qwen ASR/TTS models. Tests that need startup behavior should opt in
# explicitly via a separate `client_with_startup` fixture (TBD).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def app():
    """Import the FastAPI app once per session."""
    from app.main import app as fastapi_app

    return fastapi_app


@pytest.fixture
def client(app):
    """FastAPI TestClient without startup-event side effects."""
    from fastapi.testclient import TestClient

    return TestClient(app)


# ---------------------------------------------------------------------------
# Isolated data directories.
#
# The recordings code uses module-level `Path` constants for data dirs.
# This fixture redirects them to a per-test tmp_path so tests don't share
# state and don't write to /workspace/voice-ai-pipeline/data.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch, app) -> Path:
    """
    Redirect all recording storage to a per-test tmp directory.

    Two-pronged isolation:
    1. Monkeypatch the file_storage module-level constants used by legacy code
       paths (pipeline.py, metadata.py) that still read RAW_DIR etc.
    2. Inject a fresh RecordingsService pointing at this tmp data_root via
       FastAPI dependency_overrides — the canonical pattern for DI-based tests
       and the only way to defeat the cached singleton on app.state.
    """
    data_root = tmp_path / "data"
    recordings_root = data_root / "recordings"
    raw_dir = recordings_root / "raw"
    denoised_dir = recordings_root / "denoised"
    enhanced_dir = recordings_root / "enhanced"
    voice_profiles_dir = data_root / "voice_profiles"
    models_dir = data_root / "models"

    for d in (raw_dir, denoised_dir, enhanced_dir, voice_profiles_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    import app.services.recordings.file_storage as fs

    monkeypatch.setattr(fs, "DATA_DIR", data_root, raising=True)
    monkeypatch.setattr(fs, "RECORDINGS_DIR", recordings_root, raising=True)
    monkeypatch.setattr(fs, "RAW_DIR", raw_dir, raising=True)
    monkeypatch.setattr(fs, "DENOISED_DIR", denoised_dir, raising=True)
    monkeypatch.setattr(fs, "ENHANCED_DIR", enhanced_dir, raising=True)
    monkeypatch.setattr(fs, "VOICE_PROFILES_DIR", voice_profiles_dir, raising=True)
    monkeypatch.setattr(fs, "MODELS_DIR", models_dir, raising=True)
    # RECORDINGS_INDEX_FILE + _recordings_cache used to live on the file_storage
    # module — they were removed when the legacy index/cache was deleted in
    # favor of JsonRecordingsRepository. No replacement needed: tests that need
    # an isolated index use the repository injected via dependency_overrides
    # below.

    # Inject fresh services into the FastAPI app via dependency_overrides so
    # the cached singletons on app.state can't leak across tests.
    from app.api._dependencies import (
        get_corpus_service,
        get_ingestion_service,
        get_listeners_service,
        get_personas_service,
        get_recordings_service,
        get_training_service,
        make_corpus_service_for_testing,
        make_ingestion_service_for_testing,
        make_listeners_service_for_testing,
        make_personas_service_for_testing,
        make_recordings_service_for_testing,
        make_training_service_for_testing,
    )

    test_personas_service = make_personas_service_for_testing(data_root)
    test_listeners_service = make_listeners_service_for_testing(data_root)
    test_recordings_service = make_recordings_service_for_testing(data_root)
    test_training_service = make_training_service_for_testing(
        data_root, recordings_service=test_recordings_service
    )
    test_corpus_service = make_corpus_service_for_testing(data_root)
    test_ingestion_service = make_ingestion_service_for_testing(
        data_root, corpus_service=test_corpus_service,
    )
    app.dependency_overrides[get_personas_service] = lambda: test_personas_service
    app.dependency_overrides[get_listeners_service] = lambda: test_listeners_service
    app.dependency_overrides[get_recordings_service] = lambda: test_recordings_service
    app.dependency_overrides[get_training_service] = lambda: test_training_service
    app.dependency_overrides[get_corpus_service] = lambda: test_corpus_service
    app.dependency_overrides[get_ingestion_service] = lambda: test_ingestion_service

    # Also reset the legacy module-level singletons so the back-compat
    # function-style API (`list_personas()` etc.) sees the per-test data
    # root instead of leaking state from a previous test.
    from app.services.listeners import _reset_default_service_for_testing as _r_l
    from app.services.personas import _reset_default_service_for_testing as _r_p

    _r_p()
    _r_l()
    # `_resolve_data_root` honors DATA_ROOT env; point it at the tmp dir so
    # the legacy callers see the isolated data too.
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    # Clear cached singletons on app.state.
    for attr in (
        "_recordings_service",
        "_training_service",
        "_personas_service",
        "_listeners_service",
        "_corpus_service",
        "_ingestion_service",
    ):
        if hasattr(app.state, attr):
            delattr(app.state, attr)

    yield data_root

    app.dependency_overrides.pop(get_personas_service, None)
    app.dependency_overrides.pop(get_listeners_service, None)
    app.dependency_overrides.pop(get_recordings_service, None)
    app.dependency_overrides.pop(get_training_service, None)
    app.dependency_overrides.pop(get_corpus_service, None)
    app.dependency_overrides.pop(get_ingestion_service, None)
    _r_p()
    _r_l()


# ---------------------------------------------------------------------------
# Audio byte factories — produce valid WAV bytes for upload tests without
# hitting tests/fixtures/. Used when a test wants a parameterized duration.
# ---------------------------------------------------------------------------
WavBytesFactory = Callable[..., bytes]


def _build_wav_bytes(
    duration_seconds: float = 5.0,
    sample_rate: int = 24000,
    frequency_hz: float = 220.0,
    amplitude: float = 0.2,
) -> bytes:
    """
    Generate WAV bytes containing a sine wave.

    Default 5s sine tone at 24kHz is well within the API's 3s..300s window.
    """
    import io
    import math

    n_frames = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n_frames):
            value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
            w.writeframes(struct.pack("<h", value))
    return buf.getvalue()


@pytest.fixture
def wav_bytes() -> WavBytesFactory:
    """Return a factory: wav_bytes(duration_seconds=5.0) -> bytes."""
    return _build_wav_bytes


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory holding committed JSON + binary fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def silent_wav_bytes() -> WavBytesFactory:
    """Return a factory for silent WAVs (used when only duration matters)."""

    def factory(duration_seconds: float = 5.0, sample_rate: int = 24000) -> bytes:
        return _build_wav_bytes(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            amplitude=0.0,
        )

    return factory


# ---------------------------------------------------------------------------
# Schema assertions — used by contract tests in tests/contract/ to pin
# response shapes against Pydantic models without importing them everywhere.
# ---------------------------------------------------------------------------
@pytest.fixture
def assert_matches_schema():
    """
    Assert a JSON response matches a Pydantic model.

    Usage:
        def test_x(client, assert_matches_schema):
            from app.services.recordings.models import Recording
            r = client.get("/api/recordings/foo").json()
            assert_matches_schema(Recording, r)
    """
    from pydantic import BaseModel, ValidationError

    def _assert(model_cls: type[BaseModel], data: dict) -> BaseModel:
        try:
            return model_cls.model_validate(data)
        except ValidationError as e:
            pytest.fail(f"Response does not match {model_cls.__name__}: {e}")

    return _assert


# ---------------------------------------------------------------------------
# Legacy fixtures retained for tests not yet migrated to the new pattern.
# Marked for removal in Phase 1.1 once the contract tests replace them.
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_audio_chunk() -> bytes:
    """100ms of silence at 24kHz mono Int16. Legacy — used by test_ws_asr.py."""
    return b"\x00\x00" * 2400


@pytest.fixture
def audio_config() -> dict:
    """Legacy WS config payload."""
    return {
        "type": "config",
        "audio": {"sample_rate": 24000, "channels": 1, "format": "pcm"},
    }


@pytest.fixture
def commit_message() -> str:
    """Legacy WS commit control message."""
    return '{"type": "control", "action": "commit_utterance"}'


@pytest.fixture
def sample_recording_metadata() -> dict:
    """Legacy metadata dict — used until Recording Pydantic model lands (Task #7)."""
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
            "segments": [{"start": 0.0, "end": 5.0, "text": "這是測試"}],
        },
        "quality_metrics": {
            "snr_db": 25.5,
            "rms_volume": -12.3,
            "silence_ratio": 0.1,
            "clarity_score": 0.85,
            "training_ready": True,
        },
        "status": "processed",
        "speaker_segments": [
            {"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 12.5},
        ],
        "created_at": "2026-03-29T12:00:00",
        "updated_at": "2026-03-29T12:00:30",
    }
