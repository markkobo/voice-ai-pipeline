"""Integration tests for recordings API."""

import pytest
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


class TestRecordingsAPI:
    """Test recordings API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create a 5-second 48kHz mono WAV — long enough to pass min-duration
        validation and at the canonical target rate so the service's
        stdlib-WAV-verbatim path runs without ffmpeg."""
        import wave
        import struct

        wav_path = tmp_path / "test.wav"
        sample_rate = 48000
        duration_s = 5
        with wave.open(str(wav_path), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for _ in range(sample_rate * duration_s):
                wav_file.writeframes(struct.pack('<h', 0))

        return wav_path

    def test_list_recordings_empty(self, client):
        """Test listing recordings when empty."""
        response = client.get("/api/recordings/")
        assert response.status_code == 200
        data = response.json()
        # Response is now paginated dict
        assert isinstance(data, dict)
        assert "recordings" in data
        assert data["recordings"] == []
        assert data["total"] == 0

    def test_upload_invalid_listener_id(self, client, temp_audio_file):
        """Test upload with invalid listener_id."""
        with open(temp_audio_file, 'rb') as f:
            response = client.post(
                "/api/recordings/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"listener_id": "invalid", "persona_id": "xiao_s"}
            )
        assert response.status_code == 400

    def test_upload_invalid_persona_id(self, client, temp_audio_file):
        """Test upload with invalid persona_id."""
        with open(temp_audio_file, 'rb') as f:
            response = client.post(
                "/api/recordings/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"listener_id": "child", "persona_id": "invalid"}
            )
        assert response.status_code == 400

    def test_get_nonexistent_recording(self, client):
        """Test getting a recording that doesn't exist."""
        response = client.get("/api/recordings/nonexistent-id")
        assert response.status_code == 404

    def test_delete_nonexistent_recording(self, client):
        """Test deleting a recording that doesn't exist."""
        response = client.delete("/api/recordings/nonexistent-id")
        assert response.status_code == 404

    def test_update_nonexistent_recording(self, client):
        """Test updating a recording that doesn't exist."""
        response = client.patch(
            "/api/recordings/nonexistent-id",
            json={"title": "Test"}
        )
        assert response.status_code == 404

    def test_upload_and_get_recording(self, client, temp_audio_file):
        """Test uploading a file and retrieving it."""
        # Upload
        with open(temp_audio_file, 'rb') as f:
            response = client.post(
                "/api/recordings/upload",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"listener_id": "child", "persona_id": "xiao_s"}
            )
        assert response.status_code == 200
        data = response.json()
        recording_id = data["recording_id"]
        assert recording_id is not None
        assert data["status"] == "raw"

        # Get the recording
        response = client.get(f"/api/recordings/{recording_id}")
        assert response.status_code == 200
        rec_data = response.json()
        assert rec_data["recording_id"] == recording_id
        assert rec_data["listener_id"] == "child"
        assert rec_data["persona_id"] == "xiao_s"

