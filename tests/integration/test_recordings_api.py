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
        """Create a temporary WAV file for testing."""
        import wave
        import struct

        # Create a simple 1-second 24kHz mono WAV
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)
            # Generate 1 second of silence
            for _ in range(24000):
                wav_file.writeframes(struct.pack('<h', 0))

        return wav_path

    def test_list_recordings_empty(self, client):
        """Test listing recordings when empty."""
        response = client.get("/api/recordings/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_personas(self, client):
        """Test getting personas list."""
        response = client.get("/api/personas")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any(p["id"] == "xiao_s" for p in data)

    def test_get_listeners(self, client):
        """Test getting listeners list."""
        response = client.get("/api/listeners")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any(l["id"] == "child" for l in data)

    def test_get_recording_stats(self, client):
        """Test getting recording statistics."""
        response = client.get("/api/recordings/stats")
        assert response.status_code == 200
        data = response.json()
        assert "raw_size_bytes" in data
        assert "total_recordings" in data

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


class TestRecordingsUI:
    """Test recordings UI page."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from app.main import app
        return TestClient(app)

    def test_recordings_page_loads(self, client):
        """Test that recordings page loads successfully."""
        response = client.get("/ui/recordings")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "錄音管理" in response.text

    def test_recordings_page_has_recorder(self, client):
        """Test that recordings page has WebRTC recorder elements."""
        response = client.get("/ui/recordings")
        html = response.text
        assert 'id="recBtn"' in html
        assert 'id="stopBtn"' in html
        assert 'id="dbMeterFill"' in html
        assert 'id="listenerSelect"' in html
        assert 'id="personaSelect"' in html

    def test_recordings_page_has_upload_area(self, client):
        """Test that recordings page has upload area."""
        response = client.get("/ui/recordings")
        html = response.text
        assert 'id="uploadArea"' in html
        assert 'id="fileInput"' in html

    def test_recordings_page_has_debug_panel(self, client):
        """Test that recordings page has debug panel."""
        response = client.get("/ui/recordings")
        html = response.text
        assert 'id="debugPanel"' in html
        assert 'id="debugLogs"' in html

    def test_recordings_page_has_recordings_list(self, client):
        """Test that recordings page has recordings list."""
        response = client.get("/ui/recordings")
        html = response.text
        assert 'id="recordingsList"' in html

    def test_back_button_links_to_ui(self, client):
        """Test that back button links to main UI."""
        response = client.get("/ui/recordings")
        html = response.text
        assert 'href="/ui"' in html
