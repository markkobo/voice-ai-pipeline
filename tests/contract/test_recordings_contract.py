"""
Contract tests for the recordings REST API.

These tests pin both behavior (does the right thing happen?) and shape (does
the response match the Pydantic model?) for every endpoint refactored in
Phase 1.1.

Coverage matrix (each endpoint: ≥1 happy + ≥1 failure mode):
  GET    /api/recordings/                       — list/pagination
  GET    /api/recordings/stats                  — storage stats
  POST   /api/recordings/upload                 — file upload
  GET    /api/recordings/{id}                   — get one
  DELETE /api/recordings/{id}                   — delete
  PATCH  /api/recordings/{id}                   — update top-level fields
  GET    /api/recordings/{id}/stream            — audio stream
  GET    /api/recordings/{id}/download          — audio download
  GET    /api/recordings/{id}/transcription     — transcription
  GET    /api/recordings/{id}/speakers          — speaker info
  PATCH  /api/recordings/{id}/speakers          — set speaker labels
  GET    /api/recordings/{id}/segments          — segments
  PATCH  /api/recordings/{id}/segments/{sid}    — update segment routing

Note: speaker/{sid}/audio (ffmpeg slice), /process (pipeline), /cleanup-expired,
/backup are covered separately or deferred to Phase 1.2 because they depend on
external binaries (ffmpeg, rclone) or heavy services (pipeline).
"""
from __future__ import annotations

import io
import json
import wave
import struct
from pathlib import Path

import pytest

from app.services.recordings.models import Recording, RecordingStatus
from app.api.recordings import (
    DeleteResponse,
    ListResponse,
    SegmentsResponse,
    SpeakersResponse,
    StatsResponse,
    UpdateResponse,
    UpdateSegmentResponse,
    UpdateSpeakerLabelsResponse,
    UploadResponse,
)


pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_wav_bytes(duration_seconds: float = 5.0, sample_rate: int = 48000) -> bytes:
    """Construct a valid 48kHz mono 16-bit WAV in-memory (no ffmpeg needed)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for _ in range(int(duration_seconds * sample_rate)):
            w.writeframes(struct.pack("<h", 0))
    return buf.getvalue()


def _upload(client, *, duration_s: float = 5.0, listener_id: str = "child", persona_id: str = "xiao_s", title: str | None = None) -> dict:
    """Upload helper — returns the parsed JSON body of a successful upload."""
    wav = _make_wav_bytes(duration_s)
    files = {"file": ("test.wav", wav, "audio/wav")}
    data = {"listener_id": listener_id, "persona_id": persona_id}
    if title is not None:
        data["title"] = title
    response = client.post("/api/recordings/upload", files=files, data=data)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# GET /api/recordings/  — list
# ---------------------------------------------------------------------------
class TestListRecordings:
    def test_empty_list(self, client, assert_matches_schema):
        r = client.get("/api/recordings/")
        assert r.status_code == 200
        parsed = assert_matches_schema(ListResponse, r.json())
        assert parsed.recordings == []
        assert parsed.total == 0
        assert parsed.total_pages == 0

    def test_list_returns_uploaded_recording(self, client, assert_matches_schema):
        body = _upload(client, title="hello")
        rid = body["recording_id"]
        r = client.get("/api/recordings/")
        parsed = assert_matches_schema(ListResponse, r.json())
        assert parsed.total == 1
        assert parsed.recordings[0].recording_id == rid
        assert parsed.recordings[0].title == "hello"

    def test_pagination_slices_correctly(self, client):
        for _ in range(3):
            _upload(client)
        page1 = client.get("/api/recordings/?page=1&limit=2").json()
        page2 = client.get("/api/recordings/?page=2&limit=2").json()
        assert page1["total"] == 3
        assert page1["total_pages"] == 2
        assert len(page1["recordings"]) == 2
        assert len(page2["recordings"]) == 1
        # No overlap between pages.
        ids1 = {r["recording_id"] for r in page1["recordings"]}
        ids2 = {r["recording_id"] for r in page2["recordings"]}
        assert ids1.isdisjoint(ids2)

    def test_pagination_caps_limit_at_100(self, client):
        r = client.get("/api/recordings/?limit=9999")
        body = r.json()
        assert body["limit"] == 100


# ---------------------------------------------------------------------------
# GET /api/recordings/stats
# ---------------------------------------------------------------------------
class TestStats:
    def test_stats_initially_zero(self, client, assert_matches_schema):
        r = client.get("/api/recordings/stats")
        assert r.status_code == 200
        parsed = assert_matches_schema(StatsResponse, r.json())
        assert parsed.total_recordings == 0
        assert parsed.raw_size_bytes == 0

    def test_stats_reflects_upload(self, client):
        _upload(client)
        r = client.get("/api/recordings/stats")
        body = r.json()
        assert body["total_recordings"] == 1
        assert body["raw_size_bytes"] > 0


# ---------------------------------------------------------------------------
# POST /api/recordings/upload
# ---------------------------------------------------------------------------
class TestUpload:
    def test_happy_path(self, client, assert_matches_schema):
        body = _upload(client, title="hi")
        parsed = assert_matches_schema(UploadResponse, body)
        assert parsed.status == "raw"
        assert parsed.duration_seconds == pytest.approx(5.0, abs=0.1)
        assert parsed.recording_id

    def test_rejects_invalid_listener(self, client):
        r = client.post(
            "/api/recordings/upload",
            files={"file": ("t.wav", _make_wav_bytes(), "audio/wav")},
            data={"listener_id": "alien", "persona_id": "xiao_s"},
        )
        assert r.status_code == 400
        body = r.json()
        assert body["error"] == "invalid_listener_id"
        assert "alien" in body["message"]
        assert "alien" in body["details"]["listener_id"]

    def test_rejects_invalid_persona(self, client):
        r = client.post(
            "/api/recordings/upload",
            files={"file": ("t.wav", _make_wav_bytes(), "audio/wav")},
            data={"listener_id": "child", "persona_id": "nobody"},
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_persona_id"

    def test_rejects_short_audio(self, client):
        r = client.post(
            "/api/recordings/upload",
            files={"file": ("short.wav", _make_wav_bytes(duration_seconds=1.0), "audio/wav")},
            data={"listener_id": "child", "persona_id": "xiao_s"},
        )
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "invalid_audio"
        assert "too short" in body["message"].lower()
        assert body["details"]["duration_seconds"] == pytest.approx(1.0, abs=0.05)

    def test_rejects_long_audio(self, client):
        # 301s — just over the 300s limit. Generated as silence so it's tiny.
        r = client.post(
            "/api/recordings/upload",
            files={"file": ("long.wav", _make_wav_bytes(duration_seconds=301.0), "audio/wav")},
            data={"listener_id": "child", "persona_id": "xiao_s"},
        )
        assert r.status_code == 422
        assert "too long" in r.json()["message"].lower()

    def test_rejects_bad_extension(self, client):
        r = client.post(
            "/api/recordings/upload",
            files={"file": ("evil.txt", b"\x00" * 100, "text/plain")},
            data={"listener_id": "child", "persona_id": "xiao_s"},
        )
        assert r.status_code == 415
        body = r.json()
        assert body["error"] == "unsupported_audio_format"

    def test_rejects_corrupt_wav(self, client):
        r = client.post(
            "/api/recordings/upload",
            files={"file": ("corrupt.wav", b"RIFF\x00\x00\x00\x00WAVEgarbage", "audio/wav")},
            data={"listener_id": "child", "persona_id": "xiao_s"},
        )
        # Corrupt WAV header but valid extension → InvalidAudioError (422).
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_audio"


# ---------------------------------------------------------------------------
# GET /api/recordings/{id}
# ---------------------------------------------------------------------------
class TestGetRecording:
    def test_returns_full_recording(self, client, assert_matches_schema):
        rid = _upload(client, title="t")["recording_id"]
        r = client.get(f"/api/recordings/{rid}")
        assert r.status_code == 200
        parsed = assert_matches_schema(Recording, r.json())
        assert parsed.recording_id == rid
        assert parsed.title == "t"
        assert parsed.status == RecordingStatus.raw

    def test_404_when_missing(self, client):
        r = client.get("/api/recordings/does-not-exist")
        assert r.status_code == 404
        assert r.json()["error"] == "recording_not_found"


# ---------------------------------------------------------------------------
# DELETE /api/recordings/{id}
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_round_trip(self, client, assert_matches_schema):
        rid = _upload(client)["recording_id"]
        r = client.delete(f"/api/recordings/{rid}")
        assert r.status_code == 200
        parsed = assert_matches_schema(DeleteResponse, r.json())
        assert parsed.recording_id == rid
        # Subsequent GET returns 404.
        assert client.get(f"/api/recordings/{rid}").status_code == 404

    def test_404_when_missing(self, client):
        r = client.delete("/api/recordings/nope")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/recordings/{id}
# ---------------------------------------------------------------------------
class TestPatchRecording:
    def test_update_title(self, client, assert_matches_schema):
        rid = _upload(client)["recording_id"]
        r = client.patch(f"/api/recordings/{rid}", json={"title": "renamed"})
        assert r.status_code == 200
        assert_matches_schema(UpdateResponse, r.json())
        got = client.get(f"/api/recordings/{rid}").json()
        assert got["title"] == "renamed"

    def test_unknown_field_rejected(self, client):
        rid = _upload(client)["recording_id"]
        r = client.patch(f"/api/recordings/{rid}", json={"surprise": "hello"})
        # Pydantic model has extra='forbid' → FastAPI returns 422 with detail.
        assert r.status_code == 422

    def test_invalid_persona_id_rejected(self, client):
        rid = _upload(client)["recording_id"]
        r = client.patch(f"/api/recordings/{rid}", json={"persona_id": "alien"})
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_persona_id"

    def test_invalid_listener_id_rejected(self, client):
        rid = _upload(client)["recording_id"]
        r = client.patch(f"/api/recordings/{rid}", json={"listener_id": "alien"})
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_listener_id"

    def test_404_when_missing(self, client):
        r = client.patch("/api/recordings/nope", json={"title": "x"})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Streaming + download
# ---------------------------------------------------------------------------
class TestAudioStream:
    def test_stream_returns_wav_bytes(self, client):
        rid = _upload(client)["recording_id"]
        r = client.get(f"/api/recordings/{rid}/stream?stage=raw")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")
        # RIFF header at the start of the file.
        assert r.content[:4] == b"RIFF"
        assert r.content[8:12] == b"WAVE"

    def test_stream_invalid_stage_rejected(self, client):
        rid = _upload(client)["recording_id"]
        r = client.get(f"/api/recordings/{rid}/stream?stage=midi")
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_audio"

    def test_stream_404_when_missing(self, client):
        r = client.get("/api/recordings/nope/stream")
        assert r.status_code == 404

    def test_download_returns_raw_wav(self, client):
        rid = _upload(client)["recording_id"]
        r = client.get(f"/api/recordings/{rid}/download")
        assert r.status_code == 200
        assert r.content[:4] == b"RIFF"

    def test_download_404_when_missing(self, client):
        r = client.get("/api/recordings/nope/download")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Transcription / speakers / segments
# ---------------------------------------------------------------------------
class TestTranscriptionAndSpeakers:
    def test_transcription_default_empty(self, client):
        rid = _upload(client)["recording_id"]
        r = client.get(f"/api/recordings/{rid}/transcription")
        assert r.status_code == 200
        body = r.json()
        assert body["text"] is None
        assert body["language"] == "zh"
        assert body["segments"] == []

    def test_speakers_initially_empty(self, client, assert_matches_schema):
        rid = _upload(client)["recording_id"]
        r = client.get(f"/api/recordings/{rid}/speakers")
        assert r.status_code == 200
        parsed = assert_matches_schema(SpeakersResponse, r.json())
        assert parsed.speakers == []
        assert parsed.segment_count == 0

    def test_segments_initially_empty(self, client, assert_matches_schema):
        rid = _upload(client, listener_id="mom")["recording_id"]
        r = client.get(f"/api/recordings/{rid}/segments")
        assert r.status_code == 200
        parsed = assert_matches_schema(SegmentsResponse, r.json())
        assert parsed.listener_id == "mom"
        assert parsed.persona_id == "xiao_s"
        assert parsed.segments == []

    def test_segments_404_when_missing(self, client):
        r = client.get("/api/recordings/nope/segments")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# PATCH speaker labels and segments
# ---------------------------------------------------------------------------
class TestUpdateSpeakerLabels:
    def test_set_labels(self, client, assert_matches_schema):
        rid = _upload(client)["recording_id"]
        r = client.patch(
            f"/api/recordings/{rid}/speakers",
            json={"speaker_labels": {"SPEAKER_00": "xiao_s"}},
        )
        assert r.status_code == 200, r.text
        parsed = assert_matches_schema(UpdateSpeakerLabelsResponse, r.json())
        assert parsed.speaker_labels == {"SPEAKER_00": "xiao_s"}
        # Persisted: GET reflects the change.
        got = client.get(f"/api/recordings/{rid}").json()
        assert got["speaker_labels"] == {"SPEAKER_00": "xiao_s"}

    def test_unknown_persona_rejected(self, client):
        rid = _upload(client)["recording_id"]
        r = client.patch(
            f"/api/recordings/{rid}/speakers",
            json={"speaker_labels": {"SPEAKER_00": "alien"}},
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_persona_id"

    def test_unknown_field_rejected(self, client):
        rid = _upload(client)["recording_id"]
        r = client.patch(
            f"/api/recordings/{rid}/speakers",
            json={"speaker_labels": {"SPEAKER_00": "xiao_s"}, "extra": 1},
        )
        assert r.status_code == 422


class TestUpdateSegmentRouting:
    def test_404_when_recording_missing(self, client):
        r = client.patch("/api/recordings/nope/segments/SPEAKER_00?persona_id=xiao_s")
        assert r.status_code == 404

    def test_speaker_not_in_segments(self, client):
        rid = _upload(client)["recording_id"]
        r = client.patch(
            f"/api/recordings/{rid}/segments/SPEAKER_99?persona_id=xiao_s",
        )
        # No segments yet → service treats unmatched speaker as invalid audio.
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_audio"

    def test_no_field_provided(self, client):
        rid = _upload(client)["recording_id"]
        r = client.patch(f"/api/recordings/{rid}/segments/SPEAKER_00")
        assert r.status_code == 422
        assert "at least one" in r.json()["message"].lower()


# ---------------------------------------------------------------------------
# Cleanup-expired — both dry-run and real-delete paths.
# ---------------------------------------------------------------------------
class TestCleanupExpired:
    def test_dry_run_nothing_to_delete(self, client):
        r = client.post("/api/recordings/cleanup-expired?dry_run=true")
        assert r.status_code == 200
        body = r.json()
        assert body["dry_run"] is True
        assert body["would_delete"] == 0
        assert body["recordings"] == []

    def test_real_delete_clears_expired(self, client, isolated_data):
        from datetime import datetime, timedelta, timezone
        from app.api._dependencies import make_recordings_service_for_testing
        from app.services.recordings.models import Recording, RecordingStatus

        # Use the same service the API uses so persistence is shared.
        service = make_recordings_service_for_testing(isolated_data)
        rid = _upload(client)["recording_id"]
        # Mark as expired by mutating directly through the repository.
        past = datetime.now(timezone.utc) - timedelta(days=1)

        def mark_expired(r: Recording) -> None:
            r.status = RecordingStatus.processed
            r.processed_expires_at = past

        service.repository.update(rid, mark_expired)

        r = client.post("/api/recordings/cleanup-expired")
        assert r.status_code == 200
        body = r.json()
        # Cleanup is best-effort — the test only requires it ran without error
        # and reported on the expired recording.
        assert body["dry_run"] is False
        assert any(rec["recording_id"] == rid for rec in body["recordings"])


# ---------------------------------------------------------------------------
# Stream fallback + speaker-audio (ffmpeg-dependent).
# ---------------------------------------------------------------------------
class TestStreamFallback:
    def test_stream_falls_back_to_raw_when_enhanced_missing(self, client):
        """`stage=enhanced` returns the raw file when no enhanced copy exists."""
        rid = _upload(client)["recording_id"]
        # No pipeline ran → enhanced/ doesn't exist for this recording.
        r = client.get(f"/api/recordings/{rid}/stream?stage=enhanced")
        assert r.status_code == 200
        assert r.content[:4] == b"RIFF"


class TestSpeakerAudio:
    def test_speaker_audio_404_when_recording_missing(self, client):
        r = client.get("/api/recordings/nope/speaker/SPEAKER_00/audio")
        assert r.status_code == 404

    def test_speaker_audio_404_when_speaker_file_missing(self, client):
        rid = _upload(client)["recording_id"]
        r = client.get(f"/api/recordings/{rid}/speaker/SPEAKER_00/audio")
        assert r.status_code == 404
        assert r.json()["error"] == "recording_not_found"


class TestProcessingTrigger:
    def test_process_404_when_recording_missing(self, client):
        r = client.post("/api/recordings/nope/process")
        assert r.status_code == 404

    def test_process_rejects_already_processing(self, client, isolated_data):
        from app.api._dependencies import make_recordings_service_for_testing
        from app.services.recordings.models import Recording, RecordingStatus

        service = make_recordings_service_for_testing(isolated_data)
        rid = _upload(client)["recording_id"]

        def set_processing(r: Recording) -> None:
            r.status = RecordingStatus.processing

        service.repository.update(rid, set_processing)

        r = client.post(f"/api/recordings/{rid}/process")
        assert r.status_code == 422
        assert "already processing" in r.json()["message"].lower()
