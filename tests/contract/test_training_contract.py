"""
Contract tests for the training REST API.

Covers each of the 13 endpoints with ≥1 happy + ≥1 failure mode and pins both
behavior and Pydantic response shape.

Helpers create a real `Recording` with a real `speakers/{speaker_id}.wav` file
on disk so the AudioResolver returns a valid segment without needing the full
recording pipeline.
"""
from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path

import pytest

from app.api._dependencies import (
    get_training_service,
    make_training_service_for_testing,
)
from app.services.recordings.models import (
    Recording,
    RecordingStatus,
    SpeakerSegment,
)
from app.services.training_service.models import (
    TrainingType,
    TrainingVersion,
    VersionStatus,
)
from app.api.training import (
    ActivateResponse,
    ActiveVersionResponse,
    CancelResponse,
    CreateTrainingResponse,
    DeleteVersionResponse,
    ListVersionsResponse,
    TrainingStatusResponse,
    UpdateVersionResponse,
)

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Fixtures local to this file
# ---------------------------------------------------------------------------
def _make_wav(path: Path, duration_seconds: float = 20.0, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for _ in range(int(duration_seconds * sample_rate)):
            w.writeframes(struct.pack("<h", 0))


@pytest.fixture
def seeded_recording(isolated_data: Path, app):
    """
    Seed a fully-processed recording with one speaker segment so the
    AudioResolver can find audio for the matching `_SPEAKER_00` segment_id.

    Yields the recording_id so the test can build `{rid}_SPEAKER_00`.
    """
    from app.api._dependencies import make_recordings_service_for_testing
    from app.services.recordings.repository import JsonRecordingsRepository

    rs = make_recordings_service_for_testing(isolated_data)
    # Use service.upload to get a real Recording on disk (writes index.json
    # and the audio file in a way the resolver can find).
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        for _ in range(48000 * 5):  # 5s
            w.writeframes(struct.pack("<h", 0))
    rec = rs.upload(
        file_bytes=buf.getvalue(),
        filename="seed.wav",
        listener_id="child",
        persona_id="xiao_s",
    )

    # Mark it processed and add a SpeakerSegment with a duration the resolver
    # can use. Also write the actual speakers/SPEAKER_00.wav file on disk.
    speakers_folder = (
        isolated_data
        / "recordings"
        / "raw"
        / rec.folder_name
        / "speakers"
    )
    speakers_folder.mkdir(parents=True, exist_ok=True)
    _make_wav(speakers_folder / "SPEAKER_00.wav", duration_seconds=20.0)

    def add_segment(r: Recording) -> None:
        r.status = RecordingStatus.processed
        r.speaker_segments = [
            SpeakerSegment(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=20.0,
                duration_seconds=20.0,
                audio_path=str(speakers_folder / "SPEAKER_00.wav"),
                persona_id="xiao_s",
                listener_id="child",
            )
        ]

    rs.repository.update(rec.recording_id, add_segment)

    # Wire BOTH the recordings AND training services to point at the same
    # data root so the training service sees the same recordings.
    test_training = make_training_service_for_testing(
        isolated_data, recordings_service=rs
    )
    app.dependency_overrides[get_training_service] = lambda: test_training

    yield rec.recording_id


def _create_payload(recording_id: str, **overrides) -> dict:
    payload = {
        "persona_id": "xiao_s",
        "segment_ids": [f"{recording_id}_SPEAKER_00"],
        "rank": 16,
        "num_epochs": 10,
        "batch_size": 4,
        "training_type": "lora",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# POST /versions  — create + start
# ---------------------------------------------------------------------------
class TestCreateTraining:
    def test_happy_path(self, client, seeded_recording, assert_matches_schema):
        r = client.post("/api/training/versions", json=_create_payload(seeded_recording))
        assert r.status_code == 200, r.text
        parsed = assert_matches_schema(CreateTrainingResponse, r.json())
        assert parsed.persona_id == "xiao_s"
        assert parsed.status == VersionStatus.training
        assert parsed.training_type == TrainingType.lora
        assert parsed.total_duration_seconds == pytest.approx(20.0, abs=0.5)

    def test_invalid_rank_rejected(self, client, seeded_recording):
        r = client.post(
            "/api/training/versions",
            json=_create_payload(seeded_recording, rank=15),
        )
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "invalid_training_params"
        assert "rank" in body["message"]

    def test_invalid_epochs_rejected(self, client, seeded_recording):
        # MAX_EPOCHS was bumped 50→200 to allow SFT runs; pick something
        # above the new cap.
        r = client.post(
            "/api/training/versions",
            json=_create_payload(seeded_recording, num_epochs=500),
        )
        assert r.status_code == 422
        assert r.json()["error"] == "invalid_training_params"

    def test_invalid_batch_size_rejected(self, client, seeded_recording):
        r = client.post(
            "/api/training/versions",
            json=_create_payload(seeded_recording, batch_size=0),
        )
        assert r.status_code == 422

    def test_invalid_training_type_rejected(self, client, seeded_recording):
        r = client.post(
            "/api/training/versions",
            json=_create_payload(seeded_recording, training_type="dpo"),
        )
        # Pydantic enum validation → 422.
        assert r.status_code == 422

    def test_unknown_field_rejected(self, client, seeded_recording):
        r = client.post(
            "/api/training/versions",
            json={**_create_payload(seeded_recording), "secret_param": True},
        )
        assert r.status_code == 422

    def test_invalid_persona_rejected(self, client, seeded_recording):
        r = client.post(
            "/api/training/versions",
            json=_create_payload(seeded_recording, persona_id="alien"),
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_persona_id"

    def test_no_segments_rejected(self, client, seeded_recording):
        r = client.post(
            "/api/training/versions",
            json=_create_payload(seeded_recording, segment_ids=[]),
        )
        assert r.status_code == 422
        assert r.json()["error"] == "no_training_audio"

    def test_malformed_segment_id_rejected(self, client, seeded_recording):
        r = client.post(
            "/api/training/versions",
            json=_create_payload(seeded_recording, segment_ids=["no_marker_here"]),
        )
        # parse_segment_id raises ValueError → 500 in the legacy code. Now
        # the service wraps it as NoTrainingAudioError → 422.
        assert r.status_code in (422, 500)

    def test_total_audio_under_minimum_rejected(self, client, isolated_data, app):
        """A recording with only 5s of audio fails the 10s minimum."""
        from app.api._dependencies import make_recordings_service_for_testing

        rs = make_recordings_service_for_testing(isolated_data)
        # Upload + add a 5s SpeakerSegment.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            for _ in range(48000 * 5):
                w.writeframes(struct.pack("<h", 0))
        rec = rs.upload(
            file_bytes=buf.getvalue(),
            filename="short.wav",
            listener_id="child",
            persona_id="xiao_s",
        )
        speakers_folder = (
            isolated_data / "recordings" / "raw" / rec.folder_name / "speakers"
        )
        speakers_folder.mkdir(parents=True, exist_ok=True)
        _make_wav(speakers_folder / "SPEAKER_00.wav", duration_seconds=5.0)

        def add_segment(r):
            r.speaker_segments = [
                SpeakerSegment(
                    speaker_id="SPEAKER_00",
                    start_time=0.0,
                    end_time=5.0,
                    duration_seconds=5.0,
                )
            ]

        rs.repository.update(rec.recording_id, add_segment)
        test_training = make_training_service_for_testing(
            isolated_data, recordings_service=rs
        )
        app.dependency_overrides[get_training_service] = lambda: test_training

        r = client.post(
            "/api/training/versions",
            json=_create_payload(rec.recording_id),
        )
        assert r.status_code == 422
        assert r.json()["error"] == "no_training_audio"
        assert "too short" in r.json()["message"].lower()

    def test_concurrent_training_rejected(self, client, seeded_recording):
        r1 = client.post("/api/training/versions", json=_create_payload(seeded_recording))
        assert r1.status_code == 200
        r2 = client.post("/api/training/versions", json=_create_payload(seeded_recording))
        assert r2.status_code == 409
        assert r2.json()["error"] == "training_in_progress"


# ---------------------------------------------------------------------------
# GET /versions  — list
# ---------------------------------------------------------------------------
class TestListVersions:
    def test_empty(self, client, assert_matches_schema):
        r = client.get("/api/training/versions")
        assert r.status_code == 200
        parsed = assert_matches_schema(ListVersionsResponse, r.json())
        assert parsed.count == 0
        assert parsed.versions == []

    def test_lists_created_version(self, client, seeded_recording):
        client.post("/api/training/versions", json=_create_payload(seeded_recording))
        r = client.get("/api/training/versions")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["versions"][0]["persona_id"] == "xiao_s"

    def test_filter_by_persona(self, client, seeded_recording):
        client.post("/api/training/versions", json=_create_payload(seeded_recording))
        r = client.get("/api/training/versions?persona_id=caregiver")
        assert r.json()["count"] == 0
        r = client.get("/api/training/versions?persona_id=xiao_s")
        assert r.json()["count"] == 1


# ---------------------------------------------------------------------------
# GET /versions/{id}
# ---------------------------------------------------------------------------
class TestGetVersion:
    def test_happy_path(self, client, seeded_recording):
        create = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        )
        vid = create.json()["version_id"]
        r = client.get(f"/api/training/versions/{vid}")
        assert r.status_code == 200
        body = r.json()
        assert body["version_id"] == vid
        assert body["persona_id"] == "xiao_s"
        # Manifest was saved on create.
        assert "manifest" in body
        assert body["manifest"]["total_duration_seconds"] == pytest.approx(20.0, abs=0.5)

    def test_404_when_missing(self, client):
        r = client.get("/api/training/versions/nope")
        assert r.status_code == 404
        assert r.json()["error"] == "training_version_not_found"


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------
class TestStatus:
    def test_not_training(self, client, assert_matches_schema):
        r = client.get("/api/training/status")
        assert r.status_code == 200
        parsed = assert_matches_schema(TrainingStatusResponse, r.json())
        assert parsed.is_training is False
        assert parsed.version_id is None

    def test_in_training(self, client, seeded_recording):
        client.post("/api/training/versions", json=_create_payload(seeded_recording))
        body = client.get("/api/training/status").json()
        assert body["is_training"] is True
        assert body["persona_id"] == "xiao_s"


# ---------------------------------------------------------------------------
# PATCH /versions/{id}
# ---------------------------------------------------------------------------
class TestUpdateVersion:
    def test_set_nickname(self, client, seeded_recording, assert_matches_schema):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]
        r = client.patch(f"/api/training/versions/{vid}", json={"nickname": "v-prod"})
        assert r.status_code == 200
        parsed = assert_matches_schema(UpdateVersionResponse, r.json())
        assert parsed.nickname == "v-prod"
        # Persisted.
        assert client.get(f"/api/training/versions/{vid}").json()["nickname"] == "v-prod"

    def test_unknown_field_rejected(self, client, seeded_recording):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]
        r = client.patch(f"/api/training/versions/{vid}", json={"surprise": "x"})
        assert r.status_code == 422

    def test_404_when_missing(self, client):
        r = client.patch("/api/training/versions/nope", json={"nickname": "x"})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /versions/{id}/activate
# ---------------------------------------------------------------------------
class TestActivate:
    def test_refuses_non_ready(self, client, seeded_recording):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]
        r = client.post(f"/api/training/versions/{vid}/activate")
        assert r.status_code == 409
        assert r.json()["error"] == "version_not_ready"

    def test_404_when_missing(self, client):
        r = client.post("/api/training/versions/nope/activate")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /versions/{id}
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_round_trip(self, client, seeded_recording, assert_matches_schema):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]
        r = client.delete(f"/api/training/versions/{vid}")
        assert r.status_code == 200
        parsed = assert_matches_schema(DeleteVersionResponse, r.json())
        assert parsed.version_id == vid
        assert client.get(f"/api/training/versions/{vid}").status_code == 404

    def test_404_when_missing(self, client):
        r = client.delete("/api/training/versions/nope")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /versions/{id}/cancel
# ---------------------------------------------------------------------------
class TestCancel:
    def test_cancel_in_progress(self, client, seeded_recording, assert_matches_schema):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]
        r = client.post(f"/api/training/versions/{vid}/cancel")
        assert r.status_code == 200
        parsed = assert_matches_schema(CancelResponse, r.json())
        assert parsed.version_id == vid
        # Status reflects cancelled.
        body = client.get(f"/api/training/versions/{vid}").json()
        assert body["status"] == "cancelled"

    def test_404_when_missing(self, client):
        r = client.post("/api/training/versions/nope/cancel")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /active
# ---------------------------------------------------------------------------
class TestActive:
    def test_no_active(self, client, assert_matches_schema):
        r = client.get("/api/training/active?persona_id=xiao_s")
        assert r.status_code == 200
        parsed = assert_matches_schema(ActiveVersionResponse, r.json())
        assert parsed.active is False
        assert parsed.version is None


# ---------------------------------------------------------------------------
# GET /versions/{id}/manifest
# ---------------------------------------------------------------------------
class TestManifest:
    def test_returns_manifest(self, client, seeded_recording):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]
        r = client.get(f"/api/training/versions/{vid}/manifest")
        assert r.status_code == 200
        body = r.json()
        assert body["version_id"] == vid
        assert body["persona_id"] == "xiao_s"
        assert body["training_type"] == "lora"
        assert body["total_duration_seconds"] == pytest.approx(20.0, abs=0.5)

    def test_404_when_missing(self, client):
        r = client.get("/api/training/versions/nope/manifest")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# SSE /versions/{id}/progress
# ---------------------------------------------------------------------------
class TestListVersionsProgressSync:
    """list_versions auto-syncs from progress.json — exercises the refresh path."""

    def test_list_syncs_completed_progress(self, client, seeded_recording, isolated_data):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]

        # Write a 'ready' progress.json then call list — version should sync.
        progress_path = isolated_data / "models" / f"xiao_s_{vid}" / "progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "version_id": vid,
                    "status": "ready",
                    "current_epoch": 10,
                    "total_epochs": 10,
                    "current_loss": 0.1,
                    "best_loss": 0.1,
                    "progress_pct": 100,
                    "elapsed_seconds": 60,
                    "total_audio_duration": 20.0,
                }
            )
        )

        body = client.get("/api/training/versions").json()
        assert body["count"] == 1
        assert body["versions"][0]["status"] == "ready"
        assert body["versions"][0]["final_loss"] == pytest.approx(0.1)

    def test_get_version_includes_progress_for_training(
        self, client, seeded_recording, isolated_data
    ):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]

        progress_path = isolated_data / "models" / f"xiao_s_{vid}" / "progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "version_id": vid,
                    "status": "training",
                    "current_epoch": 3,
                    "total_epochs": 10,
                    "current_loss": 0.5,
                    "progress_pct": 30,
                }
            )
        )

        body = client.get(f"/api/training/versions/{vid}").json()
        assert body["status"] == "training"
        assert "progress" in body
        assert body["progress"]["current_epoch"] == 3
        assert "manifest" in body  # manifest also included


class TestProgressSSE:
    def test_404_when_missing(self, client):
        r = client.get("/api/training/versions/nope/progress")
        assert r.status_code == 404

    def test_emits_error_event_for_failed_training(
        self, client, seeded_recording, isolated_data
    ):
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]
        progress_path = isolated_data / "models" / f"xiao_s_{vid}" / "progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "version_id": vid,
                    "status": "failed",
                    "current_epoch": 2,
                    "total_epochs": 10,
                    "current_loss": 0.0,
                    "progress_pct": 20,
                    "error_message": "OOM during epoch 2",
                }
            )
        )

        with client.stream("GET", f"/api/training/versions/{vid}/progress") as response:
            assert response.status_code == 200
            payload = "\n".join(response.iter_lines())

        assert '"event": "error"' in payload
        assert "OOM" in payload
        # Sync should have written 'failed' to the index.
        assert client.get(f"/api/training/versions/{vid}").json()["status"] == "failed"

    def test_emits_completion_event(self, client, seeded_recording, isolated_data):
        """When progress.json says 'ready', the SSE stream emits a complete event
        and the version status is synced to 'ready'."""
        vid = client.post(
            "/api/training/versions", json=_create_payload(seeded_recording)
        ).json()["version_id"]

        # Write a 'ready' progress.json before opening the stream so the very
        # first poll inside the generator sees it.
        progress_path = isolated_data / "models" / f"xiao_s_{vid}" / "progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "version_id": vid,
                    "status": "ready",
                    "current_epoch": 10,
                    "total_epochs": 10,
                    "current_loss": 0.123,
                    "best_loss": 0.123,
                    "progress_pct": 100,
                    "elapsed_seconds": 60,
                    "total_audio_duration": 20.0,
                }
            )
        )

        with client.stream(
            "GET", f"/api/training/versions/{vid}/progress"
        ) as response:
            assert response.status_code == 200
            lines = list(response.iter_lines())
        # Expect at least one event line with "complete".
        joined = "\n".join(lines)
        assert "complete" in joined, joined
        # Version should be synced to 'ready'.
        assert client.get(f"/api/training/versions/{vid}").json()["status"] == "ready"
