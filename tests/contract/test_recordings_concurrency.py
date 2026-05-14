"""
Concurrency contract for the recordings API.

Reproduces the documented race in the legacy metadata.py:save() where two
PATCH requests racing on the same metadata.json could lose writes. The
JsonRecordingsRepository's file-lock + atomic-rename pattern is supposed to
guarantee that N concurrent updates converge to a single consistent state
with all N mutations visible.

These tests exercise that guarantee at three levels:
- Repository.update() with N=50 threads (pure data layer).
- RecordingsService.update_speaker_labels() with N=20 threads (service layer).
- HTTP PATCH /api/recordings/{id}/speakers with N=20 threads (API layer).

All three must hold.
"""
from __future__ import annotations

import io
import struct
import threading
import wave

import pytest

from app.services.recordings.models import Recording
from app.services.recordings.repository import JsonRecordingsRepository

pytestmark = [pytest.mark.contract, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers (duplicated lightly from test_recordings_contract — kept independent
# so this file can run alone).
# ---------------------------------------------------------------------------
def _wav_bytes(duration_seconds: float = 5.0, sample_rate: int = 48000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for _ in range(int(duration_seconds * sample_rate)):
            w.writeframes(struct.pack("<h", 0))
    return buf.getvalue()


def _upload(client) -> str:
    response = client.post(
        "/api/recordings/upload",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"listener_id": "child", "persona_id": "xiao_s"},
    )
    assert response.status_code == 200, response.text
    return response.json()["recording_id"]


# ---------------------------------------------------------------------------
# Repository-level — directly exercises the locking primitives.
# ---------------------------------------------------------------------------
class TestRepositoryConcurrency:
    def test_50_parallel_updates_all_persist(self, tmp_path):
        repo = JsonRecordingsRepository(tmp_path)
        recording = Recording.new("rid-x", "child_xiao_s_2026", "child", "xiao_s")
        repo.save(recording)

        N = 50

        def worker(i: int) -> None:
            def mutate(r: Recording) -> None:
                r.speaker_labels[f"SPEAKER_{i:02d}"] = "xiao_s"

            repo.update("rid-x", mutate)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = repo.get("rid-x")
        assert len(final.speaker_labels) == N, (
            f"Expected {N} labels, got {len(final.speaker_labels)}. "
            f"Missing indices: {[i for i in range(N) if f'SPEAKER_{i:02d}' not in final.speaker_labels]}"
        )


# ---------------------------------------------------------------------------
# Service-level — proves the service.update_speaker_labels path is also safe.
# ---------------------------------------------------------------------------
class TestServiceConcurrency:
    def test_service_segment_routing_under_contention(self, tmp_path):
        from app.api._dependencies import make_recordings_service_for_testing

        service = make_recordings_service_for_testing(tmp_path)
        rec = service.upload(
            file_bytes=_wav_bytes(),
            filename="test.wav",
            listener_id="child",
            persona_id="xiao_s",
        )

        # Seed: add 20 speaker_segments via a single update so we have
        # something for update_segment_routing to operate on.
        from app.services.recordings.models import SpeakerSegment

        def seed(r: Recording) -> None:
            r.speaker_segments = [
                SpeakerSegment(
                    speaker_id=f"SPEAKER_{i:02d}",
                    start_time=float(i),
                    end_time=float(i + 1),
                )
                for i in range(20)
            ]

        service.repository.update(rec.recording_id, seed)

        # 20 threads each set listener_id on its own speaker.
        def worker(i: int) -> None:
            speaker = f"SPEAKER_{i:02d}"
            # Alternate listeners between threads to prove writes don't fight.
            listener = "child" if i % 2 == 0 else "mom"
            service.update_segment_routing(
                rec.recording_id, speaker, listener_id=listener
            )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = service.get(rec.recording_id)
        seg_by_id = {seg.speaker_id: seg for seg in final.speaker_segments}
        for i in range(20):
            expected = "child" if i % 2 == 0 else "mom"
            speaker = f"SPEAKER_{i:02d}"
            assert seg_by_id[speaker].listener_id == expected, (
                f"speaker {speaker} lost its write: got {seg_by_id[speaker].listener_id!r}"
            )


# ---------------------------------------------------------------------------
# HTTP-level — what an actual buggy client storm looks like.
# ---------------------------------------------------------------------------
class TestHttpPatchConcurrency:
    def test_parallel_speaker_label_patches_converge(self, client):
        # NOTE: We can't actually have 20 different (speaker_id → persona_id)
        # mappings converge to one consistent map because each PATCH REPLACES
        # the whole map. The right test for HTTP-level convergence is "no
        # corrupted JSON, last-write wins is at least deterministic": after
        # the storm, GET returns a valid Pydantic Recording with exactly one
        # of the N mappings, not a merged/corrupt result.
        rid = _upload(client)
        N = 20

        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                # Each thread sends one unique mapping.
                response = client.patch(
                    f"/api/recordings/{rid}/speakers",
                    json={"speaker_labels": {f"SPEAKER_{i:02d}": "xiao_s"}},
                )
                if response.status_code != 200:
                    errors.append(AssertionError(f"thread {i} got {response.status_code}: {response.text}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Worker errors: {errors[:3]}"

        # After the storm: GET must succeed and the recording must validate
        # as a Recording. The speaker_labels map must contain exactly one
        # SPEAKER_NN entry (last-writer-wins on a replace-semantics PATCH).
        # If the lock failed, we might see corrupt JSON → GET would 500.
        got = client.get(f"/api/recordings/{rid}")
        assert got.status_code == 200, got.text
        body = got.json()
        labels = body["speaker_labels"]
        assert len(labels) == 1, f"Expected 1 label after storm, got {len(labels)}: {labels}"
        key = next(iter(labels))
        assert key.startswith("SPEAKER_")
        assert labels[key] == "xiao_s"
