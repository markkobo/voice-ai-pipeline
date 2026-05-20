"""
Unit tests for JsonRecordingsRepository.

The headline regression: production discovered that some `metadata.json`
files on disk carry tz-naive `created_at` (legacy `datetime.utcnow()` writes)
while others carry tz-aware `created_at` (`datetime.now(timezone.utc)` writes).
Pydantic happily parses both, but the `list()` sort by `created_at` then
explodes with `TypeError: can't compare offset-naive and offset-aware
datetimes`. This file pins the boundary-coercion behavior so the bug stays
fixed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.services.recordings.repository import (
    JsonRecordingsRepository,
    _coerce_naive_datetimes_to_utc,
)


@pytest.fixture
def repo(tmp_path):
    # JsonRecordingsRepository expects a data_root and creates
    # data_root/recordings/{index.json, raw/}. Match the prod layout.
    return JsonRecordingsRepository(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_recording_on_disk(
    repo: JsonRecordingsRepository,
    *,
    recording_id: str,
    folder_name: str,
    created_at_iso: str,
    updated_at_iso: str | None = None,
) -> None:
    """
    Write a Recording metadata.json + index entry directly to disk, bypassing
    the repository's save() so we control the exact ISO string format used
    for `created_at`. This lets us simulate "old naive writes" alongside
    "new aware writes" without monkeypatching datetime.
    """
    folder = repo.raw_root / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "recording_id": recording_id,
        "folder_name": folder_name,
        "listener_id": "child",
        "persona_id": "xiao_s",
        "created_at": created_at_iso,
        "updated_at": updated_at_iso or created_at_iso,
    }
    (folder / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Update index.json (the repository's `_index_set` would do this, but
    # writing it directly keeps the test hermetic.)
    index_path = repo.index_path
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing[recording_id] = folder_name
    index_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Regression: tz-naive + tz-aware mix must not crash list()
# ---------------------------------------------------------------------------
class TestListToleratesMixedDatetimeAwareness:
    """The bug: `results.sort(key=lambda r: r.created_at)` raised when
    metadata.json files mixed tz-naive and tz-aware ISO timestamps."""

    def test_list_with_naive_and_aware_created_at(self, repo):
        # Older recording — naive ISO (no offset suffix).
        _write_recording_on_disk(
            repo,
            recording_id="old-naive",
            folder_name="2026-04-12_old",
            created_at_iso="2026-04-12T03:23:11",
        )
        # Newer recording — tz-aware ISO with +00:00 offset.
        _write_recording_on_disk(
            repo,
            recording_id="new-aware",
            folder_name="2026-05-17_new",
            created_at_iso="2026-05-17T03:23:11+00:00",
        )

        # Before the fix this raised:
        #   TypeError: can't compare offset-naive and offset-aware datetimes
        results = repo.list()

        assert len(results) == 2
        # Sorted newest first, so the aware "2026-05-17" comes before the
        # naive "2026-04-12" (which we coerce to UTC for the purpose of the
        # comparison).
        assert results[0].recording_id == "new-aware"
        assert results[1].recording_id == "old-naive"

        # Both datetimes should now be aware after load.
        for r in results:
            assert r.created_at.tzinfo is not None, (
                f"Recording {r.recording_id} created_at should be tz-aware "
                f"after load, got naive {r.created_at!r}"
            )
            assert r.updated_at.tzinfo is not None

    def test_list_with_z_suffixed_and_naive(self, repo):
        """The other tz-aware shape we see on disk uses the trailing `Z`."""
        _write_recording_on_disk(
            repo,
            recording_id="z-aware",
            folder_name="2026-05-20_z",
            created_at_iso="2026-05-20T02:04:51.163932Z",
        )
        _write_recording_on_disk(
            repo,
            recording_id="naive",
            folder_name="2026-04-14_naive",
            created_at_iso="2026-04-14T15:14:28.636433",
        )

        results = repo.list()
        assert [r.recording_id for r in results] == ["z-aware", "naive"]

    def test_get_also_coerces_to_utc(self, repo):
        """The coercion lives in `_read_metadata_locked`, so single-item
        reads via get() should also return aware datetimes."""
        _write_recording_on_disk(
            repo,
            recording_id="solo",
            folder_name="2026-04-12_solo",
            created_at_iso="2026-04-12T03:23:11",
        )
        rec = repo.get("solo")
        assert rec.created_at.tzinfo is not None
        # Coerced to UTC, same wall-clock time.
        assert rec.created_at == datetime(2026, 4, 12, 3, 23, 11, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper-level unit tests — keeps the walker honest as the model evolves.
# ---------------------------------------------------------------------------
class TestSweepStranded:
    """RecordingsService.sweep_stranded resets recordings stuck in
    `processing` state to `failed` on startup (mirrors the corpus
    sweep at `app/services/corpus/ingestion.py::sweep_stranded`).

    Lives here, not in a separate service test file, because there
    is no test_recordings_service.py yet and these fixtures already
    set up a working repo on disk.
    """

    @pytest.fixture
    def service(self, tmp_path):
        from app.services.recordings.repository import JsonRecordingsRepository
        from app.services.recordings.service import RecordingsService

        class _AlwaysValid:
            def is_valid(self, _id):
                return True

            def list_ids(self):
                return {"xiao_s", "child"}

        repo = JsonRecordingsRepository(tmp_path)
        return RecordingsService(
            repository=repo,
            persona_validator=_AlwaysValid(),
            listener_validator=_AlwaysValid(),
            audio_root=tmp_path / "recordings" / "raw",
        )

    def _plant(self, service, *, recording_id, status, segments=None):
        """Create a recording on disk with a given status (and optional
        speaker_segments to verify they're preserved across the sweep)."""
        from app.services.recordings.models import (
            Recording,
            RecordingStatus,
            SpeakerSegment,
        )

        recording = Recording.new(
            recording_id=recording_id,
            folder_name=f"folder_{recording_id}",
            listener_id="child",
            persona_id="xiao_s",
        )
        recording.status = RecordingStatus(status)
        if segments:
            recording.speaker_segments = [SpeakerSegment(**s) for s in segments]
        service.repository.save(recording)
        return recording

    def test_sweep_resets_stranded_processing_to_failed(self, service):
        self._plant(service, recording_id="rec-stranded", status="processing")
        count = service.sweep_stranded()
        assert count == 1
        rec = service.repository.get("rec-stranded")
        assert rec.status.value == "failed"
        assert "interrupted" in (rec.error_message or "")

    def test_sweep_leaves_other_statuses_alone(self, service):
        self._plant(service, recording_id="rec-raw", status="raw")
        self._plant(service, recording_id="rec-processed", status="processed")
        self._plant(service, recording_id="rec-failed", status="failed")
        count = service.sweep_stranded()
        assert count == 0
        assert service.repository.get("rec-raw").status.value == "raw"
        assert service.repository.get("rec-processed").status.value == "processed"
        assert service.repository.get("rec-failed").status.value == "failed"

    def test_sweep_preserves_partial_speaker_segments(self, service):
        """If processing got partway and wrote speaker_segments before
        crashing, the sweep MUST keep them — only the top-level status
        flips. This is the "don't destroy partial work" guarantee."""
        self._plant(
            service,
            recording_id="rec-partial",
            status="processing",
            segments=[
                {
                    "speaker_id": "SPEAKER_00",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "transcription": "嗨大家好",
                }
            ],
        )
        service.sweep_stranded()
        rec = service.repository.get("rec-partial")
        assert rec.status.value == "failed"
        assert len(rec.speaker_segments) == 1
        assert rec.speaker_segments[0].speaker_id == "SPEAKER_00"
        assert rec.speaker_segments[0].transcription == "嗨大家好"

    def test_sweep_tolerates_concurrent_delete(self, service):
        """If a recording is deleted between list() and update(),
        the sweep must not crash — just skip and continue. Mirrors
        the corpus regression at TestSweepHandlesConcurrentDelete."""
        from unittest.mock import patch
        from app.services.recordings.repository import RecordingNotFound

        self._plant(service, recording_id="rec-a", status="processing")
        self._plant(service, recording_id="rec-b", status="processing")

        real_update = service.repository.update
        calls = {"n": 0}

        def flaky_update(recording_id, mutator):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RecordingNotFound(recording_id)
            return real_update(recording_id, mutator)

        with patch.object(service.repository, "update", side_effect=flaky_update):
            # Should not raise — just skip the "missing" item.
            count = service.sweep_stranded()
        # One survived to be flipped (the other was "concurrently deleted").
        assert count == 1


class TestCoerceNaiveDatetimesToUtc:
    def test_walks_nested_model_fields(self, repo):
        """Nested ProcessingSteps.{step}.started_at / completed_at should
        also be coerced, not just the top-level created_at/updated_at."""
        from app.services.recordings.models import (
            ProcessingStep,
            ProcessingStepStatus,
            Recording,
            RecordingStatus,
        )

        # Build directly via Pydantic (intentionally with naive datetimes).
        naive = datetime(2026, 4, 14, 15, 14, 28)
        recording = Recording(
            recording_id="rec-1",
            folder_name="rec-1",
            listener_id="child",
            persona_id="xiao_s",
            created_at=naive,
            updated_at=naive,
            status=RecordingStatus.processing,
        )
        recording.processing_steps.denoise = ProcessingStep(
            status=ProcessingStepStatus.in_progress,
            started_at=naive,
        )

        _coerce_naive_datetimes_to_utc(recording)

        assert recording.created_at.tzinfo == timezone.utc
        assert recording.updated_at.tzinfo == timezone.utc
        assert recording.processing_steps.denoise.started_at is not None
        assert recording.processing_steps.denoise.started_at.tzinfo == timezone.utc

    def test_no_op_on_already_aware(self):
        from app.services.recordings.models import Recording

        aware = datetime(2026, 5, 17, 3, 23, 11, tzinfo=timezone.utc)
        recording = Recording(
            recording_id="rec-1",
            folder_name="rec-1",
            listener_id="child",
            persona_id="xiao_s",
            created_at=aware,
            updated_at=aware,
        )
        _coerce_naive_datetimes_to_utc(recording)
        assert recording.created_at == aware  # unchanged
