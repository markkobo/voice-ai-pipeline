"""
Realistic integration tests for the recording → parsing → training handoff
pipeline.

These tests deliberately reproduce **on-disk states that the existing test
suite never sees**:

- Mixed tz-naive / tz-aware `created_at` timestamps in `metadata.json`.
- Old-schema metadata files missing fields that newer code added.
- Unknown forward-compatible enum values (e.g. `status="needs_review"`).
- Corrupt / partially-valid metadata files alongside good ones.
- A `Recording` with `speaker_segments[]` populated but no `speakers/` dir.
- Real concurrent uploads through `CorpusService.upload`.

Each test pins a specific past-bug failure mode — see the docstring on each
test for the commit hash where the fix landed.

All tests use ``tmp_path`` via the existing ``isolated_data`` fixture; the
real production data under ``data/recordings/raw/`` is never touched.

Mark this module's slow tests with ``@pytest.mark.integration`` so quick CI
runs can skip them via ``pytest -m "not integration"``.
"""
from __future__ import annotations

import io
import json
import os
import struct
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from app.api._dependencies import (
    get_recordings_service,
    get_training_service,
    make_recordings_service_for_testing,
    make_training_service_for_testing,
)
from app.api._errors import NoTrainingAudioError
from app.services.corpus.models import CorpusItemKind
from app.services.corpus.repository import JsonCorpusRepository
from app.services.corpus.service import CorpusService
from app.services.recordings.models import (
    Recording,
    RecordingStatus,
    SpeakerSegment,
)
from app.services.recordings.repository import (
    CorruptMetadata,
    JsonRecordingsRepository,
)
from app.services.training_service.audio_resolver import (
    RecordingsAudioResolver,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers — build on-disk shapes that bypass the service-layer guards. The
# whole point of these tests is to simulate what's actually on disk in
# production, not what a freshly-uploaded recording looks like.
# ---------------------------------------------------------------------------
def _write_metadata_json(folder: Path, payload: dict) -> None:
    """Persist a metadata.json directly + register in index.json.

    Bypasses the repository entirely: tests need to write old-shape /
    forward-shape JSON that the service-layer constructors wouldn't allow.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _add_to_index(recordings_root: Path, recording_id: str, folder_name: str) -> None:
    """Maintain the index.json that the repository's list() reads."""
    index_path = recordings_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {}
    index[recording_id] = folder_name
    index_path.write_text(json.dumps(index, indent=2))


def _seg_payload(speaker_id: str, *, duration: float = 20.0) -> dict:
    """Minimal speaker_segments[] entry that the resolver can consume."""
    return {
        "speaker_id": speaker_id,
        "start_time": 0.0,
        "end_time": duration,
        "duration_seconds": duration,
        "audio_path": f"speakers/{speaker_id}.wav",
        "persona_id": "xiao_s",
        "listener_id": "child",
    }


def _make_wav(path: Path, duration_seconds: float = 5.0, sample_rate: int = 24000) -> None:
    """Write a silent WAV at the given path. Used to materialize per-speaker
    audio so the resolver finds an actual file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for _ in range(int(duration_seconds * sample_rate)):
            w.writeframes(struct.pack("<h", 0))


def _wav_bytes(duration_seconds: float = 5.0, sample_rate: int = 48000) -> bytes:
    """Stand-alone WAV bytes generator. Mirrors the conftest factory so this
    file can be reasoned about in isolation."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for _ in range(int(duration_seconds * sample_rate)):
            w.writeframes(struct.pack("<h", 0))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pre-populated "production-like" data fixture.
#
# Most tests under `tests/integration/` start from a clean tmp_path. The
# point of these integration tests is the OPPOSITE — to load metadata.json
# files that look like what's been written to disk over many code revisions.
# This fixture seeds a recordings tree containing six metadata.json files,
# each exercising a different on-disk shape:
#
#   1. tz-naive `created_at` (legacy `datetime.utcnow()` writer)
#   2. tz-aware `created_at` (current writer)
#   3. extra unknown fields from a hypothetical future version
#   4. missing fields that were added after the file was written
#   5. processed recording with speaker_segments[] + speakers/*.wav on disk
#   6. processing-state recording (mid-pipeline crash signature)
#
# The fixture's data_root matches `isolated_data`'s — so it composes with
# the FastAPI test client without extra wiring.
# ---------------------------------------------------------------------------
@pytest.fixture
def production_like_recordings(isolated_data: Path):
    """Pre-populate the raw/ tree with metadata.json files exercising real
    on-disk format drift.

    Yields a dict mapping a short label → recording_id so tests can assert
    against specific cases.
    """
    raw_root = isolated_data / "recordings" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    recordings_root = isolated_data / "recordings"

    ids = {}

    # 1. Legacy tz-naive metadata. Pre-8c789de this would crash `list()`
    #    once anything tz-aware was also on disk.
    legacy = {
        "recording_id": "rec_legacy_naive",
        "folder_name": "child_xiao_s_20260101_120000_naive",
        "listener_id": "child",
        "persona_id": "xiao_s",
        "title": "legacy naive",
        "duration_seconds": 12.0,
        "file_size_bytes": 240_000,
        "transcription": {"text": None, "confidence": None, "language": "zh", "segments": []},
        "quality_metrics": {"snr_db": None, "rms_volume": None, "silence_ratio": None,
                            "clarity_score": None, "training_ready": None},
        "status": "raw",
        "processing_steps": {
            "denoise": {"status": "pending", "progress": 0, "error_message": None,
                        "started_at": None, "completed_at": None},
            "enhance": {"status": "pending", "progress": 0, "error_message": None,
                        "started_at": None, "completed_at": None},
            "diarize": {"status": "pending", "progress": 0, "error_message": None,
                        "started_at": None, "completed_at": None},
            "transcribe": {"status": "pending", "progress": 0, "error_message": None,
                           "started_at": None, "completed_at": None},
        },
        "speaker_segments": [],
        "speaker_labels": {},
        "pipeline_metrics": {"denoise_ms": None, "enhance_ms": None, "diarize_ms": None,
                             "transcribe_ms": None, "total_ms": None},
        # NB: naive datetime — no `+00:00`
        "created_at": "2026-01-01T12:00:00",
        "updated_at": "2026-01-01T12:00:00",
        "processed_at": None,
        "processed_expires_at": None,
        "error_message": None,
    }
    _write_metadata_json(raw_root / legacy["folder_name"], legacy)
    _add_to_index(recordings_root, legacy["recording_id"], legacy["folder_name"])
    ids["legacy_naive"] = legacy["recording_id"]

    # 2. Modern tz-aware metadata — what newer code writes.
    modern = dict(legacy)
    modern["recording_id"] = "rec_modern_aware"
    modern["folder_name"] = "child_xiao_s_20260501_120000_aware"
    modern["created_at"] = "2026-05-01T12:00:00+00:00"
    modern["updated_at"] = "2026-05-01T12:00:00+00:00"
    _write_metadata_json(raw_root / modern["folder_name"], modern)
    _add_to_index(recordings_root, modern["recording_id"], modern["folder_name"])
    ids["modern_aware"] = modern["recording_id"]

    # 3. Future-shape metadata: same valid fields, but with extra unknown
    #    top-level keys that a hypothetical newer slice wrote in.
    #    Repository has `extra="forbid"` on Recording, so this is the
    #    interesting case — does load() handle it gracefully? See test
    #    assertions below.
    future_shape = dict(legacy)
    future_shape["recording_id"] = "rec_future_shape"
    future_shape["folder_name"] = "child_xiao_s_20260901_120000_future"
    future_shape["created_at"] = "2026-09-01T12:00:00+00:00"
    future_shape["updated_at"] = "2026-09-01T12:00:00+00:00"
    future_shape["new_unknown_field"] = "hypothetical_2027_value"
    future_shape["another_extra"] = {"nested": "stuff"}
    _write_metadata_json(raw_root / future_shape["folder_name"], future_shape)
    _add_to_index(recordings_root, future_shape["recording_id"], future_shape["folder_name"])
    ids["future_shape"] = future_shape["recording_id"]

    # 4. Old-shape metadata MISSING fields that were added later. We omit
    #    `processed_at`, `processed_expires_at`, `pipeline_metrics`,
    #    `speaker_labels`, `speaker_segments`. Pydantic should default
    #    them via the model's `Field(default_factory=...)`.
    old_shape = {
        "recording_id": "rec_missing_fields",
        "folder_name": "child_xiao_s_20260201_120000_old",
        "listener_id": "child",
        "persona_id": "xiao_s",
        "title": "missing latter fields",
        "duration_seconds": 10.0,
        "file_size_bytes": 200_000,
        "transcription": {"text": None, "confidence": None, "language": "zh", "segments": []},
        "quality_metrics": {"snr_db": None, "rms_volume": None, "silence_ratio": None,
                            "clarity_score": None, "training_ready": None},
        "status": "raw",
        "processing_steps": {
            "denoise": {"status": "pending", "progress": 0, "error_message": None,
                        "started_at": None, "completed_at": None},
            "enhance": {"status": "pending", "progress": 0, "error_message": None,
                        "started_at": None, "completed_at": None},
            "diarize": {"status": "pending", "progress": 0, "error_message": None,
                        "started_at": None, "completed_at": None},
            "transcribe": {"status": "pending", "progress": 0, "error_message": None,
                           "started_at": None, "completed_at": None},
        },
        # naive: yet another mixed-format case in the same set
        "created_at": "2026-02-01T12:00:00",
        "updated_at": "2026-02-01T12:00:00",
    }
    _write_metadata_json(raw_root / old_shape["folder_name"], old_shape)
    _add_to_index(recordings_root, old_shape["recording_id"], old_shape["folder_name"])
    ids["missing_fields"] = old_shape["recording_id"]

    # 5. A *processed* recording with a real speakers/SPEAKER_00.wav on
    #    disk — the only shape that survives end-to-end training input.
    processed = dict(legacy)
    processed["recording_id"] = "rec_fully_processed"
    processed["folder_name"] = "child_xiao_s_20260401_120000_done"
    processed["status"] = "processed"
    processed["created_at"] = "2026-04-01T12:00:00+00:00"
    processed["updated_at"] = "2026-04-01T12:00:00+00:00"
    processed["processed_at"] = "2026-04-01T12:01:00+00:00"
    processed["speaker_segments"] = [_seg_payload("SPEAKER_00", duration=20.0)]
    processed["speaker_labels"] = {"SPEAKER_00": "xiao_s"}
    _write_metadata_json(raw_root / processed["folder_name"], processed)
    _add_to_index(recordings_root, processed["recording_id"], processed["folder_name"])
    _make_wav(
        raw_root / processed["folder_name"] / "speakers" / "SPEAKER_00.wav",
        duration_seconds=20.0,
    )
    ids["processed"] = processed["recording_id"]

    # 6. Mid-pipeline crash: status=processing, no speakers/ dir.
    in_progress = dict(legacy)
    in_progress["recording_id"] = "rec_stuck_processing"
    in_progress["folder_name"] = "child_xiao_s_20260301_120000_stuck"
    in_progress["status"] = "processing"
    in_progress["created_at"] = "2026-03-01T12:00:00+00:00"
    in_progress["updated_at"] = "2026-03-01T12:00:00+00:00"
    _write_metadata_json(raw_root / in_progress["folder_name"], in_progress)
    _add_to_index(recordings_root, in_progress["recording_id"], in_progress["folder_name"])
    ids["stuck_processing"] = in_progress["recording_id"]

    return {"data_root": isolated_data, "ids": ids}


# ===========================================================================
# 1. Recordings list — tolerate on-disk format drift
# ===========================================================================
class TestRecordingsListToleratesDiskFormatDrift:
    """Pins commit 8c789de from multiple directions simultaneously.

    The fix coerces tz-naive datetimes to UTC at the load boundary. But the
    REAL test isn't "one naive + one aware" (which the existing unit test
    covers) — it's "what happens when six different shapes coexist in
    raw/ AND the same list() call has to load and sort them all".
    """

    def test_list_returns_all_supported_shapes(
        self, isolated_data, production_like_recordings
    ):
        """Load six metadata.json files spanning legacy / modern / future
        shapes — the five supported ones (legacy naive, modern aware,
        missing-fields, processed, stuck-processing) must materialize.

        The future-shape (extra top-level fields) is intentionally
        REJECTED by Recording's ``extra="forbid"`` posture (per
        _multi_commit_review_followups.md #19 — the recording side picks
        strict validation for the on-disk shape). That row should be
        skipped with a warning, NOT crash list().
        """
        repo = JsonRecordingsRepository(isolated_data)
        results = repo.list()

        ids = production_like_recordings["ids"]
        loaded_ids = {r.recording_id for r in results}
        assert ids["legacy_naive"] in loaded_ids
        assert ids["modern_aware"] in loaded_ids
        assert ids["missing_fields"] in loaded_ids, (
            "Optional fields missing should default via Pydantic factories."
        )
        assert ids["processed"] in loaded_ids
        assert ids["stuck_processing"] in loaded_ids
        # Future-shape with unknown top-level fields → forbidden by the
        # Recording model. Confirmed visible behavior.
        assert ids["future_shape"] not in loaded_ids

    def test_list_sort_order_is_newest_first_across_mixed_tz(
        self, isolated_data, production_like_recordings
    ):
        """The bug 8c789de fixed surfaced through sort(). Pin the
        sort order across naive + aware to make sure both sides of
        the coercion are working: 2026-09 > 2026-05 > 2026-04 > 2026-03
        > 2026-02 > 2026-01 — and the sort must complete without raising
        TypeError."""
        repo = JsonRecordingsRepository(isolated_data)
        results = repo.list()
        order = [r.created_at for r in results]
        # All should be tz-aware after coercion.
        assert all(d.tzinfo is not None for d in order), (
            f"Expected all tz-aware after coercion; got tz info: "
            f"{[d.tzinfo for d in order]}"
        )
        # Reverse-chronological order — newest first.
        assert order == sorted(order, reverse=True), (
            f"list() did not return newest-first ordering: {order}"
        )

    def test_list_via_api_does_not_500(
        self, client, isolated_data, production_like_recordings
    ):
        """End-to-end: GET /api/recordings/ on a mixed-format disk must
        succeed. This is the actual user-facing regression for 8c789de.

        Five of the six seeded shapes load cleanly (the sixth is rejected
        by the Pydantic ``extra="forbid"`` posture on Recording — verified
        in test_list_returns_all_supported_shapes above)."""
        response = client.get("/api/recordings/")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["total"] == 5
        assert len(data["recordings"]) == 5


# ===========================================================================
# 2. Handoff: recordings → training audio resolver
# ===========================================================================
class TestRecordingsToAudioResolverHandoff:
    """Pins audit handoff C — the recording.folder_name + speaker_id ↔
    speakers/{speaker_id}.wav contract between pipeline.py and the
    training audio resolver. Easy to drift; subtle to debug."""

    def test_resolver_finds_speaker_wav_for_processed_recording(
        self, isolated_data, production_like_recordings
    ):
        """Happy path: processed recording with speakers/SPEAKER_00.wav
        on disk resolves cleanly to a ResolvedSegment of the right
        duration."""
        rec_id = production_like_recordings["ids"]["processed"]

        rs = make_recordings_service_for_testing(isolated_data)
        resolver = RecordingsAudioResolver(
            recordings_service=rs,
            audio_root=isolated_data / "recordings" / "raw",
        )
        resolved = resolver.resolve_segments([f"{rec_id}_SPEAKER_00"])
        assert len(resolved) == 1
        seg = resolved[0]
        assert seg.recording_id == rec_id
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.audio_path.exists()
        assert seg.audio_path.name == "SPEAKER_00.wav"
        assert seg.duration_seconds == pytest.approx(20.0)

    def test_resolver_raises_NoTrainingAudioError_when_speakers_dir_missing(
        self, isolated_data, production_like_recordings
    ):
        """A recording stuck in `processing` has speaker_segments[] empty
        AND no speakers/ dir. Resolver must refuse cleanly, NOT crash."""
        # `rec_stuck_processing` has no speaker_segments either, so we
        # need to make the segment id appear plausible but the file
        # absent. We achieve that by hand-crafting a recording with a
        # speaker_segments[] entry but no speakers/ dir.
        raw_root = isolated_data / "recordings" / "raw"
        folder = raw_root / "child_xiao_s_phantom"
        meta = {
            "recording_id": "rec_phantom",
            "folder_name": folder.name,
            "listener_id": "child",
            "persona_id": "xiao_s",
            "duration_seconds": 30.0,
            "file_size_bytes": 600_000,
            "speaker_segments": [_seg_payload("SPEAKER_00", duration=20.0)],
            "speaker_labels": {},
            "status": "processed",
            "created_at": "2026-04-01T12:00:00+00:00",
            "updated_at": "2026-04-01T12:00:00+00:00",
        }
        _write_metadata_json(folder, meta)
        _add_to_index(isolated_data / "recordings", "rec_phantom", folder.name)
        # NB: no folder/speakers/ created.

        rs = make_recordings_service_for_testing(isolated_data)
        resolver = RecordingsAudioResolver(
            recordings_service=rs,
            audio_root=raw_root,
        )
        with pytest.raises(NoTrainingAudioError) as exc:
            resolver.resolve_segments(["rec_phantom_SPEAKER_00"])
        assert "Speaker audio missing" in str(exc.value)

    def test_resolver_sums_duration_across_multi_turn_speaker(
        self, isolated_data, production_like_recordings
    ):
        """A multi-turn speaker (e.g. podcast host with 50+ alternating
        turns) must report the SUM of all segment durations, not just
        the first. Previously the loop `break`ed after the first match,
        which on a 18-minute recording returned 4.4s and tripped the
        trainer's `Total audio duration too short` guard (user-reported
        2026-05-21)."""
        raw_root = isolated_data / "recordings" / "raw"
        folder = raw_root / "child_test_multiturn"
        # 5 turns × 200s each = 1000s for SPEAKER_00; 3 turns × 1s for SPEAKER_01.
        turns = (
            [_seg_payload("SPEAKER_00", duration=200.0)] * 5
            + [_seg_payload("SPEAKER_01", duration=1.0)] * 3
        )
        meta = {
            "recording_id": "rec_multiturn",
            "folder_name": folder.name,
            "listener_id": "child",
            "persona_id": "test",
            "duration_seconds": 1003.0,
            "file_size_bytes": 100_000_000,
            "speaker_segments": turns,
            "speaker_labels": {},
            "status": "processed",
            "created_at": "2026-05-21T02:03:34+00:00",
            "updated_at": "2026-05-21T02:51:00+00:00",
        }
        _write_metadata_json(folder, meta)
        _add_to_index(isolated_data / "recordings", "rec_multiturn", folder.name)
        _make_wav(folder / "speakers" / "SPEAKER_00.wav", duration_seconds=1000.0)
        _make_wav(folder / "speakers" / "SPEAKER_01.wav", duration_seconds=3.0)

        rs = make_recordings_service_for_testing(isolated_data)
        resolver = RecordingsAudioResolver(
            recordings_service=rs,
            audio_root=raw_root,
        )
        resolved = resolver.resolve_segments(["rec_multiturn_SPEAKER_00"])
        assert len(resolved) == 1
        assert resolved[0].duration_seconds == pytest.approx(1000.0)

    def test_resolver_raises_when_speaker_wav_missing_but_dir_exists(
        self, isolated_data, production_like_recordings
    ):
        """Partial state: speakers/ exists but the specific
        SPEAKER_00.wav inside is missing (e.g. user deleted one)."""
        raw_root = isolated_data / "recordings" / "raw"
        folder = raw_root / "child_xiao_s_partial"
        meta = {
            "recording_id": "rec_partial",
            "folder_name": folder.name,
            "listener_id": "child",
            "persona_id": "xiao_s",
            "duration_seconds": 30.0,
            "file_size_bytes": 600_000,
            "speaker_segments": [
                _seg_payload("SPEAKER_00", duration=20.0),
                _seg_payload("SPEAKER_01", duration=15.0),
            ],
            "speaker_labels": {},
            "status": "processed",
            "created_at": "2026-04-01T12:00:00+00:00",
            "updated_at": "2026-04-01T12:00:00+00:00",
        }
        _write_metadata_json(folder, meta)
        _add_to_index(isolated_data / "recordings", "rec_partial", folder.name)
        # Only SPEAKER_01 has audio.
        _make_wav(folder / "speakers" / "SPEAKER_01.wav", duration_seconds=15.0)

        rs = make_recordings_service_for_testing(isolated_data)
        resolver = RecordingsAudioResolver(
            recordings_service=rs,
            audio_root=raw_root,
        )
        with pytest.raises(NoTrainingAudioError):
            resolver.resolve_segments(["rec_partial_SPEAKER_00"])

        # SPEAKER_01 should still resolve.
        resolved = resolver.resolve_segments(["rec_partial_SPEAKER_01"])
        assert len(resolved) == 1
        assert resolved[0].audio_path.name == "SPEAKER_01.wav"


# ===========================================================================
# 3. Training activation — persona_id underscore brittleness
# ===========================================================================
class TestTrainingVersionActivateWithPersonaIdUnderscores:
    """Pins commit 2289f4f: the legacy ``parts[:3]`` slice broke for
    persona_ids with 0 or 3+ underscores.

    With ``merged_path`` stored explicitly, activate_version succeeds
    regardless of the underscore count. We test the four arities the
    fix's commit message specifically called out: 0, 1, 2, 3 underscores
    in persona_id."""

    @pytest.mark.parametrize(
        "persona_id",
        [
            "foo",              # 0 underscores
            "foo_bar",          # 1 underscore
            "foo_bar_baz",      # 2 (the assumption parts[:3] encoded)
            "foo_bar_baz_qux",  # 3 (broke before 2289f4f)
        ],
    )
    def test_activate_with_stored_merged_path_resolves_for_any_arity(
        self, tmp_path, persona_id
    ):
        """With merged_path stored on the version, activation finds the
        merged-model directory regardless of underscore count."""
        from app.services.training import TrainingVersion
        from app.services.tts.qwen_tts_engine import FasterQwenTTSEngine

        version_id = "v1_20260520_120000"
        lora_dir = tmp_path / f"{persona_id}_{version_id}"
        lora_dir.mkdir(parents=True)
        # The merged dir name encodes the FULL persona_id — that's the
        # whole point of storing merged_path explicitly.
        merged_dir = tmp_path / f"merged_qwen3_tts_{persona_id}_v1"
        merged_dir.mkdir(parents=True)

        version = TrainingVersion(
            version_id=version_id,
            persona_id=persona_id,
            status="ready",
            lora_path=str(lora_dir),
            merged_path=str(merged_dir),
        )

        with patch("torch.cuda.is_available", return_value=False):
            engine = FasterQwenTTSEngine(model_size="1.7B", device="cpu")
        with patch("app.services.training.get_version_manager") as mock_mgr:
            mock_mgr.return_value.get_version.return_value = version
            engine.activate_version(version_id)

        assert engine._merged_model_path == str(merged_dir.resolve()), (
            f"persona_id={persona_id!r}: expected activation to use "
            f"stored merged_path {merged_dir}, but engine has "
            f"{engine._merged_model_path!r}"
        )


# ===========================================================================
# 4. Corpus item skips unknown enum
# ===========================================================================
class TestCorpusItemSkipsUnknownStatusEnum:
    """Pins the LIMITATION documented in corpus/models.py:59-67 —
    ``extra="ignore"`` covers new FIELDS only, not new ENUM VALUES.

    When old code parses metadata.json with status="needs_review", Pydantic
    raises ValidationError → CorruptCorpusMetadata → list() logs + skips.
    Net effect: items with unknown statuses become invisible, never crash
    the list."""

    def test_list_skips_unknown_enum_value_without_crashing(
        self, isolated_data, caplog
    ):
        import logging

        personas_root = isolated_data / "personas"
        repo = JsonCorpusRepository(personas_root)
        corpus_root = repo.corpus_root("xiao_s")
        corpus_root.mkdir(parents=True, exist_ok=True)

        # Write one valid item.
        good_dir = corpus_root / "text" / "11111111-2222-3333-4444-555555555555"
        good_dir.mkdir(parents=True)
        good_payload = {
            "item_id": "11111111-2222-3333-4444-555555555555",
            "persona_id": "xiao_s",
            "kind": "text",
            "filename": "ok.txt",
            "size_bytes": 10,
            "status": "uploaded",
            "created_at": "2026-05-01T12:00:00+00:00",
            "updated_at": "2026-05-01T12:00:00+00:00",
        }
        (good_dir / "metadata.json").write_text(json.dumps(good_payload))

        # Write one item with an unknown enum value status. This is the
        # forward-shape that the corpus models.py comment explicitly
        # documents as the limit of extra="ignore".
        bad_dir = corpus_root / "text" / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        bad_dir.mkdir(parents=True)
        bad_payload = dict(good_payload)
        bad_payload["item_id"] = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        bad_payload["filename"] = "from_future.txt"
        bad_payload["status"] = "needs_review"  # hypothetical future value
        (bad_dir / "metadata.json").write_text(json.dumps(bad_payload))

        # Both registered in index.json.
        index = {
            good_payload["item_id"]: f"text/{good_payload['item_id']}",
            bad_payload["item_id"]: f"text/{bad_payload['item_id']}",
        }
        (corpus_root / "index.json").write_text(json.dumps(index))

        with caplog.at_level(logging.WARNING):
            items = repo.list("xiao_s")

        # The valid one should survive.
        assert len(items) == 1
        assert items[0].item_id == good_payload["item_id"]

        # The skip should be visible in the warning log.
        skipped = [
            r for r in caplog.records
            if "Skipping corrupt item" in r.getMessage()
            and bad_payload["item_id"] in r.getMessage()
        ]
        assert skipped, (
            f"Expected a 'Skipping corrupt item' warning for the "
            f"unknown-enum item; got log records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )


# ===========================================================================
# 5. End-to-end lifecycle — upload → process → resolve
# ===========================================================================
class TestFullUploadToResolveLifecycle:
    """End-to-end: upload through the API, run a mocked pipeline that
    writes a deterministic speakers/SPEAKER_00.wav, then verify the
    training audio resolver can pick it up.

    This catches the brittle naming convention between
    `pipeline._extract_speakers` (writes to
    ``{folder}/speakers/{speaker_id}.wav``) and
    `RecordingsAudioResolver.resolve_segments` (reads from
    ``{folder}/speakers/{speaker_id}.wav``). They must agree."""

    def test_upload_then_mocked_process_then_resolve(
        self, client, isolated_data, app
    ):
        # 1. Upload a real 5-sec WAV via the API.
        resp = client.post(
            "/api/recordings/upload",
            files={"file": ("seed.wav", _wav_bytes(duration_seconds=5.0), "audio/wav")},
            data={"listener_id": "child", "persona_id": "xiao_s"},
        )
        assert resp.status_code == 200, resp.text
        rec_id = resp.json()["recording_id"]
        folder_name = resp.json()["folder_name"]

        # 2. Mock the pipeline. We don't actually run pyannote / whisper —
        #    we write the on-disk side-effects the resolver depends on:
        #      - speakers/SPEAKER_00.wav
        #      - speaker_segments[] on the recording metadata
        #      - status=processed
        raw_root = isolated_data / "recordings" / "raw"
        speakers_folder = raw_root / folder_name / "speakers"
        speakers_folder.mkdir(parents=True, exist_ok=True)
        _make_wav(speakers_folder / "SPEAKER_00.wav", duration_seconds=4.5)

        # Mutate the recording directly through the test service (same
        # data_root as the FastAPI app sees, via isolated_data fixture).
        rs = make_recordings_service_for_testing(isolated_data)

        def mutate(rec: Recording) -> None:
            rec.status = RecordingStatus.processed
            rec.speaker_segments = [
                SpeakerSegment(
                    speaker_id="SPEAKER_00",
                    start_time=0.0,
                    end_time=4.5,
                    duration_seconds=4.5,
                    audio_path=str(speakers_folder / "SPEAKER_00.wav"),
                    persona_id="xiao_s",
                    listener_id="child",
                ),
            ]
            rec.speaker_labels = {"SPEAKER_00": "xiao_s"}

        rs.repository.update(rec_id, mutate)

        # 3. GET /api/recordings/{id} must reflect processed status.
        resp = client.get(f"/api/recordings/{rec_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "processed"
        assert len(body["speaker_segments"]) == 1
        assert body["speaker_segments"][0]["speaker_id"] == "SPEAKER_00"

        # 4. Resolver picks up the segment_id and finds the wav.
        resolver = RecordingsAudioResolver(
            recordings_service=rs,
            audio_root=raw_root,
        )
        resolved = resolver.resolve_segments([f"{rec_id}_SPEAKER_00"])
        assert len(resolved) == 1
        assert resolved[0].audio_path == speakers_folder / "SPEAKER_00.wav"
        assert resolved[0].audio_path.exists()
        assert resolved[0].duration_seconds == pytest.approx(4.5)


# ===========================================================================
# 6. Training refuses to start with no speakers dir
# ===========================================================================
class TestTrainingWithNoSpeakersDir:
    """A recording that was uploaded but never processed has no
    speakers/ dir. The training service must refuse via
    `NoTrainingAudioError` at create-time, NOT crash later inside the
    subprocess.

    This is the pre-flight check that saves a 5-hour SFT run from
    discovering its training data was an empty directory."""

    def test_create_version_with_unprocessed_recording_refuses(
        self, client, isolated_data, wav_bytes, app
    ):
        # Upload a fresh recording but DON'T process it.
        resp = client.post(
            "/api/recordings/upload",
            files={"file": ("raw.wav", wav_bytes(duration_seconds=5.0), "audio/wav")},
            data={"listener_id": "child", "persona_id": "xiao_s"},
        )
        assert resp.status_code == 200
        rec_id = resp.json()["recording_id"]

        # Try to start training pointed at a segment that doesn't exist
        # on disk (no speakers/ folder, no speaker_segments[] entry).
        resp = client.post(
            "/api/training/versions",
            json={
                "persona_id": "xiao_s",
                "segment_ids": [f"{rec_id}_SPEAKER_00"],
                "rank": 16,
                "num_epochs": 10,
                "batch_size": 4,
                "training_type": "lora",
            },
        )
        # Domain error → 4xx, not 500.
        assert 400 <= resp.status_code < 500, (
            f"Expected a 4xx refusal, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body["error"] == "no_training_audio", body


# ===========================================================================
# 7. Corrupt metadata doesn't take down list
# ===========================================================================
class TestCorruptMetadataDoesntTakeDownList:
    """Pin the resilience of JsonRecordingsRepository.list() against
    individual bad metadata.json files.

    Mirrors the corpus repo's behavior (test 4). The recordings repo
    already converts these to CorruptMetadata and skips them — we just
    need to PROVE it for multiple failure shapes at once."""

    def test_list_returns_valid_skips_bad(self, isolated_data, caplog):
        import logging

        recordings_root = isolated_data / "recordings"
        raw_root = recordings_root / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)

        # Three good ones.
        good_ids = []
        for i in range(3):
            folder = raw_root / f"child_xiao_s_2026050{i+1}_good"
            payload = {
                "recording_id": f"rec_good_{i}",
                "folder_name": folder.name,
                "listener_id": "child",
                "persona_id": "xiao_s",
                "duration_seconds": 5.0,
                "file_size_bytes": 100_000,
                "created_at": f"2026-05-0{i+1}T12:00:00+00:00",
                "updated_at": f"2026-05-0{i+1}T12:00:00+00:00",
            }
            _write_metadata_json(folder, payload)
            _add_to_index(recordings_root, payload["recording_id"], folder.name)
            good_ids.append(payload["recording_id"])

        # One: schema violation (Pydantic ValidationError → CorruptMetadata).
        bad_folder = raw_root / "child_xiao_s_20260601_bad_schema"
        bad_folder.mkdir(parents=True)
        (bad_folder / "metadata.json").write_text(
            json.dumps({"corrupt": "missing required fields"})
        )
        _add_to_index(recordings_root, "rec_bad_schema", bad_folder.name)

        # One: not JSON at all (JSONDecodeError → CorruptMetadata).
        bin_folder = raw_root / "child_xiao_s_20260602_bin"
        bin_folder.mkdir(parents=True)
        (bin_folder / "metadata.json").write_text("<not json at all>")
        _add_to_index(recordings_root, "rec_bad_json", bin_folder.name)

        repo = JsonRecordingsRepository(isolated_data)
        with caplog.at_level(logging.WARNING):
            results = repo.list()

        loaded_ids = {r.recording_id for r in results}
        for rid in good_ids:
            assert rid in loaded_ids, (
                f"Valid recording {rid} should have loaded; got {loaded_ids}"
            )
        assert "rec_bad_schema" not in loaded_ids
        assert "rec_bad_json" not in loaded_ids

        # Each bad one should be visible as a warning.
        warnings = [r.getMessage() for r in caplog.records
                    if "Skipping corrupt" in r.getMessage()]
        assert any("rec_bad_schema" in w for w in warnings), warnings
        assert any("rec_bad_json" in w for w in warnings), warnings


# ===========================================================================
# 8. Concurrent corpus upload dedup — TOCTOU race
# ===========================================================================
class TestConcurrentUploadDedupHoldsUnderLoad:
    """Pins slice 2B BLOCKER #3: ``CorpusService.upload`` previously had a
    TOCTOU race where two concurrent uploads of the same content both
    saw "no existing hash" and both created new items.

    The fix (per c7ee1f4 review #3) wraps the dedup-scan + save in a
    per-persona upload lock. This test fires 5 concurrent uploads of
    IDENTICAL bytes and asserts exactly one corpus item ends up on disk.
    """

    def test_five_concurrent_identical_uploads_produce_one_item(
        self, isolated_data
    ):
        personas_root = isolated_data / "personas"
        repo = JsonCorpusRepository(personas_root)
        service = CorpusService(repo)

        # Identical bytes ⇒ identical SHA256 ⇒ all 5 should dedup-hit.
        content = b"shared text body that is the same across every uploader\n"

        errors = []

        def do_upload(idx: int):
            try:
                return service.upload(
                    persona_id="xiao_s",
                    kind=CorpusItemKind.text,
                    file_bytes=content,
                    filename=f"upload_{idx}.txt",
                )
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                raise

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(do_upload, i) for i in range(5)]
            results = [f.result() for f in as_completed(futures)]

        assert not errors, f"Concurrent uploads raised: {errors}"

        # All 5 calls should have returned the SAME item_id.
        item_ids = {r.item_id for r in results}
        assert len(item_ids) == 1, (
            f"Expected exactly one item_id across 5 concurrent identical "
            f"uploads; got {len(item_ids)}: {item_ids}"
        )

        # On-disk: exactly one corpus item directory.
        items = repo.list("xiao_s")
        assert len(items) == 1, (
            f"Expected exactly 1 item on disk after dedup, got {len(items)}: "
            f"{[i.item_id for i in items]}"
        )


# ===========================================================================
# 9. Quality metrics backfill path
# ===========================================================================
class TestRecordingQualityMetricsBackfillPath:
    """Per _phase2_followups.md §2: the "quality metrics backfill" fix.

    The legacy `metadata.py::update_quality_metrics` has an
    "only-update-existing-keys" guard:

        for key, value in metrics.items():
            if key in self._data["quality_metrics"]:
                self._data["quality_metrics"][key] = value

    If `quality_metrics` is ``None`` on disk (early-bug shape), the dict
    lookup fails and ALL values are silently dropped.

    The Phase 2 fix wrote a backfill that uses the model-level
    `apply_quality_metrics` instead of metadata.py. This test pins THAT
    code path — the proper Pydantic-model-driven backfill — so a future
    refactor doesn't accidentally re-introduce the silent-drop behavior.
    """

    def test_apply_quality_metrics_fills_all_fields_on_null_initial_state(
        self, isolated_data
    ):
        # Build a Recording whose quality_metrics has all None fields
        # — mirrors the post-reset state the phase-2 followups describe.
        repo = JsonRecordingsRepository(isolated_data)
        rec = Recording.new(
            recording_id="rec_qm_test",
            folder_name="child_xiao_s_qm_test",
            listener_id="child",
            persona_id="xiao_s",
        )
        # Sanity: defaults to all-None.
        assert rec.quality_metrics.snr_db is None
        assert rec.quality_metrics.clarity_score is None
        assert rec.quality_metrics.training_ready is None
        repo.save(rec)

        # Run the equivalent of the backfill: use apply_quality_metrics
        # to compute SNR / clarity / silence / RMS, persist via the
        # repository's atomic update.
        def mutate(r: Recording) -> None:
            r.apply_quality_metrics(
                snr_db=20.0,
                rms_volume=-12.5,
                silence_ratio=0.08,
                clarity_score=0.8,
            )

        repo.update("rec_qm_test", mutate)

        # All four fields landed AND training_ready was derived.
        loaded = repo.get("rec_qm_test")
        assert loaded.quality_metrics.snr_db == 20.0
        assert loaded.quality_metrics.rms_volume == -12.5
        assert loaded.quality_metrics.silence_ratio == 0.08
        assert loaded.quality_metrics.clarity_score == 0.8
        assert loaded.quality_metrics.training_ready is True, (
            "snr_db=20 > 15 AND clarity=0.8 > 0.6 → training_ready should be True"
        )

    def test_apply_quality_metrics_on_disk_round_trip(self, isolated_data):
        """Round-trip: write to disk, re-read, verify all 5 derived
        quality fields persisted. The phase-2 backfill bug specifically
        manifested as values landing in memory but not on disk — this
        forces a re-read to catch that."""
        repo = JsonRecordingsRepository(isolated_data)
        rec = Recording.new(
            recording_id="rec_qm_disk",
            folder_name="child_xiao_s_qm_disk",
            listener_id="child",
            persona_id="xiao_s",
        )
        repo.save(rec)

        def mutate(r: Recording) -> None:
            r.apply_quality_metrics(
                snr_db=12.0,
                rms_volume=-20.0,
                silence_ratio=0.4,
                clarity_score=0.4,
            )

        repo.update("rec_qm_disk", mutate)

        # Force a fresh load by going through a fresh repo instance.
        repo2 = JsonRecordingsRepository(isolated_data)
        loaded = repo2.get("rec_qm_disk")
        assert loaded.quality_metrics.snr_db == 12.0
        assert loaded.quality_metrics.rms_volume == -20.0
        assert loaded.quality_metrics.silence_ratio == 0.4
        assert loaded.quality_metrics.clarity_score == 0.4
        # snr=12 < 15 OR clarity=0.4 < 0.6 → training_ready should be False.
        assert loaded.quality_metrics.training_ready is False
