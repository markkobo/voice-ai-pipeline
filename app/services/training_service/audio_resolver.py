"""
Resolve segment_ids → audio paths for training.

Decouples TrainingService from the recordings module: the service holds an
AudioResolver protocol; the API layer wires a concrete impl backed by
RecordingsService.

Critical contract: an AudioResolver MUST refuse to return a segment whose
duration is unknown. The old `get_training_audio_for_persona` silently used
30.0s as a fallback, which let recordings of unknown duration pass the
10-second minimum check (audit defect #7).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import parse_segment_id

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSegment:
    recording_id: str
    speaker_id: str
    audio_path: Path
    duration_seconds: float
    persona_id: str
    listener_id: str


class AudioResolver(Protocol):
    def resolve_segments(self, segment_ids: list[str]) -> list[ResolvedSegment]: ...


class RecordingsAudioResolver:
    """
    Resolve segments through `RecordingsService`.

    The recording's `folder_name` plus the speaker_id locate the per-speaker
    wav file written by the diarization step
    (`{folder}/speakers/{speaker_id}.wav`). Duration comes from the matching
    `SpeakerSegment.duration_seconds` on the recording; if absent, the
    resolver raises rather than silently lying about training data.
    """

    def __init__(self, recordings_service, audio_root: Path) -> None:
        self._rs = recordings_service
        self.audio_root = Path(audio_root)

    def resolve_segments(self, segment_ids: list[str]) -> list[ResolvedSegment]:
        from app.api._errors import NoTrainingAudioError, RecordingNotFoundError

        resolved: list[ResolvedSegment] = []
        seen_recordings: dict[str, object] = {}  # recording_id → Recording (cache)

        for seg_id in segment_ids:
            try:
                recording_id, speaker_id = parse_segment_id(seg_id)
            except ValueError as e:
                # Wrap parser errors so the API gets a 422 instead of a 500.
                from app.api._errors import NoTrainingAudioError as _NTAE
                raise _NTAE(
                    str(e),
                    details={"segment_id": seg_id},
                ) from e
            recording = seen_recordings.get(recording_id)
            if recording is None:
                try:
                    recording = self._rs.get(recording_id)
                except RecordingNotFoundError as e:
                    raise NoTrainingAudioError(
                        f"Recording not found for segment {seg_id!r}",
                        details={"segment_id": seg_id, "recording_id": recording_id},
                    ) from e
                seen_recordings[recording_id] = recording

            # Speaker audio path (Phase 1.1 service uses self.audio_root for the raw layout).
            audio_path = self.audio_root / recording.folder_name / "speakers" / f"{speaker_id}.wav"
            if not audio_path.exists():
                raise NoTrainingAudioError(
                    f"Speaker audio missing: {audio_path}",
                    details={"segment_id": seg_id, "expected_path": str(audio_path)},
                )

            # Sum the duration across ALL segments for this speaker.
            # The on-disk `speakers/{speaker_id}.wav` is the concatenation
            # of every turn this speaker took (pipeline.py:705-709), so
            # the training-relevant duration is the SUM of all
            # SpeakerSegment.duration_seconds with matching speaker_id —
            # not just the first one. Previously the loop `break`ed
            # after the first match, which on a 18-minute podcast with
            # 190 alternating turns reported only the first 4.4s and
            # tripped the trainer's "Total audio duration too short:
            # 4.4s (minimum 10.0s)" guard (user-reported 2026-05-21).
            durations = [
                seg.duration_seconds
                for seg in recording.speaker_segments
                if seg.speaker_id == speaker_id
                and seg.duration_seconds is not None
            ]
            if not durations:
                raise NoTrainingAudioError(
                    f"No segments with known duration for speaker "
                    f"{speaker_id!r} in recording {recording_id!r}; refuse "
                    "to train on data with unknown duration (old code "
                    "silently defaulted to 30s)",
                    details={"segment_id": seg_id},
                )
            duration = sum(durations)

            resolved.append(
                ResolvedSegment(
                    recording_id=recording_id,
                    speaker_id=speaker_id,
                    audio_path=audio_path,
                    duration_seconds=duration,
                    persona_id=recording.persona_id,
                    listener_id=recording.listener_id,
                )
            )
            log.debug(
                "Resolved %s → %s (%.1fs)", seg_id, audio_path, duration
            )
        return resolved
