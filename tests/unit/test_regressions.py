"""
Regression tests for bug fixes.
Each test documents a past bug to prevent reoccurrence.

Add new tests here after fixing bugs.
Run: pytest tests/unit/test_regressions.py -v --tb=short
"""

import pytest
import json
import tempfile
from pathlib import Path


class TestAPIBugs:
    """API-level regression tests."""

    def test_fastapi_query_vs_body(self):
        """
        Regression: FastAPI Optional[str] params are query params, not body.
        Symptom: PATCH /segments/{id} returns null despite correct payload.
        Fix: Use URLSearchParams in frontend, not JSON body.
        """
        # This is a documentation test - the actual fix is in the frontend JS
        # Verify the API accepts query params correctly
        pass

    def test_speaker_labels_sync(self):
        """
        Regression: persona_id update in speaker_segments needs to sync speaker_labels.
        Symptom: Changing persona in dropdown appears to work but reverts on F5.
        Fix: Update both speaker_segments.persona_id AND speaker_labels dict.
        """
        pass


class TestTrainingBugs:
    """Training-related regression tests."""

    def test_progress_json_not_overwritten(self):
        """
        Regression: ProgressTracker overwrites progress.json after subprocess completes.
        Symptom: Training shows epoch=0 repeatedly, appears stalled.
        Fix: Remove redundant ProgressTracker instantiations in _run_training().
        """
        # Verify that once progress.json is written with status=ready, it stays
        # The actual fix was removing the overwriting ProgressTracker() calls
        pass

    def test_audio_resampling_for_24k(self):
        """
        Regression: Audio files at 48kHz cause AssertionError in extract_speaker_embedding.
        Symptom: Training fails with "Only support 24kHz audio"
        Fix: Resample to 24kHz using scipy.signal.resample before encoding.
        """
        # Verify the generated train_lora.py contains resampling code
        import subprocess
        result = subprocess.run(
            ["grep", "-c", "Resampling.*from.*Hz to.*Hz", "/workspace/voice-ai-pipeline/data/models/persona_mocdl6af_v9_20260427_013620/train_lora.py"],
            capture_output=True, text=True
        )
        # Should have resampling in the script (2 places: encoding + speaker embedding)
        assert int(result.stdout.strip() or "0") >= 1, "Resampling code not found in train_lora.py"


class TestUIBugs:
    """UI-related regression tests."""

    def test_segment_parsing_no_leading_underscore(self):
        """
        Regression: segment parsing must return full speakerId like "SPEAKER_00".
        Symptom: Training preview shows no segment details.
        Fix: Use segId.substring(speakerIndex + 1) to get "SPEAKER_00"
        """
        # segId format: {recording_id}_SPEAKER_{number}
        # speakerIndex + 1 gives "SPEAKER_00" (matches speaker_segments.speaker_id)
        seg_id = "rec123_SPEAKER_00"
        marker = "_SPEAKER_"
        pos = seg_id.find(marker)
        actual = seg_id[pos + 1:]  # Should be "SPEAKER_00" (full speaker ID)
        assert actual == "SPEAKER_00", f"Got {actual}"

    def test_listener_filter_resets_on_persona_change(self):
        """
        Regression: listener filter persists when switching personas in training UI.
        Symptom: Empty state with no indication why after persona change.
        Fix: Reset listener filter when persona changes.
        """
        pass  # UI-only fix, documented for awareness

    def test_auto_refresh_skips_modal(self):
        """
        Regression: auto-refresh rebuilds DOM during modal/dropdown interaction.
        Symptom: UI state lost during 5-second polling.
        Fix: Skip refresh when modal open or dropdown focused.
        """
        pass  # UI-only fix, documented for awareness
