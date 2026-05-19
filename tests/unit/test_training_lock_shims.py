"""
Unit tests for the training-lock no-op shims.

Demo-readiness #2: ``training_service/training_job.py::_release_training_locks``
imports ``set_tts_training_lock`` from ``app.services.tts.qwen_tts_engine`` and
``set_asr_training_lock`` from ``app.services.asr.engine`` after every SFT run.
Before the fix those symbols didn't exist and the surrounding generic
``except Exception`` swallowed the ``ImportError``, logging a scary warning.

These tests pin the public API surface so the warning can't come back
silently — if either shim is renamed/removed in a refactor, this test
will scream.
"""
from __future__ import annotations

import importlib

import pytest


class TestTrainingLockShims:
    def test_tts_shim_is_importable(self):
        """The exact import the training-job release path uses must work."""
        from app.services.tts.qwen_tts_engine import set_tts_training_lock

        assert callable(set_tts_training_lock)

    def test_asr_shim_is_importable(self):
        """The exact import the training-job release path uses must work."""
        from app.services.asr.engine import set_asr_training_lock

        assert callable(set_asr_training_lock)

    @pytest.mark.parametrize("active", [True, False])
    def test_tts_shim_accepts_bool_and_returns_none(self, active):
        from app.services.tts.qwen_tts_engine import set_tts_training_lock

        assert set_tts_training_lock(active) is None

    @pytest.mark.parametrize("active", [True, False])
    def test_asr_shim_accepts_bool_and_returns_none(self, active):
        from app.services.asr.engine import set_asr_training_lock

        assert set_asr_training_lock(active) is None

    def test_training_job_release_path_does_not_warn(self, caplog):
        """
        End-to-end: the same import + call sequence used by
        ``_release_training_locks`` (training_job.py:952-962). Before
        the fix this raised ImportError caught by ``except Exception``,
        which logged "Failed to release ... lock: No module named ...".
        After the fix the imports resolve and the calls return cleanly.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            from app.services.tts.qwen_tts_engine import set_tts_training_lock

            set_tts_training_lock(False)
            from app.services.asr.engine import set_asr_training_lock

            set_asr_training_lock(False)

        # No "Failed to release" warning should be in caplog.
        failure_warnings = [
            r for r in caplog.records if "Failed to release" in r.getMessage()
        ]
        assert failure_warnings == [], (
            "Lock shims must not produce 'Failed to release' warnings — "
            f"got {[r.getMessage() for r in failure_warnings]}"
        )
