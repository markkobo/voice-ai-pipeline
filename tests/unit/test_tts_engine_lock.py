"""
Unit test for the TTS engine's load lock.

The Phase 0 audit flagged a documented race: two concurrent
`generate_streaming()` calls would both observe `is_loaded=False`, both
trigger `_ensure_loaded()`, and double-load the model — leaking VRAM.

Phase 2 wraps load + activate in a threading.RLock. This test proves the
common-case + worst-case behavior using a FasterQwenTTSEngine instance
with the model-loading internals replaced by a counting fake.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest


@pytest.fixture
def engine():
    """Build a FasterQwenTTSEngine without triggering CUDA / model downloads."""
    from app.services.tts.qwen_tts_engine import FasterQwenTTSEngine

    with patch("torch.cuda.is_available", return_value=False):
        return FasterQwenTTSEngine(model_size="1.7B", device="cpu")


class TestLoadLock:
    def test_concurrent_ensure_loaded_loads_once(self, engine):
        """50 threads racing _ensure_loaded — exactly one load happens."""
        load_count = 0
        load_lock = threading.Lock()

        def fake_load_impl():
            nonlocal load_count
            with load_lock:
                load_count += 1
            time.sleep(0.05)  # widen the race window
            engine._model = object()
            engine._is_loaded = True

        with patch.object(engine, "_ensure_loaded_locked", side_effect=fake_load_impl):
            threads = [
                threading.Thread(target=engine._ensure_loaded) for _ in range(50)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert load_count == 1, (
            f"Expected exactly 1 load under contention, got {load_count}. "
            "The load lock failed to serialize."
        )
        assert engine._is_loaded is True

    def test_reentry_does_not_deadlock(self, engine):
        """activate_version → _ensure_loaded must not deadlock the RLock."""
        # Simulate: caller holds the lock and calls a method that also tries
        # to acquire it. RLock allows the same thread to re-enter.
        with engine._load_lock:
            # This would deadlock with a plain Lock; the RLock should let it through.
            with engine._load_lock:
                pass  # success means no deadlock

    def test_fast_path_skips_lock(self, engine):
        """If already loaded, _ensure_loaded returns without taking the lock."""
        engine._is_loaded = True
        # Replace the lock with one that fails if acquired — proves fast path.
        sentinel_lock = threading.Lock()
        original_lock = engine._load_lock
        try:
            engine._load_lock = _NoAcquireLock()
            engine._ensure_loaded()  # must not touch the lock
        finally:
            engine._load_lock = original_lock


class _NoAcquireLock:
    """Lock substitute that raises if any acquire attempt is made."""

    def __enter__(self):
        raise AssertionError("Fast path should not acquire the load lock")

    def __exit__(self, *args):
        return False

    def acquire(self, *args, **kwargs):
        raise AssertionError("Fast path should not acquire the load lock")

    def release(self, *args, **kwargs):
        raise AssertionError("Fast path should not release the load lock")
