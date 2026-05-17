"""
Unit tests for the stale-cancel race fix in state_manager
(commit 6c9a87a review #21).

The bug: cancel_llm_task latched llm_pending_cancel unconditionally; if
a cancel arrived between utterance N's clear_llm_task and utterance N+1's
set_llm_task, the new utterance was silently killed.

The fix: every cancel stamps the latch with the CURRENT utterance seq
(set by begin_utterance). set_llm_task honors the latch only when the
seq matches its own utterance_seq.
"""
from __future__ import annotations

import asyncio

import pytest

from app.core.state_manager import StateManager
from app.services.asr import MockASR


@pytest.fixture
def manager():
    m = StateManager(asr=MockASR(), use_qwen=False)
    return m


@pytest.fixture
def session_id(manager):
    sid = "test-session"
    state = manager.create_session(sid)
    # `set_llm_task` requires the session exists.
    return sid


class TestStaleCancelRace:
    def test_cancel_for_prior_utterance_does_not_kill_next(
        self, manager, session_id,
    ):
        """The headline bug: cancel arriving between utterance N and N+1
        must not fire on N+1."""
        # Utterance N is in flight.
        seq_n = manager.begin_utterance(session_id)
        evt_n = asyncio.Event()
        task_n = asyncio.get_event_loop().create_task(asyncio.sleep(0.01))
        manager.set_llm_task(session_id, task_n, evt_n, utterance_seq=seq_n)
        # N completes cleanly.
        manager.clear_llm_task(session_id)
        # A stale cancel arrives (think: barge-in mistakenly fired, or
        # an old ws_explicit_cancel queued up).
        manager.cancel_llm_task(session_id, origin="test_stale")
        # Utterance N+1 starts.
        seq_np1 = manager.begin_utterance(session_id)
        evt_np1 = asyncio.Event()
        task_np1 = asyncio.get_event_loop().create_task(asyncio.sleep(0.01))
        manager.set_llm_task(session_id, task_np1, evt_np1, utterance_seq=seq_np1)
        # Critical assertion: the new utterance's cancellation_event is
        # NOT set, so it streams normally.
        assert not evt_np1.is_set(), (
            "Stale cancel from prior utterance leaked through and killed "
            "the new utterance — review #21 regression."
        )

    def test_cancel_during_startup_still_fires(self, manager, session_id):
        """The original 'cancel-before-set' race must still work — a
        cancel arriving in the window between begin_utterance and
        set_llm_task should fire on the new utterance as soon as it
        registers."""
        seq = manager.begin_utterance(session_id)
        # Cancel arrives BEFORE set_llm_task (the legitimate startup race).
        manager.cancel_llm_task(session_id, origin="test_startup_race")
        # Now set_llm_task registers.
        evt = asyncio.Event()
        task = asyncio.get_event_loop().create_task(asyncio.sleep(0.01))
        manager.set_llm_task(session_id, task, evt, utterance_seq=seq)
        # The latched cancel must fire now.
        assert evt.is_set(), (
            "Cancel-during-startup race regression — set_llm_task didn't "
            "honor the legitimate latched cancel."
        )

    def test_seq_increments_monotonically(self, manager, session_id):
        s1 = manager.begin_utterance(session_id)
        s2 = manager.begin_utterance(session_id)
        s3 = manager.begin_utterance(session_id)
        assert s1 < s2 < s3

    def test_legacy_caller_without_utterance_seq(self, manager, session_id):
        """Backward-compat: callers that don't pass utterance_seq still
        get the old behavior (latched cancel fires)."""
        seq = manager.begin_utterance(session_id)
        manager.cancel_llm_task(session_id, origin="legacy_test")
        evt = asyncio.Event()
        task = asyncio.get_event_loop().create_task(asyncio.sleep(0.01))
        # No utterance_seq kwarg.
        manager.set_llm_task(session_id, task, evt)
        assert evt.is_set()


class TestCancelOriginLogging:
    def test_origin_logged_at_warn(self, manager, session_id, caplog):
        """Review-driven contract: every cancel_llm_task call logs
        `origin=<value>` at WARN — future refactors must not silently
        strip this."""
        import logging
        with caplog.at_level(logging.WARNING):
            manager.cancel_llm_task(session_id, origin="vad_barge_in")
        log_text = " ".join(r.getMessage() for r in caplog.records)
        assert "origin=vad_barge_in" in log_text
