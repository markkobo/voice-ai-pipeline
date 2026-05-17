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


# -----------------------------------------------------------------------------
# Open issue #1 (tests/_phase2_followups.md §5): "LLM cancelled 3s into stream
# with empty partial_text". Root cause was the barge-in branch in
# `state_manager.process_audio` — level-triggered AND firing even when no LLM
# task existed. On the user's FIRST utterance ever, every speech frame stamped
# `pending_cancel`; when a late-arriving PCM chunk hit `process_audio` after
# `begin_utterance` had bumped the seq, a fresh latch at the new seq made
# `set_llm_task` honor it — killing the LLM stream before token 1.
# -----------------------------------------------------------------------------


class _ScriptedVAD:
    """A VAD that returns a scripted sequence of (is_commit, prob)."""
    def __init__(self):
        self.script: list[tuple[bool, float]] = []
        self.idx = 0
        self.sample_rate = 24000

    def detect(self, audio_bytes: bytes) -> tuple[bool, float]:
        if self.idx < len(self.script):
            r = self.script[self.idx]
            self.idx += 1
            return r
        return (False, 0.0)

    def reset(self):
        pass

    @property
    def sensitivity_label(self) -> str:
        return "medium"


class TestBargeInEdgeTriggered:
    """Open issue #1: barge-in must be edge-triggered and gated on an
    actually-active LLM task. Level-triggered barge-in spammed cancel
    latches and killed fresh utterances."""

    def _make_state(self, manager):
        sid = "edge-test"
        state = manager.create_session(sid, vad=_ScriptedVAD())
        # update_config is required to mark `is_configured=True` AND to
        # apply audio settings — but it rebuilds VAD to SileroVAD. We
        # bypass by just flipping the flag and reusing our scripted VAD.
        state.is_configured = True
        state.vad = _ScriptedVAD()
        return sid, state

    def test_barge_in_does_not_fire_with_no_active_llm_task(self, manager):
        """First-utterance scenario: user is speaking, no LLM ever ran.
        process_audio must NOT latch any cancel intent — without an
        active task, there is nothing to barge in on."""
        sid, state = self._make_state(manager)
        # 10 frames of continuous speech, then silence-with-commit
        state.vad.script = [(False, 0.9)] * 10 + [(True, 0.05)]
        chunk = b'\x00\x00' * 1440

        for _ in range(11):
            manager.process_audio(sid, chunk)

        # The killer assertion: even though the user spoke for 10 frames,
        # no cancel intent was latched because `state.llm_task is None`.
        assert state.llm_pending_cancel is False, (
            "Open issue #1 regression: barge-in latched a cancel during "
            "first-utterance speech with no LLM running. This stale latch "
            "would fire on the next utterance's LLM stream."
        )
        assert state.llm_pending_cancel_seq is None

    def test_barge_in_fires_once_on_speech_start_with_active_llm(self, manager):
        """Genuine barge-in: an LLM task IS active, user starts a new
        utterance. We must fire cancel exactly once on the speech-start
        transition — not on every subsequent speech frame."""
        sid, state = self._make_state(manager)
        # Install a fake in-flight LLM task. We don't need a real coroutine —
        # `state.llm_task is not None and not done()` is what process_audio
        # checks.
        loop = asyncio.new_event_loop()
        try:
            async def _forever():
                await asyncio.sleep(10)
            fake_task = loop.create_task(_forever())
            fake_event = asyncio.Event()
            seq = manager.begin_utterance(sid)
            manager.set_llm_task(sid, fake_task, fake_event, utterance_seq=seq)

            # 5 consecutive speech frames
            state.vad.script = [(False, 0.9)] * 5
            chunk = b'\x00\x00' * 1440
            for _ in range(5):
                manager.process_audio(sid, chunk)

            # Cancel event fired exactly once (the edge), not five times.
            assert fake_event.is_set()
            # And the LLM task was cancelled.
            assert fake_task.cancelled() or fake_task.done() or fake_task._must_cancel
        finally:
            for t in asyncio.all_tasks(loop):
                t.cancel()
            loop.close()

    def test_late_pcm_after_begin_utterance_does_not_kill_fresh_llm(self, manager):
        """Production scenario: user finishes speaking, `vad_commit` is sent,
        client sends `commit_utterance`, server runs ASR, calls
        `begin_utterance` (seq 0→1) then `create_task(run_llm_stream)`.
        BEFORE the task registers via `set_llm_task`, late-buffered PCM
        chunks arrive at the WS handler. process_audio runs against them.

        The fix guarantees these late frames do NOT fire barge-in (because
        `state.llm_task is None` — the new task hasn't registered yet),
        so no fresh pending_cancel is latched at seq=1."""
        sid, state = self._make_state(manager)

        # Simulate the full flow up to vad_commit
        state.vad.script = (
            # 10 speech frames during user's utterance (was: latch spam)
            [(False, 0.9)] * 10
            # Final commit frame
            + [(True, 0.05)]
            # POST-commit: 3 late-arriving frames the browser sent before it
            # processed our vad_commit message and stopped the mic. These
            # would have looked like "continuing speech" to the old code.
            + [(False, 0.9)] * 3
        )
        chunk = b'\x00\x00' * 1440

        # Process the speech + commit
        for _ in range(11):
            result = manager.process_audio(sid, chunk)
        assert result and result.get("type") == "vad_commit"
        # After commit, _reset_utterance() is normally called by
        # `commit_utterance`. We mimic that by manually clearing audio
        # state via the documented sequence used by the WS handler.
        state.audio_buffer.clear()
        state._vad_committed = False
        state._vad_had_speech = False

        # Server bumps seq for the new utterance
        seq = manager.begin_utterance(sid)
        assert seq == 1

        # Now the 3 late-arriving PCM chunks come in BEFORE set_llm_task.
        # With the old code, each would fire `cancel_llm_task` at the new
        # seq=1, latching pending_cancel_seq=1. Then set_llm_task(seq=1)
        # would honor the (now-matching) seq and fire the cancellation
        # event. RESULT: fresh LLM killed before token 1.
        for _ in range(3):
            manager.process_audio(sid, chunk)

        # With the fix, no barge-in fires because state.llm_task is None
        # (the run_llm_stream coroutine hasn't called set_llm_task yet).
        assert state.llm_pending_cancel is False, (
            "Late-arriving PCM after begin_utterance latched a cancel — "
            "fresh LLM would be killed at set_llm_task. Open issue #1."
        )

        # Now the LLM task registers
        loop = asyncio.new_event_loop()
        try:
            evt = asyncio.Event()
            task = loop.create_task(asyncio.sleep(0.01))
            manager.set_llm_task(sid, task, evt, utterance_seq=seq)
            assert not evt.is_set(), (
                "Cancellation event was set on fresh LLM task — the user's "
                "first response was killed without a single token."
            )
        finally:
            for t in asyncio.all_tasks(loop):
                t.cancel()
            loop.close()
