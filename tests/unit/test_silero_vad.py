"""
Unit tests for the SileroVAD state machine with hysteresis.

The previous averaging-over-window implementation produced three live
failure modes (see app/services/asr/silero_vad.py module docstring):

 1. medium preset never commits — averaged prob hovers at threshold.
 2. high preset cuts after ~0.5s — min_silence too low.
 3. low preset cuts mid-sentence — inter-word dip plus accumulating
    silence counter.

These tests pin the new state machine against scripted probability
sequences so the same bugs can't quietly come back.

We bypass the ONNX session entirely by patching
`SileroVAD._run_inference` to return a scripted sequence of floats.
That lets each test express the test scenario as "the model would
have said: prob=0.8 ten times, then prob=0.1 twenty times" without
faking the model bytes.
"""
from __future__ import annotations

from typing import List

import pytest

from app.services.asr.silero_vad import (
    SileroVAD,
    SileroVADConfig,
    _STATE_SILENCE,
    _STATE_SPEECH,
)


# -----------------------------------------------------------------------------
# Scripted-prob harness.
# -----------------------------------------------------------------------------


class _ScriptedVAD(SileroVAD):
    """SileroVAD subclass that ignores audio_bytes and returns scripted
    probabilities from a list, advancing one index per call.

    When the script is exhausted, returns the last value forever — that
    keeps tests that step "until commit" terminating predictably even if
    a bug makes them not commit.
    """

    def __init__(self, script: List[float], **kwargs):
        super().__init__(**kwargs)
        # Mark as loaded so _ensure_loaded() inside _run_inference is a
        # no-op even though _session is None.
        self._is_loaded = True
        self._script = list(script)
        self._script_idx = 0

    def _run_inference(self, audio_bytes: bytes):  # type: ignore[override]
        if not self._script:
            return None
        if self._script_idx >= len(self._script):
            return self._script[-1]
        v = self._script[self._script_idx]
        self._script_idx += 1
        return v


# An arbitrary 60ms-ish chunk of bytes — the contents don't matter
# because _run_inference is mocked.
_CHUNK = b"\x00\x00" * 1440


def _run(vad: SileroVAD, n_frames: int) -> List[tuple]:
    """Step the VAD n_frames times, returning the (is_commit, prob)
    results."""
    return [vad.detect(_CHUNK) for _ in range(n_frames)]


# -----------------------------------------------------------------------------
# 1. Headline bug — medium preset MUST commit after 0.7s of silence
#    following speech. With the old averaging code it never did.
# -----------------------------------------------------------------------------


class TestMediumPresetCommitsAfterSilence:
    def test_medium_commits_after_700ms_silence(self):
        """Medium preset: 1s of clear speech, then 0.7s+ of clear silence
        → exactly one commit fires within the silence window."""
        # 60ms/frame; ~17 frames = 1.02s of speech, then 14 frames = 840ms
        # of silence (more than the 0.7s threshold).
        script = [0.95] * 17 + [0.05] * 14
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")

        results = _run(vad, len(script))
        commits = [i for i, (c, _) in enumerate(results) if c]
        assert len(commits) == 1, (
            f"Expected exactly one commit, got {len(commits)} at "
            f"frames {commits}"
        )
        # min_silence=0.7s = 12 frames at 60ms (ceil). Commit should be
        # ~12 frames into the silence run, i.e. around frame 17+12-1=28.
        # Allow a small tolerance for the ceil-rounding.
        assert 27 <= commits[0] <= 29, (
            f"Commit landed at frame {commits[0]} — expected ~28 "
            f"(speech_end=17 + min_silence_frames=12)"
        )

    def test_medium_does_not_commit_during_speech(self):
        """While still in the speech run, no commit ever fires."""
        script = [0.95] * 30  # 1.8s of solid speech
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")
        results = _run(vad, len(script))
        assert not any(c for c, _ in results), (
            "VAD committed during a run of solid speech frames — "
            "state machine is leaking false commits."
        )


# -----------------------------------------------------------------------------
# 2. Inter-word gap MUST NOT trigger commit.
# -----------------------------------------------------------------------------


class TestNaturalInterWordPauseDoesNotCommit:
    def test_medium_survives_200ms_inter_word_gap(self):
        """A 200ms gap mid-sentence (4 frames) is well below the
        medium 0.7s commit threshold. The state machine must absorb it
        without committing.

        With the old averaging code, the 200ms dip drove avg_prob below
        threshold AND ticked the silence counter (which kept counting
        as long as avg stayed sub-threshold), producing a premature
        commit. New code: _consec_below counter is the only thing that
        gates commit, and it resets on any frame >= silence_threshold."""
        # speech / 200ms-pause / speech / 200ms-pause / speech
        script = (
            [0.9] * 10  # 600ms speech
            + [0.1] * 4  # 240ms gap
            + [0.9] * 10  # 600ms speech
            + [0.1] * 3  # 180ms gap
            + [0.9] * 10  # 600ms speech
        )
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")
        results = _run(vad, len(script))
        commits = [i for i, (c, _) in enumerate(results) if c]
        assert commits == [], (
            f"Inter-word pause caused premature commit at {commits}. "
            f"min_silence_frames={vad._min_silence_frames} should "
            f"have absorbed gaps of 3-4 frames."
        )

    def test_low_preset_more_forgiving_than_medium_for_dips(self):
        """Low preset has min_silence=0.9s vs medium's 0.7s. A 13-frame
        (~780ms) gap commits on medium but NOT on low."""
        # 600ms speech, 780ms gap (13 frames), 600ms speech.
        # 13 * 60ms = 780ms. medium needs 12 frames (~720ms) to commit;
        # low needs 15 frames (~900ms).
        script = (
            [0.9] * 10  # 600ms speech
            + [0.1] * 13  # 780ms gap
            + [0.9] * 10  # 600ms speech
        )
        med = _ScriptedVAD(list(script), sample_rate=24000, sensitivity="medium")
        low = _ScriptedVAD(list(script), sample_rate=24000, sensitivity="low")
        med_results = _run(med, len(script))
        low_results = _run(low, len(script))
        med_commits = [i for i, (c, _) in enumerate(med_results) if c]
        low_commits = [i for i, (c, _) in enumerate(low_results) if c]
        assert len(med_commits) == 1, (
            f"medium should have committed once during the 780ms gap, "
            f"got {med_commits}"
        )
        assert low_commits == [], (
            f"low should NOT have committed during 780ms gap "
            f"(needs 900ms+), got {low_commits}"
        )


# -----------------------------------------------------------------------------
# 3. High preset commits faster than medium.
# -----------------------------------------------------------------------------


class TestHighPresetFasterThanMedium:
    def test_high_commits_earlier_than_medium_on_same_input(self):
        # 600ms speech then long silence — both will commit, but high
        # should commit at frame ~10+8=18, medium at ~10+12=22.
        script = [0.95] * 10 + [0.05] * 20
        high = _ScriptedVAD(list(script), sample_rate=24000, sensitivity="high")
        med = _ScriptedVAD(list(script), sample_rate=24000, sensitivity="medium")

        h_results = _run(high, len(script))
        m_results = _run(med, len(script))
        h_commit = next((i for i, (c, _) in enumerate(h_results) if c), None)
        m_commit = next((i for i, (c, _) in enumerate(m_results) if c), None)

        assert h_commit is not None, "high never committed"
        assert m_commit is not None, "medium never committed"
        assert h_commit < m_commit, (
            f"high should commit earlier than medium: "
            f"high@{h_commit}, medium@{m_commit}"
        )

    def test_high_min_silence_safely_above_stop_consonant_gap(self):
        """The original bug had high.min_silence=0.2s, which cut on
        normal stop-consonant gaps. The new preset must be >= 0.35s."""
        vad = SileroVAD(sample_rate=24000, sensitivity="high")
        assert vad.min_silence_duration >= 0.35, (
            f"high.min_silence_duration={vad.min_silence_duration} is "
            f"below the 0.35s minimum that the headline bug triaged "
            f"as cutting users mid-word."
        )


# -----------------------------------------------------------------------------
# 4. Hysteresis prevents flicker.
# -----------------------------------------------------------------------------


class TestHysteresisPreventsFlicker:
    def test_oscillation_in_hysteresis_band_does_not_flip_state(self):
        """A probability sequence that oscillates between the two
        thresholds must not flip the state. Specifically:
          medium: speech_thresh=0.50, silence_thresh=0.35
          A scripted prob of [0.95, 0.95, 0.95, 0.42, 0.42, ...] enters
          SPEECH then stays in SPEECH because 0.42 is in the hysteresis
          band (>= silence_thresh, < speech_thresh). The OLD code would
          have started ticking silence_frames because 0.42 < 0.5; the
          new code does NOT tick because 0.42 >= silence_thresh=0.35.
        """
        # 3 frames of clear speech to enter SPEECH state, then many
        # frames at 0.42 in the hysteresis band.
        script = [0.95] * 3 + [0.42] * 30
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")
        results = _run(vad, len(script))
        assert not any(c for c, _ in results), (
            "VAD committed while probs were in the hysteresis band — "
            "single-threshold flicker regression."
        )
        # And we should still be in SPEECH state at the end.
        assert vad._vad_state == _STATE_SPEECH

    def test_dip_below_silence_thresh_then_back_does_not_commit_early(self):
        """A SINGLE-frame dip below silence_threshold during speech
        should reset _consec_below — not accumulate toward commit.
        Simulates a brief glottal stop or microphone hiccup."""
        # Enter speech, then dip-for-1-frame, then speech for ages.
        # If _consec_below were a soft-accumulating counter (the old
        # bug), repeated 1-frame dips spaced out would eventually
        # accumulate to min_silence_frames and commit. With reset-on-
        # any-non-silence, they don't.
        script = (
            [0.9] * 3
            + ([0.1] + [0.9] * 5) * 10  # 10 cycles of 1-frame dip + 5 speech frames
        )
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")
        results = _run(vad, len(script))
        commits = [i for i, (c, _) in enumerate(results) if c]
        assert commits == [], (
            f"Single-frame dips accumulated toward a false commit: {commits}"
        )


# -----------------------------------------------------------------------------
# 5. current_probability reflects the most recent raw prob (not avg).
# -----------------------------------------------------------------------------


class TestCurrentProbability:
    def test_current_probability_returns_last_raw_prob(self):
        script = [0.1, 0.5, 0.9, 0.7, 0.3]
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")
        for expected in script:
            vad.detect(_CHUNK)
            assert vad.current_probability == pytest.approx(expected), (
                f"current_probability should be the raw prob, got "
                f"{vad.current_probability} != {expected}"
            )


# -----------------------------------------------------------------------------
# 6. Preset wiring + backward compatibility.
# -----------------------------------------------------------------------------


class TestPresetWiringAndBackcompat:
    @pytest.mark.parametrize("sensitivity, expected_st, expected_sit", [
        ("low", 0.40, 0.30),
        ("medium", 0.50, 0.35),
        ("high", 0.60, 0.45),
    ])
    def test_preset_thresholds_match_calibration_table(
        self, sensitivity, expected_st, expected_sit
    ):
        vad = SileroVAD(sample_rate=24000, sensitivity=sensitivity)
        assert vad.speech_threshold == pytest.approx(expected_st)
        assert vad.silence_threshold == pytest.approx(expected_sit)
        assert vad.silence_threshold < vad.speech_threshold, (
            "Hysteresis must hold: silence_thresh < speech_thresh"
        )

    def test_legacy_single_threshold_derives_hysteresis(self):
        """A caller that only passes `threshold=` (no sensitivity, no
        explicit silence_threshold) must still get hysteresis — derived
        as threshold - 0.15 clamped >= 0.20."""
        vad = SileroVAD(
            sample_rate=24000, threshold=0.6, sensitivity="custom-noop"
        )
        assert vad.speech_threshold == pytest.approx(0.6)
        assert vad.silence_threshold == pytest.approx(0.45)

    def test_legacy_low_threshold_clamps_silence_floor(self):
        """A pathological caller passing threshold=0.3 would derive
        silence_threshold=0.15 — but we clamp to >= 0.20 to avoid
        silly noise floors."""
        vad = SileroVAD(
            sample_rate=24000, threshold=0.30, sensitivity="custom-noop"
        )
        assert vad.silence_threshold == pytest.approx(0.20)

    def test_sensitivity_label_round_trip(self):
        for name in ("low", "medium", "high"):
            vad = SileroVAD(sample_rate=24000, sensitivity=name)
            assert vad.sensitivity_label == name

    def test_config_dataclass_exposes_new_fields(self):
        """SileroVADConfig has the new threshold fields. Legacy
        `threshold` field is kept for back-compat."""
        cfg = SileroVADConfig()
        assert hasattr(cfg, "speech_threshold")
        assert hasattr(cfg, "silence_threshold")
        assert hasattr(cfg, "threshold")
        assert cfg.silence_threshold < cfg.speech_threshold


# -----------------------------------------------------------------------------
# 7. State-machine internals — sanity-check that reset() restores
# SILENCE so the WS handler can use the same VAD across utterances.
# -----------------------------------------------------------------------------


class TestResetBehavior:
    def test_reset_returns_to_silence_state(self):
        # Get into SPEECH state.
        script = [0.95] * 5
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")
        _run(vad, 5)
        assert vad._vad_state == _STATE_SPEECH
        vad.reset()
        assert vad._vad_state == _STATE_SILENCE
        assert vad._consec_above == 0
        assert vad._consec_below == 0

    def test_reset_does_not_affect_detect_return_signature(self):
        """The WS handler calls reset() in process_audio. After reset,
        the next detect() must still return a (bool, float) tuple."""
        script = [0.9] * 3 + [0.1] * 20
        vad = _ScriptedVAD(script, sample_rate=24000, sensitivity="medium")
        _run(vad, 23)
        # After whatever's happened, reset and verify the next call has
        # the right shape.
        vad.reset()
        r = vad.detect(_CHUNK)
        assert isinstance(r, tuple) and len(r) == 2
        assert isinstance(r[0], bool)
        assert isinstance(r[1], float)


# -----------------------------------------------------------------------------
# 8. Integration with state_manager.process_audio's barge-in edge-trigger:
#    `_vad_had_speech` flips False→True cleanly when SileroVAD reports
#    is_commit=False on the first speech frame. This test pins the
#    behavior that commit `0cfd810` relies on.
# -----------------------------------------------------------------------------


class TestIntegrationWithStateManagerBargeInEdge:
    def test_first_speech_frame_returns_is_commit_false(self):
        """state_manager.process_audio's barge-in path treats
        (is_commit=False AND _vad_had_speech was False) as the speech-
        start edge. SileroVAD must return is_commit=False on the very
        first speech frame so this edge fires correctly."""
        vad = _ScriptedVAD([0.95] * 10, sample_rate=24000, sensitivity="medium")
        is_commit, _ = vad.detect(_CHUNK)
        assert is_commit is False, (
            "First speech frame returned is_commit=True — barge-in "
            "edge-trigger in state_manager.process_audio would never "
            "fire."
        )
