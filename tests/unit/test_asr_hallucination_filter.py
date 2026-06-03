"""
Locks the ASR hallucination filter shipped during 06/02 demo prep.

Qwen3-ASR (Whisper-family) emits a fixed set of stock phrases when fed
near-silent audio:
  - "The first was the first to be built."
  - "《大明宫词》"
  - "Thank you.", "Thanks for watching.", "Bye.", "So.", "Okay.", etc.

The filter in app/services/asr/engine.py Qwen3ASR.recognize() uses
**peak amplitude as the primary discriminator**:
  - peak >= 0.12 → keep whatever ASR returned (it's real speech).
  - peak <  0.12 → drop. Known-text set is logged separately from
                   "SILENCE DROPPED" for diagnostics, but both branches
                   end in text = "".

GPT-5 review 2026-06-03 caught the previous bug where the known-text
branch ran regardless of peak — that meant a loud "Okay" or "Thank
you" got dropped. Fix: gate the known-text drop on peak<threshold too.

These tests don't load the real Qwen3-ASR model — they patch the inference
call to return canned ASR outputs and verify the filter behavior.
"""
import struct

import pytest

from app.services.asr.engine import (
    Qwen3ASR,
    _HALLUCINATION_TEXTS,
    _SILENCE_PEAK_THRESHOLD,
    _normalize_for_halluc_match,
)


def _pcm_bytes(peak: float, length_samples: int = 24000) -> bytes:
    """Build a PCM16 buffer with a single peak sample at `peak` amplitude
    (in normalized [-1, 1]) and zeros elsewhere. Length defaults to ~1s
    at 24 kHz so the buffer is long enough to look like an utterance."""
    samples = [0] * length_samples
    if length_samples > 0:
        samples[length_samples // 2] = int(peak * 32767)
    return struct.pack(f"{length_samples}h", *samples)


class _FakeModel:
    """Stand-in for the real Qwen3-ASR model. The `text` arg is what
    transcribe() will return so each test can pin the ASR output."""

    def __init__(self, text: str):
        self._text = text

    def transcribe(self, audio_tuple):
        class _Result:
            def __init__(self, t):
                self.text = t

        return [_Result(self._text)]


@pytest.fixture
def make_asr():
    def _make(canned_text: str) -> Qwen3ASR:
        asr = Qwen3ASR.__new__(Qwen3ASR)
        asr._model = _FakeModel(canned_text)
        asr._sample_rate = 24000
        asr.latency_ms = 0
        return asr

    return _make


# ------------------------------------------------------------------
# Silence drop (peak < threshold) — this is the primary defense
# ------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "halluc_text",
    [
        "The first was the first to be built.",
        "《大明宫词》",
        "Thank you.",
        "Thank you",
        "Thanks for watching.",
        "Bye.",
        "So.",
        "Okay.",
        "OK.",
        "The.",
        "You.",
        "《小猪佩奇》",
    ],
)
async def test_hallucination_dropped_when_audio_is_silent(make_asr, halluc_text):
    """Known stock phrases get dropped when audio peak is below silence
    threshold (the hallucination case from demo logs)."""
    asr = make_asr(halluc_text)
    audio = _pcm_bytes(peak=0.05)  # well below 0.12 threshold
    result = await asr.recognize(audio)
    assert result["text"] == "", (
        f"hallucination {halluc_text!r} on silent buffer should be filtered"
    )


# ------------------------------------------------------------------
# Real speech (peak >= threshold) — must NOT be dropped, even if text
# matches a hallucination word. GPT-5 caught this bug 2026-06-03.
# ------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "word",
    ["Okay.", "Thank you.", "Bye.", "OK.", "So.", "The."],
)
async def test_loud_hallucination_word_is_kept(make_asr, word):
    """A loud, single common word must NOT be dropped — the user might
    actually say it. The previous filter dropped these regardless of
    audio peak (regression caught by GPT-5 code review)."""
    asr = make_asr(word)
    audio = _pcm_bytes(peak=0.5)  # well above 0.12 — clearly real speech
    result = await asr.recognize(audio)
    assert result["text"] == word, (
        f"loud {word!r} was incorrectly filtered out — text loss"
    )


@pytest.mark.asyncio
async def test_real_speech_long_sentence_passes(make_asr):
    """Plausible real-speech transcription with healthy peak survives."""
    asr = make_asr("Hi, I'm Mark. I build EverHome.")
    audio = _pcm_bytes(peak=0.5)
    result = await asr.recognize(audio)
    assert result["text"] == "Hi, I'm Mark. I build EverHome."


# ------------------------------------------------------------------
# Edge: silence drop fires regardless of whether text matches hallucination
# set — long ASR output on silent audio is still hallucination.
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_silent_audio_drops_non_stock_text(make_asr):
    """When audio is silent, any ASR text gets dropped — not just the
    known stock phrases. (e.g., the 'speed dough competition' case
    observed in demo logs.)"""
    asr = make_asr("This girl was about to compete in a speed dough competition.")
    audio = _pcm_bytes(peak=0.05)
    result = await asr.recognize(audio)
    assert result["text"] == ""


# ------------------------------------------------------------------
# Threshold-boundary behavior
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_peak_just_above_threshold_passes(make_asr):
    """Audio peak of 0.13 (just above 0.12) keeps the ASR output. The
    threshold is intentionally permissive — false-negative silence drops
    are worse than letting a low-peak speech sample through."""
    asr = make_asr("That's a quieter sentence.")
    audio = _pcm_bytes(peak=0.13)
    result = await asr.recognize(audio)
    assert result["text"] == "That's a quieter sentence."


@pytest.mark.asyncio
async def test_peak_just_below_threshold_drops(make_asr):
    """Peak of 0.11 (just below 0.12) drops, even for non-stock text."""
    asr = make_asr("Some random transcription.")
    audio = _pcm_bytes(peak=0.11)
    result = await asr.recognize(audio)
    assert result["text"] == ""


# ------------------------------------------------------------------
# Empty input
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_asr_output_returns_empty(make_asr):
    """If the model genuinely returns empty text, we pass through empty."""
    asr = make_asr("")
    audio = _pcm_bytes(peak=0.5)
    result = await asr.recognize(audio)
    assert result["text"] == ""


# ------------------------------------------------------------------
# Normalization function — unit tests (no model load)
# ------------------------------------------------------------------

class TestNormalization:
    """Direct tests on _normalize_for_halluc_match."""

    def test_trailing_period_stripped(self):
        assert _normalize_for_halluc_match("Okay.") == "okay"

    def test_cjk_brackets_stripped(self):
        assert _normalize_for_halluc_match("《大明宫词》") == "大明宫词"

    def test_full_width_punctuation_stripped(self):
        assert _normalize_for_halluc_match("好。") == "好"
        assert _normalize_for_halluc_match("好？") == "好"

    def test_internal_whitespace_collapsed(self):
        assert _normalize_for_halluc_match("Thank   you") == "thank you"

    def test_quotes_stripped(self):
        assert _normalize_for_halluc_match('"Okay"') == "okay"

    def test_known_stock_phrase_normalizes_into_the_set(self):
        """Sanity: every entry in _HALLUCINATION_TEXTS, after normalization,
        must still be matched by the set (round-trip stability). Note
        that "." normalizes to "" — both are in the set, which is fine."""
        for phrase in _HALLUCINATION_TEXTS:
            normalized = _normalize_for_halluc_match(phrase)
            assert normalized in _HALLUCINATION_TEXTS, (
                f"{phrase!r} normalizes to {normalized!r} which is not in the set"
            )


# ------------------------------------------------------------------
# Constants sanity
# ------------------------------------------------------------------

def test_silence_threshold_is_positive_below_one():
    """Sanity check on the empirical constant."""
    assert 0.0 < _SILENCE_PEAK_THRESHOLD < 1.0


def test_hallucination_set_includes_observed_phrases():
    """The phrases observed in 2026-06-02 demo logs must be in the set."""
    assert "the first was the first to be built" in _HALLUCINATION_TEXTS
    assert "大明宫词" in _HALLUCINATION_TEXTS
    assert "thank you" in _HALLUCINATION_TEXTS
    assert "okay" in _HALLUCINATION_TEXTS
    assert "bye" in _HALLUCINATION_TEXTS
