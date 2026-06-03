"""
Locks the ASR hallucination filter shipped during 06/02 demo prep.

Qwen3-ASR (Whisper-family) emits a fixed set of stock phrases when fed
near-silent audio:
  - "The first was the first to be built."
  - "《大明宫词》"
  - "Thank you.", "Thanks for watching.", "Bye.", "So.", "Okay.", etc.

The filter in app/services/asr/engine.py Qwen3ASR.recognize() drops these
via two paths:
  1. Known-text set (exact match after lowercase + trailing-punct strip).
  2. Peak amplitude floor: max(|min|, |max|) < 0.12 = silence/noise.

These tests don't load the real Qwen3-ASR model — they patch the inference
call to return canned ASR outputs and verify the filter behavior.
"""
import struct

import pytest

from app.services.asr.engine import Qwen3ASR


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
    ],
)
async def test_known_hallucination_phrases_dropped(make_asr, halluc_text):
    """Known stock phrases get dropped regardless of audio peak."""
    asr = make_asr(halluc_text)
    audio = _pcm_bytes(peak=0.5)  # peak high enough that silence guard won't trip
    result = await asr.recognize(audio)
    assert result["text"] == "", (
        f"hallucination {halluc_text!r} should be filtered out"
    )


@pytest.mark.asyncio
async def test_real_speech_text_passes_with_high_peak(make_asr):
    """Plausible real-speech transcription with healthy peak survives."""
    asr = make_asr("Hi, I'm Mark. I build EverHome.")
    audio = _pcm_bytes(peak=0.5)
    result = await asr.recognize(audio)
    assert result["text"] == "Hi, I'm Mark. I build EverHome."


@pytest.mark.asyncio
async def test_silence_peak_drops_any_text(make_asr):
    """Audio with peak<0.12 = silence/noise. Drop whatever ASR returned."""
    asr = make_asr("This is a long sentence ASR thinks it heard.")
    audio = _pcm_bytes(peak=0.05)  # below 0.12 threshold
    result = await asr.recognize(audio)
    assert result["text"] == ""


@pytest.mark.asyncio
async def test_peak_exactly_at_threshold_passes(make_asr):
    """Audio peak of 0.13 (just above 0.12) keeps the ASR output. The
    threshold is intentionally permissive — false-negative silence drops
    are worse than letting a low-peak speech sample through."""
    asr = make_asr("That's a quieter sentence.")
    audio = _pcm_bytes(peak=0.13)
    result = await asr.recognize(audio)
    assert result["text"] == "That's a quieter sentence."


@pytest.mark.asyncio
async def test_empty_asr_output_returns_empty(make_asr):
    """If the model genuinely returns empty text, we pass through empty
    (no exception, no special handling)."""
    asr = make_asr("")
    audio = _pcm_bytes(peak=0.5)
    result = await asr.recognize(audio)
    assert result["text"] == ""
