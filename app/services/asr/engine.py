"""ASR (Automatic Speech Recognition) engine module."""
import asyncio
import logging
import re
import time
import numpy as np
import struct
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# Peak amplitude below which we treat the whole buffer as silence/noise.
# Empirically 0.12 separates AGC'd mic-noise floor (~0.09) from real
# speech (>=0.15 even whispered). See engine.py:158-165 for the demo-day
# observation set. Real-speech-low-AGC is the false-negative risk; we
# accept it because the alternative (letting ASR hallucinations through)
# is worse for downstream LLM behavior.
_SILENCE_PEAK_THRESHOLD = 0.12

# Stock phrases Qwen3-ASR / Whisper-family models emit when fed silent
# audio. Dropped ONLY when peak is also below silence threshold — a loud
# "Okay" or "Thank you" is real speech and must pass through.
_HALLUCINATION_TEXTS = frozenset({
    "the first was the first to be built",
    "the",
    "thank you",
    "thanks for watching",
    "bye",
    "you",
    "thanks",
    "so",
    "okay",
    "ok",
    ".",
    "",
    "大明宫词",
    "小猪佩奇",
})

# Punctuation set used for normalization before comparing against
# _HALLUCINATION_TEXTS. Covers ASCII, CJK fullwidth, and the CJK title
# brackets («》「」『』〈〉【】) that wrap stock phrases like 《大明宫词》.
_HALLUC_NORMALIZE_STRIP = " .!?,;:。！？，；：…《》「」『』〈〉【】（）()[]\"'`"


def _normalize_for_halluc_match(text: str) -> str:
    """Lowercase + strip wrapping punctuation + collapse whitespace.
    Empty result is intentional — caller checks against _HALLUCINATION_TEXTS
    which includes the empty string for the case where ASR returns "."
    style noise tokens."""
    if not text:
        return ""
    stripped = text.strip().strip(_HALLUC_NORMALIZE_STRIP)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.lower()


def set_asr_training_lock(active: bool) -> None:
    """No-op shim for the training-job lock-release path.

    Demo-readiness #2: ``training_service/training_job.py::_release_training_locks``
    imports and calls this after every SFT run. Without the symbol the
    surrounding ``try/except Exception`` swallowed an ``ImportError`` and
    logged a scary "Failed to release ASR lock" warning after every
    successful training run.

    Today there is nothing to release — ASR runs serially per WebSocket
    session and the global ``_cuda_lock`` in the engine module already
    serializes GPU operations across ASR/TTS. This shim exists so the
    training-job release path stays a normal import + call, not an
    ImportError caught by a generic handler. Re-enable into a real lock
    if/when a future architecture needs a coarser "training in progress"
    flag visible to ASR.

    Args:
        active: ignored — kept for API compatibility with the call site.
    """
    logger.debug("set_asr_training_lock(%s) — no-op shim", active)


class BaseASR(ABC):
    """Abstract base class for ASR engines."""

    @abstractmethod
    async def recognize(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Recognize speech from audio bytes.

        Args:
            audio_bytes: Raw PCM audio bytes

        Returns:
            Dict containing recognized text and telemetry
        """
        pass


class Qwen3ASR(BaseASR):
    """Qwen3-ASR engine using the qwen-asr package.

    Requires model to be loaded via from_pretrained().
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-1.7B", latency_ms: int = 0):
        """
        Initialize Qwen3 ASR.

        Args:
            model_name: HuggingFace model name or local path
            latency_ms: Additional simulated latency (for testing)
        """
        self.model_name = model_name
        self.latency_ms = latency_ms
        self._model: Optional[Any] = None
        self._sample_rate = 24000

    def load_model(self) -> None:
        """Load the Qwen3-ASR model."""
        import torch
        from qwen_asr import Qwen3ASRModel
        print(f"Loading Qwen3-ASR model: {self.model_name}...")

        # Determine device
        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.bfloat16
        else:
            device_map = "cpu"
            dtype = torch.float32

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=512
        )
        print("Qwen3-ASR model loaded successfully")

        # Warmup pass — first real transcribe call triggers torch graph capture
        # + kernel launches that take ~10-13s on the A10G. Doing it during
        # startup means the user's first utterance doesn't pay that cost.
        # 0.5 s of silence is enough to trigger the warmup; the result is
        # discarded.
        try:
            silence = np.zeros(int(0.5 * self._sample_rate), dtype=np.float32)
            print("Warming up Qwen3-ASR (first-call latency mitigation)...")
            self._model.transcribe((silence, self._sample_rate))
            print("Qwen3-ASR warmup complete")
        except Exception as e:
            # Warmup is best-effort. If it fails the model still works; the
            # first real call just pays the cold-start cost.
            print(f"Qwen3-ASR warmup skipped: {e}")

    async def recognize(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Recognize speech using Qwen3-ASR.

        Args:
            audio_bytes: Raw PCM 16-bit audio bytes

        Returns:
            Dict with text and telemetry
        """
        if self._model is None:
            self.load_model()

        # Convert bytes to numpy array
        num_samples = len(audio_bytes) // 2  # 16-bit = 2 bytes
        samples = struct.unpack(f"{num_samples}h", audio_bytes)
        audio_array = np.array(samples, dtype=np.float32) / 32768.0  # Normalize to [-1, 1]

        # Debug: check audio stats
        audio_min = float(np.min(audio_array))
        audio_max = float(np.max(audio_array))
        audio_mean = float(np.mean(np.abs(audio_array)))
        print(f"[ASR] audio_bytes={len(audio_bytes)}, num_samples={num_samples}, min={audio_min:.3f}, max={audio_max:.3f}, mean_abs={audio_mean:.3f}")

        # Run inference
        start_time = time.perf_counter()

        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe((audio_array, self._sample_rate))
        )

        inference_time = int((time.perf_counter() - start_time) * 1000)
        inference_time += self.latency_ms

        # Extract text from result
        text = ""
        if result and len(result) > 0:
            text = result[0].text.strip()

        print(f"[ASR] result={result}, text='{text}'")

        # Hallucination filter — Qwen3-ASR (like Whisper-family models) emits
        # a fixed set of stock phrases when fed near-silent audio. The
        # discriminator is **peak amplitude**, not the text. Real loud "Okay"
        # peaks easily above 0.12; the same word emerging from a silent
        # buffer is a hallucination that peaks below 0.12. The known-text
        # set is a secondary signal: when peak is borderline AND the text
        # is in the stock-phrase set, drop. When peak is comfortably above
        # the speech floor, keep — even if the text matches a stock phrase.
        # Empirical observations from 2026-06-02 demo prep:
        #   real "嗯。"     peak=1.000, mean_abs=0.028  → kept
        #   real "Hi I'm Mark..." peak=0.517, mean_abs=0.022 → kept
        #   halluc "The."  peak=0.086, mean_abs=0.008  → drop (peak<0.12)
        #   halluc "《大明宫词》" peak=0.109, mean_abs=0.009 → drop (peak<0.12)
        #   halluc "speed dough..." peak=0.009, mean_abs=0.0005 → drop (peak<0.12)
        audio_peak = max(abs(audio_min), abs(audio_max))
        if text and audio_peak < _SILENCE_PEAK_THRESHOLD:
            stripped = _normalize_for_halluc_match(text)
            if stripped in _HALLUCINATION_TEXTS:
                print(f"[ASR] HALLUCINATION DROPPED: '{text}' (peak={audio_peak:.3f} mean_abs={audio_mean:.4f})")
            else:
                print(f"[ASR] SILENCE DROPPED: '{text}' (peak={audio_peak:.3f} mean_abs={audio_mean:.4f})")
            text = ""

        return {
            "text": text,
            "asr_inference_ms": inference_time
        }


class MockASR(BaseASR):
    """Mock ASR engine for testing and development.

    Simulates inference latency and returns placeholder text.
    """

    def __init__(self, latency_ms: int = 50):
        """
        Initialize Mock ASR.

        Args:
            latency_ms: Simulated inference latency in milliseconds
        """
        self.latency_ms = latency_ms

    async def recognize(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Simulate ASR recognition.

        Args:
            audio_bytes: Raw PCM audio bytes (unused in mock)

        Returns:
            Dict with mock text and telemetry
        """
        # Simulate inference delay
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Return mock result
        return {
            "text": "模擬語音辨識結果...",
            "asr_inference_ms": self.latency_ms
        }
