"""
Silero VAD (Voice Activity Detection) using ONNX runtime.

Silero VAD is a high-quality voice activity detection model that provides
better accuracy than energy-based VAD, with lower false positive/negative rates.

Model: Silero VAD (snakers4/silero-vad)
- Homepage: https://github.com/snakers4/silero-vad
- Delivers reliable voice activity detection with built-in smoothing.
"""
import os
import struct
import time
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np

from app.logging_config import get_logger

log = get_logger(__name__, component="silero_vad")


@dataclass
class SileroVADConfig:
    """Silero VAD configuration."""
    # Audio settings
    sample_rate: int = 16000  # Silero expects 16kHz
    input_sample_rate: int = 24000  # Our pipeline uses 24kHz

    # Detection threshold (speech probability to consider as talking)
    threshold: float = 0.5

    # Minimum speech duration to register (seconds)
    min_speech_duration: float = 0.1

    # Minimum silence duration after speech to confirm end (seconds)
    min_silence_duration: float = 0.3

    # Window size for smoothing (number of frames)
    smoothing_window: int = 5


class SileroVAD:
    """
    Silero VAD wrapper using ONNX Runtime.

    Implements the same interface as EnergyVAD for drop-in replacement:
    - detect(audio_bytes: bytes) -> (is_committing: bool, confidence: float)
    """

    # Default model version - using public ONNX repo
    # silero_vad_op18_ifless.onnx works with onnxruntime; other variants have shape mismatches
    MODEL_REPO = "istupakov/silero-vad-onnx"
    MODEL_FILE = "silero_vad_op18_ifless.onnx"

    def __init__(
        self,
        sample_rate: int = 24000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.3,
        sensitivity: str = "medium",
    ):
        """
        Initialize Silero VAD.

        Args:
            sample_rate: Input audio sample rate (default 24kHz from pipeline)
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration: Minimum speech duration to register
            min_silence_duration: Minimum silence after speech to trigger commit
            sensitivity: Preset ("low", "medium", "high") - adjusts threshold
        """
        self.sample_rate = sample_rate
        self.input_sr = sample_rate
        self._model_sr = 16000  # Silero internal sample rate

        # Apply sensitivity presets
        if sensitivity == "low":
            threshold = 0.3
            min_silence_duration = 0.5
        elif sensitivity == "high":
            threshold = 0.7
            min_silence_duration = 0.2
        # "medium" uses provided defaults

        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        # ONNX model state (stateful VAD requires passing hidden state)
        self._model = None
        self._session = None
        self._is_loaded = False
        self._state: Optional[np.ndarray] = None  # [2, 1, 128] float32
        self._sr: Optional[np.ndarray] = None     # [1] int64

        # Detection state
        self._speech_probs: list[float] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._last_prob = 0.0

        # Calculate frame counts from durations
        # At 24kHz, one frame = ~60ms chunk (1440 samples)
        self._chunk_samples = int(sample_rate * 0.06)  # 60ms chunks
        self._min_speech_frames = max(1, int(min_speech_duration / 0.06))
        self._min_silence_frames = max(1, int(min_silence_duration / 0.06))

    def _ensure_loaded(self):
        """Lazy load the ONNX model."""
        if self._is_loaded:
            return

        import onnxruntime as ort

        log.info("Loading Silero VAD model...")

        # Try to find cached model
        cache_dir = os.path.expanduser("~/.cache/silero_vad")
        model_path = os.path.join(cache_dir, "silero_vad.onnx")

        if not os.path.exists(model_path):
            os.makedirs(cache_dir, exist_ok=True)
            self._download_model(model_path)

        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options, providers=providers)
        except Exception as e:
            log.warning(f"CUDA provider failed ({e}), falling back to CPU")
            self._session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

        # Initialize state for stateful VAD
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array([self._model_sr], dtype=np.int64)

        self._is_loaded = True
        log.info("Silero VAD model loaded")

    def _download_model(self, model_path: str):
        """Download Silero VAD model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        log.info(f"Downloading Silero VAD model to {model_path}...")

        try:
            local_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILE,
                cache_dir=None,
            )
            import shutil
            shutil.copy(local_path, model_path)
            log.info("Silero VAD model downloaded successfully")
        except Exception as e:
            log.error(f"Failed to download Silero VAD model: {e}")
            raise

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if from_sr == to_sr:
            return audio

        duration = len(audio) / from_sr
        new_length = int(duration * to_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def detect(self, audio_bytes: bytes) -> Tuple[bool, float]:
        """
        Detect speech in audio chunk using Silero VAD.

        Args:
            audio_bytes: Raw PCM audio bytes (16-bit signed, mono)

        Returns:
            Tuple of (is_committing: bool, confidence: float)
            - is_committing: True when VAD detects end of speech phrase
            - confidence: Speech probability from Silero model (0.0-1.0)
        """
        self._ensure_loaded()

        if len(audio_bytes) < 2:
            return False, 0.0

        # Convert bytes to numpy array
        num_samples = len(audio_bytes) // 2
        try:
            samples = struct.unpack(f"{num_samples}h", audio_bytes[:num_samples * 2])
        except struct.error:
            return False, 0.0

        # Normalize to float32 [-1, 1]
        audio_float = np.array(samples, dtype=np.float32) / 32768.0

        # Resample from input_sr to 16kHz if needed
        if self.input_sr != self._model_sr:
            audio_float = self._resample(audio_float, self.input_sr, self._model_sr)

        # Ensure minimum length (30ms = 480 samples at 16kHz)
        min_len = int(0.03 * self._model_sr)
        if len(audio_float) < min_len:
            return False, 0.0

        # Run Silero VAD inference
        # Model expects: (1, num_samples) float32
        audio_input = audio_float[np.newaxis, :].astype(np.float32)

        try:
            # Stateful Silero VAD requires state and sr inputs
            # output[0] = speech probability, output[1] = new state
            output = self._session.run(
                None,
                {"input": audio_input, "state": self._state, "sr": self._sr}
            )
            speech_prob = float(output[0][0, 0])
            self._state = output[1]  # Update state for next call
        except Exception as e:
            log.warning(f"Silero VAD inference failed: {e}")
            return False, 0.0

        self._last_prob = speech_prob

        # Smoothing: maintain rolling window of probabilities
        self._speech_probs.append(speech_prob)
        if len(self._speech_probs) > self._min_speech_frames + self._min_silence_frames:
            self._speech_probs.pop(0)

        # Use smoothed probability for decision
        avg_prob = np.mean(self._speech_probs) if self._speech_probs else speech_prob

        is_speech = avg_prob >= self.threshold

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            return False, avg_prob
        else:
            self._silence_frames += 1
            self._speech_frames = 0

            # Commit if we had enough speech and enough silence after
            if (len(self._speech_probs) >= self._min_speech_frames
                and self._silence_frames >= self._min_silence_frames):
                self.reset()
                return True, avg_prob

            return False, avg_prob

    def reset(self):
        """Reset VAD state for new utterance."""
        self._speech_probs.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._last_prob = 0.0
        if self._state is not None:
            self._state.fill(0.0)

    @property
    def current_probability(self) -> float:
        """Get the last computed speech probability."""
        return self._last_prob

    @property
    def sensitivity_label(self) -> str:
        """Get the current sensitivity label."""
        if self.threshold <= 0.35:
            return "low"
        elif self.threshold >= 0.65:
            return "high"
        return "medium"


# Alias for consistency with BaseVAD naming
SileroVADEngine = SileroVAD
