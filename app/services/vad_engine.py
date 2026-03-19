"""VAD (Voice Activity Detection) engine module."""
import struct
from abc import ABC, abstractmethod
from typing import Tuple


class BaseVAD(ABC):
    """Abstract base class for Voice Activity Detection."""

    @abstractmethod
    def detect(self, audio_chunk: bytes) -> Tuple[bool, float]:
        """
        Detect if speech is present in the audio chunk.

        Args:
            audio_chunk: Raw PCM audio bytes (16-bit signed integer)

        Returns:
            Tuple of (is_speaking: bool, energy: float)
        """
        pass


class EnergyVAD(BaseVAD):
    """Energy-based Voice Activity Detection.

    Uses RMS energy threshold to detect speech presence.
    """

    def __init__(self, sample_rate: int = 24000, energy_threshold: float = 0.01):
        """
        Initialize Energy VAD.

        Args:
            sample_rate: Audio sample rate (default: 24kHz)
            energy_threshold: RMS energy threshold for speech detection
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self._bytes_per_sample = 2  # 16-bit PCM

    def detect(self, audio_chunk: bytes) -> Tuple[bool, float]:
        """
        Detect speech using RMS energy.

        Args:
            audio_chunk: Raw PCM audio bytes (16-bit signed)

        Returns:
            Tuple of (is_speaking: bool, energy: float)
        """
        if len(audio_chunk) < self._bytes_per_sample:
            return False, 0.0

        # Convert bytes to 16-bit signed integers
        num_samples = len(audio_chunk) // self._bytes_per_sample
        samples = struct.unpack(f"{num_samples}h", audio_chunk)

        # Calculate RMS energy
        if num_samples == 0:
            return False, 0.0

        # RMS = sqrt(sum(x^2) / n)
        sum_squares = sum(s * s for s in samples)
        rms = (sum_squares / num_samples) ** 0.5

        # Normalize to [-1, 1] range
        normalized_rms = rms / 32768.0

        is_speaking = normalized_rms >= self.energy_threshold

        return is_speaking, normalized_rms
