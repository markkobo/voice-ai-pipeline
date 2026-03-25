"""
VAD (Voice Activity Detection) engine module with sensitivity presets.

Uses RMS energy threshold to detect speech presence.
"""
import struct
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional


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
    """
    Energy-based Voice Activity Detection with configurable sensitivity.

    Supports three sensitivity presets and tracks silence duration
    for automatic utterance commit.
    """

    # Sensitivity presets: (energy_threshold, silence_duration_to_commit, min_speech_duration)
    PRESETS = {
        "low": {
            "energy_threshold": 0.005,
            "silence_duration_to_commit": 2.0,
            "min_speech_duration": 0.5,
        },
        "medium": {
            "energy_threshold": 0.02,
            "silence_duration_to_commit": 1.5,
            "min_speech_duration": 0.3,
        },
        "high": {
            "energy_threshold": 0.05,
            "silence_duration_to_commit": 1.0,
            "min_speech_duration": 0.2,
        },
    }

    def __init__(
        self,
        sample_rate: int = 24000,
        sensitivity: str = "medium",
        energy_threshold: Optional[float] = None,
        silence_duration_to_commit: Optional[float] = None,
        min_speech_duration: Optional[float] = None,
    ):
        """
        Initialize Energy VAD.

        Args:
            sample_rate: Audio sample rate (default: 24kHz)
            sensitivity: Preset name ("low", "medium", "high")
            energy_threshold: Override RMS threshold (0.0-1.0)
            silence_duration_to_commit: Seconds of silence before commit
            min_speech_duration: Minimum speech duration to register
        """
        self.sample_rate = sample_rate
        self._bytes_per_sample = 2  # 16-bit PCM

        # Load preset or use overrides
        preset = self.PRESETS.get(sensitivity, self.PRESETS["medium"])

        self.energy_threshold = (
            energy_threshold
            if energy_threshold is not None
            else preset["energy_threshold"]
        )
        self.silence_duration_to_commit = (
            silence_duration_to_commit
            if silence_duration_to_commit is not None
            else preset["silence_duration_to_commit"]
        )
        self.min_speech_duration = (
            min_speech_duration
            if min_speech_duration is not None
            else preset["min_speech_duration"]
        )

        # Internal state
        self._silence_frames = 0
        self._speech_frames = 0
        self._is_committing = False
        self._last_energy = 0.0

        # Calculate frames from duration
        # Assuming ~60ms of audio per chunk (1440 samples at 24kHz)
        self._chunk_duration_sec = 1440 / self.sample_rate
        self._silence_frames_to_commit = int(
            self.silence_duration_to_commit / self._chunk_duration_sec
        )
        self._min_speech_frames = int(
            self.min_speech_duration / self._chunk_duration_sec
        )

    def detect(self, audio_chunk: bytes) -> Tuple[bool, float]:
        """
        Detect speech using RMS energy.

        Args:
            audio_chunk: Raw PCM audio bytes (16-bit signed)

        Returns:
            Tuple of (is_committing: bool, energy: float)

        Note:
            Returns (True, energy) when VAD decides to COMMIT the utterance
            (silence was detected for enough frames after speech).
            Returns (False, energy) during normal speech detection.
        """
        if len(audio_chunk) < self._bytes_per_sample:
            return False, 0.0

        # Convert bytes to 16-bit signed integers
        num_samples = len(audio_chunk) // self._bytes_per_sample
        # Only use complete sample pairs — drop trailing byte if odd length
        num_samples = num_samples * 2 // 2  # ensure even
        if num_samples == 0:
            return False, 0.0
        try:
            samples = struct.unpack(f"{num_samples}h", audio_chunk[:num_samples * 2])
        except struct.error:
            # Defensive: skip chunks that can't be unpacked
            return False, 0.0

        # Calculate RMS energy
        if num_samples == 0:
            return False, 0.0

        sum_squares = sum(s * s for s in samples)
        rms = (sum_squares / num_samples) ** 0.5
        normalized_rms = rms / 32768.0
        self._last_energy = normalized_rms

        is_speech = normalized_rms >= self.energy_threshold

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            # Don't commit while still speaking
            return False, normalized_rms
        else:
            self._silence_frames += 1
            # Check if we should commit
            # Only commit if we've had enough speech first
            if (
                self._speech_frames >= self._min_speech_frames
                and self._silence_frames >= self._silence_frames_to_commit
            ):
                # Commit the utterance
                self._speech_frames = 0
                self._silence_frames = 0
                return True, normalized_rms
            return False, normalized_rms

    def reset(self):
        """Reset VAD state for new utterance."""
        self._silence_frames = 0
        self._speech_frames = 0
        self._is_committing = False

    @property
    def current_energy(self) -> float:
        """Get the last computed energy level."""
        return self._last_energy

    @property
    def sensitivity_label(self) -> str:
        """Get the current sensitivity preset name."""
        for name, preset in self.PRESETS.items():
            if (
                abs(self.energy_threshold - preset["energy_threshold"]) < 0.001
                and abs(self.silence_duration_to_commit - preset["silence_duration_to_commit"]) < 0.1
            ):
                return name
        return "custom"
