"""
Audio quality analysis module.

Analyzes audio files for:
- SNR (Signal-to-Noise Ratio)
- RMS volume
- Silence ratio
- Clarity score
- Training readiness
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Quality thresholds
SNR_THRESHOLD_DB = 15.0  # dB
RMS_THRESHOLD_DB = -40.0  # dB (minimum)
SILENCE_RATIO_THRESHOLD = 0.8  # 80%
CLARITY_THRESHOLD = 0.6


class AudioQualityAnalyzer:
    """Analyzes audio quality for training readiness."""

    def __init__(self, audio_path: Path):
        self.audio_path = Path(audio_path)
        self._samples: Optional[np.ndarray] = None
        self._sample_rate: Optional[int] = None

    def load_audio(self) -> tuple[np.ndarray, int]:
        """Load audio file and return samples and sample rate."""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(self.audio_path))
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            samples = waveform.squeeze().numpy()
            self._samples = samples
            self._sample_rate = sr
            return samples, sr
        except ImportError:
            # Fallback to pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(self.audio_path))
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            self._samples = samples
            self._sample_rate = audio.frame_rate
            return samples, audio.frame_rate

    @property
    def samples(self) -> np.ndarray:
        """Get audio samples (load if not already loaded)."""
        if self._samples is None:
            self.load_audio()
        return self._samples

    @property
    def sample_rate(self) -> int:
        """Get sample rate (load if not already loaded)."""
        if self._sample_rate is None:
            self.load_audio()
        return self._sample_rate

    def calculate_snr(self) -> float:
        """
        Calculate Signal-to-Noise Ratio in dB.

        Uses spectral subtraction approach: estimates noise from silent portions.
        """
        samples = self.samples
        if len(samples) == 0:
            return 0.0

        # Calculate RMS of entire signal
        signal_rms = np.sqrt(np.mean(samples ** 2))

        # Estimate noise from low-energy portions (bottom 20%)
        abs_samples = np.abs(samples)
        threshold = np.percentile(abs_samples, 20)
        noise_mask = abs_samples < threshold
        noise_samples = samples[noise_mask]

        if len(noise_samples) > 0:
            noise_rms = np.sqrt(np.mean(noise_samples ** 2))
        else:
            noise_rms = np.sqrt(np.mean(samples ** 2)) * 0.1  # Fallback

        # Avoid division by zero
        if noise_rms < 1e-10:
            noise_rms = 1e-10

        snr_db = 20 * np.log10(signal_rms / noise_rms)
        return float(snr_db)

    def calculate_rms_volume(self) -> float:
        """
        Calculate RMS volume in dB.

        Returns dB relative to max possible value (0 dB = clipping).
        """
        samples = self.samples
        if len(samples) == 0:
            return -np.inf

        # Normalize samples to [-1, 1] range if needed
        max_val = np.max(np.abs(samples))
        if max_val > 1.0:
            samples = samples / max_val

        rms = np.sqrt(np.mean(samples ** 2))

        if rms < 1e-10:
            return -96.0  # Effectively silent

        rms_db = 20 * np.log10(rms)
        return float(rms_db)

    def calculate_silence_ratio(self) -> float:
        """
        Calculate ratio of silence in audio.

        Silence = portions below a threshold (relative to RMS).
        """
        samples = self.samples
        if len(samples) == 0:
            return 1.0

        # Calculate RMS
        rms = np.sqrt(np.mean(samples ** 2))

        # Threshold for silence (very quiet relative to RMS, or very low absolute value)
        # If RMS is very low (near zero), treat entire signal as silence
        if rms < 1e-6:
            return 1.0

        silence_threshold = rms * 0.1

        # Count silent samples
        silent_samples = np.abs(samples) < silence_threshold
        silence_ratio = np.mean(silent_samples)

        return float(silence_ratio)

    def calculate_clarity_score(self) -> float:
        """
        Calculate clarity score (0-1).

        Based on:
        - Spectral centroid (brightness)
        - Spectral flatness (tonality vs noise)
        - Zero-crossing rate consistency
        """
        try:
            samples = self.samples
            if len(samples) < 256:
                return 0.5

            # Use FFT to calculate spectral features
            fft = np.fft.rfft(samples)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(samples), 1.0 / self.sample_rate)

            # Spectral centroid (brightness)
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0

            # Spectral flatness (geometric mean / arithmetic mean)
            # Close to 1 = noise-like (flat), close to 0 = tonal
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude) + 1e-10
            flatness = geometric_mean / arithmetic_mean

            # Normalize centroid to 0-1 range (assuming 0-8000 Hz typical speech range)
            centroid_norm = np.clip(centroid / 4000, 0, 1)

            # Clarity: good speech should be somewhat bright but not too flat
            # Mid-range centroid + low flatness = clear speech
            clarity = (centroid_norm * 0.5) + ((1 - flatness) * 0.5)

            return float(np.clip(clarity, 0, 1))

        except Exception as e:
            logger.warning(f"Clarity calculation failed: {e}")
            return 0.5  # Default to medium clarity

    def analyze(self) -> dict:
        """
        Perform full quality analysis.

        Returns:
            dict with snr_db, rms_volume, silence_ratio, clarity_score, training_ready
        """
        logger.info(f"Analyzing audio quality: {self.audio_path}")

        snr_db = self.calculate_snr()
        rms_volume = self.calculate_rms_volume()
        silence_ratio = self.calculate_silence_ratio()
        clarity_score = self.calculate_clarity_score()

        # Determine training readiness
        snr_ok = snr_db >= SNR_THRESHOLD_DB
        rms_ok = rms_volume >= RMS_THRESHOLD_DB
        silence_ok = silence_ratio <= SILENCE_RATIO_THRESHOLD
        clarity_ok = clarity_score >= CLARITY_THRESHOLD

        training_ready = snr_ok and rms_ok and silence_ok and clarity_ok

        results = {
            "snr_db": round(snr_db, 2),
            "rms_volume": round(rms_volume, 2),
            "silence_ratio": round(silence_ratio, 4),
            "clarity_score": round(clarity_score, 4),
            "training_ready": training_ready,
            "quality_warnings": [],
        }

        # Add warnings for specific issues
        if not snr_ok:
            results["quality_warnings"].append(f"SNR too low: {snr_db:.1f} dB (min: {SNR_THRESHOLD_DB} dB)")
        if not rms_ok:
            results["quality_warnings"].append(f"Volume too low: {rms_volume:.1f} dB (min: {RMS_THRESHOLD_DB} dB)")
        if not silence_ok:
            results["quality_warnings"].append(f"Too much silence: {silence_ratio*100:.1f}% (max: {SILENCE_RATIO_THRESHOLD*100}%)")
        if not clarity_ok:
            results["quality_warnings"].append(f"Clarity too low: {clarity_score:.2f} (min: {CLARITY_THRESHOLD})")

        logger.info(f"Quality analysis complete: training_ready={training_ready}, snr={snr_db:.1f}dB, rms={rms_volume:.1f}dB, silence={silence_ratio*100:.1f}%, clarity={clarity_score:.2f}")

        return results


def analyze_segment(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Analyze quality of an audio segment (not from file).

    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        dict with snr_db, rms_volume, silence_ratio, clarity_score, quality_score, quality_flags
    """
    # Create a minimal analyzer without file path
    class SegmentAnalyzer:
        def __init__(self, samples, sample_rate):
            self._samples = samples
            self._sample_rate = sample_rate
            self.samples = samples
            self.sample_rate = sample_rate

    analyzer = SegmentAnalyzer(samples, sample_rate)

    snr_db = _calculate_snr_static(samples)
    rms_volume = _calculate_rms_static(samples)
    silence_ratio = _calculate_silence_static(samples, rms_volume)
    clarity_score = _calculate_clarity_static(samples, sample_rate)

    # Determine quality flags
    quality_flags = {
        "has_overlap": False,  # Would need VAD to detect properly
        "low_energy": rms_volume < RMS_THRESHOLD_DB,
        "high_noise": snr_db < SNR_THRESHOLD_DB,
        "too_short": len(samples) / sample_rate < 1.0,
    }

    # Calculate overall quality score (0-1)
    snr_norm = max(0.0, min(1.0, snr_db / 30.0))  # 0-30 dB normalized
    duration = len(samples) / sample_rate
    duration_norm = 1.0 if 3.0 <= duration <= 30.0 else 0.5 if 1.0 <= duration < 3.0 else 0.0
    clarity_norm = clarity_score
    silence_penalty = max(0, 1.0 - silence_ratio * 2)  # High silence = penalty

    quality_score = (
        clarity_norm * 0.3 +
        snr_norm * 0.3 +
        duration_norm * 0.15 +
        silence_penalty * 0.15 +
        (1.0 if not any(quality_flags.values()) else 0.5) * 0.1  # Bonus if no flags
    )
    quality_score = round(max(0.0, min(1.0, quality_score)), 3)

    # Overall training ready
    training_ready = (
        snr_db >= SNR_THRESHOLD_DB and
        rms_volume >= RMS_THRESHOLD_DB and
        silence_ratio <= SILENCE_RATIO_THRESHOLD and
        clarity_score >= CLARITY_THRESHOLD and
        duration >= 1.0
    )

    return {
        "snr_db": round(snr_db, 2),
        "rms_volume": round(rms_volume, 2),
        "silence_ratio": round(silence_ratio, 4),
        "clarity_score": round(clarity_score, 4),
        "quality_score": quality_score,
        "quality_flags": quality_flags,
        "training_ready": training_ready,
    }


def _calculate_snr_static(samples: np.ndarray) -> float:
    """Calculate SNR for raw samples."""
    if len(samples) == 0:
        return 0.0
    signal_rms = np.sqrt(np.mean(samples ** 2))
    abs_samples = np.abs(samples)
    threshold = np.percentile(abs_samples, 20)
    noise_samples = samples[abs_samples < threshold]
    if len(noise_samples) > 0:
        noise_rms = np.sqrt(np.mean(noise_samples ** 2))
    else:
        noise_rms = np.sqrt(np.mean(samples ** 2)) * 0.1
    if noise_rms < 1e-10:
        noise_rms = 1e-10
    return float(20 * np.log10(signal_rms / noise_rms))


def _calculate_rms_static(samples: np.ndarray) -> float:
    """Calculate RMS volume for raw samples."""
    if len(samples) == 0:
        return -np.inf
    max_val = np.max(np.abs(samples))
    if max_val > 1.0:
        samples = samples / max_val
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-10:
        return -96.0
    return float(20 * np.log10(rms))


def _calculate_silence_static(samples: np.ndarray, rms: float) -> float:
    """Calculate silence ratio for raw samples."""
    if len(samples) == 0:
        return 1.0
    if rms < 1e-6:
        return 1.0
    silence_threshold = rms * 0.1
    silent_samples = np.abs(samples) < silence_threshold
    return float(np.mean(silent_samples))


def _calculate_clarity_static(samples: np.ndarray, sample_rate: int) -> float:
    """Calculate clarity score for raw samples."""
    try:
        if len(samples) < 256:
            return 0.5
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude) + 1e-10
        flatness = geometric_mean / arithmetic_mean
        centroid_norm = np.clip(centroid / 4000, 0, 1)
        clarity = (centroid_norm * 0.5) + ((1 - flatness) * 0.5)
        return float(np.clip(clarity, 0, 1))
    except Exception:
        return 0.5


def analyze_audio(audio_path: Path) -> dict:
    """
    Convenience function to analyze audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        dict with quality metrics
    """
    analyzer = AudioQualityAnalyzer(audio_path)
    return analyzer.analyze()
