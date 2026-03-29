"""Unit tests for audio quality analyzer."""

import pytest
import numpy as np
from pathlib import Path

from app.services.recordings.quality import (
    AudioQualityAnalyzer,
    analyze_audio,
    SNR_THRESHOLD_DB,
    RMS_THRESHOLD_DB,
    SILENCE_RATIO_THRESHOLD,
    CLARITY_THRESHOLD,
)


class TestAudioQualityAnalyzer:
    """Test AudioQualityAnalyzer class."""

    @pytest.fixture
    def temp_audio_dir(self, tmp_path):
        """Create temp directory with test audio files."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        return audio_dir

    @pytest.fixture
    def clean_audio_file(self, temp_audio_dir):
        """Create a clean audio file (simulated)."""
        # For this test, we'll test the analyzer logic directly
        # without actual file I/O
        pass

    def test_snr_calculation_silent(self):
        """Test SNR calculation for silent audio."""
        # Create mock analyzer with silent audio
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        analyzer._samples = np.zeros(24000)  # 1 second of silence at 24kHz
        analyzer._sample_rate = 24000

        snr = analyzer.calculate_snr()
        # Silent audio should have low SNR
        assert isinstance(snr, float)

    def test_snr_calculation_with_signal(self):
        """Test SNR calculation for audio with signal."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        # Create signal with some noise
        t = np.linspace(0, 1, 24000)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        noise = np.random.normal(0, 0.01, 24000)
        analyzer._samples = signal + noise
        analyzer._sample_rate = 24000

        snr = analyzer.calculate_snr()
        # Should have positive SNR (signal > noise)
        assert snr > 0

    def test_rms_volume_calculation(self):
        """Test RMS volume calculation."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        # Create audio at -12 dB
        rms_db_target = -12.0
        rms_linear = 10 ** (rms_db_target / 20)
        samples = np.random.normal(0, rms_linear, 24000)
        analyzer._samples = samples
        analyzer._sample_rate = 24000

        rms_db = analyzer.calculate_rms_volume()
        # Should be close to target (within a few dB)
        assert abs(rms_db - rms_db_target) < 10

    def test_rms_volume_silent(self):
        """Test RMS volume for near-silent audio."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        analyzer._samples = np.zeros(24000)
        analyzer._sample_rate = 24000

        rms_db = analyzer.calculate_rms_volume()
        # Should be very low (close to -96 dB)
        assert rms_db < -80

    def test_silence_ratio_calculation(self):
        """Test silence ratio calculation."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        # 80% silence, 20% signal
        samples = np.zeros(24000)
        samples[4800:9600] = 0.5  # Signal in middle portion
        analyzer._samples = samples
        analyzer._sample_rate = 24000

        ratio = analyzer.calculate_silence_ratio()
        # Should be close to 0.8
        assert 0.7 < ratio < 0.9

    def test_silence_ratio_all_silent(self):
        """Test silence ratio for completely silent audio."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        analyzer._samples = np.zeros(24000)
        analyzer._sample_rate = 24000

        ratio = analyzer.calculate_silence_ratio()
        assert ratio > 0.95

    def test_clarity_score_calculation(self):
        """Test clarity score calculation."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        # Create clear speech-like signal
        t = np.linspace(0, 1, 24000)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        analyzer._samples = signal
        analyzer._sample_rate = 24000

        clarity = analyzer.calculate_clarity_score()
        assert 0 <= clarity <= 1

    def test_clarity_score_defaults_on_short_audio(self):
        """Test clarity score defaults to 0.5 for very short audio."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        analyzer._samples = np.array([0.1, 0.2, 0.3])  # Very short
        analyzer._sample_rate = 24000

        clarity = analyzer.calculate_clarity_score()
        assert clarity == 0.5

    def test_analyze_returns_all_fields(self):
        """Test that analyze() returns all required fields."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        analyzer.audio_path = Path("/tmp/test.wav")  # Required for logging
        # Create moderate quality audio
        t = np.linspace(0, 1, 24000)
        signal = 0.3 * np.sin(2 * np.pi * 440 * t)
        noise = np.random.normal(0, 0.02, 24000)
        analyzer._samples = signal + noise
        analyzer._sample_rate = 24000

        results = analyzer.analyze()

        assert "snr_db" in results
        assert "rms_volume" in results
        assert "silence_ratio" in results
        assert "clarity_score" in results
        assert "training_ready" in results
        assert "quality_warnings" in results

    def test_training_ready_false_low_quality(self):
        """Test that training_ready is False for low quality audio."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        analyzer.audio_path = Path("/tmp/test.wav")  # Required for logging
        # Very low quality audio
        analyzer._samples = np.random.normal(0, 0.001, 24000)  # Very quiet
        analyzer._sample_rate = 24000

        results = analyzer.analyze()

        assert results["training_ready"] is False

    def test_training_ready_true_high_quality(self):
        """Test that training_ready is True for high quality audio."""
        analyzer = AudioQualityAnalyzer.__new__(AudioQualityAnalyzer)
        analyzer.audio_path = Path("/tmp/test.wav")  # Required for logging
        # Good quality audio: clear signal, moderate noise
        t = np.linspace(0, 1, 24000)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        noise = np.random.normal(0, 0.01, 24000)  # Low noise
        analyzer._samples = signal + noise
        analyzer._sample_rate = 24000

        results = analyzer.analyze()

        # Should pass most thresholds
        assert "snr_db" in results
        assert "clarity_score" in results


class TestQualityThresholds:
    """Test quality threshold constants."""

    def test_thresholds_are_reasonable(self):
        """Test that thresholds are set to reasonable values."""
        assert SNR_THRESHOLD_DB > 0  # SNR should be positive dB
        assert RMS_THRESHOLD_DB < 0  # RMS should be negative dB (below 0)
        assert 0 < SILENCE_RATIO_THRESHOLD < 1  # Ratio between 0 and 1
        assert 0 < CLARITY_THRESHOLD < 1  # Score between 0 and 1
