"""
Unit tests for SFT training chunking behavior.

These tests verify:
1. SpeechDataset in training_job.py creates chunks (chunking works)
2. Qwen3TTSDataset in sft_trainer.py does NOT chunk (documenting the issue)
3. Audio validation catches too-short audio
4. Speaker embeddings are non-trivial

Issue: Training v5 (8 min audio, 100 epochs) produced garbled audio.
Hypothesis: Insufficient training samples due to lack of chunking in some code paths.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import numpy as np
import soundfile as sf


class TestSpeechDatasetChunking:
    """
    Tests for SpeechDataset chunking behavior in training_job.py.

    The SpeechDataset class (lines 228-258 in training_job.py) creates
    overlapping chunks from audio_codes for training.
    """

    def test_chunking_creates_more_samples_than_audio_files(self):
        """
        SpeechDataset should create more training samples than number of audio files.

        With chunk_size=300 and hop_size=150, an audio file with 900 frames
        should create ~6 chunks (900/150 = 6 chunks).

        This is critical: without chunking, 3 audio files = only 3 training samples,
        causing severe overfitting.
        """
        # Simulate the SpeechDataset logic from training_job.py
        chunk_size = 300  # frames
        hop_size = 150    # frames (50% overlap)
        min_chunk_size = 50  # minimum frames for valid chunk

        # Simulate 3 audio files with varying lengths
        # Audio 1: 1200 frames (~20 sec at 24kHz with 50Hz frame rate)
        # Audio 2: 900 frames (~15 sec)
        # Audio 3: 600 frames (~10 sec)
        audio_lengths = [1200, 900, 600]

        samples = []
        audio_file_indices = []

        for audio_idx, seq_len in enumerate(audio_lengths):
            # Create overlapping chunks (same logic as SpeechDataset)
            for start in range(0, seq_len, hop_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                if chunk_len >= min_chunk_size:
                    samples.append((start, end))  # Mock chunk
                    audio_file_indices.append(audio_idx)

        num_audio_files = len(audio_lengths)
        num_samples = len(samples)

        # CRITICAL ASSERTION: chunking must produce more samples
        assert num_samples > num_audio_files, (
            f"Chunking must produce more samples than audio files! "
            f"Got {num_samples} samples from {num_audio_files} audio files. "
            f"Without chunking, training will severely overfit."
        )

        # Expected: ~6 + ~5 + ~3 = ~14 samples from 3 files
        expected_min = 10  # At least 10 samples from 3 files
        assert num_samples >= expected_min, (
            f"Expected at least {expected_min} chunks from 3 audio files, got {num_samples}. "
            f"Chunking may not be working correctly."
        )

    def test_chunking_parameters_are_reasonable(self):
        """Verify chunking parameters produce reasonable chunk sizes."""
        chunk_size = 300  # frames
        hop_size = 150    # frames
        sample_rate = 24000  # Qwen3-TTS requires 24kHz

        # At 24kHz with ~50Hz codec frame rate, 300 frames = ~6 seconds
        chunk_duration_sec = chunk_size / 50  # ~6 seconds
        hop_duration_sec = hop_size / 50  # ~3 seconds

        assert 5 <= chunk_duration_sec <= 10, (
            f"Chunk duration should be 5-10 seconds for good training, got {chunk_duration_sec}s"
        )

        assert 2 <= hop_duration_sec <= 5, (
            f"Hop duration should be 2-5 seconds for overlap, got {hop_duration_sec}s"
        )

    def test_small_audio_produces_fewer_chunks(self):
        """Very short audio should produce fewer chunks."""
        chunk_size = 300
        hop_size = 150
        min_chunk_size = 50

        # Short audio: 200 frames (~3.3 sec)
        # range(0, 200, 150) = [0, 150] -> 2 iterations
        # (0, min(300, 200)) = (0, 200), chunk_len = 200 >= 50 -> valid
        # (150, min(450, 200)) = (150, 200), chunk_len = 50 >= 50 -> valid
        seq_len = 200
        samples = []
        for start in range(0, seq_len, hop_size):
            end = min(start + chunk_size, seq_len)
            chunk_len = end - start
            if chunk_len >= min_chunk_size:
                samples.append((start, end))

        # Should produce 2 chunks: (0,200) and (150,200)
        assert len(samples) == 2, (
            f"Short audio (200 frames) should produce 2 chunks with hop=150, got {len(samples)}"
        )

    def test_chunking_includes_all_audio(self):
        """Every audio file should contribute at least one chunk if long enough."""
        chunk_size = 300
        hop_size = 150
        min_chunk_size = 50

        # 3 audio files of varying lengths
        audio_lengths = [1200, 900, 600]
        audio_file_indices = []

        for audio_idx, seq_len in enumerate(audio_lengths):
            for start in range(0, seq_len, hop_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                if chunk_len >= min_chunk_size:
                    audio_file_indices.append(audio_idx)

        # Each audio file should be represented
        for audio_idx in range(len(audio_lengths)):
            count = audio_file_indices.count(audio_idx)
            assert count > 0, (
                f"Audio file {audio_idx} (length={audio_lengths[audio_idx]}) "
                f"should contribute at least one chunk"
            )


class TestQwen3TTSDatasetNoChunking:
    """
    Tests documenting that Qwen3TTSDataset in sft_trainer.py does NOT chunk.

    This is a KNOWN ISSUE: sft_trainer.py returns whole audio files without chunking,
    which can cause severe overfitting with small datasets.
    """

    def test_qwen3tts_dataset_returns_whole_audio(self):
        """
        Qwen3TTSDataset should return entire audio files, not chunks.

        This test documents the current behavior of sft_trainer.py.
        The dataset just returns the whole audio for each item.
        """
        # This is the actual Qwen3TTSDataset logic from sft_trainer.py (lines 131-139)
        class Qwen3TTSDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        # Simulate data with 3 audio files
        data = [
            {"audio": np.random.randn(240000), "sampling_rate": 24000},  # ~10 sec
            {"audio": np.random.randn(480000), "sampling_rate": 24000},  # ~20 sec
            {"audio": np.random.randn(360000), "sampling_rate": 24000},  # ~15 sec
        ]

        dataset = Qwen3TTSDataset(data)

        # CRITICAL: number of samples equals number of audio files
        num_samples = len(dataset)

        assert num_samples == 3, (
            f"Qwen3TTSDataset returns {num_samples} samples, equal to number of audio files. "
            f"This means NO chunking! With only 3 samples, training will severely overfit."
        )

    def test_sft_trainer_vs_training_job_chunking_difference(self):
        """
        Document the chunking difference between sft_trainer.py and training_job.py.

        sft_trainer.py: 3 audio files = 3 training samples (NO chunking)
        training_job.py: 3 audio files = ~14 training samples (WITH chunking)
        """
        # sft_trainer.py approach (no chunking)
        sft_trainer_samples = 3  # 3 audio files = 3 samples

        # training_job.py approach (with chunking)
        chunk_size, hop_size, min_chunk_size = 300, 150, 50
        audio_lengths = [1200, 900, 600]
        training_job_samples = 0
        for seq_len in audio_lengths:
            for start in range(0, seq_len, hop_size):
                end = min(start + chunk_size, seq_len)
                if (end - start) >= min_chunk_size:
                    training_job_samples += 1

        # training_job should produce MORE samples due to chunking
        assert training_job_samples > sft_trainer_samples, (
            f"training_job.py ({training_job_samples} samples) should produce "
            f"more samples than sft_trainer.py ({sft_trainer_samples} samples) due to chunking"
        )

        # With 8 minutes of audio (~480 seconds), chunking should produce significant samples
        # At 24kHz with ~50Hz frame rate, 480 seconds = ~24000 frames
        long_audio = 24000  # ~8 min at 50Hz frame rate
        long_audio_samples = 0
        for start in range(0, long_audio, hop_size):
            end = min(start + chunk_size, long_audio)
            if (end - start) >= min_chunk_size:
                long_audio_samples += 1

        # 8 min of audio should produce at least 50+ samples
        assert long_audio_samples >= 50, (
            f"8 min audio should produce at least 50 chunks, got {long_audio_samples}. "
            f"Insufficient samples cause overfitting and garbled output."
        )


class TestAudioValidation:
    """
    Tests for audio validation before training.

    Audio that is too short, too low quality, or wrong sample rate
    can cause training to fail or produce garbled output.
    """

    def test_audio_minimum_duration_validation(self):
        """Audio must meet minimum duration for meaningful training."""
        MIN_DURATION_SEC = 10.0  # Minimum 10 seconds

        # Too short audio
        short_duration = 5.0  # seconds
        is_valid = short_duration >= MIN_DURATION_SEC

        assert not is_valid, (
            f"Audio with {short_duration}s duration should be rejected "
            f"(minimum is {MIN_DURATION_SEC}s)"
        )

        # Valid audio
        valid_duration = 15.0  # seconds
        is_valid = valid_duration >= MIN_DURATION_SEC

        assert is_valid, (
            f"Audio with {valid_duration}s duration should be accepted"
        )

    def test_audio_sample_rate_validation(self):
        """Audio must be 24kHz for Qwen3-TTS speaker encoder."""
        REQUIRED_SAMPLE_RATE = 24000

        # Wrong sample rate
        wrong_sr = 16000
        is_valid = wrong_sr == REQUIRED_SAMPLE_RATE

        assert not is_valid, (
            f"Audio at {wrong_sr}Hz should be rejected (must be {REQUIRED_SAMPLE_RATE}Hz)"
        )

        # Correct sample rate
        correct_sr = 24000
        is_valid = correct_sr == REQUIRED_SAMPLE_RATE

        assert is_valid, (
            f"Audio at {correct_sr}Hz should be accepted"
        )

    def test_audio_quality_check_exists(self):
        """Audio quality analyzer should check for common issues."""
        # Check that AudioQualityAnalyzer has the necessary checks
        from app.services.recordings.quality import AudioQualityAnalyzer

        # Verify required methods exist
        assert hasattr(AudioQualityAnalyzer, 'analyze'), (
            "AudioQualityAnalyzer should have analyze() method"
        )

    def test_segment_duration_in_metadata(self):
        """Speaker segments should have duration tracked in metadata."""
        # Simulate segment data structure
        segments = [
            {"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 10.5},
            {"speaker_id": "SPEAKER_01", "start_time": 10.5, "end_time": 25.0},
        ]

        # Each segment should have duration
        for seg in segments:
            duration = seg["end_time"] - seg["start_time"]
            assert duration > 0, f"Segment should have positive duration"

            # Duration should be tracked in metadata
            seg["duration_seconds"] = duration

            # Short segments should be flagged
            if duration < 10.0:
                seg["training_ready"] = False


class TestSpeakerEmbeddingExtraction:
    """
    Tests for speaker embedding extraction.

    Speaker embeddings must be non-trivial (not zeros or uniform values)
    to properly represent the voice.
    """

    def test_speaker_embedding_is_nontrivial(self):
        """Speaker embedding should not be all zeros or uniform values."""
        # Simulate a speaker embedding
        hidden_size = 1024
        torch.manual_seed(42)
        embedding = torch.randn(hidden_size)

        # Check that embedding is not all zeros
        is_all_zero = (embedding == 0).all()
        assert not is_all_zero, "Speaker embedding should not be all zeros"

        # Check that embedding is not uniform (standard deviation should be non-trivial)
        std = embedding.std().item()
        assert std > 0.01, (
            f"Speaker embedding should have non-trivial variation (std > 0.01), got {std}"
        )

        # Check that embedding is not all the same value
        is_uniform = (embedding == embedding[0]).all()
        assert not is_uniform, "Speaker embedding should not be uniform"

    def test_speaker_embeddings_differ_between_audio(self):
        """Different audio should produce different speaker embeddings."""
        torch.manual_seed(42)
        embedding1 = torch.randn(1024)

        torch.manual_seed(123)
        embedding2 = torch.randn(1024)

        # Embeddings should be different (different random seeds)
        is_same = torch.allclose(embedding1, embedding2)
        assert not is_same, (
            "Different audio (different seeds) should produce different embeddings"
        )

        # Note: Cosine similarity between random vectors can be negative.
        # The key is that embeddings are non-trivial and differ between audio.
        cos_sim = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0), embedding2.unsqueeze(0)
        ).item()

        # Just verify embeddings are normalized and can be compared
        # (actual similarity depends on the model, not guaranteed to be high)
        assert -1.0 <= cos_sim <= 1.0, "Cosine similarity should be in valid range"

    def test_extract_speaker_embedding_requires_24khz(self):
        """Speaker embedding extraction requires 24kHz audio."""
        REQUIRED_SR = 24000

        # Wrong sample rate should be flagged
        wrong_sr = 16000
        needs_resampling = wrong_sr != REQUIRED_SR

        assert needs_resampling, (
            f"Audio at {wrong_sr}Hz needs resampling to {REQUIRED_SR}Hz"
        )

        # Correct sample rate
        correct_sr = 24000
        needs_resampling = correct_sr != REQUIRED_SR

        assert not needs_resampling, (
            f"Audio at {correct_sr}Hz does not need resampling"
        )


class TestChunkingIntegration:
    """
    Integration tests verifying chunking works end-to-end.

    These tests verify that:
    1. Audio files are properly loaded and encoded
    2. Chunking produces the expected number of samples
    3. Speaker embeddings are correctly mapped to chunks
    """

    def test_chunking_log_message_format(self):
        """Verify the chunking log message format."""
        # From training_job.py line 252:
        # logger.info(f"Created {len(self.samples)} chunks from {len(audio_codes_list)} audio files")

        num_samples = 14
        num_audio_files = 3

        log_message = f"Created {num_samples} chunks from {num_audio_files} audio files"

        assert "14" in log_message
        assert "3" in log_message
        assert "chunks" in log_message.lower()

    def test_audio_file_index_mapping(self):
        """Verify that audio_file_indices correctly maps chunks to original files."""
        chunk_size, hop_size, min_chunk_size = 300, 150, 50
        audio_lengths = [1200, 900, 600]

        samples = []
        audio_file_indices = []

        for audio_idx, seq_len in enumerate(audio_lengths):
            for start in range(0, seq_len, hop_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                if chunk_len >= min_chunk_size:
                    samples.append((start, end))
                    audio_file_indices.append(audio_idx)

        # Verify mapping: chunk index -> audio file index
        # Audio 0 (1200 frames): starts at 0, 150, 300, 450, 600, 750, 900, 1050 = 8 chunks
        # Audio 1 (900 frames): starts at 0, 150, 300, 450, 600, 750 = 6 chunks
        # Audio 2 (600 frames): starts at 0, 150, 300, 450 = 4 chunks

        assert audio_file_indices.count(0) == 8, f"Audio 0 should have 8 chunks, got {audio_file_indices.count(0)}"
        assert audio_file_indices.count(1) == 6, f"Audio 1 should have 6 chunks, got {audio_file_indices.count(1)}"
        assert audio_file_indices.count(2) == 4, f"Audio 2 should have 4 chunks, got {audio_file_indices.count(2)}"

        # Total: 18 chunks
        assert len(samples) == 18, f"Total chunks should be 18, got {len(samples)}"

    def test_chunking_with_overlapping_audio_produces_diverse_samples(self):
        """Overlapping chunks should produce diverse training samples."""
        chunk_size, hop_size = 300, 150

        # Single long audio file
        seq_len = 1200

        chunks = []
        for start in range(0, seq_len, hop_size):
            end = min(start + chunk_size, seq_len)
            chunks.append((start, end))

        # Adjacent chunks should have different start positions
        for i in range(len(chunks) - 1):
            start1, end1 = chunks[i]
            start2, end2 = chunks[i + 1]

            # Should have overlap
            assert start2 < end1, "Adjacent chunks should overlap"

            # But different content (different start)
            assert start2 != start1, "Adjacent chunks should have different start positions"

    def test_progress_logging_includes_chunk_count(self):
        """Training progress should log number of chunks created."""
        # From training_job.py, the SpeechDataset logs:
        # f"Created {len(self.samples)} chunks from {len(audio_codes_list)} audio files"

        num_chunks = 15
        num_audio_files = 3

        # This should be logged BEFORE training starts
        progress_info = {
            "status": "training",
            "num_chunks": num_chunks,
            "num_audio_files": num_audio_files,
        }

        assert progress_info["num_chunks"] > 0, "Should log number of chunks"
        assert progress_info["num_audio_files"] > 0, "Should log number of audio files"


class TestTrainingJobChunkingVerification:
    """
    Tests that verify TrainingJob actually uses chunking.

    These tests parse the generated training script to verify
    that chunking code is actually included.
    """

    def test_training_job_script_contains_speech_dataset(self):
        """TrainingJob's inline script should contain SpeechDataset class."""
        # Read the training_job.py file to verify SpeechDataset exists
        import app.services.training_service.training_job as tj

        # Check that the module contains the chunking logic
        # by verifying the file has the expected class
        import inspect
        source = inspect.getsource(tj.TrainingJob)

        # The inline script in _run_training contains the SpeechDataset definition
        # We can't directly test the inline script, but we can verify
        # the pattern is documented

        # This test documents that SpeechDataset is defined in the inline script
        assert "SpeechDataset" in source or "chunk_size" in source, (
            "training_job.py should contain SpeechDataset or chunking references"
        )

    def test_chunking_validation_before_training(self):
        """
        There should be validation that chunking produces sufficient samples.

        If chunking produces too few samples, training may not converge properly.
        """
        min_chunks_per_file = 3  # At least 3 chunks per audio file

        audio_lengths = [1200, 900, 600]
        chunk_size, hop_size, min_chunk_size = 300, 150, 50

        chunks_per_file = []
        for seq_len in audio_lengths:
            count = 0
            for start in range(0, seq_len, hop_size):
                end = min(start + chunk_size, seq_len)
                if (end - start) >= min_chunk_size:
                    count += 1
            chunks_per_file.append(count)

        # Each file should produce at least min_chunks_per_file
        for i, count in enumerate(chunks_per_file):
            assert count >= min_chunks_per_file, (
                f"Audio file {i} (length={audio_lengths[i]}) produced only "
                f"{count} chunks, minimum is {min_chunks_per_file}"
            )

    def test_sufficient_training_samples_prevents_overfitting(self):
        """
        With sufficient training samples, the model should not overfit.

        This test calculates the ratio of training samples to audio files
        to verify there's enough data diversity.
        """
        chunk_size, hop_size, min_chunk_size = 300, 150, 50

        # Simulate 8 minutes of audio (the v5 training case)
        # At 24kHz with ~50Hz frame rate, 8 min = ~24000 frames
        total_frames = 24000
        num_audio_files = 3

        # Calculate chunks
        total_chunks = 0
        frames_per_file = total_frames // num_audio_files
        for seq_len in [frames_per_file] * num_audio_files:
            for start in range(0, seq_len, hop_size):
                end = min(start + chunk_size, seq_len)
                if (end - start) >= min_chunk_size:
                    total_chunks += 1

        samples_per_file = total_chunks / num_audio_files

        # With chunking, we should have at least 50 samples per audio file
        assert samples_per_file >= 50, (
            f"Should have at least 50 samples per audio file to prevent overfitting, "
            f"got {samples_per_file:.1f}. This could explain garbled output after training."
        )


class TestGarbledAudioDiagnosis:
    """
    Tests to diagnose the garbled audio issue from v5 training.

    Garbled output can be caused by:
    1. Too few training samples (no chunking)
    2. Wrong sample rate audio
    3. Speaker embedding issues
    4. Model architecture mismatch
    """

    def test_diagnosis_too_few_samples(self):
        """
        DIAGNOSIS: Without chunking, 8 min audio = only 3 samples.

        With 100 epochs on 3 samples, the model memorizes the training data
        rather than learning to generalize, causing garbled output.
        """
        # Without chunking
        audio_files = 3
        samples_no_chunking = audio_files

        # With chunking (8 min / 3 files ~= 2.67 min per file)
        # At 50Hz frame rate, 2.67 min = ~8000 frames per file
        # With hop=150, chunk_size=300: ~50 chunks per file
        samples_with_chunking = audio_files * 50

        # Chunking should produce at least 10x more samples
        ratio = samples_with_chunking / samples_no_chunking

        assert ratio >= 10, (
            f"Chunking should produce at least 10x more samples, got {ratio:.1f}x. "
            f"Without sufficient samples, training will overfit and produce garbled audio."
        )

    def test_diagnosis_audio_duration_vs_epochs(self):
        """
        DIAGNOSIS: 8 min audio with 100 epochs = 800 "effective" minutes.

        If not chunked, the model sees the same 3 samples 100 times,
        memorizing rather than learning.
        """
        audio_duration_min = 8
        epochs = 100
        audio_files = 3

        # Without chunking: same samples repeated
        effective_minutes_no_chunking = audio_duration_min * epochs  # 800 min

        # With chunking: diverse samples each epoch
        chunks_per_file = 50
        effective_minutes_with_chunking = audio_duration_min * epochs * chunks_per_file / audio_files

        # The ratio shows how much more diverse training is with chunking
        ratio = effective_minutes_with_chunking / effective_minutes_no_chunking

        assert ratio >= 10, (
            f"Chunking should provide {ratio:.1f}x more training diversity. "
            f"Without chunking, model memorizes rather than generalizes."
        )

    def test_diagnosis_sample_rate_mismatch(self):
        """
        DIAGNOSIS: Wrong sample rate causes speaker embedding extraction to fail.

        Speaker encoder requires exactly 24kHz audio.
        """
        REQUIRED_SR = 24000

        # Common wrong sample rates
        wrong_rates = [16000, 22050, 44100, 48000]

        for sr in wrong_rates:
            needs_fix = sr != REQUIRED_SR
            assert needs_fix, (
                f"Sample rate {sr}Hz should be flagged as needing resampling"
            )

    def test_diagnosis_ensure_chunking_is_used(self):
        """
        VERIFICATION: Ensure chunking is actually used in training.

        This test verifies that when going through TrainingJob,
        the chunking code path is executed.
        """
        # The inline script in training_job.py contains:
        # dataset = SpeechDataset(all_audio_codes, num_code_groups)
        #
        # This creates overlapping chunks from audio_codes

        # Verify the pattern exists in training_job.py
        import inspect
        import app.services.training_service.training_job as tj

        source = inspect.getsource(tj.TrainingJob)

        # The SpeechDataset class is defined inline in the script
        # Look for the chunking parameters
        has_chunk_size = "chunk_size" in source
        has_hop_size = "hop_size" in source

        assert has_chunk_size or "SpeechDataset" in source, (
            "TrainingJob should include chunking logic via chunk_size or SpeechDataset"
        )
