"""Unit tests for audio processing pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.services.recordings.pipeline import (
    AudioProcessingPipeline,
    run_processing_pipeline,
    ProcessingResult,
)


class TestAudioProcessingPipeline:
    """Test AudioProcessingPipeline class."""

    def test_processing_result_dataclass(self):
        """Test ProcessingResult dataclass."""
        result = ProcessingResult(
            success=True,
            recording_id="test-123",
            metrics={"total_ms": 1000}
        )
        assert result.success is True
        assert result.recording_id == "test-123"
        assert result.error_message is None
        assert result.metrics["total_ms"] == 1000

    def test_processing_result_failure(self):
        """Test ProcessingResult with failure."""
        result = ProcessingResult(
            success=False,
            recording_id="test-123",
            error_message="Test error"
        )
        assert result.success is False
        assert result.error_message == "Test error"

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = AudioProcessingPipeline("test-recording-id")
        assert pipeline.recording_id == "test-recording-id"
        assert pipeline.paths is None
        assert pipeline.metadata is None

    def test_find_recording_not_found(self):
        """Test finding a recording that doesn't exist."""
        with patch('app.services.recordings.file_storage.list_all_recordings', return_value=[]):
            pipeline = AudioProcessingPipeline("nonexistent-id")
            result = pipeline._find_recording()

        assert result is False
        assert pipeline.paths is None
        assert pipeline.metadata is None

    def test_find_recording_found(self):
        """Test finding an existing recording."""
        mock_paths = MagicMock()
        mock_paths.recording_id = "test-123"

        with patch('app.services.recordings.file_storage.list_all_recordings', return_value=[mock_paths]):
            with patch('app.services.recordings.pipeline.RecordingMetadata'):
                pipeline = AudioProcessingPipeline("test-123")
                result = pipeline._find_recording()

        assert result is True
        assert pipeline.paths == mock_paths
        assert pipeline.metadata is not None

    def test_run_recording_not_found(self):
        """Test running pipeline for nonexistent recording."""
        with patch('app.services.recordings.file_storage.list_all_recordings', return_value=[]):
            pipeline = AudioProcessingPipeline("nonexistent-id")
            result = pipeline.run()

        assert result.success is False
        assert result.error_message == "Recording not found"


class TestRunProcessingPipeline:
    """Test run_processing_pipeline convenience function."""

    def test_run_processing_pipeline_creates_pipeline(self):
        """Test that run_processing_pipeline creates and runs pipeline."""
        pipeline = MagicMock()
        pipeline.run.return_value = ProcessingResult(
            success=True,
            recording_id="test-123"
        )

        with patch('app.services.recordings.pipeline.AudioProcessingPipeline', return_value=pipeline):
            result = run_processing_pipeline("test-123")

        assert result.success is True
        pipeline.run.assert_called_once()
