"""
Audio processing pipeline.

Pipeline steps:
1. Quality Check
2. Noise Reduction (rnnoise)
3. Voice Enhancement (speechbrain)
4. Speaker Diarization (pyannote)
5. Transcription (Whisper)
"""

import logging
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from .file_storage import RecordingPaths
from .metadata import RecordingMetadata
from .quality import AudioQualityAnalyzer

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds, exponential backoff


@dataclass
class ProcessingResult:
    """Result of processing a recording."""
    success: bool
    recording_id: str
    error_message: Optional[str] = None
    metrics: Optional[dict] = None


class AudioProcessingPipeline:
    """Processes audio recordings through the pipeline."""

    def __init__(self, recording_id: str):
        self.recording_id = recording_id
        self.paths: Optional[RecordingPaths] = None
        self.metadata: Optional[RecordingMetadata] = None

    def _find_recording(self) -> bool:
        """Find the recording by ID."""
        from .file_storage import list_all_recordings

        for paths in list_all_recordings():
            if paths.recording_id == self.recording_id:
                self.paths = paths
                self.metadata = RecordingMetadata(paths)
                return True
        return False

    def _log(self, message: str, level: str = "INFO"):
        """Log a message with component identifier."""
        logger.info(f"[PIPELINE:{self.recording_id[:8]}] {message}")

    def _run_with_retry(self, step_fn: Callable, step_name: str) -> any:
        """
        Run a pipeline step with retry logic.

        Args:
            step_fn: The step function to execute
            step_name: Human-readable name for logging

        Returns:
            Result of step_fn

        Raises:
            The last exception if all retries fail
        """
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return step_fn()
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        f"[PIPELINE:{self.recording_id[:8]}] {step_name} failed "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}), retrying in {sleep_time}s: {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"[PIPELINE:{self.recording_id[:8]}] {step_name} failed "
                        f"(all {MAX_RETRIES} attempts): {e}"
                    )
        raise last_error

    def run(self) -> ProcessingResult:
        """
        Run the full processing pipeline.

        Returns:
            ProcessingResult with success status and metrics
        """
        self._log("Starting processing pipeline")

        # Find recording
        if not self._find_recording():
            return ProcessingResult(
                success=False,
                recording_id=self.recording_id,
                error_message="Recording not found"
            )

        try:
            # Step 1: Quality Check
            self._run_quality_check()

            # Step 2: Noise Reduction
            self._run_denoise()

            # Step 3: Voice Enhancement
            self._run_enhance()

            # Step 4: Speaker Diarization
            self._run_diarize()

            # Step 5: Transcription
            self._run_transcribe()

            # Update status to processed
            self.metadata.update_status("processed")

            self._log(f"Pipeline complete. Total: {self.metadata.data['pipeline_metrics']['total_ms']}ms")

            return ProcessingResult(
                success=True,
                recording_id=self.recording_id,
                metrics=self.metadata.data["pipeline_metrics"]
            )

        except Exception as e:
            self._log(f"Pipeline failed: {e}", "ERROR")
            self.metadata.add_error(str(e))
            return ProcessingResult(
                success=False,
                recording_id=self.recording_id,
                error_message=str(e)
            )

    def _run_quality_check(self):
        """Run quality check on raw audio."""
        self._log("Step 1: Quality Check")
        self.metadata.update_processing_step("denoise", "in_progress", progress=0)

        start_time = time.time()

        try:
            analyzer = AudioQualityAnalyzer(self.paths.raw_audio_path)
            quality_metrics = analyzer.analyze()
            self.metadata.update_quality_metrics(quality_metrics)

            elapsed_ms = int((time.time() - start_time) * 1000)
            self.metadata.update_processing_step(
                "denoise", "done",
                progress=100,
                duration_ms=elapsed_ms
            )

            if not quality_metrics["training_ready"]:
                warnings = quality_metrics.get("quality_warnings", [])
                self._log(f"Quality warnings: {warnings}", "WARNING")
            else:
                self._log(f"Quality OK: SNR={quality_metrics['snr_db']}dB, clarity={quality_metrics['clarity_score']}")

        except Exception as e:
            self._log(f"Quality check failed: {e}", "ERROR")
            # Quality check failure is not fatal - continue with pipeline but mark as skipped
            self.metadata.update_processing_step(
                "denoise", "skipped",
                progress=0,
                error_message=str(e)
            )

    def _run_denoise(self):
        """Run noise reduction using rnnoise."""
        self._log("Step 2: Noise Reduction (rnnoise)")
        self.metadata.update_processing_step("enhance", "in_progress", progress=0)

        start_time = time.time()
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                # TODO: Implement actual rnnoise processing
                # For now, just simulate the step
                if self.metadata.data.get("duration_seconds"):
                    # Roughly 1x realtime
                    estimated_ms = int(self.metadata.data["duration_seconds"] * 1000)
                else:
                    estimated_ms = 1000

                time.sleep(estimated_ms / 1000)  # Simulate processing
                break  # Success
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Denoise attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    self._log(f"Denoise failed after {MAX_RETRIES} attempts: {e}", "ERROR")
                    raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.metadata.update_processing_step(
            "enhance", "done",
            progress=100,
            duration_ms=elapsed_ms
        )

        self._log(f"Denoise complete: {elapsed_ms}ms")

    def _run_enhance(self):
        """Run voice enhancement using speechbrain."""
        self._log("Step 3: Voice Enhancement (speechbrain)")
        self.metadata.update_processing_step("diarize", "in_progress", progress=0)

        start_time = time.time()
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                # TODO: Implement actual speechbrain processing
                estimated_ms = 5000  # Placeholder
                time.sleep(estimated_ms / 1000)
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Enhance attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    self._log(f"Enhance failed after {MAX_RETRIES} attempts: {e}", "ERROR")
                    raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.metadata.update_processing_step(
            "diarize", "done",
            progress=100,
            duration_ms=elapsed_ms
        )

        self._log(f"Enhance complete: {elapsed_ms}ms")

    def _run_diarize(self):
        """Run speaker diarization using pyannote."""
        self._log("Step 4: Speaker Diarization (pyannote)")
        self.metadata.update_processing_step("transcribe", "in_progress", progress=0)

        start_time = time.time()
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                # TODO: Implement actual pyannote processing
                estimated_ms = 8000  # Placeholder
                time.sleep(estimated_ms / 1000)
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Diarize attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    self._log(f"Diarize failed after {MAX_RETRIES} attempts: {e}", "ERROR")
                    raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.metadata.update_processing_step(
            "transcribe", "done",
            progress=100,
            duration_ms=elapsed_ms
        )

        self._log(f"Diarize complete: {elapsed_ms}ms")

    def _run_transcribe(self):
        """Run transcription using Whisper."""
        self._log("Step 5: Transcription (Whisper)")
        start_time = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                # TODO: Implement actual Whisper transcription
                estimated_ms = 5000  # Placeholder
                time.sleep(estimated_ms / 1000)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Transcribe attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    self._log(f"Transcribe failed after {MAX_RETRIES} attempts: {e}", "ERROR")
                    raise

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Simulate transcription result
        transcription_text = "這是模擬的轉錄文字稿。"
        self.metadata.update_transcription(
            text=transcription_text,
            confidence=0.95,
            segments=[{"start": 0.0, "end": 5.0, "text": transcription_text}]
        )
        self.metadata.save_transcription_text(transcription_text)

        self._log(f"Transcription complete: {elapsed_ms}ms - '{transcription_text}'")


def run_processing_pipeline(recording_id: str) -> ProcessingResult:
    """
    Convenience function to run processing pipeline.

    Args:
        recording_id: ID of the recording to process

    Returns:
        ProcessingResult with success status and metrics
    """
    pipeline = AudioProcessingPipeline(recording_id)
    return pipeline.run()
