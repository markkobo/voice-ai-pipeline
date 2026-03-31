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
import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np

from .file_storage import RecordingPaths
from .metadata import RecordingMetadata
from .quality import AudioQualityAnalyzer

logger = logging.getLogger(__name__)

# Semaphore to serialize CUDA transcription and prevent OOM
# Only one transcription can run on CUDA at a time
_cuda_transcribe_lock = threading.Semaphore(1)

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

            # Step 4 & 5: Diarize and Transcribe can run in parallel
            # (both depend only on enhanced audio output)
            # TODO(P2): Use asyncio.gather for parallel execution when
            # actual implementations are added (pyannote + whisper)
            self._run_diarize()
            self._run_transcribe()

            # Step 6: Extract speaker audio files
            self._extract_speakers()

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
        """Run noise reduction using noisereduce.

        Falls back to copying raw audio if processing fails.
        """
        self._log("Step 2: Noise Reduction (noisereduce)")
        self.metadata.update_processing_step("enhance", "in_progress", progress=0)

        start_time = time.time()
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                import soundfile as sf
                import noisereduce as nr
                import numpy as np
                import torch

                # Read audio
                audio, sample_rate = sf.read(str(self.paths.raw_audio_path))

                # Use stationary noise reduction (better for speech)
                # Estimate noise from first 0.5 seconds
                noise_sample = int(0.5 * sample_rate)
                noise_clip = audio[:noise_sample]

                # Denoise using noisereduce
                # Use GPU if available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._log(f"Denoising with noisereduce on {device}...")

                # Stationary noise reduction (frame-level)
                reduced_noise = nr.reduce_noise(
                    y=audio,
                    sr=sample_rate,
                    y_noise=noise_clip,
                    stationary=True,
                    device=device
                )

                # Create denoised folder and save
                self.paths.denoised_folder.mkdir(parents=True, exist_ok=True)
                sf.write(str(self.paths.denoised_audio_path), reduced_noise, sample_rate)

                elapsed_ms = int((time.time() - start_time) * 1000)
                self.metadata.update_processing_step(
                    "enhance", "done",
                    progress=100,
                    duration_ms=elapsed_ms
                )
                self._log(f"Denoise complete: {elapsed_ms}ms, saved to {self.paths.denoised_audio_path}")
                break

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Denoise attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    # Fallback: just copy raw audio as denoised
                    self._log(f"Denoise failed, falling back to raw audio: {e}", "WARNING")
                    try:
                        import shutil
                        self.paths.denoised_folder.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(self.paths.raw_audio_path), str(self.paths.denoised_audio_path))
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        self.metadata.update_processing_step(
                            "enhance", "done",
                            progress=100,
                            duration_ms=elapsed_ms
                        )
                        self._log(f"Denoise fallback: copied raw audio")
                        break
                    except Exception as fallback_error:
                        self._log(f"Denoise fallback also failed: {fallback_error}", "ERROR")
                        raise

    def _run_enhance(self):
        """Run voice enhancement using speechbrain Sepformer.

        Falls back to copying denoised audio if processing fails.
        """
        self._log("Step 3: Voice Enhancement (speechbrain Sepformer)")
        self.metadata.update_processing_step("diarize", "in_progress", progress=0)

        start_time = time.time()
        last_error = None

        # Use denoised audio if available, otherwise use raw
        audio_path = self.paths.denoised_audio_path
        if not audio_path.exists():
            audio_path = self.paths.raw_audio_path
            self._log("Denoised audio not found, using raw audio for enhancement")

        for attempt in range(MAX_RETRIES):
            try:
                import soundfile as sf
                import torch
                from speechbrain.pretrained import SepformerSeparation

                # Load Sepformer model
                self._log("Loading speechbrain Sepformer model...")
                model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wham16k",
                    savedir="pretrained_models/sepformer-wham16k",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
                )

                # Load audio
                audio, sample_rate = sf.read(str(audio_path))
                audio_tensor = torch.from_numpy(audio).float()

                # Separate (enhance)
                self._log("Running speech enhancement...")
                with torch.no_grad():
                    enhanced = model(audio_tensor.unsqueeze(0))
                    enhanced = enhanced.squeeze(0).cpu().numpy()

                # Create enhanced folder and save
                self.paths.enhanced_folder.mkdir(parents=True, exist_ok=True)
                sf.write(str(self.paths.enhanced_audio_path), enhanced, sample_rate)

                elapsed_ms = int((time.time() - start_time) * 1000)
                self.metadata.update_processing_step(
                    "diarize", "done",
                    progress=100,
                    duration_ms=elapsed_ms
                )
                self._log(f"Enhance complete: {elapsed_ms}ms, saved to {self.paths.enhanced_audio_path}")
                break

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Enhance attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    # Fallback: copy denoised as enhanced
                    self._log(f"Enhance failed, falling back to denoised audio: {e}", "WARNING")
                    try:
                        import shutil
                        self.paths.enhanced_folder.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(audio_path), str(self.paths.enhanced_audio_path))
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        self.metadata.update_processing_step(
                            "diarize", "done",
                            progress=100,
                            duration_ms=elapsed_ms
                        )
                        self._log(f"Enhance fallback: copied audio")
                        break
                    except Exception as fallback_error:
                        self._log(f"Enhance fallback also failed: {fallback_error}", "ERROR")
                        raise

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.metadata.update_processing_step(
            "diarize", "done",
            progress=100,
            duration_ms=elapsed_ms
        )

        self._log(f"Enhance complete: {elapsed_ms}ms")

    def _run_diarize(self):
        """Run speaker diarization using pyannote.audio.

        Falls back to empty speaker segments if processing fails.
        """
        self._log("Step 4: Speaker Diarization (pyannote)")
        self.metadata.update_processing_step("transcribe", "in_progress", progress=0)

        start_time = time.time()
        last_error = None

        # Use enhanced audio if available, otherwise use denoised, otherwise use raw
        audio_path = self.paths.enhanced_audio_path
        if not audio_path.exists():
            audio_path = self.paths.denoised_audio_path
        if not audio_path.exists():
            audio_path = self.paths.raw_audio_path

        for attempt in range(MAX_RETRIES):
            try:
                import torch
                from pyannote.audio import Pipeline

                # Load pyannote pipeline
                self._log("Loading pyannote diarization model...")
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    raise ValueError("HF_TOKEN not set in environment")
                self._log(f"Using HF_TOKEN: {hf_token[:8]}...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )

                # Move to GPU if available
                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")

                # Run diarization
                self._log(f"Running diarization on: {audio_path}")
                diarization = pipeline(str(audio_path))

                # Extract speaker segments
                speaker_segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append({
                        "speaker_id": speaker,
                        "start_time": turn.start,
                        "end_time": turn.end
                    })

                elapsed_ms = int((time.time() - start_time) * 1000)

                # Store speaker segments in metadata
                self.metadata.update_speaker_segments(speaker_segments)

                self.metadata.update_processing_step(
                    "transcribe", "done",
                    progress=100,
                    duration_ms=elapsed_ms
                )

                self._log(f"Diarize complete: {elapsed_ms}ms, found {len(speaker_segments)} speaker segments")
                break

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Diarize attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    # Fallback: empty speaker segments
                    self._log(f"Diarize failed, using empty segments: {e}", "WARNING")
                    self.metadata.update_speaker_segments([])

                    elapsed_ms = int((time.time() - start_time) * 1000)
                    self.metadata.update_processing_step(
                        "transcribe", "done",
                        progress=100,
                        duration_ms=elapsed_ms
                    )
                    self._log(f"Diarize fallback: no speaker segments")

    def _extract_speakers(self):
        """Extract audio for each speaker from diarization segments.

        Creates individual WAV files for each speaker in the speakers/ folder.
        Each file is named: SPEAKER_XX.wav
        """
        self._log("Step 6: Extracting speaker audio")

        speaker_segments = self.metadata._data.get("speaker_segments", [])
        if not speaker_segments:
            self._log("No speaker segments to extract")
            return

        # Find the best available audio (enhanced > denoised > raw)
        audio_path = self.paths.enhanced_audio_path
        if not audio_path.exists():
            audio_path = self.paths.denoised_audio_path
        if not audio_path.exists():
            audio_path = self.paths.raw_audio_path

        if not audio_path.exists():
            self._log(f"Audio file not found: {audio_path}")
            return

        try:
            import soundfile as sf

            # Load full audio
            full_audio, sample_rate = sf.read(str(audio_path))
            self._log(f"Loaded audio: {len(full_audio)} samples at {sample_rate}Hz")

            # Create speakers folder
            self.paths.speakers_folder.mkdir(parents=True, exist_ok=True)

            # Group segments by speaker
            speaker_audio = {}
            for seg in speaker_segments:
                speaker_id = seg["speaker_id"]
                start_sample = int(seg["start_time"] * sample_rate)
                end_sample = int(seg["end_time"] * sample_rate)

                if speaker_id not in speaker_audio:
                    speaker_audio[speaker_id] = []
                speaker_audio[speaker_id].append(full_audio[start_sample:end_sample])

            # Save each speaker's audio
            speaker_audio_data = []
            for speaker_id, audio_chunks in speaker_audio.items():
                speaker_audio_combined = np.concatenate(audio_chunks)
                speaker_path = self.paths.speakers_folder / f"{speaker_id}.wav"
                sf.write(str(speaker_path), speaker_audio_combined, sample_rate)
                self._log(f"Extracted {speaker_id}: {len(speaker_audio_combined)} samples ({len(audio_chunks)} segments)")

                # Duration in seconds
                duration_sec = len(speaker_audio_combined) / sample_rate
                # Simple quality proxy: longer audio = more usable (capped at 30s)
                quality_score = min(1.0, duration_sec / 30.0)
                # Training ready: at least 3s of audio
                training_ready = duration_sec >= 3.0 and quality_score >= 0.1

                speaker_audio_data.append({
                    "speaker_id": speaker_id,
                    "duration_seconds": duration_sec,
                    "audio_path": str(speaker_path),
                    "transcription": self.metadata._data.get("transcription", {}).get("text", ""),
                    "transcription_confidence": self.metadata._data.get("transcription", {}).get("confidence", 0.0),
                    "quality_score": round(quality_score, 3),
                    "training_ready": training_ready,
                })

            # Enrich segments with extracted data
            self.metadata.enrich_speaker_segments(speaker_audio_data)

            self._log(f"Speaker extraction complete: {len(speaker_audio)} speakers")

        except Exception as e:
            self._log(f"Speaker extraction failed: {e}", "WARNING")

    def _run_transcribe(self):
        """Run transcription using Whisper (faster-whisper)."""
        self._log("Step 5: Transcription (Whisper)")
        start_time = time.time()

        # Use enhanced audio if available, otherwise use denoised, otherwise use raw
        audio_path = self.paths.enhanced_audio_path
        if not audio_path.exists():
            audio_path = self.paths.denoised_audio_path
        if not audio_path.exists():
            audio_path = self.paths.raw_audio_path

        self._log(f"Transcribing audio from: {audio_path}")

        for attempt in range(MAX_RETRIES):
            try:
                from faster_whisper import WhisperModel

                # Serialize CUDA transcription to prevent OOM from parallel runs
                # Only hold lock during model load + transcription, not during retry sleep
                with _cuda_transcribe_lock:
                    # Use small model for speed, medium for quality
                    # auto-detect GPU and use it if available
                    model = WhisperModel(
                        "medium",  # Model size: tiny/base/small/medium/large
                        device="cuda" if __import__('torch').cuda.is_available() else "cpu",
                        compute_type="float16" if __import__('torch').cuda.is_available() else "int8"
                    )

                    self._log("Whisper model loaded, starting transcription...")

                    # Run transcription
                    segments, info = model.transcribe(
                        str(audio_path),
                        language="zh",  # Chinese
                        beam_size=5,
                        vad_filter=True,  # Voice activity detection filter
                    )

                    # Collect results
                    transcription_segments = []
                    full_text = []
                    total_confidence = 0
                    segment_count = 0

                    for segment in segments:
                        seg_dict = {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip()
                        }
                        transcription_segments.append(seg_dict)
                        full_text.append(seg_dict["text"])
                        if segment.avg_logprob is not None:
                            total_confidence += segment.avg_logprob
                            segment_count += 1

                    transcription_text = "".join(full_text).strip()

                    # Calculate average confidence from log probabilities
                    if segment_count > 0:
                        avg_logprob = total_confidence / segment_count
                        # Convert log probability to confidence (0-1)
                        confidence = min(1.0, max(0.0, np.exp(avg_logprob)))
                    else:
                        confidence = 0.0

                    elapsed_ms = int((time.time() - start_time) * 1000)

                    self.metadata.update_transcription(
                        text=transcription_text,
                        confidence=confidence,
                        segments=transcription_segments
                    )
                    self.metadata.save_transcription_text(transcription_text)

                    self._log(f"Transcription complete: {elapsed_ms}ms, confidence={confidence:.2f}")
                    self._log(f"Transcription text: {transcription_text[:100]}...")

                    # Free GPU memory after transcription
                    del model
                    if __import__('torch').cuda.is_available():
                        __import__('torch').cuda.empty_cache()

                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"Transcribe attempt {attempt + 1} failed, retrying in {sleep_time}s: {e}", "WARNING")
                    time.sleep(sleep_time)
                else:
                    self._log(f"Transcribe failed after {MAX_RETRIES} attempts: {e}", "ERROR")
                    raise


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
