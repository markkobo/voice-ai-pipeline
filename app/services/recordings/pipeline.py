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

# Semaphore to serialize ALL CUDA operations to prevent OOM
# Only one CUDA operation (enhance, diarize, or transcribe) can run at a time
_cuda_lock = threading.Semaphore(1)

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
            # (both depend only on enhanced audio output, both share _cuda_lock anyway)
            t_diarize = threading.Thread(target=self._run_diarize, name="diarize")
            t_transcribe = threading.Thread(target=self._run_transcribe, name="transcribe")
            t_diarize.start()
            t_transcribe.start()
            t_diarize.join()
            t_transcribe.join()

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
                # Estimate noise from the lowest-energy portions of the ENTIRE audio,
                # not just the first 0.5s (which would be wrong if speech starts immediately).
                frame_length = int(0.5 * sample_rate)  # 500ms frames
                hop = int(0.25 * sample_rate)           # 250ms hop (50% overlap)
                energies = []
                for start in range(0, len(audio) - frame_length, hop):
                    frame = audio[start:start + frame_length]
                    energies.append(np.sqrt(np.mean(frame ** 2)))
                energies = np.array(energies)
                # Use the quietest 20% of frames as noise estimate
                energy_threshold = np.percentile(energies, 20)
                noise_frames = energies <= energy_threshold
                # Collect all quiet frames and concatenate
                noise_audio = np.concatenate([
                    audio[start:start + frame_length]
                    for i, (start, is_quiet) in enumerate(zip(range(0, len(audio) - frame_length, hop), noise_frames))
                    if is_quiet
                ])
                # Limit to 3 seconds of noise samples max to avoid dilution
                max_noise_samples = int(3.0 * sample_rate)
                if len(noise_audio) > max_noise_samples:
                    # Uniformly sample 3 seconds from the noise collection
                    indices = np.linspace(0, len(noise_audio) - 1, max_noise_samples).astype(int)
                    noise_clip = noise_audio[indices]
                else:
                    noise_clip = noise_audio

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

                # Acquire CUDA lock to serialize GPU operations
                with _cuda_lock:
                    # Load Sepformer model
                    self._log("Loading speechbrain Sepformer model...")
                    # sepformer-wsj02mix is trained on WSJ0-2Mix (2-speaker clean speech separation)
                    # This is the right model for voice enhancement/denoising of speech.
                    # sepformer-wham-enhancement was wrong — trained on environmental noise datasets.
                    model = SepformerSeparation.from_hparams(
                        source="speechbrain/sepformer-wsj02mix",
                        savedir="pretrained_models/sepformer-wsj02mix",
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

                    # Free GPU memory after enhancement
                    del model
                    del audio_tensor
                    del enhanced
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

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

        Uses waveform dict approach to bypass torchcodec (CUDA 12/13 mismatch).
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
                import soundfile as sf
                from pyannote.audio import Pipeline

                # Acquire CUDA lock to serialize GPU operations
                with _cuda_lock:
                    # Load audio using soundfile (bypasses torchcodec)
                    self._log(f"Loading audio from: {audio_path}")
                    audio, sr = sf.read(str(audio_path))

                    # Convert to (channels, samples) format for pyannote
                    if audio.ndim == 1:
                        audio = audio.reshape(1, -1)  # mono -> (1, samples)
                    else:
                        audio = audio.T  # (samples, channels) -> (channels, samples)

                    waveform = torch.from_numpy(audio).float()
                    audio_dict = {'waveform': waveform, 'sample_rate': sr}

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
                        pipeline = pipeline.to(torch.device("cuda"))

                    # Run diarization with waveform dict
                    self._log("Running diarization...")
                    with torch.no_grad():
                        diarization_output = pipeline(audio_dict)

                    # Use exclusive_speaker_diarization to get merged segments
                    # (removes overlapping speech turns)
                    raw_segments = []
                    for turn, _, speaker in diarization_output.exclusive_speaker_diarization.itertracks(yield_label=True):
                        raw_segments.append({
                            "speaker_id": speaker,
                            "start_time": turn.start,
                            "end_time": turn.end
                        })

                    # Merge consecutive same-speaker segments with small gaps (<=0.5s)
                    speaker_segments = self._merge_consecutive_segments(raw_segments, gap_threshold=0.5)

                    elapsed_ms = int((time.time() - start_time) * 1000)

                    # Store speaker segments in metadata
                    self.metadata.update_speaker_segments(speaker_segments)

                    self.metadata.update_processing_step(
                        "transcribe", "done",
                        progress=100,
                        duration_ms=elapsed_ms
                    )

                    self._log(f"Diarize complete: {elapsed_ms}ms, found {len(speaker_segments)} speaker segments")

                    # Free GPU memory after diarization
                    del pipeline
                    del diarization_output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

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

    def _merge_consecutive_segments(self, segments: list, gap_threshold: float = 0.5) -> list:
        """Merge consecutive segments from the same speaker if gap is <= gap_threshold.

        Parameters
        ----------
        segments : list of dict
            List of {"speaker_id": str, "start_time": float, "end_time": float}
        gap_threshold : float
            Maximum gap in seconds to merge consecutive segments (default: 0.5s)

        Returns
        -------
        list of dict
            Merged segments
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: (x["speaker_id"], x["start_time"]))

        merged = []
        current = sorted_segments[0].copy()

        for seg in sorted_segments[1:]:
            if (seg["speaker_id"] == current["speaker_id"] and
                seg["start_time"] - current["end_time"] <= gap_threshold):
                # Merge: extend current segment
                current["end_time"] = seg["end_time"]
            else:
                # Save current and start new
                merged.append(current)
                current = seg.copy()

        # Don't forget the last one
        merged.append(current)

        return merged

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
            from .quality import analyze_segment

            # Load full audio - soundfile returns (samples,) for mono or (samples, channels) for multi-channel
            full_audio, sample_rate = sf.read(str(audio_path))
            # Ensure 2D: (channels, samples) format
            if full_audio.ndim == 1:
                # Mono: (2700226,) -> (1, 2700226)
                full_audio = full_audio.reshape(1, -1)
            else:
                # Multi-channel: (samples, channels) -> (channels, samples)
                full_audio = full_audio.T
            self._log(f"Loaded audio: {full_audio.shape[1]} samples, {full_audio.shape[0]} channels at {sample_rate}Hz")

            # Create speakers folder
            self.paths.speakers_folder.mkdir(parents=True, exist_ok=True)

            # Calculate quality for each segment AND group by speaker
            speaker_audio = {}
            for i, seg in enumerate(speaker_segments):
                speaker_id = seg["speaker_id"]
                start_sample = int(seg["start_time"] * sample_rate)
                end_sample = int(seg["end_time"] * sample_rate)

                # Get segment audio (channel 0 for mono)
                seg_audio = full_audio[0, start_sample:end_sample]

                # Calculate per-segment quality
                quality_result = analyze_segment(seg_audio, sample_rate)

                # Update segment with quality data
                seg["quality_score"] = quality_result["quality_score"]
                seg["quality_flags"] = quality_result["quality_flags"]
                seg["snr_db"] = quality_result["snr_db"]
                seg["clarity_score"] = quality_result["clarity_score"]
                seg["training_ready"] = quality_result["training_ready"]
                # Keep individual duration (not combined)
                seg["duration_seconds"] = seg["end_time"] - seg["start_time"]

                # Group for speaker audio file creation
                if speaker_id not in speaker_audio:
                    speaker_audio[speaker_id] = []
                speaker_audio[speaker_id].append(seg_audio)

            # Save each speaker's combined audio
            speaker_audio_data = []
            for speaker_id, audio_chunks in speaker_audio.items():
                speaker_audio_combined = np.concatenate(audio_chunks)
                # Ensure correct format for soundfile: (samples,) for mono
                # soundfile expects 2D array (samples, channels) or handles 1D
                speaker_path = self.paths.speakers_folder / f"{speaker_id}.wav"
                sf.write(str(speaker_path), speaker_audio_combined, sample_rate, subtype='PCM_16')
                self._log(f"Extracted {speaker_id}: {len(speaker_audio_combined)} samples ({len(audio_chunks)} segments)")

                # Speaker-level duration (combined)
                duration_sec = len(speaker_audio_combined) / sample_rate

                speaker_audio_data.append({
                    "speaker_id": speaker_id,
                    "duration_seconds": duration_sec,
                    "audio_path": str(speaker_path),
                    "transcription": self.metadata._data.get("transcription", {}).get("text", ""),
                    "transcription_confidence": self.metadata._data.get("transcription", {}).get("confidence", 0.0),
                })

            # Enrich segments with audio path (but NOT duration - we keep per-segment duration)
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

                # Serialize CUDA operations to prevent OOM from parallel runs
                # Only hold lock during model load + transcription, not during retry sleep
                with _cuda_lock:
                    # Use large-v3 for best Chinese transcription quality
                    # faster-whisper large-v3: RTF ~0.006, excellent Chinese fluency
                    model = WhisperModel(
                        "large-v3",
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
                        # Convert log probability to confidence (0-1).
                        # faster-whisper avg_logprob is base-10 log, so use 10^logprob.
                        # np.exp() would be wrong (that's for natural log).
                        confidence = min(1.0, max(0.0, np.power(10.0, avg_logprob)))
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
                    import gc
                    gc.collect()

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
