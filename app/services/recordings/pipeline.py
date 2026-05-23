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
from .repository import JsonRecordingsRepository, RecordingsRepository

logger = logging.getLogger(__name__)

# Semaphore to serialize ALL CUDA operations to prevent OOM
# Only one CUDA operation (enhance, diarize, or transcribe) can run at a time
_cuda_lock = threading.Semaphore(1)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds, exponential backoff

# DeepFilterNet — modern DNN denoiser, single-speaker, 48 kHz native.
# Replaced noisereduce (spectral-gating) which was over-aggressive:
# observed 33% peak loss + bandwidth-shaped artifacts on user recordings
# from 2026-05-22. DF is loaded once per process and reused; cost is
# ~0.6s/30s audio on cuda, ~3s/30s on cpu. Threadsafe; uses _cuda_lock.
_df_model = None
_df_state = None
_df_lock = threading.Lock()


def _get_df_denoiser():
    """Lazy-init DeepFilterNet model. Returns (model, df_state) or (None, None)."""
    global _df_model, _df_state
    if _df_model is not None:
        return _df_model, _df_state
    with _df_lock:
        if _df_model is not None:
            return _df_model, _df_state
        try:
            from df.enhance import init_df
            m, s, _ = init_df()
            _df_model, _df_state = m, s
            logger.info(f"[DENOISE] DeepFilterNet ready (sr={s.sr()})")
            return _df_model, _df_state
        except Exception as e:
            logger.warning(f"[DENOISE] DeepFilterNet init failed: {e}; will fall back to noisereduce")
            return None, None


def _loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float = -16.0, peak_limit_dbfs: float = -1.0) -> np.ndarray:
    """LUFS-normalize then peak-limit. Speech-broadcast standard (-16 LUFS).

    Without the peak limit, normalize.loudness() can push dynamic speech
    samples beyond ±1.0 (clipping) — observed 2026-05-23 with raw peak
    1.475 on a quiet input. After LUFS normalization we hard-limit any
    samples above the peak ceiling to prevent that.
    """
    try:
        import pyloudnorm as pyln
        # pyloudnorm needs at least 400ms of audio for integrated loudness
        if len(audio) / sr < 0.4:
            # Too short — just peak-normalize
            peak = float(np.abs(audio).max())
            if peak > 0:
                limit = 10 ** (peak_limit_dbfs / 20)  # -1 dBFS = 0.891
                return audio * (limit / peak)
            return audio
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if loudness == float("-inf") or np.isnan(loudness):
            # Silent input — return as-is
            return audio
        normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
        # Peak limit (hard clip at -1 dBFS to be safe; mostly inaudible for speech)
        limit = 10 ** (peak_limit_dbfs / 20)
        peak = float(np.abs(normalized).max())
        if peak > limit:
            normalized = normalized * (limit / peak)
        return normalized
    except Exception as e:
        logger.warning(f"[NORM] LUFS normalize failed: {e}; falling back to peak-norm")
        peak = float(np.abs(audio).max())
        if peak > 0:
            limit = 10 ** (peak_limit_dbfs / 20)
            return audio * (limit / peak)
        return audio


@dataclass
class ProcessingResult:
    """Result of processing a recording."""
    success: bool
    recording_id: str
    error_message: Optional[str] = None
    metrics: Optional[dict] = None


class AudioProcessingPipeline:
    """Processes audio recordings through the pipeline."""

    def __init__(
        self,
        recording_id: str,
        repository: Optional[RecordingsRepository] = None,
    ):
        """Create a pipeline for one recording.

        ``repository`` is the canonical source of truth for "where is this
        recording_id on disk". When omitted, falls back to a fresh
        :class:`JsonRecordingsRepository` pointing at the configured data
        root — matches how the FastAPI ``BackgroundTasks`` entrypoint
        invokes the pipeline (``run_processing_pipeline(recording_id)``)
        from `app/api/recordings.py`.

        Tests can inject a stub repository to assert lookup behavior in
        isolation. Direct construction of ``RecordingPaths`` here is now
        forbidden — that path used to assign fresh random UUIDs to every
        folder it scanned and silently broke ``_find_recording`` (the
        e0ae9b0 incident).
        """
        self.recording_id = recording_id
        if repository is None:
            from app import config as _cfg
            repository = JsonRecordingsRepository(_cfg.data_root())
        self.repository: RecordingsRepository = repository
        self.paths: Optional[RecordingPaths] = None
        self.metadata: Optional[RecordingMetadata] = None

    def _find_recording(self) -> bool:
        """Resolve recording_id → on-disk paths via the repository.

        Repository ``get_or_none`` is keyed by recording_id (the index is
        the source of truth, with an orphan sweep fallback for folders
        that lost their index entry). When found, we build a pure
        :class:`RecordingPaths` from the recording's ``folder_name`` —
        no UUID assignment, no FS scan, no cache.
        """
        recording = self.repository.get_or_none(self.recording_id)
        if recording is None:
            return False
        self.paths = RecordingPaths(
            folder_name=recording.folder_name,
            listener_id=recording.listener_id,
            persona_id=recording.persona_id,
            recording_id=recording.recording_id,
        )
        self.metadata = RecordingMetadata(self.paths)
        return True

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
        """Run noise reduction using DeepFilterNet (DNN-based, 48 kHz native).

        Falls back to noisereduce (spectral-gating) if DF init fails, then
        to copying raw audio if even that fails.

        Step-name fix 2026-05-23: writes to "denoise" not "enhance"
        (previous code path mis-targeted "enhance" — left "denoise" step
        permanently in_progress in the UI).
        """
        self._log("Step 2: Noise Reduction (DeepFilterNet)")
        self.metadata.update_processing_step("denoise", "in_progress", progress=0)

        start_time = time.time()

        # Try DeepFilterNet first (modern DNN, single-speaker, 48 kHz)
        df_model, df_state = _get_df_denoiser()
        if df_model is not None:
            try:
                import soundfile as sf
                import torch
                import torchaudio
                from df.enhance import enhance

                audio, sample_rate = sf.read(str(self.paths.raw_audio_path))
                audio_np = np.asarray(audio, dtype=np.float32)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                audio_t = torch.from_numpy(audio_np).unsqueeze(0)

                df_sr = df_state.sr()
                if sample_rate != df_sr:
                    self._log(f"Resampling {sample_rate} → {df_sr} Hz for DF")
                    audio_t = torchaudio.functional.resample(audio_t, sample_rate, df_sr)

                with _cuda_lock:
                    self._log(f"Denoising with DeepFilterNet ({df_sr} Hz, {audio_t.shape[1]/df_sr:.1f}s)...")
                    enhanced = enhance(df_model, df_state, audio_t)

                # If we resampled, send back to original SR so downstream
                # stages and the manifest see consistent rates.
                if sample_rate != df_sr:
                    enhanced = torchaudio.functional.resample(enhanced, df_sr, sample_rate)

                enhanced_np = enhanced.squeeze(0).cpu().numpy()

                self.paths.denoised_folder.mkdir(parents=True, exist_ok=True)
                sf.write(str(self.paths.denoised_audio_path), enhanced_np, sample_rate)

                elapsed_ms = int((time.time() - start_time) * 1000)
                self.metadata.update_processing_step(
                    "denoise", "done",
                    progress=100,
                    duration_ms=elapsed_ms,
                )
                self._log(f"DeepFilterNet denoise complete: {elapsed_ms}ms")
                return
            except Exception as e:
                self._log(f"DeepFilterNet failed: {e}; falling back to noisereduce", "WARNING")
                # fall through to noisereduce attempt below

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                import soundfile as sf
                import noisereduce as nr
                import torch

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
                noise_audio = np.concatenate([
                    audio[start:start + frame_length]
                    for i, (start, is_quiet) in enumerate(zip(range(0, len(audio) - frame_length, hop), noise_frames))
                    if is_quiet
                ])
                max_noise_samples = int(3.0 * sample_rate)
                if len(noise_audio) > max_noise_samples:
                    indices = np.linspace(0, len(noise_audio) - 1, max_noise_samples).astype(int)
                    noise_clip = noise_audio[indices]
                else:
                    noise_clip = noise_audio

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._log(f"Denoising with noisereduce on {device}...")

                reduced_noise = nr.reduce_noise(
                    y=audio,
                    sr=sample_rate,
                    y_noise=noise_clip,
                    stationary=False,
                    n_fft=2048,
                    prop_decrease=0.8,
                    device=device,
                )

                self.paths.denoised_folder.mkdir(parents=True, exist_ok=True)
                sf.write(str(self.paths.denoised_audio_path), reduced_noise, sample_rate)

                elapsed_ms = int((time.time() - start_time) * 1000)
                self.metadata.update_processing_step(
                    "denoise", "done",
                    progress=100,
                    duration_ms=elapsed_ms,
                )
                self._log(f"noisereduce fallback complete: {elapsed_ms}ms")
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
                            "denoise", "done",
                            progress=100,
                            duration_ms=elapsed_ms,
                            error_message=f"All denoise paths failed: {type(e).__name__}: {e}",
                        )
                        self._log(f"Denoise fallback: copied raw audio")
                        break
                    except Exception as fallback_error:
                        self._log(f"Denoise fallback also failed: {fallback_error}", "ERROR")
                        raise

    def _run_enhance(self):
        """Enhance stage — now a passthrough copy of denoised → enhanced.

        Sepformer-wsj02mix (the previous implementation) is a 2-speaker
        SEPARATION model trained on WSJ0-2Mix, not a denoiser. On
        single-speaker recordings it either passed audio through
        unchanged or silently fell back to copying denoised when the
        forward errored — bit-identical output either way (verified
        2026-05-22). With DeepFilterNet handling the actual denoising
        upstream, there's nothing meaningful for an additional
        "enhance" stage to do, so we just propagate the denoised file
        forward. Keeping the stage + on-disk folder structure preserves
        downstream callers (diarize reads enhanced > denoised > raw).
        """
        self._log("Step 3: Enhance (passthrough — Sepformer dropped 2026-05-23)")
        self.metadata.update_processing_step("enhance", "in_progress", progress=0)

        start_time = time.time()

        audio_path = self.paths.denoised_audio_path
        if not audio_path.exists():
            audio_path = self.paths.raw_audio_path
            self._log("Denoised audio not found, using raw audio for enhance passthrough")

        try:
            import shutil
            self.paths.enhanced_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(audio_path), str(self.paths.enhanced_audio_path))
            elapsed_ms = int((time.time() - start_time) * 1000)
            self.metadata.update_processing_step(
                "enhance", "done",
                progress=100,
                duration_ms=elapsed_ms,
            )
            self._log(f"Enhance passthrough complete: {elapsed_ms}ms")
            return
        except Exception as e:
            self._log(f"Enhance passthrough failed: {e}", "ERROR")
            self.metadata.update_processing_step(
                "enhance", "failed",
                progress=0,
                error_message=f"passthrough failed: {type(e).__name__}: {e}",
            )
            raise

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

                # Patch torch.load BEFORE pyannote import to handle cached checkpoints
                # that were saved with older PyTorch (contain TorchVersion class)
                _original_torch_load = torch.load
                def _patched_torch_load(*args, **kwargs):
                    # Force weights_only=False since cached checkpoints contain
                    # TorchVersion class from older PyTorch that is not allowlisted
                    kwargs['weights_only'] = False
                    return _original_torch_load(*args, **kwargs)
                torch.load = _patched_torch_load

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
                        use_auth_token=hf_token
                    )

                    # Move to GPU if available
                    if torch.cuda.is_available():
                        pipeline = pipeline.to(torch.device("cuda"))

                    # Run diarization with waveform dict.
                    # Speaker-count hints: if the recording's folder_name
                    # contains "podcast", assume 2 speakers (host + guest).
                    # IG clips are mostly single-speaker. Otherwise let
                    # pyannote auto-detect within a 1-4 range to avoid the
                    # spurious 3rd-speaker classification we observed on
                    # quiet recordings.
                    pipeline_kwargs: dict = {}
                    folder = self.paths.folder_name.lower()
                    if "podcast" in folder:
                        pipeline_kwargs = {"num_speakers": 2}
                    elif "ig" in folder.split("_") or "ig_" in folder:
                        pipeline_kwargs = {"num_speakers": 1}
                    else:
                        pipeline_kwargs = {"min_speakers": 1, "max_speakers": 4}
                    self._log(f"Running diarization (hints={pipeline_kwargs})...")
                    with torch.no_grad():
                        diarization_output = pipeline(audio_dict, **pipeline_kwargs)

                    # pyannote 3.4 returns a plain Annotation. The old
                    # `.exclusive_speaker_diarization` attribute is gone in
                    # 3.x — calling it here silently AttributeError'd on
                    # EVERY recording and the generic except fell through
                    # to the 1-speaker fallback. That's why every podcast
                    # showed `[SPEAKER_00]`.
                    raw_segments = []
                    for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                        raw_segments.append({
                            "speaker_id": speaker,
                            "start_time": turn.start,
                            "end_time": turn.end
                        })

                    # Merge consecutive same-speaker segments with small gaps (<=0.5s)
                    speaker_segments = self._merge_consecutive_segments(raw_segments, gap_threshold=0.5)

                    # If no segments found (diarization succeeded but found no speakers),
                    # create a single segment covering the whole audio for training
                    if len(speaker_segments) == 0:
                        self._log(f"Diarize found 0 speaker segments, creating single whole-audio segment", "WARNING")
                        audio_path = self.paths.enhanced_audio_path
                        if not audio_path.exists():
                            audio_path = self.paths.denoised_audio_path
                        if not audio_path.exists():
                            audio_path = self.paths.raw_audio_path
                        if audio_path.exists():
                            import soundfile as sf
                            info = sf.info(str(audio_path))
                            duration = info.duration
                            speaker_segments = [{
                                "speaker_id": "SPEAKER_00",
                                "start_time": 0.0,
                                "end_time": duration
                            }]
                            self._log(f"Created single segment for whole audio: {duration:.1f}s")
                            self.metadata.update_speaker_segments(speaker_segments)
                        else:
                            self._log(f"Could not find audio file for single segment fallback", "ERROR")
                            self.metadata.update_speaker_segments([])

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
                    # Fallback: no speaker segments found (single-speaker or diarization failed)
                    # Create a single segment covering the whole audio for training purposes
                    self._log(f"Diarize found no speaker segments: {e}", "WARNING")
                    # Find the best available audio to get duration
                    audio_path = self.paths.enhanced_audio_path
                    if not audio_path.exists():
                        audio_path = self.paths.denoised_audio_path
                    if not audio_path.exists():
                        audio_path = self.paths.raw_audio_path

                    if audio_path.exists():
                        import soundfile as sf
                        info = sf.info(str(audio_path))
                        duration = info.duration
                        single_segment = [{
                            "speaker_id": "SPEAKER_00",
                            "start_time": 0.0,
                            "end_time": duration
                        }]
                        self._log(f"Created single segment for whole audio: {duration:.1f}s")
                        self.metadata.update_speaker_segments(single_segment)
                    else:
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
            dropped_ghosts = []
            for speaker_id, audio_chunks in speaker_audio.items():
                speaker_audio_combined = np.concatenate(audio_chunks)

                # Ghost-speaker filter (2026-05-23): diarization on solo
                # recordings hallucinates a second speaker from room noise
                # floor (observed: SPEAKER_01 peak=0.02, 100% silent
                # frames, eff_bw 277 Hz). Drop any "speaker" whose
                # concatenated audio has peak < 0.01 (-40 dBFS) — well
                # below speech, well above realistic room noise.
                peak = float(np.abs(speaker_audio_combined).max()) if speaker_audio_combined.size else 0.0
                if peak < 0.01:
                    self._log(
                        f"Dropping ghost speaker {speaker_id}: peak={peak:.4f} (<0.01) "
                        f"— almost certainly diarization noise, not real speech"
                    )
                    dropped_ghosts.append(speaker_id)
                    continue

                # Loudness-normalize to broadcast-voice standard (-16 LUFS)
                # with a peak limit at -1 dBFS. Without this, recordings
                # captured at low gain come out at -40 dBFS (barely
                # audible), and the trainer reads near-silent audio. The
                # peak limit prevents LUFS normalization from clipping on
                # quiet inputs that need a lot of gain (observed: a -38
                # LUFS input normalized to peak 1.47 — clipped).
                try:
                    normalized = _loudness_normalize(
                        speaker_audio_combined.astype(np.float32),
                        sample_rate,
                        target_lufs=-16.0,
                        peak_limit_dbfs=-1.0,
                    )
                    new_peak = float(np.abs(normalized).max())
                    new_rms = float(np.sqrt((normalized ** 2).mean()))
                    self._log(
                        f"LUFS-normalized {speaker_id}: "
                        f"peak {peak:.3f}→{new_peak:.3f}, "
                        f"rms_db {20*np.log10(max(np.sqrt((speaker_audio_combined**2).mean()), 1e-9)):.1f}→{20*np.log10(max(new_rms, 1e-9)):.1f}"
                    )
                    speaker_audio_combined = normalized
                except Exception as norm_err:
                    self._log(
                        f"Loudness normalize failed for {speaker_id}: {norm_err} "
                        f"— writing unnormalized audio",
                        "WARNING",
                    )

                speaker_path = self.paths.speakers_folder / f"{speaker_id}.wav"
                sf.write(str(speaker_path), speaker_audio_combined, sample_rate, subtype='PCM_16')
                self._log(f"Extracted {speaker_id}: {len(speaker_audio_combined)} samples ({len(audio_chunks)} segments)")

                # Speaker-level duration (combined)
                duration_sec = len(speaker_audio_combined) / sample_rate

                # Voice-cloning audit: bandwidth, clipping, levels. Runs on the
                # concatenated speaker wav (the file the trainer reads). Stored
                # per-segment so the recordings tree can badge each speaker.
                # Cheap (<1s for 15min wav) — no GPU.
                voice_audit = None
                try:
                    from .quality import audit_voice_training_quality
                    voice_audit = audit_voice_training_quality(speaker_path)
                    self._log(
                        f"Voice audit for {speaker_id}: level={voice_audit['level']} "
                        f"eff_bw={voice_audit['metrics'].get('effective_bandwidth_hz', 0):.0f}Hz "
                        f"warnings={len(voice_audit['warnings'])}"
                    )
                except Exception as audit_err:
                    self._log(f"Voice audit failed for {speaker_id}: {audit_err}", "WARNING")

                speaker_audio_data.append({
                    "speaker_id": speaker_id,
                    "duration_seconds": duration_sec,
                    "audio_path": str(speaker_path),
                    "voice_audit": voice_audit,
                    "transcription": self.metadata._data.get("transcription", {}).get("text", ""),
                    "transcription_confidence": self.metadata._data.get("transcription", {}).get("confidence", 0.0),
                })

            # Drop any speaker_segments belonging to ghost speakers. The
            # per-segment loop above writes one entry per diarized turn,
            # including for ghosts. enrich_speaker_segments would happily
            # populate audio_path/voice_audit on those segments even though
            # the wav file was never written → 404s downstream when
            # training tries to resolve them.
            if dropped_ghosts:
                self.metadata._data["speaker_segments"] = [
                    seg for seg in self.metadata._data.get("speaker_segments", [])
                    if seg.get("speaker_id") not in dropped_ghosts
                ]
                self.metadata.save()
                self._log(f"Dropped {len(dropped_ghosts)} ghost speakers: {dropped_ghosts}")

            # Enrich segments with audio path (but NOT duration - we keep per-segment duration)
            self.metadata.enrich_speaker_segments(speaker_audio_data)

            self._log(f"Speaker extraction complete: {len(speaker_audio_data)} real speakers (dropped {len(dropped_ghosts)} ghosts)")

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

        # Free any VRAM held by previous pipeline steps (enhance, diarize).
        # Without this, Whisper large-v3 OOMs on a 24 GB A10G because the
        # pyannote pipeline + speechbrain SepFormer are still in VRAM.
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        for attempt in range(MAX_RETRIES):
            try:
                from faster_whisper import WhisperModel

                # Serialize CUDA operations to prevent OOM from parallel runs
                # Only hold lock during model load + transcription, not during retry sleep
                with _cuda_lock:
                    # Use large-v3 for best Chinese transcription quality.
                    # int8_float16 quantization cuts VRAM ~50% vs float16
                    # with negligible quality loss — keeps headroom for the
                    # other pipeline models still in VRAM.
                    use_cuda = _torch.cuda.is_available()
                    model = WhisperModel(
                        "large-v3",
                        device="cuda" if use_cuda else "cpu",
                        compute_type="int8_float16" if use_cuda else "int8",
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


def run_processing_pipeline(
    recording_id: str,
    repository: Optional[RecordingsRepository] = None,
) -> ProcessingResult:
    """
    Convenience function to run processing pipeline.

    Args:
        recording_id: ID of the recording to process
        repository: optional injected repository (defaults to a fresh
            ``JsonRecordingsRepository`` rooted at ``app.config.data_root()``).

    Returns:
        ProcessingResult with success status and metrics
    """
    pipeline = AudioProcessingPipeline(recording_id, repository=repository)
    return pipeline.run()
