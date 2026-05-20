"""
Silero VAD (Voice Activity Detection) using ONNX runtime.

Silero VAD is a high-quality voice activity detection model that provides
better accuracy than energy-based VAD, with lower false positive/negative rates.

Model: Silero VAD (snakers4/silero-vad)
- Homepage: https://github.com/snakers4/silero-vad
- Delivers reliable voice activity detection with built-in smoothing.

State-machine design (rewritten 2026-05-19 — see _phase2_followups.md §4
follow-up). The previous implementation averaged speech_prob across a
rolling window, then compared the average to a single threshold for both
SPEECH-enter and SPEECH-exit transitions. That collapsed the per-frame
signal to a value that hovered indecisively around the threshold during
normal Chinese speech with natural inter-word gaps, with three live
failure modes:

  * medium (threshold=0.5): avg sat near 0.5 forever → never crossed →
    barge-in never fired, end-of-speech never committed.
  * high (min_silence=0.2s): below the typical Mandarin stop-consonant
    gap (~250-350ms) — cut speakers off mid-word.
  * low (smoothing tail): inter-word pauses dipped the avg below
    threshold AND filled the silence counter — premature commit at 4
    words.

Fix: per-frame raw probability + hysteresis (separate enter/exit
thresholds) + consecutive-frame counters for state transitions. No
smoothing-average. See `detect()` for the canonical state machine.
"""
import os
import struct
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np

from app.logging_config import get_logger

log = get_logger(__name__, component="silero_vad")


# Module-level state constants. Keeping them at module scope (not Enum)
# so the `int` comparison in detect() stays trivially cheap on the hot
# audio path.
_STATE_SILENCE = 0
_STATE_SPEECH = 1


@dataclass
class SileroVADConfig:
    """Silero VAD configuration.

    `speech_threshold` and `silence_threshold` provide hysteresis: the
    SILENCE→SPEECH transition requires raw_prob >= speech_threshold for
    `min_speech_duration`, while SPEECH→SILENCE requires
    raw_prob < silence_threshold for `min_silence_duration`. Setting
    silence_threshold < speech_threshold prevents flicker when a
    speaker's prob hovers in the ambiguous band.

    The legacy `threshold` field is deprecated — if you only set
    `threshold`, the constructor derives silence_threshold = threshold
    - 0.15 (clamped >= 0.20).
    """
    # Audio settings
    sample_rate: int = 16000  # Silero expects 16kHz
    input_sample_rate: int = 24000  # Our pipeline uses 24kHz

    # Hysteresis thresholds — see class docstring.
    speech_threshold: float = 0.50
    silence_threshold: float = 0.35

    # DEPRECATED — retained for backward-compat. Use speech_threshold.
    threshold: float = 0.50

    # Minimum speech duration to confirm SPEECH state (seconds).
    min_speech_duration: float = 0.15

    # Minimum silence duration after SPEECH to confirm commit (seconds).
    # 0.45s minimum to ride out word-final stop consonants without
    # cutting the speaker off.
    min_silence_duration: float = 0.70


class SileroVAD:
    """
    Silero VAD wrapper using ONNX Runtime.

    Implements the same interface as EnergyVAD for drop-in replacement:
    - detect(audio_bytes: bytes) -> (is_committing: bool, confidence: float)

    See module docstring for the state-machine design rationale.
    """

    # Default model version - using public ONNX repo
    # silero_vad_op18_ifless.onnx works with onnxruntime; other variants have shape mismatches
    MODEL_REPO = "istupakov/silero-vad-onnx"
    MODEL_FILE = "silero_vad_op18_ifless.onnx"

    # Preset table — see _apply_sensitivity. Centralized so tests can
    # introspect what each preset commits to.
    _PRESETS = {
        # (speech_thresh, silence_thresh, min_speech_dur, min_silence_dur)
        # low: forgiving — accepts softer onsets, waits longer to commit.
        "low":    (0.40, 0.30, 0.15, 0.90),
        # medium: default conversation.
        "medium": (0.50, 0.35, 0.15, 0.70),
        # high: fast-paced exchanges — quicker to commit, stricter on
        # speech onset. 0.45s min_silence is still above the typical
        # stop-consonant gap so we don't cut mid-word.
        "high":   (0.60, 0.45, 0.10, 0.45),
    }

    def __init__(
        self,
        sample_rate: int = 24000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.15,
        min_silence_duration: float = 0.70,
        sensitivity: str = "medium",
        speech_threshold: Optional[float] = None,
        silence_threshold: Optional[float] = None,
    ):
        """
        Initialize Silero VAD.

        Args:
            sample_rate: Input audio sample rate (default 24kHz from pipeline)
            threshold: DEPRECATED. Use `speech_threshold` instead. If only
                `threshold` is passed, `silence_threshold` is derived as
                threshold - 0.15 (clamped >= 0.20) to give hysteresis.
            min_speech_duration: Minimum speech duration to register
            min_silence_duration: Minimum silence after speech to trigger commit
            sensitivity: Preset ("low", "medium", "high") — overrides
                the explicit thresholds + durations. Pass None / "" to
                keep the explicit values.
            speech_threshold: Raw-prob threshold for SILENCE→SPEECH.
            silence_threshold: Raw-prob threshold for SPEECH→SILENCE.
                Must be < speech_threshold for hysteresis.
        """
        self.sample_rate = sample_rate
        self.input_sr = sample_rate
        self._model_sr = 16000  # Silero internal sample rate

        # Resolve thresholds in priority order:
        # 1) sensitivity preset (most common UI path)
        # 2) explicit speech_threshold / silence_threshold kwargs
        # 3) legacy `threshold` kwarg with derived silence_threshold
        if sensitivity in self._PRESETS:
            (
                resolved_speech_thresh,
                resolved_silence_thresh,
                min_speech_duration,
                min_silence_duration,
            ) = self._PRESETS[sensitivity]
        else:
            if speech_threshold is not None:
                resolved_speech_thresh = speech_threshold
            else:
                resolved_speech_thresh = threshold

            if silence_threshold is not None:
                resolved_silence_thresh = silence_threshold
            else:
                # Legacy single-threshold callers: derive hysteresis.
                resolved_silence_thresh = max(0.20, resolved_speech_thresh - 0.15)

        # Defensive: silence_threshold must be strictly below
        # speech_threshold or we have no hysteresis.
        if resolved_silence_thresh >= resolved_speech_thresh:
            resolved_silence_thresh = max(
                0.20, resolved_speech_thresh - 0.10
            )

        self.speech_threshold = resolved_speech_thresh
        self.silence_threshold = resolved_silence_thresh
        # Back-compat: `threshold` attribute mirrors speech_threshold so
        # callers reading vad.threshold still see a sensible value.
        self.threshold = resolved_speech_thresh
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        # ONNX model state (stateful VAD requires passing hidden state)
        self._model = None
        self._session = None
        self._is_loaded = False
        self._state: Optional[np.ndarray] = None  # [2, 1, 128] float32
        self._sr: Optional[np.ndarray] = None     # [1] int64

        # State machine state. All previous counters (_speech_probs list,
        # _speech_frames, _silence_frames, _last_prob in their old roles)
        # are gone — the new contract is:
        #   _vad_state       : SILENCE or SPEECH
        #   _consec_above    : consecutive frames with raw_prob >= speech_threshold
        #   _consec_below    : consecutive frames with raw_prob <  silence_threshold
        #   _last_prob       : most recent raw per-frame prob (for current_probability)
        self._vad_state = _STATE_SILENCE
        self._consec_above = 0
        self._consec_below = 0
        self._last_prob = 0.0

        # Frame counts. At 24kHz, one frame = ~60ms chunk (1440 samples)
        # since the WS handler streams 60ms-sized chunks. Compute counts
        # rounded up so durations are a lower bound (e.g. 0.45s → 8
        # frames at 60ms, ceiling).
        self._chunk_samples = int(sample_rate * 0.06)  # 60ms chunks
        self._min_speech_frames = max(
            1, int(np.ceil(min_speech_duration / 0.06))
        )
        self._min_silence_frames = max(
            1, int(np.ceil(min_silence_duration / 0.06))
        )

    def _ensure_loaded(self):
        """Lazy load the ONNX model."""
        if self._is_loaded:
            return

        import onnxruntime as ort

        log.info("Loading Silero VAD model...")

        # Try to find cached model
        cache_dir = os.path.expanduser("~/.cache/silero_vad")
        model_path = os.path.join(cache_dir, "silero_vad.onnx")

        if not os.path.exists(model_path):
            os.makedirs(cache_dir, exist_ok=True)
            self._download_model(model_path)

        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options, providers=providers)
        except Exception as e:
            log.warning(f"CUDA provider failed ({e}), falling back to CPU")
            self._session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

        # Initialize state for stateful VAD
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array([self._model_sr], dtype=np.int64)

        self._is_loaded = True
        log.info("Silero VAD model loaded")

    def _download_model(self, model_path: str):
        """Download Silero VAD model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        log.info(f"Downloading Silero VAD model to {model_path}...")

        try:
            local_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILE,
                cache_dir=None,
            )
            import shutil
            shutil.copy(local_path, model_path)
            log.info("Silero VAD model downloaded successfully")
        except Exception as e:
            log.error(f"Failed to download Silero VAD model: {e}")
            raise

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if from_sr == to_sr:
            return audio

        duration = len(audio) / from_sr
        new_length = int(duration * to_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _run_inference(self, audio_bytes: bytes) -> Optional[float]:
        """
        Run the Silero ONNX model on a single audio chunk and return the
        raw speech probability.

        Returns None if the chunk is too short or inference fails — the
        caller should treat that as "no-op frame" (state unchanged).
        Pulled out as a separate method so the unit tests can monkey-
        patch this to inject scripted probabilities without faking the
        ONNX session.
        """
        self._ensure_loaded()

        if len(audio_bytes) < 2:
            return None

        # Convert bytes to numpy array
        num_samples = len(audio_bytes) // 2
        try:
            samples = struct.unpack(f"{num_samples}h", audio_bytes[:num_samples * 2])
        except struct.error:
            return None

        # Normalize to float32 [-1, 1]
        audio_float = np.array(samples, dtype=np.float32) / 32768.0

        # Resample from input_sr to 16kHz if needed
        if self.input_sr != self._model_sr:
            audio_float = self._resample(audio_float, self.input_sr, self._model_sr)

        # Ensure minimum length (30ms = 480 samples at 16kHz)
        min_len = int(0.03 * self._model_sr)
        if len(audio_float) < min_len:
            return None

        # Run Silero VAD inference
        # Model expects: (1, num_samples) float32
        audio_input = audio_float[np.newaxis, :].astype(np.float32)

        try:
            # Stateful Silero VAD requires state and sr inputs
            # output[0] = speech probability, output[1] = new state
            output = self._session.run(
                None,
                {"input": audio_input, "state": self._state, "sr": self._sr}
            )
            speech_prob = float(output[0][0, 0])
            self._state = output[1]  # Update state for next call
            return speech_prob
        except Exception as e:
            log.warning(f"Silero VAD inference failed: {e}")
            return None

    def detect(self, audio_bytes: bytes) -> Tuple[bool, float]:
        """
        Detect speech in an audio chunk using Silero VAD.

        State machine (see module docstring for rationale):

            SILENCE:
                raw_prob >= speech_threshold → _consec_above += 1
                    if _consec_above >= min_speech_frames:
                        state := SPEECH; reset counters
                else: _consec_above := 0
                returns (is_commit=False, raw_prob)

            SPEECH:
                raw_prob <  silence_threshold → _consec_below += 1
                    if _consec_below >= min_silence_frames:
                        state := SILENCE; reset counters
                        returns (is_commit=True, raw_prob)
                else: _consec_below := 0
                returns (is_commit=False, raw_prob)

        Hysteresis: speech_threshold > silence_threshold prevents
        flicker. Probs in the band (silence_threshold, speech_threshold)
        are "no-op" — they neither push us toward speech-onset nor
        toward commit. The relevant counter ticks down (resets to 0) on
        any frame outside its sustaining range, so we genuinely need
        sustained behavior to flip state.

        Args:
            audio_bytes: Raw PCM audio bytes (16-bit signed, mono)

        Returns:
            Tuple of (is_committing: bool, confidence: float)
            - is_committing: True when VAD detects end of speech phrase
            - confidence: Most recent raw speech probability (0.0-1.0)
        """
        prob = self._run_inference(audio_bytes)
        if prob is None:
            # Chunk too short or inference failed — return last known
            # prob, no state change.
            return False, self._last_prob

        self._last_prob = prob

        if self._vad_state == _STATE_SILENCE:
            if prob >= self.speech_threshold:
                self._consec_above += 1
                if self._consec_above >= self._min_speech_frames:
                    # Confirmed SPEECH onset.
                    self._vad_state = _STATE_SPEECH
                    self._consec_above = 0
                    self._consec_below = 0
                    log.info(
                        f"[VAD] SILENCE→SPEECH prob={prob:.3f} "
                        f"min_silence_frames={self._min_silence_frames}"
                    )
            else:
                self._consec_above = 0
            return False, prob

        # _STATE_SPEECH
        if prob < self.silence_threshold:
            self._consec_below += 1
            if self._consec_below >= self._min_silence_frames:
                # Confirmed end-of-speech — commit.
                log.info(
                    f"[VAD] SPEECH→SILENCE commit prob={prob:.3f} "
                    f"silent_frames={self._consec_below}/{self._min_silence_frames}"
                )
                self._vad_state = _STATE_SILENCE
                self._consec_above = 0
                self._consec_below = 0
                return True, prob
        else:
            # Hysteresis band — speech is still going (or noise floor
            # transient); reset the silence streak.
            self._consec_below = 0
        return False, prob

    def reset(self):
        """Reset VAD state for new utterance.

        After commit, the WS handler calls this so the next utterance
        starts from a clean SILENCE state. We DON'T zero the ONNX hidden
        state — keeping it preserves the model's continuity across the
        utterance boundary (the model was trained on continuous audio).
        """
        self._vad_state = _STATE_SILENCE
        self._consec_above = 0
        self._consec_below = 0
        self._last_prob = 0.0

    @property
    def current_probability(self) -> float:
        """Get the last computed raw speech probability (no smoothing)."""
        return self._last_prob

    @property
    def sensitivity_label(self) -> str:
        """Best-effort reverse lookup of preset name from current
        speech_threshold. The exact value bands map onto the preset
        table, so this is exact for callers that constructed with a
        sensitivity= kwarg; for callers that passed custom thresholds
        we fall back to the closest band."""
        st = self.speech_threshold
        # Exact match first.
        for name, (sp_t, _, _, _) in self._PRESETS.items():
            if abs(st - sp_t) < 1e-6:
                return name
        # Otherwise pick nearest band.
        if st <= 0.45:
            return "low"
        if st >= 0.55:
            return "high"
        return "medium"


# Alias for consistency with BaseVAD naming
SileroVADEngine = SileroVAD
