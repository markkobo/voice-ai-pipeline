"""State manager for WebSocket sessions, audio buffers, utterance tracking, and LLM/TTS tasks."""
import asyncio
import uuid
import time
from typing import Optional, Dict, Any

from app.logging_config import get_logger
from app.services.asr import BaseVAD, SileroVAD, EnergyVAD, BaseASR, MockASR, Qwen3ASR
from telemetry import metrics

log = get_logger(__name__)


class AudioConfig:
    """Audio configuration for a session."""
    sample_rate: int = 24000
    channels: int = 1
    format: str = "pcm"


class SessionState:
    """State for a single WebSocket session."""
    session_id: str
    utterance_id: str
    audio_config: AudioConfig
    vad: BaseVAD
    asr: BaseASR
    audio_buffer: bytearray
    is_configured: bool
    start_time: float
    vad_latency_ms: int

    # Persona / listener (M2.2)
    persona_id: Optional[str]
    listener_id: Optional[str]

    # LLM task tracking
    llm_cancellation_event: Optional[asyncio.Event]
    llm_task: Optional[asyncio.Task]
    # Sticky-cancel flag: a cancel that arrives BEFORE set_llm_task is called
    # used to be a silent no-op. We now latch the intent here and the next
    # set_llm_task call honors it.
    llm_pending_cancel: bool

    # TTS task tracking (M1)
    tts_session_id: Optional[str]
    tts_task: Optional[asyncio.Task]
    tts_cancellation_event: Optional[asyncio.Event]

    # LLM model selection
    llm_model: Optional[str]

    # TTS model selection
    tts_model: Optional[str]

    def __init__(
        self,
        session_id: str,
        vad: BaseVAD,
        asr: BaseASR,
    ):
        self.session_id = session_id
        self.utterance_id = str(uuid.uuid4())
        self.audio_config = AudioConfig()
        self.vad = vad
        self.asr = asr
        self.audio_buffer = bytearray()
        self.is_configured = False
        self.start_time = time.time()
        self.vad_latency_ms = 0

        # Persona / listener
        self.persona_id = None
        self.listener_id = None

        # LLM
        self.llm_cancellation_event = None
        self.llm_task = None
        self.llm_pending_cancel = False
        # Utterance sequence stamping for stale-cancel-race fix
        # (review #21 of commit 6c9a87a). Each begin_utterance increments
        # the seq; cancel_llm_task stamps the latch with the current seq;
        # set_llm_task only honors the latch if the seqs match.
        self.llm_utterance_seq: int = 0
        self.llm_pending_cancel_seq: Optional[int] = None

        # TTS
        self.tts_session_id = None
        self.tts_task = None
        self.tts_cancellation_event = None
        self.tts_model: Optional[str] = None  # was missing from __init__ —
        # caused AttributeError → WS disconnect right after asr_result.

        # LLM model selection
        self.llm_model: Optional[str] = None

        # VAD tracking
        self._vad_had_speech = False  # True if we detected speech in this utterance
        self._vad_committed = False   # True if we've already committed this utterance

        # Model
        self.llm_model = None


class StateManager:
    """Manages WebSocket sessions, audio buffers, utterance tracking, and LLM/TTS tasks."""

    def __init__(self, vad: Optional[BaseVAD] = None, asr: Optional[BaseASR] = None, use_qwen: bool = True):
        """
        Initialize StateManager.

        Args:
            vad: VAD engine instance (defaults to EnergyVAD).
                 Switched from SileroVAD to EnergyVAD 2026-05-20 — the
                 Silero ONNX model returned ~0 probability on healthy
                 speech audio (RMS up to 0.15, prob 0.001) even after
                 a hysteresis-state-machine rewrite, sample-rate fix,
                 client noise-gate removal, AGC enable, and 512-sample
                 windowing. Root cause inside Silero never narrowed
                 down — but per-chunk RMS in the diag log is wildly
                 distinguishable between speech and silence on real
                 audio, so a deterministic energy-threshold VAD is the
                 reliable demo path. SileroVAD code is preserved for
                 later return.
            asr: ASR engine instance (defaults to Qwen3ASR or MockASR)
            use_qwen: If True and no asr provided, use Qwen3ASR
        """
        self._sessions: Dict[str, SessionState] = {}
        self._default_vad = vad or EnergyVAD()

        if asr is not None:
            self._default_asr = asr
        elif use_qwen:
            self._default_asr = Qwen3ASR()
        else:
            self._default_asr = MockASR()

    def create_session(self, session_id: str, vad: Optional[BaseVAD] = None,
                       asr: Optional[BaseASR] = None) -> SessionState:
        """Create a new session."""
        state = SessionState(
            session_id=session_id,
            vad=vad or self._default_vad,
            asr=asr or self._default_asr,
        )
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session state by ID."""
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove a session after cancelling any active LLM + TTS tasks."""
        state = self._sessions.get(session_id)
        if state:
            self.cancel_llm_task(session_id, origin="remove_session")
            self.cancel_tts_task(session_id)
        self._sessions.pop(session_id, None)

    def update_config(self, session_id: str, config: Dict[str, Any]) -> bool:
        """
        Update session configuration.

        Args:
            session_id: Session identifier
            config: Config dict with audio settings, persona_id, listener_id, vad, etc.

        Returns:
            True if config was applied, False otherwise
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        # Audio config
        audio_config = config.get("audio", {})
        state.audio_config.sample_rate = audio_config.get("sample_rate", 24000)
        state.audio_config.channels = audio_config.get("channels", 1)
        state.audio_config.format = audio_config.get("format", "pcm")

        # VAD sensitivity - use provided value or preserve existing.
        # 2026-05-20: switched from SileroVAD to EnergyVAD — see __init__
        # comment block. EnergyVAD has the same (is_commit, energy)
        # interface and the same low/medium/high preset names.
        vad_sensitivity = config.get("vad")
        if vad_sensitivity:
            state.vad = EnergyVAD(
                sample_rate=state.audio_config.sample_rate,
                sensitivity=vad_sensitivity,
                adaptive=False,  # avoid the speech-during-calibration trap
            )
        else:
            # Recreate VAD with new sample rate, preserve sensitivity
            current_sensitivity = getattr(state.vad, "sensitivity_label", "medium")
            state.vad = EnergyVAD(
                sample_rate=state.audio_config.sample_rate,
                sensitivity=current_sensitivity,
                adaptive=False,
            )

        # Persona / listener
        if "persona_id" in config:
            state.persona_id = config.get("persona_id")
        if "listener_id" in config:
            state.listener_id = config.get("listener_id")

        # LLM model
        if "model" in config:
            state.llm_model = config.get("model")

        # TTS model
        if "tts_model" in config:
            state.tts_model = config.get("tts_model")

        state.is_configured = True
        return True

    def update_vad_sensitivity(self, session_id: str, sensitivity: str) -> bool:
        """
        Update VAD sensitivity for a session.

        Args:
            session_id: Session identifier
            sensitivity: "low", "medium", or "high"

        Returns:
            True if updated, False if session not found
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        state.vad = SileroVAD(
            sample_rate=state.audio_config.sample_rate,
            sensitivity=sensitivity,
        )
        return True

    def add_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """
        Add audio chunk to the session buffer without VAD processing.
        Called when client sends accumulated audio before commit_utterance.

        Args:
            session_id: Session identifier
            audio_chunk: Binary PCM audio data
        """
        state = self._sessions.get(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")
        if not state.is_configured:
            raise RuntimeError("Session not configured")
        state.audio_buffer.extend(audio_chunk)

    def process_audio(self, session_id: str, audio_chunk: bytes) -> Optional[Dict[str, Any]]:
        """
        Process audio chunk through VAD for continuous monitoring.

        SileroVAD.detect() returns (is_commit, prob):
          - is_commit=True  → silence after sufficient speech → end of utterance
          - is_commit=False → currently in speech or pre-speech silence

        Returns:
            None if speech continues, dict if VAD commits (auto-send)
        """
        state = self._sessions.get(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        if not state.is_configured:
            raise RuntimeError("Session not configured")

        # Run VAD detection
        is_commit, avg_prob = state.vad.detect(audio_chunk)

        # DIAGNOSTIC (remove after VAD investigation): per-chunk prob.
        # Logs the first 30 chunks at INFO so we can see what Silero is
        # actually producing without spamming logs forever.
        if not hasattr(state, "_vad_diag_count"):
            state._vad_diag_count = 0
        if state._vad_diag_count < 30:
            import struct as _struct
            n_samples = len(audio_chunk) // 2
            if n_samples > 0:
                samples = _struct.unpack(f"{n_samples}h", audio_chunk[:n_samples * 2])
                rms = (sum(s * s for s in samples) / n_samples) ** 0.5 / 32768.0
            else:
                rms = 0.0
            log.info(
                f"[VAD-DIAG] chunk={state._vad_diag_count} "
                f"bytes={len(audio_chunk)} samples={n_samples} "
                f"rms={rms:.4f} prob={avg_prob:.3f} commit={is_commit}"
            )
            state._vad_diag_count += 1

        # Accumulate audio
        state.audio_buffer.extend(audio_chunk)

        # Track speech state. Edge-triggered: `_vad_had_speech` flips False→True
        # on the FIRST speech frame of this utterance; we use that edge to
        # decide whether this chunk is a barge-in candidate. Previously the
        # barge-in branch was level-triggered (fired on every speech frame
        # of the utterance) which spammed `cancel_llm_task` and latched
        # `pending_cancel` repeatedly — see open issue #1 in
        # tests/_phase2_followups.md §5: "LLM cancelled 3s into stream with
        # empty partial_text". A later barge-in stamp landing at the new
        # utterance's seq (via late-arriving PCM frames after
        # `begin_utterance`) would fire the cancellation_event on the
        # fresh LLM task, killing it before any token streamed.
        speech_started_this_chunk = (not is_commit) and (not state._vad_had_speech)
        if not is_commit:
            state._vad_had_speech = True

        # Barge-in: user starts speaking WHILE an LLM/TTS response is in
        # flight. Three guards (see open issue #1):
        #   1. Edge-triggered: only on transition silence→speech
        #      (`speech_started_this_chunk`). No more level-trigger spam.
        #   2. There must be an actual in-flight LLM task to barge in on
        #      (`state.llm_task is not None`). Without this, the very first
        #      utterance of a session — where no LLM has ever run — latches
        #      a stale `pending_cancel` that the next utterance picks up.
        #   3. Not already committed for this utterance (prevents firing
        #      after VAD already saw end-of-speech).
        llm_is_active = (
            state.llm_task is not None and not state.llm_task.done()
        )
        if (
            speech_started_this_chunk
            and llm_is_active
            and not state._vad_committed
        ):
            log.info(f"[VAD] Barge-in detected (new speech in active utterance)")
            self.cancel_llm_task(session_id, origin="vad_barge_in")
            self.cancel_tts_task(session_id)

        # End-of-speech: silence after sufficient speech → commit (auto-send)
        if is_commit and state._vad_had_speech and not state._vad_committed:
            log.info(f"[VAD] End of speech (silence detected), committing")
            state._vad_committed = True  # Prevent double-commit
            state.vad.reset()  # Reset VAD for next utterance
            state._vad_had_speech = False
            state.utterance_id = str(uuid.uuid4())  # New ID for next utterance
            return {
                "type": "vad_commit",
                "utterance_id": state.utterance_id,
                "energy": avg_prob,
                "telemetry": {"vad_latency_ms": state.vad_latency_ms}
            }

        return None

    async def commit_utterance(self, session_id: str) -> Dict[str, Any]:
        """
        Finalize current utterance and run ASR.

        Args:
            session_id: Session identifier

        Returns:
            Final ASR result dict
        """
        state = self._sessions.get(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        if not state.audio_buffer:
            print(f"[commit_utterance] WARNING: audio_buffer is empty for session {session_id}")
            result = {
                "type": "asr_result",
                "utterance_id": state.utterance_id,
                "is_final": True,
                "text": "",
                "telemetry": {
                    "vad_latency_ms": state.vad_latency_ms,
                    "asr_inference_ms": 0
                }
            }
            self._reset_utterance(session_id)
            return result

        print(f"[commit_utterance] session={session_id}, audio_buffer len={len(state.audio_buffer)} bytes, samples={len(state.audio_buffer)//2}")

        # Run ASR on accumulated audio with telemetry
        audio_data = bytes(state.audio_buffer)
        asr_start = time.perf_counter()
        asr_result = await state.asr.recognize(audio_data)
        asr_latency = time.perf_counter() - asr_start

        model_name = getattr(state.asr, 'model_name', 'mock')
        metrics.asr_latency.labels(
            component="asr",
            model=model_name
        ).observe(asr_latency)

        result = {
            "type": "asr_result",
            "utterance_id": state.utterance_id,
            "is_final": True,
            "text": asr_result["text"],
            "telemetry": {
                "vad_latency_ms": state.vad_latency_ms,
                "asr_inference_ms": asr_result["asr_inference_ms"]
            }
        }

        self._reset_utterance(session_id)
        return result

    # -------------------------------------------------------------------------
    # LLM Task Management
    # -------------------------------------------------------------------------

    def cancel_llm_task(self, session_id: str, origin: str = "unknown") -> bool:
        """
        Cancel any ongoing LLM streaming task for a session (barge-in).

        Latches the cancel intent stamped with the CURRENT utterance seq
        (review #21 of commit 6c9a87a — stale-cancel race fix). A cancel
        that arrives between utterance N completing and utterance N+1's
        `set_llm_task` registration used to silently kill N+1; the
        sequence stamp ensures the latch is honored only for the
        utterance it was meant for.

        `origin` is logged so the LLM-cancel mystery (open issue #1 in
        _phase2_followups.md) is diagnosable from the server log alone.

        Returns:
            True if a task was cancelled OR the cancel intent was latched.
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        had_task = state.llm_task is not None and not state.llm_task.done()
        had_event = state.llm_cancellation_event is not None and not state.llm_cancellation_event.is_set()
        log.warning(
            f"[{session_id}] cancel_llm_task origin={origin} "
            f"had_task={had_task} had_event={had_event} "
            f"current_seq={state.llm_utterance_seq}"
        )

        # Latch the intent stamped with the current seq. set_llm_task
        # increments to the next seq before checking — so a cancel for
        # seq=N never fires on seq=N+1.
        state.llm_pending_cancel = True
        state.llm_pending_cancel_seq = state.llm_utterance_seq
        was_cancelled = False

        if state.llm_cancellation_event is not None and not state.llm_cancellation_event.is_set():
            state.llm_cancellation_event.set()
            was_cancelled = True

        if state.llm_task is not None and not state.llm_task.done():
            state.llm_task.cancel()
            was_cancelled = True

        return was_cancelled or state.llm_pending_cancel

    def begin_utterance(self, session_id: str) -> int:
        """Increment the utterance seq before kicking off a new LLM task.

        Called from ws_asr.py:commit_utterance immediately before
        `asyncio.create_task(run_llm_stream(...))`. Returns the new seq
        so the caller can hand it to set_llm_task.
        """
        state = self._sessions.get(session_id)
        if not state:
            return 0
        state.llm_utterance_seq += 1
        return state.llm_utterance_seq

    def set_llm_task(
        self,
        session_id: str,
        task: asyncio.Task,
        cancellation_event: asyncio.Event,
        utterance_seq: Optional[int] = None,
    ) -> None:
        """Register an active LLM streaming task for a session.

        Honors a latched `llm_pending_cancel` ONLY if it was stamped for
        the current utterance seq (review #21). A stale cancel from a
        prior utterance is discarded silently here.
        """
        state = self._sessions.get(session_id)
        if not state:
            return
        state.llm_task = task
        state.llm_cancellation_event = cancellation_event

        if state.llm_pending_cancel:
            # The seq comparison: the latch is for `llm_pending_cancel_seq`;
            # we honor it iff it matches the seq this set_llm_task is for.
            # If utterance_seq isn't passed (legacy callers), fall back to
            # the current state seq — which preserves the old behavior.
            target = utterance_seq if utterance_seq is not None else state.llm_utterance_seq
            if state.llm_pending_cancel_seq == target:
                cancellation_event.set()
                log.info(
                    f"[{session_id}] set_llm_task fired latched cancel "
                    f"for seq={target}"
                )
            else:
                log.info(
                    f"[{session_id}] discarding stale latched cancel "
                    f"(latched_for_seq={state.llm_pending_cancel_seq}, "
                    f"current_seq={target})"
                )
            state.llm_pending_cancel = False
            state.llm_pending_cancel_seq = None

    def clear_llm_task(self, session_id: str) -> None:
        """Clear LLM task references after completion or cancellation.

        Also clears any latched llm_pending_cancel — fresh utterances start
        with no pending intent.
        """
        state = self._sessions.get(session_id)
        if state:
            state.llm_task = None
            state.llm_pending_cancel = False
            state.llm_pending_cancel_seq = None
            state.llm_cancellation_event = None

    # -------------------------------------------------------------------------
    # TTS Task Management
    # -------------------------------------------------------------------------

    def cancel_tts_task(self, session_id: str) -> bool:
        """
        Cancel any ongoing TTS streaming task.

        Returns:
            True if a task was cancelled, False otherwise
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        was_cancelled = False

        if state.tts_cancellation_event is not None and not state.tts_cancellation_event.is_set():
            state.tts_cancellation_event.set()
            was_cancelled = True

        if state.tts_task is not None and not state.tts_task.done():
            state.tts_task.cancel()
            was_cancelled = True

        return was_cancelled

    def set_tts_task(
        self,
        session_id: str,
        task: asyncio.Task,
        cancellation_event: asyncio.Event,
        tts_session_id: Optional[str] = None,
    ) -> None:
        """Register an active TTS streaming task."""
        state = self._sessions.get(session_id)
        if state:
            state.tts_task = task
            state.tts_cancellation_event = cancellation_event
            state.tts_session_id = tts_session_id or str(uuid.uuid4())

    def clear_tts_task(self, session_id: str) -> None:
        """Clear TTS task references."""
        state = self._sessions.get(session_id)
        if state:
            state.tts_task = None
            state.tts_cancellation_event = None

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _reset_utterance(self, session_id: str) -> None:
        """Reset utterance state for next sentence."""
        state = self._sessions.get(session_id)
        if state:
            state.audio_buffer.clear()
            state.utterance_id = str(uuid.uuid4())
            state.start_time = time.time()
            state.vad_latency_ms = 0
            # Reset VAD state too
            if hasattr(state.vad, "reset"):
                state.vad.reset()
            state._vad_committed = False
            state._vad_had_speech = False
