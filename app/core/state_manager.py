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

        # TTS
        self.tts_session_id = None
        self.tts_task = None
        self.tts_cancellation_event = None

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
            vad: VAD engine instance (defaults to SileroVAD)
            asr: ASR engine instance (defaults to Qwen3ASR or MockASR)
            use_qwen: If True and no asr provided, use Qwen3ASR
        """
        self._sessions: Dict[str, SessionState] = {}
        self._default_vad = vad or SileroVAD()

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
            self.cancel_llm_task(session_id)
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

        # VAD sensitivity - use provided value or preserve existing
        vad_sensitivity = config.get("vad")
        if vad_sensitivity:
            state.vad = SileroVAD(
                sample_rate=state.audio_config.sample_rate,
                sensitivity=vad_sensitivity,
            )
        else:
            # Recreate VAD with new sample rate, preserve sensitivity
            current_sensitivity = getattr(state.vad, "sensitivity_label", "medium")
            state.vad = SileroVAD(
                sample_rate=state.audio_config.sample_rate,
                sensitivity=current_sensitivity,
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

        # Accumulate audio
        state.audio_buffer.extend(audio_chunk)

        # Track speech state
        if not is_commit:
            state._vad_had_speech = True

        # Barge-in: user starts speaking in an active (not-yet-committed) utterance
        # SileroVAD returns is_commit=False during active speech
        if not is_commit and state._vad_had_speech and not state._vad_committed:
            log.info(f"[VAD] Barge-in detected (new speech in active utterance)")
            self.cancel_llm_task(session_id)
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

    def cancel_llm_task(self, session_id: str) -> bool:
        """
        Cancel any ongoing LLM streaming task for a session (barge-in).

        Returns:
            True if a task was cancelled, False otherwise
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        was_cancelled = False

        if state.llm_cancellation_event is not None and not state.llm_cancellation_event.is_set():
            state.llm_cancellation_event.set()
            was_cancelled = True

        if state.llm_task is not None and not state.llm_task.done():
            state.llm_task.cancel()
            was_cancelled = True

        return was_cancelled

    def set_llm_task(
        self,
        session_id: str,
        task: asyncio.Task,
        cancellation_event: asyncio.Event,
    ) -> None:
        """Register an active LLM streaming task for a session."""
        state = self._sessions.get(session_id)
        if state:
            state.llm_task = task
            state.llm_cancellation_event = cancellation_event

    def clear_llm_task(self, session_id: str) -> None:
        """Clear LLM task references after completion or cancellation."""
        state = self._sessions.get(session_id)
        if state:
            state.llm_task = None
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
