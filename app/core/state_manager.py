"""State manager for WebSocket sessions, audio buffers, utterance tracking, and LLM tasks."""
import asyncio
import uuid
import time
from typing import Optional, Dict, Any

from app.services.asr import BaseVAD, EnergyVAD, BaseASR, MockASR, Qwen3ASR
from telemetry import metrics


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

    # Milestone 2.1: LLM and speaker tracking
    speaker_id: Optional[str]  # From client config or future voice-print
    llm_cancellation_event: Optional[asyncio.Event]  # Set to cancel ongoing LLM stream
    llm_task: Optional[asyncio.Task]  # Active LLM streaming task

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
        self.speaker_id = None
        self.llm_cancellation_event = None
        self.llm_task = None


class StateManager:
    """Manages WebSocket sessions, audio buffers, utterance tracking, and LLM tasks."""

    def __init__(self, vad: Optional[BaseVAD] = None, asr: Optional[BaseASR] = None, use_qwen: bool = True):
        """
        Initialize StateManager.

        Args:
            vad: VAD engine instance (defaults to EnergyVAD)
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
        """
        Create a new session.

        Args:
            session_id: Unique session identifier
            vad: Optional VAD instance (uses default if not provided)
            asr: Optional ASR instance (uses default if not provided)

        Returns:
            New SessionState instance
        """
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
        """Remove a session after cancelling any active LLM task."""
        state = self._sessions.get(session_id)
        if state:
            self.cancel_llm_task(session_id)
        self._sessions.pop(session_id, None)

    def update_config(self, session_id: str, config: Dict[str, Any]) -> bool:
        """
        Update session audio configuration and speaker identity.

        Args:
            session_id: Session identifier
            config: Config dict with audio settings and optional speaker_id

        Returns:
            True if config was applied, False otherwise
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        audio_config = config.get("audio", {})
        state.audio_config.sample_rate = audio_config.get("sample_rate", 24000)
        state.audio_config.channels = audio_config.get("channels", 1)
        state.audio_config.format = audio_config.get("format", "pcm")

        # Recreate VAD with new sample rate
        state.vad = EnergyVAD(sample_rate=state.audio_config.sample_rate)

        # Accept speaker_id from client config (Milestone 2.1)
        if "speaker_id" in config:
            state.speaker_id = config.get("speaker_id")

        state.is_configured = True
        return True

    def process_audio(self, session_id: str, audio_chunk: bytes) -> Optional[Dict[str, Any]]:
        """
        Process audio chunk through VAD.

        Args:
            session_id: Session identifier
            audio_chunk: Binary audio data

        Returns:
            None if speech continues, dict with partial result if VAD triggers,
            or raises exception if session not found
        """
        state = self._sessions.get(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        if not state.is_configured:
            raise RuntimeError("Session not configured")

        # Cancel any ongoing LLM task when new speech is detected (barge-in)
        is_speaking, energy = state.vad.detect(audio_chunk)
        if is_speaking:
            self.cancel_llm_task(session_id)

        # Accumulate audio
        state.audio_buffer.extend(audio_chunk)

        # VAD detection telemetry
        vad_start = time.perf_counter()
        is_speaking, energy = state.vad.detect(audio_chunk)
        vad_end = time.perf_counter()
        vad_latency = vad_end - vad_start
        state.vad_latency_ms = int(vad_latency * 1000)

        metrics.vad_latency.labels(
            component="vad",
            model="energy"
        ).observe(vad_latency)

        # If speech detected, return partial
        if is_speaking:
            return {
                "type": "asr_result",
                "utterance_id": state.utterance_id,
                "is_final": False,
                "text": "...",
                "telemetry": {
                    "vad_latency_ms": state.vad_latency_ms,
                    "asr_inference_ms": 0
                }
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

        # Run ASR on accumulated audio with telemetry
        audio_data = bytes(state.audio_buffer)
        asr_start = time.perf_counter()
        asr_result = await state.asr.recognize(audio_data)
        asr_end = time.perf_counter()
        asr_latency = asr_end - asr_start

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

    def cancel_llm_task(self, session_id: str) -> bool:
        """
        Cancel any ongoing LLM streaming task for a session (barge-in).

        Args:
            session_id: Session identifier

        Returns:
            True if a task was cancelled, False otherwise
        """
        state = self._sessions.get(session_id)
        if not state:
            return False

        if state.llm_cancellation_event is not None and not state.llm_cancellation_event.is_set():
            state.llm_cancellation_event.set()
            return True

        if state.llm_task is not None and not state.llm_task.done():
            state.llm_task.cancel()
            return True

        return False

    def set_llm_task(
        self,
        session_id: str,
        task: asyncio.Task,
        cancellation_event: asyncio.Event,
    ) -> None:
        """
        Register an active LLM streaming task for a session.

        Args:
            session_id: Session identifier
            task: The asyncio.Task running the LLM stream
            cancellation_event: Event to set when cancelling
        """
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

    def _reset_utterance(self, session_id: str) -> None:
        """Reset utterance state for next sentence."""
        state = self._sessions.get(session_id)
        if state:
            state.audio_buffer.clear()
            state.utterance_id = str(uuid.uuid4())
            state.start_time = time.time()
            state.vad_latency_ms = 0
