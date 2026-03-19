"""State manager for WebSocket sessions, audio buffers, and utterance tracking."""
import uuid
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from app.services.vad_engine import BaseVAD
from app.services.asr_engine import BaseASR


@dataclass
class AudioConfig:
    """Audio configuration for a session."""
    sample_rate: int = 24000
    channels: int = 1
    format: str = "pcm"


@dataclass
class SessionState:
    """State for a single WebSocket session."""
    session_id: str
    utterance_id: str
    audio_config: AudioConfig
    vad: BaseVAD
    asr: BaseASR
    audio_buffer: bytearray = field(default_factory=bytearray)
    is_configured: bool = False
    start_time: float = field(default_factory=time.time)
    vad_latency_ms: int = 0


class StateManager:
    """Manages WebSocket sessions and audio buffer state."""

    def __init__(self, vad: Optional[BaseVAD] = None, asr: Optional[BaseASR] = None, use_qwen: bool = True):
        """
        Initialize StateManager.

        Args:
            vad: VAD engine instance (defaults to EnergyVAD)
            asr: ASR engine instance (defaults to Qwen3ASR or MockASR)
            use_qwen: If True and no asr provided, use Qwen3ASR
        """
        from app.services.vad_engine import EnergyVAD
        from app.services.asr_engine import MockASR, Qwen3ASR

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
            utterance_id=str(uuid.uuid4()),
            audio_config=AudioConfig(),
            vad=vad or self._default_vad,
            asr=asr or self._default_asr
        )
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session state by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise
        """
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)

    def update_config(self, session_id: str, config: Dict[str, Any]) -> bool:
        """
        Update session audio configuration.

        Args:
            session_id: Session identifier
            config: Config dict with audio settings

        Returns:
            True if config was applied, False otherwise
        """
        from app.services.vad_engine import EnergyVAD

        state = self._sessions.get(session_id)
        if not state:
            return False

        audio_config = config.get("audio", {})
        state.audio_config.sample_rate = audio_config.get("sample_rate", 24000)
        state.audio_config.channels = audio_config.get("channels", 1)
        state.audio_config.format = audio_config.get("format", "pcm")

        # Recreate VAD with new sample rate
        state.vad = EnergyVAD(sample_rate=state.audio_config.sample_rate)

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

        # Accumulate audio
        state.audio_buffer.extend(audio_chunk)

        # VAD detection
        vad_start = time.perf_counter()
        is_speaking, energy = state.vad.detect(audio_chunk)
        vad_end = time.perf_counter()
        state.vad_latency_ms = int((vad_end - vad_start) * 1000)

        # If speech detected, return partial (in real impl, would stream)
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
            # Empty buffer - return empty result
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

        # Run ASR on accumulated audio
        audio_data = bytes(state.audio_buffer)
        asr_result = await state.asr.recognize(audio_data)

        # Build final response
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

    def _reset_utterance(self, session_id: str) -> None:
        """Reset utterance state for next sentence."""
        state = self._sessions.get(session_id)
        if state:
            state.audio_buffer.clear()
            state.utterance_id = str(uuid.uuid4())
            state.start_time = time.time()
            state.vad_latency_ms = 0
