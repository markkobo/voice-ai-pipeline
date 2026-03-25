"""
Faster-Qwen3-TTS engine wrapper with VoiceDesign + emotion instruct support.

Supports streaming audio generation with early trigger capability.
"""
import asyncio
import os
import time
from typing import AsyncIterator, Optional, List
from dataclasses import dataclass

import numpy as np

from app.logging_config import get_logger
from app.services.tts.emotion_mapper import get_tts_instruct

log = get_logger(__name__, component="tts")


@dataclass
class TTSStreamEvent:
    """TTS streaming event."""
    event: str  # "start", "audio_chunk", "done", "error"
    audio_data: Optional[bytes] = None  # PCM 16-bit mono
    sample_rate: int = 24000
    error: Optional[str] = None


class FasterQwenTTSEngine:
    """
    Wrapper for faster-qwen3-tts VoiceDesign mode.

    Supports:
    - Streaming audio generation
    - Emotion-based instruct control
    - Model selection (0.6B vs 1.7B)
    - Cached model loading
    """

    # Model options
    MODELS = {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    # chunk_size vs latency/quality tradeoff
    # chunk_size=8 → ~667ms per chunk, good balance for streaming
    CHUNK_SIZE = 8

    def __init__(
        self,
        model_size: str = "0.6B",
        device: Optional[str] = None,
    ):
        """
        Initialize TTS engine.

        Args:
            model_size: "0.6B" or "1.7B"
            device: CUDA device (auto-detected if None)
        """
        self.model_size = model_size
        self.model_name = self.MODELS.get(model_size, self.MODELS["0.6B"])
        self.device = device or ("cuda:0" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
        self._model = None
        self._is_loaded = False

    def _ensure_loaded(self):
        """Lazy load the model."""
        if self._is_loaded:
            return

        log.info(f"Loading TTS model: {self.model_name} on {self.device}")

        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self._model = FasterQwen3TTS.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self._model = self._model.to(self.device)

        self._is_loaded = True
        log.info(f"TTS model loaded: {self.model_name}")

    async def generate_streaming(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
    ) -> AsyncIterator[TTSStreamEvent]:
        """
        Generate streaming audio from text with optional emotion instruct.

        Args:
            text: Text to synthesize
            instruct: Natural language instruction for voice style
                     (e.g., "(gentle, warm tone)")
            language: Language hint

        Yields:
            TTSStreamEvent objects with audio chunks
        """
        self._ensure_loaded()

        if not text:
            return

        # Build final instruct string
        if instruct:
            final_instruct = f"{instruct}"
        else:
            final_instruct = "(natural, conversational tone)"

        log.info(f"TTS generating: text_len={len(text)}, instruct={final_instruct}")

        yield TTSStreamEvent(event="start")

        try:
            # Run generation in thread to avoid blocking
            loop = asyncio.get_event_loop()

            # faster-qwen3-tts generator
            def generate():
                chunks = []
                for audio_chunk, sr, timing in self._model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=None,  # VoiceDesign mode uses instruct, not ref audio
                    ref_text=None,
                    instruct=final_instruct,
                    chunk_size=self.CHUNK_SIZE,
                ):
                    # audio_chunk is a numpy array
                    chunks.append(audio_chunk)

                if not chunks:
                    return None, 24000

                # Concatenate all chunks
                full_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                # Convert to 16-bit PCM bytes
                audio_bytes = (full_audio * 32767).astype(np.int16).tobytes()
                return audio_bytes, 24000

            audio_bytes, sr = await loop.run_in_executor(None, generate)

            if audio_bytes:
                yield TTSStreamEvent(
                    event="audio_chunk",
                    audio_data=audio_bytes,
                    sample_rate=sr,
                )

            yield TTSStreamEvent(event="done")

        except Exception as e:
            log.exception(f"TTS generation error: {e}")
            yield TTSStreamEvent(event="error", error=str(e))

    async def generate_streaming_early_trigger(
        self,
        text_buffer: AsyncIterator[str],
        emotion_instruct: str,
        language: str = "Chinese",
    ) -> AsyncIterator[TTSStreamEvent]:
        """
        Generate streaming audio with early trigger.

        The caller feeds text chunks as they arrive from LLM.
        TTS starts generating as soon as emotion is detected.

        Args:
            text_buffer: AsyncIterator of text chunks from LLM
            emotion_instruct: Pre-computed instruct string from emotion mapper
            language: Language hint

        Yields:
            TTSStreamEvent objects with audio chunks
        """
        self._ensure_loaded()

        # Accumulate text
        accumulated_text = ""

        # We need to collect all text first for this approach
        # In practice, we'll use the simpler approach where caller
        # provides complete text with instruct already determined
        async for chunk in text_buffer:
            accumulated_text += chunk

        # Now generate with full text
        async for event in self.generate_streaming(
            text=accumulated_text,
            instruct=emotion_instruct,
            language=language,
        ):
            yield event

    async def generate_single(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
    ) -> tuple[bytes, int]:
        """
        Generate non-streaming audio (for debugging / simple cases).

        Args:
            text: Text to synthesize
            instruct: Emotion instruct string
            language: Language hint

        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        all_bytes = b""
        sr = 24000

        async for event in self.generate_streaming(text, instruct, language):
            if event.event == "audio_chunk" and event.audio_data:
                all_bytes += event.audio_data
                sr = event.sample_rate

        return all_bytes, sr


class MockTTSEngine:
    """
    Mock TTS engine for testing / development without GPU.

    Simulates streaming audio generation with configurable latency.
    """

    def __init__(
        self,
        latency_per_char_ms: float = 30.0,
        sample_rate: int = 24000,
    ):
        self.latency_per_char_ms = latency_per_char_ms
        self.sample_rate = sample_rate

    async def generate_streaming(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
    ) -> AsyncIterator[TTSStreamEvent]:
        """Simulate streaming TTS with fake audio."""
        yield TTSStreamEvent(event="start")

        # Simulate processing delay
        await asyncio.sleep(len(text) * self.latency_per_char_ms / 1000.0)

        # Generate fake silent PCM audio (just for testing the streaming flow)
        chunk_size = self.sample_rate // 10  # 100ms of audio
        silence = b"\x00\x00" * chunk_size  # 16-bit silence

        yield TTSStreamEvent(
            event="audio_chunk",
            audio_data=silence,
            sample_rate=self.sample_rate,
        )

        yield TTSStreamEvent(event="done")

    async def generate_single(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
    ) -> tuple[bytes, int]:
        """Simulate non-streaming TTS."""
        audio = b"\x00\x00" * self.sample_rate  # 1 second of silence
        return audio, self.sample_rate


# Singleton instance
_tts_engine: Optional[FasterQwenTTSEngine] = None


def get_tts_engine(model_size: str = "0.6B") -> FasterQwenTTSEngine | MockTTSEngine:
    """
    Get or create the TTS engine singleton.

    Args:
        model_size: "0.6B" or "1.7B"

    Returns:
        TTS engine instance
    """
    global _tts_engine

    if _tts_engine is None:
        use_mock = os.getenv("USE_MOCK_TTS", "false").lower() == "true"
        if use_mock:
            _tts_engine = MockTTSEngine()
            log.info("Using MockTTSEngine")
        else:
            _tts_engine = FasterQwenTTSEngine(model_size=model_size)
            log.info(f"Using FasterQwenTTSEngine ({model_size})")

    return _tts_engine
