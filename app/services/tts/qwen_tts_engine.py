"""
Qwen3-TTS engine with VoiceDesign + emotion instruct support.

Uses FasterQwen3TTS for streaming when CUDA graphs work,
falls back to Qwen3TTSModel (non-streaming) on CUDA errors.
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
    Wrapper for Qwen3-TTS VoiceDesign mode with streaming + fallback.

    Tries FasterQwen3TTS (streaming, CUDA graphs) first.
    Falls back to Qwen3TTSModel (non-streaming) if CUDA graph capture fails.
    """

    # Model options
    MODELS = {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    def __init__(
        self,
        model_size: str = "1.7B",
        device: Optional[str] = None,
    ):
        self.model_size = model_size
        self.model_name = self.MODELS.get(model_size, self.MODELS["1.7B"])
        import torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None  # FasterQwen3TTS (streaming)
        self._raw_model = None  # Qwen3TTSModel (fallback)
        self._is_loaded = False
        self._use_fallback = False
        self._warmed_up = False

    def _ensure_loaded(self):
        """Lazy load the model(s)."""
        if self._is_loaded:
            return

        log.info(f"Loading TTS model: {self.model_name} on {self.device}")

        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        try:
            self._model = FasterQwen3TTS.from_pretrained(self.model_name, device=self.device)
            log.info(f"TTS model loaded (FasterQwen3TTS): {self.model_name}")
        except Exception as e:
            log.warning(f"FasterQwen3TTS failed to load: {e}, trying raw Qwen3TTSModel")
            from qwen_tts import Qwen3TTSModel
            self._raw_model = Qwen3TTSModel.from_pretrained(self.model_name, device_map="auto")
            self._use_fallback = True
            log.info(f"TTS model loaded (raw Qwen3TTSModel): {self.model_name}")

        self._is_loaded = True

    def warmup(self):
        """Warm up the model to capture CUDA graphs. Call once after loading."""
        if self._warmed_up:
            return

        # Ensure model is loaded first
        self._ensure_loaded()

        if self._use_fallback or self._raw_model is not None:
            log.info("Warmup skipped (using fallback model)")
            self._warmed_up = True
            return

        log.info("Warming up TTS model (capturing CUDA graphs)...")
        import time
        start = time.time()
        try:
            # Run one inference to capture CUDA graphs
            for _ in self._model.generate_voice_design_streaming(
                text="測試",
                instruct="(natural)",
                language="Chinese",
                chunk_size=8,
            ):
                pass
            log.info(f"TTS warmup done in {time.time()-start:.1f}s")
        except Exception as e:
            log.warning(f"TTS warmup failed: {e}")
        self._warmed_up = True

    async def generate_streaming(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
    ) -> AsyncIterator[TTSStreamEvent]:
        """
        Generate audio from text with optional emotion instruct.
        Yields audio as a single chunk (non-streaming) if FasterQwen3TTS
        fails, otherwise yields chunks as they are generated.
        """
        self._ensure_loaded()

        if not text:
            return

        if instruct:
            final_instruct = f"{instruct}"
        else:
            final_instruct = "(natural, conversational tone)"

        log.info(f"TTS generating: text_len={len(text)}, instruct={final_instruct}, fallback={self._use_fallback}")

        yield TTSStreamEvent(event="start")

        loop = asyncio.get_event_loop()

        def generate():
            try:
                if self._use_fallback or self._raw_model is not None:
                    # Non-streaming fallback
                    audio_arrays, sr = self._raw_model.generate_voice_design(
                        text=text,
                        instruct=final_instruct,
                        language=language,
                    )
                    # audio_arrays is a list of arrays, sr is int
                    if isinstance(audio_arrays, (list, tuple)):
                        full_audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
                    else:
                        full_audio = audio_arrays
                    audio_bytes = (full_audio * 32767).astype(np.int16).tobytes()
                    return audio_bytes, sr or 24000, None
                else:
                    # Streaming with FasterQwen3TTS
                    chunks = []
                    for audio_chunk, sr, timing in self._model.generate_voice_design_streaming(
                        text=text,
                        instruct=final_instruct,
                        language=language,
                        chunk_size=12,
                    ):
                        chunks.append(audio_chunk)

                    if not chunks:
                        return None, 24000, None

                    full_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                    audio_bytes = (full_audio * 32767).astype(np.int16).tobytes()
                    return audio_bytes, sr, None

            except RuntimeError as e:
                if "CUDA" in str(e) or "capture" in str(e):
                    log.warning(f"CUDA error in streaming, falling back to non-streaming: {e}")
                    # Switch to fallback model
                    if self._raw_model is None:
                        from qwen_tts import Qwen3TTSModel
                        self._raw_model = Qwen3TTSModel.from_pretrained(
                            self.model_name, device_map=self.device
                        )
                        self._use_fallback = True
                    audio_arrays, sr = self._raw_model.generate_voice_design(
                        text=text,
                        instruct=final_instruct,
                        language=language,
                    )
                    if isinstance(audio_arrays, (list, tuple)):
                        full_audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
                    else:
                        full_audio = audio_arrays
                    audio_bytes = (full_audio * 32767).astype(np.int16).tobytes()
                    return audio_bytes, sr or 24000, None
                raise

        try:
            if self._use_fallback or self._raw_model is not None:
                # Non-streaming fallback - run whole generation in executor
                result = await loop.run_in_executor(None, generate)
                if result is None:
                    yield TTSStreamEvent(event="error", error="No audio generated")
                    return
                audio_bytes, sr, _ = result
                if audio_bytes:
                    yield TTSStreamEvent(event="audio_chunk", audio_data=audio_bytes, sample_rate=sr)
                yield TTSStreamEvent(event="done")
            else:
                # True streaming - yield chunks as they arrive using queue
                import queue
                chunk_queue = queue.Queue()

                def chunk_producer():
                    try:
                        for audio_chunk, sr, timing in self._model.generate_voice_design_streaming(
                            text=text,
                            instruct=final_instruct,
                            language=language,
                            chunk_size=12,
                        ):
                            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                            chunk_queue.put((audio_bytes, sr or 24000))
                        chunk_queue.put(None)  # Sentinel to signal done
                    except Exception as e:
                        chunk_queue.put(e)

                # Run producer in thread pool
                import concurrent.futures
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = pool.submit(chunk_producer)

                # Yield chunks as they arrive
                while True:
                    result = await loop.run_in_executor(None, chunk_queue.get)
                    if result is None:
                        break
                    if isinstance(result, Exception):
                        yield TTSStreamEvent(event="error", error=str(result))
                        break
                    audio_bytes, sr = result
                    yield TTSStreamEvent(event="audio_chunk", audio_data=audio_bytes, sample_rate=sr)

                pool.shutdown(wait=False)
                yield TTSStreamEvent(event="done")

        except Exception as e:
            log.exception(f"TTS generation error: {e}")
            yield TTSStreamEvent(event="error", error=str(e))


class MockTTSEngine:
    """Mock TTS engine for testing without GPU."""

    def __init__(self, model_size: str = "0.6B"):
        self.model_size = model_size
        self._is_loaded = True

    async def generate_streaming(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
    ) -> AsyncIterator[TTSStreamEvent]:
        yield TTSStreamEvent(event="start")
        # Generate ~1 second of silence as mock audio
        mock_samples = int(24000 * 1.0)  # 1 second at 24kHz
        audio = np.zeros(mock_samples, dtype=np.float32)
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        yield TTSStreamEvent(event="audio_chunk", audio_data=audio_bytes, sample_rate=24000)
        yield TTSStreamEvent(event="done")


_tts_engine: Optional[FasterQwenTTSEngine | MockTTSEngine] = None


def get_tts_engine(model_size: str = "1.7B") -> FasterQwenTTSEngine | MockTTSEngine:
    """Get or create the TTS engine singleton."""
    global _tts_engine

    if _tts_engine is None:
        use_mock = os.getenv("USE_MOCK_TTS", "false").lower() == "true"
        if use_mock:
            _tts_engine = MockTTSEngine(model_size=model_size)
            log.info("Using MockTTSEngine")
        else:
            _tts_engine = FasterQwenTTSEngine(model_size=model_size)
            log.info(f"Using FasterQwenTTSEngine ({model_size})")

    return _tts_engine


def preload_tts():
    """Preload and warmup TTS engine at startup. Returns immediately if already loaded."""
    engine = get_tts_engine()
    if hasattr(engine, 'warmup') and not getattr(engine, '_warmed_up', False):
        engine.warmup()
    return engine

    return _tts_engine
