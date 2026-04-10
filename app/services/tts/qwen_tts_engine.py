"""
Qwen3-TTS engine with VoiceDesign + emotion instruct support.

Uses FasterQwen3TTS for streaming when CUDA graphs work,
falls back to Qwen3TTSModel (non-streaming) on CUDA errors.
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional, List
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
    Supports LoRA adapter loading for voice cloning.
    """

    # Model options - VoiceDesign for streaming, Base for LoRA
    VOICEDESIGN_MODELS = {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }
    BASE_MODELS = {
        "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    }
    # Merged LoRA models (weight merging replaces need for PEFT at inference)
    MERGED_MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/voice-ai-pipeline/data/models"))

    def __init__(
        self,
        model_size: str = "1.7B",
        device: Optional[str] = None,
    ):
        self.model_size = model_size
        self.voicedesign_name = self.VOICEDESIGN_MODELS.get(model_size, self.VOICEDESIGN_MODELS["1.7B"])
        self.base_model_name = self.BASE_MODELS.get(model_size, self.BASE_MODELS["1.7B"])
        import torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = None  # FasterQwen3TTS (streaming, VoiceDesign)
        self._raw_model = None  # Qwen3TTSModel (fallback for voice clone)
        self._lora_model = None  # Qwen3TTSForConditionalGeneration + LoRA
        self._is_loaded = False
        self._use_fallback = False
        self._warmed_up = False
        self._current_lora_path: Optional[str] = None
        self._merged_model_path: Optional[str] = None
        self._model_type: Optional[str] = None  # "voicedesign", "custom_voice", or "voice_clone"
        self._speaker_name: Optional[str] = None  # speaker name for custom_voice models
        self._ref_audio_path: Optional[str] = None  # reference audio for voice_clone mode

    def _ensure_loaded(self):
        """Lazy load the model(s)."""
        if self._is_loaded:
            return

        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        # Check for merged model path first
        if hasattr(self, '_merged_model_path') and self._merged_model_path:
            model_path = self._merged_model_path
            log.info(f"Loading merged TTS model from: {model_path}")
            try:
                self._model = FasterQwen3TTS.from_pretrained(model_path, device=self.device)
                log.info(f"TTS model loaded (FasterQwen3TTS, merged): {model_path}")
                self._is_loaded = True
                return
            except Exception as e:
                log.warning(f"Failed to load merged model: {e}, falling back to base")

        # Determine which model to load based on mode
        is_voice_clone = getattr(self, '_model_type', None) == "voice_clone"
        model_name = self.base_model_name if is_voice_clone else self.voicedesign_name

        log.info(f"Loading TTS model: {model_name} on {self.device} (mode: {self._model_type or 'voicedesign'})")

        try:
            self._model = FasterQwen3TTS.from_pretrained(model_name, device=self.device)
            log.info(f"TTS model loaded (FasterQwen3TTS): {model_name}")
        except Exception as e:
            log.warning(f"FasterQwen3TTS failed to load: {e}, trying raw Qwen3TTSModel")
            from qwen_tts import Qwen3TTSModel
            self._raw_model = Qwen3TTSModel.from_pretrained(model_name, device_map="auto")
            self._use_fallback = True
            log.info(f"TTS model loaded (raw Qwen3TTSModel): {model_name}")

        self._is_loaded = True

    def warmup(self):
        """Warm up the model to capture CUDA graphs. Call once after loading."""
        if self._warmed_up:
            return

        # Ensure model is loaded first
        self._ensure_loaded()

        # Actually trigger CUDA graph capture
        # _warmup() captures both predictor and talker graphs
        import torch
        if hasattr(self._model, '_warmup'):
            log.info("Capturing CUDA graphs...")
            # Capture with a reasonable prefill length
            self._model._warmup(prefill_len=128)
            self._warmed_up = True
            log.info("CUDA graphs captured and ready")
        elif hasattr(self._model, '_ensure_warmed_up'):
            self._model._ensure_warmed_up()
            self._warmed_up = True
        else:
            log.warning("Model does not have _warmup method, skipping")

    def activate_version(self, version_id: str):
        """
        Activate a merged LoRA model for voice cloning.

        Uses weight-merging approach: LoRA weights are merged into base model
        to produce a standalone model that works with FasterQwen3TTS streaming.

        Args:
            version_id: The training version ID to activate
        """
        from app.services.training import get_version_manager

        manager = get_version_manager()
        version = manager.get_version(version_id)

        if not version:
            log.warning(f"[TTS] Version {version_id} not found")
            return

        if version.status != "ready":
            log.warning(f"[TTS] Version {version_id} not ready (status: {version.status})")
            return

        if not version.lora_path:
            log.warning(f"[TTS] Version {version_id} has no lora_path")
            return

        # Look for merged model: merged_{lora_dir_name_without_timestamp}
        # e.g., data/models/xiao_s_v11_20260330_204755 -> data/models/merged_qwen3_tts_xiao_s_v11
        lora_dir = Path(version.lora_path)
        parent_dir = lora_dir.parent
        # Extract version base name (e.g., "xiao_s_v11" from "xiao_s_v11_20260330_204755")
        # Name format: {persona}_{version}_{timestamp}
        parts = lora_dir.name.split('_')
        # First 3 parts: xiao, s, v11 -> xiao_s_v11
        version_base = '_'.join(parts[:3])
        merged_name = f"merged_qwen3_tts_{version_base}"
        merged_path = parent_dir / merged_name

        if not merged_path.exists():
            log.warning(f"[TTS] Merged model not found at: {merged_path}")
            log.warning(f"[TTS] Run merge script first to create merged model")
            return

        # Convert to absolute path for proper loading
        merged_path_str = str(merged_path.resolve())
        if self._merged_model_path == merged_path_str and self._is_loaded:
            log.info(f"[TTS] Merged model already active: {merged_path}")
            return

        # Detect model type from config
        config_path = merged_path / "config.json"
        model_type = "voicedesign"
        speaker_name = version.persona_id  # default to persona_id
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("tts_model_type", "voicedesign")
            # Extract speaker name from talker_config.spk_id
            talker_config = config.get("talker_config", {})
            spk_id = talker_config.get("spk_id", {})
            if spk_id:
                speaker_name = list(spk_id.keys())[0]  # Get first speaker name
        log.info(f"[TTS] Model type: {model_type}, speaker: {speaker_name}")

        self._merged_model_path = merged_path_str
        self._current_lora_path = str(lora_dir)
        self._model_type = model_type
        self._speaker_name = speaker_name
        log.info(f"[TTS] Activated merged model: {merged_path}")

        # Reload model if already loaded to use merged weights
        if self._is_loaded:
            self._is_loaded = False
            self._warmed_up = False  # Need to re-warmup with new model
            self._ensure_loaded()

    def deactivate_lora(self):
        """Deactivate merged model and use base VoiceDesign model."""
        self._current_lora_path = None
        self._merged_model_path = None
        self._model_type = None
        self._speaker_name = None
        self._ref_audio_path = None
        log.info("[TTS] Merged model deactivated")

    @staticmethod
    def find_reference_audio(persona_id: str) -> Optional[str]:
        """
        Find the best reference audio file for a persona.

        Looks for extracted speaker audio in recordings, sorted by file size
        (largest = best quality).

        Args:
            persona_id: The persona ID (e.g., "xiao_s")

        Returns:
            Path to the best reference audio file, or None if not found
        """
        from pathlib import Path
        import os

        # Map persona_id to folder prefix pattern
        # e.g., "xiao_s" -> "default_xiao_s_*"
        persona_prefix_map = {
            "xiao_s": "default_xiao_s_",
            "caregiver": "default_caregiver_",
        }

        prefix = persona_prefix_map.get(persona_id, f"default_{persona_id}_")
        recordings_raw = Path("/workspace/voice-ai-pipeline/data/recordings/raw")

        if not recordings_raw.exists():
            return None

        # Find all matching folders
        matching_dirs = [d for d in recordings_raw.iterdir() if d.is_dir() and d.name.startswith(prefix)]

        # Find the speaker audio in each folder, track by size
        best_audio = None
        best_size = 0

        for folder in matching_dirs:
            speaker_audio = folder / "speakers" / "SPEAKER_00.wav"
            if speaker_audio.exists():
                size = speaker_audio.stat().st_size
                if size > best_size:
                    best_size = size
                    best_audio = str(speaker_audio)

        if best_audio:
            log = get_logger(__name__, component="tts")
            log.info(f"[TTS] Found reference audio for {persona_id}: {best_audio} ({best_size} bytes)")
        else:
            log = get_logger(__name__, component="tts")
            log.warning(f"[TTS] No reference audio found for {persona_id}")

        return best_audio

    def activate_voice_clone(self, persona_id: str, ref_audio_path: Optional[str] = None):
        """
        Activate voice clone mode using x-vector extraction from reference audio.

        Uses the Base model with generate_voice_clone_streaming and xvec_only=True.
        This approach extracts the speaker embedding at inference time, providing
        better voice matching than SFT with limited training data.

        Args:
            persona_id: The persona ID (e.g., "xiao_s")
            ref_audio_path: Path to reference audio file (extracted speaker audio).
                           If None, will auto-find from recordings.
        """
        import os
        log = get_logger(__name__, component="tts")

        # Auto-find reference audio if not provided
        if ref_audio_path is None:
            ref_audio_path = self.find_reference_audio(persona_id)
            if ref_audio_path is None:
                log.warning(f"[TTS] Could not find reference audio for {persona_id}")
                return
        elif not os.path.exists(ref_audio_path):
            log.warning(f"[TTS] Reference audio not found: {ref_audio_path}")
            return

        self._ref_audio_path = ref_audio_path
        self._model_type = "voice_clone"
        self._speaker_name = persona_id
        self._current_lora_path = None
        self._merged_model_path = None

        log.info(f"[TTS] Activated voice clone mode for {persona_id} with ref: {ref_audio_path}")

        # Reload model if already loaded to use Base model for voice clone
        if self._is_loaded:
            self._is_loaded = False
            self._warmed_up = False
            self._ensure_loaded()

    async def generate_streaming(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
        reference_audio: Optional[str] = None,
        voice_clone_prompt=None,
    ) -> AsyncIterator[TTSStreamEvent]:
        """
        Generate audio from text with optional emotion instruct.

        Args:
            text: Text to synthesize
            instruct: Emotion instruct string (e.g., "(gentle, warm)")
            language: Language code
            reference_audio: Path to reference audio for voice cloning
            voice_clone_prompt: Pre-built voice clone prompt (not needed with merged model)
        """
        self._ensure_loaded()

        if not text:
            return

        # Path B: If instruct is None, pass empty string to minimize instruct influence
        # This allows the merged model's baked voice characteristics to come through
        if instruct is None:
            final_instruct = ""
        else:
            final_instruct = instruct

        # With merged model, the voice is baked in - no reference needed
        using_merged = hasattr(self, '_merged_model_path') and self._merged_model_path is not None

        log.info(f"TTS generating: text_len={len(text)}, instruct={final_instruct}, "
                 f"merged_model={using_merged}, fallback={self._use_fallback}")

        yield TTSStreamEvent(event="start")

        loop = asyncio.get_event_loop()

        # Use streaming path with FasterQwen3TTS (works with merged model too)
        if self._model is not None and not self._use_fallback:
            import queue
            chunk_queue = queue.Queue()

            # Determine generation method based on model type
            is_custom_voice = getattr(self, '_model_type', None) == "custom_voice"
            is_voice_clone = getattr(self, '_model_type', None) == "voice_clone"
            speaker_name = getattr(self, '_speaker_name', None) or "default"
            ref_audio = getattr(self, '_ref_audio_path', None)

            def chunk_producer():
                try:
                    if is_voice_clone:
                        # Voice clone uses Base model with x-vector extraction from reference audio
                        # xvec_only=True: use only speaker embedding, no phoneme bleed-through
                        import torch
                        torch.manual_seed(42)
                        torch.cuda.manual_seed_all(42)
                        for audio_chunk, sr, timing in self._model.generate_voice_clone_streaming(
                            text=text,
                            language=language,
                            ref_audio=ref_audio,
                            ref_text="",  # Required but ignored when xvec_only=True
                            xvec_only=True,
                            do_sample=False,
                            temperature=0.0,
                            chunk_size=12,
                        ):
                            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                            chunk_queue.put((audio_bytes, sr or 24000))
                    elif is_custom_voice:
                        # Custom voice model uses speaker name instead of instruct
                        # Use deterministic generation to ensure consistent voice
                        import torch
                        torch.manual_seed(42)
                        torch.cuda.manual_seed_all(42)
                        for audio_chunk, sr, timing in self._model.generate_custom_voice_streaming(
                            text=text,
                            speaker=speaker_name,
                            language=language,
                            instruct=final_instruct if final_instruct else None,
                            do_sample=False,
                            temperature=0.0,
                            chunk_size=12,
                        ):
                            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                            chunk_queue.put((audio_bytes, sr or 24000))
                    else:
                        # VoiceDesign model uses instruct
                        for audio_chunk, sr, timing in self._model.generate_voice_design_streaming(
                            text=text,
                            instruct=final_instruct,
                            language=language,
                            chunk_size=12,
                        ):
                            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                            chunk_queue.put((audio_bytes, sr or 24000))
                    chunk_queue.put(None)
                except Exception as e:
                    log.error(f"Chunk producer error: {e}")
                    # Mark fallback for next call so we don't retry the failing streaming path
                    self._use_fallback = True
                    chunk_queue.put(e)

            import concurrent.futures
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(chunk_producer)

            streaming_err = None
            while True:
                result = await loop.run_in_executor(None, chunk_queue.get)
                if result is None:
                    break
                if isinstance(result, Exception):
                    streaming_err = result
                    break
                audio_bytes, sr = result
                yield TTSStreamEvent(event="audio_chunk", audio_data=audio_bytes, sample_rate=sr)

            pool.shutdown(wait=False)

            # If streaming failed, fall back to non-streaming within the same call
            if streaming_err is not None:
                log.warning(f"Streaming failed ({streaming_err}), falling back to non-streaming")
                raw_model = self._raw_model if self._raw_model is not None else self._model
                def generate():
                    try:
                        if is_voice_clone:
                            audio_arrays, sr = raw_model.generate_voice_clone(
                                text=text,
                                language=language,
                                ref_audio=ref_audio,
                                ref_text="",
                                xvec_only=True,
                            )
                        elif is_custom_voice:
                            audio_arrays, sr = raw_model.generate_custom_voice(
                                text=text,
                                speaker=speaker_name,
                                language=language,
                                instruct=final_instruct if final_instruct else None,
                            )
                        else:
                            audio_arrays, sr = raw_model.generate_voice_design(
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
                    except Exception as e:
                        log.error(f"Fallback generation error: {e}")
                        return None, 24000, None

                result = await loop.run_in_executor(None, generate)
                if result is None or result[0] is None:
                    yield TTSStreamEvent(event="error", error="Fallback also failed")
                    return
                audio_bytes, sr, _ = result
                yield TTSStreamEvent(event="audio_chunk", audio_data=audio_bytes, sample_rate=sr)
                yield TTSStreamEvent(event="done")
                return

            yield TTSStreamEvent(event="done")
            return

        # Fallback path (non-streaming)
        # Triggered when: (a) _raw_model was loaded as primary model, OR
        # (b) streaming path failed with CUDA graph error (_use_fallback=True, _raw_model=None)
        if self._use_fallback or self._raw_model is not None:
            # Use _raw_model if available, otherwise fall back to self._model.generate_voice_design
            raw_model = self._raw_model if self._raw_model is not None else self._model
            is_custom_voice = getattr(self, '_model_type', None) == "custom_voice"
            is_voice_clone = getattr(self, '_model_type', None) == "voice_clone"
            speaker_name = getattr(self, '_speaker_name', None) or "default"
            ref_audio = getattr(self, '_ref_audio_path', None)
            def generate():
                try:
                    if is_voice_clone:
                        # Voice clone fallback - use non-streaming generate_voice_clone
                        audio_arrays, sr = raw_model.generate_voice_clone(
                            text=text,
                            language=language,
                            ref_audio=ref_audio,
                            ref_text="",  # Required but ignored when xvec_only=True
                            xvec_only=True,
                        )
                    elif is_custom_voice:
                        audio_arrays, sr = raw_model.generate_custom_voice(
                            text=text,
                            speaker=speaker_name,
                            language=language,
                            instruct=final_instruct if final_instruct else None,
                        )
                    else:
                        audio_arrays, sr = raw_model.generate_voice_design(
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
                except Exception as e:
                    log.error(f"Generation error: {e}")
                    return None, 24000, None

            result = await loop.run_in_executor(None, generate)
            if result is None or result[0] is None:
                yield TTSStreamEvent(event="error", error="No audio generated")
                return
            audio_bytes, sr, _ = result
            yield TTSStreamEvent(event="audio_chunk", audio_data=audio_bytes, sample_rate=sr)
            yield TTSStreamEvent(event="done")
            return


class MockTTSEngine:
    """Mock TTS engine for testing without GPU."""

    def __init__(self, model_size: str = "0.6B"):
        self.model_size = model_size
        self._is_loaded = True
        self._current_lora_path: Optional[str] = None

    def warmup(self):
        """No-op warmup for mock engine."""
        pass

    def activate_version(self, version_id: str):
        """Mock activation - does nothing."""
        pass

    def deactivate_lora(self):
        """Mock deactivation - does nothing."""
        pass

    async def generate_streaming(
        self,
        text: str,
        instruct: Optional[str] = None,
        language: str = "Chinese",
        reference_audio: Optional[Any] = None,
    ) -> AsyncIterator[TTSStreamEvent]:
        """Async generator - proper async iterable for ws_asr.py."""
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
