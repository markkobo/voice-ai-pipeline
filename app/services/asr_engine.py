"""ASR (Automatic Speech Recognition) engine module."""
import asyncio
import os
import time
import numpy as np
import struct
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseASR(ABC):
    """Abstract base class for ASR engines."""

    @abstractmethod
    async def recognize(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Recognize speech from audio bytes.

        Args:
            audio_bytes: Raw PCM audio bytes

        Returns:
            Dict containing recognized text and telemetry
        """
        pass


class Qwen3ASR(BaseASR):
    """Qwen3-ASR engine using the qwen-asr package.

    Requires model to be loaded via from_pretrained().
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-1.7B", latency_ms: int = 0):
        """
        Initialize Qwen3 ASR.

        Args:
            model_name: HuggingFace model name or local path
            latency_ms: Additional simulated latency (for testing)
        """
        self.model_name = model_name
        self.latency_ms = latency_ms
        self._model: Optional[Any] = None
        self._sample_rate = 24000

    def load_model(self) -> None:
        """Load the Qwen3-ASR model."""
        import torch
        from qwen_asr import Qwen3ASRModel
        print(f"Loading Qwen3-ASR model: {self.model_name}...")

        # Determine device
        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.bfloat16
        else:
            device_map = "cpu"
            dtype = torch.float32

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=512
        )
        print("Qwen3-ASR model loaded successfully")

    async def recognize(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Recognize speech using Qwen3-ASR.

        Args:
            audio_bytes: Raw PCM 16-bit audio bytes

        Returns:
            Dict with text and telemetry
        """
        if self._model is None:
            self.load_model()

        # Convert bytes to numpy array
        num_samples = len(audio_bytes) // 2  # 16-bit = 2 bytes
        samples = struct.unpack(f"{num_samples}h", audio_bytes)
        audio_array = np.array(samples, dtype=np.float32) / 32768.0  # Normalize to [-1, 1]

        # Run inference
        start_time = time.perf_counter()

        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe((audio_array, self._sample_rate))
        )

        inference_time = int((time.perf_counter() - start_time) * 1000)
        inference_time += self.latency_ms

        # Extract text from result
        text = ""
        if result and len(result) > 0:
            text = result[0].text.strip()

        return {
            "text": text,
            "asr_inference_ms": inference_time
        }


class MockASR(BaseASR):
    """Mock ASR engine for testing and development.

    Simulates inference latency and returns placeholder text.
    """

    def __init__(self, latency_ms: int = 50):
        """
        Initialize Mock ASR.

        Args:
            latency_ms: Simulated inference latency in milliseconds
        """
        self.latency_ms = latency_ms

    async def recognize(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Simulate ASR recognition.

        Args:
            audio_bytes: Raw PCM audio bytes (unused in mock)

        Returns:
            Dict with mock text and telemetry
        """
        # Simulate inference delay
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Return mock result
        return {
            "text": "模擬語音辨識結果...",
            "asr_inference_ms": self.latency_ms
        }
