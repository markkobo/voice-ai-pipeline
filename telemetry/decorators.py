"""
Decorators for automatic telemetry collection.

Provides convenient decorators to wrap functions with automatic metrics recording.
"""

import time
from functools import wraps
from typing import Optional, Callable, Any

from .metrics import (
    vad_latency,
    asr_latency,
    llm_ttft,
    tts_ttfb,
    e2e_latency,
    streaming_chunk_latency,
)


def track_vad_latency(model: str = "default"):
    """
    Decorator to track VAD latency.

    Usage:
        @track_vad_latency(model="energy_vad")
        def detect_silence(audio):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with vad_latency.labels(component="vad", model=model).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_asr_latency(model: str = "default"):
    """
    Decorator to track ASR processing latency.

    Usage:
        @track_asr_latency(model="qwen3-asr")
        def transcribe(audio):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with asr_latency.labels(component="asr", model=model).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_llm_ttft(model: str = "default"):
    """
    Decorator to track LLM Time to First Token.

    Note: For streaming, consider using StreamingRateTracker instead.

    Usage:
        @track_llm_ttft(model="llama3-70b")
        def generate_response(prompt):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with llm_ttft.labels(component="llm", model=model).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_tts_ttfb(model: str = "default"):
    """
    Decorator to track TTS Time to First Byte.

    Usage:
        @track_tts_ttfb(model="tts-1")
        def synthesize(text):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with tts_ttfb.labels(component="tts", model=model).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_e2e_latency():
    """
    Decorator to track end-to-end latency.

    Usage:
        @track_e2e_latency()
        def process_utterance(audio):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with e2e_latency.labels(component="pipeline").time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_streaming_chunk(stream_type: str = "llm_tokens", component: str = "llm"):
    """
    Decorator to track streaming chunk latency.

    Usage:
        @track_streaming_chunk(stream_type="llm_tokens", component="llm")
        def stream_token(token):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with streaming_chunk_latency.labels(
                component=component,
                stream_type=stream_type,
            ).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator
