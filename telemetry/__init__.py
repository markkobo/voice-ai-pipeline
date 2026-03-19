"""
Telemetry module for Voice AI Pipeline.

Provides Prometheus metrics for monitoring latency, throughput, and streaming quality.
"""

from .metrics import (
    # Latency metrics
    vad_latency,
    asr_latency,
    llm_ttft,
    tts_ttfb,
    e2e_latency,
    streaming_chunk_latency,
    # Throughput metrics
    audio_chunks_total,
    utterances_total,
    asr_results_total,
    llm_tokens_total,
    tts_chunks_total,
    ws_connections_total,
    ws_messages_total,
    # Streaming metrics
    llm_tokens_per_second,
    tts_audio_bytes_per_second,
    audio_buffer_size_bytes,
    active_sessions,
    # Error metrics
    errors_total,
    asr_failures_total,
    llm_failures_total,
    tts_failures_total,
)
from .collector import TelemetryCollector

__all__ = [
    # Latency
    "vad_latency",
    "asr_latency",
    "llm_ttft",
    "tts_ttfb",
    "e2e_latency",
    "streaming_chunk_latency",
    # Throughput
    "audio_chunks_total",
    "utterances_total",
    "asr_results_total",
    "llm_tokens_total",
    "tts_chunks_total",
    "ws_connections_total",
    "ws_messages_total",
    # Streaming
    "llm_tokens_per_second",
    "tts_audio_bytes_per_second",
    "audio_buffer_size_bytes",
    "active_sessions",
    # Errors
    "errors_total",
    "asr_failures_total",
    "llm_failures_total",
    "tts_failures_total",
    # Collector
    "TelemetryCollector",
]
