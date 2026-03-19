"""
Telemetry collector utilities for Voice AI Pipeline.

Provides helper functions for metrics collection and Prometheus server management.
"""

import time
import threading
from contextlib import contextmanager
from typing import Optional, Callable
from functools import wraps

from prometheus_client import start_http_server, Counter, Histogram, Gauge

from . import metrics


class TelemetryCollector:
    """
    Centralized telemetry collector for managing metrics collection.

    Usage:
        collector = TelemetryCollector()
        collector.start_server(port=9090)
    """

    def __init__(self, port: int = 9090, enable_at_start: bool = False):
        """
        Initialize the telemetry collector.

        Args:
            port: Port to expose Prometheus metrics (default: 9090)
            enable_at_start: Whether to start the metrics server immediately
        """
        self.port = port
        self._server_thread: Optional[threading.Thread] = None
        self._start_time = time.time()

        if enable_at_start:
            self.start_server()

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        if self._server_thread is not None:
            return

        self._server_thread = threading.Thread(
            target=start_http_server,
            args=(self.port,),
            daemon=True,
        )
        self._server_thread.start()
        print(f"Metrics server started on http://localhost:{self.port}/metrics")

    def stop_server(self) -> None:
        """Stop the Prometheus metrics HTTP server."""
        # Note: HTTP server doesn't have a clean stop method in prometheus_client
        # For production, use a proper server lifecycle management
        self._server_thread = None

    @property
    def uptime_seconds(self) -> float:
        """Get the collector uptime in seconds."""
        return time.time() - self._start_time

    # =========================================================================
    # Helper methods for common operations
    # =========================================================================

    def record_audio_chunk(self, session_id: str) -> None:
        """Record an incoming audio chunk."""
        metrics.audio_chunks_total.labels(
            component="ws",
            session_id=session_id,
        ).inc()

    def record_utterance_start(self, session_id: str) -> None:
        """Record the start of a new utterance."""
        metrics.utterances_total.labels(
            component="pipeline",
            session_id=session_id,
        )
        metrics.active_sessions.labels(component="pipeline").inc()

    def record_utterance_end(self, session_id: str) -> None:
        """Record the end of an utterance."""
        metrics.active_sessions.labels(component="pipeline").dec()

    def record_ws_message(
        self,
        direction: str,
        message_type: str,
    ) -> None:
        """Record a WebSocket message."""
        metrics.ws_messages_total.labels(
            component="ws",
            direction=direction,
            message_type=message_type,
        ).inc()

    def record_error(
        self,
        component: str,
        error_type: str,
        model: Optional[str] = None,
    ) -> None:
        """Record an error occurrence."""
        metrics.errors_total.labels(
            component=component,
            error_type=error_type,
            model=model or "",
        ).inc()

        # Also record to specific component failure counter
        if component == "asr":
            metrics.asr_failures_total.labels(
                component=component,
                model=model or "",
                error_type=error_type,
            ).inc()
        elif component == "llm":
            metrics.llm_failures_total.labels(
                component=component,
                model=model or "",
                error_type=error_type,
            ).inc()
        elif component == "tts":
            metrics.tts_failures_total.labels(
                component=component,
                model=model or "",
                error_type=error_type,
            ).inc()


# ============================================================================
# Decorator utilities
# =========================================================================


def time_latency(histogram: Histogram, **labels):
    """
    Decorator to automatically measure and record latency.

    Usage:
        @time_latency(metrics.asr_latency, component="asr", model="qwen3")
        def process_audio(audio):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with histogram.labels(**labels).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Context manager utilities
# =========================================================================


@contextmanager
def measure_latency(histogram: Histogram, **labels):
    """
    Context manager to measure latency.

    Usage:
        with measure_latency(metrics.asr_latency, component="asr", model="qwen3"):
            do_work()
    """
    with histogram.labels(**labels).time():
        yield


class StreamingRateTracker:
    """
    Track streaming rates (tokens/sec, bytes/sec).

    Usage:
        tracker = StreamingRateTracker(metrics.llm_tokens_per_second, model="llama3")
        tracker.start()

        # ... during streaming ...
        tracker.add_tokens(10)

        # ... at end ...
        tracker.stop()
    """

    def __init__(self, gauge: Gauge, **labels):
        self.gauge = gauge
        self.labels = labels
        self._start_time: Optional[float] = None
        self._count = 0

    def start(self) -> None:
        """Start tracking."""
        self._start_time = time.time()
        self._count = 0

    def add(self, value: float = 1) -> None:
        """Add to the count."""
        self._count += value

    def stop(self) -> None:
        """Stop tracking and record final rate."""
        if self._start_time is None:
            return

        elapsed = time.time() - self._start_time
        if elapsed > 0:
            rate = self._count / elapsed
            self.gauge.labels(**self.labels).set(rate)

    def update_rate(self) -> None:
        """Update the gauge with current rate (call periodically during streaming)."""
        if self._start_time is None:
            return

        elapsed = time.time() - self._start_time
        if elapsed > 0:
            rate = self._count / elapsed
            self.gauge.labels(**self.labels).set(rate)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ============================================================================
# Singleton instance for easy access
# =========================================================================

_default_collector: Optional[TelemetryCollector] = None


def get_collector() -> TelemetryCollector:
    """Get the default telemetry collector instance."""
    global _default_collector
    if _default_collector is None:
        _default_collector = TelemetryCollector()
    return _default_collector
