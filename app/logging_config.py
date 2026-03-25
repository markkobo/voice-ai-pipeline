"""
Structured JSON logging configuration for Voice AI Pipeline.

All logs are written to /logs/app.log in JSON format for easy
parsing and remote troubleshooting.
"""
import logging
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "component"):
            log_entry["component"] = record.component
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_json_logging(log_dir: str = "/workspace/voice-ai-pipeline-1/logs"):
    """
    Setup structured JSON logging for the application.

    Args:
        log_dir: Directory for log files
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "app.log"

    # Create JSON formatter
    json_formatter = JSONFormatter()

    # File handler (JSON)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)

    # Console handler (for development - human readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str, component: str = None) -> logging.Logger:
    """
    Get a logger with optional component tag.

    Args:
        name: Logger name (typically __name__)
        component: Component name for log tagging (e.g., "ws_asr", "tts", "llm")

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    if component:
        logger = logging.getLogger(f"{name}.{component}")
    return logger


class StructuredLogger:
    """
    Wrapper around logging.Logger that adds structured context fields.

    Usage:
        log = StructuredLogger("ws_asr", component="ws")
        log.info("Session started", session_id="abc123")
    """

    def __init__(self, name: str, component: str = None):
        self._logger = get_logger(name, component)
        self._component = component

    def _make_extra(self, **kwargs) -> dict:
        extra = {"component": self._component} if self._component else {}
        extra.update(kwargs)
        return extra

    def debug(self, msg: str, **kwargs):
        self._logger.debug(msg, extra=self._make_extra(**kwargs))

    def info(self, msg: str, **kwargs):
        self._logger.info(msg, extra=self._make_extra(**kwargs))

    def warning(self, msg: str, **kwargs):
        self._logger.warning(msg, extra=self._make_extra(**kwargs))

    def error(self, msg: str, **kwargs):
        self._logger.error(msg, extra=self._make_extra(**kwargs))

    def exception(self, msg: str, **kwargs):
        self._logger.exception(msg, extra=self._make_extra(**kwargs))
