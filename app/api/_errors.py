"""
Centralized error handling for the API layer.

Domain layer (services) raises plain Python exceptions. API layer registers
one FastAPI handler per domain exception → HTTP status code mapping.

The response shape is uniform:
    { "error": <error_code>, "message": <human str>, "details": <dict|null> }

This is the single source of truth — services should not raise HTTPException
directly, and routes should not return ad-hoc dicts on failure.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base domain exception
# ---------------------------------------------------------------------------
class DomainError(Exception):
    """
    Base class for all domain-layer errors that map to specific HTTP codes.

    Subclasses set `status_code`, `error_code`, and an optional `details` dict.
    The HTTP handler at the bottom of this module uses those to build the
    JSON response without ever touching the service layer's internals.
    """

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ---------------------------------------------------------------------------
# Recording-related domain errors
# ---------------------------------------------------------------------------
class RecordingNotFoundError(DomainError):
    status_code = 404
    error_code = "recording_not_found"


class InvalidAudioError(DomainError):
    """Audio fails validation: duration, format, codec, etc."""

    status_code = 422
    error_code = "invalid_audio"


class AudioTooLargeError(DomainError):
    status_code = 413
    error_code = "audio_too_large"


class UnsupportedAudioFormatError(DomainError):
    status_code = 415
    error_code = "unsupported_audio_format"


class CorruptMetadataError(DomainError):
    """metadata.json failed to parse or schema-validate."""

    status_code = 500
    error_code = "corrupt_metadata"


class DiskFullError(DomainError):
    status_code = 507
    error_code = "disk_full"


# ---------------------------------------------------------------------------
# Persona / listener / training cross-cutting errors
# ---------------------------------------------------------------------------
class InvalidPersonaIdError(DomainError):
    status_code = 400
    error_code = "invalid_persona_id"


class InvalidListenerIdError(DomainError):
    status_code = 400
    error_code = "invalid_listener_id"


class PersonaConflictError(DomainError):
    """Two segments in a training job disagree on persona_id."""

    status_code = 409
    error_code = "persona_conflict"


class TrainingInProgressError(DomainError):
    """Cannot mutate state while a training job is running."""

    status_code = 409
    error_code = "training_in_progress"


# ---------------------------------------------------------------------------
# Training-specific errors (Phase 1.2).
# ---------------------------------------------------------------------------
class TrainingVersionNotFoundError(DomainError):
    status_code = 404
    error_code = "training_version_not_found"


class InvalidTrainingParamsError(DomainError):
    """Rank / epochs / batch size / training_type out of range."""

    status_code = 422
    error_code = "invalid_training_params"


class NoTrainingAudioError(DomainError):
    """Selected segments resolve to <10s of audio or zero existing files."""

    status_code = 422
    error_code = "no_training_audio"


class VersionNotReadyError(DomainError):
    """Cannot activate / preview a version whose status isn't 'ready'."""

    status_code = 409
    error_code = "version_not_ready"


class ActiveVersionLockedError(DomainError):
    """Cannot delete the currently-active version."""

    status_code = 409
    error_code = "active_version_locked"


class MergedModelMissingError(DomainError):
    """LoRA merge step did not produce the expected output directory."""

    status_code = 500
    error_code = "merged_model_missing"


# ---------------------------------------------------------------------------
# Persona / listener errors (Phase 1.3).
# ---------------------------------------------------------------------------
class PersonaNotFoundError(DomainError):
    status_code = 404
    error_code = "persona_not_found"


class ListenerNotFoundError(DomainError):
    status_code = 404
    error_code = "listener_not_found"


class FixedPersonaReadonlyError(DomainError):
    """Fixed personas cannot be renamed or deleted via the API."""

    status_code = 400
    error_code = "fixed_persona_readonly"


class SeedListenerReadonlyError(DomainError):
    """Seeded listeners (`child`, `mom`, …) cannot be deleted via the API."""

    status_code = 400
    error_code = "seed_listener_readonly"


class InvalidEmotionError(DomainError):
    """Listener `default_emotion` must be one of VALID_EMOTIONS."""

    status_code = 400
    error_code = "invalid_emotion"


class InvalidIdFormatError(DomainError):
    """persona_id / listener_id must match `^[a-z][a-z0-9_]*$`."""

    status_code = 400
    error_code = "invalid_id_format"


class DuplicateIdError(DomainError):
    """A persona / listener with this id already exists."""

    status_code = 409
    error_code = "duplicate_id"


# ---------------------------------------------------------------------------
# Generic input errors (rare — prefer specific subclasses).
# ---------------------------------------------------------------------------
class ValidationError(DomainError):
    status_code = 422
    error_code = "validation_failed"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    # 5xx errors get logged with stack; 4xx are user-driven, just info-log.
    if exc.status_code >= 500:
        log.exception(
            "DomainError on %s %s: %s",
            request.method,
            request.url.path,
            exc.message,
        )
    else:
        log.info(
            "DomainError on %s %s: %s [%s]",
            request.method,
            request.url.path,
            exc.error_code,
            exc.message,
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            # Modern shape (introduced in Phase 1.1).
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details or None,
            # Legacy FastAPI shape — the existing UI reads `err.detail` first
            # (see app/api/recordings_ui.py:1950 and 4 other call sites).
            # Mirroring `message` here keeps the UI's error-toast text correct
            # without forcing a UI-side refactor.
            "detail": exc.message,
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """
    Wire DomainError + subclasses to the FastAPI exception machinery.

    Routes call services. Services raise DomainError subclasses. FastAPI
    dispatches to `domain_error_handler` based on the exception class.
    """
    app.add_exception_handler(DomainError, domain_error_handler)  # type: ignore[arg-type]
