"""
Personas package.

New code uses `PersonasService` via FastAPI Depends. The legacy
function-style API (`list_personas`, `get_persona`, …) is kept here as a
thin backwards-compat shim that constructs a lazy process-wide repo —
needed by code paths like `app.services.llm.prompt_manager` that load at
startup, not per-request.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .models import FIXED_PERSONAS, Persona, PersonaType
from .repository import JsonPersonaRepository, PersonaNotFound
from .service import PersonasService

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy process-wide service used by legacy function-style callers only.
# Per-request consumers go through Depends(get_personas_service) instead.
# ---------------------------------------------------------------------------
_default_service: Optional[PersonasService] = None


def _resolve_data_root() -> Path:
    explicit = os.environ.get("DATA_ROOT")
    if explicit:
        return Path(explicit)
    legacy = Path("/workspace/voice-ai-pipeline/data")
    if legacy.parent.exists():
        return legacy
    return Path("data")


def _get_default_service() -> PersonasService:
    global _default_service
    if _default_service is None:
        _default_service = PersonasService(JsonPersonaRepository(_resolve_data_root()))
    return _default_service


# ---------------------------------------------------------------------------
# Legacy function-style API — kept for backwards compat with callers that
# don't go through FastAPI Depends. Internally delegates to the new service.
# ---------------------------------------------------------------------------
def list_personas() -> list[dict]:
    return [p.model_dump(mode="json") for p in _get_default_service().list_personas()]


def get_persona(persona_id: str) -> Optional[dict]:
    persona = _get_default_service().repository.get_or_none(persona_id)
    return persona.model_dump(mode="json") if persona else None


def is_fixed_persona(persona_id: str) -> bool:
    return _get_default_service().is_fixed(persona_id)


def create_persona(persona_id: str, name: str, is_family: bool = True) -> dict:
    """Legacy create — raises ValueError on validation failure (the old
    contract). Modern callers should use PersonasService.create() directly."""
    from app.api._errors import (
        DuplicateIdError,
        FixedPersonaReadonlyError,
        InvalidIdFormatError,
    )

    try:
        return _get_default_service().create(
            persona_id=persona_id, name=name, is_family=is_family
        ).model_dump(mode="json")
    except (InvalidIdFormatError, DuplicateIdError, FixedPersonaReadonlyError) as e:
        raise ValueError(e.message) from e


def update_persona(persona_id: str, name: Optional[str] = None) -> dict:
    from app.api._errors import FixedPersonaReadonlyError, PersonaNotFoundError

    try:
        return _get_default_service().update(persona_id, name=name).model_dump(mode="json")
    except PersonaNotFoundError as e:
        raise ValueError(e.message) from e
    except FixedPersonaReadonlyError as e:
        raise ValueError(e.message) from e


def delete_persona(persona_id: str) -> bool:
    from app.api._errors import FixedPersonaReadonlyError, PersonaNotFoundError

    try:
        _get_default_service().delete(persona_id)
        return True
    except PersonaNotFoundError as e:
        raise ValueError(e.message) from e
    except FixedPersonaReadonlyError as e:
        raise ValueError(e.message) from e


# ---------------------------------------------------------------------------
# Test-only helper to reset the lazy singleton between tests.
# ---------------------------------------------------------------------------
def _reset_default_service_for_testing() -> None:
    global _default_service
    _default_service = None


__all__ = [
    "FIXED_PERSONAS",
    "Persona",
    "PersonaType",
    "JsonPersonaRepository",
    "PersonaNotFound",
    "PersonasService",
    "list_personas",
    "get_persona",
    "is_fixed_persona",
    "create_persona",
    "update_persona",
    "delete_persona",
]
