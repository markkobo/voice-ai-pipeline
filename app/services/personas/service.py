"""PersonasService — orchestrator for persona CRUD."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from app.api._errors import (
    DuplicateIdError,
    FixedPersonaReadonlyError,
    InvalidIdFormatError,
    PersonaNotFoundError,
)

from .models import Persona, PersonaType
from .repository import JsonPersonaRepository, PersonaNotFound

log = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class PersonasService:
    """Pure business logic around persona CRUD.

    Repository handles locking + serialization. Service enforces:
    - id format (`^[a-z][a-z0-9_]*$`)
    - duplicate-id rejection
    - fixed-persona readonly guard
    """

    def __init__(self, repository: JsonPersonaRepository) -> None:
        self.repository = repository

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def list_personas(self) -> list[Persona]:
        return self.repository.list()

    def get(self, persona_id: str) -> Persona:
        try:
            return self.repository.get(persona_id)
        except PersonaNotFound as e:
            raise PersonaNotFoundError(
                f"Persona not found: {persona_id}",
                details={"persona_id": persona_id},
            ) from e

    def exists(self, persona_id: str) -> bool:
        return self.repository.exists(persona_id)

    def is_fixed(self, persona_id: str) -> bool:
        existing = self.repository.get_or_none(persona_id)
        return existing is not None and existing.is_fixed()

    # ------------------------------------------------------------------
    # Mutate
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        persona_id: str,
        name: str,
        is_family: bool = True,
    ) -> Persona:
        if not _ID_PATTERN.match(persona_id):
            raise InvalidIdFormatError(
                "persona_id must be lowercase letters, numbers, underscores, "
                "starting with a letter",
                details={"persona_id": persona_id, "pattern": _ID_PATTERN.pattern},
            )
        existing = self.repository.get_or_none(persona_id)
        if existing is not None:
            if existing.is_fixed():
                raise FixedPersonaReadonlyError(
                    f"Cannot create: {persona_id!r} is a fixed persona",
                    details={"persona_id": persona_id},
                )
            raise DuplicateIdError(
                f"Persona already exists: {persona_id}",
                details={"persona_id": persona_id},
            )
        persona = Persona(
            persona_id=persona_id,
            name=name,
            type=PersonaType.dynamic,
            is_family=is_family,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.repository.save(persona)
        log.info("[PERSONA] Created %s", persona_id)
        return persona

    def update(
        self,
        persona_id: str,
        *,
        name: Optional[str] = None,
    ) -> Persona:
        existing = self.get(persona_id)
        if existing.is_fixed():
            raise FixedPersonaReadonlyError(
                f"Cannot rename fixed persona: {persona_id!r}",
                details={"persona_id": persona_id},
            )
        if name is not None:
            existing.name = name
        self.repository.save(existing)
        return existing

    def delete(self, persona_id: str) -> None:
        existing = self.get(persona_id)
        if existing.is_fixed():
            raise FixedPersonaReadonlyError(
                f"Cannot delete fixed persona: {persona_id!r}",
                details={"persona_id": persona_id},
            )
        try:
            self.repository.delete(persona_id)
        except PersonaNotFound as e:
            raise PersonaNotFoundError(
                f"Persona not found: {persona_id}",
                details={"persona_id": persona_id},
            ) from e
