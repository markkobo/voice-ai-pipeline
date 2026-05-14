"""
Personas REST API — routes only.

Phase 1.3: every route uses Pydantic with `extra="forbid"`, delegates to
PersonasService via Depends, and lets the app-wide DomainError handler map
errors to HTTP. No more HTTPException raised in this module.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.api._dependencies import get_personas_service
from app.services.personas import Persona, PersonasService

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/personas", tags=["personas"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class PersonaCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    name: str
    is_family: bool = True


class PersonaUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None


class PersonasListResponse(BaseModel):
    personas: list[Persona]


class DeletePersonaResponse(BaseModel):
    status: str = "deleted"
    persona_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/", response_model=PersonasListResponse)
async def api_list_personas(
    service: PersonasService = Depends(get_personas_service),
) -> PersonasListResponse:
    return PersonasListResponse(personas=service.list_personas())


@router.get("/{persona_id}", response_model=Persona)
async def api_get_persona(
    persona_id: str,
    service: PersonasService = Depends(get_personas_service),
) -> Persona:
    return service.get(persona_id)


@router.post("/", response_model=Persona)
async def api_create_persona(
    body: PersonaCreateRequest,
    service: PersonasService = Depends(get_personas_service),
) -> Persona:
    return service.create(
        persona_id=body.persona_id,
        name=body.name,
        is_family=body.is_family,
    )


@router.patch("/{persona_id}", response_model=Persona)
async def api_update_persona(
    persona_id: str,
    body: PersonaUpdateRequest,
    service: PersonasService = Depends(get_personas_service),
) -> Persona:
    return service.update(persona_id, name=body.name)


@router.delete("/{persona_id}", response_model=DeletePersonaResponse)
async def api_delete_persona(
    persona_id: str,
    service: PersonasService = Depends(get_personas_service),
) -> DeletePersonaResponse:
    service.delete(persona_id)
    return DeletePersonaResponse(persona_id=persona_id)
