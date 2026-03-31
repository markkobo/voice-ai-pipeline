"""
Persona API endpoints.

GET/POST/PATCH/DELETE for persona definitions.
"""

import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.personas import (
    list_personas,
    get_persona,
    create_persona,
    update_persona,
    delete_persona,
    is_fixed_persona,
)

router = APIRouter(prefix="/api/personas", tags=["personas"])


class PersonaCreate(BaseModel):
    persona_id: str
    name: str
    is_family: bool = True


class PersonaUpdate(BaseModel):
    name: Optional[str] = None


@router.get("/")
async def api_list_personas():
    """List all personas."""
    return {"personas": list_personas()}


@router.get("/{persona_id}")
async def api_get_persona(persona_id: str):
    """Get a single persona."""
    persona = get_persona(persona_id)
    if not persona:
        raise HTTPException(404, f"Persona not found: {persona_id}")
    return persona


@router.post("/")
async def api_create_persona(body: PersonaCreate):
    """Create a dynamic (non-fixed) persona."""
    # Validate ID format: lowercase letters, numbers, underscores
    if not re.match(r'^[a-z][a-z0-9_]*$', body.persona_id):
        raise HTTPException(400, "persona_id must be lowercase letters, numbers, underscores, starting with a letter")

    if is_fixed_persona(body.persona_id):
        raise HTTPException(400, f"Cannot create: '{body.persona_id}' is a fixed persona")

    try:
        persona = create_persona(body.persona_id, body.name, body.is_family)
        return persona
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.patch("/{persona_id}")
async def api_update_persona(persona_id: str, body: PersonaUpdate):
    """Update persona name."""
    if is_fixed_persona(persona_id):
        raise HTTPException(400, "Cannot rename fixed persona via this API")
    try:
        persona = update_persona(persona_id, name=body.name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return persona


@router.delete("/{persona_id}")
async def api_delete_persona(persona_id: str):
    """Delete a dynamic persona. Fixed personas cannot be deleted."""
    if is_fixed_persona(persona_id):
        raise HTTPException(400, f"Cannot delete fixed persona: '{persona_id}'")
    try:
        delete_persona(persona_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"status": "deleted", "persona_id": persona_id}
