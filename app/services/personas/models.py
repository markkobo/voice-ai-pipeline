"""Pydantic domain models for personas."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PersonaType(str, Enum):
    fixed = "fixed"
    dynamic = "dynamic"


class Persona(BaseModel):
    """A persona — either a fixed family member or a dynamic guest."""

    # `extra="ignore"` tolerates legacy fields the old code wrote into
    # to_dict (e.g. `display_name`) that aren't real state.
    model_config = ConfigDict(extra="ignore")

    persona_id: str
    name: str
    type: PersonaType = PersonaType.dynamic
    is_family: bool = True
    created_at: Optional[str] = None  # ISO-8601 string; old JSON used "...Z"

    def is_fixed(self) -> bool:
        return self.type == PersonaType.fixed


# The four fixed family personas — seeded into a fresh repository and
# protected from deletion / rename via the service layer.
FIXED_PERSONAS: list[Persona] = [
    Persona(persona_id="xiao_s", name="小S", type=PersonaType.fixed, is_family=True, created_at="2026-03-01T00:00:00Z"),
    Persona(persona_id="caregiver", name="照護者", type=PersonaType.fixed, is_family=True, created_at="2026-03-01T00:00:00Z"),
    Persona(persona_id="elder_gentle", name="長輩-溫柔", type=PersonaType.fixed, is_family=True, created_at="2026-03-01T00:00:00Z"),
    Persona(persona_id="elder_playful", name="長輩-活潑", type=PersonaType.fixed, is_family=True, created_at="2026-03-01T00:00:00Z"),
]
