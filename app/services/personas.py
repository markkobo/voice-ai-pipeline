"""
Persona service — manages persona definitions (fixed family + dynamic guests).

Personas are stored in data/personas/personas.json
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/personas")
DATA_FILE = DATA_DIR / "personas.json"


# Fixed personas (family members — cannot be deleted if they have trained versions)
FIXED_PERSONAS = [
    {"persona_id": "xiao_s", "name": "小S", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
    {"persona_id": "caregiver", "name": "照護者", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
    {"persona_id": "elder_gentle", "name": "長輩-溫柔", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
    {"persona_id": "elder_playful", "name": "長輩-活潑", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
]


def _load_personas() -> list[dict]:
    """Load personas from JSON file, seeding defaults if missing."""
    if not DATA_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _save_personas(FIXED_PERSONAS)
        return FIXED_PERSONAS.copy()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        stored = json.load(f)

    # Merge fixed personas that aren't in stored
    stored_ids = {p["persona_id"] for p in stored}
    for fp in FIXED_PERSONAS:
        if fp["persona_id"] not in stored_ids:
            stored.append(fp)

    return stored


def _save_personas(personas: list[dict]) -> None:
    """Save personas to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)


def list_personas() -> list[dict]:
    """Return all personas."""
    return _load_personas()


def get_persona(persona_id: str) -> Optional[dict]:
    """Return a single persona by ID."""
    personas = _load_personas()
    for p in personas:
        if p["persona_id"] == persona_id:
            return p
    return None


def create_persona(persona_id: str, name: str, is_family: bool = True) -> dict:
    """
    Create a dynamic (non-fixed) persona.

    Raises:
        ValueError: if persona_id already exists or is a fixed persona
    """
    personas = _load_personas()

    # Check fixed
    fixed_ids = {p["persona_id"] for p in personas if p.get("type") == "fixed"}
    if persona_id in fixed_ids:
        raise ValueError(f"Cannot create: '{persona_id}' is a fixed persona")

    # Check duplicate
    existing = {p["persona_id"] for p in personas}
    if persona_id in existing:
        raise ValueError(f"Persona '{persona_id}' already exists")

    persona = {
        "persona_id": persona_id,
        "name": name,
        "type": "dynamic",
        "is_family": is_family,
        "created_at": datetime.now().isoformat() + "Z",
    }
    personas.append(persona)
    _save_personas(personas)
    return persona


def update_persona(persona_id: str, name: Optional[str] = None) -> Optional[dict]:
    """
    Update persona name. Only dynamic personas can be updated via this API.
    Fixed personas can only have their display name updated.
    """
    personas = _load_personas()
    for p in personas:
        if p["persona_id"] == persona_id:
            if name is not None:
                p["name"] = name
            _save_personas(personas)
            return p
    return None


def delete_persona(persona_id: str) -> bool:
    """
    Delete a dynamic persona.
    Fixed personas cannot be deleted.
    Returns True if deleted, False if not found or fixed.
    """
    personas = _load_personas()
    fixed_ids = {p["persona_id"] for p in personas if p.get("type") == "fixed"}

    if persona_id in fixed_ids:
        raise ValueError(f"Cannot delete fixed persona: '{persona_id}'")

    new_personas = [p for p in personas if p["persona_id"] != persona_id]
    if len(new_personas) == len(personas):
        return False

    _save_personas(new_personas)
    return True


def is_fixed_persona(persona_id: str) -> bool:
    """Check if persona is a fixed (non-deletable) persona."""
    personas = _load_personas()
    return any(p["persona_id"] == persona_id and p.get("type") == "fixed" for p in personas)
