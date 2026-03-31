"""
Persona service — manages persona definitions (fixed family + dynamic guests).

Personas are stored in data/personas/personas.json
"""

import fcntl
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/personas")
DATA_FILE = DATA_DIR / "personas.json"
_LOCK_FILE = DATA_DIR / ".personas.lock"


def _with_lock(mode: str, callback):
    """
    Execute a callback while holding an exclusive flock on _LOCK_FILE.

    Args:
        mode: "r" to open read-only, "r+" to open read-write (creates if missing)
        callback: function that receives the open file object and returns the result.
                  For "r" mode the file is positioned at start, for "r+" at end of load.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _LOCK_FILE.touch(exist_ok=True)
    with open(_LOCK_FILE, "r") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            f = open(DATA_FILE, mode, encoding="utf-8")
            try:
                result = callback(f)
            finally:
                f.close()
            return result
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


# Fixed personas (family members — cannot be deleted if they have trained versions)
FIXED_PERSONAS = [
    {"persona_id": "xiao_s", "name": "小S", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
    {"persona_id": "caregiver", "name": "照護者", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
    {"persona_id": "elder_gentle", "name": "長輩-溫柔", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
    {"persona_id": "elder_playful", "name": "長輩-活潑", "type": "fixed", "is_family": True, "created_at": "2026-03-01T00:00:00Z"},
]


def _load_personas_unlocked() -> list[dict]:
    """Load personas from JSON file, seeding defaults if missing. Caller must hold lock."""
    if not DATA_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _save_personas_unlocked(FIXED_PERSONAS.copy())
        return FIXED_PERSONAS.copy()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        stored = json.load(f)

    # Merge fixed personas that aren't in stored
    stored_ids = {p["persona_id"] for p in stored}
    for fp in FIXED_PERSONAS:
        if fp["persona_id"] not in stored_ids:
            stored.append(fp)

    return stored


def _save_personas_unlocked(personas: list[dict]) -> None:
    """Save personas to JSON file. Caller must hold lock."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)


def _load_personas() -> list[dict]:
    """Load personas (thread-safe)."""
    return _with_lock("r", lambda f: _load_personas_unlocked())


def _save_personas(personas: list[dict]) -> None:
    """Save personas (thread-safe)."""
    _with_lock("r", lambda f: _save_personas_unlocked(personas))


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
    def _txn(f):
        personas = _load_personas_unlocked()

        fixed_ids = {p["persona_id"] for p in personas if p.get("type") == "fixed"}
        if persona_id in fixed_ids:
            raise ValueError(f"Cannot create: '{persona_id}' is a fixed persona")

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
        _save_personas_unlocked(personas)
        return persona

    return _with_lock("r", _txn)


def update_persona(persona_id: str, name: Optional[str] = None) -> dict:
    """
    Update persona name.

    Raises:
        ValueError: if persona_id not found
    """
    def _txn(f):
        personas = _load_personas_unlocked()
        for p in personas:
            if p["persona_id"] == persona_id:
                if name is not None:
                    p["name"] = name
                _save_personas_unlocked(personas)
                return p
        raise ValueError(f"Persona not found: '{persona_id}'")

    return _with_lock("r", _txn)


def delete_persona(persona_id: str) -> bool:
    """
    Delete a dynamic persona.
    Fixed personas cannot be deleted.
    Returns True if deleted.

    Raises:
        ValueError: if persona_id is fixed or not found
    """
    def _txn(f):
        personas = _load_personas_unlocked()
        fixed_ids = {p["persona_id"] for p in personas if p.get("type") == "fixed"}

        if persona_id in fixed_ids:
            raise ValueError(f"Cannot delete fixed persona: '{persona_id}'")

        new_personas = [p for p in personas if p["persona_id"] != persona_id]
        if len(new_personas) == len(personas):
            raise ValueError(f"Persona not found: '{persona_id}'")

        _save_personas_unlocked(new_personas)
        return True

    return _with_lock("r", _txn)


def is_fixed_persona(persona_id: str) -> bool:
    """Check if persona is a fixed (non-deletable) persona."""
    personas = _load_personas()
    return any(p["persona_id"] == persona_id and p.get("type") == "fixed" for p in personas)
