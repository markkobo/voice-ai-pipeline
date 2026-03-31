"""
Listener service — manages listener definitions (family + dynamic guests).

Listeners are stored in data/listeners/listeners.json
"""

import fcntl
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/listeners")
DATA_FILE = DATA_DIR / "listeners.json"
_LOCK_FILE = DATA_DIR / ".listeners.lock"


def _with_lock(mode: str, callback):
    """
    Execute a callback while holding an exclusive flock on _LOCK_FILE.
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


# Pre-seeded listeners
SEED_LISTENERS = [
    {"listener_id": "child", "name": "小孩", "is_family": True, "default_emotion": "撒嬌", "created_at": "2026-03-01T00:00:00Z"},
    {"listener_id": "mom", "name": "媽媽", "is_family": True, "default_emotion": "撒嬌", "created_at": "2026-03-01T00:00:00Z"},
    {"listener_id": "dad", "name": "爸爸", "is_family": True, "default_emotion": "溫和", "created_at": "2026-03-01T00:00:00Z"},
    {"listener_id": "friend", "name": "朋友", "is_family": True, "default_emotion": "幽默", "created_at": "2026-03-01T00:00:00Z"},
    {"listener_id": "elder", "name": "長輩", "is_family": True, "default_emotion": "溫和", "created_at": "2026-03-01T00:00:00Z"},
    {"listener_id": "reporter", "name": "記者", "is_family": False, "default_emotion": "溫和", "created_at": "2026-03-01T00:00:00Z"},
    {"listener_id": "default", "name": "預設", "is_family": False, "default_emotion": "撒嬌", "created_at": "2026-03-01T00:00:00Z"},
]


def _load_listeners_unlocked() -> list[dict]:
    """Load listeners from JSON file, seeding defaults if missing. Caller must hold lock."""
    if not DATA_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _save_listeners_unlocked(SEED_LISTENERS.copy())
        return SEED_LISTENERS.copy()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        stored = json.load(f)

    # Merge seed listeners that aren't in stored
    stored_ids = {l["listener_id"] for l in stored}
    for sl in SEED_LISTENERS:
        if sl["listener_id"] not in stored_ids:
            stored.append(sl)

    return stored


def _save_listeners_unlocked(listeners: list[dict]) -> None:
    """Save listeners to JSON file. Caller must hold lock."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(listeners, f, ensure_ascii=False, indent=2)


def _load_listeners() -> list[dict]:
    """Load listeners (thread-safe)."""
    return _with_lock("r", lambda f: _load_listeners_unlocked())


def _save_listeners(listeners: list[dict]) -> None:
    """Save listeners (thread-safe)."""
    _with_lock("r", lambda f: _save_listeners_unlocked(listeners))


def list_listeners() -> list[dict]:
    """Return all listeners."""
    return _load_listeners()


def get_listener(listener_id: str) -> Optional[dict]:
    """Return a single listener by ID."""
    listeners = _load_listeners()
    for l in listeners:
        if l["listener_id"] == listener_id:
            return l
    return None


def create_listener(listener_id: str, name: str, is_family: bool = False,
                   default_emotion: str = "溫和") -> dict:
    """
    Create a new listener.

    Raises:
        ValueError: if listener_id already exists
    """
    def _txn(f):
        listeners = _load_listeners_unlocked()

        existing_ids = {l["listener_id"] for l in listeners}
        if listener_id in existing_ids:
            raise ValueError(f"Listener '{listener_id}' already exists")

        listener = {
            "listener_id": listener_id,
            "name": name,
            "is_family": is_family,
            "default_emotion": default_emotion,
            "created_at": datetime.now().isoformat() + "Z",
        }
        listeners.append(listener)
        _save_listeners_unlocked(listeners)
        return listener

    return _with_lock("r", _txn)


def update_listener(listener_id: str, name: Optional[str] = None,
                    default_emotion: Optional[str] = None) -> dict:
    """
    Update listener name or default emotion.

    Raises:
        ValueError: if listener_id not found
    """
    def _txn(f):
        listeners = _load_listeners_unlocked()
        for l in listeners:
            if l["listener_id"] == listener_id:
                if name is not None:
                    l["name"] = name
                if default_emotion is not None:
                    l["default_emotion"] = default_emotion
                _save_listeners_unlocked(listeners)
                return l
        raise ValueError(f"Listener not found: '{listener_id}'")

    return _with_lock("r", _txn)


def delete_listener(listener_id: str) -> bool:
    """
    Delete a listener by ID.

    Raises:
        ValueError: if listener_id not found
    """
    def _txn(f):
        listeners = _load_listeners_unlocked()
        new_listeners = [l for l in listeners if l["listener_id"] != listener_id]
        if len(new_listeners) == len(listeners):
            raise ValueError(f"Listener not found: '{listener_id}'")
        _save_listeners_unlocked(new_listeners)
        return True

    return _with_lock("r", _txn)
