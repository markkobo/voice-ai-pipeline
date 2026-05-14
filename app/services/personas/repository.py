"""
Persona repository — fcntl-locked, atomic-rename JSON storage.

Same locking + atomic-write pattern as JsonRecordingsRepository and
JsonTrainingRepository. On first init, seeds FIXED_PERSONAS so the
seeded list always contains the four family members.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .models import FIXED_PERSONAS, Persona

log = logging.getLogger(__name__)


class PersonaNotFound(LookupError):
    """Raised when a persona_id is not in the repository."""


class JsonPersonaRepository:
    """JSON-backed persona repository with file locking + atomic writes."""

    INDEX_FILENAME = "personas.json"

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self.personas_dir = self.data_root / "personas"
        self.index_path = self.personas_dir / self.INDEX_FILENAME
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        # Seed on first init so the fixed family is always present.
        if not self.index_path.exists():
            self._write_all([p.model_copy() for p in FIXED_PERSONAS])

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def list(self) -> list[Persona]:
        return self._read_all()

    def get(self, persona_id: str) -> Persona:
        persona = self.get_or_none(persona_id)
        if persona is None:
            raise PersonaNotFound(persona_id)
        return persona

    def get_or_none(self, persona_id: str) -> Optional[Persona]:
        for p in self._read_all():
            if p.persona_id == persona_id:
                return p
        return None

    def exists(self, persona_id: str) -> bool:
        return self.get_or_none(persona_id) is not None

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------
    def save(self, persona: Persona) -> None:
        """Insert or replace a persona."""
        with self._exclusive_lock(self.index_path):
            personas = self._read_all_locked()
            replaced = False
            for i, p in enumerate(personas):
                if p.persona_id == persona.persona_id:
                    personas[i] = persona
                    replaced = True
                    break
            if not replaced:
                personas.append(persona)
            self._write_all_locked(personas)

    def delete(self, persona_id: str) -> None:
        with self._exclusive_lock(self.index_path):
            personas = self._read_all_locked()
            new_list = [p for p in personas if p.persona_id != persona_id]
            if len(new_list) == len(personas):
                raise PersonaNotFound(persona_id)
            self._write_all_locked(new_list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_all(self) -> list[Persona]:
        if not self.index_path.exists():
            return []
        with self._shared_lock(self.index_path):
            return self._read_all_locked()

    def _read_all_locked(self) -> list[Persona]:
        if not self.index_path.exists():
            return []
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return []
            data = json.loads(content)
        except json.JSONDecodeError as e:
            log.error("personas.json is corrupt: %s", e)
            return []
        personas: list[Persona] = []
        for raw in data if isinstance(data, list) else []:
            try:
                personas.append(Persona.model_validate(raw))
            except Exception as e:
                log.warning("Skipping invalid persona entry %s: %s", raw, e)
        return personas

    def _write_all(self, personas: list[Persona]) -> None:
        with self._exclusive_lock(self.index_path):
            self._write_all_locked(personas)

    def _write_all_locked(self, personas: list[Persona]) -> None:
        payload = [p.model_dump(mode="json") for p in personas]
        _atomic_write_json(self.index_path, payload)

    # ------------------------------------------------------------------
    # POSIX file-locking primitives.
    # ------------------------------------------------------------------
    @contextmanager
    def _exclusive_lock(self, path: Path) -> Iterator[None]:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @contextmanager
    def _shared_lock(self, path: Path) -> Iterator[None]:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _atomic_write_json(path: Path, data: object) -> None:
    """tempfile → fsync → rename. Same pattern as the other repos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
