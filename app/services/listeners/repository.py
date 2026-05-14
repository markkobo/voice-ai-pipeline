"""Listener repository — same locking + atomic-write pattern as personas."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .models import SEED_LISTENERS, Listener

log = logging.getLogger(__name__)


class ListenerNotFound(LookupError):
    """Raised when a listener_id is not in the repository."""


class JsonListenerRepository:
    """JSON-backed listener repository."""

    INDEX_FILENAME = "listeners.json"

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self.listeners_dir = self.data_root / "listeners"
        self.index_path = self.listeners_dir / self.INDEX_FILENAME
        self.listeners_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_all([l.model_copy() for l in SEED_LISTENERS])

    def list(self) -> list[Listener]:
        return self._read_all()

    def get(self, listener_id: str) -> Listener:
        listener = self.get_or_none(listener_id)
        if listener is None:
            raise ListenerNotFound(listener_id)
        return listener

    def get_or_none(self, listener_id: str) -> Optional[Listener]:
        for l in self._read_all():
            if l.listener_id == listener_id:
                return l
        return None

    def exists(self, listener_id: str) -> bool:
        return self.get_or_none(listener_id) is not None

    def save(self, listener: Listener) -> None:
        with self._exclusive_lock(self.index_path):
            listeners = self._read_all_locked()
            replaced = False
            for i, l in enumerate(listeners):
                if l.listener_id == listener.listener_id:
                    listeners[i] = listener
                    replaced = True
                    break
            if not replaced:
                listeners.append(listener)
            self._write_all_locked(listeners)

    def delete(self, listener_id: str) -> None:
        with self._exclusive_lock(self.index_path):
            listeners = self._read_all_locked()
            new_list = [l for l in listeners if l.listener_id != listener_id]
            if len(new_list) == len(listeners):
                raise ListenerNotFound(listener_id)
            self._write_all_locked(new_list)

    def _read_all(self) -> list[Listener]:
        if not self.index_path.exists():
            return []
        with self._shared_lock(self.index_path):
            return self._read_all_locked()

    def _read_all_locked(self) -> list[Listener]:
        if not self.index_path.exists():
            return []
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return []
            data = json.loads(content)
        except json.JSONDecodeError as e:
            log.error("listeners.json is corrupt: %s", e)
            return []
        listeners: list[Listener] = []
        for raw in data if isinstance(data, list) else []:
            try:
                listeners.append(Listener.model_validate(raw))
            except Exception as e:
                log.warning("Skipping invalid listener entry %s: %s", raw, e)
        return listeners

    def _write_all(self, listeners: list[Listener]) -> None:
        with self._exclusive_lock(self.index_path):
            self._write_all_locked(listeners)

    def _write_all_locked(self, listeners: list[Listener]) -> None:
        payload = [l.model_dump(mode="json") for l in listeners]
        _atomic_write_json(self.index_path, payload)

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
