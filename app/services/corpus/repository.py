"""
JSON-backed repository for corpus items.

Mirrors `JsonRecordingsRepository`:
- shared/exclusive POSIX flock on a sentinel `.lock` file
- atomic-rename writes via tempfile + fsync + os.replace
- separate index.json for fast lookup, per-item metadata.json as source of
  truth

Storage:

    {corpus_root}/index.json                            {item_id: rel_path}
    {corpus_root}/<kind>/<item_id>/metadata.json        CorpusItem JSON
    {corpus_root}/<kind>/<item_id>/original.<ext>       raw upload bytes

`corpus_root` is `data_root/personas/<persona_id>/corpus`. Each persona has
its own corpus tree — no cross-persona index, no cross-persona lookup.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

from .models import CorpusItem

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions — repository layer; service maps these to domain errors.
# ---------------------------------------------------------------------------
class CorpusItemNotFound(LookupError):
    """Raised when an item_id is not present in the persona's index."""


class CorruptCorpusMetadata(ValueError):
    """Raised when an item metadata.json fails to parse / validate."""


# ---------------------------------------------------------------------------
# Protocol — service depends on this, not the concrete impl.
# ---------------------------------------------------------------------------
class CorpusRepository(Protocol):
    def get(self, persona_id: str, item_id: str) -> CorpusItem: ...
    def get_or_none(self, persona_id: str, item_id: str) -> Optional[CorpusItem]: ...
    def list(self, persona_id: str) -> list[CorpusItem]: ...
    def save(self, item: CorpusItem) -> None: ...
    def delete(self, persona_id: str, item_id: str) -> None: ...
    def item_dir(self, persona_id: str, item_id: str) -> Path: ...
    def update(
        self,
        persona_id: str,
        item_id: str,
        mutator: Callable[[CorpusItem], None],
    ) -> CorpusItem: ...


# ---------------------------------------------------------------------------
# JSON-backed implementation
# ---------------------------------------------------------------------------
class JsonCorpusRepository:
    INDEX_FILENAME = "index.json"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, personas_root: Path) -> None:
        """
        Args:
            personas_root: data_root/personas (NOT per-persona; this repo
                handles all personas, dispatching by persona_id).
        """
        self.personas_root = Path(personas_root)
        self.personas_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def corpus_root(self, persona_id: str) -> Path:
        return self.personas_root / persona_id / "corpus"

    def _index_path(self, persona_id: str) -> Path:
        return self.corpus_root(persona_id) / self.INDEX_FILENAME

    def item_dir(self, persona_id: str, item_id: str) -> Path:
        """Per-item directory. Reads the index to find the kind subdir."""
        rel = self._lookup_rel(persona_id, item_id)
        if rel is None:
            raise CorpusItemNotFound(item_id)
        return self.corpus_root(persona_id) / rel

    def item_dir_for_kind(self, persona_id: str, kind: str, item_id: str) -> Path:
        return self.corpus_root(persona_id) / kind / item_id

    # ------------------------------------------------------------------
    # Public CRUD
    # ------------------------------------------------------------------
    def get(self, persona_id: str, item_id: str) -> CorpusItem:
        item = self.get_or_none(persona_id, item_id)
        if item is None:
            raise CorpusItemNotFound(item_id)
        return item

    def get_or_none(self, persona_id: str, item_id: str) -> Optional[CorpusItem]:
        rel = self._lookup_rel(persona_id, item_id)
        if rel is None:
            return None
        meta_path = self.corpus_root(persona_id) / rel / self.METADATA_FILENAME
        if not meta_path.exists():
            return None
        with self._shared_lock(meta_path):
            return self._read_metadata_locked(meta_path)

    def list(self, persona_id: str) -> list[CorpusItem]:
        index = self._read_index(persona_id)
        results: list[CorpusItem] = []
        root = self.corpus_root(persona_id)
        for iid, rel in index.items():
            meta_path = root / rel / self.METADATA_FILENAME
            if not meta_path.exists():
                log.warning("Dangling index entry %s → %s (no metadata.json)", iid, rel)
                continue
            try:
                with self._shared_lock(meta_path):
                    results.append(self._read_metadata_locked(meta_path))
            except CorruptCorpusMetadata as e:
                log.warning("Skipping corrupt item %s: %s", iid, e)
                continue
        results.sort(key=lambda i: i.created_at, reverse=True)
        return results

    def save(self, item: CorpusItem) -> None:
        kind = item.kind.value
        item_path = self.corpus_root(item.persona_id) / kind / item.item_id
        item_path.mkdir(parents=True, exist_ok=True)
        meta_path = item_path / self.METADATA_FILENAME

        item.updated_at = datetime.now(timezone.utc)
        payload = item.model_dump(mode="json")

        with self._exclusive_lock(meta_path):
            self._atomic_write_json(meta_path, payload)

        rel = f"{kind}/{item.item_id}"
        self._index_set(item.persona_id, item.item_id, rel)

    def delete(self, persona_id: str, item_id: str) -> None:
        """Delete an item — race-safe vs a concurrent `update` on the
        same item (review #12).

        Strategy: hold the item's exclusive metadata lock across
        rmtree + index-remove. A concurrent `update` blocks on the lock,
        then sees the metadata file missing (FileNotFoundError) and
        bails with CorpusItemNotFound — vs the prior failure where it
        could resurrect a deleted item dir via mkdir(parents=True).
        """
        rel = self._lookup_rel(persona_id, item_id)
        if rel is None:
            raise CorpusItemNotFound(item_id)
        item_path = self.corpus_root(persona_id) / rel
        meta_path = item_path / self.METADATA_FILENAME

        with self._exclusive_lock(meta_path):
            if item_path.exists():
                shutil.rmtree(item_path)
            self._index_remove(persona_id, item_id)

    def update(
        self,
        persona_id: str,
        item_id: str,
        mutator: Callable[[CorpusItem], None],
    ) -> CorpusItem:
        rel = self._lookup_rel(persona_id, item_id)
        if rel is None:
            raise CorpusItemNotFound(item_id)
        meta_path = self.corpus_root(persona_id) / rel / self.METADATA_FILENAME
        with self._exclusive_lock(meta_path):
            # Re-check inside the lock — a concurrent delete may have
            # rmtree'd the dir between our index lookup and our lock
            # acquisition (review #12). Without this re-check, the
            # subsequent _atomic_write_json would mkdir(parents=True)
            # and resurrect a deleted item dir.
            if not meta_path.exists():
                raise CorpusItemNotFound(item_id)
            item = self._read_metadata_locked(meta_path)
            mutator(item)
            item.updated_at = datetime.now(timezone.utc)
            self._atomic_write_json(meta_path, item.model_dump(mode="json"))
        return item

    # ------------------------------------------------------------------
    # Index helpers — per-persona index.json under exclusive lock.
    # ------------------------------------------------------------------
    def _lookup_rel(self, persona_id: str, item_id: str) -> Optional[str]:
        return self._read_index(persona_id).get(item_id)

    def _read_index(self, persona_id: str) -> dict[str, str]:
        path = self._index_path(persona_id)
        if not path.exists():
            return {}
        with self._shared_lock(path):
            return self._read_index_locked(path)

    def _read_index_locked(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.error("corpus index.json %s is corrupt: %s", path, e)
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}

    def _index_set(self, persona_id: str, item_id: str, rel: str) -> None:
        path = self._index_path(persona_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock(path):
            index = self._read_index_locked(path)
            index[item_id] = rel
            self._atomic_write_json(path, index)

    def _index_remove(self, persona_id: str, item_id: str) -> None:
        path = self._index_path(persona_id)
        if not path.exists():
            return
        with self._exclusive_lock(path):
            index = self._read_index_locked(path)
            index.pop(item_id, None)
            self._atomic_write_json(path, index)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _read_metadata_locked(self, meta_path: Path) -> CorpusItem:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptCorpusMetadata(
                f"{meta_path} is not valid JSON: {e}"
            ) from e
        try:
            item = CorpusItem.model_validate(data)
        except Exception as e:  # pydantic ValidationError + anything else
            raise CorruptCorpusMetadata(
                f"{meta_path} failed schema validation: {e}"
            ) from e
        # Coerce any tz-naive datetimes (legacy `datetime.utcnow()` writes)
        # to UTC so the sort by `created_at` in `list()` never mixes naive and
        # aware datetimes. See note in recordings/repository.py.
        _coerce_naive_datetimes_to_utc(item)
        return item

    # ------------------------------------------------------------------
    # POSIX flock + atomic-rename primitives — copied from
    # JsonRecordingsRepository (same semantics).
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

    @staticmethod
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
                json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise


def _json_default(o: object) -> object:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _coerce_naive_datetimes_to_utc(obj: object) -> None:
    """Walk a Pydantic model and replace tz-naive datetimes with UTC-tagged
    equivalents. See `app/services/recordings/repository.py` for the full
    rationale — kept as a per-module local to avoid a shared utils module."""
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover
        return

    if isinstance(obj, BaseModel):
        for field_name in obj.__class__.model_fields:
            value = getattr(obj, field_name, None)
            if isinstance(value, datetime) and value.tzinfo is None:
                setattr(obj, field_name, value.replace(tzinfo=timezone.utc))
            else:
                _coerce_naive_datetimes_to_utc(value)
    elif isinstance(obj, list):
        for item in obj:
            _coerce_naive_datetimes_to_utc(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            _coerce_naive_datetimes_to_utc(item)
