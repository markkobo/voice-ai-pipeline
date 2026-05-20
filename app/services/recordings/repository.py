"""
Recordings repository — load/save Recording aggregates.

Two layers:
- `RecordingsRepository` protocol — minimal interface for the service.
- `JsonRecordingsRepository` — file-backed implementation with POSIX file
  locking, atomic rename writes, and an index.json for fast lookup.

The locking strategy fixes the documented race in metadata.py:save() where
concurrent PATCH requests could clobber each other's writes:
  https://github.com/.../voice-ai-pipeline/blob/master/app/services/recordings/metadata.py#L80

Locking rules:
- Read paths take a shared lock (LOCK_SH) on the file.
- Write paths take an exclusive lock (LOCK_EX) and use atomic rename.
- `update(recording_id, mutator)` is the canonical read-modify-write — it
  holds an exclusive lock for the duration of the mutator callback so two
  concurrent PATCHes converge on a single consistent state.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

from .models import Recording

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions raised by the repository — service layer converts these to
# domain errors, API layer converts those to HTTP. The repository itself
# does not know about HTTP.
# ---------------------------------------------------------------------------
class RecordingNotFound(LookupError):
    """Raised when a recording_id is not present in the index."""


class CorruptMetadata(ValueError):
    """Raised when metadata.json fails to parse or schema-validate."""


# ---------------------------------------------------------------------------
# Protocol — service depends on this, not the concrete impl.
# ---------------------------------------------------------------------------
class RecordingsRepository(Protocol):
    def get(self, recording_id: str) -> Recording: ...
    def get_or_none(self, recording_id: str) -> Optional[Recording]: ...
    def list(self) -> list[Recording]: ...
    def save(self, recording: Recording) -> None: ...
    def delete(self, recording_id: str) -> None: ...
    def exists(self, recording_id: str) -> bool: ...
    def update(
        self,
        recording_id: str,
        mutator: Callable[[Recording], None],
    ) -> Recording: ...


# ---------------------------------------------------------------------------
# JSON-backed implementation
# ---------------------------------------------------------------------------
class JsonRecordingsRepository:
    """
    File-backed repository.

    Storage layout:
        {data_root}/recordings/index.json       — { recording_id: folder_name }
        {data_root}/recordings/raw/{folder}/metadata.json  — Recording JSON

    The index is the source of truth for "which recordings exist". metadata.json
    is the source of truth for the recording's state.
    """

    INDEX_FILENAME = "index.json"
    METADATA_FILENAME = "metadata.json"
    RAW_DIRNAME = "raw"

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self.recordings_root = self.data_root / "recordings"
        self.raw_root = self.recordings_root / self.RAW_DIRNAME
        self.index_path = self.recordings_root / self.INDEX_FILENAME

        self.recordings_root.mkdir(parents=True, exist_ok=True)
        self.raw_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, recording_id: str) -> Recording:
        recording = self.get_or_none(recording_id)
        if recording is None:
            raise RecordingNotFound(recording_id)
        return recording

    def get_or_none(self, recording_id: str) -> Optional[Recording]:
        folder = self._lookup_folder(recording_id)
        if folder is None:
            return None
        return self._read_metadata(folder)

    def list(self) -> list[Recording]:
        index = self._read_index()
        results: list[Recording] = []
        for rid, folder in index.items():
            try:
                results.append(self._read_metadata(folder))
            except (RecordingNotFound, CorruptMetadata) as e:
                log.warning("Skipping corrupt or missing recording %s: %s", rid, e)
                continue
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results

    def exists(self, recording_id: str) -> bool:
        return self._lookup_folder(recording_id) is not None

    def save(self, recording: Recording) -> None:
        """
        Persist a Recording, taking an exclusive lock.

        Also registers the recording in the index if it's not there yet.
        """
        folder = recording.folder_name
        folder_path = self.raw_root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        metadata_path = folder_path / self.METADATA_FILENAME

        recording.updated_at = datetime.now(timezone.utc)
        payload = recording.model_dump(mode="json")

        with self._exclusive_lock(metadata_path):
            self._atomic_write_json(metadata_path, payload)

        # Maintain the index — also under exclusive lock to avoid lost writes
        # when two recordings are saved concurrently.
        self._index_set(recording.recording_id, folder)

    def delete(self, recording_id: str) -> None:
        folder = self._lookup_folder(recording_id)
        if folder is None:
            raise RecordingNotFound(recording_id)
        folder_path = self.raw_root / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
        self._index_remove(recording_id)

    def update(
        self,
        recording_id: str,
        mutator: Callable[[Recording], None],
    ) -> Recording:
        """
        Atomic read-modify-write.

        Two concurrent calls converge to a single consistent state: the second
        sees the first's mutation in its `mutator(recording)` call.
        """
        folder = self._lookup_folder(recording_id)
        if folder is None:
            raise RecordingNotFound(recording_id)
        metadata_path = self.raw_root / folder / self.METADATA_FILENAME

        with self._exclusive_lock(metadata_path):
            recording = self._read_metadata_locked(metadata_path)
            mutator(recording)
            recording.updated_at = datetime.now(timezone.utc)
            self._atomic_write_json(metadata_path, recording.model_dump(mode="json"))
        return recording

    # ------------------------------------------------------------------
    # Index helpers — small and well-locked.
    # ------------------------------------------------------------------
    def _lookup_folder(self, recording_id: str) -> Optional[str]:
        index = self._read_index()
        return index.get(recording_id)

    def _read_index(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        with self._shared_lock(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                log.error("index.json is corrupt: %s", e)
                return {}
        # Tolerate both new shape ({rid: folder}) and legacy shape
        # ({"recordings": [{recording_id, folder_name, ...}, ...]}).
        if isinstance(data, dict) and "recordings" in data and isinstance(data["recordings"], list):
            return {
                r["recording_id"]: r["folder_name"]
                for r in data["recordings"]
                if isinstance(r, dict) and "recording_id" in r and "folder_name" in r
            }
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}

    def _index_set(self, recording_id: str, folder_name: str) -> None:
        with self._exclusive_lock(self.index_path):
            index = self._read_index_locked()
            index[recording_id] = folder_name
            self._atomic_write_json(self.index_path, index)

    def _index_remove(self, recording_id: str) -> None:
        with self._exclusive_lock(self.index_path):
            index = self._read_index_locked()
            index.pop(recording_id, None)
            self._atomic_write_json(self.index_path, index)

    def _read_index_locked(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.error("index.json is corrupt during locked read: %s", e)
            return {}
        if isinstance(data, dict) and "recordings" in data and isinstance(data["recordings"], list):
            return {
                r["recording_id"]: r["folder_name"]
                for r in data["recordings"]
                if isinstance(r, dict) and "recording_id" in r and "folder_name" in r
            }
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}

    # ------------------------------------------------------------------
    # Metadata read helpers
    # ------------------------------------------------------------------
    def _read_metadata(self, folder_name: str) -> Recording:
        metadata_path = self.raw_root / folder_name / self.METADATA_FILENAME
        if not metadata_path.exists():
            raise RecordingNotFound(folder_name)
        with self._shared_lock(metadata_path):
            return self._read_metadata_locked(metadata_path)

    def _read_metadata_locked(self, metadata_path: Path) -> Recording:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptMetadata(f"{metadata_path} is not valid JSON: {e}") from e
        try:
            recording = Recording.model_validate(data)
        except Exception as e:  # ValidationError or anything from Pydantic
            raise CorruptMetadata(f"{metadata_path} failed schema validation: {e}") from e
        # Older metadata.json files were written with `datetime.utcnow().isoformat()`
        # — tz-naive. Newer code writes `datetime.now(timezone.utc).isoformat()` —
        # tz-aware. Mixing the two crashes any code that compares the resulting
        # datetimes (e.g. the sort in `list()`). Coerce naive → UTC at the boundary
        # so downstream code can assume aware datetimes everywhere. The on-disk
        # representation is left untouched.
        _coerce_naive_datetimes_to_utc(recording)
        return recording

    # ------------------------------------------------------------------
    # File locking + atomic write primitives.
    # ------------------------------------------------------------------
    @contextmanager
    def _exclusive_lock(self, path: Path) -> Iterator[None]:
        """
        Hold an exclusive POSIX flock on a sentinel file next to `path`.

        We lock a separate `.lock` file so the atomic rename of the target
        doesn't interfere with the lock — renaming a locked file is fine on
        POSIX, but using a sentinel keeps the semantics obvious.
        """
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
        """Hold a shared POSIX flock — multiple readers OK, writers blocked."""
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
        """
        Write JSON atomically: tempfile in same dir → fsync → rename.

        This guarantees a reader never sees a partially-written file. Combined
        with the surrounding flock, it also guarantees no lost writes between
        concurrent writers.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # Same-directory temp ensures rename() is atomic on POSIX.
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


# ---------------------------------------------------------------------------
# JSON default — handles datetime which pydantic's model_dump(mode="json")
# already converts to ISO strings, but covers raw datetimes that might sneak
# in via the model_dump output.
# ---------------------------------------------------------------------------
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _json_default(o: object) -> object:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _coerce_naive_datetimes_to_utc(obj: object) -> None:
    """
    Recursively replace any tz-naive `datetime` fields on a Pydantic model
    (and nested models / lists / dicts thereof) with the same instant tagged
    as UTC.

    Rationale: some legacy `metadata.json` files on disk were written with
    `datetime.utcnow()` (tz-naive), while newer code writes
    `datetime.now(timezone.utc)` (tz-aware). Pydantic happily parses both,
    but comparing them — as `list()` does when sorting by `created_at` —
    raises `TypeError: can't compare offset-naive and offset-aware datetimes`.
    Coercing at the load boundary keeps downstream code simple.

    Mutates `obj` in place. Safe to call on any value — non-models are no-ops.
    """
    # Local import to avoid a top-level dependency cycle: pydantic is already
    # imported transitively via .models, but we want this helper resilient.
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover - pydantic is a hard dep
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
