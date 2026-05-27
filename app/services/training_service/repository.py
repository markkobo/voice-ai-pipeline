"""
Training repository — load/save TrainingVersion + ActiveVersion + manifests.

Backing storage:
    {models_dir}/index.json                            — version list + active version
    {models_dir}/{persona}_{version_id}/manifest.json  — per-version manifest
    {models_dir}/{persona}_{version_id}/progress.json  — written by subprocess

Concurrency model — same as JsonRecordingsRepository in Phase 1.1:
- Reads take a shared POSIX flock (LOCK_SH).
- Writes take an exclusive lock (LOCK_EX) and use atomic rename.
- `update(version_id, mutator)` is the canonical read-modify-write — the lock
  is held for the full duration of the mutator callback so concurrent
  status updates don't lose writes.

Unlike the legacy `VersionManager`, there is NO in-memory cache: every read
hits disk under the shared lock. The subprocess that writes progress.json
sees a coherent file; the API reading the version list sees the latest state.
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

from .models import (
    ActiveVersion,
    TrainingManifest,
    TrainingProgressSnapshot,
    TrainingVersion,
    VersionStatus,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions (the service maps these to DomainError subclasses).
# ---------------------------------------------------------------------------
class TrainingVersionNotFound(LookupError):
    """Raised when a version_id is not present in the index."""


class CorruptTrainingIndex(ValueError):
    """Raised when index.json fails to parse or schema-validate."""


# ---------------------------------------------------------------------------
# Protocol — service depends on this, not the concrete impl.
# ---------------------------------------------------------------------------
class TrainingRepository(Protocol):
    def get(self, version_id: str) -> TrainingVersion: ...
    def get_or_none(self, version_id: str) -> Optional[TrainingVersion]: ...
    def list(self, persona_id: Optional[str] = None) -> list[TrainingVersion]: ...
    def save(self, version: TrainingVersion) -> None: ...
    def delete(self, version_id: str) -> None: ...
    def exists(self, version_id: str) -> bool: ...
    def update(
        self,
        version_id: str,
        mutator: Callable[[TrainingVersion], None],
    ) -> TrainingVersion: ...
    def get_active(self, persona_id: str) -> Optional[ActiveVersion]: ...
    def set_active(self, active: ActiveVersion) -> None: ...
    def clear_active_if(self, version_id: str) -> None: ...
    def save_manifest(self, version_id: str, manifest: TrainingManifest) -> None: ...
    def get_manifest(self, version_id: str) -> Optional[TrainingManifest]: ...
    def read_progress(self, version_id: str) -> Optional[TrainingProgressSnapshot]: ...


# ---------------------------------------------------------------------------
# JSON-backed implementation
# ---------------------------------------------------------------------------
class JsonTrainingRepository:
    """
    File-backed repository.

    The index file's on-disk shape is:
        {
            "versions": [TrainingVersion.to_legacy_dict() for each],
            "active_version": {"persona_id": ..., "version_id": ...} | null
        }

    That matches what the legacy `VersionManager` writes, so a fresh
    JsonTrainingRepository pointed at a directory the old code wrote into
    will load cleanly. New writes use the same shape.
    """

    INDEX_FILENAME = "index.json"
    MANIFEST_FILENAME = "manifest.json"
    PROGRESS_FILENAME = "progress.json"

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self.index_path = self.models_dir / self.INDEX_FILENAME
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------
    def get(self, version_id: str) -> TrainingVersion:
        version = self.get_or_none(version_id)
        if version is None:
            raise TrainingVersionNotFound(version_id)
        return version

    def get_or_none(self, version_id: str) -> Optional[TrainingVersion]:
        for v in self._read_versions():
            if v.version_id == version_id:
                return v
        return None

    def list(self, persona_id: Optional[str] = None) -> list[TrainingVersion]:
        versions = self._read_versions()
        if persona_id is not None:
            versions = [v for v in versions if v.persona_id == persona_id]
        # Sort newest first by created_at, falling back to version_id.
        # `created_at` may be missing (None) for legacy entries; we substitute
        # an aware sentinel so we never mix naive/aware datetimes in the
        # comparison. `_coerce_naive_datetimes_to_utc` (in _read_index_locked)
        # ensures the populated created_at values are also tz-aware.
        _EPOCH_UTC = datetime.min.replace(tzinfo=timezone.utc)
        versions.sort(
            key=lambda v: (v.created_at or _EPOCH_UTC, v.version_id),
            reverse=True,
        )
        return versions

    def exists(self, version_id: str) -> bool:
        return self.get_or_none(version_id) is not None

    def get_active(self, persona_id: str) -> Optional[ActiveVersion]:
        active = self._read_active()
        if active and active.persona_id == persona_id:
            return active
        return None

    def get_manifest(self, version_id: str) -> Optional[TrainingManifest]:
        version = self.get_or_none(version_id)
        if version is None or not version.lora_path:
            return None
        manifest_path = Path(version.lora_path) / self.MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        with self._shared_lock(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                log.warning("Manifest for %s is corrupt: %s", version_id, e)
                return None
        try:
            return TrainingManifest.model_validate(data)
        except Exception as e:  # pydantic ValidationError
            log.warning("Manifest schema mismatch for %s: %s", version_id, e)
            return None

    def read_progress(self, version_id: str) -> Optional[TrainingProgressSnapshot]:
        """
        Read the subprocess-written progress.json under a shared lock.

        Returns None if the file does not exist or fails to parse — the SSE
        generator treats those cases as "no update available yet".

        Resilience: `version.lora_path` is an absolute path captured at job
        creation time, so it can go stale if the project is moved between
        hosts (observed: a version created with cwd `/workspace/...` now
        running under `/home/rding/...`). When the stored path's
        progress.json is missing, fall back to the canonical layout
        `{models_dir}/{persona}_{version_id}/progress.json` — the
        repository's own root — before giving up. Otherwise the status-bar
        training pill loses its epoch/% indicators after a directory move.
        """
        version = self.get_or_none(version_id)
        if version is None or not version.lora_path:
            return None
        progress_path = Path(version.lora_path) / self.PROGRESS_FILENAME
        if not progress_path.exists():
            # Fallback: derive from this repository's models_dir + the
            # standard `{persona}_{version_id}` directory name.
            fallback_dir = self.models_dir / f"{version.persona_id}_{version_id}"
            fallback_path = fallback_dir / self.PROGRESS_FILENAME
            if fallback_path.exists():
                log.debug(
                    "read_progress: stored lora_path %s stale; using fallback %s",
                    version.lora_path, fallback_dir,
                )
                progress_path = fallback_path
            else:
                return None
        with self._shared_lock(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                log.warning("progress.json for %s is corrupt: %s", version_id, e)
                return None
        # The training subprocess (training_job.py) historically wrote a
        # sparse progress.json that did not include `version_id`, while
        # `TrainingProgressSnapshot` requires it. The path already encodes
        # the version_id (we read it from {version_dir}/progress.json), so
        # inject it as a defensive fallback. Without this, validation
        # silently fails → refresh_status_from_progress() returns None →
        # the index entry stays in `status=training` forever even after
        # the subprocess writes `status=ready` to progress.json
        # (observed for v1_20260521_104108_618646 on 2026-05-21).
        if not isinstance(data, dict):
            log.warning("progress.json for %s is not an object: %r", version_id, type(data))
            return None
        data.setdefault("version_id", version_id)
        try:
            return TrainingProgressSnapshot.model_validate(data)
        except Exception as e:
            log.warning("progress.json schema mismatch for %s: %s", version_id, e)
            return None

    # ------------------------------------------------------------------
    # Public write API — all under exclusive lock + atomic rename.
    # ------------------------------------------------------------------
    def save(self, version: TrainingVersion) -> None:
        """Insert or replace a version in the index."""
        with self._exclusive_lock(self.index_path):
            versions, active = self._read_index_locked()
            # Replace if exists, else append.
            replaced = False
            for i, v in enumerate(versions):
                if v.version_id == version.version_id:
                    versions[i] = version
                    replaced = True
                    break
            if not replaced:
                versions.append(version)
            self._write_index_locked(versions, active)

    def delete(self, version_id: str) -> None:
        """Remove a version + its on-disk artifacts.

        Merged-dir deletion uses `target.merged_path` as the SINGLE source
        of truth — never re-derives the path from `lora_path.name`. The
        re-derive path is what caused the 2026-05-25 catastrophe where
        deleting a bad SFT v2_20260525 also wiped the user's working
        v2_20260514 merged dir (both derived to the same legacy
        `parts[:3]` name `merged_qwen3_tts_xiao_s_v2`).

        Before deleting the merged dir, we also scan remaining versions
        for any whose stored `merged_path` points at the same directory
        — if found, we skip the rmtree (still drop the index entry) and
        warn. Defense-in-depth in case a future code path manages to set
        two versions' `merged_path` to the same on-disk dir.
        """
        with self._exclusive_lock(self.index_path):
            versions, active = self._read_index_locked()
            target: Optional[TrainingVersion] = None
            for v in versions:
                if v.version_id == version_id:
                    target = v
                    break
            if target is None:
                raise TrainingVersionNotFound(version_id)
            # Refuse to delete the active version — the service should clear
            # the active pointer first if it really wants to delete.
            if active and active.version_id == version_id:
                raise ValueError(
                    f"Refusing to delete active version {version_id}; "
                    "clear active first via clear_active_if()"
                )
            remaining = [v for v in versions if v.version_id != version_id]
            self._write_index_locked(remaining, active)

        # Best-effort filesystem cleanup outside the lock.
        if target.lora_path:
            lora_path = Path(target.lora_path)
            if lora_path.exists():
                try:
                    shutil.rmtree(lora_path)
                except OSError as e:
                    log.warning("Failed to remove %s: %s", lora_path, e)

        # Merged dir: use stored merged_path only. No re-derivation.
        if target.merged_path:
            merged_path = Path(target.merged_path)
            # Guard against shared paths — refuse to rm if any other
            # version still claims this same merged_path.
            sharers = [
                v.version_id
                for v in remaining
                if v.merged_path and Path(v.merged_path) == merged_path
            ]
            if sharers:
                log.warning(
                    "Refusing to delete merged dir %s for %s — also "
                    "claimed by %s. Index entry removed; files left "
                    "on disk.",
                    merged_path, version_id, sharers,
                )
            elif merged_path.exists():
                try:
                    shutil.rmtree(merged_path)
                except OSError as e:
                    log.warning("Failed to remove merged %s: %s", merged_path, e)
        elif target.model_type in ("sft", "custom_voice", "custom_voice_compatible"):
            # No merged_path on the version → don't guess. Log so the
            # operator notices and can clean up manually.
            log.warning(
                "Version %s is %s but has no merged_path set; not "
                "attempting to derive/delete a merged dir.",
                version_id, target.model_type,
            )

    def update(
        self,
        version_id: str,
        mutator: Callable[[TrainingVersion], None],
    ) -> TrainingVersion:
        """Atomic read-modify-write on a single version."""
        with self._exclusive_lock(self.index_path):
            versions, active = self._read_index_locked()
            target_idx: Optional[int] = None
            for i, v in enumerate(versions):
                if v.version_id == version_id:
                    target_idx = i
                    break
            if target_idx is None:
                raise TrainingVersionNotFound(version_id)
            target = versions[target_idx]
            mutator(target)
            versions[target_idx] = target
            self._write_index_locked(versions, active)
            return target

    def set_active(self, active: ActiveVersion) -> None:
        with self._exclusive_lock(self.index_path):
            versions, _ = self._read_index_locked()
            # Sanity-check that the target version exists and is ready.
            target = next(
                (v for v in versions if v.version_id == active.version_id),
                None,
            )
            if target is None:
                raise TrainingVersionNotFound(active.version_id)
            if target.status != VersionStatus.ready:
                raise ValueError(
                    f"Cannot activate version {active.version_id} with status "
                    f"{target.status.value!r} (must be 'ready')"
                )
            self._write_index_locked(versions, active)

    def clear_active_if(self, version_id: str) -> None:
        """Clear the active pointer iff it points at `version_id`."""
        with self._exclusive_lock(self.index_path):
            versions, active = self._read_index_locked()
            if active is None or active.version_id != version_id:
                return
            self._write_index_locked(versions, None)

    def save_manifest(self, version_id: str, manifest: TrainingManifest) -> None:
        version = self.get_or_none(version_id)
        if version is None or not version.lora_path:
            raise TrainingVersionNotFound(version_id)
        manifest_path = Path(version.lora_path) / self.MANIFEST_FILENAME
        with self._exclusive_lock(manifest_path):
            self._atomic_write_json(manifest_path, manifest.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_versions(self) -> list[TrainingVersion]:
        versions, _ = self._read_index_unlocked()
        return versions

    def _read_active(self) -> Optional[ActiveVersion]:
        _, active = self._read_index_unlocked()
        return active

    def _read_index_unlocked(self) -> tuple[list[TrainingVersion], Optional[ActiveVersion]]:
        """Read with a shared lock so we don't see a partial write."""
        if not self.index_path.exists():
            return ([], None)
        with self._shared_lock(self.index_path):
            return self._read_index_locked()

    def _read_index_locked(self) -> tuple[list[TrainingVersion], Optional[ActiveVersion]]:
        """Read assuming the caller already holds a lock on index_path."""
        if not self.index_path.exists():
            return ([], None)
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.error("index.json is corrupt during locked read: %s", e)
            raise CorruptTrainingIndex(str(e)) from e
        raw_versions = data.get("versions", []) if isinstance(data, dict) else []
        versions: list[TrainingVersion] = []
        for raw in raw_versions:
            try:
                v = TrainingVersion.model_validate(raw)
            except Exception as e:
                log.warning("Skipping invalid version entry: %s (%s)", raw, e)
                continue
            # Legacy index.json entries may carry tz-naive `created_at` /
            # `completed_at`. Coerce so the `list()` sort never mixes naive
            # and aware datetimes.
            _coerce_naive_datetimes_to_utc(v)
            versions.append(v)
        raw_active = data.get("active_version") if isinstance(data, dict) else None
        active: Optional[ActiveVersion] = None
        if raw_active:
            try:
                active = ActiveVersion.model_validate(raw_active)
            except Exception as e:
                log.warning("Active version entry invalid, ignoring: %s (%s)", raw_active, e)
            else:
                _coerce_naive_datetimes_to_utc(active)
        return (versions, active)

    def _write_index_locked(
        self,
        versions: list[TrainingVersion],
        active: Optional[ActiveVersion],
    ) -> None:
        """Write index.json atomically. Caller must hold the exclusive lock."""
        payload = {
            "versions": [v.to_legacy_dict() for v in versions],
            "active_version": active.model_dump(mode="json") if active else None,
        }
        self._atomic_write_json(self.index_path, payload)

    # ------------------------------------------------------------------
    # POSIX file locking + atomic write primitives.
    # Same shape as JsonRecordingsRepository for consistency.
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
                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )
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
    equivalents. See `app/services/recordings/repository.py` for rationale.
    Kept as a per-module local to avoid a shared utils module."""
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
