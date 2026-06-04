"""
JSON-backed repository for consent records.

Storage layout (mirrors the per-persona corpus tree):

    {personas_root}/{persona_id}/consent/index.json     {consent_id: filename}
    {personas_root}/{persona_id}/consent/{consent_id}.json   single record

Each persona has its own consent tree. No cross-persona lookup —
each consent record belongs to exactly one persona, identified by
`persona_id` on the record itself.

Concurrency: file-level POSIX flock on the index file for index reads
+ writes. Per-record writes are atomic-rename (tempfile + fsync +
os.replace) to keep the on-disk record always parseable.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Protocol

from .models import ConsentRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ConsentRecordNotFound(LookupError):
    """Raised when a consent_id is not present in the persona's index."""


class CorruptConsentRecord(ValueError):
    """Raised when a stored consent.json fails to parse / validate."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
class ConsentRepository(Protocol):
    def get(self, persona_id: str, consent_id: str) -> ConsentRecord: ...
    def get_or_none(self, persona_id: str, consent_id: str) -> Optional[ConsentRecord]: ...
    def list(self, persona_id: str) -> list[ConsentRecord]: ...
    def save(self, record: ConsentRecord) -> None: ...
    def delete(self, persona_id: str, consent_id: str) -> None: ...


# ---------------------------------------------------------------------------
# JSON-backed implementation
# ---------------------------------------------------------------------------
class JsonConsentRepository:
    INDEX_FILENAME = "index.json"

    def __init__(self, personas_root: Path) -> None:
        """
        Args:
            personas_root: data_root/personas. The repo handles all
                personas, dispatching by persona_id like the corpus
                repo does.
        """
        self.personas_root = Path(personas_root)
        self.personas_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def consent_root(self, persona_id: str) -> Path:
        return self.personas_root / persona_id / "consent"

    def _index_path(self, persona_id: str) -> Path:
        return self.consent_root(persona_id) / self.INDEX_FILENAME

    def _record_path(self, persona_id: str, consent_id: str) -> Path:
        return self.consent_root(persona_id) / f"{consent_id}.json"

    # ------------------------------------------------------------------
    # Index lock
    # ------------------------------------------------------------------
    @contextmanager
    def _index_locked(
        self, persona_id: str, exclusive: bool = False
    ) -> Iterator[dict]:
        """Context-managed read or write of the per-persona index.

        Yields a dict mapping {consent_id: filename}. If exclusive=True,
        writes the (possibly-mutated) dict back atomically when the
        block exits cleanly.
        """
        self.consent_root(persona_id).mkdir(parents=True, exist_ok=True)
        idx_path = self._index_path(persona_id)
        # Lock against a sentinel so the index file itself can be
        # rewritten atomically without losing the lock.
        lock_path = idx_path.with_suffix(".json.lock")
        # Touch the lock file (create if absent).
        lock_path.touch(exist_ok=True)

        with open(lock_path, "r+") as lock_fh:
            fcntl.flock(
                lock_fh.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            try:
                if idx_path.exists():
                    with open(idx_path, "r", encoding="utf-8") as f:
                        index = json.load(f)
                else:
                    index = {}
                yield index
                if exclusive:
                    self._atomic_write_json(idx_path, index)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write_json(target: Path, data: dict) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
            suffix=".tmp",
        ) as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2, default=str)
            tf.flush()
            os.fsync(tf.fileno())
            tmp_path = Path(tf.name)
        os.replace(tmp_path, target)

    # ------------------------------------------------------------------
    # Record I/O
    # ------------------------------------------------------------------
    def get(self, persona_id: str, consent_id: str) -> ConsentRecord:
        rec = self.get_or_none(persona_id, consent_id)
        if rec is None:
            raise ConsentRecordNotFound(consent_id)
        return rec

    def get_or_none(
        self, persona_id: str, consent_id: str
    ) -> Optional[ConsentRecord]:
        path = self._record_path(persona_id, consent_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptConsentRecord(
                f"{path}: invalid JSON ({e})"
            ) from e
        try:
            return ConsentRecord.model_validate(data)
        except Exception as e:  # pydantic ValidationError
            raise CorruptConsentRecord(
                f"{path}: schema violation ({e})"
            ) from e

    def list(self, persona_id: str) -> list[ConsentRecord]:
        out: list[ConsentRecord] = []
        with self._index_locked(persona_id, exclusive=False) as index:
            for consent_id in list(index.keys()):
                rec = self.get_or_none(persona_id, consent_id)
                if rec is not None:
                    out.append(rec)
                else:
                    # Index entry but no record on disk = inconsistent
                    # state (mid-write crash, manual rm, etc.). Surface
                    # loudly per `feedback_fail_loud` memory + Gemini
                    # review f28120b §latent-3.
                    log.warning(
                        "Inconsistent consent state: persona=%s consent_id=%s "
                        "in index but record file missing",
                        persona_id,
                        consent_id,
                    )
        # Stable order — newest first by created_at.
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out

    def save(self, record: ConsentRecord) -> None:
        """Insert or update. Index is updated in the same lock."""
        record_path = self._record_path(record.persona_id, record.consent_id)
        # Write the record file FIRST (atomic), then update the index
        # under the lock. If the index update fails, we have an orphan
        # file but no inconsistent state.
        self._atomic_write_json(
            record_path,
            json.loads(record.model_dump_json()),
        )
        with self._index_locked(record.persona_id, exclusive=True) as index:
            index[record.consent_id] = record_path.name
        log.info(
            "Saved consent_id=%s persona_id=%s status=%s",
            record.consent_id,
            record.persona_id,
            record.status,
        )

    def delete(self, persona_id: str, consent_id: str) -> None:
        """Hard delete from disk + index. Use sparingly; the normal
        flow is revoke (which preserves the tombstone). Hard delete is
        only for tests + admin cleanup."""
        with self._index_locked(persona_id, exclusive=True) as index:
            if consent_id not in index:
                raise ConsentRecordNotFound(consent_id)
            del index[consent_id]
        path = self._record_path(persona_id, consent_id)
        if path.exists():
            path.unlink()
        log.info("Hard-deleted consent_id=%s persona_id=%s", consent_id, persona_id)
