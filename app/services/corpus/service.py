"""
CorpusService — upload / list / get / delete per-persona corpus items.

This first slice only persists raw uploaded bytes + metadata. The
heavy-lift ingestion (PDF→text, EPUB→text, audio→ASR→speaker-filter,
chunking + embedding) lives in a follow-up slice, intentionally — pinning
the storage + API surface first lets the ingestion work iterate without
breaking the UI / RFC contract.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import (
    ALLOWED_EXTENSIONS_BY_KIND,
    CorpusItem,
    CorpusItemKind,
    CorpusItemStatus,
    CorpusManifest,
    CorpusManifestThresholds,
    LORA_ORGANIC_MIN_CHARS,
    LORA_SYNTHETIC_MIN_CHARS,
    MAX_UPLOAD_BYTES,
    RAG_MIN_CHUNKS,
)
from .repository import (
    CorpusItemNotFound,
    JsonCorpusRepository,
)


# Persona/item id format — same as personas service. Locks down path
# segments so `persona_id="../other"` / `"/"` / `""` / `"foo\x00bar"`
# can't traverse the corpus tree. Server-generated UUIDs match this
# pattern by construction; we still validate item_id on lookup paths
# so client-supplied IDs in GET/DELETE/INGEST can't escape.
_PERSONA_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ITEM_ID_PATTERN = re.compile(r"^[a-f0-9-]{8,64}$")  # uuid4 form

# Filename sanitization (review #16). Strip control chars + path
# separators + NUL bytes before persisting. We keep the raw filename
# field but make it safe for log lines and JSON consumers.
_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f/\\]")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service-layer exceptions — API maps these to HTTP via _errors.py.
# ---------------------------------------------------------------------------
class CorpusUploadError(ValueError):
    """Generic 4xx-class error during upload."""


class UnsupportedCorpusFormatError(CorpusUploadError):
    """Filename extension not allowed for this kind."""


class CorpusTooLargeError(CorpusUploadError):
    """Upload exceeds MAX_UPLOAD_BYTES."""


class CorpusEmptyError(CorpusUploadError):
    """Upload is zero bytes."""


class InvalidCorpusIdError(CorpusUploadError):
    """persona_id or item_id failed the regex check — likely path-traversal
    attempt or empty string. Maps to 400."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class CorpusService:
    ORIGINAL_PREFIX = "original"   # stored as original.<ext>

    def __init__(
        self,
        repository: JsonCorpusRepository,
        *,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        self.repository = repository
        self.max_upload_bytes = max_upload_bytes

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload(
        self,
        *,
        persona_id: str,
        kind: CorpusItemKind,
        file_bytes: bytes,
        filename: str,
        mime_type: Optional[str] = None,
        source: Optional[str] = None,
        source_date: Optional[datetime] = None,
        listener_tag: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> CorpusItem:
        """Persist a new corpus item from raw upload bytes.

        Validates persona_id format (security — review #11), extension,
        size, non-empty. Returns the new CorpusItem in `uploaded` state —
        ingestion is a separate later step.

        Orphan-cleanup on save-failure (review #13): if metadata write
        crashes after the original bytes have been written, we rmtree the
        item dir so disk doesn't drift.
        """
        _validate_persona_id(persona_id)

        if not file_bytes:
            raise CorpusEmptyError("Empty upload")

        if len(file_bytes) > self.max_upload_bytes:
            raise CorpusTooLargeError(
                f"File too large: {len(file_bytes)} bytes "
                f"(max {self.max_upload_bytes})"
            )

        ext = self._extension(filename)
        allowed = ALLOWED_EXTENSIONS_BY_KIND[kind]
        if ext not in allowed:
            raise UnsupportedCorpusFormatError(
                f"Unsupported extension {ext!r} for kind={kind.value}. "
                f"Allowed: {sorted(allowed)}"
            )

        item_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        sanitized_filename = _UNSAFE_FILENAME_CHARS.sub("_", filename)
        item = CorpusItem(
            item_id=item_id,
            persona_id=persona_id,
            kind=kind,
            filename=sanitized_filename,
            mime_type=mime_type,
            size_bytes=len(file_bytes),
            status=CorpusItemStatus.uploaded,
            source=source,
            source_date=source_date,
            listener_tag=listener_tag,
            notes=notes,
            created_at=now,
            updated_at=now,
        )

        # Write the original bytes BEFORE saving metadata so a crash
        # mid-upload doesn't leave a dangling index entry. If metadata
        # save then fails (disk full, JSON encode error), rmtree the item
        # dir so the original bytes aren't orphaned on disk (review #13).
        item_dir = self.repository.item_dir_for_kind(persona_id, kind.value, item_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        original_path = item_dir / f"{self.ORIGINAL_PREFIX}{ext}"
        original_path.write_bytes(file_bytes)

        try:
            self.repository.save(item)
        except Exception:
            import shutil
            shutil.rmtree(item_dir, ignore_errors=True)
            raise

        log.info(
            "Corpus upload: persona=%s kind=%s item=%s filename=%s bytes=%d",
            persona_id, kind.value, item_id, sanitized_filename, len(file_bytes),
        )
        return item

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def list(self, persona_id: str) -> list[CorpusItem]:
        _validate_persona_id(persona_id)
        return self.repository.list(persona_id)

    def get(self, persona_id: str, item_id: str) -> CorpusItem:
        _validate_persona_id(persona_id)
        _validate_item_id(item_id)
        return self.repository.get(persona_id, item_id)

    def delete(self, persona_id: str, item_id: str) -> None:
        _validate_persona_id(persona_id)
        _validate_item_id(item_id)
        self.repository.delete(persona_id, item_id)

    # ------------------------------------------------------------------
    # Manifest — rolled-up view for the UI and downstream consumers.
    # ------------------------------------------------------------------
    def compute_manifest(self, persona_id: str) -> CorpusManifest:
        _validate_persona_id(persona_id)
        items = self.repository.list(persona_id)

        by_kind: dict[str, int] = {k.value: 0 for k in CorpusItemKind}
        total_bytes = 0
        total_chars = 0
        total_chunks = 0

        for item in items:
            by_kind[item.kind.value] = by_kind.get(item.kind.value, 0) + 1
            total_bytes += item.size_bytes
            total_chars += item.extracted_chars or 0
            total_chunks += item.chunk_count or 0

        thresholds = CorpusManifestThresholds(
            ready_for_rag=total_chunks >= RAG_MIN_CHUNKS,
            ready_for_lora_synthetic=total_chars >= LORA_SYNTHETIC_MIN_CHARS,
            ready_for_lora_organic=total_chars >= LORA_ORGANIC_MIN_CHARS,
        )

        return CorpusManifest(
            persona_id=persona_id,
            total_items=len(items),
            by_kind=by_kind,
            total_bytes=total_bytes,
            extracted_chars=total_chars,
            thresholds=thresholds,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _extension(filename: str) -> str:
        _, ext = os.path.splitext(filename)
        return ext.lower()


# ---------------------------------------------------------------------------
# ID validators — module-level so repository helpers and tests can call them
# (review #11 — cross-persona path traversal). Cheap, runs on every
# read/write path that takes a persona_id from the API.
# ---------------------------------------------------------------------------
def _validate_persona_id(persona_id: str) -> None:
    if not persona_id or not _PERSONA_ID_PATTERN.match(persona_id):
        raise InvalidCorpusIdError(
            f"Invalid persona_id {persona_id!r}: must match "
            f"{_PERSONA_ID_PATTERN.pattern!r}"
        )


def _validate_item_id(item_id: str) -> None:
    if not item_id or not _ITEM_ID_PATTERN.match(item_id):
        raise InvalidCorpusIdError(
            f"Invalid item_id {item_id!r}: must match "
            f"{_ITEM_ID_PATTERN.pattern!r}"
        )
