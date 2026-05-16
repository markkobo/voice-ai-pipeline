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

        Validates extension/size/non-empty. Returns the new CorpusItem in
        `uploaded` state — ingestion is a separate later step.
        """
        if not persona_id:
            raise CorpusUploadError("persona_id is required")

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
        item = CorpusItem(
            item_id=item_id,
            persona_id=persona_id,
            kind=kind,
            filename=filename,
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
        # mid-upload doesn't leave a dangling index entry.
        item_dir = self.repository.item_dir_for_kind(persona_id, kind.value, item_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        original_path = item_dir / f"{self.ORIGINAL_PREFIX}{ext}"
        original_path.write_bytes(file_bytes)

        self.repository.save(item)
        log.info(
            "Corpus upload: persona=%s kind=%s item=%s filename=%s bytes=%d",
            persona_id, kind.value, item_id, filename, len(file_bytes),
        )
        return item

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def list(self, persona_id: str) -> list[CorpusItem]:
        return self.repository.list(persona_id)

    def get(self, persona_id: str, item_id: str) -> CorpusItem:
        return self.repository.get(persona_id, item_id)

    def delete(self, persona_id: str, item_id: str) -> None:
        self.repository.delete(persona_id, item_id)

    # ------------------------------------------------------------------
    # Manifest — rolled-up view for the UI and downstream consumers.
    # ------------------------------------------------------------------
    def compute_manifest(self, persona_id: str) -> CorpusManifest:
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
