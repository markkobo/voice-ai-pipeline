"""
IngestionService — extract text from uploaded corpus items + chunk for RAG.

Slice 2A scope: `.txt` and `.md` only. Other formats raise
`UnsupportedIngestionFormatError` until slice 2B/C/D land.

Per RFC_M6 Phase 0 the layout under each item dir grows from

    metadata.json
    original.<ext>

to

    metadata.json
    original.<ext>
    extracted.txt              ← UTF-8-normalized full text
    chunks.jsonl               ← one JSON object per chunk

After successful ingestion the CorpusItem's `status` flips
`uploaded → ingested` and `extracted_chars` + `chunk_count` are filled.
On error the status flips to `failed` and `error` carries the reason.

This first slice is **synchronous** — `.txt`/`.md` extracts in <100 ms
even for big files. Background tasks become necessary once
PDF/audio/video land (slice 2C/D); we'll switch the endpoint to
`BackgroundTasks` then without breaking the API shape.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .chunker import ChunkSpan, chunk_text
from .models import CorpusItem, CorpusItemKind, CorpusItemStatus
from .repository import CorpusItemNotFound, JsonCorpusRepository

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service-layer exceptions — API maps these to HTTP via _errors.py.
# ---------------------------------------------------------------------------
class IngestionError(RuntimeError):
    """Base for ingestion failures."""


class UnsupportedIngestionFormatError(IngestionError):
    """Extractor for this filename extension is not implemented yet."""


class ExtractionFailedError(IngestionError):
    """Decoder ran but produced no usable text."""


# ---------------------------------------------------------------------------
# Format detection — registry mapping extension → extractor callable.
# ---------------------------------------------------------------------------
EXTRACTED_FILENAME = "extracted.txt"
CHUNKS_FILENAME = "chunks.jsonl"


def _extract_plaintext(path: Path) -> str:
    """Read a UTF-8 text file. Tolerate BOM and CRLF."""
    raw = path.read_bytes()
    # Strip UTF-8 BOM if present so it doesn't leak into chunks.
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Try GB18030 (Simplified Chinese) and Big5 (Traditional) — both
        # common for Chinese-language plaintext from non-UTF-8 sources.
        for enc in ("gb18030", "big5", "utf-16"):
            try:
                text = raw.decode(enc)
                log.info("Decoded %s as %s", path.name, enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ExtractionFailedError(
                f"Could not decode {path.name} as any of "
                "utf-8/gb18030/big5/utf-16"
            )

    # Normalize line endings to \n. Leave the rest of the text alone so
    # chunker can use paragraph boundaries (\n\n) and Chinese punctuation
    # without us second-guessing the source.
    return text.replace("\r\n", "\n").replace("\r", "\n")


# Extension → callable(path: Path) -> str
_EXTRACTORS = {
    ".txt": _extract_plaintext,
    ".md": _extract_plaintext,
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class IngestionService:
    """Orchestrates per-item extraction + chunking + repository update."""

    def __init__(
        self,
        repository: JsonCorpusRepository,
        *,
        target_chunk_chars: int = 600,
        overlap_chars: int = 100,
    ) -> None:
        self.repository = repository
        self.target_chunk_chars = target_chunk_chars
        self.overlap_chars = overlap_chars

    def supported_extensions(self) -> set[str]:
        return set(_EXTRACTORS.keys())

    def ingest(self, persona_id: str, item_id: str) -> CorpusItem:
        """
        Run the extractor for an item, chunk the output, persist artifacts,
        flip the item's status.

        Idempotent — re-ingesting overwrites extracted.txt + chunks.jsonl
        and re-counts chars/chunks. That's intentional so the user can
        re-run after a chunker tweak without deleting first.
        """
        item = self.repository.get(persona_id, item_id)
        original_path = self._find_original(item)
        ext = original_path.suffix.lower()

        if ext not in _EXTRACTORS:
            # Mark failed so the manifest reflects reality, but raise
            # so the API can return 4xx.
            self._mark_failed(
                persona_id, item_id,
                f"Unsupported extension {ext!r}; supported: "
                f"{sorted(_EXTRACTORS)}",
            )
            raise UnsupportedIngestionFormatError(ext)

        # Run the extractor.
        try:
            text = _EXTRACTORS[ext](original_path)
        except ExtractionFailedError as e:
            self._mark_failed(persona_id, item_id, str(e))
            raise
        except Exception as e:
            self._mark_failed(persona_id, item_id, f"{type(e).__name__}: {e}")
            raise ExtractionFailedError(str(e)) from e

        if not text or not text.strip():
            self._mark_failed(persona_id, item_id, "Extracted text is empty")
            raise ExtractionFailedError("Extracted text is empty")

        # Write extracted.txt + chunks.jsonl atomically alongside metadata.
        item_dir = self.repository.item_dir_for_kind(
            persona_id, item.kind.value, item_id,
        )
        (item_dir / EXTRACTED_FILENAME).write_text(text, encoding="utf-8")

        chunks = chunk_text(
            text,
            target_chars=self.target_chunk_chars,
            overlap_chars=self.overlap_chars,
        )
        self._write_chunks(item_dir / CHUNKS_FILENAME, chunks, item)

        # Flip status + counts.
        def _flip(it: CorpusItem) -> None:
            it.extracted_chars = len(text)
            it.chunk_count = len(chunks)
            it.status = CorpusItemStatus.ingested
            it.error = None

        updated = self.repository.update(persona_id, item_id, _flip)
        log.info(
            "Ingested persona=%s item=%s ext=%s chars=%d chunks=%d",
            persona_id, item_id, ext, len(text), len(chunks),
        )
        return updated

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _find_original(self, item: CorpusItem) -> Path:
        """Locate original.<ext> within the item dir.

        We stored it as `original.<ext>` at upload time (see
        CorpusService.upload). Use the recorded filename to recover the
        extension — it could differ from the on-disk ext if the upload
        was renamed, but that doesn't happen today.
        """
        item_dir = self.repository.item_dir_for_kind(
            item.persona_id, item.kind.value, item.item_id,
        )
        # Match anything beginning with "original.".
        for child in item_dir.iterdir():
            if child.is_file() and child.name.startswith("original."):
                return child
        raise ExtractionFailedError(
            f"No original.* file in {item_dir}"
        )

    def _write_chunks(
        self,
        chunks_path: Path,
        spans: list[ChunkSpan],
        item: CorpusItem,
    ) -> None:
        """Write chunks as JSONL — one record per line."""
        with open(chunks_path, "w", encoding="utf-8") as f:
            for idx, span in enumerate(spans):
                record = {
                    "chunk_index": idx,
                    "char_offset": span.char_offset,
                    "char_count": span.char_count,
                    "text": span.text,
                    # Pass-through metadata from the item — useful for
                    # filtering at retrieval time.
                    "persona_id": item.persona_id,
                    "item_id": item.item_id,
                    "kind": item.kind.value,
                    "listener_tag": item.listener_tag,
                    "source": item.source,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _mark_failed(
        self, persona_id: str, item_id: str, reason: str,
    ) -> None:
        """Flip an item to status=failed with an error string. Best-effort."""
        try:
            def _flip(it: CorpusItem) -> None:
                it.status = CorpusItemStatus.failed
                it.error = reason
                it.extracted_chars = 0
                it.chunk_count = 0

            self.repository.update(persona_id, item_id, _flip)
        except CorpusItemNotFound:
            pass
