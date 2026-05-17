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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .chat_parsers import (
    detect_chat_format,
    messages_to_text,
    parse_line,
    parse_whatsapp,
    parse_wechat_csv,
)
from .chunker import CHUNKER_VERSION, ChunkSpan, chunk_text
from .models import CorpusItem, CorpusItemKind, CorpusItemStatus
from .repository import CorpusItemNotFound, JsonCorpusRepository

# Lazy import: Extractor protocol + default registry. extractors.py
# imports back from this module via TYPE_CHECKING — keep concrete
# imports inside functions to avoid cycles at module-load time.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .extractors import Extractor, ExtractResult

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


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via tempfile + os.replace (review #6).

    Same pattern as JsonCorpusRepository._atomic_write_json. Same-directory
    tempfile ensures os.replace is atomic on POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _extract_plaintext(path: Path) -> str:
    """Read a text file with codec auto-detection.

    Strategy (review #1 — Big5 silent-mis-decode disaster):

    1. UTF-8 BOM → utf-8-sig
    2. UTF-16 BE/LE BOM → utf-16-{be,le} (review #10)
    3. utf-8 strict
    4. **big5 before gb18030** — gb18030 is permissive enough to accept
       Big5 byte sequences and decode them to mojibake without raising.
       Try the stricter codec first.
    5. gb18030
    6. utf-16 (without BOM, both endians) with sanity check

    Each non-UTF-8 candidate is sanity-checked: if the decoded text has
    too many U+FFFD replacement chars OR is dominated by non-CJK garbage
    in private-use blocks (the Big5-as-gb18030 mojibake signature), it
    fails over to the next codec.
    """
    raw = path.read_bytes()

    # BOM-based fast paths.
    if raw.startswith(b"\xef\xbb\xbf"):
        return _normalize_newlines(raw[3:].decode("utf-8"))
    if raw.startswith(b"\xff\xfe"):
        return _normalize_newlines(raw[2:].decode("utf-16-le"))
    if raw.startswith(b"\xfe\xff"):
        return _normalize_newlines(raw[2:].decode("utf-16-be"))

    # Strict UTF-8 first — covers the dominant case.
    try:
        return _normalize_newlines(raw.decode("utf-8"))
    except UnicodeDecodeError:
        pass

    # Candidate codecs in priority order. big5 strict comes BEFORE
    # gb18030 — gb18030's permissive coverage causes silent mojibake on
    # Big5 input (review #1). For each candidate that decodes without
    # raising, sanity-check the output and accept only if it looks like
    # real text.
    for enc in ("big5", "gb18030", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if _looks_like_real_text(text):
            log.info("Decoded %s as %s", path.name, enc)
            return _normalize_newlines(text)
        else:
            log.debug(
                "Decode %s as %s produced mojibake-shaped output, "
                "trying next codec", path.name, enc,
            )

    raise ExtractionFailedError(
        f"Could not decode {path.name} as any of "
        "utf-8/big5/gb18030/utf-16; file may be binary or corrupt"
    )


def _normalize_newlines(text: str) -> str:
    """Normalize CR/LF/CRLF + line/paragraph separators to \n.

      LINE SEPARATOR and   PARAGRAPH SEPARATOR are produced
    by some Word→.txt and Mac exports; the chunker's paragraph regex
    won't see them otherwise (review #13).
    """
    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace(" ", "\n")
        .replace(" ", "\n\n")
        .replace("\x00", "")  # strip embedded NULs (review #12)
    )


def _looks_like_real_text(text: str) -> bool:
    """Cheap sanity check on a decoded string.

    Designed to catch the Big5-as-gb18030 mojibake case (review #1).
    Returns False when:
      - more than 0.5% of chars are U+FFFD replacement chars
      - more than 30% of non-ASCII chars fall in CJK Compatibility
        Ideographs / Private Use Areas (the gb18030 mojibake signature)
      - empty after stripping
    """
    if not text or not text.strip():
        return False
    sample = text[:8000]  # cheap — only look at the head
    n = len(sample)
    if n == 0:
        return True

    replacement_chars = sample.count("�")
    if replacement_chars > n * 0.005:
        return False

    # Count chars in suspicious Unicode blocks. Big5-decoded-as-gb18030
    # tends to land disproportionately in Private Use Area (U+E000..F8FF)
    # and CJK Compatibility (U+F900..FAFF).
    suspicious = sum(
        1 for c in sample if 0xE000 <= ord(c) <= 0xFAFF
    )
    non_ascii = sum(1 for c in sample if ord(c) > 0x7F)
    if non_ascii > 50 and suspicious > non_ascii * 0.30:
        return False

    return True


def _extract_conversation_with_metadata(path: Path) -> tuple[str, dict]:
    """Conversation extractor — sniffs the chat-export format and routes.

    Returns `(canonical_text, metadata)` where metadata reports which
    parser ran and how many messages were produced. Used by the
    ConversationExtractor (extractors.py) to fill `ExtractResult.metadata`.

    Supported sources (slice 2B):
      - WhatsApp .txt
      - Line .txt
      - WeChat CSV (any of the common third-party-tool schemas)

    Falls back to plaintext for ambiguous .txt: still useful because a
    free-form chat dump is just a transcript.
    """
    ext = path.suffix.lower()

    if ext == ".csv":
        # CSV path is unambiguously WeChat for now.
        raw = _extract_plaintext(path)
        msgs = parse_wechat_csv(raw)
        if not msgs:
            raise ExtractionFailedError(
                "WeChat CSV parser produced no messages — check column headers"
            )
        return messages_to_text(msgs), {
            "format": "wechat_csv",
            "message_count": len(msgs),
        }

    if ext == ".txt":
        raw = _extract_plaintext(path)
        fmt = detect_chat_format(raw)
        if fmt == "whatsapp":
            msgs = parse_whatsapp(raw)
        elif fmt == "line":
            msgs = parse_line(raw)
        elif fmt == "wechat":
            # Detected wechat-shaped header in a .txt — try the CSV
            # parser. Do NOT fall back to WhatsApp (review #9 of 3f2f55e):
            # if the CSV parser fails the file is malformed wechat, not
            # a WhatsApp export — silently retrying with the wrong
            # parser produces garbage.
            msgs = parse_wechat_csv(raw)
        else:
            # Unknown shape — treat as freeform conversation transcript.
            log.info(
                "Chat-export format not detected for %s; using plaintext fallback",
                path.name,
            )
            return raw, {"format": "freeform_text"}
        if not msgs:
            raise ExtractionFailedError(
                f"Chat parser ({fmt}) ran but produced no messages"
            )
        return messages_to_text(msgs), {
            "format": fmt,
            "message_count": len(msgs),
        }

    if ext == ".json":
        # JSON shapes are too varied to auto-parse — leave for a later
        # platform-specific extractor. Reject explicitly so the failure
        # message is actionable.
        raise ExtractionFailedError(
            ".json chat exports need a platform-specific parser; not yet implemented"
        )

    raise ExtractionFailedError(f"No conversation extractor for {ext}")


# Back-compat helper for any direct callers still using the old signature.
def _extract_conversation(path: Path) -> str:
    text, _meta = _extract_conversation_with_metadata(path)
    return text


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
        extractors: "list[Extractor] | None" = None,
    ) -> None:
        self.repository = repository
        self.target_chunk_chars = target_chunk_chars
        self.overlap_chars = overlap_chars
        # Lazy default — keeps the import chain clean for tests that
        # construct IngestionService directly.
        if extractors is None:
            from .extractors import default_extractors
            extractors = default_extractors()
        self._extractors = extractors

    def supported_extensions(self) -> set[str]:
        """Flat set of supported extensions across all kinds.

        Used by the API error envelope as a hint to callers. Kind-aware
        dispatch is handled inside `ingest()`.
        """
        # Probe every extractor across every kind / common-ext combo.
        # Kept brute-force-but-cheap since the lists are O(10).
        common_exts = {
            ".txt", ".md", ".csv", ".json", ".pdf", ".epub", ".docx",
            ".mp3", ".wav", ".m4a", ".mp4",
        }
        out: set[str] = set()
        for kind in CorpusItemKind:
            for ext in common_exts:
                if self._find_extractor(kind, ext) is not None:
                    out.add(ext)
        return out

    def supported_for_kind(self, kind: CorpusItemKind) -> set[str]:
        common_exts = {
            ".txt", ".md", ".csv", ".json", ".pdf", ".epub", ".docx",
            ".mp3", ".wav", ".m4a", ".mp4",
        }
        return {
            ext for ext in common_exts
            if self._find_extractor(kind, ext) is not None
        }

    def _find_extractor(
        self, kind: CorpusItemKind, ext: str,
    ) -> "Extractor | None":
        for extractor in self._extractors:
            if extractor.supports(kind, ext):
                return extractor
        return None

    def sweep_stranded(self, persona_id: str) -> int:
        """Reset any items stuck in `ingesting` status to `failed`.

        Run at server startup so crashed/killed-mid-ingest items don't
        stay stuck forever (task 62C). Synchronous ingest today means
        an item in `ingesting` state at startup definitely means the
        process died before completing.

        Returns the number of items reset.
        """
        count = 0
        for item in self.repository.list(persona_id):
            if item.status == CorpusItemStatus.ingesting:
                def _flip(it: CorpusItem, _id=item.item_id) -> None:
                    it.status = CorpusItemStatus.failed
                    it.error = "interrupted (server restarted mid-ingest)"
                self.repository.update(persona_id, item.item_id, _flip)
                count += 1
                log.warning(
                    "Reset stranded ingesting item: persona=%s item=%s",
                    persona_id, item.item_id,
                )
        return count

    def ingest(self, persona_id: str, item_id: str) -> CorpusItem:
        """
        Run the extractor for an item, chunk the output, persist artifacts,
        flip the item's status.

        Idempotent — re-ingesting overwrites extracted.txt + chunks.jsonl
        and re-counts chars/chunks. That's intentional so the user can
        re-run after a chunker tweak without deleting first.
        """
        # Validate IDs early — repository.get will resolve paths from
        # these values; an unvalidated `persona_id="../other"` would
        # bypass the corpus tree (review #11).
        from .service import _validate_item_id, _validate_persona_id
        _validate_persona_id(persona_id)
        _validate_item_id(item_id)

        item = self.repository.get(persona_id, item_id)

        # Flip to `ingesting` so a crash mid-ingest leaves the item in
        # a recoverable state (task 62C / review #4 of 8161535). The
        # startup-sweep in IngestionService.sweep_stranded() resets
        # any items stuck in `ingesting` to `failed("interrupted")`.
        def _start(it: CorpusItem) -> None:
            it.status = CorpusItemStatus.ingesting
            it.error = None
        self.repository.update(persona_id, item_id, _start)

        original_path = self._find_original(item)
        ext = original_path.suffix.lower()

        extractor = self._find_extractor(item.kind, ext)
        if extractor is None:
            # Mark failed so the manifest reflects reality, but raise
            # so the API can return 4xx.
            self._mark_failed(
                persona_id, item_id,
                f"Unsupported (kind={item.kind.value}, ext={ext!r}); supported for kind: "
                f"{sorted(self.supported_for_kind(item.kind))}",
            )
            raise UnsupportedIngestionFormatError(f"{item.kind.value}/{ext}")

        # Run the extractor.
        try:
            result = extractor.extract(original_path)
            text = result.text
            for warning in result.warnings:
                log.warning(
                    "[%s/%s] extractor warning: %s",
                    persona_id, item_id, warning,
                )
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
        # tempfile+os.replace pair avoids torn writes when two concurrent
        # /ingest calls race on the same item (review #6).
        item_dir = self.repository.item_dir_for_kind(
            persona_id, item.kind.value, item_id,
        )
        chunks = chunk_text(
            text,
            target_chars=self.target_chunk_chars,
            overlap_chars=self.overlap_chars,
        )
        _atomic_write_text(item_dir / EXTRACTED_FILENAME, text)
        self._write_chunks_atomic(item_dir / CHUNKS_FILENAME, chunks, item)

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

        Uses the extension recorded in `item.filename` rather than
        glob-matching `original.*` (review #3 — glob could match
        `.lock` sentinel files, `.bak` files from manual recovery, or
        future siblings).
        """
        item_dir = self.repository.item_dir_for_kind(
            item.persona_id, item.kind.value, item.item_id,
        )
        _, ext = os.path.splitext(item.filename)
        expected = item_dir / f"original{ext.lower()}"
        if expected.is_file():
            return expected
        raise ExtractionFailedError(
            f"No {expected.name} file in {item_dir}"
        )

    def _write_chunks_atomic(
        self,
        chunks_path: Path,
        spans: list[ChunkSpan],
        item: CorpusItem,
    ) -> None:
        """Write chunks as JSONL via tempfile + os.replace.

        Two concurrent /ingest calls on the same item used to interleave
        writes (review #6). Now each writer constructs its own tempfile
        and atomically replaces — last writer wins on the final file but
        readers never see a partial JSONL.
        """
        lines: list[str] = []
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
                "persona_speaker_alias": item.persona_speaker_alias,
                "source": item.source,
                # Chunker-algorithm version stamp (task 62B / review #5).
                # Downstream vector indexes detect a mismatch and
                # re-embed when the chunker is re-tuned.
                "chunker_version": CHUNKER_VERSION,
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        _atomic_write_text(chunks_path, "\n".join(lines) + ("\n" if lines else ""))

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
