"""
Extractor protocol — per-format text extraction for the corpus pipeline.

Each extractor implements `supports(kind, ext)` + `extract(path)` and
publishes flags (`is_async`, `needs_gpu`) so the orchestration layer
knows whether to run inline or hand off to a background task.

This is RFC_M6 Phase 0 slice 2 task 62A — the registry refactor that
unblocks slice 2C (PDF/EPUB/DOCX) and slice 2D (audio/video). The prior
`(CorpusItemKind, ext) → callable` table didn't carry per-extractor
metadata and would have ballooned awkwardly as more formats landed.

Concrete extractors today:
- `PlaintextExtractor` — `.txt`/`.md` under `kind=text` or
  `kind=transcript`. Encoding-cascade with Big5-strict-before-gb18030.
- `ConversationExtractor` — `.txt`/`.csv`/`.json` under
  `kind=conversation`. WhatsApp / Line / WeChat sniff-and-dispatch.

Both share the `_extract_plaintext` and `_normalize_newlines` helpers
that already live in `ingestion.py` (kept there for now to avoid an
import-cycle; will collapse once the legacy `_EXTRACTORS` table is
removed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import CorpusItemKind


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractResult:
    """What an extractor returns.

    text: canonical extracted text, ready for the chunker.
    warnings: non-fatal issues (e.g. "decoded as gb18030, may be wrong").
    metadata: extractor-specific structured info (e.g.
        {"format": "whatsapp", "message_count": 142}). Chunk records can
        eventually consume this for per-chunk provenance (slice 62B work).
    """
    text: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Extractor(Protocol):
    """One per-format text extractor.

    The two flag attributes drive orchestration decisions:
    - is_async: extractor runs awaitable IO (network ASR, ffmpeg subprocess).
      Today's text extractors are False. Audio/video will be True (slice 2D).
    - needs_gpu: extractor competes with the live serving GPU (e.g. ASR).
      The dispatcher must serialize these against the training GPU lock.
    """
    is_async: bool
    needs_gpu: bool

    def supports(self, kind: CorpusItemKind, ext: str) -> bool: ...

    def extract(self, path: Path) -> ExtractResult: ...


# ---------------------------------------------------------------------------
# Concrete extractors — thin wrappers over ingestion.py helpers.
#
# These stay short on purpose. The actual decode/parse logic lives in
# `_extract_plaintext` / `_extract_conversation` / `chat_parsers.py` —
# moving it here would create a cycle with `ingestion.py`. Once the
# legacy `_EXTRACTORS` table is removed from ingestion.py the helpers
# can migrate here.
# ---------------------------------------------------------------------------
class PlaintextExtractor:
    """`.txt` and `.md` for the `text` and `transcript` kinds."""
    is_async = False
    needs_gpu = False

    _SUPPORTED = frozenset({
        (CorpusItemKind.text, ".txt"),
        (CorpusItemKind.text, ".md"),
        (CorpusItemKind.transcript, ".txt"),
        (CorpusItemKind.transcript, ".md"),
    })

    def supports(self, kind: CorpusItemKind, ext: str) -> bool:
        return (kind, ext.lower()) in self._SUPPORTED

    def extract(self, path: Path) -> ExtractResult:
        # Defer to the existing helper so the encoding cascade (review
        # #1 fix) stays in one place.
        from .ingestion import _extract_plaintext
        text = _extract_plaintext(path)
        return ExtractResult(text=text, metadata={"format": "plaintext"})


class ConversationExtractor:
    """`.txt` / `.csv` / `.json` for the `conversation` kind.

    Routes to the right chat parser via `detect_chat_format`. Falls back
    to plaintext for freeform chat dumps. JSON is reserved for a
    platform-specific parser (not implemented in slice 2B).
    """
    is_async = False
    needs_gpu = False

    _SUPPORTED = frozenset({
        (CorpusItemKind.conversation, ".txt"),
        (CorpusItemKind.conversation, ".csv"),
        (CorpusItemKind.conversation, ".json"),
    })

    def supports(self, kind: CorpusItemKind, ext: str) -> bool:
        return (kind, ext.lower()) in self._SUPPORTED

    def extract(self, path: Path) -> ExtractResult:
        from .ingestion import _extract_conversation_with_metadata
        text, metadata = _extract_conversation_with_metadata(path)
        return ExtractResult(text=text, metadata=metadata)


# ---------------------------------------------------------------------------
# Default registry — IngestionService asks `default_extractors()` at
# construction. Tests + future extractors (PDF, audio) can compose their
# own list.
# ---------------------------------------------------------------------------
def default_extractors() -> list[Extractor]:
    return [
        PlaintextExtractor(),
        ConversationExtractor(),
    ]
