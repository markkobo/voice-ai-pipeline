"""
Corpus service — per-persona text/transcript/conversation ingestion.

Implements RFC_M6 §3 Phase 0: storage layout + repository + service.
Heavy ingestion (PDF/EPUB/audio→text extraction, chunking, embedding) is
deferred to a follow-up slice. This module owns "raw bytes on disk" today.
"""
from .models import (
    CorpusItem,
    CorpusItemKind,
    CorpusItemStatus,
    CorpusManifest,
)
from .repository import (
    JsonCorpusRepository,
    CorpusItemNotFound,
    CorruptCorpusMetadata,
)
from .service import CorpusService

__all__ = [
    "CorpusItem",
    "CorpusItemKind",
    "CorpusItemStatus",
    "CorpusManifest",
    "CorpusService",
    "JsonCorpusRepository",
    "CorpusItemNotFound",
    "CorruptCorpusMetadata",
]
