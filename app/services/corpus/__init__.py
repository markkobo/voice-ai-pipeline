"""
Corpus service — per-persona text/transcript/conversation ingestion.

Implements RFC_M6 §3 Phase 0: storage layout + repository + service.
Heavy ingestion (PDF/EPUB/audio→text extraction, chunking, embedding) is
deferred to a follow-up slice. This module owns "raw bytes on disk" today.
"""
from .chat_parsers import (
    ChatMessage,
    detect_chat_format,
    messages_to_text,
    parse_line,
    parse_whatsapp,
    parse_wechat_csv,
)
from .chunker import ChunkSpan, chunk_text
from .extractors import (
    ConversationExtractor,
    Extractor,
    ExtractResult,
    PlaintextExtractor,
    default_extractors,
)
from .ingestion import (
    ExtractionFailedError,
    IngestionError,
    IngestionService,
    UnsupportedIngestionFormatError,
)
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
    "ChatMessage",
    "ChunkSpan",
    "chunk_text",
    "ConversationExtractor",
    "CorpusItem",
    "CorpusItemKind",
    "CorpusItemStatus",
    "CorpusManifest",
    "CorpusService",
    "default_extractors",
    "detect_chat_format",
    "Extractor",
    "ExtractionFailedError",
    "ExtractResult",
    "IngestionError",
    "IngestionService",
    "JsonCorpusRepository",
    "CorpusItemNotFound",
    "CorruptCorpusMetadata",
    "messages_to_text",
    "parse_line",
    "parse_whatsapp",
    "parse_wechat_csv",
    "PlaintextExtractor",
    "UnsupportedIngestionFormatError",
]
