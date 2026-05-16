"""
Pydantic domain models for the per-persona corpus.

Storage layout (RFC_M6 §3 Phase 0):

    {data_root}/personas/<persona_id>/corpus/
    ├── index.json                            # {item_id: rel_dir}
    ├── manifest.json                         # rolled-up stats (cached)
    └── <kind>/<item_id>/
        ├── metadata.json                     # CorpusItem
        └── original.<ext>                    # raw upload bytes

Pure data classes — no IO, no business logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class CorpusItemKind(str, Enum):
    """Top-level classification of an uploaded corpus item."""
    text = "text"                  # books, articles, .txt/.md/.pdf/.epub/.docx
    transcript = "transcript"      # podcast/interview/video → ASR'd text
    conversation = "conversation"  # Line/WeChat/WhatsApp exports, chat logs


class CorpusItemStatus(str, Enum):
    """Where the item is in the ingest pipeline.

    Today only `uploaded` is reachable. The heavier states are reserved for
    the follow-up ingestion slice; defining them up front avoids a Pydantic
    migration when that slice lands.
    """
    uploaded = "uploaded"      # raw file persisted, no extraction yet
    ingesting = "ingesting"    # extraction/chunking running
    ingested = "ingested"      # text extracted + chunked, ready for RAG
    failed = "failed"


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------
class CorpusItem(BaseModel):
    """One uploaded artifact in the per-persona corpus."""
    model_config = ConfigDict(extra="forbid")

    item_id: str
    persona_id: str
    kind: CorpusItemKind
    filename: str                         # original upload filename
    mime_type: Optional[str] = None
    size_bytes: int
    status: CorpusItemStatus = CorpusItemStatus.uploaded

    # Provenance — optional but encouraged. `source` is a free-text label
    # ("私房書 ch.3", "podcast 2024-11-12", "Line export 2024-Q4"). Source
    # date helps date-filter RAG to avoid stale 2005 opinions about parenting.
    source: Optional[str] = None
    source_date: Optional[datetime] = None

    # If a conversation is with a specific listener (e.g. user's child),
    # tag it so per-listener retrieval can filter accordingly.
    listener_tag: Optional[str] = None

    notes: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Ingestion outputs — populated by the follow-up slice. Kept here so
    # the schema is stable across slices.
    extracted_chars: Optional[int] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None


class CorpusManifestThresholds(BaseModel):
    """Computed flags signalling whether the corpus is ready for downstream
    consumers. Thresholds come from RFC_M6 §3 Phase 0 / §6.
    """
    model_config = ConfigDict(extra="forbid")

    # RAG threshold — once we have at least 50 ingested chunks the dual-
    # index retriever has enough to be useful.
    ready_for_rag: bool = False

    # Persona-LoRA SFT thresholds — synthetic expansion floor and organic
    # floor. Both keyed on extracted-character count as a rough proxy for
    # "turns" (Chinese ~30 chars/turn average).
    ready_for_lora_synthetic: bool = False
    ready_for_lora_organic: bool = False


class CorpusManifest(BaseModel):
    """Rolled-up view of a persona's corpus."""
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    total_items: int
    by_kind: dict[str, int]    # CorpusItemKind value → count
    total_bytes: int
    extracted_chars: int       # sum across ingested items (0 today)
    thresholds: CorpusManifestThresholds
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Threshold constants — referenced by service.compute_manifest()
# ---------------------------------------------------------------------------
RAG_MIN_CHUNKS = 50
LORA_SYNTHETIC_MIN_CHARS = 15_000     # ~500 turns × 30 chars/turn
LORA_ORGANIC_MIN_CHARS = 150_000      # ~5_000 turns × 30 chars/turn

# Upload guardrails. Mirrors the recordings service's bounds style.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024    # 200 MB — books/transcripts dominate
ALLOWED_EXTENSIONS_BY_KIND: dict[CorpusItemKind, frozenset[str]] = {
    CorpusItemKind.text: frozenset({".txt", ".md", ".pdf", ".epub", ".docx"}),
    CorpusItemKind.transcript: frozenset({".txt", ".md", ".srt", ".vtt", ".json"}),
    CorpusItemKind.conversation: frozenset({".txt", ".json", ".csv", ".zip"}),
}
