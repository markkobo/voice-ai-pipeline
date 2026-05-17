"""
Corpus REST API — per-persona ingestion surface (RFC_M6 Phase 0).

Endpoints:
    POST   /api/corpus/upload          multipart: persona_id, kind, file, ...
    GET    /api/corpus/{persona_id}    list items
    GET    /api/corpus/{persona_id}/manifest   rolled-up stats + threshold flags
    GET    /api/corpus/{persona_id}/items/{item_id}    single item
    DELETE /api/corpus/{persona_id}/items/{item_id}    remove

Heavy ingestion (PDF/EPUB/audio→text extraction, chunking, embedding) is
a follow-up slice; these endpoints only persist raw bytes + metadata
today. Item status will stay `uploaded` until ingestion lands.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict

from app.api._dependencies import get_corpus_service, get_ingestion_service
from app.api._errors import (
    CorpusEmptyError,
    CorpusIngestionFailedError,
    CorpusIngestionUnsupportedError,
    CorpusItemNotFoundError,
    CorpusTooLargeError,
    InvalidCorpusIdError,
    UnsupportedCorpusFormatError,
)
from app.services.corpus import (
    CorpusItem,
    CorpusItemKind,
    CorpusManifest,
    CorpusService,
    ExtractionFailedError,
    IngestionService,
    UnsupportedIngestionFormatError,
)
from app.services.corpus.repository import CorpusItemNotFound
from app.services.corpus.service import (
    CorpusEmptyError as SvcCorpusEmptyError,
    CorpusTooLargeError as SvcCorpusTooLargeError,
    InvalidCorpusIdError as SvcInvalidCorpusIdError,
    UnsupportedCorpusFormatError as SvcUnsupportedCorpusFormatError,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/corpus", tags=["corpus"])


# ---------------------------------------------------------------------------
# Pydantic response models — explicit shapes pinned by contract tests.
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    persona_id: str
    kind: str
    status: str
    size_bytes: int


class ListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    items: list[CorpusItem]
    count: int


class DeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "deleted"
    item_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=UploadResponse)
async def api_upload_corpus(
    file: UploadFile = File(...),
    persona_id: str = Form(...),
    kind: CorpusItemKind = Form(...),
    source: Optional[str] = Form(None),
    source_date: Optional[datetime] = Form(None),
    listener_tag: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    service: CorpusService = Depends(get_corpus_service),
) -> UploadResponse:
    file_bytes = await file.read()
    try:
        item = service.upload(
            persona_id=persona_id,
            kind=kind,
            file_bytes=file_bytes,
            filename=file.filename or "upload",
            mime_type=file.content_type,
            source=source,
            source_date=source_date,
            listener_tag=listener_tag,
            notes=notes,
        )
    except SvcInvalidCorpusIdError as e:
        raise InvalidCorpusIdError(
            str(e), details={"persona_id": persona_id},
        ) from e
    except SvcCorpusEmptyError as e:
        raise CorpusEmptyError(str(e)) from e
    except SvcCorpusTooLargeError as e:
        raise CorpusTooLargeError(
            str(e),
            details={"size_bytes": len(file_bytes)},
        ) from e
    except SvcUnsupportedCorpusFormatError as e:
        raise UnsupportedCorpusFormatError(
            str(e),
            details={"filename": file.filename, "kind": kind.value},
        ) from e

    return UploadResponse(
        item_id=item.item_id,
        persona_id=item.persona_id,
        kind=item.kind.value,
        status=item.status.value,
        size_bytes=item.size_bytes,
    )


@router.get("/{persona_id}", response_model=ListResponse)
async def api_list_corpus(
    persona_id: str,
    service: CorpusService = Depends(get_corpus_service),
) -> ListResponse:
    try:
        items = service.list(persona_id)
    except SvcInvalidCorpusIdError as e:
        raise InvalidCorpusIdError(
            str(e), details={"persona_id": persona_id},
        ) from e
    return ListResponse(
        persona_id=persona_id,
        items=items,
        count=len(items),
    )


@router.get("/{persona_id}/manifest", response_model=CorpusManifest)
async def api_get_manifest(
    persona_id: str,
    service: CorpusService = Depends(get_corpus_service),
) -> CorpusManifest:
    try:
        return service.compute_manifest(persona_id)
    except SvcInvalidCorpusIdError as e:
        raise InvalidCorpusIdError(
            str(e), details={"persona_id": persona_id},
        ) from e


@router.get("/{persona_id}/items/{item_id}", response_model=CorpusItem)
async def api_get_corpus_item(
    persona_id: str,
    item_id: str,
    service: CorpusService = Depends(get_corpus_service),
) -> CorpusItem:
    try:
        return service.get(persona_id, item_id)
    except SvcInvalidCorpusIdError as e:
        raise InvalidCorpusIdError(
            str(e),
            details={"persona_id": persona_id, "item_id": item_id},
        ) from e
    except CorpusItemNotFound as e:
        raise CorpusItemNotFoundError(
            f"Corpus item {item_id!r} not found for persona {persona_id!r}",
            details={"persona_id": persona_id, "item_id": item_id},
        ) from e


@router.delete(
    "/{persona_id}/items/{item_id}", response_model=DeleteResponse
)
async def api_delete_corpus_item(
    persona_id: str,
    item_id: str,
    service: CorpusService = Depends(get_corpus_service),
) -> DeleteResponse:
    try:
        service.delete(persona_id, item_id)
    except SvcInvalidCorpusIdError as e:
        raise InvalidCorpusIdError(
            str(e),
            details={"persona_id": persona_id, "item_id": item_id},
        ) from e
    except CorpusItemNotFound as e:
        raise CorpusItemNotFoundError(
            f"Corpus item {item_id!r} not found for persona {persona_id!r}",
            details={"persona_id": persona_id, "item_id": item_id},
        ) from e
    return DeleteResponse(item_id=item_id)


@router.post(
    "/{persona_id}/items/{item_id}/ingest", response_model=CorpusItem
)
async def api_ingest_corpus_item(
    persona_id: str,
    item_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> CorpusItem:
    """Run the extractor on a previously-uploaded item.

    Idempotent — re-ingesting overwrites extracted.txt + chunks.jsonl.
    Returns the updated CorpusItem with extracted_chars + chunk_count
    + status=ingested (or status=failed if extraction failed; the item's
    `error` field carries the reason in that case).

    Slice 2A supports `.txt`/`.md` only. PDF/EPUB/DOCX/audio/video and
    chat-export parsers land in later slices (RFC_M6 §3 Phase 0).
    """
    try:
        return ingestion.ingest(persona_id, item_id)
    except SvcInvalidCorpusIdError as e:
        raise InvalidCorpusIdError(
            str(e),
            details={"persona_id": persona_id, "item_id": item_id},
        ) from e
    except CorpusItemNotFound as e:
        raise CorpusItemNotFoundError(
            f"Corpus item {item_id!r} not found for persona {persona_id!r}",
            details={"persona_id": persona_id, "item_id": item_id},
        ) from e
    except UnsupportedIngestionFormatError as e:
        raise CorpusIngestionUnsupportedError(
            f"No extractor available for {str(e)!r}; "
            "supported in slice 2A: .txt, .md",
            details={
                "persona_id": persona_id,
                "item_id": item_id,
                "extension": str(e),
                "supported": sorted(ingestion.supported_extensions()),
            },
        ) from e
    except ExtractionFailedError as e:
        raise CorpusIngestionFailedError(
            str(e),
            details={"persona_id": persona_id, "item_id": item_id},
        ) from e
