"""
Consent REST API — gates corpus ingestion + future TTS synthesis.

Endpoints:
    POST   /api/consent/                       create record (returns 201)
    GET    /api/consent/{persona_id}           list records for persona
    GET    /api/consent/{persona_id}/{consent_id}   single record
    DELETE /api/consent/{persona_id}/{consent_id}   REVOKE (not hard-delete)
    GET    /api/consent/{persona_id}/check?purpose=...   bool gate check

Hard-delete is intentionally NOT exposed. Revocation preserves the
record + adds a tombstone (per RFC §M-Consent — audit trail
requirement under NO FAKES Act + EU AI Act).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app import config as _cfg
from app.services.consent import (
    ConsentRecord,
    ConsentRepository,
    ConsentService,
    JsonConsentRepository,
    NoActiveConsentError,
    ConsentRecordRevokedError,
    ConsentRecordExpiredError,
)
from app.services.consent.repository import ConsentRecordNotFound

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/consent", tags=["consent"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------
def get_consent_service() -> ConsentService:
    repo = JsonConsentRepository(personas_root=_cfg.personas_dir())
    return ConsentService(repository=repo)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class RevokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revoking_party_name: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    derived_artifact_status: Optional[str] = None


class CheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    purpose: str
    allowed: bool
    # When allowed, the consent record that authorized it. Useful for
    # the UI to display "covered until 2046".
    covering_consent_id: Optional[str] = None


class ListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_id: str
    records: list[ConsentRecord]
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/", response_model=ConsentRecord, status_code=status.HTTP_201_CREATED)
async def api_create_consent(
    record: ConsentRecord,
    service: ConsentService = Depends(get_consent_service),
) -> ConsentRecord:
    """Create a consent record. Validation lives in the service
    (allowed relationships, known purposes, post-mortem expiry cap)."""
    try:
        return service.create(record)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{persona_id}", response_model=ListResponse)
async def api_list_consent(
    persona_id: str,
    service: ConsentService = Depends(get_consent_service),
) -> ListResponse:
    records = service.list(persona_id)
    return ListResponse(
        persona_id=persona_id,
        records=records,
        count=len(records),
    )


@router.get(
    "/{persona_id}/check",
    response_model=CheckResponse,
)
async def api_check_consent(
    persona_id: str,
    purpose: str = Query(..., min_length=1),
    recording_id: Optional[str] = Query(default=None),
    service: ConsentService = Depends(get_consent_service),
) -> CheckResponse:
    """Non-raising gate check. Returns allowed=true/false + the
    covering consent_id when allowed. UI uses this to decide whether
    to show the upload button."""
    try:
        rec = service.assert_allowed(persona_id, purpose, recording_id)
        return CheckResponse(
            persona_id=persona_id,
            purpose=purpose,
            allowed=True,
            covering_consent_id=rec.consent_id,
        )
    except (
        NoActiveConsentError,
        ConsentRecordRevokedError,
        ConsentRecordExpiredError,
    ):
        return CheckResponse(
            persona_id=persona_id,
            purpose=purpose,
            allowed=False,
            covering_consent_id=None,
        )


@router.get("/{persona_id}/{consent_id}", response_model=ConsentRecord)
async def api_get_consent(
    persona_id: str,
    consent_id: str,
    service: ConsentService = Depends(get_consent_service),
) -> ConsentRecord:
    try:
        return service.get(persona_id, consent_id)
    except ConsentRecordNotFound:
        raise HTTPException(status_code=404, detail=f"consent {consent_id} not found")


@router.delete("/{persona_id}/{consent_id}", response_model=ConsentRecord)
async def api_revoke_consent(
    persona_id: str,
    consent_id: str,
    body: RevokeRequest,
    service: ConsentService = Depends(get_consent_service),
) -> ConsentRecord:
    """DELETE = revoke (NOT hard-delete). Returns the post-revoke
    record with the tombstone attached."""
    try:
        return service.revoke(
            persona_id=persona_id,
            consent_id=consent_id,
            revoking_party_name=body.revoking_party_name,
            reason=body.reason,
            derived_artifact_status=body.derived_artifact_status,
        )
    except ConsentRecordNotFound:
        raise HTTPException(status_code=404, detail=f"consent {consent_id} not found")
