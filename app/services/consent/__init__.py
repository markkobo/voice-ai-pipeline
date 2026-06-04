"""
Consent capture, lookup, and revocation — M-Consent milestone.

Driven by NO FAKES Act (US), EU AI Act, CA AB 1836, CA AB 2602, and
HIPAA (B2B2C eldercare path per Gemini review 2026-06-03). Server-side
consent records gate corpus ingestion (POST /api/corpus/ingest fails
if no active consent record exists for the persona).
"""
from .models import (
    ConsentRecord,
    ConsentScope,
    ConsentingParty,
    ConsentJurisdiction,
    ConsentRevocation,
    PersonaState,
    ConsentStatus,
)
from .repository import (
    ConsentRepository,
    JsonConsentRepository,
    ConsentRecordNotFound,
)
from .service import (
    ConsentService,
    NoActiveConsentError,
    ConsentRecordRevokedError,
    ConsentRecordExpiredError,
)

__all__ = [
    "ConsentRecord",
    "ConsentScope",
    "ConsentingParty",
    "ConsentJurisdiction",
    "ConsentRevocation",
    "PersonaState",
    "ConsentStatus",
    "ConsentRepository",
    "JsonConsentRepository",
    "ConsentRecordNotFound",
    "ConsentService",
    "NoActiveConsentError",
    "ConsentRecordRevokedError",
    "ConsentRecordExpiredError",
]
