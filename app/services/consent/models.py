"""
Pydantic models for consent records.

Schema versioning is explicit — `schema_version` is on the record so
revocation flows can detect old shapes during migrations. v1 ships
2026-06-04 per RFC_M6 §M-Consent.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = 1


class PersonaState(str, Enum):
    """Whether the persona is still alive (pre-mortem) or has passed
    (post-mortem). Different jurisdictions have different rules; per
    CA AB 1836 post-mortem rights last 70 years. The UI must surface
    this distinction at consent capture per RFC_M6 §M-Consent."""

    PRE_MORTEM = "pre_mortem"
    POST_MORTEM = "post_mortem"


class ConsentStatus(str, Enum):
    """Computed status of the record at read time. Storage stores
    `status=active` + an optional `revocation` block; expiry is
    derived from `scope.expires_at` vs `datetime.now(timezone.utc)`."""

    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Sub-records
# ---------------------------------------------------------------------------
class ConsentingParty(BaseModel):
    """Who actually granted the consent. May or may not be the persona.
    For post-mortem personas, the consenting party is usually a
    next-of-kin or estate executor."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    relationship_to_persona: str = Field(..., min_length=1)
    # Allowed values per RFC §M-Consent — checked at the service layer,
    # not here, so future relationship types don't require a model bump.
    captured_at: datetime
    # Where the consent was captured (optional, for audit trail).
    capture_location: Optional[str] = None


class ConsentScope(BaseModel):
    """What the consent allows. Narrow scopes are encouraged — broad
    "everything" consents are a regulatory red flag."""

    model_config = ConfigDict(extra="forbid")

    # What the data + voice may be used for.
    purposes: list[str] = Field(..., min_length=1)
    # Who is allowed to hear / read the synthesized output. e.g.
    # ["self", "family", "all_listeners"]. Empty list = no listeners
    # allowed = useless but explicitly valid (drift detection).
    listener_scope: list[str] = Field(default_factory=list)
    # Hard expiry on the consent itself. None = no expiry. Note CA
    # AB 1836 caps post-mortem at 70 years — service layer enforces.
    expires_at: Optional[datetime] = None

    @field_validator("purposes")
    @classmethod
    def _purposes_no_blank(cls, v: list[str]) -> list[str]:
        for p in v:
            if not p or not p.strip():
                raise ValueError("purpose entries must be non-blank")
        return v


class ConsentJurisdiction(BaseModel):
    """Which laws apply. Captured at intake so a later revocation can
    correctly apply the right SLA (EU AI Act vs NO FAKES Act vs
    CA AB 1836 differ)."""

    model_config = ConfigDict(extra="forbid")

    # ISO 3166-1 alpha-2 country code.
    country_code: str = Field(..., min_length=2, max_length=2)
    # Subdivision (e.g. "CA" for California). Optional.
    region_code: Optional[str] = None
    # Free-form list of named laws this consent is governed under.
    # Validation lives in the service layer — keep model flexible.
    applicable_laws: list[str] = Field(default_factory=list)


class ConsentRevocation(BaseModel):
    """Set when status=revoked. Persisted as a tombstone — the original
    consent block stays so audit trails can show what was revoked."""

    model_config = ConfigDict(extra="forbid")

    revoked_at: datetime
    reason: str
    revoking_party_name: str
    # Free-form note explaining what derived artifacts (LoRAs,
    # synthesized audio) are pending re-training / removal.
    derived_artifact_status: Optional[str] = None


# ---------------------------------------------------------------------------
# Main record
# ---------------------------------------------------------------------------
class ConsentRecord(BaseModel):
    """Single consent record. One record per (persona_id, scope) — a
    persona can have multiple records covering different purposes
    granted at different times by different parties."""

    model_config = ConfigDict(extra="forbid")

    consent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona_id: str = Field(..., min_length=1)
    # If set, this consent applies to one specific recording only. If
    # None, it covers all recordings + derived corpus for the persona.
    recording_id: Optional[str] = None

    consenting_party: ConsentingParty
    scope: ConsentScope
    jurisdiction: ConsentJurisdiction

    persona_state: PersonaState

    # Storage-level status — the COMPUTED status (factoring in
    # expiry) is derived by the service layer's `current_status()`.
    status: ConsentStatus = ConsentStatus.ACTIVE
    revocation: Optional[ConsentRevocation] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    schema_version: int = SCHEMA_VERSION

    def is_revoked(self) -> bool:
        return self.status == ConsentStatus.REVOKED or self.revocation is not None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.scope.expires_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now >= self.scope.expires_at

    def current_status(self, now: Optional[datetime] = None) -> ConsentStatus:
        """Computed status — REVOKED > EXPIRED > ACTIVE."""
        if self.is_revoked():
            return ConsentStatus.REVOKED
        if self.is_expired(now):
            return ConsentStatus.EXPIRED
        return ConsentStatus.ACTIVE

    def covers_purpose(self, purpose: str) -> bool:
        return purpose in self.scope.purposes
