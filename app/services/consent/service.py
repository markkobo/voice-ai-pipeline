"""
Consent service — domain operations on top of the repository.

Responsibilities:
- Validate inputs that the model layer keeps loose (e.g.,
  `relationship_to_persona` allowed values, post-mortem expiry caps).
- Provide the "is the persona allowed to be ingested / synthesized
  RIGHT NOW for purpose X" check that gates corpus + TTS endpoints.
- Apply revocation and emit a tombstone instead of hard-deleting.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import (
    ConsentRecord,
    ConsentRevocation,
    ConsentStatus,
    PersonaState,
)
from .repository import ConsentRepository, ConsentRecordNotFound

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class NoActiveConsentError(PermissionError):
    """Raised when an action is requested but no active consent record
    covers the (persona, purpose) tuple. Maps to HTTP 403 at the API
    edge — refusing the action, with a clear reason for the user."""


class ConsentRecordRevokedError(PermissionError):
    """Raised when the consent existed but has been revoked."""


class ConsentRecordExpiredError(PermissionError):
    """Raised when the consent existed but is past its expires_at."""


# ---------------------------------------------------------------------------
# Allowed-values policy (kept here, not in the model, so additions
# don't require a schema bump).
# ---------------------------------------------------------------------------
ALLOWED_RELATIONSHIPS = frozenset({
    "self",
    "guardian",
    "next_of_kin",
    "estate_executor",
    "legal_representative",
})

KNOWN_PURPOSES = frozenset({
    "voice_cloning",          # TTS LoRA training on this persona's audio
    "persona_lora_training",  # M9 LLM persona LoRA
    "rag_corpus",             # M8 memory retrieval over text + transcripts
    "synthesis_for_family",   # produce voice output for family listeners
    "synthesis_for_self",     # produce voice output for the persona only
    "audit_review",           # internal review for safety + compliance
})

KNOWN_LISTENER_SCOPES = frozenset({
    "self",
    "family",
    "specific_listener",  # paired with a per-listener allow-list elsewhere
    "all_listeners",
})

# Maximum expiry for post-mortem consent — CA AB 1836 caps at 70 years
# from the year of death; we approximate with a 70y window from creation.
POST_MORTEM_MAX_EXPIRY = timedelta(days=365 * 70)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class ConsentService:
    def __init__(self, repository: ConsentRepository) -> None:
        self.repository = repository

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(record: ConsentRecord) -> None:
        if record.consenting_party.relationship_to_persona not in ALLOWED_RELATIONSHIPS:
            raise ValueError(
                f"relationship_to_persona must be one of "
                f"{sorted(ALLOWED_RELATIONSHIPS)}; got "
                f"{record.consenting_party.relationship_to_persona!r}"
            )
        for purpose in record.scope.purposes:
            if purpose not in KNOWN_PURPOSES:
                raise ValueError(
                    f"unknown purpose {purpose!r}; allowed: "
                    f"{sorted(KNOWN_PURPOSES)}"
                )
        for ls in record.scope.listener_scope:
            if ls not in KNOWN_LISTENER_SCOPES:
                raise ValueError(
                    f"unknown listener_scope {ls!r}; allowed: "
                    f"{sorted(KNOWN_LISTENER_SCOPES)}"
                )
        # Post-mortem expiry cap
        if (
            record.persona_state == PersonaState.POST_MORTEM
            and record.scope.expires_at is not None
        ):
            window = record.scope.expires_at - record.created_at
            if window > POST_MORTEM_MAX_EXPIRY:
                raise ValueError(
                    "post-mortem consent expiry exceeds CA AB 1836 70-year "
                    "cap relative to created_at"
                )

    # ------------------------------------------------------------------
    # Create / read / revoke
    # ------------------------------------------------------------------
    def create(self, record: ConsentRecord) -> ConsentRecord:
        self._validate(record)
        self.repository.save(record)
        log.info(
            "Created consent persona_id=%s consent_id=%s purposes=%s",
            record.persona_id,
            record.consent_id,
            record.scope.purposes,
        )
        return record

    def get(self, persona_id: str, consent_id: str) -> ConsentRecord:
        return self.repository.get(persona_id, consent_id)

    def list(self, persona_id: str) -> list[ConsentRecord]:
        return self.repository.list(persona_id)

    def revoke(
        self,
        persona_id: str,
        consent_id: str,
        revoking_party_name: str,
        reason: str,
        derived_artifact_status: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ConsentRecord:
        """Stamp a revocation tombstone on the record. Does NOT delete
        the record from disk — audit trails need the original consent
        block plus the revocation."""
        record = self.repository.get(persona_id, consent_id)
        if record.is_revoked():
            log.warning(
                "Revoke called on already-revoked consent_id=%s", consent_id
            )
            return record
        record.status = ConsentStatus.REVOKED
        record.revocation = ConsentRevocation(
            revoked_at=now or datetime.now(timezone.utc),
            reason=reason,
            revoking_party_name=revoking_party_name,
            derived_artifact_status=derived_artifact_status,
        )
        self.repository.save(record)
        log.info(
            "Revoked consent_id=%s persona_id=%s reason=%s",
            consent_id,
            persona_id,
            reason,
        )
        return record

    # ------------------------------------------------------------------
    # The gate — what corpus + TTS endpoints call.
    # ------------------------------------------------------------------
    def assert_allowed(
        self,
        persona_id: str,
        purpose: str,
        recording_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ConsentRecord:
        """Return the active consent record that covers (persona,
        purpose). If recording_id is given, prefer a record scoped to
        that recording but accept a persona-wide record as fallback.

        Raises:
            NoActiveConsentError: no record covers this tuple at all.
            ConsentRecordRevokedError: a record exists but is revoked.
            ConsentRecordExpiredError: a record exists but is expired.
        """
        now = now or datetime.now(timezone.utc)
        records = self.repository.list(persona_id)

        # First pass: records that cover the (purpose) + match
        # recording_id if specified.
        candidates: list[ConsentRecord] = []
        for r in records:
            if not r.covers_purpose(purpose):
                continue
            if recording_id is not None:
                if r.recording_id is not None and r.recording_id != recording_id:
                    continue
            candidates.append(r)

        if not candidates:
            raise NoActiveConsentError(
                f"no consent record covers persona={persona_id!r} "
                f"purpose={purpose!r}"
            )

        # Among candidates, find the freshest ACTIVE one.
        active = [
            r for r in candidates
            if r.current_status(now) == ConsentStatus.ACTIVE
        ]
        if active:
            # Most recent first (list() already sorts desc by created_at).
            return active[0]

        # No active — surface the most informative failure.
        revoked = [r for r in candidates if r.is_revoked()]
        if revoked:
            raise ConsentRecordRevokedError(
                f"consent for persona={persona_id!r} purpose={purpose!r} "
                f"was revoked: {revoked[0].revocation.reason!r}"
            )
        # Must be expired.
        raise ConsentRecordExpiredError(
            f"consent for persona={persona_id!r} purpose={purpose!r} "
            f"has expired"
        )

    def is_allowed(
        self,
        persona_id: str,
        purpose: str,
        recording_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Non-raising variant. True iff an active record covers the
        request. Useful for UI hints (show/hide upload button) where
        the caller doesn't want to handle three exception types."""
        try:
            self.assert_allowed(persona_id, purpose, recording_id, now)
            return True
        except (
            NoActiveConsentError,
            ConsentRecordRevokedError,
            ConsentRecordExpiredError,
        ):
            return False
