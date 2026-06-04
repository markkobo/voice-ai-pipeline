"""
Contract tests for /api/consent/* endpoints + the corpus ingest gate.

M-Consent first commit (2026-06-04). Lock in:
- POST /api/consent/ creates a record (returns 201 + ConsentRecord)
- GET  /api/consent/{persona_id} lists records
- GET  /api/consent/{persona_id}/{consent_id} returns single record
- GET  /api/consent/{persona_id}/check?purpose=... gate check
- DELETE /api/consent/{persona_id}/{consent_id} revokes (not hard-delete)
- POST /api/corpus/upload returns 403 unless an active consent record
  exists for the persona (purpose=rag_corpus)
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.consent import ConsentRecord


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Each test gets its own personas_dir so records don't leak.
    Monkey-patch app.config.personas_dir directly — DATA_ROOT env
    is unreliable because the legacy-workspace fallback short-
    circuits in this dev environment and any prod-cached value
    sticks around across tests."""
    personas_root = tmp_path / "personas"
    personas_root.mkdir(parents=True, exist_ok=True)
    import app.config as _cfg
    monkeypatch.setattr(_cfg, "personas_dir", lambda: personas_root)
    # Also clear the lru_cache on the original so any code that
    # imported the original function rather than calling _cfg.personas_dir
    # still gets the patched value next call.
    _cfg.personas_dir.cache_clear() if hasattr(_cfg.personas_dir, "cache_clear") else None
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_consent_payload(
    persona_id: str = "test",
    purposes: list[str] | None = None,
    relationship: str = "self",
    expires_at: str | None = None,
    persona_state: str = "pre_mortem",
) -> dict:
    return {
        "persona_id": persona_id,
        "consenting_party": {
            "name": "Mark Ko",
            "relationship_to_persona": relationship,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
        "scope": {
            "purposes": purposes or ["rag_corpus", "voice_cloning"],
            "listener_scope": ["family"],
            "expires_at": expires_at,
        },
        "jurisdiction": {
            "country_code": "US",
            "region_code": "CA",
            "applicable_laws": ["NO_FAKES_ACT", "CA_AB_1836"],
        },
        "persona_state": persona_state,
    }


# ---------------------------------------------------------------------------
# Create + read
# ---------------------------------------------------------------------------
class TestCreateAndRead:
    def test_create_returns_201_and_full_record(self, client):
        r = client.post("/api/consent/", json=_make_consent_payload())
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["persona_id"] == "test"
        assert body["status"] == "active"
        assert body["revocation"] is None
        assert "consent_id" in body
        assert "created_at" in body
        assert body["schema_version"] == 1

    def test_get_by_id_roundtrip(self, client):
        r = client.post("/api/consent/", json=_make_consent_payload())
        consent_id = r.json()["consent_id"]
        r2 = client.get(f"/api/consent/test/{consent_id}")
        assert r2.status_code == 200
        assert r2.json()["consent_id"] == consent_id

    def test_get_unknown_returns_404(self, client):
        r = client.get("/api/consent/test/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_list_empty(self, client):
        r = client.get("/api/consent/nonexistent")
        assert r.status_code == 200
        assert r.json() == {
            "persona_id": "nonexistent",
            "records": [],
            "count": 0,
        }

    def test_list_returns_newest_first(self, client):
        client.post("/api/consent/", json=_make_consent_payload())
        client.post("/api/consent/", json=_make_consent_payload(purposes=["audit_review"]))
        r = client.get("/api/consent/test")
        body = r.json()
        assert body["count"] == 2
        # Newest first by created_at — the audit_review one was added second.
        assert "audit_review" in body["records"][0]["scope"]["purposes"]

    def test_create_rejects_unknown_purpose(self, client):
        r = client.post(
            "/api/consent/",
            json=_make_consent_payload(purposes=["spam_for_profit"]),
        )
        assert r.status_code == 400
        assert "unknown purpose" in r.text

    def test_create_rejects_unknown_relationship(self, client):
        r = client.post(
            "/api/consent/",
            json=_make_consent_payload(relationship="stranger"),
        )
        assert r.status_code == 400
        assert "relationship_to_persona" in r.text


# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------
class TestRevoke:
    def test_revoke_stamps_tombstone_not_hard_delete(self, client):
        r = client.post("/api/consent/", json=_make_consent_payload())
        consent_id = r.json()["consent_id"]
        rev = client.request(
            "DELETE",
            f"/api/consent/test/{consent_id}",
            json={
                "revoking_party_name": "Mark Ko",
                "reason": "demo purposes",
                "derived_artifact_status": "v12 LoRA pending retrain",
            },
        )
        assert rev.status_code == 200
        body = rev.json()
        assert body["status"] == "revoked"
        assert body["revocation"]["reason"] == "demo purposes"
        assert body["revocation"]["revoking_party_name"] == "Mark Ko"

        # Record is still fetchable — tombstone, not hard delete.
        getr = client.get(f"/api/consent/test/{consent_id}")
        assert getr.status_code == 200
        assert getr.json()["status"] == "revoked"

    def test_revoke_unknown_returns_404(self, client):
        r = client.request(
            "DELETE",
            "/api/consent/test/00000000-0000-0000-0000-000000000000",
            json={"revoking_party_name": "x", "reason": "y"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Check (gate query)
# ---------------------------------------------------------------------------
class TestCheck:
    def test_check_no_record_returns_allowed_false(self, client):
        r = client.get(
            "/api/consent/test/check",
            params={"purpose": "rag_corpus"},
        )
        assert r.status_code == 200
        assert r.json()["allowed"] is False
        assert r.json()["covering_consent_id"] is None

    def test_check_with_active_record_returns_true(self, client):
        c = client.post("/api/consent/", json=_make_consent_payload())
        consent_id = c.json()["consent_id"]
        r = client.get(
            "/api/consent/test/check",
            params={"purpose": "rag_corpus"},
        )
        body = r.json()
        assert body["allowed"] is True
        assert body["covering_consent_id"] == consent_id

    def test_check_wrong_purpose_returns_false(self, client):
        client.post(
            "/api/consent/",
            json=_make_consent_payload(purposes=["audit_review"]),
        )
        r = client.get(
            "/api/consent/test/check",
            params={"purpose": "voice_cloning"},
        )
        assert r.json()["allowed"] is False

    def test_check_after_revoke_returns_false(self, client):
        c = client.post("/api/consent/", json=_make_consent_payload())
        consent_id = c.json()["consent_id"]
        client.request(
            "DELETE",
            f"/api/consent/test/{consent_id}",
            json={"revoking_party_name": "x", "reason": "y"},
        )
        r = client.get(
            "/api/consent/test/check",
            params={"purpose": "rag_corpus"},
        )
        assert r.json()["allowed"] is False


# ---------------------------------------------------------------------------
# Corpus upload gate
# ---------------------------------------------------------------------------
class TestCorpusUploadGate:
    """POST /api/corpus/upload now 403s without an active consent record."""

    def _upload(self, client, persona_id="test"):
        return client.post(
            "/api/corpus/upload",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
            data={"persona_id": persona_id, "kind": "text"},
        )

    def test_upload_without_consent_returns_403(self, client):
        r = self._upload(client)
        assert r.status_code == 403, r.text
        assert "consent" in r.text.lower()

    def test_upload_with_active_consent_succeeds(self, client):
        client.post("/api/consent/", json=_make_consent_payload())
        r = self._upload(client)
        assert r.status_code == 200, r.text
        assert r.json()["persona_id"] == "test"
        assert r.json()["kind"] == "text"

    def test_upload_with_revoked_consent_returns_403(self, client):
        c = client.post("/api/consent/", json=_make_consent_payload())
        consent_id = c.json()["consent_id"]
        client.request(
            "DELETE",
            f"/api/consent/test/{consent_id}",
            json={"revoking_party_name": "x", "reason": "y"},
        )
        r = self._upload(client)
        assert r.status_code == 403

    def test_upload_with_wrong_purpose_consent_returns_403(self, client):
        """Consent must cover rag_corpus specifically — not just exist."""
        client.post(
            "/api/consent/",
            json=_make_consent_payload(purposes=["audit_review"]),
        )
        r = self._upload(client)
        assert r.status_code == 403
