"""
Contract tests for the personas REST API.

Covers each of the 5 endpoints with ≥1 happy + ≥1 failure mode, asserts
both behavior and Pydantic response shape.
"""
from __future__ import annotations

import threading

import pytest

from app.services.personas import Persona
from app.api.personas import DeletePersonaResponse, PersonasListResponse

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# GET /api/personas/
# ---------------------------------------------------------------------------
class TestList:
    def test_seeded_fixed_personas_present(self, client, assert_matches_schema):
        r = client.get("/api/personas/")
        assert r.status_code == 200
        parsed = assert_matches_schema(PersonasListResponse, r.json())
        ids = {p.persona_id for p in parsed.personas}
        # All 4 fixed family members must be in the seeded list.
        assert {"xiao_s", "caregiver", "elder_gentle", "elder_playful"} <= ids
        # All fixed personas have type=fixed.
        fixed = {p.persona_id for p in parsed.personas if p.type.value == "fixed"}
        assert fixed >= {"xiao_s", "caregiver", "elder_gentle", "elder_playful"}


# ---------------------------------------------------------------------------
# GET /api/personas/{id}
# ---------------------------------------------------------------------------
class TestGet:
    def test_get_seeded(self, client, assert_matches_schema):
        r = client.get("/api/personas/xiao_s")
        assert r.status_code == 200
        parsed = assert_matches_schema(Persona, r.json())
        assert parsed.persona_id == "xiao_s"
        assert parsed.is_fixed()

    def test_404_unknown(self, client):
        r = client.get("/api/personas/nope")
        assert r.status_code == 404
        body = r.json()
        assert body["error"] == "persona_not_found"
        assert body["details"]["persona_id"] == "nope"


# ---------------------------------------------------------------------------
# POST /api/personas/
# ---------------------------------------------------------------------------
class TestCreate:
    def test_happy_path(self, client, assert_matches_schema):
        r = client.post(
            "/api/personas/",
            json={"persona_id": "guest1", "name": "Guest One", "is_family": False},
        )
        assert r.status_code == 200, r.text
        parsed = assert_matches_schema(Persona, r.json())
        assert parsed.persona_id == "guest1"
        assert parsed.type.value == "dynamic"
        # Verify it's in the list.
        listing = client.get("/api/personas/").json()
        assert any(p["persona_id"] == "guest1" for p in listing["personas"])

    def test_invalid_id_format(self, client):
        r = client.post(
            "/api/personas/",
            json={"persona_id": "Bad-ID", "name": "x"},
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_id_format"

    def test_duplicate(self, client):
        client.post(
            "/api/personas/",
            json={"persona_id": "guest1", "name": "Guest", "is_family": False},
        )
        r = client.post(
            "/api/personas/",
            json={"persona_id": "guest1", "name": "Other", "is_family": False},
        )
        assert r.status_code == 409
        assert r.json()["error"] == "duplicate_id"

    def test_collision_with_fixed_persona(self, client):
        r = client.post(
            "/api/personas/",
            json={"persona_id": "xiao_s", "name": "Imposter"},
        )
        assert r.status_code == 400
        assert r.json()["error"] == "fixed_persona_readonly"

    def test_unknown_field_rejected(self, client):
        r = client.post(
            "/api/personas/",
            json={"persona_id": "guest1", "name": "G", "bonus": True},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# PATCH /api/personas/{id}
# ---------------------------------------------------------------------------
class TestUpdate:
    def test_rename_dynamic(self, client):
        client.post(
            "/api/personas/",
            json={"persona_id": "guest1", "name": "Original"},
        )
        r = client.patch("/api/personas/guest1", json={"name": "Renamed"})
        assert r.status_code == 200
        assert r.json()["name"] == "Renamed"

    def test_cannot_rename_fixed(self, client):
        r = client.patch("/api/personas/xiao_s", json={"name": "Other"})
        assert r.status_code == 400
        assert r.json()["error"] == "fixed_persona_readonly"

    def test_404_unknown(self, client):
        r = client.patch("/api/personas/nope", json={"name": "x"})
        assert r.status_code == 404

    def test_unknown_field_rejected(self, client):
        client.post(
            "/api/personas/",
            json={"persona_id": "guest1", "name": "G"},
        )
        r = client.patch("/api/personas/guest1", json={"surprise": "x"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /api/personas/{id}
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_dynamic(self, client, assert_matches_schema):
        client.post(
            "/api/personas/",
            json={"persona_id": "guest1", "name": "G"},
        )
        r = client.delete("/api/personas/guest1")
        assert r.status_code == 200
        parsed = assert_matches_schema(DeletePersonaResponse, r.json())
        assert parsed.persona_id == "guest1"
        # Verify gone.
        assert client.get("/api/personas/guest1").status_code == 404

    def test_cannot_delete_fixed(self, client):
        r = client.delete("/api/personas/xiao_s")
        assert r.status_code == 400
        assert r.json()["error"] == "fixed_persona_readonly"

    def test_404_unknown(self, client):
        r = client.delete("/api/personas/nope")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
class TestConcurrency:
    def test_parallel_creates_all_persist(self, client):
        """30 concurrent POSTs of distinct personas — all must land."""
        N = 30
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                r = client.post(
                    "/api/personas/",
                    json={"persona_id": f"guest{i:02d}", "name": f"Guest {i}"},
                )
                if r.status_code != 200:
                    errors.append(AssertionError(f"thread {i}: {r.status_code} {r.text}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Worker errors: {errors[:3]}"
        ids = {p["persona_id"] for p in client.get("/api/personas/").json()["personas"]}
        for i in range(N):
            assert f"guest{i:02d}" in ids, f"Lost write: guest{i:02d}"
