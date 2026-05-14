"""
Contract tests for the listeners REST API.

Mirrors the personas contract test layout. Covers 5 endpoints with ≥1
happy + ≥1 failure mode + concurrency.
"""
from __future__ import annotations

import threading

import pytest

from app.services.listeners import Listener
from app.api.listeners import DeleteListenerResponse, ListenersListResponse

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# GET /api/listeners/
# ---------------------------------------------------------------------------
class TestList:
    def test_seeded_listeners_present(self, client, assert_matches_schema):
        r = client.get("/api/listeners/")
        assert r.status_code == 200
        parsed = assert_matches_schema(ListenersListResponse, r.json())
        ids = {l.listener_id for l in parsed.listeners}
        assert {"child", "mom", "dad", "friend", "elder", "reporter", "default"} <= ids
        # All seeded listeners have is_seed=True.
        seeded = {l.listener_id for l in parsed.listeners if l.is_seed}
        assert seeded >= {"child", "mom", "dad", "friend", "elder", "reporter", "default"}


# ---------------------------------------------------------------------------
# GET /api/listeners/{id}
# ---------------------------------------------------------------------------
class TestGet:
    def test_get_seeded(self, client, assert_matches_schema):
        r = client.get("/api/listeners/child")
        assert r.status_code == 200
        parsed = assert_matches_schema(Listener, r.json())
        assert parsed.listener_id == "child"
        assert parsed.is_seed is True

    def test_404_unknown(self, client):
        r = client.get("/api/listeners/nope")
        assert r.status_code == 404
        body = r.json()
        assert body["error"] == "listener_not_found"


# ---------------------------------------------------------------------------
# POST /api/listeners/
# ---------------------------------------------------------------------------
class TestCreate:
    def test_happy_path(self, client, assert_matches_schema):
        r = client.post(
            "/api/listeners/",
            json={
                "listener_id": "boss",
                "name": "Boss",
                "is_family": False,
                "default_emotion": "溫和",
            },
        )
        assert r.status_code == 200, r.text
        parsed = assert_matches_schema(Listener, r.json())
        assert parsed.listener_id == "boss"
        assert parsed.is_seed is False
        assert parsed.default_emotion == "溫和"

    def test_invalid_id_format(self, client):
        r = client.post(
            "/api/listeners/",
            json={"listener_id": "Bad-ID", "name": "x"},
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_id_format"

    def test_invalid_emotion(self, client):
        r = client.post(
            "/api/listeners/",
            json={
                "listener_id": "boss",
                "name": "Boss",
                "default_emotion": "fury",  # not in VALID_EMOTIONS
            },
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_emotion"

    def test_duplicate(self, client):
        client.post(
            "/api/listeners/",
            json={"listener_id": "boss", "name": "Boss"},
        )
        r = client.post(
            "/api/listeners/",
            json={"listener_id": "boss", "name": "Other"},
        )
        assert r.status_code == 409
        assert r.json()["error"] == "duplicate_id"

    def test_duplicate_with_seed(self, client):
        # `child` is a seeded listener — create should reject as duplicate.
        r = client.post(
            "/api/listeners/",
            json={"listener_id": "child", "name": "Imposter"},
        )
        assert r.status_code == 409

    def test_unknown_field_rejected(self, client):
        r = client.post(
            "/api/listeners/",
            json={"listener_id": "boss", "name": "B", "extra": True},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# PATCH /api/listeners/{id}
# ---------------------------------------------------------------------------
class TestUpdate:
    def test_rename(self, client):
        r = client.patch("/api/listeners/child", json={"name": "Kid"})
        assert r.status_code == 200
        assert r.json()["name"] == "Kid"

    def test_change_emotion(self, client):
        r = client.patch("/api/listeners/child", json={"default_emotion": "幽默"})
        assert r.status_code == 200
        assert r.json()["default_emotion"] == "幽默"

    def test_invalid_emotion_rejected(self, client):
        r = client.patch("/api/listeners/child", json={"default_emotion": "fury"})
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_emotion"

    def test_404_unknown(self, client):
        r = client.patch("/api/listeners/nope", json={"name": "x"})
        assert r.status_code == 404

    def test_unknown_field_rejected(self, client):
        r = client.patch("/api/listeners/child", json={"surprise": "x"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /api/listeners/{id}
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_user_created(self, client, assert_matches_schema):
        client.post(
            "/api/listeners/",
            json={"listener_id": "boss", "name": "B"},
        )
        r = client.delete("/api/listeners/boss")
        assert r.status_code == 200
        parsed = assert_matches_schema(DeleteListenerResponse, r.json())
        assert parsed.listener_id == "boss"
        assert client.get("/api/listeners/boss").status_code == 404

    def test_cannot_delete_seed(self, client):
        """New in Phase 1.3: seeded listeners are read-only.

        The legacy code allowed `DELETE /api/listeners/child`, which was a
        real bug — downstream code assumes the seven seeded listeners exist.
        """
        r = client.delete("/api/listeners/child")
        assert r.status_code == 400
        assert r.json()["error"] == "seed_listener_readonly"

    def test_404_unknown(self, client):
        r = client.delete("/api/listeners/nope")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
class TestConcurrency:
    def test_parallel_creates_all_persist(self, client):
        N = 30
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                r = client.post(
                    "/api/listeners/",
                    json={"listener_id": f"guest{i:02d}", "name": f"Guest {i}"},
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
        ids = {l["listener_id"] for l in client.get("/api/listeners/").json()["listeners"]}
        for i in range(N):
            assert f"guest{i:02d}" in ids, f"Lost write: guest{i:02d}"
