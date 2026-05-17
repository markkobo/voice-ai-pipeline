"""
Contract tests for /api/corpus/* (RFC_M6 Phase 0).

Pins request/response shapes the UI + downstream consumers depend on:
- upload returns UploadResponse with stable keys
- list returns ListResponse with `items` + `count`
- manifest returns CorpusManifest with thresholds block + flags
- delete is idempotent-ish (404 on second call)
- domain errors use the {error, message, details, detail} envelope

Heavy ingestion (PDF/EPUB/audio→text) is NOT exercised here — that's a
follow-up slice. We only check storage + API surface.
"""
from __future__ import annotations

import io

import pytest

from app.api.corpus import DeleteResponse, ListResponse, UploadResponse
from app.services.corpus import CorpusItem, CorpusManifest

pytestmark = pytest.mark.contract


PERSONA_ID = "xiao_s"


# ---------------------------------------------------------------------------
# Small upload helper
# ---------------------------------------------------------------------------
def _upload(
    client,
    *,
    persona_id: str = PERSONA_ID,
    kind: str = "text",
    content: bytes = b"hello world",
    filename: str = "note.txt",
    extra_form: dict | None = None,
):
    """POST /api/corpus/upload with multipart payload."""
    form = {
        "persona_id": persona_id,
        "kind": kind,
    }
    if extra_form:
        form.update(extra_form)
    return client.post(
        "/api/corpus/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
        data=form,
    )


# ---------------------------------------------------------------------------
# Upload — happy paths + error envelopes
# ---------------------------------------------------------------------------
class TestCorpusUpload:
    def test_upload_text_happy_path(self, client, assert_matches_schema):
        r = _upload(client)
        assert r.status_code == 200, r.text
        parsed = assert_matches_schema(UploadResponse, r.json())
        assert parsed.persona_id == PERSONA_ID
        assert parsed.kind == "text"
        assert parsed.status == "uploaded"
        assert parsed.size_bytes == len(b"hello world")
        assert parsed.item_id  # non-empty UUID-ish string

    def test_upload_unknown_extension_415(self, client):
        r = _upload(client, filename="note.exe", content=b"x")
        assert r.status_code == 415
        body = r.json()
        assert body["error"] == "unsupported_corpus_format"
        # Envelope mirrors detail for legacy UI consumers.
        assert body["detail"] == body["message"]
        assert "details" in body
        assert body["details"]["kind"] == "text"

    def test_upload_empty_file_422(self, client):
        r = _upload(client, content=b"")
        assert r.status_code == 422
        body = r.json()
        assert body["error"] == "corpus_empty"
        assert body["detail"]

    def test_upload_missing_persona_id_422(self, client):
        # No persona_id → FastAPI's own validation kicks in.
        r = client.post(
            "/api/corpus/upload",
            files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
            data={"kind": "text"},
        )
        assert r.status_code == 422

    def test_upload_extension_pinned_per_kind(self, client):
        """.csv is allowed for conversation kind but not for text kind.

        Updated after slice 2B review #3: ALLOWED_EXTENSIONS_BY_KIND was
        tightened so upload only accepts what ingestion can process.
        .json/.zip/.srt/.vtt land when their respective extractors do
        (slice 2C/2D).
        """
        ok = _upload(
            client,
            kind="conversation",
            filename="export.csv",
            content=b"time,sender,message\n2024,me,hi\n",
        )
        assert ok.status_code == 200, ok.text

        bad = _upload(
            client,
            kind="text",
            filename="export.csv",
            content=b"a,b\n1,2\n",
        )
        assert bad.status_code == 415

    def test_upload_optional_provenance_fields_preserved(self, client):
        r = _upload(
            client,
            extra_form={
                "source": "私房書 ch.3",
                "source_date": "2014-09-01T00:00:00Z",
                "listener_tag": "child",
                "notes": "from EPUB rip",
            },
        )
        assert r.status_code == 200, r.text
        item_id = r.json()["item_id"]

        got = client.get(f"/api/corpus/{PERSONA_ID}/items/{item_id}")
        assert got.status_code == 200
        body = got.json()
        assert body["source"] == "私房書 ch.3"
        assert body["listener_tag"] == "child"
        assert body["notes"] == "from EPUB rip"
        assert body["source_date"].startswith("2014-09-01")


# ---------------------------------------------------------------------------
# List + get + delete
# ---------------------------------------------------------------------------
class TestCorpusListGetDelete:
    def test_list_empty_persona(self, client, assert_matches_schema):
        r = client.get(f"/api/corpus/{PERSONA_ID}")
        assert r.status_code == 200
        parsed = assert_matches_schema(ListResponse, r.json())
        assert parsed.count == 0
        assert parsed.items == []
        assert parsed.persona_id == PERSONA_ID

    def test_list_after_uploads(self, client, assert_matches_schema):
        for i in range(3):
            r = _upload(client, content=f"content {i}".encode(), filename=f"n{i}.txt")
            assert r.status_code == 200

        r = client.get(f"/api/corpus/{PERSONA_ID}")
        assert r.status_code == 200
        parsed = assert_matches_schema(ListResponse, r.json())
        assert parsed.count == 3
        assert len(parsed.items) == 3
        for item in parsed.items:
            assert isinstance(item, CorpusItem)
            assert item.persona_id == PERSONA_ID

    def test_get_404_for_missing_item(self, client):
        r = client.get(f"/api/corpus/{PERSONA_ID}/items/no-such-id")
        assert r.status_code == 404
        body = r.json()
        assert body["error"] == "corpus_item_not_found"
        assert body["details"]["item_id"] == "no-such-id"

    def test_delete_removes_item(self, client):
        up = _upload(client)
        item_id = up.json()["item_id"]

        d = client.delete(f"/api/corpus/{PERSONA_ID}/items/{item_id}")
        assert d.status_code == 200
        assert d.json() == {"status": "deleted", "item_id": item_id}

        # Now 404 — gone for good.
        g = client.get(f"/api/corpus/{PERSONA_ID}/items/{item_id}")
        assert g.status_code == 404

    def test_delete_404_for_missing_item(self, client):
        r = client.delete(f"/api/corpus/{PERSONA_ID}/items/no-such-id")
        assert r.status_code == 404
        assert r.json()["error"] == "corpus_item_not_found"


# ---------------------------------------------------------------------------
# Manifest — rolled-up shape + threshold flags
# ---------------------------------------------------------------------------
class TestCorpusManifest:
    def test_manifest_empty(self, client, assert_matches_schema):
        r = client.get(f"/api/corpus/{PERSONA_ID}/manifest")
        assert r.status_code == 200
        parsed = assert_matches_schema(CorpusManifest, r.json())
        assert parsed.persona_id == PERSONA_ID
        assert parsed.total_items == 0
        assert parsed.total_bytes == 0
        assert parsed.extracted_chars == 0
        assert parsed.by_kind == {"text": 0, "transcript": 0, "conversation": 0}
        assert parsed.thresholds.ready_for_rag is False
        assert parsed.thresholds.ready_for_lora_synthetic is False
        assert parsed.thresholds.ready_for_lora_organic is False

    def test_manifest_counts_by_kind(self, client, assert_matches_schema):
        _upload(client, kind="text", filename="a.txt", content=b"a")
        _upload(client, kind="text", filename="b.md", content=b"bb")
        _upload(client, kind="transcript", filename="c.txt", content=b"ccc")
        _upload(
            client, kind="conversation", filename="d.csv",
            content=b"time,sender,message\n2024,me,hi here\n",
        )

        r = client.get(f"/api/corpus/{PERSONA_ID}/manifest")
        parsed = assert_matches_schema(CorpusManifest, r.json())
        assert parsed.total_items == 4
        assert parsed.by_kind["text"] == 2
        assert parsed.by_kind["transcript"] == 1
        assert parsed.by_kind["conversation"] == 1
        # 1 + 2 + 3 + len(CSV body) bytes
        assert parsed.total_bytes == 1 + 2 + 3 + len(
            b"time,sender,message\n2024,me,hi here\n"
        )

    def test_manifest_no_extra_fields_drift(self, client):
        """Body keys are frozen — if a future change adds a field, the
        CorpusManifest model must list it (extra='forbid' enforces this
        through assert_matches_schema's parse step)."""
        r = client.get(f"/api/corpus/{PERSONA_ID}/manifest")
        body = r.json()
        expected_top = {
            "persona_id",
            "total_items",
            "by_kind",
            "total_bytes",
            "extracted_chars",
            "thresholds",
            "updated_at",
        }
        assert set(body.keys()) == expected_top
        expected_thresholds = {
            "ready_for_rag",
            "ready_for_lora_synthetic",
            "ready_for_lora_organic",
        }
        assert set(body["thresholds"].keys()) == expected_thresholds
