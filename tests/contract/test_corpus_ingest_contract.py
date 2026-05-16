"""
Contract tests for POST /api/corpus/{persona_id}/items/{item_id}/ingest
(RFC_M6 Phase 0 slice 2A).

Covers .txt/.md happy path, error envelopes for unsupported formats and
empty extraction, idempotency, and that the manifest reflects post-ingest
counts.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from app.services.corpus import CorpusItem

pytestmark = pytest.mark.contract


PERSONA_ID = "xiao_s"


# ---------------------------------------------------------------------------
# Upload helper — same shape as test_corpus_contract.py
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
    form = {"persona_id": persona_id, "kind": kind}
    if extra_form:
        form.update(extra_form)
    return client.post(
        "/api/corpus/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
        data=form,
    )


# ---------------------------------------------------------------------------
# .txt / .md ingestion happy path
# ---------------------------------------------------------------------------
class TestIngestPlainText:
    def test_ingest_txt_marks_status_ingested(self, client, isolated_data, assert_matches_schema):
        # Big enough to produce multiple chunks (target_chars=600 with overlap=100).
        body = (
            "這是第一段。" * 60 + "\n\n"
            + "這是第二段。" * 60 + "\n\n"
            + "這是第三段。" * 60
        )
        up = _upload(client, content=body.encode("utf-8"), filename="story.txt")
        assert up.status_code == 200, up.text
        item_id = up.json()["item_id"]

        r = client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")
        assert r.status_code == 200, r.text
        parsed = assert_matches_schema(CorpusItem, r.json())
        assert parsed.status.value == "ingested"
        assert parsed.extracted_chars == len(body)
        assert parsed.chunk_count > 1
        assert parsed.error is None

    def test_ingest_writes_extracted_and_chunks_files(self, client, isolated_data: Path):
        body = "短測試。" * 50
        up = _upload(client, content=body.encode("utf-8"), filename="x.txt")
        item_id = up.json()["item_id"]
        client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")

        item_dir = isolated_data / "personas" / PERSONA_ID / "corpus" / "text" / item_id
        extracted = item_dir / "extracted.txt"
        chunks_path = item_dir / "chunks.jsonl"
        assert extracted.exists(), "extracted.txt not written"
        assert chunks_path.exists(), "chunks.jsonl not written"
        assert extracted.read_text("utf-8") == body

        # Each line is a chunk record.
        with open(chunks_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) >= 1
        for rec in records:
            assert rec["persona_id"] == PERSONA_ID
            assert rec["item_id"] == item_id
            assert rec["char_count"] == len(rec["text"])
            assert rec["kind"] == "text"

    def test_ingest_md_supported(self, client):
        body = b"# Header\n\nSome content here.\n\nMore content."
        up = _upload(client, content=body, filename="note.md")
        item_id = up.json()["item_id"]
        r = client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")
        assert r.status_code == 200
        assert r.json()["status"] == "ingested"
        assert r.json()["chunk_count"] >= 1

    def test_ingest_is_idempotent(self, client):
        body = "重複跑兩次。" * 100
        up = _upload(client, content=body.encode("utf-8"), filename="x.txt")
        item_id = up.json()["item_id"]
        r1 = client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")
        r2 = client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both runs report the same shape.
        assert r1.json()["extracted_chars"] == r2.json()["extracted_chars"]
        assert r1.json()["chunk_count"] == r2.json()["chunk_count"]

    def test_ingest_preserves_listener_tag_in_chunks(
        self, client, isolated_data: Path,
    ):
        body = "與小孩說話的測試內容。" * 40
        up = _upload(
            client,
            content=body.encode("utf-8"),
            filename="x.txt",
            extra_form={"listener_tag": "child"},
        )
        item_id = up.json()["item_id"]
        client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")

        chunks_path = (
            isolated_data / "personas" / PERSONA_ID / "corpus"
            / "text" / item_id / "chunks.jsonl"
        )
        records = [
            json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert all(r["listener_tag"] == "child" for r in records)


# ---------------------------------------------------------------------------
# Error envelopes
# ---------------------------------------------------------------------------
class TestIngestErrors:
    def test_ingest_unknown_item_404(self, client):
        r = client.post(f"/api/corpus/{PERSONA_ID}/items/no-such-id/ingest")
        assert r.status_code == 404
        body = r.json()
        assert body["error"] == "corpus_item_not_found"

    def test_ingest_unsupported_format_415(self, client):
        # Upload a .csv via the "conversation" kind (CSV is allowed for
        # that kind today). Ingestion for it isn't implemented yet, so
        # slice 2A should reject with 415.
        up = _upload(
            client,
            kind="conversation",
            content=b"sender,msg\nalice,hi\n",
            filename="chat.csv",
        )
        assert up.status_code == 200
        item_id = up.json()["item_id"]

        r = client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")
        assert r.status_code == 415, r.text
        body = r.json()
        assert body["error"] == "ingestion_unsupported_format"
        assert ".txt" in body["details"]["supported"]
        assert ".md" in body["details"]["supported"]


# ---------------------------------------------------------------------------
# Manifest reflects post-ingest state
# ---------------------------------------------------------------------------
class TestManifestAfterIngest:
    def test_manifest_extracted_chars_after_ingest(self, client):
        # Empty manifest first.
        r0 = client.get(f"/api/corpus/{PERSONA_ID}/manifest")
        assert r0.json()["extracted_chars"] == 0
        assert r0.json()["thresholds"]["ready_for_rag"] is False

        # Upload two items, ingest both.
        for i in range(2):
            up = _upload(
                client,
                content=("片段內容。" * 100).encode("utf-8"),
                filename=f"n{i}.txt",
            )
            client.post(
                f"/api/corpus/{PERSONA_ID}/items/{up.json()['item_id']}/ingest"
            )

        r = client.get(f"/api/corpus/{PERSONA_ID}/manifest")
        body = r.json()
        assert body["extracted_chars"] > 0
        # Two short items at ~500 chars each give a handful of chunks —
        # not enough for the RAG threshold (50). Stays False.
        assert body["thresholds"]["ready_for_rag"] is False
