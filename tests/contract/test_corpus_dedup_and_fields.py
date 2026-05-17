"""
Contract tests for task 62B data-model additions:
- content_sha256 + upload-time dedup (review #18 of 6c9a87a)
- persona_speaker_alias (review #15 — Phase 3 LoRA needs persona-side filter)
- chunker_version stamped in chunks.jsonl (review #5 of 8161535 —
  prevents silent index drift when chunker is re-tuned)
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


PERSONA_ID = "xiao_s"


def _upload(client, *, content: bytes, kind: str = "text",
            filename: str = "note.txt", extra_form: dict | None = None):
    form = {"persona_id": PERSONA_ID, "kind": kind}
    if extra_form:
        form.update(extra_form)
    return client.post(
        "/api/corpus/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
        data=form,
    )


class TestContentSha256Dedup:
    def test_same_bytes_dedupes_to_one_item(self, client):
        body = b"identical content here\n" * 50
        r1 = _upload(client, content=body)
        r2 = _upload(client, content=body, filename="copy.txt")
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Same item_id returned both times — second upload was a no-op.
        assert r1.json()["item_id"] == r2.json()["item_id"]

        # Manifest counts only once.
        m = client.get(f"/api/corpus/{PERSONA_ID}/manifest")
        assert m.json()["total_items"] == 1

    def test_different_bytes_different_items(self, client):
        r1 = _upload(client, content=b"alpha bravo charlie\n")
        r2 = _upload(client, content=b"different content entirely\n")
        assert r1.json()["item_id"] != r2.json()["item_id"]

    def test_same_bytes_different_kinds_not_deduped(self, client):
        # A `.txt` uploaded as text vs the same bytes uploaded as
        # transcript are conceptually different — dedup is scoped by
        # (persona, kind, hash).
        body = b"may appear in both contexts\n"
        r1 = _upload(client, content=body, kind="text")
        r2 = _upload(client, content=body, kind="transcript")
        assert r1.json()["item_id"] != r2.json()["item_id"]

    def test_sha_persisted_on_item(self, client):
        r = _upload(client, content=b"check the hash field works\n")
        item_id = r.json()["item_id"]
        got = client.get(f"/api/corpus/{PERSONA_ID}/items/{item_id}").json()
        assert got["content_sha256"] is not None
        assert len(got["content_sha256"]) == 64  # sha256 hex
        # All lowercase hex.
        assert all(c in "0123456789abcdef" for c in got["content_sha256"])


class TestPersonaSpeakerAlias:
    def test_field_optional_default_none(self, client):
        r = _upload(client, content=b"plain text\n")
        item_id = r.json()["item_id"]
        body = client.get(f"/api/corpus/{PERSONA_ID}/items/{item_id}").json()
        assert body["persona_speaker_alias"] is None

    def test_field_persists_when_provided(self, client):
        r = _upload(
            client,
            content=b"conversation\n",
            kind="conversation",
            extra_form={"persona_speaker_alias": "媽"},
        )
        item_id = r.json()["item_id"]
        body = client.get(f"/api/corpus/{PERSONA_ID}/items/{item_id}").json()
        assert body["persona_speaker_alias"] == "媽"


class TestChunkerVersionStamp:
    def test_chunks_carry_version_field(self, client, isolated_data: Path):
        body = ("段落內容。" * 100).encode("utf-8")
        r = _upload(client, content=body)
        item_id = r.json()["item_id"]
        ing = client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")
        assert ing.status_code == 200

        chunks_path = (
            isolated_data / "personas" / PERSONA_ID / "corpus"
            / "text" / item_id / "chunks.jsonl"
        )
        records = [
            json.loads(line)
            for line in chunks_path.read_text("utf-8").splitlines() if line.strip()
        ]
        assert records, "expected at least one chunk"
        for rec in records:
            assert "chunker_version" in rec
            assert isinstance(rec["chunker_version"], int)
            assert rec["chunker_version"] >= 1

    def test_chunks_carry_persona_speaker_alias_when_set(
        self, client, isolated_data: Path,
    ):
        body = ("和小孩聊天的內容。" * 80).encode("utf-8")
        r = _upload(
            client,
            content=body,
            kind="conversation",
            filename="memo.txt",
            extra_form={"persona_speaker_alias": "媽"},
        )
        item_id = r.json()["item_id"]
        client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")

        chunks_path = (
            isolated_data / "personas" / PERSONA_ID / "corpus"
            / "conversation" / item_id / "chunks.jsonl"
        )
        records = [
            json.loads(line)
            for line in chunks_path.read_text("utf-8").splitlines() if line.strip()
        ]
        assert all(r["persona_speaker_alias"] == "媽" for r in records)
