"""
Regression tests for the BLOCKER + MAJOR fixes from the staff review of
c7ee1f4 (62B+62C).

Each test ties back to a numbered review comment for traceability.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.corpus import CorpusItemStatus

pytestmark = pytest.mark.contract


PERSONA_ID = "xiao_s"


def _upload(client, *, content: bytes, kind: str = "text",
            filename: str = "note.txt"):
    return client.post(
        "/api/corpus/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
        data={"persona_id": PERSONA_ID, "kind": kind},
    )


class TestDedupSkipsFailedItems:
    """Review BLOCKER #1: re-uploading after a failed ingest used to
    return the same broken item_id. Now must return a NEW item."""

    def test_failed_item_not_returned_by_dedup(
        self, client, isolated_data: Path,
    ):
        body = b"some content that will be force-failed\n" * 50
        r1 = _upload(client, content=body)
        item_id_1 = r1.json()["item_id"]

        # Manually flip item to status=failed (simulating a failed
        # ingest attempt).
        from app.api._dependencies import _get_or_create_corpus_service
        # Have to go through the app's state to get the test service.
        # Easier: write the metadata.json directly.
        import json
        meta_path = (
            isolated_data / "personas" / PERSONA_ID / "corpus"
            / "text" / item_id_1 / "metadata.json"
        )
        meta = json.loads(meta_path.read_text("utf-8"))
        meta["status"] = "failed"
        meta["error"] = "fake failure for test"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        # Re-upload the same bytes.
        r2 = _upload(client, content=body)
        item_id_2 = r2.json()["item_id"]

        # MUST be a different item — the user is retrying.
        assert item_id_1 != item_id_2

    def test_non_failed_dedup_still_works(self, client):
        # Sanity: the dedup we added in 62B still fires for normal items.
        body = b"normal content\n" * 30
        r1 = _upload(client, content=body)
        r2 = _upload(client, content=body)
        assert r1.json()["item_id"] == r2.json()["item_id"]


class TestSweepHandlesConcurrentDelete:
    """Review BLOCKER #2: sweep_stranded crashed on CorpusItemNotFound
    when a concurrent delete removed an item mid-iteration."""

    def test_sweep_tolerates_item_disappearing(self, isolated_data: Path):
        from app.services.corpus import (
            CorpusItemKind,
            IngestionService,
            JsonCorpusRepository,
        )
        from app.services.corpus.models import CorpusItem
        from app.services.corpus.repository import CorpusItemNotFound
        from datetime import datetime, timezone

        repo = JsonCorpusRepository(isolated_data / "personas")
        svc = IngestionService(repository=repo)

        # Plant two stranded items.
        for i in range(2):
            item = CorpusItem(
                item_id=f"00000000-0000-4000-8000-00000000010{i}",
                persona_id=PERSONA_ID,
                kind=CorpusItemKind.text,
                filename="x.txt",
                size_bytes=5,
                status=CorpusItemStatus.ingesting,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            repo.save(item)

        # Monkey-patch update to raise CorpusItemNotFound on first call
        # (simulating concurrent delete) — sweep MUST keep going.
        real_update = repo.update
        calls = {"n": 0}
        def flaky_update(persona_id, item_id, mutator):
            calls["n"] += 1
            if calls["n"] == 1:
                raise CorpusItemNotFound(item_id)
            return real_update(persona_id, item_id, mutator)
        with patch.object(repo, "update", side_effect=flaky_update):
            # Should not raise — just skip the missing item.
            count = svc.sweep_stranded(PERSONA_ID)

        # Two stranded items, first raised, second succeeded → count=1.
        assert count == 1


class TestReingestConcurrencyGuard:
    """Review MAJOR #9: a second /ingest call landing on an item
    already in status=ingesting must 409, not silently re-run."""

    def test_double_ingest_returns_409(self, client, isolated_data: Path):
        # Upload + manually flip to ingesting (simulating an in-flight
        # ingest from another request).
        body = b"some text\n" * 30
        item_id = _upload(client, content=body).json()["item_id"]
        import json
        meta_path = (
            isolated_data / "personas" / PERSONA_ID / "corpus"
            / "text" / item_id / "metadata.json"
        )
        meta = json.loads(meta_path.read_text("utf-8"))
        meta["status"] = "ingesting"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        # /ingest on an already-ingesting item → 409.
        r = client.post(f"/api/corpus/{PERSONA_ID}/items/{item_id}/ingest")
        assert r.status_code == 409, r.text
        assert r.json()["error"] == "ingestion_in_progress"


class TestForwardEnumLimitationDocumented:
    """Review MAJOR #7: extra='ignore' doesn't help with new enum values.
    Verify the documented behavior — items with an unknown status are
    silently skipped by list(), not crashing it."""

    def test_unknown_status_item_skipped_not_crashing_list(
        self, client, isolated_data: Path,
    ):
        # First upload a normal item so the index exists.
        good = _upload(client, content=b"ok\n").json()["item_id"]

        # Now plant a forward-written item with status=needs_review
        # (hypothetical future enum value).
        import json
        from datetime import datetime, timezone
        bad_id = "00000000-0000-4000-8000-000000000999"
        bad_dir = (
            isolated_data / "personas" / PERSONA_ID / "corpus" / "text" / bad_id
        )
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "original.txt").write_bytes(b"bytes")
        (bad_dir / "metadata.json").write_text(json.dumps({
            "item_id": bad_id,
            "persona_id": PERSONA_ID,
            "kind": "text",
            "filename": "x.txt",
            "size_bytes": 5,
            "status": "needs_review",  # unknown to current binary
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False))
        # Hand-edit the index to include the bad item.
        idx_path = (
            isolated_data / "personas" / PERSONA_ID / "corpus" / "index.json"
        )
        idx = json.loads(idx_path.read_text("utf-8"))
        idx[bad_id] = f"text/{bad_id}"
        idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2))

        # `list()` should NOT crash. It silently skips the bad item
        # and returns only the good one.
        r = client.get(f"/api/corpus/{PERSONA_ID}")
        assert r.status_code == 200
        ids = [it["item_id"] for it in r.json()["items"]]
        assert good in ids
        assert bad_id not in ids
