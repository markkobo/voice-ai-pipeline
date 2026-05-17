"""
Tests for task 62C — on-disk schema posture + ingesting state machine.

- `extra="ignore"` on CorpusItem so forward-written metadata.json
  doesn't crash old binaries
- API response models keep `extra="forbid"` (covered in existing
  test_corpus_contract.py via assert_matches_schema)
- ingest() flips to `ingesting` before running, and sweep_stranded()
  resets stranded items.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.services.corpus import (
    CorpusItemKind,
    CorpusItemStatus,
    IngestionService,
    JsonCorpusRepository,
)
from app.services.corpus.models import CorpusItem


class TestSchemaForwardCompat:
    def test_extra_field_ignored_not_rejected(self):
        """A future slice adds `extraction_method`. Old binary must
        load the JSON without crashing — `extra="ignore"`."""
        raw = {
            "item_id": "00000000-0000-4000-8000-000000000001",
            "persona_id": "xiao_s",
            "kind": "text",
            "filename": "x.txt",
            "size_bytes": 100,
            "status": "uploaded",
            "created_at": "2026-05-17T00:00:00Z",
            "updated_at": "2026-05-17T00:00:00Z",
            # FUTURE FIELD — old binary must tolerate it.
            "extraction_method": "tesseract",
            "another_future_field": 42,
        }
        item = CorpusItem.model_validate(raw)
        # The unknown fields are silently dropped (extra="ignore"
        # default); known fields parse correctly.
        assert item.item_id == raw["item_id"]
        assert item.kind == CorpusItemKind.text


class TestIngestingState:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        repo = JsonCorpusRepository(tmp_path / "personas")
        svc = IngestionService(repository=repo)
        # Create an item on disk.
        item = CorpusItem(
            item_id="00000000-0000-4000-8000-000000000010",
            persona_id="xiao_s",
            kind=CorpusItemKind.text,
            filename="original.txt",
            size_bytes=5,
            status=CorpusItemStatus.uploaded,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        item_dir = repo.item_dir_for_kind("xiao_s", "text", item.item_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "original.txt").write_bytes(b"hello content body\n" * 50)
        repo.save(item)
        return repo, svc, item

    def test_ingest_flips_to_ingested_on_success(self, setup):
        repo, svc, item = setup
        result = svc.ingest("xiao_s", item.item_id)
        # Final state is `ingested`, not `ingesting`.
        assert result.status == CorpusItemStatus.ingested

    def test_sweep_resets_stranded_ingesting_to_failed(self, setup):
        repo, svc, item = setup
        # Simulate a crashed ingest: item is stuck in `ingesting`.
        def _strand(it):
            it.status = CorpusItemStatus.ingesting
        repo.update("xiao_s", item.item_id, _strand)
        assert repo.get("xiao_s", item.item_id).status == CorpusItemStatus.ingesting

        count = svc.sweep_stranded("xiao_s")
        assert count == 1

        recovered = repo.get("xiao_s", item.item_id)
        assert recovered.status == CorpusItemStatus.failed
        assert "interrupted" in (recovered.error or "")

    def test_sweep_leaves_other_statuses_alone(self, setup):
        repo, svc, item = setup
        # Item is in `uploaded` — sweep should NOT touch it.
        count = svc.sweep_stranded("xiao_s")
        assert count == 0
        assert repo.get("xiao_s", item.item_id).status == CorpusItemStatus.uploaded
