"""
Unit tests for the Extractor Protocol (RFC_M6 Phase 0 task 62A).

The protocol is what slice 2C (PDF/EPUB/DOCX) and 2D (audio/video)
will plug into. These tests pin its surface so the next round of
extractors doesn't drift.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.corpus import (
    ConversationExtractor,
    Extractor,
    ExtractResult,
    PlaintextExtractor,
    default_extractors,
)
from app.services.corpus.models import CorpusItemKind


class TestExtractResultShape:
    def test_minimal_result(self):
        r = ExtractResult(text="hi")
        assert r.text == "hi"
        assert r.warnings == []
        assert r.metadata == {}

    def test_full_result(self):
        r = ExtractResult(
            text="hi",
            warnings=["decoded as gb18030"],
            metadata={"format": "plaintext"},
        )
        assert r.warnings == ["decoded as gb18030"]
        assert r.metadata["format"] == "plaintext"


class TestPlaintextExtractor:
    def test_satisfies_protocol(self):
        e = PlaintextExtractor()
        assert isinstance(e, Extractor)

    def test_supports_text_kind(self):
        e = PlaintextExtractor()
        assert e.supports(CorpusItemKind.text, ".txt") is True
        assert e.supports(CorpusItemKind.text, ".md") is True
        assert e.supports(CorpusItemKind.text, ".pdf") is False

    def test_supports_transcript_kind(self):
        e = PlaintextExtractor()
        assert e.supports(CorpusItemKind.transcript, ".txt") is True

    def test_does_not_support_conversation(self):
        # Conversation .txt goes to ConversationExtractor, not Plaintext.
        e = PlaintextExtractor()
        assert e.supports(CorpusItemKind.conversation, ".txt") is False

    def test_extract_returns_result(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        p.write_text("hello\nworld", encoding="utf-8")
        r = PlaintextExtractor().extract(p)
        assert r.text == "hello\nworld"
        assert r.metadata.get("format") == "plaintext"

    def test_is_async_false(self):
        assert PlaintextExtractor().is_async is False

    def test_needs_gpu_false(self):
        assert PlaintextExtractor().needs_gpu is False


class TestConversationExtractor:
    def test_satisfies_protocol(self):
        assert isinstance(ConversationExtractor(), Extractor)

    def test_supports_conversation_txt_csv_json(self):
        e = ConversationExtractor()
        assert e.supports(CorpusItemKind.conversation, ".txt") is True
        assert e.supports(CorpusItemKind.conversation, ".csv") is True
        assert e.supports(CorpusItemKind.conversation, ".json") is True

    def test_does_not_support_text_kind(self):
        e = ConversationExtractor()
        assert e.supports(CorpusItemKind.text, ".txt") is False

    def test_extract_whatsapp(self, tmp_path: Path):
        body = "12/03/24, 14:32 - John: Hello\n"
        p = tmp_path / "chat.txt"
        p.write_bytes(body.encode("utf-8"))
        r = ConversationExtractor().extract(p)
        assert "John: Hello" in r.text
        assert r.metadata["format"] == "whatsapp"
        assert r.metadata["message_count"] == 1

    def test_extract_wechat_csv(self, tmp_path: Path):
        body = (
            "StrTime,IsSender,Message\n"
            "2024-03-05 08:30,0,早安\n"
        )
        p = tmp_path / "wc.csv"
        p.write_bytes(body.encode("utf-8"))
        r = ConversationExtractor().extract(p)
        assert "early" not in r.text.lower()  # sanity
        assert "早安" in r.text
        assert r.metadata["format"] == "wechat_csv"

    def test_freeform_text_metadata(self, tmp_path: Path):
        body = "Just some freeform memo about Bob.\n" * 20
        p = tmp_path / "memo.txt"
        p.write_bytes(body.encode("utf-8"))
        r = ConversationExtractor().extract(p)
        assert r.metadata["format"] == "freeform_text"


class TestDefaultRegistry:
    def test_default_extractors_returns_both(self):
        es = default_extractors()
        assert len(es) >= 2
        # Order matters — PlaintextExtractor first so it wins for
        # kind=text. Both implement the Protocol.
        for e in es:
            assert isinstance(e, Extractor)

    def test_no_overlap_between_extractors(self):
        """A given (kind, ext) shouldn't have multiple extractors saying
        yes — would make dispatch non-deterministic."""
        es = default_extractors()
        for kind in CorpusItemKind:
            for ext in (".txt", ".md", ".csv", ".json"):
                supporters = [e for e in es if e.supports(kind, ext)]
                assert len(supporters) <= 1, (
                    f"Multiple extractors claim (kind={kind.value}, ext={ext}): "
                    f"{[type(e).__name__ for e in supporters]}"
                )


class TestIngestionServiceWithCustomExtractors:
    """The ingestion service accepts a custom extractor list. Use this
    in slice 2C tests to inject a fake PDF extractor without modifying
    the default registry."""

    def test_custom_registry_takes_precedence(self, tmp_path: Path):
        from app.services.corpus import IngestionService, JsonCorpusRepository
        from app.services.corpus.models import CorpusItem, CorpusItemStatus
        from datetime import datetime, timezone

        repo = JsonCorpusRepository(tmp_path / "personas")
        # Create a real item on disk so we can call ingest() against it.
        item = CorpusItem(
            item_id="00000000-0000-4000-8000-000000000001",
            persona_id="xiao_s",
            kind=CorpusItemKind.text,
            filename="original.fake",
            size_bytes=5,
            status=CorpusItemStatus.uploaded,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        item_dir = repo.item_dir_for_kind("xiao_s", "text", item.item_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "original.fake").write_bytes(b"hello")
        repo.save(item)

        # Custom extractor for the bogus .fake extension.
        class FakeExtractor:
            is_async = False
            needs_gpu = False
            def supports(self, kind, ext):
                return kind == CorpusItemKind.text and ext == ".fake"
            def extract(self, path):
                return ExtractResult(text="REPLACED", metadata={"format": "fake"})

        svc = IngestionService(
            repository=repo,
            extractors=[FakeExtractor()],
        )
        result = svc.ingest("xiao_s", item.item_id)
        assert result.status.value == "ingested"
        assert result.extracted_chars == len("REPLACED")
