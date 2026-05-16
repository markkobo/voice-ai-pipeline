"""
Unit tests for corpus chunker — pure function, no IO.

Per CLAUDE.md "Development Philosophy": parsers/streaming code get tests
first. Chunker is a state machine over text spans and benefits from
direct unit coverage in addition to the contract tests in
test_corpus_ingest_contract.py.
"""
from __future__ import annotations

import pytest

from app.services.corpus.chunker import ChunkSpan, chunk_text


class TestChunkTextBasics:
    def test_empty_input_returns_empty_list(self):
        assert chunk_text("") == []

    def test_short_input_returns_single_chunk(self):
        text = "短輸入測試。"
        spans = chunk_text(text, target_chars=600)
        assert len(spans) == 1
        assert spans[0].char_offset == 0
        assert spans[0].char_count == len(text)
        assert spans[0].text == text

    def test_offsets_cover_source(self):
        text = "段落一。" * 200 + "\n\n" + "段落二。" * 200
        spans = chunk_text(text, target_chars=600, overlap_chars=100)
        assert len(spans) > 1
        # First chunk starts at 0.
        assert spans[0].char_offset == 0
        # Last chunk ends at len(text).
        assert spans[-1].char_end == len(text)
        # Char-counts are correct.
        for s in spans:
            assert s.text == text[s.char_offset:s.char_end]


class TestChunkTextBoundaries:
    def test_prefers_paragraph_break(self):
        """When a paragraph break exists near target, the chunk ends there."""
        text = ("aaaaa" * 100) + "\n\n" + ("bbbbb" * 100)
        spans = chunk_text(text, target_chars=500, overlap_chars=0)
        # First chunk should end at the paragraph break (~char 500).
        assert spans[0].text.endswith("\n\n") or spans[0].text.endswith("aaaaa\n\n"), \
            f"Expected paragraph-aligned end, got tail={spans[0].text[-20:]!r}"

    def test_prefers_sentence_break_when_no_paragraph(self):
        """When no paragraph break, chunk ends on Chinese full-stop."""
        text = "句子一。" * 200  # all sentence-terminated, no \n\n
        spans = chunk_text(text, target_chars=500, overlap_chars=0)
        for s in spans[:-1]:
            assert s.text.endswith("。"), f"Expected sentence end, got {s.text[-5:]!r}"

    def test_hard_cut_when_no_friendly_boundary(self):
        """Solid string of chars with no breaks falls back to hard cut."""
        text = "a" * 2000
        spans = chunk_text(text, target_chars=500, overlap_chars=100)
        # Should produce ~5 chunks, each close to 500 chars.
        assert len(spans) >= 3
        for s in spans:
            # Tolerant: hard cut might land within target ± a few chars.
            assert 400 <= s.char_count <= 700


class TestChunkTextOverlap:
    def test_overlap_produces_repeated_content(self):
        text = "a" * 2000
        spans = chunk_text(text, target_chars=500, overlap_chars=100)
        # Adjacent chunks overlap by ~100 chars.
        for i in range(len(spans) - 1):
            assert spans[i + 1].char_offset < spans[i].char_end

    def test_zero_overlap_is_clean_split(self):
        text = "a" * 2000
        spans = chunk_text(text, target_chars=500, overlap_chars=0)
        for i in range(len(spans) - 1):
            assert spans[i + 1].char_offset == spans[i].char_end


class TestChunkTextRuntFolding:
    def test_trailing_short_fragment_folds_into_previous(self):
        text = "段落內容。" * 100 + "短"  # tiny tail
        spans = chunk_text(text, target_chars=600, overlap_chars=0, min_chunk_chars=100)
        # The trailing "短" should NOT be its own chunk.
        assert spans[-1].char_count >= 100 or len(spans) == 1


class TestChunkTextDoesNotInfiniteLoop:
    def test_pathological_short_target_terminates(self):
        """Pathological input: target=1, overlap=1 — must terminate."""
        text = "abcde" * 20
        # If the chunker can't make forward progress, this hangs.
        spans = chunk_text(text, target_chars=1, overlap_chars=1)
        assert len(spans) >= 1
        # And the span text doesn't double up infinitely.
        assert sum(s.char_count for s in spans) >= len(text)
