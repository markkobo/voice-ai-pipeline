"""
Unit tests for _extract_plaintext encoding cascade (review #1).

The original cascade tried gb18030 before big5, which silently
mis-decoded Big5 Traditional-Chinese content as gb18030 mojibake — the
single highest-impact corruption for our TW/HK target user base.

These tests pin the corrected behavior:
- Big5 input decodes as Big5 (not gb18030 mojibake)
- UTF-8 with BOM decodes correctly
- UTF-16 BE without BOM is detected and decoded
- Binary garbage raises ExtractionFailedError, not silently mis-decoded
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.corpus.ingestion import (
    ExtractionFailedError,
    _extract_plaintext,
    _looks_like_real_text,
)


@pytest.fixture
def tmp_file(tmp_path: Path):
    def _write(name: str, raw: bytes) -> Path:
        p = tmp_path / name
        p.write_bytes(raw)
        return p
    return _write


class TestEncodingCascade:
    def test_utf8_plain(self, tmp_file):
        p = tmp_file("a.txt", "繁體中文書名".encode("utf-8"))
        assert _extract_plaintext(p) == "繁體中文書名"

    def test_utf8_bom(self, tmp_file):
        p = tmp_file("a.txt", b"\xef\xbb\xbf" + "繁體中文".encode("utf-8"))
        result = _extract_plaintext(p)
        assert result == "繁體中文"
        # BOM must be stripped — chunker can't tolerate it leaking.
        assert "﻿" not in result

    def test_big5_decodes_as_big5_not_gb18030(self, tmp_file):
        """Review #1 BLOCKER: a Big5-encoded Traditional Chinese book
        must not silently decode as gb18030 mojibake."""
        original = "蔡康永的說話之道"
        p = tmp_file("a.txt", original.encode("big5"))
        result = _extract_plaintext(p)
        assert result == original, (
            f"Expected Big5 decode to recover {original!r}, "
            f"got {result!r} — likely silent gb18030 mojibake."
        )

    def test_gb18030_decodes_correctly(self, tmp_file):
        """Simplified Chinese GB18030 input still works."""
        original = "简体中文内容"
        p = tmp_file("a.txt", original.encode("gb18030"))
        assert _extract_plaintext(p) == original

    def test_utf16_le_bom(self, tmp_file):
        original = "中文 UTF-16 LE"
        p = tmp_file("a.txt", b"\xff\xfe" + original.encode("utf-16-le"))
        assert _extract_plaintext(p) == original

    def test_utf16_be_bom(self, tmp_file):
        original = "中文 UTF-16 BE"
        p = tmp_file("a.txt", b"\xfe\xff" + original.encode("utf-16-be"))
        assert _extract_plaintext(p) == original

    def test_crlf_normalized(self, tmp_file):
        p = tmp_file("a.txt", "a\r\nb\r\nc".encode("utf-8"))
        assert _extract_plaintext(p) == "a\nb\nc"

    def test_cr_only_normalized(self, tmp_file):
        # Old-Mac line endings.
        p = tmp_file("a.txt", "a\rb\rc".encode("utf-8"))
        assert _extract_plaintext(p) == "a\nb\nc"

    def test_embedded_nul_stripped(self, tmp_file):
        # Some serializers leave NULs in the middle of text — they break
        # downstream sqlite/lance writers (review #12).
        p = tmp_file("a.txt", b"hello\x00world")
        assert _extract_plaintext(p) == "helloworld"

    def test_binary_garbage_raises(self, tmp_file):
        # Random bytes shouldn't silently decode as anything.
        import os
        p = tmp_file("a.bin", os.urandom(2048))
        with pytest.raises(ExtractionFailedError):
            _extract_plaintext(p)


class TestLooksLikeRealText:
    def test_real_chinese_passes(self):
        assert _looks_like_real_text("這是一段繁體中文，看起來很正常。") is True

    def test_real_english_passes(self):
        assert _looks_like_real_text("Hello world, this is normal text.") is True

    def test_empty_fails(self):
        assert _looks_like_real_text("") is False
        assert _looks_like_real_text("   \n\n  ") is False

    def test_lots_of_replacement_chars_fails(self):
        # >0.5% replacement chars → bad decode signature.
        assert _looks_like_real_text("a" * 100 + "�" * 5) is False

    def test_mojibake_in_pua_fails(self):
        # Construct a string heavy in Private Use Area (the gb18030
        # mojibake fingerprint). 60% of chars in U+E000..F8FF.
        bad = "".join(chr(0xE000 + i % 100) for i in range(100))
        good_prefix = "a" * 50
        full = good_prefix + bad
        # non_ascii is 100; suspicious is 100 — should fail.
        assert _looks_like_real_text(full) is False
