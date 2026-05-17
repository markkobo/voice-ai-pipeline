"""
Chunking strategy for corpus text → RAG-ready chunks.

Slice 2A scope: sliding-window over normalized text with paragraph-aware
boundary preference. Chunk size ~600 Chinese chars, ~100-char overlap.
These numbers come from RFC_M6 §6 (Anthropic contextual retrieval
benchmark sweet spot for Chinese is 400-800 chars; we land in the middle
to leave room for an Anthropic-style context prefix added later).

Deliberately NOT implementing the Anthropic context-prefix-via-LLM step
here — that requires the OpenAI client + cost considerations that
belong in Phase 1 when the LLM pipeline is end-to-end wired. The
chunker outputs the raw chunks; a later pass can enrich them with
context blurbs.

The chunker is pure: in = string, out = list of (offset, text) tuples.
The IngestionService wraps offsets/text into CorpusChunk records.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Bump when the chunking algorithm changes in a way that invalidates
# previously-produced chunk indices (e.g. boundary preferences shift,
# target_chars default changes, runt-folding rules change).
# Task 62B / review #5 of 8161535 — downstream vector indexes can use
# this to detect "the chunks I have in LanceDB were built under v=1 but
# the corpus now reports v=2 → re-embed needed."
CHUNKER_VERSION = 1


# Sentence-boundary regex: Chinese full stops + English period/?/! + closing
# quotes. Matches the same pattern used in app/api/ws_asr.py SENTENCE_SPLIT_RE
# so chunks fall on TTS-friendly boundaries.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；.!?」』])")

# Paragraph-boundary regex: two or more newlines.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class ChunkSpan:
    """One produced chunk: position in the source text + the text itself."""
    char_offset: int       # start offset in the source
    char_count: int        # number of characters in `text`
    text: str

    @property
    def char_end(self) -> int:
        return self.char_offset + self.char_count


def chunk_text(
    text: str,
    *,
    target_chars: int = 600,
    overlap_chars: int = 100,
    min_chunk_chars: int = 50,
) -> list[ChunkSpan]:
    """
    Split `text` into overlapping chunks of ~target_chars each.

    Boundary preference order (each step finishes a chunk earlier than
    target_chars when a "nicer" boundary is found nearby):
      1. paragraph boundary (most preferred)
      2. sentence boundary
      3. raw target (hard cut)

    Args:
        text: Source string (already normalized — see ingestion.py for
            encoding/whitespace normalization).
        target_chars: Approximate chunk size in characters.
        overlap_chars: Characters of overlap between adjacent chunks.
            Set to 0 to disable overlap.
        min_chunk_chars: Trailing fragments below this size are folded
            into the previous chunk rather than emitted as their own.

    Returns:
        List of ChunkSpan in offset order.
    """
    if not text:
        return []

    spans: list[ChunkSpan] = []
    n = len(text)
    pos = 0

    while pos < n:
        # Hard target end position.
        target_end = min(pos + target_chars, n)

        if target_end == n:
            # Last chunk — just take everything to the end.
            chunk_text_ = text[pos:n]
            if spans and len(chunk_text_) < min_chunk_chars:
                # Fold tail into the previous chunk to avoid runt chunks.
                prev = spans[-1]
                merged = text[prev.char_offset:n]
                spans[-1] = ChunkSpan(
                    char_offset=prev.char_offset,
                    char_count=len(merged),
                    text=merged,
                )
            else:
                spans.append(ChunkSpan(pos, len(chunk_text_), chunk_text_))
            break

        # Look for a paragraph break within the soft window
        # [target_end - 200, target_end + 200] — prefer breaking there.
        window_lo = max(pos, target_end - 200)
        window_hi = min(n, target_end + 200)
        end = _find_best_break(text, window_lo, window_hi, target_end)

        chunk_text_ = text[pos:end]
        spans.append(ChunkSpan(pos, len(chunk_text_), chunk_text_))

        # Advance with overlap. Never move backward; pin to at least one
        # char of forward progress so we can't loop on pathological input.
        next_pos = max(end - overlap_chars, pos + 1)
        if next_pos >= n:
            break
        pos = next_pos

    return spans


def _find_best_break(
    text: str, lo: int, hi: int, target: int
) -> int:
    """
    Pick the best end position in [lo, hi] near `target`.

    Preference: paragraph break > sentence break > raw target.
    Returns the chosen end offset (exclusive).
    """
    # Paragraph break: search for "\n\n" pattern in window.
    window = text[lo:hi]
    para_matches = list(_PARAGRAPH_SPLIT_RE.finditer(window))
    if para_matches:
        # Pick the paragraph break closest to target.
        best = min(
            para_matches,
            key=lambda m: abs((lo + m.end()) - target),
        )
        return lo + best.end()

    # Sentence break: split on sentence-terminator chars.
    sentence_matches = list(_SENTENCE_SPLIT_RE.finditer(window))
    if sentence_matches:
        best = min(
            sentence_matches,
            key=lambda m: abs((lo + m.end()) - target),
        )
        return lo + best.end()

    # Fall back to hard target.
    return target
