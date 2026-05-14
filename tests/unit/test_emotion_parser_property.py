"""
Property-based tests for the emotion parser using Hypothesis.

These complement the example-based tests in test_emotion_parser.py. The
goal: for ANY way an LLM splits the bytes of `[E:情緒]內容`, the parser
must produce the same final (emotion, full_content) once drained. This
catches streaming-fragility bugs that example tests miss.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from app.services.tts.emotion_mapper import (
    DEFAULT_EMOTION,
    EMOTION_TEXT_ENHANCEMENT,
    EmotionParser,
    enhance_text,
)


KNOWN_EMOTIONS = sorted(EMOTION_TEXT_ENHANCEMENT.keys() - {DEFAULT_EMOTION})

# A modest content alphabet — Han chars + ASCII letters + punctuation.
# Excludes `[` and `]` so generated content never collides with the tag
# delimiters. The parser handles in-content brackets correctly, but it
# muddies the property assertions.
content_chars = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Zs"),
        blacklist_characters="[]",
    ),
    min_size=1,
    max_size=40,
)

emotions = st.sampled_from(KNOWN_EMOTIONS)


def _split_into_chunks(s: str, chunks: list[int]) -> list[str]:
    """Cut `s` at the cumulative positions in `chunks` (modulo len)."""
    if not chunks or not s:
        return [s] if s else []
    cuts = sorted({c % len(s) for c in chunks if c > 0})
    cuts = [c for c in cuts if 0 < c < len(s)]
    parts: list[str] = []
    prev = 0
    for c in cuts:
        parts.append(s[prev:c])
        prev = c
    parts.append(s[prev:])
    return parts


def _drain(parser: EmotionParser, chunks: list[str]) -> tuple[str | None, str]:
    """Feed chunks one at a time, then drain. Returns (emotion, full_text)."""
    emitted_emotion: str | None = None
    emitted_text_parts: list[str] = []
    for ch in chunks:
        result = parser.update(ch)
        while result is not None:
            emo, txt = result
            if emo is not None and emitted_emotion is None:
                emitted_emotion = emo
            if txt:
                emitted_text_parts.append(txt)
            result = parser.update("")
    # Final drain in case there's still buffered content.
    for _ in range(64):
        result = parser.update("")
        if result is None:
            break
        emo, txt = result
        if emo is not None and emitted_emotion is None:
            emitted_emotion = emo
        if txt:
            emitted_text_parts.append(txt)
    return emitted_emotion, "".join(emitted_text_parts)


# ---------------------------------------------------------------------------
# Property 1: ANY chunking of a well-formed `[E:emo]content` input must
# produce (emo, content) after draining.
# ---------------------------------------------------------------------------
@given(
    emotion=emotions,
    content=content_chars,
    cuts=st.lists(st.integers(min_value=1, max_value=64), max_size=10),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_arbitrary_chunking_preserves_emotion_and_content(emotion, content, cuts):
    full_input = f"[E:{emotion}]{content}"
    chunks = _split_into_chunks(full_input, cuts)
    parser = EmotionParser()
    emitted_emotion, emitted_text = _drain(parser, chunks)
    assert emitted_emotion == emotion, (
        f"Expected emotion={emotion!r}, got {emitted_emotion!r} for chunks={chunks}"
    )
    assert emitted_text == content, (
        f"Expected content={content!r}, got {emitted_text!r} for chunks={chunks}"
    )


# ---------------------------------------------------------------------------
# Property 2: content-only input (no tag) — every byte must round-trip,
# with the default emotion locked.
# ---------------------------------------------------------------------------
@given(
    content=content_chars.filter(lambda s: not s.startswith("[")),
    cuts=st.lists(st.integers(min_value=1, max_value=64), max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_pure_content_uses_default_emotion(content, cuts):
    chunks = _split_into_chunks(content, cuts)
    parser = EmotionParser()
    emitted_emotion, emitted_text = _drain(parser, chunks)
    assert emitted_emotion == DEFAULT_EMOTION, (
        f"Expected default emotion {DEFAULT_EMOTION!r}, got {emitted_emotion!r}"
    )
    assert emitted_text == content


# ---------------------------------------------------------------------------
# Property 3: bounded drain — once the parser has been fully fed and
# drained, additional `update('')` calls must converge to None in O(1).
# This is the termination invariant the original code relied on by
# convention; the property test pins it.
# ---------------------------------------------------------------------------
@given(emotion=emotions, content=content_chars)
@settings(max_examples=50)
def test_drain_converges_after_completion(emotion, content):
    parser = EmotionParser()
    parser.update(f"[E:{emotion}]{content}")
    # Drain to completion.
    while parser.update("") is not None:
        pass
    # Further drains must return None immediately, not loop forever.
    for _ in range(16):
        assert parser.update("") is None


# ---------------------------------------------------------------------------
# Property 4: partial-tag inputs that get cut off must NEVER crash and
# must wait for more data (return None on the truncated chunks).
# ---------------------------------------------------------------------------
@given(
    emotion=emotions,
    truncate_at=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50)
def test_partial_tag_never_crashes(emotion, truncate_at):
    full_input = f"[E:{emotion}]"
    truncated = full_input[:truncate_at]
    assume(len(truncated) < len(full_input))
    parser = EmotionParser()
    # Feeding a partial tag should never raise.
    result = parser.update(truncated)
    # No content yet, so result is None (or could be (None, '') for some prefixes).
    if result is not None:
        emo, txt = result
        # If something IS emitted, it cannot be the locked emotion before ] is seen.
        assert txt == ""
    # Bounded drain still terminates.
    for _ in range(16):
        if parser.update("") is None:
            break
    else:
        pytest.fail("drain on partial input did not converge")


# ---------------------------------------------------------------------------
# Property 5: enhance_text never crashes on any (text, emotion) pair, even
# unknown emotions — they route to the default enhancer.
# ---------------------------------------------------------------------------
@given(
    text=st.text(max_size=80),
    emotion=st.text(max_size=10),
)
@settings(max_examples=100)
def test_enhance_text_handles_arbitrary_emotion(text, emotion):
    # Doesn't raise. Doesn't return None. Returns a string.
    result = enhance_text(text, emotion)
    assert isinstance(result, str)
