"""
Unit tests for EmotionParser - streaming emotion and content parser.

Format: [E:情緒]內容
- [E: marks start of emotion tag
- ] is clear delimiter after emotion value
- Content follows after ]
- Emotion must be one of: 幽默, 寵溺, 撒嬌, 毒舌, 溫和, 調皮, 開心, 感動, 生氣, 認真
"""
import pytest
from app.services.tts.emotion_mapper import EmotionParser, DEFAULT_EMOTION


def drain_parser(parser, text, chunk_size=1):
    """Helper to drain a parser with given text in chunks."""
    results = []
    pos = 0
    while pos < len(text):
        chunk = text[pos:pos+chunk_size]
        result = parser.update(chunk)
        if result is not None:
            results.append(result)
        pos += chunk_size
    # Keep calling update('') until None
    while True:
        result = parser.update('')
        if result is None:
            break
        results.append(result)
    return results


class TestNewFormatBasic:
    """Minimal tests for [E:情緒]內容 format."""

    def test_full_format_one_chunk(self):
        """Full format in one chunk - emits one char at a time for streaming."""
        parser = EmotionParser()
        result = parser.update('[E:調皮]哈哈哈')
        # First emit has emotion
        assert result == ('調皮', '哈')
        # Subsequent emits via update('')
        result = parser.update('')
        assert result == (None, '哈')
        result = parser.update('')
        assert result == (None, '哈')
        result = parser.update('')
        assert result is None

    def test_full_format_streaming(self):
        """Full format character by character."""
        parser = EmotionParser()
        results = drain_parser(parser, '[E:調皮]哈哈哈', 1)
        # 4 chars: 1st with emotion, remaining 3 without
        assert len(results) == 3
        assert results[0] == ('調皮', '哈')
        assert results[1] == (None, '哈')
        assert results[2] == (None, '哈')

    def test_emotion_locked_after_bracket(self):
        """Emotion locked when ] found, before content."""
        parser = EmotionParser()
        parser.update('[E:調皮]')
        assert parser.current_emotion == '調皮'
        assert parser.is_emotion_locked is True
        # First content char triggers emit
        result = parser.update('哈')
        assert result == ('調皮', '哈')

    def test_pure_content_no_tag(self):
        """Pure content without [E: tag uses default emotion."""
        parser = EmotionParser()
        results = drain_parser(parser, '哈哈哈', 1)
        assert results[0][0] == DEFAULT_EMOTION
        assert results[0][1] == '哈'

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        parser = EmotionParser()
        parser.update('[E:調皮]哈')
        assert parser.current_emotion == '調皮'
        parser.reset()
        assert parser.current_emotion is None
        assert parser._state == EmotionParser.S_EMPTY
        assert parser._buffer == ''

    def test_is_ready_after_emotion(self):
        """is_ready True after emotion is locked."""
        parser = EmotionParser()
        assert parser.is_ready is False
        parser.update('[E:調皮]')
        assert parser.is_emotion_locked is True
        assert parser.is_ready is True


class TestNewFormatEdgeCases:
    """Edge cases for [E:情緒]內容 format."""

    def test_partial_tag_waits(self):
        """Partial [E: without closing ] waits for more."""
        parser = EmotionParser()
        result = parser.update('[E:')
        assert result is None
        assert parser.current_emotion is None

    def test_partial_bracket_after_e(self):
        """After E:emo, ] closes tag - wait for content to emit."""
        parser = EmotionParser()
        parser.update('[E:調皮')
        result = parser.update(']')
        assert result is None  # No content yet
        assert parser.current_emotion == '調皮'
        # First content char triggers emit
        result = parser.update('哈')
        assert result == ('調皮', '哈')

    def test_empty_content_after_tag(self):
        """[E:emo] with no content immediately after."""
        parser = EmotionParser()
        result = parser.update('[E:調皮]')
        assert result is None  # No content to emit yet
        result = parser.update('哈')
        assert result == ('調皮', '哈')

    def test_unknown_emotion_uses_default_instruct(self):
        """Unknown emotion uses default TTS instruct."""
        parser = EmotionParser()
        parser.update('[E:未知情緒]哈')
        assert parser.current_emotion == '未知情緒'
        assert parser.current_instruct == "(natural, conversational tone, warm and engaging)"

    def test_pure_content_single_e(self):
        """Single 'E' without [ is content."""
        parser = EmotionParser()
        result = parser.update('E')
        assert result[0] == DEFAULT_EMOTION

    def test_text_before_tag(self):
        """Text before [E: tag is emitted with default emotion."""
        parser = EmotionParser()
        results = drain_parser(parser, '好啊，[E:調皮]哈', 1)
        # '好啊，' emitted char by char with default emotion
        # First chars have default emotion
        assert results[0][0] == DEFAULT_EMOTION
        assert results[0][1] == '好'
        # After [E:調皮], emotion changes
        # Find when emotion becomes '調皮'
        emotion_changes = [r for r in results if r[0] == '調皮']
        assert len(emotion_changes) >= 1


class TestNewFormatStreaming:
    """Streaming edge cases for [E:情緒]內容 format."""

    def test_rapid_tokens(self):
        """Simulate rapid LLM token streaming."""
        parser = EmotionParser()
        tokens = ['[', 'E:', '調皮', ']', '哈', '哈', '哈']
        results = []
        for token in tokens:
            result = parser.update(token)
            if result is not None:
                results.append(result)
        assert len(results) >= 1
        assert results[0][0] == '調皮'

    def test_very_slow_streaming(self):
        """Simulate very slow character-by-character streaming."""
        parser = EmotionParser()
        results = drain_parser(parser, '[E:調皮]哈', 1)
        assert len(results) >= 1
        assert results[0][0] == '調皮'

    def test_multibyte_chars(self):
        """Multibyte Chinese characters work correctly."""
        parser = EmotionParser()
        result = parser.update('[E:寵溺]寶貝你好棒')
        assert result == ('寵溺', '寶')
