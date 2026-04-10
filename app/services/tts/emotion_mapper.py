"""
Emotion parser: parses emotion and content from streaming LLM output.

Format: [E:情緒]內容
- [E: marks start of emotion tag
- ] is clear delimiter after emotion value
- Content follows after ]
- Emotion must be one of: 幽默, 寵溺, 撒嬌, 毒舌, 溫和, 調皮, 開心, 感動, 生氣, 認真
- Falls back to pure content with default emotion if no tag found

Supports legacy tag format: [情感:調皮]哈哈哈... (fallback)
"""
from typing import Optional
import re


DEFAULT_EMOTION = "默認"

# Text enhancement rules for Path B (text prosody)
# Adds emotional markers to text based on emotion
def _make_enhancer(add_suffix: str, should_add: callable = None):
    """Create an enhancer that adds suffix if condition is met."""
    def enhancer(t: str) -> str:
        if not t:
            return t
        # Check if suffix should be added
        if should_add and should_add(t):
            return t
        return t + add_suffix
    return enhancer

def _enhance_sajiao(t: str) -> str:
    """撒嬌: Add ～, prolong sounds"""
    if not t:
        return t
    # Add ～ at end if not present
    if not t.endswith("～"):
        t = t + "～"
    # Prolong existing ～
    t = t.replace("～", "～～")
    return t

def _enhance_shengqi(t: str) -> str:
    """生氣: Add emphasis with ！"""
    if not t:
        return t
    # Replace final punctuation with ！
    if t.endswith(("。", "！", "～")):
        t = t[:-1] + "！！！"
    else:
        t = t + "！！！"
    return t

def _enhance_kaixin(t: str) -> str:
    """開心: Add ！"""
    if not t:
        return t
    if not t.endswith(("！", "～")):
        return t + "！"
    return t

def _enhance_wenhe(t: str) -> str:
    """溫和: Add ～ soft ending"""
    if not t:
        return t
    if not t.endswith(("～", "。")):
        return t + "～"
    return t

def _enhance_yuemo(t: str) -> str:
    """幽默: Add casual ending"""
    if not t:
        return t
    if not t.endswith(("～", "！")):
        return t + "～"
    return t

def _enhance_chongni(t: str) -> str:
    """寵溺: Add loving tone"""
    if not t:
        return t
    if not t.endswith("～"):
        return t + "～"
    return t

def _enhance_dushe(t: str) -> str:
    """毒舌: Make it sharp"""
    if not t:
        return t
    if "。" in t:
        t = t.replace("。", "！")
    elif not t.endswith("！"):
        t = t + "！"
    return t

def _enhance_tiao_pi(t: str) -> str:
    """調皮: Playful ending"""
    if not t:
        return t
    if not t.endswith(("～", "！")):
        return t + "～"
    return t

def _enhance_gandan(t: str) -> str:
    """感動: Emotional ellipsis"""
    if not t:
        return t
    if not t.endswith(("...", "。")):
        return t + "..."
    return t

def _enhance_renzhen(t: str) -> str:
    """認真: Keep as-is"""
    return t

EMOTION_TEXT_ENHANCEMENT: dict = {
    "撒嬌": _enhance_sajiao,
    "生氣": _enhance_shengqi,
    "開心": _enhance_kaixin,
    "溫和": _enhance_wenhe,
    "幽默": _enhance_yuemo,
    "寵溺": _enhance_chongni,
    "毒舌": _enhance_dushe,
    "調皮": _enhance_tiao_pi,
    "感動": _enhance_gandan,
    "認真": _enhance_renzhen,
    DEFAULT_EMOTION: _enhance_renzhen,  # No enhancement for default
}

EMOTION_TAG_PATTERN = re.compile(r'[\[［](?:情感|感情)[:：]\s*(.*?)[\]］]\s*')

# New format markers (TAG_START not used directly but kept for reference)
TAG_START = "[E:"


def get_tts_instruct(emotion: str, custom_map: Optional[dict] = None) -> str:
    """Get TTS instruction string for given emotion. DEPRECATED - returns None for Path B."""
    # Path B: We use text enhancement instead of instruct strings
    return None


def enhance_text(text: str, emotion: str) -> str:
    """
    Enhance text with prosody markers based on emotion (Path B).

    Adds emotional markers like ～, ！, etc. to trigger TTS's natural
    prosody generation.

    Args:
        text: Original text content
        emotion: Detected emotion (e.g., "撒嬌", "生氣")

    Returns:
        Enhanced text with prosody markers
    """
    if not text:
        return text

    enhancer = EMOTION_TEXT_ENHANCEMENT.get(emotion, EMOTION_TEXT_ENHANCEMENT.get(DEFAULT_EMOTION))
    return enhancer(text)


class EmotionParser:
    """
    Streaming emotion + content parser.

    Format: [E:情緒]內容
    - [E: marks start of emotion tag
    - ] is clear delimiter after emotion value
    - Content follows after ]

    Output:
        (emotion, content) - when both emotion AND content are ready
        None - needs more data

    State machine states:
        S_EMPTY   = 0  # No [ found yet
        S_TAG     = 1  # Found [, looking for E:
        S_EMOTION = 2  # Found E:, reading emotion until ]
        S_CONTENT = 3  # Found ], emitting content
    """

    # State constants
    S_EMPTY = 0
    S_TAG = 1      # Found [ looking for E:
    S_EMOTION = 2  # Found E:, reading emotion until ]
    S_CONTENT = 3  # Found ], emitting content

    def __init__(self, custom_map: Optional[dict] = None):
        self.custom_map = custom_map
        self.current_emotion: Optional[str] = None
        self.current_instruct: Optional[str] = None
        self.is_emotion_locked: bool = False

        # Internal parsing state
        self._buffer: str = ""      # Accumulated input not yet processed
        self._state: int = self.S_EMPTY
        self._first_content_emitted: bool = False

    @property
    def is_ready(self) -> bool:
        """True if emotion is locked and instruct is available."""
        return self.is_emotion_locked and self.current_instruct is not None

    def update(self, text: str) -> Optional[tuple]:
        """
        Process incoming text stream.

        Args:
            text: New text chunk from LLM (can be 1 char or many)

        Returns:
            None if needs more data
            (emotion, content) when both ready (first char triggers emission)
            (None, content) for subsequent content chars
        """
        if text:
            self._buffer += text

        if self._buffer:
            return self._parse_buffer()

        return None

    def _parse_buffer(self) -> Optional[tuple]:
        """
        Parse accumulated buffer. Mutates self._buffer to remove consumed chars.
        """
        while len(self._buffer) > 0:
            if self._state == self.S_EMPTY:
                # Look for [ to start tag
                bracket_pos = self._buffer.find('[')
                if bracket_pos < 0:
                    # No [ found - pure content, use default emotion
                    if not self.is_emotion_locked:
                        self._lock_emotion(DEFAULT_EMOTION)
                    content = self._buffer
                    self._buffer = ""
                    return self._emit_content(content)
                elif bracket_pos > 0:
                    # Content before [ - emit it with default emotion
                    self._lock_emotion(DEFAULT_EMOTION)
                    content = self._buffer[:bracket_pos]
                    self._buffer = self._buffer[bracket_pos:]
                    self._state = self.S_TAG
                    return self._emit_content(content)
                else:
                    # Starts with [
                    self._state = self.S_TAG
                    self._buffer = self._buffer[1:]
                    continue

            elif self._state == self.S_TAG:
                # Looking for E: after [
                # Buffer has [ removed, so check for E: not [E:
                if len(self._buffer) < 1:
                    return None  # Need more data

                # Partial marker case - wait if buffer is just 'E' or 'E' followed by non-':'
                if self._buffer == 'E':
                    return None  # Could be 'E:' - wait for ':'
                if len(self._buffer) >= 2 and self._buffer.startswith('E:') and self._buffer[2:].find(']') < 0:
                    # Buffer starts with 'E:' but no ] yet
                    return None  # Wait for closing bracket

                if self._buffer.startswith('E:'):
                    # Found 'E:' - now look for ]
                    bracket_pos = self._buffer.find(']')
                    if bracket_pos < 0:
                        # No ] yet - shouldn't happen given above check, but safety check
                        return None
                    # Found ] - extract emotion, transition to S_CONTENT
                    # [E: + emotion_val + ] + content
                    # After removing [, buffer starts with E: (2 chars), emotion at buffer[2:bracket_pos]
                    emotion_val = self._buffer[2:bracket_pos]
                    self._lock_emotion(emotion_val if emotion_val else DEFAULT_EMOTION)
                    self._buffer = self._buffer[bracket_pos + 1:]
                    self._state = self.S_CONTENT
                    if len(self._buffer) == 0:
                        return None
                    continue
                elif self._buffer.startswith('['):
                    # Double [ - skip first
                    self._buffer = self._buffer[1:]
                    continue
                else:
                    # Not E: - treat as content (e.g. plain text starting with [)
                    self._state = self.S_CONTENT
                    if not self.is_emotion_locked:
                        self._lock_emotion(DEFAULT_EMOTION)
                    continue

            elif self._state == self.S_CONTENT:
                # Emit one character at a time (for streaming TTS)
                if len(self._buffer) == 0:
                    return None
                char = self._buffer[0]
                self._buffer = self._buffer[1:]
                return self._emit_content(char)

        return None

    def _emit_content(self, content: str) -> Optional[tuple]:
        """Helper to emit content with proper (emotion, content) format."""
        if not content:
            return None
        if not self._first_content_emitted:
            self._first_content_emitted = True
            return (self.current_emotion, content)
        return (None, content)

    def _lock_emotion(self, emotion: str):
        """Lock in the emotion value."""
        self.current_emotion = emotion
        self.current_instruct = get_tts_instruct(emotion, self.custom_map)
        self.is_emotion_locked = True
        self._first_content_emitted = False

    def reset(self):
        """Reset parser to initial state."""
        self.current_emotion = None
        self.current_instruct = None
        self.is_emotion_locked = False
        self._buffer = ""
        self._state = self.S_EMPTY
        self._first_content_emitted = False


class EmotionMapper:
    """
    Wrapper for EmotionParser with backward compatibility.

    Supports both:
    - New format: [E:情緒]內容
    - Legacy format: [情感:調皮]哈哈哈...
    """

    def __init__(self, custom_map: Optional[dict] = None):
        self.custom_map = custom_map
        self.current_emotion: Optional[str] = None
        self.current_instruct: Optional[str] = None
        self._emotion_locked = False
        self._buffer = ""
        self._parser = EmotionParser(custom_map)

    def update(self, text: str) -> tuple:
        """
        Update with new text. Returns (emotion, content).
        """
        # If emotion already locked, drain accumulated content
        if self._emotion_locked:
            # Drain parser buffer first
            result = self._parser.update('')
            if result is not None:
                _, buffered_content = result
                if buffered_content:
                    return (None, buffered_content)
            # Then return accumulated text
            if self._buffer:
                content = self._buffer
                self._buffer = ""
                return (None, content)
            # If new text provided, accumulate it
            if text:
                self._buffer += text
                return (None, '')
            return (None, '')  # Signal "nothing more"

        # Emotion not locked - always use EmotionParser first
        # It handles partial markers (like 'E:' without ']') by buffering
        result = self._parser.update(text)
        if result is not None:
            emotion, content = result
            if emotion is not None:
                self.current_emotion = emotion
                self.current_instruct = get_tts_instruct(emotion, self.custom_map)
                self._emotion_locked = True
                self._buffer = ""  # Clear buffer - old content was partial tag
            if content:
                return (emotion, content)
            # Content was empty, but there might be more in buffer
            # Drain remaining content
            while True:
                drain_result = self._parser.update('')
                if drain_result is None:
                    break
                _, drain_content = drain_result
                if drain_content:
                    return (emotion, drain_content) if emotion else (None, drain_content)
                else:
                    break
            # Still nothing, might have leftover buffer in parser
            if text:
                self._buffer += text
            return (None, '')  # Signal "nothing more"

        # Parser returned None - accumulate and check legacy format
        self._buffer += text
        return self._parse_legacy_tag()

    def _parse_legacy_tag(self) -> tuple:
        """Fallback parser for legacy [情感: xxx] tag format."""
        match = EMOTION_TAG_PATTERN.search(self._buffer)
        if match:
            emotion = match.group(1).strip()
            self._buffer = EMOTION_TAG_PATTERN.sub('', self._buffer, count=1)
            self.current_emotion = emotion
            self.current_instruct = get_tts_instruct(emotion, self.custom_map)
            self._emotion_locked = True
            return emotion, self._buffer

        if self._buffer.startswith('{') or self._buffer.startswith('['):
            return None, ""

        result = self._buffer
        self._buffer = ""
        return None, result

    def reset(self):
        """Reset mapper to initial state."""
        self.current_emotion = None
        self.current_instruct = None
        self._emotion_locked = False
        self._buffer = ""
        self._parser.reset()

    @property
    def is_ready(self) -> bool:
        """True if emotion is locked and instruct is available."""
        return self._emotion_locked and self.current_instruct is not None
