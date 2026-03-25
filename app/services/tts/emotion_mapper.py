"""
Emotion mapper: converts [情感: xxx] tags to TTS natural language instructions.

Used by TTS engine (VoiceDesign mode) to control tone, pace, and style.
"""
from typing import Optional, Dict


# Default emotion → TTS instruct mapping
# Format: "(natural language description of desired voice style)"
DEFAULT_EMOTION_MAP: Dict[str, str] = {
    "寵溺": "(gentle, high-pitched, warm and loving tone, soft delivery)",
    "撒嬌": "(coquettish, soft, slightly slower pace, endearing inflection)",
    "毒舌": "(witty, fast-paced, sarcastic but playful tone, confident delivery)",
    "幽默": "(playful, light-hearted, occasional laughs, casual and funny)",
    "認真": "(serious, thoughtful, measured pace, clear and deliberate)",
    "溫和": "(calm, gentle, warm, relaxed and reassuring tone)",
    "調皮": "(mischievous, playful, slightly teasing, energetic)",
    "感動": "(emotional, sincere, heartfelt, slower and softer)",
    "生氣": "(annoyed, frustrated, slightly elevated pitch, impatient)",
    "開心": "(happy, bright, enthusiastic, faster pace with positive energy)",
    # Fallback
    "默認": "(natural, conversational tone, warm and engaging)",
}


# Compiled regex for extracting emotion tags from LLM output
# Matches patterns like: [情感: 撒嬌] or [情感:撒嬌] or [情感: 撒嬌]  (with or without space)
import re
EMOTION_TAG_PATTERN = re.compile(r'^\[情感[:：]\s*(.*?)\]\s*')


def parse_emotion_tag(text: str) -> tuple[Optional[str], str]:
    """
    Extract emotion tag from the beginning of text.

    Args:
        text: LLM output text (may contain [情感: xxx] at start)

    Returns:
        Tuple of (emotion_string, cleaned_text_with_tag_removed)
        If no tag found, returns (None, original_text)
    """
    match = EMOTION_TAG_PATTERN.match(text)
    if match:
        emotion = match.group(1).strip()
        cleaned = EMOTION_TAG_PATTERN.sub('', text, count=1)
        return emotion, cleaned
    return None, text


def get_tts_instruct(
    emotion: str,
    custom_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Get TTS instruction string for a given emotion.

    Args:
        emotion: Emotion tag string (e.g., "撒嬌", "寵溺")
        custom_map: Optional custom emotion→instruct mapping to override defaults

    Returns:
        TTS instruct string, e.g. "(gentle, high-pitched, warm and loving tone)"
    """
    mapping = custom_map or DEFAULT_EMOTION_MAP
    return mapping.get(emotion, mapping.get("默認"))


class EmotionMapper:
    """
    Stateful emotion mapper that tracks current emotion across streaming.

    Usage:
        mapper = EmotionMapper()
        mapper.update("「[情感: 撒嬌]好啦～")
        # mapper.current_emotion = "撒嬌"
        # mapper.current_instruct = "(coquettish, soft, slightly slower pace...)"
        # mapper.cleaned_text = "「好啦～"
    """

    def __init__(self, custom_map: Optional[Dict[str, str]] = None):
        self.custom_map = custom_map
        self.current_emotion: Optional[str] = None
        self.current_instruct: Optional[str] = None
        self._emotion_locked = False
        self._buffer = ""  # Accumulate text until we can match the emotion tag
        self._buffer_returned_len = 0  # Track how much of buffer has been returned

    def update(self, text: str) -> tuple[Optional[str], str]:
        """
        Update emotion state from new text.

        Accumulates text in a buffer and checks for emotion tags.
        The emotion tag must be at the BEGINNING of the accumulated text.
        Once emotion is detected, subsequent text is returned incrementally.

        Args:
            text: New text chunk from LLM

        Returns:
            Tuple of (newly_detected_emotion_or_None, new_text_since_last_call)
        """
        if self._emotion_locked:
            # Already detected emotion, return text incrementally
            return None, text

        # Accumulate
        self._buffer += text

        # Try to find emotion tag at the beginning
        match = EMOTION_TAG_PATTERN.match(self._buffer)
        if match:
            emotion = match.group(1).strip()
            # Remove the tag from buffer
            self._buffer = EMOTION_TAG_PATTERN.sub('', self._buffer, count=1)
            # Return ALL buffered text (new content since last return + emotion tag removed)
            new_text = self._buffer[self._buffer_returned_len:]
            self._buffer_returned_len = len(self._buffer)
            self.current_emotion = emotion
            self.current_instruct = get_tts_instruct(emotion, self.custom_map)
            self._emotion_locked = True
            return emotion, new_text

        # No complete tag yet
        # Return only the NEW portion of the buffer
        new_text = self._buffer[self._buffer_returned_len:]
        self._buffer_returned_len = len(self._buffer)
        return None, new_text

    def reset(self):
        """Reset emotion state for new conversation turn."""
        self.current_emotion = None
        self.current_instruct = None
        self._emotion_locked = False
        self._buffer = ""
        self._buffer_returned_len = 0

    @property
    def is_ready(self) -> bool:
        """True if emotion has been detected and TTS can start."""
        return self._emotion_locked and self.current_instruct is not None
