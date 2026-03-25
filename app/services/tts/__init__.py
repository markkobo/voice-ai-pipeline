"""TTS services module."""
from app.services.tts.qwen_tts_engine import (
    FasterQwenTTSEngine,
    MockTTSEngine,
    get_tts_engine,
    TTSStreamEvent,
)
from app.services.tts.emotion_mapper import (
    EmotionMapper,
    parse_emotion_tag,
    get_tts_instruct,
    DEFAULT_EMOTION_MAP,
)

__all__ = [
    "FasterQwenTTSEngine",
    "MockTTSEngine",
    "get_tts_engine",
    "TTSStreamEvent",
    "EmotionMapper",
    "parse_emotion_tag",
    "get_tts_instruct",
    "DEFAULT_EMOTION_MAP",
]
