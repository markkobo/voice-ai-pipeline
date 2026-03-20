"""
Prompt manager with speaker-aware persona system.

Provides contextual system prompts based on speaker identity (speaker_id)
for future voice-print recognition integration. Currently supports
configurable persona types with fallback for unknown speakers.
"""
from typing import Optional, Dict, Any


class PersonaType:
    """Persona type constants."""
    CAREGIVER = "caregiver"      # Default caring assistant for family use
    ELDER_GENTLE = "elder_gentle"  # Warm, patient tone for elderly family members
    ELDER_PLAYFUL = "elder_playful"  # Slightly witty, respectful for elders
    CHILD = "child"            # Educational, encouraging for children
    EXTERNAL = "external"       # Sarcastic, witty (小S mode) for non-family users
    DEFAULT = "default"         # Neutral helpful assistant


# ============================================================================
# Persona Prompt Templates
# ============================================================================

_PERSONA_PROMPTS: Dict[PersonaType, str] = {
    PersonaType.CAREGIVER: (
        "你是一個貼心的家庭照護 AI 語音助理。你的主要使用者是家庭照護者。 "
        "請用溫暖、耐心、專業的口吻回應。提供實用建議，適時提醒健康注意事項。"
    ),
    PersonaType.ELDER_GENTLE: (
        "你是一位溫柔體貼的 AI 語音伙伴，請用輕聲細語、緩慢節奏的方式與長輩說話。 "
        "避免使用複雜術語，語氣要像對待家人一樣充滿愛與耐心。"
    ),
    PersonaType.ELDER_PLAYFUL: (
        "你是一位幽默風趣的 AI 伙伴，專門陪伴長輩。 "
        "說話可以帶點小俏皮，但不失尊重。適時說些溫暖的話語，讓長輩感到開心。"
    ),
    PersonaType.CHILD: (
        "你是一位親切有趣的老師，用小朋友听得懂的方式說話。 "
        "可以活潑一點，多使用簡單的例子，適時鼓勵小朋友。"
    ),
    PersonaType.EXTERNAL: (
        "你是小S風格的毒舌 AI，面對外人時可以用俏皮諷刺的口吻說話，"
        "但不失機智與幽默。注意保持輕鬆愉快的氛圍。"
    ),
    PersonaType.DEFAULT: (
        "你是一個語音 AI 助理，請根據上下文提供有幫助的回覆。"
    ),
}

# Fallback prompts when speaker_id is unknown
_DEFAULT_FALLBACK_PROMPT = (
    "你是一個有用的語音 AI 助理。請用清晰、簡潔的方式回覆。"
)

# Speaker ID to PersonaType mapping
# In the future, this will be populated dynamically via voice-print recognition
_SPEAKER_PERSONA_MAP: Dict[str, PersonaType] = {
    # Example mappings - expand as needed
    # "dad": PersonaType.ELDER_GENTLE,
    # "mom": PersonaType.ELDER_GENTLE,
    # "grandpa": PersonaType.ELDER_GENTLE,
    # "child_name": PersonaType.CHILD,
}


class PromptManager:
    """
    Manages system prompts with speaker-aware persona support.

    Usage:
        pm = PromptManager()
        system_prompt = pm.get_prompt(PersonaType.ELDER_GENTLE, speaker_id="dad")
    """

    def __init__(self, custom_personas: Optional[Dict[str, str]] = None):
        """
        Initialize PromptManager.

        Args:
            custom_personas: Optional dict of {persona_type: prompt_string}
                             to override default personas.
        """
        self._personas: Dict[str, str] = {**_PERSONA_PROMPTS}
        if custom_personas:
            for key, val in custom_personas.items():
                self._personas[key] = val
        self._speaker_map: Dict[str, str] = dict(_SPEAKER_PERSONA_MAP)

    def get_prompt(
        self,
        persona_type: str = PersonaType.DEFAULT,
        speaker_id: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get system prompt for the given persona type and speaker.

        Args:
            persona_type: PersonaType constant (str for serialization compat)
            speaker_id: Optional speaker identifier; if provided and a mapping
                       exists, persona_type is overridden by the mapping.
            extra_context: Optional dict of context variables to inject
                            (e.g., {"user_name": "爸爸", "time_of_day": "morning"})

        Returns:
            System prompt string
        """
        # Resolve speaker_id to persona_type if a mapping exists
        if speaker_id is not None and speaker_id in self._speaker_map:
            persona_type = self._speaker_map[speaker_id]

        # Get base prompt
        base_prompt = self._personas.get(persona_type, _DEFAULT_FALLBACK_PROMPT)

        # Inject extra context if provided
        if extra_context:
            context_lines = []
            for key, val in extra_context.items():
                context_lines.append(f"[Context: {key} = {val}]")
            if context_lines:
                base_prompt += "\n" + "\n".join(context_lines)

        return base_prompt

    def register_speaker(
        self,
        speaker_id: str,
        persona_type: str,
    ) -> None:
        """
        Register a speaker ID with a persona type.

        Args:
            speaker_id: Unique speaker identifier
            persona_type: PersonaType constant to assign
        """
        self._speaker_map[speaker_id] = persona_type

    def unregister_speaker(self, speaker_id: str) -> bool:
        """
        Unregister a speaker ID.

        Args:
            speaker_id: Speaker identifier to remove

        Returns:
            True if removed, False if not found
        """
        return self._speaker_map.pop(speaker_id, None) is not None

    def set_persona_prompt(self, persona_type: str, prompt: str) -> None:
        """
        Override or set a persona prompt.

        Args:
            persona_type: Persona type key
            prompt: New prompt string
        """
        self._personas[persona_type] = prompt
