"""
Prompt manager with speaker-aware persona system.

Provides contextual system prompts based on persona_id and listener_id.
Persona definitions are loaded from JSON files in app/resources/personas/.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


# ============================================================================
# Built-in Persona Types (fallback if JSON not available)
# ============================================================================

class PersonaType:
    """Persona type constants."""
    CAREGIVER = "caregiver"
    ELDER_GENTLE = "elder_gentle"
    ELDER_PLAYFUL = "elder_playful"
    CHILD = "child"
    EXTERNAL = "external"
    DEFAULT = "default"


# ============================================================================
# Built-in Persona Prompts (used when JSON loading fails or for development)
# ============================================================================

_BUILTIN_PERSONA_PROMPTS: Dict[str, str] = {
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


# ============================================================================
# Persona Manager
# ============================================================================

class PersonaManager:
    """
    Manages persona definitions loaded from JSON files.

    Usage:
        pm = PersonaManager()
        system_prompt = pm.get_prompt("xiao_s", listener_id="child")
    """

    def __init__(
        self,
        persona_dir: Optional[str] = None,
        custom_personas: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize PersonaManager.

        Args:
            persona_dir: Directory containing persona JSON files.
                        Defaults to app/resources/personas/
            custom_personas: Optional dict of {persona_type: prompt_string}
                            to override builtin personas.
        """
        if persona_dir is None:
            # Find app/resources/personas relative to this file
            base_dir = Path(__file__).parent.parent.parent
            persona_dir = base_dir / "resources" / "personas"

        self.persona_dir = Path(persona_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._custom_personas = custom_personas or {}

    def _load_persona_json(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """
        Load persona definition from JSON file.

        Args:
            persona_id: Persona identifier (e.g., "xiao_s")

        Returns:
            Persona dict or None if not found
        """
        if persona_id in self._cache:
            return self._cache[persona_id]

        persona_file = self.persona_dir / f"{persona_id}.json"

        if not persona_file.exists():
            return None

        try:
            with open(persona_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cache[persona_id] = data
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"[PromptManager] Failed to load persona {persona_id}: {e}")
            return None

    def _get_builtin_prompt(self, persona_type: str) -> str:
        """Get built-in prompt for a persona type."""
        return _BUILTIN_PERSONA_PROMPTS.get(persona_type, _BUILTIN_PERSONA_PROMPTS[PersonaType.DEFAULT])

    def get_prompt(
        self,
        persona_id: str,
        listener_id: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get system prompt for the given persona + listener.

        Args:
            persona_id: Persona identifier (e.g., "xiao_s")
            listener_id: Optional listener identifier (e.g., "child", "mom")
            extra_context: Optional dict of context variables to inject

        Returns:
            System prompt string ready to send to LLM
        """
        # Check custom personas first
        if persona_id in self._custom_personas:
            base_prompt = self._custom_personas[persona_id]
        else:
            # Try to load from JSON
            persona_data = self._load_persona_json(persona_id)

            if persona_data:
                # Build prompt from JSON structure
                parts = []

                # Base personality
                base = persona_data.get("base_personality", "")
                if base:
                    parts.append(base)

                # Relationship modifier
                relationships = persona_data.get("relationships", {})
                if listener_id and listener_id in relationships:
                    rel = relationships[listener_id]
                else:
                    rel = relationships.get(
                        persona_data.get("default_relationship", "default"),
                        ""
                    )
                if rel:
                    parts.append(f"\n\n與對方說話時：{rel}")

                # Emotion instruction
                emotion_instr = persona_data.get("emotion_instruction", "")
                if emotion_instr:
                    parts.append(f"\n\n{emotion_instr}")

                base_prompt = "\n".join(parts)
            else:
                # Fallback to built-in
                base_prompt = self._get_builtin_prompt(persona_id)

        # Inject extra context
        if extra_context:
            context_parts = []
            for key, val in extra_context.items():
                context_parts.append(f"[Context: {key} = {val}]")
            if context_parts:
                base_prompt += "\n" + "\n".join(context_parts)

        return base_prompt

    def get_available_personas(self) -> list[str]:
        """Get list of available persona IDs."""
        personas = set(self._custom_personas.keys())

        if self.persona_dir.exists():
            for f in self.persona_dir.glob("*.json"):
                personas.add(f.stem)

        # Add built-in personas
        personas.update(_BUILTIN_PERSONA_PROMPTS.keys())

        return sorted(personas)

    def get_available_listeners(self, persona_id: str) -> list[str]:
        """
        Get list of available listener IDs for a persona.

        Args:
            persona_id: Persona identifier

        Returns:
            List of listener IDs
        """
        if persona_id in self._custom_personas:
            return []

        persona_data = self._load_persona_json(persona_id)
        if persona_data:
            return list(persona_data.get("relationships", {}).keys())

        return []

    def reload(self, persona_id: Optional[str] = None):
        """
        Reload persona data from disk.

        Args:
            persona_id: Specific persona to reload, or None for all
        """
        if persona_id:
            self._cache.pop(persona_id, None)
        else:
            self._cache.clear()


# ============================================================================
# Legacy PromptManager for backward compatibility
# ============================================================================

class PromptManager(PersonaManager):
    """
    Legacy PromptManager — kept for backward compatibility.

    Use PersonaManager for new code.
    """

    def __init__(self, custom_personas: Optional[Dict[str, str]] = None):
        super().__init__(custom_personas=custom_personas)
