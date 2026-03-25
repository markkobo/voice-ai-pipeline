"""
LLM (Large Language Model) service module.

Provides async streaming LLM client with OpenAI-compatible API,
prompt management with speaker-aware persona support, and RAG stub.
"""
from .openai_client import OpenAIClient, MockLLMClient, LLMStreamEvent
from .prompt_manager import PromptManager, PersonaManager, PersonaType

__all__ = [
    "OpenAIClient",
    "MockLLMClient",
    "LLMStreamEvent",
    "PromptManager",
    "PersonaManager",
    "PersonaType",
]
