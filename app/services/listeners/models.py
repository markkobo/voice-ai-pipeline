"""Pydantic domain models for listeners."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


# Valid TTS emotion tags. Single source of truth for listener-side
# validation; was duplicated in api/listeners.py before Phase 1.3.
VALID_EMOTIONS: frozenset[str] = frozenset(
    {"寵溺", "撒嬌", "幽默", "毒舌", "溫和", "開心", "認真", "默認"}
)


class Listener(BaseModel):
    """A listener — who the AI is currently speaking to."""

    model_config = ConfigDict(extra="ignore")

    listener_id: str
    name: str
    is_family: bool = False
    default_emotion: str = "溫和"
    created_at: Optional[str] = None
    # Seeded listeners can't be deleted; this flag is set by the seed data.
    is_seed: bool = False


# Default listeners every fresh install gets. The audit flagged that the
# legacy code allowed deleting these — Phase 1.3 protects them with
# `is_seed=True`.
SEED_LISTENERS: list[Listener] = [
    Listener(listener_id="child", name="小孩", is_family=True, default_emotion="撒嬌", created_at="2026-03-01T00:00:00Z", is_seed=True),
    Listener(listener_id="mom", name="媽媽", is_family=True, default_emotion="撒嬌", created_at="2026-03-01T00:00:00Z", is_seed=True),
    Listener(listener_id="dad", name="爸爸", is_family=True, default_emotion="溫和", created_at="2026-03-01T00:00:00Z", is_seed=True),
    Listener(listener_id="friend", name="朋友", is_family=True, default_emotion="幽默", created_at="2026-03-01T00:00:00Z", is_seed=True),
    Listener(listener_id="elder", name="長輩", is_family=True, default_emotion="溫和", created_at="2026-03-01T00:00:00Z", is_seed=True),
    Listener(listener_id="reporter", name="記者", is_family=False, default_emotion="溫和", created_at="2026-03-01T00:00:00Z", is_seed=True),
    Listener(listener_id="default", name="預設", is_family=False, default_emotion="撒嬌", created_at="2026-03-01T00:00:00Z", is_seed=True),
]
