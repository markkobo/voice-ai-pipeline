"""ListenersService — orchestrator for listener CRUD."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from app.api._errors import (
    DuplicateIdError,
    InvalidEmotionError,
    InvalidIdFormatError,
    ListenerNotFoundError,
    SeedListenerReadonlyError,
)

from .models import VALID_EMOTIONS, Listener
from .repository import JsonListenerRepository, ListenerNotFound

log = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class ListenersService:
    """Pure business logic around listener CRUD."""

    def __init__(self, repository: JsonListenerRepository) -> None:
        self.repository = repository

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def list_listeners(self) -> list[Listener]:
        return self.repository.list()

    def get(self, listener_id: str) -> Listener:
        try:
            return self.repository.get(listener_id)
        except ListenerNotFound as e:
            raise ListenerNotFoundError(
                f"Listener not found: {listener_id}",
                details={"listener_id": listener_id},
            ) from e

    def exists(self, listener_id: str) -> bool:
        return self.repository.exists(listener_id)

    # ------------------------------------------------------------------
    # Mutate
    # ------------------------------------------------------------------
    def create(
        self,
        *,
        listener_id: str,
        name: str,
        is_family: bool = False,
        default_emotion: str = "溫和",
    ) -> Listener:
        if not _ID_PATTERN.match(listener_id):
            raise InvalidIdFormatError(
                "listener_id must be lowercase letters, numbers, underscores, "
                "starting with a letter",
                details={"listener_id": listener_id, "pattern": _ID_PATTERN.pattern},
            )
        if default_emotion not in VALID_EMOTIONS:
            raise InvalidEmotionError(
                f"Invalid emotion: {default_emotion!r}",
                details={"default_emotion": default_emotion, "valid": sorted(VALID_EMOTIONS)},
            )
        if self.repository.exists(listener_id):
            raise DuplicateIdError(
                f"Listener already exists: {listener_id}",
                details={"listener_id": listener_id},
            )
        listener = Listener(
            listener_id=listener_id,
            name=name,
            is_family=is_family,
            default_emotion=default_emotion,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_seed=False,
        )
        self.repository.save(listener)
        log.info("[LISTENER] Created %s", listener_id)
        return listener

    def update(
        self,
        listener_id: str,
        *,
        name: Optional[str] = None,
        default_emotion: Optional[str] = None,
    ) -> Listener:
        existing = self.get(listener_id)
        if default_emotion is not None:
            if default_emotion not in VALID_EMOTIONS:
                raise InvalidEmotionError(
                    f"Invalid emotion: {default_emotion!r}",
                    details={"default_emotion": default_emotion, "valid": sorted(VALID_EMOTIONS)},
                )
            existing.default_emotion = default_emotion
        if name is not None:
            existing.name = name
        self.repository.save(existing)
        return existing

    def delete(self, listener_id: str) -> None:
        existing = self.get(listener_id)
        if existing.is_seed:
            # New in Phase 1.3: seeded listeners cannot be deleted via API.
            # The legacy code allowed this, which broke any downstream code
            # that assumes `child`/`mom`/`default` exist.
            raise SeedListenerReadonlyError(
                f"Cannot delete seeded listener: {listener_id!r}",
                details={"listener_id": listener_id},
            )
        try:
            self.repository.delete(listener_id)
        except ListenerNotFound as e:
            raise ListenerNotFoundError(
                f"Listener not found: {listener_id}",
                details={"listener_id": listener_id},
            ) from e
