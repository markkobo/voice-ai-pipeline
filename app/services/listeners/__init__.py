"""
Listeners package — same shape as personas/.

Modern callers use `ListenersService` via Depends. Legacy function-style
API is preserved as a thin shim for back-compat.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .models import SEED_LISTENERS, VALID_EMOTIONS, Listener
from .repository import JsonListenerRepository, ListenerNotFound
from .service import ListenersService

log = logging.getLogger(__name__)


_default_service: Optional[ListenersService] = None


def _resolve_data_root() -> Path:
    """Thin wrapper around app.config.data_root() — kept for back-compat."""
    from app import config as _cfg
    return _cfg.data_root()


def _get_default_service() -> ListenersService:
    global _default_service
    if _default_service is None:
        _default_service = ListenersService(JsonListenerRepository(_resolve_data_root()))
    return _default_service


def list_listeners() -> list[dict]:
    return [l.model_dump(mode="json") for l in _get_default_service().list_listeners()]


def get_listener(listener_id: str) -> Optional[dict]:
    listener = _get_default_service().repository.get_or_none(listener_id)
    return listener.model_dump(mode="json") if listener else None


def create_listener(
    listener_id: str,
    name: str,
    is_family: bool = False,
    default_emotion: str = "溫和",
) -> dict:
    from app.api._errors import (
        DuplicateIdError,
        InvalidEmotionError,
        InvalidIdFormatError,
    )

    try:
        return _get_default_service().create(
            listener_id=listener_id,
            name=name,
            is_family=is_family,
            default_emotion=default_emotion,
        ).model_dump(mode="json")
    except (InvalidIdFormatError, InvalidEmotionError, DuplicateIdError) as e:
        raise ValueError(e.message) from e


def update_listener(
    listener_id: str,
    name: Optional[str] = None,
    default_emotion: Optional[str] = None,
) -> dict:
    from app.api._errors import InvalidEmotionError, ListenerNotFoundError

    try:
        return _get_default_service().update(
            listener_id, name=name, default_emotion=default_emotion
        ).model_dump(mode="json")
    except ListenerNotFoundError as e:
        raise ValueError(e.message) from e
    except InvalidEmotionError as e:
        raise ValueError(e.message) from e


def delete_listener(listener_id: str) -> bool:
    from app.api._errors import ListenerNotFoundError, SeedListenerReadonlyError

    try:
        _get_default_service().delete(listener_id)
        return True
    except ListenerNotFoundError as e:
        raise ValueError(e.message) from e
    except SeedListenerReadonlyError as e:
        raise ValueError(e.message) from e


def _reset_default_service_for_testing() -> None:
    global _default_service
    _default_service = None


__all__ = [
    "SEED_LISTENERS",
    "VALID_EMOTIONS",
    "Listener",
    "JsonListenerRepository",
    "ListenerNotFound",
    "ListenersService",
    "list_listeners",
    "get_listener",
    "create_listener",
    "update_listener",
    "delete_listener",
]
