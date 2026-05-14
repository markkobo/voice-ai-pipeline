"""
Listeners REST API — routes only.

Phase 1.3: every route uses Pydantic with `extra="forbid"`, delegates to
ListenersService via Depends, and lets the app-wide DomainError handler map
errors to HTTP. No more HTTPException raised in this module.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from app.api._dependencies import get_listeners_service
from app.services.listeners import Listener, ListenersService

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/listeners", tags=["listeners"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class ListenerCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listener_id: str
    name: str
    is_family: bool = False
    default_emotion: str = "溫和"


class ListenerUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    default_emotion: Optional[str] = None


class ListenersListResponse(BaseModel):
    listeners: list[Listener]


class DeleteListenerResponse(BaseModel):
    status: str = "deleted"
    listener_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/", response_model=ListenersListResponse)
async def api_list_listeners(
    service: ListenersService = Depends(get_listeners_service),
) -> ListenersListResponse:
    return ListenersListResponse(listeners=service.list_listeners())


@router.get("/{listener_id}", response_model=Listener)
async def api_get_listener(
    listener_id: str,
    service: ListenersService = Depends(get_listeners_service),
) -> Listener:
    return service.get(listener_id)


@router.post("/", response_model=Listener)
async def api_create_listener(
    body: ListenerCreateRequest,
    service: ListenersService = Depends(get_listeners_service),
) -> Listener:
    return service.create(
        listener_id=body.listener_id,
        name=body.name,
        is_family=body.is_family,
        default_emotion=body.default_emotion,
    )


@router.patch("/{listener_id}", response_model=Listener)
async def api_update_listener(
    listener_id: str,
    body: ListenerUpdateRequest,
    service: ListenersService = Depends(get_listeners_service),
) -> Listener:
    return service.update(
        listener_id, name=body.name, default_emotion=body.default_emotion
    )


@router.delete("/{listener_id}", response_model=DeleteListenerResponse)
async def api_delete_listener(
    listener_id: str,
    service: ListenersService = Depends(get_listeners_service),
) -> DeleteListenerResponse:
    service.delete(listener_id)
    return DeleteListenerResponse(listener_id=listener_id)
