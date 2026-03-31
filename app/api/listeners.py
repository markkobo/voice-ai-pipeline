"""
Listener API endpoints.

GET/POST/PATCH/DELETE for listener definitions.
"""

import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.listeners import (
    list_listeners,
    get_listener,
    create_listener,
    update_listener,
    delete_listener,
)

router = APIRouter(prefix="/api/listeners", tags=["listeners"])

VALID_EMOTIONS = {"寵溺", "撒嬌", "幽默", "毒舌", "溫和", "開心", "認真", "默認"}


class ListenerCreate(BaseModel):
    listener_id: str
    name: str
    is_family: bool = False
    default_emotion: str = "溫和"


class ListenerUpdate(BaseModel):
    name: Optional[str] = None
    default_emotion: Optional[str] = None


@router.get("/")
async def api_list_listeners():
    """List all listeners."""
    return {"listeners": list_listeners()}


@router.get("/{listener_id}")
async def api_get_listener(listener_id: str):
    """Get a single listener."""
    listener = get_listener(listener_id)
    if not listener:
        raise HTTPException(404, f"Listener not found: {listener_id}")
    return listener


@router.post("/")
async def api_create_listener(body: ListenerCreate):
    """Create a new listener."""
    # Validate ID format
    if not re.match(r'^[a-z][a-z0-9_]*$', body.listener_id):
        raise HTTPException(400, "listener_id must be lowercase letters, numbers, underscores, starting with a letter")

    try:
        listener = create_listener(
            body.listener_id,
            body.name,
            body.is_family,
            body.default_emotion,
        )
        return listener
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.patch("/{listener_id}")
async def api_update_listener(listener_id: str, body: ListenerUpdate):
    """Update listener name or default emotion."""
    if body.default_emotion and body.default_emotion not in VALID_EMOTIONS:
        raise HTTPException(400, f"Invalid emotion: {body.default_emotion}. Valid: {VALID_EMOTIONS}")

    listener = update_listener(listener_id, name=body.name, default_emotion=body.default_emotion)
    if not listener:
        raise HTTPException(404, f"Listener not found: {listener_id}")
    return listener


@router.delete("/{listener_id}")
async def api_delete_listener(listener_id: str):
    """Delete a listener."""
    success = delete_listener(listener_id)
    if not success:
        raise HTTPException(404, f"Listener not found: {listener_id}")
    return {"status": "deleted", "listener_id": listener_id}
