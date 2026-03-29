"""
Reference data endpoints.

Returns available personas and listener types.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["reference"])


@router.get("/personas")
async def list_personas():
    """List available personas."""
    return [
        {"id": "xiao_s", "name": "小S", "description": "小S 活潑調皮的性格"},
        {"id": "caregiver", "name": "照護者", "description": "溫柔體貼的照護者"},
        {"id": "elder_gentle", "name": "長輩-溫柔", "description": "溫柔的長輩語氣"},
        {"id": "elder_playful", "name": "長輩-活潑", "description": "活潑的長輩語氣"},
    ]


@router.get("/listeners")
async def list_listeners():
    """List available listener types."""
    return [
        {"id": "child", "name": "小孩", "description": "Child listener"},
        {"id": "mom", "name": "媽媽", "description": "Mom listener"},
        {"id": "dad", "name": "爸爸", "description": "Dad listener"},
        {"id": "friend", "name": "朋友", "description": "Friend listener"},
        {"id": "reporter", "name": "記者", "description": "Reporter/Interviewer"},
        {"id": "elder", "name": "長輩", "description": "Elder listener"},
        {"id": "default", "name": "預設", "description": "Default listener"},
    ]
