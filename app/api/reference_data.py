"""
Reference data endpoints.

Note: /api/personas and /api/listeners are now served by app/api/personas.py
and app/api/listeners.py respectively. This file keeps the router for
any remaining reference endpoints.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["reference"])

# Placeholder — remaining endpoints moved to personas.py / listeners.py
