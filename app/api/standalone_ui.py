"""
Standalone Voice AI dev UI — Jinja2 + static JS (RFC_M6 Phase 0-pre).

HTML lives in `app/templates/standalone.html`. CSS in
`app/static/css/standalone.css`. JS in `app/static/js/standalone.js`.
"""
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()


def _asset_version() -> str:
    """mtime of standalone.js — bumps the ?v= query string on every deploy
    so Cloudflare's edge cache + browser disk cache don't serve a stale
    bundle after a code change. 2026-06-02: CF was holding the JS for
    4hr (max-age=14400) and the user's browser kept the old code that
    didn't include the language field, so language=None reached the server.
    """
    js = Path(__file__).resolve().parent.parent / "static" / "js" / "standalone.js"
    try:
        return str(int(js.stat().st_mtime))
    except OSError:
        return "0"


@router.get("/ui")
async def serve_ui(request: Request):
    # Starlette ≥1.0 signature: (request, name, context=None).
    # Pre-1.0 was (name, {"request": request}) — passing the dict as
    # `name` makes Jinja2 try to use it as a cache key and raises
    # `TypeError: unhashable type: 'dict'`.
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "standalone.html", {"asset_version": _asset_version()}
    )
