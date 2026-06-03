"""
Standalone Voice AI dev UI — Jinja2 + static JS (RFC_M6 Phase 0-pre).

HTML lives in `app/templates/standalone.html`. CSS in
`app/static/css/standalone.css`. JS in `app/static/js/standalone.js`.
"""
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()


# Static assets used by standalone.html. The cache buster watches all of
# them so a CSS-only change still bumps the version. GPT-5 review of
# commit 3703f20 (2026-06-03) flagged that the original implementation
# only watched standalone.js — CSS edits would silently fall under the
# stale Cloudflare edge cache (max-age=14400).
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_WATCHED_ASSETS = (
    "js/standalone.js",
    "js/_status_bar.js",
    "css/standalone.css",
    "css/_status_bar.css",
)


def _asset_version() -> str:
    """Max mtime across every watched static asset. Bumps the ?v= query
    string on every deploy so Cloudflare's edge cache + browser disk
    cache don't serve a stale bundle after a code change.

    2026-06-02 incident: CF was holding standalone.js for 4hr
    (max-age=14400) and the user's browser kept old code that didn't
    include the language field, so language=None reached the server.
    2026-06-03 review: extended from JS-only to all served assets so
    CSS edits also bust."""
    latest = 0
    for rel in _WATCHED_ASSETS:
        try:
            mtime = int((_STATIC_DIR / rel).stat().st_mtime)
            if mtime > latest:
                latest = mtime
        except OSError:
            continue
    return str(latest) if latest else "0"


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
