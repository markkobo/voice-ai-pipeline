"""
Standalone Voice AI dev UI — Jinja2 + static JS (RFC_M6 Phase 0-pre).

HTML lives in `app/templates/standalone.html`. CSS in
`app/static/css/standalone.css`. JS in `app/static/js/standalone.js`.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()


@router.get("/ui")
async def serve_ui(request: Request):
    # Starlette ≥1.0 signature: (request, name, context=None).
    # Pre-1.0 was (name, {"request": request}) — passing the dict as
    # `name` makes Jinja2 try to use it as a cache key and raises
    # `TypeError: unhashable type: 'dict'`.
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(request, "standalone.html")
