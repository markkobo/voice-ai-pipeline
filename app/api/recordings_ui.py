"""
Recordings management dev UI — Jinja2 + static JS (RFC_M6 Phase 0-pre).

HTML lives in `app/templates/recordings.html`. CSS/JS in
`app/static/{css,js}/recordings.{css,js}`.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["ui"])


@router.get("/ui/recordings")
async def recordings_page(request: Request):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(request, "recordings.html")
