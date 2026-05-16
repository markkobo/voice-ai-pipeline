"""
Training management dev UI — Jinja2 + static JS (RFC_M6 Phase 0-pre).

HTML lives in `app/templates/training.html`. CSS/JS in
`app/static/{css,js}/training.{css,js}`.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["ui"])


@router.get("/ui/training")
async def training_page(request: Request):
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(request, "training.html")
