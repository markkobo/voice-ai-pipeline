"""
Smoke tests for the dev UI Jinja2 migration (RFC_M6 Phase 0-pre).

These don't lock the HTML content — that changes too often. They lock
the *integration*: route returns 200, returns HTML, references the
static CSS/JS by the expected paths, and the static files are served.

The whole point of this test file is to catch the class of bug we just
hit: the Starlette ≥1.0 TemplateResponse signature change which made
`/ui` 500 without any test surface noticing.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract


class TestStandaloneUI:
    def test_ui_returns_html(self, client):
        r = client.get("/ui")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
        # The template must reference the static assets — if it
        # regresses to inline strings the migration was undone.
        assert "/static/css/standalone.css" in r.text
        assert "/static/js/standalone.js" in r.text

    def test_static_css_served(self, client):
        r = client.get("/static/css/standalone.css")
        assert r.status_code == 200
        assert len(r.text) > 100
        # Sentinel content — first CSS rule in the file.
        assert "box-sizing" in r.text

    def test_static_js_served(self, client):
        r = client.get("/static/js/standalone.js")
        assert r.status_code == 200
        assert len(r.text) > 100
        # Sentinel — the WebSocket connection function name.
        assert "function connect" in r.text or "connect()" in r.text


class TestRecordingsUI:
    def test_ui_returns_html(self, client):
        r = client.get("/ui/recordings")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
        assert "/static/css/recordings.css" in r.text
        assert "/static/js/recordings.js" in r.text

    def test_static_css_served(self, client):
        r = client.get("/static/css/recordings.css")
        assert r.status_code == 200
        assert len(r.text) > 100

    def test_static_js_served(self, client):
        r = client.get("/static/js/recordings.js")
        assert r.status_code == 200
        assert len(r.text) > 100


class TestTrainingUI:
    def test_ui_returns_html(self, client):
        r = client.get("/ui/training")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
        assert "/static/css/training.css" in r.text
        assert "/static/js/training.js" in r.text

    def test_static_css_served(self, client):
        r = client.get("/static/css/training.css")
        assert r.status_code == 200
        assert len(r.text) > 100

    def test_static_js_served(self, client):
        r = client.get("/static/js/training.js")
        assert r.status_code == 200
        assert len(r.text) > 100
