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

    def test_fsm_state_constants_present(self, client):
        """v27 chat-UI refactor introduced a 6-state FSM. The state
        constants live in standalone.js and drive all button labels.
        Regressing this back to scattered boolean flags
        (isRecording/isThinking/…) is exactly the class of UX bug
        the refactor was meant to prevent."""
        r = client.get("/static/js/standalone.js")
        assert r.status_code == 200
        body = r.text
        for sym in ("IDLE", "CONNECTING", "READY", "LISTENING", "THINKING", "SPEAKING"):
            assert f"'{sym}'" in body, f"FSM constant {sym!r} missing"
        # The window.CHAT_STATE export is what external tests/devtools
        # use to assert on FSM state.
        assert "window.CHAT_STATE" in body
        # Single primary button rather than the legacy 4-button row.
        assert "onPrimaryClick" in body

    def test_auto_continue_checkbox_present(self, client):
        """Bug 1 fix: after the AI finishes replying, the mic should
        auto-re-arm so the user can keep talking. The checkbox lets
        users opt out and fall back to push-to-talk."""
        r = client.get("/ui")
        assert r.status_code == 200
        assert 'id="autoContinueChk"' in r.text
        assert "自動繼續對話" in r.text
        # And the JS must actually consult the checkbox.
        js = client.get("/static/js/standalone.js").text
        assert "autoContinueChk" in js
        assert "maybeFinishResponse" in js

    def test_legacy_buttons_removed(self, client):
        """The four redundant buttons (開始對話/停止對話/開始錄音/
        停止錄音 plus the separate 強制送出/取消) collapsed into one
        primary button. If they reappear, the refactor regressed."""
        r = client.get("/ui")
        html = r.text
        for legacy_id in ("startStopBtn", "recordBtn", "commitBtn", "cancelBtn"):
            assert f'id="{legacy_id}"' not in html, f"legacy button {legacy_id} reintroduced"


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
