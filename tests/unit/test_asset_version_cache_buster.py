"""
Locks the asset-version cache buster behavior (standalone_ui.py).

GPT-5 review of commit 3703f20 (2026-06-03) caught that the original
implementation only watched standalone.js — a CSS-only change wouldn't
bump the version, and Cloudflare's edge cache (max-age=14400 = 4 hours)
would serve stale CSS to browsers.

Fix: watch all four served static assets (standalone.js, _status_bar.js,
standalone.css, _status_bar.css) and return the MAX mtime. Any one
file's edit bumps the version.
"""
import os
import time

import pytest

from app.api.standalone_ui import _asset_version, _WATCHED_ASSETS, _STATIC_DIR


def test_asset_version_returns_non_empty():
    """Baseline: at least one watched file exists and mtime is returned."""
    v = _asset_version()
    assert v
    assert v != "0", "expected a real mtime, got fallback '0'"
    # Should parse as an int (epoch seconds)
    int(v)


def test_asset_version_uses_max_mtime(tmp_path, monkeypatch):
    """When two watched files have different mtimes, the newer one wins.
    This is the GPT-5-flagged regression: bumping CSS without touching
    JS should still bust the cache."""
    # Build a fake static dir with the watched layout
    js_dir = tmp_path / "js"
    css_dir = tmp_path / "css"
    js_dir.mkdir()
    css_dir.mkdir()

    js_file = js_dir / "standalone.js"
    css_file = css_dir / "standalone.css"
    sbar_js = js_dir / "_status_bar.js"
    sbar_css = css_dir / "_status_bar.css"

    for f in (js_file, css_file, sbar_js, sbar_css):
        f.write_text("// stub")

    # Set staggered mtimes — CSS is newest
    base = int(time.time()) - 1000
    os.utime(js_file, (base, base))
    os.utime(sbar_js, (base, base))
    os.utime(sbar_css, (base, base))
    css_mtime = base + 500
    os.utime(css_file, (css_mtime, css_mtime))

    monkeypatch.setattr("app.api.standalone_ui._STATIC_DIR", tmp_path)

    v = _asset_version()
    assert v == str(css_mtime), (
        f"expected max mtime ({css_mtime}) but got {v} — CSS edits won't bust"
    )


def test_asset_version_handles_missing_file(tmp_path, monkeypatch):
    """If one watched asset is missing, fall back to the max of the rest
    instead of returning 0. (e.g., if _status_bar files were removed
    in a future refactor.)"""
    js_dir = tmp_path / "js"
    js_dir.mkdir()
    js_file = js_dir / "standalone.js"
    js_file.write_text("// stub")
    expected_mtime = int(js_file.stat().st_mtime)
    # css/ directory + status bar files intentionally absent

    monkeypatch.setattr("app.api.standalone_ui._STATIC_DIR", tmp_path)

    v = _asset_version()
    assert v == str(expected_mtime)


def test_asset_version_all_missing_returns_zero(tmp_path, monkeypatch):
    """If literally nothing exists at the path, return '0' rather than
    crashing. Lets the page still render in a broken-deploy state."""
    monkeypatch.setattr("app.api.standalone_ui._STATIC_DIR", tmp_path)
    assert _asset_version() == "0"


def test_watched_assets_includes_both_js_and_css():
    """Sanity: the watched set covers both .js and .css. Catches a
    regression where someone narrows it back to JS-only."""
    has_js = any(name.endswith(".js") for name in _WATCHED_ASSETS)
    has_css = any(name.endswith(".css") for name in _WATCHED_ASSETS)
    assert has_js, "watched set lost JS coverage"
    assert has_css, "watched set lost CSS coverage — GPT-5 regression"


def test_real_static_dir_resolves():
    """The default _STATIC_DIR (when not monkey-patched) points to a
    real directory in the repo. Guards against a path-relative bug
    creeping in if the file moves."""
    assert _STATIC_DIR.exists(), f"_STATIC_DIR does not resolve: {_STATIC_DIR}"
    assert _STATIC_DIR.is_dir()
