#!/usr/bin/env python3
"""
Automated smoke-test runbook for the demo-readiness audit.

Exercises the 10 steps from the audit's "Smoke Tier" against a running
server. Each step prints one of:

    [PASS] step N: <description>
    [FAIL] step N: <reason>
    [MANUAL] step N: <what the human still has to do>

Exits 0 iff every automatic step passes; exits 1 on any FAIL. MANUAL
steps don't gate the exit code — they're reminders for the operator.

Usage:
    python scripts/demo_smoke.py
    python scripts/demo_smoke.py --base-url http://localhost:8080
    python scripts/demo_smoke.py --expect-version v2_20260514_152118_456516
    python scripts/demo_smoke.py --skip-pytest

The script defaults to checking whichever version is currently active
(read from /api/system/status), so it stays correct as we ship new
models. Pass --expect-version to pin a specific id.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Wall-clock budget for the non-pytest steps. pytest itself is ~3min and
# is gated on --skip-pytest.
HTTP_TIMEOUT_S = 5
PREVIEW_TIMEOUT_S = 15

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Tally
# ---------------------------------------------------------------------------
class Tally:
    """Single source of truth for pass/fail/manual counts + exit code."""

    def __init__(self) -> None:
        self.passes = 0
        self.fails = 0
        self.manuals = 0

    def passed(self, step: int, msg: str) -> None:
        self.passes += 1
        print(f"[PASS] step {step}: {msg}", flush=True)

    def failed(self, step: int, msg: str) -> None:
        self.fails += 1
        print(f"[FAIL] step {step}: {msg}", flush=True)

    def manual(self, step: int, msg: str) -> None:
        self.manuals += 1
        print(f"[MANUAL] step {step}: {msg}", flush=True)


# ---------------------------------------------------------------------------
# HTTP helpers — stdlib only so the script has no extra deps.
# ---------------------------------------------------------------------------
def _http_get(
    url: str, timeout: float = HTTP_TIMEOUT_S
) -> tuple[int, dict[str, str], bytes]:
    """GET. Returns (status, headers, body)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() or b""


def _http_post_json(
    url: str, payload: dict, timeout: float = HTTP_TIMEOUT_S
) -> tuple[int, dict[str, str], bytes]:
    """POST JSON. Returns (status, headers, body)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() or b""


# ---------------------------------------------------------------------------
# Per-step checks. Each takes (tally, base_url, state) and mutates state +
# tally. `state` carries forward across steps (system_status, version_id, ...).
# ---------------------------------------------------------------------------
def step1_system_status_active_version(
    tally: Tally,
    base_url: str,
    state: dict,
    expect_version: Optional[str],
) -> None:
    """tts.active_version equals expected (or "any non-null" by default)."""
    try:
        status, _, body = _http_get(f"{base_url}/api/system/status")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tally.failed(1, f"GET /api/system/status failed: {e}")
        return
    if status != 200:
        tally.failed(1, f"GET /api/system/status returned HTTP {status}")
        return
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        tally.failed(1, f"system_status body not JSON: {e}")
        return
    state["system_status"] = data
    actual = (data.get("tts") or {}).get("active_version")
    if expect_version is None:
        if actual:
            tally.passed(1, f"tts.active_version={actual!r}")
            state["active_version"] = actual
        else:
            tally.failed(
                1, "tts.active_version is null — no model is currently active"
            )
        return
    if actual == expect_version:
        tally.passed(
            1, f"tts.active_version == {expect_version!r}"
        )
        state["active_version"] = actual
    else:
        tally.failed(
            1,
            f"tts.active_version expected {expect_version!r}, got {actual!r}",
        )


def step2_resources_within_budget(
    tally: Tally, base_url: str, state: dict
) -> None:
    """vram.util_pct < 70 and disk_free_gb > 50."""
    data = state.get("system_status")
    if data is None:
        tally.failed(2, "no system_status payload from step 1")
        return
    vram = data.get("vram") or {}
    util = vram.get("util_pct", 0)
    disk = data.get("disk_free_gb", 0.0)
    available = vram.get("available", False)
    problems: list[str] = []
    if available and util >= 70:
        problems.append(f"vram.util_pct={util} (>=70)")
    if disk <= 50:
        problems.append(f"disk_free_gb={disk} (<=50)")
    if problems:
        tally.failed(2, "; ".join(problems))
    else:
        tally.passed(
            2,
            f"vram.util_pct={util} (available={available}), disk_free_gb={disk}",
        )


def step3_ui_serves_status_bar(
    tally: Tally, base_url: str
) -> None:
    """GET /ui returns 200 + body contains #sysStatusBar."""
    try:
        status, _, body = _http_get(f"{base_url}/ui")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tally.failed(3, f"GET /ui failed: {e}")
        return
    if status != 200:
        tally.failed(3, f"GET /ui returned HTTP {status}")
        return
    if b"sysStatusBar" not in body:
        tally.failed(3, "/ui body missing #sysStatusBar marker")
        return
    tally.passed(3, "/ui contains #sysStatusBar")


def step4_manual_mic_test(tally: Tally) -> None:
    tally.manual(
        4,
        "open /ui in a browser, hold-to-record 你好, expect AI audio reply",
    )


def step5_manual_tts_before_llm_done(tally: Tally) -> None:
    tally.manual(
        5,
        "grep server log for tts_start ordered before llm_done in same utterance",
    )


def step6_manual_barge_in(tally: Tally) -> None:
    tally.manual(
        6,
        "interrupt AI mid-reply by speaking — TTS should cut within ~500ms",
    )


def step7_recordings_processed(
    tally: Tally, base_url: str
) -> None:
    """At least one recording is processed; at least one has multi-speaker."""
    try:
        status, _, body = _http_get(
            f"{base_url}/api/recordings/?page=1&limit=50"
        )
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tally.failed(7, f"GET /api/recordings/ failed: {e}")
        return
    if status != 200:
        tally.failed(7, f"GET /api/recordings/ returned HTTP {status}")
        return
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        tally.failed(7, f"recordings body not JSON: {e}")
        return
    recordings = data.get("recordings", [])
    if not recordings:
        tally.failed(7, "no recordings returned")
        return
    # The audit spec says `processing_status`; the actual API uses
    # `status`. Accept either to stay tolerant of contract drift.
    processed = [
        r
        for r in recordings
        if r.get("status") == "processed"
        or r.get("processing_status") == "processed"
    ]
    multi_speaker = [
        r
        for r in recordings
        if len(
            {
                seg.get("speaker_id")
                for seg in (r.get("speaker_segments") or [])
                if seg.get("speaker_id")
            }
        )
        >= 2
    ]
    if not processed:
        tally.failed(
            7,
            f"no recordings with status=processed (have {len(recordings)} total)",
        )
        return
    if not multi_speaker:
        tally.failed(
            7,
            f"no recordings with multi-speaker segments "
            f"(have {len(processed)} processed)",
        )
        return
    tally.passed(
        7,
        f"{len(processed)} processed, {len(multi_speaker)} multi-speaker "
        f"(out of {len(recordings)} returned)",
    )


def step8_training_versions_ready_active(
    tally: Tally, base_url: str, state: dict
) -> None:
    """At least one version status=ready and active=true. Capture final_loss."""
    try:
        status, _, body = _http_get(f"{base_url}/api/training/versions")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tally.failed(8, f"GET /api/training/versions failed: {e}")
        return
    if status != 200:
        tally.failed(8, f"GET /api/training/versions returned HTTP {status}")
        return
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        tally.failed(8, f"versions body not JSON: {e}")
        return
    versions = data.get("versions", [])
    ready = [v for v in versions if v.get("status") == "ready"]
    active = [v for v in versions if v.get("active") is True]
    if not ready:
        tally.failed(8, f"no version with status=ready (out of {len(versions)})")
        return
    if not active:
        # `active` may not be a per-version field; fall back to checking
        # active_version on system_status (already captured in step 1).
        # Mind the namespacing: training-versions list returns the BARE
        # version_id (`v2_20260514_…`) but system_status reports the
        # TTS-merged-model name (`xiao_s_v2_20260514_…`). Match by suffix.
        ss = state.get("system_status") or {}
        ss_active = (ss.get("tts") or {}).get("active_version") or ""
        match = None
        for v in ready:
            vid = v.get("version_id") or ""
            if vid and (vid == ss_active or ss_active.endswith("_" + vid) or ss_active.endswith(vid)):
                match = v
                break
        if match:
            state["preview_version_id"] = match["version_id"]
            final_loss = match.get("final_loss")
            tally.passed(
                8,
                f"ready+active via system_status: vid={match['version_id']!r} "
                f"tts_active={ss_active!r} final_loss={final_loss}",
            )
            return
        tally.failed(
            8,
            f"no version flagged active=true and no match for tts.active_version="
            f"{ss_active!r} among ready vids={[v.get('version_id') for v in ready]}",
        )
        return
    target = active[0]
    state["preview_version_id"] = target.get("version_id")
    final_loss = target.get("final_loss")
    tally.passed(
        8,
        f"ready+active: {target.get('version_id')!r} final_loss={final_loss}",
    )


def step9_preview_returns_wav(
    tally: Tally, base_url: str, state: dict
) -> None:
    """POST preview for the active version, expect audio/wav > 1KB."""
    version_id = state.get("preview_version_id") or state.get("active_version")
    if not version_id:
        tally.failed(9, "no version_id captured to preview")
        return
    url = f"{base_url}/api/training/versions/{version_id}/preview"
    t0 = time.time()
    try:
        status, headers, body = _http_post_json(
            url,
            {"text": "你好，這是我的聲音測試"},
            timeout=PREVIEW_TIMEOUT_S,
        )
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tally.failed(9, f"POST {url} failed: {e}")
        return
    elapsed = time.time() - t0
    if status != 200:
        tally.failed(9, f"preview returned HTTP {status} after {elapsed:.1f}s")
        return
    ctype = headers.get("Content-Type", headers.get("content-type", ""))
    if "audio/wav" not in ctype:
        tally.failed(9, f"preview Content-Type={ctype!r} (expected audio/wav)")
        return
    size = len(body)
    if size <= 1024:
        tally.failed(9, f"preview body too small: {size} bytes")
        return
    if elapsed > 10:
        tally.failed(9, f"preview took {elapsed:.1f}s (>10s budget)")
        return
    tally.passed(
        9,
        f"preview OK: {size} bytes audio/wav in {elapsed:.1f}s "
        f"(version={version_id})",
    )


def step10_pytest(tally: Tally, skip: bool) -> None:
    """Run pytest, assert 0 failures."""
    if skip:
        tally.manual(10, "pytest skipped via --skip-pytest")
        return
    python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        tally.failed(10, f"python not found at {python}")
        return
    cmd = [str(python), "-m", "pytest", "tests/", "-q"]
    env_extras = {"USE_QWEN_ASR": "false", "USE_MOCK_TTS": "true"}
    import os

    env = {**os.environ, **env_extras}
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        tally.failed(10, "pytest timed out after 600s")
        return
    tail = result.stdout.splitlines()[-5:] + result.stderr.splitlines()[-5:]
    summary = " | ".join(line.strip() for line in tail if line.strip())
    if result.returncode == 0 and "failed" not in (result.stdout + result.stderr).lower():
        tally.passed(10, f"pytest exit 0 — {summary[:140]}")
    elif "0 failed" in result.stdout:
        tally.passed(10, f"pytest reports 0 failed — {summary[:140]}")
    else:
        tally.failed(
            10,
            f"pytest exit {result.returncode}: {summary[:200]}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Automated demo-readiness smoke test. "
            "Returns 0 iff every automatic step passes."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--expect-version",
        default=None,
        help=(
            "Pin tts.active_version to this string. "
            "Default: accept whatever is currently active."
        ),
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip step 10 (pytest is ~3min — useful for quick re-runs).",
    )
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    tally = Tally()
    state: dict[str, Any] = {}

    step1_system_status_active_version(tally, base_url, state, args.expect_version)
    step2_resources_within_budget(tally, base_url, state)
    step3_ui_serves_status_bar(tally, base_url)
    step4_manual_mic_test(tally)
    step5_manual_tts_before_llm_done(tally)
    step6_manual_barge_in(tally)
    step7_recordings_processed(tally, base_url)
    step8_training_versions_ready_active(tally, base_url, state)
    step9_preview_returns_wav(tally, base_url, state)
    step10_pytest(tally, skip=args.skip_pytest)

    print("")
    print(
        f"{tally.passes} PASS, {tally.manuals} MANUAL, {tally.fails} FAIL"
    )
    return 0 if tally.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
