"""
Central path + runtime configuration.

Every module that needs a filesystem location should import from here
instead of hardcoding `/workspace/voice-ai-pipeline/...`. The legacy
default still resolves to that path so production deployments keep working;
DATA_ROOT / MODELS_DIR / LOG_DIR env vars override on dev boxes.

Resolution order for each setting:
1. Explicit env var (DATA_ROOT, MODELS_DIR, LOG_DIR, etc.)
2. /workspace/voice-ai-pipeline/... if /workspace exists (production /
   entrypoint.sh path)
3. Relative-to-CWD fallback for ad-hoc dev / tests

The functions are wrapped in `lru_cache` so the resolution happens once
per process, but tests can swap env vars by clearing the cache (see
`app.config.reset_caches`).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — overridable via env var.
# ---------------------------------------------------------------------------
_LEGACY_WORKSPACE = Path("/workspace/voice-ai-pipeline")


def _legacy_workspace_available() -> bool:
    """True if /workspace exists (production / entrypoint.sh deployment)."""
    return _LEGACY_WORKSPACE.parent.exists()


@lru_cache(maxsize=1)
def data_root() -> Path:
    """Root for all persistent state: recordings, personas, listeners, models.

    Env: DATA_ROOT. Default: /workspace/voice-ai-pipeline/data (production)
    or ./data (dev fallback).
    """
    explicit = os.environ.get("DATA_ROOT")
    if explicit:
        return Path(explicit)
    if _legacy_workspace_available():
        return _LEGACY_WORKSPACE / "data"
    return Path("data").resolve()


@lru_cache(maxsize=1)
def models_dir() -> Path:
    """LoRA + merged model directory.

    Env: MODELS_DIR. Default: {data_root}/models.
    """
    explicit = os.environ.get("MODELS_DIR")
    if explicit:
        return Path(explicit)
    return data_root() / "models"


@lru_cache(maxsize=1)
def recordings_dir() -> Path:
    """Recordings parent dir. Holds raw/ denoised/ enhanced/ subdirs."""
    return data_root() / "recordings"


@lru_cache(maxsize=1)
def raw_dir() -> Path:
    return recordings_dir() / "raw"


@lru_cache(maxsize=1)
def denoised_dir() -> Path:
    return recordings_dir() / "denoised"


@lru_cache(maxsize=1)
def enhanced_dir() -> Path:
    return recordings_dir() / "enhanced"


@lru_cache(maxsize=1)
def voice_profiles_dir() -> Path:
    return data_root() / "voice_profiles"


@lru_cache(maxsize=1)
def personas_dir() -> Path:
    """Per-persona artifact root.

    Holds per-persona corpus (RFC_M6 Phase 0) and future per-persona LoRA
    adapters. Lives at {data_root}/personas/<persona_id>/...
    """
    return data_root() / "personas"


@lru_cache(maxsize=1)
def log_dir() -> Path:
    """Application log directory.

    Env: LOG_DIR. Default: /workspace/voice-ai-pipeline/logs (production)
    or /tmp/voice-ai-logs (dev fallback — /var/log requires sudo).
    """
    explicit = os.environ.get("LOG_DIR")
    if explicit:
        return Path(explicit)
    if _legacy_workspace_available():
        return _LEGACY_WORKSPACE / "logs"
    return Path("/tmp/voice-ai-logs")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def reset_caches() -> None:
    """Clear all cached resolutions. Tests call this after monkeypatching env."""
    for fn in (data_root, models_dir, recordings_dir, raw_dir,
               denoised_dir, enhanced_dir, voice_profiles_dir,
               personas_dir, log_dir):
        fn.cache_clear()
