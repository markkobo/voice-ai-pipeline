"""
Unit tests for app/config.py path resolution.

These pin the env-var override + legacy-fallback behavior so future changes
to deployment paths surface in CI instead of mysteriously breaking on a
fresh box. Backstop for the audit-flagged "8 hardcoded /workspace paths".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import config


@pytest.fixture(autouse=True)
def _reset_caches():
    """Each test starts with a clean lru_cache so env-var changes take effect."""
    config.reset_caches()
    yield
    config.reset_caches()


class TestDataRoot:
    def test_env_var_overrides_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "explicit"))
        assert config.data_root() == tmp_path / "explicit"

    def test_legacy_workspace_when_present(self, monkeypatch):
        monkeypatch.delenv("DATA_ROOT", raising=False)
        # On THIS host /workspace exists (the symlink to the repo). The legacy
        # branch should fire.
        if config._legacy_workspace_available():
            assert config.data_root() == Path("/workspace/voice-ai-pipeline/data")
        else:
            # On a host without /workspace, falls back to ./data.
            assert config.data_root() == Path("data").resolve()

    def test_dependent_paths_chain_from_data_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        assert config.recordings_dir() == tmp_path / "recordings"
        assert config.raw_dir() == tmp_path / "recordings" / "raw"
        assert config.denoised_dir() == tmp_path / "recordings" / "denoised"
        assert config.enhanced_dir() == tmp_path / "recordings" / "enhanced"
        assert config.voice_profiles_dir() == tmp_path / "voice_profiles"


class TestModelsDir:
    def test_models_env_var_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODELS_DIR", str(tmp_path / "m"))
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
        assert config.models_dir() == tmp_path / "m"

    def test_models_defaults_under_data_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MODELS_DIR", raising=False)
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        assert config.models_dir() == tmp_path / "models"


class TestLogDir:
    def test_log_env_var_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        assert config.log_dir() == tmp_path / "logs"

    def test_log_default_per_environment(self, monkeypatch):
        monkeypatch.delenv("LOG_DIR", raising=False)
        if config._legacy_workspace_available():
            assert config.log_dir() == Path("/workspace/voice-ai-pipeline/logs")
        else:
            assert config.log_dir() == Path("/tmp/voice-ai-logs")


class TestCacheBehavior:
    def test_reset_caches_picks_up_env_changes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "first"))
        first = config.data_root()
        # Same call returns cached value.
        assert config.data_root() is first
        # Change env, reset, expect new value.
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "second"))
        assert config.data_root() == tmp_path / "first"  # still cached
        config.reset_caches()
        assert config.data_root() == tmp_path / "second"
