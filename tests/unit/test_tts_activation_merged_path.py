"""
Unit tests for activate_version path resolution.

Demo-readiness #4: ``qwen_tts_engine.activate_version`` used to derive
the merged-model directory from ``lora_dir.name.split("_")[:3]``. That
encodes the assumption that ``persona_id`` is exactly 2 underscore-
segments (works for ``xiao_s``; breaks for ``elder_gentle_friendly``).

Fix: training_job persists the actual merged path on the
``TrainingVersion`` at completion, and activate_version reads
``version.merged_path`` first, only falling back to the convention for
legacy index.json entries.

These tests pin both code paths and verify the legacy fallback still
loads via the ``from_dict`` filter (i.e. an old index.json that lacks
the field doesn't blow up).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def engine():
    """A FasterQwenTTSEngine without CUDA / model downloads."""
    from app.services.tts.qwen_tts_engine import FasterQwenTTSEngine

    with patch("torch.cuda.is_available", return_value=False):
        return FasterQwenTTSEngine(model_size="1.7B", device="cpu")


def _make_version(
    tmp_path: Path,
    persona_id: str,
    *,
    merged_path: str | None = None,
    create_merged_dir: bool = True,
    create_convention_dir: bool = False,
):
    """Build a TrainingVersion + optionally create the merged model dir on disk."""
    from app.services.training import TrainingVersion

    version_id = "v1_20260520_120000"
    lora_dir = tmp_path / f"{persona_id}_{version_id}"
    lora_dir.mkdir(parents=True)

    if create_merged_dir and merged_path is not None:
        Path(merged_path).mkdir(parents=True, exist_ok=True)
        # activate_version reads config.json if present; absence is fine.

    if create_convention_dir:
        parts = lora_dir.name.split("_")
        version_base = "_".join(parts[:3])
        convention_dir = lora_dir.parent / f"merged_qwen3_tts_{version_base}"
        convention_dir.mkdir(parents=True, exist_ok=True)

    return TrainingVersion(
        version_id=version_id,
        persona_id=persona_id,
        status="ready",
        lora_path=str(lora_dir),
        merged_path=merged_path,
    )


class TestActivateVersionMergedPath:
    """activate_version() must use version.merged_path when set."""

    def test_stored_merged_path_takes_precedence(self, engine, tmp_path):
        """
        Multi-underscore persona (`elder_gentle_friendly`): the legacy
        `parts[:3]` slice would yield `elder_gentle_friendly` — but
        without the version-id suffix in `version_base`, the convention
        dir name doesn't match where the trainer actually wrote. With
        `merged_path` stored explicitly, activation just works.
        """
        # Note: we don't actually trigger model load — activate_version
        # short-circuits before that because _is_loaded=False.
        explicit_merged = tmp_path / "merged_qwen3_tts_elder_gentle_friendly_v1"
        version = _make_version(
            tmp_path,
            persona_id="elder_gentle_friendly",
            merged_path=str(explicit_merged),
            create_merged_dir=True,
        )

        from app.services.training import TrainingVersion

        with patch("app.services.training.get_version_manager") as mock_mgr:
            mock_mgr.return_value.get_version.return_value = version
            engine.activate_version(version.version_id)

        assert engine._merged_model_path == str(explicit_merged.resolve())

    def test_legacy_fallback_when_merged_path_unset(self, engine, tmp_path):
        """Old version (no merged_path) → falls back to convention path."""
        version = _make_version(
            tmp_path,
            persona_id="xiao_s",
            merged_path=None,
            create_convention_dir=True,
        )
        expected_convention = (
            tmp_path / "merged_qwen3_tts_xiao_s_v1"
        )
        assert expected_convention.exists(), "fixture sanity"

        with patch("app.services.training.get_version_manager") as mock_mgr:
            mock_mgr.return_value.get_version.return_value = version
            engine.activate_version(version.version_id)

        assert engine._merged_model_path == str(expected_convention.resolve())

    def test_stored_path_missing_falls_back_to_convention(
        self, engine, tmp_path, caplog
    ):
        """
        If the persisted merged_path is stale (dir was moved/deleted),
        log a warning and fall back to the convention. Lets the system
        self-heal for partially-corrupted state.
        """
        bogus = tmp_path / "does_not_exist"
        version = _make_version(
            tmp_path,
            persona_id="xiao_s",
            merged_path=str(bogus),
            create_merged_dir=False,
            create_convention_dir=True,
        )
        expected_convention = tmp_path / "merged_qwen3_tts_xiao_s_v1"

        import logging

        with caplog.at_level(logging.WARNING):
            with patch("app.services.training.get_version_manager") as mock_mgr:
                mock_mgr.return_value.get_version.return_value = version
                engine.activate_version(version.version_id)

        assert engine._merged_model_path == str(expected_convention.resolve())
        assert any(
            "Stored merged_path" in r.getMessage() for r in caplog.records
        )


class TestLegacyIndexJsonLoad:
    """index.json written before #4 lands must still load cleanly."""

    def test_from_dict_tolerates_missing_merged_path(self):
        """TrainingVersion.from_dict({...no merged_path...}) → field is None."""
        from app.services.training import TrainingVersion

        legacy = {
            "version_id": "v1_20260514_152118_456516",
            "persona_id": "xiao_s",
            "status": "ready",
            "lora_path": "/data/models/xiao_s_v1_20260514_152118",
            "rank": 16,
            "learning_rate": 1e-4,
            "num_epochs": 30,
            "batch_size": 4,
            "final_loss": 9.6312,
        }
        v = TrainingVersion.from_dict(legacy)
        assert v.merged_path is None
        assert v.version_id == "v1_20260514_152118_456516"

    def test_from_dict_round_trips_merged_path(self):
        """New writes persist merged_path; reads round-trip it."""
        from app.services.training import TrainingVersion

        new_form = {
            "version_id": "v1_20260520_120000",
            "persona_id": "elder_gentle_friendly",
            "status": "ready",
            "lora_path": "/data/models/elder_gentle_friendly_v1_20260520_120000",
            "merged_path": "/data/models/merged_qwen3_tts_elder_gentle_friendly_v1",
        }
        v = TrainingVersion.from_dict(new_form)
        assert (
            v.merged_path
            == "/data/models/merged_qwen3_tts_elder_gentle_friendly_v1"
        )

    def test_pydantic_mirror_tolerates_missing_merged_path(self):
        """
        Same coverage on the Pydantic-side mirror in training_service —
        the new repository writes via this model and must accept old
        rows (per the `extra="ignore"` posture from task 62C).
        """
        from app.services.training_service.models import TrainingVersion as PydV

        legacy_row = {
            "version_id": "v1_20260514_152118_456516",
            "persona_id": "xiao_s",
            "status": "ready",
            "lora_path": "/data/models/xiao_s_v1_20260514_152118",
            "rank": 16,
            "num_recordings_used": 18,  # legacy-only field
        }
        v = PydV.model_validate(legacy_row)
        assert v.merged_path is None

    def test_update_version_status_persists_merged_path(self, tmp_path):
        """
        End-to-end through VersionManager: the kwarg flows from the
        training job into index.json on disk.
        """
        from app.services.training import VersionManager

        with patch("app.services.training.MODELS_DIR", tmp_path):
            with patch(
                "app.services.training.VERSION_INDEX_FILE",
                tmp_path / "index.json",
            ):
                mgr = VersionManager()
                version = mgr.create_version(
                    "elder_gentle_friendly",
                    recording_ids=["r1", "r2", "r3", "r4", "r5"],
                )
                mgr.update_version_status(
                    version.version_id,
                    status="ready",
                    final_loss=0.5,
                    training_time_seconds=120,
                    merged_path="/data/models/merged_qwen3_tts_xyz",
                )
                # Reload to confirm persisted to disk
                mgr2 = VersionManager()
                reloaded = mgr2.get_version(version.version_id)
                assert (
                    reloaded.merged_path
                    == "/data/models/merged_qwen3_tts_xyz"
                )
