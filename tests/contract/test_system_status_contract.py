"""
Contract tests for GET /api/system/status.

Pins the response shape so the UI's polling can rely on a stable
schema. Two paths exercised: nothing-training and training-active.
The training-active simulation mutates the test-isolated TrainingService
to return is_training=True.
"""
from __future__ import annotations

import pytest

from app.api._system import SystemStatusResponse

pytestmark = pytest.mark.contract


class TestSystemStatus:
    def test_shape_on_idle_system(self, client, assert_matches_schema):
        r = client.get("/api/system/status")
        assert r.status_code == 200
        parsed = assert_matches_schema(SystemStatusResponse, r.json())
        # The probes never raise; defaults are sensible.
        assert parsed.disk_free_gb >= 0
        assert parsed.training.active is False
        # `vram.available` reflects whether CUDA is callable in the test
        # process. Either way the rest of the shape is filled.
        assert isinstance(parsed.vram.available, bool)
        assert parsed.vram.used_mb >= 0
        assert parsed.vram.total_mb >= 0

    def test_shape_when_training_active(self, client, isolated_data, monkeypatch):
        """If TrainingService.get_training_status reports active, the
        endpoint surfaces the version_id + persona_id + (optional)
        progress fields."""
        from app.api._dependencies import get_training_service

        # Override the dependency to return a fake service.
        class FakeService:
            class _Repo:
                def read_progress(self, version_id):
                    from app.services.training_service.models import (
                        ProgressStatus,
                        TrainingProgressSnapshot,
                    )
                    return TrainingProgressSnapshot(
                        version_id=version_id,
                        status=ProgressStatus.training,
                        current_epoch=7,
                        total_epochs=30,
                        current_loss=9.63,
                        progress_pct=23,
                    )

            repository = _Repo()

            def get_training_status(self):
                return {
                    "is_training": True,
                    "version_id": "v_test_123",
                    "persona_id": "xiao_s",
                    "status": "training",
                }

        client.app.dependency_overrides[get_training_service] = lambda: FakeService()
        try:
            r = client.get("/api/system/status")
            assert r.status_code == 200
            body = r.json()
            assert body["training"]["active"] is True
            assert body["training"]["version_id"] == "v_test_123"
            assert body["training"]["persona_id"] == "xiao_s"
            assert body["training"]["current_epoch"] == 7
            assert body["training"]["total_epochs"] == 30
            assert body["training"]["progress_pct"] == 23
            assert body["training"]["current_loss"] == pytest.approx(9.63)
        finally:
            client.app.dependency_overrides.pop(get_training_service, None)

    def test_unknown_field_rejected(self, client):
        """Response model has extra='forbid' — drift surfaces here."""
        r = client.get("/api/system/status")
        body = r.json()
        # All top-level keys must be exactly the model fields.
        assert set(body.keys()) == {"vram", "tts", "asr_ready", "training", "disk_free_gb"}
