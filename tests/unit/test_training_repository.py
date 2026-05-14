"""
Unit tests for the new locked JsonTrainingRepository.

Companions to the contract tests in tests/contract/ — these exercise the
repository in isolation, not through the FastAPI layer.
"""
from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.services.training_service.models import (
    ActiveVersion,
    ManifestRecording,
    TrainingManifest,
    TrainingManifestConfig,
    TrainingType,
    TrainingVersion,
    VersionStatus,
)
from app.services.training_service.repository import (
    JsonTrainingRepository,
    TrainingVersionNotFound,
)


@pytest.fixture
def repo(tmp_path):
    return JsonTrainingRepository(tmp_path / "models")


def _make_version(version_id: str, persona_id: str = "xiao_s", *, lora_root: Path | None = None) -> TrainingVersion:
    v = TrainingVersion(
        version_id=version_id,
        persona_id=persona_id,
        recording_ids_used=["rec-1"],
        created_at=datetime.now(timezone.utc),
    )
    if lora_root is not None:
        lora_path = lora_root / f"{persona_id}_{version_id}"
        lora_path.mkdir(parents=True, exist_ok=True)
        v.lora_path = str(lora_path)
    return v


class TestBasicCrud:
    def test_save_and_get(self, repo, tmp_path):
        v = _make_version("v1_x", lora_root=tmp_path)
        repo.save(v)
        got = repo.get("v1_x")
        assert got.version_id == "v1_x"

    def test_get_or_none_returns_none(self, repo):
        assert repo.get_or_none("nope") is None

    def test_get_raises(self, repo):
        with pytest.raises(TrainingVersionNotFound):
            repo.get("nope")

    def test_list_sorts_newest_first(self, repo, tmp_path):
        v1 = _make_version("v1_a", lora_root=tmp_path)
        v1.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        v2 = _make_version("v1_b", lora_root=tmp_path)
        v2.created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
        repo.save(v1)
        repo.save(v2)
        assert [v.version_id for v in repo.list()] == ["v1_b", "v1_a"]

    def test_list_filters_by_persona(self, repo, tmp_path):
        repo.save(_make_version("v1_a", "xiao_s", lora_root=tmp_path))
        repo.save(_make_version("v1_b", "caregiver", lora_root=tmp_path))
        ids = {v.version_id for v in repo.list(persona_id="xiao_s")}
        assert ids == {"v1_a"}

    def test_update_round_trip(self, repo, tmp_path):
        repo.save(_make_version("v1_x", lora_root=tmp_path))

        def mutate(v: TrainingVersion) -> None:
            v.nickname = "renamed"

        updated = repo.update("v1_x", mutate)
        assert updated.nickname == "renamed"
        assert repo.get("v1_x").nickname == "renamed"

    def test_update_404(self, repo):
        with pytest.raises(TrainingVersionNotFound):
            repo.update("nope", lambda v: None)

    def test_delete(self, repo, tmp_path):
        repo.save(_make_version("v1_x", lora_root=tmp_path))
        repo.delete("v1_x")
        assert not repo.exists("v1_x")

    def test_delete_404(self, repo):
        with pytest.raises(TrainingVersionNotFound):
            repo.delete("nope")


class TestActiveVersionLifecycle:
    def test_set_active_refuses_non_ready(self, repo, tmp_path):
        repo.save(_make_version("v1_x", lora_root=tmp_path))
        with pytest.raises(ValueError, match="status"):
            repo.set_active(ActiveVersion(persona_id="xiao_s", version_id="v1_x"))

    def test_set_active_happy_path(self, repo, tmp_path):
        repo.save(_make_version("v1_x", lora_root=tmp_path))

        def mark_ready(v: TrainingVersion) -> None:
            v.status = VersionStatus.ready

        repo.update("v1_x", mark_ready)
        repo.set_active(ActiveVersion(persona_id="xiao_s", version_id="v1_x"))
        active = repo.get_active("xiao_s")
        assert active is not None
        assert active.version_id == "v1_x"

    def test_delete_active_refused(self, repo, tmp_path):
        repo.save(_make_version("v1_x", lora_root=tmp_path))
        repo.update("v1_x", lambda v: setattr(v, "status", VersionStatus.ready))
        repo.set_active(ActiveVersion(persona_id="xiao_s", version_id="v1_x"))
        with pytest.raises(ValueError, match="active"):
            repo.delete("v1_x")

    def test_clear_active_then_delete(self, repo, tmp_path):
        repo.save(_make_version("v1_x", lora_root=tmp_path))
        repo.update("v1_x", lambda v: setattr(v, "status", VersionStatus.ready))
        repo.set_active(ActiveVersion(persona_id="xiao_s", version_id="v1_x"))
        repo.clear_active_if("v1_x")
        assert repo.get_active("xiao_s") is None
        repo.delete("v1_x")  # no longer active → ok


class TestManifestAndProgress:
    def test_manifest_round_trip(self, repo, tmp_path):
        v = _make_version("v1_x", lora_root=tmp_path)
        repo.save(v)
        manifest = TrainingManifest(
            version_id="v1_x",
            persona_id="xiao_s",
            segment_ids=["rec-1_SPEAKER_00"],
            training_type=TrainingType.lora,
            recordings=[
                ManifestRecording(
                    recording_id="rec-1",
                    folder_name="f",
                    audio_path="/tmp/a.wav",
                    duration_seconds=20.0,
                )
            ],
            total_duration_seconds=20.0,
            training_config=TrainingManifestConfig(
                rank=16,
                learning_rate=1e-4,
                num_epochs=10,
                batch_size=4,
                training_type=TrainingType.lora,
            ),
        )
        repo.save_manifest("v1_x", manifest)
        loaded = repo.get_manifest("v1_x")
        assert loaded is not None
        assert loaded.total_duration_seconds == 20.0
        assert loaded.recordings[0].recording_id == "rec-1"

    def test_progress_round_trip(self, repo, tmp_path):
        v = _make_version("v1_x", lora_root=tmp_path)
        repo.save(v)
        progress_path = Path(v.lora_path) / "progress.json"
        progress_path.write_text(
            json.dumps(
                {
                    "version_id": "v1_x",
                    "status": "training",
                    "current_epoch": 3,
                    "total_epochs": 10,
                    "current_loss": 0.5,
                    "progress_pct": 30,
                }
            )
        )
        snap = repo.read_progress("v1_x")
        assert snap is not None
        assert snap.current_epoch == 3
        assert snap.status.value == "training"

    def test_progress_corrupt_returns_none(self, repo, tmp_path):
        v = _make_version("v1_x", lora_root=tmp_path)
        repo.save(v)
        progress_path = Path(v.lora_path) / "progress.json"
        progress_path.write_text("not json")
        assert repo.read_progress("v1_x") is None


class TestConcurrency:
    """The whole point of this repository over the legacy VersionManager."""

    def test_parallel_updates_all_persist(self, repo, tmp_path):
        v = _make_version("v1_x", lora_root=tmp_path)
        v.segment_ids_used = []
        repo.save(v)

        N = 50

        def worker(i: int) -> None:
            def m(rec: TrainingVersion) -> None:
                rec.segment_ids_used.append(f"rec-{i}_SPEAKER_00")

            repo.update("v1_x", m)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = repo.get("v1_x")
        assert len(final.segment_ids_used) == N, (
            f"Expected {N}, got {len(final.segment_ids_used)}. "
            f"Missing: {[i for i in range(N) if f'rec-{i}_SPEAKER_00' not in final.segment_ids_used]}"
        )

    def test_concurrent_saves_no_lost_versions(self, repo, tmp_path):
        """N threads creating distinct versions should all land in the index."""
        N = 30
        for i in range(N):
            (tmp_path / f"xiao_s_v{i}").mkdir(parents=True, exist_ok=True)

        def worker(i: int) -> None:
            v = TrainingVersion(
                version_id=f"v{i}",
                persona_id="xiao_s",
                recording_ids_used=["rec-1"],
                lora_path=str(tmp_path / f"xiao_s_v{i}"),
                created_at=datetime.now(timezone.utc),
            )
            repo.save(v)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_versions = repo.list()
        assert len(all_versions) == N, (
            f"Expected {N} versions, got {len(all_versions)}. Index may have lost writes."
        )
