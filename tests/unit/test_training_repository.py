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

    def test_progress_missing_version_id_is_injected_from_path(self, repo, tmp_path):
        """The training_job subprocess historically wrote progress.json
        without `version_id`; the reader must inject it from the path
        rather than silently failing validation. Regression for the
        2026-05-21 "v1_20260521_104108_618646 stuck in training" bug:
        subprocess wrote `{"status":"ready",...}` (no version_id),
        TrainingProgressSnapshot rejected it as missing-field, and the
        index entry never flipped from training → ready.
        """
        v = _make_version("v1_x", lora_root=tmp_path)
        repo.save(v)
        progress_path = Path(v.lora_path) / "progress.json"
        progress_path.write_text(
            json.dumps(
                {
                    # NOTE: no "version_id" — matches the bad on-disk format.
                    "status": "ready",
                    "current_epoch": 10,
                    "progress_pct": 100,
                    "persona_id": "test",
                    "training_type": "lora",
                    "current_loss": 0.0,
                    "best_loss": 0.0,
                }
            )
        )
        snap = repo.read_progress("v1_x")
        assert snap is not None
        assert snap.version_id == "v1_x"
        assert snap.status.value == "ready"
        assert snap.progress_pct == 100


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


class TestDeleteMergedPathSafety:
    """Regression tests for the 2026-05-25 incident: deleting one SFT version
    must NOT remove another version's merged model dir, even when the two
    versions' legacy `parts[:3]`-derived merged-dir names would have
    collided.

    The fix has two layers of defense:
      1. `delete()` uses `target.merged_path` as the single source of
         truth — no re-derivation from `lora_path.name`.
      2. Before rmtree-ing the merged dir, it scans surviving versions
         for any whose stored `merged_path` matches; if so, it refuses
         to delete and logs a warning.
    """

    def _sft(
        self,
        version_id: str,
        lora_dir: Path,
        merged_dir: Path,
        persona_id: str = "xiao_s",
    ) -> TrainingVersion:
        """Build a custom_voice (SFT) TrainingVersion with both paths set."""
        lora_dir.mkdir(parents=True, exist_ok=True)
        merged_dir.mkdir(parents=True, exist_ok=True)
        # Drop a sentinel file in merged_dir so we can detect rmtree.
        (merged_dir / "model.safetensors").write_text("dummy")
        v = TrainingVersion(
            version_id=version_id,
            persona_id=persona_id,
            recording_ids_used=["rec-1"],
            lora_path=str(lora_dir),
            merged_path=str(merged_dir.resolve()),
            model_type="custom_voice",
            created_at=datetime.now(timezone.utc),
        )
        return v

    def test_delete_does_not_wipe_other_versions_shared_merged_dir(
        self, repo, tmp_path
    ):
        """The 2026-05-25 scenario: two SFTs whose lora dirs are different
        timestamps (`xiao_s_v2_20260514_...` and `xiao_s_v2_20260525_...`)
        would BOTH have legacy-collapsed to `merged_qwen3_tts_xiao_s_v2`.
        Suppose they both ended up pointing at the same on-disk merged dir
        (e.g. because one was trained on top of the other's output). When
        we delete one, the OTHER's merged dir must survive.
        """
        # Legacy-style collision: both lora names collapse to
        # "xiao_s_v2" under parts[:3].
        lora_a = tmp_path / "xiao_s_v2_20260514_120000_111111"
        lora_b = tmp_path / "xiao_s_v2_20260525_214017_209227"
        shared_merged = tmp_path / "merged_qwen3_tts_xiao_s_v2"

        v_old = self._sft("v2_20260514_120000_111111", lora_a, shared_merged)
        v_new = self._sft("v2_20260525_214017_209227", lora_b, shared_merged)
        repo.save(v_old)
        repo.save(v_new)

        # Delete the NEW (bad) version. The OLD version's merged dir
        # must still exist on disk.
        repo.delete("v2_20260525_214017_209227")

        assert not repo.exists("v2_20260525_214017_209227")
        # Surviving version's merged_path must still resolve.
        survivor = repo.get("v2_20260514_120000_111111")
        assert survivor.merged_path == str(shared_merged.resolve())
        assert shared_merged.exists(), (
            "Shared merged dir was wiped — the 2026-05-25 catastrophe "
            "would have recurred."
        )
        assert (shared_merged / "model.safetensors").exists()

    def test_delete_uses_stored_merged_path_not_derived(self, repo, tmp_path):
        """If the version's stored `merged_path` points somewhere OTHER
        than what the old `parts[:3]` rule would have derived, the delete
        must follow the stored path, not re-derive.
        """
        lora = tmp_path / "xiao_s_v2_20260525_214017_209227"
        # Stored merged dir uses the NEW naming convention (full lora name).
        merged_new = tmp_path / "merged_qwen3_tts_xiao_s_v2_20260525_214017_209227"
        # Legacy-derived name; an unrelated dir we don't want touched.
        legacy_derived = tmp_path / "merged_qwen3_tts_xiao_s_v2"
        legacy_derived.mkdir()
        (legacy_derived / "unrelated.bin").write_text("keep me")

        v = self._sft("v2_20260525_214017_209227", lora, merged_new)
        repo.save(v)
        repo.delete("v2_20260525_214017_209227")

        # The actually-stored merged dir got removed.
        assert not merged_new.exists()
        # The legacy-derived dir was NEVER touched (no re-derivation).
        assert legacy_derived.exists()
        assert (legacy_derived / "unrelated.bin").exists()

    def test_delete_skips_when_no_merged_path(self, repo, tmp_path):
        """SFT version with merged_path=None: don't guess, just log + skip."""
        lora = tmp_path / "xiao_s_v3_20260525_111111_111111"
        lora.mkdir()
        # A plausible legacy-derived merged dir sitting nearby — we must
        # NOT touch it.
        plausible = tmp_path / "merged_qwen3_tts_xiao_s_v3"
        plausible.mkdir()
        (plausible / "sentinel").write_text("dont touch")

        v = TrainingVersion(
            version_id="v3_20260525_111111_111111",
            persona_id="xiao_s",
            recording_ids_used=["rec-1"],
            lora_path=str(lora),
            merged_path=None,  # explicitly unset
            model_type="custom_voice",
            created_at=datetime.now(timezone.utc),
        )
        repo.save(v)
        repo.delete("v3_20260525_111111_111111")

        # Lora dir is gone.
        assert not lora.exists()
        # Plausible merged dir was NOT derived/removed.
        assert plausible.exists()
        assert (plausible / "sentinel").exists()


class TestSweepStranded:
    """TrainingService.sweep_stranded() — reconcile orphaned `training` versions
    on startup. Regression for the 2026-05-21 "stuck in training" bug.
    """

    def _service(self, repo, tmp_path):
        from app.services.training_service.audio_resolver import AudioResolver
        from app.services.training_service.service import TrainingService

        class _AcceptingValidator:
            def is_valid(self, _): return True
            def list_ids(self): return set()

        class _NullResolver(AudioResolver):
            def resolve_segments(self, segment_ids): return []

        return TrainingService(
            repository=repo,
            persona_validator=_AcceptingValidator(),
            audio_resolver=_NullResolver(),
            models_dir=tmp_path / "models",
        )

    def test_sweep_flips_terminal_progress_to_ready(self, repo, tmp_path):
        """progress.json says ready → index entry gets flipped to ready
        (the exact path the v1_20260521_104108_618646 bug took: training
        actually finished but the parent never updated index.json).
        """
        v = _make_version("v1_x", lora_root=tmp_path)
        v.status = VersionStatus.training
        repo.save(v)
        # Subprocess-style progress.json *without* version_id — the bad
        # on-disk format the reader-side fix now tolerates.
        progress_path = Path(v.lora_path) / "progress.json"
        progress_path.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "current_epoch": 10,
                    "progress_pct": 100,
                    "persona_id": "xiao_s",
                    "training_type": "lora",
                    "current_loss": 0.5,
                    "best_loss": 0.5,
                }
            )
        )
        svc = self._service(repo, tmp_path)
        n = svc.sweep_stranded()
        assert n == 1
        assert repo.get("v1_x").status == VersionStatus.ready

    def test_sweep_flips_no_progress_to_failed(self, repo, tmp_path):
        """No progress.json + status=training → subprocess died before
        writing terminal state; the sweep should flip to failed so the
        UI shows "失敗 (interrupted)" instead of "訓練中" forever.
        """
        v = _make_version("v1_x", lora_root=tmp_path)
        v.status = VersionStatus.training
        repo.save(v)
        svc = self._service(repo, tmp_path)
        n = svc.sweep_stranded()
        assert n == 1
        got = repo.get("v1_x")
        assert got.status == VersionStatus.failed
        assert got.error_message and "interrupted" in got.error_message

    def test_sweep_leaves_ready_alone(self, repo, tmp_path):
        v = _make_version("v1_x", lora_root=tmp_path)
        v.status = VersionStatus.ready
        repo.save(v)
        svc = self._service(repo, tmp_path)
        assert svc.sweep_stranded() == 0
        assert repo.get("v1_x").status == VersionStatus.ready

    def test_sweep_backfills_merged_path_when_dir_exists(self, repo, tmp_path):
        """When the merged model directory is on disk but the index entry
        has no `merged_path` (parent died after subprocess merged but
        before updating the index), the sweep should record it.
        Activation can't find the model otherwise.
        """
        # Create a lora_path with the canonical naming pattern.
        v = _make_version("v1_x", persona_id="test", lora_root=tmp_path)
        # _make_version sets lora_path to {lora_root}/{persona}_{version_id}.
        # merge_lora() derives the merged dir from the first 3 underscore
        # parts of that directory name → here 'test_v1_x' has only 2 parts,
        # so we need a directory that splits into ≥3 parts to mirror real
        # life (e.g. test_v1_20260521).
        lora_dir = tmp_path / "test_v1_20260521"
        lora_dir.mkdir()
        merged_dir = tmp_path / "merged_qwen3_tts_test_v1_20260521"
        merged_dir.mkdir()
        v.lora_path = str(lora_dir)
        v.status = VersionStatus.training
        # Subprocess-style progress.json (no version_id, status=ready).
        (lora_dir / "progress.json").write_text(
            json.dumps({"status": "ready", "current_epoch": 10, "progress_pct": 100})
        )
        repo.save(v)
        svc = self._service(repo, tmp_path)
        assert svc.sweep_stranded() == 1
        got = repo.get("v1_x")
        assert got.status == VersionStatus.ready
        assert got.merged_path == str(merged_dir.resolve())
