"""
Recording filesystem path helpers.

Pure path math: given a folder name (or listener/persona/timestamp parts),
build the on-disk paths for a recording's raw / denoised / enhanced audio,
metadata.json, transcription.txt, and speakers/ folder.

History note — this module used to own a side-channel "recordings index" that
scanned ``RAW_DIR`` on every call to ``list_all_recordings()`` and assigned
fresh random UUIDs to every folder it found. That caused silent
``recording_id`` drift between upload time and pipeline lookup
(``_find_recording``) — see the e0ae9b0 incident. The single source of truth
for "which recordings exist" is now :class:`JsonRecordingsRepository`
(``app/services/recordings/repository.py``). This file is intentionally pure
path math with no I/O beyond an optional storage-stats sweep.

The module-level path constants (``RAW_DIR``, ``DENOISED_DIR`` ...) are kept
as re-exports of :mod:`app.config` so the test conftest can monkeypatch them
per test for isolation. New code should prefer :mod:`app.config` directly.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid


# Base data directory — resolved through app.config which honors DATA_ROOT
# env var, falls back to /workspace/voice-ai-pipeline/data on production
# deployments (entrypoint.sh), then to ./data for local dev.
#
# These are module-level constants so the test conftest (`isolated_data`)
# can monkeypatch them per-test. New production code should import from
# `app.config` directly instead.
from app import config as _config

DATA_DIR = _config.data_root()
RECORDINGS_DIR = _config.recordings_dir()
RAW_DIR = _config.raw_dir()
DENOISED_DIR = _config.denoised_dir()
ENHANCED_DIR = _config.enhanced_dir()
VOICE_PROFILES_DIR = _config.voice_profiles_dir()
MODELS_DIR = _config.models_dir()


# Valid IDs (also defined as class attributes in RecordingPaths). Kept as
# module constants for back-compat with imports from
# `app.services.recordings` (`__init__.py` re-exports these).
VALID_LISTENER_IDS = {"child", "mom", "dad", "friend", "reporter", "elder", "default"}
VALID_PERSONA_IDS = {"xiao_s", "caregiver", "elder_gentle", "elder_playful"}


def _parse_folder_name(folder_name: str) -> Optional[tuple[str, str, str]]:
    """Best-effort parse of ``{listener_id}_{persona_id}_{timestamp}``.

    ``persona_id`` may itself contain underscores (e.g. ``xiao_s``), so we
    greedily match the longest valid persona prefix. Returns
    ``(listener_id, persona_id, timestamp)`` on success, ``None`` if the
    folder name does not match the expected shape — in which case the caller
    should treat the folder as opaque.
    """
    parts = folder_name.split("_")
    if len(parts) < 3:
        return None
    listener_id = parts[0]
    if listener_id not in VALID_LISTENER_IDS:
        return None
    for i in range(1, len(parts)):
        candidate = "_".join(parts[1:i + 1])
        if candidate in VALID_PERSONA_IDS:
            timestamp_parts = parts[i + 1:]
            if not timestamp_parts:
                return None
            return listener_id, candidate, "_".join(timestamp_parts)
    return None


class RecordingPaths:
    """Pure path helper for a single recording.

    Two construction forms are supported:

    1. ``RecordingPaths(folder_name="child_xiao_s_20260329_120000")`` —
       preferred for production code that already knows the folder from the
       repository. ``listener_id`` / ``persona_id`` / ``timestamp`` are
       best-effort-parsed from the folder; ``recording_id`` defaults to
       ``None`` unless the caller passes it explicitly.

    2. ``RecordingPaths(listener_id=..., persona_id=..., timestamp=...)`` —
       legacy form retained so existing unit tests keep working. Validates
       the listener/persona against the known sets. When ``timestamp`` is
       omitted, ``datetime.now()`` is used. When ``recording_id`` is omitted,
       a fresh UUID is assigned. **Never call this form from production code
       — it is the historical landmine that the e0ae9b0 incident fixed.**
    """

    VALID_LISTENER_IDS = VALID_LISTENER_IDS
    VALID_PERSONA_IDS = VALID_PERSONA_IDS

    def __init__(
        self,
        listener_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        recording_id: Optional[str] = None,
        *,
        folder_name: Optional[str] = None,
    ):
        if folder_name is not None:
            # Preferred new API — caller knows the folder name already.
            self.folder_name = folder_name
            parsed = _parse_folder_name(folder_name)
            if parsed is not None:
                p_listener, p_persona, p_ts = parsed
                self.listener_id = listener_id or p_listener
                self.persona_id = persona_id or p_persona
                self.timestamp = timestamp or p_ts
            else:
                # Folder name doesn't follow the convention — keep whatever
                # the caller passed (may be None). Pure path methods don't
                # care, but legacy `metadata.py::_load_or_create` does fall
                # back to these when seeding a new metadata.json.
                self.listener_id = listener_id
                self.persona_id = persona_id
                self.timestamp = timestamp
            self.recording_id = recording_id
            return

        # Legacy API — validate listener/persona, build folder from parts.
        if listener_id not in self.VALID_LISTENER_IDS:
            raise ValueError(f"Invalid listener_id: {listener_id}")
        if persona_id not in self.VALID_PERSONA_IDS:
            raise ValueError(f"Invalid persona_id: {persona_id}")

        self.listener_id = listener_id
        self.persona_id = persona_id
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_id = recording_id or str(uuid.uuid4())
        self.folder_name = f"{listener_id}_{persona_id}_{self.timestamp}"

    # ------------------------------------------------------------------
    # Pure path properties — no I/O, no UUID assignment, no caching.
    # ------------------------------------------------------------------
    @property
    def raw_folder(self) -> Path:
        return RAW_DIR / self.folder_name

    @property
    def denoised_folder(self) -> Path:
        return DENOISED_DIR / self.folder_name

    @property
    def enhanced_folder(self) -> Path:
        return ENHANCED_DIR / self.folder_name

    @property
    def raw_audio_path(self) -> Path:
        return self.raw_folder / "audio.wav"

    @property
    def denoised_audio_path(self) -> Path:
        return self.denoised_folder / "audio.wav"

    @property
    def enhanced_audio_path(self) -> Path:
        return self.enhanced_folder / "audio.wav"

    @property
    def metadata_path(self) -> Path:
        """Metadata stored in raw folder (same as WAV)."""
        return self.raw_folder / "metadata.json"

    @property
    def transcription_path(self) -> Path:
        """Transcription as plain text (easy to move/copy)."""
        return self.raw_folder / "transcription.txt"

    @property
    def speakers_folder(self) -> Path:
        """Folder for extracted speaker audio files."""
        return self.raw_folder / "speakers"

    # ------------------------------------------------------------------
    # Filesystem operations — still pure of any indexing side-effects.
    # ------------------------------------------------------------------
    def create_folders(self) -> None:
        """Create all recording folders."""
        for folder in (self.raw_folder, self.denoised_folder, self.enhanced_folder, self.speakers_folder):
            folder.mkdir(parents=True, exist_ok=True)

    def save_audio(self, source_path: Path, stage: str = "raw") -> Path:
        """Copy ``source_path`` to the appropriate stage folder."""
        if stage == "raw":
            dest = self.raw_audio_path
        elif stage == "denoised":
            dest = self.denoised_audio_path
        elif stage == "enhanced":
            dest = self.enhanced_audio_path
        else:
            raise ValueError(f"Invalid stage: {stage}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        return dest

    def delete_all(self) -> None:
        """Delete all folders and files for this recording."""
        for folder in (self.raw_folder, self.denoised_folder, self.enhanced_folder):
            if folder.exists():
                shutil.rmtree(folder)


def get_storage_stats() -> dict:
    """Sweep the on-disk folders and report total bytes per stage.

    Pure read-only filesystem stat — does not touch the recordings index and
    does not construct any RecordingPaths objects (no UUID drift risk).
    """
    stats = {
        "raw_size_bytes": 0,
        "denoised_size_bytes": 0,
        "enhanced_size_bytes": 0,
        "total_recordings": 0,
    }

    if RAW_DIR.exists():
        for folder in RAW_DIR.iterdir():
            if folder.is_dir():
                stats["total_recordings"] += 1
                for f in folder.rglob("*"):
                    if f.is_file():
                        stats["raw_size_bytes"] += f.stat().st_size

    if DENOISED_DIR.exists():
        for folder in DENOISED_DIR.iterdir():
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.is_file():
                        stats["denoised_size_bytes"] += f.stat().st_size

    if ENHANCED_DIR.exists():
        for folder in ENHANCED_DIR.iterdir():
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.is_file():
                        stats["enhanced_size_bytes"] += f.stat().st_size

    return stats
