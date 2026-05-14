"""
Regenerate any binary fixtures that live under tests/fixtures/.

Most audio is built on-demand by the wav_bytes() factory in conftest.py,
so this script is currently only needed if we add committed binary fixtures
(e.g. corrupt WAVs that exercise specific decode paths).

Run: .venv/bin/python tests/fixtures/_generate.py
"""
from __future__ import annotations

import struct
import wave
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent
AUDIO_DIR = FIXTURE_DIR / "audio"


def _write_silence_wav(path: Path, duration_seconds: float, sample_rate: int = 24000) -> None:
    """Write a mono 16-bit silent WAV at the given duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for _ in range(n_frames):
            w.writeframes(struct.pack("<h", 0))


def _write_corrupt_wav(path: Path) -> None:
    """Write bytes that look like a WAV header but with an inconsistent size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # RIFF header with garbage payload — decoders should reject this.
    header = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 32
    path.write_bytes(header + b"corrupt_audio_payload" * 4)


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    _write_silence_wav(AUDIO_DIR / "short_1s.wav", duration_seconds=1.0)
    _write_corrupt_wav(AUDIO_DIR / "corrupt.wav")
    print(f"Wrote fixtures to {AUDIO_DIR}")


if __name__ == "__main__":
    main()
