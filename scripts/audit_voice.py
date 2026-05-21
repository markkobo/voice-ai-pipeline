#!/usr/bin/env python3
"""Audit a training audio file or trained-model source for voice-cloning quality.

Use cases:
  - Before training: check if a recording is broadband enough to clone well.
  - After training: when a preview sounds bad, see whether the source audio
    is to blame (low bandwidth, clipping, noise) vs the training pipeline.

Usage:
  scripts/audit_voice.py <wav_path>
  scripts/audit_voice.py --version <version_id>          # looks up training source
  scripts/audit_voice.py --recording <recording_id>      # all speakers in a recording

The TTS speaker_encoder extracts a timbre embedding from this audio. If the
audio is band-limited, distorted, or silent, the embedding will be too — and
the cloned voice will inherit those defects. Garbage in, garbage out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def red(s): return _color(s, "31")
def yellow(s): return _color(s, "33")
def green(s): return _color(s, "32")
def bold(s): return _color(s, "1")


def audit_wav(wav_path: Path) -> dict:
    """Return a dict of audio quality metrics + warnings."""
    info = sf.info(str(wav_path))
    audio, sr = sf.read(str(wav_path))
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio ** 2).mean()))
    crest = peak / max(rms, 1e-9)
    clipped = int((np.abs(audio) > 0.999).sum())

    # Frame-based silence/loudness analysis (100ms windows)
    frame_size = int(0.1 * sr)
    n_frames = len(audio) // max(frame_size, 1)
    frame_peaks = np.array([
        np.abs(audio[i*frame_size:(i+1)*frame_size]).max()
        for i in range(n_frames)
    ]) if n_frames else np.array([0.0])
    silent_pct = float((frame_peaks < 0.005).mean() * 100)
    loud_pct = float((frame_peaks > 0.5).mean() * 100)

    # Spectral characteristics — analyze first 10s of voiced content
    voiced_mask = frame_peaks > 0.02  # rough voiced detection
    voiced_idx = np.where(voiced_mask)[0]
    if len(voiced_idx) > 0:
        # Concatenate first ~10 seconds of voiced frames for a stable spectrum
        chunks = []
        for i in voiced_idx[:int(10 * sr / frame_size)]:
            chunks.append(audio[i*frame_size:(i+1)*frame_size])
        sample = np.concatenate(chunks) if chunks else audio[:sr*10]
    else:
        sample = audio[:sr*10]
    sample = sample[:sr*10]  # cap at 10s
    spec = np.abs(np.fft.rfft(sample * np.hanning(len(sample))))
    freqs = np.fft.rfftfreq(len(sample), 1/sr)
    power = spec ** 2

    def band_ratio(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(power[m].sum() / max(power.sum(), 1e-9) * 100)

    e_low = band_ratio(0, 4000)
    e_mid = band_ratio(4000, 8000)
    e_high = band_ratio(8000, 12000)
    cumul = power.cumsum() / max(power.sum(), 1e-9)
    idx_95 = int(np.searchsorted(cumul, 0.95))
    eff_bw = float(freqs[idx_95])

    return {
        "path": str(wav_path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": info.duration,
        "format": f"{info.format}/{info.subtype}",
        "peak": peak,
        "peak_dbfs": 20 * np.log10(max(peak, 1e-9)),
        "rms": rms,
        "rms_dbfs": 20 * np.log10(max(rms, 1e-9)),
        "crest_factor": crest,
        "clipped_samples": clipped,
        "silent_pct": silent_pct,
        "loud_pct": loud_pct,
        "energy_0_4khz_pct": e_low,
        "energy_4_8khz_pct": e_mid,
        "energy_8_12khz_pct": e_high,
        "effective_bandwidth_hz": eff_bw,
    }


def warnings_for(m: dict) -> list[str]:
    out = []
    if m["sample_rate"] < 24000:
        out.append(red(f"Sample rate {m['sample_rate']} Hz < 24000 — upsampled audio cannot recover lost bandwidth"))
    if m["peak"] > 0.99 or m["clipped_samples"] > 100:
        out.append(red(f"Clipping detected ({m['clipped_samples']} samples) — distortion baked into the source"))
    elif m["peak"] < 0.2:
        out.append(yellow(f"Very quiet (peak {m['peak']:.2f}) — speaker_embedding may underrepresent timbre"))
    if m["silent_pct"] > 30:
        out.append(yellow(f"{m['silent_pct']:.0f}% silence — VAD/diarization may have left long gaps"))
    if m["effective_bandwidth_hz"] < 4000:
        out.append(red(f"Effective bandwidth {m['effective_bandwidth_hz']:.0f} Hz < 4 kHz — telephone-grade or heavily low-passed source; clone will be muffled"))
    elif m["effective_bandwidth_hz"] < 6500:
        out.append(yellow(f"Effective bandwidth {m['effective_bandwidth_hz']:.0f} Hz < 6.5 kHz — narrow band, clone will lack brightness"))
    if m["energy_8_12khz_pct"] < 0.5 and m["sample_rate"] >= 24000:
        out.append(yellow(f"Almost no energy above 8 kHz ({m['energy_8_12khz_pct']:.2f}%) — source lost high frequencies somewhere"))
    if m["crest_factor"] < 8:
        out.append(yellow(f"Crest factor {m['crest_factor']:.1f}x is low — source may be heavily compressed/normalized, losing dynamics"))
    return out


def print_report(m: dict) -> None:
    print()
    print(bold(f"  {m['path']}"))
    print(f"  Format:      {m['format']} @ {m['sample_rate']} Hz, {m['channels']} ch, {m['duration_seconds']:.1f}s ({m['duration_seconds']/60:.1f} min)")
    print(f"  Peak/RMS:    {m['peak']:.3f} / {m['rms']:.4f}  ({m['peak_dbfs']:.1f} / {m['rms_dbfs']:.1f} dBFS)")
    print(f"  Crest:       {m['crest_factor']:.1f}x   Clipped: {m['clipped_samples']}   Silent: {m['silent_pct']:.0f}%   Loud: {m['loud_pct']:.0f}%")
    print(f"  Spectrum:    0-4kHz {m['energy_0_4khz_pct']:5.1f}%   4-8kHz {m['energy_4_8khz_pct']:5.1f}%   8-12kHz {m['energy_8_12khz_pct']:5.1f}%")
    print(f"  Eff. BW (95%): {m['effective_bandwidth_hz']:.0f} Hz")
    warns = warnings_for(m)
    if warns:
        print()
        for w in warns:
            print(f"  ⚠️  {w}")
    else:
        print()
        print(f"  {green('No quality warnings.')}")
    print()


def resolve_from_version(version_id: str) -> list[Path]:
    """Map version_id → list of source wav files (one per speaker_id used)."""
    sys.path.insert(0, str(REPO_ROOT))
    from app.services.training_service.repository import JsonTrainingRepository
    from app import config as cfg
    repo = JsonTrainingRepository(cfg.models_dir())
    v = repo.get_or_none(version_id)
    if v is None:
        raise SystemExit(f"Version {version_id!r} not found")
    manifest = repo.get_manifest(version_id)
    paths = []
    if manifest:
        for rec in manifest.recordings:
            paths.append(Path(rec.audio_path))
    else:
        for seg_id in v.segment_ids_used:
            rec_id, _, spk_id = seg_id.partition("_")
            # Look in raw/{folder_name}/speakers/{spk_id}.wav
            for folder in (cfg.recordings_dir() / "raw").iterdir():
                wav = folder / "speakers" / f"{spk_id}.wav"
                if wav.exists():
                    paths.append(wav)
                    break
    return paths


def resolve_from_recording(recording_id: str) -> list[Path]:
    sys.path.insert(0, str(REPO_ROOT))
    from app import config as cfg
    root = cfg.recordings_dir() / "raw"
    out = []
    for folder in root.iterdir():
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        if meta.get("recording_id") == recording_id:
            for wav in (folder / "speakers").glob("*.wav"):
                out.append(wav)
            break
    return out


def audit_pipeline_stages(recording_id: str) -> None:
    """Compare raw → denoised → enhanced → speakers/* across pipeline stages.

    Answers two questions:
    1. Did the pipeline degrade the source? (each stage's effective bandwidth
       should be ≥ the previous stage's, with maybe a small drop from the
       denoise noise floor.)
    2. Is enhance actually doing anything? (output identical to denoised
       means the Sepformer fallback fired silently.)
    """
    sys.path.insert(0, str(REPO_ROOT))
    from app import config as cfg

    folder = None
    for f in cfg.raw_dir().iterdir():
        meta_file = f / "metadata.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        if meta.get("recording_id") == recording_id or f.name == recording_id:
            folder = f
            break
    if folder is None:
        print(red(f"Recording {recording_id!r} not found."))
        return

    stages = [
        ("raw", cfg.raw_dir() / folder.name / "audio.wav"),
        ("denoised", cfg.denoised_dir() / folder.name / "audio.wav"),
        ("enhanced", cfg.enhanced_dir() / folder.name / "audio.wav"),
    ]
    speakers = list((folder / "speakers").glob("*.wav"))

    print()
    print(bold(f"  Pipeline stages for {folder.name}:"))
    print()
    print(f"  {'STAGE':18} {'SR':>5}  {'PEAK':>5}  {'RMS dB':>7}  {'SILENT':>6}  {'EFF.BW':>9}  {'>4 kHz':>8}  {'>8 kHz':>8}")
    print(f"  {'-'*18} {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}")

    prev_hash = None
    for label, path in stages + [(f"speakers/{w.stem}", w) for w in speakers]:
        if not path.exists():
            print(f"  {label:18} {red('(missing)')}")
            continue
        m = audit_wav(path)
        # Detect bit-identical stage (silent-noop fallback)
        import hashlib
        h = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        marker = ""
        if prev_hash and prev_hash == h:
            marker = red("  ← identical to previous stage (enhance no-op)")
        prev_hash = h
        print(
            f"  {label:18} {m['sample_rate']:>5}  "
            f"{m['peak']:>5.2f}  {m['rms_dbfs']:>7.1f}  "
            f"{m['silent_pct']:>5.1f}%  "
            f"{m['effective_bandwidth_hz']:>7.0f}Hz  "
            f"{m['energy_4_8khz_pct']:>7.1f}%  "
            f"{m['energy_8_12khz_pct']:>7.1f}%"
            f"{marker}"
        )

    # Read metadata.json for stage status + errors
    meta_file = folder / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        ps = meta.get("processing_steps", {})
        print()
        print(bold(f"  Pipeline step status:"))
        for step in ("denoise", "enhance", "diarize", "transcribe"):
            entry = ps.get(step, {})
            status = entry.get("status", "?")
            err = entry.get("error_message")
            line = f"  {step:12} {status:8}"
            if err:
                line += f"  {red(err[:80])}"
            print(line)

    # Final read on the file that training would consume
    if speakers:
        print()
        print(bold("  Training input audit (the file the trainer reads):"))
        m = audit_wav(speakers[0])
        warns = warnings_for(m)
        if warns:
            for w in warns:
                print(f"  ⚠️  {w}")
        else:
            print(f"  {green('No quality warnings — source is suitable for training.')}")
    print()


def backfill_all_recordings() -> None:
    """Re-audit every recording's speakers/*.wav and write voice_audit into
    metadata.json. Idempotent: re-running just overwrites with fresh metrics.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from app import config as cfg
    from app.services.recordings.quality import audit_voice_training_quality

    n = 0
    for folder in cfg.raw_dir().iterdir():
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            print(red(f"Skipping {folder.name}: corrupt metadata.json"))
            continue
        segs = meta.get("speaker_segments") or []
        if not segs:
            continue
        # Build per-speaker audits (one per unique speaker_id, since speakers/
        # is per-speaker not per-segment).
        speaker_audits = {}
        for spk_wav in (folder / "speakers").glob("*.wav"):
            try:
                speaker_audits[spk_wav.stem] = audit_voice_training_quality(spk_wav)
            except Exception as e:
                print(red(f"Audit failed for {spk_wav}: {e}"))
        if not speaker_audits:
            continue
        # Stamp every segment for that speaker with the audit
        for seg in segs:
            sid = seg.get("speaker_id")
            if sid in speaker_audits:
                seg["voice_audit"] = speaker_audits[sid]
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        n += 1
        summary = ", ".join(f"{k}={v['level']}" for k, v in speaker_audits.items())
        print(f"  {folder.name}: audited {len(speaker_audits)} speakers — {summary}")
    print()
    print(green(f"Backfilled {n} recordings."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="Path to a .wav file to audit.")
    ap.add_argument("--version", help="Audit the source audio used by a trained version.")
    ap.add_argument("--recording", help="Audit all speakers in a recording (by recording_id).")
    ap.add_argument("--pipeline", help="Compare raw / denoised / enhanced / speakers stages of a recording (by folder name or recording_id).")
    ap.add_argument("--backfill-all", action="store_true", help="Run voice_audit on every existing recording's speakers and write into metadata.json.")
    args = ap.parse_args()

    if args.pipeline:
        audit_pipeline_stages(args.pipeline)
        return

    if args.backfill_all:
        backfill_all_recordings()
        return

    if args.path:
        files = [Path(args.path)]
    elif args.version:
        files = resolve_from_version(args.version)
    elif args.recording:
        files = resolve_from_recording(args.recording)
    else:
        ap.print_help()
        sys.exit(2)

    if not files:
        print(red("No audio files resolved."))
        sys.exit(1)

    for f in files:
        if not f.exists():
            print(red(f"Missing: {f}"))
            continue
        m = audit_wav(f)
        print_report(m)


if __name__ == "__main__":
    main()
