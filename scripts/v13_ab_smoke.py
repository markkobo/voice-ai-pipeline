"""V13 — generate V12 vs V13 audio side-by-side for blind A/B listening.

For each test prompt, produces:
    data/training/test_v13/ab_smoke/<prompt_idx>_v12.wav
    data/training/test_v13/ab_smoke/<prompt_idx>_v13.wav

V12 path: load the merged custom-voice model (existing production
          inference path, V12 SFT result).
V13 path: load the BASE model, attach the V13 PEFT adapter on the
          talker, call generate_voice_clone with a reference WAV.

Listen side-by-side via /ui/v13_review (works) or any local player.

Run (server must be stopped — single GPU):
    bash scripts/restart.sh --stop
    .venv/bin/python scripts/v13_ab_smoke.py --persona test
    bash scripts/restart.sh

ETA: ~3-5 min wall-clock for 6 prompts × 2 models.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v13_ab_smoke")

ROOT = Path(__file__).resolve().parent.parent
TARGET_SR = 24000

# Test prompts — mirror V13_IMPLEMENTATION_PLAN §5 buckets.
PROMPTS = [
    # Taiwan-vocab (the main accent test)
    ("01_taiwan_short", "好啦，不要生氣嘛。"),
    ("02_taiwan_narrative", "我每次想起阿嬤，就想起她炒菜的時候哼的那首歌。"),
    ("03_taiwan_casual", "那個齁，我跟你講，這件事其實沒那麼複雜啦。"),
    # Generic Mandarin (sanity check — shouldn't regress)
    ("04_generic_short", "今天天氣很好。"),
    ("05_generic_long", "我們正在開發一個讓家人可以保留長輩聲音的系統。"),
    # English (verify bilingual capability isn't lost)
    ("06_english_short", "Hi, how was your day?"),
]


def find_merged_v12(persona: str) -> Optional[Path]:
    models_dir = ROOT / "data/models"
    candidates = sorted(models_dir.glob(f"merged_qwen3_tts_{persona}_v12_*"))
    if not candidates:
        return None
    return candidates[-1]


def find_v13_adapter(persona: str, version: str = "v13-lora") -> Optional[Path]:
    p = ROOT / "data/training" / persona / "versions" / version / "adapter"
    return p if p.exists() else None


def find_reference_audio(persona: str) -> Optional[Path]:
    base = ROOT / "data/recordings/denoised"
    candidates = sorted(p for p in base.glob(f"*_{persona}_*/audio.wav"))
    return candidates[0] if candidates else None


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(str(path), sr, audio)


def generate_v12(merged_dir: Path, prompts: list[tuple[str, str]],
                 out_dir: Path, device: str, persona: str) -> None:
    """V12 path — load merged custom-voice model + generate_custom_voice."""
    log.info("V12: loading merged model %s", merged_dir.name)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    log.info("V12: model loaded, tts_model_type=%s", model.model.tts_model_type)
    # custom_voice requires a speaker id (the persona name baked into the
    # merged config). Confirm the persona is in the supported list.
    try:
        supported = model.model.get_supported_speakers()
        log.info("V12: supported speakers: %s", supported)
    except Exception:
        supported = [persona]
    speaker_id = persona if persona in supported else (supported[0] if supported else persona)

    for tag, text in prompts:
        out_path = out_dir / f"{tag}_v12.wav"
        if out_path.exists():
            log.info("V12: skip (exists) %s", out_path.name)
            continue
        log.info("V12: synthesizing %s (speaker=%s) ...", tag, speaker_id)
        try:
            audios, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker_id,
                language="auto",
            )
        except Exception as e:
            log.error("V12: failed on %s: %s", tag, e, exc_info=True)
            continue
        audio = audios[0] if isinstance(audios, list) else audios
        if isinstance(audio, tuple):
            audio = audio[0]
        save_wav(out_path, np.asarray(audio), sr)
        log.info("V12: wrote %s (%.2fs)", out_path.name, len(audio) / sr)

    # Free GPU before V13 loads
    del model
    torch.cuda.empty_cache()


def generate_v13(adapter_dir: Path, ref_audio_path: Path,
                 prompts: list[tuple[str, str]], out_dir: Path,
                 device: str) -> None:
    """V13 path — base model + PEFT adapter, generate_voice_clone with ref."""
    log.info("V13: loading base model + adapter %s", adapter_dir.parent.name)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from peft import PeftModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    log.info("V13: base loaded, tts_model_type=%s", model.model.tts_model_type)

    # Attach the adapter to the talker
    log.info("V13: attaching PEFT adapter to talker...")
    model.model.talker = PeftModel.from_pretrained(
        model.model.talker, str(adapter_dir)
    )
    log.info("V13: adapter attached")

    # Load reference audio (raw float32 24kHz)
    sr_ref, audio_ref = wavfile.read(str(ref_audio_path))
    if audio_ref.dtype == np.int16:
        audio_ref = audio_ref.astype(np.float32) / 32768.0
    elif audio_ref.dtype == np.int32:
        audio_ref = audio_ref.astype(np.float32) / 2147483648.0
    if audio_ref.ndim > 1:
        audio_ref = audio_ref.mean(axis=1)
    if sr_ref != TARGET_SR:
        from scipy import signal as sps
        audio_ref = sps.resample(
            audio_ref, int(len(audio_ref) * TARGET_SR / sr_ref)
        ).astype(np.float32)
        sr_ref = TARGET_SR
    # Cap to 15s — long ref slows clone significantly
    audio_ref = audio_ref[: 15 * sr_ref]
    log.info("V13: ref audio %.2fs", len(audio_ref) / sr_ref)

    for tag, text in prompts:
        out_path = out_dir / f"{tag}_v13.wav"
        if out_path.exists():
            log.info("V13: skip (exists) %s", out_path.name)
            continue
        log.info("V13: synthesizing %s ...", tag)
        try:
            # x_vector_only_mode=True → only the speaker_embedding from
            # ref audio is used (no ICL ref_text needed). Closest match
            # to the V12 custom_voice path's "baked speaker_embedding".
            audios, sr = model.generate_voice_clone(
                text=text,
                language="auto",
                ref_audio=(audio_ref, sr_ref),
                x_vector_only_mode=True,
                # Hard cap on generation length — without this the
                # LoRA-adapted talker can fail to emit the codec EOS
                # token and produce minutes of garbage. 200 frames
                # @ 12Hz codec = ~16s, generous for our 6 short prompts.
                max_new_tokens=200,
            )
        except Exception as e:
            log.error("V13: failed on %s: %s", tag, e, exc_info=True)
            continue
        audio = audios[0] if isinstance(audios, list) else audios
        if isinstance(audio, tuple):
            audio = audio[0]
        save_wav(out_path, np.asarray(audio), sr)
        log.info("V13: wrote %s (%.2fs)", out_path.name, len(audio) / sr)

    del model
    torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True)
    p.add_argument("--v13_version", default="v13-lora")
    p.add_argument("--ref_audio", default=None,
                   help="reference WAV for V13 voice_clone; default first denoised file")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    merged_v12 = find_merged_v12(args.persona)
    if not merged_v12:
        log.error("No merged V12 model for persona %r", args.persona)
        return 1
    log.info("V12 merged: %s", merged_v12)

    adapter = find_v13_adapter(args.persona, args.v13_version)
    if not adapter:
        log.error("No V13 adapter at %s",
                  ROOT / "data/training" / args.persona / "versions" / args.v13_version / "adapter")
        return 1
    log.info("V13 adapter: %s", adapter)

    ref_path = Path(args.ref_audio) if args.ref_audio else find_reference_audio(args.persona)
    if not ref_path or not ref_path.exists():
        log.error("No reference audio for persona %r", args.persona)
        return 1
    log.info("V13 ref: %s", ref_path)

    # Use a version-tagged subdir so V13 and V13.1 results don't collide.
    out_dir = ROOT / "data/training" / f"{args.persona}_v13" / f"ab_smoke_{args.v13_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    start = time.time()
    log.info("=" * 70); log.info("V12 GENERATION"); log.info("=" * 70)
    generate_v12(merged_v12, PROMPTS, out_dir, device, args.persona)

    log.info("=" * 70); log.info("V13 GENERATION"); log.info("=" * 70)
    generate_v13(adapter, ref_path, PROMPTS, out_dir, device)

    log.info("Total wall-clock: %.1f min", (time.time() - start) / 60)
    log.info("Listen via /ui/v13_review or direct files at %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
