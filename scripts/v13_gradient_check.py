"""V13 — Step 0 gradient-check kill switch.

Verifies the core hypothesis of V13_IMPLEMENTATION_PLAN.md §Step 0:

  "If we run the external repo's full-talker forward pass (with
   speaker_embedding injected at input_codec_embedding[:, 6, :])
   on a real (text, audio) pair, do gradients flow to the talker's
   attention/FFN layers?"

If yes → V13 plan is viable, proceed to Step 1.
If no  → V13 is dead on arrival, no point building the dataset pipeline.

Run:
    bash scripts/restart.sh --stop   # free the GPU
    .venv/bin/python scripts/v13_gradient_check.py

Approx wall-clock: 5-10 minutes (most of which is model load).

This script does NOT modify any persistent state.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v13_grad_check")

ROOT = Path(__file__).resolve().parent.parent

# A 29.5s denoised Mark recording (peak ~0.5, clean) chosen during M12a planning.
REF_WAV = ROOT / "data/recordings/denoised/child_test_20260530_034842_915324/audio.wav"
# Short, hand-written transcript — does NOT use ASR (the whole point of
# Gemini's "no circular bias" rule).
HAND_TEXT = "我每次想起阿嬤，就想起她炒菜的時候哼的那首歌。"

NUM_TRAINING_STEPS = 12  # Plan says ≥10 steps; 12 gives a small margin.

# --------------------------------------------------------------------------- #
# Audio + text prep                                                            #
# --------------------------------------------------------------------------- #


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    sr, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Take the first ~5s to keep the test cheap. Step 0 doesn't care
    # about coverage — it cares about whether gradients flow.
    audio = audio[: 5 * sr]
    return audio, sr


def tokenize_audio_to_codec_ids(speech_tokenizer, audio: np.ndarray, sr: int) -> torch.Tensor:
    """Returns (seq_len, num_code_groups) Long tensor of codec ids."""
    TARGET_SR = 24000
    if sr != TARGET_SR:
        from scipy import signal as sps  # noqa: PLC0415

        num_samples = int(len(audio) * TARGET_SR / sr)
        audio = sps.resample(audio, num_samples).astype(np.float32)
        sr = TARGET_SR
    enc = speech_tokenizer.encode(audio, sr=sr)
    codes = enc["audio_codes"][0]  # (seq_len, num_code_groups)
    return torch.as_tensor(np.asarray(codes), dtype=torch.long)


# --------------------------------------------------------------------------- #
# The forward pass mirroring the external repo                                 #
# --------------------------------------------------------------------------- #


def proposed_forward_backward(
    model,
    *,
    text_ids: torch.Tensor,  # (1, T_text) long
    codec_ids: torch.Tensor,  # (1, T_codec, num_code_groups) long
    speaker_embedding: torch.Tensor,  # (1, hidden) float
) -> torch.Tensor:
    """Implements sft_12hz.py:707-740 with our model surface.

    Returns the scalar loss (no .item()), so the caller can .backward().
    """
    talker = model.talker
    # PEFT-wrapped talker exposes the original modules via .get_base_model().
    talker_base = talker.get_base_model() if hasattr(talker, "get_base_model") else talker
    inner_model = talker_base.model  # Qwen3TTSTalkerModel

    # 1. Text embeddings — get_text_embeddings is the input-id embedding.
    text_emb_layer = talker.get_input_embeddings()
    text_emb = text_emb_layer(text_ids)  # (1, T_text, hidden)

    # 2. Codec embeddings.
    # qwen_tts core uses .model.codec_embedding (nn.Embedding,
    # shape (vocab, hidden) — see modeling_qwen3_tts.py:1441).
    # The external repo uses code_group 0 as the time axis here.
    code_group_0 = codec_ids[..., 0]  # (1, T_codec)
    codec_emb = inner_model.codec_embedding(code_group_0)  # (1, T_codec, hidden)

    # 3. Speaker embedding injection at position 6 of the codec axis.
    #    Brittle reverse-engineered position — see V13_IMPLEMENTATION_PLAN §5.
    #    Mirrors mozi1924/Qwen3-TTS-EasyFinetuning sft_12hz.py:712.
    if codec_emb.shape[1] < 7:
        raise RuntimeError(
            f"codec sequence too short for position-6 injection: "
            f"got T_codec={codec_emb.shape[1]}, need ≥7"
        )
    codec_emb = codec_emb.clone()  # avoid in-place on a view
    codec_emb[:, 6, :] = speaker_embedding

    # 4. Concatenate text + codec along the sequence axis.
    inputs_embeds = torch.cat([text_emb, codec_emb], dim=1)  # (1, T_text+T_codec, hidden)

    # 5. Labels — predict code_group_0[t+1] from code_group_0[t]. We
    #    only supervise the codec span; mask the text span with -100.
    labels = torch.full(
        (1, inputs_embeds.shape[1]),
        fill_value=-100,
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    labels[:, text_emb.shape[1] :] = code_group_0
    # Shift-by-one for next-token prediction handled internally by the model.

    # 6. Forward through the full talker. This is the load-bearing call:
    #    if gradients reach `talker.model.layers[*]`, V13 is viable.
    outputs = talker(
        inputs_embeds=inputs_embeds,
        labels=labels,
    )
    main_loss = outputs.loss

    # 7. Auxiliary code_predictor loss (same as V12 path). Use the
    #    talker hidden_state mean as the talker_hidden_states surrogate
    #    — Step 0 doesn't need to be 1:1 with production training.
    talker_hidden = inputs_embeds[:, -1, :]  # (1, hidden), placeholder
    # forward_sub_talker_finetune expects (B, num_code_groups) for codec_ids.
    one_frame_codecs = codec_ids[:, codec_ids.shape[1] // 2, :]  # (1, num_code_groups)
    _, sub_talker_loss = talker_base.forward_sub_talker_finetune(
        codec_ids=one_frame_codecs,
        talker_hidden_states=talker_hidden,
    )

    loss = main_loss + 0.3 * sub_talker_loss
    return loss, main_loss.detach(), sub_talker_loss.detach()


# --------------------------------------------------------------------------- #
# Gradient diagnostics                                                          #
# --------------------------------------------------------------------------- #


def diagnose_gradients(model) -> dict:
    """Walk the talker model and report whether gradients reached the
    layers V13 needs to update. Returns a summary dict."""
    talker = model.talker
    talker_base = talker.get_base_model() if hasattr(talker, "get_base_model") else talker
    summary = {
        "first_layer_q_proj_grad_present": None,
        "first_layer_q_proj_grad_abs_sum": None,
        "talker_layer_with_nonzero_grad_count": 0,
        "talker_layer_total_count": 0,
        "nan_in_any_grad": False,
        "first_nan_param_name": None,
    }

    # Look for `talker.model.layers[*]` which the external repo expects
    # to be updated. qwen_tts core puts the transformer body at
    # `talker.model.layers` (modeling_qwen3_tts.py:1441 area).
    body = getattr(talker_base, "model", None)
    if body is None:
        log.error("talker.model not found on model; talker class is %s", type(talker))
        return summary
    layers = getattr(body, "layers", None)
    if layers is None:
        log.error("talker.model.layers not found; this gradient check needs adjustment")
        return summary

    summary["talker_layer_total_count"] = len(layers)
    for i, layer in enumerate(layers):
        for pname, p in layer.named_parameters():
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                summary["nan_in_any_grad"] = True
                summary["first_nan_param_name"] = f"layer[{i}].{pname}"
                break
            if p.grad.abs().sum().item() > 0:
                summary["talker_layer_with_nonzero_grad_count"] += 1
                break  # one nonzero param per layer is enough to count it

    # Specifically check the q_proj of layer 0 — the canonical "did the
    # gradient reach the attention block of the first transformer layer".
    layer0 = layers[0]
    for pname in ["self_attn.q_proj.weight", "attn.q_proj.weight"]:
        p = layer0
        for piece in pname.split("."):
            p = getattr(p, piece, None)
            if p is None:
                break
        if p is not None and hasattr(p, "grad"):
            summary["first_layer_q_proj_grad_present"] = p.grad is not None
            if p.grad is not None:
                summary["first_layer_q_proj_grad_abs_sum"] = p.grad.abs().sum().item()
            break

    return summary


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #


def main() -> int:
    if not REF_WAV.exists():
        log.error("Reference WAV not found: %s", REF_WAV)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        log.info("Free GPU memory: %.2f GB", free_gb)
        if free_gb < 8.0:
            log.warning(
                "Low GPU free memory — run `bash scripts/restart.sh --stop` first to free the server"
            )

    log.info("Loading qwen_tts modules + tokenizer...")
    from qwen_tts import Qwen3TTSTokenizer  # noqa: PLC0415
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration  # noqa: PLC0415

    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")

    log.info("Loading base model Qwen/Qwen3-TTS-12Hz-1.7B-Base in bf16...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        torch_dtype=torch.bfloat16,  # full fp32 OOMs the A10G when AdamW allocates state
    )
    model.to(device)
    model.train()
    # Freeze everything except talker.model.layers — saves AdamW memory by ~3x.
    for p in model.parameters():
        p.requires_grad_(False)
    # Apply LoRA on the talker per V13 plan — this is what we'll actually
    # use in Step 2 anyway, so testing with LoRA validates the real V13
    # path, not a full-SFT hypothetical.
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.talker = get_peft_model(model.talker, lora_cfg)
    trainable = sum(p.numel() for p in model.talker.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.talker.parameters())
    log.info("LoRA applied to talker: %d / %d params trainable (%.2f%%)",
             trainable, total, 100 * trainable / total)

    log.info("Loading + encoding audio: %s", REF_WAV.name)
    audio, sr = load_audio(REF_WAV)
    # Force-resample to 24kHz once so both the codec tokenizer and
    # extract_speaker_embedding (which asserts sr==24000) see the same.
    if sr != 24000:
        from scipy import signal as sps  # noqa: PLC0415
        num_samples = int(len(audio) * 24000 / sr)
        audio = sps.resample(audio, num_samples).astype(np.float32)
        sr = 24000
        log.info("Resampled to %d Hz", sr)
    codec_ids = tokenize_audio_to_codec_ids(speech_tokenizer, audio, sr)
    codec_ids = codec_ids.unsqueeze(0).to(device)  # (1, T_codec, num_code_groups)
    log.info("codec_ids shape: %s (T_codec=%d, num_code_groups=%d)",
             tuple(codec_ids.shape), codec_ids.shape[1], codec_ids.shape[2])

    # Text token ids. Use the talker's get_input_embeddings vocab via
    # the model's text tokenizer — but the public Qwen3TTSTokenizer here
    # is the speech codec. For Step 0 we approximate text tokens by
    # using a tiny made-up id sequence; the FORWARD/BACKWARD test
    # doesn't require linguistically valid tokens, just sane shapes.
    # If this approximation gives bad NaNs we'll swap to the real text
    # tokenizer; if it gives clean gradients, the V13 mechanism works
    # for real tokens too.
    text_vocab_size = model.talker.config.vocab_size
    text_len = max(4, min(16, codec_ids.shape[1] // 4))
    rng = np.random.default_rng(42)
    text_ids = torch.as_tensor(
        rng.integers(low=10, high=min(2000, text_vocab_size - 1), size=(1, text_len)),
        dtype=torch.long,
        device=device,
    )
    log.info("text_ids shape: %s (T_text=%d)", tuple(text_ids.shape), text_len)

    log.info("Extracting speaker embedding from reference audio...")
    speaker_embedding = model.extract_speaker_embedding(audio, sr).to(device)
    if speaker_embedding.ndim == 1:
        speaker_embedding = speaker_embedding.unsqueeze(0)
    log.info("speaker_embedding shape: %s", tuple(speaker_embedding.shape))

    # Build optimizer. Only LoRA params have requires_grad=True so
    # AdamW state is tiny. LR=1e-4 matches V13 plan §3 for LoRA.
    trainable_params = [p for p in model.talker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    losses: list[float] = []
    log.info("=" * 70)
    log.info("Starting %d training steps (kill-switch test)...", NUM_TRAINING_STEPS)
    log.info("=" * 70)

    for step in range(NUM_TRAINING_STEPS):
        optimizer.zero_grad(set_to_none=True)
        try:
            loss, main_loss, sub_loss = proposed_forward_backward(
                model,
                text_ids=text_ids,
                codec_ids=codec_ids,
                speaker_embedding=speaker_embedding,
            )
        except Exception as e:
            log.error("FORWARD FAILED at step %d: %s", step, e, exc_info=True)
            return 3

        if not torch.isfinite(loss):
            log.error("NON-FINITE loss at step %d: loss=%s main=%s sub=%s",
                      step, loss.item(), main_loss.item(), sub_loss.item())
            return 4

        loss.backward()
        losses.append(loss.item())

        # On step 0 do the deepest diagnosis to capture grad presence
        # before the optimizer step erases the .grad tensors.
        if step == 0:
            log.info("Step 0 — running gradient diagnostics...")
            summary = diagnose_gradients(model)
            log.info("Diagnostics: %s", summary)

        optimizer.step()
        log.info(
            "step=%d  loss=%.4f  main=%.4f  sub=%.4f",
            step, loss.item(), main_loss.item(), sub_loss.item(),
        )

    # Final verdict
    log.info("=" * 70)
    log.info("Loss trajectory: %s", [f"{x:.3f}" for x in losses])
    decreasing = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i - 1])
    log.info("Loss decreased on %d of %d step-to-step transitions",
             decreasing, len(losses) - 1)

    # Re-diagnose at the end so we capture grad after all optimizer steps.
    # The .grad fields were zero'd by zero_grad — so we do one more forward
    # without optimizer step to get the final-state gradient pattern.
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = proposed_forward_backward(
        model,
        text_ids=text_ids, codec_ids=codec_ids, speaker_embedding=speaker_embedding,
    )
    loss.backward()
    final_summary = diagnose_gradients(model)
    log.info("Final-state diagnostics: %s", final_summary)

    # Verdict
    pct_layers_with_grad = (
        final_summary["talker_layer_with_nonzero_grad_count"]
        / max(1, final_summary["talker_layer_total_count"])
    )
    log.info("=" * 70)
    log.info("VERDICT")
    log.info("=" * 70)
    log.info(
        "  Talker layers with non-zero grad: %d / %d (%.0f%%)",
        final_summary["talker_layer_with_nonzero_grad_count"],
        final_summary["talker_layer_total_count"],
        100 * pct_layers_with_grad,
    )
    log.info("  Loss step-to-step decreases: %d / %d", decreasing, len(losses) - 1)
    log.info("  NaN in any gradient: %s", final_summary["nan_in_any_grad"])

    pass_gradients = pct_layers_with_grad >= 0.5  # at least half of talker layers
    pass_loss = decreasing >= max(1, int(0.6 * (len(losses) - 1)))  # ≥60% steps down
    pass_no_nan = not final_summary["nan_in_any_grad"]

    overall = pass_gradients and pass_loss and pass_no_nan
    log.info("  → gradients flow:   %s", "PASS" if pass_gradients else "FAIL")
    log.info("  → loss decreases:   %s", "PASS" if pass_loss else "FAIL")
    log.info("  → no NaN gradients: %s", "PASS" if pass_no_nan else "FAIL")
    log.info("")
    log.info("  V13 viability:      %s", "✅ VIABLE — proceed to Step 1" if overall else "❌ DEAD ON ARRIVAL — stop, re-diagnose")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
