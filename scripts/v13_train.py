"""V13 Step 2 — train a persona LoRA on the full talker.

Implements V13_IMPLEMENTATION_PLAN §Step 2 + §Step 3:

  - Loads train_prepared_filtered.jsonl (Step 1 output)
  - Applies LoRA to the talker's transformer layers (q/k/v/o/gate/up/down)
  - For each batch:
      input_text_embedding + input_codec_embedding with
      speaker_embedding injected at position [:, 6, :]
      → talker.forward(inputs_embeds=..., labels=codec_group_0)
      → loss = outputs.loss + 0.3 * sub_talker_loss
  - AdamW LR=1e-4 (LoRA conventional), epochs=3, batch=2 grad_accum=4
  - bf16 (full fp32 OOMs A10G during AdamW state allocation)
  - Saves PEFT adapter at data/training/<persona>/versions/v<version>/adapter/

Run:
    bash scripts/restart.sh --stop      # free GPU
    .venv/bin/python scripts/v13_train.py \
        --persona test --version v13-lora

Estimated time: ~1-2 hour on A10G for 3 epochs × 1800 chunks × bs 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v13_train")

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class V13Dataset(Dataset):
    """Each record: {audio, text, speaker_id, audio_codes (T_codec, num_code_groups)}."""

    def __init__(self, jsonl_path: Path):
        self.records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "audio_codes" not in rec:
                    continue  # need prepared records only
                self.records.append(rec)
        if not self.records:
            raise RuntimeError(f"no records with audio_codes in {jsonl_path}")
        log.info("Dataset: %d records from %s", len(self.records), jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        codec_ids = torch.as_tensor(rec["audio_codes"], dtype=torch.long)  # (T_codec, num_code_groups)
        return {
            "text": rec["text"],
            "audio_rel": rec["audio"],
            "codec_ids": codec_ids,
        }


def collate(batch: list[dict]) -> dict:
    """Pad codec_ids to the longest in the batch. Texts are tokenized
    later (need the processor). Returns lists + padded codec tensor."""
    texts = [b["text"] for b in batch]
    audio_rels = [b["audio_rel"] for b in batch]
    codec_list = [b["codec_ids"] for b in batch]
    max_t = max(c.shape[0] for c in codec_list)
    num_groups = codec_list[0].shape[1]
    B = len(batch)
    codec_padded = torch.full((B, max_t, num_groups), fill_value=0, dtype=torch.long)
    codec_lengths = torch.zeros(B, dtype=torch.long)
    for i, c in enumerate(codec_list):
        codec_padded[i, : c.shape[0]] = c
        codec_lengths[i] = c.shape[0]
    return {"texts": texts, "audio_rels": audio_rels,
            "codec_ids": codec_padded, "codec_lengths": codec_lengths}


# --------------------------------------------------------------------------- #
# Forward pass — mirrors gradient_check                                        #
# --------------------------------------------------------------------------- #

def proposed_forward(
    model,
    processor,
    *,
    texts: list[str],
    codec_ids: torch.Tensor,      # (B, T_codec, num_code_groups)
    codec_lengths: torch.Tensor,  # (B,)
    speaker_embedding: torch.Tensor,  # (1, hidden) — shared across batch (same speaker)
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (loss, main_loss, sub_loss). All scalar tensors."""
    talker = model.talker
    talker_base = talker.get_base_model() if hasattr(talker, "get_base_model") else talker
    inner_model = talker_base.model

    B = codec_ids.shape[0]

    # 1. Tokenize text. Wrap each in the assistant-text template so the
    #    talker sees the same prefix shape as inference time.
    assistant_texts = [f"<|im_start|>assistant\n{t}<|im_end|>\n" for t in texts]
    # Pad text tokens to the batch.
    text_inputs = processor(text=assistant_texts, return_tensors="pt", padding=True)
    text_ids = text_inputs["input_ids"].to(device)         # (B, T_text)
    text_attn = text_inputs["attention_mask"].to(device)   # (B, T_text)

    # 2. Build text embeddings via the talker's TEXT embedding (vocab=151936),
    #    NOT get_input_embeddings() which returns the CODEC embedding (vocab=3072).
    #    Mirrors mozi1924/Qwen3-TTS-EasyFinetuning sft_12hz.py:707
    #    (`unwrap_model.talker.get_text_embeddings()(input_text_ids)`).
    text_emb = talker_base.get_text_embeddings()(text_ids)  # (B, T_text, hidden)

    # 3. Build codec embeddings.
    code_group_0 = codec_ids[..., 0].to(device)  # (B, T_codec)
    codec_emb = inner_model.codec_embedding(code_group_0)  # (B, T_codec, hidden)

    # 4. Speaker-embedding injection at position 6.
    #    Matches mozi1924/Qwen3-TTS-EasyFinetuning sft_12hz.py:712.
    #    If T_codec < 7 for some sample, skip the inject for that row.
    if codec_emb.shape[1] >= 7:
        codec_emb = codec_emb.clone()
        # speaker_embedding is (1, hidden); broadcast across the batch.
        codec_emb[:, 6, :] = speaker_embedding.to(codec_emb.dtype)

    # 5. Concatenate text + codec.
    inputs_embeds = torch.cat([text_emb, codec_emb], dim=1)  # (B, T_text + T_codec, hidden)

    # 6. Build labels — only supervise the codec span.
    T_text = text_emb.shape[1]
    T_codec = codec_emb.shape[1]
    labels = torch.full(
        (B, T_text + T_codec),
        fill_value=-100,
        dtype=torch.long,
        device=device,
    )
    labels[:, T_text:] = code_group_0
    # Mask padded codec positions with -100 so loss is computed only on real codec.
    for b in range(B):
        valid_codec = codec_lengths[b].item()
        if valid_codec < T_codec:
            labels[b, T_text + valid_codec :] = -100

    # 7. Forward — the load-bearing call.
    outputs = talker(inputs_embeds=inputs_embeds, labels=labels)
    main_loss = outputs.loss

    # 8. Sub-talker auxiliary loss on a single codec frame mid-utterance.
    #    Use the (rough) mean of inputs_embeds along time as talker_hidden_states.
    talker_hidden = inputs_embeds.mean(dim=1)  # (B, hidden)
    # forward_sub_talker_finetune wants (B, num_code_groups) for codec_ids.
    mid = codec_ids.shape[1] // 2
    one_frame_codecs = codec_ids[:, mid, :].to(device)
    _, sub_loss = talker_base.forward_sub_talker_finetune(
        codec_ids=one_frame_codecs,
        talker_hidden_states=talker_hidden,
    )

    loss = main_loss + 0.3 * sub_loss
    return loss, main_loss.detach(), sub_loss.detach()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True)
    p.add_argument("--version", default="v13-lora",
                   help="version tag for the saved adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_records", type=int, default=None,
                   help="cap on number of records for trial run")
    p.add_argument("--ref_audio", type=str, default=None,
                   help="path to reference WAV for speaker_embedding (default: first dataset chunk)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        log.info("Free GPU memory: %.2f GB", free_gb)

    # Resolve paths
    dataset_root = ROOT / "data/training" / f"{args.persona}_v13"
    jsonl_path = dataset_root / "train_prepared_filtered.jsonl"
    if not jsonl_path.exists():
        # Fall back to unfiltered if filter wasn't run
        jsonl_path = dataset_root / "train_prepared.jsonl"
        log.warning("Using UNFILTERED dataset: %s", jsonl_path)
    if not jsonl_path.exists():
        log.error("No dataset at %s", jsonl_path)
        return 1

    save_dir = ROOT / "data/training" / args.persona / "versions" / args.version / "adapter"
    save_dir.mkdir(parents=True, exist_ok=True)
    log.info("Adapter will save to: %s", save_dir)

    # Load model
    log.info("Loading qwen_tts...")
    from qwen_tts import Qwen3TTSTokenizer
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
    from transformers import AutoProcessor

    log.info("Loading base model in bf16...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.train()

    # Freeze everything; LoRA will add trainable adapters.
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # Apply LoRA to the talker
    log.info("Applying LoRA (rank=%d)...", args.lora_rank)
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.talker = get_peft_model(model.talker, lora_cfg)
    trainable = sum(p.numel() for p in model.talker.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.talker.parameters())
    log.info("Trainable params: %d / %d (%.2f%%)",
             trainable, total, 100 * trainable / total)

    # Text processor for tokenizing transcripts
    log.info("Loading processor for text tokenization...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    # Speaker embedding — extract once from a reference WAV.
    # Use a healthy ~5-30s sample of this speaker so the embedding is stable.
    log.info("Extracting speaker embedding from reference audio...")
    if args.ref_audio:
        ref_path = Path(args.ref_audio)
    else:
        # Use the first source recording (full-length, before chunking)
        # so the speaker_embedding has plenty of phoneme coverage.
        denoised_root = ROOT / "data/recordings/denoised"
        candidates = sorted(p for p in denoised_root.glob(f"*_{args.persona}_*/audio.wav"))
        if not candidates:
            log.error("No reference WAV for persona %r", args.persona)
            return 1
        ref_path = candidates[0]
    log.info("  ref: %s", ref_path)

    from scipy.io import wavfile
    from scipy import signal as sps
    sr, ref_audio = wavfile.read(str(ref_path))
    if ref_audio.dtype == np.int16:
        ref_audio = ref_audio.astype(np.float32) / 32768.0
    elif ref_audio.dtype == np.int32:
        ref_audio = ref_audio.astype(np.float32) / 2147483648.0
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    if sr != 24000:
        num_samples = int(len(ref_audio) * 24000 / sr)
        ref_audio = sps.resample(ref_audio, num_samples).astype(np.float32)
        sr = 24000
    # Cap to 30s to keep speaker_embedding extraction fast.
    ref_audio = ref_audio[: 30 * sr]
    speaker_embedding = model.extract_speaker_embedding(ref_audio, sr).to(device)
    if speaker_embedding.ndim == 1:
        speaker_embedding = speaker_embedding.unsqueeze(0)
    log.info("  speaker_embedding shape: %s", tuple(speaker_embedding.shape))

    # Build dataset + loader
    dataset = V13Dataset(jsonl_path)
    if args.max_records is not None:
        dataset.records = dataset.records[: args.max_records]
        log.info("Capped dataset to %d records (--max-records)", len(dataset.records))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    # Optimizer — only LoRA params have requires_grad=True.
    trainable_params = [p for p in model.talker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Training loop
    log.info("=" * 70)
    log.info("Starting training: epochs=%d, batch_size=%d, grad_accum=%d, lr=%g",
             args.epochs, args.batch_size, args.grad_accum, args.lr)
    log.info("Steps per epoch (after grad_accum): %d",
             len(loader) // args.grad_accum)
    log.info("=" * 70)

    global_step = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_step_count = 0
        for batch_idx, batch in enumerate(loader):
            try:
                loss, main_loss, sub_loss = proposed_forward(
                    model, processor,
                    texts=batch["texts"],
                    codec_ids=batch["codec_ids"],
                    codec_lengths=batch["codec_lengths"],
                    speaker_embedding=speaker_embedding,
                    device=device,
                )
            except torch.cuda.OutOfMemoryError as e:
                log.error("OOM at batch %d (codec lens=%s): %s",
                          batch_idx, batch["codec_lengths"].tolist(), e)
                torch.cuda.empty_cache()
                continue

            if not torch.isfinite(loss):
                log.warning("non-finite loss at batch %d (skip): main=%.4f sub=%.4f",
                            batch_idx, main_loss.item(), sub_loss.item())
                continue

            # Scale loss by grad_accum so the gradient magnitude matches
            # a single-step batch of (batch_size * grad_accum).
            (loss / args.grad_accum).backward()
            accum_count += 1
            epoch_loss_sum += loss.item()
            epoch_step_count += 1

            if accum_count >= args.grad_accum:
                # Clip aggressively to avoid LoRA collapse at LR=1e-4.
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_step += 1
                if global_step % 5 == 0:
                    elapsed = time.time() - start_time
                    log.info(
                        "epoch=%d batch=%d/%d step=%d  loss=%.4f main=%.4f sub=%.4f  elapsed=%.0fs",
                        epoch + 1, batch_idx + 1, len(loader), global_step,
                        loss.item(), main_loss.item(), sub_loss.item(), elapsed,
                    )

        avg = epoch_loss_sum / max(1, epoch_step_count)
        log.info("=" * 70)
        log.info("EPOCH %d done: avg_loss=%.4f over %d batches",
                 epoch + 1, avg, epoch_step_count)
        log.info("=" * 70)

    # Final flush — apply leftover gradients.
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    log.info("Training complete in %.1f minutes", (time.time() - start_time) / 60)

    # Save LoRA adapter
    log.info("Saving LoRA adapter to %s ...", save_dir)
    model.talker.save_pretrained(str(save_dir))

    meta = {
        "version": args.version,
        "persona": args.persona,
        "training_type": "sft_text_lora",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": 2 * args.lora_rank,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "dataset_records": len(dataset.records),
        "dataset_jsonl": str(jsonl_path.relative_to(ROOT)),
        "ref_audio": str(ref_path.relative_to(ROOT)),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "global_steps": global_step,
        "wall_clock_minutes": (time.time() - start_time) / 60,
    }
    (save_dir.parent / "v13_train_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote metadata to %s", save_dir.parent / "v13_train_metadata.json")

    log.info("DONE. Adapter saved. To use: load Qwen3-TTS-12Hz-1.7B-Base + apply this PEFT adapter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
