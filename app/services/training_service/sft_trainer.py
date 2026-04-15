"""
SFT (Supervised Fine-Tuning) Trainer for Qwen3-TTS.

Trains all parameters (no LoRA) for maximum voice quality.
Requires more GPU memory than LoRA but captures voice nuances better.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """SFT Training configuration."""
    learning_rate: float = 1e-6  # Lower LR for full model SFT
    num_epochs: int = 10
    batch_size: int = 1  # Small batch for full model
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 8  # Effective batch = 8
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    use_gradient_checkpointing: bool = True  # Save memory


@dataclass
class SFTResult:
    """Result of SFT training."""
    success: bool
    model_path: Optional[str] = None
    final_loss: Optional[float] = None
    training_time_seconds: Optional[int] = None
    error: Optional[str] = None


class SftTrainer:
    """
    SFT trainer for Qwen3-TTS - trains full model without LoRA.

    Usage:
        config = SFTConfig(learning_rate=1e-6, num_epochs=10)
        trainer = SftTrainer(version_id, persona_id, audio_paths, config)
        result = trainer.train()
    """

    def __init__(
        self,
        version_id: str,
        persona_id: str,
        audio_paths: list[Path],
        output_dir: Path,
        config: SFTConfig,
    ):
        self.version_id = version_id
        self.persona_id = persona_id
        self.audio_paths = [Path(p) for p in audio_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self.train_script = self.output_dir / "train_sft.py"

    def _create_training_script(self) -> Path:
        """Create the SFT training script."""
        audio_paths_json = json.dumps([str(p) for p in self.audio_paths], ensure_ascii=False)

        script = f'''
"""
Auto-generated SFT training script for {self.version_id}
Trains full Qwen3-TTS model without LoRA for maximum voice quality.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
from qwen_tts import Qwen3TTSTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_PATHS = {audio_paths_json}
OUTPUT_DIR = "{self.output_dir}"
BASE_MODEL = "{self.config.base_model}"
LEARNING_RATE = {self.config.learning_rate}
NUM_EPOCHS = {self.config.num_epochs}
BATCH_SIZE = {self.config.batch_size}
GRADIENT_ACCUMULATION = {self.config.gradient_accumulation_steps}
WARMUP_STEPS = {self.config.warmup_steps}
USE_GRADIENT_CHECKPOINTING = {str(self.config.use_gradient_checkpointing).lower()}
TRACKER_PATH = Path(OUTPUT_DIR) / "progress.json"

def prepare_dataset(audio_paths):
    """Prepare dataset from audio files."""
    data = []
    for path in audio_paths:
        if not Path(path).exists():
            logger.warning(f"Audio file not found: {{path}}")
            continue
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Stereo to mono
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        data.append({{
            "audio": audio,
            "sampling_rate": sr,
            "path": str(path),
        }})
        logger.info(f"Loaded: {{path}}, {{len(audio)/sr:.1f}}s")
    return data

class Qwen3TTSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Pad or truncate to same length
    max_len = max(len(item["audio"]) for item in batch)
    audios = []
    for item in batch:
        audio = item["audio"]
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]
        audios.append(audio)
    return {
        "audio": torch.FloatTensor(np.stack(audios)),
        "sampling_rate": batch[0]["sampling_rate"],
    }

def main():
    start_time = time.time()
    logger.info(f"Starting SFT training: {{OUTPUT_DIR}}")
    logger.info(f"Base model: {{BASE_MODEL}}")
    logger.info(f"Audio files: {{len(AUDIO_PATHS)}}")

    # Load speech tokenizer
    logger.info("Loading speech tokenizer...")
    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")

    # Load model
    logger.info("Loading base model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    logger.info(f"Model loaded: {{type(model).__name__}}")

    # Enable gradient checkpointing to save memory
    if USE_GRADIENT_CHECKPOINTING:
        logger.info("Enabling gradient checkpointing...")
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        elif hasattr(model, 'use_cache'):
            model.use_cache = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {{total_params:,}}")
    logger.info(f"Trainable parameters: {{trainable_params:,}} (100% - full SFT)")

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset_data = prepare_dataset(AUDIO_PATHS)
    dataset = Qwen3TTSDataset(dataset_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    logger.info(f"Starting training: {{NUM_EPOCHS}} epochs, {{len(dataloader)}} batches/epoch")
    global_step = 0
    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            audio = batch["audio"].to(model.device)
            sr = batch["sampling_rate"][0]

            try:
                # Encode audio
                with torch.no_grad():
                    encoded = speech_tokenizer.encode(audio.cpu().numpy(), sr=sr)
                    input_codes = torch.LongTensor(encoded["audio_codes"][0]).to(model.device)
                    labels = input_codes.clone()

                # Forward with gradient checkpointing
                if USE_GRADIENT_CHECKPOINTING:
                    # Use checkpoint to save memory
                    outputs = torch.utils.checkpoint.checkpoint(
                        model.forward_sub_talker_finetune,
                        input_codes,
                        labels,
                        use_reentrant=False,
                    )
                else:
                    outputs = model.forward_sub_talker_finetune(input_codes, labels)

                loss = outputs.loss
                loss = loss / GRADIENT_ACCUMULATION
                loss.backward()

                epoch_loss += loss.item() * GRADIENT_ACCUMULATION
                num_batches += 1

                # Gradient accumulation
                if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Update progress
                    progress = {{
                        "step": global_step,
                        "epoch": epoch + 1,
                        "total_epochs": NUM_EPOCHS,
                        "loss": float(epoch_loss / max(1, num_batches)),
                        "lr": scheduler.get_last_lr()[0],
                    }}
                    with open(TRACKER_PATH, "w") as f:
                        json.dump(progress, f)

                    logger.info(f"Step {{global_step}}: loss={{loss.item() * GRADIENT_ACCUMULATION:.4f}}, lr={{scheduler.get_last_lr()[0]:.2e}}")

            except Exception as e:
                logger.error(f"Training error at batch {{batch_idx}}: {{e}}")
                continue

        # End of epoch
        avg_loss = epoch_loss / max(1, num_batches)
        elapsed = time.time() - start_time
        logger.info(f"Epoch {{epoch + 1}}/{{NUM_EPOCHS}} complete: avg_loss={{avg_loss:.4f}}, elapsed={{elapsed:.0f}}s")

    # Save final model
    logger.info("Saving SFT model...")
    save_path = Path(OUTPUT_DIR) / "sft_model"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    speech_tokenizer.save_pretrained(save_path)

    # Save training info
    training_info = {{
        "version_id": "{self.version_id}",
        "persona_id": "{self.persona_id}",
        "training_type": "SFT",
        "base_model": BASE_MODEL,
        "final_loss": float(avg_loss),
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "training_time_seconds": int(time.time() - start_time),
        "audio_paths": AUDIO_PATHS,
    }}
    with open(save_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    logger.info(f"SFT training complete! Model saved to: {{save_path}}")
    logger.info(f"Total time: {{time.time() - start_time:.0f}}s")
    return save_path

if __name__ == "__main__":
    main()
'''
        with open(self.train_script, 'w') as f:
            f.write(script)

        return self.train_script

    def train(self) -> SFTResult:
        """Run SFT training."""
        import subprocess
        import time

        start_time = time.time()

        try:
            script_path = self._create_training_script()
            logger.info(f"Created SFT training script: {script_path}")

            # Run training script
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.parent.parent)

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode != 0:
                return SFTResult(
                    success=False,
                    error=result.stderr[-2000:] if result.stderr else "Unknown error",
                )

            return SFTResult(
                success=True,
                model_path=str(self.output_dir / "sft_model"),
                training_time_seconds=int(time.time() - start_time),
            )

        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            return SFTResult(success=False, error=str(e))