"""
LoRA Trainer for Qwen3-TTS.

Implements fine-tuning using PEFT + Transformers.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    rank: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@dataclass
class TrainingResult:
    """Result of training."""
    success: bool
    lora_path: Optional[str] = None
    final_loss: Optional[float] = None
    training_time_seconds: Optional[int] = None
    error: Optional[str] = None


class LoraTrainer:
    """
    LoRA fine-tuning trainer for Qwen3-TTS.

    Usage:
        config = TrainingConfig(rank=16, num_epochs=10)
        trainer = LoraTrainer(version_id, persona_id, audio_paths, config)
        result = trainer.train()
    """

    def __init__(
        self,
        version_id: str,
        persona_id: str,
        audio_paths: list[Path],
        output_dir: Path,
        config: TrainingConfig,
    ):
        self.version_id = version_id
        self.persona_id = persona_id
        self.audio_paths = [Path(p) for p in audio_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Create training script
        self.train_script = self.output_dir / "train_lora.py"

    def _create_training_script(self) -> Path:
        """Create the training script."""
        audio_paths_json = json.dumps([str(p) for p in self.audio_paths], ensure_ascii=False)

        script = f'''
"""
Auto-generated LoRA training script for {self.version_id}
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import torch
from datasets import load_dataset, Audio
from transformers import (
    Qwen2AudioForConditionalGeneration,
    Qwen2AudioProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
AUDIO_PATHS = {audio_paths_json}
OUTPUT_DIR = "{self.output_dir}"
BASE_MODEL = "{self.config.base_model}"
RANK = {self.config.rank}
LEARNING_RATE = {self.config.learning_rate}
NUM_EPOCHS = {self.config.num_epochs}
BATCH_SIZE = {self.config.batch_size}

def prepare_dataset(audio_paths):
    """Prepare dataset from audio files."""
    data = []
    for path in audio_paths:
        if not Path(path).exists():
            logger.warning(f"Audio file not found: {{path}}")
            continue
        data.append({{
            "audio": str(path),
            "text": "",  # Qwen3-TTS uses audio for voice cloning
        }})
    return data

def main():
    logger.info(f"Loading base model: {{BASE_MODEL}}")
    processor = Qwen2AudioProcessor.from_pretrained(BASE_MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    dataset_data = prepare_dataset(AUDIO_PATHS)

    # Create a simple dataset
    from datasets import Dataset
    ds = Dataset.from_list(dataset_data)
    ds = ds.cast_column("audio", Audio())

    def preprocess(example):
        audio = example["audio"]
        inputs = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["text"],
            return_tensors="pt",
        )
        return {{
            "input_ids": inputs["input_ids"],
            "labels": inputs["labels"],
            "attention_mask": inputs["attention_mask"],
        }}

    ds = ds.map(preprocess)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        warmup_steps={self.config.warmup_steps},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=processor,
    )

    # Train
    start_time = time.time()
    trainer.train()
    training_time = int(time.time() - start_time)

    # Save LoRA adapter
    lora_path = Path(OUTPUT_DIR) / "adapter"
    model.save_pretrained(lora_path)

    # Save final loss
    final_loss = trainer.state.log_history[-1].get("loss", 0.0) if trainer.state.log_history else 0.0

    # Write result
    result = {{
        "success": True,
        "lora_path": str(lora_path),
        "final_loss": float(final_loss),
        "training_time_seconds": training_time,
    }}
    with open(Path(OUTPUT_DIR) / "training_result.json", "w") as f:
        json.dump(result, f)

    logger.info(f"Training complete! Loss: {{final_loss:.4f}}, Time: {{training_time}}s")
    return result

if __name__ == "__main__":
    main()
'''
        with open(self.train_script, "w", encoding="utf-8") as f:
            f.write(script)

        return self.train_script

    def train(self) -> TrainingResult:
        """Run training and return result."""
        import shutil

        logger.info(f"[TRAINING:{self.version_id[:8]}] Starting training")
        logger.info(f"[TRAINING:{self.version_id[:8]}] Audio files: {len(self.audio_paths)}")
        logger.info(f"[TRAINING:{self.version_id[:8]}] Output: {self.output_dir}")

        # Validate audio files exist
        missing = [p for p in self.audio_paths if not p.exists()]
        if missing:
            logger.warning(f"[TRAINING:{self.version_id[:8]}] Missing audio files: {missing}")

        if not self.audio_paths or all(not p.exists() for p in self.audio_paths):
            return TrainingResult(
                success=False,
                error="No valid audio files found",
            )

        # Create training script
        self._create_training_script()
        logger.info(f"[TRAINING:{self.version_id[:8]}] Training script: {self.train_script}")

        # Run training in subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)

        try:
            result = subprocess.run(
                [sys.executable, str(self.train_script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=7200,  # 2 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"[TRAINING:{self.version_id[:8]}] Training failed: {result.stderr}")
                return TrainingResult(
                    success=False,
                    error=result.stderr[-500:],  # Last 500 chars
                )

            # Load result
            result_file = self.output_dir / "training_result.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    data = json.load(f)
                return TrainingResult(
                    success=True,
                    lora_path=data.get("lora_path"),
                    final_loss=data.get("final_loss"),
                    training_time_seconds=data.get("training_time"),
                )

            return TrainingResult(success=False, error="No result file found")

        except subprocess.TimeoutExpired:
            return TrainingResult(success=False, error="Training timeout (2 hours)")
        except Exception as e:
            logger.error(f"[TRAINING:{self.version_id[:8]}] Training exception: {e}")
            return TrainingResult(success=False, error=str(e))
