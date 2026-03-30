"""
Background training job runner.

Manages the training process in a background thread/process.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .lora_trainer import LoraTrainer, TrainingConfig, TrainingResult
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class TrainingJob:
    """
    Background training job.

    Usage:
        job = TrainingJob(version_id, version_dir, audio_paths, config)
        job.start()
        # Or in async context:
        await job.start_async()
    """

    def __init__(
        self,
        version_id: str,
        version_dir: Path,
        audio_paths: list[Path],
        config: TrainingConfig,
        total_audio_duration: float,
    ):
        self.version_id = version_id
        self.version_dir = Path(version_dir)
        self.audio_paths = audio_paths
        self.config = config
        self.total_audio_duration = total_audio_duration

        self._thread: Optional[threading.Thread] = None
        self._result: Optional[TrainingResult] = None
        self._cancelled = False

    def start(self):
        """Start training in background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning(f"[TRAINING:{self.version_id[:8]}] Already running")
            return

        self._cancelled = False
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()
        logger.info(f"[TRAINING:{self.version_id[:8]}] Training started in background")

    def _run_training(self):
        """Run training synchronously."""
        try:
            # Initialize progress tracker
            tracker = ProgressTracker(
                version_id=self.version_id,
                version_dir=self.version_dir,
                total_epochs=self.config.num_epochs,
                total_audio_duration=self.total_audio_duration,
            )

            # Create trainer
            trainer = LoraTrainer(
                version_id=self.version_id,
                persona_id="",  # Set by caller
                audio_paths=self.audio_paths,
                output_dir=self.version_dir,
                config=self.config,
            )

            # Monitor training via progress file
            import subprocess
            import sys
            import os

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent.parent)

            # Create training script
            train_script = self.version_dir / "train_lora.py"
            audio_paths_json = trainer._create_training_script()

            # Actually create the script properly
            audio_paths_json = [str(p) for p in self.audio_paths]
            script = f'''
import os, sys, json, time, logging
from pathlib import Path
import torch
from datasets import load_dataset, Audio
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_PATHS = {json.dumps(audio_paths_json, ensure_ascii=False)}
OUTPUT_DIR = "{self.version_dir}"
BASE_MODEL = "{self.config.base_model}"
RANK = {self.config.rank}
LEARNING_RATE = {self.config.learning_rate}
NUM_EPOCHS = {self.config.num_epochs}
BATCH_SIZE = {self.config.batch_size}

def main():
    try:
        logger.info(f"Loading model: {{BASE_MODEL}}")
        processor = Qwen2AudioProcessor.from_pretrained(BASE_MODEL)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )

        lora_config = LoraConfig(
            r=RANK, lora_alpha=RANK*2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load audio files
        from datasets import Dataset
        data = []
        for path in AUDIO_PATHS:
            p = Path(path)
            if p.exists():
                data.append({{"audio": str(p), "text": ""}})

        if not data:
            raise ValueError("No audio files found")

        ds = Dataset.from_list(data)
        ds = ds.cast_column("audio", Audio())

        def preprocess(example):
            audio = example["audio"]
            inputs = processor(
                audio=audio["array"], sampling_rate=audio["sampling_rate"],
                text=example["text"], return_tensors="pt"
            )
            return {{
                "input_ids": inputs["input_ids"],
                "labels": inputs["labels"],
                "attention_mask": inputs["attention_mask"],
            }}

        ds = ds.map(precess)

        args = TrainingArguments(
            output_dir=OUTPUT_DIR, per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
            fp16=True, logging_steps=1, save_strategy="no",
            report_to="none", warmup_steps={self.config.warmup_steps},
            gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        )

        trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=processor)
        start_time = time.time()

        # Train epoch by epoch for progress tracking
        for epoch in range(NUM_EPOCHS):
            logger.info(f"Starting epoch {{epoch+1}}/{{NUM_EPOCHS}}")
            tracker_path = Path("{self.version_dir}") / "progress.json"

            # Update progress - epoch started
            if tracker_path.exists():
                with open(tracker_path) as f:
                    prog = json.load(f)
                prog["current_epoch"] = epoch + 1
                prog["progress_pct"] = int(((epoch + 1) / NUM_EPOCHS) * 100)
                with open(tracker_path, "w") as f:
                    json.dump(prog, f)

            trainer.train()

            # Get loss from trainer state
            log_history = trainer.state.log_history
            epoch_loss = log_history[-1].get("loss", 0.0) if log_history else 0.0

            # Update progress - epoch complete
            if tracker_path.exists():
                with open(tracker_path) as f:
                    prog = json.load(f)
                prog["current_loss"] = float(epoch_loss)
                if epoch_loss < prog.get("best_loss", float('inf')):
                    prog["best_loss"] = float(epoch_loss)
                prog["current_epoch"] = epoch + 1
                prog["progress_pct"] = int(((epoch + 1) / NUM_EPOCHS) * 100)
                with open(tracker_path, "w") as f:
                    json.dump(prog, f)

        training_time = int(time.time() - start_time)

        # Save LoRA
        lora_path = Path(OUTPUT_DIR) / "adapter"
        model.save_pretrained(lora_path)

        final_loss = trainer.state.log_history[-1].get("loss", 0.0) if trainer.state.log_history else 0.0

        result = {{
            "success": True,
            "lora_path": str(lora_path),
            "final_loss": float(final_loss),
            "training_time_seconds": training_time,
        }}
        with open(Path(OUTPUT_DIR) / "training_result.json", "w") as f:
            json.dump(result, f)

        # Update final progress
        if tracker_path.exists():
            with open(tracker_path) as f:
                prog = json.load(f)
            prog["status"] = "ready"
            prog["final_loss"] = float(final_loss)
            prog["progress_pct"] = 100
            prog["training_time"] = training_time
            with open(tracker_path, "w") as f:
                json.dump(prog, f)

        logger.info(f"Training complete! Loss: {{final_loss:.4f}}")

    except Exception as e:
        logger.error(f"Training failed: {{e}}")
        import traceback
        traceback.print_exc()

        tracker_path = Path("{self.version_dir}") / "progress.json"
        if tracker_path.exists():
            with open(tracker_path) as f:
                prog = json.load(f)
            prog["status"] = "failed"
            prog["error_message"] = str(e)
            with open(tracker_path, "w") as f:
                json.dump(prog, f)

        result = {{
            "success": False,
            "error": str(e),
        }}
        with open(Path(OUTPUT_DIR) / "training_result.json", "w") as f:
            json.dump(result, f)

if __name__ == "__main__":
    main()
'''
            with open(train_script, "w", encoding="utf-8") as f:
                f.write(script)

            # Run training script
            process = subprocess.Popen(
                [sys.executable, str(train_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )

            # Monitor progress
            tracker = ProgressTracker(
                version_id=self.version_id,
                version_dir=self.version_dir,
                total_epochs=self.config.num_epochs,
                total_audio_duration=self.total_audio_duration,
            )

            # Wait for process to complete
            stdout, _ = process.communicate()

            # Load result
            result_file = self.version_dir / "training_result.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    self._result = TrainingResult(**json.load(f))
            else:
                self._result = TrainingResult(
                    success=False,
                    error=f"Training script exited with code {process.returncode}\n{stdout[-500:]}",
                )

        except Exception as e:
            logger.error(f"[TRAINING:{self.version_id[:8]}] Training error: {e}")
            self._result = TrainingResult(success=False, error=str(e))

    def poll(self) -> Optional[TrainingResult]:
        """Poll for result (returns None if still running)."""
        return self._result

    def cancel(self):
        """Cancel training."""
        self._cancelled = True
        logger.info(f"[TRAINING:{self.version_id[:8]}] Cancelling training")

    def is_running(self) -> bool:
        """Check if training is still running."""
        return self._thread is not None and self._thread.is_alive()
