"""
Background training job runner.

Manages the training process in a background thread/process.
"""

import json
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
            # Use native PyTorch training with correct forward_sub_talker_finetune approach
            script = '''
# INLINE SCRIPT MARKER - Correct LoRA training for Qwen3-TTS voice cloning
# Uses forward_sub_talker_finetune which is the proper training method
import os, sys, json, time, logging
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
# Use native PyTorch training without AMP

from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
from qwen_tts import Qwen3TTSTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_PATHS = ''' + json.dumps([str(p) for p in self.audio_paths], ensure_ascii=False) + '''
OUTPUT_DIR = "''' + str(self.version_dir) + '''"
BASE_MODEL = "''' + self.config.base_model + '''"
RANK = ''' + str(self.config.rank) + '''
LEARNING_RATE = ''' + str(self.config.learning_rate) + '''
NUM_EPOCHS = ''' + str(self.config.num_epochs) + '''
BATCH_SIZE = ''' + str(self.config.batch_size) + '''
TRACKER_PATH = Path(OUTPUT_DIR) / "progress.json"

def main():
    try:
        logger.info(f"Loading model: {BASE_MODEL}")
        logger.info(f"AUDIO_PATHS: {AUDIO_PATHS}")

        # Load speech tokenizer
        logger.info("Loading speech tokenizer...")
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")

        # Load model
        logger.info("Loading base model...")
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        logger.info(f"Model loaded: {type(model)}")
        logger.info(f"num_code_groups: {model.talker.config.num_code_groups}")

        # Apply LoRA to both talker and code_predictor
        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            r=RANK, lora_alpha=RANK*2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none",
        )
        # LoRA on the talker model (text + speaker encoding)
        model.talker.model = get_peft_model(model.talker.model, lora_config, adapter_name="talker_lora")
        # LoRA on the code_predictor
        model.talker.code_predictor = get_peft_model(model.talker.code_predictor, lora_config, adapter_name="codec_lora")

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

        # Encode audio to tokens
        logger.info("Encoding audio files...")
        all_audio_codes = []  # List of (seq_len, num_code_groups) tensors
        for path in AUDIO_PATHS:
            p = Path(path)
            if p.exists():
                audio, sr = sf.read(str(p))
                # Ensure audio is float32
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                enc = speech_tokenizer.encode(audio, sr=sr)
                codes = enc["audio_codes"][0]  # (seq_len, num_code_groups)
                all_audio_codes.append(codes)
                logger.info(f"Encoded {path}: shape={codes.shape}")
            else:
                logger.warning(f"Audio file not found: {path}")

        if not all_audio_codes:
            raise ValueError("No audio files could be encoded")

        num_code_groups = all_audio_codes[0].shape[1]  # Should be 16
        logger.info(f"num_code_groups: {num_code_groups}")

        # Create dataset that yields (codec_ids, talker_hidden_state) pairs
        # For voice cloning: we use audio_codes as both input and target
        # talker_hidden_state comes from processing the audio through speaker encoder
        class SpeechDataset(Dataset):
            def __init__(self, audio_codes_list, num_code_groups):
                self.audio_codes = audio_codes_list
                self.num_code_groups = num_code_groups

            def __len__(self):
                return len(self.audio_codes)

            def __getitem__(self, idx):
                codes = self.audio_codes[idx]  # (seq_len, num_code_groups)
                # Use middle portion of audio for training (skip very start/end)
                seq_len = codes.shape[0]
                if seq_len > 10:
                    start = seq_len // 4
                    end = seq_len - seq_len // 4
                    codes = codes[start:end]

                # Target: all code groups at each time step
                # Input: same sequence length for predicting next step
                return codes  # (seq_len, num_code_groups)

        dataset = SpeechDataset(all_audio_codes, num_code_groups)

        # Use native PyTorch training with correct forward approach
        model.train()

        # Only train LoRA parameters
        trainable_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
        logger.info(f"Training with {len(trainable_params)} trainable LoRA parameters")

        logger.info(f"Starting training: {len(dataset)} samples, {NUM_EPOCHS} epochs")
        logger.info(f"Using forward_sub_talker_finetune for proper loss computation")

        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            num_batches = 0
            num_steps = 0

            # Update progress
            try:
                if TRACKER_PATH.exists():
                    with open(TRACKER_PATH) as f:
                        prog = json.load(f)
                    prog["current_epoch"] = epoch + 1
                    prog["progress_pct"] = int(((epoch + 1) / NUM_EPOCHS) * 100)
                    with open(TRACKER_PATH, "w") as f:
                        json.dump(prog, f)
                    logger.info(f"Progress: epoch {epoch+1}, {prog['progress_pct']}%")
            except Exception as e:
                logger.error(f"Progress update error: {e}")

            # Training loop - process each audio file
            for sample_idx in range(len(dataset)):
                sample_codes = dataset[sample_idx]  # (seq_len, num_code_groups)
                seq_len = sample_codes.shape[0]

                # Process each time step
                for step in range(min(seq_len - 1, 50)):  # Limit steps per sample
                    # Get codec_ids for this and next step
                    # codec_ids: (num_code_groups,) - current step
                    # target_ids: (num_code_groups,) - next step (for labels)
                    device = next(model.parameters()).device
                    codec_ids = sample_codes[step].to(device)  # (num_code_groups,) - move to device

                    # Get speaker embedding from reference audio
                    # Use average of audio embeddings as speaker representation
                    # In practice, you'd use the speaker_encoder, but for simplicity
                    # we use a learned or fixed representation

                    # For voice cloning, we create a "pseudo" talker_hidden_state
                    # by averaging the audio embeddings
                    with torch.no_grad():
                        # Get audio embeddings from code_predictor's embeddings
                        audio_embeds = []
                        # First code group uses talker's embeddings
                        embed = model.talker.get_input_embeddings()(
                            codec_ids[0].unsqueeze(0).to(device)
                        )  # (1, hidden_size)
                        audio_embeds.append(embed)
                        # Remaining code groups use code_predictor's embeddings
                        for g in range(1, num_code_groups):
                            embed = model.talker.code_predictor.get_input_embeddings()[g-1](
                                codec_ids[g].unsqueeze(0).to(device)
                            )  # (1, hidden_size)
                            audio_embeds.append(embed)
                        audio_embeds = torch.stack(audio_embeds, dim=1).squeeze(0)  # (num_code_groups, hidden_size)

                        # Average as speaker representation
                        talker_hidden = audio_embeds.mean(dim=0, keepdim=True)  # (1, hidden_size)

                    # Now use forward_sub_talker_finetune
                    # It expects codec_ids as (batch, num_code_groups) and talker_hidden as (batch, hidden_size)
                    try:
                        # Prepare inputs - move to model device
                        codec_ids_batch = codec_ids.unsqueeze(0).long().to(device)  # (1, num_code_groups)
                        talker_hidden_batch = talker_hidden.to(device)  # (1, hidden_size)

                        # Call forward_sub_talker_finetune
                        # This computes loss properly by:
                        # 1. Concatenating talker_hidden with audio code embeddings
                        # 2. Running through code_predictor.forward_finetune
                        # 3. Computing loss between predicted and actual next codes
                        _, loss = model.talker.forward_sub_talker_finetune(
                            codec_ids=codec_ids_batch,
                            talker_hidden_states=talker_hidden_batch,
                        )

                        if loss is not None and not torch.isnan(loss):
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            epoch_loss += loss.item()
                            num_batches += 1
                            num_steps += 1

                            if num_steps % 10 == 0:
                                logger.info(f"Epoch {epoch+1} step {num_steps}: loss={loss.item():.6f}")
                        else:
                            logger.warning(f"Skipping step {num_steps}: loss is None or NaN")

                    except Exception as e:
                        logger.error(f"Batch error at step {num_steps}: {e}")
                        continue

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.6f}, batches={num_batches}")

            # Update progress after epoch
            try:
                if TRACKER_PATH.exists():
                    with open(TRACKER_PATH) as f:
                        prog = json.load(f)
                    prog["current_loss"] = float(avg_loss) if not np.isnan(avg_loss) else 0.0
                    if avg_loss < (prog.get("best_loss") or float('inf')):
                        prog["best_loss"] = float(avg_loss) if not np.isnan(avg_loss) else 0.0
                    prog["current_epoch"] = epoch + 1
                    prog["progress_pct"] = int(((epoch + 1) / NUM_EPOCHS) * 100)
                    with open(TRACKER_PATH, "w") as f:
                        json.dump(prog, f)
            except Exception as e:
                logger.error(f"Progress save error: {e}")

        training_time = int(time.time() - start_time)

        # Save LoRA adapter weights
        lora_path = Path(OUTPUT_DIR) / "adapter"
        lora_path.mkdir(parents=True, exist_ok=True)

        # Save only the LoRA trainable parameters (not full model)
        from safetensors.torch import save_file

        # Collect all LoRA state dicts
        state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                state_dict[name] = param.cpu()

        # Save as safetensors
        save_file(state_dict, lora_path / "adapter_model.safetensors")

        # Save adapter config with proper PEFT format
        adapter_config = {
            "base_model_name_or_path": BASE_MODEL,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "r": RANK,
            "lora_alpha": RANK * 2,
            "lora_dropout": 0.05,
            "bias": "none",
        }
        import json as json_module
        with open(lora_path / "adapter_config.json", "w") as f:
            json_module.dump(adapter_config, f)

        logger.info(f"LoRA saved to: {lora_path}")

        # Save result
        result = {
            "success": True,
            "lora_path": str(lora_path),
            "final_loss": float(avg_loss) if not np.isnan(avg_loss) else 0.0,
            "training_time_seconds": training_time,
        }
        with open(Path(OUTPUT_DIR) / "training_result.json", "w") as f:
            json.dump(result, f)

        # Update progress
        try:
            if TRACKER_PATH.exists():
                with open(TRACKER_PATH) as f:
                    prog = json.load(f)
                prog["status"] = "ready"
                prog["progress_pct"] = 100
                with open(TRACKER_PATH, "w") as f:
                    json.dump(prog, f)
        except:
            pass

        logger.info(f"Training complete! Loss: {avg_loss:.6f}, Time: {training_time}s")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

        try:
            if TRACKER_PATH.exists():
                with open(TRACKER_PATH) as f:
                    prog = json.load(f)
                prog["status"] = "failed"
                prog["error_message"] = str(e)
                with open(TRACKER_PATH, "w") as f:
                    json.dump(prog, f)
        except:
            pass

        result = {"success": False, "error": str(e)}
        with open(Path(OUTPUT_DIR) / "training_result.json", "w") as f:
            json.dump(result, f)

if __name__ == "__main__":
    main()
'''
            with open(train_script, "w", encoding="utf-8") as f:
                f.write(script)

            # Verify script is valid Python
            import py_compile
            try:
                py_compile.compile(str(train_script), doraise=True)
                logger.info(f"[TRAINING:{self.version_id[:8]}] Script verified: {train_script}")
            except py_compile.PyCompileError as e:
                logger.error(f"[TRAINING:{self.version_id[:8]}] Script compile error: {e}")
                self._result = TrainingResult(success=False, error=f"Script compile error: {e}")
                return

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

            # Log output for debugging
            if stdout:
                logger.info(f"[TRAINING:{self.version_id[:8]}] Training output:\n{stdout[:2000]}")

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

            # Auto-merge after successful training
            if self._result.success and not self._cancelled:
                logger.info(f"[TRAINING:{self.version_id[:8]}] Training succeeded, starting merge...")
                try:
                    # Update status to merging
                    tracker_file = self.version_dir / "progress.json"
                    if tracker_file.exists():
                        with open(tracker_file) as f:
                            prog = json.load(f)
                        prog["status"] = "merging"
                        prog["progress_pct"] = 100
                        with open(tracker_file, "w") as f:
                            json.dump(prog, f)

                    merged_path = merge_lora(self.version_dir)
                    if merged_path and merged_path.exists():
                        logger.info(f"[TRAINING:{self.version_id[:8]}] Merge complete: {merged_path}")

                        # Update tracker status
                        if tracker_file.exists():
                            with open(tracker_file) as f:
                                prog = json.load(f)
                            prog["status"] = "ready"
                            with open(tracker_file, "w") as f:
                                json.dump(prog, f)

                        # Auto-activate the new merged model
                        try:
                            from app.services.tts import get_tts_engine
                            engine = get_tts_engine()
                            # Extract version_id from version_dir name (e.g. xiao_s_v12_20260330_223729)
                            # The TrainingVersion.version_id is the first part (e.g. v12_20260330_223729)
                            # We need to find the right version_id for the version manager
                            # Use the version_dir name to find matching version
                            from app.services.training import get_version_manager
                            vm = get_version_manager()
                            # Find version by lora_path matching our version_dir
                            matching = [v for v in vm.list_versions() if v.lora_path and Path(v.lora_path) == self.version_dir]
                            if matching:
                                latest = matching[0]
                                logger.info(f"[TRAINING:{self.version_id[:8]}] Auto-activating version: {latest.version_id}")
                                engine.activate_version(latest.version_id)
                            else:
                                logger.warning(f"[TRAINING:{self.version_id[:8]}] No matching version found in manager")
                        except Exception as act_err:
                            logger.warning(f"[TRAINING:{self.version_id[:8]}] Auto-activate failed: {act_err}")
                    else:
                        logger.error(f"[TRAINING:{self.version_id[:8]}] Merge failed or returned None")
                        # Update status to failed since merge is critical
                        if tracker_file.exists():
                            with open(tracker_file) as f:
                                prog = json.load(f)
                            prog["status"] = "failed"
                            prog["error_message"] = "Merge failed"
                            with open(tracker_file, "w") as f:
                                json.dump(prog, f)
                except Exception as merge_err:
                    logger.error(f"[TRAINING:{self.version_id[:8]}] Merge exception: {merge_err}")
                    import traceback
                    traceback.print_exc()

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


# =============================================================================
# LoRA Merge
# =============================================================================

def merge_lora(lora_dir: Path, base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base") -> Optional[Path]:
    """
    Merge LoRA adapter weights into base Qwen3-TTS model and save.

    This replaces PEFT's merge_and_unload() which was removed in PEFT 0.18.
    Instead we manually compute: W_merged = W_base + (alpha/rank) * W_B @ W_A

    Args:
        lora_dir: Path to version directory containing adapter/ (e.g. data/models/xiao_s_v12_timestamp)
        base_model: HuggingFace model ID for base Qwen3-TTS

    Returns:
        Path to merged model directory, or None if merge failed
    """
    import re
    import shutil
    import torch
    from safetensors.torch import load_file, save_file
    from huggingface_hub import snapshot_download

    adapter_path = lora_dir / "adapter"
    if not adapter_path.exists():
        logger.error(f"[MERGE] Adapter not found: {adapter_path}")
        return None

    adapter_file = adapter_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        logger.error(f"[MERGE] adapter_model.safetensors not found: {adapter_file}")
        return None

    # Compute merged model path
    # e.g. xiao_s_v12_20260330_223729 -> xiao_s_v12 -> merged_qwen3_tts_xiao_s_v12
    parts = lora_dir.name.split('_')
    version_base = '_'.join(parts[:3])  # first 3 parts
    merged_name = f"merged_qwen3_tts_{version_base}"
    merged_path = lora_dir.parent / merged_name

    if merged_path.exists():
        logger.info(f"[MERGE] Merged model already exists: {merged_path}")
        return merged_path

    logger.info(f"[MERGE] Starting merge: {adapter_file} -> {merged_path}")

    # Load adapter state dict
    adapter_state = load_file(adapter_file)
    logger.info(f"[MERGE] Loaded {len(adapter_state)} adapter keys")

    # Load LoRA config to get rank and alpha
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        with open(config_file) as f:
            lora_config = json.load(f)
        rank = lora_config.get("r", 16)
        lora_alpha = lora_config.get("lora_alpha", rank * 2)
        scaling = lora_alpha / rank
    else:
        rank = 16
        scaling = 2.0

    logger.info(f"[MERGE] LoRA rank={rank}, alpha={lora_alpha}, scaling={scaling}")

    # Download base model files (don't load full model into memory)
    logger.info(f"[MERGE] Downloading base model: {base_model}")
    try:
        base_model_path = Path(snapshot_download(base_model, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"]))
    except Exception as e:
        logger.error(f"[MERGE] Failed to download base model: {e}")
        return None

    # Build base model state dict from safetensors
    base_state = {}
    base_safetensor_files = list(base_model_path.glob("*.safetensors"))
    for sf in base_safetensor_files:
        state = load_file(sf)
        base_state.update(state)

    logger.info(f"[MERGE] Loaded {len(base_state)} base model keys")

    # Build mapping: for each base key, find its LoRA A and B keys
    # Adapter key pattern: talker.code_predictor.base_model.model.model.layers.{i}.self_attn.{proj}.lora_A.codec_lora.weight
    # Base key pattern:     talker.code_predictor.model.layers.{i}.self_attn.{proj}.weight
    #
    # PEFT wrapping: base_model.model.model.layers.X in LoRA → model.layers.X in base
    # The "base_model.model.model" PEFT path maps to just "model" in the original model
    # We use string replacement to do the correct mapping
    lora_pattern = re.compile(
        r'^(talker\.code_predictor\.base_model\.model\.model\.)(.+)\.lora_A\.codec_lora\.weight$'
    )
    lora_pattern_B = re.compile(
        r'^(talker\.code_predictor\.base_model\.model\.model\.)(.+)\.lora_B\.codec_lora\.weight$'
    )
    # Also build W_A and W_B storage
    lora_params: dict[str, dict] = {}  # base_key -> {'A': tensor, 'B': tensor}

    for key, tensor in adapter_state.items():
        # Try LoRA A pattern
        match_A = lora_pattern.match(key)
        if match_A:
            prefix = match_A.group(1)  # "talker.code_predictor.base_model.model.model."
            peft_suffix = match_A.group(2)  # e.g. "layers.0.self_attn.q_proj"
            # Map PEFT path to base model path: "base_model.model.model." → "model."
            # So "talker.code_predictor.base_model.model.model.layers.X" → "talker.code_predictor.model.layers.X"
            base_key = f"talker.code_predictor.model.{peft_suffix}.weight"
            if base_key not in lora_params:
                lora_params[base_key] = {}
            lora_params[base_key]['A'] = tensor
            continue

        # Try LoRA B pattern
        match_B = lora_pattern_B.match(key)
        if match_B:
            prefix = match_B.group(1)
            peft_suffix = match_B.group(2)
            base_key = f"talker.code_predictor.model.{peft_suffix}.weight"
            if base_key not in lora_params:
                lora_params[base_key] = {}
            lora_params[base_key]['B'] = tensor

    logger.info(f"[MERGE] Found {len(lora_params)} LoRA parameter groups to merge")

    # Apply merge
    merged_state = dict(base_state)  # start with base
    for base_key, lora_pair in lora_params.items():
        if 'A' not in lora_pair or 'B' not in lora_pair:
            logger.warning(f"[MERGE] Missing A or B for {base_key}, skipping")
            continue
        W_A = lora_pair['A']  # (rank, in_features)
        W_B = lora_pair['B']  # (out_features, rank)
        W_base = base_state.get(base_key)
        if W_base is None:
            logger.warning(f"[MERGE] Base key not found: {base_key}")
            continue
        # W_merged = W_base + scaling * W_B @ W_A
        W_merged = W_base.float() + (scaling * W_B.float() @ W_A.float()).to(W_base.dtype)
        merged_state[base_key] = W_merged
        logger.debug(f"[MERGE] Merged {base_key}: base={W_base.shape}, delta_norm={torch.norm(W_merged - W_base.float()).item():.4f}")

    # Create merged model directory
    merged_path.mkdir(parents=True, exist_ok=True)

    # Copy all base model files, replacing safetensors with merged
    for item in base_model_path.iterdir():
        if item.suffix == ".safetensors":
            continue  # Skip base safetensors, we'll write our merged one
        dest = merged_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # Save merged safetensor
    logger.info(f"[MERGE] Saving merged model to {merged_path}")
    save_file(merged_state, merged_path / "model.safetensors")

    logger.info(f"[MERGE] Done! Merged model saved to: {merged_path}")
    return merged_path
