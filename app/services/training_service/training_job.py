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
        training_type: str = "lora",
        persona_id: str = "persona_new",
    ):
        self.version_id = version_id
        self.version_dir = Path(version_dir)
        self.audio_paths = audio_paths
        self.config = config
        self.total_audio_duration = total_audio_duration
        self.training_type = training_type
        self.persona_id = persona_id

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
            use_lora = (self.training_type == "lora")
            # Use native PyTorch training with correct forward_sub_talker_finetune approach
            script = f'''
# INLINE SCRIPT MARKER - {"LoRA" if use_lora else "SFT"} training for Qwen3-TTS voice cloning
# Uses forward_sub_talker_finetune which is the proper training method
import os, sys, json, time, logging
from pathlib import Path
import torch

# Patch: add missing float8_e8m0fnu dtype for PyTorch 2.6 compatibility with PEFT 0.19
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e5m2  # Use float8_e5m2 as fallback

import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

USE_LORA = {use_lora}
if USE_LORA:
    from peft import LoraConfig, get_peft_model

from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
from qwen_tts import Qwen3TTSTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_PATHS = ''' + json.dumps([str(p) for p in self.audio_paths], ensure_ascii=False) + '''
OUTPUT_DIR = "''' + str(self.version_dir) + '''"
BASE_MODEL = "''' + self.config.base_model + '''"
RANK = ''' + str(getattr(self.config, 'rank', 16)) + '''
LEARNING_RATE = ''' + str(self.config.learning_rate) + '''
NUM_EPOCHS = ''' + str(self.config.num_epochs) + '''
BATCH_SIZE = ''' + str(self.config.batch_size) + '''
TRACKER_PATH = Path(OUTPUT_DIR) / "progress.json"
PERSONA_ID = "''' + self.persona_id + '''"

# Initialize progress.json before training starts
def init_progress():
    prog = {
        "status": "training",
        "current_epoch": 0,
        "progress_pct": 0,
        "persona_id": PERSONA_ID,
        "training_type": "lora" if USE_LORA else "sft",
    }
    with open(TRACKER_PATH, "w") as f:
        json.dump(prog, f)
    logger.info(f"Progress initialized: {TRACKER_PATH}")

def main():
    try:
        # Initialize progress tracking
        init_progress()

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

        if USE_LORA:
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

            # Only train LoRA parameters
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # SFT mode - train ALL parameters
            logger.info("SFT mode: training all parameters (no LoRA)")
            for param in model.parameters():
                param.requires_grad = True
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

        # Encode audio to tokens
        logger.info("Encoding audio files...")
        all_audio_codes = []  # List of (seq_len, num_code_groups) tensors
        TARGET_SR = 24000  # Qwen3-TTS requires 24kHz
        for path in AUDIO_PATHS:
            p = Path(path)
            if p.exists():
                audio, sr = sf.read(str(p))
                # Ensure audio is float32
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                # Resample to 24kHz if needed (Qwen3-TTS only supports 24kHz)
                if sr != TARGET_SR:
                    logger.info(f"Resampling {path} from {sr}Hz to {TARGET_SR}Hz")
                    # Use scipy for high-quality resampling
                    from scipy import signal
                    num_samples = int(len(audio) * TARGET_SR / sr)
                    audio = signal.resample(audio, num_samples)
                    sr = TARGET_SR
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
        #
        # IMPORTANT: Chunk audio into smaller segments to increase training samples.
        # Original design had only 3 samples (one per audio file), causing overfitting.
        # Now we create overlapping chunks of ~5-10 seconds each for 50-100+ samples.
        class SpeechDataset(Dataset):
            def __init__(self, audio_codes_list, num_code_groups, chunk_size=300, hop_size=150):
                """
                Args:
                    audio_codes_list: List of (seq_len, num_code_groups) tensors
                    num_code_groups: Number of code groups (typically 16)
                    chunk_size: Max frames per chunk (default 300 ~5-10 sec)
                    hop_size: Hop size for overlapping chunks (default 150)
                """
                self.num_code_groups = num_code_groups
                self.samples = []  # List of (chunk_tensor, audio_file_index)
                self.audio_file_indices = []  # Maps sample index -> audio file index

                for audio_idx, audio_codes in enumerate(audio_codes_list):
                    seq_len = audio_codes.shape[0]
                    # Create overlapping chunks
                    for start in range(0, seq_len, hop_size):
                        end = min(start + chunk_size, seq_len)
                        chunk_len = end - start
                        if chunk_len >= 50:  # Minimum chunk size for meaningful training
                            chunk = audio_codes[start:end].clone()
                            self.samples.append(chunk)
                            self.audio_file_indices.append(audio_idx)

                logger.info(f"Created {len(self.samples)} chunks from {len(audio_codes_list)} audio files")

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx], self.audio_file_indices[idx]

        dataset = SpeechDataset(all_audio_codes, num_code_groups)

        # Validate dataset before training
        total_chunks = len(dataset)
        MIN_CHUNKS = 30  # Minimum for meaningful SFT training

        logger.info(f"[VALIDATION] Training dataset: {total_chunks} chunks from {len(AUDIO_PATHS)} audio files")
        logger.info(f"[VALIDATION] Expected chunks per file: ~{total_chunks // max(len(AUDIO_PATHS), 1)}")

        if total_chunks < MIN_CHUNKS:
            logger.error(f"[VALIDATION] FAILED: Insufficient training samples! {total_chunks} < {MIN_CHUNKS} minimum required")
            raise ValueError(f"Insufficient training samples for SFT: {total_chunks} < {MIN_CHUNKS}. Need more/longer audio.")

        logger.info(f"[VALIDATION] PASSED: {total_chunks} chunks >= {MIN_CHUNKS} minimum")

        # Log dataset statistics
        chunk_size = 300  # Frames per chunk
        hop_size = 150    # Hop size for overlapping chunks
        min_chunk_size = 50  # Minimum chunk size for meaningful training
        total_frames = sum(s.shape[0] for s in dataset.samples)
        avg_chunk_len = total_frames / len(dataset.samples)
        logger.info(f"[DATASET] Total chunks: {len(dataset)}, Total frames: {total_frames}, Avg chunk len: {avg_chunk_len:.1f}")
        logger.info(f"[DATASET] Chunk size: {chunk_size}, Hop size: {hop_size}, Min chunk: {min_chunk_size}")

        # For SFT: extract and cache speaker embeddings BEFORE training
        # We need one embedding per ORIGINAL audio file, not per chunk
        # Since dataset is now chunked, we need to map chunk -> audio file
        audio_for_speaker_embeds = []
        TARGET_SR = 24000  # Qwen3-TTS requires 24kHz
        if not USE_LORA:
            logger.info("SFT mode: pre-extracting speaker embeddings...")
            # First load all audio for speaker embedding extraction
            for path in AUDIO_PATHS:
                p = Path(path)
                if p.exists():
                    audio_data, sr = sf.read(str(p))
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    # Resample to 24kHz if needed
                    if sr != TARGET_SR:
                        logger.info(f"Resampling {path} from {sr}Hz to {TARGET_SR}Hz")
                        from scipy import signal
                        num_samples = int(len(audio_data) * TARGET_SR / sr)
                        audio_data = signal.resample(audio_data, num_samples)
                        sr = TARGET_SR
                    # Normalize
                    max_val = np.abs(audio_data).max()
                    if max_val > 0:
                        audio_data = audio_data / max_val
                    audio_for_speaker_embeds.append((audio_data, sr))
                    logger.info(f"Loaded: {path}, {len(audio_data)/sr:.1f}s")
                else:
                    audio_for_speaker_embeds.append((None, None))
                    logger.warning(f"Audio file not found: {path}")

            # Pre-extract speaker embeddings for each audio file
            # Create a list that maps dataset index -> speaker embedding
            # We need to track which chunk comes from which audio file
            logger.info("Extracting speaker embeddings from each audio file...")
            speaker_embeddings_cache = []  # List of (1, hidden_size) tensors per audio file
            for audio_data, sr in audio_for_speaker_embeds:
                if audio_data is not None:
                    with torch.no_grad():
                        emb = model.extract_speaker_embedding(audio_data, sr)
                        speaker_embeddings_cache.append(emb.unsqueeze(0))
                        logger.info(f"Extracted embedding shape={emb.shape}")
                else:
                    speaker_embeddings_cache.append(None)
            logger.info(f"Cached {len(speaker_embeddings_cache)} speaker embeddings")

            # Validate speaker embeddings
            for i, emb in enumerate(speaker_embeddings_cache):
                if emb is None:
                    logger.error(f"[VALIDATION] Speaker embedding {i} is None!")
                    raise ValueError(f"Failed to extract speaker embedding from audio file {i}")

                emb_std = emb.std().item()
                emb_mean = emb.abs().mean().item()

                logger.info(f"[SPEAKER_EMB] File {i}: mean={emb_mean:.6f}, std={emb_std:.6f}")

                if emb_std < 0.01:
                    logger.warning(f"[SPEAKER_EMB] File {i} has suspiciously low std={emb_std:.6f} - may indicate silent/corrupt audio")

        # Use native PyTorch training with correct forward approach
        model.train()

        if USE_LORA:
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
        else:
            # SFT mode - train all parameters with lower LR
            for param in model.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            logger.info(f"SFT training: all {sum(1 for _ in model.parameters())} parameters")

        logger.info(f"Starting training: {len(dataset)} samples, {NUM_EPOCHS} epochs")
        logger.info(f"Using forward_sub_talker_finetune for proper loss computation")

        start_time = time.time()

        # For SFT with Base model, load audio files once for speaker embedding extraction
        audio_for_speaker_embeds = []
        TARGET_SR = 24000  # Qwen3-TTS requires 24kHz
        if not USE_LORA:
            logger.info("SFT mode: loading audio for speaker embedding extraction...")
            for path in AUDIO_PATHS:
                p = Path(path)
                if p.exists():
                    audio_data, sr = sf.read(str(p))
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    # Resample to 24kHz if needed (Qwen3-TTS only supports 24kHz)
                    if sr != TARGET_SR:
                        logger.info(f"Resampling {path} from {sr}Hz to {TARGET_SR}Hz for speaker embedding")
                        from scipy import signal
                        num_samples = int(len(audio_data) * TARGET_SR / sr)
                        audio_data = signal.resample(audio_data, num_samples)
                        sr = TARGET_SR
                    # Normalize audio
                    max_val = np.abs(audio_data).max()
                    if max_val > 0:
                        audio_data = audio_data / max_val
                    audio_for_speaker_embeds.append((audio_data, sr))
                    logger.info(f"Loaded audio for speaker embedding: {path}, {len(audio_data)/sr:.1f}s")
                else:
                    audio_for_speaker_embeds.append((None, None))
                    logger.warning(f"Audio file not found for speaker embedding: {path}")

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

            # Training loop - process each chunk
            for sample_idx in range(len(dataset)):
                sample_codes, audio_file_idx = dataset[sample_idx]  # Unpack chunk and audio file index
                seq_len = sample_codes.shape[0]
                device = next(model.parameters()).device

                # For SFT: use cached speaker embedding from the audio file this chunk came from
                # For LoRA: use code embedding averaging
                if not USE_LORA and speaker_embeddings_cache and speaker_embeddings_cache[audio_file_idx] is not None:
                    # Use cached speaker embedding from pre-extracted list
                    talker_hidden = speaker_embeddings_cache[audio_file_idx]
                    logger.debug(f"Chunk {sample_idx} from audio {audio_file_idx}: using cached speaker embedding")
                else:
                    # Fallback: use code embedding averaging
                    with torch.no_grad():
                        audio_embeds = []
                        # First code group uses talker's embeddings
                        embed = model.talker.get_input_embeddings()(
                            sample_codes[0].unsqueeze(0).to(device)
                        )
                        audio_embeds.append(embed)
                        # Remaining code groups use code_predictor's embeddings
                        for g in range(1, num_code_groups):
                            embed = model.talker.code_predictor.get_input_embeddings()[g-1](
                                sample_codes[g].unsqueeze(0).to(device)
                            )
                            audio_embeds.append(embed)
                        audio_embeds = torch.stack(audio_embeds, dim=1).squeeze(0)

                        # Average as speaker representation
                        talker_hidden = audio_embeds.mean(dim=0, keepdim=True)

                # Process each time step
                for step in range(min(seq_len - 1, 50)):
                    codec_ids = sample_codes[step].to(device)

                    try:
                        codec_ids_batch = codec_ids.unsqueeze(0).long().to(device)
                        talker_hidden_batch = talker_hidden.to(device)

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
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}: processing {len(dataset)} chunks, loss={avg_loss:.6f}, batches={num_batches}")

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

        if USE_LORA:
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
            result = {
                "success": True,
                "lora_path": str(lora_path),
                "final_loss": float(avg_loss) if not np.isnan(avg_loss) else 0.0,
                "training_time_seconds": training_time,
            }
        else:
            # SFT mode - save full model
            # Note: model.save_pretrained() fails due to KeyError: 'dtype' in transformers'
            # recursive_diff_dict when comparing config against default. We save manually instead.
            sft_path = Path(OUTPUT_DIR) / "sft_model"
            sft_path.mkdir(parents=True, exist_ok=True)

            # For the merged model directory, use base model as template
            # (has speech_tokenizer/, merges.txt, etc.) and replace model.safetensors
            import shutil
            from huggingface_hub import snapshot_download
            base_model_path = Path(snapshot_download(BASE_MODEL))

            # Copy base model directory structure
            parts = Path(OUTPUT_DIR).name.split('_')
            version_base = '_'.join(parts[:3])  # xiao_s_v33
            merged_name = f"merged_qwen3_tts_{version_base}"
            merged_path = Path(OUTPUT_DIR).parent / merged_name

            if merged_path.exists():
                logger.info(f"[MERGE] Merged SFT model already exists: {merged_path}")
            else:
                # Copy base model directory structure
                shutil.copytree(base_model_path, merged_path, dirs_exist_ok=True)
                logger.info(f"[MERGE] Copied base model to: {merged_path}")

            # =====================================================================
            # OPTION A: Convert Base model to CustomVoice by baking speaker embeddings
            # =====================================================================
            # Step 1: Extract speaker embedding from training audio using speaker_encoder
            # Step 2: Get the speaker embedding index (spk_id) for this persona
            # Step 3: Copy the speaker embedding into talker.codec_embedding at that index
            # Step 4: Remove speaker_encoder from state dict (not needed for CustomVoice)
            # Step 5: Update config to tts_model_type="custom_voice"
            # =====================================================================

            logger.info("[MERGE] Extracting speaker embedding for CustomVoice conversion...")

            # Load first audio file for speaker embedding extraction
            if audio_for_speaker_embeds and audio_for_speaker_embeds[0][0] is not None:
                ref_audio, ref_sr = audio_for_speaker_embeds[0]
            else:
                # Fallback: load from first AUDIO_PATH
                ref_path = Path(AUDIO_PATHS[0])
                if ref_path.exists():
                    ref_audio, ref_sr = sf.read(str(ref_path))
                    if ref_audio.dtype != np.float32:
                        ref_audio = ref_audio.astype(np.float32)
                    max_val = np.abs(ref_audio).max()
                    if max_val > 0:
                        ref_audio = ref_audio / max_val
                else:
                    raise ValueError("No audio available for speaker embedding extraction")

            # Extract speaker embedding using trained model's speaker_encoder
            with torch.no_grad():
                speaker_embedding = model.extract_speaker_embedding(ref_audio, ref_sr)
                # speaker_embedding shape: (hidden_size,) - 1D array
                logger.info(f"[MERGE] Extracted speaker embedding: shape={speaker_embedding.shape}")

            # Get the trained state dict and modify it
            state_dict = model.state_dict()

            # Find the spk_id for this persona from base model config
            # The base model config has spk_id dict like {"xiao_s": 1, "persona_new": 2, ...}
            # We need to find the next available index or use a specific index
            merged_config_path = merged_path / "config.json"
            with open(merged_config_path) as f:
                merged_config = json.load(f)

            # Get or create spk_id for PERSONA_ID
            spk_id_dict = merged_config.get("talker_config", {}).get("spk_id", {})
            if PERSONA_ID.lower() in spk_id_dict:
                spk_id = spk_id_dict[PERSONA_ID.lower()]
            else:
                # Assign next available index (find max + 1)
                max_id = max(spk_id_dict.values()) if spk_id_dict else 0
                spk_id = max_id + 1
                spk_id_dict[PERSONA_ID.lower()] = spk_id
                logger.info(f"[MERGE] Assigned new spk_id={spk_id} for {PERSONA_ID}")

            # Update config with new spk_id
            if "talker_config" not in merged_config:
                merged_config["talker_config"] = {}
            merged_config["talker_config"]["spk_id"] = spk_id_dict

            # Mark speaker as using Chinese dialect to preserve accent
            # spk_is_dialect[speaker] should be the dialect name string (e.g., "chinese")
            # NOT a boolean True - the inference code uses it as codec_language_id[dialect] key
            spk_is_dialect = merged_config.get("talker_config", {}).get("spk_is_dialect", {})
            spk_is_dialect[PERSONA_ID.lower()] = "chinese"  # Use "chinese" for Chinese language preservation
            merged_config["talker_config"]["spk_is_dialect"] = spk_is_dialect

            # Update tts_model_type to custom_voice
            merged_config["tts_model_type"] = "custom_voice"

            # Remove speaker_encoder keys from state dict (not needed for CustomVoice)
            keys_to_remove = [k for k in state_dict.keys() if 'speaker_encoder' in k]
            for k in keys_to_remove:
                del state_dict[k]
                logger.info(f"[MERGE] Removed speaker_encoder key: {k}")

            # =====================================================================
            # Bake speaker embedding into talker.codec_embedding at spk_id index
            # =====================================================================
            # The talker.codec_embedding is an nn.Embedding that stores speaker embeddings
            # indexed by spk_id. For CustomVoice, we copy our extracted speaker embedding
            # into this table at the persona's spk_id position.
            codec_embed_key = "talker.model.codec_embedding.weight"
            if codec_embed_key in state_dict:
                embed_weight = state_dict[codec_embed_key]  # (vocab_size, hidden_size)
                hidden_size = embed_weight.shape[1]

                # Ensure speaker_embedding has correct shape
                if speaker_embedding.shape[0] != hidden_size:
                    # Resample if needed (should match hidden_size)
                    logger.warning(f"[MERGE] Speaker embedding size mismatch: {speaker_embedding.shape[0]} vs {hidden_size}")
                    # Use first hidden_size elements or pad
                    if speaker_embedding.shape[0] > hidden_size:
                        speaker_embedding = speaker_embedding[:hidden_size]
                    else:
                        # Pad with zeros
                        padding = torch.zeros(hidden_size - speaker_embedding.shape[0])
                        speaker_embedding = torch.cat([speaker_embedding, padding])

                # Copy speaker embedding into codec_embedding at spk_id position
                embed_weight[spk_id] = speaker_embedding.to(embed_weight.dtype)
                state_dict[codec_embed_key] = embed_weight
                logger.info(f"[MERGE] Baked speaker embedding into {codec_embed_key} at index {spk_id}")
            else:
                logger.warning(f"[MERGE] {codec_embed_key} not found in state dict. Available keys: {list(state_dict.keys())[:10]}...")

            # Remove speaker_encoder_config from config (not needed for CustomVoice)
            merged_config.pop("speaker_encoder_config", None)
            # Keep tts_model_type = "custom_voice"

            # Save modified state dict to merged path
            from safetensors.torch import save_file
            save_file(state_dict, merged_path / "model.safetensors")
            logger.info(f"[MERGE] Saved trained model (without speaker_encoder) to: {merged_path}")

            # Save updated config
            with open(merged_config_path, "w") as f:
                json.dump(merged_config, f, indent=2)
            logger.info(f"[MERGE] Updated merged config: tts_model_type=custom_voice, spk_id={spk_id_dict}")

            # Also save filtered config.json in sft_model for reference
            import json as json_module
            config_dict = model.config.to_dict()
            # Keys that Qwen3TTSSpeakerEncoderConfig accepts
            speaker_encoder_valid_keys = {
                'mel_dim', 'enc_dim', 'enc_channels', 'enc_kernel_sizes',
                'enc_dilations', 'enc_attention_channels', 'enc_res2net_scale',
                'enc_se_channels', 'sample_rate'
            }
            # Filter speaker_encoder_config if present
            if 'speaker_encoder_config' in config_dict:
                spk_cfg = config_dict['speaker_encoder_config']
                if isinstance(spk_cfg, dict):
                    config_dict['speaker_encoder_config'] = {
                        k: v for k, v in spk_cfg.items() if k in speaker_encoder_valid_keys
                    }
            # Remove dtype/torch_dtype at top level
            config_dict.pop('dtype', None)
            config_dict.pop('torch_dtype', None)
            with open(sft_path / "config.json", "w") as f:
                json_module.dump(config_dict, f)
            logger.info(f"SFT model saved to: {sft_path}")

            result = {
                "success": True,
                "sft_path": str(sft_path),
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

            # Wait for process with timeout - read output in real-time
            stdout_lines = []
            start_time = time.time()
            # SFT training is slow - full model training with chunked audio can take 4-8 hours
            # Increase from 2 hours to 8 hours to allow full training
            timeout_seconds = 28800  # 8 hour max for SFT training

            # Read stdout in real-time using non-blocking reads
            import select
            while True:
                # Check if process exited
                retcode = process.poll()
                if retcode is not None:
                    # Process exited
                    remaining = process.stdout.read()
                    if remaining:
                        stdout_lines.append(remaining)
                    break

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    process.kill()
                    logger.error(f"[TRAINING:{self.version_id[:8]}] Training timed out after {timeout_seconds}s")
                    stdout_lines.append(f"\n[TIMEOUT] Training killed after {timeout_seconds}s")
                    # Update progress.json status to "failed" so UI shows correct status
                    try:
                        tracker_file = self.version_dir / "progress.json"
                        if tracker_file.exists():
                            with open(tracker_file) as f:
                                prog = json.load(f)
                            prog["status"] = "failed"
                            prog["error_message"] = f"Training timed out after {timeout_seconds}s"
                            with open(tracker_file, "w") as f:
                                json.dump(prog, f)
                            logger.info(f"[TRAINING:{self.version_id[:8]}] Updated progress.json to failed status")
                    except Exception as e:
                        logger.warning(f"[TRAINING:{self.version_id[:8]}] Failed to update progress.json: {e}")
                    break

                # Read available stdout (non-blocking)
                try:
                    if select.select([process.stdout], [], [], 1.0)[0]:
                        chunk = process.stdout.read(4096)
                        if chunk:
                            stdout_lines.append(chunk)
                            # Also log chunk for real-time monitoring
                            for line in chunk.splitlines(keepends=True):
                                if line.strip():
                                    logger.info(f"[TRAINING:{self.version_id[:8]}] {line.rstrip()}")
                except (OSError, IOError):
                    pass

            stdout = "".join(stdout_lines)

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

                    # For SFT, the model is already saved to merged_qwen3_tts_{version_base}
                    # in the parent directory (same location as LoRA merged models)
                    if self.training_type == "sft":
                        parts = self.version_dir.name.split('_')
                        version_base = '_'.join(parts[:3])  # xiao_s_v32
                        merged_name = f"merged_qwen3_tts_{version_base}"
                        merged_path = self.version_dir.parent / merged_name
                        if merged_path.exists():
                            logger.info(f"[TRAINING:{self.version_id[:8]}] SFT model ready: {merged_path}")
                        else:
                            merged_path = None
                            logger.error(f"[TRAINING:{self.version_id[:8]}] SFT model not found at {merged_path}")
                    else:
                        # LoRA training - need to merge adapter with base model
                        merged_path = merge_lora(self.version_dir)
                        if merged_path and merged_path.exists():
                            logger.info(f"[TRAINING:{self.version_id[:8]}] Merge complete: {merged_path}")
                        else:
                            logger.error(f"[TRAINING:{self.version_id[:8]}] Merge failed or returned None")

                    if merged_path and merged_path.exists():
                        # Update tracker status
                        if tracker_file.exists():
                            with open(tracker_file) as f:
                                prog = json.load(f)
                            prog["status"] = "ready"
                            with open(tracker_file, "w") as f:
                                json.dump(prog, f)

                        # Update version manager status to "ready" in index.json
                        try:
                            from app.services.training import get_version_manager
                            vm = get_version_manager()
                            # Extract version_id from version_dir name (e.g., "test_v1_20260424_005711")
                            # The version_id in index.json is like "v1_20260424_005711"
                            # We need to find by lora_path
                            matching = [v for v in vm.list_versions() if v.lora_path and Path(v.lora_path) == self.version_dir]
                            if matching:
                                latest = matching[0]
                                vm.update_version_status(
                                    latest.version_id,
                                    "ready",
                                    final_loss=self._result.final_loss if hasattr(self._result, 'final_loss') else None,
                                    training_time_seconds=self._result.training_time_seconds if hasattr(self._result, 'training_time_seconds') else None
                                )
                                logger.info(f"[TRAINING:{self.version_id[:8]}] Updated index.json status to ready for {latest.version_id}")
                        except Exception as vm_err:
                            logger.warning(f"[TRAINING:{self.version_id[:8]}] Failed to update version manager: {vm_err}")

                        # Auto-activate the new merged model
                        try:
                            from app.services.tts import get_tts_engine
                            engine = get_tts_engine()
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
                        logger.error(f"[TRAINING:{self.version_id[:8]}] Model preparation failed (SFT: sft_model missing or LoRA: merge failed)")
                        # Update status to failed since model is critical
                        if tracker_file.exists():
                            with open(tracker_file) as f:
                                prog = json.load(f)
                            prog["status"] = "failed"
                            prog["error_message"] = "Model preparation failed"
                            with open(tracker_file, "w") as f:
                                json.dump(prog, f)
                except Exception as merge_err:
                    logger.error(f"[TRAINING:{self.version_id[:8]}] Merge exception: {merge_err}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            logger.error(f"[TRAINING:{self.version_id[:8]}] Training error: {e}")
            self._result = TrainingResult(success=False, error=str(e))
        finally:
            # Always release training locks after training completes (success or failure)
            self._release_training_locks()

    def _release_training_locks(self):
        """Release TTS/ASR training locks after SFT training completes."""
        if self.training_type != "sft":
            return
        logger.info(f"[TRAINING:{self.version_id[:8]}] Releasing SFT training locks...")
        try:
            from app.services.tts.qwen_tts_engine import set_tts_training_lock
            set_tts_training_lock(False)
            logger.info("[TRAINING] TTS training lock released")
        except Exception as e:
            logger.warning(f"[TRAINING] Failed to release TTS lock: {e}")
        try:
            from app.services.asr.engine import set_asr_training_lock
            set_asr_training_lock(False)
            logger.info("[TRAINING] ASR training lock released")
        except Exception as e:
            logger.warning(f"[TRAINING] Failed to release ASR lock: {e}")

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
