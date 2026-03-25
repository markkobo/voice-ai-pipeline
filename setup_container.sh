#!/bin/bash
# =============================================================================
# Voice AI Pipeline — Container Setup Script
# Run this ONCE on a new container to get everything working.
# =============================================================================

set -e

echo "=== Voice AI Pipeline Container Setup ==="
echo ""

# Detect location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check for .env
if [ ! -f ".env" ]; then
    echo "[1/6] Creating .env from template..."
    cat > .env << 'ENVEOF'
# Copy your API keys here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: HuggingFace token for gated models (not needed if models already cached)
# HF_TOKEN=hf_your_token_here

# Optional: Use mock services for testing (set to true for testing without GPU)
# USE_MOCK_TTS=true
# USE_MOCK_LLM=true
# USE_QWEN_ASR=false
ENVEOF
    echo "  → Created .env — please fill in OPENAI_API_KEY!"
else
    echo "[1/6] .env already exists — skipping"
fi

# 2. Check CUDA
echo ""
echo "[2/6] Checking CUDA..."
python3 -c "import torch; print(f'  → PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')" 2>/dev/null || echo "  → PyTorch not installed or no CUDA"

# 3. Fix torchaudio CUDA version if needed
echo ""
echo "[3/6] Ensuring torchaudio matches PyTorch CUDA version..."
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
if [ "$TORCH_CUDA" != "none" ]; then
    TORCHAUDIO_CUDA=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null | grep -oP 'cu\d+' || echo "none")
    echo "  → PyTorch CUDA: $TORCH_CUDA, torchaudio CUDA: $TORCHAUDIO_CUDA"
    if [ "$TORCHAUDIO_CUDA" != "$TORCH_CUDA" ]; then
        echo "  → Mismatch! Reinstalling torchaudio for CUDA $TORCH_CUDA..."
        pip install "torchaudio==2.4.1+cu121" --index-url "https://download.pytorch.org/whl/cu121"
    else
        echo "  → Match — no action needed"
    fi
else
    echo "  → No CUDA PyTorch — skipping torchaudio fix"
fi

# 4. Set HF_HOME to workspace (persistent)
echo ""
echo "[4/6] Configuring HuggingFace cache..."
if [ -d ".cache/huggingface/hub" ] && [ $(du -s .cache/huggingface/hub | cut -f1) -gt 1000 ]; then
    echo "  → HF cache found in workspace ($(du -sh .cache/huggingface/hub | cut -f1))"
    echo "  → Setting HF_HOME to $(pwd)/.cache"
    export HF_HOME="$(pwd)/.cache/huggingface"
    if ! grep -q "HF_HOME" .env 2>/dev/null; then
        echo "HF_HOME=$(pwd)/.cache/huggingface" >> .env
    fi
    echo "  → HF_HOME=$HF_HOME"
    echo "  → Cached models:"
    ls -lh .cache/huggingface/hub/ 2>/dev/null | grep "^d" || echo "    (none found)"
else
    echo "  → No local HF cache found in workspace"
    echo "  → Models will download on first use (need ~9GB)"
fi

# 5. Check required Python packages
echo ""
echo "[5/6] Checking Python packages..."
python3 -c "import fastapi, uvicorn, pydub, websockets, transformers" 2>/dev/null && echo "  → Core packages OK" || echo "  → Missing packages — run: pip install -r requirements.txt"
python3 -c "from faster_qwen3_tts import FasterQwen3TTS" 2>/dev/null && echo "  → faster-qwen3-tts OK" || echo "  → faster-qwen3-tts not installed"
python3 -c "from qwen_tts import Qwen3TTSModel" 2>/dev/null && echo "  → qwen-tts OK" || echo "  → qwen-tts not installed"

# 6. Quick model check
echo ""
echo "[6/6] Checking cached models..."
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
for model_dir in "$HF_HOME/hub"/*/; do
    if [ -d "$model_dir" ]; then
        name=$(basename "$model_dir" | sed 's/models--//' | sed 's/--/\//g' | sed 's/-/ /g')
        size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
        echo "  → $name ($size)"
    fi
done

# Final summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your OPENAI_API_KEY"
echo "  2. Run: python -m app.main"
echo "  3. Open: http://localhost:8080/ui"
echo ""
echo "For testing without GPU:"
echo "  USE_MOCK_TTS=true USE_MOCK_LLM=true USE_QWEN_ASR=false python -m app.main"
echo ""
