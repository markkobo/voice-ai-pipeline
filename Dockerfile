FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-dev \
    libsndfile1 \
    portaudio19-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install torchaudio matching CUDA 12.1
RUN pip install --no-cache-dir \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install faster-qwen3-tts and qwen-tts
RUN pip install --no-cache-dir \
    faster-qwen3-tts \
    qwen-tts

# Install flash-attn for faster inference (optional, reduces TTS latency)
# RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Set HuggingFace cache to workspace (persistent volume)
ENV HF_HOME=/workspace/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

# Copy project
COPY . /workspace/voice-ai-pipeline-1/
WORKDIR /workspace/voice-ai-pipeline-1

# Expose ports
EXPOSE 8080 9090

# Default: run with real services
CMD ["python", "-m", "app.main"]
