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

# Install PyTorch (CUDA 12.4) - 2.6+ needed for CUDA graphs
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install torchaudio matching CUDA 12.4
RUN pip install --no-cache-dir \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install faster-qwen3-tts and qwen-tts
RUN pip install --no-cache-dir \
    faster-qwen3-tts \
    qwen-tts

# Set HuggingFace cache - mounted from host at /root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=0

# Working directory
WORKDIR /workspace/voice-ai-pipeline-1

# Copy project (host mounts volume to this path)
COPY . .

# Expose ports
EXPOSE 8080 9090

# Default: run with real services
CMD ["python", "-m", "app.main"]
