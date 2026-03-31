#!/bin/bash
#
# Voice AI Pipeline - Restart Script
# Usage: ./scripts/restart.sh [options]
#
# Options:
#   --mock-tts    Use mock TTS (no GPU required)
#   --mock-asr    Use mock ASR (no GPU required)
#   --port PORT   Override default port 8080
#   --help        Show this help
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
USE_MOCK_TTS="${USE_MOCK_TTS:-false}"
USE_QWEN_ASR="${USE_QWEN_ASR:-true}"
PORT="${PORT:-8080}"
LOG_FILE="/tmp/voice-ai-server.log"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock-tts)
            USE_MOCK_TTS="true"
            shift
            ;;
        --mock-asr)
            USE_QWEN_ASR="false"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | head -20 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}=== Voice AI Pipeline Restart ===${NC}"
echo "Port: $PORT"
echo "Mock TTS: $USE_MOCK_TTS"
echo "Mock ASR: $USE_QWEN_ASR"
echo ""

# Step 1: Kill existing processes
echo -e "${YELLOW}[1/5] Killing existing processes...${NC}"

# Kill by PID file if it exists
if [ -f /tmp/voice-ai-server.pid ]; then
    kill -9 $(cat /tmp/voice-ai-server.pid) 2>/dev/null || true
    rm -f /tmp/voice-ai-server.pid
fi

# Kill any process on our ports
for pid in $(lsof -ti :$PORT 2>/dev/null || true); do
    kill -9 $pid 2>/dev/null || true
done
for pid in $(lsof -ti :9090 2>/dev/null || true); do
    kill -9 $pid 2>/dev/null || true
done

# Also kill any stray app.main processes
pkill -9 -f "python3 -m app.main" 2>/dev/null || true
pkill -9 -f "uvicorn app.main" 2>/dev/null || true

# Wait for ports to be released
sleep 3

# Verify ports are free
if lsof -i :$PORT 2>/dev/null; then
    echo -e "${RED}Port $PORT still in use after kill${NC}"
fi

echo -e "${GREEN}✓ Processes killed${NC}"

# Step 2: Check torch/CUDA
echo -e "${YELLOW}[2/5] Checking PyTorch + CUDA...${NC}"
if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    torch_version=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓ PyTorch $torch_version with CUDA${NC}"
else
    echo -e "${YELLOW}⚠ PyTorch CUDA not available (or using mock)${NC}"
fi

# Step 3: Start server
echo -e "${YELLOW}[3/5] Starting server on port $PORT...${NC}"

# Set environment variables
export USE_MOCK_TTS
export USE_QWEN_ASR
export PYTHONUNBUFFERED=1

# Start in background
cd /workspace/voice-ai-pipeline-1
nohup python3 -m app.main > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/voice-ai-server.pid

echo "Server PID: $SERVER_PID"
echo "Log file: $LOG_FILE"

# Step 4: Wait for server to start
echo -e "${YELLOW}[4/5] Waiting for server startup...${NC}"

max_wait=60
counter=0
while [ $counter -lt $max_wait ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready!${NC}"
        break
    fi

    # Check if process died
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${RED}✗ Server process died! Check log:${NC}"
        tail -30 "$LOG_FILE"
        exit 1
    fi

    # Show startup progress
    if [ $((counter % 10)) -eq 0 ]; then
        echo "  Waiting... ($counter/$max_wait seconds)"
    fi

    sleep 1
    counter=$((counter + 1))
done

if [ $counter -ge $max_wait ]; then
    echo -e "${RED}✗ Server startup timeout!${NC}"
    echo -e "${YELLOW}Last 50 lines of log:${NC}"
    tail -50 "$LOG_FILE"
    exit 1
fi

# Step 5: Verify endpoints
echo -e "${YELLOW}[5/5] Verifying endpoints...${NC}"

endpoints=(
    "/health"
    "/api/personas"
    "/api/listeners"
    "/api/training/versions"
)

all_ok=true
for endpoint in "${endpoints[@]}"; do
    if curl -s "http://localhost:$PORT$endpoint" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $endpoint${NC}"
    else
        echo -e "${RED}✗ $endpoint failed${NC}"
        all_ok=false
    fi
done

echo ""
echo -e "${GREEN}=== Server Started Successfully ===${NC}"
echo ""
echo "  Main UI:      http://localhost:$PORT/ui"
echo "  Recordings:   http://localhost:$PORT/ui/recordings"
echo "  Training:     http://localhost:$PORT/ui/training"
echo "  API Docs:     http://localhost:$PORT/docs"
echo "  Metrics:      http://localhost:$PORT:9090/metrics"
echo "  Log:          $LOG_FILE"
echo ""
echo "  PID:          $SERVER_PID"
echo ""

# Show recent log if there were issues
if [ "$all_ok" = false ]; then
    echo -e "${YELLOW}Some endpoints failed. Recent log:${NC}"
    tail -20 "$LOG_FILE"
fi
