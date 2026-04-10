#!/bin/bash
#
# Voice AI Pipeline - Container Setup Script
# Usage: ./setup_container.sh [command]
#
# Commands:
#   build   Build Docker image
#   start   Start container (creates if not exists)
#   stop    Stop and remove container
#   restart Restart container (stop + start)
#   logs    Show container logs
#   shell   Get interactive shell in container
#   clean   Remove container AND all data volumes (stateless reset)
#
# Options:
#   --mock   Use mock TTS and ASR (no GPU required)
#   --force  Force rebuild image

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Config
CONTAINER_NAME="voice-ai-pipeline"
IMAGE_NAME="voice-ai-pipeline:latest"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Env file (user manages this locally)
ENV_FILE="$PROJECT_DIR/.env"

# Parse args
COMMAND="${1:-start}"
USE_MOCK=false
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mock)  USE_MOCK=true; shift ;;
        --force) FORCE_REBUILD=true; shift ;;
        *)       COMMAND="$1"; shift ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================

docker_running() {
    docker info &>/dev/null
}

image_exists() {
    docker image inspect "$IMAGE_NAME" &>/dev/null
}

container_exists() {
    docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

container_running() {
    docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# =============================================================================
# Build Docker image
# =============================================================================
do_build() {
    echo -e "${CYAN}[Build]${NC} Building Docker image..."

    if [ "$FORCE_REBUILD" = true ]; then
        echo -e "${YELLOW}[Build]${NC} --force: Removing old image..."
        docker rmi "$IMAGE_NAME" 2>/dev/null || true
    fi

    docker build -t "$IMAGE_NAME" "$PROJECT_DIR"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build complete${NC}"
    else
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi
}

# =============================================================================
# Start container
# =============================================================================
do_start() {
    echo -e "${CYAN}[Start]${NC} Starting container..."

    if ! docker_running; then
        echo -e "${RED}✗ Docker is not running${NC}"
        exit 1
    fi

    # Build if needed
    if ! image_exists; then
        echo -e "${YELLOW}[Start]${NC} Image not found, building..."
        do_build
    fi

    # Stop existing container if running
    if container_running; then
        echo -e "${YELLOW}[Start]${NC} Container already running, restarting..."
        docker stop "$CONTAINER_NAME" &>/dev/null || true
    fi

    # Remove old container if exists
    if container_exists; then
        docker rm "$CONTAINER_NAME" &>/dev/null || true
    fi

    # Build docker run command
    DOCKER_RUN=(
        docker run
        --name "$CONTAINER_NAME"
        --gpus all
        --privileged
        -p 8080:8080
        -p 9090:9090
        -v "$PROJECT_DIR:/workspace/voice-ai-pipeline"
        -v "$HF_CACHE_DIR:/root/.cache/huggingface"
        -w /workspace/voice-ai-pipeline
        --restart unless-stopped
        -d
    )

    # Add mock flags if requested
    if [ "$USE_MOCK" = true ]; then
        echo -e "${CYAN}[Start]${NC} Using mock TTS/ASR (no GPU required)"
        DOCKER_RUN+=(-e USE_MOCK_TTS=true -e USE_QWEN_ASR=false)
    fi

    # Source .env and pass environment variables
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
        DOCKER_RUN+=(
            -e OPENAI_API_KEY="$OPENAI_API_KEY"
            -e OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
            -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
            -e HF_TOKEN="$HF_TOKEN"
            -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
            -e USE_QWEN_ASR="${USE_QWEN_ASR:-true}"
            -e USE_MOCK_LLM="${USE_MOCK_LLM:-false}"
            -e USE_MOCK_TTS="${USE_MOCK_TTS:-false}"
        )
    else
        echo -e "${YELLOW}[Start]${NC} Warning: .env not found, using defaults"
    fi

    DOCKER_RUN+=("$IMAGE_NAME" python -m app.main)

    echo -e "${CYAN}[Start]${NC} Running container..."
    "${DOCKER_RUN[@]}"

    # Wait for health
    echo -e "${YELLOW}[Start]${NC} Waiting for server..."
    for i in {1..60}; do
        if docker exec "$CONTAINER_NAME" curl -s http://localhost:8080/health &>/dev/null 2>&1; then
            echo -e "${GREEN}✓ Server ready!${NC}"
            echo ""
            echo -e "${GREEN}=== Voice AI Pipeline ===${NC}"
            echo "  Main UI:      http://localhost:8080/ui"
            echo "  Recordings:   http://localhost:8080/ui/recordings"
            echo "  Training:    http://localhost:8080/ui/training"
            echo "  API Docs:    http://localhost:8080/docs"
            echo "  Metrics:     http://localhost:9090/metrics"
            echo ""
            echo "Container: $CONTAINER_NAME"
            return 0
        fi
        sleep 2
    done

    echo -e "${RED}✗ Server failed to start${NC}"
    echo "Logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -30
    return 1
}

# =============================================================================
# Stop container
# =============================================================================
do_stop() {
    echo -e "${CYAN}[Stop]${NC} Stopping container..."

    if container_running; then
        docker stop "$CONTAINER_NAME"
        echo -e "${GREEN}✓ Stopped${NC}"
    else
        echo -e "${YELLOW}Container not running${NC}"
    fi
}

# =============================================================================
# Restart container
# =============================================================================
do_restart() {
    do_stop
    sleep 2
    do_start
}

# =============================================================================
# Show logs
# =============================================================================
do_logs() {
    if container_exists; then
        docker logs -f "$CONTAINER_NAME" 2>&1
    else
        echo -e "${RED}Container not found${NC}"
    fi
}

# =============================================================================
# Interactive shell
# =============================================================================
do_shell() {
    if container_exists; then
        docker exec -it "$CONTAINER_NAME" /bin/bash
    else
        echo -e "${RED}Container not found${NC}"
    fi
}

# =============================================================================
# Clean everything (stateless reset)
# =============================================================================
do_clean() {
    echo -e "${RED}[Clean]${NC} This will remove ALL data and the container!"
    echo -e "${RED}  - Container: $CONTAINER_NAME${NC}"
    echo -e "${RED}  - Data dir: $DATA_DIR${NC}"
    echo -e "${RED}  - Recordings, models, voice_profiles will be DELETED${NC}"
    echo ""
    read -p "Are you sure? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi

    # Stop and remove container
    if container_exists; then
        echo -e "${YELLOW}[Clean]${NC} Removing container..."
        docker rm -f "$CONTAINER_NAME" &>/dev/null || true
    fi

    # Clean data (except .gitkeep)
    echo -e "${YELLOW}[Clean]${NC} Cleaning data directory..."
    for dir in recordings models voice_profiles; do
        if [ -d "$DATA_DIR/$dir" ]; then
            rm -rf "$DATA_DIR/$dir"/*
            echo "  Cleaned $DATA_DIR/$dir/"
        fi
    done

    echo -e "${GREEN}✓ Clean complete${NC}"
    echo "Next: run './setup_container.sh start' to start fresh"
}

# =============================================================================
# Status
# =============================================================================
do_status() {
    if container_running; then
        echo -e "${GREEN}✓ Container running${NC}"
        echo "  Name: $CONTAINER_NAME"
        echo "  Image: $IMAGE_NAME"
        echo ""
        echo "  URLs:"
        echo "    UI:         http://localhost:8080/ui"
        echo "    Health:     http://localhost:8080/health"
        echo "    Metrics:    http://localhost:9090/metrics"
    elif container_exists; then
        echo -e "${YELLOW}⚠ Container stopped${NC}"
        echo "  Run './setup_container.sh start' to start"
    else
        echo -e "${RED}✗ Container not found${NC}"
        echo "  Run './setup_container.sh build && start' to create"
    fi
}

# =============================================================================
# Main
# =============================================================================
echo -e "${CYAN}=== Voice AI Container Setup ===${NC}"
echo "Project: $PROJECT_DIR"
echo "Data: $DATA_DIR"
echo "Command: $COMMAND"
echo ""

case "$COMMAND" in
    build)  do_build ;;
    start)  do_start ;;
    stop)   do_stop ;;
    restart) do_restart ;;
    logs)   do_logs ;;
    shell)  do_shell ;;
    clean)  do_clean ;;
    status) do_status ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo "Usage: $0 [build|start|stop|restart|logs|shell|clean|status] [--mock] [--force]"
        exit 1
        ;;
esac
