#!/bin/bash
#
# Voice AI Pipeline - Smart Restart Script
# Usage: ./scripts/restart.sh [options]
#
# Options:
#   --mock-tts    Use mock TTS (no GPU required)
#   --mock-asr    Use mock ASR (no GPU required)
#   --port PORT   Override default port 8080
#   --force       Force full restart even if code hasn't changed
#   --code        Only restart if Python code changed (default: auto-detect)
#   --ui          Only check UI is accessible (no restart)
#   --help        Show this help

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PORT="${PORT:-8080}"
LOG_FILE="/tmp/voice-ai-server.log"
PID_FILE="/tmp/voice-ai-server.pid"
LAST_COMMIT_FILE="/tmp/voice-ai-last-commit.txt"

# Parse arguments
MODE="auto"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock-tts)  USE_MOCK_TTS="true"; shift ;;
        --mock-asr)  USE_QWEN_ASR="false"; shift ;;
        --port)      PORT="$2"; shift 2 ;;
        --force)     MODE="force"; shift ;;
        --code)      MODE="code"; shift ;;
        --ui)        MODE="ui"; shift ;;
        --help)
            grep "^#" "$0" | head -15 | sed 's/^# //'
            exit 0 ;;
        *)  echo -e "${RED}Unknown: $1${NC}"; exit 1 ;;
    esac
done

# Check if server is currently running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        kill -0 "$pid" 2>/dev/null && return 0
    fi
    # Also check by port
    lsof -ti :$PORT &>/dev/null && return 0
    return 1
}

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        lsof -ti :$PORT 2>/dev/null || echo ""
    fi
}

# Get last git commit hash for code change detection
last_git_commit() {
    git -C /workspace/voice-ai-pipeline-1 rev-parse HEAD 2>/dev/null
}

# Check if code changed since last restart
code_changed() {
    local current_commit=$(last_git_commit)
    local last_commit=""

    if [ -f "$LAST_COMMIT_FILE" ]; then
        last_commit=$(cat "$LAST_COMMIT_FILE")
    fi

    # Changed if commits differ or file doesn't exist
    [ "$current_commit" != "$last_commit" ]
}

echo -e "${YELLOW}=== Voice AI Smart Restart ===${NC}"
echo "Mode: $MODE | Port: $PORT"

# =============================================================================
# MODE: ui (no restart — just verify UI is accessible)
# =============================================================================
if [ "$MODE" = "ui" ]; then
    echo -e "${CYAN}[UI Check]${NC} Verifying server is accessible..."
    if is_running && curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is running and healthy${NC}"
        echo "  UI: http://localhost:$PORT/ui"
        exit 0
    else
        echo -e "${RED}✗ Server not running${NC}"
        echo "  Run without --ui to restart"
        exit 1
    fi
fi

# =============================================================================
# Check current server status
# =============================================================================
if is_running; then
    PID=$(get_pid)
    echo -e "${GREEN}✓ Server already running (PID: $PID)${NC}"

    case "$MODE" in
        auto)
            if code_changed; then
                echo -e "${YELLOW}[Code Change Detected]${NC} Need to restart"
            else
                echo -e "${CYAN}[No Change]${NC} Server already up-to-date"
                echo ""
                echo -e "  ${CYAN}Hint:${NC} If you just want to verify, run: $0 --ui"
                echo -e "  ${CYAN}Hint:${NC} To force restart, run: $0 --force"
                echo ""
                # Just verify endpoints are healthy
                if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
                    echo -e "${GREEN}✓ All endpoints OK${NC}"
                    echo "  Main UI: http://localhost:$PORT/ui"
                    exit 0
                else
                    echo -e "${RED}✗ Server unhealthy, forcing restart${NC}"
                    MODE="force"
                fi
            fi
            ;;
        force)
            echo -e "${YELLOW}[Force Restart]${NC}"
            ;;
    esac
else
    echo -e "${YELLOW}[Cold Start]${NC} Server not running, starting fresh"
    MODE="force"
fi

# =============================================================================
# Restart needed — kill old process
# =============================================================================
if [ "$MODE" = "force" ] || [ "$MODE" = "auto" ]; then
    echo -e "${YELLOW}[1/4] Stopping existing server...${NC}"

    # Kill by PID file
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            kill "$OLD_PID" 2>/dev/null || true
            # Wait for graceful shutdown
            for i in {1..5}; do
                if ! kill -0 "$OLD_PID" 2>/dev/null; then
                    echo -e "${GREEN}✓ Server stopped gracefully (PID $OLD_PID)${NC}"
                    break
                fi
                sleep 1
            done
            # Force kill if still alive
            if kill -0 "$OLD_PID" 2>/dev/null; then
                kill -9 "$OLD_PID" 2>/dev/null || true
                echo -e "${YELLOW}✓ Force killed PID $OLD_PID${NC}"
            fi
        fi
        rm -f "$PID_FILE"
    fi

    # Kill any stray processes on ports
    for pid in $(lsof -ti :$PORT 2>/dev/null || true); do
        [ "$pid" = "$OLD_PID" ] && continue
        kill "$pid" 2>/dev/null || true
    done
    for pid in $(lsof -ti :9090 2>/dev/null || true); do
        kill "$pid" 2>/dev/null || true
    done

    sleep 2
fi

# =============================================================================
# Start server (only if killed or cold start)
# =============================================================================
if [ "$MODE" = "force" ] || [ "$MODE" = "auto" ] || ! is_running; then
    echo -e "${YELLOW}[2/4] Starting server on port $PORT...${NC}"

    export USE_MOCK_TTS="${USE_MOCK_TTS:-false}"
    export USE_QWEN_ASR="${USE_QWEN_ASR:-true}"
    export PYTHONUNBUFFERED=1

    cd /workspace/voice-ai-pipeline-1
    nohup python3 -m app.main > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"

    # Record commit hash for next time
    last_git_commit > "$LAST_COMMIT_FILE"

    echo "  PID: $SERVER_PID | Log: $LOG_FILE"

    # =============================================================================
    # Wait for startup
    # =============================================================================
    echo -e "${YELLOW}[3/4] Waiting for server ready...${NC}"

    max_wait=120
    counter=0
    while [ $counter -lt $max_wait ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server ready!${NC}"
            break
        fi

        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo -e "${RED}✗ Server crashed!${NC}"
            tail -20 "$LOG_FILE"
            exit 1
        fi

        if [ $((counter % 15)) -eq 0 ]; then
            echo "  Still starting... ($counter/$max_wait)s"
        fi
        sleep 1
        counter=$((counter + 1))
    done

    if [ $counter -ge $max_wait ]; then
        echo -e "${RED}✗ Startup timeout!${NC}"
        tail -30 "$LOG_FILE"
        exit 1
    fi

    # =============================================================================
    # Verify endpoints
    # =============================================================================
    echo -e "${YELLOW}[4/4] Verifying endpoints...${NC}"

    all_ok=true
    for ep in "/health" "/api/personas" "/api/listeners" "/api/training/versions"; do
        if curl -s "http://localhost:$PORT$ep" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓$ep${NC}"
        else
            echo -e "  ${RED}✗$ep${NC}"
            all_ok=false
        fi
    done

    echo ""
    echo -e "${GREEN}=== Server Ready ===${NC}"
    echo ""
    echo "  Main UI:      http://localhost:$PORT/ui"
    echo "  Recordings:    http://localhost:$PORT/ui/recordings"
    echo "  Training:     http://localhost:$PORT/ui/training"
    echo "  API Docs:     http://localhost:$PORT/docs"
    echo "  Metrics:      http://localhost:$PORT:9090/metrics"
    echo "  Log:          $LOG_FILE"
    echo "  PID:          $SERVER_PID"
    echo ""

    if [ "$all_ok" = false ]; then
        echo -e "${YELLOW}Some endpoints failed. Recent log:${NC}"
        tail -10 "$LOG_FILE"
    fi
fi
