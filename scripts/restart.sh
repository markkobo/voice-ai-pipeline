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
#   --watch       Watch for file changes and auto-restart (background daemon)
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
WATCHER_PID_FILE="/tmp/voice-ai-watcher.pid"

# Parse arguments
MODE="auto"
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock-tts)  USE_MOCK_TTS="true"; shift ;;
        --mock-asr)  USE_QWEN_ASR="false"; shift ;;
        --port)      PORT="$2"; shift 2 ;;
        --force)     MODE="force"; shift ;;
        --code)      MODE="code"; shift ;;
        --watch)     MODE="watch"; shift ;;
        --ui)        MODE="ui"; shift ;;
        --help)
            grep "^#" "$0" | head -15 | sed 's/^# //'
            exit 0 ;;
        *)  echo -e "${RED}Unknown: $1${NC}"; exit 1 ;;
    esac
done

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        kill -0 "$pid" 2>/dev/null && return 0
    fi
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

last_git_commit() {
    git -C /workspace/voice-ai-pipeline-1 rev-parse HEAD 2>/dev/null
}

code_changed() {
    local current_commit=$(last_git_commit)
    local last_commit=""
    if [ -f "$LAST_COMMIT_FILE" ]; then
        last_commit=$(cat "$LAST_COMMIT_FILE")
    fi
    # True if commit changed OR if there are uncommitted changes
    [ "$current_commit" != "$last_commit" ] || [ -n "$(git -C /workspace/voice-ai-pipeline-1 diff --name-only HEAD 2>/dev/null)" ]
}

# Get list of changed files (both committed AND uncommitted) since last restart
get_changed_files() {
    local last_commit=""
    if [ -f "$LAST_COMMIT_FILE" ]; then
        last_commit=$(cat "$LAST_COMMIT_FILE")
    fi

    local changed=""

    # Committed changes since last recorded commit
    if [ -n "$last_commit" ]; then
        changed=$(git -C /workspace/voice-ai-pipeline-1 diff --name-only "$last_commit" HEAD 2>/dev/null)
    fi

    # Add uncommitted changes (diff HEAD vs working tree)
    local uncommitted=$(git -C /workspace/voice-ai-pipeline-1 diff --name-only HEAD 2>/dev/null)
    if [ -n "$uncommitted" ]; then
        if [ -n "$changed" ]; then
            changed="$changed"$'\n'"$uncommitted"
        else
            changed="$uncommitted"
        fi
    fi

    echo "$changed"
}

# Categorize changed files by component
# Returns: ui_only | server | full_restart
categorize_change() {
    local changed_files=$(get_changed_files | sort -u | grep -v '^$')
    local has_ui=false
    local has_python=false

    for f in $changed_files; do
        case "$f" in
            *.html|*.js|*.css)
                has_ui=true
                ;;
            *.py)
                # All Python files require server restart (even standalone_ui.py embeds HTML in a string)
                # The server must reload the Python module to pick up changes
                has_python=true
                ;;
        esac
    done

    if [ "$has_python" = true ]; then
        echo "full_restart"
    elif [ "$has_ui" = true ]; then
        echo "ui_only"
    else
        echo "none"
    fi
}

# =============================================================================
# Stop server (graceful, with fallback to kill -9)
# =============================================================================
stop_server() {
    echo -e "${YELLOW}[Stop]${NC} Stopping server..."

    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            kill "$OLD_PID" 2>/dev/null || true
            for i in {1..5}; do
                if ! kill -0 "$OLD_PID" 2>/dev/null; then
                    echo -e "${GREEN}✓ Stopped gracefully (PID $OLD_PID)${NC}"
                    break
                fi
                sleep 1
            done
            if kill -0 "$OLD_PID" 2>/dev/null; then
                kill -9 "$OLD_PID" 2>/dev/null || true
                echo -e "${YELLOW}✓ Force killed PID $OLD_PID${NC}"
            fi
        fi
        rm -f "$PID_FILE"
    fi

    # Kill stray processes on ports
    for pid in $(lsof -ti :$PORT 2>/dev/null || true); do
        [ "$pid" = "$OLD_PID" ] && continue
        kill "$pid" 2>/dev/null || true
    done
    for pid in $(lsof -ti :9090 2>/dev/null || true); do
        kill "$pid" 2>/dev/null || true
    done

    sleep 2
}

# =============================================================================
# Start server (returns PID)
# =============================================================================
start_server() {
    echo -e "${YELLOW}[Start]${NC} Starting server on port $PORT..."

    export USE_MOCK_TTS="${USE_MOCK_TTS:-false}"
    export USE_QWEN_ASR="${USE_QWEN_ASR:-true}"
    export PYTHONUNBUFFERED=1

    cd /workspace/voice-ai-pipeline-1
    nohup python3 -m app.main > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"

    # Record commit hash
    last_git_commit > "$LAST_COMMIT_FILE"

    echo "  PID: $SERVER_PID | Log: $LOG_FILE"
}

# =============================================================================
# Wait for server to be ready
# =============================================================================
wait_ready() {
    echo -e "${YELLOW}[Wait]${NC} Waiting for server ready..."

    max_wait=120
    counter=0
    while [ $counter -lt $max_wait ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ready!${NC}"
            return 0
        fi

        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo -e "${RED}✗ Server crashed!${NC}"
            tail -20 "$LOG_FILE"
            return 1
        fi

        if [ $((counter % 15)) -eq 0 ]; then
            echo "  Starting... ($counter/$max_wait)s"
        fi
        sleep 1
        counter=$((counter + 1))
    done

    echo -e "${RED}✗ Startup timeout!${NC}"
    tail -30 "$LOG_FILE"
    return 1
}

# =============================================================================
# Verify endpoints
# =============================================================================
verify_endpoints() {
    echo -e "${YELLOW}[Verify]${NC} Checking endpoints..."
    local all_ok=true
    for ep in "/health" "/api/personas" "/api/listeners" "/api/training/versions"; do
        if curl -s "http://localhost:$PORT$ep" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓$ep${NC}"
        else
            echo -e "  ${RED}✗$ep${NC}"
            all_ok=false
        fi
    done
    [ "$all_ok" = true ]
}

# =============================================================================
# Full restart (stop + start + wait + verify)
# =============================================================================
do_restart() {
    local reason="${1:-code change}"
    echo -e "${CYAN}[Restart]${NC} Restarting: $reason"

    stop_server
    start_server

    if ! wait_ready; then
        echo -e "${RED}✗ Restart failed${NC}"
        return 1
    fi

    verify_endpoints

    echo ""
    echo -e "${GREEN}=== Server Ready (PID $(get_pid)) ===${NC}"
    echo "  Main UI:      http://localhost:$PORT/ui"
    echo "  Recordings:   http://localhost:$PORT/ui/recordings"
    echo "  Training:      http://localhost:$PORT/ui/training"
    echo "  API Docs:     http://localhost:$PORT/docs"
    echo "  Metrics:      http://localhost:$PORT:9090/metrics"
}

# =============================================================================
# Watch mode: background daemon that monitors code and auto-restarts
# =============================================================================
do_watch() {
    echo -e "${YELLOW}=== Watch Mode ===${NC}"
    echo "Monitoring code changes and auto-restarting server..."
    echo "PID file: $WATCHER_PID_FILE"
    echo ""

    # Kill existing watcher if any
    if [ -f "$WATCHER_PID_FILE" ]; then
        OLD_WATCHER=$(cat "$WATCHER_PID_FILE")
        kill "$OLD_WATCHER" 2>/dev/null || true
        rm -f "$WATCHER_PID_FILE"
    fi

    # Check prerequisites
    if ! command -v inotifywait &>/dev/null; then
        echo -e "${YELLOW}Installing inotify-tools...${NC}"
        sudo apt-get install -y inotify-tools 2>/dev/null || apt install -y inotify-tools 2>/dev/null || true
    fi

    if ! command -v inotifywait &>/dev/null; then
        echo -e "${RED}inotifywait not available. Install inotify-tools.${NC}"
        echo "  Debian/Ubuntu: sudo apt install inotify-tools"
        echo "  Or use --code mode instead of --watch"
        exit 1
    fi

    # Start watching in background
    (
        LAST_COMMIT=$(last_git_commit 2>/dev/null)

        # Watch app/, telemetry/, and scripts/ directories
        # Exclude: __pycache__, .pyc, .pyo, data/, .git/
        inotifywait -m -r \
            --exclude '(__pycache__|\.pyc|\.pyo|\.git|data|\.claude)' \
            -e modify,create,delete \
            /workspace/voice-ai-pipeline-1/app \
            /workspace/voice-ai-pipeline-1/telemetry \
            /workspace/voice-ai-pipeline-1/scripts \
            2>/dev/null | while read directory filename event; do

            # Check if it's a Python file or shell script
            if [[ "$filename" =~ \.(py|sh)$ ]]; then
                CURRENT_COMMIT=$(last_git_commit 2>/dev/null)

                if [ "$CURRENT_COMMIT" != "$LAST_COMMIT" ]; then
                    CHANGE_TYPE=$(categorize_change)
                    echo ""
                    echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} Code changed: $CHANGE_TYPE"
                    # Re-sync with LAST_COMMIT_FILE after restart
                    if [ -f "$LAST_COMMIT_FILE" ]; then
                        LAST_COMMIT=$(cat "$LAST_COMMIT_FILE")
                    else
                        LAST_COMMIT="$CURRENT_COMMIT"
                    fi

                    case "$CHANGE_TYPE" in
                        ui_only)
                            echo -e "  ${CYAN}→${NC} UI/HTML/JS only — refresh browser (no restart)"
                            ;;
                        full_restart)
                            echo "  Triggering full restart..."
                            if is_running; then
                                echo "  Stopping server..."
                                stop_server
                                sleep 1
                                start_server
                                if wait_ready; then
                                    verify_endpoints
                                    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Auto-restart complete"
                                fi
                            fi
                            ;;
                    esac
                fi
            fi
        done
    ) &
    WATCHER_PID=$!
    echo $WATCHER_PID > "$WATCHER_PID_FILE"

    echo -e "${GREEN}✓ Watcher started (PID $WATCHER_PID)${NC}"
    echo ""
    echo "Watching for .py and .sh changes in:"
    echo "  app/  telemetry/  scripts/"
    echo ""
    echo "Server:"
    if is_running; then
        echo -e "  ${GREEN}✓ Running (PID $(get_pid))${NC}"
        echo "  UI: http://localhost:$PORT/ui"
    else
        echo -e "  ${RED}✗ Not running${NC}"
    fi
    echo ""
    echo "To stop watcher: kill \$(cat $WATCHER_PID_FILE)"
    echo "Or: pkill -f 'inotifywait.*voice-ai'"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
echo -e "${YELLOW}=== Voice AI Smart Restart ===${NC}"
echo "Mode: $MODE | Port: $PORT"

case "$MODE" in
    ui)
        echo -e "${CYAN}[UI Check]${NC}"
        if is_running && curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server healthy (PID $(get_pid))${NC}"
            echo "  http://localhost:$PORT/ui"
            exit 0
        else
            echo -e "${RED}✗ Server not running${NC}"
            exit 1
        fi
        ;;

    watch)
        do_watch
        ;;

    auto)
        if is_running; then
            PID=$(get_pid)
            echo -e "${GREEN}✓ Server running (PID $PID)${NC}"

            if code_changed; then
                CHANGE_TYPE=$(categorize_change)
                echo -e "${YELLOW}[Code Changed]${NC} ($CHANGE_TYPE)"

                case "$CHANGE_TYPE" in
                    ui_only)
                        echo -e "${CYAN}[UI Only]${NC} HTML/JS changed — no restart needed"
                        echo -e "  ${GREEN}✓${NC} Refresh browser: http://localhost:$PORT/ui"
                        echo -e "  ${CYAN}→${NC} Server running (PID $PID) — no restart triggered"
                        exit 0
                        ;;
                    full_restart)
                        do_restart "Python code updated"
                        ;;
                    *)
                        echo -e "${CYAN}[No Change]${NC} No relevant file changes"
                        ;;
                esac
            else
                echo -e "${CYAN}[No Change]${NC} Server up-to-date"
                if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
                    echo -e "${GREEN}✓ Endpoints OK${NC}"
                    echo "  http://localhost:$PORT/ui"
                else
                    echo -e "${YELLOW}Endpoints unhealthy, restarting...${NC}"
                    do_restart "unhealthy endpoints"
                fi
            fi
        else
            echo -e "${YELLOW}[Cold Start]${NC}"
            do_restart "cold start"
        fi
        ;;

    force|code)
        if is_running; then
            do_restart "force restart"
        else
            do_restart "cold start"
        fi
        ;;
esac
