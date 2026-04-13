#!/bin/bash
# 遇到任何錯誤立即停止執行，不要帶著殘破的狀態繼續跑
set -e

# 正確的 ANSI 顏色代碼
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENV_FILE=".env"
LOG_FILE="litellm.log"

echo -e "${YELLOW}>>> Starting Environment Check (Root/Ubuntu)...${NC}"

# 1. Node.js Check & Install (升級目標設為 20 以確保最大相容性)
NODE_MAJOR=20
CURRENT_NODE_VER=$(node -v 2>/dev/null | cut -d'.' -f1 | tr -d 'v')

if [ -z "$CURRENT_NODE_VER" ] || [ "$CURRENT_NODE_VER" -lt "$NODE_MAJOR" ]; then
    echo -e "${YELLOW}[!] Node.js missing or < v$NODE_MAJOR. Upgrading to v$NODE_MAJOR...${NC}"
    apt-get update && apt-get install -y curl ca-certificates gnupg
    # 修正：補上缺失的 /setup_ 路徑
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -
    apt-get install -y nodejs
    hash -r
else
    echo -e "${GREEN}[✓] Node.js $(node -v) is sufficient.${NC}"
fi

# 2. Dependency Check
echo -e "${YELLOW}>>> Checking Python & System Dependencies...${NC}"
command -v lsof &> /dev/null || apt-get install -y lsof
command -v pip3 &> /dev/null || apt-get install -y python3-pip
command -v litellm &> /dev/null || pip3 install 'litellm[proxy]'

echo -e "${YELLOW}>>> Installing Claude Code...${NC}"
npm install -g @anthropic-ai/claude-code

# 3. Handle API Key (資安隔離)
if [ -f "$ENV_FILE" ]; then
    # 安全地讀取 .env，忽略註解
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

if [ -z "$MINIMAX_API_KEY" ]; then
    echo -e "${YELLOW}[?] MiniMax API Key not found in $ENV_FILE.${NC}"
    echo -n "Please paste your NEW MiniMax API Key (Do not use the revoked one): "
    read -s USER_KEY
    echo ""
    echo "MINIMAX_API_KEY=$USER_KEY" >> "$ENV_FILE"
    export MINIMAX_API_KEY=$USER_KEY
fi

# 4. LiteLLM Config Generation (核心路由修正)
echo -e "${YELLOW}>>> Generating LiteLLM Config...${NC}"
cat <<EOF > litellm_config.yaml
model_list:
  - model_name: anthropic/MiniMax-M2.7
    litellm_params:
      model: anthropic/MiniMax-M2.7
      api_base: "https://api.minimaxi.com/anthropic"
      api_key: "os.environ/MINIMAX_API_KEY"
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: anthropic/MiniMax-M2.7
      api_base: "https://api.minimaxi.com/anthropic"
      api_key: "os.environ/MINIMAX_API_KEY"
litellm_settings:
  drop_params: true
EOF

# 5. Export Claude Code Environment Variables
# 這兩行是欺騙 Claude SDK 的關鍵，不可省略
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_API_KEY="sk-fake-key-for-litellm"
export ANTHROPIC_MODEL="anthropic/MiniMax-M2.7"

# 6. Start Proxy and Pipe Logs
echo -e "${YELLOW}>>> Starting LiteLLM Proxy on port 4000...${NC}"
lsof -ti:4000 | xargs kill -9 2>/dev/null || true
echo "--- Session Start $(date) ---" > "$LOG_FILE"

nohup litellm --config litellm_config.yaml --port 4000 >> "$LOG_FILE" 2>&1 &
PROXY_PID=$!

# 等待 Proxy 啟動並驗證存活
sleep 3
if ! kill -0 $PROXY_PID 2>/dev/null; then
    echo -e "${RED}[!] LiteLLM failed to start. Check $LOG_FILE${NC}"
    cat "$LOG_FILE"
    exit 1
fi
echo -e "${GREEN}[✓] LiteLLM Proxy is running (PID: $PROXY_PID)${NC}"

# 7. Auto-cleanup
cleanup() { 
    echo -e "${YELLOW}>>> Shutting down Proxy (PID: $PROXY_PID)...${NC}"
    kill $PROXY_PID 2>/dev/null || true
}
trap cleanup EXIT

# 8. Start Claude Code
echo -e "${GREEN}=========================================${NC}"
echo -e "  Claude Code Proxy Env is Ready."
echo -e "${GREEN}=========================================${NC}"

# 直接呼叫 claude，hash -r 和 set -e 會確保它存在且可執行
claude
