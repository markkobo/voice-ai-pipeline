#!/usr/bin/env bash
# Run the Cloudflare *named* tunnel for everhome.mkk.dev (and voice.mkk.dev).
#
# Reads CF_TOKEN (tunnel connector JWT) from .env.
# Idempotent: kills any previously-running cloudflared (quick or named) first.
#
# The tunnel UUID this token is bound to: 42db264b-080f-4dde-8f15-f6fbfac1138a
# (tunnel name: voice-ai-pipeline)
#
# Ingress is managed remotely via Cloudflare API (config_src=cloudflare), so
# this script just runs the connector. To change hostname routing, PUT to
#   /accounts/{account_id}/cfd_tunnel/{uuid}/configurations
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load CF_TOKEN from .env
if [[ ! -f .env ]]; then
  echo "ERROR: .env not found in $ROOT_DIR" >&2
  exit 1
fi
# shellcheck disable=SC1091
set -a; source .env; set +a

if [[ -z "${CF_TOKEN:-}" ]]; then
  echo "ERROR: CF_TOKEN not set in .env" >&2
  exit 1
fi

CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$HOME/.local/bin/cloudflared}"
if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  CLOUDFLARED_BIN="$(command -v cloudflared || true)"
fi
if [[ -z "$CLOUDFLARED_BIN" || ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "ERROR: cloudflared binary not found" >&2
  exit 1
fi

LOG_FILE="${CF_LOG_FILE:-/tmp/cloudflared-named.log}"

# Kill any existing cloudflared (quick tunnel or stale named connector)
EXISTING_PIDS="$(pgrep -f '^cloudflared ' || true)"
if [[ -n "$EXISTING_PIDS" ]]; then
  echo "Killing existing cloudflared PIDs: $EXISTING_PIDS"
  # shellcheck disable=SC2086
  kill $EXISTING_PIDS 2>/dev/null || true
  sleep 2
  # Force kill if still alive
  STILL_ALIVE="$(pgrep -f '^cloudflared ' || true)"
  if [[ -n "$STILL_ALIVE" ]]; then
    # shellcheck disable=SC2086
    kill -9 $STILL_ALIVE 2>/dev/null || true
  fi
fi

echo "Starting named tunnel connector -> https://everhome.mkk.dev"
echo "Log: $LOG_FILE"
nohup "$CLOUDFLARED_BIN" tunnel --no-autoupdate run --token "$CF_TOKEN" \
  > "$LOG_FILE" 2>&1 &
NEW_PID=$!
disown "$NEW_PID" 2>/dev/null || true
echo "cloudflared PID: $NEW_PID"
echo "Tail logs: tail -f $LOG_FILE"
