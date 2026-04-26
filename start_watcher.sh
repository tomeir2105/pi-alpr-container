#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_FILE="$SCRIPT_DIR/.env"

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE"; exit 1; }

"$SCRIPT_DIR/stop_watcher.sh" || true

APP_ENV_FILE=$(grep -m1 '^APP_ENV_FILE=' "$ENV_FILE" | cut -d= -f2-)
APP_ENV_FILE=${APP_ENV_FILE:-/mnt/localdisk/pi-alpr/config/.env}
HOST_DATA_DIR=$(grep -m1 '^HOST_DATA_DIR=' "$ENV_FILE" | cut -d= -f2-)
HOST_DATA_DIR=${HOST_DATA_DIR:-/mnt/localdisk/pi-alpr}

echo "Syncing $ENV_FILE to $APP_ENV_FILE"
mkdir -p "$(dirname "$APP_ENV_FILE")"
if ! cp "$ENV_FILE" "$APP_ENV_FILE" 2>/dev/null; then
  sudo cp "$ENV_FILE" "$APP_ENV_FILE"
  sudo chown "$(id -u):$(id -g)" "$APP_ENV_FILE" 2>/dev/null || true
fi

mkdir -p "$HOST_DATA_DIR/events/images" "$HOST_DATA_DIR/events/videos/recording-tmp" "$HOST_DATA_DIR/logs"
echo "Recording storage:"
df -h "$HOST_DATA_DIR" || true

SWAP_MB=$(awk '/SwapTotal/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "${SWAP_MB:-0}" -lt 2048 ]; then
  echo "Warning: swap is ${SWAP_MB:-0} MB. Consider 2048-4096 MB swap on external drive."
fi

if command -v vcgencmd >/dev/null 2>&1; then
  vcgencmd measure_temp || true
  vcgencmd get_throttled || true
fi

echo "Active stream config:"
grep -nE '^(RTSP_URL|HIKVISION_HOST|HIKVISION_CHANNEL|USE_CAMERA_MOTION_API)=' "$APP_ENV_FILE" || true

cd "$SCRIPT_DIR" || exit 1
docker compose up -d --build --force-recreate --remove-orphans watcher

# Optional cleanup (run only when explicitly requested):
# [ "${PRUNE_IMAGES:-0}" = "1" ] && docker image prune -f
