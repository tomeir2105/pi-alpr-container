#!/bin/sh
set -eu

SWAP_FILE=${1:-/mnt/localdisk/pi-alpr/pi-alpr.swap}
SWAP_MB=${2:-4096}

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo $0 $SWAP_FILE $SWAP_MB" >&2
  exit 1
fi

mkdir -p "$(dirname "$SWAP_FILE")"
if [ ! -f "$SWAP_FILE" ]; then
  fallocate -l "${SWAP_MB}M" "$SWAP_FILE" 2>/dev/null || dd if=/dev/zero of="$SWAP_FILE" bs=1M count="$SWAP_MB"
  chmod 600 "$SWAP_FILE"
  mkswap "$SWAP_FILE"
fi

if ! grep -qs " $SWAP_FILE " /etc/fstab; then
  printf '%s none swap sw 0 0\n' "$SWAP_FILE" >> /etc/fstab
fi

swapon "$SWAP_FILE" 2>/dev/null || true
swapon --show
