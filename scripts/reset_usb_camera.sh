#!/usr/bin/env bash
#
# Reset a USB camera by toggling its `authorized` flag in sysfs.
# Useful when a camera (e.g. /dev/video0) gets stuck and stops streaming.
#
# Usage:
#   ./reset_usb_camera.sh [video_device]
#
# Examples:
#   ./reset_usb_camera.sh            # defaults to /dev/video0
#   ./reset_usb_camera.sh /dev/video2
#
# NOTE: Linux only (relies on udevadm and /sys/bus/usb). Requires sudo.

set -euo pipefail

VIDEO_DEV="${1:-/dev/video0}"

if [[ ! -e "$VIDEO_DEV" ]]; then
  echo "Error: video device '$VIDEO_DEV' does not exist." >&2
  exit 1
fi

USB_PATH=$(udevadm info -q path -n "$VIDEO_DEV" |
  grep -oE '/usb[0-9]+/[0-9]+-[0-9]+(\.[0-9]+)*' |
  tail -1 |
  xargs basename)

if [[ -z "$USB_PATH" ]]; then
  echo "Error: could not determine USB path for '$VIDEO_DEV'." >&2
  exit 1
fi

AUTHORIZED_FILE="/sys/bus/usb/devices/$USB_PATH/authorized"

if [[ ! -e "$AUTHORIZED_FILE" ]]; then
  echo "Error: '$AUTHORIZED_FILE' not found." >&2
  exit 1
fi

echo "Resetting USB device: $USB_PATH (from $VIDEO_DEV)"

echo 0 | sudo tee "$AUTHORIZED_FILE" >/dev/null
sleep 2
echo 1 | sudo tee "$AUTHORIZED_FILE" >/dev/null

echo "Done."
