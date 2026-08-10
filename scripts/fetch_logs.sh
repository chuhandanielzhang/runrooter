#!/bin/bash
# PC: pull the LAST Jetson session (CSV logs + camera recordings) into this repo.
#
#   bash scripts/fetch_logs.sh                      # -> logs/sessions/run_<BJ time>/
#   bash scripts/fetch_logs.sh outdoor              # -> logs/sessions/outdoor_<BJ time>/
#   bash scripts/fetch_logs.sh outdoor 192.168.1.123
#   bash scripts/fetch_logs.sh --all [ip]           # pull entire Jetson logs/ tree
#
# Dest (always under this checkout):
#   <repo>/upper_controller_pc/hopper_controller/logs/...
# Jetson source:
#   /home/nvidia/hopper_upper/hopper_controller/logs/
set -e
unset LD_LIBRARY_PATH

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST_ROOT="${HERE}/upper_controller_pc/hopper_controller/logs"
RLOGS="/home/nvidia/hopper_upper/hopper_controller/logs"

if [ "${1:-}" = "--all" ]; then
  JETSON_IP="${2:-192.168.1.100}"
  J="nvidia@${JETSON_IP}"
  mkdir -p "${DEST_ROOT}"
  rsync -a "${J}:${RLOGS}/" "${DEST_ROOT}/"
  echo "all Jetson logs + cam -> ${DEST_ROOT}/"
  du -sh "${DEST_ROOT}" 2>/dev/null
  exit 0
fi

# Last session only (default).
exec bash "${HERE}/scripts/save_last.sh" "${1:-run}" "${2:-192.168.1.100}"
