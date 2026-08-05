#!/bin/bash
# Fetch CSV logs from the Jetson upper layer and plot the newest one on the PC.
# plot_torque_fz.py saves a PNG under logs/figs/ and opens it in the system
# image viewer (xdg-open) automatically.
#
# Usage (on PC):
#   bash scripts/fetch_plot.sh                      # newest log
#   bash scripts/fetch_plot.sh 192.168.1.100 <log>  # specific log file
set -e
unset LD_LIBRARY_PATH   # drop conda OpenSSL so system ssh works

JETSON_IP="${1:-192.168.1.100}"
HC_DIR="/home/abc/Hopper/robot_runtime/upper_controller_pc/hopper_controller"

rsync -a "nvidia@${JETSON_IP}:hopper_upper/hopper_controller/logs/" "${HC_DIR}/logs/"
echo "logs synced -> ${HC_DIR}/logs"

cd "${HC_DIR}"
if [ -n "${2:-}" ]; then
  python3 tools/plot_torque_fz.py "$2"
else
  python3 tools/plot_torque_fz.py
fi
