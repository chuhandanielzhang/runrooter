#!/bin/bash
# Sync ONLY the upper-layer code (PC -> Jetson), nothing else.
# If hopper-upper is running on the Jetson it is restarted so the new code
# takes effect; if it is stopped it stays stopped.
#
# Usage (on PC):
#   bash scripts/sync_upper.sh              # default IP .100
#   bash scripts/sync_upper.sh 192.168.1.123
set -e
unset LD_LIBRARY_PATH   # drop conda OpenSSL so system ssh works

JETSON_IP="${1:-192.168.1.100}"
UPPER_SRC="/home/abc/Hopper/robot_runtime/upper_controller_pc/"

rsync -a --delete \
  --exclude 'hopper_controller/logs' \
  --exclude '__pycache__' \
  "${UPPER_SRC}" "nvidia@${JETSON_IP}:/home/nvidia/hopper_upper/"
echo "upper synced -> nvidia@${JETSON_IP}:/home/nvidia/hopper_upper"

if ssh "nvidia@${JETSON_IP}" 'systemctl is-active --quiet hopper-upper.service'; then
  ssh "nvidia@${JETSON_IP}" 'sudo systemctl restart hopper-upper.service'
  echo "hopper-upper was RUNNING -> restarted with the new code"
else
  echo "hopper-upper not running (start: ssh nvidia@${JETSON_IP} sudo systemctl start hopper-upper)"
fi

# Jetson-local perception (D435 + AprilTag; services/hopper-perception.service).
if ssh "nvidia@${JETSON_IP}" 'systemctl is-active --quiet hopper-perception.service'; then
  ssh "nvidia@${JETSON_IP}" 'sudo systemctl restart hopper-perception.service'
  echo "hopper-perception was RUNNING -> restarted with the new code"
else
  echo "hopper-perception not running (start: ssh nvidia@${JETSON_IP} sudo systemctl start hopper-perception)"
fi
