#!/bin/bash
# PC: push upper-layer code to Jetson and restart services if they are running.
#
#   bash scripts/update_upper.sh              # default 192.168.1.100
#   bash scripts/update_upper.sh 192.168.1.123
#
# PC source : <repo>/upper_controller_pc/
# Jetson    : /home/nvidia/hopper_upper/
# Excludes  : hopper_controller/logs, __pycache__
set -e
unset LD_LIBRARY_PATH

HERE="$(cd "$(dirname "$0")/.." && pwd)"
JETSON_IP="${1:-192.168.1.100}"
UPPER_SRC="${HERE}/upper_controller_pc/"
J="nvidia@${JETSON_IP}"
SSH="ssh -o ConnectTimeout=8 ${J}"

if [ ! -d "${UPPER_SRC}" ]; then
  echo "missing upper source: ${UPPER_SRC}" >&2
  exit 1
fi

rsync -a --delete \
  --exclude 'hopper_controller/logs' \
  --exclude '__pycache__' \
  "${UPPER_SRC}" "${J}:/home/nvidia/hopper_upper/"
echo "upper synced -> ${J}:/home/nvidia/hopper_upper"

if $SSH 'systemctl is-active --quiet hopper-upper.service'; then
  $SSH 'sudo systemctl restart hopper-upper.service'
  echo "hopper-upper was RUNNING -> restarted"
else
  echo "hopper-upper not running (start: ssh ${J} sudo systemctl start hopper-upper)"
fi

if $SSH 'systemctl is-active --quiet hopper-perception.service'; then
  $SSH 'sudo systemctl restart hopper-perception.service'
  echo "hopper-perception was RUNNING -> restarted"
else
  echo "hopper-perception not running (start: ssh ${J} sudo systemctl start hopper-perception)"
fi
