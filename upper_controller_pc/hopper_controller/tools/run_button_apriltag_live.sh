#!/usr/bin/env bash
# Live AprilTag button viewer + setpoint publisher for MANIPULATION.
#
# Geometry: button = tag + right 16.5 cm, down 2.6 cm, protrude 5 cm; press 1 cm.
#
# Usage:
#   bash tools/run_button_apriltag_live.sh
#   bash tools/run_button_apriltag_live.sh 192.168.1.100
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
JETSON_IP="${1:-192.168.1.100}"
cd "${HC_DIR}"
# Reuse camera bring-up from terrain gate script if present.
if [[ -x "${SCRIPT_DIR}/run_terrain_gate_live.sh" ]]; then
  START_SERVER=1 bash "${SCRIPT_DIR}/run_terrain_gate_live.sh" "${JETSON_IP}" --help >/dev/null 2>&1 || true
fi
# Ensure server (same helper pattern as terrain gate).
unset LD_LIBRARY_PATH
if ! python3 - <<PY
import socket,sys
s=socket.socket(); s.settimeout(2)
try:
 s.connect(("${JETSON_IP}",5556)); sys.exit(0)
except Exception:
 sys.exit(1)
finally:
 s.close()
PY
then
  echo "[..] starting D435 server on ${JETSON_IP}"
  ssh -o BatchMode=yes -o ConnectTimeout=6 "nvidia@${JETSON_IP}" \
    'cd /home/nvidia/hopper_upper/hopper_controller/tools && nohup python3 -u d435_net_server.py >/tmp/d435_net_server.log 2>&1 </dev/null &'
  sleep 4
fi
exec python3 tools/button_apriltag_live.py \
  --net "${JETSON_IP}" --tag-id 1 --tag-size 0.09 \
  --right-m 0.165 --down-m 0.026 --protrude-m 0.05 --press-m 0.01 \
  --rotate 90
