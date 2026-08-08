#!/usr/bin/env bash
# Live D435 terrain-gate viewer (PC) + ensure Jetson camera server.
#
# Usage:
#   bash tools/run_terrain_gate_live.sh
#   bash tools/run_terrain_gate_live.sh 192.168.1.100
#   bash tools/run_terrain_gate_live.sh 192.168.1.100 --rotate 90
#   START_SERVER=0 bash tools/run_terrain_gate_live.sh   # viewer only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
unset LD_LIBRARY_PATH   # drop conda OpenSSL so system ssh works

JETSON_IP="${1:-192.168.1.100}"
if [[ $# -ge 1 ]]; then
  shift
fi
ROTATE="${ROTATE:-90}"
START_SERVER="${START_SERVER:-1}"
SSH_USER="${SSH_USER:-nvidia}"
REMOTE_TOOLS="/home/nvidia/hopper_upper/hopper_controller/tools"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new
     "${SSH_USER}@${JETSON_IP}")

extra_args=("$@")
has_rotate=0
for a in "${extra_args[@]+"${extra_args[@]}"}"; do
  if [[ "$a" == "--rotate" ]]; then
    has_rotate=1
    break
  fi
done
if [[ ${#extra_args[@]} -eq 0 ]]; then
  viewer_args=(--net "${JETSON_IP}" --rotate "${ROTATE}")
elif [[ $has_rotate -eq 0 ]]; then
  viewer_args=(--net "${JETSON_IP}" --rotate "${ROTATE}" "${extra_args[@]}")
else
  viewer_args=(--net "${JETSON_IP}" "${extra_args[@]}")
fi

port_open() {
  python3 - <<PY
import socket, sys
s = socket.socket()
s.settimeout(2.0)
try:
    s.connect(("${JETSON_IP}", 5556))
except Exception:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

if [[ "${START_SERVER}" == "1" ]]; then
  if port_open; then
    echo "[ok] Jetson D435 server already listening on ${JETSON_IP}:5556"
  else
    echo "[..] starting d435_net_server.py on ${JETSON_IP}"
    "${SSH[@]}" "bash -lc '
      set -e
      cd ${REMOTE_TOOLS}
      if ! python3 - <<EOF
import pyrealsense2 as rs
import sys
sys.exit(0 if len(list(rs.context().query_devices())) else 1)
EOF
      then
        echo \"[err] no RealSense on Jetson (check USB)\"
        exit 2
      fi
      pkill -f \"[p]ython3.*d435_net_server.py\" 2>/dev/null || true
      sleep 0.3
      nohup python3 -u d435_net_server.py >/tmp/d435_net_server.log 2>&1 </dev/null &
      for i in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        if ss -ltn | grep -q \":5556\"; then
          tail -5 /tmp/d435_net_server.log || true
          exit 0
        fi
      done
      echo \"[err] server failed to bind :5556\"
      tail -30 /tmp/d435_net_server.log || true
      exit 3
    '"
    if ! port_open; then
      echo "[err] PC still cannot connect to ${JETSON_IP}:5556"
      exit 1
    fi
    echo "[ok] D435 server up on ${JETSON_IP}:5556"
  fi
fi

cd "${HC_DIR}"
echo "[..] python3 tools/terrain_gate_live.py ${viewer_args[*]}"
exec python3 tools/terrain_gate_live.py "${viewer_args[@]}"
