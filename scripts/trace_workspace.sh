#!/bin/bash
# Launch the work-envelope tracer ON the Jetson from this PC.
#
#   bash scripts/trace_workspace.sh              # default IP .100
#   bash scripts/trace_workspace.sh 192.168.1.100 --speed 0.06
#
# Why Jetson: hopper_data_lcmt is published by hopper-driver on the robot.
# Running the tracer on the PC needs multicast; this path talks to LCM
# locally on the Jetson (same as hopper-upper).
#
# Stops hopper-upper (only one hopper_cmd publisher), waits for gamepad X,
# traces BOTTOM then TOP. Does NOT restart hopper-upper.
set -e
unset LD_LIBRARY_PATH

HERE="$(cd "$(dirname "$0")/.." && pwd)"
JETSON_IP="${1:-192.168.1.100}"
if [[ "$JETSON_IP" == --* ]]; then
  JETSON_IP="192.168.1.100"
else
  shift || true
fi
J="nvidia@${JETSON_IP}"
SSH="ssh -o ConnectTimeout=8 ${J}"
REMOTE="/home/nvidia/hopper_upper/hopper_controller"

echo "== sync tracer -> ${J}"
rsync -a \
  "${HERE}/upper_controller_pc/hopper_controller/tools/trace_workspace.py" \
  "${J}:${REMOTE}/tools/trace_workspace.py"

echo "== hopper-driver must be up"
if ! $SSH 'systemctl is-active --quiet hopper-driver.service'; then
  echo "XX hopper-driver is not running on ${JETSON_IP}"
  echo "   ssh ${J} sudo systemctl start hopper-driver"
  exit 1
fi

echo "== stop hopper-upper on ${JETSON_IP} (stays stopped)"
bash "${HERE}/scripts/upper.sh" stop "${JETSON_IP}"

echo "== run tracer ON Jetson. Press gamepad X when ready. Ctrl-C = quit."
set +e
# -t: Ctrl-C from this terminal reaches the remote python.
ssh -t -o ConnectTimeout=8 "${J}" \
  "cd ${REMOTE} && python3 -u tools/trace_workspace.py --log-dir logs $*"
RC=$?
set -e

echo "== pull logs to PC"
mkdir -p "${HERE}/upper_controller_pc/hopper_controller/logs_local"
rsync -a --include='workspace_trace_*' --exclude='*' \
  "${J}:${REMOTE}/logs/" \
  "${HERE}/upper_controller_pc/hopper_controller/logs_local/" || true

echo "hopper-upper left STOPPED. When finished:  bash scripts/upper.sh start"
echo "logs: ${HERE}/upper_controller_pc/hopper_controller/logs_local/workspace_trace_*.csv"
exit "$RC"
