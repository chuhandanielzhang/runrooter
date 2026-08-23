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
# traces BOTTOM then MID then TOP, then starts hopper-upper again.
set -e
unset LD_LIBRARY_PATH

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="${HERE}/upper_controller_pc/hopper_controller"
JETSON_IP="${1:-192.168.1.100}"
if [[ "$JETSON_IP" == --* ]]; then
  JETSON_IP="192.168.1.100"
else
  shift || true
fi
J="nvidia@${JETSON_IP}"
SSH="ssh -o ConnectTimeout=8 ${J}"
REMOTE="/home/nvidia/hopper_upper/hopper_controller"

echo "== replay the planned path offline (joint stops, keep-outs, floor)"
if ! (cd "${LOCAL}" && python3 tools/check_workspace_path.py); then
  echo "XX the planned path does not pass its own check; not driving it"
  exit 1
fi

echo "== sync tracer -> ${J}"
rsync -a \
  "${LOCAL}/tools/trace_workspace.py" \
  "${LOCAL}/tools/workspace_envelope.py" \
  "${LOCAL}/tools/check_workspace_path.py" \
  "${J}:${REMOTE}/tools/"

echo "== hopper-driver must be up"
if ! $SSH 'systemctl is-active --quiet hopper-driver.service'; then
  echo "XX hopper-driver is not running on ${JETSON_IP}"
  echo "   ssh ${J} sudo systemctl start hopper-driver"
  exit 1
fi

echo "== stop hopper-upper on ${JETSON_IP} (restarted after the trace)"
bash "${HERE}/scripts/upper.sh" stop "${JETSON_IP}"

echo "== run tracer ON Jetson. Press gamepad X when ready. Ctrl-C = quit."
set +e
# -t: Ctrl-C from this terminal reaches the remote python.
ssh -t -o ConnectTimeout=8 "${J}" \
  "cd ${REMOTE} && python3 -u tools/trace_workspace.py --log-dir logs $*"
RC=$?
set -e

echo "== pull logs to PC"
mkdir -p "${LOCAL}/logs_local"
rsync -a --include='workspace_trace_*' --exclude='*' \
  "${J}:${REMOTE}/logs/" "${LOCAL}/logs_local/" || true

echo "== start hopper-upper on ${JETSON_IP}"
bash "${HERE}/scripts/upper.sh" start "${JETSON_IP}" || true

NEWEST="$(ls -t "${LOCAL}"/logs_local/workspace_trace_*.csv 2>/dev/null | head -1)"
if [[ -n "${NEWEST}" ]]; then
  (cd "${LOCAL}" && python3 tools/plot_workspace_trace.py "${NEWEST}") || true
fi
echo "logs: ${LOCAL}/logs_local/workspace_trace_*.csv"
exit "$RC"
