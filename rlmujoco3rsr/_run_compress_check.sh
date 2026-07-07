#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
KZ="${1:-1100}"        # stance spring stiffness
GIF="${2:-hfa_compress_check.gif}"
pkill -f "run_cao_on_our" 2>/dev/null
pkill -f "cao_fake_robot" 2>/dev/null
sleep 2

echo "=== Cao stance_kp_z = $KZ N/m  ->  $GIF ==="
# 1) start Cao HFA QP controller (high level) in background, fully detached
CAO_L0=0.42 CAO_HOP_H=0.20 CAO_MODE=3 CAO_TAU=25 CAO_KZ=$KZ \
  setsid python -u run_cao_on_our_model.py > run_cao.log 2>&1 < /dev/null &
CTRL_PID=$!
sleep 5
echo "controller pid=$CTRL_PID alive=$(ps -p $CTRL_PID -o pid= 2>/dev/null | wc -l)"

# 2) run our model as fake robot (plant) in FOREGROUND, record gif + log compression
FAKE_TAU=25 python -u cao_fake_robot.py --duration-s 18 --drop-start-s 1.5 \
  --record-gif "$GIF" 2>&1 | grep -v -E "warp|Warn|Failed|EGL|egl|NoneType"

# 3) cleanup
kill $CTRL_PID 2>/dev/null
pkill -f "run_cao_on_our" 2>/dev/null
echo "done"
