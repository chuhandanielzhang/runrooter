#!/bin/bash
# A/B: mode 1 (legacy stance PD) vs mode 2 (rate KF + TD reference shaping +
# error-scheduled damping), pure leg, isolated LCM bus (7669, ttl=0 -- NEVER
# touches the real robot on 7667).
set -u
cd "$(dirname "$0")"
export LCM_DEFAULT_URL="udpm://239.255.76.67:7669?ttl=0"

DUR=${DUR:-15}
MODES=${MODES:-"1 2"}

for MODE in $MODES; do
  pkill -9 -f modee_fake_robot 2>/dev/null
  pkill -9 -f run_cao_on_our_model 2>/dev/null
  sleep 0.5
  TAG="mode${MODE}_ab"
  FAKE_TAU=${FAKE_TAU:-25} python3 -u modee_fake_robot.py --duration-s "$DUR" --drop-start-s 1.5 \
    > "/tmp/ab_${TAG}.log" 2>&1 &
  MJ=$!
  sleep 2
  CAO_MODE="$MODE" CAO_PURE=1 CAO_TAU=${CAO_TAU:-25} \
    timeout $((DUR + 30)) python3 -u run_cao_on_our_model.py > "/tmp/ab_${TAG}_ctl.log" 2>&1 &
  CT=$!
  wait $MJ
  kill $CT 2>/dev/null
  sleep 0.5
  echo "=== $TAG ==="
  grep -E "RESULT|late-phase|SLIP total|THRUST range|Traceback|Error" "/tmp/ab_${TAG}.log" | head -20 || true
  grep -E "Traceback|Error" "/tmp/ab_${TAG}_ctl.log" | head -5 || true
done
