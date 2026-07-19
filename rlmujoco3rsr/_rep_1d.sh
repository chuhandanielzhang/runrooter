#!/bin/bash
# Repeat the same 1D leg-only config N times to expose the intermittent "sticks
# to the floor" failure (LCM timing races make runs non-deterministic).
set -u
cd "$(dirname "$0")"
N=${N:-5}
for k in $(seq 1 $N); do
  pkill -9 -f modee_fake_robot 2>/dev/null
  pkill -9 -f run_cao_on_our_model 2>/dev/null
  sleep 0.5
  FAKE_TAU=20 python3 -u modee_fake_robot.py --duration-s ${DUR:-15} --strict-1d --drop-z ${DROPZ:-0.62} \
    > /tmp/rep1d_mj_$k.log 2>&1 &
  MJ=$!
  sleep 2
  env CAO_TAU=20 CAO_1D=1 CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 \
      CAO_SW_KP=150 CAO_SW_KD=4 \
    timeout 120 python3 -u run_cao_on_our_model.py > /tmp/rep1d_ct_$k.log 2>&1 &
  CT=$!
  wait $MJ
  kill $CT 2>/dev/null
  sleep 0.3
  echo "run$k: $(grep -o 'late-phase.*' /tmp/rep1d_mj_$k.log | head -1)"
done
