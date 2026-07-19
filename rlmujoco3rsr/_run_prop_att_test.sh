#!/bin/bash
# Prop-only attitude sign test: hips held at home by position actuators, base pinned
# in the air with attitude FREE, initial roll +10 deg. If the prop attitude loop has
# the correct sign, roll converges to ~0; if flipped, it diverges.
set -u
cd "$(dirname "$0")"
pkill -9 -f modee_fake_robot 2>/dev/null
pkill -9 -f run_cao_on_our_model 2>/dev/null
sleep 0.5

FAKE_TAU=25 python3 -u modee_fake_robot.py --duration-s ${DUR:-12} \
  --hold-hips --hold-att-free --print-roll \
  --init-roll-deg ${ROLL0:-10} --drop-start-s 999 > /tmp/att.log 2>&1 &
MJ=$!
sleep 2
CAO_L0=${L0:-0.42} CAO_HOP_H=0.15 CAO_TAU=25 \
  timeout 100 python3 -u run_cao_on_our_model.py > /tmp/att_ctl.log 2>&1 &
CT=$!
wait $MJ
sleep 0.5
kill $CT 2>/dev/null
echo "=== roll trace (plant) ==="
grep "roll=" /tmp/att.log | awk 'NR % 4 == 1'
