#!/bin/bash
# Strict-1D vertical hopping: modee_fake_robot + runtime ModeE (CAO_1D=1).
set -u
cd "$(dirname "$0")"
pkill -9 -f modee_fake_robot 2>/dev/null
pkill -9 -f run_cao_on_our_model 2>/dev/null
sleep 0.5

FAKE_TAU=${FAKE_TAU:-25} python3 -u modee_fake_robot.py --duration-s ${DUR:-20} --strict-1d \
  --record-gif "${GIF:-/tmp/modee_1d.gif}" > /tmp/mfr1d.log 2>&1 &
MJ=$!
sleep 2
CAO_L0=${CAO_L0:-0.42} CAO_HOP_H=${CAO_HOP_H:-0.20} CAO_TAU=${CAO_TAU:-25} CAO_1D=1 \
  timeout 150 python3 -u run_cao_on_our_model.py > /tmp/mfr1d_ctl.log 2>&1 &
CT=$!
wait $MJ
sleep 0.5
kill $CT 2>/dev/null
sleep 0.5
echo "=== plant ==="
tail -5 /tmp/mfr1d.log
echo "=== ctl counts (stance/flight) ==="
grep -c STANCE /tmp/mfr1d_ctl.log
grep -c FLIGHT /tmp/mfr1d_ctl.log
echo "=== ctl tail ==="
tail -3 /tmp/mfr1d_ctl.log
