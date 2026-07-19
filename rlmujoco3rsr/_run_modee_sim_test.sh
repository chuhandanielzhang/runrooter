#!/bin/bash
# Closed-loop test: modee_fake_robot (3RSR plant, current conventions) + runtime ModeE controller.
set -u
cd "$(dirname "$0")"
pkill -9 -f modee_fake_robot 2>/dev/null
pkill -9 -f run_cao_on_our_model 2>/dev/null
sleep 0.5

CAO_L0=${CAO_L0:-0.42}
CAO_HOP_H=${CAO_HOP_H:-0.20}

FAKE_TAU=${FAKE_TAU:-25} MFR_PERM_BRANCH=${MFR_PERM_BRANCH:-0} python3 -u modee_fake_robot.py --duration-s ${DUR:-15} --drop-start-s 1.5 \
  --record-gif "${GIF:-/tmp/modee_on_sim.gif}" > /tmp/mfr.log 2>&1 &
MJ=$!
sleep 2
CAO_L0=$CAO_L0 CAO_HOP_H=$CAO_HOP_H CAO_MASS=${CAO_MASS:-2.73} CAO_TAU=${CAO_TAU:-25} \
  CAO_ST_KR=${CAO_ST_KR:-} CAO_ST_KW=${CAO_ST_KW:-} CAO_FL_KR=${CAO_FL_KR:-} CAO_FL_KW=${CAO_FL_KW:-} \
  CAO_PROP_BASE=${CAO_PROP_BASE:-} CAO_SW_KP=${CAO_SW_KP:-} CAO_SW_KD=${CAO_SW_KD:-} \
  timeout 120 python3 -u run_cao_on_our_model.py > /tmp/mfr_ctl.log 2>&1 &
CT=$!
wait $MJ
sleep 0.5
kill $CT 2>/dev/null
sleep 0.5
echo "=== plant ==="
tail -6 /tmp/mfr.log
echo "=== ctl head ==="
head -2 /tmp/mfr_ctl.log
echo "=== ctl tail ==="
tail -3 /tmp/mfr_ctl.log
