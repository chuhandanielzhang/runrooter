#!/bin/bash
# Param sweep: find a surviving 3D config for the runtime ModeE on the 3RSR sim plant.
set -u
cd "$(dirname "$0")"

run_one () {
  local mode=$1 hoph=$2 l0=$3 tag=$4
  pkill -9 -f modee_fake_robot 2>/dev/null
  pkill -9 -f run_cao_on_our_model 2>/dev/null
  sleep 0.5
  FAKE_TAU=25 python3 -u modee_fake_robot.py --duration-s ${DUR:-15} > /tmp/sw_mj_$tag.log 2>&1 &
  local MJ=$!
  sleep 2
  CAO_L0=$l0 CAO_HOP_H=$hoph CAO_MODE=$mode CAO_TAU=25 \
    timeout 120 python3 -u run_cao_on_our_model.py > /tmp/sw_ct_$tag.log 2>&1 &
  local CT=$!
  wait $MJ
  kill $CT 2>/dev/null
  sleep 0.3
  local R
  R=$(grep RESULT /tmp/sw_mj_$tag.log | head -1)
  echo "mode=$mode hop_h=$hoph l0=$l0 -> $R"
}

run_one 2 0.10 0.42 a
run_one 2 0.20 0.42 b
run_one 3 0.10 0.42 c
run_one 3 0.20 0.42 d
run_one 3 0.15 0.45 e
