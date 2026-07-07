#!/bin/bash
# Sweep attitude gains / prop authority for 3D survival on the sim plant.
set -u
cd "$(dirname "$0")"

run_one () {
  local tag=$1; shift
  pkill -9 -f modee_fake_robot 2>/dev/null
  pkill -9 -f run_cao_on_our_model 2>/dev/null
  sleep 0.5
  FAKE_TAU=25 python3 -u modee_fake_robot.py --duration-s ${DUR:-18} > /tmp/sa_mj_$tag.log 2>&1 &
  local MJ=$!
  sleep 2
  env "$@" CAO_L0=0.42 CAO_HOP_H=0.15 CAO_MODE=3 CAO_TAU=25 \
    timeout 150 python3 -u run_cao_on_our_model.py > /tmp/sa_ct_$tag.log 2>&1 &
  local CT=$!
  wait $MJ
  kill $CT 2>/dev/null
  sleep 0.3
  echo "$tag [$*] -> $(grep RESULT /tmp/sa_mj_$tag.log | head -1)"
}

run_one base
run_one kr40 CAO_ST_KR=40 CAO_ST_KW=12
run_one fl_hi CAO_FL_KR=150 CAO_FL_KW=25
run_one prop20 CAO_PROP_BASE=0.20
run_one combo CAO_ST_KR=40 CAO_ST_KW=12 CAO_FL_KR=150 CAO_FL_KW=25 CAO_PROP_BASE=0.20
