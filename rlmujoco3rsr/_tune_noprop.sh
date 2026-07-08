#!/bin/bash
# Staged leg-only (no propeller) tuning: 1D -> 2D -> 3D.
# Usage: STAGE=1d|2d|3d ./_tune_noprop.sh  (params via env, one line result per run)
set -u
cd "$(dirname "$0")"

run_one () {
  local stage=$1 tag=$2; shift 2
  pkill -9 -f modee_fake_robot 2>/dev/null
  pkill -9 -f run_cao_on_our_model 2>/dev/null
  sleep 0.5
  local plant_flags=""
  local ctl_extra=""
  case "$stage" in
    1d) plant_flags="--strict-1d"; ctl_extra="CAO_1D=1" ;;
    2d) plant_flags="--strict-2d" ;;
    3d) plant_flags="" ;;
  esac
  FAKE_TAU=${FAKE_TAU:-20} python3 -u modee_fake_robot.py --duration-s ${DUR:-15} $plant_flags \
    ${GIF:+--record-gif $GIF} > /tmp/tn_mj_$tag.log 2>&1 &
  local MJ=$!
  sleep 2
  env CAO_MODE=1 CAO_TAU=20 $ctl_extra "$@" \
    timeout 150 python3 -u run_cao_on_our_model.py > /tmp/tn_ct_$tag.log 2>&1 &
  local CT=$!
  wait $MJ
  kill $CT 2>/dev/null
  sleep 0.3
  local res osc
  res=$(grep RESULT /tmp/tn_mj_$tag.log | head -1 | sed 's/RESULT: //')
  osc=$(grep -o "late-phase.*" /tmp/tn_mj_$tag.log | head -1)
  echo "[$stage/$tag] $* -> $res | $osc"
}

case "${STAGE:-1d}" in
1d)
  run_one 1d a CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4
  run_one 1d b CAO_L0=0.455 CAO_HOP_H=0.15 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4
  run_one 1d c CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1600 CAO_KDZ=30 CAO_SW_KP=150 CAO_SW_KD=4
  run_one 1d d CAO_L0=0.42  CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4
  ;;
2d)
  # stance attitude PD sweep (leg-only pitch control), best 1D params assumed via env B1D_*
  for KR in 23 45 70 100; do
    for KW in 7 14 25; do
      run_one 2d kr${KR}kw${KW} CAO_L0=${B_L0:-0.455} CAO_HOP_H=${B_H:-0.10} \
        CAO_KZ=${B_KZ:-1100} CAO_KDZ=${B_KDZ:-20} CAO_SW_KP=150 CAO_SW_KD=4 \
        CAO_ST_KR=$KR CAO_ST_KW=$KW
    done
  done
  ;;
2dm)
  # mechanism-level levers (gain-insensitive divergence -> change the physics of the cycle)
  run_one 2dm lowhop  CAO_L0=0.455 CAO_HOP_H=0.06 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=14
  run_one 2dm soft    CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=700  CAO_KDZ=15 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=14
  run_one 2dm bigkw   CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=40
  run_one 2dm shortl0 CAO_L0=0.42  CAO_HOP_H=0.08 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=14
  run_one 2dm swkd    CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=8 CAO_ST_KR=45 CAO_ST_KW=14
  ;;
2dx)
  # authority probes: (a) unlock torque budget, (b) pure Raibert (no stance attitude torque)
  FAKE_TAU=35 run_one 2dx tau35   CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=14 CAO_TAU=35
  run_one 2dx raibert CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=0.001 CAO_ST_KW=0.001
  FAKE_TAU=35 run_one 2dx both    CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=0.001 CAO_ST_KW=0.001 CAO_TAU=35
  ;;
3d)
  run_one 3d final CAO_L0=${B_L0:-0.455} CAO_HOP_H=${B_H:-0.10} \
    CAO_KZ=${B_KZ:-1100} CAO_KDZ=${B_KDZ:-20} CAO_SW_KP=${B_SWKP:-150} CAO_SW_KD=${B_SWKD:-4} \
    CAO_ST_KR=${B_KR:-45} CAO_ST_KW=${B_KW:-14}
  ;;
esac
