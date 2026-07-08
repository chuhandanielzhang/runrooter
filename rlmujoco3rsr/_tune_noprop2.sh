#!/bin/bash
# Pure-leg 2D/3D with the NEW core.py mechanisms (pure_leg_mode + no-prop LO
# omega gate + mode-3 HLIP S2S placement).
set -u
cd "$(dirname "$0")"

run_one () {
  local stage=$1 tag=$2; shift 2
  pkill -9 -f modee_fake_robot 2>/dev/null
  pkill -9 -f run_cao_on_our_model 2>/dev/null
  sleep 0.5
  local plant_flags=""
  case "$stage" in
    2d) plant_flags="--strict-2d" ;;
    3d) plant_flags="" ;;
  esac
  FAKE_TAU=${FAKE_TAU:-20} python3 -u modee_fake_robot.py --duration-s ${DUR:-15} $plant_flags \
    --drop-z ${DROPZ:-0.62} ${GIF:+--record-gif $GIF} > /tmp/tn2_mj_$tag.log 2>&1 &
  local MJ=$!
  sleep 2
  env CAO_TAU=20 CAO_PURE=1 "$@" \
    timeout 150 python3 -u run_cao_on_our_model.py > /tmp/tn2_ct_$tag.log 2>&1 &
  local CT=$!
  wait $MJ
  kill $CT 2>/dev/null
  sleep 0.3
  echo "[$stage/$tag] $* -> $(grep -o 'late-phase.*' /tmp/tn2_mj_$tag.log | head -1)"
}

case "${STAGE:-2d}" in
2d)
  # mode1 = pure-leg QP;  mode3 = HLIP S2S placement (new core)
  run_one 2d m1      CAO_MODE=1 CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4
  run_one 2d m3      CAO_MODE=3 CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4
  run_one 2d m3kr45  CAO_MODE=3 CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=14
  run_one 2d m1kr45  CAO_MODE=1 CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=14
  ;;
3d)
  run_one 3d final   CAO_MODE=${B_MODE:-3} CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 \
    CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=${B_KR:-45} CAO_ST_KW=${B_KW:-14}
  ;;
trim)
  # v_des trim sweep (operator-style trim against the systematic per-stance pitch impulse)
  for SX in -0.30 -0.15 +0.15 +0.30; do
    export MFR_STICK_X=$SX MFR_STICK_Y=0 MFR_STICK_AT_S=0
    run_one 2d trim$SX CAO_MODE=1 CAO_L0=0.455 CAO_HOP_H=0.10 CAO_KZ=1100 CAO_KDZ=20 \
      CAO_SW_KP=150 CAO_SW_KD=4 CAO_ST_KR=45 CAO_ST_KW=14
    unset MFR_STICK_X MFR_STICK_Y MFR_STICK_AT_S
  done
  ;;
esac
