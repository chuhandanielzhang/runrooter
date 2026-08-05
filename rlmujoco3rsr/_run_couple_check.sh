#!/bin/bash
# Closed-loop regression for the PogoX prop/leg height split (prop_hop_*):
#   TAG=5cm  HOP=0.05  -> rho_hop = 0, must reproduce the pre-change limit cycle
#   TAG=10cm HOP=0.10  -> rho_hop = 0.5, props carry the extra height share
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-5cm}"
HOP="${HOP:-0.05}"
DUR="${DUR:-25}"
BUS="udpm://239.255.76.67:7669?ttl=0"
OUT="$(pwd)/demos/pogox_couple_${TAG}.gif"

pkill -9 -f modee_fake_robot.py 2>/dev/null || true
pkill -9 -f "hopper_controller/run_modee.py" 2>/dev/null || true
sleep 0.5

LCM_DEFAULT_URL="$BUS" FAKE_TAU=40 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
python3 -u modee_fake_robot.py \
  --duration-s "$DUR" \
  --total-mass 5.61 \
  --floor-mu 0.5 \
  --drop-z 0.68 \
  --record-gif "$OUT" \
  > "/tmp/couple_${TAG}_mj.log" 2>&1 &
MJ_PID=$!

sleep 2

python3 -u ../upper_controller_pc/hopper_controller/run_modee.py \
  --lcm-url "$BUS" \
  --hop-height "$HOP" \
  ${EXTRA_CTL:-} \
  > "/tmp/couple_${TAG}_ctl.log" 2>&1 &
CTL_PID=$!

set +e
wait "$MJ_PID"
MJ_RC=$?
kill "$CTL_PID" 2>/dev/null
wait "$CTL_PID" 2>/dev/null
set -e

echo "=== plant tail (rc=$MJ_RC, hop=$HOP) ==="
tail -12 "/tmp/couple_${TAG}_mj.log"
echo "GIF: $OUT"
exit "$MJ_RC"
