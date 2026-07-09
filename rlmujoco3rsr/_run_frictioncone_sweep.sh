#!/bin/bash
# Friction-cone modulation A/B (2026-07-07): low-mu floor x stance prop downforce.
# Physics under test: stance collective REVERSE thrust F_dn presses the body down,
# the leg adds +F_dn to fz -> CoM dynamics unchanged, contact normal force +F_dn,
# friction cone |fxy| <= mu*(fz+F_dn) widens (most at touchdown when fz ~ 0).
# Metric: per-stance foot slip (mm) + survival (late-phase hopping oscillation).
set -u
cd "$(dirname "$0")"

# Isolated LCM bus: port 7669, ttl=0 (loopback only). MUST differ from the
# REAL robot bus (7667): ttl=0 only stops packets leaving this host -- a
# run_modee.py session on the SAME PC still receives same-port multicast via
# local loopback and forwards it to the Jetson (this actuated the real robot
# twice on 2026-07-09). Different port = hard isolation.
export LCM_DEFAULT_URL="udpm://239.255.76.67:7669?ttl=0"

MUS=${MUS:-"0.9 0.4 0.25"}
DFS=${DFS:-"0 15 30"}
DUR=${DUR:-15}
# EXTRA_FLAGS e.g. "--strict-2d" to isolate the slip physics from 3D roll issues
EXTRA_FLAGS=${EXTRA_FLAGS:-}

for MU in $MUS; do for DF in $DFS; do
  pkill -9 -f modee_fake_robot 2>/dev/null
  pkill -9 -f run_cao_on_our_model 2>/dev/null
  sleep 0.5
  TAG="mu${MU}_df${DF}"
  FAKE_TAU=${FAKE_TAU:-25} python3 -u modee_fake_robot.py --duration-s "$DUR" --drop-start-s 1.5 \
    --floor-mu "$MU" $EXTRA_FLAGS > "/tmp/fc_${TAG}.log" 2>&1 &
  MJ=$!
  sleep 2
  CAO_MODE=${CAO_MODE:-3} CAO_TAU=${CAO_TAU:-25} CAO_MU="$MU" CAO_DOWNFORCE="$DF" \
    CAO_DOWNFORCE_TD=${CAO_DOWNFORCE_TD:-} \
    timeout $((DUR + 30)) python3 -u run_cao_on_our_model.py > "/tmp/fc_${TAG}_ctl.log" 2>&1 &
  CT=$!
  wait $MJ
  kill $CT 2>/dev/null
  sleep 0.5
  echo "=== $TAG ==="
  grep -E "RESULT|late-phase|SLIP|THRUST range" "/tmp/fc_${TAG}.log" | grep -v "^\[SLIP\]" || echo "(no result -- see /tmp/fc_${TAG}.log)"
done; done
