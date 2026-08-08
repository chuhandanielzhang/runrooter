#!/bin/bash
# Run the CURRENT production core.py / run_modee.py defaults against the
# closed-chain 3-RSR MuJoCo plant and record a GIF.
#
# No ModeEConfig gains/limits are overridden. Simulation-only adaptations:
#   - isolated LCM bus (port 7669, ttl=0; never reaches the Jetson)
#   - MuJoCo mass/inertia uniformly scaled to current core mass (5.61 kg)
#   - floor friction set to current controller mu (0.5)
#   - pre-release hold/drop to bootstrap the hopping limit cycle
set -euo pipefail
cd "$(dirname "$0")"

BUS="udpm://239.255.76.67:7669?ttl=0"
OUT="${GIF:-$(pwd)/demos/modee_current_core_defaults.gif}"
DURATION="${DUR:-25}"

pkill -9 -f modee_fake_robot.py 2>/dev/null || true
pkill -9 -f "upper_controller_pc/hopper_controller/run_modee.py" 2>/dev/null || true
sleep 0.5

mkdir -p "$(dirname "$OUT")"

# Prefer the normal NVIDIA EGL path. The explicit Mesa driver path + system
# libstdc++ also provides a software fallback when the running NVIDIA kernel
# module and userspace driver temporarily mismatch (e.g. before a reboot).
LCM_DEFAULT_URL="$BUS" FAKE_TAU=40 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
python3 -u modee_fake_robot.py \
  --duration-s "$DURATION" \
  --total-mass 5.61 \
  --floor-mu 0.5 \
  --drop-z 0.68 \
  --record-gif "$OUT" \
  > /tmp/current_core_mj.log 2>&1 &
MJ_PID=$!

sleep 2

# Only route LCM to the isolated simulation bus. Every controller parameter is
# read directly from the latest ModeEConfig / ModeELCMConfig defaults.
python3 -u ../upper_controller_pc/hopper_controller/run_modee.py \
  --lcm-url "$BUS" \
  > /tmp/current_core_ctl.log 2>&1 &
CTL_PID=$!

set +e
wait "$MJ_PID"
MJ_RC=$?
kill "$CTL_PID" 2>/dev/null
wait "$CTL_PID" 2>/dev/null
set -e

echo "=== CURRENT core.py defaults ==="
python3 - <<'PY'
import sys
sys.path.insert(0, "../upper_controller_pc/hopper_controller")
from modee.core import ModeEConfig
c = ModeEConfig()
for key in (
    "mass_kg", "leg_l0_m", "hop_height_m", "stance_kp_z", "stance_kd_z",
    "stance_kpp", "stance_kpd", "flight_kR", "flight_kW",
    "swing_kp_xy", "swing_kd_xy", "prop_base_thrust_ratio",
    "thrust_total_ratio_max", "thrust_max_each_n", "pwm_max_us",
    "prop_k_thrust", "mu", "tau_cmd_max_nm",
):
    print(f"{key}={getattr(c, key)}")
PY

echo "=== plant result ==="
tail -10 /tmp/current_core_mj.log
echo "=== controller head/tail ==="
head -3 /tmp/current_core_ctl.log
tail -3 /tmp/current_core_ctl.log
echo "GIF: $OUT"
exit "$MJ_RC"
