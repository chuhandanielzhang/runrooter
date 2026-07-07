#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="$(pwd)/videos"
mkdir -p "$OUT_DIR"

OUT_MP4="$OUT_DIR/case_modee_serial_fwd_3s_5s_3s.mp4"

echo "=== CASE / Hopper-aero: record ModeE (serial) demo (3s in-place, 5s forward, 3s in-place) ==="
echo "Output: $OUT_MP4"

# Clean up old processes (best-effort)
pkill -f mujoco_lcm_fake_robot.py 2>/dev/null || true
pkill -f run_modee.py 2>/dev/null || true
sleep 1

# Start MuJoCo fake robot (serial plant)
# Timeline (SIM time):
# - hold-level-s 3.0: hold base level for 3s (controller warm-start)
# - after release:
#   - cmd_vx0 for 3s (in-place hopping)
#   - cmd_vx1 for 5s (forward at 0.3 m/s)
#   - cmd_vx2 for 3s (back to in-place hopping)
python3 mujoco_lcm_fake_robot.py \
  --arm \
  --realtime \
  --model "../../HW/Hopper-aero/model/hopper_serial.xml" \
  --q-sign 1 \
  --q-offset 0 \
  --hold-level-s 3.0 \
  --fake-gamepad \
  --fake-gamepad-y-hold-s 2.0 \
  --cmd-vx0 0.0 \
  --cmd-vy0 0.0 \
  --cmd-vx1 0.30 \
  --cmd-vy1 0.0 \
  --cmd-switch-after-s 3.0 \
  --cmd-vx2 0.0 \
  --cmd-vy2 0.0 \
  --cmd-switch2-after-s 8.0 \
  --duration-s 14 \
  --record-mp4 "$OUT_MP4" \
  --hud \
  > /tmp/case_hopper_sim_modee_fwd_mj.log 2>&1 &
MJ_PID=$!

sleep 1

# Start ModeE controller (serial leg model)
python3 run_modee.py \
  --leg-model serial \
  --tau-out-max 2500 \
  --pwm-max 1400 \
  --use-hopper4-pwm \
  --thrust-ratio 0.03 \
  --thrust-max-each 20 \
  --thrust-min-each 0.30 \
  --stance-use-props \
  --hop-peak-z 0.70 \
  --print-hz 0 \
  > /tmp/case_hopper_sim_modee_fwd_ctl.log 2>&1 &
CTL_PID=$!

echo "Running... (MuJoCo PID=$MJ_PID, controller PID=$CTL_PID)"
wait "$MJ_PID" || true

echo "Stopping controller..."
kill "$CTL_PID" 2>/dev/null || true
wait "$CTL_PID" 2>/dev/null || true

if [ -f "$OUT_MP4" ]; then
  echo "✅ Done: $OUT_MP4"
  ls -lh "$OUT_MP4"
else
  echo "❌ Video not found: $OUT_MP4"
  echo "--- tail mujoco log ---"
  tail -80 /tmp/case_hopper_sim_modee_fwd_mj.log 2>/dev/null || true
  echo "--- tail controller log ---"
  tail -80 /tmp/case_hopper_sim_modee_fwd_ctl.log 2>/dev/null || true
  exit 1
fi

