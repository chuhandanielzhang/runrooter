#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="$(pwd)/videos"
mkdir -p "$OUT_DIR"

OUT_MP4="$OUT_DIR/case_modee_serial_3d_free_2d_cmd_3s_3s.mp4"

# Resolve model path across different worktrees/layouts.
MODEL_PATH="../../HW/Hopper-aero/model/hopper_serial.xml"
if [ ! -f "$MODEL_PATH" ]; then
  MODEL_PATH="../../../mjcf/hopper_serial.xml"
fi

echo "=== CASE / Hopper-aero: record ModeE (serial) 3D-free + 2D-command demo ==="
echo "Task: 3s in-place -> 3s forward (+X), attitude free from start"
echo "Model: $MODEL_PATH"
echo "Output: $OUT_MP4"

# Clean up old processes (best-effort)
pkill -f mujoco_lcm_fake_robot.py 2>/dev/null || true
pkill -f run_modee.py 2>/dev/null || true
sleep 1

# Start ModeE controller:
# - keep all tuned 1D params in core.py untouched
# - force runtime 2D by setting mode_1d=False via --2d-mode
# Launch controller first, then MuJoCo immediately after (no delay) to minimize startup skew.
python3 run_modee.py \
  --leg-model serial \
  --tau-out-max 2500 \
  --tau-max 80 \
  --pwm-max 1400 \
  --thrust-ratio 0.00 \
  --thrust-max-each 0.0 \
  --thrust-min-each 0.0 \
  --cmd-dv-max 0.08 \
  --foot-vel-lpf-tau 0.03 \
  --swing-kp-xy 35 \
  --swing-kd-xy 6 \
  --swing-kp-z 350 \
  --swing-kd-z 8 \
  --flight-kR-roll 36 \
  --flight-kW-roll 20 \
  --flight-kR-pitch 36 \
  --flight-kW-pitch 20 \
  --flight-tau-rp-max 70 \
  --2d-mode \
  --energy-kp 0.8 \
  --hop-height 0.01 \
  --hop-peak-z 0.60 \
  --print-hz 0 \
  > /tmp/case_hopper_sim_modee_2d_ctl.log 2>&1 &
CTL_PID=$!

# Start MuJoCo fake robot (serial plant)
# Timeline (no rail constraints, 3D free body):
# - 0~3s : in-place hopping (vx=0)
# - 3~6s : forward hopping along +X (vx=0.01)
# - >=6s : back to in-place (vx=0)
# Release height request: base starts ~30cm above nominal foot contact
# (serial leg nominal l0≈0.565m -> init-base-z≈0.865m).
python3 mujoco_lcm_fake_robot.py \
  --arm \
  --realtime \
  --model "$MODEL_PATH" \
  --q-sign 1 \
  --q-offset 0 \
  --init-base-z 0.865 \
  --fake-gamepad \
  --fake-gamepad-y-hold-s 1.0 \
  --cmd-vx0 0.0 \
  --cmd-vy0 0.0 \
  --cmd-vx1 0.01 \
  --cmd-vy1 0.0 \
  --cmd-switch-after-s 3.0 \
  --cmd-vx2 0.0 \
  --cmd-vy2 0.0 \
  --cmd-switch2-after-s 6.0 \
  --cmd-ramp-s 5.0 \
  --duration-s 8 \
  --record-mp4 "$OUT_MP4" \
  --hud \
  > /tmp/case_hopper_sim_modee_2d_mj.log 2>&1 &
MJ_PID=$!

echo "Running 2D sim... (MuJoCo PID=$MJ_PID, controller PID=$CTL_PID)"
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
  tail -80 /tmp/case_hopper_sim_modee_2d_mj.log 2>/dev/null || true
  echo "--- tail controller log ---"
  tail -80 /tmp/case_hopper_sim_modee_2d_ctl.log 2>/dev/null || true
  exit 1
fi
