#!/usr/bin/env python3
"""
MuJoCo "fake robot" process for Hopper-aero ModeE (LCM IO parity with real robot).

Goal
----
Run your *real PC-side controller* (`hopper_controller/run_modee.py`) against MuJoCo,
without changing any LCM channel names or lcmt types.

This process emulates the Pi-side `hopper_driver` IO:
  Publishes (sensor side):
    - `hopper_data_lcmt` : q, qd, tauIq
    - `hopper_imu_lcmt`  : quat(wxyz), gyro(body), acc(body), rpy(xyz)
  Subscribes (command side):
    - `hopper_cmd_lcmt`  : tau_ff (and optional joint PD fields)
    - `motor_pwm_lcmt`   : 6-channel PWM + control_mode (arm gate)

Important semantic parity with the real robot code (`Hopper-aero/src/hardware/hopper_hardware.cpp`):
  - Joint sign/offset on LCM:
      q_lcm  = q_sign * motor_pos + q_offset
      qd_lcm = q_sign * motor_vel
      tauIq  = q_sign * motor_tau
    Defaults match the real robot:
      q_sign   = -1
      q_offset = -1.047

  - Command sign:
      motor_tau = q_sign * tau_ff_lcm

  - IMU accel convention matches ModeECore:
      `hopper_imu_lcmt.acc` is published as **-(specific force)** in body frame:
        acc_b = R_wb^T * (g_w - a_w) = - R_wb^T * (a_w - g_w)
      This yields:
        - at rest: acc_b ≈ [0,0,-9.81]
        - in free-fall: acc_b ≈ [0,0,0]

Usage
-----
Terminal A (fake robot / MuJoCo):
  cd Hopper_sim/model_aero
  python3 mujoco_lcm_fake_robot.py --arm

Terminal B (your real controller):
  cd Hopper_sim/model_aero
  python3 run_modee.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Sequence

import numpy as np
import mujoco
import lcm

# --- LCM types (same pattern as hopper_controller/modee/lcm_controller.py) ---
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_LCM_TYPES_DIR = os.path.join(_CUR_DIR, "..", "hopper_lcm_types", "lcm_types")
sys.path.append(_LCM_TYPES_DIR)

from python.hopper_cmd_lcmt import hopper_cmd_lcmt  # type: ignore  # noqa: E402
from python.hopper_data_lcmt import hopper_data_lcmt  # type: ignore  # noqa: E402
from python.hopper_imu_lcmt import hopper_imu_lcmt  # type: ignore  # noqa: E402
from python.gamepad_lcmt import gamepad_lcmt  # type: ignore  # noqa: E402
from python.motor_pwm_lcmt import motor_pwm_lcmt  # type: ignore  # noqa: E402

from modee.controllers.motor_utils import MotorTableModel  # noqa: E402
from forward_kinematics import ForwardKinematics  # noqa: E402


def _quat_to_R_wb(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) -> rotation matrix R_wb (body->world)."""
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = (q / n).astype(float)
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _R_to_rpy_xyz(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (XYZ intrinsic) from R_wb."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(max(1e-12, R[2, 1] ** 2 + R[2, 2] ** 2))))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


def _quat_from_rpy_xyz_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build quaternion (wxyz) from roll/pitch/yaw in radians.
    Convention: intrinsic XYZ (roll about X, pitch about Y, yaw about Z),
    equivalent to q = qz(yaw) * qy(pitch) * qx(roll).
    """
    hr = 0.5 * float(roll)
    hp = 0.5 * float(pitch)
    hy = 0.5 * float(yaw)
    cr, sr = float(np.cos(hr)), float(np.sin(hr))
    cp, sp = float(np.cos(hp)), float(np.sin(hp))
    cy, sy = float(np.cos(hy)), float(np.sin(hy))
    # q = qz * qy * qx
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return np.array([w, x, y, z], dtype=float)


def _vee_so3(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=float).reshape(3, 3)
    return np.array([E[2, 1], E[0, 2], E[1, 0]], dtype=float)


def _parse_int_list(s: str, *, n: int) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    out = [int(p) for p in parts]
    if len(out) != int(n):
        raise ValueError(f"expected {n} ints, got {len(out)} from '{s}'")
    return out


def _parse_float_list(s: str, *, n: int) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    out = [float(p) for p in parts]
    if len(out) != int(n):
        raise ValueError(f"expected {n} floats, got {len(out)} from '{s}'")
    return out


def _parse_arm_groups(s: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """
    Parse prop PWM groups string.
    Format: "2;1;3"  -> ((2,), (1,), (3,))
            "0,4;1,5;2,3" -> ((0,4),(1,5),(2,3))
    """
    arms = [a.strip() for a in str(s).split(";")]
    if len(arms) != 3:
        raise ValueError(f"expected 3 arm groups split by ';', got {len(arms)} from '{s}'")
    groups: list[tuple[int, ...]] = []
    for a in arms:
        if a == "":
            groups.append(tuple())
            continue
        idxs = tuple(int(x.strip()) for x in a.split(",") if x.strip() != "")
        groups.append(idxs)
    return (groups[0], groups[1], groups[2])


def _fmt_vec(v: np.ndarray, *, fmt: str = "{:+.2f}") -> str:
    v = np.asarray(v, dtype=float).reshape(-1)
    return "[" + ",".join(fmt.format(float(x)) for x in v.tolist()) + "]"


def _draw_hud_rgb(frame_rgb: np.ndarray, *, lines: list[str], font_size: int = 18) -> np.ndarray:
    """
    Draw a simple HUD overlay onto an RGB uint8 frame.
    Uses PIL if available; otherwise returns the original frame.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return frame_rgb

    frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        return frame_rgb

    im = Image.fromarray(frame_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=int(max(10, int(font_size))))
    except Exception:
        font = ImageFont.load_default()

    # Layout
    x0, y0 = 12, 10
    line_h = int(max(12, int(font_size * 1.15)))
    pad = 8

    text = "\n".join([str(s) for s in lines if str(s).strip() != ""])
    if text.strip() == "":
        return frame_rgb

    # Compute bounding box and draw semi-transparent background
    try:
        bbox = draw.multiline_textbbox((x0, y0), text, font=font, spacing=2)
        bx0, by0, bx1, by1 = [int(v) for v in bbox]
    except Exception:
        bx0, by0, bx1, by1 = x0, y0, x0 + 420, y0 + (line_h * max(1, len(lines)))
    draw.rectangle((bx0 - pad, by0 - pad, bx1 + pad, by1 + pad), fill=(0, 0, 0, 150))

    # Text with light outline
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.multiline_text((x0 + dx, y0 + dy), text, font=font, fill=(0, 0, 0, 220), spacing=2)
    draw.multiline_text((x0, y0), text, font=font, fill=(255, 255, 255, 255), spacing=2)

    out = Image.alpha_composite(im, overlay).convert("RGB")
    return np.asarray(out, dtype=np.uint8)


def _add_frame_axes_to_scene(
    scene: mujoco.MjvScene,
    *,
    origin_w: np.ndarray,
    R_wf: np.ndarray,
    axis_len_m: float = 0.30,
    width_px: float = 6.0,
    rgba_x: np.ndarray | None = None,
    rgba_y: np.ndarray | None = None,
    rgba_z: np.ndarray | None = None,
) -> None:
    """Add a 3-axis coordinate frame (x,y,z) to the current MuJoCo scene as colored line geoms (WORLD coords)."""
    try:
        origin = np.asarray(origin_w, dtype=float).reshape(3)
        R = np.asarray(R_wf, dtype=float).reshape(3, 3)
        L = float(max(0.0, float(axis_len_m)))
        if L <= 1e-9:
            return
        if rgba_x is None:
            rgba_x = np.array([1.0, 0.1, 0.1, 0.95], dtype=np.float32)
        if rgba_y is None:
            rgba_y = np.array([0.1, 1.0, 0.1, 0.95], dtype=np.float32)
        if rgba_z is None:
            rgba_z = np.array([0.2, 0.4, 1.0, 0.95], dtype=np.float32)

        def _add_line(p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray) -> None:
            if int(scene.ngeom) >= int(scene.maxgeom):
                return
            geom = scene.geoms[int(scene.ngeom)]
            size = np.zeros(3, dtype=np.float64)
            pos = np.zeros(3, dtype=np.float64)
            mat = np.zeros(9, dtype=np.float64)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, pos, mat, np.asarray(rgba, dtype=np.float32).reshape(4))
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                float(width_px),
                np.asarray(p0, dtype=np.float64).reshape(3),
                np.asarray(p1, dtype=np.float64).reshape(3),
            )
            scene.ngeom = int(scene.ngeom) + 1

        ex = origin + L * R[:, 0]
        ey = origin + L * R[:, 1]
        ez = origin + L * R[:, 2]
        _add_line(origin, ex, rgba_x)
        _add_line(origin, ey, rgba_y)
        _add_line(origin, ez, rgba_z)
    except Exception:
        return


def _add_world_line_to_scene(
    scene: mujoco.MjvScene,
    *,
    p0_w: np.ndarray,
    p1_w: np.ndarray,
    rgba: np.ndarray,
    width_px: float = 5.0,
) -> None:
    """Add a single world-space line segment to the scene (visualization only)."""
    try:
        if int(scene.ngeom) >= int(scene.maxgeom):
            return
        geom = scene.geoms[int(scene.ngeom)]
        size = np.zeros(3, dtype=np.float64)
        pos = np.zeros(3, dtype=np.float64)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, pos, mat, np.asarray(rgba, dtype=np.float32).reshape(4))
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            float(width_px),
            np.asarray(p0_w, dtype=np.float64).reshape(3),
            np.asarray(p1_w, dtype=np.float64).reshape(3),
        )
        scene.ngeom = int(scene.ngeom) + 1
    except Exception:
        return


def main() -> None:
    default_model = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mjcf", "hopper_serial.xml"))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="MuJoCo MJCF path. Recommended: Hopper_sim/mjcf/hopper_serial.xml (serial-equivalent roll/pitch/shift plant).",
    )
    ap.add_argument("--lcm-url", type=str, default="udpm://239.255.76.67:7667?ttl=255")
    ap.add_argument("--duration-s", type=float, default=0.0, help="<=0 means run forever.")
    ap.add_argument("--arm", action="store_true", help="Force-arm outputs regardless of motor_pwm_lcmt.control_mode")
    ap.add_argument("--viewer", action="store_true", help="Open a MuJoCo viewer window (best for debugging).")
    ap.add_argument("--realtime", action="store_true", help="Sleep to approximate real-time stepping (useful with --viewer).")
    ap.add_argument("--print-hz", type=float, default=2.0, help="Console print rate (Hz). 0 disables printing.")
    ap.add_argument("--init-roll-deg", type=float, default=0.0, help="Initial base roll angle (deg).")
    ap.add_argument("--init-pitch-deg", type=float, default=0.0, help="Initial base pitch angle (deg).")
    ap.add_argument("--init-yaw-deg", type=float, default=0.0, help="Initial base yaw angle (deg).")
    ap.add_argument("--init-base-z", type=float, default=None, help="Initial base height z (m).")
    ap.add_argument(
        "--init-joint-pos",
        type=str,
        default=None,
        help="Override initial 3 joint positions (comma-separated). Example: '0,0,0.05'.",
    )

    # Recording
    ap.add_argument("--record-gif", type=str, default=None, help="Write an animated GIF to this path (offscreen render).")
    ap.add_argument("--record-mp4", type=str, default=None, help="Write an MP4 video to this path (offscreen render).")
    ap.add_argument("--record-fps", type=float, default=20.0, help="Recording FPS (for --record-gif).")
    ap.add_argument("--record-width", type=int, default=640, help="Recording frame width.")
    ap.add_argument("--record-height", type=int, default=480, help="Recording frame height.")
    ap.add_argument("--hud", action="store_true", help="Overlay HUD text (phase/PWM/forces) onto recorded frames.")
    ap.add_argument("--hud-font-size", type=int, default=18, help="HUD font size (pixels).")
    ap.add_argument("--cam-follow-tau", type=float, default=0.35, help="Camera follow smoothing time constant (s).")
    ap.add_argument("--cam-lookat-z", type=float, default=0.75, help="Camera lookat Z (m).")
    ap.add_argument("--draw-frames", action="store_true", help="Draw WORLD/base/IMU coordinate frames in the rendered scene.")
    ap.add_argument("--frame-axis-len", type=float, default=0.30, help="Axis length (m) for --draw-frames.")
    ap.add_argument("--frame-width-px", type=float, default=6.0, help="Line width (px) for --draw-frames/--draw-leg-line.")
    ap.add_argument("--imu-axis-len-scale", type=float, default=1.25, help="IMU axis length scale relative to --frame-axis-len (IMU longer by default).")
    ap.add_argument("--imu-width-scale", type=float, default=0.60, help="IMU axis line width scale relative to --frame-width-px (IMU thinner by default).")
    ap.add_argument("--draw-leg-line", action="store_true", help="Draw base->foot leg direction line in the rendered scene.")

    # Small 'realistic' randomization (visual + tiny init perturbations)
    ap.add_argument("--random-scene", action="store_true", help="Enable small scene randomization (camera jitter, init pose noise).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random-scene. 0 uses time-based seed.")
    ap.add_argument("--cam-jitter-m", type=float, default=0.0, help="Per-frame camera lookat XY jitter (m) when recording.")
    ap.add_argument("--cam-az-jitter-deg", type=float, default=0.0, help="Per-frame camera azimuth jitter (deg) when recording.")
    ap.add_argument("--cam-el-jitter-deg", type=float, default=0.0, help="Per-frame camera elevation jitter (deg) when recording.")
    ap.add_argument("--rand-init-rpy-deg", type=float, default=0.0, help="Random init roll/pitch/yaw noise magnitude (deg).")
    ap.add_argument("--rand-init-xy-m", type=float, default=0.0, help="Random init base XY noise magnitude (m).")
    ap.add_argument("--rand-friction-mu", type=float, default=0.0, help="Randomize ground friction mu by +/- this amount.")

    # Scripted demo helpers (no real gamepad needed)
    ap.add_argument("--hold-level-s", type=float, default=0.0, help="Hold robot level (frozen pose) for this many seconds of SIM time.")
    ap.add_argument("--hold-tilt-s", type=float, default=0.0, help="Then tilt while still holding for this many seconds of SIM time.")
    ap.add_argument("--hold-pitch-deg", type=float, default=0.0, help="Pitch target during hold-tilt stage (deg).")
    ap.add_argument("--hold-roll-deg", type=float, default=0.0, help="Roll target during hold-tilt stage (deg).")
    ap.add_argument("--hold-kp-pos", type=float, default=2500.0, help="Hold (hover) PD: position kp (N/m).")
    ap.add_argument("--hold-kd-pos", type=float, default=260.0, help="Hold (hover) PD: position kd (N/(m/s)).")
    ap.add_argument("--hold-kp-rot", type=float, default=240.0, help="Hold (hover) PD: attitude kp (Nm/rad).")
    ap.add_argument("--hold-kd-rot", type=float, default=45.0, help="Hold (hover) PD: attitude kd (Nm/(rad/s)).")
    ap.add_argument("--hold-force-max", type=float, default=250.0, help="Max magnitude of hold force (N) applied to base.")
    ap.add_argument("--hold-tau-max", type=float, default=80.0, help="Max magnitude of hold torque (Nm) applied to base.")

    ap.add_argument("--fake-gamepad", action="store_true", help="Publish synthetic gamepad_lcmt (vx/vy commands) for run_modee.py.")
    ap.add_argument(
        "--fake-gamepad-y-once",
        action="store_true",
        help="If set, publish gamepad Y=1 for ONE message at startup (rising edge) to trigger ModeE LOG START + user_reset().",
    )
    ap.add_argument(
        "--fake-gamepad-y-hold-s",
        type=float,
        default=0.0,
        help="If >0, hold gamepad Y=1 for this many *simulation* seconds (more robust than --fake-gamepad-y-once if the controller starts late).",
    )
    ap.add_argument("--gamepad-hz", type=float, default=50.0, help="Synthetic gamepad publish rate (Hz).")
    ap.add_argument("--gamepad-max-cmd-vel", type=float, default=0.8, help="Must match run_modee.py --max-cmd-vel.")
    ap.add_argument(
        "--gamepad-vx-sign",
        type=float,
        default=-1.0,
        help="Sign applied to commanded vx when mapping to gamepad stick. Default -1 matches ModeE WORLD(+x) convention to MuJoCo forward in this sim.",
    )
    ap.add_argument("--cmd-vx0", type=float, default=0.0, help="Desired vx (m/s) after release, before switch.")
    ap.add_argument("--cmd-vy0", type=float, default=0.0, help="Desired vy (m/s) after release, before switch.")
    ap.add_argument("--cmd-vx1", type=float, default=0.30, help="Desired vx (m/s) after switch.")
    ap.add_argument("--cmd-vy1", type=float, default=0.0, help="Desired vy (m/s) after switch.")
    ap.add_argument("--cmd-switch-after-s", type=float, default=5.0, help="Switch to (vx1,vy1) this many seconds AFTER release.")
    ap.add_argument("--cmd-vx2", type=float, default=0.0, help="Desired vx (m/s) after second switch (optional).")
    ap.add_argument("--cmd-vy2", type=float, default=0.0, help="Desired vy (m/s) after second switch (optional).")
    ap.add_argument(
        "--cmd-switch2-after-s",
        type=float,
        default=1.0e9,
        help="Second switch time (s AFTER release) to (vx2,vy2). Set huge to disable.",
    )
    ap.add_argument("--cmd-ramp-s", type=float, default=0.6, help="Smooth ramp duration (s) for vx/vy change.")
    ap.add_argument(
        "--init-motor-deg",
        type=float,
        default=None,
        help="(3RSR model) Override all 3 motor joint angles to this value (deg) at reset.",
    )

    # Match real robot sign/offset conventions (see HopperHardware::_fill_in_motor_data_to_lcm()).
    ap.add_argument("--q-sign", type=float, default=-1.0, help="LCM joint sign: q_lcm = q_sign*motor_pos + q_offset")
    ap.add_argument("--q-offset", type=float, default=-1.047, help="LCM joint offset (rad)")
    ap.add_argument(
        "--lcm-to-mj-order",
        type=str,
        default="0,1,2",
        help="Mapping from LCM motor index -> MuJoCo motor index (0..2). Example: '1,2,0'",
    )
    ap.add_argument(
        "--leg-l0-m",
        type=float,
        default=0.464,
        help="ModeE nominal leg length (m). Used ONLY for debug prints (q_shift estimate).",
    )

    # Prop mapping in simulation: map PWM indices to arm tips (same convention as ModeEConfig.prop_pwm_idx_per_arm).
    ap.add_argument(
        "--prop-pwm-idx-per-arm",
        type=str,
        default="2;1;3",
        help="PWM indices per arm (RED;GREEN;BLUE). Format: '2;1;3' or '0,4;1,5;2,3'",
    )
    ap.add_argument("--prop-arm-len", type=float, default=0.569451, help="Prop arm length (m)")
    ap.add_argument("--prop-z-off", type=float, default=0.03, help="Coaxial z offset (m) when an arm has 2+ PWM channels")
    ap.add_argument("--strict-1d", action="store_true",
                    help="Lock base x/y/roll/pitch/yaw every step → pure 1-DOF vertical hopping.")
    ap.add_argument("--strict-2d", action="store_true",
                    help="Lock base y every step (X-Z plane motion).")
    ap.add_argument("--strict-2d-unlock-att-after-s", type=float, default=0.0,
                    help="If >0 with --strict-2d, keep attitude locked before this many seconds after RELEASE, then unlock.")

    args = ap.parse_args()
    if bool(getattr(args, "strict_1d", False)) and bool(getattr(args, "strict_2d", False)):
        ap.error("--strict-1d and --strict-2d are mutually exclusive.")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    # If the model defines a 'home' keyframe, prefer it for stable initialization.
    try:
        kid = int(model.key("home").id)
        mujoco.mj_resetDataKeyframe(model, data, kid)
        mujoco.mj_forward(model, data)
    except Exception:
        mujoco.mj_forward(model, data)

    # Optional initial base orientation override
    r0 = float(np.deg2rad(float(args.init_roll_deg)))
    p0 = float(np.deg2rad(float(args.init_pitch_deg)))
    y0 = float(np.deg2rad(float(args.init_yaw_deg)))
    if (abs(r0) > 1e-12) or (abs(p0) > 1e-12) or (abs(y0) > 1e-12):
        q0 = _quat_from_rpy_xyz_wxyz(r0, p0, y0)
        data.qpos[3:7] = q0.reshape(4)
        # zero base angular velocity to match "initial state" intent
        data.qvel[3:6] = 0.0
        mujoco.mj_forward(model, data)

    # Detect plant type by joint names:
    # - 3RSR parallel plant: motor1_joint/motor2_joint/motor3_joint
    # - serial equivalent plant: Leg_Joint_Roll/Leg_Joint_Pitch/Leg_Joint_Shift
    is_3rsr = False
    try:
        _ = model.joint("motor1_joint").id
        is_3rsr = True
    except Exception:
        is_3rsr = False

    if is_3rsr:
        motor_joint_names = ["motor1_joint", "motor2_joint", "motor3_joint"]
        actuator_names = ["motor1", "motor2", "motor3"]
        foot_body_name = "foot_link"
    else:
        motor_joint_names = ["Leg_Joint_Roll", "Leg_Joint_Pitch", "Leg_Joint_Shift"]
        actuator_names = ["roll_motor", "pitch_motor", "shift_motor"]
        foot_body_name = "Foot_Link"

    motor_joint_ids = [int(model.joint(n).id) for n in motor_joint_names]
    motor_qpos_adr = [int(model.jnt_qposadr[jid]) for jid in motor_joint_ids]
    motor_qvel_adr = [int(model.jnt_dofadr[jid]) for jid in motor_joint_ids]
    actuator_ids = [int(model.actuator(n).id) for n in actuator_names]

    # Optional initial motor angles override (useful for '初始状态 XX度' experiments on 3RSR model)
    if is_3rsr and (getattr(args, "init_motor_deg", None) is not None):
        th = float(np.deg2rad(float(args.init_motor_deg)))
        for a in motor_qpos_adr:
            data.qpos[int(a)] = th
        for a in motor_qvel_adr:
            data.qvel[int(a)] = 0.0
        mujoco.mj_forward(model, data)

    # Optional initial base height override.
    # For the serial hopper model, default XML base_z=0.5 puts the fully-extended foot below ground.
    # Use a safer default if user didn't specify.
    init_base_z = args.init_base_z
    if init_base_z is None and (not is_3rsr):
        init_base_z = 0.65
    if init_base_z is not None:
        data.qpos[2] = float(init_base_z)
        data.qvel[0:3] = 0.0
        mujoco.mj_forward(model, data)

    # Optional joint position override (applied after any keyframe / init_motor_deg).
    if args.init_joint_pos is not None:
        q0 = np.asarray(_parse_float_list(args.init_joint_pos, n=3), dtype=float).reshape(3)
        for i, a in enumerate(motor_qpos_adr):
            data.qpos[int(a)] = float(q0[i])
        for a in motor_qvel_adr:
            data.qvel[int(a)] = 0.0
        mujoco.mj_forward(model, data)

    # Ground geom id (for contact/GRF HUD)
    ground_geom_id = None
    try:
        ground_geom_id = int(model.geom("ground").id)
    except Exception:
        ground_geom_id = None

    # Base for IMU + prop forces
    base_body_id = int(model.body("base_link").id)
    foot_body_id = int(model.body(foot_body_name).id)
    imu_site_id = None
    try:
        imu_site_id = int(model.site("imu_site").id)
    except Exception:
        imu_site_id = None

    # Optional small randomization (init pose + friction). This only affects simulation; LCM semantics stay identical.
    rng = None
    if bool(getattr(args, "random_scene", False)):
        seed = int(getattr(args, "seed", 0))
        if seed <= 0:
            seed = int(time.time() * 1e6) & 0xFFFFFFFF
        rng = np.random.default_rng(int(seed))
        print(f"[mujoco_lcm_fake_robot] random_scene=1 seed={seed}")

        # Randomize ground friction mu slightly (if ground geom exists)
        dmu = float(max(0.0, float(getattr(args, "rand_friction_mu", 0.0))))
        if (dmu > 0.0) and (ground_geom_id is not None):
            try:
                mu0 = float(model.geom_friction[int(ground_geom_id), 0])
                mu1 = float(max(0.0, mu0 + float(rng.uniform(-dmu, +dmu))))
                model.geom_friction[int(ground_geom_id), 0] = mu1
                print(f"[mujoco_lcm_fake_robot] ground mu: {mu0:.3f} -> {mu1:.3f}")
            except Exception:
                pass

        # Randomize initial base pose slightly (tiny, for realism)
        dxy = float(max(0.0, float(getattr(args, "rand_init_xy_m", 0.0))))
        drpy_deg = float(max(0.0, float(getattr(args, "rand_init_rpy_deg", 0.0))))
        if dxy > 0.0:
            data.qpos[0] = float(data.qpos[0] + float(rng.uniform(-dxy, +dxy)))
            data.qpos[1] = float(data.qpos[1] + float(rng.uniform(-dxy, +dxy)))
        if drpy_deg > 0.0:
            try:
                R0 = _quat_to_R_wb(np.asarray(data.qpos[3:7], dtype=float).reshape(4))
                rpy0 = _R_to_rpy_xyz(R0)
                dr = float(np.deg2rad(drpy_deg))
                rpy1 = rpy0 + rng.uniform(-dr, +dr, size=3)
                data.qpos[3:7] = _quat_from_rpy_xyz_wxyz(float(rpy1[0]), float(rpy1[1]), float(rpy1[2])).reshape(4)
                data.qvel[3:6] = 0.0
            except Exception:
                pass
        mujoco.mj_forward(model, data)

    fk = ForwardKinematics()

    # LCM
    lc = lcm.LCM(str(args.lcm_url))

    # Commands
    tau_ff_lcm = np.zeros(3, dtype=float)
    pwm_us_cmd = np.ones(6, dtype=float) * 1000.0
    cmd_armed = False

    def _on_cmd(_chan: str, payload: bytes):
        nonlocal tau_ff_lcm
        try:
            msg = hopper_cmd_lcmt.decode(payload)
            tau_ff_lcm = np.asarray(msg.tau_ff, dtype=float).reshape(3).copy()
        except Exception:
            return

    def _on_pwm(_chan: str, payload: bytes):
        nonlocal pwm_us_cmd, cmd_armed
        try:
            msg = motor_pwm_lcmt.decode(payload)
            pwm_us_cmd = np.asarray(msg.pwm_values, dtype=float).reshape(6).copy()
            cmd_armed = bool(int(getattr(msg, "control_mode", 0)) > 0)
        except Exception:
            return

    lc.subscribe("hopper_cmd_lcmt", _on_cmd)
    lc.subscribe("motor_pwm_lcmt", _on_pwm)

    # Motor model (PWM -> thrust)
    motor_table = MotorTableModel.default_from_table()

    # Prop geometry (body frame)
    L = float(args.prop_arm_len)
    # order: [RED, GREEN, BLUE] (GREEN forward +X)
    prop_arm_pos_b = np.array(
        [
            [-0.5 * L, +np.sqrt(3.0) * 0.5 * L, 0.0],
            [+1.0 * L, 0.0, 0.0],
            [-0.5 * L, -np.sqrt(3.0) * 0.5 * L, 0.0],
        ],
        dtype=float,
    )
    pwm_groups = _parse_arm_groups(str(args.prop_pwm_idx_per_arm))
    motor_pos_b = np.zeros((6, 3), dtype=float)
    for arm_i in range(3):
        idxs = pwm_groups[arm_i]
        if len(idxs) == 0:
            continue
        for k_i, pwm_idx in enumerate(idxs):
            if pwm_idx < 0 or pwm_idx > 5:
                continue
            z_off = 0.0
            if len(idxs) >= 2:
                z_off = float(args.prop_z_off) * (1.0 if (k_i % 2 == 0) else -1.0)
            motor_pos_b[int(pwm_idx), :] = prop_arm_pos_b[arm_i, :] + np.array([0.0, 0.0, z_off], dtype=float)

    # LCM sign/offset mapping
    q_sign = float(args.q_sign)
    q_offset = float(args.q_offset)
    phys_to_mj: Sequence[int] = _parse_int_list(str(args.lcm_to_mj_order), n=3)
    if len(set(phys_to_mj)) != 3:
        raise ValueError(f"--lcm-to-mj-order must be a permutation of 0,1,2. Got {phys_to_mj}")

    dt = float(model.opt.timestep)
    t0 = float(time.time())
    last_print_t = 0.0

    viewer = None
    if bool(args.viewer):
        try:
            import mujoco.viewer as mj_viewer  # type: ignore

            viewer = mj_viewer.launch_passive(model, data)
        except Exception as e:
            print(f"[mujoco_lcm_fake_robot] WARNING: failed to launch viewer ({e}). Running headless.")
            viewer = None

    # Optional recording (GIF)
    gif_writer = None
    mp4_proc = None
    renderer = None
    cam = None
    cam_base_az = None
    cam_base_el = None
    cam_lookat_f = None
    cam_az_f = None
    cam_el_f = None
    record_path = getattr(args, "record_gif", None)
    record_mp4_path = getattr(args, "record_mp4", None)
    record_fps = float(max(1e-3, float(getattr(args, "record_fps", 20.0))))
    record_dt = 1.0 / record_fps
    next_frame_sim_t = 0.0
    sim_t = 0.0
    if record_path or record_mp4_path:
        try:
            if record_path:
                import imageio.v2 as imageio  # type: ignore

                os.makedirs(os.path.dirname(os.path.abspath(str(record_path))), exist_ok=True)
                gif_writer = imageio.get_writer(str(record_path), mode="I", duration=record_dt)
            renderer = mujoco.Renderer(model, width=int(args.record_width), height=int(args.record_height))
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            # A nice default view for hopping
            cam.azimuth = 90.0
            cam.elevation = -15.0
            cam.distance = 2.8
            cam.lookat[:] = np.array([0.0, 0.0, float(getattr(args, "cam_lookat_z", 0.75))], dtype=float)
            if rng is not None:
                # Mild camera randomization for a more "real" look
                cam.azimuth = float(cam.azimuth + float(rng.uniform(-12.0, +12.0)))
                cam.elevation = float(cam.elevation + float(rng.uniform(-4.0, +4.0)))
                cam.distance = float(cam.distance * float(rng.uniform(0.92, 1.08)))
            cam_base_az = float(cam.azimuth)
            cam_base_el = float(cam.elevation)
            cam_lookat_f = np.asarray(cam.lookat, dtype=float).reshape(3).copy()
            cam_az_f = float(cam.azimuth)
            cam_el_f = float(cam.elevation)

            # Optional MP4 writer via ffmpeg rawvideo pipe (no extra Python deps)
            if record_mp4_path:
                os.makedirs(os.path.dirname(os.path.abspath(str(record_mp4_path))), exist_ok=True)
                w = int(args.record_width)
                h = int(args.record_height)
                fps = float(record_fps)
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    f"{fps:.6f}",
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(record_mp4_path),
                ]
                mp4_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                print(f"[mujoco_lcm_fake_robot] recording MP4 -> {os.path.abspath(str(record_mp4_path))}")

            if record_path:
                print(f"[mujoco_lcm_fake_robot] recording GIF -> {os.path.abspath(str(record_path))}")
        except Exception as e:
            print(f"[mujoco_lcm_fake_robot] WARNING: failed to start recording ({e}).")
            gif_writer = None
            mp4_proc = None
            renderer = None
            cam = None

    # Scripted demo: hold -> tilt -> release timeline in SIM time.
    hold_level_s = float(max(0.0, float(getattr(args, "hold_level_s", 0.0))))
    hold_tilt_s = float(max(0.0, float(getattr(args, "hold_tilt_s", 0.0))))
    hold_total_s = float(hold_level_s + hold_tilt_s)
    hold_qpos0 = np.asarray(data.qpos, dtype=float).copy()
    # Lock ONLY the floating base during HOLD; let the leg joints move under control.
    hold_base_pos0 = np.asarray(hold_qpos0[0:3], dtype=float).reshape(3).copy()
    hold_base_quat0 = np.asarray(hold_qpos0[3:7], dtype=float).reshape(4).copy()
    strict2d_quat_lock = hold_base_quat0.copy()
    # Keep yaw constant during hold-tilt (like “tilt by hand”)
    try:
        hold_yaw0 = float(_R_to_rpy_xyz(_quat_to_R_wb(hold_base_quat0))[2])
    except Exception:
        hold_yaw0 = 0.0
    hold_pitch_tgt = float(np.deg2rad(float(getattr(args, "hold_pitch_deg", 0.0))))
    hold_roll_tgt = float(np.deg2rad(float(getattr(args, "hold_roll_deg", 0.0))))
    released = False
    release_sim_t = 0.0

    # Synthetic gamepad publish schedule (relative to release)
    fake_gamepad = bool(getattr(args, "fake_gamepad", False))
    fake_gamepad_y_once = bool(getattr(args, "fake_gamepad_y_once", False))
    fake_gamepad_y_hold_s = float(max(0.0, float(getattr(args, "fake_gamepad_y_hold_s", 0.0))))
    y_once_sent = False
    gp_hz = float(max(1e-3, float(getattr(args, "gamepad_hz", 50.0))))
    gp_period = 1.0 / gp_hz
    next_gp_sim_t = 0.0
    gp_max_v = float(max(1e-6, float(getattr(args, "gamepad_max_cmd_vel", 0.8))))
    gp_vx_sign = float(getattr(args, "gamepad_vx_sign", -1.0))
    cmd_vx0 = float(getattr(args, "cmd_vx0", 0.0))
    cmd_vy0 = float(getattr(args, "cmd_vy0", 0.0))
    cmd_vx1 = float(getattr(args, "cmd_vx1", 0.30))
    cmd_vy1 = float(getattr(args, "cmd_vy1", 0.0))
    cmd_switch_after = float(max(0.0, float(getattr(args, "cmd_switch_after_s", 5.0))))
    cmd_vx2 = float(getattr(args, "cmd_vx2", 0.0))
    cmd_vy2 = float(getattr(args, "cmd_vy2", 0.0))
    cmd_switch2_after = float(max(0.0, float(getattr(args, "cmd_switch2_after_s", 1.0e9))))
    cmd_ramp = float(max(1e-6, float(getattr(args, "cmd_ramp_s", 0.6))))
    strict_2d_unlock_att_after = float(max(0.0, float(getattr(args, "strict_2d_unlock_att_after_s", 0.0))))
    vx_cmd = 0.0
    vy_cmd = 0.0
    stick_x = 0.0
    stick_y = 0.0

    # Total mass (for hold gravity compensation)
    m_tot = float(np.sum(np.asarray(model.body_mass, dtype=float)))
    g_mag = float(abs(float(model.opt.gravity[2])))
    g_w_const = np.array([0.0, 0.0, -g_mag], dtype=float)

    while True:
        loop_t0 = float(time.time())

        # Stop condition (SIM time). This keeps demo timing consistent even if rendering/recording is slow.
        if float(args.duration_s) > 0.0 and float(sim_t) >= float(args.duration_s):
            break

        # Drain inbound LCM packets (non-blocking)
        for _ in range(16):
            if lc.handle_timeout(0) <= 0:
                break

        # HOLD sequence for the first hold_total_s seconds of SIM time:
        # - Base is "held" using an external PD wrench (more physical than teleporting qpos).
        # - Leg joints are free to move under controller torques.
        hold_active = bool(hold_total_s > 0.0) and (float(sim_t) < float(hold_total_s))
        hold_base_quat_cmd = None
        hold_force_w = np.zeros(3, dtype=float)
        hold_torque_w = np.zeros(3, dtype=float)
        if hold_active:
            # Apply tilt during the second stage (smooth ramp to target over hold_tilt_s).
            # We set the base quaternion explicitly; joint targets in FLIGHT will then rotate accordingly.
            roll = 0.0
            pitch = 0.0
            if hold_tilt_s > 1e-9 and float(sim_t) >= float(hold_level_s) - 1e-12:
                u = float((float(sim_t) - float(hold_level_s)) / float(hold_tilt_s))
                u = float(np.clip(u, 0.0, 1.0))
                u = float(u * u * (3.0 - 2.0 * u))  # smoothstep
                roll = float(u * hold_roll_tgt)
                pitch = float(u * hold_pitch_tgt)
            hold_base_quat_cmd = _quat_from_rpy_xyz_wxyz(float(roll), float(pitch), float(hold_yaw0)).reshape(4)

            # External PD wrench to hold base position + attitude
            try:
                # Current base state
                p_now = np.asarray(data.qpos[0:3], dtype=float).reshape(3)
                v_now = np.asarray(data.qvel[0:3], dtype=float).reshape(3)  # world linear vel
                q_now = np.asarray(data.qpos[3:7], dtype=float).reshape(4)
                R_now = _quat_to_R_wb(q_now)
                R_des = _quat_to_R_wb(hold_base_quat_cmd)
                # orientation error in body frame
                E = (R_des.T @ R_now) - (R_now.T @ R_des)
                e_R_b = 0.5 * _vee_so3(E)
                omega_b = np.asarray(data.qvel[3:6], dtype=float).reshape(3)  # body ang vel

                kp_p = float(max(0.0, float(getattr(args, "hold_kp_pos", 2500.0))))
                kd_p = float(max(0.0, float(getattr(args, "hold_kd_pos", 260.0))))
                kp_R = float(max(0.0, float(getattr(args, "hold_kp_rot", 240.0))))
                kd_R = float(max(0.0, float(getattr(args, "hold_kd_rot", 45.0))))

                # Force in world: PD + gravity compensation
                F_w = kp_p * (hold_base_pos0.reshape(3) - p_now.reshape(3)) - kd_p * v_now.reshape(3)
                F_w = (F_w - float(m_tot) * g_w_const).reshape(3)  # subtract g_w (down) -> add upward

                # Torque: PD in body then rotate to world
                tau_b = (-kp_R * e_R_b - kd_R * omega_b).reshape(3)
                tau_w = (R_now @ tau_b.reshape(3)).reshape(3)

                # Clip
                fmax = float(max(0.0, float(getattr(args, "hold_force_max", 250.0))))
                tmax = float(max(0.0, float(getattr(args, "hold_tau_max", 80.0))))
                if fmax > 1e-9:
                    fn = float(np.linalg.norm(F_w))
                    if fn > fmax:
                        F_w = (F_w * (fmax / fn)).astype(float)
                if tmax > 1e-9:
                    tn = float(np.linalg.norm(tau_w))
                    if tn > tmax:
                        tau_w = (tau_w * (tmax / tn)).astype(float)

                hold_force_w = np.asarray(F_w, dtype=float).reshape(3)
                hold_torque_w = np.asarray(tau_w, dtype=float).reshape(3)
            except Exception:
                hold_force_w[:] = 0.0
                hold_torque_w[:] = 0.0
        else:
            if (not bool(released)) and bool(hold_total_s > 0.0):
                released = True
                release_sim_t = float(sim_t)

        # Synthetic gamepad command profile:
        # - During HOLD: publish vx/vy (default 0) so the controller is already "alive".
        # - After RELEASE: start with (vx0,vy0),
        #   then at t=cmd_switch_after ramp to (vx1,vy1),
        #   then at t=cmd_switch2_after ramp to (vx2,vy2) (optional; disable with huge cmd_switch2_after).
        t_after_release = float(max(0.0, float(sim_t) - float(release_sim_t)))
        # Ensure ordering (backward compatible): if cmd_switch2_after <= cmd_switch_after, disable the second switch.
        t_sw1 = float(cmd_switch_after)
        t_sw2 = float(cmd_switch2_after)
        if t_sw2 <= t_sw1 + 1e-9:
            t_sw2 = 1.0e9

        if t_after_release < float(t_sw1):
            vx_cmd = float(cmd_vx0)
            vy_cmd = float(cmd_vy0)
        elif t_after_release < float(t_sw2):
            u = float((t_after_release - float(t_sw1)) / float(cmd_ramp))
            u = float(np.clip(u, 0.0, 1.0))
            u = float(u * u * (3.0 - 2.0 * u))  # smoothstep
            vx_cmd = float((1.0 - u) * float(cmd_vx0) + u * float(cmd_vx1))
            vy_cmd = float((1.0 - u) * float(cmd_vy0) + u * float(cmd_vy1))
        else:
            u = float((t_after_release - float(t_sw2)) / float(cmd_ramp))
            u = float(np.clip(u, 0.0, 1.0))
            u = float(u * u * (3.0 - 2.0 * u))  # smoothstep
            vx_cmd = float((1.0 - u) * float(cmd_vx1) + u * float(cmd_vx2))
            vy_cmd = float((1.0 - u) * float(cmd_vy1) + u * float(cmd_vy2))

        stick_x = float(np.clip((vx_cmd * gp_vx_sign) / gp_max_v, -1.0, 1.0))
        stick_y = float(np.clip(vy_cmd / gp_max_v, -1.0, 1.0))
        if bool(fake_gamepad) and (float(sim_t) >= float(next_gp_sim_t) - 1e-12):
            try:
                gp = gamepad_lcmt()
                gp.rightStickAnalog = [float(stick_x), float(stick_y)]
                if fake_gamepad_y_hold_s > 1e-9:
                    # Hold Y high for a short window so the controller reliably sees the rising edge
                    # even if it starts after MuJoCo.
                    gp.y = 1 if float(sim_t) < fake_gamepad_y_hold_s else 0
                elif bool(fake_gamepad_y_once) and (not bool(y_once_sent)):
                    # Rising edge for one publish is enough for ModeE logging trigger.
                    gp.y = 1
                    y_once_sent = True
                else:
                    gp.y = 0
                lc.publish("gamepad_lcmt", gp.encode())
            except Exception:
                pass
            next_gp_sim_t = float(next_gp_sim_t + gp_period)

        armed = bool(args.arm) or bool(cmd_armed)

        # --- Apply joint torques (tau_ff only) ---
        tau_motor_phys = (q_sign * np.asarray(tau_ff_lcm, dtype=float).reshape(3)).reshape(3) if armed else np.zeros(3, dtype=float)
        tau_apply = np.zeros(3, dtype=float)
        if is_3rsr:
            # reorder to MuJoCo motor indices
            for i in range(3):
                j = int(phys_to_mj[i])
                tau_apply[j] = float(tau_motor_phys[i])
        else:
            # serial plant uses LCM order directly: [roll, pitch, shift]
            tau_apply = tau_motor_phys.copy()
        for j, aid in enumerate(actuator_ids):
            data.ctrl[int(aid)] = float(tau_apply[j])

        # --- Apply propeller forces as external wrench on base_link ---
        pwm_apply = pwm_us_cmd if armed else (np.ones(6, dtype=float) * 1000.0)
        thrusts6 = motor_table.thrust_from_pwm(np.asarray(pwm_apply, dtype=float).reshape(6))

        # Compute base rotation in world
        body_quat = np.asarray(data.qpos[3:7], dtype=float).reshape(4)  # [w,x,y,z]
        R_wb = _quat_to_R_wb(body_quat)

        # Sum forces/torques in BODY then rotate to WORLD
        total_force_w = np.zeros(3, dtype=float)
        total_torque_w = np.zeros(3, dtype=float)
        ez_b = np.array([0.0, 0.0, 1.0], dtype=float)
        for idx in range(6):
            thrust = float(thrusts6[idx])
            if thrust == 0.0:
                continue
            r_b = np.asarray(motor_pos_b[idx], dtype=float).reshape(3)
            f_b = ez_b * thrust
            tau_b = np.cross(r_b, f_b)
            total_force_w += (R_wb @ f_b.reshape(3)).reshape(3)
            total_torque_w += (R_wb @ tau_b.reshape(3)).reshape(3)

        # Apply wrench on base body (world frame): props + (optional) hold wrench
        data.xfrc_applied[base_body_id, 0:3] = (total_force_w + hold_force_w).reshape(3)
        data.xfrc_applied[base_body_id, 3:6] = (total_torque_w + hold_torque_w).reshape(3)

        # Step physics
        mujoco.mj_step(model, data)

        # ---- strict 1-D rail: lock everything except z translation ----
        if bool(getattr(args, "strict_1d", False)):
            data.qpos[0] = 0.0   # x
            data.qpos[1] = 0.0   # y
            data.qpos[3] = 1.0   # quat w
            data.qpos[4] = 0.0   # quat x
            data.qpos[5] = 0.0   # quat y
            data.qpos[6] = 0.0   # quat z
            data.qvel[0] = 0.0   # vx
            data.qvel[1] = 0.0   # vy
            data.qvel[3] = 0.0   # omega_x
            data.qvel[4] = 0.0   # omega_y
            data.qvel[5] = 0.0   # omega_z
            mujoco.mj_forward(model, data)
        elif bool(getattr(args, "strict_2d", False)):
            # X-Z planar rail: keep lateral DOF locked.
            data.qpos[1] = 0.0   # y
            data.qvel[1] = 0.0   # vy

            # Optional staged unlock: keep attitude fixed for early hopping, then free it.
            if strict_2d_unlock_att_after > 1e-9 and t_after_release < strict_2d_unlock_att_after:
                data.qpos[3:7] = strict2d_quat_lock.reshape(4)
                data.qvel[3:6] = 0.0  # omega xyz
            mujoco.mj_forward(model, data)

        # Advance our sim clock regardless (recording/command schedules are in sim-time)
        sim_t += dt

        # --- Publish state to LCM ---
        # Motor states in MuJoCo
        motor_pos_mj = np.array([float(data.qpos[a]) for a in motor_qpos_adr], dtype=float).reshape(3)
        motor_vel_mj = np.array([float(data.qvel[a]) for a in motor_qvel_adr], dtype=float).reshape(3)

        # Reorder to LCM physical motor order (only needed for 3RSR parallel plant).
        if is_3rsr:
            motor_pos_phys = np.zeros(3, dtype=float)
            motor_vel_phys = np.zeros(3, dtype=float)
            for i in range(3):
                j = int(phys_to_mj[i])
                motor_pos_phys[i] = float(motor_pos_mj[j])
                motor_vel_phys[i] = float(motor_vel_mj[j])
        else:
            motor_pos_phys = motor_pos_mj.copy()
            motor_vel_phys = motor_vel_mj.copy()

        q_lcm = (q_sign * motor_pos_phys + q_offset).astype(float)
        qd_lcm = (q_sign * motor_vel_phys).astype(float)
        tauIq_lcm = (q_sign * tau_motor_phys).astype(float)

        jd = hopper_data_lcmt()
        jd.q = [float(v) for v in q_lcm.reshape(3)]
        jd.qd = [float(v) for v in qd_lcm.reshape(3)]
        jd.tauIq = [float(v) for v in tauIq_lcm.reshape(3)]
        lc.publish("hopper_data_lcmt", jd.encode())

        # IMU
        # NOTE (MuJoCo freejoint convention):
        # - data.qvel[3:6] for a free joint is the angular velocity expressed in the BODY/LOCAL frame.
        # - Therefore this already matches the real robot LCMT convention: gyro in body frame (FLU).
        gyro_b = np.asarray(data.qvel[3:6], dtype=float).reshape(3)
        rpy = _R_to_rpy_xyz(R_wb)

        # Publish accel as -(specific force) in body:
        #   specific_force_w = a_w - g_w
        #   acc_b = -R^T*(a_w - g_w) = R^T*(g_w - a_w)
        g_w = np.array([0.0, 0.0, -9.81], dtype=float)
        a_w = np.asarray(data.qacc[0:3], dtype=float).reshape(3)
        # During HOLD we "support" the robot externally (frozen pose), so IMU should look like stationary:
        # a_w ≈ 0  => acc_b ≈ R^T * g_w (gravity/down vector in body frame).
        if bool(hold_active):
            a_w = np.zeros(3, dtype=float)
        specific_force_w = (a_w - g_w).reshape(3)
        specific_force_b = (R_wb.T @ specific_force_w.reshape(3)).reshape(3)
        acc_b = (-specific_force_b).astype(float)

        im = hopper_imu_lcmt()
        im.quat = [float(v) for v in body_quat.reshape(4)]
        im.gyro = [float(v) for v in gyro_b.reshape(3)]
        im.acc = [float(v) for v in acc_b.reshape(3)]
        im.rpy = [float(v) for v in rpy.reshape(3)]
        lc.publish("hopper_imu_lcmt", im.encode())

        # Record frames (sim-time)
        if (renderer is not None) and (cam is not None) and ((gif_writer is not None) or (mp4_proc is not None)):
            if sim_t >= next_frame_sim_t:
                try:
                    # Smooth-follow camera: track base in XY (low-pass), keep a fixed Z lookat.
                    base_pos = np.asarray(data.xpos[base_body_id], dtype=float).reshape(3)
                    target_lookat = np.array(
                        [float(base_pos[0]), float(base_pos[1]), float(getattr(args, "cam_lookat_z", 0.75))],
                        dtype=float,
                    )

                    # Noise targets (kept small; filtered below for "soft" movement)
                    az_target = float(cam_base_az) if cam_base_az is not None else float(cam.azimuth)
                    el_target = float(cam_base_el) if cam_base_el is not None else float(cam.elevation)
                    if rng is not None:
                        cam_j = float(max(0.0, float(getattr(args, "cam_jitter_m", 0.0))))
                        if cam_j > 0.0:
                            target_lookat[0] = float(target_lookat[0] + float(rng.normal(0.0, cam_j)))
                            target_lookat[1] = float(target_lookat[1] + float(rng.normal(0.0, cam_j)))
                        az_j = float(max(0.0, float(getattr(args, "cam_az_jitter_deg", 0.0))))
                        el_j = float(max(0.0, float(getattr(args, "cam_el_jitter_deg", 0.0))))
                        if az_j > 0.0:
                            az_target = float(az_target + float(rng.normal(0.0, az_j)))
                        if el_j > 0.0:
                            el_target = float(el_target + float(rng.normal(0.0, el_j)))

                    cam_tau = float(max(0.0, float(getattr(args, "cam_follow_tau", 0.35))))
                    a_cam = float(record_dt / (cam_tau + record_dt)) if cam_tau > 1e-9 else 1.0

                    if cam_lookat_f is None:
                        cam_lookat_f = target_lookat.copy()
                    else:
                        cam_lookat_f = ((1.0 - a_cam) * np.asarray(cam_lookat_f, dtype=float).reshape(3) + a_cam * target_lookat).astype(float)
                    cam.lookat[:] = np.asarray(cam_lookat_f, dtype=float).reshape(3)

                    if cam_az_f is None:
                        cam_az_f = float(cam.azimuth)
                    cam_az_f = float((1.0 - a_cam) * float(cam_az_f) + a_cam * float(az_target))
                    cam.azimuth = float(cam_az_f)

                    if cam_el_f is None:
                        cam_el_f = float(cam.elevation)
                    cam_el_f = float((1.0 - a_cam) * float(cam_el_f) + a_cam * float(el_target))
                    cam.elevation = float(cam_el_f)

                    renderer.update_scene(data, camera=cam)

                    # === Debug draw: frames + leg direction (no effect on physics/controller) ===
                    if bool(getattr(args, "draw_frames", False)):
                        try:
                            scene = renderer.scene
                            axis_len = float(max(1e-6, float(getattr(args, "frame_axis_len", 0.30))))
                            w_px = float(max(1.0, float(getattr(args, "frame_width_px", 6.0))))
                            imu_len = float(axis_len * float(getattr(args, "imu_axis_len_scale", 1.25)))
                            imu_w_px = float(max(1.0, w_px * float(getattr(args, "imu_width_scale", 0.60))))
                            # WORLD frame at origin
                            _add_frame_axes_to_scene(
                                scene,
                                origin_w=np.zeros(3, dtype=float),
                                R_wf=np.eye(3, dtype=float),
                                axis_len_m=axis_len,
                                width_px=w_px,
                                rgba_x=np.array([1.0, 0.2, 0.2, 0.85], dtype=np.float32),
                                rgba_y=np.array([0.2, 1.0, 0.2, 0.85], dtype=np.float32),
                                rgba_z=np.array([0.2, 0.4, 1.0, 0.85], dtype=np.float32),
                            )

                            # Robot/base frame at base_link origin (R_wb: body->world)
                            base_pos_w = np.asarray(data.xpos[base_body_id], dtype=float).reshape(3)
                            R_wb_vis = np.asarray(data.xmat[base_body_id], dtype=float).reshape(3, 3)
                            _add_frame_axes_to_scene(
                                scene,
                                origin_w=base_pos_w,
                                R_wf=R_wb_vis,
                                axis_len_m=axis_len,
                                width_px=(w_px + 1.5),
                            )

                            # IMU frame at imu_site (if present)
                            if imu_site_id is not None:
                                imu_pos_w = np.asarray(data.site_xpos[int(imu_site_id)], dtype=float).reshape(3)
                                R_wi_vis = np.asarray(data.site_xmat[int(imu_site_id)], dtype=float).reshape(3, 3)
                                _add_frame_axes_to_scene(
                                    scene,
                                    origin_w=imu_pos_w,
                                    R_wf=R_wi_vis,
                                    axis_len_m=float(max(1e-6, imu_len)),
                                    width_px=float(max(1.0, imu_w_px)),
                                    rgba_x=np.array([1.0, 1.0, 0.2, 0.90], dtype=np.float32),
                                    rgba_y=np.array([0.7, 0.2, 1.0, 0.90], dtype=np.float32),
                                    rgba_z=np.array([0.2, 1.0, 1.0, 0.90], dtype=np.float32),
                                )

                            # Leg direction: base->foot (MuJoCo ground truth)
                            if bool(getattr(args, "draw_leg_line", False)):
                                foot_pos_w = np.asarray(data.xpos[foot_body_id], dtype=float).reshape(3)
                                _add_world_line_to_scene(
                                    scene,
                                    p0_w=base_pos_w,
                                    p1_w=foot_pos_w,
                                    rgba=np.array([1.0, 1.0, 1.0, 0.85], dtype=np.float32),
                                    width_px=(w_px + 1.0),
                                )
                        except Exception:
                            pass

                    frame = renderer.render()
                    # ensure RGB uint8
                    frame = np.asarray(frame, dtype=np.uint8)
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]

                    # HUD overlay (phase/pwm/forces) - computed from the SAME sim step as the rendered frame (no lag)
                    if bool(getattr(args, "hud", False)):
                        # Phase: stance/flight from foot-ground contact
                        in_contact = False
                        if ground_geom_id is not None:
                            try:
                                for ci in range(int(data.ncon)):
                                    c = data.contact[ci]
                                    g1 = int(c.geom1)
                                    g2 = int(c.geom2)
                                    if (g1 == int(ground_geom_id) and int(model.geom_bodyid[g2]) == int(foot_body_id)) or (
                                        g2 == int(ground_geom_id) and int(model.geom_bodyid[g1]) == int(foot_body_id)
                                    ):
                                        in_contact = True
                                        break
                            except Exception:
                                in_contact = False
                        phase = "STANCE" if bool(in_contact) else "FLIGHT"

                        # GRF estimate (sum of ground-foot contacts; world frame)
                        F_grf_w = np.zeros(3, dtype=float)
                        if ground_geom_id is not None:
                            try:
                                for ci in range(int(data.ncon)):
                                    c = data.contact[ci]
                                    g1 = int(c.geom1)
                                    g2 = int(c.geom2)
                                    if not (
                                        (g1 == int(ground_geom_id) and int(model.geom_bodyid[g2]) == int(foot_body_id))
                                        or (g2 == int(ground_geom_id) and int(model.geom_bodyid[g1]) == int(foot_body_id))
                                    ):
                                        continue
                                    cf = np.zeros(6, dtype=float)
                                    mujoco.mj_contactForce(model, data, int(ci), cf)
                                    n_w = np.asarray(c.frame[0:3], dtype=float).reshape(3)
                                    t1_w = np.asarray(c.frame[3:6], dtype=float).reshape(3)
                                    t2_w = np.asarray(c.frame[6:9], dtype=float).reshape(3)
                                    R_cw = np.column_stack([n_w, t1_w, t2_w]).astype(float)
                                    f_w = (R_cw @ np.asarray(cf[0:3], dtype=float).reshape(3)).reshape(3)
                                    # Heuristic sign fix: in world Z-up, GRF on robot should have +Fz during stance.
                                    if float(f_w[2]) < 0.0:
                                        f_w = (-f_w).astype(float)
                                    F_grf_w += f_w
                            except Exception:
                                F_grf_w[:] = 0.0

                        F_prop_w = np.asarray(total_force_w, dtype=float).reshape(3)
                        F_tot_w = (F_grf_w + F_prop_w).reshape(3)

                        # === Extra kinematic debug (helps diagnose sign flips) ===
                        try:
                            base_pos_w = np.asarray(data.xpos[base_body_id], dtype=float).reshape(3)
                            foot_pos_w = np.asarray(data.xpos[foot_body_id], dtype=float).reshape(3)
                            foot_rel_w = (foot_pos_w - base_pos_w).reshape(3)
                            foot_b_gt = (np.asarray(R_wb, dtype=float).reshape(3, 3).T @ foot_rel_w.reshape(3)).reshape(3)

                            # World-vertical flight target: foot_des_w = [0,0,-l0] (yaw doesn't matter for vertical)
                            l0_ser = 0.5653  # from hopper_serial.xml geometry (0.0416 + 0.5237)
                            foot_des_w_vert = np.array([0.0, 0.0, -float(l0_ser)], dtype=float)
                            foot_des_b_vert = (np.asarray(R_wb, dtype=float).reshape(3, 3).T @ foot_des_w_vert.reshape(3)).reshape(3)

                            # Joint positions in LCM convention (for serial, with q_sign=1,q_offset=0 these match mj qpos)
                            motor_pos_mj_now = np.array([float(data.qpos[a]) for a in motor_qpos_adr], dtype=float).reshape(3)
                            q_lcm_now = (q_sign * motor_pos_mj_now + q_offset).astype(float).reshape(3)
                        except Exception:
                            foot_b_gt = np.full(3, np.nan, dtype=float)
                            foot_des_b_vert = np.full(3, np.nan, dtype=float)
                            q_lcm_now = np.full(3, np.nan, dtype=float)

                        # Compact HUD text
                        hud_lines = [
                            f"t={sim_t:6.3f}s  {phase}  armed={int(armed)}  rel={t_after_release:4.1f}s",
                            f"v_cmd=[{vx_cmd:+.2f},{vy_cmd:+.2f}]m/s stick=[{stick_x:+.2f},{stick_y:+.2f}]",
                            f"base_p_w={_fmt_vec(np.asarray(data.qpos[0:3], dtype=float).reshape(3), fmt='{:+.3f}')}  base_v_w={_fmt_vec(np.asarray(data.qvel[0:3], dtype=float).reshape(3), fmt='{:+.3f}')}",
                            f"rpy_deg=[{np.rad2deg(rpy[0]):+.1f},{np.rad2deg(rpy[1]):+.1f},{np.rad2deg(rpy[2]):+.1f}]  q=[{q_lcm_now[0]:+.3f},{q_lcm_now[1]:+.3f},{q_lcm_now[2]:+.3f}]",
                            f"foot_b_gt={_fmt_vec(foot_b_gt, fmt='{:+.3f}')}  foot_des_b_vert={_fmt_vec(foot_des_b_vert, fmt='{:+.3f}')}",
                            f"tau_ff_lcm={_fmt_vec(tau_ff_lcm, fmt='{:+.1f}')}",
                            f"tau_apply={_fmt_vec(tau_apply, fmt='{:+.1f}')}",
                            f"pwm_us={_fmt_vec(pwm_apply, fmt='{:.0f}')}",
                            f"thrust6={_fmt_vec(thrusts6, fmt='{:.1f}')}  sum={float(np.sum(thrusts6)):.1f}N",
                            f"F_grf_w={_fmt_vec(F_grf_w, fmt='{:+.1f}')}N",
                            f"F_prop_w={_fmt_vec(F_prop_w, fmt='{:+.1f}')}N",
                            f"F_tot_w={_fmt_vec(F_tot_w, fmt='{:+.1f}')}N",
                        ]
                        frame = _draw_hud_rgb(frame, lines=hud_lines, font_size=int(getattr(args, "hud_font_size", 18)))

                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
                    if gif_writer is not None:
                        gif_writer.append_data(frame)
                    if (mp4_proc is not None) and (mp4_proc.stdin is not None):
                        mp4_proc.stdin.write(frame.tobytes())
                except Exception:
                    pass
                next_frame_sim_t += record_dt

        # Viewer + debug printing
        if viewer is not None:
            try:
                viewer.sync()
            except Exception:
                viewer = None

        phz = float(args.print_hz)
        if phz > 0.0:
            now = float(time.time())
            if (now - last_print_t) >= (1.0 / max(1e-6, phz)):
                last_print_t = now
                # Measured (MuJoCo) leg length (base origin -> foot origin)
                base_pos_w = np.asarray(data.xpos[base_body_id], dtype=float).reshape(3)
                foot_pos_w = np.asarray(data.xpos[foot_body_id], dtype=float).reshape(3)
                leg_len_mj = float(np.linalg.norm(foot_pos_w - base_pos_w))
                # Estimated leg length from published q_lcm (only meaningful for 3RSR delta motor angles).
                if is_3rsr:
                    try:
                        foot_vicon, _ = fk.forward_kinematics(q_lcm)
                        leg_len_est = float(np.linalg.norm(np.asarray(foot_vicon, dtype=float).reshape(3)))
                        q_shift_est = float(leg_len_est - float(args.leg_l0_m))
                    except Exception:
                        leg_len_est = float("nan")
                        q_shift_est = float("nan")
                else:
                    # serial plant: use MuJoCo base-foot distance as "leg length" proxy
                    leg_len_est = float(leg_len_mj)
                    # default serial l0 from hopper_serial.xml geometry (0.0416 + 0.5237)
                    l0_ser = 0.5653
                    q_shift_est = float(leg_len_est - l0_ser)
                print(
                    "[mujoco_lcm_fake_robot] "
                    f"armed={int(armed)} "
                    f"q_lcm={q_lcm.reshape(3)} "
                    f"leg_len_mj={leg_len_mj:.3f} "
                    f"leg_len_est={leg_len_est:.3f} "
                    f"q_shift_est={q_shift_est:+.3f}"
                )

        # Realtime pacing (optional). Deterministic sim doesn't require sleeping.
        if bool(args.realtime):
            dt_wall = float(time.time() - loop_t0)
            if dt_wall < dt:
                time.sleep(dt - dt_wall)
        else:
            time.sleep(0.0)

    if gif_writer is not None:
        try:
            gif_writer.close()
        except Exception:
            pass
    if mp4_proc is not None:
        try:
            if mp4_proc.stdin is not None:
                mp4_proc.stdin.close()
        except Exception:
            pass
        try:
            mp4_proc.wait(timeout=5.0)
        except Exception:
            try:
                mp4_proc.kill()
            except Exception:
                pass
    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


