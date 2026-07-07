"""
Frame / signal conventions — MUST match the Jetson/PC runtime stack.

Everything here is copied verbatim from the running stack so the demo speaks the
exact same "wire language" as the real robot. Sources (READ-ONLY, never edited):

  - robot_runtime/upper_controller_pc/hopper_controller/modee/core.py   (ModeEConfig)
  - hopperHFAcase2026/hopper_controller/mujoco_lcm_fake_robot.py        (sim parity layer)
  - hopperHFAcase2026/model/hopper_serial.xml                           (plant)

Signal semantics (identical to hopper_data_lcmt / hopper_imu_lcmt / hopper_cmd_lcmt
/ motor_pwm_lcmt as emulated by mujoco_lcm_fake_robot.py for the serial plant):

  sensor side (what the controller receives):
    q   (3,)  joint positions  [roll, pitch, shift],  q_lcm = Q_SIGN*q_mj + Q_OFFSET
    qd  (3,)  joint velocities,                        qd_lcm = Q_SIGN*qd_mj
    quat(4,)  base attitude wxyz (body->world)
    gyro(3,)  body-frame angular velocity  (MuJoCo freejoint qvel[3:6] is body-local)
    acc (3,)  -(specific force) in body frame:  acc_b = R_wb^T (g_w - a_w)
              -> at rest [0,0,-9.81], in free fall [0,0,0]

  command side (what the controller sends):
    tau_ff (3,) joint feed-forward torques; plant applies Q_SIGN*tau_ff
    pwm_us (6,) propeller ESC PWM in microseconds (1000..2000), mapped to thrust
                through the measured 1950KV motor table.

Frames (serial plant):
    world: +Z up.  body/base: +X forward, +Y left, +Z up.
    Propeller thrust acts along body +Z at each arm tip.
"""

from __future__ import annotations

import numpy as np

# --- LCM joint mapping for the SERIAL plant (see record_modee_serial_*.sh: --q-sign 1 --q-offset 0)
Q_SIGN: float = 1.0
Q_OFFSET: float = 0.0

# --- physical parameters (ModeEConfig in core.py) ---
MASS_KG: float = 3.75
GRAVITY: float = 9.81
COM_B = np.array([-2.79376456e-04, 1.68299070e-06, -5.72937376e-02])
# whole-body inertia diag about COM, computed from this very MJCF (core.py:223)
I_BODY_DIAG = np.array([0.0716072799, 0.0716088488, 0.0579831725])

# --- serial leg geometry (core.py serial_* + hopper_serial.xml) ---
SERIAL_HIP_Z_OFF_M: float = 0.0416
SERIAL_FOOT_Z_M: float = 0.5237
LEG_L0_M: float = SERIAL_HIP_Z_OFF_M + SERIAL_FOOT_Z_M  # 0.5653, matches fake-robot HUD

# --- propeller geometry: EXACTLY the fake-robot layout (mujoco_lcm_fake_robot.py:606) ---
# order: [RED, GREEN, BLUE]; GREEN points forward (+X)
PROP_ARM_LEN_M: float = 0.569451
PROP_ARM_POS_B = np.array(
    [
        [-0.5 * PROP_ARM_LEN_M, +np.sqrt(3.0) * 0.5 * PROP_ARM_LEN_M, 0.0],  # RED
        [+1.0 * PROP_ARM_LEN_M, 0.0, 0.0],                                   # GREEN
        [-0.5 * PROP_ARM_LEN_M, -np.sqrt(3.0) * 0.5 * PROP_ARM_LEN_M, 0.0],  # BLUE
    ]
)
# PWM channel index per arm (fake robot default "2;1;3"): RED<-pwm[2], GREEN<-pwm[1], BLUE<-pwm[3]
PROP_PWM_IDX_PER_ARM = (2, 1, 3)

# --- control timing (matches the runtime stack) ---
CONTROL_DT: float = 0.002       # 500 Hz controller tick
MJ_TIMESTEP: float = 0.001      # hopper_serial.xml <option timestep>

# --- contact FSM thresholds (ModeEConfig) ---
TOUCHDOWN_DELTA_M: float = 0.020

# --- actuator limits for the SERIAL plant (hopper_serial.xml ctrlrange;
#     runtime launches run_modee.py with --tau-out-max 2500 for this plant) ---
# roll/pitch are hinge torques (Nm); shift is a prismatic FORCE (N).
TAU_LIMITS = np.array([27.0, 27.0, 2500.0])
MU_FRICTION: float = 0.4


def quat_to_R_wb(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) -> rotation matrix R_wb (body->world). Same as runtime stack."""
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    q = q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_rpy_xyz(R: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (XYZ intrinsic) from R_wb. Same as runtime stack."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(max(1e-12, R[2, 1] ** 2 + R[2, 2] ** 2))))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw])
