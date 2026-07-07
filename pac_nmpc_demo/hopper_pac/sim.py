"""
MuJoCo plant with the SAME interface `u` and coordinates as the Jetson stack.

This is an in-process re-implementation of the IO semantics of
`hopperHFAcase2026/hopper_controller/mujoco_lcm_fake_robot.py` (serial plant path):

  step(tau_ff, pwm_us) -> SensorData(q, qd, quat, gyro, acc)

  - tau_ff (3,): joint torques in LCM convention [roll, pitch, shift];
                 plant applies Q_SIGN * tau_ff to actuators (serial: Q_SIGN=1).
  - pwm_us (6,): ESC PWM (us); thrust via the measured motor table, applied along
                 body +Z at each arm tip as an external wrench on base_link.
  - q/qd:  LCM convention (q_lcm = Q_SIGN*q_mj + Q_OFFSET).
  - quat:  wxyz body->world.
  - gyro:  body-frame angular velocity.
  - acc:   -(specific force) in body: rest [0,0,-9.81], free fall [0,0,0].

Uncertainty injection (for PAC experiments) is done HERE, in the plant — the
controller never sees the true parameters:
  - ground friction mu
  - ground height offset (early/late touchdown -> multi-modal contact timing)
  - payload mass added to the base
  - per-motor thrust scale
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np
import mujoco

from .conventions import Q_SIGN, Q_OFFSET, PROP_ARM_POS_B, PROP_PWM_IDX_PER_ARM, MJ_TIMESTEP
from .motor_table import MotorTableModel

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "hopper_serial.xml")

_JOINT_NAMES = ["Leg_Joint_Roll", "Leg_Joint_Pitch", "Leg_Joint_Shift"]
_ACT_NAMES = ["roll_motor", "pitch_motor", "shift_motor"]


@dataclass
class SensorData:
    q: np.ndarray
    qd: np.ndarray
    quat: np.ndarray   # wxyz
    gyro: np.ndarray   # body frame
    acc: np.ndarray    # -(specific force), body frame
    # ground-truth extras (for logging/metrics only; controller must not use)
    gt_base_pos_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gt_base_vel_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gt_contact: bool = False


@dataclass
class PlantUncertainty:
    """True-plant parameters (unknown to the controller)."""
    friction_mu: float = 0.3          # MJCF default
    ground_z_offset_m: float = 0.0    # + raises the ground (early touchdown)
    payload_mass_kg: float = 0.0      # extra mass rigidly added to base
    thrust_scale: np.ndarray = field(default_factory=lambda: np.ones(6))
    gyro_noise_std: float = 0.0
    acc_noise_std: float = 0.0


class HopperMujocoPlant:
    def __init__(
        self,
        model_path: str = _MODEL_PATH,
        uncertainty: PlantUncertainty | None = None,
        seed: int = 0,
        init_base_z: float = 0.65,
    ):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(model_path))
        self.data = mujoco.MjData(self.model)
        self.unc = uncertainty or PlantUncertainty()
        self.rng = np.random.default_rng(seed)

        self._joint_ids = [int(self.model.joint(n).id) for n in _JOINT_NAMES]
        self._qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self._joint_ids]
        self._qvel_adr = [int(self.model.jnt_dofadr[j]) for j in self._joint_ids]
        self._act_ids = [int(self.model.actuator(n).id) for n in _ACT_NAMES]
        self._base_bid = int(self.model.body("base_link").id)
        self._foot_bid = int(self.model.body("Foot_Link").id)
        self._ground_gid = int(self.model.geom("ground").id)

        # --- apply plant-side (true) uncertainty ---
        self.model.geom_friction[self._ground_gid, 0] = float(self.unc.friction_mu)
        if abs(self.unc.ground_z_offset_m) > 1e-12:
            self.model.geom_pos[self._ground_gid, 2] += float(self.unc.ground_z_offset_m)
        if self.unc.payload_mass_kg > 1e-9:
            self.model.body_mass[self._base_bid] += float(self.unc.payload_mass_kg)

        self.motor_table = MotorTableModel.default_from_table()

        # prop mount points per PWM channel (body frame), fake-robot mapping "2;1;3"
        self._pwm_pos_b = np.zeros((6, 3))
        for arm_i, pwm_idx in enumerate(PROP_PWM_IDX_PER_ARM):
            self._pwm_pos_b[pwm_idx] = PROP_ARM_POS_B[arm_i]
        self._pwm_active = np.zeros(6, dtype=bool)
        self._pwm_active[list(PROP_PWM_IDX_PER_ARM)] = True

        self.reset(init_base_z=init_base_z)

    # ------------------------------------------------------------------ reset
    def reset(self, init_base_z: float = 0.65, rpy0: np.ndarray | None = None) -> SensorData:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = float(init_base_z)
        if rpy0 is not None:
            r, p, y = [float(v) for v in np.asarray(rpy0).reshape(3)]
            hr, hp, hy = 0.5 * r, 0.5 * p, 0.5 * y
            cr, sr = np.cos(hr), np.sin(hr)
            cp, sp = np.cos(hp), np.sin(hp)
            cy, sy = np.cos(hy), np.sin(hy)
            self.data.qpos[3:7] = [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr,
                sy * cp * cr - cy * sp * sr,
            ]
        mujoco.mj_forward(self.model, self.data)
        return self._read_sensors()

    # ------------------------------------------------------------------ step
    def step(self, tau_ff: np.ndarray, pwm_us: np.ndarray, n_substeps: int = 2,
             extra_force_w: np.ndarray | None = None) -> SensorData:
        """Advance the plant by one 500 Hz controller tick (2 x 1 ms physics steps).

        extra_force_w: optional world-frame disturbance force on the base (push tests).
        """
        tau_mj = Q_SIGN * np.asarray(tau_ff, dtype=float).reshape(3)
        thrusts6 = self.motor_table.thrust_from_pwm(np.asarray(pwm_us, dtype=float).reshape(6))
        thrusts6 = thrusts6 * np.asarray(self.unc.thrust_scale, dtype=float).reshape(6)

        for _ in range(int(n_substeps)):
            for j, aid in enumerate(self._act_ids):
                self.data.ctrl[aid] = float(tau_mj[j])

            # propeller wrench on base (body +Z thrust at each arm tip)
            quat = np.asarray(self.data.qpos[3:7], dtype=float)
            R_wb = _quat_to_R(quat)
            F_w = np.zeros(3)
            T_w = np.zeros(3)
            ez_b = np.array([0.0, 0.0, 1.0])
            for idx in range(6):
                if not self._pwm_active[idx] or thrusts6[idx] <= 0.0:
                    continue
                f_b = ez_b * float(thrusts6[idx])
                tau_b = np.cross(self._pwm_pos_b[idx], f_b)
                F_w += R_wb @ f_b
                T_w += R_wb @ tau_b
            if extra_force_w is not None:
                F_w = F_w + np.asarray(extra_force_w, dtype=float).reshape(3)
            self.data.xfrc_applied[self._base_bid, 0:3] = F_w
            self.data.xfrc_applied[self._base_bid, 3:6] = T_w

            mujoco.mj_step(self.model, self.data)

        return self._read_sensors()

    def apply_push(self, force_w: np.ndarray, duration_s: float, tau_ff: np.ndarray, pwm_us: np.ndarray):
        """Apply an impulsive world-frame push on the base for duration_s (helper for demos)."""
        n = max(1, int(round(duration_s / (2 * MJ_TIMESTEP))))
        out = None
        for _ in range(n):
            self.data.xfrc_applied[self._base_bid, 0:3] += np.asarray(force_w, dtype=float)
            out = self.step(tau_ff, pwm_us)
        return out

    # ------------------------------------------------------------------ sensors
    def _read_sensors(self) -> SensorData:
        q_mj = np.array([self.data.qpos[a] for a in self._qpos_adr])
        qd_mj = np.array([self.data.qvel[a] for a in self._qvel_adr])
        q_lcm = Q_SIGN * q_mj + Q_OFFSET
        qd_lcm = Q_SIGN * qd_mj

        quat = np.asarray(self.data.qpos[3:7], dtype=float).copy()
        R_wb = _quat_to_R(quat)
        gyro_b = np.asarray(self.data.qvel[3:6], dtype=float).copy()  # body-local

        g_w = np.array([0.0, 0.0, -9.81])
        a_w = np.asarray(self.data.qacc[0:3], dtype=float)
        acc_b = -(R_wb.T @ (a_w - g_w))

        if self.unc.gyro_noise_std > 0:
            gyro_b = gyro_b + self.rng.normal(0.0, self.unc.gyro_noise_std, 3)
        if self.unc.acc_noise_std > 0:
            acc_b = acc_b + self.rng.normal(0.0, self.unc.acc_noise_std, 3)

        contact = False
        for ci in range(int(self.data.ncon)):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])
            if (g1 == self._ground_gid and b2 == self._foot_bid) or (
                g2 == self._ground_gid and b1 == self._foot_bid
            ):
                contact = True
                break

        return SensorData(
            q=q_lcm,
            qd=qd_lcm,
            quat=quat,
            gyro=gyro_b,
            acc=acc_b,
            gt_base_pos_w=np.asarray(self.data.qpos[0:3], dtype=float).copy(),
            gt_base_vel_w=np.asarray(self.data.qvel[0:3], dtype=float).copy(),
            gt_contact=contact,
        )


def _quat_to_R(q_wxyz: np.ndarray) -> np.ndarray:
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
