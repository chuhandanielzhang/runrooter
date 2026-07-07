"""
Runtime-compatible hopping controller: PAC-NMPC outer loop + 500 Hz inner loop.

Mirrors the layering of the real stack (ModeECore -> WBC), with the stance
wrench reference produced by PAC-NMPC instead of HFA:

  outer loop (50 Hz):  PAC-NMPC solve -> u0 = [f_c_w(3), t_arms(3)] + PAC bound
  inner loop (500 Hz):
    - phase FSM from leg length (touchdown l <= l0 - delta, liftoff l >= l0),
      same rule as ModeECore
    - STANCE: tau = J^T (-R_wb^T f_c)   (leg realizes the planned GRF)
    - FLIGHT: Raibert foot placement (kv = 0.16, CASE Table I) via IK + joint PD
    - props: outer-loop thrusts + small 500 Hz attitude-PD differential thrust
      (stands in for the runtime's WBC-QP tracking layer), then thrust -> PWM
      via the measured motor table

Interface to the plant is IDENTICAL to the Jetson stack: consumes SensorData
(q, qd, quat, gyro, acc), produces (tau_ff(3), pwm_us(6)).

State estimation note: this demo reads base position/velocity from the plant's
ground truth (the real stack has its own leg-odometry estimator; re-building it
is orthogonal to the PAC-NMPC story). Attitude/rates come from quat/gyro as on
the real robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import (
    quat_to_R_wb, R_to_rpy_xyz, PROP_ARM_POS_B, PROP_PWM_IDX_PER_ARM, COM_B,
    TAU_LIMITS, TOUCHDOWN_DELTA_M, CONTROL_DT,
)
from .leg import fk_jac, ik, grf_to_tau, LEG_L0_M
from .motor_table import MotorTableModel
from .pac_nmpc import PACNMPC, PACNMPCConfig
from .sim import SensorData


@dataclass
class InnerLoopConfig:
    outer_decimation: int = 10           # 500/10 = 50 Hz outer loop (= rollout dt 0.02)
    kv_raibert: float = 0.16             # CASE Table I
    kp_joint: float = 60.0               # flight roll/pitch joint PD (Nm/rad)
    kd_joint: float = 2.0
    kp_shift: float = 400.0              # flight shift PD (N/m — prismatic force channel)
    kd_shift: float = 20.0
    kp_att_prop: float = 20.0            # 500 Hz residual attitude PD on props (Nm/rad)
    kd_att_prop: float = 4.0
    t_max_each: float = 10.0
    shift_ext_tau: float = 10.0          # flight: gentle leg extension bias (N on shift)
    fxy_lpf_alpha: float = 0.35          # smooth sampled horizontal GRF between solves
    # stance feasibility guard — mirrors the runtime WBC-QP constraints
    mu_cone: float = 0.4                 # |f_xy| <= mu * f_z   (ModeEConfig.mu)
    stance_fz_min: float = 45.0          # ModeEConfig.stance_fz_min / mpc_fz_min
    tau_rp_flight_max: float = 12.0      # clamp flight roll/pitch joint PD (anti-spike)


class PACHoppingController:
    def __init__(self, nmpc: PACNMPC | None = None, inner: InnerLoopConfig | None = None):
        self.nmpc = nmpc or PACNMPC(PACNMPCConfig())
        self.inner = inner or InnerLoopConfig()
        self.motor_table = MotorTableModel.default_from_table()

        # prop moment map (body frame, thrust along +Z), about COM
        r = PROP_ARM_POS_B - COM_B.reshape(1, 3)
        ez = np.array([0.0, 0.0, 1.0])
        self._M_t = np.stack([np.cross(r[i], ez) for i in range(3)], axis=1)  # (3, 3)
        self._M_t_rp_pinv = np.linalg.pinv(self._M_t[0:2, :])                 # roll/pitch rows

        self._stance = False
        self._tick = 0
        self._u0 = np.zeros(6)
        self._u0[3:6] = 0.3
        self._fxy_filt = np.zeros(2)   # smooth sampled horizontal GRF (same idea as runtime mpc_fxy_lpf)
        self._last = {"bound": float("nan"), "j_hat": float("nan"), "viol_hat": float("nan")}

    # ------------------------------------------------------------------ step
    def step(self, sens: SensorData, v_xy_des: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        q = np.asarray(sens.q, dtype=float).reshape(3)
        qd = np.asarray(sens.qd, dtype=float).reshape(3)
        R_wb = quat_to_R_wb(sens.quat)
        rpy = R_to_rpy_xyz(R_wb)
        omega_b = np.asarray(sens.gyro, dtype=float).reshape(3)

        foot_b, _ = fk_jac(q[0], q[1], q[2])
        l = float(np.linalg.norm(foot_b))

        # --- phase FSM (same thresholds as ModeECore) ---
        if not self._stance and l <= LEG_L0_M - TOUCHDOWN_DELTA_M:
            self._stance = True
        elif self._stance and l >= LEG_L0_M:
            self._stance = False

        # --- outer loop: PAC-NMPC at 50 Hz ---
        if self._tick % self.inner.outer_decimation == 0:
            x0 = np.array([
                sens.gt_base_pos_w[0], sens.gt_base_pos_w[1], sens.gt_base_pos_w[2],
                sens.gt_base_vel_w[0], sens.gt_base_vel_w[1], sens.gt_base_vel_w[2],
                rpy[0], rpy[1], omega_b[0], omega_b[1],
            ])
            # measured foot anchor about COM (world) for the current stance
            r_foot0_w = R_wb @ (foot_b - COM_B) if self._stance else None
            res = self.nmpc.solve(x0=x0, v_xy_des=v_xy_des, in_contact=self._stance,
                                  r_foot0_w=r_foot0_w)
            self._u0 = res["u0"]
            a = self.inner.fxy_lpf_alpha
            self._fxy_filt = (1.0 - a) * self._fxy_filt + a * self._u0[0:2]
            self._u0[0:2] = self._fxy_filt
            self._last = {k: res[k] for k in ("bound", "j_hat", "viol_hat")}
        self._tick += 1

        f_c_w = self._u0[0:3].copy()
        t_arms = self._u0[3:6].copy()

        # --- inner loop: leg ---
        tau = np.zeros(3)
        if self._stance:
            # feasibility guard (runtime WBC-QP equivalents): fz floor + friction cone
            f_c_w[2] = max(f_c_w[2], self.inner.stance_fz_min)
            fxy = float(np.hypot(f_c_w[0], f_c_w[1]))
            fxy_max = self.inner.mu_cone * f_c_w[2]
            if fxy > fxy_max:
                f_c_w[0:2] *= fxy_max / fxy
            tau = grf_to_tau(f_c_w, R_wb, q)
        else:
            # Raibert foot placement (body-aligned horizontal frame)
            v_xy = sens.gt_base_vel_w[0:2]
            r_td = self.inner.kv_raibert * v_xy - 0.14 * np.asarray(v_xy_des, dtype=float)
            r_norm = float(np.linalg.norm(r_td))
            r_max = 0.35 * LEG_L0_M
            if r_norm > r_max:
                r_td = r_td * (r_max / r_norm)
            # rotate desired world-frame offset into body frame (yaw only)
            yaw = rpy[2]
            cy, sy = np.cos(yaw), np.sin(yaw)
            r_td_b = np.array([cy * r_td[0] + sy * r_td[1], -sy * r_td[0] + cy * r_td[1]])
            z_b = -np.sqrt(max(LEG_L0_M ** 2 - float(r_td_b @ r_td_b), 1e-6))
            q_des = ik(np.array([r_td_b[0], r_td_b[1], z_b]))
            tau = self.inner.kp_joint * (q_des - q) - self.inner.kd_joint * qd
            m = self.inner.tau_rp_flight_max
            tau[0:2] = np.clip(tau[0:2], -m, m)
            tau[2] = self.inner.kp_shift * (q_des[2] - q[2]) - self.inner.kd_shift * qd[2]
            tau[2] -= self.inner.shift_ext_tau  # bias toward full extension
            self._fxy_filt *= 0.9               # bleed stale stance force during flight
        tau = np.clip(tau, -TAU_LIMITS, TAU_LIMITS)

        # --- inner loop: props (outer thrust + 500 Hz attitude PD residual) ---
        tau_rp_pd = (-self.inner.kp_att_prop * rpy[0:2]
                     - self.inner.kd_att_prop * omega_b[0:2])
        dt_arms = self._M_t_rp_pinv @ tau_rp_pd
        t_cmd = np.clip(t_arms + dt_arms, 0.0, self.inner.t_max_each)

        pwm_us = np.full(6, 1000.0)
        pwm_arms = self.motor_table.pwm_from_thrust(t_cmd)
        for arm_i, pwm_idx in enumerate(PROP_PWM_IDX_PER_ARM):
            pwm_us[pwm_idx] = float(pwm_arms[arm_i])

        info = {
            "stance": self._stance,
            "leg_len": l,
            "rpy": rpy,
            "f_c_cmd": f_c_w,
            "t_arms_cmd": t_cmd,
            **self._last,
        }
        return tau, pwm_us, info
