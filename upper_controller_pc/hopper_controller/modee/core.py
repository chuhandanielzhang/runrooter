from __future__ import annotations

"""
ModeE core controller (real-robot version)
=========================================

This is the "modee" architecture used in MuJoCo:
  - Event-based hop phases (TD/COMP/PUSH/FLIGHT/APEX)
  - Stance wrench reference from PD + impulse shaping (no MPC)
  - Closed-form SRB leg forces + lstsq prop thrust allocation (no WBC-QP)
  - All control uses IMU + encoders only (no MuJoCo ground truth)

This file is MuJoCo-free and is meant to run on the real robot via LCM.
"""

from dataclasses import dataclass
import math
import os
import time as _time
import numpy as np

# NOTE: hopper_controller is not a Python package by default (no __init__.py).
# Keep imports relative to the folder that runs the controller (same style as Hopper4.py).
from forward_kinematics import ForwardKinematics, InverseJacobian

from modee.controllers.motor_utils import MotorTableModel


def _leg_native_to_imu_body(p: np.ndarray) -> np.ndarray:
    """Leg FK frame == IMU frame (FRD: +X fwd, +Y right, +Z down). No conversion."""
    return np.asarray(p, dtype=float).reshape(3).copy()


def _imu_body_to_leg_native(p: np.ndarray) -> np.ndarray:
    """IMU frame == leg FK frame (FRD). No conversion."""
    return np.asarray(p, dtype=float).reshape(3).copy()


def _skew(r: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in np.asarray(r, dtype=float).reshape(3)]
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _vee_so3(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=float).reshape(3, 3)
    return np.array([E[2, 1], E[0, 2], E[1, 0]], dtype=float)


def _Rz(yaw: float) -> np.ndarray:
    c = float(math.cos(float(yaw)))
    s = float(math.sin(float(yaw)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = (q / n).astype(float)
    # keep w>=0 (avoid discontinuities)
    if float(q[0]) < 0.0:
        q = (-q).astype(float)
    return q


def _quat_to_R_wb(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix R_wb: body -> world. Quaternion is wxyz."""
    q = _quat_normalize_wxyz(q_wxyz)
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
    """Roll/pitch/yaw from R_wb (body FRD -> world NED, both +Z down).

    Standard aerospace ZYX (yaw-pitch-roll) extraction:
      right tilt (right side down) => roll > 0
      nose down                    => pitch < 0
      level body => R ~ Rz(yaw), roll = pitch = 0.
    This helper is retained for logging/debug only; control roll/pitch
    comes from e_R (built directly from R_wb) and hopper_imu_lcmt.rpy.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    roll = float(math.atan2(R[2, 1], R[2, 2]))
    pitch = float(-math.asin(_clipf(R[2, 0], -1.0, 1.0)))
    yaw = float(math.atan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


def _cross3(a, b) -> np.ndarray:
    """Fast 3-vector cross product.

    np.cross costs ~90us/call on the Jetson (generic dispatch + moveaxis);
    with ~10 calls per control step that alone breaks the 500Hz budget.
    """
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array(
        [a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], dtype=float
    )


def _inv3(A: np.ndarray) -> np.ndarray | None:
    """Closed-form 3x3 inverse (adjugate). Returns None when near-singular.

    np.linalg.solve costs ~30us/call on the Jetson (LAPACK dispatch overhead
    dominates at this size); the closed form is ~5us. Callers fall back to
    numpy when None is returned.
    """
    a, b, c = float(A[0, 0]), float(A[0, 1]), float(A[0, 2])
    d, e, f = float(A[1, 0]), float(A[1, 1]), float(A[1, 2])
    g, h, i = float(A[2, 0]), float(A[2, 1]), float(A[2, 2])
    co00 = e * i - f * h
    co01 = f * g - d * i
    co02 = d * h - e * g
    det = a * co00 + b * co01 + c * co02
    scale = max(abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), 1e-300)
    # second term guards underflow of scale**3 (e.g. zero matrix)
    if abs(det) < max(1e-12 * scale * scale * scale, 1e-300):
        return None
    inv_det = 1.0 / det
    return np.array(
        [
            [co00 * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
            [co01 * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
            [co02 * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
        ],
        dtype=float,
    )


def _lstsq_minnorm(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-norm least-squares solve, same result as np.linalg.lstsq for
    full-rank wide/square A but ~40x faster (lstsq runs a full SVD; ~80us/call
    on the Jetson and it's called twice per control step)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(A.shape[0])
    AAt = A @ A.T
    # tiny Tikhonov floor keeps this safe when A loses rank (e.g. rz ~ 0)
    lam = 1e-12 * max(1.0, float(np.trace(AAt)))
    if AAt.shape == (2, 2):
        # closed-form 2x2 solve (avoids LAPACK dispatch overhead per step)
        a = float(AAt[0, 0]) + lam
        bb = float(AAt[0, 1])
        cc = float(AAt[1, 0])
        d = float(AAt[1, 1]) + lam
        det = a * d - bb * cc
        if abs(det) > 1e-300:
            y = np.array(
                [
                    (d * float(b[0]) - bb * float(b[1])) / det,
                    (a * float(b[1]) - cc * float(b[0])) / det,
                ],
                dtype=float,
            )
            return A.T @ y
    try:
        y = np.linalg.solve(AAt + lam * np.eye(A.shape[0]), b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    return A.T @ y


def _tau_limit_proportional(tau: np.ndarray, tau_max: np.ndarray) -> tuple[np.ndarray, float]:
    """Direction-preserving torque limiting.

    If any |tau[i]| exceeds tau_max[i], the WHOLE vector is scaled by the single
    factor min_i(tau_max[i]/|tau[i]|) so the torque (and thus the foot-force)
    DIRECTION is preserved -- never clip axes independently.
    Returns (tau_scaled, scale).
    """
    tau = np.asarray(tau, dtype=float).reshape(3)
    tau_max = np.asarray(tau_max, dtype=float).reshape(3)
    scale = 1.0
    for i in range(3):
        ai = abs(float(tau[i]))
        if ai > 1e-9:
            si = float(tau_max[i]) / ai
            if si < scale:
                scale = si
    if scale >= 1.0:
        return tau, 1.0
    return (tau * float(scale)).astype(float), float(scale)


def _minimum_norm_side_force(
    r_foot_b: np.ndarray, tau_body_des: np.ndarray
) -> np.ndarray:
    """Minimum-norm side force for the current Mode1 force signs.

    Current ``f_b`` is the robot-on-ground force and body reaction moment is
    ``-r x f``.  Therefore ``f_side=(r x tau_des)/|r|^2`` realizes
    ``-r x f_side=tau_des`` after the leg-axis torque component is removed.
    """
    r = np.asarray(r_foot_b, dtype=float).reshape(3)
    tau = np.asarray(tau_body_des, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    if r2 <= 1e-12:
        return np.zeros(3, dtype=float)
    return (_cross3(r, tau) / r2).astype(float)


def _clipf(x, lo: float, hi: float) -> float:
    """Fast scalar clip (np.clip on a python float costs ~35us on the Jetson)."""
    x = float(x)
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return x


def _quat_from_omega_dt(omega_b: np.ndarray, dt: float) -> np.ndarray:
    w = np.asarray(omega_b, dtype=float).reshape(3)
    th = float(np.linalg.norm(w) * float(dt))
    if th < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = (w / np.linalg.norm(w)).astype(float)
    half = 0.5 * th
    return _quat_normalize_wxyz(np.array([math.cos(half), *(math.sin(half) * axis)], dtype=float))


def _quat_mul(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in np.asarray(q1_wxyz, dtype=float).reshape(4)]
    w2, x2, y2, z2 = [float(v) for v in np.asarray(q2_wxyz, dtype=float).reshape(4)]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _up_dir_w(gravity: float) -> np.ndarray:
    """Unit UP vector in WORLD (+Z DOWN frame): opposite to gravity = [0,0,-1]."""
    up_w = np.array([0.0, 0.0, -float(gravity)], dtype=float)
    return (up_w / max(1e-12, float(np.linalg.norm(up_w)))).astype(float)


class SimpleIMUAttitudeEstimator:
    """
    MATLAB com_filter-style attitude estimator (2026-07-11), COORDINATE FRAME
    UNCHANGED (body FRD, world +Z DOWN):
      - EVERY tick: pure gyro integration (q <- q (x) dq(omega,dt)).
      - STANCE MID-WINDOW ONLY: nudge the tilt toward the specific-force
        direction (the steady push points ~ up along the leg), by a fraction
        `accel_weight` of the full correction -- NO |a|~g gate (direction only,
        exactly like MATLAB, which normalizes imu_accel and uses its direction
        even while |a| >> g during the push).
      - FLIGHT: pure gyro (free-fall specific force is not "up", so no accel).
      - reset(): next update re-initializes the tilt straight from the accel
        (MATLAB orient_reset), yaw = 0.
    Yaw is never corrected here (no mag / no accel yaw info); the caller fuses
    yaw separately (lidar).
    """

    def __init__(self, kp_acc: float = 0.6, acc_g_min: float = 0.90, acc_g_max: float = 1.10):
        self.kp_acc = float(kp_acc)  # legacy (unused in MATLAB mode)
        self.acc_g_min = float(acc_g_min)
        self.acc_g_max = float(acc_g_max)
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # wxyz, body->world
        self._need_init = True

    def reset(self) -> None:
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._need_init = True

    @staticmethod
    def _tilt_from_accel(acc_b: np.ndarray) -> np.ndarray:
        """Level tilt-only quaternion whose body UP aligns with the specific
        force direction (yaw = 0). q=I gives up_b=[0,0,-1]; rotate it onto a_b."""
        acc_b = np.asarray(acc_b, dtype=float).reshape(3)
        a_n = float(np.linalg.norm(acc_b))
        if a_n <= 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        a_b = (acc_b / a_n).astype(float)
        up0 = np.array([0.0, 0.0, -1.0], dtype=float)  # up_b at q = I
        # We want body->world R with R @ a_b = up_w, i.e. R rotates the measured
        # body-up onto world-up. axis = a_b x up0 (NOT up0 x a_b).
        axis = _cross3(a_b, up0)
        axis_n = float(np.linalg.norm(axis))
        cos_a = float(_clipf(float(np.dot(a_b, up0)), -1.0, 1.0))
        if axis_n < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        ang = float(math.acos(cos_a))
        return _quat_from_omega_dt((axis / axis_n) * ang, 1.0)

    def update(
        self,
        *,
        omega_b: np.ndarray,
        acc_b: np.ndarray,
        dt: float,
        correct: bool = False,
        accel_weight: float = 0.0,
    ) -> np.ndarray:
        dt = float(dt)
        omega_b = np.asarray(omega_b, dtype=float).reshape(3)
        acc_b = np.asarray(acc_b, dtype=float).reshape(3)

        # MATLAB orient_reset: seed the tilt straight from the accelerometer.
        if bool(self._need_init):
            self._q = _quat_normalize_wxyz(self._tilt_from_accel(acc_b))
            self._need_init = False

        # gyro integration (every tick)
        dq = _quat_from_omega_dt(omega_b, dt)
        self._q = _quat_mul(self._q, dq)
        self._q = _quat_normalize_wxyz(self._q)

        # stance mid-window accel tilt correction (direction only, no |a| gate)
        if bool(correct) and float(accel_weight) > 0.0:
            a_norm = float(np.linalg.norm(acc_b))
            if a_norm > 1e-9:
                a_b = (acc_b / a_norm).astype(float)
                R_wb = _quat_to_R_wb(self._q)
                up_b = (R_wb.T @ _up_dir_w(9.81)).reshape(3)
                # Body-frame nudge q<-q*dq moves up_b by dR^T, so to rotate the
                # ESTIMATED up_b toward the MEASURED a_b we rotate about
                # e = a_b x up_b (NOT up_b x a_b -- that sign diverges).
                e = _cross3(a_b, up_b)
                e_n = float(np.linalg.norm(e))
                if e_n > 1e-9:
                    # MATLAB: correcting angle = accel_weight * asin(|cross|),
                    # rotate about the (body-frame) correction axis. NO dt here.
                    ang = float(accel_weight) * float(math.asin(min(1.0, e_n)))
                    dq2 = _quat_from_omega_dt((e / e_n) * ang, 1.0)
                    self._q = _quat_mul(self._q, dq2)
                    self._q = _quat_normalize_wxyz(self._q)

        return self._q.copy()


@dataclass
class ModeEConfig:
    """ModeE controller configuration.

    Parameters changed during normal experiments are grouped first. Robot
    geometry and optional features are kept below as advanced configuration.
    """

    # ======================================================================
    # ★ HOP 日常调参（高度 / 能量律 / 落点 / 姿态 / 速度收敛 —— 都在这里）
    # ======================================================================
    # ---- 跳高 ----
    # 目标 apex（相对 l0）。pogox 下这是唯一高度旋钮；k_v / r* / Raibert
    # 中性点都从它推导。
    hop_height_m: float = 0.05
    # 可用行程 X（设计弹簧 k_v 的分母）。一般别动。
    leg_stroke_max_m: float = 0.090
    # 站立能量律：当前用 "pogox"。可选 "fbslip" | "nrc" | "mode1"。
    stance_energy_law: str = "pogox"
    # ---- pogox 能量泵 ----
    pogox_kr: float = 250.0                 # 泵增益：大=一两跳到位，力更大
    pogox_vz_lag_s: float = 0.05            # 泵用 vz 的模型超前补偿
    pogox_seed_vz_mps: float = 0.05         # 静止起振速度门
    pogox_seed_weight_frac: float = 0.4     # 静止种子支撑重量比
    pogox_min_spring_frac: float = 0.5      # 负泵最多把弹簧削到这个比例
    pogox_lo_taper_m: float = 0.015         # 离地前最后这段力→0
    # ---- 泵平滑：只靠模型输入滤波，不硬切泵输出 ----
    # 泵专用 vz 滤波时间常数（只影响 _px_vz_ax）。平滑来自输入，
    # 这样泵项是物理律的直接结果，hop_height 旋钮才能线性兑现。
    pogox_vz_ax_lpf_tau_s: float = 0.03
    # 泵项斜率限幅 (N/s)：0 = 关闭。硬切会把能量注入掐死在自设
    # 墙上（log 205009: 64% tick 顶着 ±4 N），破坏单一旋钮。保留
    # 参数只作诊断开关，默认不开。
    pogox_pump_slew_nps: float = 0.0
    # ---- 每跳 apex 回授（冷启动 trim，避免前几跳必然偏低）----
    nrc_apex_trim_init: float = 1.4
    # ---- Raibert 落点（水平速度收敛，靠落脚，不靠倾斜）----
    flight_kv: float = 0.13                 # 速度误差反馈 [m/(m/s)]
    flight_stepper_lim_m: float = 0.18      # |foot_xy| 上限
    # ---- 飞行姿态（桨调平；与速度收敛独立）----
    flight_kR: float = 40.0
    flight_kW: float = 8.0
    flight_tau_rp_max: float = 100.0
    # ---- ★ 姿态收敛速度开关（飞行水平速度 → fl_tilt）----
    # False = 只调平，不倾斜刹水平速度（推荐先关着调跳高）
    # True  = 开 fl_tilt，用桨矢量刹 v_xy
    flight_vel_ctrl_enable: bool = False
    flight_vel_kv: float = 0.5              # 开了才生效：阻尼带宽 [1/s]
    flight_vel_tilt_max_deg: float = 10.0
    flight_vel_tilt_slew_dps: float = 180.0
    flight_level_settle_s: float = 0.03
    # ---- 桨怠速（姿态差动的底座）----
    # Idle PWM 1200: T≈2.7 N → ratio≈0.0491 @ m=5.61
    prop_base_thrust_ratio: float = 0.0491

    # ======================================================================
    # PRIMARY TUNING
    # ======================================================================

    # Mode1: world-z impedance + push energy compensation, stance attitude PD
    # with closed-form leg force allocation, propeller residual attitude
    # overlay, classic Raibert placement, and flight propeller attitude PD.
    dt: float = 0.002
    mode_1d: bool = False

    # ==================== STANCE Z: impedance + push spring ===============
    # 2026-07-19 rollback (per user): the adaptive-stiffness compression
    # spring, its k ceiling, and the command-force LPF were removed as
    # unphysical patches -- the auto-sized k (3300..8000 N/m) excited the
    # ~33 Hz leg mode through the sensing/actuation delay, which the legacy
    # fixed-gain impedance never did. What remains:
    #   COMPRESSION: the original world-height impedance (physical,
    #     delay-stable):  f = kz*(l0 + hop_height_m - h_com) - bz*vz_up.
    #     The compression depth is whatever m, v_td and kz give.
    #   PUSH (world-Z gated, latched): one spring re-size at the bottom so
    #     the work released from depth x0 back to l0 leaves the body at the
    #     takeoff speed for hop_height_m plus the learned per-hop loss:
    #       0.5*k_push*x0^2 = 0.5*m*v_to^2 + m*g_st*x0 + E_loss,
    #       v_to = sqrt(2*g_up*hop_height_m).
    #     Force blends over stance_push_blend_tau_s and is spring-shaped:
    #     max at the bottom, zero at l0 (replaces the old chattering
    #     energy pump).
    #   DISCRETE layer (hop to hop): at every touchdown the measured apex
    #   of the hop that just ended updates E_loss by
    #   gamma*m*g*(hop_height_m - h_apex) (mode1_apex_adapt_gamma). Ideal
    #   apex-error dynamics: e(n+1) = (1-gamma)*e(n) -- a 1-D contraction,
    #   deadbeat at gamma = 1 (Koditschek-Buehler / Saranli energy
    #   regulation).
    use_energy_compensation: bool = True
    # 2026-08-02 (log 112743): 0.07 -> 0.12.  A 7 cm target sits at the
    # ground-chatter floor (flight ~0.24 s; with the trim floor the
    # effective target could fall to 2.8 cm -> vz_lo ~0.7 m/s, barely
    # separates).  The robot demonstrated clean 0.13 m hops; 0.12 gives
    # ~0.31 s flights -- clean flight-time apex, TD detection, and the
    # trim update always inside its 0.12 s window.
    # 2026-08-03: 0.12 -> 0.07 -> 0.05.  Real apex was running well
    # above the knob (flight Fz float + noisy stance vz pumping); lower
    # the setpoint while the estimator/Fz caps catch up.
    # PUSH gate on WORLD-Z upward velocity (m/s) + consecutive-tick confirm.
    stance_push_vz_mps: float = 0.05
    stance_push_confirm_steps: int = 3
    # First-order blend of the push spring force at the latch (s).
    stance_push_blend_tau_s: float = 0.01
    # Sensor LPF on the world-Z velocity used by the PUSH gate AND the
    # NRC pump state x2 (leg vibration puts +-2 m/s spikes on the
    # kinematic vz estimate).  2026-08-02: 15 -> 8 ms -- the NRC pump
    # over-injects in proportion to kR * (estimator lag); halving the
    # LPF share of that lag works WITH the kR=150 + extension-fade +
    # apex-trim fix (log 073212 read r/r*=0.82 at push end while the
    # true vz was already 1.32 vs the 1.166 m/s target).
    # 2026-08-02 later (log 115132): 8 -> 20 ms (~8 Hz).  f_ref_w2 was
    # jumping +-20-47 N PER TICK (std 20.6 N) because the pump multiplies
    # the raw-ish vz (which flips +-0.2-0.3 m/s tick-to-tick in stance)
    # by ~2*m*kR*|r-r*|.  20 ms still lags the ~0.25 s stance by <10%
    # of the stroke; the leftover per-hop apex bias is what the trim
    # return map exists for.
    stance_vz_lpf_tau_s: float = 0.02
    # Earliest PUSH latch after touchdown (s). Real bottoms arrive >= 30 ms
    # after TD at these drop heights; chatter stances latched PUSH at 16 ms.
    stance_push_min_stance_s: float = 0.03

    # Leg length, contact phases, and vertical stance force.
    leg_l0_m: float = 0.461
    # Original world-height impedance gains (proven delay-stable; the ~33 Hz
    # leg mode only rang once stiffness went well above this level).
    stance_kp_z: float = 1200.0
    stance_kd_z: float = 5.0
    stance_fz_min: float = 0.0
    # Per-sample cap on the commanded vertical force; the downstream
    # joint-torque rescale remains the hardware protection.
    stance_fz_max: float = 500.0
    hopper4_td_threshold_m: float = 0.02
    hopper4_lo_threshold_m: float = 0.0
    # Minimum dwell (ticks) in a phase before TD/LO can fire again. 23:14 log:
    # with 1 tick (2 ms) a hard landing produced 16-28 ms fake stance/flight
    # chatter cycles, each firing a freshly-sized 200-500 N spring. 10 ticks
    # = 20 ms; real stances/flights here are >= 100/200 ms.
    # 2026-08-03: 10 -> 50 (100 ms). Log 011106: the swing leg retracted
    # past -2 cm within ~30 ms of liftoff while still ASCENDING, so TD
    # re-fired mid-air. Real flights at 0.07 m apex last >= 160 ms.
    hopper4_phase_min_steps: int = 50
    # TD additionally requires the body to be descending (or nearly
    # stopped): vz_up <= this bound (UP-POSITIVE, i.e. -v_hat_z).
    # Blocks mid-air stance re-entry from swing-leg retraction;
    # generous enough for stand-start (vz ~ 0).
    hopper4_td_vz_max_mps: float = 0.15
    # Encoder bypass of the vz gate (log 200745 hop 7): world vz drifted
    # "ascending" during a real landing and the gate above rejected TD for
    # 180 ms of q < -2 cm.  If the leg-length rate (joint encoders, no world
    # estimate involved) shows the leg compressing faster than this while
    # already past the -2 cm depth, accept TD regardless of world vz.
    # Real landing read qd_shift = -0.82 m/s at the -2 cm crossing; flight
    # qd noise is ~0.11 m/s std, so 0.05 fires immediately on contact and
    # single-tick noise cannot fake the 2 cm depth requirement.
    hopper4_td_qd_comp_mps: float = 0.05

    # Stance attitude PD (one gain for roll and pitch).
    # 2026-08-02: bumped ~1.4x for stronger balance authority.
    # 2026-08-03 (log 000102): 70/1.5/20 -> 40/1.0/6.  tau_des_w1 hit
    # 8.7 Nm alone and with f_ref=100 N pinned tau1 at the 9 Nm wall for
    # ~30% of every stance.  Soften the PD and hard-cap the attitude
    # demand so Fz + att stay inside the motor budget.
    stance_kpp: float = 40.0
    stance_kpd: float = 1.0
    stance_tau_rp_max: float = 6.0
    # LEG SHARE of the stance attitude torque.  0 = leg targets the FULL
    # stance PD demand (never reduced for the props).
    # HISTORY: 2026-08-02 HFA split the demand (leg capped, props took
    # residual) and fell (log 230314) -- abandoned.  2026-08-03: true
    # DECOUPLE -- leg keeps the full stance_kpp/kpd demand AND stance
    # props independently track the same R_des with flight_kR/kW (same
    # gains as flight).  Additive, not a residual split; neither channel
    # steals authority from the other.
    stance_leg_att_tau_max_nm: float = 0.0
    stance_mu: float = 0.0
    stance_fxy_max: float = 0.0

    # MATLAB/SLX EMA applied to the CAN-reported qd.
    # lambda is the weight of the previous estimate; 0 leaves raw CAN qd.
    qd_ema_lambda: float = 0.4

    # Flight placement (Raibert return map) / swing PD.
    # 2026-08-02 restructure: the placement law is now the textbook
    # two-term Raibert law
    #     p_f = (T_st/2) * v_hat + flight_kv * (v_hat - v_des)
    # where the NEUTRAL POINT coefficient T_st/2 = (pi/2)*sqrt(m/k) is
    # computed LIVE from the stance spring nrc_k_n_m -- retune the
    # height channel and the placement law re-times itself (before this
    # the neutral share was buried inside one hand-tuned gain).
    # flight_kv is now the PURE feedback gain [m per m/s of velocity
    # error]: the deadbeat value of the linearized return map is
    # ~T_st/2, so useful range is 0.03 (gentle) .. 0.10 (near-deadbeat).
    # Old flight_kv=0.17 folded both terms (T_st/2 ~ 0.096 at k=1800);
    # 0.08 keeps the same total braking at v_des = 0.
    # 2026-08-02 later (log 105820): |foot_des_xy| p99 pinned at the
    # 0.13 m stepper lim with |vxy|_LO ~0.9 -- placement was saturating,
    # not under-gained.  lim 0.13 -> 0.18 (still inside l0=0.461 workspace)
    # and kv 0.08 -> 0.10 (near-deadbeat) so the unclipped demand can
    # actually grow: at v=0.9, (T_st/2+kv)*v ~ 0.18 m.
    # DEPRECATED (2026-08-02): the desired-velocity feedforward is now
    # the -flight_kv*v_des term of the return-map law above (correct
    # Raibert sign: command forward -> foot lands BEHIND neutral).
    flight_kr: float = 0.09
    swing_kp_xy: float = 60.0
    swing_kd_xy: float = 1.1
    swing_kp_z: float = 1300.0
    swing_kd_z: float = 20.0

    # Propeller/HFA. Flight attitude PD uses one gain for roll and pitch.
    # 2026-08-02: bumped ~1.5x for stronger flight balance.
    # 2026-08-02 later (log 103423): tau_des~28 Nm but tau_props only ~6 Nm
    # -- props felt sluggish; raise kR/kW for snappier unsaturated response
    # (authority ceiling raised in prop_att_thrust_max_each_n below).
    # Collective baseline.  History: 2026-08-01 idle was PWM 1100 (user:
    # "base 是 1100, propeller 在 energy 能量补充的时候才加 Fz", 0.675 N
    # total = 0.0103 m*g).  2026-08-02 (user approved, log 103423): at the
    # 1100 idle the differential had ~0.1 N down-headroom -- every torque
    # needed the collective-lift spool-up, and the quadratic PWM->thrust
    # map made low-rpm response sluggish.  Raised to the true tri-rotor
    # hover-headroom form: PWM ~1250 -> per motor t = k*250^2 = 1.41 N,
    # 4.22 N total = 0.064 * m*g of standing lift.  g_eff/t_td/descent
    # f_z all consume this ratio automatically (longer predicted flights
    # -> bigger tilt budget; descent tilt always has thrust to vector).
    # 2026-08-03 (user "关掉其他 propeller就1500怠速 然后只负责姿态"):
    # Fixed idle; collective never plans extra Fz, only attitude
    # differential moves arms off idle.
    # 2026-08-03 (4): idle -> 1450.  t_each = k*(450)^2 = 4.556 N,
    # Idle PWM 1200: t_each = k*(200)^2 = 0.900 N, T = 2.700 N.
    # 2026-08-03: m 6.7 -> 5.61 => ratio = T/(m*g) ≈ 0.0491.

    # ===== PogoX / tri-rotor FLIGHT VELOCITY CONVERGENCE (2026-08-01) =====
    # Outer velocity loop closed on the LIVE flight velocity every tick
    # (IMU-propagated from the liftoff latch -- NOT held constant), the
    # standard multirotor cascade (Salazar-Cruz CEP'09; Lee CDC'10; PogoX
    # ICRA'24):  a_des = kv*(v_des - v(t)) -> desired thrust direction ->
    # reduced-attitude R_des (yaw kept).  The props then track R_des with
    # the same flight_kR/kW geometric PD -- no separate control path.
    # Landing protection is a CONTINUOUS budget, not a switch: a lean of
    # angle theta needs theta/slew seconds to ramp level plus a settle
    # margin, so the tilt target is capped by
    #     theta_budget = slew * max(0, t_td - flight_level_settle_s),
    # which shrinks to zero exactly at the slew rate -> body is level
    # `settle` seconds before ballistic touchdown, with no discontinuity.
    # 2026-08-03 (2): back ON per user "flight_vel_ctrl_enable 开起来" --
    # PogoX-style velocity damper tilts the thrust axis in flight so the
    # machine converges horizontal velocity instead of only leveling.
    # 2026-08-03 afternoon: OFF -- user "先關閉姿態收斂速度"; flight only
    # levels (R_des = Rz(yaw)), no fl_tilt from horizontal velocity error.
    # 2026-08-02 later (log 114427): kv=4 / max=18 / slew=240 slammed
    # fl_tilt_cmd to 18° for 40-70% of every flight (sat threshold only
    # ~3 cm/s of |ev| at base 0.064). Soften so the damper stays in the
    # linear atan region for typical |ev| ~ 0.1-0.3 m/s: kv 4->2,
    # max 18->10, slew 240->120.
    # 2026-08-02 log 214643: with settle 0.06 / slew 120 and the symmetric
    # -arc t_td (fixed alongside, now max'd with the previous hop's real
    # airtime) the budget zeroed at mid-flight -- the machine was level
    # for the whole descent ("机器不要只水平").  slew 180 repays the 10°
    # cap in 56 ms, so settle 0.03 + repay window ~= 90 ms before TD and
    # tilt authority holds through most of the 0.3 s hop.

    # ===== PROPELLER ENERGY SUPPLEMENT (2026-08-01, decoupled) =====
    # Mode1's push spring is capped by the leg force budget
    # (k_push <= stance_fz_max/x0), so at shallow bottoms the leg
    # physically cannot store the full E_need and the hop falls short.
    # The unmet share is routed to the propeller COLLECTIVE, acting over
    # the PUSH stroke AND the flight ascent (props have no stroke limit):
    #     E_leg  = 0.5*k_boost*x0^2          (what the capped spring stores)
    #     E_def  = max(0, E_need - E_leg)    (deficit)
    #     F_prop = min(E_def/(x0 + h_tgt), prop_energy_max_ratio*m*g)
    # Latched once per stance at the PUSH latch; in flight it fades
    # CONTINUOUSLY with the remaining ascent,
    #     F(t) = F_prop * clip(vz_up/prop_energy_apex_fade_vz, 0, 1),
    # reaching zero exactly at apex (vz_up = 0) -- physical rolloff, no
    # switch.  Enters ONLY the Fz channel of the allocator, so attitude
    # torque is untouched; apex overshoot from the ascent share is
    # absorbed by the apex return map (E_loss) hop-to-hop.
    # 2026-08-03 (user): OFF -- no planned prop Fz in stance/flight.
    # Collective stays at the 1500 idle; props only do attitude differential.
    prop_energy_supplement_enable: bool = False
    # 2026-08-02: 0.35 -> 0.60 (larger stance push assist), but flight cap
    # is now zero because the supplement is consumed in stance.
    # 2026-08-02 night (fbslip): 0.60 -> 1.50 so the prop collective can
    # absorb the Fz the leg ceiling refuses (~100 N at m=6.7) without the
    # residual being clipped while the leg rides the 9 Nm torque wall.
    prop_energy_max_ratio: float = 1.50   # supplement collective cap (x m*g)
    # PUSH-only scale on prop_energy_fz (COMP keeps full residual).
    # 2026-08-03 (user "push ... fz propeller还是太大力"): 1.0 -> 0.25 so
    # once the bottom latches, props only hand 25% of the Fz leftover;
    # compression catch assist is unchanged.
    prop_energy_push_scale: float = 0.25
    # vz below which the ascent supplement rolls off (m/s): the last
    # ~vz_fade of upward speed ramps the force linearly to zero at apex.
    # Now only relevant for the stance->liftoff handoff (not active in flight).
    prop_energy_apex_fade_vz: float = 0.30

    # ===== NRC stance energy law (Lo/Chu/Au, ACC 2020) ============
    # "A Norm-Regulation-Based Limit Cycle Control of Vertical Hoppers":
    # ONE continuous force law for the WHOLE stance -- replaces Mode1's
    # COMP spring + PUSH latch + k re-solve (no latch, no blend, no
    # per-hop E_loss map needed: convergence happens INSIDE the stance).
    #
    # Normalized phase state (up-positive, world-Z):
    #   omega = sqrt(k/m)
    #   x1 = (h_com - l0) + m*g_st/k     (spring coord, gravity-shifted)
    #   x2 = vz_up / omega               (velocity, same units as x1)
    # On the limit cycle the stance orbit is a circle ||x|| = r*.  The
    # TARGET radius is built from the HEIGHT TARGET (this is the
    # height <-> energy coupling):
    #   v_to = sqrt(2*g_up*hop_height_m)          (prop-assisted ascent)
    #   r*   = hypot(m*g_st/k, v_to/omega)
    # Control law (paper's NRC-2 smooth variant, F = m*f):
    #   F_pump = -2*m*kR*omega * x2 * (||x|| - r*)
    #   F_leg  = clip(k*(l0-h_com) - bz*vz + F_pump, 0, stance_fz_max)
    # d/dt||x|| = -2*kR*x2^2*(1 - r*/||x||): the radius error decays
    # monotonically DURING the stance -- disturbance rejection within
    # one hop, and F_pump -> 0 on the limit cycle (pure spring remains).
    # F_pump = 0 exactly at the bottom (x2 = 0): no bottom force spike,
    # energy is pumped over the whole stroke -> no tau chatter.
    #
    # PROPELLER FUSION (continuous, decoupled): the stance plant is
    #   m*hddot = -m*g + F_leg + T_sum*cos(theta)        (world-Z, up+)
    # where T_sum is the prop collective along body -Z and
    # cos(theta) = R_wb[2,2] projects it onto the world vertical.  The
    # NRC law computes ONE total world-Z demand
    #   F_des = k*(l0 - h_com) - bz*vz + F_pump
    # and it is split lexicographically, leg first, every tick of the
    # WHOLE stance (compression included -- no phase gate):
    #   F_leg  = clip(F_des, 0, nrc_leg_fz_max)
    #   T_sum  = clip(F_des - nrc_leg_fz_max, 0,
    #                 prop_energy_max_ratio*m*g) / max(cos(theta), 0.5)
    # The 1/cos(theta) de-projection makes the DELIVERED world-Z equal
    # the NRC residual regardless of body tilt, so leg + prop reproduce
    # F_des exactly until both saturate.  The residual rides the pure
    # Fz channel of the allocator; the attitude differential never sees
    # it.  In FLIGHT the props regulate the SAME height target through
    # the ballistic apex prediction (see prop_height_kh below).
    #
    # ---- the tuning set (in order of importance) ----
    #   hop_height_m            -> apex target: the ONE height knob
    #   nrc_k_n_m               -> stance stiffness: bottom depth + stance time
    #                              (depth ~ v_td*sqrt(m/k), T_st ~ pi*sqrt(m/k))
    #   nrc_kR                  -> energy pump gain: bigger = converge in one
    #                              stance but higher mid-stroke force
    #   nrc_bz                  -> small damping (anti-vibration only; the
    #                              energy it eats is repumped by kR)
    #   nrc_leg_fz_max          -> where the leg stops and the props start
    #   prop_energy_max_ratio   -> prop collective ceiling (x m*g)
    #   prop_height_kh          -> flight apex-error gain (0 = leg only)
    #   prop_energy_apex_fade_vz-> ascent force rolls to 0 at apex
    # =====================================================================
    # NORM-REGULATED ENERGY-ANCHORED STANCE ("pogox" key, 2026-08-03)
    # -- ONE height knob.  NOT a PogoX copy: this is a hybrid of
    #   (a) NRC norm regulation (Lo, Chu, Au -- ACC 2020): the pump is
    #       linear in the phase-radius error, so its force is BOUNDED by
    #       the error (1D sim: Fpk ~ 200 N vs 1.3-9 kN for a raw
    #       energy-error pump under estimator lag) and the gain kR is
    #       the citable NRC knob;
    #   (b) the PogoX prop-assisted plant (ICRA'24): effective gravities
    #       g_st/g_up absorb the propeller idle collective, so leg and
    #       prop parameters stay coupled through the model;
    #   (c) our own augments: stroke-anchored stiffness, model-based
    #       vz lag compensation, standstill seed, liftoff unload taper,
    #       and the no-intermediate-cap contract.
    # NO intermediate force caps anywhere in this law -- the only
    # saturations are physical: unilateral contact (F >= 0) and the
    # FINAL joint-torque proportional rescale (9 Nm hardware wall,
    # whose clipped lift rides the prop collective via Fz assist).
    #
    # MODEL (MATLAB SRB leg coordinate x = l0 - l = -q_shift >= 0):
    #   m*z̈ = F - m*g_st,   g_st = g*(1 - rho_st),  g_up = g*(1 - rho_up)
    # Everything derives from ONE knob h* = hop_height_m plus physical
    # constants (m, g, l0, usable stroke X = leg_stroke_max_m):
    #   v*  = sqrt(2*g_up*h*)              (ballistic apex map)
    #   k_v = (m*v*^2 + 2*m*g_st*X)/X^2    (stiffness: the FULL stroke X
    #         stores exactly the limit-cycle energy 0.5*k*X^2 =
    #         0.5*m*v*^2 + m*g_st*X -- hardware anchor, not a gain)
    #   om  = sqrt(k_v/m),  x_eq = m*g_st/k_v
    #   r   = hypot(x - x_eq, vz/om)       (phase radius, NRC coords)
    #   r*  = hypot(x_eq, v*/om)           (limit cycle through liftoff
    #                                       x = 0 at vz = v*)
    #   F   = k_v*x - 2*m*pogox_kr*vz*(r - r*)
    # Contact power dE/dt = vz*(F - k_v*x) = -2*m*kr*vz^2*(r - r*):
    # the radius converges GLOBALLY to r* in BOTH stance sub-phases (no
    # COMP/BOTTOM/PUSH machine, no latch, no per-hop re-solve).  At
    # liftoff r = r* => vz = v* => apex h* above l0.
    # ---- NRC norm-regulation gain (rate knob, NOT a height knob).
    # 1D sim @ h*=0.07: kR 80-300 all converge; 150 settles in ~3 hops.
    # 2026-08-03: 150 -> 100 -> 80 -- softened because real apex sat
    # above hop_height.  ROOT CAUSE found in log 105750: the overshoot
    # came from a corrupted world-kinematic vz re-firing the pump at
    # push end (+60 N impulse), not from nominal cycle energy.  With the
    # pump now fed by the ENCODER axial rate (slip-proof), back to 150:
    # at 80 the first hop under-pumped (vz peak 0.65 vs 0.99 needed)
    # and stalled 1 cm short of liftoff.
    # 2026-08-03 (log 113953): still one rebound short on hop 1 -- from
    # standstill the body only sinks ~5 cm, the spring stores ~2.2 J of
    # the ~5.2 J liftoff budget and kr=150 pumps ~1 J per half-cycle.
    # 250 (sim-converged range goes to 300) roughly doubles the pump so
    # the FIRST push clears q_shift >= 0.
    # MODEL-BASED lag compensation: the vz feeding r and the pump is
    # LPF'd (stance_vz_lpf_tau_s) + kinematic-chain delayed; delayed
    # velocity phase-shifts the pump (under-compensated = parasitic
    # damping -- 1D sim: bootstrap from stand FAILED at 30 ms real lag
    # with 35 ms comp; over-compensation only adds ~2 cm apex).  Predict
    # forward with the APPLIED force (known, no estimator):
    #   vz_n = vz_f + (F_prev/m - g_st)*pogox_vz_lag_s.
    # Sim sweep, comp = 50 ms: real lag 0-40 ms all hop, steady apex
    # 6.6-13.4 cm on h* = 0.07, Fpk <= 216 N.  Err on the HIGH side.
    # Standstill seed: the pump has zero authority at vz = 0 (power =
    # F*vz).  Near the STANDING equilibrium only (spring force <=
    # weight -- deep-compression zero crossings at the oscillation
    # bottom must keep the spring law, else the seed dumps the stored
    # energy every cycle and the buildup deadlocks; found in 1D sim)
    # command F = frac*m*g_st so gravity starts the oscillation.
    # Pump slack limit: minimum fraction of the k_v*x spring the total
    # stance force may be softened down to by a NEGATIVE pump (descent
    # energy-gain).  1.0 = pump may never soften the spring; 0.0 = pump
    # may cut force to zero (pure 1D law; slack foot at TD -> bottom
    # bounce, log 114543 hop 2).
    # Liftoff unload boundary condition (NOT a cap): MATLAB uses
    # spring_force = -k*(l-l0), so force vanishes with LEG compression,
    # not world-Z COM deficit.  Use x_leg=max(0,-q_shift):
    # F *= min(1, x_leg/d), continuous at l=l0 even when body is tilted.
    # =====================================================================
    # FB-SLIP STANCE LAW  (37a1475 port; hopped reliably 2026-07-25/26)
    # =====================================================================
    # MODEL (SRB, world-Z).  m*z̈ = F_leg + rho_st*m*g - m*g, so the leg
    # works against the EFFECTIVE stance gravity g_st = g*(1 - rho_st)
    # (props idle at 1500 carry rho_st*m*g).  Spring coordinate is the
    # world-Z height deficit x = l0 - h_com (x >= 0, x=0 at full
    # extension).  All quantities below derive from FOUR inputs --
    # (m, g_st, takeoff_height_m, leg_force_budget_g) -- so the tuning
    # stays COUPLED through the model instead of independent knobs:
    #
    #   v_to  = sqrt(2*g_up*takeoff_height_m)          (ballistic apex)
    #   F_max = leg_force_budget_g * m*g               (peak budget)
    #   x0    : (w_push*F_max - m*g_st)*x0 = 0.5*m*v_to^2   (design map:
    #           reception target depth s_tgt = x0 - x_td)
    #   F_push: energy balance INCLUDING the liftoff taper (see below)
    #           F_push*(x0 - d/2) = 0.5*m*v_to^2 + m*g_st*x0
    #
    # PHASES:
    #  1) RECEPTION (positional, NO velocity estimate in the loop):
    #     force ramps from preload*m*g_st to ~m*g_st over the stroke to
    #     s_tgt (gravity guarantees the sink), then a stiff CATCH ramps
    #     to F_max within the catch span.
    #  2) BOTTOM (one physical event): vz settled OR position rebound.
    #  3) PUSH (open loop, sized once at the bottom): constant F_push,
    #     blended in over stance_push_blend_tau_s, then TAPERED linearly
    #     to ZERO over the last stance_push_taper_m of extension --
    #     F(x) = F_push * min(1, x/d).  The force is therefore
    #     CONTINUOUS through the stance->flight switch (no step drop at
    #     liftoff / 起飞前几拍慢慢变小), the leg is unloaded exactly at
    #     full extension, and the taper's lost work d/2*F_push is repaid
    #     by the F_push sizing above -- the parameters stay coupled.
    #
    # ---- (a) model inputs ----
    # Ballistic height that sizes ONLY the takeoff speed.
    # 2026-08-03: 0.08 -> 0.07 (match hop_height; log 000102 flew ~14 cm
    # on 0.08 -- lower v_to + softer gains below to stop the overshoot).
    takeoff_height_m: float = 0.05
    # Hard peak-force budget (F_max = beta*m*g ~ 230 N at m=6.7; the AK60
    # torque rescale is the true hardware protection below it).
    leg_force_budget_g: float = 3.5
    # ---- (b) reception ----
    # Preload fraction at touch, weight fraction AT the target depth
    # (< 1 keeps a net downward pull), catch span past the target.
    stance_recv_preload_frac: float = 0.3
    stance_recv_tgt_weight_frac: float = 0.95
    stance_recv_catch_span_m: float = 0.015
    # Brake ramp after TD + first-order blend on the reception force.
    stance_brake_ramp_s: float = 0.015
    stance_fz_blend_tau_s: float = 0.012
    # ---- (c) design map (reception target depth from push balance) ----
    #   (w_push*F_max - m*g_st) * x0 = 0.5*m*v_to^2,  s_tgt = x0 - x_td
    # (squat tracks the height target and the prop g_st automatically).
    stance_push_force_frac: float = 0.42
    stance_travel_min_m: float = 0.020
    # ---- (d) bottom latch ----
    # vz settle band + confirm window + position rebound.
    stance_bottom_settle_mps: float = 0.10
    stance_bottom_confirm_s: float = 0.030
    stance_bottom_rebound_m: float = 0.004
    # ---- (e) push taper (2026-08-03, "push最后几拍要慢慢变小") ----
    # Linear roll-off of the push force to ZERO over the last d meters
    # of extension: F(x) = F_push*min(1, x/d), x = l0 - h_com.  Kills
    # the step discontinuity at the stance->flight switch (the old
    # constant push held full force until the phase machine cut it in
    # one tick).  The taper is POSITION-driven (deterministic, no
    # estimator in the loop, same philosophy as the reception) and its
    # lost work is repaid inside the F_push energy balance:
    #   F_push = (0.5*m*v_to^2 + m*g_st*x0) / (x0 - d/2)
    # which reduces to the classic m*g_st + m*v_to^2/(2*x0) at d = 0.
    # 0 disables (back to the hard cut).
    stance_push_taper_m: float = 0.015
    # 2026-08-02 (2): 2200 -> 1800 per user -- moderate the leg force
    # again; the compression deceleration the softer spring gives up is
    # picked up by the prop collective (the NRC residual above
    # nrc_leg_fz_max now fires through compression with the 1/cos(theta)
    # world-Z de-projection, see PROPELLER FUSION above).
    # 2026-08-02 (3): 1800 -> 3200 per user "stance 停留时间不能久, 快速
    # 压缩快速起跳".  T_st ~ pi*sqrt(m/k): 0.192 -> 0.144 s (logged
    # stances 0.25 s -> expect ~0.18 s).  Bottom depth shrinks v_td/omega
    # ~ 5.5 -> 4 cm at v_td 1.2.  Peak passive demand v_td*sqrt(k*m)+mg
    # ~ 240 N briefly exceeds leg(150)+prop(39) around the bottom -- the
    # cap rides for a few ticks and the per-hop apex trim absorbs the
    # clipped energy; the Raibert neutral point T_st/2 re-times itself
    # from this knob automatically.
    nrc_k_n_m: float = 3200.0            # virtual spring stiffness [N/m]
    # 2026-08-02 history: 400 -> 150 -> 350.  At 400 with no other
    # protection, the ~25-50 ms vz-estimate lag made the pump keep
    # firing past the target (log 073212: 0.15 m hops on a 0.07 m
    # setpoint).  150 fixed the overshoot but gutted the injection: log
    # 080815's first stance peaked at 133 N total demand -- barely the
    # spring, 3 N left for the props, no liftoff ("propeller根本没出力").
    # With the extension fade + per-hop apex trim now absorbing the
    # lag-induced overshoot, the pump can be strong again: at 350 the
    # sim (30 ms lag) launches cold, props saturate their 39 N cap each
    # hop, and apex walks 10.6 -> 8.1 -> 7.4 -> 7.0 cm in 4 hops.
    # 2026-08-02 later (log 115132): 350 -> 150.  The pump's noise gain
    # is 2*m*kR*|r-r*| ~ 4700*|r-r*| N/(m/s) at 350 -- stance vz noise
    # became the +-20-47 N/tick f_ref jitter the user saw as "腿部
    # torque cmd 一直波动" (hip tau jumped 1-3 Nm/tick; the attitude PD
    # contributed only 0.3).  At 150 (paired with the 20 ms vz LPF) the
    # jitter drops ~5x; slower in-stance convergence is covered hop-to-
    # hop by the apex trim return map.
    # 2026-08-02 (log 231912, stuck on ground): COLD START from stand --
    # v_td = 0, so the pump (∝ vz) injects from zero.  At kR=150 the
    # stroke peaked at 114 N / vz 0.45 m/s, stalled 4 mm short of
    # liftoff, fell back and dribbled at 20-44 N (< mg) forever.  300
    # doubles the per-stroke injection so the stand-start escapes in
    # 1-2 pumps; the 20 ms vz LPF + extension fade + apex trim absorb
    # the extra noise/overshoot that killed 350 before.
    nrc_kR: float = 300.0                # norm-regulation gain [1/(m*s)]
    nrc_bz: float = 8.0                  # virtual damping [N*s/m]
    # Pump extension fade [m]: F_pump *= clip((l0 - h_com)/fade, 0, 1).
    # Energy injection belongs to the mid-stroke; the last few cm before
    # liftoff are exactly where estimator lag turns residual pumping into
    # apex overshoot, so the pump ramps SMOOTHLY to zero as the leg
    # approaches full extension (continuous -- no hard cut).  0 disables.
    # 2026-08-02 (log 112743 hop 3): 0.04 -> 0.02.  With the spring term
    # already zero at natural length, fading the pump over the last 4 cm
    # left the final ~2 cm of stroke with NO upward force -- vz stalled
    # at ~0.6 m/s just short of the q_shift +1 cm liftoff threshold, the
    # robot fell back and double-pumped (0.61 s stance).  2 cm keeps the
    # anti-overshoot intent while carrying the push through liftoff.
    # 2026-08-02 (log 231912): 0.02 -> 0.01, same failure again at the
    # stiffer k=3200 -- the whole cold-start stroke is only ~3.6 cm, so
    # a 2 cm fade muted the pump over most of the push (vz stalled 4 mm
    # short of liftoff).  1 cm frees the mid-stroke while still zeroing
    # the pump right at the exit.
    nrc_pump_ext_fade_m: float = 0.01
    # ---- per-hop apex return map (Koditschek-Buehler layer for NRC) ----
    # The in-stance NRC loop regulates the ESTIMATED energy; any
    # systematic vz-estimate bias/lag lands at a biased apex no matter
    # the gains.  Close the loop on the MEASURED apex instead: at each
    # touchdown the flight-time apex (z_apex_actual, drift-free) updates
    # a slow multiplicative trim on the height target,
    #   trim <- clip(trim - gain*(h_apex - h_tgt)/h_tgt, min, max)
    # so h_tgt_eff = hop_height * trim.  Error dynamics are the same 1-D
    # contraction as the Mode1 E_loss map: e+ = (1-gain)*e, deadbeat at
    # gain=1.  Sim: converges to 7.0 cm in 3-4 hops under 25-70 ms lag
    # or a 20% velocity-estimate bias.  gain=0 disables.
    nrc_apex_trim_gain: float = 0.5
    # 2026-08-02 (log 112743): min 0.4 -> 0.7.  With hop_height 0.12 the
    # effective target floor is 8.4 cm -- always above the liftoff
    # feasibility line, so the trim can never command a hop the leg
    # cannot physically separate from.
    nrc_apex_trim_min: float = 0.7
    # 2026-08-05 (log 205009): 1.6 -> 2.5.  With hop_height=5 cm the
    # trim pinned at 1.6 from hop 4 onward while measured apex stuck at
    # 3-4.5 cm -- that self-imposed ceiling was the height wall.  2.5
    # leaves headroom for the return map to keep lifting h* until the
    # knobs actually deliver; err_clip still bounds each hop step.
    nrc_apex_trim_max: float = 2.5
    # Per-hop clip on the RELATIVE apex error fed to the trim (0 = off).
    # 2026-08-02: the first hop-off transient apexes at 0.01-0.03 m (-70%
    # error) and one gain=0.5 step yanked trim to ~1.35 -> hop 2 overshot
    # to 0.13-0.14 m, then trim oscillated in big steps (1.0->0.645->0.40,
    # hitting the floor).  Clipping |e_apex| <= 0.4 bounds each trim step
    # to gain*0.4 = 0.2: transients converge smoothly in 2-3 hops and
    # normal hops (error < 40%) are unaffected.
    nrc_apex_trim_err_clip: float = 0.4
    # ANTI-RATCHET (2026-08-02, log 112743): the trim update used to be
    # skipped entirely when the flight was shorter than the 0.12 s apex
    # window.  That made the return map a one-way ratchet: overshoot
    # hops pull the trim DOWN, but ground-chatter "hops" (T_fl ~ 0.10 s,
    # apex 1-2 mm) never push it back UP -- once trim fell to 0.588 the
    # robot stuck to the ground permanently (跳4-6 粘地).  Now a flight
    # shorter than the window counts as apex ~ 0: e_apex = -err_clip,
    # so the trim recovers by gain*err_clip (+0.2) per failed hop and
    # the robot climbs out of the chatter trap on its own.
    #
    # Stance-duration guard: a stance longer than this is a double-pump
    # (missed liftoff -> fell back -> pumped again; the log-112743 hop-3
    # anomaly was 0.61 s and its 0.141 m apex wrongly slammed trim
    # 0.788 -> 0.588).  Such a hop's apex does not enter the trim update.
    # The chatter recovery above still fires regardless (it uses no apex
    # measurement).  0 disables.
    # 2026-08-02: 0.45 -> 0.35 with the k=3200 stiffening (normal stance
    # now ~0.18 s, so 2x normal ~ 0.36 marks a double-pump).
    nrc_trim_stance_max_s: float = 0.35
    # Split point between the leg and the props for the stance energy
    # demand [N].  This is the LEG'S REAL AUTHORITY, not the stance_fz_max
    # safety clamp: the AK60 hip-torque limit is what actually bounds the
    # axial force (~15 N per Nm of tau_out_max in this geometry, so the
    # 10 Nm bring-up cap => ~150 N).  Demand above this rides the prop
    # collective instead of being commanded to a leg that cannot deliver
    # it (the 2026-08-02 063655 log commanded 430 N, the torque limiter
    # scaled it to ~1/3, and the props stayed idle because the residual
    # was measured against 500 N).  Raise it together with tau_out_max.
    # Ignored when the props are disarmed (leg is then asked for all).
    # 2026-08-02: 160 -> 130 per user (moderate the leg, more prop in
    # compression).  Props now engage once the demand passes 130 N --
    # 54% of the stance in sim vs 41% before -- instead of only at the
    # very bottom.  Leg peak force drops by the same 30 N.
    # 2026-08-02 later (log 110447): f_ref_w2 p99 pinned at 130 while
    # nrc_f_des peaked 874 N -- user "放开 cut".  Raised to stance_fz_max
    # so the only remaining clip is the 500 N safety clamp; residual above
    # that still cannot exist in f_ref, and prop_energy_max_ratio still
    # caps the prop share independently.
    # 2026-08-02 night (user "fz上加大propeller减小腿部 不要碰到limit"):
    # 150 -> 100.  ~15 N/Nm * 9 Nm ≈ 135 N is the HARD wall; 100 N leaves
    # ~35 N of torque headroom for attitude so the proportional tau rescale
    # never fires on Fz alone.  Residual rides prop_energy_fz.
    # 2026-08-03 (log 000102 still pinned tau1): 100 -> 70.  More Fz to
    # the props, more torque headroom for the (now softer) attitude PD.
    nrc_leg_fz_max: float = 70.0
    # FLIGHT height regulation via props: while ascending, predict the
    # ballistic apex h_pred = h + vz^2/(2g) and command
    #   F = clip(kh*m*g*(h_apex_tgt - h_pred)/hop_height, 0, cap) * fade(vz)
    # Pure feedback on the CURRENT arc: zero when the arc already reaches
    # the target, fades to zero at apex.  kh = 1 means a 100% relative
    # apex shortfall asks for one extra m*g of collective (then capped).
    # 2026-08-02: 1.0 -> 0.4.  Log 070306 apexes hit 13.5/18.1 cm vs the
    # 7 cm target with 5-19 N of flight collective riding this feedback:
    # its absolute-height reference (-p_hat_w2 vs l0+hop) carries a few cm
    # of drift, and at kh=1 that bias turns into the same overshoot.  At
    # 0.4 the props only rescue large shortfalls instead of inflating the
    # arc.  Set 0 for pure-leg apex control.
    # 2026-08-02 later (log 114427): 0.4 -> 0.2.  First hop still rode
    # ~13 N of flight prop_energy through ascent (apex 0.16-0.21 vs 0.12);
    # halve the gain so the flight channel only rescues large shortfalls.
    prop_height_kh: float = 0.2

    # RB gamepad "big jump": the NEXT stance solves the push spring for
    # big_jump_height_gain * hop_height_m, one hop only.
    big_jump_height_gain: float = 1.8

    # ===== RT stand-and-unfold first hop (runtime compat, 2026-07-24) =====
    # During the RT transition the FIRST stance is a standing launch, not a
    # landing: lcm_controller sets rt_first_hop_spring_active and the stance
    # Z law becomes a plain virtual spring from the static P4 pose.
    rt_first_hop_spring_active: bool = False
    rt_first_hop_spring_k_n_m: float = 1800.0
    rt_first_hop_spring_d_n_s_m: float = 6.0
    # Stance-phase propeller idle (collective).
    # 2026-08-01: PWM-1100 idle, props add planned Fz only through the
    # PUSH energy supplement.  2026-08-02: raised to match the flight
    # base (~PWM 1250, see prop_base_thrust_ratio) so the motors stay in
    # the linearized PWM region and the flight differential spools up
    # without a dead zone at liftoff.  (The stance attitude-residual
    # rationale from the HFA experiment is obsolete -- stance props are
    # collective-only under the decoupled contract.)
    # Match flight idle (PWM 1200 / ratio 0.0491 at m=5.61).
    prop_stance_base_thrust_ratio: float = 0.0491
    stance_use_props: bool = True
    # ===== Hybrid leg-prop Z (TA-SLIP, 2026-07-19) =====
    # Props shape EFFECTIVE GRAVITY (continuous, low authority); the leg
    # shapes CONTACT ENERGY (impulsive, high authority). Three couplings,
    # all closed-form inside the stance-Z law:
    #   ascent   g_up = g*(1 - prop_base_thrust_ratio)
    #            -> v_to = sqrt(2*g_up*hop_height_m) shrinks the push spring;
    #   descent  g_dn = g*(1 - prop_flight_brake_ratio)
    #            -> aerial braking lowers v_td, softening the landing spring
    #               BEFORE contact;
    #   stance   g_st = g*(1 - prop_stance_base_thrust_ratio)
    #            -> props carry part of the weight, shrinking both springs'
    #               gravity terms.
    # The flight-time apex measurement uses the asymmetric-arc formula
    # h = T^2 / (2*(1/sqrt(g_up)+1/sqrt(g_dn))^2) so the discrete apex layer
    # stays unbiased when brake != ascent ratio. All couplings vanish when
    # props are disarmed (ratios treated as 0).
    # Extra collective while DESCENDING in flight (ratio of m*g).
    # 2026-08-01 OFF: Fz is only planned for the PUSH energy supplement;
    # descent keeps the same PWM-1100 idle base as the rest of the cycle.
    prop_flight_brake_ratio: float = 0.0
    # Descent detection threshold on world vz (m/s, up-positive).
    prop_flight_brake_vz_mps: float = 0.10
    # ===== Descent VELOCITY-BRAKE collective (2026-08-02) =====
    # The tilt loop's lateral force is f_z*tan(theta): during ASCENT f_z
    # carries the height channel's energy supplement (10-40 N) and the
    # tilt brakes for real, but during DESCENT the collective is the
    # 0.68 N idle -- an 18 deg lean delivers 0.2 N (~0.03 m/s^2),
    # nothing.  This grants the descent a BOUNDED collective purely so
    # the tilt has something to vector:
    #   ratio = prop_descent_brake_ratio
    #           * clip(|e_v|/prop_descent_brake_ev_mps, 0, 1)   (error gate)
    #           * clip((t_td - settle)/settle, 0, 1)            (TD fade)
    # Proportional to the horizontal velocity error (zero when
    # converged), fading to zero on the same time-to-touchdown clock as
    # the tilt budget (level AND idle before landing).  It is a
    # VELOCITY-channel force by intent but a pure Fz by allocation; the
    # per-hop apex trim absorbs its (small) vertical energy over hops.
    # 0.10 * m*g = 6.6 N -> 2.1 N lateral at 18 deg.  0 disables.
    # 2026-08-02: 0.10 -> 0.15 (~10 N -> ~3 N lateral at 18 deg) now that
    # the settle=0.06 window actually lets the descent brake fire.
    # 2026-08-03: OFF with the idle-1500 / attitude-only prop mode.
    prop_descent_brake_ratio: float = 0.0
    prop_descent_brake_ev_mps: float = 0.40
    # 2026-07-19 (per user): 3D / bidirectional thrust DISABLED everywhere.
    # prop_bidir=False makes negative thrust idle at pwm_min in the PWM map,
    # forces forward-only floors in the stance overlays / daisy chain, and
    # disables the stance downforce experiment. "auto" cannot reverse with
    # bidir off, so flight is forward-only too.
    prop_flight_reverse: str = "auto"
    prop_bidir: bool = False
    # Total Z budget = thrust_total_ratio_max * m*g.  2026-08-03 (user
    # "propeller所有 maximum是1950"): 0.60 -> 0.93 so the collective can
    # use the full hardware (3 * 20.3 N ≈ 60.9 N ≈ 0.93 m*g).  The old
    # 0.60 cap pinned thrust_sum at 39 N (~PWM 1667) and made the 1950 us
    # per-arm ceiling unreachable.  Used as the STANCE ceiling; flight
    # uses the tighter flight_thrust_sum_max_ratio below.
    thrust_total_ratio_max: float = 0.93
    # FLIGHT collective ceiling (attitude-only contract).  Idle at PWM
    # 1200 is only ~0.041 m*g; the mixer lifts collective to realize
    # torque when the low arm floors at 1000.  Log 031217: thrust_ref
    # stayed 2.7 N but thrust_sum p95 hit 30 N (~0.46 m*g) -- the body
    # floated, T_fl ~0.52 s ≈ 33 cm ballistic vs hop_height=0.07.
    # Cap total flight thrust so attitude cannot inflate apex.
    # 2026-08-03: 0.18 -> 0.10 was too tight -- log 103253 delivered
    # only ~18% of tau_des (1.7/9.3 Nm) with thrust_sum pinned at 5.5 N
    # and PWM max 1485; attitude felt powerless.  0.25*m*g ≈ 13.8 N at
    # m=5.61 restores differential headroom without the old 0.46 mg float.
    flight_thrust_sum_max_ratio: float = 0.25
    # Per-arm ceiling, kept CONSISTENT with the PWM ceiling below:
    #   thrust(pwm) = k*(pwm-1000)^2 -> 2.25e-5 * 950^2 = 20.3 N at 1950 us.
    # 2026-08-02: was 10.0 N, a number derived back when k = 1.24e-5 (where
    # 10 N did sit near 1900 us).  After the thrust-stand recalibration to
    # k = 2.25e-5 the same 10 N is only 1667 us, so this cap -- not the PWM
    # ceiling -- was silently binding: the 0801/0802 logs show all three
    # arms pinned at exactly 1667 us.  Now the hardware PWM max is the
    # binding limit and the total budget above governs the collective.
    thrust_max_each_n: float = 20.3
    pwm_min_us: float = 1000.0
    # ESC/prop hardware maximum (user): 1950 us, not 2000.
    pwm_max_us: float = 1950.0
    # 2026-07-2x bench recalibration (thrust-stand): k = 2.25e-5 N/us^2.
    prop_k_thrust: float = 2.25e-5

    # Hop-to-hop apex return-map gain for the active push spring.
    mode1_apex_adapt_gamma: float = 0.4

    # State estimation.
    use_fc_quat: bool = False
    att_accel_weight: float = -0.01
    att_stance_bound_lo: int = 90
    att_stance_bound_hi: int = 130
    # Liftoff XY latch window [ticks]: median of the last N stance samples
    # (push tail, ~40 ms @500Hz).
    # 2026-08-02 later (user): revert skip/mid-push window -- same as MATLAB.
    vel_push_tail_n: int = 20
    # 2026-08-02 (user): reverted to the MATLAB unconditional window.
    # 2026-08-03 evening: RE-ENABLED at 0.25 for the Raibert convergence
    # work -- logs 105750/115333 show the unloaded push tail (foot slip /
    # body pivot) writes -5..-6 m/s garbage into the ring, the LO latch
    # then throws the swing target 0.4 m off and the first flight ticks
    # slam the 9 Nm wall.  With the gate, the ring holds the last N
    # samples taken while f_ref_z > 0.25*m*g (foot firmly planted); the
    # placement law finally sees the real takeoff velocity.  Set back to
    # 0.0 for the pure MATLAB behavior.
    vel_latch_fz_min_ratio: float = 0.25

    # Motor command limits.
    # 2026-08-02: 40 -> 9 (real motor limit, per user "腿部 torque max 是
    # 9Nm").  With 40 the software limit never bound and the TRUE
    # saturation happened downstream in the motor driver as a PER-AXIS
    # hard cut: foot-force direction distorted AND the prop residual pass
    # (saturation-aware, see "_tau_limit_proportional" + the final
    # allocation) never saw the deficit.  At 9 the saturation is modelled
    # INSIDE the core: the whole vector scales (direction preserved) and
    # the props pick up exactly the attitude moment the leg cannot
    # deliver -- the hybrid leg+prop system as intended.
    tau_cmd_max_nm: tuple[float, float, float] = (9.0, 9.0, 9.0)
    tau_cmd_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ======================================================================
    # ROBOT MODEL AND ADVANCED OPTIONS
    # ======================================================================

    mode_1d_disable_mpc: bool = True

    # Physical model (also affects energy and force allocation).
    # 2026-07-2x re-weighed WITH the RM folding arms + camera: 6.7 kg.
    # Sizes both stance springs (k ~ m) and the propeller collective.
    # 2026-08-03: 6.7 -> 5.61 (user; prior calibrated value 07-20).
    mass_kg: float = 5.61
    gravity: float = 9.81
    # COM offset in base frame (m). If unknown, keep zeros; tune later.
    # Computed from MuJoCo MJCF (`Hopper-modee-clean/mjcf/hopper_serial.xml`) at default pose.
    com_b: tuple[float, float, float] = (-2.79376456e-04, 1.68299070e-06, -5.72937376e-02)
    # Body inertia diagonal in BODY frame (kg*m^2). Reserved for future model-based planning.
    # Computed from MuJoCo MJCF (`Hopper-modee-clean/mjcf/hopper_serial.xml`) as whole-body inertia about COM,
    # expressed in base/body frame (diagonal approximation; off-diagonals are small).
    I_body_diag: tuple[float, float, float] = (0.0716072799, 0.0716088488, 0.0579831725)

    # Legacy trajectory/MPC timing (inactive while use_unified_stance/use_mpc
    # are false).
    hop_z0: float = 0.9
    stance_T: float = 0.20
    stance_min_T: float = 0.08
    flight_min_T: float = 0.10

    # Flight XY velocity latch: MATLAB-style mean of the last N stance
    # planted-foot samples. Falls back to the instantaneous liftoff sample.

    # DLS / ridge regularization for delta Jacobian inversions.
    # When enabled, we compute a damped pseudo-inverse:
    #   A^+ = (A^T A + λ^2 I)^(-1) A^T
    # with λ = lambda_rel * ||A||_F.
    # This prevents inv(J_inv) / inv(J_inv^T) from exploding near singularities.
    delta_jacobian_dls_enable: bool = True
    delta_jacobian_dls_lambda_rel: float = 0.002

    # ===== Unified stance reference (single-mode; no COMP/PUSH switching) =====
    # When enabled, stance is controlled by a single smooth COM-z reference trajectory:
    #   (z_td, vz_td) -> (z_min, 0) -> (z_end, v_to)
    # where z_min (compression depth) is chosen adaptively from touchdown vertical speed to "soft land".
    # Disabled for leg-axis SLIP spring (compress/push gated by q_shift, not COM-z profile).
    use_unified_stance: bool = False
    # Approximate max upward deceleration (m/s^2) during landing. Smaller => deeper compression, softer landing.
    # Reduced from 25.0 to 15.0 to increase compression depth (squat deeper before jump).
    soft_land_a_max: float = 16.0   # Softer landing → deeper but gentler compression
    # Time to reach max compression (s): t_comp ≈ |vz_td| / soft_land_a_max, clamped to keep numerics stable.
    soft_land_tc_min: float = 0.06
    soft_land_tc_max_ratio: float = 0.45  # shorter compression window -> stiffer leg feel
    # Compression depth bounds (m) relative to touchdown height (base frame).
    soft_land_depth_min_m: float = 0.0
    # NOTE: for meaningful leg-only hopping, we need enough compression travel to generate takeoff velocity
    # without forcing an extra downward motion in the "push" segment. 0.12m was too small in practice.
    # Increased from 0.25 to 0.35 to allow deeper squat.
    soft_land_depth_max_m: float = 0.15  # Allow up to 15cm compression for landing absorption
    # Optional safety guard on minimum base height during stance reference generation (m).
    z_guard: float = 0.35

    # Takeoff speed bounds (safety clamps).
    v_to_min: float = 0.40
    v_to_max: float = 1.60

    # Advanced propeller allocation constraints.
    stance_thrust_sum_min_ratio: float = 0.02
    stance_thrust_sum_max_ratio: float = 3.50

    # prop geometry in base frame (meters); default is symmetric with GREEN on +X
    prop_arm_len_m: float = 0.569451

    # ===== Prop PWM channel mapping (REAL ROBOT) =====
    # ModeE solves 3 thrust variables ordered with `prop_positions_b`:
    #   arm 0 at -90 deg (0,-L)          -> physical Motor3/PWM[3]
    #   arm 1 at +150 deg (-x,+y)        -> physical Motor1/PWM[1]
    #   arm 2 at +30 deg  (+x,+y)        -> physical Motor2/PWM[2]
    # 2026-08-01 spin-test re-alignment (spin_prop_test.py, one motor at a
    # time, user observed): pwm[3] spins the -Y arm, pwm[1] the LEFT-front
    # (-x,+y) arm, pwm[2] the RIGHT-front (+x,+y) arm.  The previous
    # 2026-07-18 map ((2,),(3,),(1,)) was one cyclic step off after the
    # motor/ESC rewiring -- attitude differential landed on the WRONG arm.
    prop_pwm_idx_per_arm: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (3,),  # arm 0 (-Y)    = Motor3
        (1,),  # arm 1 (-x,+y) = Motor1
        (2,),  # arm 2 (+x,+y) = Motor2
    )

    # Minimum forward thrust per arm keeps the propellers responsive.
    wbc_thrust_min_each_n: float = 0.1
    # ===== Attitude differential spool-up cap (2026-08-01) =====
    # At the PWM-1100 idle base a ZERO-SUM differential has only
    # (base - t_min) ~ 0.1 N of down-headroom -- no torque.  Tri-rotor
    # practice at a low idle: the low arm floors at t_min while the arms
    # that need torque spool UP above the base (direction-preserving,
    # minimal collective lift).  This cap bounds how far ANY arm may rise
    # above the collective base for the ATTITUDE channel, so a large
    # attitude demand can never stack the PWM sky-high ("不能叠加很多
    # pwm"): 3 N above the 0.225 N base = PWM ~1380 ceiling per arm.
    # Per-arm attitude differential above idle (N). 2026-08-02: 3 -> 5 so
    # the stronger flight_kR/kW has room to act (still well below the
    # 20.3 N / 1950 us per-arm ceiling).
    # 2026-08-02 later (log 103423): delivered tau_props ~6 Nm vs demand
    # 28 Nm -- the 5 N differential was the clip. 5 -> 8 -> 12 (with the
    # new 1.41 N hover base the per-arm ceiling is ~13.4 N ~ PWM 1772,
    # still clear of the 20.3 N / 1950 us hard limit).  Expected torque
    # authority ~15 Nm (~1.25 Nm per N of cap, measured from 103423).
    # Full hardware headroom to 1950 us: att_cap = 20.3 - idle_thrust.
    # Idle 1200 (0.900 N) -> att_cap ≈ 19.40.
    prop_att_thrust_max_each_n: float = 19.4

    # ===== Contact friction (controller-side) =====
    # Must match the ground/contact physics as closely as possible (e.g. MuJoCo friction).
    # This parameter is used by BOTH:
    # - stance wrench reference (friction cone)
    # - WBC-QP (friction pyramid / cone approximation)
    # Contact friction used by BOTH stance reference and WBC-QP.
    # Match Hopper_sim default (and be conservative on real floors): higher mu makes the controller request
    # larger horizontal GRF, which can cause real slip + leg jitter when the true friction is lower.
    mu: float = 0.5

    # ===== Leg kinematics backend =====
    # - "delta": real-robot 3-RSR delta motor angles (uses `forward_kinematics.py`)
    # - "serial": MuJoCo serial-equivalent leg (roll/pitch/shift) used by hopper_serial.xml
    leg_model: str = "delta"

    # Serial leg geometry (must match hopper_serial.xml):
    # base_link -> hip origin offset (m), and hip -> foot body origin offset (m).
    serial_hip_z_off_m: float = 0.0416   # base_link to hip is at z=-0.0416
    serial_foot_z_m: float = 0.5237      # Leg_Link to Foot_Link offset magnitude along -Z

    # Optional leg-axis allocation retained for comparison. False selects the
    # CASE-style world-frame SRB allocation used by both control modes.
    # True  -> decompose the stance contact force into
    #            f = f_ax * u + f_side,   u = (foot - COM)/|foot - COM|
    #          f_ax (axial, along the COM->foot line) has ZERO moment arm ->
    #          pure energy/height channel; f_side (perp to r) delivers EXACTLY
    #          the attitude torque (min-norm solution of tau = -r x f). The
    #          big push can no longer tip the body (the rx*fz lever term
    #          vanishes identically instead of being cancelled by feedforward)
    #          and stance_fxy_max above now clips ONLY the attitude channel.
    #          f_ax is chosen so the WORLD-vertical push still equals fz_cmd.
    # False -> legacy body-frame z/xy split with the rx*fz lever feedforward.
    # 2026-07-11 (log analysis, hop2 modee_latest): with the legacy split the
    # push-phase lever term rx*fz/rz reached -30..-50 N while the pitch
    # correction only asked +6..+28 N -- the sum pinned fx at the -20 N clip
    # for the WHOLE stance and the attitude channel never got authority.
    # PURE-LEG ONLY workaround: set True (axial push has no moment arm, the
    # +/-20 N budget goes entirely to attitude). With PROPS ARMED keep False:
    # the paper HFA structure (fz fixed, fxy exact torque solve) is correct
    # because the stance props compensate the clip residual (Eq. 12).
    # 2026-07-11 user: leg spring-SLIP (axial spring + side attitude), not body
    # xyz split (fx/fy/fz + rx*fz lever feedforward).
    stance_leg_frame_alloc: bool = False

    # Reverse-thrust and downforce experiment (inactive when prop_bidir=False).
    # USER REQUIREMENT (2026-07-06): total Z thrust must stay ~constant during attitude
    # correction. The differential solution is zero-sum by geometry (symmetric tri-rotor
    # => sum of attitude thrusts == 0), so the sum only changes when an arm hits this
    # floor and collective lift kicks in. Set the budget to the physical ceiling the
    # pwm mapping can deliver (pwm_rev_floor_us): with CALIBRATED k=1.24e-5,
    # k*(1000-400)^2 = 1.24e-5*600^2 ~ 4.5 N per arm (2026-07-19: opened with 3D).
    prop_reverse_max_n: float = 4.5
    # Lowest reverse pwm command (us). 600us of reverse range; reverse thrust per us is
    # aerodynamically weaker than forward anyway (fixed-pitch prop) -- calibrate before
    # deepening this.
    pwm_rev_floor_us: float = 400.0

    # ===== Stance friction-cone modulation via prop downforce (2026-07-07) =====
    # Physics: during stance the props push DOWN (collective reverse, total
    # stance_downforce_n newtons). The leg fz command is raised by the same
    # amount, so the CoM vertical dynamics (hop apex, SLIP energy) are UNCHANGED
    # -- the extra prop force and the extra ground reaction cancel on the body.
    # What DOES change is the contact normal force:  N = fz_leg = fz_slip + F_dn,
    # so the friction cone |fxy| <= mu*N widens by mu*F_dn. Crucially this holds
    # right AT touchdown when fz_slip is still near zero -- exactly the moment
    # the foot normally breaks loose on low-mu ground. Equivalent friction:
    #   mu_eff = mu * (1 + F_dn / fz_slip)   (pointwise; largest gain early stance)
    # Only active when prop_bidir (needs reverse thrust); the applied value is
    # capped by the physical reverse budget 3*prop_reverse_max_n. In stance this
    # REPLACES the positive baseline prop_base_thrust_ratio (they contradict).
    # 0 = off (default; behavior identical to before).
    stance_downforce_n: float = 0.0
    # Downforce window after touchdown (s). Sim finding (2026-07-08): full-stance
    # downforce BACKFIRES -- fz_cmd += F_dn also scales the lever-arm fxy
    # feedforward (fxy ~ rx*fz/rz), so with an off-center foot the robot pushes
    # itself horizontally harder all stance (drift/slip UP even on high-mu
    # ground). The slip that matters happens in the first tens of ms after
    # touchdown while fz_slip is still ramping from ~0; a short pulse boosts N
    # exactly there and expires before the lever-arm side effect integrates.
    # <=0 = apply for the whole stance (the naive variant, kept for A/B).
    stance_downforce_td_s: float = 0.06

    # Propeller PWM mapping method. The calibrated square-root mapping is the
    # default; set False to use the measured lookup table.
    use_hopper4_pwm_mapping: bool = True

    # Flight propeller gains are in PRIMARY TUNING. The measured motor/prop
    # lag is roughly 100 ms, so excessive rate gain can become anti-damping.
    # Stance propellers track only the residual moment after leg allocation.

    # Optional model-based stance rate observer. Disabled means the attitude
    # derivative term uses the raw gyro.
    stance_kw_obs_en: bool = False
    stance_kw_obs_k: float = 0.05

    # ===== MIT-style SRB MPC (stance force planning) =====
    # When enabled, MPC replaces the default SRB virtual-spring for stance f_ref computation.
    # MPC plans the full 3D GRF over a horizon to simultaneously achieve:
    #   1. Attitude stabilization (drive angular velocity to zero before liftoff)
    #   2. Vertical trajectory tracking (quintic polynomial)
    #   3. Horizontal velocity regulation
    # Reference: Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3", IROS 2018.
    # WARNING: the MPC model assumes a +Z-UP world; the estimator (p_hat/v_hat) is now
    # +Z-DOWN (NED). Before re-enabling MPC, convert x0/xref signs at the interface.
    use_mpc: bool = False
    # MPC timing
    mpc_dt: float = 0.02          # MPC prediction timestep (s)
    mpc_horizon: int = 15         # prediction horizon steps (15 × 0.02 = 0.30s) better covers stance for no-prop balancing

    # MPC state weights Q (per timestep)
    # State: [px, py, pz, vx, vy, vz, roll, pitch, yaw, ωx, ωy, ωz, yaw_ref]
    mpc_w_px: float = 0.0         # no XY position tracking (Raibert handles it)
    mpc_w_py: float = 0.0
    mpc_w_pz: float = 500.0
    mpc_w_vx: float = 0.0          # 1D: no horizontal tracking
    mpc_w_vy: float = 0.0
    mpc_w_vz: float = 50.0
    mpc_w_roll: float = 0.0
    mpc_w_pitch: float = 0.0
    mpc_w_yaw: float = 0.0
    mpc_w_wx: float = 0.0
    mpc_w_wy: float = 0.0
    mpc_w_wz: float = 0.0

    # MPC input weights R (force regularization)
    # no-prop mode: keep regularization low so force follows objective instead of artificial smoothness.
    mpc_alpha_u: float = 2e-4

    # MPC decimation: run MPC every N control steps (500Hz/N).
    # MIT runs MPC at 30-50 Hz; default 10 → 50 Hz (was 5→100Hz, too fast → fx oscillation).
    # Between solves, hold last f_ref.
    mpc_decimation: int = 2
    # Push phase starts after this fraction of stance_T.
    # Lower value = earlier vertical impulse build-up (more liftoff speed for short contacts).
    mpc_push_start_ratio: float = 0.55
    # MPC state-input gyro conditioning (for x0[omega]):
    # DISABLED 2026-07-11 per user: gyro 不要滤波.
    # Use a dedicated LPF/clip so optimizer is not driven by raw IMU high-frequency noise.
    mpc_omega_lpf_tau: float = 0.0
    mpc_omega_xy_clip_radps: float = 2.2
    # If MPC is enabled, keep stance in pure MPC->SRB-QP path:
    # - on solve failure, hold cached MPC force if available
    # - avoid SRB default fallback (hold MPC solution)
    mpc_hold_cache_on_fail: bool = True

    # MPC constraints
    mpc_mu: float = 0.4           # friction coefficient
    mpc_fz_min: float = 45.0
    mpc_fz_max: float = 200.0
    mpc_fxy_max: float = 80.0     # relax horizontal bound; friction constraint remains the main limiter
    mpc_fxy_lpf_alpha: float = 1.0

    # ===== LiDAR odometry fusion (Mid-360 / Point-LIO via hopper_odom_lcmt) =====
    # The LiDAR gives a DRIFT-FREE global reference; the fast loop stays
    # IMU+leg-kinematics. Fusion is a slow complementary pull, NOT a replacement:
    #   - XY position: _p_hat_w slowly pulled to the lidar position
    #   - yaw: a world-frame yaw offset (applied on top of the IMU quaternion)
    #     slowly tracks the lidar yaw, so the core world frame CONVERGES to the
    #     lidar map frame (patrol waypoints/velocities are in that map frame).
    #   - z and velocity are NOT touched (stance leg-kinematics z and the
    #     velocity KF are the control-critical fast estimates).
    # Fusion silently pauses (pure dead-reckoning) when odom is stale/degraded.
    lidar_fuse_en: bool = True
    lidar_pos_tau_s: float = 0.7    # XY pull time constant (s)
    lidar_yaw_tau_s: float = 2.0    # yaw-offset pull time constant (s)
    lidar_stale_s: float = 0.4      # ignore odom older than this (wall clock, s)
    lidar_pos_init_snap_m: float = 1e9  # first healthy fix snaps XY (no slow pull from 0)

class ModeECore:
    """
    Pure controller core (no LCM, no MuJoCo).

    Inputs:
      - joint_pos, joint_vel: delta motor angles and velocities (3,)
      - imu_*: gyro/acc (body frame), optional quat (wxyz)
      - desired_v_xy: desired world velocity [vx, vy]

    Outputs:
      - tau_cmd (3,) motor torques (Nm)
      - pwm_us (6,) prop PWM microseconds
      - debug/info dict (phases, estimates, etc)
    """

    def __init__(self, cfg: ModeEConfig):
        self.cfg = cfg
        self.dt = float(self.cfg.dt)
        self.mass = float(cfg.mass_kg)
        self.gravity = float(cfg.gravity)
        self.com_b = np.asarray(cfg.com_b, dtype=float).reshape(3)
        self.I_body = np.diag(np.asarray(cfg.I_body_diag, dtype=float).reshape(3))

        # Body frame = leg FK frame = IMU frame (FRD: +X fwd, +Y right, +Z down).

        # Leg kinematics backend selection
        self._leg_model = str(getattr(cfg, "leg_model", "delta")).strip().lower()
        if self._leg_model not in ("delta", "serial"):
            self._leg_model = "delta"

        # delta kinematics (real robot)
        self.fk = ForwardKinematics() if self._leg_model == "delta" else None
        self.kin = InverseJacobian(use_simulink_filter=False, forgetting_factor=0.95, dt=float(self.dt)) if self._leg_model == "delta" else None

        # For serial MuJoCo leg, override l0 to match the model geometry (so TD/LO detection works).
        # When roll=pitch=0 and shift=0 (joint lower limit), the foot is at:
        #   z = -(serial_hip_z_off_m + serial_foot_z_m) in base frame, so ||foot|| ≈ serial_hip_z_off_m + serial_foot_z_m.
        if self._leg_model == "serial":
            try:
                l0_ser = float(abs(float(self.cfg.serial_hip_z_off_m)) + abs(float(self.cfg.serial_foot_z_m)))
                if l0_ser > 1e-6:
                    self.cfg.leg_l0_m = l0_ser
            except Exception:
                pass
            # Serial plant uses a PRISMATIC "shift" joint, so the 3rd actuator command is a generalized FORCE (N),
            # not a torque (Nm). The default (20Nm) is far too small to support the robot weight in MuJoCo.
            # We therefore boost ONLY the 3rd limit when it looks like a real-robot torque tuple was provided.
            try:
                tmax = np.asarray(self.cfg.tau_cmd_max_nm, dtype=float).reshape(3)
                if float(abs(tmax[2])) < 200.0:
                    self.cfg.tau_cmd_max_nm = (float(tmax[0]), float(tmax[1]), 2500.0)
            except Exception:
                pass

        # MATLAB-style EMA on CAN-reported qd (qd_ema_lambda).
        self._qd_ema = np.zeros(3, dtype=float)
        self._qd_ema_init: bool = False
        # Previous hop's measured flight duration [s] (LO->TD), used to
        # correct the ballistic time-to-touchdown prediction that gates
        # the flight tilt budget and descent brake (0 = unknown).
        self._flight_dur_prev: float = 0.0
        # Flight-phase XY velocity latched at liftoff (push tail mean).
        self._flight_vel = np.zeros(3, dtype=float)
        # Per-stance push-phase ring buffer (last N samples for XY velocity).
        _tail_n = int(max(1, int(getattr(self.cfg, "vel_push_tail_n", 10))))
        self._vel_push_tail_n = _tail_n
        self._push_vel_ring = np.zeros((_tail_n, 3), dtype=float)
        self._push_vel_ring_i: int = 0
        self._push_vel_ring_cnt: int = 0
        # Mode1 push-spring state (per-stance parts reset at touchdown;
        # _mode1_Eloss persists across hops -- it is the apex adaptation).
        self._mode1_push_latched: bool = False
        self._mode1_push_confirm_count: int = 0
        self._mode1_k_comp: float | None = None
        self._mode1_k_boost: float = 0.0
        self._mode1_v_td: float = 0.0
        self._mode1_x0: float = 0.0
        self._mode1_boost_f_state: float = 0.0
        # LPF'd world-Z up-velocity used only inside the stance-Z law.
        self._mode1_vz_lpf: float | None = None
        self._mode1_Eloss: float = 0.0
        # "auto" reverse-policy hysteresis latch (see _allocate_prop_thrust).
        self._prop_rev_on: bool = False
        # Fraction of the demanded attitude torque the decoupled allocator
        # actually delivered last tick (1.0 = unclipped; telemetry).
        self._prop_att_scale: float = 1.0
        # PogoX flight velocity convergence state (see flight_vel_* config):
        # slew-limited tilt reference (2-vector: direction+angle) and logs.
        self._fl_tilt_vec = np.zeros(2, dtype=float)
        self._fl_tilt_cmd_deg: float = 0.0
        self._fl_zb_des_xy = np.zeros(2, dtype=float)
        # Horizontal-convergence telemetry: flight velocity error norm
        # and the DELIVERED lateral force f_z*tan(tilt) [N].
        self._fl_ev_xy: float = 0.0
        self._fl_lat_force_n: float = 0.0
        # Propeller PUSH energy supplement latched for the current stance (N).
        self._prop_energy_fz: float = 0.0
        # FB-SLIP stance state (37a1475 port; sized at TD, consumed per tick).
        self._fb_x_td: float = 0.0          # TD preload depth x_td [m]
        self._fb_s_tgt: float = 0.0         # design-map target travel [m]
        self._fb_trav_plan: float = 0.0     # planned max travel [m]
        self._fb_xz_max: float = 0.0        # deepest x_z this stance [m]
        self._fb_push_f: float = 0.0        # latched constant push force [N]
        self._fb_push_taper: float = 0.0    # latched liftoff taper span [m]
        # POGOX energy-shaping: last APPLIED stance force for the
        # model-based vz lag compensation (None = fresh stance).
        self._px_f_prev: float | None = None
        # POGOX axial (encoder) leg-rate: q_shift history + LPF state.
        # World-kinematic vz is invalid once the leg unloads (foot slip /
        # body pivot read as -2.5 m/s "fall", log 105750) -- the pump must
        # see only the encoder-level axial rate.
        self._px_qs_prev: float | None = None
        self._px_vz_ax: float | None = None
        # Last pump-term value for the per-tick slew limit (N).
        self._px_pump_prev: float = 0.0
        self._fb_fcomp_lpf: float | None = None
        # RB gamepad big-jump request (armed until consumed at a PUSH latch).
        self._big_jump_pending: bool = False
        # NRC stance energy law state/telemetry (see nrc_* config):
        # one-hop height gain (big jump) + last norm radius vs target.
        self._nrc_big_gain: float = 1.0
        self._nrc_r: float = 0.0
        self._nrc_r_star: float = 0.0
        self._nrc_f_des: float = 0.0
        # Per-hop apex return-map trim on the NRC height target
        # (see nrc_apex_trim_gain): h_tgt_eff = hop_height * trim.
        # Start above 1.0 so cold-start hops are not forced low while
        # the return map climbs (nrc_apex_trim_init).
        self._nrc_h_trim: float = float(_clipf(
            float(getattr(self.cfg, "nrc_apex_trim_init", 1.0)),
            float(self.cfg.nrc_apex_trim_min),
            float(self.cfg.nrc_apex_trim_max),
        ))
        # Runtime compat: lcm_controller sets this after the RT transition so
        # the first-flight apex/eta estimate skips one corrupted arc.
        self._eta_skip_once: bool = False
        # Stance kW rate observer state (see stance_kw_obs_* config):
        # roll/pitch rate estimate + the body attitude torque commanded LAST tick.
        self._kw_obs_w = np.zeros(2, dtype=float)
        self._kw_obs_tau_prev = np.zeros(2, dtype=float)
        self._kw_obs_init: bool = False

        # Runtime prop armed state (gamepad A/B switch, fed by the LCM layer via
        # set_props_armed every cycle). Replaces the deleted mode-1/pure_leg_mode:
        # False = pure-leg behavior (no prop demands, true g_eff, liftoff omega
        # gate). Defaults True so headless/sim use without the LCM layer keeps
        # normal standalone behavior.
        self._props_armed_rt: bool = True

        # ===== MIT SRB MPC (stance force planning) =====
        if bool(cfg.use_mpc):
            from .controllers.mpc import MITCondensedGRFMPC, MITCondensedGRFMPCConfig
            self._mpc = MITCondensedGRFMPC(MITCondensedGRFMPCConfig(
                dt=float(cfg.mpc_dt),
                N=int(cfg.mpc_horizon),
                mu=float(cfg.mpc_mu),
                fz_min=float(cfg.mpc_fz_min),
                fz_max=float(cfg.mpc_fz_max),
                fxy_max=float(cfg.mpc_fxy_max),
                w_px=float(cfg.mpc_w_px),
                w_py=float(cfg.mpc_w_py),
                w_pz=float(cfg.mpc_w_pz),
                w_vx=float(cfg.mpc_w_vx),
                w_vy=float(cfg.mpc_w_vy),
                w_vz=float(cfg.mpc_w_vz),
                w_roll=float(cfg.mpc_w_roll),
                w_pitch=float(cfg.mpc_w_pitch),
                w_yaw=float(cfg.mpc_w_yaw),
                w_wx=float(cfg.mpc_w_wx),
                w_wy=float(cfg.mpc_w_wy),
                w_wz=float(cfg.mpc_w_wz),
                alpha_u=float(cfg.mpc_alpha_u),
            ))
        else:
            self._mpc = None
        self._mpc_counter: int = 0  # decimation counter
        self._mpc_f_ref_cache: np.ndarray = np.zeros(3, dtype=float)  # cached MPC output
        self._f_ref_z_prev: float = 0.0  # stance fz continuity state
        self._f_ref_xy_prev: np.ndarray = np.zeros(2, dtype=float)  # stance fxy continuity state
        self._mpc_omega_lpf: np.ndarray = np.zeros(3, dtype=float)
        self._mpc_omega_lpf_init: bool = False

        # motor PWM map (thrust->PWM)
        self.use_hopper4_pwm = bool(cfg.use_hopper4_pwm_mapping)
        self.prop_k_thrust = float(cfg.prop_k_thrust)
        if not bool(self.use_hopper4_pwm):
            # Use lookup table (MotorTableModel) when Hopper4 mapping is disabled
            self.motor_table = MotorTableModel.default_from_table()
            # Clamp to FC configured range if needed
            self.motor_table.pwm_min_us = float(cfg.pwm_min_us)
            self.motor_table.pwm_max_us = float(cfg.pwm_max_us)
        else:
            self.motor_table = None

        # attitude estimator
        self.att = SimpleIMUAttitudeEstimator(kp_acc=0.6, acc_g_min=0.90, acc_g_max=1.10)
        # MATLAB com_filter stance-tick counter (accel-correction window gate).
        self._att_stance_tick: int = 0

        # estimator state
        # Leg-kinematics velocity in stance; latched at liftoff for flight XY.
        # Flight Z from raw IMU integration only (no KF / no filter).
        self._v_hat_w = np.zeros(3, dtype=float)
        self._p_hat_w = np.array([0.0, 0.0, float(cfg.hop_z0)], dtype=float)
        self._v_hat_inited = False
        # User override: freeze internal velocity estimate to zero (used to stop drift on demand).
        self._user_zero_vel_hold: bool = False
        # ---- LiDAR odometry fusion state (see lidar_* in ModeEConfig) ----
        # Written by update_lidar_odom() (LCM thread, under the runner's lock);
        # read in step(). Wall-clock stamped for the staleness gate.
        self._lidar_pos_map = np.zeros(3, dtype=float)
        self._lidar_yaw_map: float = 0.0
        self._lidar_quality: int = 0
        self._lidar_rx_walltime: float = 0.0
        self._lidar_pos_inited: bool = False
        # World-frame yaw offset applied on top of the IMU attitude:
        # R_wb_used = Rz(_lidar_yaw_off) @ R_wb_imu. Slowly tracks the lidar
        # yaw so the core world frame converges to the lidar map frame.
        self._lidar_yaw_off: float = 0.0
        self._lidar_fused_n: int = 0

        # phase state
        self.sim_time = 0.0
        self._stance = False
        self._td_t: float | None = None
        self._lo_t: float | None = None
        self._q_shift_td: float | None = None
        self._prev_vz: float | None = None  # up-positive vz for apex crossing

        # apex + swing gating (legacy up-positive vz crossing)
        self._apex_reached = False
        self._z_lo: float | None = None
        self._vz_lo: float | None = None
        self._z_apex_actual: float = float("nan")  # last measured apex h (m, for log)

        # stance reference profile (unified stance: soft landing + push-off, no discrete COMP/PUSH switching)
        self._stance_prof_inited = False
        self._stance_t_comp: float | None = None
        self._stance_depth_tgt_m: float = 0.0
        self._stance_com_off_z: float = 0.0
        # Cached stance reference endpoints for event-based retiming (COM/world-z).
        self._stance_z_end: float | None = None
        # Quintic (minimum-jerk) z(t) coefficients in COM-z (world), used by stance reference generation.
        # poly: z(t) = c0 + c1 t + ... + c5 t^5
        self._stance_poly1: np.ndarray | None = None  # [0, t_comp]
        self._stance_poly2: np.ndarray | None = None  # [t_comp, stance_T]
        self._stance_T1: float = 0.0
        self._stance_T2: float = 0.0
        self._v_to_cmd = float(cfg.v_to_min)

        # last solution hold (robustness)
        self._wbc_last_t = np.zeros(3, dtype=float)
        self._wbc_last_f = np.zeros(3, dtype=float)
        self._tau_cmd_prev = np.zeros(3, dtype=float)

        # precompute prop positions in base frame (physical robot mapping confirmed by user)
        L = float(cfg.prop_arm_len_m)
        #   arm 0: -90 deg,  ( 0,       -L)  -> Motor3/PWM[3]  (2026-07-06 remap)
        #   arm 1: +150 deg, (-sqrt3/2 L, +0.5 L) -> Motor2/PWM[2]
        #   arm 2:  +30 deg, (+sqrt3/2 L, +0.5 L) -> Motor1/PWM[1]
        self.prop_positions_b = np.array(
            [
                [0.0, -1.0 * L, 0.0],
                [-math.sqrt(3) * 0.5 * L, +0.5 * L, 0.0],
                [+math.sqrt(3) * 0.5 * L, +0.5 * L, 0.0],
            ],
            dtype=float,
        )

        # Validate prop PWM mapping (avoid silent duplicates / out-of-range indices)
        try:
            groups = tuple(tuple(int(x) for x in g) for g in cfg.prop_pwm_idx_per_arm)
            flat = [i for g in groups for i in g]
            if (len(groups) != 3) or any(len(g) < 1 for g in groups):
                raise ValueError("prop_pwm_idx_per_arm must be 3 groups, each with >= 1 index")
            if any((i < 0) or (i > 5) for i in flat):
                raise ValueError(f"prop_pwm_idx_per_arm out of range: {groups}")
            if len(set(flat)) != len(flat):
                raise ValueError(f"prop_pwm_idx_per_arm has duplicate indices: {groups}")
            self._prop_pwm_groups = groups
        except Exception:
            # Fallback to 3 motors on indices 0/1/2 (safe-ish default)
            self._prop_pwm_groups = ((0,), (1,), (2,))

        # 3-RSR torque map workspace clamp (same as MuJoCo demo)
        self._delta_ws = dict(xy=0.27, z_min=0.22, z_max=0.49, z_off=0.03)

        # ===== DEBUG TOGGLES (stance force isolation) =====
        # Set via env vars at launch; default OFF. Used to debug the stance fz/fxfy decoupling.
        #   MODEE_DBG_STANCE_ZERO_FXFY=1 -> kill stance horizontal force (Tau_des[xy]=0, f_ref[xy]=0)
        #   MODEE_DBG_STANCE_ZERO_FZ=1   -> keep fz in the QP (friction-cone budget so fxy/attitude
        #                                   can be generated), but zero the fz component ONLY when
        #                                   the solved contact force is mapped to leg joint torque.
        #                                   (Leg outputs attitude torque only; no vertical push.)
        #   MODEE_DBG_STANCE_FLIP_FZ=1   -> negate stance vertical force reference f_ref[2]
        self._dbg_stance_zero_fxfy = (os.environ.get("MODEE_DBG_STANCE_ZERO_FXFY", "0") == "1")
        self._dbg_stance_zero_fz = (os.environ.get("MODEE_DBG_STANCE_ZERO_FZ", "0") == "1")
        self._dbg_stance_flip_fz = (os.environ.get("MODEE_DBG_STANCE_FLIP_FZ", "0") == "1")

        # ===== HARDCODED DEBUG (2026-07-04): output-stage force gating =====
        # Everything upstream (force computation, logs, props) stays untouched;
        # the gate is applied ONLY at the final force->torque output:
        #   - stance_force_zero_out: commanded leg torque in stance is ZEROED
        #   - stance_fxy_only_out:   stance leg outputs ONLY the attitude fxy force;
        #                            the vertical (body-z spring/push) component is
        #                            zeroed before the force->torque mapping
        #   - flight_fxy_only_out:   flight leg outputs only the XY swing force
        # Set all to False to restore normal operation.
        # 2026-07-04 (evening): debug phase finished -- ALL gates OFF, full force
        # output restored in both stance and flight.
        self._dbg_stance_force_zero_out = False
        self._dbg_stance_fxy_only_out = False
        self._dbg_flight_fxy_only_out = False

    def user_reset(self) -> None:
        """
        User-requested reset (triggered by gamepad Y on the PC side).

        Purpose:
        - Zero drifting estimator/integrator states so a new experiment/log segment starts clean.
        - Keep the controller running; do NOT change driver mode here.
        """
        # Estimator/integrator states
        self._v_hat_w[:] = 0.0
        self._qd_ema[:] = 0.0
        self._qd_ema_init = False
        self._flight_dur_prev = 0.0
        self._flight_vel[:] = 0.0
        self._push_vel_ring[:] = 0.0
        self._push_vel_ring_i = 0
        self._push_vel_ring_cnt = 0
        self._mode1_push_latched = False
        self._mode1_push_confirm_count = 0
        self._mode1_k_comp = None
        self._mode1_k_boost = 0.0
        self._mode1_v_td = 0.0
        self._mode1_x0 = 0.0
        self._mode1_boost_f_state = 0.0
        self._mode1_vz_lpf = None
        self._mode1_Eloss = 0.0
        self._fb_x_td = 0.0
        self._fb_s_tgt = 0.0
        self._fb_trav_plan = 0.0
        self._fb_xz_max = 0.0
        self._fb_push_f = 0.0
        self._fb_push_taper = 0.0
        self._px_f_prev = None
        self._px_qs_prev = None
        self._px_vz_ax = None
        self._px_pump_prev = 0.0
        self._fb_fcomp_lpf = None
        self._prop_rev_on = False
        self._prop_att_scale = 1.0
        self._fl_tilt_vec[:] = 0.0
        self._fl_tilt_cmd_deg = 0.0
        self._fl_zb_des_xy[:] = 0.0
        self._fl_ev_xy = 0.0
        self._fl_lat_force_n = 0.0
        self._prop_energy_fz = 0.0
        self._big_jump_pending = False
        self._nrc_big_gain = 1.0
        self._nrc_r = 0.0
        self._nrc_r_star = 0.0
        self._nrc_f_des = 0.0
        self._nrc_h_trim = float(_clipf(
            float(getattr(self.cfg, "nrc_apex_trim_init", 1.0)),
            float(self.cfg.nrc_apex_trim_min),
            float(self.cfg.nrc_apex_trim_max),
        ))
        self._eta_skip_once = False
        self._kw_obs_w[:] = 0.0
        self._kw_obs_tau_prev[:] = 0.0
        self._kw_obs_init = False
        self._v_hat_inited = False
        self._prev_vz = None
        self._apex_reached = False
        self._z_lo = None
        self._vz_lo = None
        self._z_apex_actual = float("nan")
        self._f_ref_z_prev = 0.0
        self._f_ref_xy_prev[:] = 0.0
        self._mpc_omega_lpf[:] = 0.0
        self._mpc_omega_lpf_init = False
        # Reset stance reference profile (re-initialize on next touchdown)
        self._stance_prof_inited = False
        self._stance_t_comp = None
        self._stance_depth_tgt_m = 0.0
        self._stance_com_off_z = 0.0
        self._stance_z_end = None
        self._stance_poly1 = None
        self._stance_poly2 = None
        self._stance_T1 = 0.0
        self._stance_T2 = 0.0

        # Rebase XY position for nicer logs (doesn't materially change the control because references are relative)
        self._p_hat_w[0] = 0.0
        self._p_hat_w[1] = 0.0
        # LiDAR fusion: re-snap to the next healthy fix after a rebase (a slow
        # pull from the rebased origin would fight the fresh lidar position).
        self._lidar_pos_inited = False

        # Reset attitude estimator state (only used if use_fc_quat=False)
        try:
            self.att.reset()
        except Exception:
            pass
        self._att_stance_tick = 0
    def user_zero_velocity_hold(self, enable: bool) -> None:
        """
        User-requested "HARD STOP" of the internal velocity estimate.

        When enabled:
        - v_hat is forced to 0 every control step (no IMU integration drift)
        - integrators are kept at 0
        - flight foot placement stops drifting when desired_v==0

        This is a debugging / operator convenience feature (not physically enforcing real velocity to 0).
        """
        self._user_zero_vel_hold = bool(enable)
        if bool(self._user_zero_vel_hold):
            # Make the state look like a fresh start immediately.
            self.user_reset()

    def user_request_big_jump_next_stance(self) -> None:
        """
        Gamepad RB: arm a one-shot BIG JUMP.  The next PUSH latch solves the
        push spring for big_jump_height_gain * hop_height_m (see the energy
        block); the request is consumed there and normal hops resume.
        """
        self._big_jump_pending = True

    def compute_stand_swing_tau(
        self,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        leg_len_des_m: float,
        tau_max_nm: float,
        imu_quat_wxyz: np.ndarray | None = None,
        kp_z: float | None = None,
        kd_z: float | None = None,
        axial_ff_n: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """
        RT/P4 stand torque: same Cartesian swing law as FLIGHT. The target is
        WORLD-vertical (like the flight placement): foot straight below the
        body at leg length `leg_len_des_m`, rotated into the body frame with
        the IMU quaternion (foot_des_b = R_wb^T @ [0,0,+L], world FRD +Z
        down). Without a quaternion it falls back to body-centered [0,0,L].
        Each motor torque is hard-capped to +/-tau_max_nm.

        Force law (native delta frame, +Z down) matches the FLIGHT block:
          sideForce  = swing_kp/kd_xy * (x_des - x), orthogonal to the leg
          springForce = -swing_kp_z*(||x||-L_des)*u - swing_kd_z*v_axial

        SLIP-style loaded stand: with the default flight gains the body
        weight sags the leg; pass stiffer axial gains via kp_z/kd_z and the
        weight the LEG must carry via axial_ff_n (N, pushes the foot outward
        along the leg axis) so the PD only handles the residual and L_des is
        truly held.

        Returns:
          tau: (3,) motor torques (Nm)
          err_m: ||x_des - x|| foot position error (m)
          speed_mps: ||xdot|| foot speed (m/s)
        """
        q = np.asarray(joint_pos, dtype=float).reshape(3)
        qd = np.asarray(joint_vel, dtype=float).reshape(3)
        L_des = float(max(0.05, float(leg_len_des_m)))
        cap = float(abs(float(tau_max_nm)))
        x_des = np.array([0.0, 0.0, L_des], dtype=float)
        if imu_quat_wxyz is not None:
            quat = np.asarray(imu_quat_wxyz, dtype=float).reshape(4)
            if np.all(np.isfinite(quat)) and float(np.linalg.norm(quat)) > 1e-6:
                R_wb = _quat_to_R_wb(_quat_normalize_wxyz(quat))
                # World-vertical target (FRD, +Z down), rotated into body frame.
                x_des = (
                    np.asarray(R_wb, dtype=float).reshape(3, 3).T
                    @ np.array([0.0, 0.0, L_des], dtype=float)
                ).reshape(3)

        if self._leg_model == "serial":
            foot_b, J_body = self._serial_leg_fk_jac(
                q_roll=float(q[0]), q_pitch=float(q[1]), q_shift=float(q[2]),
            )
            x = _imu_body_to_leg_native(np.asarray(foot_b, dtype=float).reshape(3))
            xdot = _imu_body_to_leg_native(
                (np.asarray(J_body, dtype=float).reshape(3, 3) @ qd.reshape(3)).reshape(3)
            )
        else:
            if self.fk is None or self.kin is None:
                return np.zeros(3, dtype=float), 1e9, 1e9
            foot, _ = self.fk.forward_kinematics(q)
            x = np.asarray(foot, dtype=float).reshape(3)
            J_inv_raw, _ = self.kin.inverse_jacobian(x, qd, theta=None)
            J_inv = np.asarray(J_inv_raw, dtype=float).reshape(3, 3)
            xdot = (self._stable_inv3(J_inv) @ qd.reshape(3)).reshape(3).astype(float)

        err_m = float(np.linalg.norm(x_des - x))
        speed_mps = float(np.linalg.norm(xdot))

        leg_length = float(np.linalg.norm(x))
        if leg_length < 1e-6:
            unitSpring = np.array([0.0, 0.0, 1.0], dtype=float)
            leg_length = 0.0
        else:
            unitSpring = (x / leg_length).astype(float)
        springVel = (float(np.dot(xdot, unitSpring)) * unitSpring).astype(float)

        Khp = float(self.cfg.swing_kp_xy)
        Khd = float(self.cfg.swing_kd_xy)
        sideForce = (Khp * (x_des - x) - Khd * xdot).astype(float)
        sideForce = (sideForce - float(np.dot(sideForce, unitSpring)) * unitSpring).astype(float)

        k = float(self.cfg.swing_kp_z) if kp_z is None else float(kp_z)
        b = float(self.cfg.swing_kd_z) if kd_z is None else float(kd_z)
        force_scalar = -float(k) * float(leg_length - L_des) + float(axial_ff_n)
        springForce = (force_scalar * unitSpring - float(b) * springVel).astype(float)
        f_native = (sideForce + springForce).astype(float)

        try:
            if self._leg_model == "serial":
                f_b = _leg_native_to_imu_body(f_native)
                tau = (np.asarray(J_body, dtype=float).reshape(3, 3).T @ f_b.reshape(3)).reshape(3)
            else:
                inv_Jt = self._stable_inv3(J_inv.T)
                tau = (inv_Jt @ f_native.reshape(3)).reshape(3)
            tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
            tau = (tau_sign.reshape(3) * tau.reshape(3)).reshape(3).astype(float)
        except Exception:
            return np.zeros(3, dtype=float), err_m, speed_mps

        if cap > 0.0:
            tau = np.clip(tau, -cap, cap).astype(float)
        return tau, err_m, speed_mps

    def update_lidar_odom(
        self,
        *,
        pos_map: np.ndarray,
        yaw_map: float,
        quality: int,
        rx_walltime: float,
    ) -> None:
        """
        Feed one LiDAR odometry sample (hopper_odom_lcmt, already in the hopper
        map frame: +Z DOWN, FRD body). Called from the LCM thread under the
        runner's lock; the actual fusion happens inside step().

          pos_map:     body position in map frame (m)
          yaw_map:     body yaw in map frame (rad, aerospace ZYX)
          quality:     1 = healthy (fuse), 0 = degraded (ignore)
          rx_walltime: time.time() at receive (staleness gate in step())
        """
        p = np.asarray(pos_map, dtype=float).reshape(3)
        if not (np.all(np.isfinite(p)) and np.isfinite(float(yaw_map))):
            return
        self._lidar_pos_map = p.copy()
        self._lidar_yaw_map = float(yaw_map)
        self._lidar_quality = int(quality)
        self._lidar_rx_walltime = float(rx_walltime)

    def set_props_armed(self, armed: bool) -> None:
        """Runtime prop armed state (gamepad A/B). False = pure-leg behavior:
        no prop force/torque demands, un-assisted g_eff, liftoff omega gate."""
        self._props_armed_rt = bool(armed)

    @staticmethod
    def _pinv_ridge(A: np.ndarray, lambda_rel: float) -> np.ndarray:
        """
        Damped least-squares (ridge) pseudo-inverse:
          A^+ = (A^T A + λ^2 I)^(-1) A^T
        with λ = lambda_rel * ||A||_F.

        This is a small, dependency-free way to prevent Jacobian inversions from exploding when A
        becomes ill-conditioned (common for delta/3-RSR near workspace edges).
        """
        A = np.asarray(A, dtype=float)
        if A.shape != (3, 3):
            A = A.reshape(3, 3)
        lam_rel = float(max(0.0, float(lambda_rel)))
        if lam_rel <= 0.0:
            # Least-squares pseudo-inverse (still better than hard crash if singular)
            return np.linalg.pinv(A)
        fro = float(np.sqrt(float((A * A).sum())))
        lam = float(lam_rel * max(1e-12, fro))
        M = (A.T @ A + (lam * lam) * np.eye(3, dtype=float)).astype(float)
        M_inv = _inv3(M)
        if M_inv is not None:
            return (M_inv @ A.T).astype(float)
        try:
            return np.linalg.solve(M, A.T).astype(float)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)

    def _stable_inv3(self, A: np.ndarray) -> np.ndarray:
        """
        Robust inverse for 3x3 matrices used in delta kinematics:
        - If DLS is enabled, return ridge pseudo-inverse (stable near singularities).
        - Else, try exact inverse and fall back to pinv.
        """
        A = np.asarray(A, dtype=float).reshape(3, 3)
        if bool(getattr(self.cfg, "delta_jacobian_dls_enable", True)):
            lam_rel = float(getattr(self.cfg, "delta_jacobian_dls_lambda_rel", 0.0))
            return self._pinv_ridge(A, lam_rel)
        try:
            return np.linalg.inv(A).astype(float)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)

    def _allocate_prop_thrust(
        self,
        *,
        tau_des_w: np.ndarray,
        prop_r_w: np.ndarray,
        z_thrust_w: np.ndarray,
        thrust_sum_ref: float,
        thrust_sum_max: float,
        reverse_policy: str = "bidir",
    ) -> np.ndarray:
        """
        Tri-rotor thrust allocation: idle-base DIFFERENTIAL attitude with a
        bounded spool-up (2026-08-01, user: "propeller 在 energy 能量补充的
        时候才加 Fz, 其他时候就像 3rotor 文章一样差值控制姿态, 不能叠加
        很多 pwm"; 2026-08-02 the idle base moved 1100 -> ~1250 us for
        real down-headroom, see prop_base_thrust_ratio).

        Two channels compose every arm command
            t_i = T/3 + c + s * Delta_i :

          [Fz]  COLLECTIVE  T = clip(thrust_sum_ref, thrust_sum_max).
                T is the idle base (prop_base_thrust_ratio, ~PWM 1250)
                for the whole hop cycle; the ONLY planned addition is the
                PUSH energy supplement (prop_energy_fz).  No
                attitude-driven Fz plan anywhere.

          [tau] DIFFERENTIAL Delta = min-norm solution of
                M[:2] Delta = tau_xy, projected zero-sum (a pure moment
                for the symmetric Y-frame).  At the idle base the down-
                headroom (base - t_min) is ~0.1 N, so realizing torque
                REQUIRES the minimal collective lift c(s) that keeps the
                floored arm at t_min while the demanding arms rise above
                the base -- standard low-idle tri-rotor mixing.  The lift
                is the SMALLEST that makes s*Delta feasible, and every
                arm is HARD-CAPPED at
                    t_ceil = base + prop_att_thrust_max_each_n
                so a large attitude demand can never stack the PWM
                ("不能叠加很多 pwm"; 3 N cap = PWM ~1380 per arm).

        Saturation: torque realized as s*tau_des with the LARGEST s in
        [0,1] under floor/ceiling/sum caps -- one scalar on the whole
        differential, moment DIRECTION exact (prioritized mixing, cf.
        Faessler et al. RA-L 2017; Johansen & Fossen, Automatica 2013).

        reverse_policy (floor selection only, solver identical):
          "fwd"   forward-only floor (never cross the 1000 us stop).
          "bidir" reverse floor always available (stance downforce path).
          "auto"  forward floor first; open the reverse floor only while
                  the forward solution cannot realize the torque (s < 1),
                  with hysteresis so the low arm does not chatter across
                  the 1000 us stop.  Feasibility-triggered, not angle-
                  triggered.
        """
        tau_des_w = np.asarray(tau_des_w, dtype=float).reshape(3)
        prop_r_w = np.asarray(prop_r_w, dtype=float).reshape(3, 3)
        z_thrust_w = np.asarray(z_thrust_w, dtype=float).reshape(3)
        M_prop = np.column_stack([
            _cross3(prop_r_w[i].reshape(3), z_thrust_w.reshape(3)) for i in range(3)
        ]).astype(float)
        thrusts_att = _lstsq_minnorm(M_prop[:2, :], tau_des_w[:2]).astype(float)
        # Zero-sum projection: the differential itself is a pure moment;
        # only the explicit lift c below may move the total thrust.
        thrusts_att = (thrusts_att - float(np.mean(thrusts_att))).astype(float)
        t_max = float(self.cfg.thrust_max_each_n)
        tsum_cap = float(max(0.0, float(thrust_sum_max)))
        T_col = float(thrust_sum_ref)
        if tsum_cap > 1e-9:
            T_col = min(T_col, tsum_cap)
        base_each = min(T_col / 3.0, t_max)
        # Per-arm ceiling for the attitude channel: base + spool-up cap.
        att_cap = float(max(0.0, float(getattr(
            self.cfg, "prop_att_thrust_max_each_n", 3.0
        ))))
        # Stance and flight both pass a Tau_prop_des into this allocator;
        # the per-arm att_cap bounds the differential in both phases.
        t_ceil = min(t_max, base_each + att_cap)

        a = [float(thrusts_att[i]) for i in range(3)]
        a_min = min(a)
        a_max = max(a)

        def _solve(t_min: float) -> tuple[np.ndarray, float]:
            # Largest s in [0,1] such that t_i = base + c(s) + s*a_i is
            # feasible, where c(s) = max(0, t_min - (base + s*a_min)) is
            # the MINIMAL collective lift keeping the floored arm at
            # t_min.  Ceiling and sum caps bound the spool-up.
            s = 1.0
            if a_max > 1e-9:
                s = min(s, max(0.0, t_ceil - base_each) / a_max)
            if (base_each + s * a_min) < t_min - 1e-12:
                # Lift engaged: base+c+s*a_min == t_min exactly.
                # Ceiling: t_min + s*(a_max - a_min) <= t_ceil.
                # Sum (zero-sum a): 3*t_min - 3*s*a_min <= tsum_cap.
                s = 1.0
                d = a_max - a_min
                if d > 1e-9:
                    s = min(s, max(0.0, t_ceil - t_min) / d)
                if tsum_cap > 1e-9 and a_min < -1e-9:
                    s = min(s, max(0.0, tsum_cap - 3.0 * t_min)
                            / (-3.0 * a_min))
                s = max(0.0, min(1.0, s))
            c = max(0.0, t_min - (base_each + s * a_min))
            return (
                np.array(
                    [base_each + c + s * a[0],
                     base_each + c + s * a[1],
                     base_each + c + s * a[2]],
                    dtype=float,
                ),
                float(s),
            )

        bidir_ok = bool(getattr(self.cfg, "prop_bidir", False))
        t_min_fwd = max(0.0, float(self.cfg.wbc_thrust_min_each_n))
        t_min_rev = -abs(float(self.cfg.prop_reverse_max_n)) if bidir_ok \
            else float(self.cfg.wbc_thrust_min_each_n)
        pol = str(reverse_policy)
        if (not bidir_ok) or pol == "fwd":
            thrusts, s_used = _solve(t_min_fwd if bidir_ok else t_min_rev)
        elif pol == "bidir":
            thrusts, s_used = _solve(t_min_rev)
        else:  # "auto"
            thr_f, s_f = _solve(t_min_fwd)
            if (not self._prop_rev_on) and s_f < 1.0 - 1e-6:
                self._prop_rev_on = True
            if self._prop_rev_on:
                thr_b, s_b = _solve(t_min_rev)
                if float(np.min(thr_b)) >= t_min_fwd - 1e-9:
                    # Demand shrank: bidir solution is already all-forward,
                    # identical to the fwd one -> disengage with no jump.
                    self._prop_rev_on = False
                    thrusts, s_used = thr_f, s_f
                else:
                    thrusts, s_used = thr_b, s_b
            else:
                thrusts, s_used = thr_f, s_f
        # Telemetry: fraction of the demanded attitude torque actually
        # delivered (1.0 = unclipped).  CSV column prop_att_scale.
        self._prop_att_scale = float(s_used)
        return thrusts

    def _pwm_from_arm_thrusts(self, thrusts: np.ndarray) -> np.ndarray:
        """Per-arm thrusts (3, signed N) -> 6 PWM us via prop_pwm_idx_per_arm.

        Hopper4 sqrt law, extended for bidirectional (3D-mode) ESCs:
          thrust > 0: pwm = 1000 + sqrt( thrust / k)   in [1000, pwm_max_us]
          thrust < 0: pwm = 1000 - sqrt(|thrust| / k)  in [pwm_rev_floor_us, 1000]
        (reverse assumed symmetric in k until calibrated; bridge maps pwm<1000
        to the ESC reverse half). With prop_bidir=False negative thrust idles.
        """
        thrusts = np.asarray(thrusts, dtype=float).reshape(3)
        thrust_motor = np.zeros(6, dtype=float)
        for arm_i in range(3):
            idxs = self._prop_pwm_groups[arm_i]
            t_each = float(thrusts[arm_i]) / float(len(idxs))
            for k in idxs:
                thrust_motor[int(k)] = t_each

        def _hopper4_pwm(thrust_i: float) -> float:
            k = float(self.prop_k_thrust)
            stop = float(self.cfg.pwm_min_us)
            if k <= 1e-12 or thrust_i == 0.0:
                return stop
            if thrust_i > 0.0:
                return float(_clipf(stop + math.sqrt(thrust_i / k), stop, float(self.cfg.pwm_max_us)))
            if not bool(getattr(self.cfg, "prop_bidir", False)):
                return stop
            return float(_clipf(stop - math.sqrt(-thrust_i / k), float(self.cfg.pwm_rev_floor_us), stop))

        if bool(self.use_hopper4_pwm) or self.motor_table is None:
            return np.array([_hopper4_pwm(float(thrust_motor[i])) for i in range(6)], dtype=float)
        # MotorTableModel lookup table (forward-only; reverse not in the table)
        return self.motor_table.pwm_from_thrust(thrust_motor).astype(float).reshape(6)

    def prop_reverse_balance_pwm(
        self,
        *,
        imu_rpy: np.ndarray,
        imu_gyro_b: np.ndarray,
        base_pwm_us: float,
    ) -> np.ndarray:
        """Attitude-balance the body with the props around a REVERSE base PWM.

        For the LB switch loop after the fixed reverse spin: each arm idles at
        base_pwm_us (< 1000 = reverse), and the flight attitude PD
        (flight_kR/kW on roll/pitch, level target) adds a differential thrust
        on top, allocated with the same tri-rotor geometry as normal hopping.
        Returns 6 PWM us (bidir sqrt mapping; requires prop_bidir).
        """
        rpy = np.asarray(imu_rpy, dtype=float).reshape(3)
        omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
        # Level-target attitude PD (same gains/cap as the in-flight prop demand).
        tau_b = np.zeros(3, dtype=float)
        tau_b[0] = -float(self.cfg.flight_kR) * float(rpy[0]) - float(self.cfg.flight_kW) * float(omega_b[0])
        tau_b[1] = -float(self.cfg.flight_kR) * float(rpy[1]) - float(self.cfg.flight_kW) * float(omega_b[1])
        cap = float(self.cfg.flight_tau_rp_max)
        n = float(np.hypot(tau_b[0], tau_b[1]))
        if cap > 0.0 and n > cap and n > 1e-9:
            tau_b[:2] *= cap / n
        # Convert the reverse base PWM to a (negative) per-arm base thrust via the
        # inverse sqrt law, then allocate in BODY frame (near-level assumption,
        # attitude torque is a body-frame quantity anyway).
        k = float(self.prop_k_thrust)
        stop = float(self.cfg.pwm_min_us)
        d = float(base_pwm_us) - stop            # < 0 for reverse
        t_base_each = -k * d * d if d < 0.0 else k * d * d
        prop_r_b = (self.prop_positions_b - self.com_b.reshape(1, 3)).astype(float)
        z_thrust_b = np.array([0.0, 0.0, -1.0], dtype=float)
        thrusts = self._allocate_prop_thrust(
            tau_des_w=tau_b,                      # body frame in, body geometry below
            prop_r_w=prop_r_b,
            z_thrust_w=z_thrust_b,
            thrust_sum_ref=3.0 * float(t_base_each),
            thrust_sum_max=float(self.mass * self.gravity * float(self.cfg.thrust_total_ratio_max)),
        )
        return self._pwm_from_arm_thrusts(thrusts)

    def compute_tau_from_force_base(
        self,
        *,
        joint_pos: np.ndarray,
        f_base: np.ndarray,
        use_contact_site_map: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map a desired 3D force in BODY/BASE frame (FRD: +X forward, +Y right, +Z down)
        to delta motor torques using the same Jacobian convention as ModeE.

        Conventions:
        - FK/Jacobian are in the native analytic delta frame (+Z down).
        - IMU-body forces are converted once via `_imu_body_to_leg_native`.
        - Mapping uses: tau = inv(J_inv^T) * f_delta

        Args:
          joint_pos: (3,) motor angles [0,1,2] in physical motor order.
          f_base: (3,) force in BASE frame (FRD, +Z down).
          use_contact_site_map: if True, apply the same 3cm contact-site offset + workspace clamp
            used by the stance mapping (`A_tau_f`). This makes stand-alone force tests match stance.

        Returns:
          tau: (3,) motor torques (Nm) in physical motor order [0,1,2]
          foot_vicon: (3,) FK foot position in delta/vicon frame (+Z down)
        """
        joint_pos = np.asarray(joint_pos, dtype=float).reshape(3)
        f_base = np.asarray(f_base, dtype=float).reshape(3)

        if self._leg_model == "serial":
            foot_b, J_body = self._serial_leg_fk_jac(
                q_roll=float(joint_pos[0]),
                q_pitch=float(joint_pos[1]),
                q_shift=float(joint_pos[2]),
            )
            foot_vicon = _imu_body_to_leg_native(np.asarray(foot_b, dtype=float).reshape(3))
            tau = (np.asarray(J_body, dtype=float).reshape(3, 3).T @ f_base.reshape(3)).reshape(3)
            tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
            tau = (tau_sign.reshape(3) * tau.reshape(3)).reshape(3).astype(float)
            return tau, foot_vicon

        # delta (real robot)
        if self.fk is None or self.kin is None:
            raise RuntimeError("delta leg model requested but kinematics is not initialized")

        foot_vicon, _ = self.fk.forward_kinematics(joint_pos)
        foot_vicon = np.asarray(foot_vicon, dtype=float).reshape(3)

        x3 = foot_vicon.copy()
        if bool(use_contact_site_map):
            x3[2] = float(x3[2] + float(self._delta_ws["z_off"]))
            x3[0] = float(_clipf(x3[0], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
            x3[1] = float(_clipf(x3[1], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
            x3[2] = float(_clipf(x3[2], float(self._delta_ws["z_min"]), float(self._delta_ws["z_max"])))

        # Compute inverse Jacobian at x3 (delta/vicon frame)
        J_inv_map, _ = self.kin.inverse_jacobian(x3, np.zeros(3, dtype=float), theta=None)
        J_inv_map = np.asarray(J_inv_map, dtype=float).reshape(3, 3)
        inv_Jt = self._stable_inv3(J_inv_map.T)

        f_native = _imu_body_to_leg_native(f_base.reshape(3))

        # tau = inv(J_inv^T) * f_native
        tau = (inv_Jt @ f_native.reshape(3)).reshape(3)

        # Motor wiring/driver sign override
        tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
        tau = (tau_sign.reshape(3) * tau.reshape(3)).reshape(3).astype(float)
        return tau, foot_vicon

    def _serial_leg_fk_jac(self, *, q_roll: float, q_pitch: float, q_shift: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Serial-equivalent leg kinematics for MuJoCo `hopper_serial.xml`.

        Joint order:
          q = [roll, pitch, shift]
        Base frame:
          +X forward, +Y left, +Z up

        Returns:
          foot_b: (3,) foot origin position in base frame
          J_body: (3,3) Jacobian mapping qdot -> foot_vrel_b in base frame
        """
        # Geometry from MJCF:
        # base_link -> hip origin offset: z = -serial_hip_z_off_m
        p0 = np.array([0.0, 0.0, -float(self.cfg.serial_hip_z_off_m)], dtype=float)
        foot_z = float(self.cfg.serial_foot_z_m)

        # Rotation: roll about +X, pitch about +Y
        cr = float(np.cos(q_roll)); sr = float(np.sin(q_roll))
        cp = float(np.cos(q_pitch)); sp = float(np.sin(q_pitch))
        Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        Rp = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        R = (Rr @ Rp).astype(float)

        # Prismatic axis is +Z in the roll/pitch frame; q_shift increases SHORTENING the leg.
        v = np.array([0.0, 0.0, float(q_shift) - float(foot_z)], dtype=float)
        foot_rel = (R @ v.reshape(3)).reshape(3)
        foot_b = (p0 + foot_rel).reshape(3)

        # Jacobian columns:
        axis_roll = np.array([1.0, 0.0, 0.0], dtype=float)
        axis_pitch = (Rr @ np.array([0.0, 1.0, 0.0], dtype=float).reshape(3)).reshape(3)
        axis_shift = R[:, 2].reshape(3)
        J0 = _cross3(axis_roll, foot_rel)
        J1 = _cross3(axis_pitch, foot_rel)
        J2 = axis_shift
        J_body = np.stack([J0, J1, J2], axis=1).astype(float)
        return foot_b, J_body

    @staticmethod
    def _quintic_coeff(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
        """
        Quintic polynomial coefficients for minimum-jerk interpolation:
          p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
        satisfying (p,v,a) at t=0 and t=T.
        """
        T = float(max(1e-6, float(T)))
        p0 = float(p0); v0 = float(v0); a0 = float(a0)
        p1 = float(p1); v1 = float(v1); a1 = float(a1)
        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        M = np.array(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ],
            dtype=float,
        )
        b = np.array(
            [
                p1 - (c0 + c1 * T + c2 * T**2),
                v1 - (c1 + 2 * c2 * T),
                a1 - (2 * c2),
            ],
            dtype=float,
        )
        c3, c4, c5 = [float(x) for x in np.linalg.solve(M, b)]
        return np.array([c0, c1, c2, c3, c4, c5], dtype=float)

    @staticmethod
    def _quintic_eval(c: np.ndarray, t: float) -> tuple[float, float, float]:
        """Evaluate quintic polynomial (pos, vel, acc) at time t."""
        c = np.asarray(c, dtype=float).reshape(6)
        t = float(max(0.0, float(t)))
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        c0, c1, c2, c3, c4, c5 = [float(x) for x in c]
        p = c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5
        v = c1 + 2 * c2 * t + 3 * c3 * t2 + 4 * c4 * t3 + 5 * c5 * t4
        a = 2 * c2 + 6 * c3 * t + 12 * c4 * t2 + 20 * c5 * t3
        return float(p), float(v), float(a)

    @staticmethod
    def _smoothstep01(x: float) -> float:
        """C1 smooth step from 0->1 for x in [0,1]."""
        x = float(_clipf(float(x), 0.0, 1.0))
        return float(x * x * (3.0 - 2.0 * x))

    def _push_vel_tail_mean(self) -> np.ndarray | None:
        """Mean of the last vel_push_tail_n stance leg-kinematics samples
        (MATLAB-style avg_foot_vel rolling window).  Used only as the
        flight_vel initial condition at liftoff."""
        n = int(self._push_vel_ring_cnt)
        if n <= 0:
            return None
        cap = int(self._vel_push_tail_n)
        if n < cap:
            return np.mean(self._push_vel_ring[:n, :], axis=0)
        return np.mean(self._push_vel_ring, axis=0)

    def _init_unified_stance_profile(
        self,
        *,
        R_wb: np.ndarray,
        z_td_base: float,
        vz_td: float,
        q_shift_td: float,
    ) -> None:
        """
        Initialize a single, smooth stance COM-z reference curve:
          (z_td, vz_td) -> (z_min, 0) -> (z_end, v_to_cmd)

        IMPORTANT: this profile works in HEIGHT coordinates (up-positive):
          z_td_base = height of base above ground (m, > 0)
          vz_td     = upward vertical velocity (m/s, negative when falling into TD)
        Callers must convert from the +Z-down world (height = -p_z, v_up = -v_z).
        """
        cfg = self.cfg
        R_wb = np.asarray(R_wb, dtype=float).reshape(3, 3)
        z_td_base = float(z_td_base)
        vz_td = float(vz_td)
        q_shift_td = float(q_shift_td)

        # Time budget
        T = float(max(float(cfg.stance_min_T), float(cfg.stance_T)))

        # COM offset ABOVE the base origin (height coords, up-positive).
        # (R_wb @ com_b)[2] is the world +Z-down offset, so negate it.
        com_off_z = float(-(R_wb @ self.com_b.reshape(3))[2])
        self._stance_com_off_z = float(com_off_z)

        # Touchdown COM-z reference origin
        z0 = float(z_td_base + com_off_z)

        # Estimate COM-z at "nominal leg length" (q_shift=0) for the end of stance reference.
        # We treat this as the nominal liftoff height. (The robot may liftoff earlier in practice.)
        z_end = float((z_td_base - q_shift_td) + com_off_z)

        # Desired takeoff speed (already computed at touchdown; clamp for safety)
        # NOTE: v_to_min is a legacy guard; we will still allow v_to to be reduced for feasibility
        # if the compression/extension distance is insufficient.
        v_to = float(_clipf(float(self._v_to_cmd), float(cfg.v_to_min), float(cfg.v_to_max)))

        # Adaptive compression depth/time from touchdown vertical speed (soft landing),
        # BUT also ensure the push segment is not forced to command an extra downward motion.
        #
        # Key insight:
        #   If the stance reference constrains BOTH end position z_end (near nominal leg length)
        #   AND a large end velocity v_to over a long push duration, then a smooth polynomial
        #   will inevitably go DOWN first (negative vz) to satisfy the boundary conditions.
        #   That makes the allocator reduce vertical support (< mg), so the robot collapses instead of pushing off.
        #
        # We avoid this by choosing t_comp long enough (and thus depth large enough) that the remaining
        # extension distance dz = z_end - z_min can support a non-negative velocity profile up to v_to.
        v_in = float(max(0.0, -vz_td))
        a_max = float(max(1e-3, float(cfg.soft_land_a_max)))
        # Minimum time to brake the measured inbound speed under decel limit.
        t_comp_decel = float(v_in / a_max) if v_in > 1e-9 else 0.0
        # Pre-compression at touchdown (q_shift<0 means already shorter than nominal).
        precomp = float(max(0.0, -q_shift_td))
        # Additional requirement so push can be "mostly upward" (no extra downward motion):
        # Use a simple displacement lower bound for non-negative velocity:
        #   dz >= 0.5 * v_to * T2  where T2 = T - t_comp
        # with dz = (z_end - z_min) = depth + precomp (COM offset cancels).
        # and depth ≈ 0.5 * v_in * t_comp (area under braking from v_in to 0).
        t_comp_push = 0.0
        denom = float(v_in + v_to)
        numer = float(v_to * T - 2.0 * precomp)
        if (denom > 1e-6) and (numer > 0.0):
            t_comp_push = float(numer / denom)
        # Pick the larger of the two requirements, then clamp to stable bounds.
        t_comp = float(max(float(cfg.soft_land_tc_min), t_comp_decel, t_comp_push))
        t_comp = float(_clipf(t_comp, float(cfg.soft_land_tc_min), float(cfg.soft_land_tc_max_ratio) * T))
        t_comp = float(min(t_comp, max(1e-3, T - 1e-3)))

        depth = float(0.5 * v_in * t_comp)
        depth = float(_clipf(depth, float(cfg.soft_land_depth_min_m), float(cfg.soft_land_depth_max_m)))

        # Optional base-z guard (convert to COM-z)
        if float(cfg.z_guard) > 0.0:
            z_min_base = float(z_td_base - depth)
            if z_min_base < float(cfg.z_guard):
                z_min_base = float(cfg.z_guard)
                depth = float(max(0.0, z_td_base - z_min_base))
        z_min = float((z_td_base - depth) + com_off_z)

        # Build two quintic segments in COM-z.
        # If depth was clipped (or inbound speed estimate is small), the remaining dz may be too small
        # to reach the requested v_to without going DOWN first. Reduce v_to for feasibility.
        T1 = float(max(1e-3, t_comp))
        T2 = float(max(1e-3, T - T1))
        dz = float(z_end - z_min)
        if (dz > 1e-6) and (T2 > 1e-6):
            v_to_feas = float(2.0 * dz / T2)
            # keep a small margin to avoid numerical overshoot
            v_to = float(min(v_to, 0.98 * v_to_feas))
        poly1 = self._quintic_coeff(z0, vz_td, 0.0, z_min, 0.0, 0.0, T1)
        poly2 = self._quintic_coeff(z_min, 0.0, 0.0, z_end, v_to, 0.0, T2)

        self._stance_prof_inited = True
        self._stance_t_comp = float(T1)
        self._stance_depth_tgt_m = float(depth)
        self._stance_poly1 = poly1
        self._stance_poly2 = poly2
        self._stance_T1 = float(T1)
        self._stance_T2 = float(T2)
        self._stance_z_end = float(z_end)

    def _unified_stance_ref(self, t_in_stance: float) -> tuple[float, float, float]:
        """
        Return (z_ref, vz_ref, az_ref) in COM HEIGHT coordinates (up-positive)
        at time since touchdown.
        If profile is not initialized, falls back to holding current estimate
        (converted from the +Z-down world estimates: height = -p_z, v_up = -v_z).
        """
        t = float(max(0.0, float(t_in_stance)))
        if (not bool(self._stance_prof_inited)) or (self._stance_poly1 is None) or (self._stance_poly2 is None):
            # Fallback: hold current COM height and use current upward velocity (best effort)
            return float(-self._p_hat_w[2] + float(self._stance_com_off_z)), float(-self._v_hat_w[2]), 0.0

        T1 = float(max(1e-3, float(self._stance_T1)))
        T2 = float(max(1e-3, float(self._stance_T2)))
        # NOTE: use the second segment at the exact boundary (t==T1) to make retiming safe and
        # avoid any edge-case discontinuity.
        if t < T1:
            return self._quintic_eval(self._stance_poly1, min(t, T1))
        else:
            return self._quintic_eval(self._stance_poly2, min(t - T1, T2))

    def step(
        self,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        imu_gyro_b: np.ndarray,
        imu_acc_b: np.ndarray,
        imu_quat_wxyz: np.ndarray | None,
        imu_rpy: np.ndarray | None,
        desired_v_xy_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        One control step.

        Returns:
          tau_cmd (3,), pwm_us (6,), info dict
        """
        self.sim_time = float(self.sim_time + self.dt)

        # Copy to avoid mutating caller buffers (LCM controller may reuse arrays).
        desired_v_xy_w = np.asarray(desired_v_xy_w, dtype=float).reshape(2).copy()

        # ===== 1D MODE: force zero horizontal velocity =====
        if bool(self.cfg.mode_1d):
            desired_v_xy_w[:] = 0.0

        joint_pos = np.asarray(joint_pos, dtype=float).reshape(3)
        joint_vel = np.asarray(joint_vel, dtype=float).reshape(3)
        imu_gyro_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
        imu_acc_b = np.asarray(imu_acc_b, dtype=float).reshape(3)
        imu_rpy = (
            np.asarray(imu_rpy, dtype=float).reshape(3)
            if imu_rpy is not None
            else np.zeros(3, dtype=float)
        )

        # --- Joint-velocity filtering ---
        # MATLAB/SLX EMA on the CAN-reported joint velocity:
        #   y[k] = (1-lambda)*input[k] + lambda*y[k-1].
        lam_qd = float(_clipf(
            float(getattr(self.cfg, "qd_ema_lambda", 0.0)), 0.0, 0.95
        ))
        if lam_qd > 0.0:
            if not bool(self._qd_ema_init):
                self._qd_ema[:] = joint_vel
                self._qd_ema_init = True
            else:
                self._qd_ema = (
                    (1.0 - lam_qd) * joint_vel + lam_qd * self._qd_ema
                ).astype(float)
            qd_src = self._qd_ema.copy()
        else:
            qd_src = joint_vel.copy()

        # Conditioned qd enters both foot PD and planted-foot velocity estimate.
        joint_vel_kin = qd_src.copy()

        acc_for_att_b = imu_acc_b.copy()

        # --- Foot kinematics (native delta FK, same FRD frame as IMU) ---
        # - foot_vicon / foot_vdot_vicon: used for leg PD & Jacobian
        # - foot_b / foot_vrel_b: debug alias (same frame)
        J_inv: np.ndarray | None = None
        J_body: np.ndarray | None = None

        if self._leg_model == "serial":
            foot_b_ser, J_body = self._serial_leg_fk_jac(
                q_roll=float(joint_pos[0]),
                q_pitch=float(joint_pos[1]),
                q_shift=float(joint_pos[2]),
            )
            J_body = np.asarray(J_body, dtype=float).reshape(3, 3)
            foot_vicon = _imu_body_to_leg_native(np.asarray(foot_b_ser, dtype=float).reshape(3))
            foot_vdot_vicon = _imu_body_to_leg_native((J_body @ joint_vel_kin.reshape(3)).reshape(3))
        else:
            if self.fk is None or self.kin is None:
                raise RuntimeError("delta leg model requested but kinematics is not initialized")
            foot_vicon, _ = self.fk.forward_kinematics(joint_pos)
            foot_vicon = np.asarray(foot_vicon, dtype=float).reshape(3)
            J_inv_raw, _ = self.kin.inverse_jacobian(foot_vicon, joint_vel_kin, theta=None)
            J_inv = np.asarray(J_inv_raw, dtype=float).reshape(3, 3)
            foot_vdot_vicon = (self._stable_inv3(J_inv) @ joint_vel_kin.reshape(3)).reshape(3).astype(float)

        foot_b = _leg_native_to_imu_body(foot_vicon)
        foot_vrel_b = _leg_native_to_imu_body(foot_vdot_vicon)

        # Jacobian diagnostics (delta leg only; NaN for serial or invalid cases)
        J_inv_det = float("nan")
        J_inv_cond = float("nan")
        if J_inv is not None:
            try:
                if np.all(np.isfinite(J_inv)):
                    J_inv_det = float(np.linalg.det(np.asarray(J_inv, dtype=float).reshape(3, 3)))
                    J_inv_cond = float(np.linalg.cond(np.asarray(J_inv, dtype=float).reshape(3, 3)))
            except Exception:
                pass
        # A_tau_f diagnostics (computed later if available)
        A_tau_f_det = float("nan")
        A_tau_f_cond = float("nan")

        # ===== Equivalent shift coordinate for phase detection =====
        # delta mode: use leg-length shift q_shift = ||foot|| - l0 (negative when compressed)
        # serial mode: use the prismatic shift joint directly (q_shift_joint >= 0 means compression),
        # and map it into the SAME "delta-style" convention via: q_shift = -q_shift_joint.
        leg_length = float(np.linalg.norm(foot_vicon))
        
        # Cache g_eff and dz_tgt for logging (computed at touchdown)
        g_eff_log = float("nan")
        dz_tgt_log = float("nan")
        if self._leg_model == "serial":
            # In hopper_serial.xml, shift joint increases with COMPRESSION.
            q_shift = -float(joint_pos[2])
            qd_shift = -float(joint_vel_kin[2])
        else:
            q_shift = float(leg_length - self.cfg.leg_l0_m)
            # Leg-length velocity from filtered CAN joint velocity through the
            # leg Jacobian. Positive means extending; negative means compressing.
            if leg_length > 1e-6:
                qd_shift = float(np.dot(
                    foot_vdot_vicon,
                    foot_vicon / leg_length,
                ))
            else:
                qd_shift = 0.0

        # --- Attitude estimate (body->world) ---
        if bool(self.cfg.use_fc_quat) and (imu_quat_wxyz is not None):
            q_hat = _quat_normalize_wxyz(np.asarray(imu_quat_wxyz, dtype=float).reshape(4))
        else:
            # MATLAB com_filter AHRS: gyro integrate every tick; inject the
            # accel tilt correction ONLY inside the stance mid-window. self._stance
            # here still holds the PREVIOUS tick's phase (the phase machine runs
            # below), so the counter tracks stance ticks with a 1-tick lag.
            if bool(self._stance):
                self._att_stance_tick += 1
            else:
                self._att_stance_tick = 0
            b_lo = int(getattr(self.cfg, "att_stance_bound_lo", 90))
            b_hi = int(getattr(self.cfg, "att_stance_bound_hi", 130))
            att_correct = bool(self._stance) and (b_lo <= int(self._att_stance_tick) < b_hi)
            q_hat = self.att.update(
                omega_b=imu_gyro_b,
                acc_b=acc_for_att_b,
                dt=float(self.dt),
                correct=att_correct,
                accel_weight=float(getattr(self.cfg, "att_accel_weight", 0.5)),
            )
        R_wb_hat = _quat_to_R_wb(q_hat)

        # ---- LiDAR yaw fusion (slow complementary pull, see lidar_* config) ----
        # A world-frame yaw offset rotates the IMU attitude so the core world
        # frame converges to the lidar map frame; roll/pitch stay pure IMU.
        # Applied BEFORE R_wb_hat is used anywhere (velocities, Raibert, e_R),
        # so the whole step sees one consistent frame.
        lidar_fresh = False
        if bool(getattr(self.cfg, "lidar_fuse_en", True)) and (int(self._lidar_quality) == 1) \
           and (float(self._lidar_rx_walltime) > 0.0):
            age_s = float(_time.time()) - float(self._lidar_rx_walltime)
            lidar_fresh = age_s <= float(getattr(self.cfg, "lidar_stale_s", 0.4))
        if lidar_fresh:
            yaw_imu = float(math.atan2(R_wb_hat[1, 0], R_wb_hat[0, 0]))
            yaw_err = float(self._lidar_yaw_map) - (yaw_imu + float(self._lidar_yaw_off))
            yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))  # wrap to [-pi, pi]
            tau_yaw = float(max(1e-3, float(getattr(self.cfg, "lidar_yaw_tau_s", 2.0))))
            if not bool(self._lidar_pos_inited):
                # first healthy fix: snap the yaw offset (no slow swing-in)
                self._lidar_yaw_off = float(self._lidar_yaw_off) + yaw_err
            else:
                a_yaw = float(_clipf(float(self.dt) / (tau_yaw + float(self.dt)), 0.0, 1.0))
                self._lidar_yaw_off = float(self._lidar_yaw_off) + a_yaw * yaw_err
        if abs(float(self._lidar_yaw_off)) > 1e-12:
            R_wb_hat = (_Rz(float(self._lidar_yaw_off)) @ R_wb_hat).astype(float)
        rpy_hat = _R_to_rpy_xyz(R_wb_hat)
        z_w = np.asarray(R_wb_hat[:, 2], dtype=float).reshape(3)
        # Propeller thrust direction in WORLD. In the current FRD IMU/body convention,
        # propellers push along body -Z, so the world thrust direction is -R_wb[:, 2].
        z_thrust_w = (-z_w).astype(float).reshape(3)

        # --- Base velocity from leg kinematics (foot assumed stationary in WORLD) ---
        # v_base_w = R_wb @ ( -foot_vdot_b - 0.1*(omega_b x foot_b) )
        # MATLAB-style 0.1 omega×r (2026-08-02, per user): matches the
        # original Raibert MATLAB pipeline.
        v_base_from_foot_w = (R_wb_hat @ (
            -foot_vrel_b.reshape(3)
            - 0.1 * _cross3(imu_gyro_b.reshape(3), foot_b.reshape(3))
        )).reshape(3)

        if not bool(self._v_hat_inited):
            self._v_hat_w = np.zeros(3, dtype=float)
            self._v_hat_inited = True

        touchdown_evt = False
        liftoff_evt = False
        apex_evt = False

        # ===== Hopper4-style phase machine =====
        # Use only leg-length thresholds + minimum phase duration:
        #   Flight -> Stance: q_shift <= -td_threshold
        #   Stance -> Flight: q_shift >= +lo_threshold
        phase_min_steps = int(max(1, int(getattr(self.cfg, "hopper4_phase_min_steps", 10))))
        phase_min_t = float(phase_min_steps) * float(self.dt)
        td_thr = float(max(0.0, float(getattr(self.cfg, "hopper4_td_threshold_m", 0.02))))
        lo_thr = float(max(0.0, float(getattr(self.cfg, "hopper4_lo_threshold_m", 0.01))))

        if (not bool(self._stance)) and np.isfinite(float(q_shift)):
            lo_t = float(self._lo_t) if self._lo_t is not None else 0.0
            t_in_flight = float(self.sim_time) - lo_t
            # Mid-air false-TD guard (log 011106): the swing leg can hit
            # the -2 cm threshold while still ASCENDING. Require the body
            # to be descending (or nearly stopped) before accepting TD.
            # IMPORTANT: _v_hat_w is WORLD FRD (+Z DOWN).  The gate is
            # specified in up-positive vz (same as logged vz_up).  Log
            # 103253: used v_hat_z raw (+1.5 while falling) so
            # "vz <= 0.15" REJECTED every real landing for ~200 ms of
            # q < -2 cm -- second hop never entered stance.  Convert:
            #   vz_up = -v_hat_z,  require vz_up <= hopper4_td_vz_max.
            vz_gate_max = float(getattr(
                self.cfg, "hopper4_td_vz_max_mps", 1e9
            ))
            vz_up_now = -float(self._v_hat_w[2])
            vz_gate_ok = (
                (not np.isfinite(vz_up_now))
                or vz_up_now <= vz_gate_max
            )
            # ENCODER escape hatch for the vz gate (log 200745 hop 7): the
            # world vz estimate drifted to +0.33..+0.74 "ascending" during a
            # REAL landing, so the gate rejected TD for the entire 180 ms the
            # leg sat below -2 cm -- the controller stayed in FLIGHT and the
            # swing PD fought the ground.  The leg-length rate qd_shift comes
            # straight from the joint encoders: 2 cm of compression while
            # still compressing fast can only be ground contact (a commanded
            # swing retraction never drives the leg that deep that fast at
            # this depth).  Accept TD when EITHER the world-vz gate passes or
            # the encoder says the leg is being compressed.
            qd_comp_min = float(getattr(
                self.cfg, "hopper4_td_qd_comp_mps", 0.05
            ))
            qd_gate_ok = (
                np.isfinite(float(qd_shift))
                and float(qd_shift) <= -qd_comp_min
            )
            cond_td = (
                (float(q_shift) <= -td_thr)
                and (t_in_flight >= phase_min_t)
                and (vz_gate_ok or qd_gate_ok)
            )

            if bool(cond_td):
                touchdown_evt = True
                self._stance = True
                # Duration of the stance BEFORE the flight that just ended
                # (for the double-pump trim guard, see nrc_trim_stance_max_s).
                prev_stance_s = None
                if self._lo_t is not None and self._td_t is not None:
                    prev_stance_s = float(self._lo_t) - float(self._td_t)
                self._td_t = float(self.sim_time)
                # ---- Flight-time apex measurement ----
                # Asymmetric ballistic arc (TA-SLIP): ascent gravity g_up and
                # descent gravity g_dn differ when the aerial brake is on.
                # T = sqrt(2h/g_up) + sqrt(2h/g_dn)
                #   => h = T^2 / (2*(1/sqrt(g_up) + 1/sqrt(g_dn))^2)
                # (reduces to g*T^2/8 when g_up == g_dn).
                # Window >= 0.12 s (apex >= ~1.8 cm): the 23:14 log had 90 ms
                # chatter "flights" pass the old 0.06 s window with a fake
                # 1 cm apex, each pumping ~1.3 J into E_loss.
                if self._lo_t is not None:
                    T_fl = float(self.sim_time) - float(self._lo_t)
                    # Remember real airtime for next hop's t_td prediction
                    # (tilt budget / descent brake): the symmetric-arc
                    # formula under-predicted by ~2x (log 214643) because
                    # TD happens below LO height and prop lift slows the
                    # fall.  Only real hops count, chatter would shrink it.
                    if 0.12 <= T_fl <= 1.5:
                        self._flight_dur_prev = T_fl
                    # The apex-trim return map serves BOTH continuous energy
                    # laws: nrc and pogox regulate an ESTIMATED in-stance
                    # energy, so any vz bias / taper loss lands at a biased
                    # apex every hop.  Log 200745 (pogox): all 6 hops apexed
                    # 1.9-3.4 cm vs the 5 cm target while the trim sat
                    # frozen at 1.000 because this gate was nrc-only.
                    _nrc_on_td = (
                        str(getattr(
                            self.cfg, "stance_energy_law", "mode1"
                        )).lower() in ("nrc", "pogox")
                    )
                    _trim_gain_td = float(_clipf(float(
                        self.cfg.nrc_apex_trim_gain
                    ), 0.0, 1.0))
                    _trim_eclip_td = float(max(0.0, float(getattr(
                        self.cfg, "nrc_apex_trim_err_clip", 0.0
                    ))))
                    # Double-pump guard (see nrc_trim_stance_max_s): the
                    # apex of a hop launched by an anomalous over-long
                    # stance must not steer the trim.
                    _stance_max_td = float(getattr(
                        self.cfg, "nrc_trim_stance_max_s", 0.0
                    ))
                    _stance_anomalous = bool(
                        _stance_max_td > 0.0
                        and prev_stance_s is not None
                        and float(prev_stance_s) > _stance_max_td
                    )
                    if T_fl < 0.12:
                        # Chatter/failed hop (see the ANTI-RATCHET note at
                        # nrc_trim_stance_max_s): no usable apex, but the
                        # hop demonstrably failed -- recover the trim by
                        # one clipped step so the map is a two-way
                        # contraction instead of a downward ratchet.
                        if (_nrc_on_td and _trim_gain_td > 0.0
                                and _trim_eclip_td > 0.0):
                            self._nrc_h_trim = float(_clipf(
                                float(self._nrc_h_trim)
                                + _trim_gain_td * _trim_eclip_td,
                                float(self.cfg.nrc_apex_trim_min),
                                float(self.cfg.nrc_apex_trim_max),
                            ))
                    if 0.12 <= T_fl <= 1.5:
                        tilt_fl = float(_clipf(
                            -float(z_thrust_w[2]), 0.0, 1.0
                        ))
                        rho_up_m = (
                            float(self.cfg.prop_base_thrust_ratio) * tilt_fl
                            if bool(self._props_armed_rt) else 0.0
                        )
                        rho_dn_m = (
                            max(
                                float(self.cfg.prop_base_thrust_ratio),
                                float(self.cfg.prop_flight_brake_ratio),
                            ) * tilt_fl
                            if bool(self._props_armed_rt) else 0.0
                        )
                        g_up_m = float(max(
                            1e-3,
                            float(self.gravity) * (1.0 - rho_up_m),
                        ))
                        g_dn_m = float(max(
                            1e-3,
                            float(self.gravity) * (1.0 - rho_dn_m),
                        ))
                        s_arc = (
                            1.0 / float(np.sqrt(g_up_m))
                            + 1.0 / float(np.sqrt(g_dn_m))
                        )
                        self._z_apex_actual = float(
                            T_fl * T_fl / (2.0 * s_arc * s_arc)
                        )
                        # Discrete apex-return-map layer (Koditschek-Buehler /
                        # Saranli energy regulation): fold the apex error of
                        # the hop that just ended into E_loss, consumed by the
                        # next push-spring sizing. Ideal-model error dynamics
                        # are e(n+1) = (1-gamma)*e(n): a 1-D contraction,
                        # deadbeat at gamma=1.
                        gam = float(_clipf(
                            float(self.cfg.mode1_apex_adapt_gamma), 0.0, 1.0
                        ))
                        self._mode1_Eloss += (
                            gam * float(self.mass) * float(self.gravity)
                            * (float(self.cfg.hop_height_m)
                               - float(self._z_apex_actual))
                        )
                        self._mode1_Eloss = float(_clipf(
                            self._mode1_Eloss, -2.0, 8.0
                        ))
                        # NRC per-hop apex return map (see nrc_apex_trim_*):
                        # same 1-D contraction, but folded into a
                        # multiplicative trim on the NRC height target
                        # instead of Mode1's E_loss.  Uses the drift-free
                        # flight-time apex just computed above.  Closes the
                        # loop the in-stance NRC cannot: a lagged/biased vz
                        # estimate lands at a biased apex every hop; this
                        # removes the bias hop-over-hop.
                        if (_nrc_on_td
                                and np.isfinite(self._z_apex_actual)
                                and not _stance_anomalous):
                            h_ref = float(max(0.02, float(
                                self.cfg.hop_height_m
                            )))
                            e_apex = (
                                float(self._z_apex_actual) - h_ref
                            ) / h_ref
                            # Bound the per-hop trim step (see
                            # nrc_apex_trim_err_clip): hop-off transient
                            # apexes must not yank the trim in one step.
                            if _trim_eclip_td > 0.0:
                                e_apex = float(_clipf(
                                    e_apex, -_trim_eclip_td, _trim_eclip_td
                                ))
                            self._nrc_h_trim = float(_clipf(
                                float(self._nrc_h_trim)
                                - _trim_gain_td * e_apex,
                                float(self.cfg.nrc_apex_trim_min),
                                float(self.cfg.nrc_apex_trim_max),
                            ))
                self._apex_reached = False
                # Trigger MPC solve on the very first stance step.
                self._mpc_counter = max(1, int(self.cfg.mpc_decimation)) - 1
                self._mpc_f_ref_cache[:] = 0.0
                fz_seed = float(self.cfg.mpc_fz_min) if bool(getattr(self.cfg, "use_mpc", True)) else float(self.cfg.stance_fz_min)
                self._f_ref_z_prev = float(max(0.0, fz_seed))
                self._f_ref_xy_prev[:] = 0.0
                self._mpc_omega_lpf[:] = 0.0
                self._mpc_omega_lpf_init = False

                # latch TD shift for compression measurement
                self._q_shift_td = float(q_shift)
                # Re-seed push-phase velocity ring (COMP samples excluded).
                self._push_vel_ring[:] = 0.0
                self._push_vel_ring_i = 0
                self._push_vel_ring_cnt = 0
                # Re-seed the Mode-1 two-spring per-stance state (E_loss is the
                # cross-hop apex adaptation and intentionally survives).
                self._mode1_push_latched = False
                self._mode1_push_confirm_count = 0
                self._mode1_k_comp = None
                self._mode1_k_boost = 0.0
                self._mode1_v_td = 0.0
                self._mode1_x0 = 0.0
                self._mode1_boost_f_state = 0.0
                self._mode1_vz_lpf = None
                # The prop energy supplement served last hop's PUSH + ascent;
                # a fresh stance re-solves it at its own PUSH latch.
                self._prop_energy_fz = 0.0
                # NRC law: consume the big-jump request at TOUCHDOWN (the whole
                # continuous stance targets the taller apex; Mode1's one-shot
                # consume-at-latch does not exist here).
                if (str(getattr(self.cfg, "stance_energy_law", "mode1"))
                        .lower() == "nrc") and bool(self._big_jump_pending):
                    self._nrc_big_gain = float(max(1.0, float(
                        self.cfg.big_jump_height_gain
                    )))
                    self._big_jump_pending = False
                # touchdown z estimate from kinematics (assume foot at ground z=0).
                # World +Z DOWN: the body above the ground has NEGATIVE z (p_z = -height).
                z_td_est = -float((R_wb_hat @ foot_b.reshape(3))[2])
                self._p_hat_w[2] = float(z_td_est)

                # Takeoff speed target for desired apex (ballistic, with prop assist).
                # z_thrust_w points UP (level: [0,0,-1] in the +Z-down world), so the
                # baseline prop thrust REDUCES effective gravity: g_eff = g*(1 - rho).
                g_eff = float(self.gravity + (float(z_thrust_w[2]) * float(self.mass) * float(self.gravity) * float(self.cfg.prop_base_thrust_ratio)) / max(1e-6, float(self.mass)))
                g_eff = float(max(1e-3, g_eff))
                dz_tgt = float(max(0.05, float(self.cfg.hop_height_m)))
                g_eff_log = float(g_eff)
                dz_tgt_log = float(dz_tgt)
                v_to_nominal = float(np.sqrt(2.0 * g_eff * dz_tgt))
                self._v_to_cmd = float(_clipf(v_to_nominal, float(self.cfg.v_to_min), float(self.cfg.v_to_max)))

                # ---- FB-SLIP TD sizing (37a1475 port) ----
                # v_to from the dedicated takeoff-height knob, and the
                # DESIGN-MAP reception target depth: pushing at
                # w_push*F_max over the absolute deficit x0 = x_td + s_tgt
                # must deliver the takeoff energy,
                #   (w_push*F_max - m*g_st) * x0 = 0.5*m*v_to^2.
                if str(getattr(self.cfg, "stance_energy_law", "nrc")) \
                        .lower() == "fbslip":
                    try:
                        m_td = float(self.mass)
                        g_raw = float(self.gravity)
                        rho_st_td = (
                            float(_clipf(float(
                                self.cfg.prop_stance_base_thrust_ratio
                            ), 0.0, 0.8))
                            if (bool(self._props_armed_rt)
                                and bool(self.cfg.stance_use_props))
                            else 0.0
                        )
                        g_st_td = g_raw * (1.0 - rho_st_td)
                        v_to_fb = float(_clipf(
                            float(np.sqrt(2.0 * g_eff * float(max(
                                0.02, float(self.cfg.takeoff_height_m)
                            )))),
                            float(self.cfg.v_to_min),
                            float(self.cfg.v_to_max),
                        ))
                        self._v_to_cmd = v_to_fb
                        f_max_td = float(max(
                            2.5 * m_td * g_raw,
                            float(self.cfg.leg_force_budget_g)
                            * m_td * g_raw,
                        ))
                        h_com_td = float(-z_td_est)
                        x_td0 = float(max(
                            0.0, float(self.cfg.leg_l0_m) - h_com_td
                        ))
                        w_push = float(_clipf(float(
                            self.cfg.stance_push_force_frac
                        ), 0.05, 1.0))
                        a_net = float(max(
                            0.15 * g_raw,
                            (w_push * f_max_td) / m_td - g_st_td,
                        ))
                        x0_des = v_to_fb * v_to_fb / (2.0 * a_net)
                        stroke = float(max(
                            0.015, float(self.cfg.leg_stroke_max_m)
                        ))
                        catch = float(max(0.005, float(
                            self.cfg.stance_recv_catch_span_m
                        )))
                        trav_min = float(_clipf(float(
                            self.cfg.stance_travel_min_m
                        ), 0.0, stroke))
                        self._fb_x_td = x_td0
                        self._fb_s_tgt = float(_clipf(
                            x0_des - x_td0,
                            trav_min,
                            max(trav_min, stroke - catch),
                        ))
                        self._fb_trav_plan = float(max(
                            0.02,
                            min(stroke, float(self._fb_s_tgt) + catch),
                        ))
                        self._fb_xz_max = 0.0
                        self._fb_push_f = 0.0
                        self._fb_push_taper = 0.0
                        self._px_f_prev = None
                        self._px_qs_prev = None
                        self._px_vz_ax = None
                        self._px_pump_prev = 0.0
                        self._fb_fcomp_lpf = None
                    except Exception:
                        self._fb_s_tgt = float(getattr(
                            self.cfg, "stance_travel_min_m", 0.02
                        ))

                # Initialize smooth stance profile.
                # NOTE: the profile works in HEIGHT coordinates (up-positive), so convert
                # from the +Z-down world here: height = -p_z, v_up = -v_z.
                try:
                    vz_td_ned = float(v_base_from_foot_w[2]) if np.isfinite(float(v_base_from_foot_w[2])) else float(self._v_hat_w[2])
                    self._init_unified_stance_profile(
                        R_wb=R_wb_hat,
                        z_td_base=float(-z_td_est),
                        vz_td=float(-vz_td_ned),
                        q_shift_td=float(self._q_shift_td) if self._q_shift_td is not None else float(q_shift),
                    )
                except Exception:
                    self._stance_prof_inited = False

        # ===== Hopper4-style liftoff =====
        if bool(self._stance) and np.isfinite(float(q_shift)):
            # Liftoff: plain leg-extension threshold only.
            # 2026-08-03: dropped the stance-side phase_min dwell
            # (was 100 ms via hopper4_phase_min_steps) per user -- LO
            # fires as soon as q_shift >= lo_thr.  The min-dwell still
            # applies to Flight->Stance (TD) only, above.
            cond_lo = float(q_shift) >= lo_thr
            if bool(cond_lo):
                liftoff_evt = True
                self._stance = False
                self._lo_t = float(self.sim_time)
                # POGOX law consumes the RB big-jump flag per-tick during
                # stance (continuous law, no latch event): retire it at
                # liftoff so it stays a one-hop boost.
                if str(getattr(
                    self.cfg, "stance_energy_law", ""
                )).lower() == "pogox":
                    self._big_jump_pending = False
                # Latch flight XY.
                # MATLAB-style last-N stance mean; instantaneous LO fallback.
                v_push_tail = self._push_vel_tail_mean()
                if v_push_tail is not None:
                    v_latch = np.asarray(v_push_tail, dtype=float).reshape(3)
                elif np.all(np.isfinite(v_base_from_foot_w)):
                    v_latch = v_base_from_foot_w.reshape(3).astype(float)
                else:
                    v_latch = self._v_hat_w.reshape(3)
                self._flight_vel = np.asarray(v_latch, dtype=float).reshape(3).copy()
                self._flight_vel[2] = 0.0
                # NOTE (2026-08-01): the prop energy supplement is NOT cleared
                # here.  Unlike the leg (whose stroke ends at liftoff) the
                # props keep doing work through the ASCENT; the supplement
                # rides the collective until apex with a continuous vz fade
                # (see the flight collective block) and is re-seeded at the
                # next PUSH latch.
                # Liftoff state for apex detection / logs (up-positive vz).
                self._nrc_big_gain = 1.0  # big jump served one stance only
                self._z_lo = float(self._p_hat_w[2])
                self._vz_lo = float(-self._v_hat_w[2])
                self._prev_vz = None
                self._apex_reached = False

        # ===== Body velocity =====
        # STANCE: XY+Z = instantaneous planted-foot kinematics (hard
        #         overwrite).  MATLAB-style 0.1*omega×r is already in
        #         v_base_from_foot_w; Z stays the foot kinematic channel
        #         (user 2026-08-03: "z还是原来的 z不要改").
        # FLIGHT: XY from liftoff latch (+ IMU DR below); Z = IMU integration.
        g_w = np.array([0.0, 0.0, float(self.gravity)], dtype=float)  # +Z DOWN
        if np.all(np.isfinite(imu_acc_b)):
            a_w = (R_wb_hat @ imu_acc_b.reshape(3) + g_w.reshape(3)).reshape(3)
        else:
            a_w = np.zeros(3, dtype=float)
        vz_pred = float(self._v_hat_w[2]) + float(a_w[2]) * float(self.dt)

        if bool(getattr(self, "_user_zero_vel_hold", False)):
            self._v_hat_w[:] = 0.0
        elif bool(self._stance):
            if np.all(np.isfinite(v_base_from_foot_w)):
                self._v_hat_w[0] = float(v_base_from_foot_w[0])
                self._v_hat_w[1] = float(v_base_from_foot_w[1])
                self._v_hat_w[2] = float(v_base_from_foot_w[2])
                # Latch ring, FZ-GATED (see vel_latch_fz_min_ratio): record
                # only while last tick's stance fz reference shows the foot
                # firmly loaded; the ring then naturally holds the last N
                # well-planted samples and the liftoff median ignores the
                # slipping unload tail.
                _fz_gate = float(max(0.0, float(getattr(
                    self.cfg, "vel_latch_fz_min_ratio", 0.0
                )))) * float(self.mass) * float(self.gravity)
                if _fz_gate <= 0.0 or float(self._f_ref_z_prev) > _fz_gate:
                    i = int(self._push_vel_ring_i)
                    self._push_vel_ring[i, :] = v_base_from_foot_w.reshape(3).astype(float)
                    self._push_vel_ring_i = (i + 1) % int(self._vel_push_tail_n)
                    self._push_vel_ring_cnt = min(
                        int(self._push_vel_ring_cnt) + 1, int(self._vel_push_tail_n)
                    )
            else:
                self._v_hat_w[2] = float(vz_pred)
        else:
            # FLIGHT XY: IMU dead-reckoning from the liftoff latch (2026-08-01,
            # user: "空中的速度不能锁存...不然空中只会保持一种姿态").  The
            # latched last-N stance mean is the INITIAL CONDITION only; each
            # flight tick propagates with the world specific force
            #     v_xy += (R*a_b + g)_xy * dt,
            # so prop braking (the PogoX tilt below) actually shows up in the
            # velocity the Raibert placement and the tilt loop both consume --
            # the standard tri-rotor cascade assumption (Salazar-Cruz CEP'09).
            # Same a_w integration the Z channel has always used.
            if np.all(np.isfinite(a_w)):
                self._flight_vel[0] += float(a_w[0]) * float(self.dt)
                self._flight_vel[1] += float(a_w[1]) * float(self.dt)
            self._v_hat_w[0] = float(self._flight_vel[0])
            self._v_hat_w[1] = float(self._flight_vel[1])
            self._v_hat_w[2] = float(vz_pred)
        self._v_hat_inited = True

        # integrate position + stance z correction
        # (2026-07-10: the 50 ms stance-z LPF was DELETED per user -- in stance
        # the height comes straight from leg kinematics every tick.)
        self._p_hat_w = self._p_hat_w + self._v_hat_w * float(self.dt)
        if bool(self._stance):
            self._p_hat_w[2] = -float((R_wb_hat @ foot_b.reshape(3))[2])

        # ---- LiDAR XY position correction (slow pull; z stays leg-based) ----
        # lidar_fresh was evaluated at the attitude section this same tick.
        if lidar_fresh:
            ex = float(self._lidar_pos_map[0]) - float(self._p_hat_w[0])
            ey = float(self._lidar_pos_map[1]) - float(self._p_hat_w[1])
            if not bool(self._lidar_pos_inited):
                # First healthy fix: snap XY to the lidar (the dead-reckoned
                # origin is arbitrary anyway). Yaw offset was snapped above.
                self._p_hat_w[0] = float(self._lidar_pos_map[0])
                self._p_hat_w[1] = float(self._lidar_pos_map[1])
                self._lidar_pos_inited = True
            else:
                tau_p = float(max(1e-3, float(getattr(self.cfg, "lidar_pos_tau_s", 0.7))))
                a_p = float(_clipf(float(self.dt) / (tau_p + float(self.dt)), 0.0, 1.0))
                self._p_hat_w[0] = float(self._p_hat_w[0]) + a_p * ex
                self._p_hat_w[1] = float(self._p_hat_w[1]) + a_p * ey
            self._lidar_fused_n += 1

        # apex detection (flight): up-positive vz sign change (legacy).
        # World +Z is DOWN internally, but apex uses vz_up = -v_hat_w[2] so
        # ascending => vz_up > 0 and apex is the crossing to <= 0.
        vz_up = float(-self._v_hat_w[2])
        if self._prev_vz is None:
            self._prev_vz = float(vz_up)
        if (not bool(self._stance)) and (float(self._prev_vz) > 0.0) and (float(vz_up) <= 0.0):
            apex_evt = True
            self._apex_reached = True
            if self._z_lo is not None:
                h_act = float(self._z_lo) - float(self._p_hat_w[2])
                if np.isfinite(h_act) and h_act > 0.0:
                    self._z_apex_actual = float(h_act)
        self._prev_vz = float(vz_up)

        # ===== stance: unified reference (no discrete COMP/PUSH switching) =====
        # We keep "compress_active" as a debug label only (pre/post max-compression time),
        # but the controller itself uses a single smooth stance reference curve.
        az_des = -float(self.gravity)  # default (flight)
        compress_active = False
        depth_now = 0.0
        depth_tgt = 0.0
        depth_tgt_act = 0.0
        z_now = float(self._p_hat_w[2])
        s = 0.0
        t_in_stance = 0.0

        if bool(self._stance):
            t_td = float(self._td_t) if (self._td_t is not None) else float(self.sim_time)
            t_in_stance = float(float(self.sim_time) - t_td)
            s = float(_clipf(t_in_stance / max(1e-6, float(self.cfg.stance_T)), 0.0, 1.0))

            # Actual compression depth for logging (from leg length shift)
            q_shift_td = self._q_shift_td
            if (q_shift_td is None) or (not np.isfinite(float(q_shift_td))) or (not np.isfinite(q_shift)):
                depth_now = 0.0
            else:
                depth_now = float(max(0.0, float(q_shift_td) - float(q_shift)))

            z_now = float(self._p_hat_w[2])

            # Desired vertical acceleration from the unified stance reference (COM-z), smooth by construction.
            if bool(self.cfg.use_unified_stance) and bool(self._stance_prof_inited):
                _, _, az_ref = self._unified_stance_ref(t_in_stance)
                az_des = float(az_ref)
                depth_tgt = float(self._stance_depth_tgt_m)
                depth_tgt_act = float(depth_tgt)
                t_comp = float(self._stance_t_comp) if self._stance_t_comp is not None else 0.0
                compress_active = bool(t_in_stance < float(t_comp))
            else:
                # If the stance profile isn't initialized, keep a conservative default.
                az_des = 0.0
                depth_tgt = 0.0
                depth_tgt_act = 0.0
                compress_active = False

        # ===== Build lever arms about COM (world) =====
        r_foot_w = (R_wb_hat @ (foot_b - self.com_b).reshape(3)).reshape(3)
        prop_r_w = (R_wb_hat @ (self.prop_positions_b - self.com_b.reshape(1, 3)).T).T.copy()

        # ===== Torque mapping A_tau_f (world GRF -> delta motor torques) =====
        A_tau_f_3rsr = None
        if self._leg_model == "serial":
            try:
                if J_body is not None:
                    # RAW leg torque map (serial): tau = J^T * (R_wb^T * f). No sign flips
                    # beyond the per-motor wiring sign (tau_cmd_sign).
                    A_tau_f_3rsr = ((np.asarray(J_body, dtype=float).reshape(3, 3).T @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T)).astype(float)
                    tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                    A_tau_f_3rsr = (np.diag(tau_sign) @ A_tau_f_3rsr).astype(float)
            except Exception:
                A_tau_f_3rsr = None
        else:
            try:
                if self.kin is None:
                    raise RuntimeError("delta kinematics not initialized")
                # Use same foot offset convention as the MuJoCo demo (contact site 3cm below link origin)
                # Here x3 is already in delta/vicon frame (z positive)
                x3 = foot_vicon.copy()
                x3[2] = float(x3[2] + float(self._delta_ws["z_off"]))
                x3[0] = float(_clipf(x3[0], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
                x3[1] = float(_clipf(x3[1], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
                x3[2] = float(_clipf(x3[2], float(self._delta_ws["z_min"]), float(self._delta_ws["z_max"])))

                # Recompute inverse Jacobian at the clamped workspace point for numerical robustness.
                # Hopper4 returns J_inv such that: thetadot = J_inv * xdot  (delta/vicon frame)
                # Torque mapping: tau = inv(J_inv^T) * f_delta
                J_inv_map, _ = self.kin.inverse_jacobian(x3, np.zeros(3, dtype=float), theta=None)
                J_inv_map = np.asarray(J_inv_map, dtype=float).reshape(3, 3)
                inv_Jt = self._stable_inv3(J_inv_map.T)

                # RAW leg torque map (delta): tau = inv(J_inv^T) * (R_wb^T * f). No sign flips
                # beyond the per-motor wiring sign (tau_cmd_sign).
                A_tau_f_3rsr = ((inv_Jt @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T)).astype(float)
                # Motor torque sign convention (real robot)
                tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                A_tau_f_3rsr = (np.diag(tau_sign) @ A_tau_f_3rsr).astype(float)
            except Exception:
                A_tau_f_3rsr = None

        if A_tau_f_3rsr is not None:
            try:
                if np.all(np.isfinite(A_tau_f_3rsr)):
                    A_tau_f_det = float(np.linalg.det(np.asarray(A_tau_f_3rsr, dtype=float).reshape(3, 3)))
                    A_tau_f_cond = float(np.linalg.cond(np.asarray(A_tau_f_3rsr, dtype=float).reshape(3, 3)))
            except Exception:
                pass

        tau_cmd_max = np.asarray(self.cfg.tau_cmd_max_nm, dtype=float).reshape(3)

        # ===== Wrench-level controller =====
        # Upstream: desired net wrench (F_des from f_ref, Tau from SO(3) PD).
        # Downstream: closed-form leg forces + lstsq prop thrust (no WBC-QP).
        #   f_contact_w: GRF in world frame; z_thrust_w = -R_wb[:, 2] for prop thrust direction.
        # Collective plan (2026-08-01): PWM-1100 idle base for the WHOLE hop
        # cycle.  The ONLY planned addition is the PUSH energy supplement,
        # which -- unlike the leg -- persists through the flight ASCENT
        # (props have no stroke limit) and fades out CONTINUOUSLY at apex:
        #     F(t) = F_prop * clip(vz_up / vz_fade, 0, 1)
        # vz_up is the physical fade variable: it reaches zero exactly AT
        # apex, so the force rolls off with the remaining ascent -- no
        # switch, no timer ("不要硬 cut").  Descent adds nothing.
        if bool(self._stance):
            prop_ratio = float(self.cfg.prop_stance_base_thrust_ratio)
        elif (
            -float(self._v_hat_w[2])
            < -float(self.cfg.prop_flight_brake_vz_mps)
        ):
            prop_ratio = float(max(
                float(self.cfg.prop_base_thrust_ratio),
                float(self.cfg.prop_flight_brake_ratio),
            ))
            # Descent VELOCITY-BRAKE collective (see the
            # prop_descent_brake_* config block): error-gated, faded on
            # the same time-to-touchdown clock as the tilt budget.  Gives
            # the tilt loop a real f_z to vector while falling.
            _dbr = float(max(0.0, float(getattr(
                self.cfg, "prop_descent_brake_ratio", 0.0
            ))))
            if (_dbr > 1e-9
                    and bool(getattr(
                        self.cfg, "flight_vel_ctrl_enable", False))
                    and bool(self._props_armed_rt)):
                ev_ref = float(max(0.05, float(getattr(
                    self.cfg, "prop_descent_brake_ev_mps", 0.40
                ))))
                e_v_n = float(np.linalg.norm(
                    np.asarray(desired_v_xy_w, dtype=float).reshape(2)
                    - np.asarray(
                        self._flight_vel[0:2], dtype=float
                    ).reshape(2)
                ))
                vz_up_c = -float(self._v_hat_w[2])
                vz_lo_c = (
                    float(self._vz_lo)
                    if (self._vz_lo is not None
                        and np.isfinite(float(self._vz_lo)))
                    else 0.0
                )
                t_td_c = max(0.0, (
                    vz_up_c + max(0.0, vz_lo_c)
                ) / max(0.5, float(self.gravity)))
                # Same airtime correction as the tilt budget: the
                # symmetric-arc prediction collapses the brake window to
                # ~nothing on real hops (log 214643).
                if self._flight_dur_prev > 0.0 and self._lo_t is not None:
                    t_td_c = max(t_td_c, (
                        float(self._flight_dur_prev)
                        - (float(self.sim_time) - float(self._lo_t))
                    ))
                settle_c = float(max(0.02, float(getattr(
                    self.cfg, "flight_level_settle_s", 0.12
                ))))
                fade_c = float(_clipf(
                    (t_td_c - settle_c) / settle_c, 0.0, 1.0
                ))
                prop_ratio += (
                    _dbr
                    * float(_clipf(e_v_n / ev_ref, 0.0, 1.0))
                    * fade_c
                )
        else:
            prop_ratio = float(self.cfg.prop_base_thrust_ratio)
        thrust_sum_ref = float(self.mass * self.gravity * float(prop_ratio))
        # Prop energy supplement in FLIGHT: DISABLED (2026-08-02, user
        # "flight phase只负责控制姿态").  The supplement is consumed in
        # the stance push (see above).  Flight keeps only the idle base
        # collective + optional descent brake so the prop differential is
        # free for attitude.
        self._prop_energy_fz = 0.0
        # Global propeller enable gate. 2026-07-09: keyed off the RUNTIME armed
        # state (A switch) instead of the deleted pure_leg_mode -- when the user
        # never presses A, the controller must not assume prop assist anywhere
        # (g_eff, flight attitude demands, prop overlays all shut off here).
        props_enabled_ctrl = bool(self._props_armed_rt) and (
            bool(self.cfg.stance_use_props)
            or (float(self.cfg.prop_base_thrust_ratio) > 1e-9)
            or (float(self.cfg.prop_stance_base_thrust_ratio) > 1e-9)
        )
        if not bool(props_enabled_ctrl):
            thrust_sum_ref = 0.0
        # ===== Phase-independent attitude error (body FRD, world +Z down) =====
        # CASE HFA geometric SO(3) attitude error (Lee et al., CDC 2010).
        # R_des keeps measured yaw while commanding zero roll/pitch.
        yaw = float(rpy_hat[2])
        R_des = _Rz(yaw)
        # ---- PogoX / tri-rotor FLIGHT VELOCITY CONVERGENCE ----
        # Reduced-attitude thrust vectoring: keep yaw, tilt the desired body
        # thrust axis toward the horizontal velocity error (see the
        # flight_vel_* config block).  When the tilt command is zero this
        # construction reduces EXACTLY to _Rz(yaw), so the level-attitude
        # behavior is unchanged whenever the loop is off or converged.
        # The props then track this R_des through the SAME flight_kR/kW
        # geometric PD and the decoupled differential channel -- velocity
        # convergence rides the existing attitude path, no new actuator law.
        self._fl_tilt_cmd_deg = 0.0
        self._fl_zb_des_xy[:] = 0.0
        self._fl_ev_xy = 0.0
        self._fl_lat_force_n = 0.0
        if bool(self._stance):
            # Slew state re-arms level for the next flight.
            self._fl_tilt_vec[:] = 0.0
        elif (bool(getattr(self.cfg, "flight_vel_ctrl_enable", False))
                and bool(props_enabled_ctrl)
                and (self._lo_t is not None)):
            kv_fl = float(max(0.0, float(self.cfg.flight_vel_kv)))
            # CONTINUOUS velocity feedback: the outer loop closes on the
            # LIVE IMU-propagated flight velocity every tick,
            #     a_des(t) = kv * (v_des - v(t)),
            # so the commanded tilt shrinks as the velocity converges and
            # the body re-levels on its own (Salazar-Cruz CEP'09; Lee
            # CDC'10; PogoX ICRA'24) -- instead of freezing one attitude
            # for the whole window the way a latched velocity would.
            a_des_xy = kv_fl * (
                np.asarray(desired_v_xy_w, dtype=float).reshape(2)
                - np.asarray(self._flight_vel[0:2], dtype=float).reshape(2)
            )
            self._fl_ev_xy = float(np.linalg.norm(
                np.asarray(desired_v_xy_w, dtype=float).reshape(2)
                - np.asarray(self._flight_vel[0:2], dtype=float).reshape(2)
            ))
            # LANDING-UPRIGHT rule, CONTINUOUS form (no on/off gate): a
            # lean of theta needs theta/slew seconds to ramp level plus
            # the settle margin before touchdown, so the ballistic
            # time-to-touchdown t_td can only PAY FOR
            #     theta_budget = slew * max(0, t_td - settle).
            # Early in flight the budget is above the user cap and never
            # binds; near touchdown it shrinks to zero AT THE SLEW RATE,
            # so the slew-limited reference tracks it exactly and the
            # body is level `settle` seconds to spare.  v_td_pred = vz_lo
            # (symmetric arc) errs on levelling early -- the safe side.
            vz_up_fl = -float(self._v_hat_w[2])
            vz_lo_fl = (
                float(self._vz_lo)
                if (self._vz_lo is not None
                    and np.isfinite(float(self._vz_lo)))
                else 0.0
            )
            a_dn_fl = float(self.gravity) * (
                1.0 - _clipf(float(prop_ratio), 0.0, 0.9)
            )
            t_td_fl = max(0.0, (
                vz_up_fl + max(0.0, vz_lo_fl)
            ) / max(0.5, a_dn_fl))
            # The symmetric-arc formula under-predicts airtime ~2x (TD is
            # below LO height, prop lift slows the fall): log 214643 had
            # the budget collapse at mid-flight and the whole descent
            # forced level.  When the previous hop's REAL airtime is
            # known, take the larger of the two predictions -- steady
            # hopping keeps tilt authority through most of the descent,
            # and the budget still shrinks to zero before the (measured)
            # touchdown.
            if self._flight_dur_prev > 0.0 and self._lo_t is not None:
                t_in_fl = float(self.sim_time) - float(self._lo_t)
                t_td_fl = max(
                    t_td_fl, float(self._flight_dur_prev) - t_in_fl
                )
            else:
                # FIRST HOP / no history: the symmetric-arc prediction
                # collapses to ~0.06 s and kills the tilt before the real
                # 0.2 s flight is half over (log 220610 first hop).  Use a
                # conservative default airtime so the first takeoff also has
                # tilt authority; the cap still limits the lean magnitude.
                t_td_fl = max(t_td_fl, 0.18)
            slew_dps_fl = float(max(1.0, float(
                self.cfg.flight_vel_tilt_slew_dps
            )))
            ang_budget_r = math.radians(slew_dps_fl) * max(0.0, (
                t_td_fl - float(max(0.0, float(
                    self.cfg.flight_level_settle_s
                )))
            ))
            # VECTORED THRUST: direction AND magnitude from ONE desired
            # world force (up-positive),
            #   F_des = [ m*a_des_xy,  f_z_plan ],
            #   z_b_des = F_des/||F_des||,  T_cmd = ||F_des||,
            # so the vertical collective share is preserved BY
            # CONSTRUCTION (T grows as 1/cos of the tilt) and a tiny
            # collective simply yields a tiny delivered lateral force
            # f_z*tan(tilt) -- the damper fades out by physics, no gate.
            f_z_up = float(max(
                1e-3,
                float(self.mass) * float(self.gravity) * float(prop_ratio),
            ))
            f_xy_raw = (float(self.mass) * a_des_xy).astype(float)
            # Tilt cap = min(user cap, repayable landing budget): both are
            # physical limits, both bind continuously.
            t_max_r = min(
                math.radians(float(_clipf(
                    float(self.cfg.flight_vel_tilt_max_deg), 0.0, 25.0
                ))),
                float(ang_budget_r),
            )
            # SLEW-LIMITED tilt reference in ANGLE space: the raw desired
            # force maps to a tilt 2-vector (direction = tilt direction,
            # magnitude = angle); the reference moves toward it at most
            # flight_vel_tilt_slew_dps -- R_des stays continuous through
            # lean build-up, the budget shrink, and error sign flips.
            f_n_raw = float(np.linalg.norm(f_xy_raw))
            if np.all(np.isfinite(f_xy_raw)) and f_n_raw > 1e-9:
                ang_tgt = min(t_max_r, math.atan2(f_n_raw, f_z_up))
                tilt_tgt = (f_xy_raw / f_n_raw) * ang_tgt
            else:
                tilt_tgt = np.zeros(2, dtype=float)
            slew_r = math.radians(slew_dps_fl) * float(self.dt)
            d_tl = tilt_tgt - self._fl_tilt_vec
            d_n = float(np.linalg.norm(d_tl))
            if d_n > slew_r:
                d_tl = d_tl * (slew_r / d_n)
            self._fl_tilt_vec = (self._fl_tilt_vec + d_tl).astype(float)
            ang_now = float(np.linalg.norm(self._fl_tilt_vec))
            if ang_now > 1e-4:
                f_xy_w = (
                    (self._fl_tilt_vec / ang_now)
                    * (f_z_up * math.tan(ang_now))
                ).astype(float)
                f_n = float(np.linalg.norm(f_xy_w))
            else:
                f_xy_w = np.zeros(2, dtype=float)
                f_n = 0.0
            # Delivered lateral braking force [N]: f_z * tan(tilt).
            self._fl_lat_force_n = float(f_n)
            if f_n > 1e-6 and np.all(np.isfinite(f_xy_w)):
                f_mag = float(math.hypot(f_n, f_z_up))
                # NOTE (2026-08-01, user: "energy 补充的时候才加 Fz"):
                # the collective is NOT raised to follow the tilt vector.
                # thrust_sum_ref stays at the PWM-1100 idle base; the tilt
                # only shapes R_des and the braking moment is delivered by
                # the DIFFERENTIAL channel (3rotor style).  The delivered
                # lateral force is whatever the tilted total thrust gives.
                # Thrust points along -z_b (FRD): the desired body z-axis
                # is the negative of the desired thrust direction.  Yaw is
                # kept via the geometric-controller triad (Lee et al.).
                zb_des = np.array([
                    -float(f_xy_w[0]) / f_mag,
                    -float(f_xy_w[1]) / f_mag,
                    float(f_z_up) / f_mag,
                ], dtype=float)
                xc_yaw = np.array(
                    [math.cos(yaw), math.sin(yaw), 0.0], dtype=float
                )
                yb_des = _cross3(zb_des, xc_yaw)
                yb_n = float(np.linalg.norm(yb_des))
                if yb_n > 1e-6:
                    yb_des = yb_des / yb_n
                    xb_des = _cross3(yb_des, zb_des)
                    R_des = np.column_stack(
                        [xb_des, yb_des, zb_des]
                    ).astype(float)
                    self._fl_tilt_cmd_deg = math.degrees(
                        math.atan2(f_n, f_z_up)
                    )
                    self._fl_zb_des_xy[0] = float(zb_des[0])
                    self._fl_zb_des_xy[1] = float(zb_des[1])
        E_R = (R_des.T @ R_wb_hat) - (R_wb_hat.T @ R_des)
        e_R = (0.5 * _vee_so3(E_R)).astype(float)
        e_R[2] = 0.0

        f_ref = np.zeros(3, dtype=float)
        mpc_status = "off"
        mpc_u0 = np.zeros(3, dtype=float)
        mpc_used = False  # Track whether MPC provided f_ref this step
        if bool(self._stance):
            # ===== Stance force reference: MPC or SRB virtual spring =====
            mpc_used = False
            use_mpc_stance = bool(self._mpc is not None) and bool(self.cfg.use_mpc)
            if bool(self.cfg.mode_1d) and bool(getattr(self.cfg, "mode_1d_disable_mpc", True)):
                use_mpc_stance = False
                mpc_status = "disabled_1d"

            mpc_dec = max(1, int(self.cfg.mpc_decimation))
            run_mpc_now = False
            if use_mpc_stance:
                self._mpc_counter += 1
                run_mpc_now = (self._mpc_counter >= mpc_dec)
            else:
                self._mpc_counter = 0

            if use_mpc_stance and run_mpc_now:
                self._mpc_counter = 0
                try:
                    # --- Build MPC state x0 (13D) ---
                    # State layout: [px, py, pz, vx, vy, vz, roll, pitch, yaw, ωx, ωy, ωz, yaw_ref]
                    mpc_x0 = np.zeros(13, dtype=float)
                    mpc_x0[0] = float(self._p_hat_w[0])
                    mpc_x0[1] = float(self._p_hat_w[1])
                    mpc_x0[2] = float(self._p_hat_w[2])
                    mpc_x0[3] = float(self._v_hat_w[0])
                    mpc_x0[4] = float(self._v_hat_w[1])
                    mpc_x0[5] = float(self._v_hat_w[2])
                    mpc_x0[6] = float(imu_rpy[0])   # roll
                    mpc_x0[7] = float(imu_rpy[1])   # pitch
                    mpc_x0[8] = float(imu_rpy[2])   # yaw
                    # Feed MPC with conditioned angular rates and keep frame consistency.
                    # Dynamics/cost are expressed in world coordinates; use omega_w = R_wb * omega_b.
                    omega_mpc_b = np.asarray(imu_gyro_b, dtype=float).reshape(3).copy()
                    # MPC gyro LPF (DISABLED per user 2026-07-11: gyro 不要滤波).
                    # try:
                    #     tau_om = float(max(0.0, float(getattr(self.cfg, "mpc_omega_lpf_tau", 0.0))))
                    # except Exception:
                    #     tau_om = 0.0
                    # if tau_om > 1e-9:
                    #     if not bool(self._mpc_omega_lpf_init):
                    #         self._mpc_omega_lpf = omega_mpc_b.copy()
                    #         self._mpc_omega_lpf_init = True
                    #     else:
                    #         a_om = float(_clipf(float(self.dt) / (float(tau_om) + float(self.dt)), 0.0, 1.0))
                    #         self._mpc_omega_lpf = (1.0 - a_om) * self._mpc_omega_lpf + a_om * omega_mpc_b
                    #     omega_mpc_b = np.asarray(self._mpc_omega_lpf, dtype=float).reshape(3).copy()
                    # Convert to world-frame angular velocity for MPC state consistency.
                    omega_mpc = (R_wb_hat @ omega_mpc_b.reshape(3)).reshape(3)
                    try:
                        wclip_mpc = float(max(0.0, float(getattr(self.cfg, "mpc_omega_xy_clip_radps", 0.0))))
                    except Exception:
                        wclip_mpc = 0.0
                    if wclip_mpc > 1e-9:
                        omega_mpc[0] = float(_clipf(float(omega_mpc[0]), -wclip_mpc, +wclip_mpc))
                        omega_mpc[1] = float(_clipf(float(omega_mpc[1]), -wclip_mpc, +wclip_mpc))
                    mpc_x0[9] = float(omega_mpc[0])   # ωx (body ≈ world for small angles)
                    mpc_x0[10] = float(omega_mpc[1])  # ωy
                    mpc_x0[11] = float(omega_mpc[2])  # ωz
                    mpc_x0[12] = float(imu_rpy[2])     # yaw_ref

                    # --- Build reference trajectory (N, 13) ---
                    # NOTE: We use a DIRECT velocity ramp to v_to_cmd for vz, NOT
                    # the quintic polynomial. The quintic's v_to is limited by
                    # compression depth (v_to_feas = 2*dz/T2), which can be near
                    # zero when touchdown velocity is small (e.g. first hop from
                    # hand). This caused MPC to always output fz=fz_min, leaving
                    # all energy injection to the independent energy_comp (which
                    # MPC doesn't know about), breaking force consistency.
                    #
                    # With a direct ramp, MPC plans the full push trajectory itself,
                    # ensuring fx/fy (for attitude) and fz (for height) are jointly
                    # optimized within friction cone constraints.
                    N_mpc = int(self.cfg.mpc_horizon)
                    dt_mpc = float(self.cfg.mpc_dt)
                    mpc_xref = np.zeros((N_mpc, 13), dtype=float)
                    vx_des = float(desired_v_xy_w[0])
                    vy_des = float(desired_v_xy_w[1])
                    yaw_now = float(imu_rpy[2])
                    px_now = float(self._p_hat_w[0])
                    py_now = float(self._p_hat_w[1])
                    pz_now = float(self._p_hat_w[2])

                    # Desired takeoff velocity from hop_height_m (relative height).
                    h_target = float(max(0.05, float(self.cfg.hop_height_m)))
                    v_to_target = float(np.sqrt(2.0 * float(self.gravity) * h_target))
                    T_stance_total = float(max(0.05, float(self.cfg.stance_T)))
                    # Start push early enough for short-contact hops; ratio is configurable.
                    push_ratio = float(_clipf(float(self.cfg.mpc_push_start_ratio), 0.05, 0.6))
                    t_push_start = T_stance_total * push_ratio
                    T_push = float(max(1e-6, T_stance_total - t_push_start))

                    for k in range(N_mpc):
                        tk = float(t_in_stance + (k + 1) * dt_mpc)

                        # vz reference: smooth Hermite ramp 0 → v_to_cmd
                        if tk <= t_push_start:
                            vz_ref_k = 0.0
                            dz_ref_k = 0.0
                        else:
                            frac = float(min(1.0, max(0.0, (tk - t_push_start) / T_push)))
                            vz_ref_k = float(v_to_target * (3.0 * frac * frac - 2.0 * frac * frac * frac))
                            # Analytical integral of Hermite: ∫h(s)ds = s³ - s⁴/2
                            dz_ref_k = float(v_to_target * T_push * (frac ** 3 - 0.5 * frac ** 4))

                        mpc_xref[k, 0] = px_now + vx_des * (k + 1) * dt_mpc  # px
                        mpc_xref[k, 1] = py_now + vy_des * (k + 1) * dt_mpc  # py
                        mpc_xref[k, 2] = float(pz_now + dz_ref_k)             # pz (consistent with vz ramp)
                        mpc_xref[k, 3] = vx_des                               # vx desired
                        mpc_xref[k, 4] = vy_des                               # vy desired
                        mpc_xref[k, 5] = float(vz_ref_k)                      # vz: direct ramp to takeoff
                        mpc_xref[k, 6] = 0.0                                  # roll → 0
                        mpc_xref[k, 7] = 0.0                                  # pitch → 0
                        mpc_xref[k, 8] = yaw_now                              # yaw → hold
                        mpc_xref[k, 9] = 0.0                                  # ωx → 0
                        mpc_xref[k, 10] = 0.0                                 # ωy → 0
                        mpc_xref[k, 11] = 0.0                                 # ωz → 0
                        mpc_xref[k, 12] = yaw_now                             # yaw_ref

                    # --- Contact schedule: predict liftoff, set flight steps to 0 ---
                    # MIT MPC uses a gait planner for contact schedules.  For our single-
                    # legged hopper we know the approximate stance duration, so we tell
                    # MPC exactly when the foot lifts off.  This is critical: without it
                    # MPC plans attitude-correcting forces for steps that never execute,
                    # and under-prioritises driving ω→0 before the real liftoff.
                    t_remaining_stance = float(max(0.0, T_stance_total - t_in_stance))
                    contact_sched = np.array(
                        [int((k + 1) * dt_mpc <= t_remaining_stance) for k in range(N_mpc)],
                        dtype=int,
                    )
                    # Ensure at least the first step is in contact (we are in stance now)
                    if contact_sched[0] == 0:
                        contact_sched[0] = 1

                    # --- Foot position in world (moment arm for torque) ---
                    # r_foot_w already computed above (line ~1792) as R_wb @ (foot_b - com_b)

                    # --- Solve MPC ---
                    mpc_result = self._mpc.solve(
                        x0=mpc_x0,
                        x_ref_seq=mpc_xref,
                        contact_schedule=contact_sched,
                        m=float(self.mass),
                        g=float(self.gravity),
                        I_body=self.I_body,
                        r_foot_w=r_foot_w,
                        z_w=z_thrust_w,
                        T_base=float(thrust_sum_ref),
                    )
                    mpc_status = str(mpc_result.get("status", "unknown"))
                    mpc_u0 = np.asarray(mpc_result.get("u0", np.zeros(3)), dtype=float).reshape(3)

                    if mpc_status in ("solved", "solved inaccurate", "solved_inaccurate"):
                        # Exponential low-pass filter on horizontal forces to prevent
                        # solve-to-solve oscillation (the dominant 22-28 Hz shaking).
                        # Vertical force fz passes through unfiltered for responsive push.
                        alpha_fxy = float(_clipf(float(self.cfg.mpc_fxy_lpf_alpha), 0.0, 1.0))
                        f_ref[0] = alpha_fxy * float(mpc_u0[0]) + (1.0 - alpha_fxy) * float(self._mpc_f_ref_cache[0])
                        f_ref[1] = alpha_fxy * float(mpc_u0[1]) + (1.0 - alpha_fxy) * float(self._mpc_f_ref_cache[1])
                        f_ref[2] = float(mpc_u0[2])  # fz unfiltered
                        self._mpc_f_ref_cache[:] = f_ref[:]
                        mpc_used = True
                    elif bool(getattr(self.cfg, "mpc_hold_cache_on_fail", True)):
                        # Keep pure MPC->QP structure: on transient solver degradation,
                        # hold the last valid MPC force instead of falling back to default.
                        if float(np.linalg.norm(self._mpc_f_ref_cache)) > 1e-9:
                            f_ref[:] = self._mpc_f_ref_cache
                            mpc_u0 = np.asarray(self._mpc_f_ref_cache, dtype=float).reshape(3).copy()
                            mpc_used = True
                            mpc_status = f"cached_on_fail:{mpc_status}"
                except Exception:
                    mpc_status = "exception"
                    mpc_used = False
                    if bool(getattr(self.cfg, "mpc_hold_cache_on_fail", True)):
                        if float(np.linalg.norm(self._mpc_f_ref_cache)) > 1e-9:
                            f_ref[:] = self._mpc_f_ref_cache
                            mpc_u0 = np.asarray(self._mpc_f_ref_cache, dtype=float).reshape(3).copy()
                            mpc_used = True
                            mpc_status = "cached_on_exception"
            elif use_mpc_stance and (not run_mpc_now):
                # Between MPC solves: hold cached f_ref (standard MPC practice)
                f_ref[:] = self._mpc_f_ref_cache
                # Log meaningful MPC output on cached steps as well
                # (otherwise mpc_u0 appears as zeros every decimation interval).
                mpc_u0 = np.asarray(self._mpc_f_ref_cache, dtype=float).reshape(3).copy()
                mpc_used = True
                mpc_status = "cached"

            if not mpc_used:
                # SLIP stance: f_ref[2] = leg-axis spring magnitude; horizontal
                # attitude force comes from the SLIP side channel (below).
                f_ref[:] = 0.0

            # f_ref[2] filled by leg-axis virtual spring below; f_ref[0:2] stay 0.
        else:
            f_ref[:] = 0.0

        # ===== Stance axial/vertical force =====
        # World-vertical CoM-height impedance and push spring. f_cz is applied
        # along the world vertical by the SRB allocation below:
        #   f_cz = max(0, kz*(h_des - h) - bz*vz_up + kE*(E_des - E_sys)|_push)
        # Energy term is CASE-gated: only when qd_shift > 0 (leg extending),
        # and clamped to inject-only (never extract during compression).
        energy_comp_fz = 0.0
        energy_gate = False
        try:
            if bool(self._stance) and (not mpc_used):
                l0 = float(self.cfg.leg_l0_m)
                foot_b_now = np.asarray(foot_b, dtype=float).reshape(3)
                l_leg = float(np.linalg.norm(foot_b_now))
                foot_w_now = (R_wb_hat @ foot_b_now.reshape(3)).reshape(3)
                h_com = float(foot_w_now[2])
                # ===== Mode1 two-spring stance, WORLD-Z gated =====
                # TA-SLIP hybrid: when props are armed they carry
                # rho_st*m*g of the weight in stance and rho_up*m*g in
                # the ascent, so both springs are solved against the
                # EFFECTIVE gravities instead of g (see the hybrid
                # config block).
                m = float(self.mass)
                g = float(self.gravity)
                bz = float(getattr(self.cfg, "stance_kd_z", 20.0))
                # WORLD-Z spring coordinate (2026-07-19 per user, "不要
                # 腿长"): height deficit of the body below full leg
                # extension, x = l0 - h_com (h_com is the world-vertical
                # body height above the foot). The push spring force is
                # zero exactly when the BODY HEIGHT reaches l0 -- the
                # correct condition for vertical takeoff energy -- and
                # is consistent with the compression impedance, which
                # also acts on h_com.
                x_z = float(max(0.0, l0 - h_com))
                vz_up = -float(self._v_hat_w[2])
                fz_cap = float(self.cfg.stance_fz_max)

                props_z = bool(self._props_armed_rt)
                rho_st = (
                    float(_clipf(
                        float(self.cfg.prop_stance_base_thrust_ratio),
                        0.0, 0.8,
                    ))
                    if (props_z and bool(self.cfg.stance_use_props))
                    else 0.0
                )
                rho_up = (
                    float(_clipf(
                        float(self.cfg.prop_base_thrust_ratio), 0.0, 0.8
                    ))
                    if props_z
                    else 0.0
                )
                g_st = g * (1.0 - rho_st)
                g_up = g * (1.0 - rho_up)

                # LPF the world-Z velocity for this law only: raw
                # kinematic vz carries leg-vibration spikes (23:14 log
                # hop #3: -2.2 -> +2.4 m/s in 20 ms mid-compression)
                # that both poison the ~v^2 stiffness re-solve and
                # false-fire the PUSH gate.
                tau_vz = float(max(
                    0.0, float(self.cfg.stance_vz_lpf_tau_s)
                ))
                if self._mode1_vz_lpf is None:
                    self._mode1_vz_lpf = float(vz_up)
                else:
                    a_vz = (
                        1.0
                        if tau_vz <= 1e-12
                        else float(self.dt / (tau_vz + self.dt))
                    )
                    self._mode1_vz_lpf += a_vz * (
                        float(vz_up) - float(self._mode1_vz_lpf)
                    )
                vz_f = float(self._mode1_vz_lpf)

                # COMPRESSION: the ORIGINAL world-height impedance
                # (2026-07-19 rollback per user -- the adaptive-stiffness
                # spring rang the ~33 Hz leg mode; this fixed-gain law
                # never did). Preload at touchdown ~ kz*hop_height plus
                # a modest slope; depth is whatever m, v_td and kz give.
                kz = float(self.cfg.stance_kp_z)
                h_des = l0 + float(self.cfg.hop_height_m)
                f_comp = kz * (h_des - h_com) - bz * vz_up

                # ===== NRC continuous stance energy law (ACC 2020) =====
                # See the nrc_* config block for the full derivation.  ONE
                # smooth force law for the whole stance: spring + norm-
                # regulation pump.  No latch, no k re-solve, no blend --
                # this is the anti-chatter replacement for the Mode1
                # machinery below (which stays selectable via
                # stance_energy_law = "mode1").
                _law = str(getattr(
                    self.cfg, "stance_energy_law", "mode1"
                )).lower()
                nrc_on = (_law == "nrc")
                fb_on = (_law == "fbslip")
                px_on = (_law == "pogox")
                f_nrc_leg = 0.0
                if nrc_on:
                    k_nrc = float(max(1.0, float(self.cfg.nrc_k_n_m)))
                    om_n = float(np.sqrt(k_nrc / m))
                    # Height target -> norm target (the height coupling):
                    # take off at v_to and the prop-assisted ballistic arc
                    # tops out at hop_height_m (x big-jump gain, one hop).
                    h_tgt_n = (
                        float(max(0.0, float(self.cfg.hop_height_m)))
                        * float(max(1.0, float(self._nrc_big_gain)))
                        * float(self._nrc_h_trim)
                    )
                    v_to_n = float(np.sqrt(2.0 * g_up * h_tgt_n))
                    x1_n = (float(h_com) - l0) + (m * g_st / k_nrc)
                    x2_n = float(vz_f) / om_n
                    r_n = float(np.hypot(x1_n, x2_n))
                    r_star_n = float(np.hypot(
                        m * g_st / k_nrc, v_to_n / om_n
                    ))
                    # NRC-2 pump: zero at the bottom (x2 = 0), pumps over
                    # the whole stroke, vanishes on the limit cycle.
                    f_pump_n = (
                        -2.0 * m * float(self.cfg.nrc_kR) * om_n
                        * x2_n * (r_n - r_star_n)
                    )
                    # Extension fade (see nrc_pump_ext_fade_m): the pump
                    # rolls smoothly to zero over the last few cm before
                    # full extension -- the window where estimator lag
                    # would otherwise keep 100+ N firing past the target.
                    x_fade = float(getattr(
                        self.cfg, "nrc_pump_ext_fade_m", 0.0
                    ))
                    if x_fade > 1e-6:
                        f_pump_n *= float(_clipf(
                            (l0 - float(h_com)) / x_fade, 0.0, 1.0
                        ))
                    f_des_n = (
                        k_nrc * (l0 - float(h_com))
                        - float(max(0.0, float(self.cfg.nrc_bz)))
                        * float(vz_f)
                        + f_pump_n
                    )
                    # ---- split the demand: leg first, props take the rest ----
                    # The split point is the LEG's real authority
                    # (nrc_leg_fz_max ~ what the hip torque can actually
                    # deliver), NOT the stance_fz_max safety clamp: keying
                    # off the 500 N clamp made the residual identically
                    # zero (2026-08-02 log 063655: NRC demand p95 259 N,
                    # peak 430 N -> props never fired in stance while the
                    # leg command was being scaled to ~1/3 by the 10 Nm
                    # torque limit -- "腿无力" with idle props).
                    # With no props to hand the residual to, the leg is
                    # asked for everything (old behavior).
                    self._prop_energy_fz = 0.0
                    props_supp = (
                        bool(self.cfg.prop_energy_supplement_enable)
                        and bool(self._props_armed_rt)
                    )
                    leg_ceiling = fz_cap
                    if props_supp:
                        leg_ceiling = float(_clipf(
                            float(self.cfg.nrc_leg_fz_max), 10.0, fz_cap
                        ))
                    f_nrc_leg = float(_clipf(f_des_n, 0.0, leg_ceiling))
                    # Residual rides the prop COLLECTIVE, re-evaluated
                    # every tick (pure Fz channel -- the attitude
                    # differential never sees it).  leg + prop reproduce
                    # the NRC demand exactly until both saturate.
                    if props_supp:
                        f_cap_pr = (
                            float(_clipf(float(
                                self.cfg.prop_energy_max_ratio
                            ), 0.0, 3.0)) * m * g
                        )
                        pe = float(_clipf(
                            f_des_n - leg_ceiling, 0.0, f_cap_pr
                        ))
                        if bool(self._mode1_push_latched):
                            pe *= float(_clipf(float(getattr(
                                self.cfg, "prop_energy_push_scale", 1.0
                            )), 0.0, 1.0))
                        self._prop_energy_fz = float(pe)
                    energy_comp_fz = float(f_pump_n)
                    self._nrc_r = r_n
                    self._nrc_r_star = r_star_n
                    # Total stance demand before the leg/prop split: compare
                    # against f_ref_w2 + prop_energy_fz to see the split.
                    self._nrc_f_des = float(f_des_n)

                # ===== NORM-REGULATED ENERGY-ANCHORED STANCE =====
                # (config key "pogox"; see the pogox_* block).  NRC norm
                # regulation (ACC 2020) on the stroke-anchored spring:
                #   F = k_v*x - 2*m*kR*vz_n*(r - r*)
                # dE/dt = -2*m*kR*vz_n^2*(r - r*) -> r converges globally
                # to the limit cycle -- no sub-phase machine, no caps.
                f_px = 0.0
                if px_on:
                    self._prop_energy_fz = 0.0
                    # MATLAB stance_phase_ODE.m:
                    #   l = norm(robot_pos-foot_pos)
                    #   spring_force = -k_spring*(l-l0)
                    # Our q_shift = l-l0, hence x_leg = l0-l = -q_shift.
                    # Keep this separate from x_z=l0-h_com: using x_z
                    # left ~70 N at q_shift≈0 when the body was tilted
                    # (log 104932), then flight swing torque reversed hard.
                    x_leg_px = float(max(0.0, -float(q_shift)))
                    h_star = float(max(0.0, float(self.cfg.hop_height_m)))
                    # Per-hop apex return map (shared with NRC, see
                    # nrc_apex_trim_*): the in-stance pump regulates the
                    # ESTIMATED cycle energy, so taper losses / vz bias land
                    # at a biased apex every hop (log 200745: 1.9-3.4 cm vs
                    # the 5 cm target, all 6 hops).  The trim, updated at
                    # each touchdown from the drift-free flight-time apex,
                    # scales the height target (and through it k_v and r*)
                    # until the MEASURED apex matches hop_height_m.
                    h_star *= float(_clipf(
                        float(self._nrc_h_trim),
                        float(self.cfg.nrc_apex_trim_min),
                        float(self.cfg.nrc_apex_trim_max),
                    ))
                    if bool(self._big_jump_pending):
                        # One-hop RB boost: taller apex target this stance.
                        h_star *= float(max(1.0, float(
                            self.cfg.big_jump_height_gain
                        )))
                    v_star = float(np.sqrt(2.0 * g_up * h_star))
                    X_str = float(max(0.02, float(
                        self.cfg.leg_stroke_max_m
                    )))
                    k_v = (
                        (m * v_star * v_star
                         + 2.0 * m * g_st * X_str)
                        / (X_str * X_str)
                    )
                    E_star = 0.5 * m * v_star * v_star + m * g_st * l0
                    # AXIAL (encoder) leg rate for the pump -- NOT the
                    # world-kinematic vz.  Once the leg unloads near full
                    # extension the foot slips / body pivots about it and
                    # the kinematic chain reads a phantom -2.5 m/s "fall"
                    # (log 105750): the pump then re-fired +60 N at push
                    # end.  q_shift is l-l0 straight from the shift
                    # encoder; with the foot planted d(q_shift)/dt IS the
                    # body vertical rate, and slip/pivot cannot touch it.
                    if self._px_qs_prev is None:
                        vz_ax_raw = float(vz_f)
                    else:
                        vz_ax_raw = (
                            (float(q_shift) - float(self._px_qs_prev))
                            / float(self.dt)
                        )
                    self._px_qs_prev = float(q_shift)
                    # Pump-path LPF (pogox_vz_ax_lpf_tau_s): the tick
                    # differentiation above is white-noisy at 500 Hz and
                    # the shared 20 ms stance tau let +-0.2 m/s flicker
                    # through, which the 2*m*kr*|r-r*| gain turned into
                    # +-12 N/tick force chatter (log 200745).
                    tau_ax = float(max(0.0, float(getattr(
                        self.cfg, "pogox_vz_ax_lpf_tau_s", 0.03
                    ))))
                    if self._px_vz_ax is None:
                        self._px_vz_ax = float(vz_ax_raw)
                    else:
                        a_ax = (
                            1.0
                            if tau_ax <= 1e-12
                            else float(self.dt / (tau_ax + self.dt))
                        )
                        self._px_vz_ax += a_ax * (
                            float(vz_ax_raw) - float(self._px_vz_ax)
                        )
                    vz_ax_f = float(self._px_vz_ax)
                    # Model-based lag compensation (see pogox_vz_lag_s):
                    # predict vz forward with the APPLIED force -- known
                    # exactly, no estimator in the correction.
                    t_lag_px = float(max(0.0, float(
                        self.cfg.pogox_vz_lag_s
                    )))
                    f_prev_px = (
                        float(self._px_f_prev)
                        if self._px_f_prev is not None
                        else m * g_st
                    )
                    # Sign-preserving clamp on the lag correction: its
                    # job is PHASE alignment, never sign flips.  In the
                    # liftoff taper F ~ 0, so the raw correction is a
                    # constant -g*t_lag ~ -0.47 m/s that turned a real
                    # +0.4 m/s ascent into a "descent" -- the pump then
                    # went NEGATIVE in the last 5 mm of every push and
                    # killed the jump 3 mm short (logs 113953/114543/
                    # 115333, all first pushes died at q ~ -0.003).
                    corr_px = (f_prev_px / m - g_st) * t_lag_px
                    corr_px = float(_clipf(
                        corr_px, -abs(vz_ax_f), abs(vz_ax_f)
                    ))
                    vz_n = vz_ax_f + corr_px
                    E_now = (
                        0.5 * m * vz_n * vz_n
                        + m * g_st * float(h_com)
                        + 0.5 * k_v * x_leg_px * x_leg_px
                    )
                    dE = float(E_star - E_now)
                    # NRC phase coordinates on the anchored spring.
                    om_px = float(np.sqrt(k_v / m))
                    x_eq_px = m * g_st / k_v
                    r_px = float(np.hypot(
                        x_leg_px - x_eq_px, vz_n / om_px
                    ))
                    # Liftoff margin: the target cycle must carry enough
                    # energy AT THE TAPER BOUNDARY (x = d) to coast the
                    # final dead zone -- taper gives up ~all push work
                    # over the last d meters while gravity keeps pulling.
                    # Aim r* at vz = sqrt(v*^2 + 2*g*d) instead of v* at
                    # x = 0, so the cycle crosses q_shift >= 0 with v*
                    # in hand rather than arriving exactly empty (the
                    # recurring "died 3 mm short" failure).
                    d_marg_px = float(max(0.0, float(
                        self.cfg.pogox_lo_taper_m
                    )))
                    v_star_eff = float(np.sqrt(
                        v_star * v_star + 2.0 * g_up * d_marg_px
                    ))
                    r_star_px = float(np.hypot(
                        x_eq_px, v_star_eff / om_px
                    ))
                    kr_px = float(max(0.0, float(self.cfg.pogox_kr)))
                    v_eps = float(max(0.0, float(
                        self.cfg.pogox_seed_vz_mps
                    )))
                    if ((abs(vz_n) < v_eps)
                            and (dE > 0.05 * E_star)
                            and (k_v * x_leg_px <= m * g_st)):
                        # Standstill seed: partial weight support, gravity
                        # starts the oscillation (pump power is F*vz = 0
                        # at rest).  Spring-force gate: only near the
                        # standing equilibrium -- at the deep-compression
                        # vz zero crossing the spring law must hold.
                        f_px = float(_clipf(float(
                            self.cfg.pogox_seed_weight_frac
                        ), 0.0, 1.0)) * m * g_st
                        energy_comp_fz = 0.0
                        self._px_pump_prev = 0.0
                    else:
                        f_pump_px = (
                            -2.0 * m * kr_px * vz_n
                            * (r_px - r_star_px)
                        )
                        # SLACK LIMIT: the 1D law assumes contact never
                        # breaks, so the pump happily cuts F to 0 on the
                        # way down to "gain energy".  On hardware that
                        # de-loads the foot at touchdown (log 114543 hop
                        # 2: pump -46 N -> f_ref = 0 at TD), the body
                        # free-falls onto the spring and the bottom
                        # BOUNCES -- impact loss ate the hop.  A real
                        # spring is never slacker than k*x; allow the
                        # pump to soften it at most 50%.
                        beta_px = float(_clipf(float(getattr(
                            self.cfg, "pogox_min_spring_frac", 0.5
                        )), 0.0, 1.0))
                        f_pump_px = float(max(
                            f_pump_px,
                            -(1.0 - beta_px) * k_v * x_leg_px,
                        ))
                        # Optional pump slew (pogox_pump_slew_nps): default
                        # OFF.  Hard-cutting the pump pinned energy inject
                        # on a self-imposed wall (log 205009: 64% of stance
                        # ticks at ±4 N) and broke the single hop_height
                        # knob.  Smoothness comes from the vz LPF on the
                        # pump INPUT, not from clipping the OUTPUT.
                        slew_px = float(max(0.0, float(getattr(
                            self.cfg, "pogox_pump_slew_nps", 0.0
                        ))))
                        if slew_px > 0.0:
                            dmax_px = slew_px * float(self.dt)
                            pump_prev_px = float(self._px_pump_prev)
                            f_pump_px = float(_clipf(
                                f_pump_px,
                                pump_prev_px - dmax_px,
                                pump_prev_px + dmax_px,
                            ))
                        f_px = k_v * x_leg_px + f_pump_px
                        energy_comp_fz = float(f_pump_px)
                        self._px_pump_prev = float(f_pump_px)
                    self._nrc_r = float(r_px)
                    self._nrc_r_star = float(r_star_px)
                    # Physical saturations ONLY: unilateral contact and
                    # the liftoff unload boundary condition.  The joint
                    # torque rescale downstream is the hardware wall.
                    f_px = float(max(0.0, f_px))
                    d_px = float(max(0.0, float(
                        self.cfg.pogox_lo_taper_m
                    )))
                    if d_px > 1e-6:
                        f_px *= float(_clipf(
                            x_leg_px / d_px, 0.0, 1.0
                        ))
                    self._px_f_prev = float(f_px)
                    self._nrc_f_des = float(f_px)  # telemetry reuse

                # PUSH latch on the FILTERED world-Z velocity: the body
                # has passed the bottom when vz turns positive.
                # Debounced, blocked for the first
                # stance_push_min_stance_s of the stance (real bottoms
                # arrive later; only chatter latched at 16 ms), then
                # held until liftoff.
                # 2026-08-02: the latch is now shared by BOTH energy laws.
                # NRC used to relabel COMP/PUSH from the instantaneous vz
                # sign every tick, and stance vibration (+-1..2 m/s spikes
                # on kinematic vz) flipped the label 2-5x per stance (log
                # 103423).  The debounced ONE-WAY latch gives exactly two
                # stages per stance; under NRC it feeds ONLY the phase
                # label/telemetry -- the continuous force law is untouched.
                # latch_evt (the Mode1 push-spring re-solve) stays
                # mode1-only.
                td_t_z = (
                    float(self._td_t)
                    if self._td_t is not None
                    else float(self.sim_time)
                )
                t_in_st_z = float(self.sim_time) - td_t_z
                latch_evt = False

                # ===== FB-SLIP stance (37a1475 port) =====
                # COMP: positional reception (no velocity estimate in the
                # loop) -> BOTTOM: vz-settle confirm OR position rebound
                # -> PUSH: constant force sized once at the bottom, held
                # open-loop.  See the fbslip config block.
                f_fb = 0.0
                if fb_on:
                    self._prop_energy_fz = 0.0
                    f_max_z = float(max(
                        2.5 * m * g,
                        float(self.cfg.leg_force_budget_g) * m * g,
                    ))
                    cap_fb = float(min(fz_cap, f_max_z))
                    # --- positional reception force (COMP law) ---
                    ramp = float(_clipf(
                        t_in_st_z
                        / float(max(1e-3, float(
                            self.cfg.stance_brake_ramp_s
                        ))),
                        0.0, 1.0,
                    ))
                    s_tr = float(max(0.0, x_z - float(self._fb_x_td)))
                    f0_recv = float(_clipf(float(
                        self.cfg.stance_recv_preload_frac
                    ), 0.0, 1.0)) * m * g_st
                    f_mid = float(max(
                        f0_recv,
                        float(_clipf(float(
                            self.cfg.stance_recv_tgt_weight_frac
                        ), 0.5, 1.0)) * m * g_st,
                    ))
                    trav_pl = float(max(0.02, float(self._fb_trav_plan)))
                    s_tgt = float(_clipf(
                        float(self._fb_s_tgt), 0.01, 0.9 * trav_pl
                    ))
                    if s_tr <= s_tgt:
                        f_brk = (
                            f0_recv
                            + (s_tr / max(1e-6, s_tgt))
                            * (f_mid - f0_recv)
                        )
                    else:
                        catch_span = float(_clipf(float(
                            self.cfg.stance_recv_catch_span_m
                        ), 0.005, max(0.005, trav_pl - s_tgt)))
                        f_brk = (
                            f_mid
                            + float(_clipf(
                                (s_tr - s_tgt) / catch_span, 0.0, 1.0
                            )) * max(0.0, cap_fb - f_mid)
                        )
                    f_recv = ramp * float(f_brk)
                    # First-order blend on the reception reference
                    # (feedforward shaping only, "腿抖" fix from 37a1475).
                    tau_fzb = float(max(0.0, float(
                        self.cfg.stance_fz_blend_tau_s
                    )))
                    if self._fb_fcomp_lpf is None:
                        self._fb_fcomp_lpf = 0.0
                    a_fzb = (
                        1.0 if tau_fzb <= 1e-12
                        else float(self.dt / (tau_fzb + self.dt))
                    )
                    self._fb_fcomp_lpf += a_fzb * (
                        float(f_recv) - float(self._fb_fcomp_lpf)
                    )
                    f_recv = float(self._fb_fcomp_lpf)

                    # --- BOTTOM: the single phase event ---
                    if not bool(self._mode1_push_latched):
                        self._fb_xz_max = float(max(
                            float(self._fb_xz_max), float(x_z)
                        ))
                        v_settle = float(max(1e-3, float(
                            self.cfg.stance_bottom_settle_mps
                        )))
                        gate_t = float(
                            self.cfg.stance_push_min_stance_s
                        )
                        bottom_now = (
                            float(vz_f) >= -v_settle
                            and t_in_st_z >= gate_t
                        )
                        if bottom_now:
                            self._mode1_push_confirm_count += 1
                        else:
                            self._mode1_push_confirm_count = 0
                        n_confirm = int(max(1, int(np.ceil(
                            float(self.cfg.stance_bottom_confirm_s)
                            / float(max(1e-4, self.dt))
                        ))))
                        # Position rebound: the body measurably RISING
                        # off its deepest point is past the bottom by
                        # definition (velocity-consistency gated).
                        reb_m = float(max(1e-4, float(
                            self.cfg.stance_bottom_rebound_m
                        )))
                        rebound_now = (
                            t_in_st_z >= gate_t
                            and (float(self._fb_xz_max) - float(x_z))
                            >= reb_m
                            and float(vz_f) >= -v_settle
                        )
                        if (self._mode1_push_confirm_count >= n_confirm
                                or rebound_now):
                            self._mode1_push_latched = True
                            # --- size the push force ONCE (taper-aware
                            # energy balance, see stance_push_taper_m):
                            #   F_push*(x0 - d/2)
                            #     = 0.5*m*v_to^2 + m*g_st*x0
                            # The taper F(x) = F_push*min(1, x/d) gives
                            # up d/2*F_push of work near liftoff; the
                            # denominator repays it so the coupled
                            # (height -> v_to -> F_push) chain still
                            # delivers the full takeoff energy.  d = 0
                            # reduces to m*g_st + m*v_to^2/(2*x0).
                            x0 = float(max(0.01, x_z))
                            h_gain = 1.0
                            if bool(self._big_jump_pending):
                                h_gain = float(max(1.0, float(
                                    self.cfg.big_jump_height_gain
                                )))
                                self._big_jump_pending = False
                            v_to = float(self._v_to_cmd) * float(
                                np.sqrt(h_gain)
                            )
                            d_tp = float(_clipf(
                                float(getattr(
                                    self.cfg,
                                    "stance_push_taper_m", 0.0,
                                )),
                                0.0, 0.8 * x0,
                            ))
                            self._fb_push_taper = d_tp
                            e_need = (
                                0.5 * m * v_to * v_to
                                + m * g_st * x0
                            )
                            self._fb_push_f = float(min(
                                e_need
                                / max(0.2 * x0, x0 - 0.5 * d_tp),
                                cap_fb,
                            ))
                            self._mode1_x0 = x0
                            # Blend starts from the compression force
                            # but NEVER from above the push target
                            # (down-steps just unload the leg).
                            self._mode1_boost_f_state = float(min(
                                max(0.0, f_recv),
                                float(self._fb_push_f),
                            ))
                    if bool(self._mode1_push_latched):
                        # PUSH: latched constant force, open loop,
                        # tapered to zero over the final extension so
                        # the force is continuous through liftoff.
                        tau_blend = float(max(0.0, float(
                            self.cfg.stance_push_blend_tau_s
                        )))
                        a_blend = (
                            1.0 if tau_blend <= 1e-12
                            else float(_clipf(
                                float(self.dt)
                                / (tau_blend + float(self.dt)),
                                0.0, 1.0,
                            ))
                        )
                        self._mode1_boost_f_state += a_blend * (
                            float(self._fb_push_f)
                            - float(self._mode1_boost_f_state)
                        )
                        f_fb = float(self._mode1_boost_f_state)
                        # Liftoff taper POST-blend: position-driven and
                        # deterministic (the blend LPF must not smear
                        # it, or force is still on the foot at LO).
                        d_tp = float(max(0.0, float(getattr(
                            self, "_fb_push_taper", 0.0
                        ))))
                        if d_tp > 1e-6:
                            f_fb *= float(_clipf(
                                x_z / d_tp, 0.0, 1.0
                            ))
                        energy_comp_fz = float(max(
                            0.0, f_fb - m * g_st
                        ))
                    else:
                        f_fb = float(f_recv)

                    # ---- leg / prop Fz split (stay under the 9 Nm wall) ----
                    # f_fb is the TOTAL world-Z demand from the FB-SLIP law.
                    # Cap the LEG at nrc_leg_fz_max (~100 N << ~135 N hard
                    # limit); the residual rides the prop COLLECTIVE so the
                    # torque rescale never has to steal attitude authority.
                    # f_ref / springForce keep the LEG share only -- the
                    # prop share is NOT added back into f_ref (that would
                    # re-ask the leg for the residual and defeat the cap).
                    props_supp = (
                        bool(self.cfg.prop_energy_supplement_enable)
                        and bool(self._props_armed_rt)
                    )
                    leg_ceiling = float(fz_cap)
                    if props_supp:
                        leg_ceiling = float(_clipf(
                            float(self.cfg.nrc_leg_fz_max), 10.0, fz_cap
                        ))
                    f_des_fb = float(f_fb)
                    f_fb = float(_clipf(f_des_fb, 0.0, leg_ceiling))
                    self._prop_energy_fz = 0.0
                    if props_supp:
                        f_cap_pr = (
                            float(_clipf(float(
                                self.cfg.prop_energy_max_ratio
                            ), 0.0, 3.0)) * m * g
                        )
                        pe = float(_clipf(
                            f_des_fb - leg_ceiling, 0.0, f_cap_pr
                        ))
                        # PUSH softens the prop Fz; COMP keeps the full
                        # residual so the catch still offloads the leg.
                        if bool(self._mode1_push_latched):
                            pe *= float(_clipf(float(getattr(
                                self.cfg, "prop_energy_push_scale", 1.0
                            )), 0.0, 1.0))
                        self._prop_energy_fz = float(pe)

                # For pogox the latch (and the re-arm below) must watch
                # the ENCODER axial rate, same signal as the pump: the
                # world-kinematic vz is chaotic in stance (log 112059
                # hop 2 never confirmed vz>0.05 before liftoff, so the
                # whole push stayed labeled COMP; log 113953 latched
                # DURING descent).
                vz_latch = (
                    float(self._px_vz_ax)
                    if (px_on and self._px_vz_ax is not None)
                    else float(vz_f)
                )
                if (not fb_on) and not bool(self._mode1_push_latched):
                    if (
                        vz_latch > float(self.cfg.stance_push_vz_mps)
                        and t_in_st_z >= float(
                            self.cfg.stance_push_min_stance_s
                        )
                    ):
                        self._mode1_push_confirm_count += 1
                    else:
                        self._mode1_push_confirm_count = 0
                    if (
                        self._mode1_push_confirm_count
                        >= int(max(1, int(
                            self.cfg.stance_push_confirm_steps
                        )))
                    ):
                        self._mode1_push_latched = True
                        # pogox/nrc are continuous laws: the latch is
                        # telemetry only, no Mode1 spring re-solve.
                        # ONE-WAY only: COMP -> PUSH -> FLIGHT.
                        # No PUSH -> COMP re-arm inside the same stance.
                        latch_evt = (not nrc_on) and (not px_on)

                if latch_evt:
                    # Push spring: released work from the current depth
                    # x0 back to l0 equals the takeoff kinetic energy
                    # for hop_height_m, the lift against the STANCE
                    # effective gravity, and the per-hop loss estimate
                    # learned by the apex return map:
                    #   0.5*k_push*x0^2
                    #     = 0.5*m*v_to^2 + m*g_st*x0 + E_loss
                    # with v_to = sqrt(2*g_up*h): the ascent collective
                    # keeps pushing after liftoff, so the leg only has
                    # to supply the (1 - rho_up) share of the apex
                    # energy. x0 is the WORLD-Z height deficit at the
                    # bottom, not the leg compression.
                    x0 = float(max(5e-3, x_z))
                    # RB gamepad big-jump: solve THIS push for a taller
                    # apex, one hop only.
                    h_tgt = float(max(0.0, float(self.cfg.hop_height_m)))
                    if bool(self._big_jump_pending):
                        h_tgt *= float(max(1.0, float(
                            self.cfg.big_jump_height_gain
                        )))
                        self._big_jump_pending = False
                    v_to = float(np.sqrt(2.0 * g_up * h_tgt))
                    E_need = (
                        0.5 * m * v_to * v_to
                        + m * g_st * x0
                        + float(self._mode1_Eloss)
                    )
                    E_need = float(max(0.25 * m * g_st * x0, E_need))
                    k_push = 2.0 * E_need / (x0 * x0)
                    self._mode1_k_boost = float(
                        min(k_push, fz_cap / x0)
                    )
                    self._mode1_x0 = x0
                    # ===== PROPELLER ENERGY SUPPLEMENT (decoupled) =====
                    # The leg force budget caps the spring at
                    # k_boost <= fz_cap/x0, so it can store at most
                    # E_leg = 0.5*k_boost*x0^2 of the required E_need.
                    # The unmet share rides the prop COLLECTIVE.  Unlike
                    # the leg, the props are not stroke-limited: the force
                    # acts over the PUSH stroke x0 AND the ascent h_tgt
                    # (it fades continuously to zero at apex, see the
                    # flight collective block), so the work budget is
                    #     F_prop * (x0 + h_tgt) = E_def
                    #     F_prop = min(E_def/(x0 + h_tgt),
                    #                  prop_energy_max_ratio * m*g).
                    # This enters ONLY the Fz channel of the allocator (a
                    # pure collective adds zero moment): the attitude
                    # channel never sees it -- leg and prop energy paths
                    # stay decoupled by construction.  Any apex overshoot
                    # from the ascent continuation is absorbed by the
                    # discrete apex return map (E_loss adapts down), so
                    # the two energy sources stay COUPLED hop-to-hop
                    # through one physical measurement, not through an
                    # algebraic loop inside the tick.
                    self._prop_energy_fz = 0.0
                    if (bool(self.cfg.prop_energy_supplement_enable)
                            and bool(self._props_armed_rt)):
                        E_leg = 0.5 * float(self._mode1_k_boost) * x0 * x0
                        E_def = float(max(0.0, E_need - E_leg))
                        f_cap_pr = (
                            float(_clipf(float(
                                self.cfg.prop_energy_max_ratio
                            ), 0.0, 0.8)) * m * g
                        )
                        stroke = float(max(1e-2, x0 + h_tgt))
                        self._prop_energy_fz = float(min(
                            E_def / stroke, f_cap_pr
                        ))
                    # Blend starts from the compression force for
                    # continuity.
                    self._mode1_boost_f_state = float(max(0.0, f_comp))

                compress_active = not bool(self._mode1_push_latched)
                energy_gate = (
                    (not nrc_on) and (not fb_on) and (not px_on)
                ) and bool(
                    self._mode1_push_latched
                ) and bool(
                    getattr(self.cfg, "use_energy_compensation", True)
                )
                if px_on:
                    springForce_scalar = float(f_px)
                elif nrc_on:
                    springForce_scalar = float(f_nrc_leg)
                elif fb_on:
                    springForce_scalar = float(f_fb)
                elif energy_gate:
                    # Pure spring during push (no damping), on the
                    # WORLD-Z height deficit: force reaches zero exactly
                    # when the body height reaches l0, moving at v_to.
                    f_push_tgt = float(self._mode1_k_boost) * x_z
                    tau_blend = float(max(
                        0.0, float(self.cfg.stance_push_blend_tau_s)
                    ))
                    a_blend = (
                        1.0
                        if tau_blend <= 1e-12
                        else float(_clipf(
                            float(self.dt)
                            / (tau_blend + float(self.dt)),
                            0.0,
                            1.0,
                        ))
                    )
                    self._mode1_boost_f_state += a_blend * (
                        f_push_tgt - self._mode1_boost_f_state
                    )
                    springForce_scalar = float(
                        self._mode1_boost_f_state
                    )
                    # Log the extra force above the plain impedance.
                    energy_comp_fz = float(max(
                        0.0,
                        springForce_scalar - float(max(0.0, f_comp)),
                    ))
                else:
                    springForce_scalar = float(f_comp)

                if springForce_scalar < 0.0:
                    springForce_scalar = 0.0

                # RT transition first takeoff (runtime compat): a plain
                # virtual spring from the static P4 pose.  Overrides the
                # compression/push results above; those are landing logic
                # and are not physically applicable to a standing launch.
                if bool(getattr(
                    self.cfg, "rt_first_hop_spring_active", False
                )):
                    k_rt = float(max(0.0, float(
                        self.cfg.rt_first_hop_spring_k_n_m
                    )))
                    d_rt = float(max(0.0, float(
                        self.cfg.rt_first_hop_spring_d_n_s_m
                    )))
                    springForce_scalar = float(max(
                        0.0, m * g_st + k_rt * x_z - d_rt * vz_f
                    ))
                    self._mode1_push_latched = True
                    self._mode1_k_boost = float(min(
                        springForce_scalar, fz_cap
                    ))
                    self._mode1_x0 = float(x_z)
                    self._mode1_push_confirm_count = 0
                    compress_active = False
                    energy_gate = False
                    energy_comp_fz = 0.0
                    self._prop_energy_fz = 0.0

                f_ref[2] = float(springForce_scalar)
                # Prop residual rides thrust_sum_ref (see allocator below),
                # NOT f_ref: putting it back into f_ref would re-ask the
                # leg for the same Newtons and pin it on the 9 Nm wall.
                # Exception -- Mode1 legacy: its f_ref was historically the
                # total spring; keep that A/B path unchanged.
                if (
                    (not fb_on) and (not nrc_on) and (not px_on)
                    and bool(getattr(
                        self.cfg, "prop_energy_supplement_enable", False
                    ))
                    and bool(self._props_armed_rt)
                ):
                    f_ref[2] += float(self._prop_energy_fz)
                if px_on:
                    # POGOX contract: NO intermediate force cap.  Only the
                    # unilateral contact bound here; the joint-torque
                    # proportional rescale (+ prop Fz assist) downstream
                    # is the one and only saturation.
                    f_ref[2] = float(max(0.0, float(f_ref[2])))
                else:
                    f_ref[2] = float(_clipf(
                        float(f_ref[2]),
                        float(self.cfg.stance_fz_min),
                        float(self.cfg.stance_fz_max),
                    ))
        except Exception:
            pass

        # NOTE: MODEE_DBG_STANCE_ZERO_FZ no longer zeroes f_ref[2] here. fz must stay in the QP
        # so the friction cone |fxy| <= mu*fz has a budget and the attitude (fxy) force can be
        # generated. The fz contribution is instead removed downstream, only when the solved
        # contact force is mapped to leg joint torque (see stance torque mapping below).

        if bool(self._stance):
            self._f_ref_z_prev = float(f_ref[2])
            self._f_ref_xy_prev[:] = np.asarray(f_ref[0:2], dtype=float).reshape(2)
        else:
            self._f_ref_z_prev = 0.0
            self._f_ref_xy_prev[:] = 0.0

        # Friction cone is enforced by QP constraints; no need to clip f_ref here.

        # ===== Attitude torque: direct PD on the CASE SO(3) error =====
        # Stance and flight share the same PD structure:
        #   tau_b = -kR * e_R - kW * omega
        # In stance, leg fxy satisfies r_b × f_b = tau_b (body FRD, same frame as foot_b).
        # In flight, Tau_des is realized by propeller differential thrust.
        tau_b_stance_des = np.zeros(3, dtype=float)
        tau_b_att_des = np.zeros(3, dtype=float)

        # Raw gyro for P-error (e_R) and prop D-term; stance leg kW uses observer.
        omega_raw = np.asarray(imu_gyro_b, dtype=float).reshape(3)

        # ---- Stance kW rate OBSERVER (see stance_kw_obs_* in ModeEConfig) ----
        # Runs EVERY tick (stance + flight) so it is warm at touchdown:
        #   predict with last tick's commanded body torque through J,
        #   correct toward the raw gyro at gain k_obs.
        # Flight with props off predicts tau=0 (torque-free body) -- exact.
        omega_obs_xy = np.array([float(omega_raw[0]), float(omega_raw[1])], dtype=float)
        kw_obs_on = bool(getattr(self.cfg, "stance_kw_obs_en", False))
        if kw_obs_on and np.isfinite(omega_obs_xy).all():
            k_obs = float(_clipf(float(getattr(self.cfg, "stance_kw_obs_k", 0.09)), 0.0, 1.0))
            J_diag = np.asarray(self.cfg.I_body_diag, dtype=float).reshape(3)
            if not bool(self._kw_obs_init):
                self._kw_obs_w[0] = float(omega_raw[0])
                self._kw_obs_w[1] = float(omega_raw[1])
                self._kw_obs_init = True
            else:
                self._kw_obs_w[0] += float(self._kw_obs_tau_prev[0]) / max(1e-6, float(J_diag[0])) * float(self.dt)
                self._kw_obs_w[1] += float(self._kw_obs_tau_prev[1]) / max(1e-6, float(J_diag[1])) * float(self.dt)
                self._kw_obs_w[0] += k_obs * (float(omega_raw[0]) - float(self._kw_obs_w[0]))
                self._kw_obs_w[1] += k_obs * (float(omega_raw[1]) - float(self._kw_obs_w[1]))
            if np.isfinite(self._kw_obs_w).all():
                omega_obs_xy = self._kw_obs_w.copy()
            else:
                self._kw_obs_w[0] = float(omega_raw[0])
                self._kw_obs_w[1] = float(omega_raw[1])
                omega_obs_xy = self._kw_obs_w.copy()

        if bool(self._stance):
            tau_rp_max = float(self.cfg.stance_tau_rp_max)
            kR_xy = float(self.cfg.stance_kpp)
            kW_xy = float(self.cfg.stance_kpd)
            omega_b = omega_raw
            tau_b_stance = np.zeros(3, dtype=float)
            # kW rate source: raw gyro unless the optional observer is enabled.
            tau_b_stance[0] = -kR_xy * float(e_R[0]) - kW_xy * float(omega_obs_xy[0])
            tau_b_stance[1] = -kR_xy * float(e_R[1]) - kW_xy * float(omega_obs_xy[1])
            # Project off the point-foot leg axis before norm limiting; that
            # component is physically undeliverable by a contact force.
            u_leg_att = np.asarray(foot_b, dtype=float).reshape(3)
            u_leg_n = float(np.linalg.norm(u_leg_att))
            if u_leg_n > 1e-6:
                u_leg_att = u_leg_att / u_leg_n
                tau_b_stance = (
                    tau_b_stance
                    - float(np.dot(tau_b_stance, u_leg_att)) * u_leg_att
                )
            tau_b_stance_des = tau_b_stance.copy()
            tau_b_att_des = tau_b_stance.copy()
            # DEBUG: kill stance attitude torque so QP produces NO horizontal contact force (fxfy=0).
            if bool(self._dbg_stance_zero_fxfy):
                tau_b_att_des[:] = 0.0
                tau_b_stance_des[:] = 0.0
        else:
            # Flight propeller attitude PD.
            omega_b = omega_raw
            if not bool(props_enabled_ctrl):
                # No propellers physically available: do not request flight attitude torques.
                tau_rp_max = 0.0
                tau_b_att_des[:] = 0.0
            else:
                tau_rp_max = float(self.cfg.flight_tau_rp_max)
                kR = float(self.cfg.flight_kR)
                kW = float(self.cfg.flight_kW)
                # Raw gyro (no filtering anywhere, per user 2026-07-10).
                tau_b = np.zeros(3, dtype=float)
                tau_b[0] = (-kR * float(e_R[0])) - (kW * float(omega_b[0]))
                tau_b[1] = (-kR * float(e_R[1])) - (kW * float(omega_b[1]))
                tau_b[2] = 0.0
                tau_b_att_des = tau_b.copy()
            if bool(self._stance):
                tau_b_stance_des = tau_b_att_des.copy()
        
        # Norm-based torque limiting before projection to the world-frame wrench.
        # MATLAB virtual_spring limits the FULL norm of the leg-axis-projected
        # hip torque (stance gains a small z comp from the projection; flight
        # z stays 0 so this is unchanged there).
        tau_rp_norm = float(np.linalg.norm(tau_b_att_des))
        if tau_rp_norm > tau_rp_max and tau_rp_max > 0.0 and tau_rp_norm > 1e-9:
            scale = float(tau_rp_max) / tau_rp_norm
            tau_b_att_des = (tau_b_att_des * scale).astype(float)
        if bool(self._stance):
            tau_b_stance_des = tau_b_att_des.copy()
        tau_w = (R_wb_hat @ tau_b_att_des.reshape(3)).reshape(3)
        Tau_des = np.array([float(tau_w[0]), float(tau_w[1]), 0.0], dtype=float)
        # ===== Propeller attitude demand (DECOUPLED, additive) =====
        # Flight: Tau_des IS the flight_kR/kW PD -- props track it directly.
        # Stance: Tau_des is the LEG's stance_kpp/kpd demand (full, never
        # reduced).  Props independently recompute the SAME R_des error
        # with flight_kR/kW / flight_tau_rp_max and track that through the
        # differential channel.  Additive parallel loops -- not an HFA
        # residual split -- so the leg's attitude force/torque is intact.
        Tau_prop_des = Tau_des.copy()
        Tau_des_dbg = Tau_des.copy()
        if bool(self._stance) and bool(props_enabled_ctrl):
            tau_rp_max_p = float(self.cfg.flight_tau_rp_max)
            kR_p = float(self.cfg.flight_kR)
            kW_p = float(self.cfg.flight_kW)
            tau_b_prop = np.zeros(3, dtype=float)
            tau_b_prop[0] = (
                (-kR_p * float(e_R[0])) - (kW_p * float(omega_raw[0]))
            )
            tau_b_prop[1] = (
                (-kR_p * float(e_R[1])) - (kW_p * float(omega_raw[1]))
            )
            n_p = float(np.linalg.norm(tau_b_prop))
            if n_p > tau_rp_max_p and tau_rp_max_p > 0.0 and n_p > 1e-9:
                tau_b_prop = (
                    tau_b_prop * (tau_rp_max_p / n_p)
                ).astype(float)
            tau_w_prop = (
                R_wb_hat @ tau_b_prop.reshape(3)
            ).reshape(3)
            Tau_prop_des = np.array(
                [float(tau_w_prop[0]), float(tau_w_prop[1]), 0.0],
                dtype=float,
            )
        # Observer prediction input for NEXT tick: the LEG stance PD
        # command (props are a parallel loop, not in this observer).
        if bool(self._stance) or bool(props_enabled_ctrl):
            self._kw_obs_tau_prev[0] = float(tau_b_att_des[0])
            self._kw_obs_tau_prev[1] = float(tau_b_att_des[1])
        else:
            self._kw_obs_tau_prev[:] = 0.0
        # Log the rate the stance D-term actually consumed (observer when on).
        omega_b_used_dbg = omega_b.copy()
        if kw_obs_on:
            omega_b_used_dbg[0] = float(omega_obs_xy[0])
            omega_b_used_dbg[1] = float(omega_obs_xy[1])

        # ===== Flight swing torque reference (only after apex) =====
        tau_ref = None
        # Debug: force that is fed into the Jacobian->torque mapping.
        # - f_tau_b:     BODY frame (FRD, +Z down)
        # - f_tau_delta: delta/vicon frame (+Z down)
        # In flight: this comes from swing foot-space PD.
        # In stance: we derive it from the solved contact force (GRF) so it matches the stance A_tau_f mapping.
        f_tau_b = np.zeros(3, dtype=float)
        f_tau_delta = np.zeros(3, dtype=float)
        # Debug targets for logging/printing (always populate with finite shape)
        foot_des_b_dbg = np.full(3, np.nan, dtype=float)
        foot_des_w_dbg = np.full(3, np.nan, dtype=float)       # world-frame vector (base->foot)
        p_foot_des_w_dbg = np.full(3, np.nan, dtype=float)     # world-frame point (absolute)

        xdot_for_pd = foot_vdot_vicon.copy()

        if not bool(self._stance):
            # Raibert RETURN-MAP placement in WORLD FRD (+Z down), same
            # vertical sense as body/FK footpos.  Textbook two-term law:
            #   target_xy_w = (T_st/2) * v_xy + flight_kv * (v_xy - v_des)
            #   target_z_w  = +sqrt(l0^2 - ||target_xy_w||^2)
            #   foot_des_b  = R_wb^T @ target_w   (quaternion body<-world)
            # NEUTRAL POINT synced to the height channel: the stance is a
            # half period of the nrc_k_n_m virtual spring, so
            #   T_st/2 = (pi/2)*sqrt(m/k)
            # follows the stance-stiffness knob automatically.  flight_kv
            # is the pure feedback gain (see config).  v_des enters with
            # the CORRECT Raibert sign (foot behind neutral to speed up);
            # the old +flight_kr feedforward is retired.
            l0 = float(self.cfg.leg_l0_m)
            kv = float(max(0.0, float(self.cfg.flight_kv)))
            # Neutral-point timing coupled to the ACTIVE stance law: under
            # pogox the stance is a half period of the pogox design spring
            # k_v = (m*v*^2 + 2*m*g*X)/X^2 (hop_height + stroke knobs), so
            # the Raibert neutral point re-times itself with the SAME
            # spring the height law actually commands.  nrc_k_n_m is only
            # the fallback for the other laws (it just happened to equal
            # the pogox k_v at h*=0.05/X=0.09; change either knob and the
            # old constant silently mistimes the placement).
            if str(getattr(
                self.cfg, "stance_energy_law", ""
            )).lower() == "pogox":
                h_st_rb = float(max(0.0, float(self.cfg.hop_height_m)))
                X_rb = float(max(0.02, float(self.cfg.leg_stroke_max_m)))
                v_star_rb = math.sqrt(2.0 * float(self.gravity) * h_st_rb)
                k_st = float(max(1.0, (
                    float(self.mass) * v_star_rb * v_star_rb
                    + 2.0 * float(self.mass) * float(self.gravity) * X_rb
                ) / (X_rb * X_rb)))
            else:
                k_st = float(max(1.0, float(getattr(
                    self.cfg, "nrc_k_n_m", 1800.0
                ))))
            half_t_st = 0.5 * math.pi * math.sqrt(
                float(self.mass) / k_st
            )
            step_lim = float(abs(float(self.cfg.flight_stepper_lim_m)))
            v_xy_w = np.array([float(self._v_hat_w[0]), float(self._v_hat_w[1])], dtype=float)
            vdes_xy_w = np.array([float(desired_v_xy_w[0]), float(desired_v_xy_w[1])], dtype=float)
            target_xy_w = (
                (half_t_st + kv) * v_xy_w - kv * vdes_xy_w
            ).astype(float)
            if bool(self.cfg.mode_1d):
                target_xy_w[0] = 0.0
                target_xy_w[1] = 0.0
            normTarget = float(np.linalg.norm(target_xy_w))
            norm_for_z = normTarget
            if (step_lim > 1e-9) and (normTarget > step_lim):
                target_xy_w = (
                    target_xy_w * (step_lim / max(1e-12, normTarget))
                ).astype(float)
                norm_for_z = float(np.linalg.norm(target_xy_w))
            target_z_w = float(np.sqrt(max(
                0.0, float(l0 * l0) - float(norm_for_z * norm_for_z)
            )))
            foot_des_w = np.array([float(target_xy_w[0]), float(target_xy_w[1]), float(target_z_w)], dtype=float)
            foot_des_b = (np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T @ foot_des_w.reshape(3)).reshape(3)
            foot_des_native = np.asarray(foot_des_b, dtype=float).reshape(3).copy()
            # Debug: foot_des_w is world FRD (+Z down); foot_des_b is body FRD for PD/print.
            foot_des_b_dbg = np.asarray(foot_des_b, dtype=float).reshape(3).copy()
            foot_des_w_dbg = np.asarray(foot_des_w, dtype=float).reshape(3).copy()
            p_foot_des_w_dbg = (
                np.asarray(self._p_hat_w, dtype=float).reshape(3)
                + (np.asarray(R_wb_hat, dtype=float).reshape(3, 3) @ foot_des_b_dbg.reshape(3)).reshape(3)
            )
            # ===== Hopper4 flight leg force (sideForce + springForce), native delta frame (+Z down) =====
            x = np.asarray(foot_vicon, dtype=float).reshape(3)
            targetFootPos = np.asarray(foot_des_native, dtype=float).reshape(3)
            xdot = np.asarray(xdot_for_pd, dtype=float).reshape(3)

            leg_length = float(np.linalg.norm(x))
            if leg_length < 1e-6:
                unitSpring = np.array([0.0, 0.0, 1.0], dtype=float)
                leg_length = 0.0
            else:
                unitSpring = (x / leg_length).astype(float)

            springVel = (float(np.dot(xdot, unitSpring)) * unitSpring).astype(float)

            omega_b_native = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            Khp = float(self.cfg.swing_kp_xy)
            Khd = float(self.cfg.swing_kd_xy)
            # Ground-relative foot velocity: xdot is the JOINT-only foot
            # velocity (J*qdot); subtract the body-rotation part ω×x to remove
            # the foot velocity induced by body spin. 2026-07-11: reverted back
            # to '-' per user.
            xdot_damped = xdot + 0 * _cross3(omega_b_native, x)
            sideForce = (
                Khp * (targetFootPos - x) - Khd * xdot_damped
            ).astype(float)
            sideForce = (sideForce - float(np.dot(sideForce, unitSpring)) * unitSpring).astype(float)

            k = float(self.cfg.swing_kp_z)
            b = float(self.cfg.swing_kd_z)
            force_scalar = -float(k) * float(leg_length - float(l0))
            springForce = (force_scalar * unitSpring - float(b) * springVel).astype(float)

            footForce = (sideForce + springForce).astype(float)
            f_native_cmd = footForce.copy()
            f_b_cmd = f_native_cmd.copy()

            # HARDCODED DEBUG: only the XY (side) force is applied to leg torque in
            # flight; axial/spring Z is zeroed. Computation/logging above unchanged.
            if bool(self._dbg_flight_fxy_only_out):
                f_native_cmd = np.array([float(f_native_cmd[0]), float(f_native_cmd[1]), 0.0], dtype=float)
                f_b_cmd = np.array([float(f_b_cmd[0]), float(f_b_cmd[1]), 0.0], dtype=float)

            if self._leg_model == "serial":
                f_tau_b = f_b_cmd.copy()
                f_tau_delta = f_native_cmd.copy()
                try:
                    if J_body is None:
                        raise RuntimeError("serial Jacobian missing")
                    tau_ref = (np.asarray(J_body, dtype=float).reshape(3, 3).T @ f_b_cmd.reshape(3)).reshape(3)
                    tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                    tau_ref = (tau_sign.reshape(3) * tau_ref.reshape(3)).reshape(3)
                    # Direction-preserving limit (NOT per-axis clip): keeps foot-force direction.
                    tau_ref, _ = _tau_limit_proportional(tau_ref, tau_cmd_max)
                except Exception:
                    tau_ref = None
            else:
                f_delta_cmd = f_native_cmd.copy()
                f_tau_b = f_b_cmd.copy()
                f_tau_delta = f_delta_cmd.copy()
                try:
                    if J_inv is None:
                        raise RuntimeError("delta inverse Jacobian missing")
                    inv_Jt = self._stable_inv3(np.asarray(J_inv, dtype=float).reshape(3, 3).T)
                    tau_ref = (inv_Jt @ f_delta_cmd.reshape(3)).reshape(3)
                    # Motor torque sign convention (real robot wiring/driver)
                    tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                    tau_ref = (tau_sign.reshape(3) * tau_ref.reshape(3)).reshape(3)
                    # Direction-preserving limit (NOT per-axis clip): keeps foot-force direction.
                    tau_ref, _ = _tau_limit_proportional(tau_ref, tau_cmd_max)
                except Exception:
                    tau_ref = None

        # ===== Force allocation =====
        # Mode1: closed-form leg allocation plus propeller residual overlay.
        # Pure-leg operation is selected by leaving props disarmed.
        # Stance may use the full hardware collective budget; flight is
        # attitude-only and must NOT float the ballistic apex above the
        # hop_height knob (see flight_thrust_sum_max_ratio).
        if bool(self._stance):
            thrust_sum_max = float(
                self.mass * self.gravity
                * float(self.cfg.thrust_total_ratio_max)
            )
        else:
            thrust_sum_max = float(
                self.mass * self.gravity
                * float(getattr(
                    self.cfg, "flight_thrust_sum_max_ratio", 0.18
                ))
            )
        props_on = bool(props_enabled_ctrl)
        if not props_on:
            thrust_sum_max = 0.0
        # Default values for the stance attitude split debug (info dict). If
        # we are in flight, the leg naturally takes the full attitude demand.
        tau_leg_des_w = np.asarray(Tau_des, dtype=float).reshape(3).copy()
        tau_leg_des_b = np.asarray(tau_b_att_des, dtype=float).reshape(3).copy()
        if bool(self._stance):
            # --- Stance: closed-form leg (fz + fxy) + prop collaboration ---
            F_des = np.asarray(f_ref, dtype=float).reshape(3).copy()
            # Friction-cone modulation (2026-07-07, see stance_downforce_n docs):
            # props push DOWN by f_dn total; the leg pushes UP by f_dn extra. Net
            # body force = 0 (CoM trajectory unchanged), contact normal force
            # +f_dn -> friction cone |fxy| <= mu*(fz_slip + f_dn) widens, most
            # valuably at touchdown when fz_slip ~ 0. Capped by the physical
            # reverse budget; replaces the positive stance baseline thrust.
            f_dn = 0.0
            if props_on and bool(getattr(self.cfg, "prop_bidir", False)):
                f_dn = min(
                    max(0.0, float(getattr(self.cfg, "stance_downforce_n", 0.0))),
                    3.0 * abs(float(self.cfg.prop_reverse_max_n)),
                )
                # Touchdown-window gating (see stance_downforce_td_s): boost N only
                # while fz_slip is still ramping; expire before the lever-arm fxy
                # side effect (fxy ~ rx*fz/rz) integrates into horizontal drift.
                td_win = float(getattr(self.cfg, "stance_downforce_td_s", 0.0))
                if td_win > 0.0:
                    t_td = float(self._td_t) if self._td_t is not None else float(self.sim_time)
                    if (float(self.sim_time) - t_td) > td_win:
                        f_dn = 0.0
            if f_dn > 0.0:
                thrust_sum_ref = -f_dn
            # PROPELLER ENERGY SUPPLEMENT: the NRC/PUSH demand the capped
            # leg cannot deliver rides the prop COLLECTIVE.  Pure Fz
            # channel of the decoupled allocator -- attitude differential
            # is unaffected.  Under the NRC law the residual is a per-tick
            # continuous quantity valid for the WHOLE stance (compression
            # included); under Mode1 it is latched at PUSH, so gate it on
            # the latch there.
            _pe_law = str(getattr(
                self.cfg, "stance_energy_law", "mode1"
            )).lower()
            # fbslip / nrc: residual is valid for the WHOLE stance
            # (catch + push); Mode1 only latches it at PUSH.
            _pe_gate = (
                bool(self._mode1_push_latched)
                or _pe_law in ("nrc", "fbslip")
            )
            if (_pe_gate
                    and props_on
                    and f_dn <= 0.0
                    and float(self._prop_energy_fz) > 0.0):
                # World-Z de-projection (paper plant: m*hddot = -m*g +
                # F_leg + T_sum*cos(theta)).  _prop_energy_fz is a WORLD-Z
                # force; the collective acts along body -Z, so divide by
                # cos(theta) = -z_thrust_w[2] to deliver the residual on
                # the world vertical under body tilt.  Floor 0.5 (60 deg)
                # bounds the blow-up; upright it is exactly 1.
                cos_th = float(_clipf(-float(z_thrust_w[2]), 0.5, 1.0))
                thrust_sum_ref = float(thrust_sum_ref) + float(
                    self._prop_energy_fz
                ) / cos_th
            fz_cmd = float(max(0.0, float(f_ref[2]))) + f_dn

            # (tau_leg_des_w and tau_leg_des_b were initialized above so the
            # info dict can read them even in flight; recompute here in stance.)
            tau_leg_des_w = np.asarray(Tau_des, dtype=float).reshape(3).copy()
            tau_leg_des_b = np.asarray(tau_b_att_des, dtype=float).reshape(3).copy()
            # LEG SHARE cap (stance_leg_att_tau_max_nm): scale the torque
            # the leg allocation targets; the withheld share reaches the
            # props automatically because both Eq.12 residual passes below
            # subtract the leg's DELIVERED torque from the original
            # Tau_des.  Direction preserved; both SLIP (tau_leg_des_b) and
            # SRB (tau_leg_des_w) paths read these two vectors.
            _leg_att_cap = float(max(0.0, float(getattr(
                self.cfg, "stance_leg_att_tau_max_nm", 0.0
            ))))
            if _leg_att_cap > 0.0:
                _tau_leg_n = float(np.linalg.norm(tau_leg_des_w))
                if _tau_leg_n > _leg_att_cap and _tau_leg_n > 1e-9:
                    _s_leg = _leg_att_cap / _tau_leg_n
                    tau_leg_des_w = (tau_leg_des_w * _s_leg).astype(float)
                    tau_leg_des_b = (tau_leg_des_b * _s_leg).astype(float)
            try:
                r_foot_b = (foot_b - self.com_b).reshape(3)
                rx = float(r_foot_b[0])
                ry = float(r_foot_b[1])
                rz = float(r_foot_b[2])
                tau_att_xy = np.asarray(tau_b_att_des, dtype=float).reshape(3)[:2]
                # SLX has only the hip-torque norm limit; no extra friction or
                # Cartesian side-force clip in this block.
                mu_s = float(getattr(self.cfg, "stance_mu", 0.0))
                fxy_cap = float(getattr(self.cfg, "stance_fxy_max", 0.0))
                lim = float("inf")
                if mu_s > 0.0:
                    lim = min(lim, mu_s * max(0.0, fz_cmd))
                if fxy_cap > 0.0:
                    lim = min(lim, fxy_cap)

                if bool(getattr(self.cfg, "stance_leg_frame_alloc", False)):
                    # --- SLIP-style split (see stance_leg_frame_alloc docs) ---
                    # Side force: minimum-norm f with delivered torque
                    # tau = -r x f equal to tau_att_des (perp component of r):
                    #   f_side = (r x tau) / |r|^2   (automatically perp to r)
                    # Reduces exactly to the legacy solution when rx=ry=0.
                    # SLX uses the hip-origin foot vector x (no COM offset).
                    r_side = r_foot_b
                    r_n2 = float(np.dot(r_side, r_side))
                    u_b = (r_side / max(1e-6, math.sqrt(r_n2))).reshape(3)
                    # MATLAB: full 3-vector hip torque into the min-norm map
                    # (the leg-axis component was already projected out above;
                    # any residual component along r drops out of the cross).
                    tau_v = np.asarray(tau_leg_des_b, dtype=float).reshape(3).copy()
                    f_side_b = _minimum_norm_side_force(r_side, tau_v)
                    # The fxy limit acts on the PURE attitude channel only --
                    # the axial push carries no torque, so clipping f_side no
                    # longer corrupts the height/energy channel (and vice versa).
                    if np.isfinite(lim) and lim > 0.0:
                        f_side_b[0] = float(_clipf(float(f_side_b[0]), -lim, lim))
                        f_side_b[1] = float(_clipf(float(f_side_b[1]), -lim, lim))
                    # Axial spring: f_ref[2] is the leg-axis SLIP spring magnitude;
                    # apply directly along u_b (no body-z / world-z projection).
                    f_ax = float(max(0.0, float(fz_cmd)))
                    f_contact_b_cmd = (f_ax * u_b + f_side_b).astype(float).reshape(3)
                else:
                    # --- SRB (HFA): WORLD-vertical support + WORLD torque solve ---
                    # 2026-07-11 fix: the fz support is now held along the WORLD
                    # vertical, NOT body-z. The old legacy split placed fz_cmd in
                    # f_contact_b_cmd[2] (body z) and rotated the whole vector to
                    # world, so a body tilt theta leaked a WORLD-horizontal force
                    # fz*sin(theta) that was NOT part of the attitude solve --
                    # a destabilizing side push that fed the tip-over divergence
                    # seen while hopping. Solving in the WORLD frame with the
                    # WORLD lever arm r_foot_w and WORLD demand Tau_des keeps the
                    # weight-support push vertical at any attitude.
                    #
                    # Leg reaction torque:  tau_leg = -(r_w x f_w)
                    #   tau_leg_x = rz*fy - ry*fz
                    #   tau_leg_y = rx*fz - rz*fx
                    # Fix fz = fz_cmd (world vertical, +Z DOWN = push into ground)
                    # and back-solve the world horizontal for the demanded torque:
                    #   fy = (tau_x + ry*fz)/rz ,  fx = (rx*fz - tau_y)/rz
                    r_w = np.asarray(r_foot_w, dtype=float).reshape(3)
                    rxw, ryw, rzw = float(r_w[0]), float(r_w[1]), float(r_w[2])
                    fz_w = float(fz_cmd)
                    tau_des_xy = np.asarray(tau_leg_des_w, dtype=float).reshape(3)[:2]
                    if abs(rzw) > 1e-6:
                        fx_w = (rxw * fz_w - float(tau_des_xy[1])) / rzw
                        fy_w = (float(tau_des_xy[0]) + ryw * fz_w) / rzw
                    else:
                        fx_w, fy_w = 0.0, 0.0
                    # Paper HFA friction cone (Eq. 11): ||f_xy||_2 <= mu * fz,
                    # projected by proportional scaling (direction preserved).
                    if mu_s > 0.0:
                        cone = float(mu_s * max(0.0, fz_w))
                        fxy_norm = float(np.hypot(fx_w, fy_w))
                        if fxy_norm > cone and fxy_norm > 1e-9:
                            s_fxy = cone / fxy_norm
                            fx_w *= s_fxy
                            fy_w *= s_fxy
                    # Optional extra per-axis hard cut (stance_fxy_max; 0 = off).
                    if fxy_cap > 0.0:
                        fx_w = float(_clipf(fx_w, -fxy_cap, fxy_cap))
                        fy_w = float(_clipf(fy_w, -fxy_cap, fxy_cap))
                    f_contact_w = np.array([fx_w, fy_w, fz_w], dtype=float)
                    # Keep the body-frame command consistent (logs + the shared
                    # rotate-back below reproduce exactly this world force).
                    f_contact_b_cmd = (R_wb_hat.T @ f_contact_w.reshape(3)).reshape(3)
                # Downstream variables named *_w expect world frame; convert once here.
                f_contact_w = (R_wb_hat @ f_contact_b_cmd.reshape(3)).reshape(3)
            except Exception:
                f_contact_w = np.array([0.0, 0.0, fz_cmd], dtype=float)

            if A_tau_f_3rsr is not None:
                tau_qp = (np.asarray(A_tau_f_3rsr, dtype=float).reshape(3, 3) @ f_contact_w.reshape(3)).reshape(3)
            else:
                tau_qp = np.zeros(3, dtype=float)
            status = "closed_form"
            slack = np.zeros(6, dtype=float)

            if props_on:
                try:
                    # DECOUPLED stance props: collective = vertical energy
                    # plan; DIFFERENTIAL tracks Tau_prop_des (flight_kR/kW
                    # on the same R_des).  Leg still takes the full
                    # stance PD through contact -- additive, not residual.
                    thrusts = self._allocate_prop_thrust(
                        tau_des_w=Tau_prop_des,
                        prop_r_w=prop_r_w,
                        z_thrust_w=z_thrust_w,
                        thrust_sum_ref=float(thrust_sum_ref),
                        thrust_sum_max=float(thrust_sum_max),
                    )
                except Exception:
                    thrusts = np.zeros(3, dtype=float)
            else:
                thrusts = np.zeros(3, dtype=float)

        else:
            # --- Flight: swing-leg tau_ref + props track Tau_des directly ---
            F_des = (np.asarray(f_ref, dtype=float).reshape(3) + z_thrust_w.reshape(3) * float(thrust_sum_ref)).astype(float)
            f_contact_w = np.zeros(3, dtype=float)
            if tau_ref is not None:
                tau_qp = np.asarray(tau_ref, dtype=float).reshape(3).copy()
            else:
                tau_qp = np.zeros(3, dtype=float)
            status = "closed_form"
            slack = np.zeros(6, dtype=float)

            if props_on:
                try:
                    thrusts = self._allocate_prop_thrust(
                        tau_des_w=Tau_prop_des,
                        prop_r_w=prop_r_w,
                        z_thrust_w=z_thrust_w,
                        thrust_sum_ref=float(thrust_sum_ref),
                        thrust_sum_max=float(thrust_sum_max),
                        reverse_policy=str(getattr(self.cfg, "prop_flight_reverse", "auto")),
                    )
                except Exception:
                    thrusts = np.zeros(3, dtype=float)
            else:
                thrusts = np.zeros(3, dtype=float)

        # Extra wrench debug (helps real-robot diagnosis):
        thrust_sum = float(np.sum(thrusts)) if np.all(np.isfinite(thrusts)) else float("nan")
        F_total_w = (f_contact_w + z_thrust_w.reshape(3) * thrust_sum).astype(float).reshape(3)
        tau_contact_w = _cross3(r_foot_w.reshape(3), f_contact_w.reshape(3)).astype(float).reshape(3)
        tau_props_w = np.zeros(3, dtype=float)
        try:
            for i in range(3):
                tau_props_w = (tau_props_w + _cross3(prop_r_w[i].reshape(3), (z_thrust_w.reshape(3) * float(thrusts[i])).reshape(3))).astype(float)
        except Exception:
            tau_props_w[:] = np.nan
        tau_total_w = (tau_contact_w + tau_props_w).astype(float).reshape(3)

        ok_status = str(status) in ("solved", "solved inaccurate", "solved_inaccurate", "closed_form")
        ok = bool(ok_status) and np.all(np.isfinite(f_contact_w)) and np.all(np.isfinite(thrusts)) and np.all(np.isfinite(tau_qp))
        if ok:
            self._wbc_last_t = thrusts.copy()
            if bool(self._stance):
                self._wbc_last_f = f_contact_w.copy()
        else:
            thrusts = self._wbc_last_t.copy()
            if bool(self._stance):
                f_contact_w = self._wbc_last_f.copy()
            else:
                f_contact_w[:] = 0.0
            tau_qp = self._tau_cmd_prev.copy()
            slack[:] = 0.0
            status = f"fallback({status})"

        # HFA decoupling (MODEE_DBG_STANCE_ZERO_FZ): fz stays in the QP (so the friction cone
        # |fxy| <= mu*fz gives the attitude solver a budget), but here we drop the fz component
        # before mapping the contact force to leg joint torque, so the leg commands ONLY the
        # attitude (horizontal-force) torque and no vertical push.
        if bool(self._stance) and bool(self._dbg_stance_zero_fz) and (A_tau_f_3rsr is not None):
            try:
                f_xy_only = np.array([float(f_contact_w[0]), float(f_contact_w[1]), 0.0], dtype=float)
                tau_qp = (np.asarray(A_tau_f_3rsr, dtype=float).reshape(3, 3) @ f_xy_only.reshape(3)).reshape(3)
            except Exception:
                pass

        # HARDCODED DEBUG: stance leg outputs ONLY the attitude fxy force; the BODY-z
        # (spring/energy push) component is zeroed just before the force->torque
        # mapping. Upstream computation and logs (f_contact_w, f_ref_w) are unchanged.
        if bool(self._stance) and bool(self._dbg_stance_fxy_only_out) and (A_tau_f_3rsr is not None):
            try:
                f_b_gate = (R_wb_hat.T @ np.asarray(f_contact_w, dtype=float).reshape(3)).reshape(3)
                f_b_gate[2] = 0.0
                f_w_gate = (R_wb_hat @ f_b_gate.reshape(3)).reshape(3)
                tau_qp = (np.asarray(A_tau_f_3rsr, dtype=float).reshape(3, 3) @ f_w_gate.reshape(3)).reshape(3)
            except Exception:
                pass

        # HARDCODED DEBUG: stance leg force is NOT applied. The full pipeline above
        # (spring/energy fz, attitude fxy, logs f_contact_w/f_ref_w) runs unchanged;
        # only the commanded leg torque is zeroed at this final output stage.
        if bool(self._stance) and bool(self._dbg_stance_force_zero_out):
            tau_qp = np.zeros(3, dtype=float)

        # final motor torques: scale proportionally to keep direction if any exceeds limit
        tau_qp = np.asarray(tau_qp, dtype=float).reshape(3)
        tau_cmd_max = np.asarray(tau_cmd_max, dtype=float).reshape(3)
        tau_cmd, scale = _tau_limit_proportional(tau_qp, tau_cmd_max)
        self._tau_cmd_prev = tau_cmd.copy()

        # Torque limit also limits the reported contact force: when the joint-torque cap scales
        # the commanded torque down, the effective foot force is reduced by the same factor, so
        # footforce_b reflects what the leg can actually deliver (stance, closed-form path).
        # FZ ASSIST (2026-08-02, log 220050: leg fz pinned at the 9 Nm cap,
        # f_ref 147 N vs deliverable 135 N while the prop collective idled
        # at 7.9 N): the force the cap takes away from the leg, projected
        # on the prop thrust axis, is added to the collective below.
        # (Fz channel only -- decoupled contract, no attitude coupling.)
        prop_fz_assist_n = 0.0
        if bool(self._stance) and float(scale) < 1.0:
            _f_pre_cap_w = np.asarray(f_contact_w, dtype=float).reshape(3).copy()
            f_contact_w = (_f_pre_cap_w * float(scale)).astype(float)
            try:
                prop_fz_assist_n = float(max(0.0, float(np.dot(
                    (_f_pre_cap_w - f_contact_w).reshape(3),
                    np.asarray(z_thrust_w, dtype=float).reshape(3),
                ))))
            except Exception:
                prop_fz_assist_n = 0.0

        if bool(self._stance) and props_on and prop_fz_assist_n > 0.0:
            # FZ ASSIST re-allocation: joint-torque cap scaled the leg
            # force; the lost lift rides the prop COLLECTIVE.  Keep the
            # stance attitude differential (Tau_prop_des) so the Fz bump
            # does not wipe the parallel attitude loop.
            try:
                thrusts = self._allocate_prop_thrust(
                    tau_des_w=Tau_prop_des,
                    prop_r_w=prop_r_w,
                    z_thrust_w=z_thrust_w,
                    thrust_sum_ref=float(min(
                        float(thrust_sum_ref) + float(prop_fz_assist_n),
                        float(thrust_sum_max),
                    )),
                    thrust_sum_max=float(thrust_sum_max),
                )
                self._wbc_last_t = thrusts.copy()
            except Exception:
                pass

            # Refresh wrench telemetry after the final prop allocation.
            thrust_sum = float(np.sum(thrusts)) if np.all(np.isfinite(thrusts)) else float("nan")
            F_total_w = (
                f_contact_w + z_thrust_w.reshape(3) * thrust_sum
            ).astype(float).reshape(3)
            tau_contact_w = _cross3(
                r_foot_w.reshape(3), f_contact_w.reshape(3)
            ).astype(float).reshape(3)
            try:
                tau_props_w = np.sum(
                    [
                        _cross3(
                            prop_r_w[i].reshape(3),
                            z_thrust_w.reshape(3) * float(thrusts[i]),
                        )
                        for i in range(3)
                    ],
                    axis=0,
                ).astype(float).reshape(3)
            except Exception:
                tau_props_w = np.full(3, np.nan, dtype=float)
            tau_total_w = (tau_contact_w + tau_props_w).astype(float).reshape(3)

        thrusts = np.asarray(thrusts, dtype=float).reshape(3).copy()
        if not props_on:
            thrusts[:] = 0.0
        self._wbc_last_t = np.asarray(thrusts, dtype=float).reshape(3).copy()

        # Telemetry describes the thrust sent to ESCs.
        thrust_sum = float(np.sum(thrusts)) if np.all(np.isfinite(thrusts)) else float("nan")
        F_total_w = (
            f_contact_w + z_thrust_w.reshape(3) * thrust_sum
        ).astype(float).reshape(3)
        tau_contact_w = _cross3(
            r_foot_w.reshape(3), f_contact_w.reshape(3)
        ).astype(float).reshape(3)
        tau_props_w = np.zeros(3, dtype=float)
        try:
            for i in range(3):
                tau_props_w = (
                    tau_props_w
                    + _cross3(
                        prop_r_w[i].reshape(3),
                        z_thrust_w.reshape(3) * float(thrusts[i]),
                    )
                ).astype(float)
        except Exception:
            tau_props_w[:] = np.nan
        tau_total_w = (tau_contact_w + tau_props_w).astype(float).reshape(3)

        # Stance GRF in BODY FRD (same frame as foot_b): single SO(3) R_wb^T, after torque scaling.
        if bool(self._stance):
            f_tau_b = (R_wb_hat.T @ f_contact_w.reshape(3)).reshape(3)
            f_tau_delta = np.asarray(f_tau_b, dtype=float).reshape(3).copy()

        # thrust (3 arms) -> 6 PWM (map via prop_pwm_idx_per_arm)
        pwm_us = self._pwm_from_arm_thrusts(thrusts)

        info = {
            "t": float(self.sim_time),
            "stance": int(bool(self._stance)),
            "touchdown": int(touchdown_evt),
            "liftoff": int(liftoff_evt),
            "apex": int(apex_evt),
            "compress": int(bool(compress_active)),
            "push": int(bool(self._stance) and (not bool(compress_active))),
            "desired_v_xy_w": np.asarray(desired_v_xy_w, dtype=float).reshape(2).copy(),
            "q_hat_wxyz": q_hat.copy(),
            "rpy_hat": rpy_hat.copy(),
            "p_hat_w": np.asarray(self._p_hat_w, dtype=float).reshape(3).copy(),
            "v_hat_w": np.asarray(self._v_hat_w, dtype=float).reshape(3).copy(),
            # Debug: base velocity measured from leg kinematics (foot assumed stationary in WORLD).
            "v_meas_foot_w": np.asarray(v_base_from_foot_w, dtype=float).reshape(3).copy(),
            "flight_vel_w": np.asarray(self._flight_vel, dtype=float).reshape(3).copy(),
            # LiDAR odometry fusion debug (hopper_odom_lcmt):
            "lidar_fresh": int(bool(lidar_fresh)),
            "lidar_pos_map": np.asarray(self._lidar_pos_map, dtype=float).reshape(3).copy(),
            "lidar_yaw_off": float(self._lidar_yaw_off),
            "lidar_fused_n": int(self._lidar_fused_n),
            # Foot kinematics:
            # - foot_vicon: delta/vicon frame (+Z DOWN)
            # - foot_b:     body frame (FRD, +Z DOWN)
            "foot_vicon": foot_vicon.copy(),
            "foot_b": foot_b.copy(),
            "foot_vdot_vicon": foot_vdot_vicon.copy(),
            "foot_vrel_b": foot_vrel_b.copy(),
            # Joint velocity used by kinematics/estimator: CAN qd with EMA.
            "qd_kin": np.asarray(joint_vel_kin, dtype=float).reshape(3).copy(),
            "J_inv_det": float(J_inv_det),
            "J_inv_cond": float(J_inv_cond),
            "A_tau_f_det": float(A_tau_f_det),
            "A_tau_f_cond": float(A_tau_f_cond),
            # Flight swing target (NaN during stance).
            "foot_des_b": foot_des_b_dbg.copy(),
            "foot_des_w": foot_des_w_dbg.copy(),
            "p_foot_des_w": p_foot_des_w_dbg.copy(),
            "q_shift": float(q_shift),
            "q_shift_equiv": float(q_shift),
            "qd_shift": float(qd_shift),
            "qd_shift_equiv": float(qd_shift),
            "az_des": float(az_des),
            "comp_m": float(depth_now),
            "comp_tgt_m": float(depth_tgt),
            "comp_tgt_act_m": float(depth_tgt_act),
            "z_now_m": float(z_now) if bool(self._stance) else 0.0,
            "s_stance": float(s) if bool(self._stance) else 0.0,
            "compress_active": int(bool(compress_active)),
            "push_started": int(bool(self._stance) and (not bool(compress_active))),
            "energy_comp_fz": float(energy_comp_fz),
            # Prop PUSH energy supplement actually riding the collective (N).
            "prop_energy_fz": float(self._prop_energy_fz),
            # NRC norm state: r converging to r_star = energy on target.
            "nrc_r": float(self._nrc_r),
            "nrc_r_star": float(self._nrc_r_star),
            "nrc_f_des": float(self._nrc_f_des),
            # Per-hop apex return-map trim: h_tgt_eff = hop_height * trim.
            "nrc_h_trim": float(self._nrc_h_trim),
            # Mode1 apex-adapt energy state (persistent across hops);
            # should be small and bounded when NRC+trim is converging.
            "mode1_Eloss_j": float(self._mode1_Eloss),
            # Decoupled allocator: fraction of demanded attitude torque
            # delivered (1.0 = unclipped; < 1 = raise the collective baseline).
            "prop_att_scale": float(self._prop_att_scale),
            # PogoX flight velocity convergence: commanded lean + direction.
            "fl_tilt_cmd_deg": float(self._fl_tilt_cmd_deg),
            "fl_zb_des_x": float(self._fl_zb_des_xy[0]),
            "fl_zb_des_y": float(self._fl_zb_des_xy[1]),
            # Horizontal convergence: flight |v - v_des| and the
            # DELIVERED prop lateral force f_z*tan(tilt).
            "fl_ev_xy": float(self._fl_ev_xy),
            "fl_lat_force_n": float(self._fl_lat_force_n),
            "energy_gate": int(bool(self._stance) and bool(energy_gate)),
            "vz_up": float(vz_up),
            # Stance attitude debug (HFA split): desired, leg share, and
            # residual (goes to props).  *_xy norm makes single-column trends
            # easy to read; full 3-vectors kept for component plots.
            "tau_b_att_des": np.asarray(tau_b_att_des, dtype=float).reshape(3).copy(),
            "tau_b_leg_des": np.asarray(tau_leg_des_b, dtype=float).reshape(3).copy(),
            "tau_b_leg_act": np.asarray(
                R_wb_hat.T @ tau_contact_w.reshape(3), dtype=float
            ).reshape(3).copy(),
            "tau_b_res_des": np.asarray(
                tau_b_att_des - tau_leg_des_b, dtype=float
            ).reshape(3).copy(),
            "tau_b_att_des_xy_norm": float(np.linalg.norm(tau_b_att_des[:2])),
            "tau_b_leg_des_xy_norm": float(np.linalg.norm(tau_leg_des_b[:2])),
            "tau_props_xy_norm": float(np.linalg.norm(tau_props_w[:2])),
            # Leg attitude share utilization: 1.0 means the leg got the full
            # request (cap not binding); <1 means the cap was active and
            # props should be carrying the rest.
            "leg_att_share": float(
                (float(np.linalg.norm(tau_leg_des_b[:2]))
                 / max(1e-9, float(np.linalg.norm(tau_b_att_des[:2]))))
                if np.linalg.norm(tau_b_att_des[:2]) > 1e-9 else 0.0
            ),
            # Joint-torque cap: proportional scale actually applied (<1 =
            # saturated) and the lift deficit handed to the prop collective.
            "tau_cap_scale": float(scale),
            "prop_fz_assist_n": float(prop_fz_assist_n),
            # Wrench-level debug:
            "F_des_w": np.asarray(F_des, dtype=float).reshape(3).copy(),
            "f_ref_w": np.asarray(f_ref, dtype=float).reshape(3).copy(),
            "f_h4_stance_base_w": np.zeros(3, dtype=float),
            "stance_additive_mode": 0,
            "thrust_sum_ref": float(thrust_sum_ref),
            "thrust_sum": float(thrust_sum),
            "F_total_w": np.asarray(F_total_w, dtype=float).reshape(3).copy(),
            "tau_contact_w": np.asarray(tau_contact_w, dtype=float).reshape(3).copy(),
            "tau_props_w": np.asarray(tau_props_w, dtype=float).reshape(3).copy(),
            "tau_total_w": np.asarray(tau_total_w, dtype=float).reshape(3).copy(),
            "f_contact_w": f_contact_w.copy(),
            # Contact force (GRF) expressed in BODY frame, same frame as foot_b.
            "f_contact_b": (np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T @ np.asarray(f_contact_w, dtype=float).reshape(3)).reshape(3).copy(),
            # Unified footforce, SAME coordinate system as footpos (foot_b) for ALL phases:
            #   body FRD (+X fwd, +Y right, +Z down). Only SO(3) is used (no per-axis flips):
            #     stance -> GRF (world)        --R_wb^T-->            body
            #     flight -> swing leg force (leg-native) --_leg_native_to_imu_body--> body
            #   The single leg<->IMU SO(3) lives ONLY in _leg_native_to_imu_body (identity now).
            "footforce_b": (
                np.asarray(f_tau_b, dtype=float).reshape(3).copy()
                if bool(self._stance)
                else _leg_native_to_imu_body(f_tau_b).reshape(3).copy()
            ),
            # Debug: force that is fed into the Jacobian->torque mapping.
            "f_tau_b": f_tau_b.copy(),
            "f_tau_delta": f_tau_delta.copy(),
            "thrusts_arm": thrusts.copy(),
            "tau_cmd": tau_cmd.copy(),
            "pwm_us": pwm_us.copy(),
            "slack": slack.copy(),
            "status": status,
            # Debug: attitude torque demand that the QP tries to realize (WORLD frame, yaw-free)
            "tau_des_w": Tau_des_dbg.copy(),
            # SO(3) attitude error and body-frame stance torque demand.
            "e_R": e_R.copy(),
            "tau_b_stance_des": tau_b_stance_des.copy(),
            # Debug: gyro actually used by the stance attitude torque controller (BODY frame)
            "omega_b_used": omega_b_used_dbg.copy(),
            "z_lo_m": float(self._z_lo) if self._z_lo is not None else float("nan"),
            "vz_lo_m_s": float(self._vz_lo) if self._vz_lo is not None else float("nan"),
            "v_to_cmd_m_s": float(self._v_to_cmd),
            "hop_height_m": float(self.cfg.hop_height_m),
            # Flight-time apex measurement h = g*T^2/8 (log-only telemetry).
            "z_apex_actual_m": float(self._z_apex_actual),
            # MPC debug
            "mpc_status": mpc_status,
            "mpc_u0": mpc_u0.copy(),
        }

        return tau_cmd, pwm_us, info


