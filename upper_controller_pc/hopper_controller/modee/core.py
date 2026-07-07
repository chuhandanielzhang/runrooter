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


def _clipf(x, lo: float, hi: float) -> float:
    """Fast scalar clip (np.clip on a python float costs ~35us on the Jetson)."""
    x = float(x)
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return x


def _robust_fit_eval(ts: np.ndarray, ys: np.ndarray, t_eval: float) -> float:
    """
    Robust linear fit y(t) = c0 + c1*t over one stance, evaluated at t_eval.

    Used to latch the takeoff velocity: v_meas is noisy (dq/dt + body shaking spikes
    of several m/s) and the horizontal velocity keeps CHANGING during stance, so a
    plain mean reports the mid-stance velocity (~3x low at liftoff, log-verified).
    A line fit with MAD outlier rejection captures the trend and extrapolates to
    the liftoff instant while ignoring spikes.
    """
    ts = np.asarray(ts, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    n = int(ts.size)
    if n == 0:
        return float("nan")
    if n < 8:
        return float(np.median(ys))
    mask = np.isfinite(ys) & np.isfinite(ts)
    if int(mask.sum()) < 8:
        return float(np.median(ys[np.isfinite(ys)])) if np.any(np.isfinite(ys)) else float("nan")
    c0, c1 = 0.0, 0.0
    for _ in range(3):
        tm = ts[mask]
        ym = ys[mask]
        A = np.column_stack([np.ones(tm.size), tm])
        try:
            sol, *_ = np.linalg.lstsq(A, ym, rcond=None)
        except np.linalg.LinAlgError:
            return float(np.median(ym))
        c0, c1 = float(sol[0]), float(sol[1])
        resid = ys - (c0 + c1 * ts)
        med = float(np.median(resid[mask]))
        mad = float(np.median(np.abs(resid[mask] - med)))
        if mad < 1e-9:
            break
        new_mask = np.abs(resid - med) < (3.0 * 1.4826 * mad)
        new_mask &= np.isfinite(ys)
        if int(new_mask.sum()) < 8 or bool(np.array_equal(new_mask, mask)):
            break
        mask = new_mask
    return float(c0 + c1 * float(t_eval))


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
    Minimal 'real robot style' attitude estimator:
      - propagate by gyro integration
      - correct tilt using accelerometer (no mag)
    """

    def __init__(self, kp_acc: float = 0.6, acc_g_min: float = 0.90, acc_g_max: float = 1.10, acc_lpf_tau: float = 0.25):
        self.kp_acc = float(kp_acc)
        self.acc_g_min = float(acc_g_min)
        self.acc_g_max = float(acc_g_max)
        self.acc_lpf_tau = float(acc_lpf_tau)
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # wxyz, body->world
        self._acc_f = np.zeros(3, dtype=float)
        self._inited = False

    def reset(self) -> None:
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._acc_f = np.zeros(3, dtype=float)
        self._inited = False

    def update(self, *, omega_b: np.ndarray, acc_b: np.ndarray, dt: float) -> np.ndarray:
        dt = float(dt)
        omega_b = np.asarray(omega_b, dtype=float).reshape(3)
        acc_b = np.asarray(acc_b, dtype=float).reshape(3)

        # LPF accel (for tilt correction gate)
        if not bool(self._inited):
            self._acc_f = acc_b.copy()
            self._inited = True
        else:
            tau = float(max(1e-6, self.acc_lpf_tau))
            a = float(_clipf(dt / (tau + dt), 0.0, 1.0))
            self._acc_f = (1.0 - a) * self._acc_f + a * acc_b

        # gyro integration
        dq = _quat_from_omega_dt(omega_b, dt)
        self._q = _quat_mul(self._q, dq)
        self._q = _quat_normalize_wxyz(self._q)

        # accel correction (tilt only)
        a_norm = float(np.linalg.norm(self._acc_f))
        if a_norm > 1e-9:
            g = 9.81
            g_ratio = float(a_norm / g)
            if float(self.acc_g_min) <= g_ratio <= float(self.acc_g_max):
                # Measured UP direction in body: normalized specific force
                # (at rest in FRD: [0,0,-1] = -Z_frd = up).
                a_b = (self._acc_f / a_norm).astype(float)
                R_wb = _quat_to_R_wb(self._q)
                up_b = (R_wb.T @ _up_dir_w(g)).reshape(3)
                e = _cross3(up_b, a_b)
                # small-angle correction in body frame
                omega_corr = float(self.kp_acc) * e
                dq2 = _quat_from_omega_dt(omega_corr, dt)
                self._q = _quat_mul(self._q, dq2)
                self._q = _quat_normalize_wxyz(self._q)

        return self._q.copy()


@dataclass
class ModeEConfig:
    # control rate
    dt: float = 0.002  # 500 Hz

    # ===== 1D Mode (vertical hopping only) =====
    # When enabled:
    #   - desired_v_xy is forced to 0
    #   - foot target in flight is forced to directly below (no Raibert placement)
    #   - robot only hops vertically in place
    # Use this to verify height convergence before adding horizontal movement.
    # Default to 3D tuning now; keep 1D tuned parameters below as locked baseline.
    mode_1d: bool = False
    # In the 1D tuning stage, disable stance MPC by default and use SRB + energy compensation.
    mode_1d_disable_mpc: bool = True

    # ===== Energy-based stance force compensation (pure dynamics) =====
    # When enabled, adds energy compensation to stance force during PUSH (leg extending):
    #   E_current = 0.5*m*vz² + m*g*z  (kinetic + gravitational, NO virtual spring)
    #   E_target  = m*g*(z + hop_height_m)
    #   f_compensation = energy_comp_kp * (E_target - E_current)
    # This helps achieve target hop height even with model errors.
    use_energy_compensation: bool = True
    # Hopper4-style energy loop (scaled conservative for real-robot bring-up).
    energy_comp_kp: float = 5
    # TRUE desired apex height above liftoff (m). 2026-07-05: this is now a real
    # closed-loop target -- see apex_fb_* below. Set what you actually want.
    hop_height_m: float = 0.1
    # ===== Per-hop apex feedback (closes the height loop) =====
    # The energy pump above is a P-only law: fz ~ Kp*(E_tgt - E_now). Its
    # steady-state height sits FAR below hop_height_m (0.39 commanded ->
    # ~0.09 achieved; 0.09 commanded -> no liftoff at all, log 18:05).
    # Fix: integrate the measured apex error each hop and feed the energy
    # target with h_eff = hop_height_m + apex_fb_int, so the ACTUAL apex
    # converges to hop_height_m within a few hops.
    #   apex_fb_ki:      integrator gain (m of h_eff per m of apex error/hop)
    #   apex_fb_int_init: initial boost seed. 0.25 puts the first hop at
    #                     h_eff~0.35, near the operating point that gave
    #                     ~9 cm on hardware, so hop #1 already lifts off.
    #   apex_fb_int_max: clamp on the accumulated boost.
    apex_fb_ki: float = 1.0
    apex_fb_int_init: float = 0.25
    apex_fb_int_max: float = 0.60
    # Measure the actual apex from the FLIGHT TIME between liftoff and the next
    # touchdown: h = g_eff * T^2 / 8. Needs only the phase-machine timestamps, so
    # it is immune to the vz_lo estimation error that corrupted the p_hat-based
    # apex (log 15:00: identical push forces but vz_lo -1.05/-0.86/-0.14 -> the
    # integrator wound up on phantom shortfall). The integrator then updates at
    # touchdown instead of at the in-flight apex event.
    apex_meas_flight_time: bool = True
    # LPF (seconds) on the leg-extension velocity used INSIDE the energy term only.
    # The 0.5*m*v^2 term has dF/dv = Kp*m*v ~= 28-53 N/(m/s) at push speeds, so the
    # dq/dt velocity noise (HF std ~0.5 m/s in stance) became +-15 N of fz command
    # noise and closed a ~30 Hz self-excited loop (2026-07-05 log). Energy pumping
    # is a low-bandwidth objective; a 20 ms filter kills the noise gain without
    # affecting the stance stroke. Set <=0 to use the raw kinematic velocity.
    energy_vel_lpf_tau: float = 0.02
    # physical params
    mass_kg: float = 3.75
    gravity: float = 9.81
    # COM offset in base frame (m). If unknown, keep zeros; tune later.
    # Computed from MuJoCo MJCF (`Hopper-modee-clean/mjcf/hopper_serial.xml`) at default pose.
    com_b: tuple[float, float, float] = (-2.79376456e-04, 1.68299070e-06, -5.72937376e-02)
    # Body inertia diagonal in BODY frame (kg*m^2). Reserved for future model-based planning.
    # Computed from MuJoCo MJCF (`Hopper-modee-clean/mjcf/hopper_serial.xml`) as whole-body inertia about COM,
    # expressed in base/body frame (diagonal approximation; off-diagonals are small).
    I_body_diag: tuple[float, float, float] = (0.0716072799, 0.0716088488, 0.0579831725)

    # delta leg nominal "length" (vicon/delta z coordinate, meters)
    # Nominal "extended" leg length (m).
    # - delta: this is ||foot_vicon|| when the delta leg is at its nominal length.
    # - serial (MuJoCo roll/pitch/shift model): we will auto-override this to match the model geometry.
    leg_l0_m: float = 0.45

    # hop target (relative height above liftoff/contact estimate)
    hop_z0: float = 0.9
    stance_T: float = 0.20   # Longer stance for deeper compression/push
    stance_min_T: float = 0.08
    flight_min_T: float = 0.10

    # touchdown/liftoff detection on equivalent shift coordinate:
    #   q_shift = leg_length - leg_l0_m
    #   negative: compressed (stance phase, allow up to -0.02 m compression)
    #   positive: extended (flight phase)
    td_q_shift_gate: float = -0.01  # touchdown when leg compressed by ≥1cm
    td_qd_shift_gate: float = -0.01  # touchdown when qd_shift < -0.01 (compressing)
    td_dq_shift_gate: float = -5e-5  # touchdown when dq < -5e-5 (compression increasing)
    # Hopper4-style phase thresholds:
    # - Flight -> Stance when leg compresses below (l0 - threshold)
    # - Stance -> Flight when leg extends above (l0 + threshold)
    hopper4_td_threshold_m: float = 0.020
    hopper4_lo_threshold_m: float = 0.0
    hopper4_phase_min_steps: int = 50  # ~100ms at 500Hz

    # Liftoff: switch to flight when q_shift/qd_shift gates are met.
    # Enforce a minimum stance duration so the first hop has enough time
    # to reduce attitude rate before leaving contact.
    stance_lo_min_T: float = 0.08  # shorter 1D stance to avoid over-extension before LO
    # Liftoff hysteresis (use filtered shift/velocity for robustness).
    # Prevents premature FLIGHT switch caused by raw q_shift jitter around zero.
    lo_q_shift_gate: float = 0.003
    lo_qd_shift_gate: float = 0.05
    # No-prop 3D takeoff quality gate:
    # Require low roll/pitch angular rate before liftoff so the first hop does not
    # leave stance with large residual body spin.
    lo_use_omega_gate_no_prop: bool = False
    lo_omega_gate_dps: float = 15.0
    lo_no_prop_max_stance_T: float = 0.40
    # Safety: if stance leg extends beyond this filtered shift (m), force liftoff
    # regardless of omega gate to avoid over-extension while still in PUSH.
    lo_force_liftoff_q_shift_m: float = 0.030
    # ===== Signal conditioning (real-robot robustness) =====
    # 3-RSR delta Jacobian can amplify encoder noise near workspace edges, which shows up as:
    #   - jitter in foot velocity estimate (foot_vdot)
    #   - jitter in q_shift / qd_shift (phase machine chattering)
    #   - jitter in force->torque mapping (inv(J_inv^T))
    #
    # These simple filters make the controller much more tolerant to noisy qd/Jacobian.
    #
    # Low-pass on joint velocity used in kinematics/Jacobian (seconds). Set <=0 to disable.
    # 2026-07-06: replaced by the CASE-style qd chain below (EMA + moving average);
    # keep at 0 to avoid triple-filtering.
    joint_vel_lpf_tau: float = 0.0
    # CASE/OMEGA-style joint velocity conditioning (their Simulink chain is
    # q -> discrete derivative K(z-1)/(Ts*z) -> EMA(forgetting 0.4) -> moving avg).
    # Our derivative step is qd_kin_from_q_diff below; these two complete the chain:
    #   qd_ema = ff*prev + (1-ff)*new   (ff = qd_ema_forgetting, 0 disables)
    #   then a qd_ma_window-sample moving average (<=1 disables).
    qd_ema_forgetting: float = 0.4
    qd_ma_window: int = 5
    # Kinematics qd source. 2026-07-07 user decision (REVERSED from 07-06):
    # use the CAN-reported qd carried in hopper_data_lcmt.qd (the Jetson driver
    # now publishes the AK60 internal velocity estimate); the upper layer does
    # NOT differentiate q itself. Set True to go back to PC-side dq/dt.
    qd_kin_from_q_diff: bool = False
    # Low-pass on q_shift / qd_shift used for touchdown/liftoff detection (seconds). Set <=0 to disable.
    q_shift_lpf_tau: float = 0.005
    qd_shift_lpf_tau: float = 0.005
    # ===== Liftoff XY latch (Raibert flight_vel) =====
    # Instantaneous leg kinematics AT the liftoff sample (last point only).
    use_mid_stance_vel: bool = False
    vel_clean_td_skip_s: float = 0.03      # skip this long after touchdown (impact transient)
    vel_clean_push_qd_thresh: float = 0.10 # stop accumulating once leg extends faster than this (push-off, m/s)
    vel_clean_min_samples: int = 5         # need at least this many clean samples, else fall back to window avg
    # 2026-07-05: the clean-window mean alone reports the MID-STANCE velocity;
    # any horizontal delta-v gained during push-off (where leg kinematics is
    # unusable: extension projects into XY through body tilt) was lost -- the
    # latch under-read a moving takeoff by ~3x, yet using leg kinematics there
    # latched phantom velocity when hopping in place. Fix: anchor on the clean
    # window (leg kinematics, drift-free) and add the IMU specific-force XY
    # integral from anchor to liftoff. <=0.2 s of integration re-anchored every
    # stance: no drift accumulation; zero for in-place hops (accel measures the
    # REAL motion), so no phantom either.
    vel_latch_push_dv: bool = True
    # Hard cap (m/s) on the |XY| velocity latched at liftoff for the Raibert
    # flight target (proportional scaling, direction preserved). Safety net
    # against phantom-velocity latches (push-off leg-extension projection);
    # 0.5 m/s caps the Raibert foot offset at kv*0.5 ~ 9 cm. Set <=0 to disable.
    vel_latch_clamp_mps: float = 0.5
    # ---- Liftoff XY latch: PUSH-phase LINEAR FIT extrapolated to t_LO ----
    # Least-squares line v(t) over the push-phase samples, evaluated at the
    # liftoff instant: full-window smoothing WITHOUT the mean's lag (XY velocity
    # ramps up during push, so a plain average systematically under-estimates
    # the takeoff velocity). One 3-sigma outlier-rejection pass. Fallback chain:
    # fit -> push-phase mean -> instantaneous kinematics.
    vel_latch_fit: bool = True
    vel_latch_fit_min_n: int = 8
    # ---- Velocity Kalman filter (Cheetah-style: velocity + accel bias) ----
    # State x = [v_w (3, world); b_a (3, accel bias, body frame)].
    # Predict every tick with the IMU: v_w += (R_wb @ (f_b - b_a) + g_w) * dt.
    # Update with the leg-kinematics base velocity on valid stance ticks.
    # The stance updates make the accel bias observable, so IMU integration is
    # safe in flight (this replaces the flight XY-latch-hold + ballistic vz).
    # Set False to fall back to the legacy leg-only estimator.
    use_vel_kf: bool = True
    vel_kf_sigma_acc: float = 0.6       # accel white noise -> velocity Q [m/s^2]
    vel_kf_sigma_bias: float = 0.03     # accel bias random walk [m/s^2/sqrt(s)]
    vel_kf_meas_std: float = 0.12       # leg-kinematics velocity noise [m/s]
    vel_kf_meas_std_push: float = 0.30  # inflated during push (fast leg, worst SNR)
    vel_kf_bias_max: float = 1.5        # |b_a| clamp per axis [m/s^2]
    # ---- Leg-kinematics measurement TRUST GATE ("foot planted" assumption) ----
    # The v_base_from_foot_w formula assumes the foot is pinned to the ground.
    # That fails (a) in the touchdown impact transient and (b) near liftoff:
    # once the leg extends fast the normal load -> 0 and the foot SKATES
    # sideways (2026-07-06 logs: 1-2 m/s phantom XY at LO on an IN-PLACE hop;
    # gated samples averaged (0.05, 0.03) m/s -- correct). Trust the sample
    # only when BOTH:  t_since_td >= vel_meas_td_skip_s
    #           AND    qd_shift   <  vel_meas_qd_max_mps (extension speed).
    # Compression (qd_shift < 0, foot pressed hard) stays trusted. Gated ticks:
    # KF runs predict-only and nothing enters the liftoff-latch buffers.
    vel_meas_td_skip_s: float = 0.03
    vel_meas_qd_max_mps: float = 0.25
    # Debounce: require N consecutive samples to declare touchdown/liftoff.
    # IMPORTANT (real robot): q_shift_raw can jitter around 0 near liftoff, and qd_shift can be noisy.
    # If liftoff triggers on a single noisy sample, the stance can end *way too early* (no PUSH impulse),
    # which destroys hop height and makes swing timing inconsistent.
    td_debounce_steps: int = 1
    lo_debounce_steps: int = 1

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
    use_unified_stance: bool = True
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

    # ===== Flight foot placement =====
    # Raibert (Kv/Kr) in WORLD (+Z down), then foot_des_b = R_wb^T @ target_w (quaternion).
    flight_kv: float = 0.15
    flight_kr: float = 0.0
    # If you see "摆腿太小" (world/heading XY step is small), increase this cap first.
    flight_stepper_lim_m: float = 0.2
    # swing (flight) foot-space torque reference (passed via QP tau_ref)
    # Hopper4-style decomposition:
    #   - Axial (along leg direction): kp_z/kd_z act on leg length + axial velocity
    #   - Perpendicular (⊥ leg direction): kp_xy/kd_xy act on foot Cartesian error, projected to ⊥ plane
    # NOTE: these are the exact knobs you want to tune to prevent flight over-extension.
    # Hopper4-like baseline for this controller:
    #   Khp ~= 50, Khd ~= 1, with axial spring/damping kept conservative for hardware bring-up.
    swing_kp_xy: float = 50.0
    swing_kd_xy: float = 1.3
    # Axial (virtual spring) stiffness and damping for flight leg control.
    # BALANCE: High kd prevents over-extension but amplifies velocity noise → jitter.
    # Use moderate kd (8-12) with strong LPF filtering instead of high kd.
    # Over-extension is also limited by the axial_coeff clamp logic (line ~2169).
    swing_kp_z: float = 1000.0
    stance_kp_z: float = 1000.0   # RESTORED to Cao original (was drifted to 1100)
    stance_kd_z: float = 10.0    # RESTORED to Cao original (was drifted to 20)
    swing_kd_z: float = 10.0
    # Baseline anti-chatter around nominal length (flight):
    # - Near l0, hard spring sign flips can cause leg "back-and-forth" jitter.
    # - Use a small deadband + velocity clip to keep motion spring-like and continuous.
    swing_l0_deadband_m: float = 0.025
    swing_l0_vel_deadband_mps: float = 0.35
    swing_axial_vel_clip_mps: float = 0.6
    # Over-length recovery in flight (l > l0):
    # keep pull-back stronger than near-l0 deadband behavior to avoid lingering extension.
    swing_overextend_kp_scale: float = 1.35
    # Extra noise guards for flight PD loop:
    # - clip xdot before Khd term
    # - clip roll/pitch gyro used in omega×x coupling
    swing_xdot_clip_mps: float = 0.6
    swing_omega_xy_clip_radps: float = 1.2

    # props / thrust
    # Treat 3 QP thrust variables as "per-arm total thrust" (N).
    # total baseline thrust = ratio*m*g
    # User request: props should handle more attitude work throughout stance/flight,
    # so the leg can focus more on velocity convergence.
    # Baseline thrust ratio (m*g). Non-zero keeps props active and responsive.
    # 2026-07-04: back to LOW baseline (0.02). The allocator now uses collective
    # lift: when a big corrective torque is demanded the un-floored arms spool up
    # transiently (sum rises), so idle thrust can stay low WITHOUT losing torque
    # authority -- per user: 平时低基础推力, 倾斜时迅速拉回.
    # 2026-07-05 19:43 log: at ratio 0.02 the arms idle near PWM 1030; torque
    # commands were instant/large (47 N one tick after a 6 deg tilt) but a prop
    # spooling from ~zero RPM needs 100-200 ms (the measured 134 ms lag), and
    # the low arm has no thrust to shed (one-sided differential). User chose to
    # KEEP the low idle -- so the response-speed ceiling stays; do not expect
    # sub-100ms attitude response at this baseline.
    prop_base_thrust_ratio: float = 0.02
    # Pure-leg test mode: hard-disable prop allocation in both stance and flight.
    # This keeps stance balance/hopping evaluation independent from propeller hardware state.
    pure_leg_mode: bool = False
    # Enable propeller modulation in stance for disturbance recovery.
    stance_use_props: bool = True
    thrust_total_ratio_max: float = 3.50  # QP cap on sum(thrust) <= ratio*m*g (higher prop authority)
    # Per-arm thrust cap passed to WBC-QP (N).
    # NOTE: 10N/arm was not enough roll authority in logs (roll diverged while saturated).
    # Per-arm thrust cap passed to WBC-QP (N).
    # With PWM capped at 1400us (see below), keep this conservative to avoid mapping saturation.
    thrust_max_each_n: float = 55.0

    # Stance prop policy
    stance_thrust_sum_min_ratio: float = 0.02
    stance_thrust_sum_max_ratio: float = 3.50

    # prop geometry in base frame (meters); default is symmetric with GREEN on +X
    prop_arm_len_m: float = 0.569451

    # ===== Prop PWM channel mapping (REAL ROBOT) =====
    # ModeE solves 3 thrust variables ordered with `prop_positions_b`:
    #   arm 0 at -90 deg (0,-L)          -> physical Motor3/PWM[3]
    #   arm 1 at +150 deg (-x,+y)        -> physical Motor2/PWM[2]
    #   arm 2 at +30 deg  (+x,+y)        -> physical Motor1/PWM[1]
    # 2026-07-06 remap per user: M3 sits on -Y, M1 on (+x,+y), M2 on (-x,+y)
    # (previous ((1,),(2,),(3,)) had M1<->M3 swapped -> reversed Y torque).
    prop_pwm_idx_per_arm: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (3,),  # arm 0 (-Y)    = Motor3
        (2,),  # arm 1 (-x,+y) = Motor2
        (1,),  # arm 2 (+x,+y) = Motor1
    )

    # motor torque limits (delta 3-RSR), per user: 10 Nm
    # IMPORTANT:
    # This limit is used INSIDE ModeECore (WBC-QP feasibility).
    # For bring-up safety, prefer limiting torques at the output layer in `modee/lcm_controller.py`
    # (tau_out_max / tau_out_scale) so the internal solver still behaves normally.
    tau_cmd_max_nm: tuple[float, float, float] = (20.0, 20.0, 20.0)
    # Motor torque sign convention (real robot wiring/driver):
    # Keep this as a *motor wiring/driver sign* override:
    #   +1 means "as modeled", -1 flips the commanded motor torque direction.
    # This applies to BOTH: stance torque mapping (A_tau_f) and flight swing tau_ref.
    tau_cmd_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # QP weights
    # Weight for tracking swing spring tau_ref in flight phase.
    # Higher value = QP prioritizes following swing spring torque (better leg length control).
    # CRITICAL: This must be comparable to slack weights (1e3~1e4) to actually work!
    # Previous value (1.0) was too small - QP ignored tau_ref entirely.
    # Increasing to 1e3 to enforce leg retraction in flight.
    wbc_w_tau_ref_flight: float = 1e3

    # ===== Prop thrust configuration =====
    # Minimum per-arm thrust (N). Must be <= (prop_base_thrust_ratio*m*g)/3 when we lock thrust_sum in stance.
    # Keep each arm above a small minimum so prop response is immediate.
    # (Kept at 0.1: with base ratio 0.02 a higher floor would silently raise
    # the effective idle via collective lift, which the user declined.)
    wbc_thrust_min_each_n: float = 0.1
    # - wbc_w_f_ref: soft tracking of f_ref (vertical spring force reference).
    # In SRB mode, f_ref only has vertical component; horizontal forces are determined
    # by QP to satisfy Tau_des. Keep moderate so QP can freely adjust horizontal forces.
    wbc_w_f_ref: float = 1.0

    # Optional clamp on stance horizontal contact force (world frame):
    #   |fx| <= wbc_fxy_abs_max, |fy| <= wbc_fxy_abs_max
    # This directly limits horizontal acceleration (smoothness) and prevents sudden runaway speed spikes.
    # Set <= 0 to disable.
    wbc_fxy_abs_max: float = 1000.0

    # ===== WBC-QP regularization (bias who does the work) =====
    # These weights bias the QP solution distribution between leg contact force vs prop thrust:
    # - Larger wbc_w_t => thrust is "expensive" => QP prefers using the leg when feasible.
    # - Larger wbc_w_f => contact force is "expensive" => QP prefers using thrust when feasible.
    #
    # Unified SRB-QP: leg and props are jointly optimized in BOTH stance and flight.
    # To ensure legs dominate in stance (props are weak), set wbc_w_t >> wbc_w_f.
    # The QP naturally uses legs first; props only contribute when legs hit limits.
    wbc_w_f: float = 1e-6
    wbc_w_t: float = 1e-1

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

    # ===== Stance vertical force floor (SRB-QP only) =====
    # Minimum/maximum normal force used by the stance wrench reference (N).
    # Keep min >= mg to avoid collapse during early compression.
    stance_fz_min: float = 0.0
    stance_fz_max: float = 260.0   # raised 240->260 for more vertical headroom in stance
    # Stance horizontal (attitude) force limits, applied by PROPORTIONAL scaling
    # (direction preserved) in the closed-form allocation:
    #   |fxy| <= stance_mu * fz   (friction cone: right at touchdown fz is still
    #                              small, so fxy ramps up WITH the normal force
    #                              instead of slamming in and slipping)
    #   |fxy| <= stance_fxy_max   (absolute ceiling independent of fz)
    # Set either to <=0 to disable that limit.
    stance_mu: float = 0.3
    # 15 -> 30 (2026-07-05): the fxy allocation now feeds forward the fz
    # lever-arm moment (foot 5-9 cm off-center @ fz~130 N needs ~20 N of fxy
    # just to cancel the tipping torque); 15 N clipped that cancellation and
    # 25% of stance #0 sat on the cap. The friction cone mu*fz above remains
    # the anti-slip bound (it is what protects right at touchdown).
    stance_fxy_max: float = 30.0

    # PWM limits
    pwm_min_us: float = 1000.0
    # PWM cap. Raised for stronger prop authority.
    pwm_max_us: float = 1700.0

    # ===== Bidirectional props (2026-07-06) =====
    # ESCs (T-Motor 4in1) run in 3D mode; the DDS bridge maps LCM pwm as:
    #   1000 = stop, 1000->2000 = forward (same scale as before), 1000->0 = REVERSE.
    # Physical outputs: prop1=AUX1, prop2=AUX4, prop3=AUX3 (AUX2 output damaged:
    # forward pulse widths pass, reverse-side widths do not; do NOT use AUX2/5/6).
    # NOTE: in 3D mode the forward throttle resolution is halved -> k_thrust must be
    # re-calibrated before trusting absolute thrust numbers.
    prop_bidir: bool = True
    # Per-arm REVERSE thrust budget (N) = allocator lower bound (only when prop_bidir).
    # USER REQUIREMENT (2026-07-06): total Z thrust must stay ~constant during attitude
    # correction. The differential solution is zero-sum by geometry (symmetric tri-rotor
    # => sum of attitude thrusts == 0), so the sum only changes when an arm hits this
    # floor and collective lift kicks in. Set the budget to the physical ceiling the
    # pwm mapping can deliver (pwm_rev_floor_us): k*(1000-700)^2 = 1.1e-4*300^2 ~ 9.9N,
    # so the floor practically never binds and the sum stays at the baseline.
    prop_reverse_max_n: float = 10.0
    # Lowest reverse pwm command (us). 300us of reverse range; reverse thrust per us is
    # aerodynamically weaker than forward anyway (fixed-pitch prop) -- calibrate before
    # deepening this.
    pwm_rev_floor_us: float = 700.0

    # ===== Propeller PWM mapping method =====
    # If True: use Hopper4-style k_thrust square-root relationship (pwm = 1000 + sqrt(thrust / k_thrust))
    # If False: use MotorTableModel lookup table (interpolation from measured data)
    # Hopper4-style PWM mapping (sqrt law) is the most predictable way to respect a strict PWM cap.
    use_hopper4_pwm_mapping: bool = True
    # Hopper4 thrust coefficient (N per (pwm_delta)^2, where pwm_delta = pwm - 1000)
    # Default from Hopper4: k_thrust = 1.47e-4
    # Tuned higher for sensitivity (more thrust for the same PWM command).
    # Formula: thrust = k_thrust * (pwm - 1000)^2
    # Inverse: pwm = 1000 + sqrt(thrust / k_thrust)
    # NOTE: 1.47e-5 makes even ~5N/arm saturate at pwm_max=1300 (sqrt mapping), starving attitude torque.
    prop_k_thrust: float = 1.10e-4

    # Use FC quaternion directly (recommended for real robot) vs. re-estimate from gyro+acc
    use_fc_quat: bool = True

    # ===== Flight phase attitude control gains (SO(3) roll/pitch) =====
    # Separate gains for roll and pitch axes to allow independent tuning.
    # kR: attitude error gain (proportional, larger = stiffer response)
    # kW: angular velocity damping gain (derivative, larger = more damping/braking)
    # tau_rp_max: maximum desired roll/pitch torque (Nm)  pitchdebug
    # NOTE (real robot): keep flight attitude authority reasonably high; thrust limits still provide safety.
    # kR = attitude-error->torque (proportional); kW = angular-rate->torque (damping/D).
    # RESTORED 2026-06-24 to the original CASE/Cao values (kR=60, kW=40) per user request
    # ("propeller 部分先改回去 / 不要弱化"). Earlier bench reductions (60/40 -> 28/14) were
    # attempts to tame a jitter/limit-cycle, but the prop behavior must match Cao's working
    # upper layer, not be weakened. Do NOT lower these again without the user's say-so.
    # 2026-07-01: RESTORED to the ACTUAL CASE/hopperHFAcase2026 values (verified by diffing
    # the CASE zip core.py) after they had been silently slashed to kR=5/kW=0 -> props had
    # ~1/12 the stiffness and ZERO rate damping, so "props can't balance the robot". CASE
    # ran these exact values on THIS physical robot and balanced.
    # 2026-07-04: values were SWAPPED vs CASE (had kR=40/kW=60). With the new
    # full-authority allocator the oversized kW (delayed ~20ms through ESC/props)
    # drove a 6.5Hz limit cycle (D-term std 22Nm vs P 2.6Nm in log). CASE actual:
    # kR=60, kW=40 (bench script even used 36/20). Do NOT raise kW to "add damping";
    # prop lag makes large kW anti-damping at these frequencies.
    # 2026-07-05 18:50 log: kR=100 drove a 4.2 Hz roll limit cycle. Measured
    # thrust-vs-gyro lag was ~134 ms (LCM->DDS->PX4->ESC->prop spool-up); past
    # ~90 deg of phase the D-term pumps the oscillation instead of damping it,
    # and at 4.2 Hz the lag was ~200 deg. The attitude crossover must stay
    # below ~2-3 Hz with this actuation delay -- that caps kR near the CASE
    # value of 60, NO MATTER how much more "angle authority" we want. kW cut
    # 30->15: at a 2-3 Hz crossover the delayed rate feedback contributes ~zero
    # real damping, so a big kW only adds noise drive and saturation.
    # (Prop D-term uses RAW gyro; the notch+LPF chain is leg/stance only.)
    # 2026-07-05 19:58 log (hop-to-hop divergence): with kR=120 the per-hop
    # touchdown attitude ALTERNATED sign with growing amplitude (+-1 -> +-4 ->
    # +-7 deg) and the latched velocity flipped with it (+-0.5 clamp hit) --
    # the in-flight limit cycle (~4.2 Hz, 0.35 s flight = 1.5 periods) lands
    # each hop with opposite tilt, the tilted 100 N push injects +-0.3-0.5 m/s,
    # and Raibert flips the foot each hop: velocity can NEVER converge. kR must
    # stay at 60 until the ~134 ms actuation lag is reduced (ActuatorMotors
    # direct channel / higher prop idle), NOT because 60 is "enough" but
    # because anything higher self-oscillates and destabilizes the hops.
    flight_kR_roll: float = 90.0
    flight_kW_roll: float = 35.0
    flight_kR_pitch: float = 90.0
    flight_kW_pitch: float = 35.0
    flight_tau_rp_max: float = 25.0

    # ===== Flight velocity -> attitude tilt (Raibert-style pull-back) =====
    # 2026-07-05 user request: props should NOT hold level; they should TILT
    # the body (up to prop_vel_tilt_max_deg) against the velocity error so the
    # thrust's horizontal component drags v_hat back toward the desired
    # velocity -- same philosophy as Raibert foot placement, but actuated by
    # the props during flight. Attitude target:
    #   a_des_xy = kv * (v_des - v_hat),  tan(tilt) = |a_des_xy| / g
    # kv=3.5: a 0.5 m/s velocity error asks 1.75 m/s^2 ~ 10 deg (saturates at
    # the max). Stance target stays LEVEL (leg fxy handles horizontal there).
    # 2026-07-05 19:40 DISABLED after hardware test: tilting the TARGET makes
    # the props tolerate up to 10 deg of body tilt (inside the tilted target
    # there is ~no restoring torque, hard push only beyond it). User wants the
    # opposite: restore level from the FIRST degree. Keep level target; pull-
    # back toward desired velocity is handled by Raibert foot placement only.
    prop_vel_tilt: bool = False
    prop_vel_tilt_kv: float = 3.5
    prop_vel_tilt_max_deg: float = 1.0

    # ===== Control mode switch =====
    # 1 = pure_leg:          closed-form leg, propellers OFF (stance & flight)
    # 2 = decouple_leg_prop: closed-form leg + lstsq prop overlay (stance & flight)
    # 3 = alias of 2 (legacy unified_qp flag; WBC-QP removed)
    control_mode: int = 2

    # ===== Stance phase attitude control (SRB SO(3) PD) =====
    # SRB approach: compute Tau_des = -kR*e_R - kW*omega in body frame, then let QP
    # find the optimal foot contact force f_foot such that r_foot × f_foot ≈ Tau_des,
    # subject to friction cone |fx,fy| ≤ μ·fz.
    # The QP naturally generates horizontal foot forces for attitude correction.
    # Priority: 1. Height (Apex) 2. Velocity 3. Attitude (via QP slack weights).
    # SRB stance attitude SO(3) PD gains (body-frame torque: tau = -kR*e_R - kW*omega).
    # QP maps Tau_des to horizontal foot forces via r_foot × f_foot = Tau_des.
    # RESTORED (2026-06-24) to CASE/Cao values per user ("it used to pure-leg hop").
    # The stance attitude kR had drifted down to 100 (from Cao's 250) -> 2.5x weaker
    # attitude authority, so the leg couldn't arrest tip-over fast enough before the foot
    # drifted forward under the vertical push (inverted-pendulum positive feedback) ->
    # crooked takeoff. IMU verified CORRECT (tilt test: nose-down=+pitch, right-down=+roll;
    # acc-consistency error 0.5 m/s^2), so this is a stance-gain regression, not an IMU sign.
    # 2026-06-24: ms-level log showed kpp=250 makes tau_des_y saturate (~-25) on a 6 deg
    # error -> leg torque tau1 pins at the 20 Nm cap and the QP thrashes tau0/tau2 by
    # +-5..10 Nm at ~20 Hz (the "baseline jitter" the user felt). Backed off to 140 so
    # tau_des stays in the linear (unsaturated) region for normal errors. Still well above
    # the previous 100 for authority.
    # 2026-07-01: RESTORED to ACTUAL CASE values (kpp=100, kpd=1) from the CASE zip; had been
    # cut to 5/0 (20x weaker stance attitude, no damping) -> couldn't arrest tip-over.
    # User-tuned values (2026-07-05: keep these, do NOT bulk-restore CASE).
    stance_kpp_x: float = 23.0    # leg stance kR roll
    stance_kpp_y: float = 23.0    # leg stance kR pitch
    stance_kpd_x: float = 3    # leg stance kW roll
    stance_kpd_y: float = 3    # leg stance kW pitch
    # ===== Stance D-term gyro conditioning: NOTCH + light LPF =====
    # We consume PX4 /fmu/out/sensor_combined = RAW gyro (PX4's own filter
    # pipeline only applies to vehicle_angular_velocity, which is not in the
    # DDS export list). FFT of the 04:53 log shows the stance gyro noise is
    # NARROW-BAND: 84% of the >1 Hz energy sits in 10-45 Hz (dominant
    # ~22.5 Hz, leg-motor structural vibration); flight is clean. A biquad
    # notch removes that band with almost no phase loss in the 3-6 Hz control
    # band, unlike the old 25 ms first-order LPF (-38 deg at 5 Hz).
    # Swept on the real logged stance gyro: wide notch 25 Hz / BW 25
    # (covers ~12-38 Hz) + 8 ms LPF gives D-torque jitter within ~30% of the
    # old filter while cutting phase loss at 5 Hz from -38 to -26 deg
    # (gain 0.95 vs 0.79) -> kpd can be raised without push-phase twitching.
    # Filters run CONTINUOUSLY (flight too) so touchdown sees no warm-up
    # transient. Set notch_hz <= 0 to disable the notch.
    # 2026-07-07: strengthened per user request -- notch BW 25->32 Hz (covers
    # ~11-43 Hz, the full measured noise band) and LPF 8->15 ms. Cost: phase
    # loss at 5 Hz grows back to roughly the old -38 deg level; if the push
    # phase starts twitching or D feels sluggish, revert to BW 25 / tau 0.008.
    stance_gyro_notch_hz: float = 25.0
    stance_gyro_notch_bw_hz: float = 32.0
    # Light first-order LPF after the notch (residual >45 Hz content).
    stance_gyro_lpf_tau: float = 0.015
    stance_tau_rp_max: float = 20.0

    # ===== WBC-QP slack weights (stance vs flight) =====
    # These weights decide whether the QP would rather:
    # - violate force equalities (via sF) to satisfy attitude torque (via r×f), OR
    # - accept attitude torque error (via sTau) so that forces follow references.
    #
    # Priority order (higher weight = higher priority):
    # - w_slack_Fz: vertical force tracking (height conservation). Higher = QP prioritizes tracking F_des_z.
    # - w_slack_tau_xy: roll/pitch torque tracking (attitude conservation). Higher = QP prioritizes tracking Tau_des_xy.
    # - w_slack_Fxy: horizontal force tracking (velocity convergence). Lower = allows horizontal force to deviate for attitude.
    #
    # Tuning guide:
    # - For "attitude conservation" priority: increase w_slack_tau_xy (e.g., 1e5), decrease w_slack_Fz (e.g., 5e4).
    # - For "height conservation" priority: increase w_slack_Fz (e.g., 1e5), decrease w_slack_tau_xy (e.g., 5e4).
    # - For balanced: keep both high but similar (e.g., both 8e4).
    # - For "leg-only balance": make w_slack_tau_xy very high so QP strictly tracks attitude via leg GRF (r×f).
    w_slack_Fxy: float = 1e3
    w_slack_Fz: float = 8e4
    w_slack_tau_xy: float = 3e5
    w_slack_tau_z: float = 1e3
    w_slack_Fxy_flight: float = 2e3
    w_slack_Fz_flight: float = 6e3
    # Flight: make torque slack expensive so props actually correct attitude.
    w_slack_tau_flight_xy: float = 1e6
    w_slack_tau_flight_z: float = 8e2

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
    # Use a dedicated LPF/clip so optimizer is not driven by raw IMU high-frequency noise.
    mpc_omega_lpf_tau: float = 0.020
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
        self.dt = float(cfg.dt)
        self.mass = float(cfg.mass_kg)
        self.gravity = float(cfg.gravity)
        self.com_b = np.asarray(cfg.com_b, dtype=float).reshape(3)
        self.I_body = np.diag(np.asarray(cfg.I_body_diag, dtype=float).reshape(3))

        # Body frame = leg FK frame = IMU frame (FRD: +X fwd, +Y right, +Z down).

        # Leg kinematics backend selection
        self._leg_model = str(getattr(cfg, "leg_model", "delta")).strip().lower()
        if self._leg_model not in ("delta", "serial"):
            print(f"[modee] WARN: unknown leg_model='{self._leg_model}', falling back to 'delta'")
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

        # Joint velocity LPF (used by kinematics to reduce Jacobian/qd jitter)
        self._joint_vel_lpf = np.zeros(3, dtype=float)
        self._joint_vel_lpf_init = False
        # CASE-style qd conditioning state (EMA + moving average)
        self._qd_ema = np.zeros(3, dtype=float)
        self._qd_ema_init = False
        _qd_ma_n = int(max(1, int(getattr(cfg, "qd_ma_window", 5))))
        self._qd_ma_buf = np.zeros((_qd_ma_n, 3), dtype=float)
        self._qd_ma_count = 0
        # Previous q sample for the dq/dt kinematics velocity (qd_kin_from_q_diff).
        self._q_diff_prev: np.ndarray | None = None
        # Heavily-filtered leg-extension velocity for the stance energy term
        # (anti-twitch: see energy_vel_lpf_tau).
        self._energy_vel_lpf: float = 0.0
        self._energy_vel_lpf_init: bool = False
        # Filtered gyro for the stance attitude D-term (see stance_gyro_lpf_tau).
        self._stance_gyro_lpf = np.zeros(3, dtype=float)
        self._stance_gyro_lpf_init: bool = False
        # Biquad notch state for the D-term gyro (x/y axes): [x1,x2,y1,y2] each.
        self._gyro_notch_x = np.zeros((2, 4), dtype=float)
        self._gyro_notch_init: bool = False
        self._gyro_notch_coefs: tuple | None = None

        # Shift-coordinate LPF + debounce (phase robustness)
        self._q_shift_lpf: float = 0.0
        self._qd_shift_lpf: float = 0.0
        self._shift_lpf_init: bool = False
        self._td_debounce_count: int = 0
        self._lo_debounce_count: int = 0

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
        self.att = SimpleIMUAttitudeEstimator(kp_acc=0.6, acc_g_min=0.90, acc_g_max=1.10, acc_lpf_tau=0.25)

        # estimator state
        self._v_hat_w = np.zeros(3, dtype=float)
        self._p_hat_w = np.array([0.0, 0.0, float(cfg.hop_z0)], dtype=float)
        self._v_hat_inited = False
        self._z_hat_contact_filt: float | None = None
        # com_filter.py style: rolling 10-sample window, no push/comp distinction
        self._vel_window_size: int = 10
        self._foot_vel_window = np.zeros((self._vel_window_size, 3), dtype=float)
        self._foot_pos_window = np.zeros((self._vel_window_size, 3), dtype=float)
        self._vel_window_count: int = 0
        # CASE-style push-phase average for the liftoff XY latch:
        # running sum of v_meas over the PUSH phase (after max compression).
        self._push_v_sum_w = np.zeros(3, dtype=float)
        self._push_v_sum_b = np.zeros(3, dtype=float)
        self._push_v_cnt: int = 0
        # Mid-stance clean-window accumulator (running sum of v_meas_w over the clean phase).
        self._vmeas_clean_sum = np.zeros(3, dtype=float)
        self._vmeas_clean_cnt: int = 0
        self._vmeas_clean_sum_b = np.zeros(3, dtype=float)
        # IMU XY delta-v integral since touchdown (world). The clean-window sums
        # store (v_meas - dv) anchors; the liftoff latch adds dv back at LO.
        self._stance_dv_w = np.zeros(3, dtype=float)
        # Per-stance sample buffers for the liftoff robust-fit latch (t, v_xy_w, v_xy_b).
        self._stance_v_t: list[float] = []
        self._stance_v_w: list[tuple[float, float]] = []
        self._stance_v_b: list[tuple[float, float]] = []
        # Liftoff latch source for debugging: 2=linear fit, 1=push mean, 0=instantaneous.
        self._lo_latch_src: int = 0
        # Velocity KF state (cfg.use_vel_kf): x = [v_w(3); b_a(3, body-frame accel bias)].
        # Generous initial bias covariance so b_a converges within the first ~10 hops.
        self._kf_v_w = np.zeros(3, dtype=float)
        self._kf_b_a = np.zeros(3, dtype=float)
        self._kf_P = np.diag([0.5, 0.5, 0.5, 0.25, 0.25, 0.25]).astype(float)
        self._flight_vel = np.zeros(3, dtype=float)
        self._flight_vel_b = np.zeros(3, dtype=float)
        # User override: freeze internal velocity estimate to zero (used to stop drift on demand).
        self._user_zero_vel_hold: bool = False

        # phase state
        self.sim_time = 0.0
        self._stance = False
        self._td_t: float | None = None
        self._lo_t: float | None = None
        self._q_shift_prev: float | None = None
        self._qd_shift_prev: float | None = None
        self._q_shift_td: float | None = None
        self._prev_vz: float | None = None

        # Per-hop apex feedback state (see apex_fb_* in ModeEConfig)
        self._apex_fb_int: float = float(getattr(cfg, "apex_fb_int_init", 0.0))
        self._z_apex_actual: float = float("nan")  # last measured apex h (m, for log)
        self._apex_err_last: float = float("nan")  # last apex error (m, for log)

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

        # apex + swing gating
        self._apex_reached = False
        self._z_lo: float | None = None  # liftoff height (base z, world frame)
        self._vz_lo: float | None = None  # liftoff vertical velocity (world frame)
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
        except Exception as e:
            # Fallback to 3 motors on indices 0/1/2 (safe-ish default)
            print(f"[modee] WARN: invalid prop_pwm_idx_per_arm ({e}); falling back to ((0,),(1,),(2,))")
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
        if self._dbg_stance_zero_fxfy or self._dbg_stance_zero_fz or self._dbg_stance_flip_fz:
            print(
                f"[modee][DEBUG] stance zero_fxfy={self._dbg_stance_zero_fxfy} "
                f"zero_fz={self._dbg_stance_zero_fz} flip_fz={self._dbg_stance_flip_fz}"
            )

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
        if self._dbg_stance_force_zero_out or self._dbg_stance_fxy_only_out or self._dbg_flight_fxy_only_out:
            print(
                f"[modee][DEBUG][HARDCODED] stance_force_zero_out={self._dbg_stance_force_zero_out} "
                f"stance_fxy_only_out={self._dbg_stance_fxy_only_out} "
                f"flight_fxy_only_out={self._dbg_flight_fxy_only_out}"
            )

    def user_reset(self) -> None:
        """
        User-requested reset (triggered by gamepad Y on the PC side).

        Purpose:
        - Zero drifting estimator/integrator states so a new experiment/log segment starts clean.
        - Keep the controller running; do NOT change driver mode here.
        """
        # Estimator/integrator states
        self._v_hat_w[:] = 0.0
        self._joint_vel_lpf[:] = 0.0
        self._joint_vel_lpf_init = False
        self._qd_ema[:] = 0.0
        self._qd_ema_init = False
        self._qd_ma_buf[:] = 0.0
        self._qd_ma_count = 0
        self._q_diff_prev = None
        self._energy_vel_lpf = 0.0
        self._energy_vel_lpf_init = False
        self._stance_gyro_lpf[:] = 0.0
        self._stance_gyro_lpf_init = False
        self._gyro_notch_x[:] = 0.0
        self._gyro_notch_init = False
        self._q_shift_lpf = 0.0
        self._qd_shift_lpf = 0.0
        self._shift_lpf_init = False
        self._td_debounce_count = 0
        self._lo_debounce_count = 0
        self._v_hat_inited = False
        self._z_hat_contact_filt = None
        self._foot_vel_window[:] = 0.0
        self._foot_pos_window[:] = 0.0
        self._vel_window_count = 0
        self._push_v_sum_w[:] = 0.0
        self._push_v_sum_b[:] = 0.0
        self._push_v_cnt = 0
        self._vmeas_clean_sum[:] = 0.0
        self._vmeas_clean_cnt = 0
        self._vmeas_clean_sum_b[:] = 0.0
        self._stance_dv_w[:] = 0.0
        self._stance_v_t.clear()
        self._stance_v_w.clear()
        self._stance_v_b.clear()
        self._lo_latch_src = 0
        self._kf_v_w[:] = 0.0
        self._kf_b_a[:] = 0.0
        self._kf_P = np.diag([0.5, 0.5, 0.5, 0.25, 0.25, 0.25]).astype(float)
        self._flight_vel[:] = 0.0
        self._flight_vel_b[:] = 0.0
        self._prev_vz = None
        self._apex_fb_int = float(getattr(self.cfg, "apex_fb_int_init", 0.0))
        self._z_apex_actual = float("nan")
        self._apex_err_last = float("nan")
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

        # Reset swing placement memory/gating
        self._apex_reached = False
        self._z_lo = None
        self._vz_lo = None

        # Reset attitude estimator state (only used if use_fc_quat=False)
        try:
            self.att.reset()
        except Exception:
            pass

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
    ) -> np.ndarray:
        """
        Tri-rotor thrust allocation, DIRECTION-PRESERVING with collective lift.

        Rotors can only push (t >= t_min), so at a LOW baseline a large corrective
        torque cannot come from lowering the floored arm -- instead the OTHER arms
        spool UP hard while the floored arm sits at t_min (total thrust rises
        transiently during the correction, back to baseline at rest). Realized
        torque = s * tau_des with ONE scalar s in [0,1]:
          - collective lift c keeps every arm >= t_min WITHOUT changing torque
            (uniform thrust adds zero roll/pitch moment);
          - s < 1 only when the per-arm max or the total-sum cap binds
            (whole differential scaled together -> direction exact, never
            per-arm clipping).
        Same policy in stance and flight; thrust_sum_max = thrust_total_ratio_max * m * g.
        """
        tau_des_w = np.asarray(tau_des_w, dtype=float).reshape(3)
        prop_r_w = np.asarray(prop_r_w, dtype=float).reshape(3, 3)
        z_thrust_w = np.asarray(z_thrust_w, dtype=float).reshape(3)
        M_prop = np.column_stack([
            _cross3(prop_r_w[i].reshape(3), z_thrust_w.reshape(3)) for i in range(3)
        ]).astype(float)
        thrusts_att = _lstsq_minnorm(M_prop[:2, :], tau_des_w[:2]).astype(float)
        base_each = float(thrust_sum_ref) / 3.0
        # Bidirectional ESCs: the per-arm floor becomes NEGATIVE (reverse thrust),
        # so a large corrective torque can also come from pushing the low arm below
        # zero instead of only spooling the others up -> less collective-lift
        # coupling, faster attitude response at low baseline. Same math: the
        # collective-lift/scaling logic below only assumes t_min <= t_max.
        if bool(getattr(self.cfg, "prop_bidir", False)):
            t_min = -abs(float(self.cfg.prop_reverse_max_n))
        else:
            t_min = float(self.cfg.wbc_thrust_min_each_n)
        t_max = float(self.cfg.thrust_max_each_n)
        tsum_cap = float(max(0.0, float(thrust_sum_max)))

        a = [float(thrusts_att[i]) for i in range(3)]
        a_min = min(a)
        a_max = max(a)
        a_sum = a[0] + a[1] + a[2]

        # Largest s in [0,1] such that thrusts = base + c(s) + s*att is feasible,
        # where c(s) = max(0, t_min - (base + s*a_min)) is the collective lift.
        # Case 1: no lift needed (base + s*a_min >= t_min).
        s = 1.0
        if a_max > 1e-9:
            s = min(s, max(0.0, t_max - base_each) / a_max)
        if tsum_cap > 1e-9 and a_sum > 1e-9:
            s = min(s, max(0.0, tsum_cap - 3.0 * base_each) / a_sum)
        if (base_each + s * a_min) < t_min - 1e-12:
            # Case 2: lift engaged. Arm ceiling: t_min + s*(a_max - a_min) <= t_max.
            # Sum cap: 3*t_min + s*(a_sum - 3*a_min) <= tsum_cap.
            s = 1.0
            d = a_max - a_min
            if d > 1e-9:
                s = min(s, max(0.0, t_max - t_min) / d)
            dsum = a_sum - 3.0 * a_min
            if tsum_cap > 1e-9 and dsum > 1e-9:
                s = min(s, max(0.0, tsum_cap - 3.0 * t_min) / dsum)
            s = max(0.0, min(1.0, s))

        c = max(0.0, t_min - (base_each + s * a_min))
        thrusts = np.array(
            [base_each + c + s * a[0], base_each + c + s * a[1], base_each + c + s * a[2]],
            dtype=float,
        )
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
        tau_b[0] = -float(self.cfg.flight_kR_roll) * float(rpy[0]) - float(self.cfg.flight_kW_roll) * float(omega_b[0])
        tau_b[1] = -float(self.cfg.flight_kR_pitch) * float(rpy[1]) - float(self.cfg.flight_kW_pitch) * float(omega_b[1])
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

    def _touchdown_ok(self) -> bool:
        # placeholder for future gating; keep always true in this minimal core.
        return True

    def _liftoff_ok(self) -> bool:
        if not bool(self._stance):
            return False
        t_td = float(self._td_t) if self._td_t is not None else float(self.sim_time)
        return (float(self.sim_time) - t_td) >= float(self.cfg.stance_lo_min_T)

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

    def _latch_fit_push_vxy(self, t_lo: float):
        """
        Liftoff XY latch via least-squares LINE FIT over the push-phase stance
        samples (self._stance_v_t / _v_w / _v_b), evaluated at t = t_lo (time
        since touchdown). XY velocity ramps up during push, so extrapolating a
        fit to the liftoff instant avoids the systematic lag of a plain mean
        while still smoothing over all samples. One 3-sigma outlier-rejection
        pass per axis. Returns (v_w_xy, v_b_xy) as 2-vectors, or None when
        there are not enough samples for a trustworthy fit.
        """
        min_n = int(max(4, int(getattr(self.cfg, "vel_latch_fit_min_n", 8))))
        n_all = len(self._stance_v_t)
        if n_all < min_n:
            return None
        t = np.asarray(self._stance_v_t, dtype=float)
        if bool(self._stance_prof_inited) and (self._stance_t_comp is not None):
            m = t >= float(self._stance_t_comp)
        else:
            m = t >= 0.5 * float(t_lo)  # profile not inited: second half of stance
        if int(np.count_nonzero(m)) < min_n:
            m = np.ones(t.shape, dtype=bool)
        t_f = t[m]
        if float(np.max(t_f) - np.min(t_f)) < 5.0 * float(self.dt):
            return None  # push window too short to define a slope
        # The buffers hold only TRUSTED samples (foot-planted gate), which end
        # before liftoff once the extension speed crosses the gate. Do not
        # extrapolate the fit beyond the last trusted sample: evaluate there.
        t_lo = float(min(float(t_lo), float(np.max(t_f))))
        v_w = np.asarray(self._stance_v_w, dtype=float).reshape(-1, 2)[m]
        v_b = np.asarray(self._stance_v_b, dtype=float).reshape(-1, 2)[m]
        out_w = np.zeros(2, dtype=float)
        out_b = np.zeros(2, dtype=float)
        for arr, out in ((v_w, out_w), (v_b, out_b)):
            for k in range(2):
                y = arr[:, k]
                c = np.polyfit(t_f, y, 1)
                r = y - np.polyval(c, t_f)
                s = float(np.std(r))
                if s > 1e-9:
                    keep = np.abs(r) <= 3.0 * s
                    nk = int(np.count_nonzero(keep))
                    if min_n <= nk < len(y):
                        c = np.polyfit(t_f[keep], y[keep], 1)
                out[k] = float(np.polyval(c, float(t_lo)))
        if not (np.all(np.isfinite(out_w)) and np.all(np.isfinite(out_b))):
            return None
        return out_w, out_b

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

        # --- Signal conditioning: joint velocity for kinematics ---
        # 2026-07-07: qd source = joint_vel input (hopper_data_lcmt.qd, which the
        # Jetson driver fills with the AK60 CAN-reported velocity). Set
        # qd_kin_from_q_diff=True to differentiate q on the PC instead.
        if bool(getattr(self.cfg, "qd_kin_from_q_diff", True)):
            if self._q_diff_prev is None:
                qd_src = np.zeros(3, dtype=float)
            else:
                qd_src = ((joint_pos - self._q_diff_prev) / float(self.dt)).astype(float)
            self._q_diff_prev = joint_pos.copy()
        else:
            qd_src = joint_vel.copy()

        joint_vel_kin = qd_src.copy()
        # 2026-07-06: legacy first-order LPF REPLACED by the CASE-style chain below
        # (EMA + moving average). joint_vel_lpf_tau=0; code kept for reference only.
        # try:
        #     tau = float(getattr(self.cfg, "joint_vel_lpf_tau", 0.0))
        # except Exception:
        #     tau = 0.0
        # if float(tau) > 0.0:
        #     if not bool(self._joint_vel_lpf_init):
        #         self._joint_vel_lpf = qd_src.copy()
        #         self._joint_vel_lpf_init = True
        #     else:
        #         a = float(_clipf(float(self.dt) / (float(tau) + float(self.dt)), 0.0, 1.0))
        #         self._joint_vel_lpf = (1.0 - a) * self._joint_vel_lpf + a * qd_src
        #     joint_vel_kin = np.asarray(self._joint_vel_lpf, dtype=float).reshape(3).copy()
        # CASE/OMEGA-style qd conditioning: EMA (forgetting factor) + moving average.
        # Their chain: q -> K(z-1)/(Ts*z) (that's qd_kin_from_q_diff) -> EMA(0.4) -> MA.
        ff = float(getattr(self.cfg, "qd_ema_forgetting", 0.0))
        if 0.0 < ff < 1.0:
            if not bool(self._qd_ema_init):
                self._qd_ema = joint_vel_kin.copy()
                self._qd_ema_init = True
            else:
                self._qd_ema = ff * self._qd_ema + (1.0 - ff) * joint_vel_kin
            joint_vel_kin = self._qd_ema.astype(float).reshape(3).copy()
        ma_n = int(getattr(self.cfg, "qd_ma_window", 1))
        if ma_n > 1 and self._qd_ma_buf.shape[0] == ma_n:
            self._qd_ma_buf[:-1] = self._qd_ma_buf[1:]
            self._qd_ma_buf[-1] = joint_vel_kin
            self._qd_ma_count = min(self._qd_ma_count + 1, ma_n)
            if self._qd_ma_count >= ma_n:
                joint_vel_kin = np.mean(self._qd_ma_buf, axis=0).astype(float).reshape(3)

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
            # qd_shift: rate of change of leg length (positive = extending, negative = compressing)
            # foot_vdot_vicon is in delta/vicon frame, we need the component along the leg direction
            if leg_length > 1e-6:
                leg_dir = foot_vicon / leg_length  # unit vector from base to foot
                leg_extension_vel = float(np.dot(foot_vdot_vicon, leg_dir))  # positive = extending
                qd_shift = float(leg_extension_vel)  # positive = extending, negative = compressing
            else:
                qd_shift = 0.0

        # --- LPF q_shift / qd_shift for phase detection robustness ---
        q_shift_raw = float(q_shift)
        qd_shift_raw = float(qd_shift)
        try:
            tau_q = float(getattr(self.cfg, "q_shift_lpf_tau", 0.0))
        except Exception:
            tau_q = 0.0
        try:
            tau_qd = float(getattr(self.cfg, "qd_shift_lpf_tau", 0.0))
        except Exception:
            tau_qd = 0.0
        if (float(tau_q) > 0.0) or (float(tau_qd) > 0.0):
            if not bool(self._shift_lpf_init):
                self._q_shift_lpf = q_shift_raw
                self._qd_shift_lpf = qd_shift_raw
                self._shift_lpf_init = True
            else:
                if float(tau_q) > 0.0:
                    a = float(_clipf(float(self.dt) / (float(tau_q) + float(self.dt)), 0.0, 1.0))
                    self._q_shift_lpf = float((1.0 - a) * float(self._q_shift_lpf) + a * q_shift_raw)
                else:
                    self._q_shift_lpf = q_shift_raw
                if float(tau_qd) > 0.0:
                    a = float(_clipf(float(self.dt) / (float(tau_qd) + float(self.dt)), 0.0, 1.0))
                    self._qd_shift_lpf = float((1.0 - a) * float(self._qd_shift_lpf) + a * qd_shift_raw)
                else:
                    self._qd_shift_lpf = qd_shift_raw
            # overwrite for downstream phase machine
            q_shift = float(self._q_shift_lpf) if float(tau_q) > 0.0 else q_shift_raw
            qd_shift = float(self._qd_shift_lpf) if float(tau_qd) > 0.0 else qd_shift_raw

        # --- Attitude estimate (body->world) ---
        if bool(self.cfg.use_fc_quat) and (imu_quat_wxyz is not None):
            q_hat = _quat_normalize_wxyz(np.asarray(imu_quat_wxyz, dtype=float).reshape(4))
        else:
            q_hat = self.att.update(omega_b=imu_gyro_b, acc_b=imu_acc_b, dt=float(self.dt))
        R_wb_hat = _quat_to_R_wb(q_hat)
        rpy_hat = _R_to_rpy_xyz(R_wb_hat)
        z_w = np.asarray(R_wb_hat[:, 2], dtype=float).reshape(3)
        # Propeller thrust direction in WORLD. In the current FRD IMU/body convention,
        # propellers push along body -Z, so the world thrust direction is -R_wb[:, 2].
        z_thrust_w = (-z_w).astype(float).reshape(3)

        # --- Base velocity from leg kinematics (foot assumed stationary in WORLD) ---
        # v_base_w = R_wb @ ( -foot_vdot_b - omega_b x foot_b )
        # NOTE (2026-07): fixed two legacy com_filter.py errors here: the gyro used to be
        # scaled by 0.1 (10x underestimated omega x r term) and the cross product mixed
        # world-frame omega with body-frame foot position.
        v_base_from_foot_w = (R_wb_hat @ (
            -foot_vrel_b.reshape(3) - _cross3(imu_gyro_b.reshape(3), foot_b.reshape(3))
        )).reshape(3)

        # --- Base velocity: LEG KINEMATICS ONLY (NO IMU accel integration) ---
        # User request (2026-07): never integrate IMU acceleration -- accel bias made the
        # estimate drift. Instead:
        #   STANCE: all 3 axes from leg kinematics (foot stationary in world), see below;
        #   FLIGHT: XY held at the liftoff average; vz propagated with the BALLISTIC model
        #           (vz += g_eff*dt), deterministic and re-anchored by the leg at every TD.
        if not bool(self._v_hat_inited):
            self._v_hat_w = np.zeros(3, dtype=float)
            self._v_hat_inited = True

        # cache previous shift/qd (used for debounce / retiming)
        q_shift_prev = self._q_shift_prev
        qd_shift_prev = self._qd_shift_prev

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
            cond_td = (float(q_shift) <= -td_thr) and (t_in_flight >= phase_min_t)

            if bool(cond_td):
                touchdown_evt = True
                self._stance = True
                self._td_t = float(self.sim_time)
                # ---- Flight-time apex measurement (estimator-independent) ----
                # h = g_eff * T^2 / 8 for a symmetric ballistic arc (same takeoff
                # and landing height -- true for in-place hopping). Uses ONLY the
                # LO->TD timestamps, so a bad vz_lo latch cannot corrupt the
                # height loop (see apex_meas_flight_time).
                if bool(getattr(self.cfg, "apex_meas_flight_time", True)) and \
                   bool(getattr(self.cfg, "apex_use_feedback", True)) and \
                   (self._lo_t is not None):
                    T_fl = float(self.sim_time) - float(self._lo_t)
                    if 0.06 <= T_fl <= 1.5:
                        g_eff_fl_m = float(self.gravity) * (
                            1.0 + float(z_thrust_w[2]) * float(self.cfg.prop_base_thrust_ratio)
                        )
                        g_eff_fl_m = float(max(1e-3, g_eff_fl_m))
                        h_act_ft = g_eff_fl_m * T_fl * T_fl / 8.0
                        self._z_apex_actual = float(h_act_ft)
                        err_ft = float(self.cfg.hop_height_m) - float(h_act_ft)
                        self._apex_err_last = float(err_ft)
                        ki_ft = float(getattr(self.cfg, "apex_fb_ki", 0.0))
                        i_max_ft = float(getattr(self.cfg, "apex_fb_int_max", 0.6))
                        self._apex_fb_int = float(_clipf(
                            float(self._apex_fb_int) + ki_ft * err_ft, 0.0, i_max_ft))
                self._apex_reached = False
                self._td_debounce_count = 0
                self._lo_debounce_count = 0
                # com_filter.py: do NOT clear window at touchdown (rolling)
                # Push-phase average accumulator IS per-stance: clear at touchdown.
                self._push_v_sum_w[:] = 0.0
                self._push_v_sum_b[:] = 0.0
                self._push_v_cnt = 0
                # Mid-stance clean velocity accumulator IS per-stance: clear it at touchdown.
                self._vmeas_clean_sum[:] = 0.0
                self._vmeas_clean_sum_b[:] = 0.0
                self._vmeas_clean_cnt = 0
                self._stance_dv_w[:] = 0.0
                # Per-stance sample buffers for the liftoff robust-fit latch.
                self._stance_v_t.clear()
                self._stance_v_w.clear()
                self._stance_v_b.clear()
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

                # touchdown z estimate from kinematics (assume foot at ground z=0).
                # World +Z DOWN: the body above the ground has NEGATIVE z (p_z = -height).
                z_td_est = -float((R_wb_hat @ foot_b.reshape(3))[2])
                self._z_hat_contact_filt = float(z_td_est)
                self._p_hat_w[2] = float(z_td_est)

                # Takeoff speed target for desired apex (ballistic, with prop assist).
                # z_thrust_w points UP (level: [0,0,-1] in the +Z-down world), so the
                # baseline prop thrust REDUCES effective gravity: g_eff = g*(1 - rho).
                g_eff = float(self.gravity + (float(z_thrust_w[2]) * float(self.mass) * float(self.gravity) * float(self.cfg.prop_base_thrust_ratio)) / max(1e-6, float(self.mass)))
                g_eff = float(max(1e-3, g_eff))
                dz_tgt = float(max(0.05, float(self.cfg.hop_height_m) + float(self._apex_fb_int)))
                g_eff_log = float(g_eff)
                dz_tgt_log = float(dz_tgt)
                v_to_nominal = float(np.sqrt(2.0 * g_eff * dz_tgt))
                self._v_to_cmd = float(_clipf(v_to_nominal, float(self.cfg.v_to_min), float(self.cfg.v_to_max)))

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
                except Exception as e:
                    print(f"FAILED TO INIT UNIFIED STANCE: {e}")
                    self._stance_prof_inited = False

        # track previous shift for debounce
        if np.isfinite(q_shift):
            self._q_shift_prev = float(q_shift)
        if np.isfinite(qd_shift):
            self._qd_shift_prev = float(qd_shift)

        # ===== Hopper4-style liftoff =====
        if bool(self._stance) and np.isfinite(float(q_shift)):
            td_t = float(self._td_t) if self._td_t is not None else float(self.sim_time)
            t_in_stance = float(self.sim_time) - td_t
            cond_lo = (float(q_shift) >= lo_thr) and (t_in_stance >= phase_min_t)
            # No-prop mode: do not leave stance with large roll/pitch rates.
            # This prevents "looks balanced in stance but falls in flight" failures.
            no_prop_mode = bool(getattr(self.cfg, "pure_leg_mode", False)) or (
                (not bool(self.cfg.stance_use_props)) and (float(self.cfg.prop_base_thrust_ratio) <= 1e-9)
            )
            if bool(no_prop_mode) and bool(getattr(self.cfg, "lo_use_omega_gate_no_prop", True)):
                gate_dps = float(max(0.0, float(getattr(self.cfg, "lo_omega_gate_dps", 20.0))))
                wx_dps = abs(float(np.rad2deg(float(imu_gyro_b[0])))) if np.isfinite(float(imu_gyro_b[0])) else 0.0
                wy_dps = abs(float(np.rad2deg(float(imu_gyro_b[1])))) if np.isfinite(float(imu_gyro_b[1])) else 0.0
                omega_ok = (wx_dps <= gate_dps) and (wy_dps <= gate_dps)
                # Safety escape: if the leg extends too much, force LO to avoid over-extension.
                q_force = float(max(float(lo_thr), float(getattr(self.cfg, "lo_force_liftoff_q_shift_m", 0.020))))
                force_lo = float(q_shift) >= q_force
                lo_tmax = float(max(float(phase_min_t), float(getattr(self.cfg, "lo_no_prop_max_stance_T", 0.32))))
                if bool(force_lo):
                    cond_lo = True
                else:
                    cond_lo = bool(cond_lo) and (bool(omega_ok) or (t_in_stance >= lo_tmax))
            if bool(cond_lo):
                liftoff_evt = True
                self._stance = False
                self._lo_t = float(self.sim_time)
                self._lo_debounce_count = 0
                # ---- Horizontal takeoff velocity for the Raibert flight target ----
                # Preferred: push-phase LINEAR FIT extrapolated to t_LO (smooth,
                # no mean-lag). Fallbacks: push-phase mean, then instantaneous.
                fit_res = None
                if bool(getattr(self.cfg, "vel_latch_fit", True)):
                    try:
                        fit_res = self._latch_fit_push_vxy(float(t_in_stance))
                    except Exception:
                        fit_res = None
                if fit_res is not None:
                    self._lo_latch_src = 2
                    v_fw, v_fb = fit_res
                    self._flight_vel = np.array([float(v_fw[0]), float(v_fw[1]), 0.0], dtype=float)
                    self._flight_vel_b = np.array([float(v_fb[0]), float(v_fb[1]), 0.0], dtype=float)
                elif int(self._push_v_cnt) > 0:
                    self._lo_latch_src = 1
                    inv_n = 1.0 / float(self._push_v_cnt)
                    self._flight_vel = (self._push_v_sum_w * inv_n).astype(float).reshape(3)
                    self._flight_vel_b = (self._push_v_sum_b * inv_n).astype(float).reshape(3)
                else:
                    self._lo_latch_src = 0
                    self._flight_vel_b = (
                        (-foot_vrel_b.reshape(3))
                        - _cross3(imu_gyro_b.reshape(3), foot_b.reshape(3))
                    ).astype(float).reshape(3)
                    self._flight_vel = (R_wb_hat @ self._flight_vel_b.reshape(3)).reshape(3).copy()
                self._flight_vel[2] = 0.0
                self._flight_vel_b[2] = 0.0
                # Safety clamp on the latched |XY| velocity (phantom-latch guard),
                # proportional scaling so the direction is preserved.
                v_clamp = float(getattr(self.cfg, "vel_latch_clamp_mps", 0.0))
                if v_clamp > 0.0:
                    n_w = float(np.hypot(float(self._flight_vel[0]), float(self._flight_vel[1])))
                    if n_w > v_clamp and n_w > 1e-9:
                        s_v = v_clamp / n_w
                        self._flight_vel[0] = float(self._flight_vel[0]) * s_v
                        self._flight_vel[1] = float(self._flight_vel[1]) * s_v
                    n_b = float(np.hypot(float(self._flight_vel_b[0]), float(self._flight_vel_b[1])))
                    if n_b > v_clamp and n_b > 1e-9:
                        s_vb = v_clamp / n_b
                        self._flight_vel_b[0] = float(self._flight_vel_b[0]) * s_vb
                        self._flight_vel_b[1] = float(self._flight_vel_b[1]) * s_vb
                # Record liftoff state for apex prediction (used by apex feedback loop)
                if bool(getattr(self.cfg, "apex_use_feedback", True)):
                    self._z_lo = float(self._p_hat_w[2])
                    self._vz_lo = float(self._v_hat_w[2])

        # ===== Velocity estimator =====
        # use_vel_kf=True (default): Cheetah-style KF -- IMU prediction every tick
        #   (accel bias estimated online), leg-kinematics velocity update in stance.
        # use_vel_kf=False (legacy): stance = instantaneous leg kinematics;
        #   flight = XY held at latched flight_vel, vz ballistic.
        v_meas_w = None
        in_push_now = False
        meas_trusted = False
        use_kf = bool(getattr(self.cfg, "use_vel_kf", True))
        if bool(getattr(self, "_user_zero_vel_hold", False)):
            # User "hard stop": freeze the estimate at exactly zero.
            self._v_hat_w[:] = 0.0
            self._kf_v_w[:] = 0.0
        elif bool(self._stance):
            v_meas_w = v_base_from_foot_w.reshape(3).copy()
            if np.all(np.isfinite(v_meas_w)):
                # "Foot planted" trust gate (see vel_meas_* in ModeEConfig):
                # reject the TD impact transient and the fast-extension phase
                # where the unloaded foot skates (phantom XY velocity).
                _td_t_g = float(self._td_t) if self._td_t is not None else float(self.sim_time)
                _t_st_g = float(self.sim_time) - _td_t_g
                meas_trusted = (
                    (_t_st_g >= float(getattr(self.cfg, "vel_meas_td_skip_s", 0.03)))
                    and np.isfinite(float(qd_shift))
                    and (float(qd_shift) < float(getattr(self.cfg, "vel_meas_qd_max_mps", 0.25)))
                )
                self._foot_vel_window[:-1] = self._foot_vel_window[1:]
                self._foot_vel_window[-1] = foot_vdot_vicon.copy()
                self._foot_pos_window[:-1] = self._foot_pos_window[1:]
                self._foot_pos_window[-1] = foot_vicon.copy()
                self._vel_window_count = min(self._vel_window_count + 1, self._vel_window_size)
                if not use_kf:
                    # Legacy estimator: XY only from trusted samples (hold last
                    # value through gated ticks); z (leg axial) tracks always.
                    if meas_trusted:
                        self._v_hat_w[0] = float(v_meas_w[0])
                        self._v_hat_w[1] = float(v_meas_w[1])
                    self._v_hat_w[2] = float(v_meas_w[2])
                # Collect TRUSTED stance samples (with time) for the liftoff
                # latch buffers. Gated samples (TD impact / fast-extension foot
                # skating) are excluded: on an in-place hop they read 1-2 m/s
                # of phantom XY, which no fit or average can reject.
                td_t_acc = float(self._td_t) if self._td_t is not None else float(self.sim_time)
                t_in_st_acc = float(self.sim_time) - td_t_acc
                v_meas_b_now = (
                    (-foot_vrel_b.reshape(3))
                    - _cross3(imu_gyro_b.reshape(3), foot_b.reshape(3))
                ).astype(float).reshape(3)
                if meas_trusted and len(self._stance_v_t) < 2000:
                    self._stance_v_t.append(float(t_in_st_acc))
                    self._stance_v_w.append((float(v_meas_w[0]), float(v_meas_w[1])))
                    self._stance_v_b.append((float(v_meas_b_now[0]), float(v_meas_b_now[1])))
                # PUSH-phase average accumulator for the liftoff XY latch
                # (trusted samples only; with the skate gate active this covers
                # early push, before the extension speed crosses the gate).
                if bool(self._stance_prof_inited) and (self._stance_t_comp is not None):
                    in_push_acc = t_in_st_acc >= float(self._stance_t_comp)
                else:
                    # Profile not initialized: fall back to "leg extending".
                    in_push_acc = np.isfinite(float(qd_shift)) and (float(qd_shift) > 0.0)
                if in_push_acc and meas_trusted:
                    self._push_v_sum_w += v_meas_w
                    self._push_v_sum_b += v_meas_b_now
                    self._push_v_cnt += 1
                in_push_now = bool(in_push_acc)
                if bool(getattr(self.cfg, "use_mid_stance_vel", False)):
                    # IMU XY delta-v since touchdown (world). Gravity is z-only in
                    # the +Z-down world, so world-XY accel = (R_wb @ specific_force)_xy.
                    if bool(getattr(self.cfg, "vel_latch_push_dv", False)):
                        a_w_now = (R_wb_hat @ imu_acc_b.reshape(3)).reshape(3)
                        if np.isfinite(a_w_now[0]) and np.isfinite(a_w_now[1]):
                            self._stance_dv_w[0] += float(a_w_now[0]) * float(self.dt)
                            self._stance_dv_w[1] += float(a_w_now[1]) * float(self.dt)
                    # Clean mid-stance accumulator (only when use_mid_stance_vel).
                    if (t_in_st_acc >= float(self.cfg.vel_clean_td_skip_s)) and \
                       (float(qd_shift) < float(self.cfg.vel_clean_push_qd_thresh)):
                        dv0 = float(self._stance_dv_w[0])
                        dv1 = float(self._stance_dv_w[1])
                        if not bool(getattr(self.cfg, "vel_latch_push_dv", False)):
                            dv0 = 0.0
                            dv1 = 0.0
                        self._vmeas_clean_sum[0] += float(v_meas_w[0]) - dv0
                        self._vmeas_clean_sum[1] += float(v_meas_w[1]) - dv1
                        self._vmeas_clean_sum[2] += float(v_meas_w[2])
                        self._vmeas_clean_sum_b[0] += float(v_meas_b_now[0])
                        self._vmeas_clean_sum_b[1] += float(v_meas_b_now[1])
                        self._vmeas_clean_sum_b[2] += float(v_meas_b_now[2])
                        self._vmeas_clean_cnt += 1
            else:
                # Invalid kinematics this tick: hold the previous estimate.
                v_meas_w = None
        elif not use_kf:
            # LEGACY FLIGHT: ballistic vz (deterministic, no sensor drift; re-anchored
            # by the leg at every touchdown). Baseline prop thrust reduces effective
            # gravity: g_eff = g * (1 + z_thrust_w[2] * rho), level: z_thrust_w[2] ~= -1.
            g_eff_fl = float(self.gravity) * (
                1.0 + float(z_thrust_w[2]) * float(self.cfg.prop_base_thrust_ratio)
            )
            self._v_hat_w[0] = float(self._flight_vel[0])
            self._v_hat_w[1] = float(self._flight_vel[1])
            self._v_hat_w[2] = float(self._v_hat_w[2]) + g_eff_fl * float(self.dt)

        # ---- Velocity KF: predict with IMU, update with leg kinematics in stance ----
        # x = [v_w (3); b_a (3, body-frame accel bias)]
        # Predict: v_w += (R_wb @ (f_b - b_a) + g_w) * dt, F = [[I, -R*dt], [0, I]]
        # Update (stance): z = v_base_from_foot_w, H = [I, 0]
        # IMU specific force naturally includes prop thrust and drag, so flight
        # needs no separate ballistic model; the stance updates keep b_a observable.
        if use_kf and not bool(getattr(self, "_user_zero_vel_hold", False)):
            dt_kf = float(self.dt)
            g_w_kf = np.array([0.0, 0.0, float(self.gravity)], dtype=float)  # +Z DOWN
            if np.all(np.isfinite(imu_acc_b)):
                f_b_kf = imu_acc_b.reshape(3).astype(float)
            else:
                # Missing accel this tick: coast (a_w = 0 <=> pure gravity cancel).
                f_b_kf = (R_wb_hat.T @ (-g_w_kf)).reshape(3) + self._kf_b_a
            a_w_kf = (R_wb_hat @ (f_b_kf - self._kf_b_a)).reshape(3) + g_w_kf
            self._kf_v_w = self._kf_v_w + a_w_kf * dt_kf
            F_kf = np.eye(6, dtype=float)
            F_kf[0:3, 3:6] = -R_wb_hat * dt_kf
            sa = float(getattr(self.cfg, "vel_kf_sigma_acc", 0.6))
            sb = float(getattr(self.cfg, "vel_kf_sigma_bias", 0.03))
            Q_kf = np.zeros((6, 6), dtype=float)
            Q_kf[0:3, 0:3] = (sa * sa * dt_kf) * np.eye(3)
            Q_kf[3:6, 3:6] = (sb * sb * dt_kf) * np.eye(3)
            self._kf_P = F_kf @ self._kf_P @ F_kf.T + Q_kf
            # Update only on TRUSTED samples ("foot planted" gate); gated ticks
            # (TD impact, fast-extension skating) run predict-only on the IMU.
            if (v_meas_w is not None) and meas_trusted:
                r_std = float(getattr(self.cfg, "vel_kf_meas_std", 0.12))
                if bool(in_push_now):
                    r_std = max(r_std, float(getattr(self.cfg, "vel_kf_meas_std_push", 0.30)))
                S_kf = self._kf_P[0:3, 0:3] + (r_std * r_std) * np.eye(3)
                K_kf = self._kf_P[:, 0:3] @ np.linalg.inv(S_kf)  # 6x3
                dx = (K_kf @ (v_meas_w.reshape(3) - self._kf_v_w)).reshape(6)
                self._kf_v_w = self._kf_v_w + dx[0:3]
                b_max = float(getattr(self.cfg, "vel_kf_bias_max", 1.5))
                self._kf_b_a = np.clip(self._kf_b_a + dx[3:6], -b_max, b_max)
                IKH = np.eye(6, dtype=float)
                IKH[:, 0:3] -= K_kf
                self._kf_P = IKH @ self._kf_P
                self._kf_P = 0.5 * (self._kf_P + self._kf_P.T)
            self._v_hat_w = self._kf_v_w.astype(float).reshape(3).copy()

        # integrate position + stance z correction
        self._p_hat_w = self._p_hat_w + self._v_hat_w * float(self.dt)
        if bool(self._stance):
            z_meas = -float((R_wb_hat @ foot_b.reshape(3))[2])
            if self._z_hat_contact_filt is None:
                self._z_hat_contact_filt = float(z_meas)
            z_tau = 0.05
            az = float(_clipf(float(self.dt) / (z_tau + float(self.dt)), 0.0, 1.0))
            self._z_hat_contact_filt = (1.0 - az) * float(self._z_hat_contact_filt) + az * float(z_meas)
            self._p_hat_w[2] = float(self._z_hat_contact_filt)

        # apex detection (flight): vz_hat sign change
        # World +Z is DOWN: ascending => vz < 0, descending => vz > 0.
        # Apex is the crossing from negative (up) to non-negative (down).
        vz_hat = float(self._v_hat_w[2])
        if self._prev_vz is None:
            self._prev_vz = float(vz_hat)
        if (not bool(self._stance)) and (float(self._prev_vz) < 0.0) and (float(vz_hat) >= 0.0):
            apex_evt = True
            self._apex_reached = True
            # ---- Per-hop apex feedback: integrate the measured height error ----
            # h_actual = z_lo - z_apex (world +Z DOWN: apex z is more negative).
            # The energy pump is P-only, so without this integrator the actual
            # apex sits far below hop_height_m (see apex_fb_* in ModeEConfig).
            # p_hat-based apex update: SKIPPED when apex_meas_flight_time is on
            # (the flight-time measurement at the next touchdown replaces it --
            # p_hat integrates the latched vz_lo, which log 15:00 showed can be
            # ~0.9 m/s wrong at identical push forces).
            if bool(getattr(self.cfg, "apex_use_feedback", True)) and (self._z_lo is not None) \
               and not bool(getattr(self.cfg, "apex_meas_flight_time", True)):
                h_act = float(self._z_lo) - float(self._p_hat_w[2])
                if np.isfinite(h_act):
                    self._z_apex_actual = float(h_act)
                    err = float(self.cfg.hop_height_m) - float(h_act)
                    self._apex_err_last = float(err)
                    ki = float(getattr(self.cfg, "apex_fb_ki", 0.0))
                    i_max = float(getattr(self.cfg, "apex_fb_int_max", 0.6))
                    self._apex_fb_int = float(_clipf(
                        float(self._apex_fb_int) + ki * err, 0.0, i_max))
        self._prev_vz = float(vz_hat)

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
        pure_leg_mode = bool(getattr(self.cfg, "pure_leg_mode", False))
        thrust_sum_ref = float(self.mass * self.gravity * float(self.cfg.prop_base_thrust_ratio))
        # Global propeller enable gate (for single-leg/no-prop tuning).
        props_enabled_ctrl = (not bool(pure_leg_mode)) and (
            bool(self.cfg.stance_use_props) or (float(self.cfg.prop_base_thrust_ratio) > 1e-9)
        )
        if bool(pure_leg_mode) or not bool(props_enabled_ctrl):
            thrust_sum_ref = 0.0
        # ===== Compute SO(3) attitude error EARLY =====
        # Use the quaternion published in hopper_imu_lcmt (q_hat -> R_wb_hat), and
        # the yaw published in the same LCM message to build a yaw-free level target.
        # This matches the paper form:
        #   e_R = 1/2 * vee(R_des^T R - R^T R_des)
        # while keeping yaw uncontrolled and avoiding the rpy_hat Euler branch.
        yaw_des = float(imu_rpy[2])
        R_des = _Rz(yaw_des)
        # ---- Flight velocity->tilt target (Raibert-style pull-back, props) ----
        # Replace the LEVEL flight target with a tilt that accelerates the body
        # toward the desired velocity (see prop_vel_tilt in ModeEConfig).
        # Stance keeps the level target: horizontal control there is leg fxy.
        if (not bool(self._stance)) and bool(props_enabled_ctrl) and \
           bool(getattr(self.cfg, "prop_vel_tilt", False)):
            kv_t = float(getattr(self.cfg, "prop_vel_tilt_kv", 0.0))
            a_des_x = kv_t * (float(desired_v_xy_w[0]) - float(self._flight_vel[0]))
            a_des_y = kv_t * (float(desired_v_xy_w[1]) - float(self._flight_vel[1]))
            g_t = float(self.gravity)
            a_cap = g_t * math.tan(math.radians(float(getattr(self.cfg, "prop_vel_tilt_max_deg", 10.0))))
            a_n = float(math.hypot(a_des_x, a_des_y))
            if a_n > a_cap and a_n > 1e-9:
                s_a = a_cap / a_n
                a_des_x *= s_a
                a_des_y *= s_a
            # World +Z is DOWN. Thrust must point "up + toward a_des":
            # t_dir ~ (a_des_x, a_des_y, -g), and body +Z axis = -t_dir.
            t_dir = np.array([a_des_x, a_des_y, -g_t], dtype=float)
            t_dir /= float(np.linalg.norm(t_dir))
            z_des = (-t_dir).reshape(3)
            # Keep the current yaw: x_c is the yaw heading projected to horizontal.
            x_c = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0], dtype=float)
            y_des = _cross3(z_des, x_c)
            y_n = float(np.linalg.norm(y_des))
            if y_n > 1e-6:
                y_des = y_des / y_n
                x_des = _cross3(y_des, z_des)
                R_des = np.column_stack([x_des, y_des, z_des]).astype(float)
        E = (R_des.T @ R_wb_hat) - (R_wb_hat.T @ R_des)
        e_R = 0.5 * _vee_so3(E)
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
                    try:
                        tau_om = float(max(0.0, float(getattr(self.cfg, "mpc_omega_lpf_tau", 0.0))))
                    except Exception:
                        tau_om = 0.0
                    if tau_om > 1e-9:
                        if not bool(self._mpc_omega_lpf_init):
                            self._mpc_omega_lpf = omega_mpc_b.copy()
                            self._mpc_omega_lpf_init = True
                        else:
                            a_om = float(_clipf(float(self.dt) / (float(tau_om) + float(self.dt)), 0.0, 1.0))
                            self._mpc_omega_lpf = (1.0 - a_om) * self._mpc_omega_lpf + a_om * omega_mpc_b
                        omega_mpc_b = np.asarray(self._mpc_omega_lpf, dtype=float).reshape(3).copy()
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
                # SRB stance: f_ref carries only vertical force (virtual spring, below).
                # Horizontal forces are solved by QP to satisfy Tau_des (attitude SO(3) PD).
                f_ref[:] = 0.0

            # f_ref[2] will be filled by virtual spring below; f_ref[0:2] = 0 (QP handles horizontal).
        else:
            f_ref[:] = 0.0

        # ===== Hopper4-style Virtual Spring (replaces unified stance & energy comp) =====
        # Instead of artificial COMP floors and complex energy tracking, we use a simple
        # robust virtual spring directly on the leg length error, matching Hopper4.py.
        energy_comp_fz = 0.0
        try:
            if bool(self._stance) and (not mpc_used):
                l0 = float(self.cfg.leg_l0_m)
                foot_b_now = np.asarray(foot_b, dtype=float).reshape(3)
                l_leg = float(np.linalg.norm(foot_b_now))
                
                k_spring = float(getattr(self.cfg, "stance_kp_z", 1000.0))
                b_spring = float(getattr(self.cfg, "stance_kd_z", 20.0))
                
                unitSpring_s = foot_b_now / max(1e-6, l_leg)
                xdot_s = np.asarray(foot_vrel_b, dtype=float).reshape(3)
                springVel_scalar = float(np.dot(xdot_s, unitSpring_s))
                
                # Anti-twitch: BOTH velocity consumers below (kd damping and the
                # energy term, dF/dv = Kp*m*v) get a heavily-filtered extension
                # velocity. Raw dq/dt velocity carries the 36 Hz stance vibration;
                # feeding it to kd_z=10 alone injected +-5 N of fz noise (03:43
                # log). The axial spring loop is only ~2.6 Hz, so 20 ms of lag
                # costs ~18 deg there while cutting 36 Hz ~4.6x. Seeded at
                # touchdown (init flag cleared in flight below).
                e_tau = float(getattr(self.cfg, "energy_vel_lpf_tau", 0.0))
                if e_tau > 0.0:
                    if not self._energy_vel_lpf_init:
                        self._energy_vel_lpf = float(springVel_scalar)
                        self._energy_vel_lpf_init = True
                    else:
                        a_e = float(self.dt) / (e_tau + float(self.dt))
                        self._energy_vel_lpf += a_e * (float(springVel_scalar) - self._energy_vel_lpf)
                    springVel_energy = float(self._energy_vel_lpf)
                else:
                    springVel_energy = float(springVel_scalar)

                springForce_scalar = -k_spring * (l_leg - l0) - b_spring * springVel_energy

                leg_velocity = float(qd_shift)
                if leg_velocity > 0.0 and bool(getattr(self.cfg, "use_energy_compensation", True)):
                    m = float(self.mass)
                    g = float(self.gravity)
                    # World +Z DOWN: foot below the body has foot_w[2] > 0, and the body
                    # height above the (assumed flat) ground is +foot_w[2].
                    foot_w_now = (R_wb_hat @ foot_b_now.reshape(3)).reshape(3)
                    groundHeight = float(foot_w_now[2])
                    
                    energy = (0.5 * m * springVel_energy * springVel_energy
                              + 0.5 * k_spring * (l0 - l_leg)**2
                              + m * g * groundHeight)
                    # Effective target = commanded height + apex-feedback boost
                    # (the P-only pump needs a persistent E_error to keep pushing;
                    # the integrator supplies it so the ACTUAL apex converges to
                    # hop_height_m instead of sitting ~4x low).
                    h = float(self.cfg.hop_height_m) + float(self._apex_fb_int)
                    target = m * g * (l0 + h)
                    E_error = target - energy
                    
                    Kp = float(self.cfg.energy_comp_kp)
                    energy_comp_fz = float(max(0.0, Kp * E_error))
                    springForce_scalar += energy_comp_fz
                
                if springForce_scalar < 0.0:
                    springForce_scalar = 0.0
                
                f_ref[2] = float(springForce_scalar)
                f_ref[2] = float(_clipf(float(f_ref[2]), float(self.cfg.stance_fz_min), float(self.cfg.stance_fz_max)))
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
            # Re-seed the energy velocity LPF at the next touchdown.
            self._energy_vel_lpf_init = False

        # Friction cone is enforced by QP constraints; no need to clip f_ref here.

        # ===== SO(3) attitude torque (SRB: direct PD on attitude error) =====
        # Both stance and flight use the same SO(3) PD structure:
        #   tau_b = -kR * e_R - kW * omega
        # In stance, leg fxy satisfies r_b × f_b = tau_b (body FRD, same frame as foot_b).
        # In flight, Tau_des is realized by propeller differential thrust.
        tau_b_stance_des = np.zeros(3, dtype=float)
        tau_b_att_des = np.zeros(3, dtype=float)

        # --- D-term gyro conditioning: biquad notch + light LPF, run EVERY
        # step (flight too) so touchdown sees a warm, transient-free filter.
        # Rationale/params: see stance_gyro_notch_hz in ModeEConfig.
        omega_raw = np.asarray(imu_gyro_b, dtype=float).reshape(3)
        omega_flt = omega_raw.copy()
        f0 = float(getattr(self.cfg, "stance_gyro_notch_hz", 0.0))
        if f0 > 0.0:
            key = (f0, float(self.cfg.stance_gyro_notch_bw_hz), float(self.dt))
            if self._gyro_notch_coefs is None or self._gyro_notch_coefs[0] != key:
                w0 = 2.0 * math.pi * f0 * float(self.dt)
                q_f = f0 / max(1e-6, float(self.cfg.stance_gyro_notch_bw_hz))
                alpha = math.sin(w0) / (2.0 * q_f)
                c = math.cos(w0)
                a0 = 1.0 + alpha
                # normalized: b = [1, -2c, 1]/a0 ; a = [-2c, 1-alpha]/a0
                self._gyro_notch_coefs = (
                    key,
                    1.0 / a0, -2.0 * c / a0, 1.0 / a0,
                    -2.0 * c / a0, (1.0 - alpha) / a0,
                )
            _, b0, b1, b2, a1, a2 = self._gyro_notch_coefs
            if not self._gyro_notch_init:
                # seed steady-state at current value (DC gain of notch = 1)
                for ax in range(2):
                    v = float(omega_raw[ax])
                    self._gyro_notch_x[ax, :] = (v, v, v, v)
                self._gyro_notch_init = True
            for ax in range(2):  # x, y only (z unused by the D-term)
                x1, x2, y1, y2 = self._gyro_notch_x[ax]
                x0 = float(omega_raw[ax])
                y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                self._gyro_notch_x[ax] = (x0, x1, y0, y1)
                omega_flt[ax] = y0
        g_tau = float(getattr(self.cfg, "stance_gyro_lpf_tau", 0.0))
        if g_tau > 0.0:
            if not self._stance_gyro_lpf_init:
                self._stance_gyro_lpf[:] = omega_flt
                self._stance_gyro_lpf_init = True
            else:
                a_g = float(self.dt) / (g_tau + float(self.dt))
                self._stance_gyro_lpf += a_g * (omega_flt - self._stance_gyro_lpf)
            omega_flt = self._stance_gyro_lpf.copy()

        if bool(self._stance):
            tau_rp_max = float(self.cfg.stance_tau_rp_max)
            omega_b = omega_raw
            omega_d = omega_flt
            kR_x = float(self.cfg.stance_kpp_x)
            kR_y = float(self.cfg.stance_kpp_y)
            kW_x = float(self.cfg.stance_kpd_x)
            kW_y = float(self.cfg.stance_kpd_y)
            tau_b_stance = np.zeros(3, dtype=float)
            tau_b_stance[0] = -kR_x * float(e_R[0]) - kW_x * float(omega_d[0])
            tau_b_stance[1] = -kR_y * float(e_R[1]) - kW_y * float(omega_d[1])
            tau_b_stance_des = tau_b_stance.copy()
            tau_b_att_des = tau_b_stance.copy()
            # DEBUG: kill stance attitude torque so QP produces NO horizontal contact force (fxfy=0).
            if bool(self._dbg_stance_zero_fxfy):
                tau_b_att_des[:] = 0.0
                tau_b_stance_des[:] = 0.0
        else:
            # Flight phase: separate roll/pitch gains (for propeller control).
            # NOTE: notch+LPF keep running above (no reset) so the D-term gyro
            # is already warm when the next touchdown arrives.
            omega_b = omega_raw
            if not bool(props_enabled_ctrl):
                # No propellers physically available: do not request flight attitude torques.
                tau_rp_max = 0.0
                tau_b_att_des[:] = 0.0
            else:
                tau_rp_max = float(self.cfg.flight_tau_rp_max)
                kR_roll = float(self.cfg.flight_kR_roll)
                kW_roll = float(self.cfg.flight_kW_roll)
                kR_pitch = float(self.cfg.flight_kR_pitch)
                kW_pitch = float(self.cfg.flight_kW_pitch)
                # Prop D-term uses RAW gyro (user request 2026-07-05): the
                # notch+LPF chain stays for the LEG stance D-term only.
                tau_b = np.zeros(3, dtype=float)
                tau_b[0] = (-float(kR_roll) * float(e_R[0])) - (float(kW_roll) * float(omega_b[0]))
                tau_b[1] = (-float(kR_pitch) * float(e_R[1])) - (float(kW_pitch) * float(omega_b[1]))
                tau_b[2] = 0.0
                tau_b_att_des = tau_b.copy()
        
        # Norm-based torque limiting before projection to the world-frame wrench.
        tau_rp_norm = float(np.sqrt(tau_b_att_des[0] ** 2 + tau_b_att_des[1] ** 2))
        if tau_rp_norm > tau_rp_max and tau_rp_max > 0.0 and tau_rp_norm > 1e-9:
            scale = float(tau_rp_max) / tau_rp_norm
            tau_b_att_des[0] = float(tau_b_att_des[0] * scale)
            tau_b_att_des[1] = float(tau_b_att_des[1] * scale)
        if bool(self._stance):
            tau_b_stance_des = tau_b_att_des.copy()
        tau_w = (R_wb_hat @ tau_b_att_des.reshape(3)).reshape(3)
        Tau_des = np.array([float(tau_w[0]), float(tau_w[1]), 0.0], dtype=float)
        # Propeller attitude demand uses the propeller/flight gains in both
        # stance and flight. Stance leg fxy can use separate leg gains above.
        tau_b_prop_des = np.zeros(3, dtype=float)
        if bool(props_enabled_ctrl):
            # RAW gyro for the prop D-term (user request); filtered gyro is
            # reserved for the leg stance D-term.
            tau_b_prop_des[0] = (
                -float(self.cfg.flight_kR_roll) * float(e_R[0])
                - float(self.cfg.flight_kW_roll) * float(omega_b[0])
            )
            tau_b_prop_des[1] = (
                -float(self.cfg.flight_kR_pitch) * float(e_R[1])
                - float(self.cfg.flight_kW_pitch) * float(omega_b[1])
            )
            prop_tau_max = float(self.cfg.flight_tau_rp_max)
            prop_norm = float(np.sqrt(tau_b_prop_des[0] ** 2 + tau_b_prop_des[1] ** 2))
            if prop_norm > prop_tau_max and prop_tau_max > 0.0 and prop_norm > 1e-9:
                prop_scale = prop_tau_max / prop_norm
                tau_b_prop_des[0] = float(tau_b_prop_des[0] * prop_scale)
                tau_b_prop_des[1] = float(tau_b_prop_des[1] * prop_scale)
        tau_prop_w = (R_wb_hat @ tau_b_prop_des.reshape(3)).reshape(3)
        Tau_prop_des = np.array([float(tau_prop_w[0]), float(tau_prop_w[1]), 0.0], dtype=float)
        Tau_des_dbg = Tau_des.copy()
        omega_b_used_dbg = omega_b.copy()

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
        s2s_active_dbg = 0

        xdot_for_pd = foot_vdot_vicon.copy()

        if not bool(self._stance):
            # Raibert in WORLD FRD (+Z down, 竖直向下), same vertical sense as body/FK footpos.
            #   target_xy_w = Kv * v_lo_xy_w + Kr * desired_v_xy_w
            #   target_z_w  = +sqrt(l0^2 - ||target_xy_w||^2)
            #   foot_des_b  = R_wb^T @ target_w   (quaternion body<-world)
            l0 = float(self.cfg.leg_l0_m)
            kv = float(self.cfg.flight_kv)
            kr = float(self.cfg.flight_kr)
            step_lim = float(abs(float(self.cfg.flight_stepper_lim_m)))
            target_xy_w = (
                kv * np.array([float(self._v_hat_w[0]), float(self._v_hat_w[1])], dtype=float)
                + kr * np.array([float(desired_v_xy_w[0]), float(desired_v_xy_w[1])], dtype=float)
            )
            if bool(self.cfg.mode_1d):
                target_xy_w[0] = 0.0
                target_xy_w[1] = 0.0
            normTarget = float(np.linalg.norm(target_xy_w))
            if (step_lim > 1e-9) and (normTarget > step_lim):
                target_xy_w = (target_xy_w * (step_lim / max(1e-12, normTarget))).astype(float)
                normTarget = float(np.linalg.norm(target_xy_w))
            target_z_w = float(np.sqrt(max(0.0, float(l0 * l0) - float(normTarget * normTarget))))
            foot_des_w = np.array([float(target_xy_w[0]), float(target_xy_w[1]), float(target_z_w)], dtype=float)
            foot_des_b = (np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T @ foot_des_w.reshape(3)).reshape(3)
            foot_des_native = np.asarray(foot_des_b, dtype=float).reshape(3).copy()
            s2s_active = True

            # Debug: foot_des_w is world FRD (+Z down); foot_des_b is body FRD for PD/print.
            foot_des_b_dbg = np.asarray(foot_des_b, dtype=float).reshape(3).copy()
            foot_des_w_dbg = np.asarray(foot_des_w, dtype=float).reshape(3).copy()
            p_foot_des_w_dbg = (
                np.asarray(self._p_hat_w, dtype=float).reshape(3)
                + (np.asarray(R_wb_hat, dtype=float).reshape(3, 3) @ foot_des_b_dbg.reshape(3)).reshape(3)
            )
            s2s_active_dbg = int(bool(s2s_active))

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
            sideForce = (Khp * (targetFootPos - x) - Khd * (xdot - _cross3(omega_b_native, x))).astype(float)
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

        # ===== Force allocation (closed-form leg + lstsq prop, no WBC-QP) =====
        # control_mode: 1=pure_leg, 2/3=decouple (lstsq prop overlay)
        _cmode = int(self.cfg.control_mode)
        thrust_sum_max = float(self.mass * self.gravity * float(self.cfg.thrust_total_ratio_max))
        props_on = bool(props_enabled_ctrl) and (_cmode != 1)
        if not props_on:
            thrust_sum_max = 0.0

        if bool(self._stance):
            # --- Stance: closed-form leg (fz + fxy) + optional prop overlay ---
            F_des = np.asarray(f_ref, dtype=float).reshape(3).copy()
            fz_cmd = float(max(0.0, float(f_ref[2])))
            try:
                r_foot_b = (foot_b - self.com_b).reshape(3)
                rx = float(r_foot_b[0])
                ry = float(r_foot_b[1])
                rz = float(r_foot_b[2])
                # Delivered body torque from the commanded contact force (robot
                # pushes the ground with f_cmd, body feels the reaction):
                #   tau_x = rz*fy - ry*fz
                #   tau_y = rx*fz - rz*fx
                # The old code assumed r = (0,0,rz) and DROPPED the ry*fz / rx*fz
                # lever terms. In the 04:19 log the foot sat 5-9 cm off-center in
                # x while fz reached 110-130 N -> a 5-10 Nm tipping moment the
                # attitude PD (peak ~6 Nm at kR=20) had to fight blind. Solving
                # with the full lever arm turns that known moment into a
                # feedforward instead of an attitude disturbance:
                #   fy = (tau_x + ry*fz)/rz ,  fx = (rx*fz - tau_y)/rz
                # (reduces exactly to the old solution when rx=ry=0).
                tau_att_xy = np.asarray(tau_b_att_des, dtype=float).reshape(3)[:2]
                if abs(rz) > 1e-6:
                    fxy_b = (
                        (rx * fz_cmd - float(tau_att_xy[1])) / rz,
                        (float(tau_att_xy[0]) + ry * fz_cmd) / rz,
                    )
                else:
                    fxy_b = (0.0, 0.0)
                # Friction cone + absolute cap on stance fxy (2026-07-05).
                # The old "QP enforces the cone" comment is stale -- this is the
                # closed-form path and fxy was UNLIMITED except via tau_rp_max
                # (20 Nm -> up to ~44 N horizontal). At touchdown fz is still
                # small, so a big attitude-driven fxy instantly breaks the foot
                # loose ("slips right at TD"). Proportional scaling keeps the
                # torque direction:  |fxy| <= mu*fz  and  |fxy| <= stance_fxy_max.
                fxy_norm = float(np.hypot(float(fxy_b[0]), float(fxy_b[1])))
                mu_s = float(getattr(self.cfg, "stance_mu", 0.0))
                fxy_cap = float(getattr(self.cfg, "stance_fxy_max", 0.0))
                lim = float("inf")
                if mu_s > 0.0:
                    lim = min(lim, mu_s * max(0.0, fz_cmd))
                if fxy_cap > 0.0:
                    lim = min(lim, fxy_cap)
                if np.isfinite(lim) and fxy_norm > lim and fxy_norm > 1e-9:
                    s_fxy = lim / fxy_norm
                    fxy_b = (float(fxy_b[0]) * s_fxy, float(fxy_b[1]) * s_fxy)
                f_contact_b_cmd = np.array([float(fxy_b[0]), float(fxy_b[1]), fz_cmd], dtype=float)
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
            # --- Flight: swing-leg tau_ref + optional prop overlay (same lstsq as stance) ---
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
        if bool(self._stance) and float(scale) < 1.0:
            f_contact_w = (np.asarray(f_contact_w, dtype=float).reshape(3) * float(scale)).astype(float)

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
            "flight_vel_b": np.asarray(self._flight_vel_b, dtype=float).reshape(3).copy(),
            # Debug: base velocity measured from leg kinematics (foot assumed stationary in WORLD).
            "v_meas_foot_w": np.asarray(v_base_from_foot_w, dtype=float).reshape(3).copy(),
            # Debug: running PUSH-phase average (this is what the liftoff latch uses).
            "v_push_avg_w": (
                np.asarray(self._push_v_sum_w, dtype=float).reshape(3) / float(self._push_v_cnt)
                if int(self._push_v_cnt) > 0 else np.zeros(3, dtype=float)
            ),
            "push_v_cnt": int(self._push_v_cnt),
            # Liftoff latch source: 2=push linear fit, 1=push mean, 0=instantaneous.
            "lo_latch_src": int(self._lo_latch_src),
            # Velocity-KF accel bias estimate (body frame) -- should settle to a
            # small near-constant value; a drifting/saturated bias means bad tuning.
            "kf_b_a": np.asarray(self._kf_b_a, dtype=float).reshape(3).copy(),
            # Foot kinematics:
            # - foot_vicon: delta/vicon frame (+Z DOWN)
            # - foot_b:     body frame (FRD, +Z DOWN)
            "foot_vicon": foot_vicon.copy(),
            "foot_b": foot_b.copy(),
            "foot_vdot_vicon": foot_vdot_vicon.copy(),
            "foot_vrel_b": foot_vrel_b.copy(),
            # Filtered joint velocity actually used by kinematics/estimator
            # (q-diff -> EMA(qd_ema_forgetting) -> MA(qd_ma_window)).
            # Compare against raw qd0..qd2 in the CSV to see the filter effect.
            "qd_kin": np.asarray(joint_vel_kin, dtype=float).reshape(3).copy(),
            "J_inv_det": float(J_inv_det),
            "J_inv_cond": float(J_inv_cond),
            "A_tau_f_det": float(A_tau_f_det),
            "A_tau_f_cond": float(A_tau_f_cond),
            # Flight S2S/swing target (for debugging). In stance these will be NaNs.
            "s2s_active": int(s2s_active_dbg),
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
            # Debug: SO(3) attitude error and BODY-frame attitude torque before world projection.
            "e_R": e_R.copy(),
            "tau_b_stance_des": tau_b_stance_des.copy(),
            # Debug: gyro actually used by the stance attitude torque controller (BODY frame)
            "omega_b_used": omega_b_used_dbg.copy(),
            # Apex height feedback (for debugging/convergence analysis)
            "z_lo_m": float(self._z_lo) if self._z_lo is not None else float("nan"),
            "vz_lo_m_s": float(self._vz_lo) if self._vz_lo is not None else float("nan"),
            "v_to_cmd_m_s": float(self._v_to_cmd),
            "hop_height_m": float(self.cfg.hop_height_m),
            # Apex feedback loop state: last measured apex height above LO,
            # and the current integrator boost added to the energy target.
            "z_apex_actual_m": float(self._z_apex_actual),
            "apex_err_int": float(self._apex_fb_int),
            # Falling cat debug (recovery gating)
            # MPC debug
            "mpc_status": mpc_status,
            "mpc_u0": mpc_u0.copy(),
        }

        return tau_cmd, pwm_us, info


