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
    # PRIMARY TUNING
    # ======================================================================

    # Mode1: world-z impedance + push energy compensation, stance attitude PD
    # with closed-form leg force allocation, propeller residual attitude
    # overlay, classic Raibert placement, and flight propeller attitude PD.
    dt: float = 0.002
    mode_1d: bool = False

    # ============ STANCE Z: FORCE-BUDGETED SLIP (FB-SLIP, 2026-07-21) ======
    # ONE task-level knob: hop_height_m. Everything else is derived from the
    # ballistics of the CURRENT hop plus fixed ROBOT constants (mass, leg
    # force budget, leg stroke). No per-height gain retuning, no memory of
    # previous hops.
    #
    #   Ballistic pair:  v_to = sqrt(2*g_up*h),   v_td = measured at TD.
    #
    #   CONSTANT-FORCE stance (2026-07-21 v2). v1 realized compression as a
    #   TD-sized linear spring; the budget-consistent stiffness came out at
    #   8000-11000 N/m and rang the leg mode through the sensing/actuation
    #   delay (02:45 log: vz swung -3.0 -> +2.8 m/s in 6 ms, the vibration
    #   spike latched PUSH at x_z ~ 20 mm and the remaining stroke could
    #   not reach v_to -> hops decayed). Any position-spring realization
    #   inherits that trade-off, so both sub-phases are now PURE FORCE
    #   schedules -- zero position gain, zero velocity gain, nothing to
    #   ring -- and constant force is also the MINIMUM-PEAK-FORCE solution
    #   for a given stroke (half the peak of a linear spring).
    #
    #   COMPRESSION (sized once at touchdown, from the measured v_td):
    #     Brake at the design force F_b = stance_brake_force_g * m * g_st.
    #     Constant deceleration stops the body in
    #       x_c = m*v_td^2 / (2*(F_b - m*g_st))
    #     Higher hop -> larger v_td -> DEEPER compression at the SAME gentle
    #     force; lower hop -> shallow. If x_c would exceed the stroke, the
    #     force is raised to m*g_st + m*v_td^2/(2*stroke) and capped at
    #     F_max = leg_force_budget_g * m * g. The force ramps in over
    #     stance_brake_ramp_s (no impact step at touchdown).
    #
    #   PUSH (world-Z gated, latched): constant force sized so the work
    #     over the remaining stroke x_up (current height deficit to l0)
    #     delivers the takeoff energy:
    #       F_push = m*g_st + m*v_to^2 / (2*x_up),   F_push <= F_max
    #     A current-hop velocity floor keeps pushing while vz_up < v_to;
    #     no previous-hop apex or loss estimate enters the command.
    #
    #   FLIGHT (ballistic touchdown guard): after liftoff at measured vz_lo
    #     the body physically cannot return to the ground before
    #     T_return = 2*vz_lo/g. Touchdown is accepted only after
    #     flight_td_guard_kappa * T_return AND while the body is descending,
    #     so flight-phase leg retraction/vibration can no longer enter
    #     stance mid-air.
    use_energy_compensation: bool = True
    hop_height_m: float = 0.07
    # ---- Robot constants for FB-SLIP (not per-height tuning knobs) ----
    # Hard peak-force budget as a multiple of body weight (F_max = beta*m*g).
    leg_force_budget_g: float = 5.0
    # Usable vertical leg stroke for compression (m).
    leg_stroke_max_m: float = 0.06
    # Design braking force as a multiple of stance weight (alpha*m*g_st).
    # Must be > 1 (net upward force while braking). Sets how deep a given
    # landing speed compresses: x_c = v_td^2 / (2*g_st*(alpha-1)).
    stance_brake_force_g: float = 3.0
    # Brake force rise time after touchdown (s); avoids a force step at TD.
    stance_brake_ramp_s: float = 0.015
    # Ballistic TD guard: accept TD after kappa*T_return (T_return=2*vz_lo/g).
    # 03:52 log: kappa=0.5 opens the window exactly at the apex, and the
    # kinematic vz_lo under-reads near full leg extension (logged 0.74 vs
    # 1.15 true from the 232 ms flight time), so a leg-retraction crossing
    # at 78 ms was accepted as touchdown -> 26 ms fake stance -> 275 N cap
    # kick -> tumble. The body cannot physically return before T_return
    # (descent brake only makes it later), so a large kappa is safe on
    # flat ground; 0.75 gives margin against the vz_lo under-estimate.
    flight_td_guard_kappa: float = 0.75
    # Accept TD only while descending: vz_up <= this gate (m/s).
    flight_td_descend_vz_mps: float = 0.05
    # PUSH gate on WORLD-Z upward velocity (m/s) + consecutive-tick confirm.
    stance_push_vz_mps: float = 0.02
    stance_push_confirm_steps: int = 2
    # First-order blend of the push spring force at the latch (s).
    stance_push_blend_tau_s: float = 0.01
    # Sensor LPF on the world-Z velocity used by the PUSH gate only (leg
    # vibration puts +-2 m/s spikes on the kinematic vz estimate).
    stance_vz_lpf_tau_s: float = 0.008
    # SENSOR floor on the earliest PUSH latch (s). The real gate is
    # PHYSICAL, computed per hop at TD from constant-deceleration
    # kinematics (a = v_td^2 / (2*x_c)  =>  stop time t = v_td/a):
    #   t_bottom = t_ramp/2 + 2*x_c/v_td
    # (first term: deceleration not yet built up during the force ramp).
    # PUSH is blocked before stance_push_bottom_eta * t_bottom, so impact
    # ringing right after TD can no longer latch PUSH ahead of the true
    # bottom (03:29 log ST2: +2.9 m/s spike latched at 36 ms while the
    # bottom arrived at ~60 ms).
    stance_push_min_stance_s: float = 0.01
    # Safety factor (<1) on the predicted time-to-bottom: v_td is a noisy
    # kinematic measurement and a genuinely early bottom must not be missed.
    stance_push_bottom_eta: float = 0.8
    # Current-hop takeoff-speed feedback. During PUSH:
    #   F_catch = m*g_st + kp_v*max(0, v_to-vz_up)
    # and the commanded force is max(push spring, F_catch). Units N/(m/s).
    stance_push_vz_kp: float = 100.0

    # Leg length, contact phases, and vertical stance force.
    leg_l0_m: float = 0.461
    # Fallback world-height impedance (used only if the TD-sized FB-SLIP
    # spring is unavailable, e.g. controller enabled mid-stance).
    stance_kp_z: float = 1400.0
    stance_kd_z: float = 6.0
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
    hopper4_phase_min_steps: int = 10

    # Stance attitude PD (one gain for roll and pitch).
    stance_kpp: float = 70.0
    stance_kpd: float = 1.5
    stance_tau_rp_max: float = 15.0
    stance_mu: float = 0.0
    stance_fxy_max: float = 0.0

    # MATLAB/SLX EMA applied to the CAN-reported qd.
    # lambda is the weight of the previous estimate; 0 leaves raw CAN qd.
    qd_ema_lambda: float = 0.4

    # Flight placement (classic Raibert) / swing PD.
    flight_kv: float = 0.16
    flight_kr: float = 0.09
    flight_stepper_lim_m: float = 0.13
    swing_kp_xy: float = 60.0
    swing_kd_xy: float = 1.0
    swing_kp_z: float = 1300.0
    swing_kd_z: float = 20.0

    # Propeller/HFA. Flight attitude PD uses one gain for roll and pitch.
    flight_kR: float = 40.0
    flight_kW: float = 6.0
    flight_tau_rp_max: float = 100
    prop_base_thrust_ratio: float = 0.01
    # Stance-phase propeller idle (collective). Higher than flight so props
    # unload the leg / raise effective hop energy during push. Flight still
    # uses prop_base_thrust_ratio. Typical bring-up: 0.08–0.15.
    prop_stance_base_thrust_ratio: float = 0.12
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
    prop_flight_brake_ratio: float = 0.10
    # Descent detection threshold on world vz (m/s, up-positive).
    prop_flight_brake_vz_mps: float = 0.10
    # FLIGHT world-Z prop force budget (fraction of m*g), dual to the leg's
    # F_max = beta*m*g. The attitude allocator's forward-floor collective
    # lift silently raises the TOTAL thrust far above the commanded
    # baseline (03:36 log: baseline 0.55-5.5 N commanded, 4-19.5 N
    # delivered = up to 0.35*m*g), so the effective descent gravity
    # wandered between 0.65g and g. That unmodeled vertical force breaks
    # exactly the ballistics FB-SLIP plans on (v_td, apex, TD guard) and
    # produced the weak-hop chatter. In FLIGHT the allocation sum cap is
    #   sum(thrusts) <= flight_thrust_sum_max_ratio * m * g,
    # the attitude differential is scaled direction-preserving to fit, so
    # props remain a BOUNDED gravity modulation:
    #   g_dn >= g*(1 - flight_thrust_sum_max_ratio).
    flight_thrust_sum_max_ratio: float = 0.20
    # 2026-07-19 (per user): 3D / bidirectional thrust DISABLED everywhere.
    # prop_bidir=False makes negative thrust idle at pwm_min in the PWM map,
    # forces forward-only floors in the stance overlays / daisy chain, and
    # disables the stance downforce experiment. "auto" cannot reverse with
    # bidir off, so flight is forward-only too.
    prop_flight_reverse: str = "auto"
    prop_bidir: bool = False
    # Total Z budget ~0.6*m*g ≈ 33 N (m=5.61). Per-arm 10 N ≈ pwm 1900 with
    # calibrated k=1.24e-5 (was capped at 4.5 N / 1600 us).
    thrust_total_ratio_max: float = 0.60
    thrust_max_each_n: float = 10.0
    pwm_min_us: float = 1000.0
    pwm_max_us: float = 2000.0
    prop_k_thrust: float = 1.24e-5

    # State estimation.
    use_fc_quat: bool = False
    att_accel_weight: float = -0.01
    att_stance_bound_lo: int = 90
    att_stance_bound_hi: int = 130
    vel_push_tail_n: int = 20

    # Motor command limits.
    tau_cmd_max_nm: tuple[float, float, float] = (40.0, 40.0, 40.0)
    tau_cmd_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # ======================================================================
    # ROBOT MODEL AND ADVANCED OPTIONS
    # ======================================================================

    mode_1d_disable_mpc: bool = True

    # Physical model (also affects energy and force allocation).
    # 2026-07-19 restored to the measured 5.61 kg (per user); sizes both
    # stance springs (k ~ m) and the propeller collective.
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
    #   arm 0 at -90 deg (0,-L)          -> physical Motor2/PWM[2]/MAIN2
    #   arm 1 at +150 deg (-x,+y)        -> physical Motor3/PWM[3]/MAIN3
    #   arm 2 at +30 deg  (+x,+y)        -> physical Motor1/PWM[1]/MAIN8
    # 2026-07-18 confirmed physical layout (M1 moved MAIN1 -> MAIN8):
    #   M1 -> MAIN8 at +30 deg, M2 -> MAIN2 at -90 deg, M3 -> MAIN3 at +150 deg.
    prop_pwm_idx_per_arm: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (2,),  # arm 0 (-Y)    = Motor2
        (3,),  # arm 1 (-x,+y) = Motor3
        (1,),  # arm 2 (+x,+y) = Motor1
    )

    # Minimum forward thrust per arm keeps the propellers responsive.
    wbc_thrust_min_each_n: float = 0.1

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
        # Flight-phase XY velocity latched at liftoff (push tail mean).
        self._flight_vel = np.zeros(3, dtype=float)
        # Per-stance push-phase ring buffer (last N samples for XY velocity).
        _tail_n = int(max(1, int(getattr(self.cfg, "vel_push_tail_n", 10))))
        self._vel_push_tail_n = _tail_n
        self._push_vel_ring = np.zeros((_tail_n, 3), dtype=float)
        self._push_vel_ring_i: int = 0
        self._push_vel_ring_cnt: int = 0
        # Mode1 push-spring state (reset at every touchdown).
        self._mode1_push_latched: bool = False
        self._mode1_push_confirm_count: int = 0
        self._mode1_f_brake: float | None = None
        self._mode1_x_c_plan: float = 0.0
        self._mode1_t_bottom: float = 0.0
        self._mode1_k_boost: float = 0.0
        self._mode1_v_td: float = 0.0
        self._mode1_x0: float = 0.0
        self._mode1_boost_f_state: float = 0.0
        # LPF'd world-Z up-velocity used only inside the stance-Z law.
        self._mode1_vz_lpf: float | None = None
        # "auto" reverse-policy hysteresis latch (see _allocate_prop_thrust).
        self._prop_rev_on: bool = False
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
        self._flight_vel[:] = 0.0
        self._push_vel_ring[:] = 0.0
        self._push_vel_ring_i = 0
        self._push_vel_ring_cnt = 0
        self._mode1_push_latched = False
        self._mode1_push_confirm_count = 0
        self._mode1_f_brake = None
        self._mode1_x_c_plan = 0.0
        self._mode1_t_bottom = 0.0
        self._mode1_k_boost = 0.0
        self._mode1_v_td = 0.0
        self._mode1_x0 = 0.0
        self._mode1_boost_f_state = 0.0
        self._mode1_vz_lpf = None
        self._prop_rev_on = False
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
        thrust_sum_max = thrust_total_ratio_max * m * g.

        reverse_policy (prioritized a.k.a. daisy-chain allocation, cf.
        Johansen & Fossen 2013):
          "fwd"   forward-only floor (never cross the 1000 us stop).
          "bidir" reverse floor always available (stance downforce path).
          "auto"  solve forward-only FIRST; open the reverse floor only when
                  that solution cannot realize the demanded torque (s < 1).
                  The Case-1 math then reverses the low arm only by the
                  amount actually needed. Hysteresis: once engaged, reverse
                  stays available until the bidir solution naturally returns
                  to the forward side (all arms >= forward floor), so the
                  low arm does not flip back and forth across 1000 us at the
                  engagement boundary. This replaces the angle-threshold
                  gate: the criterion is actuator FEASIBILITY, not |e_R|.
        """
        tau_des_w = np.asarray(tau_des_w, dtype=float).reshape(3)
        prop_r_w = np.asarray(prop_r_w, dtype=float).reshape(3, 3)
        z_thrust_w = np.asarray(z_thrust_w, dtype=float).reshape(3)
        M_prop = np.column_stack([
            _cross3(prop_r_w[i].reshape(3), z_thrust_w.reshape(3)) for i in range(3)
        ]).astype(float)
        thrusts_att = _lstsq_minnorm(M_prop[:2, :], tau_des_w[:2]).astype(float)
        base_each = float(thrust_sum_ref) / 3.0
        t_max = float(self.cfg.thrust_max_each_n)
        tsum_cap = float(max(0.0, float(thrust_sum_max)))

        a = [float(thrusts_att[i]) for i in range(3)]
        a_min = min(a)
        a_max = max(a)
        a_sum = a[0] + a[1] + a[2]

        def _solve(t_min: float) -> tuple[np.ndarray, float]:
            # Largest s in [0,1] such that thrusts = base + c(s) + s*att is
            # feasible, where c(s) = max(0, t_min - (base + s*a_min)) is the
            # collective lift. Case 1: no lift needed (base + s*a_min >= t_min).
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
            return (
                np.array(
                    [base_each + c + s * a[0], base_each + c + s * a[1], base_each + c + s * a[2]],
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
            thrusts, _ = _solve(t_min_fwd if bidir_ok else t_min_rev)
        elif pol == "bidir":
            thrusts, _ = _solve(t_min_rev)
        else:  # "auto"
            thr_f, s_f = _solve(t_min_fwd)
            if (not self._prop_rev_on) and s_f < 1.0 - 1e-6:
                self._prop_rev_on = True
            if self._prop_rev_on:
                thr_b, _ = _solve(t_min_rev)
                if float(np.min(thr_b)) >= t_min_fwd - 1e-9:
                    # Demand shrank: bidir solution is already all-forward,
                    # identical to the fwd one -> disengage with no jump.
                    self._prop_rev_on = False
                    thrusts = thr_f
                else:
                    thrusts = thr_b
            else:
                thrusts = thr_f
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
        (MATLAB avg_foot_vel/pos rolling window; used only for flight_vel)."""
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
        # v_base_w = R_wb @ ( -foot_vdot_b - omega_b x foot_b )
        # Full-coefficient omega×r restored (2026-07-19): with the foot pinned,
        # body rotation moves the leg relative to the body; without this term
        # that rotation is misread as horizontal CoM velocity (false Raibert
        # swing when hopping in place). MATLAB's 0.1 factor undercompensated.
        v_base_from_foot_w = (R_wb_hat @ (
            -foot_vrel_b.reshape(3)
            - _cross3(imu_gyro_b.reshape(3), foot_b.reshape(3))
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
            # ---- FB-SLIP ballistic touchdown guard ----
            # After liftoff at measured vz_lo the body cannot physically be
            # back at the ground before T_return = 2*vz_lo/g. Block TD for
            # kappa*T_return, and additionally require the body to be
            # DESCENDING: mid-flight leg retraction/vibration crosses the
            # length threshold while ascending and used to fake a stance.
            t_td_guard = float(phase_min_t)
            if (
                self._lo_t is not None
                and self._vz_lo is not None
                and np.isfinite(float(self._vz_lo))
                and float(self._vz_lo) > 0.05
            ):
                kappa_td = float(_clipf(
                    float(getattr(self.cfg, "flight_td_guard_kappa", 0.5)),
                    0.0, 1.0,
                ))
                t_return = 2.0 * float(self._vz_lo) / float(
                    max(1e-3, float(self.gravity))
                )
                t_td_guard = float(_clipf(
                    kappa_td * t_return, float(phase_min_t), 0.5
                ))
            vz_up_now = -float(self._v_hat_w[2])
            descending_ok = (not np.isfinite(vz_up_now)) or (
                vz_up_now <= float(getattr(
                    self.cfg, "flight_td_descend_vz_mps", 0.05
                ))
            )
            # Safety: the descend gate uses the IMU-integrated flight
            # velocity; if that estimate misbehaves it must never block a
            # real landing forever. Past twice the guard window (>=0.6 s)
            # fall back to the plain length threshold.
            if t_in_flight >= max(0.6, 2.0 * t_td_guard):
                descending_ok = True
            cond_td = (
                (float(q_shift) <= -td_thr)
                and (t_in_flight >= t_td_guard)
                and bool(descending_ok)
            )

            if bool(cond_td):
                touchdown_evt = True
                self._stance = True
                self._td_t = float(self.sim_time)
                # ---- Flight-time apex measurement ----
                # Asymmetric ballistic arc (TA-SLIP): ascent gravity g_up and
                # descent gravity g_dn differ when the aerial brake is on.
                # T = sqrt(2h/g_up) + sqrt(2h/g_dn)
                #   => h = T^2 / (2*(1/sqrt(g_up) + 1/sqrt(g_dn))^2)
                # (reduces to g*T^2/8 when g_up == g_dn).
                # Window >= 0.12 s rejects short chatter "flights". This apex
                # estimate is diagnostic only; it does not affect later hops.
                if self._lo_t is not None:
                    T_fl = float(self.sim_time) - float(self._lo_t)
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
                # Re-seed the Mode-1 push-spring state for this stance.
                self._mode1_push_latched = False
                self._mode1_push_confirm_count = 0
                self._mode1_f_brake = None
                self._mode1_x_c_plan = 0.0
                self._mode1_t_bottom = 0.0
                self._mode1_k_boost = 0.0
                self._mode1_v_td = 0.0
                self._mode1_x0 = 0.0
                self._mode1_boost_f_state = 0.0
                self._mode1_vz_lpf = None
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

                # ---- FB-SLIP: size the constant BRAKE force ONCE at TD ----
                # Constant-deceleration braking at the design force
                # F_b = alpha*m*g_st stops the measured v_td in
                #   x_c = m*v_td^2 / (2*(F_b - m*g_st)).
                # If x_c exceeds the stroke, raise the force to
                # m*g_st + m*v_td^2/(2*stroke) and cap at F_max. Pure
                # feedforward: no position/velocity gain, nothing to ring.
                try:
                    m_td = float(self.mass)
                    g_raw = float(self.gravity)
                    rho_st_td = (
                        float(_clipf(
                            float(self.cfg.prop_stance_base_thrust_ratio),
                            0.0, 0.8,
                        ))
                        if (
                            bool(self._props_armed_rt)
                            and bool(self.cfg.stance_use_props)
                        )
                        else 0.0
                    )
                    g_st_td = g_raw * (1.0 - rho_st_td)
                    v_dn = float(v_base_from_foot_w[2])
                    if not np.isfinite(v_dn):
                        v_dn = float(self._v_hat_w[2])
                    v_td_meas = float(max(0.0, v_dn))
                    # BALLISTIC floor on v_td from the CURRENT flight time
                    # (03:59 log: kinematic read 0.34 m/s on a 220 ms
                    # flight whose ballistics say 1.05; the low v_td
                    # collapsed t_bottom to 27 ms, PUSH latched at 28 ms
                    # while still descending, 247 N fired before the
                    # bottom -> bounce chatter). Symmetric-arc kinematics
                    #   v_td = g*T_flight/2
                    # uses only the phase-machine timestamps, no velocity
                    # estimator. With the descent brake it slightly
                    # OVER-estimates v_td, which only opens the PUSH gate
                    # later (conservative); the constant F_brake does not
                    # depend on v_td at all.
                    if self._lo_t is not None:
                        T_fl_td = float(self.sim_time) - float(self._lo_t)
                        if 0.1 <= T_fl_td <= 1.0:
                            v_td_bal = float(_clipf(
                                0.5 * g_raw * T_fl_td, 0.0, 3.0
                            ))
                            v_td_meas = float(max(v_td_meas, v_td_bal))
                    self._mode1_v_td = v_td_meas
                    f_max_td = float(max(
                        2.0 * m_td * g_raw,
                        float(self.cfg.leg_force_budget_g) * m_td * g_raw,
                    ))
                    alpha_b = float(max(
                        1.2, float(self.cfg.stance_brake_force_g)
                    ))
                    f_b = alpha_b * m_td * g_st_td
                    x_c = (
                        m_td * v_td_meas * v_td_meas
                        / max(1e-6, 2.0 * (f_b - m_td * g_st_td))
                    )
                    stroke = float(max(
                        0.015, float(self.cfg.leg_stroke_max_m)
                    ))
                    if x_c > stroke:
                        x_c = stroke
                        f_b = (
                            m_td * g_st_td
                            + m_td * v_td_meas * v_td_meas
                            / (2.0 * stroke)
                        )
                    self._mode1_x_c_plan = float(x_c)
                    self._mode1_f_brake = float(min(f_b, f_max_td))
                    # Predicted time-to-bottom (constant-deceleration
                    # kinematics + half the force ramp, during which the
                    # deceleration is not yet built up):
                    #   a = v_td^2/(2*x_c)  =>  t_stop = v_td/a = 2*x_c/v_td
                    #   t_bottom = t_ramp/2 + 2*x_c/v_td
                    # Consumed by the PUSH gate: no latch before
                    # stance_push_bottom_eta * t_bottom.
                    t_ramp_td = float(max(
                        1e-3, float(self.cfg.stance_brake_ramp_s)
                    ))
                    self._mode1_t_bottom = float(
                        0.5 * t_ramp_td
                        + 2.0 * x_c / max(0.1, v_td_meas)
                    )
                except Exception:
                    self._mode1_f_brake = None
                    self._mode1_t_bottom = 0.0

        # ===== Hopper4-style liftoff =====
        if bool(self._stance) and np.isfinite(float(q_shift)):
            td_t = float(self._td_t) if self._td_t is not None else float(self.sim_time)
            t_in_stance = float(self.sim_time) - td_t
            # (2026-07-10: the no-prop omega liftoff gate and the forced-liftoff
            # escapes (q_shift 3 cm / stance timeout) were DELETED per user --
            # liftoff is the plain leg-extension threshold + min phase time.)
            cond_lo = (float(q_shift) >= lo_thr) and (
                t_in_stance >= phase_min_t
            )
            if bool(cond_lo):
                liftoff_evt = True
                self._stance = False
                self._lo_t = float(self.sim_time)
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
                # Liftoff state for apex detection / logs (up-positive vz).
                self._z_lo = float(self._p_hat_w[2])
                self._vz_lo = float(-self._v_hat_w[2])
                self._prev_vz = None
                self._apex_reached = False

        # ===== Body velocity =====
        # XY: planted-foot odometry in stance, last-N mean latched at liftoff.
        # Z: stance leg odometry, flight IMU integration.
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
                i = int(self._push_vel_ring_i)
                self._push_vel_ring[i, :] = v_base_from_foot_w.reshape(3).astype(float)
                self._push_vel_ring_i = (i + 1) % int(self._vel_push_tail_n)
                self._push_vel_ring_cnt = min(
                    int(self._push_vel_ring_cnt) + 1, int(self._vel_push_tail_n)
                )
            else:
                self._v_hat_w[2] = float(vz_pred)
        else:
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
        # Collective idle: stance uses a higher baseline so props assist the
        # hop; flight ascent keeps the small prop_base_thrust_ratio; flight
        # DESCENT uses the aerial-brake ratio (TA-SLIP: lower v_td -> softer
        # landing spring, see hybrid config block).
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
        else:
            prop_ratio = float(self.cfg.prop_base_thrust_ratio)
        thrust_sum_ref = float(self.mass * self.gravity * float(prop_ratio))
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
                # Peak-force budget: the FB-SLIP F_max = beta*m*g, further
                # limited by the absolute hardware cap stance_fz_max.
                f_max_z = float(max(
                    2.5 * float(self.mass) * float(self.gravity),
                    float(self.cfg.leg_force_budget_g)
                    * float(self.mass) * float(self.gravity),
                ))
                fz_cap = float(min(float(self.cfg.stance_fz_max), f_max_z))

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

                # COMPRESSION (FB-SLIP v2): CONSTANT brake force sized
                # once at TD (see the TD block), ramped in over
                # stance_brake_ramp_s so there is no force step at
                # impact. Pure feedforward -- no position or velocity
                # gain, so the sensing/actuation delay has nothing to
                # ring (the TD-sized linear spring at 8000-11000 N/m
                # did, 02:45 log).
                td_t_z = (
                    float(self._td_t)
                    if self._td_t is not None
                    else float(self.sim_time)
                )
                t_in_st_z = float(self.sim_time) - td_t_z
                if self._mode1_f_brake is not None:
                    t_ramp = float(max(
                        1e-3, float(self.cfg.stance_brake_ramp_s)
                    ))
                    ramp = float(_clipf(t_in_st_z / t_ramp, 0.0, 1.0))
                    f_comp = ramp * float(self._mode1_f_brake)
                else:
                    # Fallback fixed impedance (only if the controller was
                    # enabled mid-stance and no TD sizing exists).
                    kz = float(self.cfg.stance_kp_z)
                    h_des = l0 + float(self.cfg.hop_height_m)
                    f_comp = kz * (h_des - h_com) - bz * vz_up
                f_comp = float(min(float(f_comp), f_max_z))

                # PUSH latch on the FILTERED world-Z velocity: the body
                # has passed the bottom when vz turns positive.
                # Debounced, and PHYSICALLY gated: constant-deceleration
                # braking predicts the bottom at
                #   t_bottom = t_ramp/2 + 2*x_c/v_td
                # so no latch is accepted before eta_b*t_bottom -- impact
                # ringing right after TD cannot fire PUSH ahead of the
                # true bottom. stance_push_min_stance_s remains only as a
                # small sensor floor.
                eta_b = float(_clipf(
                    float(getattr(
                        self.cfg, "stance_push_bottom_eta", 0.8
                    )),
                    0.0, 1.0,
                ))
                t_push_gate = float(max(
                    float(self.cfg.stance_push_min_stance_s),
                    eta_b * float(self._mode1_t_bottom),
                ))
                latch_evt = False
                if not bool(self._mode1_push_latched):
                    if (
                        vz_f > float(self.cfg.stance_push_vz_mps)
                        and t_in_st_z >= t_push_gate
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
                        latch_evt = True

                if latch_evt:
                    # CONSTANT push force (FB-SLIP v2): the work over the
                    # remaining stroke x0 (current height deficit to l0)
                    # delivers the takeoff energy for hop_height_m:
                    #   F_push*x0 = 0.5*m*v_to^2 + m*g_st*x0
                    #   => F_push = m*g_st + m*v_to^2/(2*x0)
                    # with v_to = sqrt(2*g_up*h): the ascent collective
                    # keeps pushing after liftoff, so the leg only has
                    # to supply the (1 - rho_up) share of the apex
                    # energy. Capped at the F_max budget. Feedforward
                    # only -- no gain on position/velocity, no ringing.
                    x0 = float(max(0.01, x_z))
                    v_to = float(np.sqrt(
                        2.0 * g_up
                        * float(max(0.0, float(self.cfg.hop_height_m)))
                    ))
                    f_push = (
                        m * g_st
                        + m * v_to * v_to / (2.0 * x0)
                    )
                    # _mode1_k_boost now stores the CONSTANT push force (N).
                    self._mode1_k_boost = float(min(f_push, fz_cap))
                    self._mode1_x0 = x0
                    # Blend starts from the compression force for
                    # continuity.
                    self._mode1_boost_f_state = float(max(0.0, f_comp))

                compress_active = not bool(self._mode1_push_latched)
                energy_gate = bool(self._mode1_push_latched) and bool(
                    getattr(self.cfg, "use_energy_compensation", True)
                )
                if energy_gate:
                    # CONSTANT push force until liftoff (LO event zeroes
                    # it); a current-hop velocity floor keeps pushing
                    # while vz_up < v_to.
                    f_push_const = float(self._mode1_k_boost)
                    v_to_now = float(np.sqrt(
                        2.0 * g_up
                        * float(max(0.0, float(self.cfg.hop_height_m)))
                    ))
                    v_err = float(max(0.0, v_to_now - vz_f))
                    f_push_catch = (
                        m * g_st
                        + float(self.cfg.stance_push_vz_kp) * v_err
                        if v_err > 0.0
                        else 0.0
                    )
                    f_push_tgt = float(max(
                        f_push_const, f_push_catch
                    ))
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

                f_ref[2] = float(springForce_scalar)
                # Hard force budget: never exceed F_max (nor the hardware
                # cap), in ANY stance sub-phase.
                f_ref[2] = float(_clipf(
                    float(f_ref[2]),
                    float(self.cfg.stance_fz_min),
                    float(fz_cap),
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
        # ===== HFA propeller demand (CASE2026 paper) =====
        # Flight: props are the primary attitude actuator and track Tau_des
        # directly (Eq. 15 in the paper; no separate prop PD/gains).
        # Stance: props compensate ONLY the residual moment left after the
        # feasible leg allocation (tau_res = tau_des - r x f, Eq. 12); that is
        # computed below in the stance allocation branch once f_contact_w is
        # known. (2026-07-11: the independent stance_prop_* PD was DELETED.)
        Tau_prop_des = Tau_des.copy()
        Tau_des_dbg = Tau_des.copy()
        # Observer prediction input for NEXT tick: the total commanded body
        # attitude moment is Tau_des in both phases under HFA (leg + residual
        # props sum to the demand); props off in flight leaves it 0.
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
            # Raibert in WORLD FRD (+Z down, 竖直向下), same vertical sense as body/FK footpos.
            #   target_xy_w = Kv * v_lo_xy_w + Kr * desired_v_xy_w
            #   target_z_w  = +sqrt(l0^2 - ||target_xy_w||^2)
            #   foot_des_b  = R_wb^T @ target_w   (quaternion body<-world)
            l0 = float(self.cfg.leg_l0_m)
            kv = float(self.cfg.flight_kv)
            kr = float(self.cfg.flight_kr)
            step_lim = float(abs(float(self.cfg.flight_stepper_lim_m)))
            v_xy_w = np.array([float(self._v_hat_w[0]), float(self._v_hat_w[1])], dtype=float)
            vdes_xy_w = np.array([float(desired_v_xy_w[0]), float(desired_v_xy_w[1])], dtype=float)
            target_xy_w = (kv * v_xy_w + kr * vdes_xy_w).astype(float)
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
        thrust_sum_max = float(self.mass * self.gravity * float(self.cfg.thrust_total_ratio_max))
        props_on = bool(props_enabled_ctrl)
        if not props_on:
            thrust_sum_max = 0.0
        elif not bool(self._stance):
            # FLIGHT: hard world-Z force budget so the props stay a BOUNDED
            # gravity modulation (see flight_thrust_sum_max_ratio). The big
            # stance cap (thrust_total_ratio_max) exists for contact attitude
            # authority and must not leak into the ballistic phase.
            thrust_sum_max = float(min(
                thrust_sum_max,
                float(self.mass) * float(self.gravity)
                * float(max(0.0, float(
                    self.cfg.flight_thrust_sum_max_ratio
                ))),
            ))
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
            fz_cmd = float(max(0.0, float(f_ref[2]))) + f_dn

            tau_leg_des_w = np.asarray(Tau_des, dtype=float).reshape(3).copy()
            tau_leg_des_b = np.asarray(tau_b_att_des, dtype=float).reshape(3).copy()
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
                    # Props realize the attitude demand minus the moment
                    # delivered by the commanded stance leg force.
                    tau_leg_w = (
                        -_cross3(r_foot_w.reshape(3), f_contact_w.reshape(3))
                    ).reshape(3)
                    tau_prop_cmd_w = np.array([
                        float(Tau_des[0]) - float(tau_leg_w[0]),
                        float(Tau_des[1]) - float(tau_leg_w[1]),
                        0.0,
                    ], dtype=float)
                    if not np.all(np.isfinite(tau_prop_cmd_w)):
                        tau_prop_cmd_w = np.zeros(3, dtype=float)
                    thrusts = self._allocate_prop_thrust(
                        tau_des_w=tau_prop_cmd_w,
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
        if bool(self._stance) and float(scale) < 1.0:
            f_contact_w = (np.asarray(f_contact_w, dtype=float).reshape(3) * float(scale)).astype(float)

        if bool(self._stance) and props_on:
            # SATURATION-AWARE residual (2026-07-19, per user: "leg unchanged,
            # props cooperate"). The first prop pass above compensated
            # Tau_des - (-r x f_cmd) with the PRE-saturation leg force, so it
            # only covered the friction-cone / fxy clipping. The joint-torque
            # cap just scaled f_contact_w by `scale` (0.57..0.83 in the hop
            # logs during compression) -- the leg silently under-delivers
            # (1-scale) of its attitude moment. Recompute the CASE Eq.12
            # residual from the FINAL deliverable leg force so the props pick
            # up exactly that deficit (clipped to their own authority inside
            # _allocate_prop_thrust). Leg commands are untouched; at scale=1
            # this reproduces the first pass bit-for-bit.
            try:
                tau_leg_final_w = (
                    -_cross3(r_foot_w.reshape(3), f_contact_w.reshape(3))
                ).reshape(3)
                tau_prop_cmd_final_w = np.array([
                    float(Tau_des[0]) - float(tau_leg_final_w[0]),
                    float(Tau_des[1]) - float(tau_leg_final_w[1]),
                    0.0,
                ], dtype=float)
                if not np.all(np.isfinite(tau_prop_cmd_final_w)):
                    tau_prop_cmd_final_w = np.zeros(3, dtype=float)
                thrusts = self._allocate_prop_thrust(
                    tau_des_w=tau_prop_cmd_final_w,
                    prop_r_w=prop_r_w,
                    z_thrust_w=z_thrust_w,
                    thrust_sum_ref=float(thrust_sum_ref),
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
            "energy_gate": int(bool(self._stance) and bool(energy_gate)),
            "vz_up": float(vz_up),
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
            # FB-SLIP telemetry: TD-sized constant brake force and plan.
            "fbslip_v_td_m_s": float(self._mode1_v_td),
            "fbslip_f_brake_n": (
                float(self._mode1_f_brake)
                if self._mode1_f_brake is not None else float("nan")
            ),
            "fbslip_x_c_plan_m": float(self._mode1_x_c_plan),
            "fbslip_f_push_n": float(self._mode1_k_boost),
            "fbslip_t_bottom_s": float(self._mode1_t_bottom),
            # Flight-time apex measurement h = g*T^2/8 (log-only telemetry).
            "z_apex_actual_m": float(self._z_apex_actual),
            # MPC debug
            "mpc_status": mpc_status,
            "mpc_u0": mpc_u0.copy(),
        }

        return tau_cmd, pwm_us, info


