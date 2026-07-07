from __future__ import annotations

"""
ModeE core controller (real-robot version)
=========================================

This is the "modee" architecture used in MuJoCo:
  - Event-based hop phases (TD/COMP/PUSH/FLIGHT/APEX)
  - Stance wrench reference from PD + impulse shaping (no MPC)
  - SRB WBC-QP (OSQP) -> solves: GRF + 3-arm thrusts + (stance-only) motor torques
  - All control uses IMU + encoders only (no MuJoCo ground truth)

This file is MuJoCo-free and is meant to run on the real robot via LCM.
"""

from dataclasses import dataclass
import math
import numpy as np

# NOTE: hopper_controller is not a Python package by default (no __init__.py).
# Keep imports relative to the folder that runs the controller (same style as Hopper4.py).
from forward_kinematics import ForwardKinematics, InverseJacobian

from modee.controllers.motor_utils import MotorTableModel
from modee.controllers.wbc_qp_osqp import WBCQP, WBCQPConfig


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
    """Roll-pitch-yaw (XYZ intrinsic) from R_wb."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # roll
    roll = float(math.atan2(R[2, 1], R[2, 2]))
    # pitch
    pitch = float(math.atan2(-R[2, 0], math.sqrt(max(1e-12, R[2, 1] ** 2 + R[2, 2] ** 2))))
    # yaw
    yaw = float(math.atan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


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
            a = float(np.clip(dt / (tau + dt), 0.0, 1.0))
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
                # measured "down" direction in body
                a_b = (self._acc_f / a_norm).astype(float)
                # estimated "down" in body (from q): down_w = [0,0,-1], so down_b = R_bw * down_w = R_wb^T * down_w
                R_wb = _quat_to_R_wb(self._q)
                down_b = (R_wb.T @ np.array([0.0, 0.0, -1.0], dtype=float)).reshape(3)
                # tilt error axis ~ cross(down_b, a_b)
                e = np.cross(down_b, a_b)
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
    energy_comp_kp: float = 12.0
    hop_height_m: float = 0.20

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
    leg_l0_m: float = 0.464

    # hop target (world z)
    hop_z0: float = 0.9
    # Target apex height (world z, meters). Primary objective: reach this apex reliably.
    hop_peak_z: float = 0.7
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
    # NOTE: Bottom-level driver (ak60_controller.cpp) already uses Kalman filter on joint velocity.
    # Setting this to 0 to avoid double-filtering (which adds latency).
    joint_vel_lpf_tau: float = 0.005
    # Low-pass on q_shift / qd_shift used for touchdown/liftoff detection (seconds). Set <=0 to disable.
    q_shift_lpf_tau: float = 0.005
    qd_shift_lpf_tau: float = 0.005
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
    # Flight foot placement uses Hopper4 Raibert (Kv/Kr) in heading frame.
    flight_kv: float = 0.16
    flight_kr: float = 0.14
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
    swing_kd_xy: float = 1.0
    # Axial (virtual spring) stiffness and damping for flight leg control.
    # BALANCE: High kd prevents over-extension but amplifies velocity noise → jitter.
    # Use moderate kd (8-12) with strong LPF filtering instead of high kd.
    # Over-extension is also limited by the axial_coeff clamp logic (line ~2169).
    swing_kp_z: float = 1500.0
    stance_kp_z: float = 1100.0
    stance_kd_z: float = 20.0
    swing_kd_z: float = 8.0
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
    # We output a 6-length `pwm_us` vector that is sent directly to Betaflight via MSP_SET_MOTOR:
    #   motor_pwm_lcmt.pwm_values[0..5] -> motors M1..M6 on the flight controller.
    #
    # ModeE solves 3 thrust variables (one per *arm*) ordered consistently with `prop_positions_b`:
    #   arm 0 = RED, arm 1 = GREEN, arm 2 = BLUE
    #
    # Your real robot may have:
    # - 3 props total (one per arm), OR
    # - 6 props total (coaxial, two per arm), etc.
    #
    # We represent this as: for each arm, a tuple of 1+ PWM indices that belong to that arm.
    # The per-arm thrust is divided equally among its indices.
    #
    # Your measured order (top view, clockwise): GREEN -> PWM[1], then PWM[3], then PWM[2].
    # And you currently have ONLY 3 propellers installed, so we map:
    #   GREEN arm -> (1,)
    #   BLUE  arm -> (3,)   (clockwise next from GREEN)
    #   RED   arm -> (2,)   (clockwise next from BLUE)
    #
    # Unused PWM indices will get 0 thrust -> pwm_min_us (1000us).
    # If you re-wire / re-map, update this tuple.
    prop_pwm_idx_per_arm: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
        (2,),  # RED arm
        (1,),  # GREEN arm
        (3,),  # BLUE arm
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
    mu: float = 0.4

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
    stance_fz_max: float = 200.0

    # PWM limits
    pwm_min_us: float = 1000.0
    # PWM cap. Raised for stronger prop authority.
    pwm_max_us: float = 1700.0

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
    # Increased kR to prevent attitude divergence during long flight phases.
    flight_kR_roll: float = 60.0
    flight_kW_roll: float = 40.0
    flight_kR_pitch: float = 60.0
    flight_kW_pitch: float = 40.0
    flight_tau_rp_max: float = 25.0

    # ===== Control mode switch =====
    # 1 = pure_leg:          leg-only QP, propellers OFF (stance & flight)
    # 2 = decouple_leg_prop: leg-only QP + lstsq prop overlay (stance), unified QP (flight)
    # 3 = unified_qp:        unified QP with leg+prop (stance & flight)
    control_mode: int = 3

    # ===== Stance phase attitude control (SRB SO(3) PD) =====
    # SRB approach: compute Tau_des = -kR*e_R - kW*omega in body frame, then let QP
    # find the optimal foot contact force f_foot such that r_foot × f_foot ≈ Tau_des,
    # subject to friction cone |fx,fy| ≤ μ·fz.
    # The QP naturally generates horizontal foot forces for attitude correction.
    # Priority: 1. Height (Apex) 2. Velocity 3. Attitude (via QP slack weights).
    # SRB stance attitude SO(3) PD gains (body-frame torque: tau = -kR*e_R - kW*omega).
    # QP maps Tau_des to horizontal foot forces via r_foot × f_foot = Tau_des.
    stance_kpp_x: float = 100.0    # kR roll  (increased from 31 to compensate for proper damping)
    stance_kpp_y: float = 100.0    # kR pitch
    stance_kpd_x: float = 1.0     # kW roll  (increased for stronger angular velocity damping)
    stance_kpd_y: float = 1.0     # kW pitch
    stance_tau_rp_max: float = 30.0

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
    w_slack_tau_xy: float = 2e5
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

        # frames: base (z up) vs delta/vicon (z down)
        self.robot2vicon = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

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

        # WBC-QP (tracks f_ref + Tau_des from SRB or MPC)
        self.wbc = WBCQP(
            WBCQPConfig(
                mu=float(cfg.mu),
                fz_min=0.0,
                fz_max=float(cfg.stance_fz_max),
                fxy_abs_max=(float(abs(float(cfg.wbc_fxy_abs_max))) if float(getattr(cfg, "wbc_fxy_abs_max", 0.0)) > 0.0 else None),
                thrust_total_ratio_max=float(cfg.thrust_total_ratio_max),
                thrust_min_each=float(max(0.0, float(cfg.wbc_thrust_min_each_n))),
                w_f=float(max(0.0, float(getattr(cfg, "wbc_w_f", 1e-4)))),
                w_t=float(max(0.0, float(getattr(cfg, "wbc_w_t", 1e-4)))),
                w_f_ref=float(max(0.0, float(cfg.wbc_w_f_ref))),
                w_t_ref=0.0,
                w_tsum_ref=0.0,
                # Slack weights are exposed in ModeEConfig for tuning.
                w_slack_Fxy=float(getattr(cfg, "w_slack_Fxy", 2e4)),
                w_slack_Fz=float(getattr(cfg, "w_slack_Fz", 8e4)),
                w_slack_tau_xy=float(getattr(cfg, "w_slack_tau_xy", 5e4)),
                w_slack_tau_z=float(getattr(cfg, "w_slack_tau_z", 1e3)),
                w_slack_Fxy_flight=float(getattr(cfg, "w_slack_Fxy_flight", 2e3)),
                w_slack_Fz_flight=float(getattr(cfg, "w_slack_Fz_flight", 6e3)),
                w_slack_tau_flight_xy=float(getattr(cfg, "w_slack_tau_flight_xy", 1e4)),
                w_slack_tau_flight_z=float(getattr(cfg, "w_slack_tau_flight_z", 8e2)),
                enable_tau_vars=True,
                w_tau=0.0,
                w_tau_ref=float(cfg.wbc_w_tau_ref_flight),
            )
        )

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
        self._flight_vel = np.zeros(3, dtype=float)
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
        # precompute prop positions in base frame (GREEN on +X)
        L = float(cfg.prop_arm_len_m)
        # order: [RED, GREEN, BLUE] (visual naming; GREEN forward)
        self.prop_positions_b = np.array(
            [
                [-0.5 * L, +math.sqrt(3) * 0.5 * L, 0.0],
                [+1.0 * L, 0.0, 0.0],
                [-0.5 * L, -math.sqrt(3) * 0.5 * L, 0.0],
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
        self._flight_vel[:] = 0.0
        self._prev_vz = None
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
        fro = float(np.linalg.norm(A, ord="fro"))
        lam = float(lam_rel * max(1e-12, fro))
        M = (A.T @ A + (lam * lam) * np.eye(3, dtype=float)).astype(float)
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

    def compute_tau_from_force_base(
        self,
        *,
        joint_pos: np.ndarray,
        f_base: np.ndarray,
        use_contact_site_map: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map a desired 3D force in BODY/BASE frame (+X forward, +Y left, +Z up)
        to delta motor torques using the same Jacobian convention as ModeE.

        Conventions:
        - FK/Jacobian are in delta/vicon frame (+Z down).
        - `robot2vicon` converts BASE(+Z up) -> DELTA(+Z down).
        - Mapping uses: tau = inv(J_inv^T) * f_delta

        Args:
          joint_pos: (3,) motor angles [0,1,2] in physical motor order.
          f_base: (3,) force in BASE frame (+Z up).
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
            foot_vicon = (self.robot2vicon @ np.asarray(foot_b, dtype=float).reshape(3)).reshape(3)
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
            x3[0] = float(np.clip(x3[0], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
            x3[1] = float(np.clip(x3[1], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
            x3[2] = float(np.clip(x3[2], float(self._delta_ws["z_min"]), float(self._delta_ws["z_max"])))

        # Compute inverse Jacobian at x3 (delta/vicon frame)
        J_inv_map, _ = self.kin.inverse_jacobian(x3, np.zeros(3, dtype=float), theta=None)
        J_inv_map = np.asarray(J_inv_map, dtype=float).reshape(3, 3)
        inv_Jt = self._stable_inv3(J_inv_map.T)

        # base(+Z up) -> delta(+Z down)
        f_delta = (self.robot2vicon @ f_base.reshape(3)).reshape(3)

        # tau = inv(J_inv^T) * f_delta
        tau = (inv_Jt @ f_delta.reshape(3)).reshape(3)

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
        J0 = np.cross(axis_roll, foot_rel)
        J1 = np.cross(axis_pitch, foot_rel)
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
        x = float(np.clip(float(x), 0.0, 1.0))
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
        """
        cfg = self.cfg
        R_wb = np.asarray(R_wb, dtype=float).reshape(3, 3)
        z_td_base = float(z_td_base)
        vz_td = float(vz_td)
        q_shift_td = float(q_shift_td)

        # Time budget
        T = float(max(float(cfg.stance_min_T), float(cfg.stance_T)))

        # COM offset in world-z (approx constant over the stance reference generation)
        com_off_z = float((R_wb @ self.com_b.reshape(3))[2])
        self._stance_com_off_z = float(com_off_z)

        # Touchdown COM-z reference origin
        z0 = float(z_td_base + com_off_z)

        # Estimate COM-z at "nominal leg length" (q_shift=0) for the end of stance reference.
        # We treat this as the nominal liftoff height. (The robot may liftoff earlier in practice.)
        z_end = float((z_td_base - q_shift_td) + com_off_z)

        # Desired takeoff speed (already computed at touchdown; clamp for safety)
        # NOTE: v_to_min is a legacy guard; we will still allow v_to to be reduced for feasibility
        # if the compression/extension distance is insufficient.
        v_to = float(np.clip(float(self._v_to_cmd), float(cfg.v_to_min), float(cfg.v_to_max)))

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
        t_comp = float(np.clip(t_comp, float(cfg.soft_land_tc_min), float(cfg.soft_land_tc_max_ratio) * T))
        t_comp = float(min(t_comp, max(1e-3, T - 1e-3)))

        depth = float(0.5 * v_in * t_comp)
        depth = float(np.clip(depth, float(cfg.soft_land_depth_min_m), float(cfg.soft_land_depth_max_m)))

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
        Return (z_ref, vz_ref, az_ref) in COM/world-z at time since touchdown.
        If profile is not initialized, falls back to holding current estimate.
        """
        t = float(max(0.0, float(t_in_stance)))
        if (not bool(self._stance_prof_inited)) or (self._stance_poly1 is None) or (self._stance_poly2 is None):
            # Fallback: hold current COM z and use current vz_hat (best effort)
            return float(self._p_hat_w[2] + float(self._stance_com_off_z)), float(self._v_hat_w[2]), 0.0

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

        # --- Signal conditioning: joint velocity LPF (used by kinematics) ---
        joint_vel_kin = joint_vel.copy()
        try:
            tau = float(getattr(self.cfg, "joint_vel_lpf_tau", 0.0))
        except Exception:
            tau = 0.0
        if float(tau) > 0.0:
            if not bool(self._joint_vel_lpf_init):
                self._joint_vel_lpf = joint_vel.copy()
                self._joint_vel_lpf_init = True
            else:
                a = float(np.clip(float(self.dt) / (float(tau) + float(self.dt)), 0.0, 1.0))
                self._joint_vel_lpf = (1.0 - a) * self._joint_vel_lpf + a * joint_vel
            joint_vel_kin = np.asarray(self._joint_vel_lpf, dtype=float).reshape(3).copy()

        # --- Foot kinematics ---
        # Variables are kept in the same naming convention as the delta version for compatibility with the rest
        # of the controller and debug logs:
        # - foot_vicon: delta/vicon frame (+Z down)
        # - foot_b:     base/body frame (+Z up)
        # - foot_vdot_vicon: foot velocity in delta/vicon frame
        # - foot_vrel_b:     foot velocity in base/body frame
        J_inv: np.ndarray | None = None
        J_body: np.ndarray | None = None

        if self._leg_model == "serial":
            # MuJoCo serial equivalent plant: q = [roll, pitch, shift]
            foot_b, J_body = self._serial_leg_fk_jac(
                q_roll=float(joint_pos[0]),
                q_pitch=float(joint_pos[1]),
                q_shift=float(joint_pos[2]),
            )
            foot_b = np.asarray(foot_b, dtype=float).reshape(3)
            J_body = np.asarray(J_body, dtype=float).reshape(3, 3)
            foot_vrel_b = (J_body @ joint_vel_kin.reshape(3)).reshape(3)
            foot_vicon = (self.robot2vicon @ foot_b.reshape(3)).reshape(3)
            foot_vdot_vicon = (self.robot2vicon @ foot_vrel_b.reshape(3)).reshape(3)
        else:
            # delta/vicon frame (+Z down)
            if self.fk is None or self.kin is None:
                raise RuntimeError("delta leg model requested but kinematics is not initialized")
            foot_vicon, _ = self.fk.forward_kinematics(joint_pos)
            foot_vicon = np.asarray(foot_vicon, dtype=float).reshape(3)
            # NOTE: we do NOT trust the raw xdot solve inside `inverse_jacobian` near singularities.
            # Always compute xdot from the returned J_inv with a stable inverse (optionally DLS).
            J_inv_raw, _ = self.kin.inverse_jacobian(foot_vicon, joint_vel_kin, theta=None)
            J_inv = np.asarray(J_inv_raw, dtype=float).reshape(3, 3)
            foot_vdot_vicon = (self._stable_inv3(J_inv) @ joint_vel_kin.reshape(3)).reshape(3).astype(float)

        # Convert to base frame (z up)
        foot_b = (self.robot2vicon @ foot_vicon.reshape(3)).reshape(3)
        foot_vrel_b = (self.robot2vicon @ foot_vdot_vicon.reshape(3)).reshape(3)

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
                    a = float(np.clip(float(self.dt) / (float(tau_q) + float(self.dt)), 0.0, 1.0))
                    self._q_shift_lpf = float((1.0 - a) * float(self._q_shift_lpf) + a * q_shift_raw)
                else:
                    self._q_shift_lpf = q_shift_raw
                if float(tau_qd) > 0.0:
                    a = float(np.clip(float(self.dt) / (float(tau_qd) + float(self.dt)), 0.0, 1.0))
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

        # --- Touchdown velocity estimate from leg kinematics (best-effort) ---
        # Uses Hopper4/com_filter.py formula: ang_vel = R_wb @ (gyro * 0.1)
        _ang_vel_dbg = (R_wb_hat @ (imu_gyro_b * 0.1).reshape(3)).reshape(3)
        v_base_from_foot_w = (
            (R_wb_hat @ (self.robot2vicon @ (-foot_vdot_vicon)).reshape(3)).reshape(3)
            + (R_wb_hat @ np.cross(_ang_vel_dbg, (self.robot2vicon @ (-foot_vicon)).reshape(3))).reshape(3)
        )

        # --- IMU propagation for base velocity ---
        # Convention (matches our SimpleIMUAttitudeEstimator and current hopper_driver):
        # - `imu_acc_b` is a *gravity / down* vector in BODY frame (units: m/s^2),
        #   i.e. when the robot is level + stationary: imu_acc_b ≈ [0, 0, -9.81] (because +Z is UP).
        # - Then linear acceleration is: a_w = g_w - R_wb * imu_acc_b
        #   (stationary: R_wb*imu_acc_b == g_w  => a_w == 0).
        g_w = np.array([0.0, 0.0, -float(self.gravity)], dtype=float)
        a_w = (g_w - (R_wb_hat @ imu_acc_b.reshape(3))).reshape(3)
        # Hopper4-like XY velocity estimator:
        # - STANCE: estimate XY from leg kinematics (foot velocity + omega×r), fused below
        # - FLIGHT: HOLD XY (do not integrate IMU accel bias)
        #
        # Default is True to match Hopper4 behavior (even if the config field is absent).
        use_h4_vxy = bool(getattr(self.cfg, "use_hopper4_vxy_estimator", True))
        # User request: allow a "hard stop" mode that keeps internal velocity estimate exactly zero
        # (prevents IMU integration drift from moving foot targets when desired_v == 0).
        if bool(getattr(self, "_user_zero_vel_hold", False)):
            self._v_hat_w[:] = 0.0
            self._v_hat_inited = True
            v_pred = np.zeros(3, dtype=float)
        else:
            if not bool(self._v_hat_inited):
                self._v_hat_w = np.zeros(3, dtype=float)
                self._v_hat_inited = True
            v_pred = (np.asarray(self._v_hat_w, dtype=float).reshape(3) + a_w * float(self.dt)).reshape(3)
            # Hopper4-like behavior: do NOT integrate IMU accel for horizontal velocity (XY).
            # - STANCE: XY will be corrected by leg-kinematics fusion below
            # - FLIGHT: hold XY to avoid drift
            if bool(use_h4_vxy):
                v_pred[0] = float(self._v_hat_w[0])
                v_pred[1] = float(self._v_hat_w[1])

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
                self._apex_reached = False
                self._td_debounce_count = 0
                self._lo_debounce_count = 0
                # com_filter.py: do NOT clear window at touchdown (rolling)
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

                # touchdown z estimate from kinematics (assume foot at ground z=0)
                z_td_est = -float((R_wb_hat @ foot_b.reshape(3))[2])
                self._z_hat_contact_filt = float(z_td_est)
                self._p_hat_w[2] = float(z_td_est)

                # Takeoff speed target for desired apex (ballistic, with prop assist).
                g_eff = float(self.gravity - (float(z_w[2]) * float(self.mass) * float(self.gravity) * float(self.cfg.prop_base_thrust_ratio)) / max(1e-6, float(self.mass)))
                g_eff = float(max(1e-3, g_eff))
                dz_tgt = float(max(0.05, float(self.cfg.hop_height_m)))
                g_eff_log = float(g_eff)
                dz_tgt_log = float(dz_tgt)
                v_to_nominal = float(np.sqrt(2.0 * g_eff * dz_tgt))
                self._v_to_cmd = float(np.clip(v_to_nominal, float(self.cfg.v_to_min), float(self.cfg.v_to_max)))

                # Initialize smooth stance profile.
                try:
                    self._init_unified_stance_profile(
                        R_wb=R_wb_hat,
                        z_td_base=float(z_td_est),
                        vz_td=float(v_base_from_foot_w[2]) if np.isfinite(float(v_base_from_foot_w[2])) else float(v_pred[2]),
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
                # com_filter.py: compute flight_vel from rolling 10-sample average
                ang_vel_lo = (R_wb_hat @ (imu_gyro_b * 0.1).reshape(3)).reshape(3)
                if self._vel_window_count < self._vel_window_size:
                    avg_fv = foot_vdot_vicon.copy()
                    avg_fp = foot_vicon.copy()
                else:
                    avg_fv = np.mean(self._foot_vel_window, axis=0)
                    avg_fp = np.mean(self._foot_pos_window, axis=0)
                self._flight_vel = (
                    (R_wb_hat @ (self.robot2vicon @ (-avg_fv)).reshape(3)).reshape(3)
                    + (R_wb_hat @ np.cross(ang_vel_lo, (self.robot2vicon @ (-avg_fp)).reshape(3))).reshape(3)
                )
                self._flight_vel[2] = 0.0
                # Record liftoff state for apex prediction (used by apex feedback loop)
                if bool(getattr(self.cfg, "apex_use_feedback", True)):
                    self._z_lo = float(self._p_hat_w[2])
                    self._vz_lo = float(self._v_hat_w[2])

        # ===== com_filter.py velocity estimator =====
        # Stance: instantaneous foot-kinematics (no averaging).
        # Flight: hold latched flight_vel.
        # Rolling 10-sample window updated every stance tick for liftoff average.
        v_meas_w = None
        if bool(self._stance):
            ang_vel_w = (R_wb_hat @ (imu_gyro_b * 0.1).reshape(3)).reshape(3)
            v_meas_w = (
                (R_wb_hat @ (self.robot2vicon @ (-foot_vdot_vicon)).reshape(3)).reshape(3)
                + (R_wb_hat @ np.cross(ang_vel_w, (self.robot2vicon @ (-foot_vicon)).reshape(3))).reshape(3)
            )
            if np.all(np.isfinite(v_meas_w)):
                self._foot_vel_window[:-1] = self._foot_vel_window[1:]
                self._foot_vel_window[-1] = foot_vdot_vicon.copy()
                self._foot_pos_window[:-1] = self._foot_pos_window[1:]
                self._foot_pos_window[-1] = foot_vicon.copy()
                self._vel_window_count = min(self._vel_window_count + 1, self._vel_window_size)
                self._v_hat_w[0] = float(v_meas_w[0])
                self._v_hat_w[1] = float(v_meas_w[1])
                self._v_hat_w[2] = float(v_pred[2])
            else:
                self._v_hat_w = np.asarray(v_pred, dtype=float).reshape(3).copy()
                v_meas_w = None
        else:
            self._v_hat_w[0] = float(self._flight_vel[0])
            self._v_hat_w[1] = float(self._flight_vel[1])
            self._v_hat_w[2] = float(v_pred[2])

        # integrate position + stance z correction
        self._p_hat_w = self._p_hat_w + self._v_hat_w * float(self.dt)
        if bool(self._stance):
            z_meas = -float((R_wb_hat @ foot_b.reshape(3))[2])
            if self._z_hat_contact_filt is None:
                self._z_hat_contact_filt = float(z_meas)
            z_tau = 0.05
            az = float(np.clip(float(self.dt) / (z_tau + float(self.dt)), 0.0, 1.0))
            self._z_hat_contact_filt = (1.0 - az) * float(self._z_hat_contact_filt) + az * float(z_meas)
            self._p_hat_w[2] = float(self._z_hat_contact_filt)

        # apex detection (flight): vz_hat sign change
        vz_hat = float(self._v_hat_w[2])
        if self._prev_vz is None:
            self._prev_vz = float(vz_hat)
        if (not bool(self._stance)) and (float(self._prev_vz) > 0.0) and (float(vz_hat) <= 0.0):
            apex_evt = True
            self._apex_reached = True
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
            s = float(np.clip(t_in_stance / max(1e-6, float(self.cfg.stance_T)), 0.0, 1.0))

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
                    # For serial model, J_body is in BASE frame (+Z up):
                    #   tau = J^T * f_leg_b,   f_leg_b = -R^T * f_grf_w
                    # => tau = -(J^T * R^T) * f_grf_w
                    A_tau_f_3rsr = (-(np.asarray(J_body, dtype=float).reshape(3, 3).T @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T)).astype(float)
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
                x3[0] = float(np.clip(x3[0], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
                x3[1] = float(np.clip(x3[1], -float(self._delta_ws["xy"]), +float(self._delta_ws["xy"])))
                x3[2] = float(np.clip(x3[2], float(self._delta_ws["z_min"]), float(self._delta_ws["z_max"])))

                # Recompute inverse Jacobian at the clamped workspace point for numerical robustness.
                # Hopper4 returns J_inv such that: thetadot = J_inv * xdot  (delta/vicon frame)
                # Torque mapping: tau = inv(J_inv^T) * f_delta
                J_inv_map, _ = self.kin.inverse_jacobian(x3, np.zeros(3, dtype=float), theta=None)
                J_inv_map = np.asarray(J_inv_map, dtype=float).reshape(3, 3)
                inv_Jt = self._stable_inv3(J_inv_map.T)

                # f_w is the GRF (ground -> robot) in WORLD frame (+Z up).
                # Our torque convention matches Hopper4/ModeE: tau maps to the force the LEG applies on the ground
                # (robot -> ground), which is the opposite of GRF. Therefore the stance torque map has a leading '-'.
                #
                # World GRF -> BODY -> DELTA (+Z down):
                #   f_delta_grf = robot2vicon * (R_wb^T * f_w)
                # Desired leg force for torque mapping (robot->ground) is: f_delta_leg = -f_delta_grf
                # tau = inv(J_inv^T) * f_delta_leg
                A_tau_f_3rsr = (-(inv_Jt @ self.robot2vicon @ np.asarray(R_wb_hat, dtype=float).reshape(3, 3).T)).astype(float)
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

        # ===== Wrench-level controller (stance) =====
        # We run the controller in a "wrench -> allocation" structure:
        #  - Upstream produces desired *net* wrench on the body: (F_des_w, Tau_des_w).
        #  - Downstream WBC-QP allocates that wrench into:
        #      decision vars: [f_contact_w, thrusts, tau_cmd, slack]
        #    using equality constraints:
        #      f + z_w*sum(t) + sF   = F_des
        #      r×f + Σ r_i×(z_w*t_i) + sTau = Tau_des
        #  - We additionally provide *references* (f_ref, thrust_ref, tau_ref) for tracking/smoothing.
        #
        # Important convention:
        #   f_contact_w is the GRF (ground -> robot) in WORLD frame (+Z up).
        #   Each thrust_i acts along body +Z, which is z_w in WORLD. When the body is tilted, z_w has XY components.
        #   Therefore, if we want zero net horizontal force, the contact force must cancel the XY component of thrust.
        pure_leg_mode = bool(getattr(self.cfg, "pure_leg_mode", False))
        thrust_sum_ref = float(self.mass * self.gravity * float(self.cfg.prop_base_thrust_ratio))
        # Minimum stance thrust sum bound (policy): allows "legs do most of the stance work" when set to 0.
        thrust_sum_min_stance = float(self.mass * self.gravity * float(getattr(self.cfg, "stance_thrust_sum_min_ratio", 0.0)))
        thrust_max_each_qp = float(self.cfg.thrust_max_each_n)
        thrust_ref = None
        # Global propeller enable gate (for single-leg/no-prop tuning).
        # If false, thrust variables are hard-clamped to zero in both stance and flight.
        props_enabled_ctrl = (not bool(pure_leg_mode)) and (
            bool(self.cfg.stance_use_props) or (float(self.cfg.prop_base_thrust_ratio) > 1e-9)
        )
        if bool(pure_leg_mode):
            thrust_sum_ref = 0.0
            thrust_sum_min_stance = 0.0
            thrust_max_each_qp = 0.0
        if not bool(props_enabled_ctrl):
            thrust_sum_ref = 0.0
            thrust_sum_min_stance = 0.0
            thrust_max_each_qp = 0.0
            thrust_ref = np.zeros(3, dtype=float)
        # ===== Compute SO(3) attitude error EARLY =====
        # Needed by stance/flight Tau_des (SO(3) PD).
        # e_R = 0.5 * vee(R_des^T R - R^T R_des) captures roll/pitch error in body frame.
        yaw = float(rpy_hat[2])
        R_des = _Rz(yaw)
        E = (R_des.T @ R_wb_hat) - (R_wb_hat.T @ R_des)
        e_R = 0.5 * _vee_so3(E)

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
                    mpc_x0[6] = float(rpy_hat[0])   # roll
                    mpc_x0[7] = float(rpy_hat[1])   # pitch
                    mpc_x0[8] = float(rpy_hat[2])   # yaw
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
                            a_om = float(np.clip(float(self.dt) / (float(tau_om) + float(self.dt)), 0.0, 1.0))
                            self._mpc_omega_lpf = (1.0 - a_om) * self._mpc_omega_lpf + a_om * omega_mpc_b
                        omega_mpc_b = np.asarray(self._mpc_omega_lpf, dtype=float).reshape(3).copy()
                    # Convert to world-frame angular velocity for MPC state consistency.
                    omega_mpc = (R_wb_hat @ omega_mpc_b.reshape(3)).reshape(3)
                    try:
                        wclip_mpc = float(max(0.0, float(getattr(self.cfg, "mpc_omega_xy_clip_radps", 0.0))))
                    except Exception:
                        wclip_mpc = 0.0
                    if wclip_mpc > 1e-9:
                        omega_mpc[0] = float(np.clip(float(omega_mpc[0]), -wclip_mpc, +wclip_mpc))
                        omega_mpc[1] = float(np.clip(float(omega_mpc[1]), -wclip_mpc, +wclip_mpc))
                    mpc_x0[9] = float(omega_mpc[0])   # ωx (body ≈ world for small angles)
                    mpc_x0[10] = float(omega_mpc[1])  # ωy
                    mpc_x0[11] = float(omega_mpc[2])  # ωz
                    mpc_x0[12] = float(rpy_hat[2])     # yaw_ref

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
                    yaw_now = float(rpy_hat[2])
                    px_now = float(self._p_hat_w[0])
                    py_now = float(self._p_hat_w[1])
                    pz_now = float(self._p_hat_w[2])

                    # Desired takeoff velocity from hop_height_m (relative height,
                    # consistent with the 1D-tuned energy compensation parameters).
                    # Using hop_height_m (0.15m) instead of hop_peak_z (0.7m absolute)
                    # gives a moderate v_to ≈ 1.72 m/s → flight ≈ 0.35s.
                    h_target = float(max(0.05, float(self.cfg.hop_height_m)))
                    v_to_target = float(np.sqrt(2.0 * float(self.gravity) * h_target))
                    T_stance_total = float(max(0.05, float(self.cfg.stance_T)))
                    # Start push early enough for short-contact hops; ratio is configurable.
                    push_ratio = float(np.clip(float(self.cfg.mpc_push_start_ratio), 0.05, 0.6))
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
                        z_w=z_w,
                        T_base=float(thrust_sum_ref),
                    )
                    mpc_status = str(mpc_result.get("status", "unknown"))
                    mpc_u0 = np.asarray(mpc_result.get("u0", np.zeros(3)), dtype=float).reshape(3)

                    if mpc_status in ("solved", "solved inaccurate", "solved_inaccurate"):
                        # Exponential low-pass filter on horizontal forces to prevent
                        # solve-to-solve oscillation (the dominant 22-28 Hz shaking).
                        # Vertical force fz passes through unfiltered for responsive push.
                        alpha_fxy = float(np.clip(float(self.cfg.mpc_fxy_lpf_alpha), 0.0, 1.0))
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
                
                springForce_scalar = -k_spring * (l_leg - l0) - b_spring * springVel_scalar
                
                leg_velocity = float(qd_shift)
                if leg_velocity > 0.0 and bool(getattr(self.cfg, "use_energy_compensation", True)):
                    m = float(self.mass)
                    g = float(self.gravity)
                    
                    groundHeight = float(np.dot(
                        R_wb_hat @ foot_b_now, np.array([0.0, 0.0, -1.0])))
                    
                    energy = (0.5 * m * springVel_scalar * springVel_scalar
                              + 0.5 * k_spring * (l0 - l_leg)**2
                              + m * g * (-1.0) * groundHeight)
                    h = float(self.cfg.hop_height_m)
                    target = m * g * (l0 + h)
                    E_error = target - energy
                    
                    Kp = float(self.cfg.energy_comp_kp)
                    energy_comp_fz = float(max(0.0, Kp * E_error))
                    springForce_scalar += energy_comp_fz
                
                if springForce_scalar < 0.0:
                    springForce_scalar = 0.0
                
                f_ref[2] = float(springForce_scalar)
                f_ref[2] = float(np.clip(float(f_ref[2]), float(self.cfg.stance_fz_min), float(self.cfg.stance_fz_max)))
        except Exception:
            pass

        if bool(self._stance):
            self._f_ref_z_prev = float(f_ref[2])
            self._f_ref_xy_prev[:] = np.asarray(f_ref[0:2], dtype=float).reshape(2)
        else:
            self._f_ref_z_prev = 0.0
            self._f_ref_xy_prev[:] = 0.0

        # Friction cone is enforced by QP constraints; no need to clip f_ref here.

        # ===== SO(3) attitude torque (SRB: direct PD on attitude error) =====
        # Both stance and flight use the same SO(3) PD structure:
        #   tau_b = -kR * e_R - kW * omega
        # In stance, QP maps Tau_des to horizontal foot forces via r_foot × f_foot = Tau_des.
        # In flight, Tau_des is realized by propeller differential thrust.
        if bool(self._stance):
            tau_rp_max = float(self.cfg.stance_tau_rp_max)
            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            kR_x = float(self.cfg.stance_kpp_x)
            kR_y = float(self.cfg.stance_kpp_y)
            kW_x = float(self.cfg.stance_kpd_x)
            kW_y = float(self.cfg.stance_kpd_y)
            tau_b_stance = np.zeros(3, dtype=float)
            tau_b_stance[0] = -kR_x * float(e_R[0]) - kW_x * float(omega_b[0])
            tau_b_stance[1] = -kR_y * float(e_R[1]) - kW_y * float(omega_b[1])
            tau_w = (R_wb_hat @ tau_b_stance.reshape(3)).reshape(3)
            Tau_des = np.array([float(tau_w[0]), float(tau_w[1]), 0.0], dtype=float)
        else:
            # Flight phase: separate roll/pitch gains (for propeller control)
            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            if not bool(props_enabled_ctrl):
                # No propellers physically available: do not request flight attitude torques.
                tau_rp_max = 0.0
                Tau_des = np.zeros(3, dtype=float)
            else:
                tau_rp_max = float(self.cfg.flight_tau_rp_max)
                kR_roll = float(self.cfg.flight_kR_roll)
                kW_roll = float(self.cfg.flight_kW_roll)
                kR_pitch = float(self.cfg.flight_kR_pitch)
                kW_pitch = float(self.cfg.flight_kW_pitch)
                tau_b = np.zeros(3, dtype=float)
                tau_b[0] = (-float(kR_roll) * float(e_R[0])) - (float(kW_roll) * float(omega_b[0]))
                tau_b[1] = (-float(kR_pitch) * float(e_R[1])) - (float(kW_pitch) * float(omega_b[1]))
                tau_b[2] = 0.0
                tau_w = (R_wb_hat @ tau_b.reshape(3)).reshape(3)
                Tau_des = np.array([float(tau_w[0]), float(tau_w[1]), 0.0], dtype=float)
        
        # Norm-based torque limiting
        if bool(self._stance):
            tau_rp_norm = float(np.sqrt(Tau_des[0]**2 + Tau_des[1]**2))
            if tau_rp_norm > tau_rp_max and tau_rp_norm > 1e-9:
                scale = tau_rp_max / tau_rp_norm
                Tau_des[0] = float(Tau_des[0] * scale)
                Tau_des[1] = float(Tau_des[1] * scale)
        else:
            Tau_des[0] = float(np.clip(Tau_des[0], -tau_rp_max, +tau_rp_max))
            Tau_des[1] = float(np.clip(Tau_des[1], -tau_rp_max, +tau_rp_max))
        Tau_des_dbg = Tau_des.copy()
        omega_b_used_dbg = omega_b.copy()

        # ===== Flight swing torque reference (only after apex) =====
        tau_ref = None
        # Debug: force that is fed into the Jacobian->torque mapping.
        # - f_tau_b:     BODY frame (+Z up)
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

        xdot_for_pd = foot_vrel_b.copy()

        if not bool(self._stance):
            # Hopper4 flight target:
            #   targetFootPos = Kv * v_xy + Kr * desired_v_xy
            #   ||targetFootPos|| clamped by stepperLim
            #   targetFootPos[2] = -sqrt(l0^2 - ||xy||^2)
            l0 = float(self.cfg.leg_l0_m)
            kv = float(self.cfg.flight_kv)
            kr = float(self.cfg.flight_kr)
            step_lim = float(abs(float(self.cfg.flight_stepper_lim_m)))
            targetFootPos_w = (
                kv * np.array([float(self._v_hat_w[0]), float(self._v_hat_w[1]), 0.0], dtype=float)
                + kr * np.array([float(desired_v_xy_w[0]), float(desired_v_xy_w[1]), 0.0], dtype=float)
            )
            if bool(self.cfg.mode_1d):
                targetFootPos_w[0] = 0.0
                targetFootPos_w[1] = 0.0
            normTarget = float(np.linalg.norm(targetFootPos_w))
            if (step_lim > 1e-9) and (normTarget > step_lim):
                targetFootPos_w = (targetFootPos_w * (step_lim / max(1e-12, normTarget))).astype(float)
                normTarget = float(np.linalg.norm(targetFootPos_w))
            targetFootPos_w[2] = -float(np.sqrt(max(0.0, float(l0 * l0) - float(normTarget * normTarget))))
            foot_des_w = np.asarray(targetFootPos_w, dtype=float).reshape(3)
            foot_des_b = (R_wb_hat.T @ foot_des_w.reshape(3)).reshape(3)
            s2s_active = True

            # Expose flight target for debug:
            # - foot_des_b is in BODY frame (+Z up) and is what the PD uses.
            # - Convert to WORLD for logging/plotting convenience.
            foot_des_b_dbg = np.asarray(foot_des_b, dtype=float).reshape(3).copy()
            foot_des_w_dbg = (np.asarray(R_wb_hat, dtype=float).reshape(3, 3) @ foot_des_b_dbg.reshape(3)).reshape(3)
            p_foot_des_w_dbg = (np.asarray(self._p_hat_w, dtype=float).reshape(3) + foot_des_w_dbg.reshape(3)).reshape(3)
            s2s_active_dbg = int(bool(s2s_active))

            # ===== Hopper4 flight leg force (sideForce + springForce), in BODY frame =====
            # Match Hopper4 lines:
            #   sideForce = Khp*(targetFootPos - x) - Khd*(xdot - omega×x)
            #   sideForce -= dot(sideForce, unitSpring)*unitSpring
            #   force = -k*(l - l0)
            #   springForce = force*unitSpring - b*springVel
            #   footForce = sideForce + springForce
            x = np.asarray(foot_b, dtype=float).reshape(3)
            targetFootPos = np.asarray(foot_des_b, dtype=float).reshape(3)
            xdot = np.asarray(xdot_for_pd, dtype=float).reshape(3)  # leg-induced foot velocity (base frame)

            leg_length = float(np.linalg.norm(x))
            if leg_length < 1e-6:
                unitSpring = np.array([0.0, 0.0, -1.0], dtype=float)
                leg_length = 0.0
            else:
                unitSpring = (x / leg_length).astype(float)

            springVel = (float(np.dot(xdot, unitSpring)) * unitSpring).astype(float)

            omega_b = np.asarray(imu_gyro_b, dtype=float).reshape(3)
            Khp = float(self.cfg.swing_kp_xy)
            Khd = float(self.cfg.swing_kd_xy)
            sideForce = (Khp * (targetFootPos - x) - Khd * (xdot - np.cross(omega_b, x))).astype(float)
            sideForce = (sideForce - float(np.dot(sideForce, unitSpring)) * unitSpring).astype(float)

            k = float(self.cfg.swing_kp_z)
            b = float(self.cfg.swing_kd_z)
            force_scalar = -float(k) * float(leg_length - float(l0))
            springForce = (force_scalar * unitSpring - float(b) * springVel).astype(float)

            footForce = (sideForce + springForce).astype(float)
            f_b_cmd = footForce.copy()

            if self._leg_model == "serial":
                # serial plant: use J_body (BASE frame) directly
                f_tau_b = f_b_cmd.copy()
                f_tau_delta = (self.robot2vicon @ f_tau_b.reshape(3)).reshape(3)
                try:
                    if J_body is None:
                        raise RuntimeError("serial Jacobian missing")
                    tau_ref = (np.asarray(J_body, dtype=float).reshape(3, 3).T @ f_b_cmd.reshape(3)).reshape(3)
                    tau_sign = np.asarray(self.cfg.tau_cmd_sign, dtype=float).reshape(3)
                    tau_ref = (tau_sign.reshape(3) * tau_ref.reshape(3)).reshape(3)
                    tau_ref = np.clip(tau_ref, -tau_cmd_max, +tau_cmd_max).astype(float)
                except Exception:
                    tau_ref = None
            else:
                # base->delta
                f_delta_cmd = (self.robot2vicon @ f_b_cmd.reshape(3)).reshape(3)
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
                    # clip to limits
                    tau_ref = np.clip(tau_ref, -tau_cmd_max, +tau_cmd_max).astype(float)
                except Exception:
                    tau_ref = None

        # ===== SRB-QP solve (stance AND flight) =====
        # control_mode: 1=pure_leg, 2=decouple_leg_prop, 3=unified_qp
        _cmode = int(self.cfg.control_mode)
        thrust_sum_max = float(self.mass * self.gravity * float(self.cfg.thrust_total_ratio_max))
        if not bool(props_enabled_ctrl) or _cmode == 1:
            thrust_sum_max = 0.0

        if bool(self._stance):
            F_des = np.asarray(f_ref, dtype=float).reshape(3)

            if _cmode == 3:
                # --- Mode 3: Unified QP (leg + prop in one QP) ---
                stance_thrust_max = float(thrust_max_each_qp) if bool(props_enabled_ctrl) and bool(self.cfg.stance_use_props) else 0.0
                stance_tsum_min = float(thrust_sum_min_stance) if stance_thrust_max > 0.0 else 0.0
                stance_tsum_max = float(thrust_sum_max) if stance_thrust_max > 0.0 else 0.0
                stance_tsum_ref = float(thrust_sum_ref) if stance_thrust_max > 0.0 else 0.0
                base_each = stance_tsum_ref / 3.0
                sol = self.wbc.update_and_solve(
                    m=float(self.mass), g=float(self.gravity), z_w=z_w,
                    r_foot_w=r_foot_w, prop_r_w=prop_r_w,
                    F_des=F_des, Tau_des=Tau_des, in_stance=True,
                    thrust_sum_target=None,
                    thrust_sum_bounds=(stance_tsum_min, stance_tsum_max),
                    thrust_sum_ref=stance_tsum_ref,
                    thrust_max_each=stance_thrust_max,
                    f_ref=f_ref,
                    thrust_ref=np.full(3, base_each, dtype=float),
                    A_tau_f=A_tau_f_3rsr, tau_cmd_max=tau_cmd_max, tau_ref=None,
                )
                status = str(sol.get("status", ""))
                f_contact_w = np.asarray(sol.get("f_foot_w", np.zeros(3)), dtype=float).reshape(3)
                thrusts = np.asarray(sol.get("thrusts", np.zeros(3)), dtype=float).reshape(3)
                tau_qp = np.asarray(sol.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)
                slack = np.asarray(sol.get("slack", np.zeros(6)), dtype=float).reshape(6)

            else:
                # --- Mode 1 & 2: Leg-only QP (props excluded from QP) ---
                sol = self.wbc.update_and_solve(
                    m=float(self.mass), g=float(self.gravity), z_w=z_w,
                    r_foot_w=r_foot_w, prop_r_w=prop_r_w,
                    F_des=F_des, Tau_des=Tau_des, in_stance=True,
                    thrust_sum_target=None,
                    thrust_sum_bounds=(0.0, 0.0),
                    thrust_sum_ref=0.0,
                    thrust_max_each=0.0,
                    f_ref=f_ref,
                    thrust_ref=np.zeros(3, dtype=float),
                    A_tau_f=A_tau_f_3rsr, tau_cmd_max=tau_cmd_max, tau_ref=None,
                )
                status = str(sol.get("status", ""))
                f_contact_w = np.asarray(sol.get("f_foot_w", np.zeros(3)), dtype=float).reshape(3)
                tau_qp = np.asarray(sol.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)
                slack = np.asarray(sol.get("slack", np.zeros(6)), dtype=float).reshape(6)

                if _cmode == 2 and bool(props_enabled_ctrl):
                    # Mode 2: lstsq prop overlay (does NOT touch leg forces)
                    try:
                        tau_contact_leg = np.cross(r_foot_w.reshape(3), f_contact_w.reshape(3)).astype(float)
                        tau_residual = (np.asarray(Tau_des, dtype=float).reshape(3) - tau_contact_leg).astype(float)
                        M_prop = np.column_stack([
                            np.cross(prop_r_w[i].reshape(3), z_w.reshape(3)) for i in range(3)
                        ]).astype(float)
                        thrusts_att = np.linalg.lstsq(M_prop[:2, :], tau_residual[:2], rcond=None)[0].astype(float)
                        base_each = float(thrust_sum_ref) / 3.0
                        t_min = float(self.cfg.wbc_thrust_min_each_n)
                        t_max = float(self.cfg.thrust_max_each_n)
                        thrusts = np.clip(thrusts_att + base_each, t_min, t_max).astype(float).reshape(3)
                    except Exception:
                        thrusts = np.zeros(3, dtype=float)
                else:
                    # Mode 1: pure leg, props OFF
                    thrusts = np.zeros(3, dtype=float)

        else:
            # --- Flight phase (all modes) ---
            if _cmode == 1:
                # Mode 1: pure leg, no props in flight either
                F_des = np.asarray(f_ref, dtype=float).reshape(3)
                sol = self.wbc.update_and_solve(
                    m=float(self.mass), g=float(self.gravity), z_w=z_w,
                    r_foot_w=r_foot_w, prop_r_w=prop_r_w,
                    F_des=F_des, Tau_des=Tau_des, in_stance=False,
                    thrust_sum_target=None,
                    thrust_sum_bounds=(0.0, 0.0),
                    thrust_sum_ref=0.0,
                    thrust_max_each=0.0,
                    f_ref=f_ref, thrust_ref=np.zeros(3, dtype=float),
                    A_tau_f=None, tau_cmd_max=tau_cmd_max, tau_ref=tau_ref,
                )
            else:
                # Mode 2 & 3: unified QP with props in flight
                F_des = (np.asarray(f_ref, dtype=float).reshape(3) + z_w.reshape(3) * float(thrust_sum_ref)).astype(float)
                sol = self.wbc.update_and_solve(
                    m=float(self.mass), g=float(self.gravity), z_w=z_w,
                    r_foot_w=r_foot_w, prop_r_w=prop_r_w,
                    F_des=F_des, Tau_des=Tau_des, in_stance=False,
                    thrust_sum_target=None,
                    thrust_sum_bounds=(0.5 * float(thrust_sum_ref), thrust_sum_max),
                    thrust_sum_ref=float(thrust_sum_ref),
                    thrust_max_each=float(thrust_max_each_qp),
                    f_ref=f_ref, thrust_ref=thrust_ref,
                    A_tau_f=None, tau_cmd_max=tau_cmd_max, tau_ref=tau_ref,
                )
            status = str(sol.get("status", ""))
            f_contact_w = np.asarray(sol.get("f_foot_w", np.zeros(3)), dtype=float).reshape(3)
            thrusts = np.asarray(sol.get("thrusts", np.zeros(3)), dtype=float).reshape(3)
            tau_qp = np.asarray(sol.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)
            slack = np.asarray(sol.get("slack", np.zeros(6)), dtype=float).reshape(6)

        # Extra wrench debug (helps real-robot diagnosis):
        thrust_sum = float(np.sum(thrusts)) if np.all(np.isfinite(thrusts)) else float("nan")
        F_total_w = (f_contact_w + z_w.reshape(3) * thrust_sum).astype(float).reshape(3)
        tau_contact_w = np.cross(r_foot_w.reshape(3), f_contact_w.reshape(3)).astype(float).reshape(3)
        tau_props_w = np.zeros(3, dtype=float)
        try:
            for i in range(3):
                tau_props_w = (tau_props_w + np.cross(prop_r_w[i].reshape(3), (z_w.reshape(3) * float(thrusts[i])).reshape(3))).astype(float)
        except Exception:
            tau_props_w[:] = np.nan
        tau_total_w = (tau_contact_w + tau_props_w).astype(float).reshape(3)

        ok_status = str(status) in ("solved", "solved inaccurate", "solved_inaccurate")
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

        # Debug: in stance, derive the force used for Jacobian->torque mapping from the solved GRF.
        # QP variable `f_contact_w` is the GRF (ground -> robot) in WORLD frame (+Z up).
        # The torque mapping uses the LEG force convention (robot -> ground), which is the opposite of GRF.
        # We expose that force in both BODY (+Z up) and DELTA (+Z down) frames for debugging.
        if bool(self._stance):
            # Apply negative sign to x and y components of contact force for torque mapping
            # (z component keeps the original negative sign from -f_contact_w)
            f_contact_w_for_tau = -f_contact_w.copy()
            f_contact_w_for_tau[0] = -float(f_contact_w_for_tau[0])
            f_contact_w_for_tau[1] = -float(f_contact_w_for_tau[1])
            f_tau_b = (R_wb_hat.T @ f_contact_w_for_tau.reshape(3)).reshape(3)
            f_tau_delta = (self.robot2vicon @ f_tau_b.reshape(3)).reshape(3)

        # final motor torques: scale proportionally to keep direction if any exceeds limit
        tau_qp = np.asarray(tau_qp, dtype=float).reshape(3)
        tau_cmd_max = np.asarray(tau_cmd_max, dtype=float).reshape(3)
        # Find scaling factor: scale = min(1.0, min(tau_cmd_max[i] / abs(tau_qp[i])) for all i)
        scale = 1.0
        for i in range(3):
            if abs(tau_qp[i]) > 1e-9:
                scale_i = float(tau_cmd_max[i]) / abs(float(tau_qp[i]))
                scale = float(min(scale, scale_i))
        tau_cmd = (tau_qp * float(scale)).astype(float)
        self._tau_cmd_prev = tau_cmd.copy()

        # thrust (3 arms) -> 6 PWM (map via prop_pwm_idx_per_arm)
        thrust_motor = np.zeros(6, dtype=float)
        for arm_i in range(3):
            idxs = self._prop_pwm_groups[arm_i]
            t_each = float(thrusts[arm_i]) / float(len(idxs))
            for k in idxs:
                thrust_motor[int(k)] = t_each
        
        # Convert thrust to PWM using selected method
        if bool(self.use_hopper4_pwm):
            # Hopper4-style: pwm = 1000 + sqrt(thrust / k_thrust)
            pwm_us = np.zeros(6, dtype=float)
            for i in range(6):
                thrust_i = float(thrust_motor[i])
                if thrust_i <= 0.0:
                    pwm_us[i] = float(self.cfg.pwm_min_us)
                else:
                    k = float(self.prop_k_thrust)
                    if k > 1e-12:
                        pwm_delta = float(math.sqrt(thrust_i / k))
                        pwm_us[i] = float(self.cfg.pwm_min_us) + pwm_delta
                    else:
                        pwm_us[i] = float(self.cfg.pwm_min_us)
                # Clamp to limits
                pwm_us[i] = float(np.clip(pwm_us[i], float(self.cfg.pwm_min_us), float(self.cfg.pwm_max_us)))
        else:
            # MotorTableModel lookup table
            if self.motor_table is None:
                # Fallback: use Hopper4 method if table not initialized
                pwm_us = np.zeros(6, dtype=float)
                for i in range(6):
                    thrust_i = float(thrust_motor[i])
                    if thrust_i <= 0.0:
                        pwm_us[i] = float(self.cfg.pwm_min_us)
                    else:
                        k = float(self.prop_k_thrust)
                        if k > 1e-12:
                            pwm_delta = float(math.sqrt(thrust_i / k))
                            pwm_us[i] = float(self.cfg.pwm_min_us) + pwm_delta
                        else:
                            pwm_us[i] = float(self.cfg.pwm_min_us)
                    pwm_us[i] = float(np.clip(pwm_us[i], float(self.cfg.pwm_min_us), float(self.cfg.pwm_max_us)))
            else:
                pwm_us = self.motor_table.pwm_from_thrust(thrust_motor).astype(float).reshape(6)

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
            # Foot kinematics:
            # - foot_vicon: delta/vicon frame (+Z DOWN)
            # - foot_b:     body frame (+Z UP)
            "foot_vicon": foot_vicon.copy(),
            "foot_b": foot_b.copy(),
            "foot_vdot_vicon": foot_vdot_vicon.copy(),
            "foot_vrel_b": foot_vrel_b.copy(),
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
            # Debug: gyro actually used by the stance attitude torque controller (BODY frame)
            "omega_b_used": omega_b_used_dbg.copy(),
            # Apex height feedback (for debugging/convergence analysis)
            "z_lo_m": float(self._z_lo) if self._z_lo is not None else float("nan"),
            "vz_lo_m_s": float(self._vz_lo) if self._vz_lo is not None else float("nan"),
            "v_to_cmd_m_s": float(self._v_to_cmd),
            "hop_peak_z_m": float(self.cfg.hop_peak_z),
            # Falling cat debug (recovery gating)
            # MPC debug
            "mpc_status": mpc_status,
            "mpc_u0": mpc_u0.copy(),
        }

        return tau_cmd, pwm_us, info


