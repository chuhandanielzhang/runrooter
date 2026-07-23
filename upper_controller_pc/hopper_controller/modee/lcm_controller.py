from __future__ import annotations

import csv
import os
import sys
import time
import threading
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import lcm

# Make LCM python types importable (same pattern as Hopper4.py)
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_LCM_TYPES_DIR = os.path.join(_CUR_DIR, "..", "..", "hopper_lcm_types", "lcm_types")
sys.path.append(_LCM_TYPES_DIR)

from python.hopper_data_lcmt import hopper_data_lcmt  # type: ignore
from python.hopper_cmd_lcmt import hopper_cmd_lcmt  # type: ignore
from python.hopper_imu_lcmt import hopper_imu_lcmt  # type: ignore
from python.gamepad_lcmt import gamepad_lcmt  # type: ignore
from python.motor_pwm_lcmt import motor_pwm_lcmt  # type: ignore
from python.hopper_odom_lcmt import hopper_odom_lcmt  # type: ignore
from python.hopper_nav_cmd_lcmt import hopper_nav_cmd_lcmt  # type: ignore
from python.wheel_cmd_lcmt import wheel_cmd_lcmt  # type: ignore

from modee.core import ModeECore, ModeEConfig


def _quat_wxyz_to_R_wb(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> rotation matrix R_wb (body->world)."""
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # normalize (avoid drift / bad packets)
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n > 1e-12:
        w, x, y, z = w / n, x / n, y / n, z / n
    # standard quaternion rotation (right-handed)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


@dataclass
class ModeELCMConfig:
    lcm_url: str = "udpm://239.255.76.67:7667?ttl=255"
    # 2026-07-11 per user (MATLAB values): posvellim = 0.15 m/s. MATLAB's
    # position loop clamps desiredVel to +-0.15; we have no position loop, so
    # the equivalent is capping the stick/nav v_des at 0.15 (was 0.8).
    max_cmd_vel: float = 0.15
    stick_deadzone: float = 0.10
    # ---- LiDAR patrol (hopper_nav_cmd_lcmt from lidar_perception/patrol.py) ----
    # SELECT toggles patrol; while engaged the nav velocity replaces the stick.
    # ANY stick input beyond the deadzone (or B) immediately disengages.
    # Nav command staleness gate: older than this -> treat as inactive (robot
    # falls back to zero-velocity stick behavior, patrol stays engaged).
    nav_cmd_stale_s: float = 0.3
    nav_cmd_vel_max: float = 0.15  # hard cap on patrol velocity (matches max_cmd_vel/posvellim)
    # Print rate (Hz). Default 5: status line only; control loop stays 500 Hz (dt=0.002).
    # Set <=0 to print every control step (500 lines/s — usually too fast).
    print_hz: float = 5.0
    # Output-side safety (does NOT affect ModeECore/QP internals):
    # - tau_out_scale: multiply final motor torques by this factor before sending (e.g. 0.1 for bring-up)
    # - tau_out_max_nm: absolute per-joint max torque sent to hardware (Nm); applied after scaling
    tau_out_scale: float = 1.0
    # Default to a conservative output torque cap for bring-up safety.
    # Increase gradually (e.g., 2, 3, 5...) as confidence grows.
    tau_out_max_nm: float | None = 30
    # SAFE flag:
    # - If triggered, we request hopper_driver to enter DAMP (same as pressing B),
    #   and pause the Python controller loop for a few seconds.
    safe_rp_deg: float = 50.0
    safe_q_min: float = -0.8553   # -49 deg (normal mechanical retract limit)
    safe_q_max: float = 1.4835  # +85 deg (mechanical extend limit; matches kAk60LcmQOffsetRad)
    # Whole-robot gait switch:
    # - OFF/DAMP + LT: no actuator motion; current RM pose is labeled q=+11.5
    #   and MOBILE is selected.
    # - OFF/DAMP + RT: no actuator motion; current RM pose is labeled q=0 and
    #   HOPPING is selected.
    # - enabled HOPPING + LT: at the next liftoff, stop legs/props and drive
    #   the three folding-arm RM2006s from 0 to +11.5, entering MOBILE.
    # - enabled MOBILE + RT: P4 holds a world-vertical foot (quaternion) at
    #   switch_rb_leg_len_m with torque cap 1 Nm; props forced ON at the
    #   switch_rb_prop_base_pwm_us baseline. After the stand delay ModeE
    #   resumes and pushes with the hop_prop_base_pwm_us FLOOR already open
    #   (2026-07-23 21:15: no gap); at the FIRST liftoff (push end) RM
    #   drives +11.5 -> 0 and the floor holds until RM reaches 0.
    # SAFE q-range guard is relaxed while legs are force-free / in P4:
    safe_q_min_switch: float = -1.0472    # -60 deg (SAFE guard only, in LT/P4 modes)
    # Enabled MOBILE + RT: stand up, then enter HOPPING (Phase 4).
    #   - legs: FLIGHT swing Cartesian PD toward a WORLD-vertical foot at
    #     switch_rb_leg_len_m (quaternion, always perpendicular to ground),
    #     torque hard-capped at switch_rb_tau_max_nm (1 Nm).
    #   - props: forced ON at the UPWARD baseline switch_rb_prop_base_pwm_us
    #     (1100, >1000 = forward thrust) for the whole phase;
    #   - after switch_rb_pushdelay_s ModeE takes over legs and begins the
    #     hopping push; RM stays at +11.5 until that push ends (liftoff),
    #     then drives +11.5 -> 0. Prop floor hop_prop_base_pwm_us opens at
    #     the hopping handoff (covers push + flight) and is released only
    #     when RM reaches 0 (2026-07-23 21:15, continuous coverage).
    # 2026-07-23 09:23 user: 0.428 -> 0.46; 2026-07-24 06:47: 0.46 -> 0.47.
    switch_rb_leg_len_m: float = 0.47      # RT/P4 / LT stand target leg length (m)
    switch_rb_tau_max_nm: float = 1.0       # RT leg torque cap (Nm)
    switch_rb_pushdelay_s: float = 1.0      # stand this long, then enter hopping (push)
    # First hop out of RT: temporarily override leg_l0_m so the spring push
    # extends toward full leg (~0.554 m at AK60 setZero / q_lcm=1.4835).
    # Restored at the first liftoff. 2026-07-24 03:52: 0.53 -> 0.54 (~1.4 cm
    # margin below full extension; stroke from 0.47 stand ~= 7 cm).
    switch_rb_first_hop_l0_m: float = 0.54
    # 2026-07-23 09:23 user: 1100 -> 1200 (more prop support during the stand;
    # ~1.8 N/arm at the calibrated k=2.25e-5, ~8% of body weight total).
    switch_rb_prop_base_pwm_us: float = 1200.0  # prop baseline during the RT stand (us)
    # RT transition prop support (ENTER HOPPING -> RM unfold done), expressed
    # as the PWM whose thrust the collective must deliver. 2026-07-24 01:36:
    # NO LONGER a per-motor PWM floor. The old max(pwm, 1600) clamp pinned
    # arms at the floor and cut ModeE's differential attitude authority in
    # half -- the 01:17 log shows pwm1/pwm2 stuck at 1600 with 40+ command
    # reversals on the free arms (visible jitter) while pitch ran +3.5 ->
    # -12 deg unchecked through flight. Now the equivalent COLLECTIVE thrust
    # (3*k*(pwm-1000)^2, = 24.3 N ~ 37% weight at 1600/k=2.25e-5) is fed into
    # ModeE as a prop_*_base_thrust_ratio override, so the allocator
    # differentiates attitude moments AROUND the support and any arm may dip
    # below 1600 when a moment needs it. Same window, same total lift.
    hop_prop_base_pwm_us: float = 1600.0
    # TEMP: RT path legs-only bring-up. When True, RT never arms props and never
    # drives RM (+11.5->0). Flip back to False once the P4 leg hold looks good.
    rt_leg_only_no_prop_rm: bool = False
    # ---- LT (HOPPING -> MOBILE) stand-and-fold (2026-07-24 06:41) ----
    # Old LT: at the next liftoff the legs went force-free MID-AIR and the
    # robot dropped onto the folding arms.  New LT: at the liftoff edge of
    # the push AFTER the LT press (push completes normally), the leg
    # immediately holds a P4-style stand -- world-vertical foot at
    # switch_rb_leg_len_m (the same length RT starts from), props OFF --
    # so the robot lands on the held leg like a SLIP pendulum.  The RM
    # arms fold 0 -> +11.5 from that same moment; when they reach the
    # mobile pose the wheels are down (support polygon formed) and only
    # then does the leg release into MOBILE.
    switch_lt_stand: bool = True
    # Torque cap while the leg carries the robot.  RT/P4 uses 1 Nm since
    # the wheels carry the weight there; holding ~72 N at 0.47 m needs
    # ~5 Nm, so 6 gives margin without unlocking hop-level forces.
    switch_lt_tau_max_nm: float = 6.0
    # Fail-safe: if RM never reports "reached", release to MOBILE anyway.
    switch_lt_timeout_s: float = 4.0
    # ---- RT (MOBILE -> HOPPING) stand-and-unfold (2026-07-24 07:05) ----
    # Old RT: P4 place -> first-hop spring push -> RM unfolds at the first
    # liftoff (prop window bridging the gap).  New RT, per user: P4 places
    # the leg exactly as before; after switch_rb_pushdelay_s the leg HOLDS
    # its length UNCAPPED (inverted-pendulum stand, leg carries the robot)
    # while the RM arms unfold +11.5 -> 0 SYNCHRONIZED; when they reach 0
    # the controller enters plain normal hopping (FLIGHT phase first --
    # no first-hop spring, no l0 override).
    # 10 Nm = the AK60 driver's own clamp, i.e. effectively uncapped.
    switch_rt_stand_tau_max_nm: float = 10.0
    switch_rt_stand_timeout_s: float = 4.0
    # Propeller SOFT-START: rate-limit how fast prop PWM may RISE (us per second). A hard
    # 1000->1200 jump spins up all props at once -> big ESC inrush current -> battery sag ->
    # brownout / CAN bus-off. Ramping the rise cuts the peak current. Falling PWM (spin-down/
    # stop) is applied IMMEDIATELY (safety). e.g. 400 us/s -> 1000->1200 takes ~0.5s.
    # Set to a very large value (e.g. 1e12) to effectively disable the ramp.
    # NOTE (2026-06): 400 us/s was throttling the prop ATTITUDE control bandwidth -> lag +
    # overshoot + "not responsive" (only rising was limited, so authority was asymmetric).
    # Raised to 8000 us/s now that the FC/ESC has independent power: essentially unthrottled
    # for closed-loop control, while still clipping the very worst startup inrush spike.
    # NOTE (2026-06-23): lowered to 6000 us/s (~12 us/frame @ 500 Hz) for a SMOOTHER prop
    # trace -- the user wants gentle, non-jittery PWM. This rate-limits how fast PWM can
    # climb so attitude-loop spikes/noise don't translate into abrupt prop jumps. (Falling
    # PWM is still immediate for safety.) If response feels too laggy, raise back toward
    # 15000-30000.
    # RESTORED (2026-06-24): the original CASE/Cao controller had NO prop slew limiter at
    # all (instant PWM). Set very high to effectively disable up-rate limiting so prop
    # response is not weakened. TRADE-OFF: removes the ESC inrush soft-start that guarded
    # against battery sag -> brownout / CAN bus-off on hard 1000->base jumps. If brownouts
    # return, drop this back to ~15000-30000 (still fast, but caps the initial arm surge).
    prop_slew_up_us_per_s: float = 1.0e7
    # Propeller master switch via gamepad: A = props ON, B = props (and legs) OFF.
    # On/off is carried in motor_pwm_lcmt.control_mode (NOT by zeroing pwm), so the real
    # pwm_values stay visible in lcm-spy even while props are OFF:
    #   control_mode = prop_ctrl_mode_on  (3) -> props ON  (px4_bridge spins per pwm_values)
    #   control_mode = prop_ctrl_mode_off (1) -> props OFF (bridge idle, data still flows)
    #   control_mode = -1                      -> SAFE/DAMP (legs damp, props off)
    prop_ctrl_mode_on: int = 3
    prop_ctrl_mode_off: int = 1
    safe_pause_s: float = 5.0
    safe_damp_kd: float = 0.8

    # AK60-side damping (MIT Kd gain):
    # - This is **motor-internal viscous damping**: tau += kd * (qd_des - qd_motor).
    # - In ModeE we normally send only tau_ff with kp=kd=0 (pure torque).
    # - Setting this > 0 helps reduce flight-phase jitter/oscillation, because it dissipates energy using
    #   the motor's own high-rate velocity estimate (not the Python/Lcm qd).
    # - Applied per-phase (FLIGHT vs STANCE) so you can add a small amount in stance too.
    ak60_flight_damp_kd: float = 0
    # 2026-07-19 stance anti-jitter: small motor-side damping using the
    # driver's own high-rate velocity (much cleaner than the ~230 Hz LCM qd).
    ak60_stance_damp_kd: float = 0

    # ===== Command shaping / demo mode =====
    # To keep the hop process smooth, we rate-limit the commanded desired velocity.
    # This prevents sudden step changes in Raibert target_xy (and resulting speed jumps).
    cmd_dv_max_mps2: float = 0.0
    # Simple demo: override desired velocity to a fixed value.
    # Keep disabled by default so user/gamepad velocity command directly drives Raibert foot placement.
    demo_enable: bool = False
    demo_vx_mps: float = 0.0
    demo_vy_mps: float = 0.0  # Zero velocity - stationary hopping with Raibert stabilization

    # Motor velocity: Mode1 uses hopper_data_lcmt.qd (AK60 CAN estimate) as
    # the input to the controller-side EMA.

    # ===== RM M2006 folding arms (3x, output-shaft rad) =====================
    # Fixed logical endpoints; all three motors use the same sign:
    #   HOPPING = 0 rad, MOBILE = +11.5 rad.
    # The driver owns the coordinate offset. rm_set_zero + rm_zero_at_rad can
    # label the current physical pose as either endpoint without energizing a
    # motor (OFF/DAMP initialization). Enabled transitions use current-mode PD.
    rm_hopping_rad: float = 0.0
    rm_mobile_rad: float = 11.5
    rm_kp_a_per_rad: float = 2.0         # current-mode PD: A per rad
    rm_kd_a_per_rad_s: float = 0.2       # current-mode PD: A per rad/s
    rm_iq_max_a: float = 5.0             # |current| cap during the drive (A)
    rm_reach_tol_rad: float = 0.3        # "in place" tolerance (all 3 motors)
    # Synchronized fold/unfold -- TEMP DISABLED 2026-07-24 04:00 (user).
    # Was: friction differs per arm, so independent PDs let fast motors finish
    # while a slow one lags. Stage-2 drove a shared reference that never leads
    # the SLOWEST motor by more than this many rad. Set >0 to re-enable.
    # kp*lead = 2.0*1.5 = 3 A on the slowest arm (cap is 5 A).
    # 2026-07-24 07:05 user: sync ON -- fast arms wait for the slowest.
    rm_sync_lead_rad: float = 1.0        # 0 = independent PD (sync off)
    rm_hold_s: float = 1.0               # hold at endpoint before idling (s)
    # Station-keeping at the HOPPING endpoint (2026-07-23 user request):
    # while gait==hopping AND legs are PD/PWMPD (and no fold transition is
    # running), keep a continuous current-mode PD on 0 rad so hop impacts
    # cannot back-drive the folding arms. Thermal-safe by construction: at
    # q=0 the PD current is ~0 A (heat ~ i^2, only disturbance transients
    # draw current), and the cap is well under the M2006 continuous rating.
    rm_hopping_hold: bool = True
    rm_hold_iq_max_a: float = 2.0        # |current| cap for the station hold (A)

    # ===== MOBILE: kiwi drive, 3 hub-motor wheels at 120 deg ================
    # Chassis (top view, body +x forward): the wheels sit evenly spaced on
    # a circle of radius wheel_base_radius_m, each hub axis pointing at the
    # chassis center, so wheel i drives along the tangent
    #   t_i = (-sin(az_i), +cos(az_i))
    # and side slip rides on the omni rollers. Inverse kinematics from the
    # body twist (vx, vy [m/s], wz [rad/s, CCW+ seen from above]):
    #   v_i     = -sin(az_i)*vx + cos(az_i)*vy + R*wz     (rim speed, m/s)
    #   omega_i = sign_i * v_i / r_wheel                   (rad/s, to LCM)
    wheel_azimuth_deg: tuple = (0.0, 120.0, 240.0)
    wheel_base_radius_m: float = 0.20    # center -> wheel contact point
    wheel_radius_m: float = 0.05         # hub wheel rolling radius
    # Per-wheel sign to absorb wiring/mounting direction; calibrate with an
    # on-robot spin test (same procedure as the prop mapping).
    wheel_dir_sign: tuple = (1.0, 1.0, 1.0)
    # Stick full scale: RIGHT stick = translation (same axis convention as
    # hopping desired_v_xy), LEFT stick X = yaw rate. Per-wheel speed cap
    # scales the whole twist uniformly so the direction is preserved.
    mobile_v_max_mps: float = 0.8
    mobile_wz_max_rad_s: float = 1.5
    # DM-H6215: rated 120 rpm = 12.6 rad/s, no-load 320 rpm = 33.5 rad/s.
    # Cap between the two -- brief transients above rated are fine, sustained
    # driving should stay near it (drop mobile_v_max_mps if wheels run hot).
    wheel_speed_max_rad_s: float = 25.0


class ModeELCMController:
    """
    Real-robot runner:
      hopper_data_lcmt + hopper_imu_lcmt (+ gamepad_lcmt) -> ModeECore -> hopper_cmd_lcmt + motor_pwm_lcmt

    Note: This controller always sends commands. The underlying hopper_driver handles mode switching
    and safety (OFF/DAMP/PD/PWMPD modes). Python layer doesn't need to manage ARM/DISARM.
    """

    def __init__(self, *, modee_cfg: ModeEConfig | None = None, lcm_cfg: ModeELCMConfig | None = None):
        self.modee_cfg = ModeEConfig() if modee_cfg is None else modee_cfg
        self.lcm_cfg = ModeELCMConfig() if lcm_cfg is None else lcm_cfg

        self.core = ModeECore(self.modee_cfg)
        self.lc = lcm.LCM(self.lcm_cfg.lcm_url)

        self.lock = threading.Lock()
        self.running = True

        self.robot_state = {
            "q": np.zeros(3),
            "qd": np.zeros(3),
            "tau": np.zeros(3),
            # RM M2006/C610 motors (relayed by the Jetson driver inside
            # hopper_data_lcmt; leg-class actuators on the Pixhawk CAN).
            "rm_q": np.zeros(3),
            "rm_qd": np.zeros(3),
            "rm_iq": np.zeros(3),
            "rm_online": 0,
            "imu_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "imu_gyro": np.zeros(3),
            "imu_acc": np.zeros(3),
            "imu_rpy": np.zeros(3),
            "gamepad": None,
            "have_motor": False,
            "have_imu": False,
            # LiDAR odometry (hopper_odom_lcmt) -- fed to core.update_lidar_odom
            "odom_pos": np.zeros(3),
            "odom_yaw": 0.0,
            "odom_quality": 0,
            "odom_rx_t": 0.0,
            # Patrol nav command (hopper_nav_cmd_lcmt)
            "nav_v_xy": np.zeros(2),
            "nav_active": 0,
            "nav_wp_index": -1,
            "nav_rx_t": 0.0,
        }

        # CSV logger (starts on gamepad Y press; stops when program exits / B)
        self._log_enabled = False
        self._log_fp = None

        # Desired velocity command smoothing (rate limiter)
        self._v_cmd_filt = np.zeros(2, dtype=float)
        self._log_writer = None
        self._log_path = None
        self._log_latest_path = None
        self._log_last_flush_t = 0.0
        self._log_rows = 0

        # PWM filter (same as Hopper4 for propeller stability)
        # Low-pass filter: smoothed_pwm = alpha * new_pwm + (1-alpha) * prev_pwm
        # User request: no PWM smoothing; send prop commands immediately.
        self.pwm_filter_alpha = 1.0
        self.prev_pwm_us = np.zeros(6, dtype=float) + float(self.modee_cfg.pwm_min_us)
        # Timestamp of last prop PWM publish, for the soft-start (rise-rate) limiter.
        self._last_pwm_pub_t: float | None = None
        self._last_y = False
        self._last_b = False
        # User "hard stop" latch (maps to gamepad `point` button; user calls it "I").
        self._zero_vel_hold = False
        self._last_point = False
        # RB one-shot big jump trigger (execute at next touchdown).
        self._last_rb = False
        # User request: pressing Y should enable the "hard hold" until we ENTER STANCE once.
        # This is useful when the robot is being held in the air: it prevents IMU drift from moving the foot target.
        self._y_hold_until_stance = False
        # Estimate the Pi-side driver mode from gamepad button edges (mirror Hopper-aero/main.cpp).
        # This is used to gate the SAFE latch so it only triggers in PD/PWMPD.
        self._mode_est: int = 0  # OFF
        self._mode_last_b: bool = False
        self._mode_last_x: bool = False
        self._mode_last_a: bool = False
        self._mode_last_start: bool = False
        # RT (P4) stand mode: while active, ignore ModeE legs and hold a
        # world-vertical foot at switch_rb_leg_len_m via flight swing PD
        # (torque-capped) with props forced ON at the 1100 baseline; exits
        # into hopping after switch_rb_pushdelay_s. TEMP leg-only keeps props off.
        self._switch_loop: bool = False
        self._mode_last_lb: bool = False
        self._mode_last_rb: bool = False
        # RB/RT phase-4 start time (fixed stand, then hop entry).
        self._rb_p4_t0: float | None = None
        # (legacy name kept) unused for liftoff-window; prop floor is gated by
        # rm_stage==2 below. Cleared on B / MOBILE entry.
        self._hop_prop_base_active: bool = False
        self._hop_prop_base_liftoff_count: int = 0
        # Explicit mechanical configuration. HOPPING is the backward-compatible
        # startup state; MOBILE always means free legs + stopped propellers.
        self._gait_mode: str = "hopping"
        # A button: standalone propeller master switch (normal mode, outside switch loop).
        self._prop_enable: bool = False
        # RM M2006 desired torque current (A); sent inside every hopper_cmd_lcmt.
        self._rm_iq_des = np.zeros(3, dtype=float)
        # RM sequence: 0 idle, 1 reserved settle, 2 drive, 3 endpoint hold.
        self._rm_stage: int = 0
        # Start poses latched on the first stage-2 tick; the synchronized
        # drive measures per-arm progress against these (see rm_sync_lead_rad).
        self._rm_sync_q0: np.ndarray | None = None
        self._rm_target: float = 0.0
        self._rm_hold_t0: float = 0.0
        # LT armed flag: set at the LT press, consumed at the next liftoff edge.
        self._rm_lt_pending: bool = False
        # LT stand-and-fold: wall time the stand began (None = inactive).
        # Entered on the liftoff edge of the push after the LT press.
        self._lt_stand_t0: float | None = None
        # RT stand-and-unfold: wall time the uncapped stand began (None =
        # inactive).  Entered when P4 finishes its pushdelay.
        self._rt_stand_t0: float | None = None
        # RT armed flag: set when P4 hands off to HOPPING; RM +11.5->0 starts
        # only at the next liftoff (push end), not during the push itself.
        self._rm_rt_pending: bool = False
        # RT first-hop l0 override: original leg_l0_m saved here while the
        # first post-RT push runs with switch_rb_first_hop_l0_m; restored at
        # the first liftoff (or on B abort).
        self._rt_l0_restore: float | None = None
        # Saved (prop_base_thrust_ratio, prop_stance_base_thrust_ratio) while
        # the RT prop-base window overrides them (see hop_prop_base_pwm_us).
        self._prop_ratio_restore: tuple | None = None
        # Logical-position initialization pulse. On the edge, the driver labels
        # the current physical RM pose as _rm_zero_at_rad without moving it.
        self._rm_zero_until: float = 0.0
        self._rm_zero_at_rad: float = 0.0

        # Patrol engage flag (SELECT toggles; stick/B disengages).
        self._patrol_enable: bool = False
        self._last_select: bool = False

        self.lc.subscribe("hopper_data_lcmt", self._handle_robot_data)
        self.lc.subscribe("hopper_imu_lcmt", self._handle_imu_data)
        self.lc.subscribe("gamepad_lcmt", self._handle_gamepad_data)
        self.lc.subscribe("hopper_odom_lcmt", self._handle_odom_data)
        self.lc.subscribe("hopper_nav_cmd_lcmt", self._handle_nav_cmd)

        # SAFE latch (upper-layer)
        self._safe_until = 0.0
        self._safe_last_t = 0.0
        # LCM receive rate estimate (for status print)
        self._rx_motor_n = 0
        self._rx_imu_n = 0
        self._rx_rate_t0 = time.time()
        self._rx_motor_hz = 0.0
        self._rx_imu_hz = 0.0
        # Timestamps of the LAST received packet per channel (link-health debug).
        # Logged as *_age_ms so a frozen lower layer / dead IMU bridge is visible in the CSV.
        self._rx_motor_last_t = 0.0
        self._rx_imu_last_t = 0.0

    def _update_mode_est(self, gamepad_msg) -> None:
        """
        Mirror Pi-side mode switching in `Hopper-aero/main.cpp`:
          B -> DAMP
          X -> PD
          A toggles props only when legs are already PD/PWMPD
          START (when in DAMP) -> OFF
        """
        # Mode constants (keep in sync with Hopper-aero/main.cpp)
        OFF, DAMP, PD, PWMPD = 0, 1, 2, 3

        b_now = bool(getattr(gamepad_msg, "b", 0)) if gamepad_msg is not None else False
        x_now = bool(getattr(gamepad_msg, "x", 0)) if gamepad_msg is not None else False
        a_now = bool(getattr(gamepad_msg, "a", 0)) if gamepad_msg is not None else False
        start_now = bool(getattr(gamepad_msg, "start", 0)) if gamepad_msg is not None else False
        # LT/RT analog triggers (driver only fills leftTriggerAnalog/rightTriggerAnalog
        # in [0,1], so threshold at 0.5). 2026-07-07 rework: the old LB leg-retraction
        # parking loop is DELETED; LT/RT drive the RM M2006 jobs.
        lb_now = (float(getattr(gamepad_msg, "leftTriggerAnalog", 0.0)) > 0.5) if gamepad_msg is not None else False
        rb_now = (float(getattr(gamepad_msg, "rightTriggerAnalog", 0.0)) > 0.5) if gamepad_msg is not None else False

        enabled = int(self._mode_est) in (PD, PWMPD)

        # LT selects MOBILE while disabled, and requests the
        # HOPPING->MOBILE transition while enabled.
        if bool(lb_now) and (not bool(self._mode_last_lb)):
            if not enabled:
                self._gait_mode = "mobile"
                self._switch_loop = False
                self._rm_lt_pending = False
                self._rm_rt_pending = False
                self._lt_stand_t0 = None
                self._rm_stage = 0
                self._rm_iq_des[:] = 0.0
                self._rm_set_logical_position(
                    float(self.lcm_cfg.rm_mobile_rad),
                    "LT disabled: select MOBILE",
                )
            elif self._gait_mode != "hopping":
                print("[gait] LT ignored (already MOBILE; use RT to enter HOPPING)")
            elif bool(self._rm_lt_pending) or bool(self._rm_rt_pending) or int(self._rm_stage) != 0:
                print("[gait] LT ignored (fold transition already pending/active)")
            else:
                self._rm_lt_pending = True
                if bool(getattr(self.lcm_cfg, "switch_lt_stand", True)):
                    print(
                        "[gait] LT ARMED -> hop normally; at the next push "
                        "complete: hold leg %.3f m vertical (props OFF), "
                        "fold RM 0 -> %.1f rad, then MOBILE"
                        % (float(self.lcm_cfg.switch_rb_leg_len_m),
                           float(self.lcm_cfg.rm_mobile_rad))
                    )
                else:
                    print(
                        "[gait] LT ARMED -> next push end: legs/props OFF, "
                        "RM 0 -> %.1f rad, enter MOBILE"
                        % float(self.lcm_cfg.rm_mobile_rad)
                    )
        self._mode_last_lb = bool(lb_now)

        if bool(rb_now) and (not bool(self._mode_last_rb)):
            # RT selects HOPPING while disabled. While enabled, only MOBILE may
            # start P4 and the MOBILE->HOPPING transition.
            if not enabled:
                self._rt_stand_t0 = None
                self._gait_mode = "hopping"
                self._switch_loop = False
                self._rm_lt_pending = False
                self._rm_rt_pending = False
                self._rm_stage = 0
                self._rm_iq_des[:] = 0.0
                self._rm_set_logical_position(
                    float(self.lcm_cfg.rm_hopping_rad),
                    "RT disabled: select HOPPING",
                )
            elif self._gait_mode != "mobile":
                print("[gait] RT ignored (already HOPPING; use LT to enter MOBILE)")
            elif bool(self._switch_loop):
                print("[switch_loop] RT ignored (P4 stand already active)")
            else:
                self._switch_loop = True
                self._rb_p4_t0 = time.time()
                self._rm_lt_pending = False
                self._rm_rt_pending = False
                self._rm_stage = 0
                self._rm_iq_des[:] = 0.0
                # RT normally forces props ON for P4 + first hop. TEMP leg-only
                # bring-up: leave props off so only the stand swing PD is visible.
                if not bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                    self._prop_enable = True
                print(
                    "[switch_loop] RT MOBILE -> P4 world-vertical L=%.3fm "
                    "(cap %.1fNm)%s; enter HOPPING after %.1fs"
                    % (
                        float(self.lcm_cfg.switch_rb_leg_len_m),
                        float(self.lcm_cfg.switch_rb_tau_max_nm),
                        (" + props OFF (leg-only)" if bool(self.lcm_cfg.rt_leg_only_no_prop_rm)
                         else (" + props %.0fus" % float(self.lcm_cfg.switch_rb_prop_base_pwm_us))),
                        float(self.lcm_cfg.switch_rb_pushdelay_s),
                    )
                )
        self._mode_last_rb = bool(rb_now)

        # Edge-triggered transitions (same priority order as Pi)
        # B = stop BOTH legs (DAMP) and props. Always clears props on the B edge.
        if bool(b_now) and (not bool(self._mode_last_b)):
            self._prop_enable = False
            self._close_prop_base_window("B abort")   # B stops everything
            # B aborts every powered transition. The selected gait is retained
            # so OFF/DAMP can later be armed into the intended configuration.
            self._rm_stage = 0
            self._rm_lt_pending = False
            self._rm_rt_pending = False
            self._lt_stand_t0 = None
            self._rt_stand_t0 = None
            self._switch_loop = False
            self._rb_p4_t0 = None
            self._restore_rt_first_hop_l0()
            self._rm_iq_des = np.zeros(3, dtype=float)
            self._rm_zero_until = 0.0
            if int(self._mode_est) != DAMP:
                self._mode_est = DAMP
            print("[prop] OFF (B) -> legs DAMP + props stop (control_mode=%d)"
                  % int(self.lcm_cfg.prop_ctrl_mode_off))
        elif bool(x_now) and (not bool(self._mode_last_x)) and (int(self._mode_est) != PD):
            self._mode_est = PD
            self._prop_enable = False
        elif bool(a_now) and (not bool(self._mode_last_a)):
            if int(self._mode_est) == PD:
                self._mode_est = PWMPD
                self._prop_enable = True
                print(
                    "[prop] ON (A): PD -> PWMPD (control_mode=%d)"
                    % int(self.lcm_cfg.prop_ctrl_mode_on)
                )
            elif int(self._mode_est) == PWMPD:
                self._mode_est = PD
                self._prop_enable = False
                print("[prop] OFF (A): PWMPD -> PD")
            else:
                self._prop_enable = False
                print("[prop] A ignored: press X first")
        elif bool(start_now) and (not bool(self._mode_last_start)) and (int(self._mode_est) == DAMP):
            self._mode_est = OFF

        self._mode_last_b = bool(b_now)
        self._mode_last_x = bool(x_now)
        self._mode_last_a = bool(a_now)
        self._mode_last_start = bool(start_now)

    def _restore_rt_first_hop_l0(self) -> None:
        """Undo the RT first-hop spring + l0 overrides (idempotent)."""
        self.core.cfg.rt_first_hop_spring_active = False
        if self._rt_l0_restore is not None:
            self.core.cfg.leg_l0_m = float(self._rt_l0_restore)
            self._rt_l0_restore = None
            print("[switch_loop] first-hop l0 restored -> %.3f m"
                  % float(self.core.cfg.leg_l0_m))

    def _open_prop_base_window(self) -> None:
        """RT prop support as a COLLECTIVE inside ModeE (no PWM clamping).

        Converts hop_prop_base_pwm_us into total thrust (3*k*(pwm-1000)^2)
        and overrides ModeE's prop_base_thrust_ratio (flight) and
        prop_stance_base_thrust_ratio (stance) so the allocator carries the
        support while keeping FULL two-sided attitude authority. The old
        per-motor max(pwm, base) clamp is gone -- it pinned arms at the floor
        and halved the available moment (01:17 log jitter + pitch runaway).
        """
        k = float(self.modee_cfg.prop_k_thrust)
        d = max(0.0, float(self.lcm_cfg.hop_prop_base_pwm_us) - 1000.0)
        mg = float(max(1e-6, float(self.core.mass) * float(self.core.gravity)))
        ratio = float(np.clip(3.0 * k * d * d / mg, 0.0, 0.5))
        if self._prop_ratio_restore is None:
            self._prop_ratio_restore = (
                float(self.core.cfg.prop_base_thrust_ratio),
                float(self.core.cfg.prop_stance_base_thrust_ratio),
            )
        self.core.cfg.prop_base_thrust_ratio = ratio
        self.core.cfg.prop_stance_base_thrust_ratio = max(
            ratio, float(self.core.cfg.prop_stance_base_thrust_ratio)
        )
        self._hop_prop_base_active = True
        print(
            "[prop] base window OPEN: collective %.1f N (%.0f%% weight, "
            "= %.0f us/arm) via ModeE ratio override"
            % (ratio * mg, 100.0 * ratio,
               float(self.lcm_cfg.hop_prop_base_pwm_us))
        )

    def _close_prop_base_window(self, reason: str) -> None:
        """Undo the RT prop-base ratio override (idempotent)."""
        was_active = bool(self._hop_prop_base_active)
        self._hop_prop_base_active = False
        if self._prop_ratio_restore is not None:
            base_fl, base_st = self._prop_ratio_restore
            self.core.cfg.prop_base_thrust_ratio = float(base_fl)
            self.core.cfg.prop_stance_base_thrust_ratio = float(base_st)
            self._prop_ratio_restore = None
            if was_active:
                print("[prop] base window CLOSED (%s) -> ModeE prop control"
                      % reason)

    def _enter_hop_from_rb(self) -> None:
        """LEGACY (unused since 2026-07-24 07:05): spring-takeoff RT handoff.

        Replaced by the RT stand-and-unfold flow (_rt_stand_t0): P4 places
        the leg, then the leg holds its length uncapped while RM unfolds
        synchronized, and hopping starts in plain FLIGHT with no first-hop
        spring / l0 override.  Kept for reference / quick rollback.

        Old behavior: enter HOPPING with one plain spring takeoff.

        RM +11.5 -> 0 is armed here and starts at the next liftoff. The first
        stance bypasses FB-SLIP and releases a MATLAB-style virtual spring from
        the static P4 pose to switch_rb_first_hop_l0_m. At liftoff the spring
        mode and l0 override are cleared; hop 2+ uses normal FB-SLIP.

        TEMP: with rt_leg_only_no_prop_rm, skip prop arming and RM arming."""
        self._switch_loop = False
        self._rb_p4_t0 = None
        self._gait_mode = "hopping"
        # First-hop simple spring + l0 override (this RT loop's push only).
        if self._rt_l0_restore is None:
            self._rt_l0_restore = float(self.core.cfg.leg_l0_m)
        self.core.cfg.leg_l0_m = float(self.lcm_cfg.switch_rb_first_hop_l0_m)
        self.core.cfg.rt_first_hop_spring_active = True
        # The in-flight leg retraction (first-hop l0 -> normal l0) stretches
        # this hop's flight arc, so its eta measurement is invalid; skip it.
        self.core._eta_skip_once = True
        leg_only = bool(self.lcm_cfg.rt_leg_only_no_prop_rm)
        # 2026-07-23 21:15 (user): the prop base must cover the WHOLE
        # transition with no gap -- P4 stand (1200 forced) -> first push ->
        # flight -> RM unfold -- and release only after RM reaches 0.
        # Applied as a ModeE collective-ratio override (see
        # _open_prop_base_window); cleared when the RT RM drive finishes.
        if not leg_only:
            self._open_prop_base_window()
        else:
            self._close_prop_base_window("leg-only RT")
        self._hop_prop_base_liftoff_count = 0
        if not leg_only:
            self._prop_enable = True
            self._rm_rt_pending = True
        else:
            self._prop_enable = False
            self._rm_rt_pending = False
        print(
            "[switch_loop] ENTER HOPPING: plain spring first takeoff, "
            "l0=%.3f m k=%.0f N/m%s"
            % (
                float(self.lcm_cfg.switch_rb_first_hop_l0_m),
                float(self.core.cfg.rt_first_hop_spring_k_n_m),
                " (leg-only: props/RM OFF)"
                if leg_only
                else (
                    "; RM +11.5 -> %.1f armed for next liftoff; "
                    "prop base %.0fus OPEN now (push+flight) until RM done"
                    % (
                        float(self.lcm_cfg.rm_hopping_rad),
                        float(self.lcm_cfg.hop_prop_base_pwm_us),
                    )
                ),
            )
        )

    def _handle_robot_data(self, channel: str, data: bytes) -> None:
        msg = hopper_data_lcmt.decode(data)
        with self.lock:
            self.robot_state["q"] = np.array(msg.q, dtype=float)
            self.robot_state["qd"] = np.array(msg.qd, dtype=float)
            self.robot_state["tau"] = np.array(msg.tauIq, dtype=float)
            self.robot_state["rm_q"] = np.array(msg.rm_q, dtype=float)
            self.robot_state["rm_qd"] = np.array(msg.rm_qd, dtype=float)
            self.robot_state["rm_iq"] = np.array(msg.rm_iq, dtype=float)
            self.robot_state["rm_online"] = int(msg.rm_online)
            self.robot_state["have_motor"] = True
            self._rx_motor_n += 1
            self._rx_motor_last_t = time.time()

    def _handle_imu_data(self, channel: str, data: bytes) -> None:
        msg = hopper_imu_lcmt.decode(data)
        with self.lock:
            self.robot_state["imu_quat"] = np.array(msg.quat, dtype=float)
            # NO sign hacks. The bridge already publishes gyro = R_mount @ gyro_b in the
            # robot frame (+X fwd, +Y left, +Z up), proven consistent with the attitude
            # quaternion (dds_gyro_check.py: omega_true=vee(R^T*Rdot) vs gyro slope +0.99,
            # no axis flipped -- the EKF integrates the gyro so they cannot disagree in sign).
            # A previous "gyro[1] = -gyro[1]" here was canceling an equal-and-opposite hack
            # in the bridge; both are now removed so the chain is clean end-to-end and what
            # lcm-spy shows is exactly what the controller consumes.
            self.robot_state["imu_gyro"] = np.array(msg.gyro, dtype=float).reshape(3)
            self.robot_state["imu_acc"] = np.array(msg.acc, dtype=float)
            self.robot_state["imu_rpy"] = np.array(msg.rpy, dtype=float)
            self.robot_state["have_imu"] = True
            self._rx_imu_n += 1
            self._rx_imu_last_t = time.time()

    def _handle_gamepad_data(self, channel: str, data: bytes) -> None:
        try:
            msg = gamepad_lcmt.decode(data)
        except Exception:
            return
        with self.lock:
            self.robot_state["gamepad"] = msg

    def _handle_odom_data(self, channel: str, data: bytes) -> None:
        try:
            msg = hopper_odom_lcmt.decode(data)
        except Exception:
            return
        with self.lock:
            self.robot_state["odom_pos"] = np.array(
                [float(msg.pos[0]), float(msg.pos[1]), float(msg.pos[2])], dtype=float
            )
            self.robot_state["odom_yaw"] = float(msg.rpy[2])
            self.robot_state["odom_quality"] = int(msg.quality)
            self.robot_state["odom_rx_t"] = float(time.time())

    def _handle_nav_cmd(self, channel: str, data: bytes) -> None:
        try:
            msg = hopper_nav_cmd_lcmt.decode(data)
        except Exception:
            return
        with self.lock:
            self.robot_state["nav_v_xy"] = np.array(
                [float(msg.v_xy_w[0]), float(msg.v_xy_w[1])], dtype=float
            )
            self.robot_state["nav_active"] = int(msg.active)
            self.robot_state["nav_wp_index"] = int(msg.wp_index)
            self.robot_state["nav_rx_t"] = float(time.time())

    def run_lcm_handler(self) -> None:
        while self.running:
            try:
                self.lc.handle_timeout(10)
            except Exception:
                time.sleep(0.01)

    def _compute_desired_v_xy(self, gamepad_msg) -> np.ndarray:
        v = np.zeros(2, dtype=float)
        if gamepad_msg is None:
            return v
        try:
            stick_x = float(gamepad_msg.rightStickAnalog[0])
            stick_y = float(gamepad_msg.rightStickAnalog[1])
        except Exception:
            return v

        dz = float(self.lcm_cfg.stick_deadzone)
        if abs(stick_x) < dz:
            stick_x = 0.0
        if abs(stick_y) < dz:
            stick_y = 0.0
        max_v = float(self.lcm_cfg.max_cmd_vel)
        # Keep the same convention as Hopper4.py for operator consistency:
        v[0] = -stick_x * max_v
        v[1] = stick_y * max_v
        return v

    def _compute_wheel_cmd(self, gamepad_msg) -> np.ndarray:
        """MOBILE kiwi-drive IK: sticks -> body twist -> 3 wheel speeds.

        Right stick = translation (same axis convention as the hopping
        desired_v_xy), left stick X = yaw rate. Returns wheel angular
        speeds (rad/s), uniformly scaled if any wheel exceeds the cap so
        the commanded twist DIRECTION is preserved.
        """
        vx = vy = wz = 0.0
        if gamepad_msg is not None:
            try:
                rs_x = float(gamepad_msg.rightStickAnalog[0])
                rs_y = float(gamepad_msg.rightStickAnalog[1])
                ls_x = float(gamepad_msg.leftStickAnalog[0])
            except Exception:
                rs_x = rs_y = ls_x = 0.0
            dz = float(self.lcm_cfg.stick_deadzone)
            rs_x = 0.0 if abs(rs_x) < dz else rs_x
            rs_y = 0.0 if abs(rs_y) < dz else rs_y
            ls_x = 0.0 if abs(ls_x) < dz else ls_x
            vx = -rs_x * float(self.lcm_cfg.mobile_v_max_mps)
            vy = rs_y * float(self.lcm_cfg.mobile_v_max_mps)
            wz = -ls_x * float(self.lcm_cfg.mobile_wz_max_rad_s)
        az = np.deg2rad(np.asarray(
            self.lcm_cfg.wheel_azimuth_deg, dtype=float
        ).reshape(3))
        r_base = float(self.lcm_cfg.wheel_base_radius_m)
        r_wheel = float(max(1e-4, float(self.lcm_cfg.wheel_radius_m)))
        sgn = np.asarray(self.lcm_cfg.wheel_dir_sign, dtype=float).reshape(3)
        # v_i = -sin(az_i)*vx + cos(az_i)*vy + R*wz (rim speed, m/s)
        v_rim = -np.sin(az) * vx + np.cos(az) * vy + r_base * wz
        omega = sgn * v_rim / r_wheel
        w_max = float(max(1e-3, float(self.lcm_cfg.wheel_speed_max_rad_s)))
        pk = float(np.max(np.abs(omega)))
        if pk > w_max:
            omega = omega * (w_max / pk)
        return omega.astype(float)

    def _publish_wheel_cmd(self, omega_rad_s: np.ndarray, *, enable: bool) -> None:
        msg = wheel_cmd_lcmt()
        msg.timestamp = int(time.time() * 1e6)
        msg.speed_des_rad_s = [
            float(x) for x in np.asarray(omega_rad_s, dtype=float).reshape(3)
        ]
        msg.enable = 1 if enable else 0
        self.lc.publish("wheel_cmd_lcmt", msg.encode())
        # For the periodic status line (MOBILE debugging).
        self._wheel_last_cmd = np.asarray(omega_rad_s, dtype=float).reshape(3)
        self._wheel_last_enable = bool(enable)

    def _publish_hopper_cmd(
        self,
        tau_cmd: np.ndarray,
        *,
        kp_joint: np.ndarray | None = None,
        kd_joint: np.ndarray | None = None,
    ) -> None:
        msg = hopper_cmd_lcmt()
        msg.tau_ff = [float(t) for t in np.asarray(tau_cmd, dtype=float).reshape(3)]
        msg.q_des = [0.0, 0.0, 0.0]
        msg.qd_des = [0.0, 0.0, 0.0]
        if kp_joint is None:
            msg.kp_joint = [0.0, 0.0, 0.0]
        else:
            msg.kp_joint = [float(x) for x in np.asarray(kp_joint, dtype=float).reshape(3)]
        if kd_joint is None:
            msg.kd_joint = [0.0, 0.0, 0.0]
        else:
            msg.kd_joint = [float(x) for x in np.asarray(kd_joint, dtype=float).reshape(3)]
        # RM M2006 current command (A). Rides inside hopper_cmd_lcmt so the Jetson
        # driver applies leg-class gating (X arms / B cuts) before forwarding to
        # the Pixhawk. Set via set_rm_iq_des(); zeros by default.
        msg.rm_iq_des = [float(x) for x in np.asarray(self._rm_iq_des, dtype=float).reshape(3)]
        # Coordinate-only RM initialization. On the pulse edge the driver makes
        # the current physical pose read rm_zero_at_rad; no motor is energized.
        msg.rm_zero_at_rad = float(self._rm_zero_at_rad)
        msg.rm_set_zero = 1 if time.time() < float(self._rm_zero_until) else 0
        self.lc.publish("hopper_cmd_lcmt", msg.encode())

    def set_rm_iq_des(self, iq_a) -> None:
        """Set the desired M2006 torque current (A, per motor, clipped +/-10).
        Takes effect on every subsequent hopper_cmd_lcmt publish; the Jetson
        driver only forwards it in PD/PWMPD mode (gamepad X), B zeroes it."""
        self._rm_iq_des = np.clip(np.asarray(iq_a, dtype=float).reshape(3), -10.0, 10.0)

    def _rm_set_logical_position(self, logical_rad: float, label: str) -> None:
        """Label the current RM physical pose without moving it."""
        self._rm_zero_at_rad = float(logical_rad)
        self._rm_zero_until = time.time() + 0.1
        print(
            "[rm] %s: current physical pose is now logical q=%+.1f rad (0 A)"
            % (label, self._rm_zero_at_rad)
        )

    def _rm_start(self, target_rad: float, label: str) -> None:
        """Drive all folding arms to an absolute logical endpoint."""
        self._rm_target = float(target_rad)
        self._rm_sync_q0 = None   # re-latch start poses on the first drive tick
        self._rm_stage = 2
        print("[rm] %s -> drive to %+.1f rad (kp %.1f kd %.2f cap %.1fA, "
              "sync lead %.1f rad)"
              % (label, self._rm_target, float(self.lcm_cfg.rm_kp_a_per_rad),
                 float(self.lcm_cfg.rm_kd_a_per_rad_s),
                 float(self.lcm_cfg.rm_iq_max_a),
                 float(getattr(self.lcm_cfg, "rm_sync_lead_rad", 1.5))))

    def _update_rm(self) -> None:
        """RM M2006 sequence, run once per control step (writes _rm_iq_des).

        Stage 2 (drive): current-mode PD toward a SHARED synchronized
        reference (never more than rm_sync_lead_rad ahead of the slowest
        arm's progress), capped at rm_iq_max_a -- see rm_sync_lead_rad.
        When ALL three are within rm_reach_tol_rad -> stage 3. Stage 1 is
        retained only as an optional settling state.
        Stage 3 (hold): keep the PD hold for rm_hold_s, then idle at the fixed
        logical endpoint. Transition endpoints are never re-zeroed: HOPPING
        remains 0 rad and MOBILE remains +11.5 rad.
        Requires fresh feedback: if rm_online != 7 the command is forced to 0 A
        (the sequence stays in its stage and resumes when feedback returns).

        Stage 0 (idle): normally 0 A, EXCEPT the HOPPING station hold (see
        rm_hopping_hold): gait==hopping + legs PD/PWMPD keeps a low-cap PD
        on 0 rad so hop impacts cannot back-drive the arms (~0 A at rest).
        """
        if int(self._rm_stage) == 0:
            self._update_rm_station_hold()
            return
        with self.lock:
            rm_q = np.asarray(self.robot_state["rm_q"], dtype=float).reshape(3).copy()
            rm_qd = np.asarray(self.robot_state["rm_qd"], dtype=float).reshape(3).copy()
            rm_online = int(self.robot_state["rm_online"])
        if rm_online != 7:
            self._rm_iq_des = np.zeros(3, dtype=float)
            return
        target = float(self._rm_target)
        kp = float(self.lcm_cfg.rm_kp_a_per_rad)
        kd = float(self.lcm_cfg.rm_kd_a_per_rad_s)
        cap = float(abs(float(self.lcm_cfg.rm_iq_max_a)))
        now_t = time.time()
        if int(self._rm_stage) == 1:
            # Pre-zero settling: 0 A until the re-zeroed rm_q feedback is in.
            self._rm_iq_des = np.zeros(3, dtype=float)
            if (now_t - float(self._rm_hold_t0)) >= 0.3:
                self._rm_stage = 2
                self._rm_sync_q0 = None   # re-latch start poses for the drive
                print("[rm] pre-zero done (q=[%+.3f %+.3f %+.3f]) -> drive to %+.1f rad"
                      % (*rm_q, target))
            return
        if int(self._rm_stage) == 2:
            if bool(np.all(np.abs(rm_q - target) <= float(self.lcm_cfg.rm_reach_tol_rad))):
                self._rm_stage = 3
                self._rm_hold_t0 = now_t
                print("[rm] reached %+.1f rad -> hold %.1fs, then re-zero"
                      % (target, float(self.lcm_cfg.rm_hold_s)))
            else:
                # --- Synchronized drive (2026-07-24 07:05 user: re-enabled;
                # "rm电机要一样的 不能一个快一个慢").  All arms track a
                # SHARED reference that never runs more than rm_sync_lead_rad
                # ahead of the slowest arm's progress, so the fast arms wait.
                lead = float(getattr(self.lcm_cfg, "rm_sync_lead_rad", 1.0))
                if self._rm_sync_q0 is None:
                    self._rm_sync_q0 = rm_q.copy()
                span = target - np.asarray(self._rm_sync_q0, dtype=float)
                if lead > 0.0 and float(np.max(np.abs(span))) > 0.3:
                    safe = np.where(np.abs(span) > 1e-6, span, 1e-6)
                    prog = np.clip((rm_q - self._rm_sync_q0) / safe, 0.0, 1.0)
                    prog = np.where(np.abs(span) > 0.3, prog, 1.0)
                    p_ref = min(
                        1.0,
                        float(np.min(prog))
                        + lead / float(max(1e-6, np.max(np.abs(span)))),
                    )
                    q_ref = self._rm_sync_q0 + p_ref * span
                    self._rm_iq_des = np.clip(
                        kp * (q_ref - rm_q) - kd * rm_qd, -cap, cap
                    )
                    return
        elif int(self._rm_stage) == 3:
            if (now_t - float(self._rm_hold_t0)) >= float(self.lcm_cfg.rm_hold_s):
                self._rm_stage = 0
                self._rm_iq_des = np.zeros(3, dtype=float)
                print(
                    "[rm] endpoint held: q=[%+.3f %+.3f %+.3f] -> idle (0 A)"
                    % tuple(rm_q)
                )
                return
        self._rm_iq_des = np.clip(kp * (target - rm_q) - kd * rm_qd, -cap, cap)

    def _update_rm_station_hold(self) -> None:
        """HOPPING station keeping (rm_stage==0 only).

        While gait==hopping AND legs are enabled (PD/PWMPD) AND no fold
        transition is pending, hold the arms at rm_hopping_rad with a
        low-cap current PD. At the endpoint the current is ~0 A, so this
        is thermally free; only hop-impact transients draw current.
        """
        PD, PWMPD = 2, 3
        active = (
            bool(getattr(self.lcm_cfg, "rm_hopping_hold", True))
            and self._gait_mode == "hopping"
            and int(self._mode_est) in (PD, PWMPD)
            and not bool(self._rm_lt_pending)
            and not bool(self._rm_rt_pending)
            and not bool(self._switch_loop)
        )
        if not active:
            self._rm_iq_des = np.zeros(3, dtype=float)
            return
        with self.lock:
            rm_q = np.asarray(self.robot_state["rm_q"], dtype=float).reshape(3).copy()
            rm_qd = np.asarray(self.robot_state["rm_qd"], dtype=float).reshape(3).copy()
            rm_online = int(self.robot_state["rm_online"])
        if rm_online != 7:
            self._rm_iq_des = np.zeros(3, dtype=float)
            return
        target = float(self.lcm_cfg.rm_hopping_rad)
        kp = float(self.lcm_cfg.rm_kp_a_per_rad)
        kd = float(self.lcm_cfg.rm_kd_a_per_rad_s)
        cap = float(abs(float(getattr(self.lcm_cfg, "rm_hold_iq_max_a", 2.0))))
        self._rm_iq_des = np.clip(kp * (target - rm_q) - kd * rm_qd, -cap, cap)

    def _apply_tau_output_limit(self, tau_raw: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Output-side torque limiting.
        This is intentionally outside ModeECore so it doesn't change the internal controller/QP solution.

        Returns:
          tau_send: torque actually sent to hardware (3,)
          scale_applied: scalar multiplier applied (includes tau_out_scale and any extra scaling due to tau_out_max)
        """
        tau_raw = np.asarray(tau_raw, dtype=float).reshape(3)
        # First apply user scaling
        scale = float(np.clip(float(self.lcm_cfg.tau_out_scale), 0.0, 1e9))
        tau = (tau_raw * scale).astype(float)

        # Then apply output max (proportional scaling to keep direction)
        if self.lcm_cfg.tau_out_max_nm is not None:
            lim = float(abs(float(self.lcm_cfg.tau_out_max_nm)))
            if lim > 0.0:
                m = float(np.max(np.abs(tau)))
                if m > lim:
                    scale2 = lim / m
                    tau = (tau * float(scale2)).astype(float)
                    scale = float(scale * scale2)
        return tau, float(scale)

    def _publish_motor_pwm(self, pwm_us: np.ndarray, *, control_mode: int = 1, force: bool = False) -> None:
        pwm_us = np.asarray(pwm_us, dtype=float).reshape(6)

        # Propeller SOFT-START: rate-limit prop PWM moving AWAY from the stop point
        # (1000us) to cut ESC inrush current (hard spool-up jumps cause battery sag
        # -> brownout / CAN bus-off). BIDIR (2026-07-06): pwm < 1000 = reverse, so
        # "spool up" can be either direction; motion TOWARD stop (spin-down, either
        # side) is applied IMMEDIATELY for safety. force=True bypasses the limiter
        # (used by the shutdown zero-out / disarm paths).
        now_t = time.time()
        if force or self._last_pwm_pub_t is None:
            limited = pwm_us
        else:
            dt = max(0.0, now_t - self._last_pwm_pub_t)
            max_up = float(self.lcm_cfg.prop_slew_up_us_per_s) * dt
            stop = float(self.modee_cfg.pwm_min_us)  # 1000us = prop stop (bidir center)
            prev_d = self.prev_pwm_us - stop
            new_d = pwm_us - stop
            # Growth of |d| is ramp-limited; shrinking toward 0 (stop) is immediate.
            upper = np.maximum(prev_d + max_up, 0.0)
            lower = np.minimum(prev_d - max_up, 0.0)
            limited = stop + np.clip(new_d, lower, upper)
        self._last_pwm_pub_t = now_t
        self.prev_pwm_us = np.asarray(limited, dtype=float).copy()
        pwm_us = limited
        
        msg = motor_pwm_lcmt()
        msg.timestamp = int(time.time() * 1e6)
        msg.pwm_values = [float(v) for v in pwm_us]
        msg.roll_error = 0.0
        msg.pitch_error = 0.0
        msg.roll_output = 0.0
        msg.pitch_output = 0.0
        msg.control_mode = int(control_mode)
        self.lc.publish("motor_pwm_lcmt", msg.encode())

    def _publish_zero_outputs(self) -> None:
        """Zero every outgoing channel so the robot does not keep acting on the last
        commands after the upper layer stops. Sends a zeroed hopper_cmd (tau=q_des=qd_des=
        kp=kd=0) and disarms the propellers (pwm_min) while requesting driver DAMP. Sent a
        few times since LCM is best-effort UDP. Called on controller shutdown."""
        try:
            pwm_min = float(self.modee_cfg.pwm_min_us)
        except Exception:
            pwm_min = 1000.0
        for _ in range(5):
            try:
                self._publish_hopper_cmd(np.zeros(3, dtype=float))
                self._publish_motor_pwm(np.full(6, pwm_min, dtype=float), control_mode=-1, force=True)
            except Exception:
                break
            time.sleep(0.01)

    def _start_log(self) -> None:
        if bool(self._log_enabled):
            return
        try:
            # Default: a logs/ folder INSIDE robot_runtime (next to this controller), so
            # everything (code + logs) stays in robot_runtime. Override via MODEE_LOG_DIR.
            _default_logs = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
            )
            logs_dir = os.path.expanduser(os.environ.get("MODEE_LOG_DIR", _default_logs))
            os.makedirs(logs_dir, exist_ok=True)
            # Unique file per Y press; also mirrored to modee_latest.csv on stop.
            stamp = time.strftime("%Y%m%d_%H%M%S")
            log_name = os.environ.get("MODEE_LOG_NAME", f"modee_{stamp}.csv")
            path = os.path.join(logs_dir, log_name)
            self._log_latest_path = os.path.join(logs_dir, "modee_latest.csv")

            fp = open(path, "w", newline="")
            writer = csv.writer(fp)

            header = [
                "wall_time_s",
                "t_s",
                "phase",
                "gait_mode",
                "driver_mode_est",
                "switch_loop",
                "prop_enable",
                "props_active",
                "prop_ctrl_mode",
                "rm_stage",
                "rm_target_rad",
                "rm_lt_pending",
                "rm_rt_pending",
                "rt_first_hop_l0_active",
                "stance",
                "compress_active",
                "push_started",
                "touchdown",
                "liftoff",
                "apex",
                "status",
                # joints (measured from hopper_data_lcmt)
                "q0",
                "q1",
                "q2",
                "qd0",
                "qd1",
                "qd2",
                # measured motor torque / Iq (from driver feedback, not command)
                "tau_meas0",
                "tau_meas1",
                "tau_meas2",
                # filtered joint velocity used by kinematics (q-diff -> EMA -> MA)
                "qd_kin0",
                "qd_kin1",
                "qd_kin2",
                # commands / mapping
                # tau* are the torques actually SENT to hardware (after output limiting)
                "tau0",
                "tau1",
                "tau2",
                # raw torques produced by ModeECore (before output limiting)
                "tau_raw0",
                "tau_raw1",
                "tau_raw2",
                # output limiter debug
                "tau_out_scale_applied",
                "f_tau_delta0",
                "f_tau_delta1",
                "f_tau_delta2",
                "f_contact_w0",
                "f_contact_w1",
                "f_contact_w2",
                # GRF in BODY frame (core computes it; log it directly so plots
                # never have to reconstruct body force from rpy + world force)
                "f_contact_b0",
                "f_contact_b1",
                "f_contact_b2",
                "thrust0",
                "thrust1",
                "thrust2",
                "pwm0",
                "pwm1",
                "pwm2",
                "pwm3",
                "pwm4",
                "pwm5",
                # kinematics
                "foot_vicon0",
                "foot_vicon1",
                "foot_vicon2",
                "foot_b0",
                "foot_b1",
                "foot_b2",
                # foot velocity (body frame, +Z up). Useful for Hopper4-like velocity estimation debugging.
                "foot_vrel_b0",
                "foot_vrel_b1",
                "foot_vrel_b2",
                "J_inv_det",
                "J_inv_cond",
                "A_tau_f_det",
                "A_tau_f_cond",
                "foot_des_b0",
                "foot_des_b1",
                "foot_des_b2",
                "foot_des_w0",
                "foot_des_w1",
                "foot_des_w2",
                "p_foot_des_w0",
                "p_foot_des_w1",
                "p_foot_des_w2",
                "leg_len_m",
                "q_shift_m",
                "qd_shift_mps",
                # compression debug
                "comp_m",
                "comp_tgt_m",
                "comp_tgt_act_m",
                "z_now_m",
                "s_stance",
                # state estimate
                "p_hat_w0",
                "p_hat_w1",
                "p_hat_w2",
                "v_hat_w0",
                "v_hat_w1",
                "v_hat_w2",
                # Debug: base velocity measured from foot kinematics, in WORLD frame
                "v_meas_foot_w0",
                "v_meas_foot_w1",
                "v_meas_foot_w2",
                "desired_vx_w",
                "desired_vy_w",
                # IMU raw (LCM)
                "imu_quat_w",
                "imu_quat_x",
                "imu_quat_y",
                "imu_quat_z",
                "imu_gyro_x",
                "imu_gyro_y",
                "imu_gyro_z",
                "imu_acc_x",
                "imu_acc_y",
                "imu_acc_z",
                "imu_rpy_roll",
                "imu_rpy_pitch",
                "imu_rpy_yaw",
                # IMU estimate (ModeE)
                "q_hat_w",
                "q_hat_x",
                "q_hat_y",
                "q_hat_z",
                "rpy_hat_roll",
                "rpy_hat_pitch",
                "rpy_hat_yaw",
                # attitude control debug (what QP is trying to realize)
                "tau_des_w0",
                "tau_des_w1",
                "tau_des_w2",
                "e_R0",
                "e_R1",
                "e_R2",
                "tau_b_stance_des0",
                "tau_b_stance_des1",
                "tau_b_stance_des2",
                "omega_b_used0",
                "omega_b_used1",
                "omega_b_used2",
                # solver debug
                "slack0",
                "slack1",
                "slack2",
                "slack3",
                "slack4",
                "slack5",
                # wrench debug (net wrench vs allocation)
                "F_des_w0",
                "F_des_w1",
                "F_des_w2",
                "f_ref_w0",
                "f_ref_w1",
                "f_ref_w2",
                # Mode1 vertical push energy
                "energy_comp_fz",
                "vz_up",
                "energy_gate",
                "thrust_sum_ref",
                "thrust_sum",
                "F_total_w0",
                "F_total_w1",
                "F_total_w2",
                "tau_contact_w0",
                "tau_contact_w1",
                "tau_contact_w2",
                "tau_props_w0",
                "tau_props_w1",
                "tau_props_w2",
                "tau_total_w0",
                "tau_total_w1",
                "tau_total_w2",
                # apex / takeoff debug
                "z_apex_actual_m",
                "v_to_cmd_m_s",
                "desired_vz_from_apex_m_s",
                "hop_height_m",
                # FB-SLIP: TD-sized constant brake/push forces + plan
                "fbslip_v_td_m_s",
                "fbslip_f_brake_n",
                "fbslip_x_c_plan_m",
                "fbslip_x_td_m",
                "fbslip_f_push_n",
                "fbslip_t_bottom_s",
                "fbslip_sink",
                "apex_eta",
                "apex_e_bias",
                "vel_kf_x",
                "vel_kf_y",
                "vel_kf_z",
                "hfb_T_st_s",
                "hfb_k_dx",
                "hfb_dx_att_x_m",
                "hfb_dx_att_y_m",
                "vz_lo_m_s",
                # MPC debug
                "mpc_status",
                "mpc_u0_fx",
                "mpc_u0_fy",
                "mpc_u0_fz",
                # LCM link health (received rates + last-packet age).
                # If q freezes / IMU dies, *_age_ms grows while the controller keeps running.
                "rx_motor_hz",
                "rx_imu_hz",
                "motor_age_ms",
                "imu_age_ms",
                # RM M2006/C610 motors (Pixhawk CAN, relayed in hopper_data_lcmt)
                "rm_q0",
                "rm_q1",
                "rm_q2",
                "rm_qd0",
                "rm_qd1",
                "rm_qd2",
                "rm_iq0",
                "rm_iq1",
                "rm_iq2",
                "rm_iq_des0",
                "rm_iq_des1",
                "rm_iq_des2",
                "rm_online",
            ]
            writer.writerow(header)
            fp.flush()

            self._log_fp = fp
            self._log_writer = writer
            self._log_path = path
            self._log_enabled = True
            self._log_last_flush_t = float(time.time())
            self._log_rows = 0
            print(f"[log] START -> {path}")
        except Exception as e:
            # Don't kill controller if logging fails.
            self._log_enabled = False
            self._log_fp = None
            self._log_writer = None
            self._log_path = None
            print(f"[log] START FAILED: {e}")

    def _stop_log(self) -> None:
        path = self._log_path
        rows = int(self._log_rows)
        if self._log_fp is not None:
            try:
                self._log_fp.flush()
            except Exception:
                pass
            try:
                self._log_fp.close()
            except Exception:
                pass
            latest = getattr(self, "_log_latest_path", None)
            if path and latest and os.path.isfile(path):
                try:
                    import shutil
                    shutil.copy2(path, latest)
                except Exception as e:
                    print(f"[log] copy to modee_latest.csv failed: {e}")
        self._log_enabled = False
        self._log_fp = None
        self._log_writer = None
        self._log_path = None
        self._log_last_flush_t = 0.0
        self._log_rows = 0
        if path:
            print(f"[log] STOP -> {path}  ({rows} rows)")

    def _handle_log_trigger(self, gamepad_msg) -> None:
        y_now = False
        try:
            y_now = bool(getattr(gamepad_msg, "y", 0)) if gamepad_msg is not None else False
        except Exception:
            y_now = False

        # Rising edge:
        # - restart CSV logging (close current file if any, then start a new one)
        # - reset controller estimates (v_hat, integrators, etc.) for a clean segment
        if bool(y_now) and (not bool(self._last_y)):
            # reset core estimates (user-requested "zero": v, integrators, etc.)
            try:
                self.core.user_reset()
            except Exception:
                pass
            # User request: also enable velocity hard-hold until we enter STANCE once.
            # (This prevents "floating" foot targets while the robot is being held in the air.)
            self._y_hold_until_stance = True
            try:
                self.core.user_zero_velocity_hold(True)
            except Exception:
                pass
            # restart logging every time Y is pressed
            if bool(self._log_enabled):
                self._stop_log()
            self._start_log()
        self._last_y = bool(y_now)

        # Stop log on B (user uses B to enter DAMP on the Pi). This is a convenience so logs
        # end cleanly when the operator aborts.
        b_now = False
        try:
            b_now = bool(getattr(gamepad_msg, "b", 0)) if gamepad_msg is not None else False
        except Exception:
            b_now = False
        if bool(b_now) and (not bool(self._last_b)):
            if bool(self._log_enabled):
                self._stop_log()
        self._last_b = bool(b_now)

    def _handle_zero_vel_trigger(self, gamepad_msg) -> None:
        """
        User request: pressing the controller "I" button should behave like a restart:
        - force internal velocity estimate to 0 (no drift)
        - keep desired velocity command at 0

        On the Xbox mapping in this repo, this is `point` (see `xbox_controller.hpp`).
        """
        point_now = False
        try:
            point_now = bool(getattr(gamepad_msg, "point", 0)) if gamepad_msg is not None else False
        except Exception:
            point_now = False

        # Rising edge toggles HOLD.
        if bool(point_now) and (not bool(self._last_point)):
            self._zero_vel_hold = not bool(self._zero_vel_hold)
            try:
                self.core.user_zero_velocity_hold(bool(self._zero_vel_hold))
            except Exception:
                pass

        self._last_point = bool(point_now)

    def _handle_big_jump_trigger(self, gamepad_msg) -> None:
        """
        RB rising edge:
        Arm one-shot big jump in core, executed at next stance touchdown.
        """
        rb_now = False
        try:
            rb_now = bool(getattr(gamepad_msg, "rightBumper", 0)) if gamepad_msg is not None else False
        except Exception:
            rb_now = False

        if bool(rb_now) and (not bool(self._last_rb)):
            try:
                self.core.user_request_big_jump_next_stance()
            except Exception:
                pass
        self._last_rb = bool(rb_now)

    def _safe_status_label(
        self,
        *,
        roll: float,
        pitch: float,
        q: np.ndarray,
        safe_armed: bool,
        now_t: float,
    ) -> str:
        """Short SAFE summary for the periodic status line."""
        if (bool(self._switch_loop) or self._gait_mode == "mobile"
                or self._lt_stand_t0 is not None):
            return "exempt"
        pause_s = float(max(0.0, float(self.lcm_cfg.safe_pause_s)))
        if pause_s > 0.0 and (now_t - float(self._safe_last_t)) < pause_s:
            return "ACTIVE"
        rp_lim = float(np.deg2rad(float(self.lcm_cfg.safe_rp_deg)))
        q_min = float(self.lcm_cfg.safe_q_min)
        q_max = float(self.lcm_cfg.safe_q_max)
        unsafe_tilt = (abs(float(roll)) > rp_lim) or (abs(float(pitch)) > rp_lim)
        unsafe_q = bool(np.any((q < q_min) | (q > q_max)))
        if bool(safe_armed) and (unsafe_tilt or unsafe_q):
            return "RISK"
        return "OK"

    def _format_status_line(
        self,
        *,
        info: dict,
        q: np.ndarray,
        roll: float,
        pitch: float,
        safe_armed: bool,
        now_t: float,
    ) -> str:
        in_stance = bool(int(info.get("stance", 0)))
        if in_stance:
            ph = "STANCE:COMP" if int(info.get("compress", 0)) else "STANCE:PUSH"
        else:
            ph = "FLIGHT"
        if self._lt_stand_t0 is not None:
            gait_tag = "LT-STAND"
        elif self._rt_stand_t0 is not None:
            gait_tag = "RT-STAND"
        elif bool(self._switch_loop):
            gait_tag = "P4"
        else:
            gait_tag = self._gait_mode.upper()
        # Driver mode estimate (mirrors the Jetson mode machine; X arms).
        mode_tag = {0: "OFF", 1: "DAMP", 2: "PD", 3: "PWMPD"}.get(
            int(self._mode_est), "?"
        )
        foot_vicon = np.asarray(info.get("foot_vicon", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        if np.all(np.isfinite(foot_vicon)):
            leg_len = float(np.linalg.norm(foot_vicon))
            line = f"[{gait_tag}|{mode_tag}|{ph}] leg_len={leg_len:.4f} m"
        else:
            line = f"[{gait_tag}|{mode_tag}|{ph}] leg_len=nan"
        if self._gait_mode == "mobile":
            w = np.asarray(
                getattr(self, "_wheel_last_cmd", np.zeros(3)), dtype=float
            ).reshape(3)
            en = "on" if bool(getattr(self, "_wheel_last_enable", False)) \
                else "off"
            line += (
                f" wheels[{en}]="
                f"{w[0]:+.1f}/{w[1]:+.1f}/{w[2]:+.1f} rad/s"
            )
        return line

    def _log_step(
        self,
        *,
        wall_time_s: float,
        q: np.ndarray,
        qd: np.ndarray,
        imu_quat: np.ndarray,
        imu_gyro: np.ndarray,
        imu_acc: np.ndarray,
        imu_rpy: np.ndarray,
        desired_v_xy: np.ndarray,
        tau_cmd: np.ndarray,
        tau_raw: np.ndarray,
        tau_out_scale_applied: float,
        pwm_us: np.ndarray,
        info: dict,
        props_active: bool = False,
        prop_ctrl_mode: int = 1,
    ) -> None:
        if (not bool(self._log_enabled)) or (self._log_writer is None) or (self._log_fp is None):
            return
        try:
            stance = int(info.get("stance", 0))
            compress = int(info.get("compress_active", info.get("compress", 0)))
            push_started = int(info.get("push_started", 0))
            touchdown = int(info.get("touchdown", 0))
            liftoff = int(info.get("liftoff", 0))
            apex = int(info.get("apex", 0))
            status = str(info.get("status", ""))

            if stance:
                phase = "STANCE:COMP" if int(info.get("compress", 0)) else "STANCE:PUSH"
            else:
                phase = "FLIGHT"

            tau_meas = np.asarray(
                self.robot_state.get("tau", [np.nan, np.nan, np.nan]), dtype=float
            ).reshape(3)

            f_tau_delta = np.asarray(info.get("f_tau_delta", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            f_contact_w = np.asarray(info.get("f_contact_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            f_contact_b = np.asarray(info.get("f_contact_b", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            thrusts_arm = np.asarray(info.get("thrusts_arm", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            pwm_us = np.asarray(pwm_us, dtype=float).reshape(6)

            foot_vicon = np.asarray(info.get("foot_vicon", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            foot_b = np.asarray(info.get("foot_b", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            qd_kin = np.asarray(info.get("qd_kin", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            foot_vrel_b = np.asarray(info.get("foot_vrel_b", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            J_inv_det = float(info.get("J_inv_det", float("nan")))
            J_inv_cond = float(info.get("J_inv_cond", float("nan")))
            A_tau_f_det = float(info.get("A_tau_f_det", float("nan")))
            A_tau_f_cond = float(info.get("A_tau_f_cond", float("nan")))
            foot_des_b = np.asarray(info.get("foot_des_b", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            foot_des_w = np.asarray(info.get("foot_des_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            p_foot_des_w = np.asarray(info.get("p_foot_des_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)

            leg_len = float(np.linalg.norm(foot_vicon)) if np.all(np.isfinite(foot_vicon)) else float("nan")
            q_shift = float(info.get("q_shift_equiv", float("nan")))
            qd_shift = float(info.get("qd_shift_equiv", float("nan")))

            comp_m = float(info.get("comp_m", float("nan")))
            comp_tgt_m = float(info.get("comp_tgt_m", float("nan")))
            comp_tgt_act_m = float(info.get("comp_tgt_act_m", float("nan")))
            z_now_m = float(info.get("z_now_m", float("nan")))
            s_stance = float(info.get("s_stance", float("nan")))

            p_hat_w = np.asarray(info.get("p_hat_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            v_hat_w = np.asarray(info.get("v_hat_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            v_meas_foot_w = np.asarray(info.get("v_meas_foot_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)

            q_hat = np.asarray(info.get("q_hat_wxyz", [np.nan, np.nan, np.nan, np.nan]), dtype=float).reshape(4)
            rpy_hat = np.asarray(info.get("rpy_hat", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)

            slack = np.asarray(info.get("slack", [np.nan] * 6), dtype=float).reshape(6)
            tau_des_w = np.asarray(info.get("tau_des_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            e_R = np.asarray(info.get("e_R", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            tau_b_stance_des = np.asarray(
                info.get("tau_b_stance_des", [np.nan, np.nan, np.nan]), dtype=float
            ).reshape(3)
            omega_b_used = np.asarray(info.get("omega_b_used", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            F_des_w = np.asarray(info.get("F_des_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            f_ref_w = np.asarray(info.get("f_ref_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            energy_comp_fz = float(info.get("energy_comp_fz", float("nan")))
            # Log the exact core signals; do not reconstruct the gate from vz.
            # CASE PUSH is physical leg extension (qd_shift > 0), while vz_up
            # remains useful for the SRB vertical-energy calculation.
            vz_up = float(info.get(
                "vz_up",
                -v_hat_w[2] if np.isfinite(float(v_hat_w[2])) else float("nan"),
            ))
            energy_gate = int(info.get("energy_gate", 0))
            thrust_sum_ref = float(info.get("thrust_sum_ref", float("nan")))
            thrust_sum = float(info.get("thrust_sum", float("nan")))
            F_total_w = np.asarray(info.get("F_total_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            tau_contact_w = np.asarray(info.get("tau_contact_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            tau_props_w = np.asarray(info.get("tau_props_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            tau_total_w = np.asarray(info.get("tau_total_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            # RM M2006 state snapshot (arrays are replaced atomically in the LCM
            # handler, so a lock-free read here is safe).
            rm_q = np.asarray(self.robot_state["rm_q"], dtype=float).reshape(3)
            rm_qd = np.asarray(self.robot_state["rm_qd"], dtype=float).reshape(3)
            rm_iq = np.asarray(self.robot_state["rm_iq"], dtype=float).reshape(3)
            rm_online = int(self.robot_state["rm_online"])
            z_apex_actual_m = float(info.get("z_apex_actual_m", float("nan")))
            v_to_cmd_m_s = float(info.get("v_to_cmd_m_s", float("nan")))
            desired_vz_from_apex_m_s = float(info.get("desired_vz_from_apex_m_s", float("nan")))
            hop_height_m = float(info.get("hop_height_m", float("nan")))
            mpc_u0 = np.asarray(info.get("mpc_u0", [0.0, 0.0, 0.0]), dtype=float).reshape(3)

            row = [
                float(wall_time_s),
                float(info.get("t", float("nan"))),
                phase,
                str(self._gait_mode),
                int(self._mode_est),
                int(bool(self._switch_loop)),
                int(bool(self._prop_enable)),
                int(bool(props_active)),
                int(prop_ctrl_mode),
                int(self._rm_stage),
                float(self._rm_target),
                int(bool(self._rm_lt_pending)),
                int(bool(self._rm_rt_pending)),
                int(self._rt_l0_restore is not None),
                stance,
                compress,
                push_started,
                touchdown,
                liftoff,
                apex,
                status,
                float(q[0]),
                float(q[1]),
                float(q[2]),
                float(qd[0]),
                float(qd[1]),
                float(qd[2]),
                float(tau_meas[0]),
                float(tau_meas[1]),
                float(tau_meas[2]),
                float(qd_kin[0]),
                float(qd_kin[1]),
                float(qd_kin[2]),
                float(tau_cmd[0]),
                float(tau_cmd[1]),
                float(tau_cmd[2]),
                float(tau_raw[0]),
                float(tau_raw[1]),
                float(tau_raw[2]),
                float(tau_out_scale_applied),
                float(f_tau_delta[0]),
                float(f_tau_delta[1]),
                float(f_tau_delta[2]),
                float(f_contact_w[0]),
                float(f_contact_w[1]),
                float(f_contact_w[2]),
                float(f_contact_b[0]),
                float(f_contact_b[1]),
                float(f_contact_b[2]),
                float(thrusts_arm[0]),
                float(thrusts_arm[1]),
                float(thrusts_arm[2]),
                float(pwm_us[0]),
                float(pwm_us[1]),
                float(pwm_us[2]),
                float(pwm_us[3]),
                float(pwm_us[4]),
                float(pwm_us[5]),
                float(foot_vicon[0]),
                float(foot_vicon[1]),
                float(foot_vicon[2]),
                float(foot_b[0]),
                float(foot_b[1]),
                float(foot_b[2]),
                float(foot_vrel_b[0]),
                float(foot_vrel_b[1]),
                float(foot_vrel_b[2]),
                float(J_inv_det),
                float(J_inv_cond),
                float(A_tau_f_det),
                float(A_tau_f_cond),
                float(foot_des_b[0]),
                float(foot_des_b[1]),
                float(foot_des_b[2]),
                float(foot_des_w[0]),
                float(foot_des_w[1]),
                float(foot_des_w[2]),
                float(p_foot_des_w[0]),
                float(p_foot_des_w[1]),
                float(p_foot_des_w[2]),
                float(leg_len),
                float(q_shift),
                float(qd_shift),
                float(comp_m),
                float(comp_tgt_m),
                float(comp_tgt_act_m),
                float(z_now_m),
                float(s_stance),
                float(p_hat_w[0]),
                float(p_hat_w[1]),
                float(p_hat_w[2]),
                float(v_hat_w[0]),
                float(v_hat_w[1]),
                float(v_hat_w[2]),
                float(v_meas_foot_w[0]),
                float(v_meas_foot_w[1]),
                float(v_meas_foot_w[2]),
                float(desired_v_xy[0]),
                float(desired_v_xy[1]),
                float(imu_quat[0]),
                float(imu_quat[1]),
                float(imu_quat[2]),
                float(imu_quat[3]),
                float(imu_gyro[0]),
                float(imu_gyro[1]),
                float(imu_gyro[2]),
                float(imu_acc[0]),
                float(imu_acc[1]),
                float(imu_acc[2]),
                float(imu_rpy[0]),
                float(imu_rpy[1]),
                float(imu_rpy[2]),
                float(q_hat[0]),
                float(q_hat[1]),
                float(q_hat[2]),
                float(q_hat[3]),
                float(rpy_hat[0]),
                float(rpy_hat[1]),
                float(rpy_hat[2]),
                float(tau_des_w[0]),
                float(tau_des_w[1]),
                float(tau_des_w[2]),
                float(e_R[0]),
                float(e_R[1]),
                float(e_R[2]),
                float(tau_b_stance_des[0]),
                float(tau_b_stance_des[1]),
                float(tau_b_stance_des[2]),
                float(omega_b_used[0]),
                float(omega_b_used[1]),
                float(omega_b_used[2]),
                float(slack[0]),
                float(slack[1]),
                float(slack[2]),
                float(slack[3]),
                float(slack[4]),
                float(slack[5]),
                float(F_des_w[0]),
                float(F_des_w[1]),
                float(F_des_w[2]),
                float(f_ref_w[0]),
                float(f_ref_w[1]),
                float(f_ref_w[2]),
                float(energy_comp_fz),
                float(vz_up),
                int(energy_gate),
                float(thrust_sum_ref),
                float(thrust_sum),
                float(F_total_w[0]),
                float(F_total_w[1]),
                float(F_total_w[2]),
                float(tau_contact_w[0]),
                float(tau_contact_w[1]),
                float(tau_contact_w[2]),
                float(tau_props_w[0]),
                float(tau_props_w[1]),
                float(tau_props_w[2]),
                float(tau_total_w[0]),
                float(tau_total_w[1]),
                float(tau_total_w[2]),
                float(z_apex_actual_m),
                float(v_to_cmd_m_s),
                float(desired_vz_from_apex_m_s),
                float(hop_height_m),
                float(info.get("fbslip_v_td_m_s", float("nan"))),
                float(info.get("fbslip_f_brake_n", float("nan"))),
                float(info.get("fbslip_x_c_plan_m", float("nan"))),
                float(info.get("fbslip_x_td_m", float("nan"))),
                float(info.get("fbslip_f_push_n", float("nan"))),
                float(info.get("fbslip_t_bottom_s", float("nan"))),
                float(info.get("fbslip_sink", 0)),
                float(info.get("apex_eta", 1.0)),
                float(info.get("apex_e_bias", 0.0)),
                float(info.get("vel_kf_x", float("nan"))),
                float(info.get("vel_kf_y", float("nan"))),
                float(info.get("vel_kf_z", float("nan"))),
                float(info.get("hfb_T_st_s", float("nan"))),
                float(info.get("hfb_k_dx", float("nan"))),
                float(info.get("hfb_dx_att_x_m", float("nan"))),
                float(info.get("hfb_dx_att_y_m", float("nan"))),
                float(info.get("vz_lo_m_s", float("nan"))),
                # MPC debug
                str(info.get("mpc_status", "")),
                float(mpc_u0[0]),
                float(mpc_u0[1]),
                float(mpc_u0[2]),
                # LCM link health
                float(self._rx_motor_hz),
                float(self._rx_imu_hz),
                float(max(0.0, (wall_time_s - float(self._rx_motor_last_t)) * 1e3) if self._rx_motor_last_t > 0.0 else float("nan")),
                float(max(0.0, (wall_time_s - float(self._rx_imu_last_t)) * 1e3) if self._rx_imu_last_t > 0.0 else float("nan")),
                float(rm_q[0]),
                float(rm_q[1]),
                float(rm_q[2]),
                float(rm_qd[0]),
                float(rm_qd[1]),
                float(rm_qd[2]),
                float(rm_iq[0]),
                float(rm_iq[1]),
                float(rm_iq[2]),
                float(self._rm_iq_des[0]),
                float(self._rm_iq_des[1]),
                float(self._rm_iq_des[2]),
                int(rm_online),
            ]
            self._log_writer.writerow(row)
            self._log_rows += 1

            # Periodic flush to reduce data loss without killing performance.
            # 2026-07-23: 0.25 -> 0.05 s; a hard kill was losing the last
            # quarter second right when the interesting event happened.
            t_now = float(time.time())
            if (t_now - float(self._log_last_flush_t)) >= 0.05:
                self._log_fp.flush()
                self._log_last_flush_t = t_now
        except Exception:
            # Never crash controller due to logging issues
            pass

    def run_controller(self) -> None:
        dt = float(self.modee_cfg.dt)
        next_t = time.time()
        last_print = 0.0

        try:
            while self.running:
                now = time.time()
                if now < next_t:
                    # Sleep most of the wait, busy-spin the last ~0.2ms: plain
                    # sleep() overshoots by 50-100us per tick, which alone costs
                    # ~5% of the 2ms budget at 500Hz.
                    remaining = next_t - now
                    if remaining > 2e-4:
                        time.sleep(remaining - 2e-4)
                    continue
                next_t = next_t + dt
                # If we fell behind by more than 2 periods (e.g. after a stall),
                # resync instead of bursting to catch up.
                if now - next_t > 2.0 * dt:
                    next_t = now + dt

                with self.lock:
                    have_motor = bool(self.robot_state["have_motor"])
                    have_imu = bool(self.robot_state["have_imu"])
                    q = np.asarray(self.robot_state["q"], dtype=float).reshape(3).copy()
                    qd = np.asarray(self.robot_state["qd"], dtype=float).reshape(3).copy()
                    imu_quat = np.asarray(self.robot_state["imu_quat"], dtype=float).reshape(4).copy()
                    imu_gyro = np.asarray(self.robot_state["imu_gyro"], dtype=float).reshape(3).copy()
                    imu_acc = np.asarray(self.robot_state["imu_acc"], dtype=float).reshape(3).copy()
                    imu_rpy = np.asarray(self.robot_state["imu_rpy"], dtype=float).reshape(3).copy()
                    gamepad_msg = self.robot_state["gamepad"]
                    odom_pos = np.asarray(self.robot_state["odom_pos"], dtype=float).reshape(3).copy()
                    odom_yaw = float(self.robot_state["odom_yaw"])
                    odom_quality = int(self.robot_state["odom_quality"])
                    odom_rx_t = float(self.robot_state["odom_rx_t"])
                    nav_v_xy = np.asarray(self.robot_state["nav_v_xy"], dtype=float).reshape(2).copy()
                    nav_active = int(self.robot_state["nav_active"])
                    nav_rx_t = float(self.robot_state["nav_rx_t"])

                # Feed the latest lidar odometry to the core estimator (fusion
                # itself happens inside core.step, gated by freshness/quality).
                if odom_rx_t > 0.0:
                    self.core.update_lidar_odom(
                        pos_map=odom_pos,
                        yaw_map=odom_yaw,
                        quality=odom_quality,
                        rx_walltime=odom_rx_t,
                    )

                # Y key logging trigger works even before we have all packets
                self._handle_log_trigger(gamepad_msg)
                # User "I"/point hard-stop (also works even before we have all packets)
                self._handle_zero_vel_trigger(gamepad_msg)
                # NOTE: RB is now repurposed as the prop-switch toggle (handled in
                # _update_mode_est). The old one-shot big-jump RB trigger is disabled.
                # self._handle_big_jump_trigger(gamepad_msg)

                if (not have_motor) or (not have_imu):
                    # Wait for first packets
                    if (now - last_print) > 1.0:
                        last_print = now
                    continue

                desired_v_xy = self._compute_desired_v_xy(gamepad_msg)

                # ===== LiDAR patrol (SELECT toggles; stick/B wins back) =====
                sel_now = bool(getattr(gamepad_msg, "select", 0)) if gamepad_msg is not None else False
                if sel_now and not self._last_select:
                    self._patrol_enable = not self._patrol_enable
                    print(f"[patrol] {'ENGAGED' if self._patrol_enable else 'OFF'} (SELECT)")
                self._last_select = sel_now
                b_now_pat = bool(getattr(gamepad_msg, "b", 0)) if gamepad_msg is not None else False
                stick_moved = bool(np.any(np.abs(desired_v_xy) > 1e-9))  # already deadzoned
                if self._patrol_enable and (b_now_pat or stick_moved):
                    self._patrol_enable = False
                    print(f"[patrol] OFF ({'B' if b_now_pat else 'stick override'})")
                if self._patrol_enable:
                    nav_fresh = (now - nav_rx_t) < float(getattr(self.lcm_cfg, "nav_cmd_stale_s", 0.3))
                    if nav_fresh and (nav_active == 1):
                        v_cap = float(getattr(self.lcm_cfg, "nav_cmd_vel_max", 0.5))
                        n_nav = float(np.hypot(float(nav_v_xy[0]), float(nav_v_xy[1])))
                        if n_nav > v_cap and n_nav > 1e-9:
                            nav_v_xy = nav_v_xy * (v_cap / n_nav)
                        desired_v_xy = nav_v_xy.astype(float).copy()
                    else:
                        # patrol engaged but no valid nav cmd -> hold position (v=0)
                        desired_v_xy = np.zeros(2, dtype=float)

                if bool(self._zero_vel_hold) or bool(self._y_hold_until_stance):
                    desired_v_xy[:] = 0.0
                # Demo override: walk slowly in -X (or any configured constant velocity)
                if bool(getattr(self.lcm_cfg, "demo_enable", False)):
                    desired_v_xy[0] = float(getattr(self.lcm_cfg, "demo_vx_mps", -0.20))
                    desired_v_xy[1] = float(getattr(self.lcm_cfg, "demo_vy_mps", 0.0))

                # Smooth command (rate limit) to avoid sudden speed jumps.
                dv_max = float(max(0.0, float(getattr(self.lcm_cfg, "cmd_dv_max_mps2", 0.0))))
                if dv_max > 1e-9:
                    dv_step = dv_max * float(dt)
                    dv = (desired_v_xy - self._v_cmd_filt).astype(float)
                    dv = np.clip(dv, -dv_step, +dv_step).astype(float)
                    self._v_cmd_filt = (self._v_cmd_filt + dv).astype(float)
                    desired_v_xy = self._v_cmd_filt.copy()
                # Track estimated driver mode (used to gate SAFE latch).
                self._update_mode_est(gamepad_msg)

                # ===== SAFE flag (upper-layer latch) =====
                # Trigger conditions:
                # 1) |roll| or |pitch| > 50deg
                # 2) any q outside [-1.06, +1.30]
                roll = float(imu_rpy[0])
                pitch = float(imu_rpy[1])
                rp_lim = float(np.deg2rad(float(self.lcm_cfg.safe_rp_deg)))
                # SAFE limit policy: the RT P4 stand and the LT zero-leg mode are
                # exempt from SAFE (P4 is constrained by its 1 Nm torque cap; the
                # LT mode sends zero force). q_min is relaxed while either is active.
                q_min = (float(self.lcm_cfg.safe_q_min_switch)
                         if (bool(self._switch_loop) or self._gait_mode == "mobile"
                             or self._lt_stand_t0 is not None)
                         else float(self.lcm_cfg.safe_q_min))
                q_max = float(self.lcm_cfg.safe_q_max)
                unsafe_tilt = (abs(roll) > rp_lim) or (abs(pitch) > rp_lim)
                unsafe_q = bool(np.any((q < q_min) | (q > q_max)))
                # User request: only trigger SAFE when the driver is in PD/PWMPD.
                # This avoids SAFE spam while the robot is in OFF/DAMP (e.g., being carried/tilted).
                safe_armed = int(self._mode_est) in (2, 3)  # PD or PWMPD
                # Automatic SAFE is disabled for debugging. B still commands DAMP + props off.
                if False and safe_armed and (not self._switch_loop) and (unsafe_tilt or unsafe_q):
                    # Request driver to enter DAMP by sending motor_pwm_lcmt.control_mode < 0.
                    # Also send a damping-like joint command (kp=0, kd=safe_damp_kd, tau=0) as an extra guard.
                    reason = []
                    if unsafe_tilt:
                        reason.append(f"tilt rp=[{np.rad2deg(roll):+.1f},{np.rad2deg(pitch):+.1f}]deg")
                    if unsafe_q:
                        reason.append(f"q_out q=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f}] (lim [{q_min:+.2f},{q_max:+.2f}])")
                    pause_s = float(max(0.0, float(self.lcm_cfg.safe_pause_s)))

                    # Stop logging on SAFE (treat as end of experiment segment)
                    if bool(self._log_enabled):
                        self._stop_log()

                    # disarm props + force DAMP in driver
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    self._publish_motor_pwm(np.full(6, pwm_min, dtype=float), control_mode=-1, force=True)
                    # LATCH props OFF: clear the arm flag so props stay stopped after the SAFE
                    # pause and only re-arm on a deliberate A press (otherwise they spin again).
                    self._prop_enable = False
                    self._close_prop_base_window("SAFE")
                    self._restore_rt_first_hop_l0()
                    # damping-like joints (in case driver is still in PD/PWMPD)
                    kd = float(self.lcm_cfg.safe_damp_kd)
                    self._publish_hopper_cmd(np.zeros(3, dtype=float), kp_joint=np.zeros(3, dtype=float), kd_joint=np.full(3, kd, dtype=float))

                    # Update our mode estimate to DAMP to prevent repeated SAFE triggers while the operator recovers.
                    self._mode_est = 1  # DAMP
                    self._safe_last_t = float(time.time())
                    print(f"[SAFE] TRIGGERED: {'; '.join(reason)} -> DAMP (pause {pause_s:.0f}s)")

                    # pause controller loop (do NOT backlog-catchup)
                    if pause_s > 0.0:
                        time.sleep(pause_s)
                    next_t = time.time() + dt
                    last_print = time.time()
                    continue

                control_enabled = int(self._mode_est) in (2, 3)
                # ModeE may use prop assistance only in an enabled HOPPING
                # configuration. MOBILE and OFF/DAMP are hard-disarmed.
                core_props_armed = bool(
                    control_enabled
                    and self._gait_mode == "hopping"
                    and self._prop_enable
                )
                self.core.set_props_armed(core_props_armed)
                # Always compute and send commands; underlying driver handles mode switching and safety
                tau_raw, pwm_us, info = self.core.step(
                    joint_pos=q,
                    joint_vel=qd,
                    imu_gyro_b=imu_gyro,
                    imu_acc_b=imu_acc,
                    imu_quat_wxyz=imu_quat,
                    imu_rpy=imu_rpy,
                    desired_v_xy_w=desired_v_xy,
                )

                # RT first-hop l0 override ends at the FIRST liftoff: the push
                # extended to switch_rb_first_hop_l0_m; hop 2 onwards uses the
                # normal leg_l0_m again.
                if (self._rt_l0_restore is not None) and int(info.get("liftoff", 0)) != 0:
                    self._restore_rt_first_hop_l0()

                # Enabled HOPPING + LT (stand-and-fold): catch the FRESH
                # stance bottom (push latch edge: vz ~ 0, leg loaded) and
                # freeze into the stand there instead of pushing.  The RM
                # fold starts immediately; MOBILE begins only when the
                # arms are down (see the _lt_stand_t0 branch below).
                # 2026-07-24 06:41 user: the stand begins AT push complete
                # (liftoff edge), not at the next bottom.  Legs hold the
                # RT stand pose (0.47 m, world-vertical foot) already in
                # the air, PROPS OFF, and the RM fold starts at the same
                # moment; the robot lands on the held leg while the arms
                # come down.
                if (bool(self._rm_lt_pending)
                        and bool(getattr(self.lcm_cfg, "switch_lt_stand", True))
                        and int(info.get("liftoff", 0)) != 0):
                    self._rm_lt_pending = False
                    self._rm_rt_pending = False
                    self._lt_stand_t0 = time.time()
                    self._close_prop_base_window("LT stand")
                    self._rm_start(
                        float(self.lcm_cfg.rm_mobile_rad),
                        "LT job (stand -> fold RM)",
                    )
                    print(
                        "[gait] LT STAND: push complete -> leg holds %.3f m "
                        "(tau cap %.1f Nm, props OFF), RM 0 -> %.1f rad; "
                        "MOBILE when RM reaches it"
                        % (float(self.lcm_cfg.switch_rb_leg_len_m),
                           float(self.lcm_cfg.switch_lt_tau_max_nm),
                           float(self.lcm_cfg.rm_mobile_rad))
                    )
                # Legacy LT (switch_lt_stand=False): at push end enter MOBILE
                # atomically; legs force-free mid-air while RM folds.
                elif bool(self._rm_lt_pending) and int(info.get("liftoff", 0)) != 0:
                    self._rm_lt_pending = False
                    self._rm_rt_pending = False
                    self._gait_mode = "mobile"
                    self._close_prop_base_window("LT -> MOBILE")
                    self._rm_start(
                        float(self.lcm_cfg.rm_mobile_rad),
                        "LT job (push end -> MOBILE)",
                    )
                    print("[gait] MOBILE: legs FORCE-FREE, props OFF")
                # RT: after P4 handoff, wait for the first push to finish
                # (liftoff) before unfolding RM +11.5 -> 0.
                # TEMP: skipped when rt_leg_only_no_prop_rm (pending never set).
                elif bool(self._rm_rt_pending) and int(info.get("liftoff", 0)) != 0:
                    self._rm_rt_pending = False
                    if not bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                        self._rm_start(
                            float(self.lcm_cfg.rm_hopping_rad),
                            "RT job (push end -> unfold RM)",
                        )
                        print(
                            "[gait] RT: push end -> RM +11.5 -> %.1f rad"
                            % float(self.lcm_cfg.rm_hopping_rad)
                        )

                if self._lt_stand_t0 is not None:
                    # LT stand-and-fold: P4-style world-vertical-foot hold
                    # from the moment the last push completes -- the LEG
                    # carries the landing and the stand (higher tau cap)
                    # while the RM arms fold to the mobile pose.  Props
                    # are OFF for the whole stand (2026-07-24 06:41 user).
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    tau_send, _err_m, _spd = self.core.compute_stand_swing_tau(
                        joint_pos=q_arr,
                        joint_vel=qd_arr,
                        leg_len_des_m=float(self.lcm_cfg.switch_rb_leg_len_m),
                        tau_max_nm=float(self.lcm_cfg.switch_lt_tau_max_nm),
                        imu_quat_wxyz=imu_quat,
                    )
                    tau_out_scale_applied = 1.0
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    pwm_us = np.full(6, pwm_min, dtype=float)
                    props_active = False
                    # Release to MOBILE once the arms are down (RM left its
                    # drive stage) or on the fail-safe timeout.
                    rm_done = int(self._rm_stage) != 2
                    timed_out = (
                        time.time() - float(self._lt_stand_t0)
                    ) >= float(self.lcm_cfg.switch_lt_timeout_s)
                    if rm_done or timed_out:
                        self._lt_stand_t0 = None
                        self._gait_mode = "mobile"
                        props_active = False
                        print(
                            "[gait] LT STAND done (%s) -> MOBILE: "
                            "legs FORCE-FREE, props OFF"
                            % ("RM reached" if rm_done else "TIMEOUT")
                        )
                elif self._rt_stand_t0 is not None:
                    # RT stand-and-unfold (2026-07-24 07:05): leg holds the
                    # P4 length UNCAPPED (inverted pendulum, leg carries
                    # the robot) while RM unfolds +11.5 -> 0 synchronized;
                    # props stay on the P4 baseline.  When RM reaches 0
                    # (or timeout) -> plain normal hopping, FLIGHT first.
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    tau_send, _err_m, _spd = self.core.compute_stand_swing_tau(
                        joint_pos=q_arr,
                        joint_vel=qd_arr,
                        leg_len_des_m=float(self.lcm_cfg.switch_rb_leg_len_m),
                        tau_max_nm=float(self.lcm_cfg.switch_rt_stand_tau_max_nm),
                        imu_quat_wxyz=imu_quat,
                    )
                    tau_out_scale_applied = 1.0
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    pwm_us = np.full(6, pwm_min, dtype=float)
                    if bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                        props_active = False
                    else:
                        prop_pwm_up = float(self.lcm_cfg.switch_rb_prop_base_pwm_us)
                        for grp in self.modee_cfg.prop_pwm_idx_per_arm:
                            for idx in grp:
                                ii = int(idx)
                                if 0 <= ii < 6:
                                    pwm_us[ii] = prop_pwm_up
                        props_active = True
                    rm_done = int(self._rm_stage) != 2
                    timed_out = (
                        time.time() - float(self._rt_stand_t0)
                    ) >= float(self.lcm_cfg.switch_rt_stand_timeout_s)
                    if rm_done or timed_out:
                        self._rt_stand_t0 = None
                        self._gait_mode = "hopping"
                        print(
                            "[gait] RT STAND done (%s) -> HOPPING "
                            "(normal FLIGHT, no first-hop overrides)"
                            % ("RM reached" if rm_done else "TIMEOUT")
                        )
                elif self._switch_loop:
                    # RT (P4) stand: ignore ModeE legs. Flight swing Cartesian PD
                    # toward a WORLD-vertical foot (quaternion) at
                    # switch_rb_leg_len_m, torque-capped. After
                    # switch_rb_pushdelay_s -> uncapped RT stand + RM unfold.
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    now_t = time.time()
                    if (self._rb_p4_t0 is not None) and \
                       (now_t - float(self._rb_p4_t0)) >= float(self.lcm_cfg.switch_rb_pushdelay_s):
                        # Leg placed -> uncapped stand; RM unfold starts NOW
                        # (synchronized drive, fast arms wait for the slow).
                        self._switch_loop = False
                        self._rb_p4_t0 = None
                        self._rt_stand_t0 = time.time()
                        if not bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                            self._rm_start(
                                float(self.lcm_cfg.rm_hopping_rad),
                                "RT stand (unfold RM, synced)",
                            )
                        print(
                            "[gait] RT STAND: leg holds %.3f m UNCAPPED "
                            "(%.0f Nm), RM +11.5 -> %.1f rad; HOPPING when "
                            "RM reaches it"
                            % (float(self.lcm_cfg.switch_rb_leg_len_m),
                               float(self.lcm_cfg.switch_rt_stand_tau_max_nm),
                               float(self.lcm_cfg.rm_hopping_rad))
                        )
                    tau_send, _err_m, _spd = self.core.compute_stand_swing_tau(
                        joint_pos=q_arr,
                        joint_vel=qd_arr,
                        leg_len_des_m=float(self.lcm_cfg.switch_rb_leg_len_m),
                        tau_max_nm=float(self.lcm_cfg.switch_rb_tau_max_nm),
                        imu_quat_wxyz=imu_quat,
                    )
                    tau_out_scale_applied = 1.0
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    pwm_us = np.full(6, pwm_min, dtype=float)
                    # TEMP leg-only: keep props OFF during P4. Otherwise force
                    # the upward 1100 us baseline.
                    if bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                        props_active = False
                    else:
                        prop_pwm_up = float(self.lcm_cfg.switch_rb_prop_base_pwm_us)
                        for grp in self.modee_cfg.prop_pwm_idx_per_arm:
                            for idx in grp:
                                ii = int(idx)
                                if 0 <= ii < 6:
                                    pwm_us[ii] = prop_pwm_up
                        props_active = True
                elif self._gait_mode == "mobile":
                    # MOBILE command: zero leg torque/gains and stopped props.
                    # Commands are still published for observation in OFF/DAMP;
                    # only the lower driver's outer PD/PWMPD mode decides
                    # whether any motor follows them.
                    tau_send = np.zeros(3, dtype=float)
                    tau_out_scale_applied = 1.0
                    props_active = False
                else:
                    # Normal mode: ModeE leg hopping ENABLED. Use ModeE's solved leg torque
                    # (output-limited). (Was temporarily zeroed for LB/RB-only testing.)
                    tau_send, tau_out_scale_applied = self._apply_tau_output_limit(tau_raw)
                    # Normal mode: props follow the A-switch. pwm_us keeps ModeE's real values
                    # (visible in lcm-spy); control_mode tells the bridge whether to spin.
                    props_active = bool(control_enabled and self._prop_enable)
                    # Prop base window (2026-07-24 01:36 rework): the support
                    # now rides INSIDE ModeE as a collective-ratio override
                    # (_open_prop_base_window), so pwm_us here is already the
                    # allocator's two-sided solution -- no per-motor clamping.
                    # Close the window once the RT unfold has finished:
                    # pending consumed (drive started at the first liftoff)
                    # and the stage has left "drive" (reached 0 -> hold/idle).
                    if (bool(self._hop_prop_base_active)
                            and (not bool(self._rm_rt_pending))
                            and int(self._rm_stage) != 2):
                        self._close_prop_base_window("RM unfold done")
                # RM M2006 sequence: refresh _rm_iq_des (rides inside hopper_cmd_lcmt below).
                self._update_rm()
                # MOBILE kiwi wheels: drive only in enabled MOBILE once the
                # RM fold is out of its drive stage (don't roll while the
                # arms are still folding); every other tick streams zeros
                # with enable=0. The Jetson driver applies its own X/B mode
                # gating on top, same class as rm_iq_des.
                if (
                    self._gait_mode == "mobile"
                    and control_enabled
                    and int(self._rm_stage) != 2
                ):
                    self._publish_wheel_cmd(
                        self._compute_wheel_cmd(gamepad_msg), enable=True
                    )
                else:
                    self._publish_wheel_cmd(
                        np.zeros(3, dtype=float), enable=False
                    )
                # Optional AK60-side damping in FLIGHT only (helps reduce oscillation without affecting takeoff).
                # We keep kp=0 and qd_des=0, so this acts like: tau += -kd * qd_motor (in motor frame).
                kd_flight = float(max(0.0, float(getattr(self.lcm_cfg, "ak60_flight_damp_kd", 0.0))))
                kd_stance = float(max(0.0, float(getattr(self.lcm_cfg, "ak60_stance_damp_kd", 0.0))))
                in_stance = int(info.get("stance", 0)) != 0
                # If Y-latched hard-hold is active, release it as soon as we enter STANCE.
                if bool(in_stance) and bool(self._y_hold_until_stance):
                    self._y_hold_until_stance = False
                    # Only disable the core hold if the user hasn't separately latched it via point/I.
                    if not bool(self._zero_vel_hold):
                        try:
                            self.core.user_zero_velocity_hold(False)
                        except Exception:
                            pass
                # P4 supplies its own damping; MOBILE must remain truly free.
                kd_use = 0.0 if (self._switch_loop or self._gait_mode == "mobile"
                                 or self._lt_stand_t0 is not None
                                 or self._rt_stand_t0 is not None) \
                    else float(kd_stance if in_stance else kd_flight)
                if kd_use > 0.0:
                    self._publish_hopper_cmd(
                        tau_send,
                        kp_joint=np.zeros(3, dtype=float),
                        kd_joint=np.full(3, kd_use, dtype=float),
                    )
                else:
                    self._publish_hopper_cmd(tau_send)
                # Propeller on/off is carried in control_mode (we do NOT zero pwm to stop):
                #   props_active -> prop_ctrl_mode_on (3); otherwise prop_ctrl_mode_off (1).
                # pwm_values are always the real commanded values, so they stay visible in
                # lcm-spy even when props are OFF; px4_bridge only spins when control_mode==3.
                # OFF/DAMP is a whole-robot disarm even if A was previously
                # latched or a stale transition flag survived.
                props_active = bool(props_active and control_enabled)
                prop_cm = (int(self.lcm_cfg.prop_ctrl_mode_on) if bool(props_active)
                           else int(self.lcm_cfg.prop_ctrl_mode_off))
                self._publish_motor_pwm(pwm_us, control_mode=prop_cm)

                # Periodic status: driver mode, hop/switch phase, SAFE, foot pos + leg length.
                print_hz = float(self.lcm_cfg.print_hz)
                if print_hz <= 0.0:
                    do_print = True
                else:
                    do_print = (now - last_print) >= (1.0 / print_hz)
                if do_print:
                    last_print = now
                    dt_rate = float(now - float(self._rx_rate_t0))
                    if dt_rate >= 0.5:
                        self._rx_motor_hz = float(self._rx_motor_n) / dt_rate
                        self._rx_imu_hz = float(self._rx_imu_n) / dt_rate
                        self._rx_motor_n = 0
                        self._rx_imu_n = 0
                        self._rx_rate_t0 = float(now)
                    print(
                        self._format_status_line(
                            info=info,
                            q=q,
                            roll=roll,
                            pitch=pitch,
                            safe_armed=safe_armed,
                            now_t=now,
                        )
                    )

                # Log (if enabled). Gate here: building all the np arrays and the
                # dict(info) copy costs real time at 500Hz even when logging is off.
                if not bool(self._log_enabled):
                    continue
                self._log_step(
                    wall_time_s=float(now),
                    q=q,
                    qd=qd,
                    imu_quat=imu_quat,
                    imu_gyro=imu_gyro,
                    imu_acc=imu_acc,
                    imu_rpy=imu_rpy,
                    desired_v_xy=desired_v_xy,
                    tau_cmd=np.asarray(tau_send, dtype=float).reshape(3),
                    tau_raw=np.asarray(tau_raw, dtype=float).reshape(3),
                    tau_out_scale_applied=float(tau_out_scale_applied),
                    pwm_us=np.asarray(pwm_us, dtype=float).reshape(6),
                    info=dict(info),
                    props_active=bool(props_active),
                    prop_ctrl_mode=int(prop_cm),
                )

        finally:
            # Zero all outgoing commands so the robot doesn't keep acting on stale values.
            self._publish_zero_outputs()
            # Ensure file is closed when controller stops
            self._stop_log()


