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
    max_cmd_vel: float = 0.8
    stick_deadzone: float = 0.10
    # ---- LiDAR patrol (hopper_nav_cmd_lcmt from lidar_perception/patrol.py) ----
    # SELECT toggles patrol; while engaged the nav velocity replaces the stick.
    # ANY stick input beyond the deadzone (or B) immediately disengages.
    # Nav command staleness gate: older than this -> treat as inactive (robot
    # falls back to zero-velocity stick behavior, patrol stays engaged).
    nav_cmd_stale_s: float = 0.3
    nav_cmd_vel_max: float = 0.5  # hard cap on patrol velocity (below max_cmd_vel)
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
    safe_q_max: float = 1.4835    # +85 deg (mechanical extend limit)
    # 2026-07-07: the old LB "parking" leg-retraction loop (P1 approach / P2
    # continue / P3 settle + reverse-prop stages) was DELETED per user request.
    # LT is now: press -> RM pre-zero + arm; at the NEXT push end (liftoff edge)
    # the legs go to ZERO force (pure tau_ff=0, kp=kd=0) and the RM motors run
    # their tuned LT job (-11 rad -> hold -> re-zero). Exit via RT (P4) or B.
    # SAFE q-range guard is relaxed while legs are force-free / in P4:
    safe_q_min_switch: float = -1.0472    # -60 deg (SAFE guard only, in LT/P4 modes)
    # RT "stand up, then enter hopping" (Phase 4), standalone (no prerequisite).
    # On RT (2026-07-07 spec):
    #   - legs: position control to switch_rb_target_rad (-0.1 rad) with the torque
    #     hard-capped at switch_rb_tau_max_nm (1 Nm);
    #   - props: forced ON at the UPWARD baseline switch_rb_prop_base_pwm_us (1100,
    #     >1000 = forward thrust) for the whole phase;
    #   - after switch_rb_pushdelay_s (2 s) the loop exits into the hopping cycle
    #     (push phase): ModeE takes over legs AND prop balancing, and the RM motors
    #     start their RT job (+11 rad, see the RM block below).
    # During the FIRST hop cycle after this handoff (until the 2nd liftoff), the
    # propeller PWM is floored at hop_prop_base_pwm_us (ModeE attitude adds on top).
    # P4 position gains (also used nowhere else now that P1-P3 are gone):
    switch_phase3_kp_nm_per_rad: float = 12.0  # P4 position gain
    switch_phase3_kd_nm_s_per_rad: float = 0.4 # P4 damping gain
    switch_rb_target_rad: float = -0.1      # RT leg stand-up target (rad, LCM q)
    switch_rb_tau_max_nm: float = 1.0       # RT leg torque cap (Nm)
    switch_rb_pushdelay_s: float = 2.0      # stand this long, then enter hopping (push)
    switch_rb_prop_base_pwm_us: float = 1100.0  # prop baseline during the RT stand (us)
    hop_prop_base_pwm_us: float = 1200.0    # prop PWM floor during the first post-RB hop cycle
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
    ak60_flight_damp_kd: float = 0.02
    ak60_stance_damp_kd: float = 0.1

    # ===== Command shaping / demo mode =====
    # To keep the hop process smooth, we rate-limit the commanded desired velocity.
    # This prevents sudden step changes in Raibert target_xy (and resulting speed jumps).
    cmd_dv_max_mps2: float = 0.0
    # Simple demo: override desired velocity to a fixed value.
    # Keep disabled by default so user/gamepad velocity command directly drives Raibert foot placement.
    demo_enable: bool = False
    demo_vx_mps: float = 0.0
    demo_vy_mps: float = 0.0  # Zero velocity - stationary hopping with Raibert stabilization

    # Motor velocity (2026-07-07, reversed from 07-06): hopper_data_lcmt.qd now
    # carries the AK60 CAN-reported velocity and ALL upper layers consume it
    # directly -- ModeECore no longer differentiates q (qd_kin_from_q_diff=False
    # in core.py; the EMA+MA conditioning chain still applies on top).

    # ===== RM M2006 (3x, output-shaft rad; the JETSON DRIVER owns the zero) =====
    # One sequence, two triggers (parameters tuned 2026-07-07 with run_rm_test.py:
    # kp 2.0 / kd 0.2 / cap 5 A):
    #   LT job (target rm_lt_target_rad = -11): starts at the NEXT push end
    #     (liftoff edge) after the LT press; the legs go force-free (tau=0) there.
    #   RT job (target rm_rt_target_rad = +11): starts at the RT->hopping handoff
    #     (when the legs finish the 2s pre-push stand and ModeE resumes).
    # Sequence: current-mode PD to the target; when ALL three are within
    # rm_reach_tol_rad, hold rm_hold_s, then pulse hopper_cmd_lcmt.rm_set_zero so
    # the Jetson driver latches THAT position as the new rm_q zero; then 0 A.
    # Current only flows while the driver is in PD/PWMPD (X armed); B aborts.
    rm_lt_target_rad: float = -11.0      # LT job target (rad, from current zero)
    rm_rt_target_rad: float = +11.0      # RT job target (rad, from current zero)
    rm_kp_a_per_rad: float = 2.0         # current-mode PD: A per rad
    rm_kd_a_per_rad_s: float = 0.2       # current-mode PD: A per rad/s
    rm_iq_max_a: float = 5.0             # |current| cap during the drive (A)
    rm_reach_tol_rad: float = 0.3        # "in place" tolerance (all 3 motors)
    rm_hold_s: float = 1.0               # hold at target before the re-zero (s)


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

        # CSV logger (starts on gamepad Y press; stops when program exits)
        self._log_enabled = False
        self._log_fp = None

        # Desired velocity command smoothing (rate limiter)
        self._v_cmd_filt = np.zeros(2, dtype=float)
        self._log_writer = None
        self._log_path = None
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
        # RT (P4) stand mode: while active, ignore ModeE legs and position-control
        # to switch_rb_target_rad with props at the 1100 baseline; exits into
        # hopping after switch_rb_pushdelay_s. (The old LB P1-P3 parking loop was
        # deleted 2026-07-07.)
        self._switch_loop: bool = False
        self._mode_last_lb: bool = False
        self._mode_last_rb: bool = False
        # RB/RT phase-4 start time (fixed 2s stand, then hop entry), and the
        # post-RB prop base-PWM window.
        self._rb_p4_t0: float | None = None
        self._hop_prop_base_active: bool = False
        self._hop_prop_base_liftoff_count: int = 0
        # LT zero-leg mode: entered at the liftoff edge after an LT press. Legs
        # get pure tau_ff = 0 (kp=kd=0, i.e. force-free) while the RM job runs;
        # exits via RT (enters P4) or B (DAMP).
        self._lt_zero_legs: bool = False
        # A button: standalone propeller master switch (normal mode, outside switch loop).
        self._prop_enable: bool = False
        # RM M2006 desired torque current (A); sent inside every hopper_cmd_lcmt.
        self._rm_iq_des = np.zeros(3, dtype=float)
        # RM sequence: 0 idle (0 A), 1 drive to _rm_target, 2 hold before re-zero.
        # LT job (-11) starts at the NEXT push end (liftoff edge) after the LT
        # press; RT job (+11) starts at the RT->hopping handoff. B resets to idle.
        self._rm_stage: int = 0
        self._rm_target: float = 0.0
        self._rm_hold_t0: float = 0.0
        # LT armed flag: set at the LT press, consumed at the next liftoff edge.
        self._rm_lt_pending: bool = False
        # rm_set_zero pulse deadline: hopper_cmd_lcmt.rm_set_zero = 1 until then
        # (the Jetson driver latches the new zero on the 0->1 edge).
        self._rm_zero_until: float = 0.0

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

        # LT: pre-zero at the press; at the NEXT push end (liftoff edge, see the
        # main loop) the legs go FORCE-FREE (tau=0, kp=kd=0) and the RM motors
        # run the tuned LT job (-11 rad -> hold -> re-zero).
        if bool(lb_now) and (not bool(self._mode_last_lb)):
            if bool(self._rm_lt_pending) or bool(self._lt_zero_legs) or int(self._rm_stage) != 0:
                print("[rm] LT ignored (LT job already pending/active)")
            else:
                self._rm_lt_pending = True
                self._rm_prezero("LT press")
                print("[rm] LT ARMED -> at NEXT push end: legs 0 force + RM drive to %+.1f rad"
                      % float(self.lcm_cfg.rm_lt_target_rad))
        self._mode_last_lb = bool(lb_now)

        if bool(rb_now) and (not bool(self._mode_last_rb)):
            # RT (2026-07-07 spec, standalone -- no LB parking prerequisite):
            # legs position-control to switch_rb_target_rad (-0.1 rad) capped at
            # switch_rb_tau_max_nm (1 Nm), props forced ON at the upward baseline
            # switch_rb_prop_base_pwm_us (1100); after switch_rb_pushdelay_s (2 s)
            # exit into hopping (_enter_hop_from_rb), where ModeE takes over
            # legs + prop balance and the RM motors start their RT job (+11).
            if bool(self._switch_loop):
                print("[switch_loop] RT ignored (P4 stand already active)")
            else:
                self._switch_loop = True
                self._rb_p4_t0 = time.time()
                self._rm_lt_pending = False
                self._lt_zero_legs = False   # RT takes over from the LT zero-leg mode
                # RM pre-zero at the PRESS: the 2s P4 stand gives the zero plenty
                # of time to latch before the RT drive starts at hop entry.
                self._rm_prezero("RT press")
                print("[switch_loop] RT -> P4 legs to %.2frad (cap %.1fNm) + props base %.0fus; "
                      "enter hopping after %.1fs"
                      % (float(self.lcm_cfg.switch_rb_target_rad),
                         float(self.lcm_cfg.switch_rb_tau_max_nm),
                         float(self.lcm_cfg.switch_rb_prop_base_pwm_us),
                         float(self.lcm_cfg.switch_rb_pushdelay_s)))
        self._mode_last_rb = bool(rb_now)

        # Edge-triggered transitions (same priority order as Pi)
        # B = stop BOTH legs (DAMP) and props. Always clears props on the B edge.
        if bool(b_now) and (not bool(self._mode_last_b)):
            self._prop_enable = False
            self._hop_prop_base_active = False   # B stops everything: drop the prop base window
            # B also aborts the RM sequence, the LT zero-leg mode and the RT P4
            # stand (driver gates everything to DAMP anyway; this keeps the
            # upper-layer state machines consistent).
            self._rm_stage = 0
            self._rm_lt_pending = False
            self._lt_zero_legs = False
            self._switch_loop = False
            self._rb_p4_t0 = None
            self._rm_iq_des = np.zeros(3, dtype=float)
            self._rm_zero_until = 0.0
            if int(self._mode_est) != DAMP:
                self._mode_est = DAMP
            print("[prop] OFF (B) -> legs DAMP + props stop (control_mode=%d)"
                  % int(self.lcm_cfg.prop_ctrl_mode_off))
        elif bool(x_now) and (not bool(self._mode_last_x)) and (int(self._mode_est) != PD):
            self._mode_est = PD
        elif bool(a_now) and (not bool(self._mode_last_a)):
            # A = propeller switch (parallel to X for legs): arm props ON. They follow the
            # commanded pwm_values; control_mode tells px4_bridge to spin. B turns it off.
            self._prop_enable = True
            print("[prop] ON (A) -> props armed (control_mode=%d, follow pwm_values)"
                  % int(self.lcm_cfg.prop_ctrl_mode_on))
            if int(self._mode_est) == PD:
                self._mode_est = PWMPD
        elif bool(start_now) and (not bool(self._mode_last_start)) and (int(self._mode_est) == DAMP):
            self._mode_est = OFF

        self._mode_last_b = bool(b_now)
        self._mode_last_x = bool(x_now)
        self._mode_last_a = bool(a_now)
        self._mode_last_start = bool(start_now)

    def _enter_hop_from_rb(self) -> None:
        """RT Phase-4 stand has finished: exit into hopping so ModeE resumes, and
        arm the propeller PWM base floor for the FIRST hop cycle (until the 2nd liftoff)."""
        self._switch_loop = False
        self._rb_p4_t0 = None
        self._hop_prop_base_active = True
        self._hop_prop_base_liftoff_count = 0
        # RM M2006 RT job: entering the push/hopping cycle -> drive to
        # rm_rt_target_rad (+11) and re-zero there (runs during hopping).
        self._rm_start(float(self.lcm_cfg.rm_rt_target_rad), "RT job (hop entry)")
        print("[switch_loop] RT -> ENTER hopping (ModeE legs + prop balance); prop PWM base=%.0f for first cycle (until 2nd liftoff)"
              % float(self.lcm_cfg.hop_prop_base_pwm_us))

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
        # RM re-zero pulse: the Jetson driver latches the current RM position as
        # the new rm_q zero on the 0->1 edge (see _update_rm).
        msg.rm_set_zero = 1 if time.time() < float(self._rm_zero_until) else 0
        self.lc.publish("hopper_cmd_lcmt", msg.encode())

    def set_rm_iq_des(self, iq_a) -> None:
        """Set the desired M2006 torque current (A, per motor, clipped +/-10).
        Takes effect on every subsequent hopper_cmd_lcmt publish; the Jetson
        driver only forwards it in PD/PWMPD mode (gamepad X), B zeroes it."""
        self._rm_iq_des = np.clip(np.asarray(iq_a, dtype=float).reshape(3), -10.0, 10.0)

    def _rm_prezero(self, label: str) -> None:
        """Pulse rm_set_zero NOW (current RM position becomes rm_q = 0).

        Called at the LT/RT PRESS, i.e. well BEFORE the drive starts (LT: the
        drive begins at the next liftoff, >= one hop away; RT: 2 s of P4 stand).
        Doing the pre-zero here means the later drive starts EXACTLY at the
        push-end / hop-entry instant with no settling delay in between."""
        self._rm_zero_until = time.time() + 0.1
        print("[rm] %s -> PRE-ZERO now (rm_q = 0 here); drive will start from this zero" % label)

    def _rm_start(self, target_rad: float, label: str) -> None:
        """Begin the RM drive IMMEDIATELY (pre-zero already happened at the
        button press): PD to target_rad, hold rm_hold_s, re-zero at the target."""
        self._rm_target = float(target_rad)
        self._rm_stage = 2
        print("[rm] %s -> drive to %+.1f rad (kp %.1f kd %.2f cap %.1fA), re-zero there"
              % (label, self._rm_target, float(self.lcm_cfg.rm_kp_a_per_rad),
                 float(self.lcm_cfg.rm_kd_a_per_rad_s), float(self.lcm_cfg.rm_iq_max_a)))

    def _update_rm(self) -> None:
        """RM M2006 sequence, run once per control step (writes _rm_iq_des).

        Stage 2 (drive): current-mode PD to _rm_target, capped at rm_iq_max_a.
        When ALL three are within rm_reach_tol_rad -> stage 3. (The pre-zero
        happens at the LT/RT press itself, see _rm_prezero; stage 1 is only
        used as a settling stage if a caller ever starts with it.)
        Stage 3 (hold): keep the PD hold for rm_hold_s, then pulse rm_set_zero
        again (THIS position becomes the new rm_q zero) and idle.
        Requires fresh feedback: if rm_online != 7 the command is forced to 0 A
        (the sequence stays in its stage and resumes when feedback returns).
        """
        if int(self._rm_stage) == 0:
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
                print("[rm] pre-zero done (q=[%+.3f %+.3f %+.3f]) -> drive to %+.1f rad"
                      % (*rm_q, target))
            return
        if int(self._rm_stage) == 2:
            if bool(np.all(np.abs(rm_q - target) <= float(self.lcm_cfg.rm_reach_tol_rad))):
                self._rm_stage = 3
                self._rm_hold_t0 = now_t
                print("[rm] reached %+.1f rad -> hold %.1fs, then re-zero"
                      % (target, float(self.lcm_cfg.rm_hold_s)))
        elif int(self._rm_stage) == 3:
            if (now_t - float(self._rm_hold_t0)) >= float(self.lcm_cfg.rm_hold_s):
                self._rm_zero_until = now_t + 0.1   # 0.1s rm_set_zero pulse
                self._rm_stage = 0
                self._rm_iq_des = np.zeros(3, dtype=float)
                print("[rm] RE-ZERO: q was [%+.3f %+.3f %+.3f] -> rm_q reads 0 here; idle (0 A)"
                      % tuple(rm_q))
                return
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
            # Overwrite the same file each run (user request).
            log_name = os.environ.get("MODEE_LOG_NAME", "modee_latest.csv")
            path = os.path.join(logs_dir, log_name)

            fp = open(path, "w", newline="")
            writer = csv.writer(fp)

            header = [
                "wall_time_s",
                "t_s",
                "phase",
                "stance",
                "compress_active",
                "push_started",
                "touchdown",
                "liftoff",
                "apex",
                "s2s_active",
                "status",
                # joints
                "q0",
                "q1",
                "q2",
                "qd0",
                "qd1",
                "qd2",
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
                "z_lo_m",
                "vz_lo_m_s",
                "z_apex_actual_m",
                "apex_err_int",
                "v_to_cmd_m_s",
                "desired_vz_from_apex_m_s",
                "hop_height_m",
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
        except Exception as e:
            # Don't kill controller if logging fails.
            self._log_enabled = False
            self._log_fp = None
            self._log_writer = None
            self._log_path = None

    def _stop_log(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.flush()
            except Exception:
                pass
            try:
                self._log_fp.close()
            except Exception:
                pass
        self._log_enabled = False
        self._log_fp = None
        self._log_writer = None
        self._log_path = None
        self._log_last_flush_t = 0.0

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
        if bool(self._switch_loop) or bool(self._lt_zero_legs):
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
        # Minimal status: phase + leg-kinematics base velocity (the ONLY velocity source
        # of the estimator now). v_leg_w = R_wb @ (-foot_vdot_b - omega_b x foot_b);
        # in STANCE it directly becomes v_hat, in FLIGHT it is informational only.
        in_stance = bool(int(info.get("stance", 0)))
        if in_stance:
            ph = "STANCE:COMP" if int(info.get("compress", 0)) else "STANCE:PUSH"
        else:
            ph = "FLIGHT"
        v_leg_w = np.asarray(info.get("v_meas_foot_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
        # LiDAR / patrol tags: +LIDAR when odom is fresh-and-fused this tick,
        # +PATROL(wpN) while the SELECT patrol mode drives the velocity command.
        lidar_tag = ""
        if int(info.get("lidar_fresh", 0)):
            p_l = np.asarray(info.get("lidar_pos_map", [np.nan] * 3), dtype=float).reshape(3)
            lidar_tag = f" +LIDAR[{p_l[0]:+.2f},{p_l[1]:+.2f}]"
        patrol_tag = ""
        if bool(self._patrol_enable):
            patrol_tag = f" +PATROL(wp{int(self.robot_state.get('nav_wp_index', -1))})"
        return (
            f"[{ph}]{lidar_tag}{patrol_tag} "
            f"v_leg_w=[{v_leg_w[0]:+.3f},{v_leg_w[1]:+.3f},{v_leg_w[2]:+.3f}]"
        )

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
            s2s_active = int(info.get("s2s_active", 0))
            status = str(info.get("status", ""))

            if stance:
                phase = "STANCE:COMP" if int(info.get("compress", 0)) else "STANCE:PUSH"
            else:
                phase = "FLIGHT"

            f_tau_delta = np.asarray(info.get("f_tau_delta", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
            f_contact_w = np.asarray(info.get("f_contact_w", [np.nan, np.nan, np.nan]), dtype=float).reshape(3)
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
            z_lo_m = float(info.get("z_lo_m", float("nan")))
            vz_lo_m_s = float(info.get("vz_lo_m_s", float("nan")))
            z_apex_actual_m = float(info.get("z_apex_actual_m", float("nan")))
            apex_err_int = float(info.get("apex_err_int", float("nan")))
            v_to_cmd_m_s = float(info.get("v_to_cmd_m_s", float("nan")))
            desired_vz_from_apex_m_s = float(info.get("desired_vz_from_apex_m_s", float("nan")))
            hop_height_m = float(info.get("hop_height_m", float("nan")))
            mpc_u0 = np.asarray(info.get("mpc_u0", [0.0, 0.0, 0.0]), dtype=float).reshape(3)

            row = [
                float(wall_time_s),
                float(info.get("t", float("nan"))),
                phase,
                stance,
                compress,
                push_started,
                touchdown,
                liftoff,
                apex,
                s2s_active,
                status,
                float(q[0]),
                float(q[1]),
                float(q[2]),
                float(qd[0]),
                float(qd[1]),
                float(qd[2]),
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
                float(z_lo_m),
                float(vz_lo_m_s),
                float(z_apex_actual_m),
                float(apex_err_int),
                float(v_to_cmd_m_s),
                float(desired_vz_from_apex_m_s),
                float(hop_height_m),
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

            # Periodic flush to reduce data loss without killing performance
            t_now = float(time.time())
            if (t_now - float(self._log_last_flush_t)) >= 0.25:
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
                         if (bool(self._switch_loop) or bool(self._lt_zero_legs))
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
                    self._hop_prop_base_active = False
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

                # LT-armed job: at the push end (liftoff edge), legs go FORCE-FREE
                # (tau=0, kp=kd=0, see below) and the RM drive to rm_lt_target_rad
                # starts. Exits via RT (P4) or B.
                if bool(self._rm_lt_pending) and int(info.get("liftoff", 0)) != 0:
                    self._rm_lt_pending = False
                    self._lt_zero_legs = True
                    self._rm_start(float(self.lcm_cfg.rm_lt_target_rad), "LT job (push end)")
                    print("[rm] LT: legs FORCE-FREE (tau=0) while the RM job runs")

                if self._switch_loop:
                    # RT (P4) stand: ignore ModeE legs. Position control to
                    # switch_rb_target_rad (-0.1 rad) hard-capped at
                    # switch_rb_tau_max_nm (1 Nm), sent as pure tau_ff (kp=kd=0
                    # to the driver). Props forced ON at the upward 1100 base.
                    # After switch_rb_pushdelay_s -> _enter_hop_from_rb.
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    now_t = time.time()
                    if (self._rb_p4_t0 is not None) and \
                       (now_t - float(self._rb_p4_t0)) >= float(self.lcm_cfg.switch_rb_pushdelay_s):
                        self._enter_hop_from_rb()
                    q_des4 = float(self.lcm_cfg.switch_rb_target_rad)
                    kp4 = float(self.lcm_cfg.switch_phase3_kp_nm_per_rad)
                    kd4 = float(self.lcm_cfg.switch_phase3_kd_nm_s_per_rad)
                    cap4 = float(abs(float(self.lcm_cfg.switch_rb_tau_max_nm)))
                    tau_send = np.clip(kp4 * (q_des4 - q_arr) - kd4 * qd_arr, -cap4, cap4).astype(float)
                    tau_out_scale_applied = 1.0
                    # Props: forced ON at the UPWARD baseline (1100us, >1000 =
                    # forward thrust) for the whole 2s phase. The hopping handoff
                    # (_enter_hop_from_rb) then floors them at hop_prop_base_pwm_us
                    # while ModeE balance takes over.
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    pwm_us = np.full(6, pwm_min, dtype=float)
                    prop_pwm_up = float(self.lcm_cfg.switch_rb_prop_base_pwm_us)
                    for grp in self.modee_cfg.prop_pwm_idx_per_arm:
                        for idx in grp:
                            ii = int(idx)
                            if 0 <= ii < 6:
                                pwm_us[ii] = prop_pwm_up
                    props_active = True
                elif self._lt_zero_legs:
                    # LT mode: legs FORCE-FREE (pure tau_ff = 0, kp=kd=0) while the
                    # RM job runs (drive to -11 -> hold -> re-zero -> idle). Stays
                    # force-free until RT (P4) or B. Props follow the normal
                    # A-switch path with ModeE's pwm untouched.
                    tau_send = np.zeros(3, dtype=float)
                    tau_out_scale_applied = 1.0
                    props_active = bool(self._prop_enable)
                else:
                    # Normal mode: ModeE leg hopping ENABLED. Use ModeE's solved leg torque
                    # (output-limited). (Was temporarily zeroed for LB/RB-only testing.)
                    tau_send, tau_out_scale_applied = self._apply_tau_output_limit(tau_raw)
                    # Normal mode: props follow the A-switch. pwm_us keeps ModeE's real values
                    # (visible in lcm-spy); control_mode tells the bridge whether to spin.
                    props_active = bool(self._prop_enable)
                    # Post-RB prop base: for the FIRST hop cycle after an RB handoff, floor the
                    # propeller PWM at hop_prop_base_pwm_us (props idle there; ModeE attitude adds
                    # on top). The window ends at the 2nd liftoff. Props still only spin if A is on.
                    if bool(self._hop_prop_base_active):
                        if int(info.get("liftoff", 0)) != 0:
                            self._hop_prop_base_liftoff_count += 1
                            if self._hop_prop_base_liftoff_count >= 2:
                                self._hop_prop_base_active = False
                        if bool(self._hop_prop_base_active):
                            base = float(self.lcm_cfg.hop_prop_base_pwm_us)
                            pwm_us = np.asarray(pwm_us, dtype=float).reshape(6).copy()
                            for grp in self.modee_cfg.prop_pwm_idx_per_arm:
                                for idx in grp:
                                    ii = int(idx)
                                    if 0 <= ii < 6 and float(pwm_us[ii]) < base:
                                        pwm_us[ii] = base
                # RM M2006 sequence: refresh _rm_iq_des (rides inside hopper_cmd_lcmt below).
                self._update_rm()
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
                # In the P4 stand the PC-side PD already includes damping, and in the
                # LT zero-leg mode the legs must be truly FORCE-FREE -> pure tau_ff,
                # no driver-side kd, in both cases.
                kd_use = 0.0 if (self._switch_loop or self._lt_zero_legs) \
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
                )

        finally:
            # Zero all outgoing commands so the robot doesn't keep acting on stale values.
            self._publish_zero_outputs()
            # Ensure file is closed when controller stops
            self._stop_log()


