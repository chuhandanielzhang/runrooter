from __future__ import annotations

import csv
import json
import math
import os
import socket
import sys
import time
import threading
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

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
from modee.approach_law import approach_twist, dead_reckon_step


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
    # Cross-host dashboard state: controller runs on Jetson, viewer on PC.
    # Subnet-directed broadcast (not 255.255.255.255): the limited broadcast
    # only egresses via the default-route iface = Jetson WiFi, never reaching
    # the PC on the wired 192.168.1.0/24 link (2026-08-08).
    dashboard_udp_host: str = "192.168.1.255"
    dashboard_udp_port: int = 5557
    # 2026-07-11 per user (MATLAB values): posvellim = 0.15 m/s. MATLAB's
    # position loop clamps desiredVel to +-0.15; we have no position loop, so
    # the equivalent is capping the stick/nav v_des at 0.15 (was 0.8).
    max_cmd_vel: float = 0.4
    stick_deadzone: float = 0.10
    # HOPPING height knob: push LEFT stick Y once, then release to neutral.
    # Linux/Xbox LY is negative upward: up-release adds one step, down-release
    # subtracts one step.  The two thresholds provide hysteresis, so holding
    # the stick or neutral noise cannot repeat the command.
    hop_height_stick_enable: bool = True
    hop_height_stick_trigger: float = 0.70
    hop_height_stick_release: float = 0.20
    hop_height_step_m: float = 0.02
    hop_height_min_m: float = 0.03
    hop_height_max_m: float = 0.13
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
    # 2026-08-02: 30 -> 9 = the real motor limit.  ModeECore now caps at
    # 9 Nm internally (proportional + prop residual), so this output-side
    # limiter is a pure backstop that should never bind.
    tau_out_max_nm: float | None = 9
    # SAFE flag:
    # - If triggered, we request hopper_driver to enter DAMP (same as pressing B),
    #   and pause the Python controller loop for a few seconds.
    safe_rp_deg: float = 50.0
    safe_q_min: float = -1.06     # CASE: retract limit (with kAk60LcmQOffsetRad=-60 deg)
    safe_q_max: float = 1.40      # CASE: extend limit (user: do not past q=1.4)
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
    # 2026-07-23 09:23 user: 0.428 -> 0.46; 2026-07-24 06:47: 0.46 -> 0.47;
    # 2026-07-24 07:15: 0.47 -> 0.475.
    switch_rb_leg_len_m: float = 0.46     # RT/P4 / LT stand target leg length (m)
    switch_rb_tau_max_nm: float = 2.0       # RT leg torque cap (Nm)
    # AK60 motor-internal velocity damping during RT/P4 leg placement.
    # Sent through the MIT kd field, using the driver's high-rate velocity;
    # this supplements the Cartesian damping already present in tau_ff.
    switch_rb_ak60_damp_kd: float = 0.2
    switch_rb_pushdelay_s: float = 1.0      # stand this long, then enter hopping (push)
    # First hop out of RT: temporarily override leg_l0_m so the spring push
    # extends toward full leg (~0.554 m at AK60 setZero / q_lcm≈-1.047).
    # Restored at the first liftoff. 2026-07-24 03:52: 0.53 -> 0.54 (~1.4 cm
    # margin below full extension; stroke from 0.47 stand ~= 7 cm).
    switch_rb_first_hop_l0_m: float = 0.55
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
    # 2026-08-03: idle -> 1200.
    hop_prop_base_pwm_us: float = 1200.0
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
    # ---- RT (MOBILE -> HOPPING), 2026-08-01 simplification ----
    # P4 places the leg (1 s, capped, props 1200); after
    # switch_rb_pushdelay_s the controller enters NORMAL HOPPING directly:
    #   - first hop = normal cycle law with ONLY a special leg length
    #     (l0 = switch_rb_first_hop_l0_m, restored at the first liftoff);
    #   - props = normal ModeE allocator from tick one (no forced 1600
    #     PWM, no base window);
    #   - RM unfolds +11.5 -> 0 IN PARALLEL with the hopping (synced
    #     drive, non-blocking).
    # The 2026-07-24 RT-STAND phase (uncapped stand + prop 1600 + wait
    # for RM) is retired; its switch_rt_stand_* params below are unused
    # but kept for rollback.
    # 10 Nm = the AK60 driver's own clamp, i.e. effectively uncapped.
    switch_rt_stand_tau_max_nm: float = 10.0
    switch_rt_stand_timeout_s: float = 4.0
    # While RM unfolds in RT-STAND, force EVERY mapped prop motor to this
    # PWM (2026-07-24 07:15: 1500; 07:21: 1600 each).  P4 place still uses
    # switch_rb_prop_base_pwm_us (1200).
    # Match hop idle 1200.
    switch_rt_stand_prop_pwm_us: float = 1200.0
    # SLIP-style loaded stand (2026-07-24 07:24 user: "PD加大 一定要像slip
    # 那样撑起来").  Log 071232 showed the leg sagging ~2 cm during the RM
    # unfold: the stand reused the FLIGHT swing gains (kp_z 1300 N/m, no
    # gravity FF), which sag mg/kp ~ 5 cm under load.  RT-STAND now uses
    # these stiffer axial gains PLUS a weight feedforward = m*g minus the
    # estimated collective prop lift at the stand PWM, so the PD only
    # corrects the residual and the leg truly holds L_des.
    switch_rt_stand_kp_z_n_m: float = 4000.0
    switch_rt_stand_kd_z_n_s_m: float = 60.0
    switch_rt_stand_weight_ff: bool = True
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
    safe_damp_kd: float = 2.0

    # AK60-side damping (MIT Kd gain):
    # - This is **motor-internal viscous damping**: tau += kd * (qd_des - qd_motor).
    # - In ModeE we normally send only tau_ff with kp=kd=0 (pure torque).
    # - Setting this > 0 helps reduce flight-phase jitter/oscillation, because it dissipates energy using
    #   the motor's own high-rate velocity estimate (not the Python/Lcm qd).
    # - Applied per-phase (FLIGHT vs STANCE) so you can add a small amount in stance too.
    ak60_flight_damp_kd: float = 0.15
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
    # 2026-07-24 07:20: 1.0 -> 3.0 so the SLOWEST arm can saturate
    # rm_iq_max_a (iq ≈ kp*lead = 2*3 = 6 -> clip 5 A).  With lead=1
    # the slowest was stuck at ~2 A and could not clear mid-travel load.
    rm_sync_lead_rad: float = 3.0        # 0 = independent PD (sync off)
    rm_hold_s: float = 1.0               # hold at endpoint before idling (s)
    # Station-keeping at the HOPPING endpoint (2026-07-23 user request):
    # while gait==hopping AND legs are PD/PWMPD (and no fold transition is
    # running), keep a continuous current-mode PD on 0 rad so hop impacts
    # cannot back-drive the folding arms. Thermal-safe by construction: at
    # q=0 the PD current is ~0 A (heat ~ i^2, only disturbance transients
    # draw current), and the cap is well under the M2006 continuous rating.
    rm_hopping_hold: bool = True
    rm_hold_iq_max_a: float = 2.0        # |current| cap for the station hold (A)

    # ===== MOBILE: kiwi drive, 3x RM M2006/C610 at 120 deg ================
    # Chassis (top view, body +x forward): the wheels sit evenly spaced on
    # a circle of radius wheel_base_radius_m, each wheel axis pointing at the
    # chassis center, so wheel i drives along the tangent
    #   t_i = (-sin(az_i), +cos(az_i))
    # and side slip rides on the omni rollers. Inverse kinematics from the
    # body twist (vx, vy [m/s], wz [rad/s, CCW+ seen from above]):
    #   v_i     = -sin(az_i)*vx + cos(az_i)*vy + R*wz     (rim speed, m/s)
    #   omega_i = sign_i * v_i / r_wheel                   (rad/s, to LCM)
    # Upper layer still publishes wheel_cmd_lcmt.speed_des_rad_s as OUTPUT
    # shaft / wheel rad/s. Jetson RmWheelController closes a local speed PI
    # and talks C610 current on can1 (IDs 1..3). Folding-arm M2006s remain
    # on Pixhawk and are unrelated to this bus.
    # 2026-08-08 field calibration: wheel 1 sits at the BACK of the drive-
    # forward (camera) direction, not the front -- the old (0,120,240)
    # was mirrored about the Y axis, which flips ONLY the lateral (vy)
    # response (vx and wz unaffected): stick-right drove the base LEFT.
    wheel_azimuth_deg: tuple = (180.0, 60.0, 300.0)
    wheel_base_radius_m: float = 0.20    # center -> wheel contact point
    # Measured 2026-08-08: wheel DIAMETER 10.1 cm -> radius 0.0505 m.
    # NOTE the chassis is open loop above the driver (C610 closes a local
    # rotor-speed PI, but the upper layer never reads wheel odometry), so
    # this radius sets the true m/s per commanded rad/s AND the accuracy
    # of dead-reckoning during tag dropouts.
    wheel_radius_m: float = 0.0505
    # Per-wheel sign to absorb wiring/mounting direction; calibrate with an
    # on-robot spin test (same procedure as the prop mapping).
    wheel_dir_sign: tuple = (1.0, 1.0, 1.0)
    # Stick full scale: RIGHT stick = translation (same axis convention as
    # hopping desired_v_xy), LEFT stick X = yaw rate. Per-wheel speed cap
    # scales the whole twist uniformly so the direction is preserved.
    mobile_v_max_mps: float = 0.8
    mobile_wz_max_rad_s: float = 1.5
    # Output-shaft cap after the M2006 36:1 gearbox. 25 rad/s ~ 239 rpm
    # shaft / ~8600 rpm rotor -- brief peaks ok, sustained driving should
    # stay near continuous rating (drop mobile_v_max_mps if wheels run hot).
    wheel_speed_max_rad_s: float = 25.0
    # ---- MOBILE leg stow (after wheels start moving) ----
    # Latched once any wheel command exceeds the motion threshold. Soft
    # joint PD swings the 3-RSR to mobile_leg_q_des with a 1 Nm torque
    # wall. After arrival, outer position P keeps q_des while AK60 MIT kd
    # supplies the only velocity damping path (avoids duplicate noisy D).
    # q from lcm-spy MOBILE stow pose (2026-08-08):
    mobile_leg_q_des: tuple = (-0.9564, 0.7446, -0.2453)
    mobile_leg_tau_max_nm: float = 1.0
    mobile_leg_kp: float = 8.0            # Nm/rad approach (clipped by tau_max)
    mobile_leg_kd_move: float = 0.0       # no outer D; AK60 handles damping
    mobile_leg_ak60_kd_move: float = 2.0
    mobile_leg_kp_hold: float = 8.0       # outer position hold (tau_ff P)
    # Do NOT apply the noisy CAN qd in tau_ff during hold. Log 142441 showed
    # qd spikes of +/-10 rad/s, making the outer D bang against +/-1 Nm.
    mobile_leg_kd_tau_hold: float = 0.0
    # One damping path only: AK60's high-rate internal velocity feedback.
    # kd=2 was stable in damping-only operation; 4 plus outer D chattered.
    mobile_leg_ak60_kd_hold: float = 2.0
    mobile_leg_arrive_rad: float = 0.08   # max |q_err| to enter hold
    mobile_wheel_motion_rad_s: float = 0.8  # |omega_cmd| latch threshold

    # ===== MANIPULATION: deployed appendages support the body ==============
    # While already deployed, each LT rising edge toggles MOBILE <->
    # MANIPULATION. MOBILE drives the kiwi wheels with the sticks;
    # MANIPULATION stops/holds all wheels. Stick fallback commands the 3-RSR
    # foot on a spherical workspace:
    #   right stick up/right -> +X/+Y tangential foot motion,
    #   left stick up/down    -> retract/extend radial leg length,
    #   z = +sqrt(L^2 - x^2 - y^2).
    # The radial direction is the sphere normal and is used as the axial
    # spring direction by compute_stand_swing_tau().
    manip_xy_rate_mps: float = 0.10
    manip_leg_len_rate_mps: float = 0.08
    manip_leg_len_min_m: float = 0.25
    # Carve the outer travel sphere so joint q stays <= manip_q_max
    # (equal-q at 1.4 -> L≈0.55 m; tilted poses hit q=1.4 earlier, so
    # Lmax/tilt are cut below the mechanical tip — user 2026-08-08).
    # Outer sphere: allow the 0.50-0.55 m READY band plus press stroke.
    manip_leg_len_max_m: float = 0.58
    manip_q_max: float = 1.4
    manip_xy_max_m: float = 0.32
    manip_tilt_max_deg: float = 42.0
    # Manipulation is intentionally low-authority: move the Cartesian target
    # incrementally like stick teleop, never snap the leg with a stiff servo.
    # User 2026-08-08 23:10: 2 Nm still delivered <1 N at the foot
    # (Jacobian + clamp fighting out-of-WS goals). Raise cruise/press caps.
    manip_tau_max_nm: float = 3.0
    # Press/hold: higher torque + open-loop wall-normal force feedforward
    # (tag may be occluded by the hand — setpoint is latched, never live).
    manip_press_tau_max_nm: float = 8.0
    manip_press_ff_n: float = 30.0
    manip_kp_z_n_m: float = 900.0
    manip_kd_z_n_s_m: float = 18.0
    # AprilTag auto uses Cartesian targets/PD so its wall-normal stroke is not
    # projected onto that sphere (tools/button_apriltag_live.py).
    # Geometry: button = tag right 16.5 cm, down 2.6 cm, protrude 5 cm,
    # press 5 cm into the wall (user 2026-08-08 23:07).
    manip_button_auto_enable: bool = True
    manip_button_setpoint_path: str = "logs/button_setpoint.json"
    manip_button_setpoint_udp_port: int = 5558
    # The PC dashboard runs AprilTag detection together with terrain fitting,
    # so valid target packets can be slower than the camera frame rate.
    manip_button_stale_s: float = 1.5
    # Keep a stabilized MOBILE target briefly through detector dropouts. The
    # target is copied permanently when LT enters MANIPULATION.
    manip_button_mobile_latch_s: float = 3.0
    manip_button_tag_id: int = 1
    manip_button_tag_size_m: float = 0.09
    manip_button_acquire_samples: int = 3
    manip_button_acquire_tol_m: float = 0.010
    # Press stroke into the wall (overrides perception packet press_m).
    manip_button_press_m: float = 0.05
    # Stage arrive / hold. Log 23:12: stuck on pre because clamp left a
    # residual > arrive — snap + looser gate (user 2026-08-08 23:13).
    manip_button_arrive_m: float = 0.025
    manip_button_hold_s: float = 0.5
    # Auto target crawl rate (m/s) — faster hand.
    manip_button_rate_mps: float = 0.45
    # Cartesian PD: higher on press/hold so contact force builds.
    manip_button_kp_n_m: float = 320.0
    manip_button_press_kp_n_m: float = 550.0
    manip_button_kd_n_s_m: float = 2.0
    manip_ak60_kd: float = 2.0
    # MOBILE -> MANIPULATION first goes "home": straight down under the
    # body at equal joint angles manip_home_q (FK => ~0.42 m leg length).
    manip_home_q: float = 0.4
    manip_home_arrive_m: float = 0.02
    # After press/hold/retract: return to MOBILE and reverse away from the
    # wall along -n (user 2026-08-08 23:47).
    manip_backup_enable: bool = True
    manip_backup_m: float = 0.30
    manip_backup_v_mps: float = 0.12

    # ===== MOBILE auto-approach (LT drives the wheels toward the tag) ======
    # LT in MOBILE with the tag READY enters MANIPULATION directly (as
    # before). LT with the tag detected but NOT ready (unreachable /
    # acquiring) now starts a wheel servo that drives the base until the
    # button targets fit the leg workspace, then auto-enters MANIPULATION.
    # Any large stick input or B cancels back to manual MOBILE.
    mobile_approach_enable: bool = True
    # Servo goal for the BUTTON hover XY in LEG frame L (leg origin).
    # Arrival / READY is NOT this XY alone — see mobile_ready_r_* below.
    # Yaw FOV keep uses drive/camera bearing; wheel twist is L→drive
    # before kiwi IK.
    mobile_approach_goal_x_m: float = 0.086
    mobile_approach_goal_y_m: float = 0.063
    # Unique stop band (user 2026-08-08): MOBILE hard-stops radial
    # advance at r_max; READY / MANIP when ||foot_face_L|| ∈ [r_min, r_max].
    mobile_ready_r_min_m: float = 0.50
    mobile_ready_r_max_m: float = 0.53
    # Approach speed (2026-08-08 23:13): faster cruise; near-scale still
    # slows the last centimetres.
    mobile_approach_kp_v: float = 0.70         # m/s per m of XY error
    mobile_approach_kp_wz: float = 1.0         # rad/s per rad of yaw error
    mobile_approach_v_max_mps: float = 0.22
    mobile_approach_wz_max_rad_s: float = 0.45
    # Keep driving on the remembered (dead-reckoned) tag pose this long
    # after the detector drops out; then stop and wait for re-capture.
    approach_memory_timeout_s: float = 3.0
    # Tiered slow-down on detection dropouts ("step and see"): full speed
    # only while packets are fresh, half speed when the last sample is
    # older than fresh_s, crawl (translation only, yaw frozen) while
    # coasting on the dead-reckoned memory past manip_button_stale_s.
    mobile_approach_fresh_s: float = 0.4
    mobile_approach_slow_scale: float = 0.5
    mobile_approach_memory_scale: float = 0.3
    # Radial FOV handoff (||pre||_L): FAR centers TAG; NEAR parks TAG
    # only a little left of image center (tag_left_deg), not hard button
    # center (log 23:00: bear_tag=-17 deg was too much).
    mobile_approach_blend_near_m: float = 0.55
    mobile_approach_blend_far_m: float = 0.85
    mobile_approach_tag_left_deg: float = 6.0
    mobile_approach_bear_gate_deg: float = 12.0
    # Far-fast / near-slow on ||pre||_L (translation only).
    mobile_approach_v_far_m: float = 0.90
    mobile_approach_v_near_m: float = 0.55
    mobile_approach_v_near_scale: float = 0.35
    # Approach corridor: forward advance is throttled to zero until the
    # lateral (wall-parallel) error is inside this gate -- translate to
    # the button's normal line first, then drive in square (2026-08-08).
    mobile_approach_lat_gate_m: float = 0.08
    # Tag must stay READY this long before the auto MANIPULATION entry.
    mobile_approach_ready_hold_s: float = 0.3
    # Stick excursion that aborts the approach back to manual driving.
    mobile_approach_stick_override: float = 0.25
    # ---- L-frame -> drive-frame yaw correction (2026-08-08 20:59) -------
    # Perception targets are expressed in the LEG calibration frame L, but
    # T_L_C shows the camera optical axis (= drive forward, the direction
    # right-stick-up moves the base) sits at atan2(R[1][2], R[0][2]) =
    # -32.9 deg yaw in L. Feeding raw L coordinates to the wheel servo
    # made a tag ~33 deg to the camera's RIGHT read as "straight ahead"
    # (robot drove forward, tag left the FOV) and left a permanent 33 deg
    # offset in the wall-normal alignment. The controller loads the yaw
    # from the calib JSON at startup; this is only the fallback.
    drive_calib_rel_path: str = "tools/apriltags_print/calib/T_L_C.json"
    drive_yaw_in_L_deg: float = -32.9

    # ===== PUSH: semi-autonomous box pushing (leg contact + wheels) ========
    # Entered from MOBILE with LT when no button tag is available but the
    # perception node reports a box face (two symmetric AprilTags or the
    # vertical-plane fallback). The leg keeps a low-gain Cartesian contact
    # on the face; the wheels supply the push. LEFT stick commands the BOX:
    # up = push forward along the face normal, left/right = steer the box
    # by offsetting the contact point from the face center (torque about
    # the box's friction center). LT exits back to MOBILE.
    push_enable: bool = True
    push_box_stale_s: float = 1.0              # box packet freshness gate
    push_contact_depth_m: float = 0.02         # press into the face (m)
    push_e_max_m: float = 0.12                 # contact offset cap (m)
    push_e_rate_mps: float = 0.08              # stick -> contact offset rate
    push_v_max_mps: float = 0.25               # max push advance speed
    push_wz_max_rad_s: float = 0.5
    push_kp_wz: float = 1.5                    # face-normal yaw alignment
    push_kp_vy: float = 0.8                    # workspace re-centering gain
    # Stable-pushing curvature cap (Mason/Lynch motion cone): the yaw rate
    # is limited to kappa_max * |v_push| so the contact does not slip.
    push_kappa_max_1_m: float = 1.5
    push_foot_rate_mps: float = 0.05           # contact target crawl rate
    push_tau_max_nm: float = 2.0
    push_kp_n_m: float = 200.0
    push_kd_n_s_m: float = 2.0


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

        # CSV logger: auto-starts with the upper controller; Y also restarts
        # a new file (and resets estimates). B stops the log.
        self._log_enabled = False
        self._log_fp = None

        # Desired velocity command smoothing (rate limiter)
        self._v_cmd_filt = np.zeros(2, dtype=float)
        # LEFT-stick-Y hop-height detent: +1 means an upward push is armed,
        # -1 a downward push, 0 means waiting for the next excursion.
        self._hop_height_stick_pending: int = 0
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
        # startup state. MOBILE and MANIPULATION share the deployed appendage
        # geometry; only their active actuator changes (wheels vs leg).
        self._gait_mode: str = "hopping"
        # MOBILE leg stow: idle -> approach (soft 1 Nm PD) -> damp hold.
        # Latched by wheel motion; cleared when leaving MOBILE.
        self._mobile_leg_active: bool = False
        self._mobile_leg_holding: bool = False
        self._mobile_leg_kd_cmd: float = 0.0
        self._wheel_pending_cmd = np.zeros(3, dtype=float)
        self._wheel_pending_enable: bool = False
        self._manip_init_pending: bool = False
        self._manip_foot_des_b = np.array(
            [0.0, 0.0, float(self.lcm_cfg.switch_rb_leg_len_m)], dtype=float
        )
        self._manip_err_m: float = float("nan")
        # True during button press/hold — use higher tau/kp into the wall.
        self._manip_press_boost: bool = False
        self._manip_press_n_L: np.ndarray | None = None
        self._manip_speed_mps: float = float("nan")
        # AprilTag button auto-press:
        # wait_tag|pre|face|press|hold|retract|done
        self._btn_stage: str = "idle"
        self._btn_hold_t0: float | None = None
        self._btn_last_setpoint: dict | None = None
        self._btn_acquire_samples: list[dict] = []
        self._btn_last_sample_t_wall: float = -1.0
        self._btn_stage_print: str = ""
        self._btn_udp_latest: dict | None = None
        self._btn_udp_rx_t: float = float("-inf")
        self._btn_udp_sock: socket.socket | None = None
        try:
            self._btn_udp_sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM
            )
            self._btn_udp_sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            self._btn_udp_sock.bind((
                "",
                int(self.lcm_cfg.manip_button_setpoint_udp_port),
            ))
            self._btn_udp_sock.setblocking(False)
        except OSError as exc:
            print(f"[MANIP] button UDP unavailable: {exc}; using local JSON")
            if self._btn_udp_sock is not None:
                self._btn_udp_sock.close()
            self._btn_udp_sock = None
        # Pre-acquire the wall-button target while driving in MOBILE. The
        # operator waits for a printed READY, then presses LT to manipulate.
        self._mobile_tag_samples: list[dict] = []
        self._mobile_tag_last_sample_t_wall: float = -1.0
        self._mobile_tag_last_rx_t: float = float("-inf")
        self._mobile_tag_poll_t: float = 0.0
        self._mobile_tag_ready_sp: dict | None = None
        self._mobile_tag_state: str = "searching"
        self._mobile_tag_reach_error_m: float = float("nan")
        # Latest raw (not yet stabilized) tag setpoint, for the approach
        # servo and the status line distance readout.
        self._mobile_tag_last_sp: dict | None = None
        self._mobile_tag_last_sp_rx: float = float("-inf")
        self._mobile_tag_cam_z_m: float = float("nan")
        # MOBILE auto-approach (LT while the tag is detected but not READY).
        self._approach_active: bool = False
        self._approach_sp: dict | None = None      # {"pre","tag","n_in"} DRIVE
        self._approach_dbg_t: float = 0.0
        # Post-press open-loop wheel reverse (drive frame unit dir, meters left).
        self._backup_active: bool = False
        self._backup_dir_D: tuple[float, float] = (-1.0, 0.0)
        self._backup_remain_m: float = 0.0
        self._backup_dbg_t: float = 0.0
        # Yaw of the drive/wheel frame's +X (camera forward) inside the L
        # frame, from the hand-eye calibration (see drive_yaw_in_L_deg).
        dy = math.radians(float(getattr(
            self.lcm_cfg, "drive_yaw_in_L_deg", -32.9)))
        try:
            calib = (Path(__file__).resolve().parent.parent
                     / str(getattr(self.lcm_cfg, "drive_calib_rel_path",
                                   "tools/apriltags_print/calib/T_L_C.json")))
            R = np.asarray(
                json.loads(calib.read_text())["R"], dtype=float
            ).reshape(3, 3)
            dy = math.atan2(float(R[1, 2]), float(R[0, 2]))
            print(f"[init] drive yaw in L from T_L_C: "
                  f"{math.degrees(dy):+.1f} deg")
        except Exception as e:
            print(f"[init] T_L_C unavailable ({e}); drive yaw fallback "
                  f"{math.degrees(dy):+.1f} deg")
        self._l2d_cs: tuple = (math.cos(dy), math.sin(dy))
        self._approach_ready_since: float | None = None
        self._approach_last_twist: tuple = (0.0, 0.0, 0.0)
        self._approach_waiting: bool = False
        # Box face (perception "box" field) + PUSH mode state.
        self._push_box: dict | None = None         # {"center","n_in","right"}
        self._push_box_rx: float = float("-inf")
        self._push_e_des_m: float = 0.0
        self._push_v_last: float = 0.0
        self._push_last_twist: tuple = (0.0, 0.0, 0.0)
        self._push_waiting: bool = False
        # Lightweight state bridge for the camera dashboard (20 Hz JSON).
        self._dashboard_status_last_t: float = 0.0
        self._dashboard_udp_sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM
        )
        self._dashboard_udp_sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_BROADCAST, 1
        )
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
        # LT/RT analog triggers (driver fills leftTriggerAnalog /
        # rightTriggerAnalog in [0,1], threshold at 0.5). Keep the explicit
        # LT names: the previous `lb_now` name was misleading -- it was always
        # leftTriggerAnalog, never the LB bumper.
        lt_now = (float(getattr(gamepad_msg, "leftTriggerAnalog", 0.0)) > 0.5) if gamepad_msg is not None else False
        rt_now = (float(getattr(gamepad_msg, "rightTriggerAnalog", 0.0)) > 0.5) if gamepad_msg is not None else False
        # LB mode intentionally disabled (2026-08-07). The physical bumper is
        # ignored here; LT owns deployed MOBILE <-> MANIPULATION selection.
        # lb_bumper_now = bool(getattr(gamepad_msg, "leftBumper", 0))

        enabled = int(self._mode_est) in (PD, PWMPD)

        # LT selects MOBILE while disabled; from HOPPING it requests the
        # existing stand/fold transition. Once appendages are deployed, every
        # LT rising edge toggles MOBILE <-> MANIPULATION without reconfiguring.
        if bool(lt_now) and (not bool(self._mode_last_lb)):
            if not enabled:
                self._gait_mode = "mobile"
                self._manip_init_pending = False
                self._mobile_tag_samples = []
                self._mobile_tag_ready_sp = None
                self._mobile_tag_state = "searching"
                self._reset_mobile_leg_stow()
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
            elif self._gait_mode == "mobile":
                if bool(self._approach_active):
                    # LT toggles the auto-approach off (manual again).
                    self._approach_stop("LT cancel")
                else:
                    ready_sp = self._mobile_tag_ready_sp
                    tag_recent = (
                        time.monotonic() - float(self._mobile_tag_last_sp_rx)
                        <= 1.0
                    )
                    if ready_sp is not None:
                        # Tag stable AND reachable: manipulate right away.
                        self._enter_manipulation(ready_sp, source="LT")
                    elif (bool(getattr(
                            self.lcm_cfg, "mobile_approach_enable", True))
                            and tag_recent):
                        # Tag seen but out of the leg workspace: drive there.
                        self._approach_start()
                    elif (bool(getattr(self.lcm_cfg, "push_enable", True))
                            and self._read_box_setpoint() is not None):
                        # No button tag but a box face: semi-auto pushing.
                        self._enter_push()
                    else:
                        print(
                            "[gait] LT ignored: no tag "
                            f"({self._mobile_tag_state}) and no box face; "
                            "stay MOBILE"
                        )
            elif self._gait_mode == "push":
                self._gait_mode = "mobile"
                self._push_box = None
                self._push_e_des_m = 0.0
                self._manip_init_pending = False
                self._reset_mobile_leg_stow()
                print("[gait] LT -> MOBILE: leave PUSH, wheels manual")
            elif self._gait_mode == "manipulation":
                self._gait_mode = "mobile"
                self._manip_init_pending = False
                self._mobile_tag_samples = []
                self._mobile_tag_ready_sp = None
                self._mobile_tag_state = "searching"
                self._reset_mobile_leg_stow()
                self._reset_manip_button()
                print(
                    "[gait] LT -> MOBILE: wheels enabled; leg stows to "
                    "mobile q after wheel motion (1 Nm -> damp hold)"
                )
            elif self._gait_mode != "hopping":
                print("[gait] LT ignored (deployed transition active)")
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
        self._mode_last_lb = bool(lt_now)

        if bool(rt_now) and (not bool(self._mode_last_rb)):
            # RT selects HOPPING while disabled. While enabled, only MOBILE may
            # start P4 and the MOBILE->HOPPING transition.
            if not enabled:
                self._rt_stand_t0 = None
                self._gait_mode = "hopping"
                self._reset_mobile_leg_stow()
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
                print(
                    "[gait] RT ignored (leave MANIPULATION with LT first, "
                    "or already HOPPING)"
                )
            elif bool(self._switch_loop):
                print("[switch_loop] RT ignored (P4 stand already active)")
            else:
                self._switch_loop = True
                self._rb_p4_t0 = time.time()
                self._reset_mobile_leg_stow()
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
        self._mode_last_rb = bool(rt_now)

        # Edge-triggered transitions (same priority order as Pi)
        # B = stop BOTH legs (DAMP) and props. Always clears props on the B edge.
        if bool(b_now) and (not bool(self._mode_last_b)):
            self._prop_enable = False
            self._close_prop_base_window("B abort")   # B stops everything
            # B aborts every powered transition. The selected gait is retained
            # so OFF/DAMP can later be armed into the intended configuration.
            self._approach_stop("B")
            if bool(self._backup_active):
                self._backup_active = False
                self._backup_remain_m = 0.0
                print("[backup] OFF (B)")
            if self._gait_mode == "push":
                # PUSH is an active contact task: B drops back to MOBILE so
                # a later X re-arm cannot resume pushing unexpectedly.
                self._gait_mode = "mobile"
                self._push_box = None
                self._push_e_des_m = 0.0
                print("[gait] B: leave PUSH -> MOBILE")
            self._rm_stage = 0
            self._rm_lt_pending = False
            self._rm_rt_pending = False
            self._lt_stand_t0 = None
            self._rt_stand_t0 = None
            self._switch_loop = False
            self._rb_p4_t0 = None
            self._reset_mobile_leg_stow()
            self._restore_rt_first_hop_l0()
            self._rm_iq_des = np.zeros(3, dtype=float)
            self._rm_zero_until = 0.0
            if int(self._mode_est) != DAMP:
                self._mode_est = DAMP
            # Hard-stop props on the B edge instead of waiting for the
            # remainder of this control tick (including core.step).  force=True
            # bypasses the soft-start limiter and also resets its previous-PWM
            # state to the stop point.  The normal end-of-tick publication sends
            # another OFF frame for best-effort UDP redundancy.
            pwm_stop = np.full(
                6, float(self.modee_cfg.pwm_min_us), dtype=float
            )
            self._publish_motor_pwm(
                pwm_stop,
                control_mode=int(self.lcm_cfg.prop_ctrl_mode_off),
                force=True,
            )
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
        # World-frame mapping (2026-08-07): stick UP = +X forward, stick
        # RIGHT = +Y right. Jetson fills rightStickAnalog[0]=rx (right = +)
        # and [1]=ry (Xbox up = -), so forward needs the minus.
        # (2026-08-08: briefly sign-flipped chasing a MOBILE lateral bug;
        # the real culprit was the mirrored wheel_azimuth_deg -- reverted.)
        v[0] = -stick_y * max_v
        v[1] = stick_x * max_v
        return v

    def _handle_hop_height_stick(self, gamepad_msg) -> None:
        """Change hop_height_m once when LEFT-Y returns to neutral.

        Up (negative Xbox LY) arms +1; down arms -1.  No change is applied
        until the operator releases the stick, so a held stick cannot scroll
        repeatedly at the 500 Hz control rate.
        """
        if not bool(getattr(
            self.lcm_cfg, "hop_height_stick_enable", True
        )):
            self._hop_height_stick_pending = 0
            return
        if gamepad_msg is None or self._gait_mode != "hopping":
            self._hop_height_stick_pending = 0
            return
        try:
            stick_y = float(gamepad_msg.leftStickAnalog[1])
        except Exception:
            self._hop_height_stick_pending = 0
            return
        if not np.isfinite(stick_y):
            self._hop_height_stick_pending = 0
            return

        trigger = float(np.clip(
            abs(float(getattr(
                self.lcm_cfg, "hop_height_stick_trigger", 0.70
            ))), 0.05, 1.0
        ))
        release = float(np.clip(
            abs(float(getattr(
                self.lcm_cfg, "hop_height_stick_release", 0.20
            ))), 0.0, trigger - 0.01
        ))
        if stick_y <= -trigger:
            self._hop_height_stick_pending = +1
            return
        if stick_y >= trigger:
            self._hop_height_stick_pending = -1
            return
        if (abs(stick_y) > release
                or int(self._hop_height_stick_pending) == 0):
            return

        direction = int(self._hop_height_stick_pending)
        self._hop_height_stick_pending = 0
        step = float(max(0.0, float(getattr(
            self.lcm_cfg, "hop_height_step_m", 0.02
        ))))
        h_min = float(getattr(
            self.lcm_cfg, "hop_height_min_m", 0.03
        ))
        h_max = float(max(h_min, float(getattr(
            self.lcm_cfg, "hop_height_max_m", 0.13
        ))))
        h_old = float(self.core.cfg.hop_height_m)
        # Round to millimetres to avoid 0.03 + n*0.02 accumulating binary
        # floating-point tails in logs/status output.
        h_new = round(float(np.clip(
            h_old + direction * step, h_min, h_max
        )), 3)
        self.core.cfg.hop_height_m = h_new
        print(
            "[hop_height] LEFT stick %s-release: %.3f -> %.3f m"
            % ("UP" if direction > 0 else "DOWN", h_old, h_new)
        )

    def _reset_mobile_leg_stow(self) -> None:
        """Clear the MOBILE leg approach/hold latch."""
        self._mobile_leg_active = False
        self._mobile_leg_holding = False
        self._mobile_leg_kd_cmd = 0.0

    def _update_mobile_leg_stow(
        self,
        *,
        q: np.ndarray,
        qd: np.ndarray,
        wheel_omega_cmd: np.ndarray,
        wheels_enabled: bool,
    ) -> np.ndarray:
        """Soft-stow legs to mobile_leg_q_des after wheels start moving.

        Approach and hold both use position P in tau_ff and one
        AK60-internal damping path:
            tau_ff = kp*(q_des - q)
            motor damping = -mobile_leg_ak60_kd_hold * qd_motor
        clipped to mobile_leg_tau_max_nm. Hold also sets
        `_mobile_leg_kd_cmd` so the AK60 MIT kd adds viscous damping.
        """
        self._mobile_leg_kd_cmd = 0.0
        if self._gait_mode != "mobile":
            self._reset_mobile_leg_stow()
            return np.zeros(3, dtype=float)

        w = np.asarray(wheel_omega_cmd, dtype=float).reshape(3)
        thr = float(max(0.0, self.lcm_cfg.mobile_wheel_motion_rad_s))
        if (not bool(self._mobile_leg_active)
                and bool(wheels_enabled)
                and thr > 0.0
                and float(np.max(np.abs(w))) >= thr):
            self._mobile_leg_active = True
            self._mobile_leg_holding = False
            print(
                "[mobile-leg] wheel motion -> soft stow to q_des="
                f"{tuple(float(x) for x in self.lcm_cfg.mobile_leg_q_des)} "
                f"(tau_max={float(self.lcm_cfg.mobile_leg_tau_max_nm):.1f} Nm)"
            )

        if not bool(self._mobile_leg_active):
            return np.zeros(3, dtype=float)

        q_now = np.asarray(q, dtype=float).reshape(3)
        qd_now = np.asarray(qd, dtype=float).reshape(3)
        q_des = np.asarray(
            self.lcm_cfg.mobile_leg_q_des, dtype=float
        ).reshape(3)
        err = (q_des - q_now).astype(float)
        arrive = float(max(0.0, self.lcm_cfg.mobile_leg_arrive_rad))
        if (not bool(self._mobile_leg_holding)
                and float(np.max(np.abs(err))) <= arrive):
            self._mobile_leg_holding = True
            print(
                "[mobile-leg] arrived (|e|_max="
                f"{float(np.max(np.abs(err))):.3f} rad) -> P + AK60-D hold "
                f"kp={float(self.lcm_cfg.mobile_leg_kp_hold):.1f} "
                f"tau_kd={float(self.lcm_cfg.mobile_leg_kd_tau_hold):.2f} "
                f"ak60_kd={float(self.lcm_cfg.mobile_leg_ak60_kd_hold):.2f}"
            )

        if bool(self._mobile_leg_holding):
            kp = float(max(0.0, self.lcm_cfg.mobile_leg_kp_hold))
            kd = float(max(0.0, self.lcm_cfg.mobile_leg_kd_tau_hold))
            self._mobile_leg_kd_cmd = float(max(
                0.0, self.lcm_cfg.mobile_leg_ak60_kd_hold
            ))
        else:
            kp = float(max(0.0, self.lcm_cfg.mobile_leg_kp))
            kd = float(max(0.0, self.lcm_cfg.mobile_leg_kd_move))
            self._mobile_leg_kd_cmd = float(max(
                0.0, self.lcm_cfg.mobile_leg_ak60_kd_move
            ))

        cap = float(max(0.0, self.lcm_cfg.mobile_leg_tau_max_nm))
        tau = kp * err - kd * qd_now
        if cap > 0.0:
            tau = np.clip(tau, -cap, cap)
        return tau.astype(float)

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
            # Same axis convention as hopping desired_v_xy: stick UP =
            # +vx forward; stick RIGHT = +vy right. Yaw: _kiwi_ik wz>0 is
            # CW from above (FRD +Z down), so stick LEFT (ls_x<0) must
            # command wz<0 = CCW = turn left (sign fixed 2026-08-08).
            vx = -rs_y * float(self.lcm_cfg.mobile_v_max_mps)
            vy = rs_x * float(self.lcm_cfg.mobile_v_max_mps)
            wz = ls_x * float(self.lcm_cfg.mobile_wz_max_rad_s)
        return self._kiwi_ik(vx, vy, wz)

    def _kiwi_ik(self, vx: float, vy: float, wz: float) -> np.ndarray:
        """Body twist -> wheel rad/s. FRD: vx fwd, vy RIGHT, and wz>0 =
        right-handed about body +Z DOWN = CW seen from above."""
        az = np.deg2rad(np.asarray(
            self.lcm_cfg.wheel_azimuth_deg, dtype=float
        ).reshape(3))
        r_base = float(self.lcm_cfg.wheel_base_radius_m)
        r_wheel = float(max(1e-4, float(self.lcm_cfg.wheel_radius_m)))
        sgn = np.asarray(self.lcm_cfg.wheel_dir_sign, dtype=float).reshape(3)
        # v_i = -sin(az_i)*vx + cos(az_i)*vy + R*wz (rim speed, m/s)
        v_rim = -np.sin(az) * float(vx) + np.cos(az) * float(vy) \
            + r_base * float(wz)
        omega = sgn * v_rim / r_wheel
        w_max = float(max(1e-3, float(self.lcm_cfg.wheel_speed_max_rad_s)))
        pk = float(np.max(np.abs(omega)))
        if pk > w_max:
            omega = omega * (w_max / pk)
        return omega.astype(float)

    @staticmethod
    def _dead_reckon_xy(
        points: list[np.ndarray],
        vectors: list[np.ndarray],
        vx: float,
        vy: float,
        wz: float,
        dt: float,
    ) -> None:
        """Advance L-frame targets by the commanded body twist (in place).

        FRD horizontal plane (x fwd, y right), wz>0 = CW from above
        (right-handed about +Z down). When the body rotates by
        dtheta = wz*dt, a world-fixed point in body coordinates rotates
        the opposite way: p <- Rz(-dtheta) (p - v dt). Directions rotate
        only. Used to keep servoing on the remembered tag/box pose
        through detector dropouts.
        """
        for p in points:
            p[0], p[1] = dead_reckon_step(
                float(p[0]), float(p[1]), vx, vy, wz, dt, is_point=True
            )
        for v in vectors:
            v[0], v[1] = dead_reckon_step(
                float(v[0]), float(v[1]), vx, vy, wz, dt, is_point=False
            )

    def _l2d_xy(self, x: float, y: float) -> tuple[float, float]:
        """L-frame horizontal coords -> drive/wheel frame (yaw only)."""
        c, s = self._l2d_cs
        return c * float(x) + s * float(y), -s * float(x) + c * float(y)

    def _d2l_xy(self, x: float, y: float) -> tuple[float, float]:
        """Drive/wheel frame horizontal coords -> L frame (yaw only)."""
        c, s = self._l2d_cs
        return c * float(x) - s * float(y), s * float(x) + c * float(y)

    # ===== MOBILE auto-approach (LT -> wheel servo toward the tag) =========

    def _approach_start(self) -> None:
        self._approach_active = True
        self._approach_sp = None
        self._approach_ready_since = None
        self._approach_waiting = False
        self._approach_last_twist = (0.0, 0.0, 0.0)
        print(
            "[approach] LT -> AUTO-APPROACH: driving wheels toward tag "
            f"({self._mobile_tag_state}); stick/B/LT cancels"
        )

    def _approach_stop(self, reason: str | None) -> None:
        if not bool(self._approach_active):
            return
        self._approach_active = False
        self._approach_sp = None
        self._approach_ready_since = None
        self._approach_waiting = False
        if reason:
            print(f"[approach] OFF ({reason})")

    def _manip_home_foot(self) -> np.ndarray:
        """Straight-down home: equal joints at manip_home_q (≈ q=0.4)."""
        qh = float(getattr(self.lcm_cfg, "manip_home_q", 0.4))
        try:
            if self.core.fk is not None:
                foot, _ = self.core.fk.forward_kinematics(
                    np.full(3, qh, dtype=float)
                )
                foot = np.asarray(foot, dtype=float).reshape(3)
                if np.all(np.isfinite(foot)):
                    return foot
        except Exception:
            pass
        # FK fallback: measured L≈0.422 m at q=0.4.
        return np.array([0.0, 0.0, 0.422], dtype=float)

    def _rebuild_button_press(
        self, sp: dict, *, press_m: float | None = None
    ) -> dict:
        """Force press stroke to manip_button_press_m (default 5 cm)."""
        out = dict(sp)
        n_in = np.asarray(
            out.get("wall_normal_in_L", [1.0, 0.0, 0.0]), dtype=float
        ).reshape(3)
        nn = float(np.linalg.norm(n_in))
        if nn > 1e-9:
            n_in = n_in / nn
        face = np.asarray(out["foot_face_L"], dtype=float).reshape(3)
        pm = float(
            press_m
            if press_m is not None
            else getattr(self.lcm_cfg, "manip_button_press_m", 0.05)
        )
        out["press_m"] = pm
        out["foot_face_L"] = face.tolist()
        out["foot_pre_L"] = (face - 0.03 * n_in).tolist()
        out["foot_press_L"] = (face + pm * n_in).tolist()
        out["wall_normal_in_L"] = n_in.tolist()
        return out

    def _enter_manipulation(self, ready_sp: dict, *, source: str) -> None:
        """Shared MOBILE -> MANIPULATION entry (LT press or auto-arrival)."""
        self._approach_stop(None)
        self._backup_active = False
        self._backup_remain_m = 0.0
        self._gait_mode = "manipulation"
        self._manip_init_pending = True
        self._reset_mobile_leg_stow()
        self._reset_manip_button()
        # Latch the button target, but first recenter the leg straight
        # down (home @ q≈0.4) before pre/face/press.
        self._btn_last_setpoint = self._rebuild_button_press(ready_sp)
        self._btn_stage = "home"
        self._btn_stage_print = "home"
        qh = float(getattr(self.lcm_cfg, "manip_home_q", 0.4))
        pm = float(getattr(self.lcm_cfg, "manip_button_press_m", 0.05))
        self._manip_press_n_L = np.asarray(
            self._btn_last_setpoint.get("wall_normal_in_L", [1.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        nn = float(np.linalg.norm(self._manip_press_n_L))
        if nn > 1e-9:
            self._manip_press_n_L /= nn
        print(
            f"[gait] {source} -> MANIPULATION: wheels STOP; "
            f"TAG setpoint LATCHED (open-loop — occlusion OK); "
            f"HOME -> PRE -> FACE -> PRESS({pm*100:.0f}cm). "
            f"Keep mode PD (do not press B)."
        )

    def _update_mobile_approach(self, gamepad_msg, dt: float) -> np.ndarray | None:
        """Wheel servo toward the button tag. Returns wheel rad/s or None.

        Closed loop on every fresh setpoint; through detector dropouts the
        last pose is dead-reckoned with the commanded twist for up to
        approach_memory_timeout_s, then the base stops and waits. When the
        pre-acquisition monitor reports READY for ready_hold_s the
        controller enters MANIPULATION without a second LT press.
        """
        if not bool(self._approach_active):
            return None
        cfg = self.lcm_cfg
        # Operator override: any meaningful stick input cancels.
        if gamepad_msg is not None:
            try:
                sticks = (
                    float(gamepad_msg.rightStickAnalog[0]),
                    float(gamepad_msg.rightStickAnalog[1]),
                    float(gamepad_msg.leftStickAnalog[0]),
                    float(gamepad_msg.leftStickAnalog[1]),
                )
            except Exception:
                sticks = ()
            thr = float(getattr(cfg, "mobile_approach_stick_override", 0.25))
            if any(abs(s) > thr for s in sticks):
                self._approach_stop("stick override")
                return None

        now_m = time.monotonic()
        age = now_m - float(self._mobile_tag_last_sp_rx)
        stale_s = float(getattr(cfg, "manip_button_stale_s", 1.5))
        memory_s = float(getattr(cfg, "approach_memory_timeout_s", 3.0))
        if age <= stale_s and self._mobile_tag_last_sp is not None:
            sp = self._mobile_tag_last_sp
            n_in = np.asarray(
                sp["wall_normal_in_L"], dtype=float
            ).reshape(3).copy()
            nn = float(np.linalg.norm(n_in))
            if nn > 1e-9:
                n_in /= nn
            pre = np.asarray(
                sp["foot_pre_L"], dtype=float
            ).reshape(3).copy()
            tag_L = sp.get("tag_center_L")
            if tag_L is None:
                # Fallback: tag ≈ face − button_right along wall (legacy
                # packets without tag_center_L). Prefer face if present.
                face = np.asarray(
                    sp.get("foot_face_L", sp["foot_pre_L"]), dtype=float
                ).reshape(3).copy()
                tag = face.copy()
            else:
                tag = np.asarray(tag_L, dtype=float).reshape(3).copy()
            # Keep targets in LEG frame L — translation judgment is from
            # the leg origin. Drive/camera views are derived per tick for
            # FOV yaw only; wheel twist is rotated L→D after the law.
            self._approach_sp = {"pre": pre, "tag": tag, "n_in": n_in}
            if bool(self._approach_waiting):
                self._approach_waiting = False
                print("[approach] tag re-captured -> resume")
        elif self._approach_sp is not None and age <= memory_s:
            # Detector dropout: advance L-frame memory by the L-frame twist.
            vx_l, vy_l, wz_l = self._approach_last_twist
            pts = [self._approach_sp["pre"]]
            if self._approach_sp.get("tag") is not None:
                pts.append(self._approach_sp["tag"])
            self._dead_reckon_xy(
                pts,
                [self._approach_sp["n_in"]],
                vx_l, vy_l, wz_l, dt,
            )
        else:
            # Memory too old: hold position and wait for re-detection.
            if not bool(self._approach_waiting):
                self._approach_waiting = True
                print(
                    "[approach] tag lost > "
                    f"{memory_s:.1f}s -> wheels stop, waiting for tag"
                )
            self._approach_last_twist = (0.0, 0.0, 0.0)
            return np.zeros(3, dtype=float)

        pre = self._approach_sp["pre"]
        n_in = self._approach_sp["n_in"]
        gx = float(getattr(cfg, "mobile_approach_goal_x_m", 0.086))
        gy = float(getattr(cfg, "mobile_approach_goal_y_m", 0.063))
        arrive_err = math.hypot(float(pre[0]) - gx, float(pre[1]) - gy)
        face_r = float(self._mobile_tag_cam_z_m) if np.isfinite(
            self._mobile_tag_cam_z_m
        ) else float(np.linalg.norm(pre))

        # UNIQUE arrival: READY means ||face||_L ∈ [r_min, r_max] and
        # workspace OK — stop wheels, enter MANIP after ready_hold_s.
        r_min = float(getattr(cfg, "mobile_ready_r_min_m", 0.50))
        r_max = float(getattr(cfg, "mobile_ready_r_max_m", 0.53))
        ready_ok = (
            self._mobile_tag_state == "ready"
            and self._mobile_tag_ready_sp is not None
            and age <= stale_s
            and r_min <= face_r <= r_max
        )
        if ready_ok:
            if self._approach_ready_since is None:
                self._approach_ready_since = now_m
                print(
                    "[approach] READY -> holding wheels, "
                    f"enter MANIP in "
                    f"{float(getattr(cfg, 'mobile_approach_ready_hold_s', 0.3)):.1f}s"
                )
            hold = float(getattr(cfg, "mobile_approach_ready_hold_s", 0.3))
            if (now_m - self._approach_ready_since) >= hold:
                self._enter_manipulation(
                    self._mobile_tag_ready_sp,
                    source=(
                        "approach arrived (READY, "
                        f"pre_L err={arrive_err*100:.1f}cm, "
                        f"|face|_L={face_r*100:.1f}cm)"
                    ),
                )
                return np.zeros(3, dtype=float)
            # Freeze while confirming READY — no more forward creep.
            self._approach_last_twist = (0.0, 0.0, 0.0)
            return np.zeros(3, dtype=float)
        self._approach_ready_since = None

        # Closed-loop PBVS in LEG frame L; FOV yaw uses drive/camera view.
        tag = self._approach_sp.get("tag")
        if tag is None:
            tag = pre
        tag_dx, tag_dy = self._l2d_xy(float(tag[0]), float(tag[1]))
        pre_dx, pre_dy = self._l2d_xy(float(pre[0]), float(pre[1]))
        n_dx, n_dy = self._l2d_xy(float(n_in[0]), float(n_in[1]))
        vx_L, vy_L, wz = approach_twist(
            float(pre[0]), float(pre[1]),
            float(n_in[0]), float(n_in[1]),
            age, cfg,
            tag_x=float(tag[0]), tag_y=float(tag[1]),
            view_x=tag_dx, view_y=tag_dy,
            n_view_x=n_dx, n_view_y=n_dy,
            pre_view_x=pre_dx, pre_view_y=pre_dy,
        )
        # Radial band gate on ||face||_L (log 22:51: crept 0.50→0.43
        # because the <=r_max branch ran before retreat). Order:
        #   1) too close (<r_min): retreat along -n
        #   2) at/inside r_max: kill wall-normal advance; keep lateral only
        nx, ny = float(n_in[0]), float(n_in[1])
        if nx * float(pre[0]) + ny * float(pre[1]) < 0.0:
            nx, ny = -nx, -ny
        nh = math.hypot(nx, ny) or 1.0
        nhx, nhy = nx / nh, ny / nh
        adv = vx_L * nhx + vy_L * nhy
        if face_r < r_min:
            # Back out until back in band (was stuck at ~0 with old elif).
            retreat = float(np.clip(0.8 * (r_min - face_r), 0.06, 0.15))
            lat_x, lat_y = vx_L - adv * nhx, vy_L - adv * nhy
            vx_L = lat_x - retreat * nhx
            vy_L = lat_y - retreat * nhy
        elif face_r <= r_max:
            # Hard-stop radial creep; lateral/yaw may still center the tag.
            if adv > 0.0:
                vx_L -= adv * nhx
                vy_L -= adv * nhy
        self._approach_last_twist = (vx_L, vy_L, wz)
        # Wheel IK is in the drive/camera frame.
        vx_D, vy_D = self._l2d_xy(vx_L, vy_L)
        if (now_m - self._approach_dbg_t) >= 1.0:
            self._approach_dbg_t = now_m
            bear_tag = math.degrees(math.atan2(tag_dy, tag_dx))
            bear_btn = math.degrees(math.atan2(pre_dy, pre_dx))
            print(
                f"[approach] pre_L=({pre[0]:+.2f},{pre[1]:+.2f}) "
                f"tag_L=({tag[0]:+.2f},{tag[1]:+.2f}) "
                f"|face|_L={face_r:.2f}m "
                f"bear_tag={bear_tag:+.1f}deg bear_btn={bear_btn:+.1f}deg "
                f"err={arrive_err*100:.1f}cm clip="
                f"{self._mobile_tag_reach_error_m*100:.1f}cm "
                f"v_L=({vx_L:+.2f},{vy_L:+.2f}) "
                f"v_D=({vx_D:+.2f},{vy_D:+.2f}) wz={wz:+.2f}"
            )
        return self._kiwi_ik(vx_D, vy_D, wz)

    def _reset_manip_button(self) -> None:
        self._btn_stage = (
            "wait_tag"
            if bool(getattr(self.lcm_cfg, "manip_button_auto_enable", False))
            else "idle"
        )
        self._btn_hold_t0 = None
        self._btn_last_setpoint = None
        self._btn_acquire_samples = []
        self._btn_last_sample_t_wall = -1.0
        self._btn_stage_print = self._btn_stage

    def _write_dashboard_status(
        self,
        *,
        now: float,
        q: np.ndarray,
        qd: np.ndarray,
        tau_cmd: np.ndarray,
        info: dict,
    ) -> None:
        """Publish gait/leg command state for terrain_gate_live dashboard."""
        if (float(now) - float(self._dashboard_status_last_t)) < 0.05:
            return
        self._dashboard_status_last_t = float(now)
        path = Path(_CUR_DIR).resolve().parent / "logs" / "dashboard_status.json"
        foot_actual = np.asarray(
            info.get("foot_vicon", [np.nan, np.nan, np.nan]), dtype=float
        ).reshape(3)
        payload = {
            "t_wall": time.time(),
            "gait_mode": str(self._gait_mode),
            "button_stage": str(self._btn_stage),
            "mobile_tag_state": str(self._mobile_tag_state),
            # Goal: |face|_L in [r_min, r_max]. clip = workspace 差值.
            "mobile_ready_r_min_m": float(
                getattr(self.lcm_cfg, "mobile_ready_r_min_m", 0.50)
            ),
            "mobile_ready_r_max_m": float(
                getattr(self.lcm_cfg, "mobile_ready_r_max_m", 0.53)
            ),
            "mobile_tag_reach_error_m": (
                float(self._mobile_tag_reach_error_m)
                if np.isfinite(self._mobile_tag_reach_error_m) else None
            ),
            "mobile_tag_r_err_m": (
                (
                    max(0.0, float(self._mobile_tag_cam_z_m) - float(
                        getattr(self.lcm_cfg, "mobile_ready_r_max_m", 0.53)
                    ))
                    if float(self._mobile_tag_cam_z_m) > float(
                        getattr(self.lcm_cfg, "mobile_ready_r_max_m", 0.53)
                    )
                    else min(0.0, float(self._mobile_tag_cam_z_m) - float(
                        getattr(self.lcm_cfg, "mobile_ready_r_min_m", 0.50)
                    ))
                )
                if np.isfinite(self._mobile_tag_cam_z_m) else None
            ),
            "foot_cmd_L": [
                float(x) for x in np.asarray(
                    self._manip_foot_des_b, dtype=float
                ).reshape(3)
            ],
            "foot_actual_L": [
                float(x) if np.isfinite(x) else None for x in foot_actual
            ],
            "q": [
                float(x) for x in np.asarray(q, dtype=float).reshape(3)
            ],
            "qd": [
                float(x) for x in np.asarray(qd, dtype=float).reshape(3)
            ],
            "tau_cmd": [
                float(x) for x in np.asarray(
                    tau_cmd, dtype=float
                ).reshape(3)
            ],
            "manip_err_m": (
                float(self._manip_err_m)
                if np.isfinite(self._manip_err_m) else None
            ),
            # MOBILE auto-approach + box/push telemetry (PC monitor views).
            "approach_active": bool(self._approach_active),
            "approach_waiting": bool(self._approach_waiting),
            "mobile_tag_cam_z_m": (
                float(self._mobile_tag_cam_z_m)
                if np.isfinite(self._mobile_tag_cam_z_m) else None
            ),
            "push_e_m": float(self._push_e_des_m),
            "push_v_mps": float(self._push_v_last),
            "box": (
                {
                    "source": str(self._push_box.get("source", "?")),
                    "center_L": [
                        float(x) for x in np.asarray(
                            self._push_box["center"], dtype=float
                        ).reshape(3)
                    ],
                    "normal_in_L": [
                        float(x) for x in np.asarray(
                            self._push_box["n_in"], dtype=float
                        ).reshape(3)
                    ],
                    "width_m": (
                        float(self._push_box.get("width_m", float("nan")))
                        if np.isfinite(float(self._push_box.get(
                            "width_m", float("nan")
                        ))) else None
                    ),
                    "age_s": round(
                        time.monotonic() - float(self._push_box_rx), 2
                    ),
                }
                if self._push_box is not None else None
            ),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            encoded = json.dumps(payload)
            tmp.write_text(encoded)
            tmp.replace(path)
            self._dashboard_udp_sock.sendto(
                encoded.encode("utf-8"),
                (
                    str(self.lcm_cfg.dashboard_udp_host),
                    int(self.lcm_cfg.dashboard_udp_port),
                ),
            )
        except Exception:
            pass

    def _read_button_setpoint(self) -> dict | None:
        """Load fresh AprilTag setpoint from PC UDP or local JSON fallback."""
        if not bool(getattr(self.lcm_cfg, "manip_button_auto_enable", False)):
            return None
        if self._btn_udp_sock is not None:
            while True:
                try:
                    raw, _peer = self._btn_udp_sock.recvfrom(65535)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    self._btn_udp_latest = json.loads(raw.decode("utf-8"))
                    self._btn_udp_rx_t = time.monotonic()
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
        from_udp = (
            self._btn_udp_latest is not None
            and time.monotonic() - self._btn_udp_rx_t
            <= float(getattr(self.lcm_cfg, "manip_button_stale_s", 0.5))
        )
        if from_udp:
            data = self._btn_udp_latest
        else:
            path = Path(getattr(
                self.lcm_cfg, "manip_button_setpoint_path",
                "logs/button_setpoint.json",
            ))
            if not path.is_absolute():
                path = Path(_CUR_DIR).resolve().parent / path
            try:
                data = json.loads(path.read_text())
            except Exception:
                return None
        if not bool(data.get("valid", False)):
            return None
        age = (
            time.monotonic() - self._btn_udp_rx_t
            if from_udp
            else time.time() - float(data.get("t_wall", 0.0))
        )
        if age < -0.1 or age > float(
            getattr(self.lcm_cfg, "manip_button_stale_s", 0.5)
        ):
            return None
        if int(data.get("tag_id", -1)) != int(
            getattr(self.lcm_cfg, "manip_button_tag_id", 1)
        ):
            return None
        if abs(
            float(data.get("tag_size_m", -1.0))
            - float(getattr(self.lcm_cfg, "manip_button_tag_size_m", 0.09))
        ) > 1e-4:
            return None
        for key in (
            "foot_pre_L",
            "foot_face_L",
            "foot_press_L",
            "wall_normal_in_L",
        ):
            if key not in data:
                return None
            try:
                value = np.asarray(data[key], dtype=float).reshape(3)
            except Exception:
                return None
            if not np.all(np.isfinite(value)):
                return None
        return data

    def _read_box_setpoint(self) -> dict | None:
        """Fresh box-face estimate from the perception node ("box" field).

        The Jetson perception node sends one JSON packet per frame on the
        button-setpoint UDP port; the box estimate rides in the same
        packet so a missing button tag ("valid": false) still delivers
        the box. Returns the raw box dict or None.
        """
        # Drains the UDP socket into _btn_udp_latest as a side effect.
        self._read_button_setpoint()
        data = self._btn_udp_latest
        if data is None:
            return None
        age = time.monotonic() - float(self._btn_udp_rx_t)
        if age > float(getattr(self.lcm_cfg, "push_box_stale_s", 1.0)):
            return None
        box = data.get("box")
        if not isinstance(box, dict) or not bool(box.get("valid", False)):
            return None
        for key in ("center_L", "normal_in_L", "face_right_L"):
            if key not in box:
                return None
            try:
                value = np.asarray(box[key], dtype=float).reshape(3)
            except Exception:
                return None
            if not np.all(np.isfinite(value)):
                return None
        return box

    def _clamp_manip_foot_b(self, target: np.ndarray) -> np.ndarray:
        """Project a body/FK foot target onto the spherical manip workspace.

        Stick teleop: keep ||p||=L with z reconstructed on the +Z hemisphere.
        """
        target = np.asarray(target, dtype=float).reshape(3).copy()
        L = float(np.linalg.norm(target))
        L = float(np.clip(
            L,
            float(self.lcm_cfg.manip_leg_len_min_m),
            float(self.lcm_cfg.manip_leg_len_max_m),
        ))
        xy = target[:2].copy()
        xy_norm = float(np.linalg.norm(xy))
        tilt_lim = math.radians(float(np.clip(
            self.lcm_cfg.manip_tilt_max_deg, 0.0, 89.0
        )))
        xy_lim = float(min(
            max(0.0, self.lcm_cfg.manip_xy_max_m),
            L * math.sin(tilt_lim),
            max(0.0, L - 1e-4),
        ))
        if xy_norm > xy_lim and xy_norm > 1e-9:
            xy *= xy_lim / xy_norm
        target[0:2] = xy
        target[2] = math.sqrt(max(
            1e-8, L * L - float(np.dot(xy, xy))
        ))
        return target

    def _clamp_manip_foot_cartesian(self, target: np.ndarray) -> np.ndarray:
        """Soft workspace clamp that preserves Cartesian direction.

        Used by AprilTag wall-button press so the wall-normal stroke is
        not destroyed by rebuilding z = sqrt(L^2 - x^2 - y^2). Also carves
        the outer sphere so joint q stays <= manip_q_max (1.4).
        """
        p = np.asarray(target, dtype=float).reshape(3).copy()
        if not np.all(np.isfinite(p)):
            return np.array([0.0, 0.0, float(self.lcm_cfg.manip_leg_len_min_m)])
        # Foot must stay in the +Z hemisphere of the FK frame.
        if float(p[2]) < 0.05:
            p[2] = 0.05
        Lmin = float(self.lcm_cfg.manip_leg_len_min_m)
        Lmax = float(self.lcm_cfg.manip_leg_len_max_m)
        L = float(np.linalg.norm(p))
        if L < 1e-9:
            return np.array([0.0, 0.0, Lmin], dtype=float)
        if L < Lmin:
            p *= Lmin / L
            L = Lmin
        elif L > Lmax:
            p *= Lmax / L
            L = Lmax
        tilt_lim = math.radians(float(np.clip(
            self.lcm_cfg.manip_tilt_max_deg, 0.0, 89.0
        )))
        xy = p[:2].copy()
        xy_norm = float(np.linalg.norm(xy))
        xy_lim = float(min(
            max(0.0, self.lcm_cfg.manip_xy_max_m),
            L * math.sin(tilt_lim),
            max(0.0, L - 1e-4),
        ))
        if xy_norm > xy_lim and xy_norm > 1e-9:
            # Shrink only xy; keep z so wall-normal delta is not remapped
            # onto the sphere.
            scale = xy_lim / xy_norm
            p[0] *= scale
            p[1] *= scale
            L2 = float(np.linalg.norm(p))
            if L2 < Lmin:
                p *= Lmin / max(L2, 1e-9)
            elif L2 > Lmax:
                p *= Lmax / L2
        # Extra carve: if the point is still past the q=1.4 ring, pull it
        # toward the +Z home axis (keeps heading, shortens radius).
        q_max = float(getattr(self.lcm_cfg, "manip_q_max", 1.4))
        # Equal-q length at q_max ≈ 0.55 m; allow only a fraction once
        # tilted (empirical: L * (1 + 0.55*sin(tilt)) ≲ L(q_max)).
        L_tip = 0.55
        tilt = math.atan2(float(np.linalg.norm(p[:2])), max(1e-6, float(p[2])))
        L_qcap = L_tip / max(1.0, 1.0 + 0.7 * math.sin(tilt))
        L_qcap = min(L_qcap, Lmax)
        Ln = float(np.linalg.norm(p))
        if Ln > L_qcap and Ln > 1e-9:
            p = p * (L_qcap / Ln)
        _ = q_max  # documented limit; geometric carve enforces it
        return p.astype(float)

    def _monitor_mobile_button_target(self) -> None:
        """Pre-acquire a stable/reachable button target while in MOBILE."""
        if self._gait_mode != "mobile":
            return
        now = time.time()
        if (now - float(self._mobile_tag_poll_t)) < 0.05:
            return
        self._mobile_tag_poll_t = now
        sp = self._read_button_setpoint()
        if sp is None:
            since_valid = time.monotonic() - self._mobile_tag_last_rx_t
            latch_s = float(max(
                0.0,
                getattr(self.lcm_cfg, "manip_button_mobile_latch_s", 3.0),
            ))
            # Do not throw away a useful acquisition because the shared
            # dashboard skipped one or two detections. A READY pose remains
            # available long enough for the operator to press LT.
            if since_valid <= latch_s and (
                self._mobile_tag_ready_sp is not None
                or len(self._mobile_tag_samples) > 0
            ):
                return
            if self._mobile_tag_state != "searching":
                print("[mobile-tag] LOST -> searching for tag id=1")
            self._mobile_tag_samples = []
            self._mobile_tag_ready_sp = None
            self._mobile_tag_state = "searching"
            self._mobile_tag_reach_error_m = float("nan")
            return

        sample_t = float(sp.get("t_wall", 0.0))
        if sample_t <= float(self._mobile_tag_last_sample_t_wall) + 1e-6:
            return
        self._mobile_tag_last_sample_t_wall = sample_t
        self._mobile_tag_last_rx_t = time.monotonic()
        # Raw sample for the approach servo + status-line distance.
        self._mobile_tag_last_sp = sp
        self._mobile_tag_last_sp_rx = time.monotonic()
        face_now = np.asarray(sp["foot_face_L"], dtype=float).reshape(3)
        # Dashboard "distance" = Euclidean range from LEG origin to BTN
        # face in L (not camera depth).
        self._mobile_tag_cam_z_m = float(np.linalg.norm(face_now))
        count = int(max(
            1, getattr(self.lcm_cfg, "manip_button_acquire_samples", 8)
        ))
        self._mobile_tag_samples.append(sp)
        self._mobile_tag_samples = self._mobile_tag_samples[-count:]
        if len(self._mobile_tag_samples) < count:
            if self._mobile_tag_state != "acquiring":
                print(f"[mobile-tag] DETECTED id=1 -> stabilizing {count} frames")
            self._mobile_tag_state = "acquiring"
            return

        faces = np.asarray(
            [s["foot_face_L"] for s in self._mobile_tag_samples],
            dtype=float,
        ).reshape(count, 3)
        face = np.median(faces, axis=0)
        spread = float(np.max(np.linalg.norm(
            faces - face.reshape(1, 3), axis=1
        )))
        tol = float(max(
            0.001,
            getattr(self.lcm_cfg, "manip_button_acquire_tol_m", 0.010),
        ))
        if spread > tol:
            self._mobile_tag_ready_sp = None
            self._mobile_tag_state = "unstable"
            return

        normals = np.asarray(
            [s["wall_normal_in_L"] for s in self._mobile_tag_samples],
            dtype=float,
        ).reshape(count, 3)
        n_in = np.median(normals, axis=0)
        n_norm = float(np.linalg.norm(n_in))
        if n_norm <= 1e-6:
            self._mobile_tag_ready_sp = None
            self._mobile_tag_state = "unstable"
            return
        n_in /= n_norm
        press_m = float(getattr(self.lcm_cfg, "manip_button_press_m", 0.05))
        latched = dict(sp)
        latched["foot_face_L"] = face.tolist()
        latched["foot_pre_L"] = (face - 0.03 * n_in).tolist()
        latched["foot_press_L"] = (face + press_m * n_in).tolist()
        latched["wall_normal_in_L"] = n_in.tolist()
        latched["press_m"] = press_m

        # READY = stable + ||face||_L in the stop band. Workspace clip is
        # reported but does NOT block READY inside the band (log 23:00:
        # |face|=0.50 m still UNREACHABLE on clip=17 cm while the user
        # stop band is exactly 0.50-0.53 — manip crawl handles residual).
        reach_error = 0.0
        for key in ("foot_pre_L", "foot_face_L", "foot_press_L"):
            raw = np.asarray(latched[key], dtype=float).reshape(3)
            clipped = self._clamp_manip_foot_cartesian(raw)
            reach_error = max(
                reach_error, float(np.linalg.norm(clipped - raw))
            )
        self._mobile_tag_reach_error_m = reach_error
        r_face = float(np.linalg.norm(face))
        r_min = float(getattr(self.lcm_cfg, "mobile_ready_r_min_m", 0.50))
        r_max = float(getattr(self.lcm_cfg, "mobile_ready_r_max_m", 0.53))
        if r_face > r_max or r_face < r_min:
            if self._mobile_tag_state != "unreachable":
                why = []
                if r_face > r_max:
                    why.append(f"|face|={r_face:.2f}>r_max={r_max:.2f}")
                if r_face < r_min:
                    why.append(f"|face|={r_face:.2f}<r_min={r_min:.2f}")
                if reach_error > 0.02:
                    why.append(f"clip={reach_error*100:.1f}cm")
                print(
                    "[mobile-tag] DETECTED/STABLE id=1 but UNREACHABLE: "
                    + ", ".join(why)
                    + f"; drive MOBILE into {r_min:.2f}-{r_max:.2f} m band"
                )
            self._mobile_tag_ready_sp = None
            self._mobile_tag_state = "unreachable"
            return

        self._mobile_tag_ready_sp = latched
        if self._mobile_tag_state != "ready":
            print(
                "[mobile-tag] READY id=1: "
                f"|face|_L={r_face:.2f} m in [{r_min:.2f},{r_max:.2f}], "
                f"clip={reach_error*100:.1f}cm (info), "
                f"spread={spread*1000:.1f}mm "
                "-> LT / auto-approach enters MANIPULATION"
            )
        self._mobile_tag_state = "ready"

    # ===== PUSH: semi-autonomous box pushing ================================

    def _store_push_box(self, box: dict) -> None:
        """Normalize a perception box dict into servo-ready L-frame arrays."""
        n_in = np.asarray(box["normal_in_L"], dtype=float).reshape(3).copy()
        nn = float(np.linalg.norm(n_in))
        if nn > 1e-9:
            n_in /= nn
        right = np.asarray(
            box["face_right_L"], dtype=float
        ).reshape(3).copy()
        rn = float(np.linalg.norm(right))
        if rn > 1e-9:
            right /= rn
        self._push_box = {
            "center": np.asarray(
                box["center_L"], dtype=float
            ).reshape(3).copy(),
            "n_in": n_in,
            "right": right,
            "source": str(box.get("source", "?")),
            "width_m": float(box.get("width_m", float("nan"))),
        }
        self._push_box_rx = time.monotonic()

    def _enter_push(self) -> None:
        """MOBILE -> PUSH: leg becomes the pusher, wheels keep driving."""
        self._approach_stop(None)
        self._gait_mode = "push"
        self._manip_init_pending = True
        self._push_e_des_m = 0.0
        self._push_last_twist = (0.0, 0.0, 0.0)
        self._push_waiting = False
        self._reset_mobile_leg_stow()
        box = self._read_box_setpoint()
        if box is not None:
            self._store_push_box(box)
        src = (self._push_box or {}).get("source", "?")
        print(
            "[gait] LT -> PUSH: box face acquired "
            f"(source={src}); LEFT stick = box fwd/steer, LT exits"
        )

    def _update_push_mode(
        self,
        gamepad_msg,
        q: np.ndarray,
        qd: np.ndarray,
        dt: float,
        control_enabled: bool,
    ) -> np.ndarray:
        """One PUSH tick: leg contact torque + wheel twist. Returns tau.

        Three-layer structure (fast to slow):
          1. contact force  -- leg Cartesian PD presses push_contact_depth_m
             into the face along the normal (quasi-admittance, capped);
          2. contact point  -- LEFT stick X offsets the contact by e along
             the face-right axis; the resulting torque about the box's
             friction center steers the box. The wheel lateral velocity
             re-centers the leg workspace under the commanded contact;
          3. transport      -- LEFT stick Y advances along the face normal;
             wz keeps the body (camera) square to the face. |wz| is capped
             by push_kappa_max_1_m * |v| (stable-pushing motion cone).
        """
        cfg = self.lcm_cfg
        if bool(self._manip_init_pending):
            # FK-seed the shared Cartesian target once at entry.
            self._update_manip_foot_target(None, q, 0.0)

        box = self._read_box_setpoint()
        now_m = time.monotonic()
        if box is not None:
            self._store_push_box(box)
            if bool(self._push_waiting):
                self._push_waiting = False
                print("[push] box re-captured -> resume")

        # Stick semantics: LEFT stick commands the BOX, not the base.
        ls_x = ls_y = 0.0
        if gamepad_msg is not None:
            try:
                ls_x = float(gamepad_msg.leftStickAnalog[0])
                ls_y = float(gamepad_msg.leftStickAnalog[1])
            except Exception:
                ls_x = ls_y = 0.0
        dz = float(cfg.stick_deadzone)
        ls_x = 0.0 if abs(ls_x) < dz else ls_x
        ls_y = 0.0 if abs(ls_y) < dz else ls_y
        v_push = -ls_y * float(getattr(cfg, "push_v_max_mps", 0.25))
        # Steering: stick RIGHT turns the BOX right (CW from above), which
        # needs the pushing torque of a contact LEFT of the friction
        # center, i.e. a negative offset along face_right.
        e_max = float(getattr(cfg, "push_e_max_m", 0.12))
        e_rate = float(getattr(cfg, "push_e_rate_mps", 0.08))
        self._push_e_des_m = float(np.clip(
            self._push_e_des_m + (-ls_x) * e_rate * float(dt),
            -e_max, e_max,
        ))
        self._push_v_last = float(v_push)

        age = now_m - float(self._push_box_rx)
        stale_s = float(getattr(cfg, "push_box_stale_s", 1.0))
        memory_s = float(getattr(cfg, "approach_memory_timeout_s", 3.0))
        lost = self._push_box is None or age > memory_s
        if (not lost) and box is None and age > 0.5 * stale_s:
            # Short dropout: advance the memory by the commanded twist.
            # The twist lives in the drive frame; the box memory in L.
            vx_l, vy_l, wz_l = self._push_last_twist
            vx_L, vy_L = self._d2l_xy(vx_l, vy_l)
            b = self._push_box
            self._dead_reckon_xy(
                [b["center"]], [b["n_in"], b["right"]],
                vx_L, vy_L, wz_l, dt,
            )

        if lost:
            if not bool(self._push_waiting):
                self._push_waiting = True
                print(
                    "[push] box lost > "
                    f"{memory_s:.1f}s -> wheels stop, leg holds contact"
                )
            self._push_last_twist = (0.0, 0.0, 0.0)
            self._wheel_pending_cmd = np.zeros(3, dtype=float)
            self._wheel_pending_enable = bool(control_enabled)
        else:
            b = self._push_box
            center = b["center"]
            n_in = b["n_in"]
            right = b["right"]
            # Desired contact point on the face (before workspace clamp).
            raw_tgt = (
                center
                + self._push_e_des_m * right
                + float(getattr(cfg, "push_contact_depth_m", 0.02)) * n_in
            )
            goal = self._clamp_manip_foot_cartesian(raw_tgt)
            cur = np.asarray(
                self._manip_foot_des_b, dtype=float
            ).reshape(3)
            rate = float(max(1e-3, getattr(cfg, "push_foot_rate_mps", 0.05)))
            delta = goal - cur
            dist = float(np.linalg.norm(delta))
            step = rate * float(max(0.0, dt))
            if dist > 1e-6:
                cur = cur + delta * (min(step, dist) / dist)
            self._manip_foot_des_b = self._clamp_manip_foot_cartesian(cur)

            # Wheel twist: advance along the face normal, re-center the leg
            # laterally, keep the body square to the face. The box memory
            # is L-frame (leg needs it); the WHEELS need drive frame.
            ndx, ndy = self._l2d_xy(n_in[0], n_in[1])
            n_h = np.asarray([ndx, ndy])
            n_hn = float(np.linalg.norm(n_h))
            n_h = n_h / n_hn if n_hn > 1e-6 else np.array([1.0, 0.0])
            rdx, rdy = self._l2d_xy(right[0], right[1])
            r_h = np.asarray([rdx, rdy])
            r_hn = float(np.linalg.norm(r_h))
            if r_hn > 1e-6:
                r_h = r_h / r_hn
            else:
                r_h = np.array([-n_h[1], n_h[0]])
            # Same n_in half-plane guard as approach_law (tag in front
            # of the camera => press direction must have +drive-x).
            if ndx * float(center[0]) + ndy * float(center[1]) < 0.0:
                ndx, ndy = -ndx, -ndy
                n_h = -n_h
            yaw_err = math.atan2(ndy, ndx)
            wz_max = float(getattr(cfg, "push_wz_max_rad_s", 0.5))
            # Same body-frame sense as mobile approach / stick yaw.
            wz = float(np.clip(
                float(getattr(cfg, "push_kp_wz", 1.5)) * yaw_err,
                -wz_max, wz_max,
            ))
            # Stable-pushing curvature cap: don't yaw faster than the
            # advance speed allows or the contact slips out of the cone.
            kappa = float(getattr(cfg, "push_kappa_max_1_m", 1.5))
            wz_cap = kappa * max(abs(v_push), 0.05)
            wz = float(np.clip(wz, -wz_cap, wz_cap))
            _, raw_tgt_dy = self._l2d_xy(raw_tgt[0], raw_tgt[1])
            v_lat = float(np.clip(
                float(getattr(cfg, "push_kp_vy", 0.8)) * raw_tgt_dy,
                -float(getattr(cfg, "push_v_max_mps", 0.25)),
                float(getattr(cfg, "push_v_max_mps", 0.25)),
            ))
            v_max = float(getattr(cfg, "push_v_max_mps", 0.25))
            vx = float(np.clip(
                v_push * n_h[0] + v_lat * r_h[0], -v_max, v_max
            ))
            vy = float(np.clip(
                v_push * n_h[1] + v_lat * r_h[1], -v_max, v_max
            ))
            self._push_last_twist = (vx, vy, wz)
            self._wheel_pending_cmd = self._kiwi_ik(vx, vy, wz)
            self._wheel_pending_enable = bool(control_enabled)

        # Leg: low-gain Cartesian PD onto the (possibly held) contact.
        tau_send, self._manip_err_m, self._manip_speed_mps = (
            self.core.compute_stand_swing_tau(
                joint_pos=np.asarray(q, dtype=float).reshape(3),
                joint_vel=np.asarray(qd, dtype=float).reshape(3),
                leg_len_des_m=float(np.linalg.norm(self._manip_foot_des_b)),
                tau_max_nm=float(getattr(cfg, "push_tau_max_nm", 2.0)),
                foot_des_b=np.asarray(
                    self._manip_foot_des_b, dtype=float
                ).reshape(3),
                kp_z=float(getattr(cfg, "push_kp_n_m", 200.0)),
                kd_z=float(getattr(cfg, "push_kd_n_s_m", 2.0)),
                axial_ff_n=0.0,
                cartesian_pd=True,
            )
        )
        return np.asarray(tau_send, dtype=float).reshape(3)

    def _update_manip_button_target(
        self,
        joint_pos: np.ndarray,
        dt: float,
    ) -> np.ndarray | None:
        """Run latched open-loop crawl: home -> pre -> face -> press -> hold.

        Returns foot_des_b or None if auto path inactive (caller uses stick).
        Once latched (from MOBILE READY / enter), the setpoint NEVER follows
        live tag again — hand occlusion during press must not cancel motion.
        """
        if not bool(getattr(self.lcm_cfg, "manip_button_auto_enable", False)):
            return None

        # Initialize once from measured FK. During tag acquisition, hold this
        # Cartesian point instead of unexpectedly handing control to sticks.
        if bool(self._manip_init_pending):
            self._update_manip_foot_target(None, joint_pos, 0.0)

        # Open-loop after latch: do not poll live perception (occlusion).
        if self._btn_last_setpoint is not None:
            sp = None  # unused; path below uses the latch only
        else:
            sp = self._read_button_setpoint()
        if self._btn_last_setpoint is None:
            self._btn_stage = "wait_tag"
            self._btn_stage_print = "wait_tag"
            if sp is not None:
                sample_t = float(sp.get("t_wall", 0.0))
                if sample_t > float(self._btn_last_sample_t_wall) + 1e-6:
                    self._btn_last_sample_t_wall = sample_t
                    self._btn_acquire_samples.append(sp)

                count = int(max(
                    1, getattr(self.lcm_cfg, "manip_button_acquire_samples", 8)
                ))
                self._btn_acquire_samples = self._btn_acquire_samples[-count:]
                if len(self._btn_acquire_samples) >= count:
                    faces = np.asarray([
                        s["foot_face_L"] for s in self._btn_acquire_samples
                    ], dtype=float).reshape(count, 3)
                    face = np.median(faces, axis=0)
                    spread = float(np.max(np.linalg.norm(
                        faces - face.reshape(1, 3), axis=1
                    )))
                    tol = float(max(
                        0.001,
                        getattr(
                            self.lcm_cfg, "manip_button_acquire_tol_m", 0.010
                        ),
                    ))
                    if spread <= tol:
                        normals = np.asarray([
                            s["wall_normal_in_L"]
                            for s in self._btn_acquire_samples
                        ], dtype=float).reshape(count, 3)
                        n_in = np.median(normals, axis=0)
                        n_norm = float(np.linalg.norm(n_in))
                        if n_norm > 1e-6:
                            n_in = n_in / n_norm
                            press_m = float(getattr(
                                self.lcm_cfg, "manip_button_press_m", 0.05
                            ))
                            # Rebuild approach and press from one latched face
                            # and one unit wall normal. This guarantees that
                            # face->press is perpendicular to the wall.
                            latched = dict(sp)
                            latched["foot_face_L"] = face.tolist()
                            latched["foot_pre_L"] = (
                                face - 0.03 * n_in
                            ).tolist()
                            latched["foot_press_L"] = (
                                face + press_m * n_in
                            ).tolist()
                            latched["wall_normal_in_L"] = n_in.tolist()
                            latched["press_m"] = press_m
                            self._btn_last_setpoint = latched
                            # If we entered without a prior HOME (rare
                            # wait_tag path), still start from home.
                            self._btn_stage = "home"
                            self._btn_hold_t0 = None
                            print(
                                "[manip-btn] tag stable -> START: "
                                f"HOME -> pre -> face -> press({press_m*100:.0f}cm) "
                                f"(tag_id={sp.get('tag_id')}, "
                                f"spread={spread*1000:.1f}mm)"
                            )
                    else:
                        # Keep a rolling acquisition window until the target
                        # is stable; never move toward a noisy pose.
                        self._btn_acquire_samples = (
                            self._btn_acquire_samples[-max(1, count - 1):]
                        )

            if self._btn_last_setpoint is None:
                return np.asarray(
                    self._manip_foot_des_b, dtype=float
                ).reshape(3)

        sp = self._btn_last_setpoint
        # Keep press stroke at the configured depth (still from LATCH only).
        sp = self._rebuild_button_press(sp)
        self._btn_last_setpoint = sp
        n_in = np.asarray(
            sp["wall_normal_in_L"], dtype=float
        ).reshape(3)
        nn = float(np.linalg.norm(n_in))
        if nn > 1e-9:
            n_in = n_in / nn
        self._manip_press_n_L = n_in.copy()
        goals = {
            "home": self._manip_home_foot(),
            "pre": np.asarray(sp["foot_pre_L"], dtype=float).reshape(3),
            "face": np.asarray(sp["foot_face_L"], dtype=float).reshape(3),
            "press": np.asarray(sp["foot_press_L"], dtype=float).reshape(3),
        }

        stage = str(self._btn_stage)
        if stage == "home":
            goal_raw = goals["home"]
        elif stage in ("press", "hold"):
            goal_raw = goals["press"]
        elif stage == "face":
            goal_raw = goals["face"]
        elif stage in ("pre", "retract", "done"):
            goal_raw = goals["pre"]
        else:
            self._btn_stage = "home"
            stage = "home"
            goal_raw = goals["home"]

        # Log 23:10: crawling toward an out-of-workspace pre then
        # hard-clamping each tick froze the foot (err stuck ~297 mm,
        # never left pre) → limp press. Crawl toward the CLAMPED goal
        # for home/pre/face so stages can finish; during press/hold
        # keep the wall-normal stroke (soft L only) so PD can push.
        if stage in ("press", "hold"):
            goal_cmd = np.asarray(goal_raw, dtype=float).reshape(3).copy()
            if float(goal_cmd[2]) < 0.05:
                goal_cmd[2] = 0.05
            L = float(np.linalg.norm(goal_cmd))
            L_soft = 0.62
            if L > L_soft and L > 1e-9:
                goal_cmd *= L_soft / L
            self._manip_press_boost = True
        else:
            goal_cmd = self._clamp_manip_foot_cartesian(goal_raw)
            self._manip_press_boost = False

        cur = np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
        rate = float(max(1e-3, self.lcm_cfg.manip_button_rate_mps))
        delta = goal_cmd - cur
        dist = float(np.linalg.norm(delta))
        step = rate * float(max(0.0, dt))
        if dist > 1e-6:
            cur = cur + delta * (min(step, dist) / dist)
        if stage in ("press", "hold"):
            self._manip_foot_des_b = cur.astype(float)
        else:
            self._manip_foot_des_b = self._clamp_manip_foot_cartesian(cur)

        arrive = float(max(0.001, self.lcm_cfg.manip_button_arrive_m))
        home_arrive = float(max(
            arrive, getattr(self.lcm_cfg, "manip_home_arrive_m", 0.02)
        ))
        # Recompute dist to the commanded goal after clamp/snap — the
        # pre-clamp residual was why pre never advanced (log 23:12).
        dist = float(np.linalg.norm(
            goal_cmd - np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
        ))
        gate = home_arrive if stage == "home" else arrive
        if dist <= gate:
            # Snap so the next stage starts from the stage goal, not a
            # nearby clamp artifact.
            self._manip_foot_des_b = np.asarray(goal_cmd, dtype=float).reshape(3)
            dist = 0.0

        pm_cm = float(getattr(self.lcm_cfg, "manip_button_press_m", 0.05)) * 100.0
        if stage == "home" and dist <= home_arrive:
            self._btn_stage = "pre"
            print("[manip-btn] HOME arrived (q≈0.4 down) -> pre")
        elif stage == "pre" and dist <= arrive:
            self._btn_stage = "face"
            print("[manip-btn] pre arrived -> face")
        elif stage == "face" and dist <= arrive:
            self._btn_stage = "press"
            print(f"[manip-btn] face arrived -> press {pm_cm:.0f} cm")
        elif stage == "press" and dist <= arrive:
            self._btn_stage = "hold"
            self._btn_hold_t0 = time.time()
            print(
                "[manip-btn] press arrived -> hold "
                f"{float(self.lcm_cfg.manip_button_hold_s):.2f}s "
                f"(boost tau/"
                f"{float(getattr(self.lcm_cfg, 'manip_press_tau_max_nm', 5.0)):.0f}Nm)"
            )
        elif stage == "hold":
            t0 = float(self._btn_hold_t0 or time.time())
            if (time.time() - t0) >= float(self.lcm_cfg.manip_button_hold_s):
                self._btn_stage = "retract"
                self._manip_press_boost = False
                print("[manip-btn] hold done -> retract")
        elif stage == "retract" and dist <= arrive:
            self._btn_stage = "done"
            print("[manip-btn] retract done -> backup away from wall")
            self._start_post_press_backup()

        self._btn_stage_print = str(self._btn_stage)
        return np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)

    def _start_post_press_backup(self) -> None:
        """Leave MANIPULATION and reverse wheels along -n (away from wall)."""
        if not bool(getattr(self.lcm_cfg, "manip_backup_enable", True)):
            print("[manip-btn] backup disabled -> stay at done/pre")
            return
        n_L = getattr(self, "_manip_press_n_L", None)
        if n_L is None and self._btn_last_setpoint is not None:
            n_L = np.asarray(
                self._btn_last_setpoint.get("wall_normal_in_L", [1.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(3)
        if n_L is None:
            n_L = np.array([1.0, 0.0, 0.0], dtype=float)
        n_L = np.asarray(n_L, dtype=float).reshape(3)
        nn = float(np.linalg.norm(n_L[:2]))
        if nn < 1e-9:
            n_dx, n_dy = 1.0, 0.0
        else:
            n_dx, n_dy = self._l2d_xy(float(n_L[0]), float(n_L[1]))
            nh = math.hypot(n_dx, n_dy) or 1.0
            n_dx, n_dy = n_dx / nh, n_dy / nh
        # n points INTO the wall → backup is opposite.
        self._backup_dir_D = (-n_dx, -n_dy)
        self._backup_remain_m = float(max(
            0.05, getattr(self.lcm_cfg, "manip_backup_m", 0.30)
        ))
        self._backup_active = True
        self._approach_stop(None)
        self._gait_mode = "mobile"
        self._manip_init_pending = False
        self._manip_press_boost = False
        self._reset_mobile_leg_stow()
        # Keep button state as done; clear latch so a new LT can re-acquire.
        self._btn_stage = "done"
        self._btn_stage_print = "backup"
        self._btn_last_setpoint = None
        print(
            f"[gait] PRESS done -> MOBILE BACKUP "
            f"{self._backup_remain_m*100:.0f} cm along "
            f"drive=({self._backup_dir_D[0]:+.2f},{self._backup_dir_D[1]:+.2f})"
        )

    def _update_post_press_backup(
        self, gamepad_msg, dt: float
    ) -> np.ndarray | None:
        """Open-loop reverse after button press. Returns wheel rad/s or None."""
        if not bool(self._backup_active):
            return None
        # Stick cancels, same idea as approach override.
        try:
            sticks = (
                float(gamepad_msg.leftStickAnalog[0]),
                float(gamepad_msg.leftStickAnalog[1]),
                float(gamepad_msg.rightStickAnalog[0]),
                float(gamepad_msg.rightStickAnalog[1]),
            )
        except Exception:
            sticks = ()
        thr = float(getattr(
            self.lcm_cfg, "mobile_approach_stick_override", 0.25
        ))
        if any(abs(s) > thr for s in sticks):
            self._backup_active = False
            self._backup_remain_m = 0.0
            print("[backup] OFF (stick override)")
            return None

        v = float(max(0.02, getattr(self.lcm_cfg, "manip_backup_v_mps", 0.12)))
        dx, dy = self._backup_dir_D
        vx, vy = v * float(dx), v * float(dy)
        self._backup_remain_m -= v * float(max(0.0, dt))
        now = time.monotonic()
        if (now - float(self._backup_dbg_t)) >= 0.5:
            self._backup_dbg_t = now
            print(
                f"[backup] remain={max(0.0, self._backup_remain_m)*100:.0f} cm "
                f"v_D=({vx:+.2f},{vy:+.2f})"
            )
        if self._backup_remain_m <= 0.0:
            self._backup_active = False
            self._backup_remain_m = 0.0
            print("[backup] done -> MOBILE manual")
            return np.zeros(3, dtype=float)
        return self._kiwi_ik(vx, vy, 0.0)

    def _update_manip_foot_target(
        self,
        gamepad_msg,
        joint_pos: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Integrate the deployed-leg command on a spherical workspace.

        The target is stored as a body/FK-frame base-to-foot vector. Right
        stick integrates its X/Y coordinates; left-stick Y changes the radius.
        Z is reconstructed from the positive hemisphere, so the commanded
        foot always lies on ||p||=L and its surface normal is p/||p||.
        """
        if bool(self._manip_init_pending):
            try:
                if self.core.fk is None:
                    raise RuntimeError("manipulation requires delta-leg FK")
                foot_now, _ = self.core.fk.forward_kinematics(
                    np.asarray(joint_pos, dtype=float).reshape(3)
                )
                foot_now = np.asarray(foot_now, dtype=float).reshape(3)
                if not np.all(np.isfinite(foot_now)):
                    raise ValueError("non-finite FK")
                self._manip_foot_des_b = foot_now.copy()
                print(
                    "[manip] target initialized from FK p_b=[%+.3f,%+.3f,%+.3f] m"
                    % tuple(float(x) for x in foot_now)
                )
            except Exception as exc:
                self._manip_foot_des_b = np.array(
                    [0.0, 0.0, float(self.lcm_cfg.switch_rb_leg_len_m)],
                    dtype=float,
                )
                print("[manip] FK init failed (%s); using vertical target" % exc)
            self._manip_init_pending = False

        target = np.asarray(
            self._manip_foot_des_b, dtype=float
        ).reshape(3).copy()
        L = float(np.linalg.norm(target))
        L = float(np.clip(
            L,
            float(self.lcm_cfg.manip_leg_len_min_m),
            float(self.lcm_cfg.manip_leg_len_max_m),
        ))
        rs_x = rs_y = ls_y = 0.0
        if gamepad_msg is not None:
            try:
                rs_x = float(gamepad_msg.rightStickAnalog[0])
                rs_y = float(gamepad_msg.rightStickAnalog[1])
                ls_y = float(gamepad_msg.leftStickAnalog[1])
            except Exception:
                rs_x = rs_y = ls_y = 0.0
        dz = float(self.lcm_cfg.stick_deadzone)
        rs_x = 0.0 if abs(rs_x) < dz else rs_x
        rs_y = 0.0 if abs(rs_y) < dz else rs_y
        ls_y = 0.0 if abs(ls_y) < dz else ls_y

        # Right stick follows the same body XY convention as MOBILE:
        # up (-Y) -> +X, right (+X) -> +Y (L frame is FRD, +Y right).
        xy_rate = float(max(0.0, self.lcm_cfg.manip_xy_rate_mps))
        target[0] += (-rs_y) * xy_rate * float(dt)
        target[1] += rs_x * xy_rate * float(dt)
        # Left stick up is physical foot-up/retract (smaller radius);
        # left stick down extends the leg.
        L += ls_y * float(max(
            0.0, self.lcm_cfg.manip_leg_len_rate_mps
        )) * float(dt)
        L = float(np.clip(
            L,
            float(self.lcm_cfg.manip_leg_len_min_m),
            float(self.lcm_cfg.manip_leg_len_max_m),
        ))

        xy = target[:2].copy()
        xy_norm = float(np.linalg.norm(xy))
        tilt_lim = math.radians(float(np.clip(
            self.lcm_cfg.manip_tilt_max_deg, 0.0, 89.0
        )))
        xy_lim = float(min(
            max(0.0, self.lcm_cfg.manip_xy_max_m),
            L * math.sin(tilt_lim),
            max(0.0, L - 1e-4),
        ))
        if xy_norm > xy_lim and xy_norm > 1e-9:
            xy *= xy_lim / xy_norm
        target[0:2] = xy
        target[2] = math.sqrt(max(
            1e-8, L * L - float(np.dot(xy, xy))
        ))
        self._manip_foot_des_b = target.copy()
        return target

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
        # Clear the latched RM current so the zeroed hopper_cmd does not carry
        # a stale nonzero rm_iq_des out with it.
        try:
            self._rm_iq_des = np.zeros(3, dtype=float)
        except Exception:
            pass
        for _ in range(5):
            try:
                self._publish_hopper_cmd(np.zeros(3, dtype=float))
                self._publish_wheel_cmd(np.zeros(3, dtype=float), enable=False)
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
                # NRC stance law + prop energy supplement
                "nrc_r",
                "nrc_r_star",
                "nrc_f_des",
                "nrc_h_trim",
                "mode1_Eloss_j",
                "fl_ev_xy",
                "fl_tilt_cmd_deg",
                "rpy_des_roll",
                "rpy_des_pitch",
                "fl_lat_force_n",
                "prop_energy_fz",
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
                # Stance attitude split (HFA debug): desired, leg target,
                # leg delivered, prop residual, and utilization.
                "tau_b_att_des0",
                "tau_b_att_des1",
                "tau_b_att_des2",
                "tau_b_leg_des0",
                "tau_b_leg_des1",
                "tau_b_leg_des2",
                "tau_b_leg_act0",
                "tau_b_leg_act1",
                "tau_b_leg_act2",
                "tau_b_res_des0",
                "tau_b_res_des1",
                "tau_b_res_des2",
                "tau_b_att_des_xy_norm",
                "tau_b_leg_des_xy_norm",
                "tau_props_xy_norm",
                "leg_att_share",
                "tau_cap_scale",
                "prop_fz_assist_n",
                # apex / takeoff debug
                "z_apex_actual_m",
                "v_to_cmd_m_s",
                "desired_vz_from_apex_m_s",
                "hop_height_m",
                "h_com_flight_m",
                "h_leg_relative_m",
                "h_foot_clear_m",
                "h_com_imu_m",
                "h_leg_imu_m",
                "h_foot_clear_imu_m",
                "v_lo_imu_m_s",
                "foot_clearance_target_m",
                "foot_clearance_control_enable",
                "foot_clearance_com_target_m",
                "flight_phase_s",
                "lo_event_q_shift_m",
                "td_event_q_shift_m",
                "lo_unload_delay_s",
                "td_impact_delay_s",
                "specific_up_mps2",
                # FB-SLIP: TD-sized constant brake/push forces + plan
                "fbslip_v_td_m_s",
                "fbslip_f_brake_n",
                "fbslip_x_c_plan_m",
                "fbslip_x_td_m",
                "fbslip_s_tgt_m",
                "fbslip_f_push_n",
                "fbslip_t_bottom_s",
                "fbslip_sink",
                "apex_eta",
                "apex_e_bias",
                # Apex-to-apex hybrid work controller
                "height_work_active",
                "height_apex_energy_j",
                "height_loss_hat_j",
                "height_loss_meas_j",
                "height_loss_ff_j",
                "height_apex_confidence",
                "height_work_req_j",
                "height_work_leg_j",
                "height_work_prop_j",
                "height_x_star_m",
                "height_x_bottom_m",
                "height_work_extension_m",
                "height_v_comp_td_mps",
                "height_compression_k_n_m",
                "height_ascent",
                "height_leg_excess_force_n",
                "height_brake_force_n",
                "height_prop_push_ratio",
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
                # MANIPULATION / AprilTag button telemetry.
                "manip_stage",
                "manip_tag_latched",
                "manip_tag_id",
                "manip_tag_age_s",
                "manip_cmd_x",
                "manip_cmd_y",
                "manip_cmd_z",
                "manip_actual_x",
                "manip_actual_y",
                "manip_actual_z",
                "manip_err_m",
                "manip_speed_mps",
                "manip_pre_x",
                "manip_pre_y",
                "manip_pre_z",
                "manip_face_x",
                "manip_face_y",
                "manip_face_z",
                "manip_press_x",
                "manip_press_y",
                "manip_press_z",
                "manip_wall_n_in_x",
                "manip_wall_n_in_y",
                "manip_wall_n_in_z",
                "manip_kp",
                "manip_outer_kd",
                "manip_ak60_kd",
                "manip_tau_max_nm",
                "manip_target_rate_mps",
                # MOBILE tag pre-acquisition + auto-approach + PUSH telemetry
                # (appended at the end so column indices stay stable).
                "mobile_tag_state",
                "mobile_tag_cam_z_m",
                "approach_active",
                "push_e_m",
                "push_v_mps",
                "box_source",
                "box_cx",
                "box_cy",
                "box_cz",
                "box_nx",
                "box_ny",
                "box_nz",
                "box_width_m",
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

        # Rising edge on Y:
        # - reset controller estimates (v_hat, integrators, etc.)
        # - enable velocity hard-hold until we enter STANCE once
        # - restart CSV logging (close current file if any, then start a new one)
        if bool(y_now) and (not bool(self._last_y)):
            try:
                self.core.user_reset(zero_yaw=True)
            except Exception:
                pass
            self._y_hold_until_stance = True
            try:
                self.core.user_zero_velocity_hold(True)
            except Exception:
                pass
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
        if (bool(self._switch_loop)
                or self._gait_mode in ("mobile", "manipulation")
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
        hop_h = float(self.core.cfg.hop_height_m)
        if np.all(np.isfinite(foot_vicon)):
            leg_len = float(np.linalg.norm(foot_vicon))
            line = (
                f"[{gait_tag}|{mode_tag}|{ph}] "
                f"leg_len={leg_len:.4f} m hop_height={hop_h:.3f} m"
            )
        else:
            line = (
                f"[{gait_tag}|{mode_tag}|{ph}] "
                f"leg_len=nan hop_height={hop_h:.3f} m"
            )
        # Leg-frame foot XYZ (same as foot_vicon used for leg_len).
        if np.all(np.isfinite(foot_vicon)):
            line += (
                f" xyz={foot_vicon[0]:+.3f}/{foot_vicon[1]:+.3f}/"
                f"{foot_vicon[2]:+.3f} m"
            )
        else:
            line += " xyz=nan/nan/nan m"
        vd = np.asarray(
            getattr(self, "_vdes_last", np.zeros(2)), dtype=float
        ).reshape(2)
        line += f" vdes={vd[0]:+.2f}/{vd[1]:+.2f} m/s"
        if self._gait_mode == "mobile":
            w = np.asarray(
                getattr(self, "_wheel_last_cmd", np.zeros(3)), dtype=float
            ).reshape(3)
            en = "on" if bool(getattr(self, "_wheel_last_enable", False)) \
                else "off"
            if bool(getattr(self, "_mobile_leg_holding", False)):
                leg_st = "hold"
            elif bool(getattr(self, "_mobile_leg_active", False)):
                leg_st = "stow"
            else:
                leg_st = "idle"
            if gait_tag == "MOBILE":
                # Task-centric MOBILE status: tag / reachability / box
                # replace the hop metrics (leg_len etc. are HOPPING data).
                line = f"[{gait_tag}|{mode_tag}]"
                tag_st = str(self._mobile_tag_state).upper()
                dist = float(getattr(
                    self, "_mobile_tag_cam_z_m", float("nan")
                ))
                tag_fresh = (
                    time.monotonic() - float(self._mobile_tag_last_sp_rx)
                ) <= float(getattr(self.lcm_cfg, "manip_button_stale_s", 1.5))
                if tag_fresh and np.isfinite(dist):
                    # "@Xm" = ||foot_face_L|| from leg origin (ready 0.50-0.53).
                    line += f" tag={tag_st}@|L|{dist:.2f}m"
                else:
                    line += f" tag={tag_st}"
                if tag_st == "READY":
                    line += " manip=OK"
                else:
                    line += " manip=FAR"
                    reach = float(self._mobile_tag_reach_error_m)
                    if np.isfinite(reach):
                        line += f"(clip={reach * 100:.1f}cm)"
                if bool(self._approach_active):
                    if bool(self._approach_waiting):
                        line += " approach=WAIT"
                    elif tag_st == "READY":
                        line += " approach=HOLD->MANIP"
                    else:
                        line += " approach=SERVO"
                box = self._read_box_setpoint()
                if box is not None:
                    line += f" box={str(box.get('source', '?')).upper()}"
                    w_box = float(box.get("width_m", float("nan")))
                    if np.isfinite(w_box):
                        line += f"({w_box:.2f}m)"
                else:
                    line += " box=NO"
            line += (
                f" wheels[{en}]="
                f"{w[0]:+.1f}/{w[1]:+.1f}/{w[2]:+.1f} rad/s"
                f" leg={leg_st}"
            )
        elif self._gait_mode == "push":
            w = np.asarray(
                getattr(self, "_wheel_last_cmd", np.zeros(3)), dtype=float
            ).reshape(3)
            en = "on" if bool(getattr(self, "_wheel_last_enable", False)) \
                else "off"
            p = np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
            box = self._push_box
            src = str((box or {}).get("source", "?")).upper()
            age = time.monotonic() - float(self._push_box_rx)
            state = "WAIT" if bool(self._push_waiting) else "TRACK"
            line += (
                f" push[{src}|{state} age={age:.1f}s]"
                f" e={self._push_e_des_m * 100:+.1f}cm"
                f" v={self._push_v_last:+.2f}m/s"
                f" foot={p[0]:+.3f}/{p[1]:+.3f}/{p[2]:+.3f} m"
                f" err={self._manip_err_m * 1000:.1f}mm"
                f" wheels[{en}]={w[0]:+.1f}/{w[1]:+.1f}/{w[2]:+.1f} rad/s"
            )
        elif self._gait_mode == "manipulation":
            p = np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
            btn = str(getattr(self, "_btn_stage_print", "") or "")
            line += (
                f" manip_p={p[0]:+.3f}/{p[1]:+.3f}/{p[2]:+.3f} m"
                f" L={np.linalg.norm(p):.3f}"
                f" err={self._manip_err_m*1000:.1f} mm"
            )
            if btn:
                line += f" btn={btn}"
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
            nrc_r = float(info.get("nrc_r", float("nan")))
            nrc_r_star = float(info.get("nrc_r_star", float("nan")))
            nrc_f_des = float(info.get("nrc_f_des", float("nan")))
            nrc_h_trim = float(info.get("nrc_h_trim", float("nan")))
            mode1_Eloss_j = float(info.get("mode1_Eloss_j", float("nan")))
            fl_ev_xy = float(info.get("fl_ev_xy", float("nan")))
            fl_tilt_cmd_deg = float(info.get("fl_tilt_cmd_deg", float("nan")))
            rpy_des = np.asarray(
                info.get("rpy_des", [np.nan, np.nan, np.nan]), dtype=float
            ).reshape(3)
            fl_lat_force_n = float(info.get("fl_lat_force_n", float("nan")))
            prop_energy_fz = float(info.get("prop_energy_fz", float("nan")))
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
                float(nrc_r),
                float(nrc_r_star),
                float(nrc_f_des),
                float(nrc_h_trim),
                float(mode1_Eloss_j),
                float(fl_ev_xy),
                float(fl_tilt_cmd_deg),
                float(rpy_des[0]),
                float(rpy_des[1]),
                float(fl_lat_force_n),
                float(prop_energy_fz),
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
                # Stance attitude split debug (full body-frame vectors).
                float(info.get("tau_b_att_des0", float("nan"))),
                float(info.get("tau_b_att_des1", float("nan"))),
                float(info.get("tau_b_att_des2", float("nan"))),
                float(info.get("tau_b_leg_des0", float("nan"))),
                float(info.get("tau_b_leg_des1", float("nan"))),
                float(info.get("tau_b_leg_des2", float("nan"))),
                float(info.get("tau_b_leg_act0", float("nan"))),
                float(info.get("tau_b_leg_act1", float("nan"))),
                float(info.get("tau_b_leg_act2", float("nan"))),
                float(info.get("tau_b_res_des0", float("nan"))),
                float(info.get("tau_b_res_des1", float("nan"))),
                float(info.get("tau_b_res_des2", float("nan"))),
                float(info.get("tau_b_att_des_xy_norm", float("nan"))),
                float(info.get("tau_b_leg_des_xy_norm", float("nan"))),
                float(info.get("tau_props_xy_norm", float("nan"))),
                float(info.get("leg_att_share", float("nan"))),
                float(info.get("tau_cap_scale", float("nan"))),
                float(info.get("prop_fz_assist_n", float("nan"))),
                float(z_apex_actual_m),
                float(v_to_cmd_m_s),
                float(desired_vz_from_apex_m_s),
                float(hop_height_m),
                float(info.get("h_com_flight_m", float("nan"))),
                float(info.get("h_leg_relative_m", float("nan"))),
                float(info.get("h_foot_clear_m", float("nan"))),
                float(info.get("h_com_imu_m", float("nan"))),
                float(info.get("h_leg_imu_m", float("nan"))),
                float(info.get("h_foot_clear_imu_m", float("nan"))),
                float(info.get("v_lo_imu_m_s", float("nan"))),
                float(info.get("foot_clearance_target_m", 0.05)),
                int(info.get("foot_clearance_control_enable", 0)),
                float(info.get("foot_clearance_com_target_m", float("nan"))),
                float(info.get("flight_phase_s", float("nan"))),
                float(info.get("lo_event_q_shift_m", float("nan"))),
                float(info.get("td_event_q_shift_m", float("nan"))),
                float(info.get("lo_unload_delay_s", float("nan"))),
                float(info.get("td_impact_delay_s", float("nan"))),
                float(info.get("specific_up_mps2", float("nan"))),
                float(info.get("fbslip_v_td_m_s", float("nan"))),
                float(info.get("fbslip_f_brake_n", float("nan"))),
                float(info.get("fbslip_x_c_plan_m", float("nan"))),
                float(info.get("fbslip_x_td_m", float("nan"))),
                float(info.get("fbslip_s_tgt_m", float("nan"))),
                float(info.get("fbslip_f_push_n", float("nan"))),
                float(info.get("fbslip_t_bottom_s", float("nan"))),
                float(info.get("fbslip_sink", 0)),
                float(info.get("apex_eta", 1.0)),
                float(info.get("apex_e_bias", 0.0)),
                int(info.get("height_work_active", 0)),
                float(info.get("height_apex_energy_j", float("nan"))),
                float(info.get("height_loss_hat_j", float("nan"))),
                float(info.get("height_loss_meas_j", float("nan"))),
                float(info.get("height_loss_ff_j", float("nan"))),
                float(info.get("height_apex_confidence", float("nan"))),
                float(info.get("height_work_req_j", float("nan"))),
                float(info.get("height_work_leg_j", float("nan"))),
                float(info.get("height_work_prop_j", float("nan"))),
                float(info.get("height_x_star_m", float("nan"))),
                float(info.get("height_x_bottom_m", float("nan"))),
                float(info.get(
                    "height_work_extension_m", float("nan")
                )),
                float(info.get("height_v_comp_td_mps", float("nan"))),
                float(info.get(
                    "height_compression_k_n_m", float("nan")
                )),
                int(info.get("height_ascent", 0)),
                float(info.get(
                    "height_leg_excess_force_n", float("nan")
                )),
                float(info.get("height_brake_force_n", float("nan"))),
                float(info.get(
                    "height_prop_push_ratio", float("nan")
                )),
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
                # MANIPULATION / button auto state.
                str(self._btn_stage),
                int(self._btn_last_setpoint is not None),
                int(
                    self._btn_last_setpoint.get("tag_id", -1)
                    if self._btn_last_setpoint is not None else -1
                ),
                float(
                    max(
                        0.0,
                        wall_time_s
                        - float(self._btn_last_setpoint.get("t_wall", 0.0)),
                    )
                    if self._btn_last_setpoint is not None else float("nan")
                ),
                float(self._manip_foot_des_b[0]),
                float(self._manip_foot_des_b[1]),
                float(self._manip_foot_des_b[2]),
                float(foot_vicon[0]),
                float(foot_vicon[1]),
                float(foot_vicon[2]),
                float(self._manip_err_m),
                float(self._manip_speed_mps),
                *[
                    float(x) for x in (
                        self._btn_last_setpoint.get(
                            "foot_pre_L", [np.nan, np.nan, np.nan]
                        )
                        if self._btn_last_setpoint is not None
                        else [np.nan, np.nan, np.nan]
                    )
                ],
                *[
                    float(x) for x in (
                        self._btn_last_setpoint.get(
                            "foot_face_L", [np.nan, np.nan, np.nan]
                        )
                        if self._btn_last_setpoint is not None
                        else [np.nan, np.nan, np.nan]
                    )
                ],
                *[
                    float(x) for x in (
                        self._btn_last_setpoint.get(
                            "foot_press_L", [np.nan, np.nan, np.nan]
                        )
                        if self._btn_last_setpoint is not None
                        else [np.nan, np.nan, np.nan]
                    )
                ],
                *[
                    float(x) for x in (
                        self._btn_last_setpoint.get(
                            "wall_normal_in_L", [np.nan, np.nan, np.nan]
                        )
                        if self._btn_last_setpoint is not None
                        else [np.nan, np.nan, np.nan]
                    )
                ],
                float(self.lcm_cfg.manip_button_kp_n_m),
                float(self.lcm_cfg.manip_button_kd_n_s_m),
                float(self.lcm_cfg.manip_ak60_kd),
                float(self.lcm_cfg.manip_tau_max_nm),
                float(self.lcm_cfg.manip_button_rate_mps),
                str(self._mobile_tag_state),
                float(self._mobile_tag_cam_z_m),
                int(bool(self._approach_active)),
                float(self._push_e_des_m),
                float(self._push_v_last),
                str(
                    self._push_box.get("source", "")
                    if self._push_box is not None else ""
                ),
                *[
                    float(x) for x in (
                        np.asarray(
                            self._push_box["center"], dtype=float
                        ).reshape(3)
                        if self._push_box is not None
                        else [np.nan, np.nan, np.nan]
                    )
                ],
                *[
                    float(x) for x in (
                        np.asarray(
                            self._push_box["n_in"], dtype=float
                        ).reshape(3)
                        if self._push_box is not None
                        else [np.nan, np.nan, np.nan]
                    )
                ],
                float(
                    self._push_box.get("width_m", float("nan"))
                    if self._push_box is not None else float("nan")
                ),
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

        # Start CSV logging as soon as the upper loop runs (not on gamepad Y).
        try:
            self._start_log()
        except Exception as e:
            print(f"[log] auto-start failed: {e}")

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
                # LEFT stick Y detent: up-release +2 cm, down-release -2 cm.
                self._handle_hop_height_stick(gamepad_msg)
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
                # For the status line: verify stick axis signs on the
                # ground (stick up should read vdes=+X) before hopping.
                self._vdes_last = np.asarray(
                    desired_v_xy, dtype=float
                ).reshape(2).copy()
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
                         if (bool(self._switch_loop)
                             or self._gait_mode in (
                                 "mobile", "manipulation", "push")
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
                    self._reset_mobile_leg_stow()
                    self._close_prop_base_window("LT -> MOBILE")
                    self._rm_start(
                        float(self.lcm_cfg.rm_mobile_rad),
                        "LT job (push end -> MOBILE)",
                    )
                    print(
                        "[gait] MOBILE: props OFF; leg stows after "
                        "wheel motion (1 Nm -> damp hold)"
                    )
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
                        self._reset_mobile_leg_stow()
                        props_active = False
                        print(
                            "[gait] LT STAND done (%s) -> MOBILE: "
                            "props OFF; leg stows after wheel motion"
                            % ("RM reached" if rm_done else "TIMEOUT")
                        )
                elif self._rt_stand_t0 is not None:
                    # RT stand-and-unfold (2026-07-24 07:05): leg holds the
                    # P4 length UNCAPPED (inverted pendulum, leg carries
                    # the robot) while RM unfolds +11.5 -> 0 synchronized;
                    # props: EVERY mapped motor forced to
                    # switch_rt_stand_prop_pwm_us (1500).  When RM reaches 0
                    # (or timeout) -> plain normal hopping, FLIGHT first.
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    # Weight the LEG must carry = m*g - collective prop lift
                    # at the stand PWM (sqrt law: thrust = k*(pwm-1000)^2),
                    # divided by the linkage force efficiency.
                    # Calibrated from sample1/sample3 false-hop holds
                    # (2026-07-25): the leg held the body statically at a
                    # COMMANDED ~77 N while weight minus prop lift was only
                    # ~54 N -- delivered/commanded = 0.69-0.72, the same
                    # linkage efficiency as the push-work calibration. A
                    # weight-only FF therefore sags and the PD has to make
                    # up ~35% of the load.
                    leg_eff = float(min(1.0, max(0.1, float(getattr(
                        self.modee_cfg, "height_leg_work_efficiency", 0.72
                    )))))
                    ff_n = 0.0
                    if bool(self.lcm_cfg.switch_rt_stand_weight_ff) \
                            and not bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                        d_pwm = max(
                            0.0,
                            float(self.lcm_cfg.switch_rt_stand_prop_pwm_us) - 1000.0,
                        )
                        lift_n = 3.0 * float(self.modee_cfg.prop_k_thrust) * d_pwm * d_pwm
                        ff_n = max(
                            0.0, float(self.modee_cfg.mass_kg) * 9.81 - lift_n
                        ) / leg_eff
                    elif bool(self.lcm_cfg.switch_rt_stand_weight_ff):
                        ff_n = float(self.modee_cfg.mass_kg) * 9.81 / leg_eff
                    tau_send, _err_m, _spd = self.core.compute_stand_swing_tau(
                        joint_pos=q_arr,
                        joint_vel=qd_arr,
                        leg_len_des_m=float(self.lcm_cfg.switch_rb_leg_len_m),
                        tau_max_nm=float(self.lcm_cfg.switch_rt_stand_tau_max_nm),
                        imu_quat_wxyz=imu_quat,
                        kp_z=float(self.lcm_cfg.switch_rt_stand_kp_z_n_m),
                        kd_z=float(self.lcm_cfg.switch_rt_stand_kd_z_n_s_m),
                        axial_ff_n=ff_n,
                    )
                    tau_out_scale_applied = 1.0
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    pwm_us = np.full(6, pwm_min, dtype=float)
                    if bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                        props_active = False
                    else:
                        prop_pwm_up = float(self.lcm_cfg.switch_rt_stand_prop_pwm_us)
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
                        # 2026-08-01 (user): NO RT-STAND special phase.  Enter
                        # normal HOPPING directly: the FIRST hop only differs
                        # by leg length (l0 = switch_rb_first_hop_l0_m, longer
                        # stroke from the stand; restored at the first
                        # liftoff) -- the stance law itself is the NORMAL
                        # cycle (NRC pumps the standing launch from zero
                        # energy like any other energy deficit).  Props run
                        # the normal ModeE allocator from tick one (1100 idle
                        # + attitude differential + energy supplement -- no
                        # forced 1600, no base window).  RM unfolds +11.5 -> 0
                        # AT THE SAME TIME, non-blocking.
                        self._switch_loop = False
                        self._rb_p4_t0 = None
                        self._gait_mode = "hopping"
                        if self._rt_l0_restore is None:
                            self._rt_l0_restore = float(self.core.cfg.leg_l0_m)
                        self.core.cfg.leg_l0_m = float(
                            self.lcm_cfg.switch_rb_first_hop_l0_m
                        )
                        # The in-flight retraction (first-hop l0 -> normal l0)
                        # stretches this arc; skip its eta/apex measurement.
                        self.core._eta_skip_once = True
                        if not bool(self.lcm_cfg.rt_leg_only_no_prop_rm):
                            self._prop_enable = True
                            self._rm_start(
                                float(self.lcm_cfg.rm_hopping_rad),
                                "RT: unfold RM alongside hopping",
                            )
                        print(
                            "[gait] RT -> HOPPING now: first hop l0=%.3f m "
                            "(normal cycle law, restored at liftoff 1), "
                            "props = normal ModeE, RM +11.5 -> %.1f rad "
                            "in parallel"
                            % (float(self.lcm_cfg.switch_rb_first_hop_l0_m),
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
                elif self._gait_mode == "manipulation":
                    # Deployed appendages carry the body. Wheels are stopped
                    # below; the released 3-RSR leg tracks a body-frame point
                    # on its spherical workspace. Prefer live AprilTag button
                    # setpoint (right 16.5 / down 2.6 / protrude 5 / press 1 cm);
                    # otherwise stick teleop.
                    manip_target_b = self._update_manip_button_target(q, dt)
                    button_auto = manip_target_b is not None
                    if manip_target_b is None:
                        manip_target_b = self._update_manip_foot_target(
                            gamepad_msg, q, dt
                        )
                    boost = bool(
                        button_auto and getattr(self, "_manip_press_boost", False)
                    )
                    f_ff = None
                    if button_auto and boost:
                        tau_cap = float(getattr(
                            self.lcm_cfg, "manip_press_tau_max_nm", 8.0
                        ))
                        kp_c = float(getattr(
                            self.lcm_cfg, "manip_button_press_kp_n_m", 550.0
                        ))
                        # Open-loop contact force along latched wall normal
                        # (does not need live tag while the hand occludes it).
                        n_L = getattr(self, "_manip_press_n_L", None)
                        ff_n = float(getattr(
                            self.lcm_cfg, "manip_press_ff_n", 30.0
                        ))
                        if n_L is not None and ff_n > 1e-6:
                            f_ff = (
                                ff_n * np.asarray(n_L, dtype=float).reshape(3)
                            )
                    elif button_auto:
                        tau_cap = float(self.lcm_cfg.manip_tau_max_nm)
                        kp_c = float(self.lcm_cfg.manip_button_kp_n_m)
                    else:
                        tau_cap = float(self.lcm_cfg.manip_tau_max_nm)
                        kp_c = float(self.lcm_cfg.manip_kp_z_n_m)
                    tau_send, self._manip_err_m, self._manip_speed_mps = (
                        self.core.compute_stand_swing_tau(
                            joint_pos=np.asarray(q, dtype=float).reshape(3),
                            joint_vel=np.asarray(qd, dtype=float).reshape(3),
                            leg_len_des_m=float(np.linalg.norm(manip_target_b)),
                            tau_max_nm=tau_cap,
                            foot_des_b=manip_target_b,
                            kp_z=kp_c,
                            kd_z=float(
                                self.lcm_cfg.manip_button_kd_n_s_m
                                if button_auto
                                else self.lcm_cfg.manip_kd_z_n_s_m
                            ),
                            axial_ff_n=0.0,
                            cartesian_pd=bool(button_auto),
                            f_ff_native=f_ff,
                        )
                    )
                    tau_out_scale_applied = 1.0
                    props_active = False
                elif self._gait_mode == "push":
                    # PUSH: leg presses the box face, wheels transport.
                    # LEFT stick commands the BOX (fwd/steer); LT exits.
                    tau_send = self._update_push_mode(
                        gamepad_msg, q, qd, dt, control_enabled
                    )
                    tau_out_scale_applied = 1.0
                    props_active = False
                elif self._gait_mode == "mobile":
                    # MOBILE: kiwi wheels from sticks. Legs stay force-free
                    # until the wheels actually start moving, then soft-stow
                    # to mobile_leg_q_des (1 Nm wall), then P + AK60-D hold.
                    self._monitor_mobile_button_target()
                    wheels_on = bool(
                        control_enabled and int(self._rm_stage) != 2
                    )
                    # LT auto-approach servo overrides the sticks while
                    # active (it cancels itself on stick/B and may enter
                    # MANIPULATION on arrival).
                    w_auto = None
                    if wheels_on:
                        # Post-press reverse has priority over approach/sticks.
                        w_auto = self._update_post_press_backup(gamepad_msg, dt)
                        if w_auto is None:
                            w_auto = self._update_mobile_approach(
                                gamepad_msg, dt
                            )
                    if w_auto is not None:
                        w_cmd = np.asarray(w_auto, dtype=float).reshape(3)
                    else:
                        w_cmd = (
                            self._compute_wheel_cmd(gamepad_msg)
                            if wheels_on else np.zeros(3, dtype=float)
                        )
                    self._wheel_pending_cmd = np.asarray(
                        w_cmd, dtype=float
                    ).reshape(3)
                    self._wheel_pending_enable = bool(wheels_on)
                    tau_send = self._update_mobile_leg_stow(
                        q=q,
                        qd=qd,
                        wheel_omega_cmd=w_cmd,
                        wheels_enabled=wheels_on,
                    )
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
                # gating on top, same class as rm_iq_des. PUSH keeps the
                # wheels live too (leg contact + wheel transport together).
                if self._gait_mode in ("mobile", "push"):
                    self._publish_wheel_cmd(
                        getattr(
                            self, "_wheel_pending_cmd", np.zeros(3)
                        ),
                        enable=bool(getattr(
                            self, "_wheel_pending_enable", False
                        )),
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
                # P4 has Cartesian damping in tau_ff and an additional AK60
                # internal kd using the motor driver's high-rate velocity.
                # MOBILE is force-free until wheel-triggered stow; once the
                # soft approach arrives, hold with tau_ff P + AK60 MIT kd.
                if bool(self._switch_loop):
                    kd_use = float(max(0.0, float(getattr(
                        self.lcm_cfg, "switch_rb_ak60_damp_kd", 0.2
                    ))))
                elif self._gait_mode == "mobile":
                    kd_use = float(max(
                        0.0, float(getattr(self, "_mobile_leg_kd_cmd", 0.0))
                    ))
                elif self._gait_mode in ("manipulation", "push"):
                    # Manipulation/button/push: outer Cartesian position P
                    # only; AK60 supplies the sole velocity damping path.
                    kd_use = float(max(
                        0.0, self.lcm_cfg.manip_ak60_kd
                    ))
                elif (self._lt_stand_t0 is not None
                      or self._rt_stand_t0 is not None):
                    kd_use = 0.0
                else:
                    kd_use = float(kd_stance if in_stance else kd_flight)
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
                self._write_dashboard_status(
                    now=now,
                    q=np.asarray(q, dtype=float).reshape(3),
                    qd=np.asarray(qd, dtype=float).reshape(3),
                    tau_cmd=np.asarray(tau_send, dtype=float).reshape(3),
                    info=info,
                )

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


