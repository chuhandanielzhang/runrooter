from __future__ import annotations

import atexit
import json
import math
import os
import queue
import socket
import sys
import time
import threading
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


class _AuxIO:
    """Single daemon thread for ALL auxiliary I/O of the upper controller.

    PRIMARY vs AUXILIARY (2026-08-20): the 500 Hz control thread must never
    block on I/O. print() under `python3 -u` is a synchronous write to the
    journald pipe; when journald stalls (disk contention with the
    out-of-process LCM CSV logger / camera recorder) the write blocks, the
    loop freezes, and the driver keeps replaying the LAST published PWM --
    in flight this looks like radio packet loss (journal 16:19:19: 695 ms
    PWM freeze right after the LT/B print burst; 16:20:36: 216 ms).

    The control thread therefore only enqueues work here (never blocks);
    this thread performs the actual print / file write / UDP send. If the
    queue is full the item is DROPPED and counted -- losing a status line
    is always better than losing a control tick.
    """

    def __init__(self, maxsize: int = 512) -> None:
        self._q: "queue.Queue" = queue.Queue(maxsize=int(maxsize))
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="aux-io", daemon=True
        )
        self._thread.start()
        # Drain + stop at exit: a daemon thread caught mid-print during
        # interpreter finalization aborts CPython ("could not acquire lock
        # for stdout"). The sentinel makes the worker return; join gives it
        # time to finish the current write.
        atexit.register(self._shutdown)

    def put(self, item) -> None:
        """Enqueue a str (printed) or callable (executed). Never blocks."""
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _shutdown(self) -> None:
        try:
            self._q.put(None, timeout=0.2)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                if callable(item):
                    item()
                else:
                    print(item, flush=True)
            except Exception:
                pass


_AUX = _AuxIO()


def jprint(*args) -> None:
    """Async print: journal text goes through the aux thread, never the
    control thread. Drop-in replacement for print() in this module."""
    _AUX.put(" ".join(str(a) for a in args))


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
    # 2026-08-10: 0.4 -> 0.65 (user: 满杆速度).
    # 2026-08-21: 0.65 -> 0.90 (user: 遥控速度再大点).
    # 2026-08-21 evening: 0.90 -> 1.20 (user: 再加大遥感速度).
    max_cmd_vel: float = 1.20
    stick_deadzone: float = 0.10
    # HOPPING foot-clearance knob (teleop): LEFT stick deflect + release.
    # Writes core.cfg.foot_clearance_target_m (active height when
    # foot_clearance_control_enable). Presets on return-to-neutral:
    #   down -> 0.02 m, left -> 0.03 m, right -> 0.01 m, up -> 0.20 m.
    hop_height_stick_enable: bool = True
    hop_height_stick_trigger: float = 0.70
    hop_height_stick_release: float = 0.20
    hop_height_stick_up_m: float = 0.08
    hop_height_stick_down_m: float = 0.02
    hop_height_stick_left_m: float = 0.05
    hop_height_stick_right_m: float = 0.06
    hop_height_min_m: float = 0.01
    hop_height_max_m: float = 0.20
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
    # 2026-08-11 (user): hopping 不要 output tau 限制；tau 限幅只用于
    # P4 摆腿到 l0（switch_rb_tau_max_nm）。None = pass-through.
    # Was 9 as a backstop after ModeECore's internal 9 Nm model.
    tau_out_max_nm: float | None = None
    # SAFE flag:
    # - If triggered, we request hopper_driver to enter DAMP (same as pressing B),
    #   and pause the Python controller loop for a few seconds.
    safe_rp_deg: float = 50.0
    safe_q_min: float = -1.06     # CASE: retract limit (with kAk60LcmQOffsetRad=-60 deg)
    # CASE: extend limit.  Was 1.40 (a policy number, not the mechanical
    # stop).  User 2026-08-23 hand-posed the leg to q=[-0.71, 1.46, 1.53],
    # foot r=0.374 at z=0.221 -- outside anything the 1.40 box could reach.
    # 1.55 is that pose plus the 0.02 rad of solver margin, no further.
    safe_q_max: float = 1.55
    # Whole-robot gait switch:
    # - OFF/DAMP + LT: no actuator motion; current RM pose is labeled q=+11.5
    #   and MOBILE is selected.
    # - OFF/DAMP + RT: no actuator motion; current RM pose is labeled q=0 and
    #   HOPPING is selected.
    # - enabled HOPPING + LT: at the next liftoff, hold switch_lt_leg_len_m
    #   and drive the three folding-arm RM2006s from 0 to +11.5, entering
    #   MOBILE.
    # - enabled MOBILE + LB: box push (needs 60 mm tags id 2/3).
    #   push_simple=True  -> snap leg to manip_hold_q, sticks drive
    #                        the base exactly like MOBILE (fwd/strafe/yaw).
    #   push_simple=False -> auto-align to the working pose, then
    #                        snap the same hold q and LEFT stick
    #                        drives the BOX (paper push law).
    # - enabled MOBILE + LT: already MOBILE, no-op (LT is gait/mobile only).
    # - enabled MOBILE + RB: AprilTag button (auto-approach then press).
    # - enabled MOBILE + RT: P4 stand then HOPPING.
    # SAFE q-range guard is relaxed while legs are force-free / in P4:
    safe_q_min_switch: float = -1.0472    # -60 deg (SAFE guard only, in LT/P4 modes)
    # P4 stand params (started by enabled MOBILE + RT).
    switch_rb_align_leg_len_m: float = 0.35
    switch_rb_align_tol_m: float = 0.015
    switch_rb_align_timeout_s: float = 1.0
    # P4 hold length.  2026-08-11 15:19 (user): 保持腿 / 第一跳腿长 0.43；
    # 正常跳跃原长仍是 core.leg_l0_m=0.455（第一跳 real TD+LO 后恢复）。
    switch_rb_leg_len_m: float = 0.43
    # Temporary leg_l0 for the FIRST hopping stance after RT/P4. Restored
    # after a real TD then LO (or B abort) — not the fake LO at P4 handoff.
    switch_rb_first_hop_l0_m: float = 0.43
    # ONLY intentional soft tau cap: P4 摆腿到 l0 (align + hold).
    # Hopping has no output/core soft tau limit (2026-08-11 user).
    switch_rb_tau_max_nm: float = 2.0       # RT/P4 leg placement torque cap (Nm)
    # AK60 motor-internal velocity damping during RT/P4 leg placement.
    # Sent through the MIT kd field, using the driver's high-rate velocity;
    # this supplements the Cartesian damping already present in tau_ff.
    switch_rb_ak60_damp_kd: float = 0.3
    switch_rb_pushdelay_s: float = 1.0      # hold at 0.43 this long, then HOPPING
    # Temporary foot_clearance while entering HOPPING from MOBILE (RT).
    # 2026-08-11 14:36 (user): 一二跳目标; 14:39: 0.07 -> 0.05.
    switch_rt_hop_height_m: float = 0.05
    switch_rt_hop_height_n: int = 2
    # Prop baseline during P4 hold (us).
    switch_rb_prop_base_pwm_us: float = 1200.0
    # ModeE collective base once HOPPING starts (RM unfold / first-hop window).
    # History: 1400 -> 1600 -> 1800 (2026-08-11); 2026-08-12 (user): 1700.
    hop_prop_base_pwm_us: float = 1700.0
    # Hard per-motor floor while the prop-base window is open
    # (2026-08-12 user: "最低只能 1600"). Collective stays at
    # hop_prop_base_pwm_us; attitude may still raise individual arms above
    # this, but none may drop below it.
    hop_prop_base_pwm_min_us: float = 1600.0
    # Window closes on first real hop LO (after a TD), or this timeout --
    # whichever first. 2026-08-12 (user): 2.0 -> 1.0 s; no longer gated on
    # RM unfold done.
    hop_prop_base_timeout_s: float = 1.0
    # ---- LT (HOPPING -> MOBILE) stand-and-fold ----
    # 2026-08-10 22:19 (user):
    #   1) next compression (TD) -> RM starts folding 0 -> +11.5
    #   2) legs finish that hop cycle normally (COMP+PUSH)
    #   3) at that stance's liftoff -> props OFF + P4-style stand hold
    #      at switch_lt_leg_len_m; MOBILE when RM reaches mobile pose
    #      (or switch_lt_timeout_s).
    # HOPPING->MOBILE stand length (independent of RT/P4 switch_rb_leg_len_m).
    switch_lt_leg_len_m: float = 0.45
    # Torque cap while the leg carries the robot.  RT/P4 uses 1 Nm since
    # the wheels carry the weight there; holding ~72 N at 0.47 m needs
    # ~5 Nm, so 6 gives margin without unlocking hop-level forces.
    # 2026-08-09 user ("大力"): 6 -> 9, the calibrated hardware torque
    # wall -- the landing catch after the LT push must never sag.  AK60
    # kd=2 (below) and the timeout / RM-reached release stay unchanged.
    switch_lt_tau_max_nm: float = 9.0
    # AK60 motor-internal velocity damping while HOPPING->MOBILE holds the
    # leg at switch_lt_leg_len_m. Outer Cartesian damping remains active.
    switch_lt_ak60_damp_kd: float = 2.0
    # Temporary hop_height for the HOPPING->MOBILE transition hop (the push
    # completed after LT is armed). Restored when LT stand begins.
    switch_lt_hop_height_m: float = 0.05
    # Fail-safe: if RM never reports "reached", release to MOBILE anyway.
    switch_lt_timeout_s: float = 4.0
    # ---- RT (MOBILE -> HOPPING), 2026-08-10: first "hop" = RT-STAND ----
    # P4 places the leg; after switch_rb_pushdelay_s enter RT-STAND:
    #   - tumbler lift+hold (length ramp to switch_rt_stand_leg_len_m,
    #     tau cap switch_rt_stand_tau_max_nm, AK60 kd switch_rt_stand_ak60_kd);
    #   - props = ModeE allocator around hop_prop_base_pwm_us (1600) for
    #     stance-like attitude balance (not a flat PWM clamp);
    #   - RM unfolds +11.5 -> 0 in parallel (synced drive);
    #   - 收腿/HOPPING when slowest RM arm within
    #     switch_rt_stand_rm_near_deg AND elapsed >= min_s (RM keeps
    #     driving to 0 / drive-timeout after the handoff).
    # 18:06 + 18:35 attempts: the 7/24 loaded-stand params (kp_z 4000,
    # FF 73 N, 10 Nm cap) launched the leg 0.45 -> 0.54 within 200 ms;
    # replayed over the 163642 demo hold they pin 7-10 Nm where the demo
    # logged <= 0.3 Nm (that demo hold WAS the P4 law).  Params below
    # retained only for rollback; the stand branch no longer reads the
    # kp/kd/FF ones.
    # 2026-08-10 18:58 "顶起长撑地": the stand now LIFTS the body like the
    # 163642 demo push (L 0.449 -> ~0.52) but statically.  Same P4 law
    # (kp_z 1200 default), target length raised to 0.55 which includes the
    # spring sag mg/kp ~= 4.6 cm -> equilibrium ~0.504 under full weight.
    # 9 Nm = the LT-stand calibrated torque wall (carries the whole robot).
    switch_rt_stand_tau_max_nm: float = 9.0
    switch_rt_stand_timeout_s: float = 4.0
    # Stand target leg length (m).  0.55 stays under the physical maximum
    # 0.554; the PD force vanishes as L approaches it, so it cannot pump
    # into a takeoff.
    switch_rt_stand_leg_len_m: float = 0.55
    # Minimum ground-support duration: HOPPING (收腿) starts only after the
    # slowest RM arm is within switch_rt_stand_rm_near_deg of target AND
    # this much time has passed (timeout still wins).  RM keeps driving
    # after the hop handoff until it hits 0 or rm_drive_timeout.
    switch_rt_stand_min_s: float = 2.0
    # Slowest-arm |q - target| threshold (deg) that allows 收腿 / HOPPING.
    switch_rt_stand_rm_near_deg: float = 10.0
    # Rate limit for the target length ramp 0.448 -> 0.55 (m/s).  A step
    # target kicked the body (18:06/18:35 logs); ~1 s ramp lifts smoothly.
    switch_rt_stand_extend_rate_mps: float = 0.1
    # AK60 motor-internal velocity damping during the stand (LT stand
    # holds the full body weight cleanly at 2.0).
    switch_rt_stand_ak60_kd: float = 2.0
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
    ak60_flight_damp_kd: float = 0.3
    # 2026-07-19 stance anti-jitter: small motor-side damping using the
    # driver's own high-rate velocity (much cleaner than the ~230 Hz LCM qd).
    # Restored to yesterday push (a12f736 / outdoor1): 0.1.
    ak60_stance_damp_kd: float = 0.1
    # 2026-08-11 (user): 第一跳 damp -- extra AK60 stance damping for
    # the FIRST hopping stance only (n_flights_done < 1: the landing right
    # after the mobile->hopping switch, arms still unfolding).  Normal hops
    # go back to ak60_stance_damp_kd.
    ak60_first_hop_stance_damp_kd: float = 0.2

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
    # Missing/stale endpoint feedback must not block HOPPING -> MOBILE forever.
    # Timeout stops RM current and clears the drive stage; it does not alter
    # encoder coordinates.
    rm_drive_timeout_s: float = 3.0
    # Station-keeping at the HOPPING endpoint (2026-07-23 user request):
    # while gait==hopping AND legs are PD/PWMPD (and no fold transition is
    # running), keep a continuous current-mode PD on 0 rad so hop impacts
    # cannot back-drive the folding arms. Thermal-safe by construction: at
    # q=0 the PD current is ~0 A (heat ~ i^2, only disturbance transients
    # draw current), and the cap is well under the M2006 continuous rating.
    rm_hopping_hold: bool = True
    # Same low-cap PD after HOPPING->MOBILE, but on +11.5 so wheel motion
    # cannot back-drive the folded arms. Same current cap as hopping.
    rm_mobile_hold: bool = True
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
    mobile_v_max_mps: float = 1.20
    mobile_wz_max_rad_s: float = 2.0
    # Output-shaft cap after the M2006 36:1 gearbox. 25 rad/s ~ 239 rpm
    # shaft / ~8600 rpm rotor -- brief peaks ok, sustained driving should
    # stay near continuous rating (drop mobile_v_max_mps if wheels run hot).
    wheel_speed_max_rad_s: float = 25.0
    # ---- MOBILE leg stow (after wheels start moving) ----
    # Two stages: 1) center (equal joints), 2) lift to the field pose the
    # user liked (lcm-spy 2026-08-10 22:14): q≈(0.248, -1.009, -0.623) →
    # foot ≈(-0.056, -0.156, +0.249).
    mobile_leg_q_des: tuple = (0.248, -1.009, -0.623)
    mobile_leg_center_q: float = 0.4       # stage-1: straight-down center
    mobile_leg_tau_max_nm: float = 3.0
    mobile_leg_kp: float = 20.0
    mobile_leg_kd_move: float = 0.0       # no outer D; AK60 handles damping
    mobile_leg_ak60_kd_move: float = 2.0
    mobile_leg_kp_hold: float = 16.0
    # Do NOT apply the noisy CAN qd in tau_ff during hold. Log 142441 showed
    # qd spikes of +/-10 rad/s, making the outer D bang against +/-1 Nm.
    mobile_leg_kd_tau_hold: float = 0.0
    # One damping path only: AK60's high-rate internal velocity feedback.
    mobile_leg_ak60_kd_hold: float = 2.0
    mobile_leg_arrive_rad: float = 0.10   # max |q_err| to advance stage / hold
    mobile_wheel_motion_rad_s: float = 0.8  # |omega_cmd| latch threshold
    # While holding the stow pose, right-stick X (same axis as wheel vy)
    # shifts the foot ±Y so the leg can track lateral wheel motion.
    mobile_leg_y_stick_m: float = 0.10
    mobile_leg_y_rate_mps: float = 0.12

    # ===== MANIPULATION: hold a fixed joint pose (push-box) ==============
    # 2026-08-13: MOBILE LB needs a fresh 60 mm box tag (id 2 or 3),
    # then holds this q (AK60 kd, no q_des slew). Wheels stay on the
    # sticks (fwd + left/right only, no yaw). LT is mobile only. RT is
    # MOBILE -> HOPPING. Button is RB (id 1).
    manip_hold_q: tuple = (1.376, 1.378, -0.096)
    manip_hold_kp: float = 20.0
    # 0 = snap q_des to manip_hold_q. Motion is damped by AK60 MIT kd
    # (manip_ak60_kd), not a joint-space rate limit.
    manip_hold_q_rate_rad_s: float = 0.0
    # LB double-click window kept for RB β presets (HOPPING only).
    # Box-hold is LB. Button auto-approach/press is RB (2026-08-14).
    # LT is hopping <-> mobile only.
    lb_double_click_s: float = 0.40
    # RB bumper: 1/2/3 clicks set deadbeat β_lo/β_hi (and turn the
    # closed-form placement on). Window matches LB.
    rb_beta_presets: tuple = (
        (0.11, 0.50),
        (0.50, 0.50),
        (0.75, 0.75),
    )
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
    # Soft stick-teleop sphere. Real deployed travel is work_* below
    # (user 2026-08-14). Two-joint-down hold q≈(1.38,1.38,-0.10) is
    # L=0.476 / r_xy=0.31 / tilt=41 deg — the old q-carve (L_qcap≈0.38)
    # rejected that pose and reported a fake 21 cm "clip".
    manip_leg_len_max_m: float = 0.58
    manip_q_max: float = 1.55
    manip_xy_max_m: float = 0.383
    manip_tilt_max_deg: float = 65.0
    # Manipulation is intentionally low-authority: move the Cartesian target
    # incrementally like stick teleop, never snap the leg with a stiff servo.
    # User 2026-08-08 23:10: 2 Nm still delivered <1 N at the foot
    # (Jacobian + clamp fighting out-of-WS goals). Raise cruise/press caps.
    manip_tau_max_nm: float = 5.0
    # Press/hold: higher torque + open-loop wall-normal force feedforward
    # (tag may be occluded by the hand — setpoint is latched, never live).
    manip_press_tau_max_nm: float = 12.0
    manip_press_ff_n: float = 45.0
    manip_kp_z_n_m: float = 900.0
    manip_kd_z_n_s_m: float = 18.0
    # AprilTag auto uses Cartesian targets/PD so its wall-normal stroke is not
    # projected onto that sphere (tools/button_apriltag_live.py).
    # Geometry: button = tag right 19.5 cm, down 5.6 cm, protrude 5 cm,
    # wheels press 5 cm into the wall (user 2026-08-14 02:27 +3cm R/D).
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
    # Hover standoff before FACE (along -n). Must be > arrive so FACE is
    # not skipped; last motion is then square into the button (user
    # 2026-08-14 正对按钮再按).
    manip_button_pre_m: float = 0.03
    # Press into the wall: wheels creep forward along +n; the foot holds
    # the FACE hover (user 2026-08-14 轮子往前按). Leg no longer stretches
    # the unreachable 5 cm stroke (log 01:42 / screenshot 02:22 z dump).
    manip_button_press_m: float = 0.05
    manip_button_press_v_mps: float = 0.04
    # Stage arrive / hold. Log 23:12: stuck on pre because clamp left a
    # residual > arrive — snap + looser gate (user 2026-08-08 23:13).
    manip_button_arrive_m: float = 0.025
    manip_button_hold_s: float = 0.5
    # PRE/FACE Cartesian target speed limit.
    manip_button_rate_mps: float = 2
    # Log 01:42: mobile 3 Nm sat 87% of PRESS, foot dumped to z=0.451
    # (floor) and buzzed 19 s. PRE/FACE/PRESS use this cap; 归中 stays
    # on mobile_leg_tau_max_nm (gentler near the floor).
    manip_button_tau_max_nm: float = 8.0
    # Give up a stage that cannot close the last centimetres (envelope
    # residual / joint stop) instead of grinding. Crawl needs more than
    # the old 1.2 s snap timeout.
    manip_button_stage_timeout_s: float = 4.0
    # Cartesian PD for the button auto path.
    manip_button_kp_n_m: float = 240.0
    manip_button_press_kp_n_m: float = 240.0
    manip_button_kd_n_s_m: float = 4.0
    manip_ak60_kd: float = 2.0
    # MOBILE -> MANIPULATION first does 归中: equal joints at
    # mobile_leg_center_q (same as MOBILE stow step 1). Then PRE/FACE/
    # PRESS on the hold-pose XY plane.
    manip_home_q: float = 0.4
    manip_home_arrive_m: float = 0.03
    # After press/hold: stay in MANIPULATION at the press pose. No retract,
    # no wheel reverse (user 2026-08-14 按完就停).
    manip_backup_enable: bool = False
    manip_backup_m: float = 0.30
    manip_backup_v_mps: float = 0.20

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
    # Legacy radial band remains dashboard-only. Actual handoff below uses
    # PRE XY error + workspace clip + wall yaw.
    mobile_ready_r_min_m: float = 0.50
    mobile_ready_r_max_m: float = 0.53
    mobile_ready_lat_m: float = 0.05
    mobile_ready_xy_m: float = 0.03
    # Current operator-approved pose leaves FACE about 3 cm beyond Lmax;
    # the arm reaches its envelope and wheels close the final gap.
    mobile_ready_clip_m: float = 0.04
    # Approach speed (user 2026-08-14 对齐和前进太快).
    mobile_approach_kp_v: float = 0.70         # m/s per m of XY error
    mobile_approach_kp_wz: float = 0.90         # rad/s per rad of yaw error
    mobile_approach_v_max_mps: float = 0.22
    mobile_approach_wz_max_rad_s: float = 0.40
    # Keep driving on the remembered (dead-reckoned) tag pose this long
    # after the detector drops out; then stop and wait for re-capture.
    approach_memory_timeout_s: float = 3.0
    # Tiered slow-down on detection dropouts ("step and see"): full speed
    # only while packets are fresh, half speed when the last sample is
    # older than fresh_s, crawl (translation only, yaw frozen) while
    # coasting on the dead-reckoned memory past manip_button_stale_s.
    mobile_approach_fresh_s: float = 0.4
    mobile_approach_slow_scale: float = 0.7
    mobile_approach_memory_scale: float = 0.4
    # Radial FOV handoff (||pre||_L): FAR centers TAG; NEAR parks TAG
    # only a little left of image center (tag_left_deg), not hard button
    # center (log 23:00: bear_tag=-17 deg was too much).
    mobile_approach_blend_near_m: float = 0.55
    mobile_approach_blend_far_m: float = 0.85
    mobile_approach_tag_left_deg: float = 6.0
    mobile_approach_bear_gate_deg: float = 18.0
    # Camera-to-wall yaw (0 = square / parallel). Near approach and READY
    # both require this; depth-plane normal is preferred over tag PnP.
    mobile_approach_yaw_ok_deg: float = 8.0
    mobile_approach_yaw_stop_deg: float = 22.0
    # Far-fast / near-slow on ||pre||_L (translation only).
    mobile_approach_v_far_m: float = 0.90
    mobile_approach_v_near_m: float = 0.50
    mobile_approach_v_near_scale: float = 0.30
    # Approach corridor: forward advance is throttled to zero until the
    # lateral (wall-parallel) error is inside this gate -- translate to
    # the button's normal line first, then drive in square (2026-08-08).
    mobile_approach_lat_gate_m: float = 0.12
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

    # ===== PUSH: box face tags (print box_apriltags_id2_id3_60mm_a4.pdf)
    # Facing the box: LEFT=ID2, RIGHT=ID3. Black square 60 mm, gap 60 mm
    # -> center-to-center 120 mm. Perception must use the same size.
    # True  = simplified: snap manip_hold_q + stick wheels (no auto-align).
    # False = paper: auto square/center/distance, then snap the same q
    #         and LEFT stick commands the box (fwd/steer + yaw align).
    push_simple: bool = True
    push_enable: bool = True
    push_box_tag_ids: tuple = (2, 3)
    push_box_tag_size_m: float = 0.060
    push_box_stale_s: float = 1.0              # box packet freshness gate
    push_contact_depth_m: float = 0.02         # press into the face (m)
    push_e_max_m: float = 0.12                 # contact offset cap (m)
    # (2026-08-13 log 225152, user: 甩尾不够 左转接右转慢) 0.08 meant a
    # full left->right e reversal took 3 s at full stick; 0.20 -> 1.2 s.
    push_e_rate_mps: float = 0.20              # stick -> contact offset rate
    push_v_max_mps: float = 0.25               # max push advance speed
    # Face-squaring servo: measured p95 yaw rate was only 18 deg/s with
    # kp 1.5 / cap 0.5 -- too soft to follow the rotating face. Same log.
    # 0.9 lost the tags at 15 fps color even after the 4 ms exposure cap
    # (2026-08-13 太快丢视野) -> 0.7 as the camera-limited compromise.
    push_wz_max_rad_s: float = 0.7
    push_kp_wz: float = 2.5                    # face-normal yaw alignment
    push_kp_vy: float = 0.8                    # workspace re-centering gain
    # (2026-08-13) Motion-cone wz cap REMOVED from the face-squaring servo:
    # it froze the yaw correction at stick release (log 224304, yaw_err
    # +69 deg, tags lost). Kept for reference/telemetry only.
    push_kappa_max_1_m: float = 1.5
    # ---- Contact-force feedback (2026-08-13 user: 甩尾时丢箱子, 加力反馈
    # 一起推动和摆姿态). F = J^-T tau (foot->environment, leg frame),
    # decomposed on the face normal (F_n) and face_right (F_t).
    # 1) Normal-force keeper: admittance on the press depth -- F_n sagging
    #    below push_f_des_n (face rotating away mid-swing) presses deeper;
    #    a spike backs off. Nominal: kp 200 N/m x 0.02 m = 4 N.
    push_f_des_n: float = 5.0
    push_f_adapt_mps_per_n: float = 0.003      # depth rate per N of F error
    push_depth_min_m: float = 0.005
    push_depth_max_m: float = 0.06
    push_f_lpf_tau_s: float = 0.15             # J^-T tau estimate is noisy
    # 2) Friction-cone slip guard: |F_t|/F_n above this ratio means the
    #    foot is about to slide off the face -> freeze further |e| growth
    #    (force-based motion cone; replaces the removed velocity cone).
    push_slip_ratio_max: float = 0.8
    # Wheel twist accel limits (2026-08-13 user: 轮子瞬时太快画面糊).
    # 0.5 m/s^2 -> full 0.25 m/s in 0.5 s; 2.0 rad/s^2 -> wz_max 0.9 in
    # 0.45 s. Smooths starts/stops/swings without dulling the response.
    push_acc_max_mps2: float = 0.5
    push_wz_acc_max_rad_s2: float = 2.0
    # PARALLEL FIRST (2026-08-13 user: 第一前提是平行, 不然箱子打到轮子).
    # Below yaw_ok the push runs at full speed; the advance fades linearly
    # and STOPS at yaw_stop, and |e| growth is frozen above yaw_ok, until
    # the tail swing re-squares the camera to the face.
    push_yaw_ok_deg: float = 8.0
    push_yaw_stop_deg: float = 18.0
    # Leg force control in PUSH deliberately has NO independent gains: it
    # reuses the wall-button press set (manip_button_kp_n_m /
    # manip_button_kd_n_s_m / manip_button_rate_mps / manip_press_tau_max_nm
    # / manip_press_ff_n) so the paper reports ONE contact-force law
    # (user 2026-08-09: "所有力和手臂摆动的速度参数都和按钮一样").
    # MOBILE box auto-approach (LT with a fresh box face, no button tag):
    # drive until the camera is square to the face (yaw), laterally centered
    # on the tag pair, at the working distance. READY needs the errors to
    # stay inside the band for push_ready_hold_s; a second LT enters PUSH.
    push_approach_dist_m: float = 0.40         # default working distance (m)
    push_ready_dist_tol_m: float = 0.04        # READY: |dist err| < 4 cm
    push_ready_yaw_deg: float = 6.0            # READY: face-normal yaw err
    push_ready_hold_s: float = 0.5             # dwell inside band -> READY
    # Exit hysteresis (log 19:43: READY then yaw jumped 0→15° from tag
    # noise and immediately re-servo'd). Once READY, only leave after
    # sitting outside the WIDER exit band for push_ready_exit_hold_s.
    push_ready_exit_dist_tol_m: float = 0.08
    push_ready_exit_yaw_deg: float = 18.0
    push_ready_exit_hold_s: float = 0.6
    # PUSH is open-loop contact (2026-08-09 user): no force closed loop.
    # Optional F_meas is log-only. Foot target is LPF'd so tag pose noise
    # does not chatter the Cartesian PD.
    push_target_lpf_tau_s: float = 0.25
    # Soft contact law (not the button-press 30 N / 8 Nm set — that
    # fought the face and jittered). Keep the same posture geometry.
    push_kp_n_m: float = 200.0
    push_kd_n_s_m: float = 4.0
    push_tau_max_nm: float = 3.0
    # Entry is a SNAP onto the face contact (2026-08-13: 摆腿太慢 with
    # 0.08 + HOME crawl); this rate only shapes the stick-driven e moves.
    push_foot_rate_mps: float = 0.30
    # PUSH foot moves ONLY in the horizontal plane z = FK(manip_hold_q).z
    # (user 2026-08-13). Delta-leg workspace at that height (0.362 m) ends
    # at xy ~= 0.31 m in every direction (sampled), and the hold pose sits
    # at xy = 0.3095 -- i.e. on the boundary. Targets past this saturate
    # and the wheel-strafe servo carries the base instead.
    push_xy_max_m: float = 0.31

    # ===== Deployed-leg WORK ENVELOPE (2026-08-14, measured on robot) =====
    # User-defined working range for every deployed-leg Cartesian target
    # (button / push / manip teleop). In the leg FK frame (+z DOWN):
    #   TOP     z >= 0.128 m — two-q-down (q0=q1=1.4 unchanged) with
    #           the third joint at the software limit q2=-1.2
    #           (FK xyz≈+0.320/-0.185/+0.128, L=0.391, r=0.37).
    #           Previous 0.176 was log 01:36 (q2≈-0.8); the foot can
    #           still lift until that third joint hits the zero-side
    #           stop. Homing FK(0).z=0.354 is the INNER length bound,
    #           not a z-floor.
    #   BOTTOM  z <= 0.440 m (foot at wheel-contact floor level),
    #   FLOOR   r_xy < 0.10 m at BOTTOM hits the floor (user 2026-08-23).
    #           Inside that disk z is lifted toward work_len_min; z_bottom
    #           is only allowed for r >= work_xy_floor_r_m.
    #   REACH   work_len_min <= L <= 0.50 m.  work_len_min is NOT 归中
    #           (FK(0)=0.354): the three keep-out ducts sit at
    #           L=hypot(0.107, 0.320)=0.337, so a clamp at 归中 can only
    #           skate their outer lip.  0.330 matches LEN_FOLD_MIN_M in
    #           tools/workspace_envelope.py; joints retract to L=0.247.
    #   SWING   horizontal r <= 0.383 m (FK(1.55,-0.97,1.55), at z=0.162).
    #   MINUS three wheel BARRELS, 120 deg symmetric about the origin
    #   (user: 左边右边空间一样). Keep-outs live in the LEG/L frame:
    #   bottom on x=0 (az=270), then +120/+240 (30 and 150).
    #   Independent of drive-frame wheel_azimuth_deg (kiwi IK).
    #   The barrel is 3D (user 2026-08-23): at z_bottom it is just the wheel
    #   disk, it opens into an ellipse at work_keep_z_wide_m and closes again
    #   symmetrically above.  The ellipse was sized from a hand-posed foot
    #   that just clears the wheel: centre = its perpendicular foot on the
    #   ray through the wheel, b = its distance to that ray, a reaches the
    #   inner edge of the wheel disk.  Because a grows exactly as fast as the
    #   centre moves out, the inner face of all three barrels is one straight
    #   cylinder at r = 0.115 m, the duct the low rings weave through.
    #   work_keep_b_wide_m sets the barrel's AZIMUTHAL width, so it alone
    #   decides how far round a ring gets; 2026-08-24 it came down from 0.150
    #   to 0.118 after a hand pose 31 mm clear of the real wheel was called
    #   blocked.  tools/workspace_envelope.py mirrors these numbers.
    work_z_top_m: float = 0.128
    work_z_bottom_m: float = 0.440
    work_len_min_m: float = 0.330
    work_len_home_m: float = 0.354   # FK(0,0,0) 归中; floor-cone apex
    work_len_max_m: float = 0.50
    work_xy_max_m: float = 0.383
    work_xy_floor_r_m: float = 0.10
    work_wheel_center_m: float = 0.17
    work_wheel_keep_m: float = 0.055
    work_wheel_azimuth_deg: tuple = (270.0, 30.0, 150.0)
    work_keep_z_wide_m: float = 0.267
    work_keep_r_wide_m: float = 0.229
    work_keep_a_wide_m: float = 0.114
    work_keep_b_wide_m: float = 0.118
    work_keep_margin_m: float = 0.008


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

        # Desired velocity command smoothing (rate limiter)
        self._v_cmd_filt = np.zeros(2, dtype=float)
        # LEFT-stick hop-height detent: "", "up", "down", "left", "right".
        # Height is applied once when the stick returns to neutral.
        self._hop_height_stick_pending: str = ""

        # PWM filter (same as Hopper4 for propeller stability)
        # Low-pass filter: smoothed_pwm = alpha * new_pwm + (1-alpha) * prev_pwm
        # User request: no PWM smoothing; send prop commands immediately.
        self.pwm_filter_alpha = 1.0
        self.prev_pwm_us = np.zeros(6, dtype=float) + float(self.modee_cfg.pwm_min_us)
        # Timestamp of last prop PWM publish, for the soft-start (rise-rate) limiter.
        self._last_pwm_pub_t: float | None = None
        self._last_y = False
        self._last_b = False
        self._last_lb_log = False
        self._last_rb_log = False
        # User "hard stop" latch (maps to gamepad `point` button; user calls it "I").
        self._zero_vel_hold = False
        self._last_point = False
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
        self._mode_last_lb_bumper: bool = False
        self._lb_click_pending: bool = False
        self._lb_click_t: float = 0.0
        self._mode_last_rb_bumper: bool = False
        self._rb_click_pending: bool = False
        self._rb_click_n: int = 0
        self._rb_click_t: float = 0.0
        # "hold" = push-box q-hold; "button" = AprilTag press sequence.
        self._manip_kind: str = "hold"
        # RB/RT phase-4 start time (fixed stand, then hop entry).
        self._rb_p4_t0: float | None = None
        # RT/P4 starts by retracting to switch_rb_align_leg_len_m while
        # returning the foot to world vertical. It then advances to the
        # switch_rb_leg_len_m final hold.
        self._rb_p4_aligning: bool = False
        # Prop-base window (collective 1700 + PWM floor 1600): open at
        # MOBILE->HOPPING handoff; close on first real hop LO (after a TD)
        # or hop_prop_base_timeout_s -- whichever first.
        self._hop_prop_base_active: bool = False
        self._hop_prop_base_t0: float | None = None
        self._hop_prop_base_saw_td: bool = False
        self._hop_prop_base_liftoff_count: int = 0  # legacy counter (unused)
        # Explicit mechanical configuration. HOPPING is the backward-compatible
        # startup state. MOBILE and MANIPULATION share the deployed appendage
        # geometry; only their active actuator changes (wheels vs leg).
        self._gait_mode: str = "hopping"
        # MOBILE leg stow: idle -> center -> lift -> hold.
        # Latched by wheel motion; cleared when leaving MOBILE.
        self._mobile_leg_active: bool = False
        self._mobile_leg_holding: bool = False
        self._mobile_leg_phase: str = "idle"   # idle|center|lift|hold
        self._mobile_leg_kd_cmd: float = 0.0
        self._mobile_leg_y_off_m: float = 0.0  # stick lateral while holding
        self._box_ready_exit_since: float | None = None
        # PUSH leg stages: home (center) -> reach (to face) -> track.
        self._push_leg_phase: str = "home"
        self._wheel_pending_cmd = np.zeros(3, dtype=float)
        self._wheel_pending_enable: bool = False
        self._manip_init_pending: bool = False
        self._manip_hold_q_cmd: np.ndarray | None = None
        self._manip_foot_des_b = np.array(
            [0.0, 0.0, float(self.lcm_cfg.switch_rb_leg_len_m)], dtype=float
        )
        self._manip_err_m: float = float("nan")
        # True during button press/hold — use higher tau/kp into the wall.
        self._manip_press_boost: bool = False
        self._manip_press_n_L: np.ndarray | None = None
        self._manip_speed_mps: float = float("nan")
        # AprilTag button auto-press. Tag acquisition happens only in MOBILE;
        # MANIPULATION consumes one frozen READY latch.
        # home|pre|face|press|hold|done|latch_missing
        self._btn_stage: str = "idle"
        self._btn_hold_t0: float | None = None
        self._btn_stage_t0: float | None = None
        self._btn_stage_seen: str = ""
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
            jprint(f"[MANIP] button UDP unavailable: {exc}; using local JSON")
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
        self._approach_kind: str = "button"        # "button" | "box"
        self._approach_sp: dict | None = None      # {"pre","tag","n_in"} DRIVE
        self._approach_dbg_t: float = 0.0
        # Post-press open-loop wheel reverse (drive frame unit dir, meters left).
        self._backup_active: bool = False
        self._backup_dir_D: tuple[float, float] = (-1.0, 0.0)
        self._backup_remain_m: float = 0.0
        self._backup_dbg_t: float = 0.0
        # FACE-then-wheel-press: open-loop creep along +n (drive frame).
        self._press_drive_active: bool = False
        self._press_dir_D: tuple[float, float] = (1.0, 0.0)
        self._press_drive_remain_m: float = 0.0
        self._press_drive_t0: float = 0.0
        self._press_drive_dbg_t: float = 0.0
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
            jprint(f"[init] drive yaw in L from T_L_C: "
                  f"{math.degrees(dy):+.1f} deg")
        except Exception as e:
            jprint(f"[init] T_L_C unavailable ({e}); drive yaw fallback "
                  f"{math.degrees(dy):+.1f} deg")
        self._l2d_cs: tuple = (math.cos(dy), math.sin(dy))
        # Work envelope.  L_min is work_len_min_m, NOT the homing length:
        # 归中 (FK(0)=0.354) cannot fold into the r=0.107 keep-out ducts at
        # L=0.337, so clamping there locks the foot out of them.
        try:
            f0, _ = self.core.fk.forward_kinematics(np.zeros(3))
            f0 = np.asarray(f0, dtype=float).reshape(3)
            l_home = float(np.linalg.norm(f0))
            self.lcm_cfg.work_len_home_m = l_home
            jprint(
                "[init] work envelope: z "
                f"{float(self.lcm_cfg.work_z_top_m):.3f}..{float(self.lcm_cfg.work_z_bottom_m):.3f} m, "
                f"L={float(self.lcm_cfg.work_len_min_m):.3f}..{float(self.lcm_cfg.work_len_max_m):.2f} m "
                f"(归中 {l_home:.3f}), "
                f"swing r<={float(self.lcm_cfg.work_xy_max_m):.3f} m "
                f"(fwd-tilt), floor-keep r<"
                f"{float(self.lcm_cfg.work_xy_floor_r_m):.2f} m at BOTTOM, "
                f"3 wheel barrels 120deg at r="
                f"{float(self.lcm_cfg.work_wheel_center_m):.2f} m: disk "
                f"{float(self.lcm_cfg.work_wheel_keep_m):.3f} m at the floor, "
                f"widest at z={float(self.lcm_cfg.work_keep_z_wide_m):.3f} m "
                f"(a={float(self.lcm_cfg.work_keep_a_wide_m):.3f} "
                f"b={float(self.lcm_cfg.work_keep_b_wide_m):.3f} "
                f"+{float(self.lcm_cfg.work_keep_margin_m) * 1e3:.0f} mm)"
            )
        except Exception as exc:
            jprint(f"[init] work envelope banner unavailable ({exc})")
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
        # Box auto-approach + working-distance latch (process-persistent:
        # the FIRST manual LT -> PUSH stores the measured face distance and
        # every later approach targets that same "现在这个状态" pose).
        self._push_work_dist_m: float | None = None
        self._box_ready: bool = False
        self._box_ready_since: float | None = None
        self._box_dist_err_m: float = float("nan")
        self._box_yaw_err_deg: float = float("nan")
        # PUSH contact-force feedback: raw F_n for CSV + LPF'd normal /
        # tangential components driving the depth admittance + slip guard.
        self._push_f_meas_n: float = float("nan")
        self._push_f_n_lpf: float = float("nan")
        self._push_f_t_lpf: float = float("nan")
        self._push_depth_cmd_m: float = 0.02
        self._push_tgt_lpf: np.ndarray | None = None
        # Foot z (FK frame) locked to FK(manip_hold_q) for the whole PUSH:
        # the leg moves in that horizontal plane only (user 2026-08-13).
        self._push_z_hold_m: float | None = None
        # Button auto: same horizontal plane as simplified push / hold pose.
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
        self._rm_drive_t0: float | None = None
        # Start poses latched on the first stage-2 tick; the synchronized
        # drive measures per-arm progress against these (see rm_sync_lead_rad).
        self._rm_sync_q0: np.ndarray | None = None
        self._rm_target: float = 0.0
        self._rm_hold_t0: float = 0.0
        # LT armed flag: set at the LT press, consumed at the next compression
        # (touchdown) where RM fold starts.
        self._rm_lt_pending: bool = False
        # After RM fold armed at compression: wait for THIS stance's liftoff
        # to turn props off and enter the stand hold (legs finish the cycle).
        self._lt_await_lo_stand: bool = False
        # LT stand-and-fold: wall time the stand began (None = inactive).
        # Entered on the liftoff after the compression that started RM fold.
        self._lt_stand_t0: float | None = None
        # RT stand-and-unfold: wall time the uncapped stand began (None =
        # inactive).  Entered when P4 finishes its pushdelay.
        self._rt_stand_t0: float | None = None
        # RT armed flag: set when P4 hands off to HOPPING; RM +11.5->0 starts
        # only at the next liftoff (push end), not during the push itself.
        self._rm_rt_pending: bool = False
        # Launch-time height / l0. Each MOBILE->HOPPING session restores
        # these so last hopping's stick height or leftover override cannot
        # leak into the next hopping.
        self._hop_height_launch_m: float = float(self.core.cfg.hop_height_m)
        self._foot_clearance_launch_m: float = float(
            self.core.cfg.foot_clearance_target_m
        )
        self._leg_l0_launch_m: float = float(self.core.cfg.leg_l0_m)
        # Temporary hop_height override across gait switches. Saved value is
        # restored after RT's first N hops or when LT stand begins / B abort.
        self._hop_height_restore: float | None = None
        self._rt_hop_height_remaining: int = 0
        # RT first-hop l0 override: original leg_l0_m saved here while the
        # first post-RT stance runs with switch_rb_first_hop_l0_m; restored
        # at that liftoff (or B abort).
        self._rt_l0_restore: float | None = None
        # Do NOT restore on the fake LO right after P4 handoff (still
        # planted → stance→flight edge). Wait for a real TD first.
        self._rt_first_hop_l0_saw_td: bool = False
        # Saved (prop_base_thrust_ratio, prop_stance_base_thrust_ratio) while
        # the RT prop-base window overrides them (see hop_prop_base_pwm_us).
        self._prop_ratio_restore: tuple | None = None
        # Logical-position initialization pulse. On the edge, the driver labels
        # the current physical RM pose as _rm_zero_at_rad without moving it.
        self._rm_zero_until: float = 0.0
        self._rm_zero_at_rad: float = 0.0

        # Patrol engage flag (stick/B disengages). SELECT (Back) is owned by
        # services/gamepad_upper_toggle.py: 1x start / 2x stop hopper-upper.
        self._patrol_enable: bool = False

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
        lb_bumper_now = bool(getattr(gamepad_msg, "leftBumper", 0)) if gamepad_msg is not None else False
        rb_bumper_now = bool(getattr(gamepad_msg, "rightBumper", 0)) if gamepad_msg is not None else False

        enabled = int(self._mode_est) in (PD, PWMPD)

        # LB bumper: simplified push hold needs only PD + MOBILE.
        # Box tags are required only by the non-simple auto-align mode.
        # While already in hold/button, LB exits back to MOBILE.
        self._flush_lb_click()
        if bool(lb_bumper_now) and (not bool(self._mode_last_lb_bumper)):
            if enabled and self._gait_mode in ("manipulation", "push"):
                self._lb_click_pending = False
                self._leave_deployed_to_mobile("LB")
            elif enabled and self._gait_mode == "mobile":
                self._try_enter_push_hold("LB")
            elif not enabled:
                jprint("[gait] LB ignored (press X for PD first, then LB)")
            else:
                jprint("[gait] LB ignored (need PD + MOBILE)")
        self._mode_last_lb_bumper = bool(lb_bumper_now)

        # RB bumper:
        #   MOBILE / MANIPULATION -> auto-approach then press the wall button
        #     (the sequence that previously worked).
        #   HOPPING -> 1/2/3 clicks set β_lo/β_hi (closed-form placement).
        self._flush_rb_click()
        if bool(rb_bumper_now) and (not bool(self._mode_last_rb_bumper)):
            if enabled and self._gait_mode in ("mobile", "manipulation"):
                self._rb_click_pending = False
                self._lb_start_button("RB")
            elif self._gait_mode == "hopping":
                now_m = time.monotonic()
                win = float(getattr(self.lcm_cfg, "lb_double_click_s", 0.40))
                if bool(self._rb_click_pending) and (
                    now_m - float(self._rb_click_t)
                ) <= win:
                    self._rb_click_n = min(3, int(self._rb_click_n) + 1)
                    self._rb_click_t = now_m
                    jprint(f"[gait] RB: click {int(self._rb_click_n)}")
                else:
                    self._rb_click_pending = True
                    self._rb_click_n = 1
                    self._rb_click_t = now_m
                    jprint("[gait] RB: click 1 (β preset if no more clicks)")
            elif not enabled:
                jprint("[gait] RB ignored (press X for PD first, then RB)")
            else:
                jprint("[gait] RB ignored (need PD + MOBILE for button)")
        self._mode_last_rb_bumper = bool(rb_bumper_now)

        # LT is mobile only: select MOBILE while disabled; from HOPPING
        # request stand/fold; from hold/button leave to MOBILE. Never
        # starts push-box (that is LB).
        if bool(lt_now) and (not bool(self._mode_last_lb)):
            if not enabled:
                self._gait_mode = "mobile"
                self._reset_hop_estimator("LT disabled -> MOBILE")
                self._manip_init_pending = False
                self._manip_hold_q_cmd = None
                self._mobile_tag_samples = []
                self._mobile_tag_ready_sp = None
                self._mobile_tag_state = "searching"
                self._reset_mobile_leg_stow()
                self._switch_loop = False
                self._rm_lt_pending = False
                self._rm_rt_pending = False
                self._lt_await_lo_stand = False
                self._lt_stand_t0 = None
                self._rm_stage = 0
                self._rm_iq_des[:] = 0.0
                self._rm_set_logical_position(
                    float(self.lcm_cfg.rm_mobile_rad),
                    "LT disabled: select MOBILE",
                )
            elif self._gait_mode in ("manipulation", "push"):
                self._leave_deployed_to_mobile("LT")
            elif self._gait_mode == "mobile":
                jprint("[gait] LT ignored (already MOBILE; LB = push-box)")
            elif self._gait_mode != "hopping":
                jprint("[gait] LT ignored (deployed transition active)")
            elif (bool(self._rm_lt_pending)
                    or bool(self._lt_await_lo_stand)
                    or self._lt_stand_t0 is not None):
                jprint("[gait] LT ignored (fold transition already pending/active)")
            else:
                # Abort any in-progress RM unfold/drive so LT can arm on
                # the next compression (touchdown).
                if int(self._rm_stage) != 0 or bool(self._rm_rt_pending):
                    self._rm_stage = 0
                    self._rm_rt_pending = False
                    self._rm_drive_t0 = None
                    self._rm_sync_q0 = None
                    self._rm_iq_des[:] = 0.0
                    jprint("[rm] abort drive so LT can arm on compression")
                self._rm_lt_pending = True
                self._lt_await_lo_stand = False
                self._override_hop_height(
                    float(self.lcm_cfg.switch_lt_hop_height_m),
                    "LT transition hop",
                )
                # LT owns the temporary height for the transition push;
                # cancel any remaining RT first-hop countdown.
                self._rt_hop_height_remaining = 0
                jprint(
                    "[gait] LT ARMED -> hop_height=%.3f m; on next "
                    "compression: fold RM 0 -> %.1f; legs finish that "
                    "hop; at liftoff: props OFF + hold leg %.3f m, "
                    "then MOBILE"
                    % (float(self.lcm_cfg.switch_lt_hop_height_m),
                       float(self.lcm_cfg.rm_mobile_rad),
                       float(self.lcm_cfg.switch_lt_leg_len_m))
                )
        self._mode_last_lb = bool(lt_now)

        if bool(rt_now) and (not bool(self._mode_last_rb)):
            # RT selects HOPPING while disabled. While PD is on, MOBILE
            # RT starts P4 then the MOBILE -> HOPPING transition.
            if not enabled:
                self._rt_stand_t0 = None
                self._gait_mode = "hopping"
                self._reset_hop_estimator(
                    "RT disabled -> HOPPING", new_session=True
                )
                self._reset_mobile_leg_stow()
                self._switch_loop = False
                self._rm_lt_pending = False
                self._rm_rt_pending = False
                self._lt_await_lo_stand = False
                self._rm_stage = 0
                self._rm_iq_des[:] = 0.0
                self._rm_set_logical_position(
                    float(self.lcm_cfg.rm_hopping_rad),
                    "RT disabled: select HOPPING",
                )
            elif self._gait_mode != "mobile":
                jprint(
                    "[gait] RT ignored (leave MANIPULATION with LT first, "
                    "or already HOPPING)"
                )
            elif bool(self._switch_loop):
                jprint("[switch_loop] RT ignored (P4 stand already active)")
            else:
                self._approach_stop("RT -> hopping")
                self._switch_loop = True
                self._rb_p4_t0 = time.time()
                self._rb_p4_aligning = True
                self._reset_mobile_leg_stow()
                self._rm_lt_pending = False
                self._rm_rt_pending = False
                self._lt_await_lo_stand = False
                self._rm_stage = 0
                self._rm_iq_des[:] = 0.0
                self._prop_enable = True
                jprint(
                    "[switch_loop] RT MOBILE -> P4: align L=%.3fm, then "
                    "extend/hold L=%.3fm (cap %.1fNm) + props %.0fus for "
                    "%.1fs, then enter FLIGHT/HOPPING (RM unfold + prop "
                    "base 1600; normal hopping from there)"
                    % (
                        float(self.lcm_cfg.switch_rb_align_leg_len_m),
                        float(self.lcm_cfg.switch_rb_leg_len_m),
                        float(self.lcm_cfg.switch_rb_tau_max_nm),
                        float(self.lcm_cfg.switch_rb_prop_base_pwm_us),
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
                jprint("[backup] OFF (B)")
            if bool(self._press_drive_active):
                self._press_drive_active = False
                self._press_drive_remain_m = 0.0
                jprint("[manip-btn] wheel press OFF (B)")
            if (
                self._gait_mode == "manipulation"
                and str(getattr(self, "_manip_kind", "")) == "button"
            ):
                self._btn_stage = "aborted"
                self._btn_stage_print = "aborted"
                self._manip_press_boost = False
                jprint("[manip-btn] sequence ABORTED (B); no later press")
            if self._gait_mode == "push":
                # PUSH is an active contact task: B drops back to MOBILE so
                # a later X re-arm cannot resume pushing unexpectedly.
                self._gait_mode = "mobile"
                self._push_box = None
                self._push_e_des_m = 0.0
                self._manip_hold_q_cmd = None
                self._reset_push_contact_state()
                jprint("[gait] B: leave PUSH -> MOBILE")
            self._rm_stage = 0
            self._rm_lt_pending = False
            self._rm_rt_pending = False
            self._lt_await_lo_stand = False
            self._lt_stand_t0 = None
            self._rt_stand_t0 = None
            self._switch_loop = False
            self._rb_p4_t0 = None
            self._rb_p4_aligning = False
            self._reset_mobile_leg_stow()
            self._lb_click_pending = False
            self._rb_click_pending = False
            self._restore_hop_height("B abort")
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
            jprint("[prop] OFF (B) -> legs DAMP + props stop (control_mode=%d)"
                  % int(self.lcm_cfg.prop_ctrl_mode_off))
        elif bool(x_now) and (not bool(self._mode_last_x)) and (int(self._mode_est) != PD):
            self._mode_est = PD
            self._prop_enable = False
        elif bool(a_now) and (not bool(self._mode_last_a)):
            if int(self._mode_est) == PD:
                self._mode_est = PWMPD
                self._prop_enable = True
                jprint(
                    "[prop] ON (A): PD -> PWMPD (control_mode=%d)"
                    % int(self.lcm_cfg.prop_ctrl_mode_on)
                )
            elif int(self._mode_est) == PWMPD:
                self._mode_est = PD
                self._prop_enable = False
                jprint("[prop] OFF (A): PWMPD -> PD")
            else:
                self._prop_enable = False
                jprint("[prop] A ignored: press X first")
        elif bool(start_now) and (not bool(self._mode_last_start)) and (int(self._mode_est) == DAMP):
            self._mode_est = OFF

        self._mode_last_b = bool(b_now)
        self._mode_last_x = bool(x_now)
        self._mode_last_a = bool(a_now)
        self._mode_last_start = bool(start_now)

    def _override_hop_height(self, h_m: float, reason: str) -> None:
        """Temporarily force the active height knob; save prior once.

        With foot_clearance_control_enable, the active knob is
        foot_clearance_target_m; otherwise hop_height_m.
        """
        h_new = float(max(0.0, float(h_m)))
        use_clear = bool(getattr(
            self.core.cfg, "foot_clearance_control_enable", False
        ))
        if use_clear:
            if self._hop_height_restore is None:
                self._hop_height_restore = float(
                    self.core.cfg.foot_clearance_target_m
                )
            h_old = float(self.core.cfg.foot_clearance_target_m)
            self.core.cfg.foot_clearance_target_m = h_new
            jprint(
                "[foot_clearance] OVERRIDE (%s): %.3f -> %.3f m "
                "(restore %.3f)"
                % (reason, h_old, h_new, float(self._hop_height_restore))
            )
        else:
            if self._hop_height_restore is None:
                self._hop_height_restore = float(self.core.cfg.hop_height_m)
            h_old = float(self.core.cfg.hop_height_m)
            self.core.cfg.hop_height_m = h_new
            jprint(
                "[hop_height] OVERRIDE (%s): %.3f -> %.3f m (restore %.3f)"
                % (reason, h_old, h_new, float(self._hop_height_restore))
            )

    def _rm_drive_busy(self) -> bool:
        """True while RM is mid fold/unfold (stage 2).

        Stage 3 is the 1 s endpoint insurance hold. After HOPPING->MOBILE
        that hold may overlap the first wheel command -- that is intended.
        Only the long drive must keep the wheels off.
        """
        return int(self._rm_stage) == 2

    def _begin_new_hopping_session(self, reason: str) -> None:
        """Restore launch knobs so this hopping does not inherit the last.

        Drops leftover RT/LT height and l0 overrides and writes the
        launch-time hop_height / foot_clearance / l0 back onto cfg.
        Stick-changed height from the previous hopping is discarded.
        """
        self._hop_height_stick_pending = ""
        self._rt_hop_height_remaining = 0
        self._hop_height_restore = None
        self._rt_first_hop_l0_saw_td = False
        if self._rt_l0_restore is not None:
            self.core.cfg.leg_l0_m = float(self._leg_l0_launch_m)
            self._rt_l0_restore = None
        self.core.cfg.hop_height_m = float(self._hop_height_launch_m)
        self.core.cfg.foot_clearance_target_m = float(
            self._foot_clearance_launch_m
        )
        jprint(
            "[gait] new hopping session (%s): clearance=%.3f hop_height=%.3f "
            "l0=%.3f (launch defaults; last session height/vel discarded)"
            % (
                reason,
                float(self.core.cfg.foot_clearance_target_m),
                float(self.core.cfg.hop_height_m),
                float(self.core.cfg.leg_l0_m),
            )
        )

    def _reset_hop_estimator(
        self,
        reason: str,
        *,
        first_lo_zero_xy: bool = False,
        new_session: bool = False,
    ) -> None:
        """Zero hop-cycle velocity (v_hat, flight latch, last-N rings).

        Called when leaving HOPPING for MOBILE so the last hop's residual
        does not sit in the estimator, and again when re-entering HOPPING
        because core.step still runs while walking and can rewrite v_hat.

        new_session: every switch TO hopping is a fresh hop bout -- restore
        launch height/l0, clear trim/E_loss via user_reset, first LO v=0.
        first_lo_zero_xy: MOBILE->HOPPING first liftoff latches v=0
        (P4/stand kinematics pollute the last-N ring). Implied by new_session.
        """
        if new_session:
            self._begin_new_hopping_session(reason)
            first_lo_zero_xy = True
        try:
            self.core.user_reset(zero_yaw=False)
        except Exception:
            pass
        self._v_cmd_filt[:] = 0.0
        if first_lo_zero_xy:
            self.core._rt_first_lo_zero_xy = True
        jprint(
            "[gait] hop estimator cleared (%s): v_hat=0 flight_vel=0%s"
            % (reason, " first-LO v=0" if first_lo_zero_xy else "")
        )

    def _arm_rt_first_hops(self, reason: str) -> None:
        """MOBILE->HOPPING first-N hop height override (disabled when N=0)."""
        n_hops = int(max(0, int(self.lcm_cfg.switch_rt_hop_height_n)))
        if n_hops <= 0:
            return
        self._override_hop_height(
            float(self.lcm_cfg.switch_rt_hop_height_m),
            "%s first %d hop(s)" % (reason, n_hops),
        )
        self._rt_hop_height_remaining = n_hops

    def _arm_rt_first_hop_l0(self, reason: str) -> None:
        """Override leg_l0 for RM-unfold flight + first hopping stance."""
        l0_new = float(self.lcm_cfg.switch_rb_first_hop_l0_m)
        if self._rt_l0_restore is None:
            self._rt_l0_restore = float(self.core.cfg.leg_l0_m)
        l0_old = float(self.core.cfg.leg_l0_m)
        self.core.cfg.leg_l0_m = l0_new
        self._rt_first_hop_l0_saw_td = False
        # Force FLIGHT so P4's planted stance cannot fire a fake LO that
        # would restore l0 and swing-extend the leg (log 152156: 0.43→0.455
        # in 20 ms while RM started).
        self.core._stance = False
        self.core._td_t = None
        self.core._lo_t = None
        jprint(
            "[switch_loop] first-hop l0 (%s): %.3f -> %.3f m (restore %.3f); "
            "force FLIGHT hold"
            % (reason, l0_old, l0_new, float(self._rt_l0_restore))
        )

    def _restore_rt_first_hop_l0(self) -> None:
        """Undo the RT first-hop l0 override (idempotent)."""
        if self._rt_l0_restore is None:
            return
        self.core.cfg.leg_l0_m = float(self._rt_l0_restore)
        self._rt_l0_restore = None
        self._rt_first_hop_l0_saw_td = False
        jprint("[switch_loop] first-hop l0 restored -> %.3f m"
              % float(self.core.cfg.leg_l0_m))

    def _restore_hop_height(self, reason: str) -> None:
        """Undo a temporary height override (idempotent)."""
        self._rt_hop_height_remaining = 0
        if self._hop_height_restore is None:
            return
        use_clear = bool(getattr(
            self.core.cfg, "foot_clearance_control_enable", False
        ))
        if use_clear:
            h_old = float(self.core.cfg.foot_clearance_target_m)
            self.core.cfg.foot_clearance_target_m = float(
                self._hop_height_restore
            )
            self._hop_height_restore = None
            jprint(
                "[foot_clearance] RESTORE (%s): %.3f -> %.3f m"
                % (reason, h_old, float(self.core.cfg.foot_clearance_target_m))
            )
        else:
            h_old = float(self.core.cfg.hop_height_m)
            self.core.cfg.hop_height_m = float(self._hop_height_restore)
            self._hop_height_restore = None
            jprint(
                "[hop_height] RESTORE (%s): %.3f -> %.3f m"
                % (reason, h_old, float(self.core.cfg.hop_height_m))
            )

    def _open_prop_base_window(self) -> None:
        """RT first-hop prop support: collective + PWM floor.

        Converts hop_prop_base_pwm_us into total thrust (3*k*(pwm-1000)^2)
        and overrides ModeE's prop_base_thrust_ratio (flight) and
        prop_stance_base_thrust_ratio (stance) so the allocator carries the
        support while keeping FULL two-sided attitude authority. A hard
        per-motor floor (hop_prop_base_pwm_min_us) is applied in the
        publish path while the window is open.

        Closes on the first REAL hop liftoff (must see a TD first, so the
        P4 handoff fake LO cannot kill it) or hop_prop_base_timeout_s,
        whichever comes first.
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
        self._hop_prop_base_t0 = time.time()
        self._hop_prop_base_saw_td = False
        jprint(
            "[prop] base window OPEN: collective %.1f N (%.0f%% weight, "
            "= %.0f us/arm, floor %.0f us) via ModeE ratio override "
            "(close on first-hop LO or %.1fs)"
            % (ratio * mg, 100.0 * ratio,
               float(self.lcm_cfg.hop_prop_base_pwm_us),
               float(getattr(self.lcm_cfg, "hop_prop_base_pwm_min_us", 1600.0)),
               float(getattr(self.lcm_cfg, "hop_prop_base_timeout_s", 1.0)))
        )

    def _close_prop_base_window(self, reason: str) -> None:
        """Undo the RT prop-base ratio override (idempotent)."""
        was_active = bool(self._hop_prop_base_active)
        self._hop_prop_base_active = False
        self._hop_prop_base_t0 = None
        self._hop_prop_base_saw_td = False
        if self._prop_ratio_restore is not None:
            base_fl, base_st = self._prop_ratio_restore
            self.core.cfg.prop_base_thrust_ratio = float(base_fl)
            self.core.cfg.prop_stance_base_thrust_ratio = float(base_st)
            self._prop_ratio_restore = None
            if was_active:
                jprint("[prop] base window CLOSED (%s) -> ModeE prop control"
                      % reason)

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
        """Queue foot_clearance_target_m when LEFT stick returns to neutral.

        Armed by a full deflection, written on release, latched at the
        next touchdown (pogox does not retune the current stance):
          up -> hop_height_stick_up_m
          down -> hop_height_stick_down_m
          left -> hop_height_stick_left_m
          right -> hop_height_stick_right_m
        """
        if not bool(getattr(
            self.lcm_cfg, "hop_height_stick_enable", True
        )):
            self._hop_height_stick_pending = ""
            return
        if gamepad_msg is None or self._gait_mode != "hopping":
            self._hop_height_stick_pending = ""
            return
        # Ignore stick while a gait-switch temporary height is active.
        if self._hop_height_restore is not None:
            self._hop_height_stick_pending = ""
            return
        try:
            stick_x = float(gamepad_msg.leftStickAnalog[0])
            stick_y = float(gamepad_msg.leftStickAnalog[1])
        except Exception:
            self._hop_height_stick_pending = ""
            return
        if (not np.isfinite(stick_x)) or (not np.isfinite(stick_y)):
            self._hop_height_stick_pending = ""
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
        # Arm the dominant axis while deflected past trigger.
        if max(abs(stick_x), abs(stick_y)) >= trigger:
            if abs(stick_y) >= abs(stick_x):
                self._hop_height_stick_pending = (
                    "up" if stick_y <= -trigger else "down"
                )
            else:
                self._hop_height_stick_pending = (
                    "left" if stick_x <= -trigger else "right"
                )
            return
        if (max(abs(stick_x), abs(stick_y)) > release
                or not self._hop_height_stick_pending):
            return

        direction = str(self._hop_height_stick_pending)
        self._hop_height_stick_pending = ""
        preset = {
            "up": float(self.lcm_cfg.hop_height_stick_up_m),
            "down": float(self.lcm_cfg.hop_height_stick_down_m),
            "left": float(self.lcm_cfg.hop_height_stick_left_m),
            "right": float(self.lcm_cfg.hop_height_stick_right_m),
        }.get(direction, float("nan"))
        if not np.isfinite(preset):
            return
        h_min = float(getattr(self.lcm_cfg, "hop_height_min_m", 0.02))
        h_max = float(max(h_min, float(getattr(
            self.lcm_cfg, "hop_height_max_m", 0.40
        ))))
        h_new = round(float(np.clip(preset, h_min, h_max)), 3)
        # Queue only. Pogox latches this at the NEXT touchdown. Do not
        # print here — journal I/O on the 500 Hz thread stalls the loop
        # (same class of hitch as in-loop CSV).
        self.core.cfg.foot_clearance_target_m = h_new

    def _reset_mobile_leg_stow(self) -> None:
        """Clear the MOBILE leg approach/hold latch."""
        self._mobile_leg_active = False
        self._mobile_leg_holding = False
        self._mobile_leg_phase = "idle"
        self._mobile_leg_kd_cmd = 0.0
        self._mobile_leg_y_off_m = 0.0

    def _mobile_leg_stow_foot(self) -> np.ndarray:
        """FK of mobile_leg_q_des (field pose, not dead-center)."""
        q_des = np.asarray(
            self.lcm_cfg.mobile_leg_q_des, dtype=float
        ).reshape(3)
        try:
            if self.core.fk is not None:
                foot, _ = self.core.fk.forward_kinematics(q_des)
                foot = np.asarray(foot, dtype=float).reshape(3)
                if np.all(np.isfinite(foot)):
                    return foot
        except Exception:
            pass
        return np.array([0.23, -0.16, 0.37], dtype=float)

    def _update_mobile_leg_stow(
        self,
        *,
        q: np.ndarray,
        qd: np.ndarray,
        wheel_omega_cmd: np.ndarray,
        wheels_enabled: bool,
        gamepad_msg=None,
        dt: float = 0.001,
    ) -> np.ndarray:
        """Soft-stow legs after wheels start moving: center, then field pose.

        Stages:
          1. center — equal joints at mobile_leg_center_q (straight down)
          2. lift   — joint PD to mobile_leg_q_des (user field pose)
          3. hold   — Cartesian hold at that foot + stick ±Y so the leg
                      can track lateral wheel motion (right stick X)
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
            self._mobile_leg_phase = "center"
            self._mobile_leg_y_off_m = 0.0
            q_c = float(getattr(self.lcm_cfg, "mobile_leg_center_q", 0.4))
            jprint(
                "[mobile-leg] wheel motion -> CENTER first "
                f"(q={q_c:.2f}), then field pose "
                f"q_des={tuple(float(x) for x in self.lcm_cfg.mobile_leg_q_des)} "
                f"(foot~(-0.06,-0.16,+0.25), "
                f"tau_max={float(self.lcm_cfg.mobile_leg_tau_max_nm):.1f} Nm)"
            )

        if not bool(self._mobile_leg_active):
            return np.zeros(3, dtype=float)

        q_now = np.asarray(q, dtype=float).reshape(3)
        qd_now = np.asarray(qd, dtype=float).reshape(3)
        arrive = float(max(0.0, self.lcm_cfg.mobile_leg_arrive_rad))
        phase = str(self._mobile_leg_phase)
        cap = float(max(0.0, self.lcm_cfg.mobile_leg_tau_max_nm))

        # ---- HOLD: Cartesian at field foot + stick lateral Y ------------
        if phase == "hold" or bool(self._mobile_leg_holding):
            self._mobile_leg_holding = True
            self._mobile_leg_phase = "hold"
            rs_x = 0.0
            if gamepad_msg is not None:
                try:
                    rs_x = float(gamepad_msg.rightStickAnalog[0])
                except Exception:
                    rs_x = 0.0
            dz = float(self.lcm_cfg.stick_deadzone)
            if abs(rs_x) < dz:
                rs_x = 0.0
            y_max = float(getattr(self.lcm_cfg, "mobile_leg_y_stick_m", 0.10))
            y_rate = float(getattr(self.lcm_cfg, "mobile_leg_y_rate_mps", 0.12))
            # Same sign as wheel vy: stick RIGHT -> +Y (body right).
            y_tgt = float(np.clip(rs_x * y_max, -y_max, y_max))
            dy = y_tgt - float(self._mobile_leg_y_off_m)
            step = y_rate * float(max(0.0, dt))
            if abs(dy) <= step:
                self._mobile_leg_y_off_m = y_tgt
            else:
                self._mobile_leg_y_off_m += math.copysign(step, dy)
            foot = self._mobile_leg_stow_foot()
            foot = foot.copy()
            foot[1] += float(self._mobile_leg_y_off_m)
            foot = self._clamp_manip_foot_cartesian(foot)
            self._mobile_leg_kd_cmd = float(max(
                0.0, self.lcm_cfg.mobile_leg_ak60_kd_hold
            ))
            tau, _, _ = self.core.compute_stand_swing_tau(
                joint_pos=q_now,
                joint_vel=qd_now,
                leg_len_des_m=float(np.linalg.norm(foot)),
                tau_max_nm=cap,
                foot_des_b=foot,
                kp_z=float(max(50.0, self.lcm_cfg.mobile_leg_kp_hold * 15.0)),
                kd_z=4.0,
                axial_ff_n=0.0,
                cartesian_pd=True,
            )
            return np.asarray(tau, dtype=float).reshape(3)

        # ---- CENTER / LIFT: joint-space toward q_des --------------------
        if phase == "center":
            q_c = float(getattr(self.lcm_cfg, "mobile_leg_center_q", 0.4))
            q_des = np.full(3, q_c, dtype=float)
            err = (q_des - q_now).astype(float)
            if float(np.max(np.abs(err))) <= arrive:
                self._mobile_leg_phase = "lift"
                jprint(
                    "[mobile-leg] CENTER done (|e|_max="
                    f"{float(np.max(np.abs(err))):.3f} rad) -> LIFT "
                    "to field pose"
                )
                q_des = np.asarray(
                    self.lcm_cfg.mobile_leg_q_des, dtype=float
                ).reshape(3)
                err = (q_des - q_now).astype(float)
        else:
            self._mobile_leg_phase = "lift"
            q_des = np.asarray(
                self.lcm_cfg.mobile_leg_q_des, dtype=float
            ).reshape(3)
            err = (q_des - q_now).astype(float)
            if float(np.max(np.abs(err))) <= arrive:
                self._mobile_leg_phase = "hold"
                self._mobile_leg_holding = True
                jprint(
                    "[mobile-leg] LIFT arrived (|e|_max="
                    f"{float(np.max(np.abs(err))):.3f} rad) -> hold + "
                    "stick±Y with wheels"
                )

        kp = float(max(0.0, self.lcm_cfg.mobile_leg_kp))
        kd = float(max(0.0, self.lcm_cfg.mobile_leg_kd_move))
        self._mobile_leg_kd_cmd = float(max(
            0.0, self.lcm_cfg.mobile_leg_ak60_kd_move
        ))
        tau = kp * err - kd * qd_now
        if cap > 0.0:
            tau = np.clip(tau, -cap, cap)
        return tau.astype(float)

    def _button_center_tau(self, q, qd) -> np.ndarray:
        """MOBILE first-step 归中: equal-joint PD to mobile_leg_center_q."""
        q_c = float(getattr(self.lcm_cfg, "mobile_leg_center_q", 0.4))
        q_des = np.full(3, q_c, dtype=float)
        q_now = np.asarray(q, dtype=float).reshape(3)
        qd_now = np.asarray(qd, dtype=float).reshape(3)
        kp = float(max(0.0, self.lcm_cfg.mobile_leg_kp))
        kd = float(max(0.0, self.lcm_cfg.mobile_leg_kd_move))
        cap = float(max(0.0, self.lcm_cfg.mobile_leg_tau_max_nm))
        tau = kp * (q_des - q_now) - kd * qd_now
        if cap > 0.0:
            tau = np.clip(tau, -cap, cap)
        return tau.astype(float)

    def _compute_wheel_cmd(
        self, gamepad_msg, *, allow_yaw: bool = True
    ) -> np.ndarray:
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
            if allow_yaw:
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

    def _approach_start(self, *, kind: str = "button") -> None:
        self._approach_active = True
        self._approach_kind = str(kind)
        self._approach_sp = None
        self._approach_ready_since = None
        self._approach_waiting = False
        self._approach_last_twist = (0.0, 0.0, 0.0)
        self._box_ready = False
        self._box_ready_since = None
        self._box_ready_exit_since = None
        if kind == "box":
            work = float(
                self._push_work_dist_m
                if self._push_work_dist_m is not None
                else getattr(self.lcm_cfg, "push_approach_dist_m", 0.40)
            )
            jprint(
                "[approach] LB -> BOX APPROACH: square to the face, "
                f"center on tags, stop at {work:.2f} m; READY -> auto "
                "PUSH (stick/B/LB cancels)"
            )
        else:
            jprint(
                "[approach] RT -> AUTO-APPROACH: driving toward tag "
                f"({self._mobile_tag_state}); stick/B/LT cancels"
            )

    def _approach_stop(self, reason: str | None) -> None:
        if not bool(self._approach_active):
            return
        self._approach_active = False
        self._approach_kind = "button"
        self._approach_sp = None
        self._approach_ready_since = None
        self._approach_waiting = False
        self._box_ready = False
        self._box_ready_since = None
        self._box_ready_exit_since = None
        if reason:
            jprint(f"[approach] OFF ({reason})")

    def _manip_home_foot(self) -> np.ndarray:
        """Straight-down 归中: equal joints at mobile_leg_center_q (≈ 0.4)."""
        qh = float(getattr(
            self.lcm_cfg, "mobile_leg_center_q",
            getattr(self.lcm_cfg, "manip_home_q", 0.4),
        ))
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

    def _lock_button_z(self, p: np.ndarray) -> np.ndarray:
        """Keep the detected button height; envelope clamp handles limits."""
        return np.asarray(p, dtype=float).reshape(3).copy()

    def _rebuild_button_press(
        self, sp: dict, *, press_m: float | None = None
    ) -> dict:
        """Build PRE/FACE/PRESS at the detected button height."""
        out = dict(sp)
        n_in = np.asarray(
            out.get("wall_normal_in_L", [1.0, 0.0, 0.0]), dtype=float
        ).reshape(3)
        n_in[2] = 0.0
        nn = float(np.linalg.norm(n_in))
        if nn > 1e-9:
            n_in = n_in / nn
        else:
            n_in = np.array([1.0, 0.0, 0.0], dtype=float)
        face = self._lock_button_z(out["foot_face_L"])
        pm = float(
            press_m
            if press_m is not None
            else getattr(self.lcm_cfg, "manip_button_press_m", 0.05)
        )
        pre_m = float(getattr(self.lcm_cfg, "manip_button_pre_m", 0.08))
        out["press_m"] = pm
        out["foot_face_L"] = face.tolist()
        out["foot_pre_L"] = (face - pre_m * n_in).tolist()
        out["foot_press_L"] = (face + pm * n_in).tolist()
        out["wall_normal_in_L"] = n_in.tolist()
        return out

    def _button_lat_err_m(self, pre, n_in) -> float:
        """Wall-parallel distance from button hover to the approach park pose."""
        gx = float(getattr(self.lcm_cfg, "mobile_approach_goal_x_m", 0.086))
        gy = float(getattr(self.lcm_cfg, "mobile_approach_goal_y_m", 0.063))
        nx = float(n_in[0])
        ny = float(n_in[1])
        px = float(pre[0])
        py = float(pre[1])
        if nx * px + ny * py < 0.0:
            nx, ny = -nx, -ny
        nh = math.hypot(nx, ny) or 1.0
        nhx, nhy = nx / nh, ny / nh
        ex, ey = px - gx, py - gy
        e_adv = ex * nhx + ey * nhy
        return math.hypot(ex - e_adv * nhx, ey - e_adv * nhy)

    def _flush_lb_click(self) -> None:
        """Clear a leftover LB click window. Hold is LT, not LB."""
        if not bool(self._lb_click_pending):
            return
        win = float(getattr(self.lcm_cfg, "lb_double_click_s", 0.40))
        if (time.monotonic() - float(self._lb_click_t)) < win:
            return
        self._lb_click_pending = False

    def _flush_rb_click(self) -> None:
        """Apply the pending RB β preset after the multi-click window.

        Only while HOPPING -- MOBILE uses RB for the button approach.
        """
        if not bool(self._rb_click_pending):
            return
        if self._gait_mode != "hopping":
            self._rb_click_pending = False
            return
        win = float(getattr(self.lcm_cfg, "lb_double_click_s", 0.40))
        if (time.monotonic() - float(self._rb_click_t)) < win:
            return
        n = int(max(1, min(3, int(self._rb_click_n))))
        self._rb_click_pending = False
        self._apply_rb_beta_preset(n)

    def _apply_rb_beta_preset(self, n: int) -> None:
        presets = getattr(
            self.lcm_cfg, "rb_beta_presets",
            ((0.11, 0.50), (0.50, 0.50), (0.75, 0.75)),
        )
        idx = int(max(1, min(3, int(n)))) - 1
        try:
            blo, bhi = presets[idx]
        except Exception:
            blo, bhi = (0.11, 0.50)
        blo = float(blo)
        bhi = float(bhi)
        self.core.cfg.flight_deadbeat_model = True
        self.core.cfg.flight_deadbeat_beta = blo
        self.core.cfg.flight_deadbeat_beta_hi = bhi
        jprint(
            f"[gait] RB x{int(n)} -> deadbeat ON  β_lo={blo:.2f}  "
            f"β_hi={bhi:.2f}"
        )

    def _leave_deployed_to_mobile(self, source: str) -> None:
        self._gait_mode = "mobile"
        self._manip_kind = "hold"
        self._manip_init_pending = False
        self._manip_hold_q_cmd = None
        self._push_box = None
        self._push_e_des_m = 0.0
        self._reset_push_contact_state()
        self._mobile_tag_samples = []
        self._mobile_tag_ready_sp = None
        self._mobile_tag_state = "searching"
        self._reset_mobile_leg_stow()
        self._reset_manip_button()
        self._approach_stop(None)
        jprint(
            f"[gait] {source} -> MOBILE: leave deployed; "
            "leg stows after wheel motion"
        )

    def _lb_start_button(self, source: str) -> None:
        """RB: AprilTag button (auto-approach if not yet READY, then press)."""
        if bool(self._approach_active):
            self._approach_stop(source)
        ready_sp = self._mobile_tag_ready_sp
        tag_recent = (
            time.monotonic() - float(self._mobile_tag_last_sp_rx) <= 1.0
        )
        if ready_sp is not None:
            self._enter_manipulation(ready_sp, source=source, kind="button")
        elif bool(getattr(self.lcm_cfg, "mobile_approach_enable", True)) and tag_recent:
            if self._gait_mode != "mobile":
                self._leave_deployed_to_mobile(source)
            self._approach_start()
            jprint(f"[gait] {source}: tag seen, auto-approach then PRESS")
        else:
            jprint(
                f"[gait] {source} ignored: no tag "
                f"({self._mobile_tag_state}); stay {self._gait_mode}"
            )

    def _enter_manipulation(
        self, ready_sp=None, *, source: str, kind: str = "hold"
    ) -> None:
        """MOBILE -> MANIPULATION. kind=hold (q-hold) or button (tag press)."""
        self._approach_stop(None)
        self._backup_active = False
        self._backup_remain_m = 0.0
        self._gait_mode = "manipulation"
        self._manip_kind = "button" if kind == "button" else "hold"
        self._reset_mobile_leg_stow()
        self._reset_manip_button()
        if self._manip_kind == "hold":
            self._manip_init_pending = False
            self._btn_stage = "hold"
            self._btn_stage_print = "q-hold"
            qh = tuple(
                float(x) for x in np.asarray(
                    getattr(
                        self.lcm_cfg, "manip_hold_q",
                        (1.376, 1.378, -0.096),
                    ),
                    dtype=float,
                ).reshape(3)
            )
            rate = float(getattr(
                self.lcm_cfg, "manip_hold_q_rate_rad_s", 0.0
            ))
            if rate <= 1e-9:
                self._manip_hold_q_cmd = np.asarray(qh, dtype=float).reshape(3)
            else:
                try:
                    self._manip_hold_q_cmd = np.asarray(
                        self.robot_state["q"], dtype=float
                    ).reshape(3).copy()
                except Exception:
                    self._manip_hold_q_cmd = None
            kd = float(self.lcm_cfg.manip_ak60_kd)
            jprint(
                f"[gait] {source} -> PUSH-BOX hold q="
                f"({qh[0]:+.3f},{qh[1]:+.3f},{qh[2]:+.3f}) rad  "
                f"{'snap' if rate <= 1e-9 else f'slew {rate:.2f} rad/s'}  "
                f"kp={float(getattr(self.lcm_cfg, 'manip_hold_kp', 20.0)):.1f} "
                f"ak60_kd={kd:.1f}; "
                f"VISION-FREE SIMPLE; wheels same as MOBILE "
                f"(fwd/strafe + yaw); "
                f"LB/LT back to MOBILE"
            )
            return
        # Button: HOME then PRE/FACE/PRESS on the latched tag.
        self._manip_init_pending = True
        if ready_sp is not None:
            self._btn_last_setpoint = self._rebuild_button_press(ready_sp)
            self._btn_stage = "home"
            self._btn_stage_print = "home"
            self._manip_press_n_L = np.asarray(
                self._btn_last_setpoint.get(
                    "wall_normal_in_L", [1.0, 0.0, 0.0]
                ),
                dtype=float,
            ).reshape(3)
            nn = float(np.linalg.norm(self._manip_press_n_L))
            if nn > 1e-9:
                self._manip_press_n_L /= nn
            pm = float(getattr(self.lcm_cfg, "manip_button_press_m", 0.05))
            q_c = float(getattr(self.lcm_cfg, "mobile_leg_center_q", 0.4))
            jprint(
                f"[gait] {source} -> BUTTON: 归中(q={q_c:.2f}) -> PRE -> "
                f"FACE -> ARM PRESS {pm*100:.0f}cm. "
                f"READY pose frozen; no more tag reads; wheels STOP. "
                f"swing tau={float(getattr(self.lcm_cfg, 'manip_button_tau_max_nm', 8.0)):.0f}Nm "
                f"ak60_kd={float(self.lcm_cfg.mobile_leg_ak60_kd_move):.1f}. "
                f"Keep PD (do not press B)."
            )
        else:
            self._btn_stage = "idle"
            self._btn_stage_print = "stick"
            jprint(
                f"[gait] {source} -> BUTTON stick teleop "
                "(no READY tag); LT/LB back to MOBILE"
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
                jprint("[approach] tag re-captured -> resume")
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
                jprint(
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

        # GitHub stop band: READY depends on radial distance only. Clip and
        # wall yaw are diagnostics; they must not make the base cross r_min.
        r_min = float(getattr(cfg, "mobile_ready_r_min_m", 0.50))
        r_max = float(getattr(cfg, "mobile_ready_r_max_m", 0.53))
        n_dx, n_dy = self._l2d_xy(float(n_in[0]), float(n_in[1]))
        pre_dx, pre_dy = self._l2d_xy(float(pre[0]), float(pre[1]))
        if n_dx * pre_dx + n_dy * pre_dy < 0.0:
            n_dx, n_dy = -n_dx, -n_dy
        yaw_wall = math.atan2(n_dy, n_dx)
        yaw_ok = math.radians(
            float(getattr(cfg, "mobile_approach_yaw_ok_deg", 8.0))
        )
        ready_ok = (
            self._mobile_tag_state == "ready"
            and self._mobile_tag_ready_sp is not None
            and age <= stale_s
            and r_min <= face_r <= r_max
        )
        if ready_ok:
            if self._approach_ready_since is None:
                self._approach_ready_since = now_m
                jprint(
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
                        f"|face|_L={face_r*100:.1f}cm, "
                        f"yaw_wall={math.degrees(yaw_wall):+.1f}deg)"
                    ),
                    kind="button",
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
        # Hard radial safety gate from the working GitHub controller:
        # retreat if too close; inside the band remove all wall-normal
        # advance while still allowing lateral alignment and yaw correction.
        nx, ny = float(n_in[0]), float(n_in[1])
        if nx * float(pre[0]) + ny * float(pre[1]) < 0.0:
            nx, ny = -nx, -ny
        nh = math.hypot(nx, ny) or 1.0
        nhx, nhy = nx / nh, ny / nh
        adv = vx_L * nhx + vy_L * nhy
        if face_r < r_min:
            retreat = float(np.clip(0.8 * (r_min - face_r), 0.06, 0.15))
            lat_x, lat_y = vx_L - adv * nhx, vy_L - adv * nhy
            vx_L = lat_x - retreat * nhx
            vy_L = lat_y - retreat * nhy
        elif face_r <= r_max and adv > 0.0:
            vx_L -= adv * nhx
            vy_L -= adv * nhy
        self._approach_last_twist = (vx_L, vy_L, wz)
        # Wheel IK is in the drive/camera frame.
        vx_D, vy_D = self._l2d_xy(vx_L, vy_L)
        if (now_m - self._approach_dbg_t) >= 1.0:
            self._approach_dbg_t = now_m
            bear_tag = math.degrees(math.atan2(tag_dy, tag_dx))
            bear_btn = math.degrees(math.atan2(pre_dy, pre_dx))
            jprint(
                f"[approach] pre_L=({pre[0]:+.2f},{pre[1]:+.2f}) "
                f"tag_L=({tag[0]:+.2f},{tag[1]:+.2f}) "
                f"|face|_L={face_r:.2f}m "
                f"bear_tag={bear_tag:+.1f}deg bear_btn={bear_btn:+.1f}deg "
                f"yaw_wall={math.degrees(yaw_wall):+.1f}deg "
                f"err={arrive_err*100:.1f}cm clip="
                f"{self._mobile_tag_reach_error_m*100:.1f}cm "
                f"v_L=({vx_L:+.2f},{vy_L:+.2f}) "
                f"v_D=({vx_D:+.2f},{vy_D:+.2f}) wz={wz:+.2f}"
            )
        return self._kiwi_ik(vx_D, vy_D, wz)

    def _reset_manip_button(self) -> None:
        self._btn_stage = "idle"
        self._btn_hold_t0 = None
        self._btn_last_setpoint = None
        self._btn_acquire_samples = []
        self._btn_last_sample_t_wall = -1.0
        self._btn_stage_print = self._btn_stage
        self._press_drive_active = False
        self._press_drive_remain_m = 0.0

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
        mobile_xy_err = float("nan")
        try:
            src = (
                self._approach_sp["pre"]
                if self._approach_sp is not None
                else self._mobile_tag_last_sp["foot_pre_L"]
            )
            gx = float(getattr(
                self.lcm_cfg, "mobile_approach_goal_x_m", 0.086
            ))
            gy = float(getattr(
                self.lcm_cfg, "mobile_approach_goal_y_m", 0.063
            ))
            mobile_xy_err = math.hypot(
                float(src[0]) - gx, float(src[1]) - gy
            )
        except Exception:
            pass
        payload = {
            "t_wall": time.time(),
            "gait_mode": str(self._gait_mode),
            "button_stage": str(self._btn_stage),
            "mobile_tag_state": str(self._mobile_tag_state),
            # Actual handoff: PRE XY error + workspace clip + wall yaw.
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
            "mobile_tag_xy_err_m": (
                float(mobile_xy_err) if np.isfinite(mobile_xy_err) else None
            ),
            "mobile_ready_xy_m": float(
                getattr(self.lcm_cfg, "mobile_ready_xy_m", 0.03)
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
            "approach_kind": str(self._approach_kind),
            "approach_waiting": bool(self._approach_waiting),
            "mobile_tag_cam_z_m": (
                float(self._mobile_tag_cam_z_m)
                if np.isfinite(self._mobile_tag_cam_z_m) else None
            ),
            "push_e_m": float(self._push_e_des_m),
            "push_v_mps": float(self._push_v_last),
            # Box-push telemetry: pose errors vs the working pose, READY,
            # latched working distance, measured push force + state.
            "box_dist_err_m": (
                float(self._box_dist_err_m)
                if np.isfinite(self._box_dist_err_m) else None
            ),
            "box_yaw_err_deg": (
                float(self._box_yaw_err_deg)
                if np.isfinite(self._box_yaw_err_deg) else None
            ),
            "box_ready": bool(self._box_ready),
            "push_work_dist_m": (
                float(self._push_work_dist_m)
                if self._push_work_dist_m is not None else None
            ),
            "push_f_meas_n": (
                float(self._push_f_meas_n)
                if np.isfinite(self._push_f_meas_n) else None
            ),
            "push_state": self._push_state_str() or None,
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
        # AUXILIARY I/O: serialization, the atomic file write and the UDP
        # send all happen on the aux thread. A 20 Hz JSON write to eMMC on
        # the control thread was one of the 500 Hz stall sources.
        sock = self._dashboard_udp_sock
        addr = (
            str(self.lcm_cfg.dashboard_udp_host),
            int(self.lcm_cfg.dashboard_udp_port),
        )

        def _flush() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            encoded = json.dumps(payload)
            tmp.write_text(encoded)
            tmp.replace(path)
            sock.sendto(encoded.encode("utf-8"), addr)

        _AUX.put(_flush)

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

    def _try_enter_push_hold(self, source: str) -> None:
        """MOBILE -> box push (60 mm tags id 2/3).

        push_simple: snap manip_hold_q, sticks drive the base.
        otherwise: auto-align then PUSH (LEFT stick drives the box).
        """
        if bool(self._approach_active) and self._approach_kind == "box":
            self._approach_stop(source)
            return
        # Simplified mode is intentionally vision-free: deploy the fixed
        # hold pose immediately and leave all wheel motion to the sticks.
        # Previously this check sat after the tag gate, so "simple" LB was
        # unavailable unless id2/id3 happened to be visible.
        if bool(getattr(self.lcm_cfg, "push_simple", False)):
            self._enter_manipulation(source=source, kind="hold")
            return
        box = self._read_box_setpoint()
        if not self._box_has_push_tags(box):
            ids = getattr(self.lcm_cfg, "push_box_tag_ids", (2, 3))
            jprint(
                f"[gait] {source} ignored: need 60 mm box tag "
                f"id={int(ids[0])} or {int(ids[1])}; stay MOBILE"
            )
            return
        self._store_push_box(box)
        if self._box_in_ready_band():
            jprint(f"[gait] {source}: already at working pose -> PUSH")
            self._enter_push()
        else:
            self._approach_start(kind="box")

    def _box_has_push_tags(self, box) -> bool:
        """True if the perception box packet saw 60 mm box ids (2 and/or 3)."""
        if not isinstance(box, dict) or not bool(box.get("valid", False)):
            return False
        want = {
            int(x) for x in getattr(self.lcm_cfg, "push_box_tag_ids", (2, 3))
        }
        try:
            seen = {int(x) for x in (box.get("tag_ids_seen") or [])}
        except Exception:
            return False
        return bool(want & seen)

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
        the outer sphere so joint q stays <= manip_q_max.
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
        # No tilted q=1.4 radial carve: it rejected two-joint-down hold
        # (L=0.476) as ~21 cm out of range. Deployed travel is work_*.
        return p.astype(float)

    def _work_wheel_az_L(self) -> tuple:
        """Wheel keep-out azimuths in the leg/L frame (rad).

        Bottom wheel on x=0 (270 deg), then 120 deg (user 2026-08-23).
        Not the drive-frame kiwi layout and not rotated by drive yaw.
        """
        az = getattr(
            self.lcm_cfg, "work_wheel_azimuth_deg", (270.0, 30.0, 150.0)
        )
        return tuple(math.radians(float(a)) for a in az)

    def _work_wheel_centers_xy(self) -> tuple:
        """Three wheel hubs in L, common radius, 120 deg apart."""
        r = float(getattr(self.lcm_cfg, "work_wheel_center_m", 0.17))
        return tuple(
            (r * math.cos(az), r * math.sin(az))
            for az in self._work_wheel_az_L()
        )

    def _work_keep_cross(self, z: float) -> tuple:
        """Wheel-barrel cross-section at height z, margin included:
        (centre radius on the wheel ray, radial semi-axis, tangential
        semi-axis). Zero semi-axes outside the barrel's z span."""
        cfg = self.lcm_cfg
        z_bot = float(getattr(cfg, "work_z_bottom_m", 0.440))
        z_wide = float(getattr(cfg, "work_keep_z_wide_m", 0.267))
        rc0 = float(getattr(cfg, "work_wheel_center_m", 0.17))
        a0 = float(getattr(cfg, "work_wheel_keep_m", 0.055))
        rc1 = float(getattr(cfg, "work_keep_r_wide_m", 0.229))
        a1 = float(getattr(cfg, "work_keep_a_wide_m", 0.114))
        b1 = float(getattr(cfg, "work_keep_b_wide_m", 0.150))
        margin = float(getattr(cfg, "work_keep_margin_m", 0.008))
        cz = z_bot - z_wide
        if cz <= 1e-9:
            return rc0, a0 + margin, a0 + margin
        t = (float(z) - z_wide) / cz
        if abs(t) > 1.0:
            return rc0, 0.0, 0.0
        s = math.sqrt(max(0.0, 1.0 - t * t))
        return (rc0 + s * (rc1 - rc0),
                a0 + s * (a1 - a0) + margin,
                a0 + s * (b1 - a0) + margin)

    def _work_keep_near_r(self, az: float, z: float) -> float:
        """Smallest r>0 where the ray at azimuth az enters a wheel barrel at
        height z; inf when the ray misses all three."""
        rc, a, b = self._work_keep_cross(z)
        if a <= 0.0 or b <= 0.0:
            return float("inf")
        best = float("inf")
        for az_w in self._work_wheel_az_L():
            d = float(az) - az_w
            c, s = math.cos(d), math.sin(d)
            qa = (c / a) ** 2 + (s / b) ** 2
            qb = -2.0 * rc * c / (a * a)
            qc = (rc / a) ** 2 - 1.0
            disc = qb * qb - 4.0 * qa * qc
            if disc <= 0.0 or qa <= 0.0:
                continue
            r1 = (-qb - math.sqrt(disc)) / (2.0 * qa)
            if 0.0 < r1 < best:
                best = r1
        return best

    def _work_keep_inside(self, x: float, y: float, z: float) -> bool:
        rc, a, b = self._work_keep_cross(z)
        if a <= 0.0 or b <= 0.0:
            return False
        for az_w in self._work_wheel_az_L():
            c, s = math.cos(az_w), math.sin(az_w)
            du = float(x) * c + float(y) * s - rc
            dn = -float(x) * s + float(y) * c
            if (du / a) ** 2 + (dn / b) ** 2 <= 1.0:
                return True
        return False

    def _work_r_cap(self, az: float, z: float) -> float:
        """Max foot xy-radius at this azimuth/height (outer disk minus
        the three wheel barrels). Connected from the origin: a ray that
        hits a barrel is capped at the inner intersection."""
        l_max = float(getattr(self.lcm_cfg, "work_len_max_m", 0.50))
        r_max = math.sqrt(max(0.0, l_max * l_max - float(z) ** 2))
        xy_max = float(getattr(self.lcm_cfg, "work_xy_max_m", 0.37))
        if xy_max > 0.0:
            r_max = min(r_max, xy_max)
        return min(r_max, self._work_keep_near_r(az, z))

    def _work_z_floor_max(self, r: float) -> float:
        """Deepest allowed z at this xy radius.

        User 2026-08-23: r_xy < 0.10 m at z_bottom hits the floor.
        Cone from work_len_home at r=0 to z_bottom at r = work_xy_floor_r_m.
        The apex is 归中, not work_len_min: the folding room work_len_min
        buys is for reaching into the keep-out ducts out at r=0.107, and
        pinning the apex to it would put 归中 itself outside the envelope.
        """
        cfg = self.lcm_cfg
        z_bot = float(getattr(cfg, "work_z_bottom_m", 0.440))
        z0 = float(getattr(cfg, "work_len_home_m", 0.354))
        r_floor = float(getattr(cfg, "work_xy_floor_r_m", 0.10))
        if r_floor <= 1e-9 or float(r) >= r_floor:
            return z_bot
        return z0 + (z_bot - z0) * max(0.0, float(r)) / r_floor

    def _clamp_work_envelope(self, target: np.ndarray) -> np.ndarray:
        """Deployed-leg WORK ENVELOPE clamp (user-defined 2026-08-14).

        z band [work_z_top, work_z_bottom], L in [work_len_min, work_len_max],
        horizontal r <= work_xy_max (forward-tilt disk ≈ 0.36 m),
        minus three identical wheel barrels (120 deg, left/right symmetric
        about drive-forward; cross-section grows with height, see
        _work_keep_cross). Inside r < work_xy_floor_r_m the lowest z
        is lifted (floor cone). Not applied to MOBILE stow.
        """
        cfg = self.lcm_cfg
        z_top = float(getattr(cfg, "work_z_top_m", 0.354))
        z_bot = float(getattr(cfg, "work_z_bottom_m", 0.440))
        p = np.asarray(target, dtype=float).reshape(3).copy()
        if not np.all(np.isfinite(p)):
            return np.array([0.0, 0.0, z_top], dtype=float)
        p[2] = float(np.clip(p[2], min(z_top, z_bot), max(z_top, z_bot)))
        l_min = float(getattr(cfg, "work_len_min_m", 0.354))
        l_max = float(getattr(cfg, "work_len_max_m", 0.50))
        L = float(np.linalg.norm(p))
        if L < 1e-9:
            return np.array([0.0, 0.0, max(z_top, l_min)], dtype=float)
        if L < l_min:
            p = p * (l_min / L)
            L = l_min
        elif L > l_max:
            p = p * (l_max / L)
            L = l_max
        p[2] = float(np.clip(p[2], min(z_top, z_bot), max(z_top, z_bot)))
        r = math.hypot(float(p[0]), float(p[1]))
        z_floor = self._work_z_floor_max(r)
        if float(p[2]) > z_floor:
            p[2] = z_floor
        xy_max = float(getattr(cfg, "work_xy_max_m", 0.37))
        r_cap = math.sqrt(max(0.0, l_max * l_max - float(p[2]) ** 2))
        if xy_max > 0.0:
            r_cap = min(r_cap, xy_max)
        az = math.atan2(float(p[1]), float(p[0]))
        # Always cap at the near intersection, even where the foot itself is
        # in the clear sliver beyond a barrel.  The leg is the segment from
        # the hip to the foot, so reaching that sliver drags the leg through
        # the wheel -- 2026-08-24, the trace hit exactly this at az=144 deg
        # on the TOP ring with the foot 9 mm clear of the barrel and the leg
        # 45 mm inside the wheel.
        r_cap = min(r_cap, self._work_keep_near_r(az, float(p[2])))
        r = math.hypot(float(p[0]), float(p[1]))
        if r > r_cap and r > 1e-9:
            p[0] *= r_cap / r
            p[1] *= r_cap / r
        return p

    def _clamp_push_foot_plane(self, target: np.ndarray) -> np.ndarray:
        """PUSH-mode clamp: fixed z = FK(manip_hold_q).z, xy-disc only.

        The generic Cartesian clamp must NOT be used here: its empirical
        q-carve caps L at ~0.38 m at 40 deg tilt, but the hold pose is
        L = 0.476 m (q near-limit but valid), so it would drag the foot
        off the plane and away from the face.
        """
        if self._push_z_hold_m is None:
            return self._clamp_manip_foot_cartesian(target)
        p = np.asarray(target, dtype=float).reshape(3).copy()
        z0 = float(self._push_z_hold_m)
        if not np.all(np.isfinite(p)):
            return np.array([0.0, 0.0, z0], dtype=float)
        p[2] = z0
        xy_max = float(max(
            0.05, getattr(self.lcm_cfg, "push_xy_max_m", 0.31)
        ))
        xy_norm = float(np.linalg.norm(p[:2]))
        if xy_norm > xy_max:
            p[:2] *= xy_max / xy_norm
        # Work envelope on top: z band + length cap + wheel sectors.
        return self._clamp_work_envelope(p)

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
                jprint("[mobile-tag] LOST -> searching for tag id=1")
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
                jprint(f"[mobile-tag] DETECTED id=1 -> stabilizing {count} frames")
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
        latched = dict(sp)
        latched["foot_face_L"] = face.tolist()
        latched["wall_normal_in_L"] = n_in.tolist()
        latched = self._rebuild_button_press(latched)

        # GitHub behavior: the 0.50-0.53 m radial band is the authoritative
        # stop/READY condition. Workspace clip remains diagnostic only; making
        # it a gate caused the base to drive through the stop band to 0.42 m.
        reach_error = 0.0
        for key in ("foot_pre_L", "foot_face_L", "foot_press_L"):
            raw = np.asarray(latched[key], dtype=float).reshape(3)
            clipped = self._clamp_work_envelope(raw)
            reach_error = max(
                reach_error, float(np.linalg.norm(clipped - raw))
            )
        self._mobile_tag_reach_error_m = reach_error
        r_face = float(np.linalg.norm(face))
        r_min = float(getattr(self.lcm_cfg, "mobile_ready_r_min_m", 0.50))
        r_max = float(getattr(self.lcm_cfg, "mobile_ready_r_max_m", 0.53))
        if r_face < r_min or r_face > r_max:
            if self._mobile_tag_state != "unreachable":
                why = []
                if r_face > r_max:
                    why.append(f"|face|={r_face:.2f}>r_max={r_max:.2f}")
                if r_face < r_min:
                    why.append(f"|face|={r_face:.2f}<r_min={r_min:.2f}")
                if reach_error > 0.02:
                    why.append(f"clip={reach_error*100:.1f}cm")
                jprint(
                    "[mobile-tag] DETECTED/STABLE id=1 but UNREACHABLE: "
                    + ", ".join(why)
                    + f"; drive MOBILE into {r_min:.2f}-{r_max:.2f} m band"
                )
            self._mobile_tag_ready_sp = None
            self._mobile_tag_state = "unreachable"
            return

        self._mobile_tag_ready_sp = latched
        if self._mobile_tag_state != "ready":
            jprint(
                "[mobile-tag] READY id=1: "
                f"|face|_L={r_face:.2f} m in [{r_min:.2f},{r_max:.2f}], "
                f"clip={reach_error*100:.1f}cm (info), "
                f"spread={spread*1000:.1f}mm "
                "-> RB / auto-approach enters arm PRESS"
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

    def _push_work_dist(self) -> float:
        """Working face distance: latched value if set, else config default."""
        if self._push_work_dist_m is not None:
            return float(self._push_work_dist_m)
        return float(getattr(self.lcm_cfg, "push_approach_dist_m", 0.40))

    def _box_pose_errors(self) -> dict | None:
        """Drive-frame box-face errors from the stored box memory.

        Returns None when no box is stored or the normal is degenerate.
        Keys: dist (m, along the inward normal to the face center),
        yaw (rad, 0 = camera square to the face), lat (m, signed along
        face-right), n_h / r_h (unit drive-XY normal / face-right).
        """
        b = self._push_box
        if b is None:
            return None
        cx, cy = self._l2d_xy(float(b["center"][0]), float(b["center"][1]))
        ndx, ndy = self._l2d_xy(float(b["n_in"][0]), float(b["n_in"][1]))
        # n_in must point INTO the box (same half-plane as the center),
        # matching the guard in _update_push_mode / approach_law.
        if ndx * cx + ndy * cy < 0.0:
            ndx, ndy = -ndx, -ndy
        nh = math.hypot(ndx, ndy)
        if nh < 1e-6:
            return None
        nhx, nhy = ndx / nh, ndy / nh
        rdx, rdy = self._l2d_xy(float(b["right"][0]), float(b["right"][1]))
        rh = math.hypot(rdx, rdy)
        if rh > 1e-6:
            rhx, rhy = rdx / rh, rdy / rh
        else:
            rhx, rhy = -nhy, nhx
        return {
            "dist": float(cx * nhx + cy * nhy),
            "yaw": float(math.atan2(nhy, nhx)),
            "lat": float(cx * rhx + cy * rhy),
            "n_h": (nhx, nhy),
            "r_h": (rhx, rhy),
        }

    def _box_in_ready_band(self) -> bool:
        """PUSH-ready test; also refreshes the telemetry error fields."""
        cfg = self.lcm_cfg
        err = self._box_pose_errors()
        if err is None:
            self._box_dist_err_m = float("nan")
            self._box_yaw_err_deg = float("nan")
            return False
        d_err = float(err["dist"]) - self._push_work_dist()
        self._box_dist_err_m = float(d_err)
        self._box_yaw_err_deg = float(math.degrees(err["yaw"]))
        return (
            abs(d_err)
            <= float(getattr(cfg, "push_ready_dist_tol_m", 0.04))
            and abs(err["yaw"])
            <= math.radians(float(getattr(cfg, "push_ready_yaw_deg", 6.0)))
        )

    def _update_box_approach(self, gamepad_msg, dt: float) -> np.ndarray | None:
        """Wheel servo toward the PUSH working pose. Returns rad/s or None.

        Goal (drive frame): camera square to the box face (yaw -> 0),
        face center on the midline (lat -> 0), face at the working
        distance. When both READY bands hold for push_ready_hold_s the
        wheels freeze and PUSH starts automatically (2026-08-13 user:
        one LB = align + deploy leg + stick drives the box).
        """
        if not (bool(self._approach_active)
                and self._approach_kind == "box"):
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
        box = self._read_box_setpoint()
        if box is not None:
            self._store_push_box(box)
            if bool(self._approach_waiting):
                self._approach_waiting = False
                jprint("[approach] box re-captured -> resume")
        age = now_m - float(self._push_box_rx)
        stale_s = float(getattr(cfg, "push_box_stale_s", 1.0))
        memory_s = float(getattr(cfg, "approach_memory_timeout_s", 3.0))
        if self._push_box is None or age > memory_s:
            if not bool(self._approach_waiting):
                self._approach_waiting = True
                jprint(
                    "[approach] box lost > "
                    f"{memory_s:.1f}s -> wheels stop, waiting for box"
                )
            self._box_ready = False
            self._box_ready_since = None
            self._box_ready_exit_since = None
            self._approach_last_twist = (0.0, 0.0, 0.0)
            return np.zeros(3, dtype=float)
        if box is None and age > 0.5 * stale_s:
            # Short dropout: advance the L-frame memory by the commanded
            # drive-frame twist (same scheme as PUSH mode).
            vx_d, vy_d, wz_d = self._approach_last_twist
            vx_L, vy_L = self._d2l_xy(vx_d, vy_d)
            b = self._push_box
            self._dead_reckon_xy(
                [b["center"]], [b["n_in"], b["right"]],
                vx_L, vy_L, wz_d, dt,
            )

        in_band = self._box_in_ready_band()
        err = self._box_pose_errors()
        if err is None:
            self._approach_last_twist = (0.0, 0.0, 0.0)
            return np.zeros(3, dtype=float)
        if in_band and age <= stale_s:
            self._box_ready_exit_since = None
            hold = float(getattr(cfg, "push_ready_hold_s", 0.5))
            if self._box_ready_since is None:
                self._box_ready_since = now_m
            if (now_m - self._box_ready_since) >= hold:
                self._box_ready = True
                jprint(
                    "[approach] box READY "
                    f"(dist err={self._box_dist_err_m * 100:+.1f}cm, "
                    f"yaw err={self._box_yaw_err_deg:+.1f}deg) "
                    "-> auto PUSH (deploy leg; LEFT stick = box)"
                )
                self._enter_push()
                return np.zeros(3, dtype=float)
            # Freeze while inside the band (confirming READY).
            self._approach_last_twist = (0.0, 0.0, 0.0)
            return np.zeros(3, dtype=float)

        # Outside the tight entry band. If already READY, keep holding
        # until the WIDER exit band is violated for exit_hold_s (tag yaw
        # noise otherwise flapped READY↔servo every frame, log 19:43).
        if bool(self._box_ready):
            d_err = float(err["dist"]) - self._push_work_dist()
            yaw_err = float(err["yaw"])
            out_exit = (
                abs(d_err)
                > float(getattr(cfg, "push_ready_exit_dist_tol_m", 0.08))
                or abs(yaw_err) > math.radians(
                    float(getattr(cfg, "push_ready_exit_yaw_deg", 18.0))
                )
            )
            if out_exit:
                if self._box_ready_exit_since is None:
                    self._box_ready_exit_since = now_m
                exit_hold = float(
                    getattr(cfg, "push_ready_exit_hold_s", 0.6)
                )
                if (now_m - self._box_ready_exit_since) >= exit_hold:
                    jprint(
                        "[approach] box left READY "
                        f"(dist err={d_err * 100:+.1f}cm, "
                        f"yaw err={math.degrees(yaw_err):+.1f}deg) "
                        "-> re-servo"
                    )
                    self._box_ready = False
                    self._box_ready_since = None
                    self._box_ready_exit_since = None
                else:
                    self._approach_last_twist = (0.0, 0.0, 0.0)
                    return np.zeros(3, dtype=float)
            else:
                # Inside the exit band: stay READY, wheels hold.
                self._box_ready_exit_since = None
                self._approach_last_twist = (0.0, 0.0, 0.0)
                return np.zeros(3, dtype=float)
        self._box_ready = False
        self._box_ready_since = None
        self._box_ready_exit_since = None

        # PBVS in the drive frame (camera forward = +x).
        kp_v = float(getattr(cfg, "mobile_approach_kp_v", 0.70))
        v_max = float(getattr(cfg, "mobile_approach_v_max_mps", 0.22))
        w_max = float(getattr(cfg, "push_wz_max_rad_s", 0.5))
        nhx, nhy = err["n_h"]
        rhx, rhy = err["r_h"]
        d_err = float(err["dist"]) - self._push_work_dist()
        yaw_err = float(err["yaw"])
        lat = float(err["lat"])
        # Gate the along-normal advance on lateral/yaw error so the robot
        # squares up before closing distance (same idea as approach_law).
        lat_gate = float(getattr(cfg, "mobile_approach_lat_gate_m", 0.08))
        bear_gate = math.radians(
            float(getattr(cfg, "mobile_approach_bear_gate_deg", 12.0))
        )
        adv_scale = max(0.0, 1.0 - abs(lat) / max(1e-6, lat_gate))
        adv_scale *= max(0.0, 1.0 - abs(yaw_err) / max(1e-6, bear_gate))
        # d_err > 0 = too far -> advance along +n (into the face).
        v_adv = kp_v * d_err * adv_scale
        v_lat = kp_v * lat
        vx = float(np.clip(v_adv * nhx + v_lat * rhx, -v_max, v_max))
        vy = float(np.clip(v_adv * nhy + v_lat * rhy, -v_max, v_max))
        if age > stale_s:
            slow = float(getattr(cfg, "mobile_approach_memory_scale", 0.3))
            vx *= slow
            vy *= slow
        wz = float(np.clip(
            float(getattr(cfg, "push_kp_wz", 1.5)) * yaw_err,
            -w_max, w_max,
        ))
        if age > stale_s:
            wz = 0.0
        self._approach_last_twist = (vx, vy, wz)
        if (now_m - self._approach_dbg_t) >= 1.0:
            self._approach_dbg_t = now_m
            jprint(
                f"[approach] box dist={err['dist']:.2f}m "
                f"(err={d_err * 100:+.1f}cm) "
                f"yaw={math.degrees(yaw_err):+.1f}deg "
                f"lat={lat * 100:+.1f}cm "
                f"v_D=({vx:+.2f},{vy:+.2f}) wz={wz:+.2f}"
            )
        return self._kiwi_ik(vx, vy, wz)

    def _push_state_str(self) -> str:
        """One-word PUSH contact state for logs/dashboard."""
        if self._gait_mode != "push":
            return ""
        if bool(self._push_waiting):
            return "WAIT"
        phase = str(getattr(self, "_push_leg_phase", "track"))
        if phase == "home":
            return "HOME"
        if phase == "reach":
            return "REACH"
        return "TRACK"

    def _reset_push_contact_state(self) -> None:
        self._push_f_meas_n = float("nan")
        self._push_f_n_lpf = float("nan")
        self._push_f_t_lpf = float("nan")
        self._push_depth_cmd_m = float(
            getattr(self.lcm_cfg, "push_contact_depth_m", 0.02)
        )
        self._push_tgt_lpf = None
        self._push_leg_phase = "home"

    def _enter_push(self) -> None:
        """MOBILE -> PUSH: snap foot onto the face, LEFT stick drives box."""
        self._approach_stop(None)
        self._gait_mode = "push"
        # init_pending=True -> first _update_push_mode tick SNAPs the foot
        # target straight onto the face contact (no HOME crawl).
        self._manip_init_pending = True
        self._push_e_des_m = 0.0
        self._push_last_twist = (0.0, 0.0, 0.0)
        self._push_waiting = False
        self._reset_push_contact_state()
        self._push_leg_phase = "track"
        self._reset_mobile_leg_stow()
        # Lock the foot height for the whole push: z = FK(manip_hold_q).z
        # (the simplified-push pose). Targets below only move in xy.
        try:
            qh = np.asarray(
                getattr(
                    self.lcm_cfg, "manip_hold_q", (1.376, 1.378, -0.096)
                ),
                dtype=float,
            ).reshape(3)
            foot_h, _ = self.core.fk.forward_kinematics(qh)
            self._push_z_hold_m = float(
                np.asarray(foot_h, dtype=float).reshape(3)[2]
            )
        except Exception:
            self._push_z_hold_m = None
        box = self._read_box_setpoint()
        if box is not None:
            self._store_push_box(box)
        # First manual entry latches the measured face distance as THE
        # working distance ("现在这个状态"), process-persistent: later box
        # approaches drive back to this same pose.
        if self._push_work_dist_m is None:
            err = self._box_pose_errors()
            if err is not None and np.isfinite(err["dist"]):
                self._push_work_dist_m = float(err["dist"])
                jprint(
                    "[push] working distance LATCHED at "
                    f"{self._push_work_dist_m:.2f} m (default was "
                    f"{float(getattr(self.lcm_cfg, 'push_approach_dist_m', 0.40)):.2f} m)"
                )
        src = (self._push_box or {}).get("source", "?")
        jprint(
            "[gait] PUSH: snap foot->face contact + AK60 kd; LEFT stick = "
            f"box fwd + e-offset steer (source={src}); LB/LT exits"
        )

    def _apply_push_twist(
        self,
        vx: float,
        vy: float,
        wz: float,
        dt: float,
        control_enabled: bool,
    ) -> None:
        """Accel-limited wheel twist for PUSH (2026-08-13: 画面糊).

        Step changes in (vx, vy, wz) shook the camera and blurred the
        tags; slew each component from the previously commanded twist so
        starts, stops and tail swings ramp instead of jumping. The slewed
        value also feeds _push_last_twist, keeping dead reckoning honest.
        """
        pvx, pvy, pwz = self._push_last_twist
        dv = float(max(
            0.0, getattr(self.lcm_cfg, "push_acc_max_mps2", 0.5)
        )) * float(max(0.0, dt))
        dw = float(max(
            0.0, getattr(self.lcm_cfg, "push_wz_acc_max_rad_s2", 2.0)
        )) * float(max(0.0, dt))
        vx = float(np.clip(vx, pvx - dv, pvx + dv))
        vy = float(np.clip(vy, pvy - dv, pvy + dv))
        wz = float(np.clip(wz, pwz - dw, pwz + dw))
        self._push_last_twist = (vx, vy, wz)
        self._wheel_pending_cmd = self._kiwi_ik(vx, vy, wz)
        self._wheel_pending_enable = (
            bool(control_enabled) and not self._rm_drive_busy()
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
             wz ONLY servos the camera square/parallel to the face (never
             cone-capped). When the e-offset torque rotates the box, this
             servo swings the tail to follow the rotating face (甩尾).
        """
        cfg = self.lcm_cfg
        box = self._read_box_setpoint()
        now_m = time.monotonic()
        if box is not None:
            self._store_push_box(box)
            if bool(self._push_waiting):
                self._push_waiting = False
                jprint("[push] box re-captured -> resume")
        if bool(self._manip_init_pending):
            # SNAP the foot target straight onto the face contact point
            # (no HOME / crawl -- user 2026-08-13: 摆腿太慢). AK60 kd
            # damps the jump; kp is the soft push PD below.
            self._manip_init_pending = False
            self._push_leg_phase = "track"
            self._push_depth_cmd_m = float(
                getattr(cfg, "push_contact_depth_m", 0.02)
            )
            b0 = self._push_box
            if b0 is not None:
                snap_tgt = (
                    np.asarray(b0["center"], dtype=float).reshape(3)
                    + self._push_depth_cmd_m
                    * np.asarray(b0["n_in"], dtype=float).reshape(3)
                )
                self._manip_foot_des_b = self._clamp_push_foot_plane(
                    snap_tgt
                )
            else:
                self._update_manip_foot_target(None, q, 0.0)
            self._push_tgt_lpf = None

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

        # ---- Contact-force feedback: F = J^-T tau (foot->environment,
        # leg frame), split into F_n (into the face) and F_t (along
        # face_right). LPF'd; feeds the depth admittance + slip guard.
        b_mem = self._push_box
        try:
            tau_meas = np.asarray(
                self.robot_state.get("tau", [np.nan, np.nan, np.nan]),
                dtype=float,
            ).reshape(3)
            f_leg_L = self.core.estimate_contact_force_native(
                joint_pos=q, tau_meas=tau_meas
            )
            if b_mem is not None and np.all(np.isfinite(f_leg_L)):
                f_n = float(np.dot(f_leg_L, b_mem["n_in"]))
                f_t = float(np.dot(f_leg_L, b_mem["right"]))
                self._push_f_meas_n = f_n
                lp = max(
                    1e-3, float(getattr(cfg, "push_f_lpf_tau_s", 0.15))
                )
                af = min(1.0, float(max(0.0, dt)) / lp)
                if not np.isfinite(self._push_f_n_lpf):
                    self._push_f_n_lpf = f_n
                    self._push_f_t_lpf = f_t
                else:
                    self._push_f_n_lpf += af * (f_n - self._push_f_n_lpf)
                    self._push_f_t_lpf += af * (f_t - self._push_f_t_lpf)
        except Exception:
            pass

        # Steering sign (verified 2026-08-09, FRD z DOWN so +torque about
        # z = CW from above): stick RIGHT (ls_x > 0) must turn the BOX
        # right = CW. Contact LEFT of the friction center (offset -e along
        # face_right, r = (0,-e,0)) with the push force F = (F,0,0) gives
        # (r x F)_z = +eF > 0 = CW. Hence e_des integrates -ls_x.
        e_max = float(getattr(cfg, "push_e_max_m", 0.12))
        e_rate = float(getattr(cfg, "push_e_rate_mps", 0.08))
        de = (-ls_x) * e_rate * float(dt)
        # Force-based motion cone: |F_t|/F_n near the friction limit means
        # the foot is about to slide off the face -- hold |e|, do not
        # command more offset torque. Only trips with a firm contact.
        slip_max = float(getattr(cfg, "push_slip_ratio_max", 0.8))
        e_grows = abs(self._push_e_des_m + de) > abs(self._push_e_des_m)
        if (
            np.isfinite(self._push_f_n_lpf)
            and self._push_f_n_lpf > 2.0
            and abs(self._push_f_t_lpf) > slip_max * self._push_f_n_lpf
            and e_grows
        ):
            de = 0.0
        # PARALLEL FIRST (2026-08-13: 不平行箱角会打到轮子): while the
        # tail swing has not caught up, do not command MORE box rotation.
        # (_box_yaw_err_deg is one tick stale here -- fine at 100 Hz.)
        yaw_ok = float(getattr(cfg, "push_yaw_ok_deg", 8.0))
        if (
            e_grows
            and np.isfinite(self._box_yaw_err_deg)
            and abs(self._box_yaw_err_deg) > yaw_ok
        ):
            de = 0.0
        self._push_e_des_m = float(np.clip(
            self._push_e_des_m + de, -e_max, e_max,
        ))
        self._push_v_last = float(v_push)
        pose_err = self._box_pose_errors()
        if pose_err is not None:
            self._box_dist_err_m = (
                float(pose_err["dist"]) - self._push_work_dist()
            )
            self._box_yaw_err_deg = float(math.degrees(pose_err["yaw"]))

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

        # ---- Wheels: box-stick law; leg tracks the offset contact ------
        if lost:
            if not bool(self._push_waiting):
                self._push_waiting = True
                jprint(
                    "[push] box lost > "
                    f"{memory_s:.1f}s -> wheels stop, leg holds contact"
                )
            self._apply_push_twist(0.0, 0.0, 0.0, dt, control_enabled)
        else:
            b = self._push_box
            center = b["center"]
            n_in = b["n_in"]
            right = b["right"]
            # ---- Normal-force keeper (甩尾时丢箱子): admittance on the
            # press depth. F_n sagging under push_f_des_n (face rotating
            # away mid-swing) presses deeper; a spike backs off.
            if np.isfinite(self._push_f_n_lpf):
                gain = float(getattr(cfg, "push_f_adapt_mps_per_n", 0.003))
                err_f = (
                    float(getattr(cfg, "push_f_des_n", 5.0))
                    - self._push_f_n_lpf
                )
                self._push_depth_cmd_m = float(np.clip(
                    self._push_depth_cmd_m
                    + gain * err_f * float(max(0.0, dt)),
                    float(getattr(cfg, "push_depth_min_m", 0.005)),
                    float(getattr(cfg, "push_depth_max_m", 0.06)),
                ))
            raw_tgt = (
                center
                + self._push_e_des_m * right
                + self._push_depth_cmd_m * n_in
            )
            # z is pinned by _clamp_push_foot_plane below: the foot only
            # translates in the FK(manip_hold_q).z horizontal plane; the
            # box tag height never drives the foot up/down.
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
            if ndx * float(center[0]) + ndy * float(center[1]) < 0.0:
                ndx, ndy = -ndx, -ndy
                n_h = -n_h
            yaw_err = math.atan2(ndy, ndx)
            wz_max = float(getattr(cfg, "push_wz_max_rad_s", 0.5))
            # PARALLEL FIRST: advancing while yaw_err is large drives the
            # box corner into the wheels. Full speed while squared
            # (<= yaw_ok), linear fade to a full stop at yaw_stop; the
            # squaring servo below keeps running at full authority.
            yaw_ok = math.radians(
                float(getattr(cfg, "push_yaw_ok_deg", 8.0))
            )
            yaw_stop = math.radians(
                float(getattr(cfg, "push_yaw_stop_deg", 18.0))
            )
            par = float(np.clip(
                (yaw_stop - abs(yaw_err)) / max(1e-6, yaw_stop - yaw_ok),
                0.0, 1.0,
            ))
            v_use = v_push * par
            # wz ONLY squares the camera to the face -- NEVER capped by
            # the motion cone. Log 20260813_224304: the old cap
            # kappa*max(|v|,0.05) froze wz at 0.075 rad/s when the stick
            # was released, yaw_err ran to +69 deg and the tags dropped
            # out (user: 摄像头必须平行于箱面). Steering is the LEG's
            # contact offset e; when the box rotates the face normal
            # rotates and this servo swings the tail to follow (甩尾).
            wz = float(np.clip(
                float(getattr(cfg, "push_kp_wz", 1.5)) * yaw_err,
                -wz_max, wz_max,
            ))
            _, raw_tgt_dy = self._l2d_xy(raw_tgt[0], raw_tgt[1])
            v_lat = float(np.clip(
                float(getattr(cfg, "push_kp_vy", 0.8)) * raw_tgt_dy,
                -float(getattr(cfg, "push_v_max_mps", 0.25)),
                float(getattr(cfg, "push_v_max_mps", 0.25)),
            ))
            v_max = float(getattr(cfg, "push_v_max_mps", 0.25))
            vx = float(np.clip(
                v_use * n_h[0] + v_lat * r_h[0], -v_max, v_max
            ))
            vy = float(np.clip(
                v_use * n_h[1] + v_lat * r_h[1], -v_max, v_max
            ))
            self._apply_push_twist(vx, vy, wz, dt, control_enabled)

            # ---- Leg: track contact = face center + e*right + depth*n.
            # THIS is the steering actuator (contact offset e -> torque
            # about the box friction center). LPF kills tag jitter; the
            # rate limit only shapes stick-driven moves (entry was a
            # snap, so no slow deploy).
            tau_lpf = max(
                1e-3, float(getattr(cfg, "push_target_lpf_tau_s", 0.25))
            )
            a = min(1.0, float(max(0.0, dt)) / tau_lpf)
            if self._push_tgt_lpf is None:
                self._push_tgt_lpf = np.asarray(raw_tgt, dtype=float).copy()
            else:
                self._push_tgt_lpf = (
                    (1.0 - a) * self._push_tgt_lpf + a * raw_tgt
                )
            goal = self._clamp_push_foot_plane(self._push_tgt_lpf)
            cur = np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
            rate = float(max(
                1e-3, getattr(cfg, "push_foot_rate_mps", 0.30)
            ))
            delta = goal - cur
            dist = float(np.linalg.norm(delta))
            step = rate * float(max(0.0, dt))
            if dist > 1e-6:
                cur = cur + delta * (min(step, dist) / dist)
            self._manip_foot_des_b = self._clamp_push_foot_plane(cur)

        # Leg: soft Cartesian PD onto the (offset) contact pose. AK60
        # MIT kd (manip_ak60_kd) is the velocity damping path.
        tau_send, self._manip_err_m, self._manip_speed_mps = (
            self.core.compute_stand_swing_tau(
                joint_pos=np.asarray(q, dtype=float).reshape(3),
                joint_vel=np.asarray(qd, dtype=float).reshape(3),
                leg_len_des_m=float(np.linalg.norm(self._manip_foot_des_b)),
                tau_max_nm=float(getattr(cfg, "push_tau_max_nm", 3.0)),
                foot_des_b=np.asarray(
                    self._manip_foot_des_b, dtype=float
                ).reshape(3),
                kp_z=float(getattr(cfg, "push_kp_n_m", 200.0)),
                kd_z=float(getattr(cfg, "push_kd_n_s_m", 4.0)),
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
        The stable MOBILE target stays latched through PRE/FACE/PRESS so
        hand occlusion and pose jitter cannot move the goal.
        """
        if not bool(getattr(self.lcm_cfg, "manip_button_auto_enable", False)):
            return None

        # Initialize once from measured FK. During tag acquisition, hold this
        # Cartesian point instead of unexpectedly handing control to sticks.
        if bool(self._manip_init_pending):
            self._update_manip_foot_target(None, joint_pos, 0.0)

        # Acquisition belongs exclusively to MOBILE. Entering BUTTON always
        # carries a frozen READY setpoint; never scan again during HOME/PRE/
        # FACE/PRESS, even if the tag becomes occluded or moves.
        if self._btn_last_setpoint is None:
            if self._btn_stage != "latch_missing":
                jprint(
                    "[manip-btn] ERROR: READY latch missing -> STOP; "
                    "return to MOBILE and acquire again"
                )
            self._btn_stage = "latch_missing"
            self._btn_stage_print = "latch_missing"
            self._manip_press_boost = False
            return np.asarray(
                self._manip_foot_des_b, dtype=float
            ).reshape(3)

        sp = self._btn_last_setpoint
        # Freeze the stable MOBILE latch through PRE/FACE/PRESS. Live updates
        # drifted the target across the workspace while the arm occluded the
        # tag (log 02:30: target moved ~35 cm during one attempt).
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
            # HOME = MOBILE first step 归中 (equal q), NOT the hold-z plane.
            "home": self._manip_home_foot(),
            "pre": self._lock_button_z(sp["foot_pre_L"]),
            "face": self._lock_button_z(sp["foot_face_L"]),
            "press": self._lock_button_z(sp["foot_press_L"]),
        }

        stage = str(self._btn_stage)
        if stage == "home":
            goal_raw = goals["home"]
        elif stage in ("press", "hold", "done"):
            # Restore the working GitHub path: the leg performs the 5 cm
            # wall-normal stroke and then remains at the pressed pose.
            goal_raw = goals["press"]
        elif stage == "face":
            goal_raw = goals["face"]
        elif stage in ("pre", "retract"):
            goal_raw = goals["pre"]
        elif stage in ("align_failed", "aborted"):
            goal_raw = np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
        else:
            self._btn_stage = "home"
            stage = "home"
            goal_raw = goals["home"]

        # PRE/FACE stay in the conservative envelope. PRESS/HOLD restore the
        # GitHub soft 0.62 m radial cap so the arm, not the wheels, can make
        # the short contact stroke. Detected button Z is preserved.
        face_c = self._clamp_work_envelope(goals["face"])
        pre_c = self._clamp_work_envelope(goals["pre"])
        if stage == "home":
            # Joint-space 归中 (MOBILE first step). Cartesian clamp would
            # not keep the three joints equal.
            goal_cmd = np.asarray(goal_raw, dtype=float).reshape(3).copy()
            self._manip_press_boost = False
        elif stage in ("press", "hold", "done"):
            goal_cmd = np.asarray(goals["press"], dtype=float).reshape(3).copy()
            if float(goal_cmd[2]) < 0.05:
                goal_cmd[2] = 0.05
            length = float(np.linalg.norm(goal_cmd))
            if length > 0.62 and length > 1e-9:
                goal_cmd *= 0.62 / length
            self._manip_press_boost = stage in ("press", "hold")
        else:
            goal_cmd = pre_c if stage in ("pre", "retract") else face_c
            self._manip_press_boost = False

        cur = np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
        rate = float(getattr(self.lcm_cfg, "manip_button_rate_mps", 0.0))
        if rate <= 1e-9:
            # Snap like MOBILE stow (AK60 damping, no crawl).
            cur = np.asarray(goal_cmd, dtype=float).reshape(3)
        else:
            delta = goal_cmd - cur
            dist = float(np.linalg.norm(delta))
            step = rate * float(max(0.0, dt))
            if dist > 1e-6:
                cur = cur + delta * (min(step, dist) / dist)
        if stage == "home":
            self._manip_foot_des_b = cur.astype(float)
        elif stage in ("press", "hold", "done"):
            self._manip_foot_des_b = cur.astype(float)
        else:
            self._manip_foot_des_b = self._clamp_work_envelope(cur)

        arrive = float(max(0.001, self.lcm_cfg.manip_button_arrive_m))
        home_arrive = float(max(
            arrive, getattr(self.lcm_cfg, "manip_home_arrive_m", 0.02)
        ))
        # Stage transitions always use measured FK error. Command residual
        # falsely reported 38 cm while measured tracking error was 11 mm
        # (log 02:30), causing timeout-driven presses before alignment.
        p_now = None
        try:
            if self.core.fk is not None:
                p_now, _ = self.core.fk.forward_kinematics(
                    np.asarray(joint_pos, dtype=float).reshape(3)
                )
                p_now = np.asarray(p_now, dtype=float).reshape(3)
        except Exception:
            p_now = None
        if rate <= 1e-9:
            if p_now is not None and np.all(np.isfinite(p_now)):
                dist = float(np.linalg.norm(goal_cmd - p_now))
            else:
                dist = float(np.linalg.norm(
                    goal_cmd - np.asarray(
                        self._manip_foot_des_b, dtype=float
                    ).reshape(3)
                ))
        else:
            dist = float(np.linalg.norm(
                goal_cmd - np.asarray(
                    self._manip_foot_des_b, dtype=float
                ).reshape(3)
            ))
            gate = home_arrive if stage == "home" else arrive
            if dist <= gate:
                self._manip_foot_des_b = np.asarray(
                    goal_cmd, dtype=float
                ).reshape(3)
                dist = 0.0
        measured_dist = (
            float(np.linalg.norm(goal_cmd - p_now))
            if p_now is not None and np.all(np.isfinite(p_now))
            else float(dist)
        )

        if stage != str(getattr(self, "_btn_stage_seen", "")):
            self._btn_stage_t0 = time.time()
            self._btn_stage_seen = str(stage)
        timeout_s = float(max(
            0.2, getattr(self.lcm_cfg, "manip_button_stage_timeout_s", 1.2)
        ))
        aged = (
            (time.time() - float(self._btn_stage_t0 or time.time()))
            >= timeout_s
        )

        pm_cm = float(getattr(self.lcm_cfg, "manip_button_press_m", 0.05)) * 100.0
        if stage == "home":
            q_c = float(getattr(self.lcm_cfg, "mobile_leg_center_q", 0.4))
            q_err = float(np.max(np.abs(
                np.full(3, q_c, dtype=float)
                - np.asarray(joint_pos, dtype=float).reshape(3)
            )))
            arrive_q = float(getattr(
                self.lcm_cfg, "mobile_leg_arrive_rad", 0.10
            ))
            if q_err <= arrive_q:
                self._btn_stage = "pre"
                jprint(
                    "[manip-btn] CENTER/归中 arrived "
                    f"(|e|_max={q_err:.3f} rad) -> pre"
                )
        elif stage == "pre" and measured_dist <= arrive:
            self._btn_stage = "face"
            jprint(
                f"[manip-btn] pre arrived (err={measured_dist*1000:.0f}mm) "
                "-> FACE (square on button, then press)"
            )
        elif stage == "face" and measured_dist <= arrive:
            self._btn_stage = "press"
            self._press_drive_active = False
            jprint(
                f"[manip-btn] face arrived (err={measured_dist*1000:.0f}mm) "
                f"-> ARM PRESS {pm_cm:.0f} cm (wheels stopped)"
            )
        elif stage in ("pre", "face") and aged:
            self._btn_stage = "align_failed"
            self._press_drive_active = False
            self._manip_press_boost = False
            self._manip_foot_des_b = np.asarray(cur, dtype=float).reshape(3)
            jprint(
                f"[manip-btn] {stage} ALIGN FAILED "
                f"(measured err={measured_dist*1000:.0f}mm) -> STOP; "
                "no wheel press"
            )
        elif stage == "press":
            # The command crawl reaches the requested stroke even when the
            # compliant contact prevents measured FK from reaching the target.
            if dist <= arrive or aged:
                self._btn_stage = "hold"
                self._btn_hold_t0 = time.time()
                jprint(
                    "[manip-btn] arm press commanded -> hold "
                    f"{float(self.lcm_cfg.manip_button_hold_s):.2f}s"
                )
        elif stage == "hold":
            t0 = float(self._btn_hold_t0 or time.time())
            if (time.time() - t0) >= float(self.lcm_cfg.manip_button_hold_s):
                self._btn_stage = "done"
                self._manip_press_boost = False
                jprint("[manip-btn] hold done -> STOP (stay at press, no backup)")
        elif stage == "retract" and (dist <= arrive or aged):
            self._btn_stage = "done"
            jprint("[manip-btn] retract done -> STOP (no backup)")

        if bool(self._press_drive_active):
            self._btn_stage_print = "wheel-press"
        else:
            self._btn_stage_print = str(self._btn_stage)
        return np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)

    def _wall_normal_drive_xy(self) -> tuple[float, float]:
        """Unit wall-inward direction in the drive/wheel frame."""
        n_L = getattr(self, "_manip_press_n_L", None)
        if n_L is None and self._btn_last_setpoint is not None:
            n_L = np.asarray(
                self._btn_last_setpoint.get(
                    "wall_normal_in_L", [1.0, 0.0, 0.0]
                ),
                dtype=float,
            ).reshape(3)
        if n_L is None:
            n_L = np.array([1.0, 0.0, 0.0], dtype=float)
        n_L = np.asarray(n_L, dtype=float).reshape(3)
        nn = float(np.linalg.norm(n_L[:2]))
        if nn < 1e-9:
            return 1.0, 0.0
        n_dx, n_dy = self._l2d_xy(float(n_L[0]), float(n_L[1]))
        # Detector normal has a 180° ambiguity. Select the hemisphere that
        # points toward the visible button. Missing this guard produced
        # drive=(-1,0), i.e. the "press" drove backward (log 02:30).
        if self._btn_last_setpoint is not None:
            face = np.asarray(
                self._btn_last_setpoint.get("foot_face_L", [1.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(3)
            face_dx, face_dy = self._l2d_xy(float(face[0]), float(face[1]))
            if n_dx * face_dx + n_dy * face_dy < 0.0:
                n_dx, n_dy = -n_dx, -n_dy
        nh = math.hypot(n_dx, n_dy) or 1.0
        return n_dx / nh, n_dy / nh

    def _start_button_wheel_press(self) -> None:
        """Creep the base into the wall; foot stays on FACE."""
        pm = float(max(
            0.02, getattr(self.lcm_cfg, "manip_button_press_m", 0.05)
        ))
        n_dx, n_dy = self._wall_normal_drive_xy()
        self._press_dir_D = (n_dx, n_dy)
        self._press_drive_remain_m = pm
        self._press_drive_active = True
        self._press_drive_t0 = time.monotonic()
        self._press_drive_dbg_t = 0.0
        v = float(getattr(self.lcm_cfg, "manip_button_press_v_mps", 0.10))
        jprint(
            f"[manip-btn] WHEEL PRESS {pm*100:.0f} cm along "
            f"drive=({n_dx:+.2f},{n_dy:+.2f}) at {v:.2f} m/s "
            "(foot holds FACE)"
        )

    def _update_button_wheel_press(self, dt: float) -> np.ndarray | None:
        """Open-loop forward creep during PRESS. Returns wheel rad/s or None."""
        if not bool(self._press_drive_active):
            return None
        v = float(max(
            0.02, getattr(self.lcm_cfg, "manip_button_press_v_mps", 0.10)
        ))
        dx, dy = self._press_dir_D
        self._press_drive_remain_m -= v * float(max(0.0, dt))
        now = time.monotonic()
        timed_out = (now - float(self._press_drive_t0 or now)) >= 4.0
        if (now - float(self._press_drive_dbg_t)) >= 0.3:
            self._press_drive_dbg_t = now
            jprint(
                f"[manip-btn] wheel press remain="
                f"{max(0.0, self._press_drive_remain_m)*100:.0f} cm "
                f"drive=({dx:+.2f},{dy:+.2f})"
            )
        if self._press_drive_remain_m <= 0.0 or timed_out:
            self._press_drive_active = False
            self._press_drive_remain_m = 0.0
            why = "timeout" if timed_out else "arrived"
            jprint(f"[manip-btn] wheel press {why}")
            return np.zeros(3, dtype=float)
        return self._kiwi_ik(v * float(dx), v * float(dy), 0.0)

    def _start_post_press_backup(self) -> None:
        """Leave MANIPULATION and reverse wheels along -n (away from wall)."""
        if not bool(getattr(self.lcm_cfg, "manip_backup_enable", False)):
            jprint("[manip-btn] backup disabled -> STOP")
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
        self._manip_hold_q_cmd = None
        self._manip_press_boost = False
        self._reset_mobile_leg_stow()
        # Keep button state as done; clear latch so a new LT can re-acquire.
        self._btn_stage = "done"
        self._btn_stage_print = "backup"
        self._btn_last_setpoint = None
        jprint(
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
            jprint("[backup] OFF (stick override)")
            return None

        v = float(max(0.02, getattr(self.lcm_cfg, "manip_backup_v_mps", 0.12)))
        dx, dy = self._backup_dir_D
        vx, vy = v * float(dx), v * float(dy)
        self._backup_remain_m -= v * float(max(0.0, dt))
        now = time.monotonic()
        if (now - float(self._backup_dbg_t)) >= 0.5:
            self._backup_dbg_t = now
            jprint(
                f"[backup] remain={max(0.0, self._backup_remain_m)*100:.0f} cm "
                f"v_D=({vx:+.2f},{vy:+.2f})"
            )
        if self._backup_remain_m <= 0.0:
            self._backup_active = False
            self._backup_remain_m = 0.0
            jprint("[backup] done -> MOBILE manual")
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
                jprint(
                    "[manip] target initialized from FK p_b=[%+.3f,%+.3f,%+.3f] m"
                    % tuple(float(x) for x in foot_now)
                )
            except Exception as exc:
                self._manip_foot_des_b = np.array(
                    [0.0, 0.0, float(self.lcm_cfg.switch_rb_leg_len_m)],
                    dtype=float,
                )
                jprint("[manip] FK init failed (%s); using vertical target" % exc)
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
        q_des: np.ndarray | None = None,
        kp_joint: np.ndarray | None = None,
        kd_joint: np.ndarray | None = None,
    ) -> None:
        msg = hopper_cmd_lcmt()
        msg.tau_ff = [float(t) for t in np.asarray(tau_cmd, dtype=float).reshape(3)]
        if q_des is None:
            msg.q_des = [0.0, 0.0, 0.0]
        else:
            msg.q_des = [
                float(x) for x in np.asarray(q_des, dtype=float).reshape(3)
            ]
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
        jprint(
            "[rm] %s: current physical pose is now logical q=%+.1f rad (0 A)"
            % (label, self._rm_zero_at_rad)
        )

    def _rm_start(self, target_rad: float, label: str) -> None:
        """Drive all folding arms to an absolute logical endpoint."""
        self._rm_target = float(target_rad)
        self._rm_sync_q0 = None   # re-latch start poses on the first drive tick
        self._rm_stage = 2
        self._rm_drive_t0 = time.time()
        jprint("[rm] %s -> drive to %+.1f rad (kp %.1f kd %.2f cap %.1fA, "
              "sync lead %.1f rad, timeout %.1fs)"
              % (label, self._rm_target, float(self.lcm_cfg.rm_kp_a_per_rad),
                 float(self.lcm_cfg.rm_kd_a_per_rad_s),
                 float(self.lcm_cfg.rm_iq_max_a),
                 float(getattr(self.lcm_cfg, "rm_sync_lead_rad", 1.5)),
                 float(getattr(self.lcm_cfg, "rm_drive_timeout_s", 3.0))))

    def _rm_max_abs_err_rad(self) -> float:
        """Largest |q_i - target| among the three RM arms (rad).

        Used as the "slowest arm remaining" distance for the RT-STAND
        early hop handoff (switch_rt_stand_rm_near_deg).
        """
        with self.lock:
            rm_q = np.asarray(
                self.robot_state["rm_q"], dtype=float
            ).reshape(3).copy()
        if not np.all(np.isfinite(rm_q)):
            return float("inf")
        return float(np.max(np.abs(rm_q - float(self._rm_target))))

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
                jprint("[rm] pre-zero done (q=[%+.3f %+.3f %+.3f]) -> drive to %+.1f rad"
                      % (*rm_q, target))
            return
        if int(self._rm_stage) == 2:
            timeout_s = float(max(
                0.0, getattr(self.lcm_cfg, "rm_drive_timeout_s", 3.0)
            ))
            if (
                timeout_s > 0.0
                and self._rm_drive_t0 is not None
                and (now_t - float(self._rm_drive_t0)) >= timeout_s
            ):
                self._rm_stage = 0
                self._rm_drive_t0 = None
                self._rm_sync_q0 = None
                self._rm_iq_des = np.zeros(3, dtype=float)
                jprint(
                    "[rm] drive TIMEOUT %.1fs: target=%+.1f "
                    "q=[%+.3f %+.3f %+.3f] -> stage 0, current OFF; "
                    "LT transition unblocked"
                    % (timeout_s, target, *rm_q)
                )
                return
            if bool(np.all(np.abs(rm_q - target) <= float(self.lcm_cfg.rm_reach_tol_rad))):
                self._rm_stage = 3
                self._rm_drive_t0 = None
                self._rm_hold_t0 = now_t
                jprint("[rm] reached %+.1f rad -> hold %.1fs, then re-zero"
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
                self._rm_drive_t0 = None
                self._rm_iq_des = np.zeros(3, dtype=float)
                jprint(
                    "[rm] endpoint held: q=[%+.3f %+.3f %+.3f] -> idle (0 A)"
                    % tuple(rm_q)
                )
                return
        self._rm_iq_des = np.clip(kp * (target - rm_q) - kd * rm_qd, -cap, cap)

    def _update_rm_station_hold(self) -> None:
        """Idle-stage station keeping at the current gait endpoint.

        HOPPING: hold 0 rad so hop impacts cannot back-drive the arms.
        MOBILE / PUSH / deployed hold: hold +11.5 so wheel motion cannot
        back-drive the folded arms. Same low-cap current PD as hopping;
        at the endpoint current is ~0 A, only disturbances draw current.
        """
        PD, PWMPD = 2, 3
        gait = str(self._gait_mode)
        hop_ok = (
            bool(getattr(self.lcm_cfg, "rm_hopping_hold", True))
            and gait == "hopping"
            and not bool(self._rm_lt_pending)
            and not bool(self._rm_rt_pending)
            and not bool(self._switch_loop)
            and self._lt_stand_t0 is None
        )
        mobile_ok = (
            bool(getattr(self.lcm_cfg, "rm_mobile_hold", True))
            and gait in ("mobile", "push", "manipulation")
            and self._lt_stand_t0 is None
            and self._rt_stand_t0 is None
            and not bool(self._switch_loop)
        )
        if hop_ok:
            target = float(self.lcm_cfg.rm_hopping_rad)
        elif mobile_ok:
            target = float(self.lcm_cfg.rm_mobile_rad)
        else:
            self._rm_iq_des = np.zeros(3, dtype=float)
            return
        if int(self._mode_est) not in (PD, PWMPD):
            self._rm_iq_des = np.zeros(3, dtype=float)
            return
        with self.lock:
            rm_q = np.asarray(self.robot_state["rm_q"], dtype=float).reshape(3).copy()
            rm_qd = np.asarray(self.robot_state["rm_qd"], dtype=float).reshape(3).copy()
            rm_online = int(self.robot_state["rm_online"])
        if rm_online != 7:
            self._rm_iq_des = np.zeros(3, dtype=float)
            return
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

    def _handle_y_trigger(self, gamepad_msg) -> None:
        y_now = False
        try:
            y_now = bool(getattr(gamepad_msg, "y", 0)) if gamepad_msg is not None else False
        except Exception:
            y_now = False

        # Rising edge on Y:
        # - reset controller estimates (v_hat, integrators, etc.)
        # - enable velocity hard-hold until we enter STANCE once
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
        self._last_y = bool(y_now)

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
        if bool(getattr(
            self.core.cfg, "foot_clearance_control_enable", False
        )):
            hop_h = float(self.core.cfg.foot_clearance_target_m)
            hop_tag = "foot_clear"
        else:
            hop_h = float(self.core.cfg.hop_height_m)
            hop_tag = "hop_height"
        if np.all(np.isfinite(foot_vicon)):
            leg_len = float(np.linalg.norm(foot_vicon))
            line = (
                f"[{gait_tag}|{mode_tag}|{ph}] "
                f"leg_len={leg_len:.4f} m {hop_tag}={hop_h:.3f} m"
            )
        else:
            line = (
                f"[{gait_tag}|{mode_tag}|{ph}] "
                f"leg_len=nan {hop_tag}={hop_h:.3f} m"
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
        if self._gait_mode == "hopping":
            if bool(getattr(self.core.cfg, "flight_deadbeat_model", False)):
                line += (
                    " β="
                    f"{float(self.core.cfg.flight_deadbeat_beta):.2f}/"
                    f"{float(self.core.cfg.flight_deadbeat_beta_hi):.2f}"
                )
            else:
                line += " β=off(Raibert)"
        if self._gait_mode == "mobile":
            w = np.asarray(
                getattr(self, "_wheel_last_cmd", np.zeros(3)), dtype=float
            ).reshape(3)
            en = "on" if bool(getattr(self, "_wheel_last_enable", False)) \
                else "off"
            if bool(getattr(self, "_mobile_leg_holding", False)):
                leg_st = "hold"
            elif bool(getattr(self, "_mobile_leg_active", False)):
                phase = str(getattr(self, "_mobile_leg_phase", "stow"))
                leg_st = phase if phase in ("center", "lift") else "stow"
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
                    elif (self._approach_kind == "box"
                            and bool(self._box_ready)):
                        line += " approach=READY->PUSH"
                    elif self._approach_kind == "box":
                        line += " approach=BOX-SERVO"
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
                    # Working-pose errors (yaw parallelism + distance):
                    # visible even without the auto approach running.
                    if np.isfinite(self._box_dist_err_m):
                        line += (
                            f" dErr={self._box_dist_err_m * 100:+.1f}cm"
                        )
                    if np.isfinite(self._box_yaw_err_deg):
                        line += f" yawErr={self._box_yaw_err_deg:+.1f}deg"
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
            state = self._push_state_str() or "TRACK"
            f_str = (
                f"{self._push_f_meas_n:+.1f}N"
                if np.isfinite(self._push_f_meas_n) else "--"
            )
            ft_str = (
                f"{self._push_f_t_lpf:+.1f}N"
                if np.isfinite(self._push_f_t_lpf) else "--"
            )
            line += (
                f" push[{src}|{state} age={age:.1f}s]"
                f" F={f_str}/Ft={ft_str}"
                f" depth={self._push_depth_cmd_m * 100:.1f}cm"
                f" e={self._push_e_des_m * 100:+.1f}cm"
                f" v={self._push_v_last:+.2f}m/s"
                f" foot={p[0]:+.3f}/{p[1]:+.3f}/{p[2]:+.3f} m"
                f" err={self._manip_err_m * 1000:.1f}mm"
                f" wheels[{en}]={w[0]:+.1f}/{w[1]:+.1f}/{w[2]:+.1f} rad/s"
            )
        elif self._gait_mode == "manipulation":
            kind = str(getattr(self, "_manip_kind", "hold"))
            if kind == "hold":
                qh = np.asarray(
                    getattr(
                        self.lcm_cfg, "manip_hold_q",
                        (1.376, 1.378, -0.096),
                    ),
                    dtype=float,
                ).reshape(3)
                err = (
                    float(self._manip_err_m)
                    if np.isfinite(self._manip_err_m) else float("nan")
                )
                line += (
                    f" HOLD q_des={qh[0]:+.3f}/{qh[1]:+.3f}/{qh[2]:+.3f}"
                    f" q_err={err:.3f} rad"
                    f" kp={float(getattr(self.lcm_cfg, 'manip_hold_kp', 20.0)):.0f}"
                    f" kd={float(self.lcm_cfg.manip_ak60_kd):.1f}"
                )
            else:
                p = np.asarray(self._manip_foot_des_b, dtype=float).reshape(3)
                btn = str(getattr(self, "_btn_stage_print", "") or "")
                line += (
                    f" BUTTON p={p[0]:+.3f}/{p[1]:+.3f}/{p[2]:+.3f} m"
                    f" L={np.linalg.norm(p):.3f}"
                    f" err={self._manip_err_m*1000:.1f} mm"
                )
                if btn:
                    line += f" btn={btn}"
        return line

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

                # Y resets controller estimates and enables the temporary
                # velocity hold; it no longer controls CSV logging.
                self._handle_y_trigger(gamepad_msg)
                # User "I"/point hard-stop (also works even before we have all packets)
                self._handle_zero_vel_trigger(gamepad_msg)
                # LEFT stick detent: queue foot_clearance for the next TD.
                self._handle_hop_height_stick(gamepad_msg)

                if (not have_motor) or (not have_imu):
                    # Wait for first packets
                    if (now - last_print) > 1.0:
                        last_print = now
                    continue

                desired_v_xy = self._compute_desired_v_xy(gamepad_msg)

                # ===== LiDAR patrol (external enable; stick/B wins back) =====
                # SELECT/Back no longer toggles patrol — see hopper-gamepad-upper.
                b_now_pat = bool(getattr(gamepad_msg, "b", 0)) if gamepad_msg is not None else False
                stick_moved = bool(np.any(np.abs(desired_v_xy) > 1e-9))  # already deadzoned
                if self._patrol_enable and (b_now_pat or stick_moved):
                    self._patrol_enable = False
                    jprint(f"[patrol] OFF ({'B' if b_now_pat else 'stick override'})")
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

                    # disarm props + force DAMP in driver
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    self._publish_motor_pwm(np.full(6, pwm_min, dtype=float), control_mode=-1, force=True)
                    # LATCH props OFF: clear the arm flag so props stay stopped after the SAFE
                    # pause and only re-arm on a deliberate A press (otherwise they spin again).
                    self._prop_enable = False
                    self._close_prop_base_window("SAFE")
                    # damping-like joints (in case driver is still in PD/PWMPD)
                    kd = float(self.lcm_cfg.safe_damp_kd)
                    self._publish_hopper_cmd(np.zeros(3, dtype=float), kp_joint=np.zeros(3, dtype=float), kd_joint=np.full(3, kd, dtype=float))

                    # Update our mode estimate to DAMP to prevent repeated SAFE triggers while the operator recovers.
                    self._mode_est = 1  # DAMP
                    self._safe_last_t = float(time.time())
                    jprint(f"[SAFE] TRIGGERED: {'; '.join(reason)} -> DAMP (pause {pause_s:.0f}s)")
                    # pause controller loop (do NOT backlog-catchup)
                    if pause_s > 0.0:
                        time.sleep(pause_s)
                    next_t = time.time() + dt
                    last_print = time.time()
                    continue

                control_enabled = int(self._mode_est) in (2, 3)
                # ModeE may use prop assistance in enabled HOPPING, and also
                # during RT-STAND (gait still mobile) so the 1600 collective
                # base can attitude-balance like a normal stance.
                core_props_armed = bool(
                    control_enabled
                    and self._prop_enable
                    and (
                        self._gait_mode == "hopping"
                        or self._rt_stand_t0 is not None
                    )
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

                # RT first-N hop_height override: count completed pushes.
                if (int(self._rt_hop_height_remaining) > 0
                        and int(info.get("liftoff", 0)) != 0):
                    self._rt_hop_height_remaining = int(
                        self._rt_hop_height_remaining
                    ) - 1
                    if int(self._rt_hop_height_remaining) <= 0:
                        self._restore_hop_height("RT first hops done")
                    else:
                        jprint(
                            "[hop_height] RT override remaining %d hop(s) "
                            "at %.3f m"
                            % (int(self._rt_hop_height_remaining),
                               float(self.core.cfg.hop_height_m))
                        )
                # First-hop l0 (0.43) stays through RM-unfold FLIGHT hold.
                # Restore to normal l0 (0.455) only after a REAL touchdown
                # then liftoff — not the fake LO at P4 handoff (log 152156).
                if self._rt_l0_restore is not None:
                    if int(info.get("touchdown", 0)) != 0:
                        self._rt_first_hop_l0_saw_td = True
                    if (bool(self._rt_first_hop_l0_saw_td)
                            and int(info.get("liftoff", 0)) != 0):
                        self._restore_rt_first_hop_l0()

                # Enabled HOPPING + LT (stand-and-fold), 2026-08-10:
                #   compression (TD) -> start RM fold (legs keep hopping)
                #   that stance's liftoff -> props OFF + stand hold
                # Exactly the NEXT stance entry (TD edge), never merely
                # "currently in COMP".  If LT is pressed midway through a
                # stance, that stance is left untouched and folding waits for
                # the following TD.
                lt_compress = int(info.get("touchdown", 0)) != 0
                if bool(self._rm_lt_pending) and lt_compress:
                    self._rm_lt_pending = False
                    self._rm_rt_pending = False
                    self._lt_await_lo_stand = True
                    self._rm_start(
                        float(self.lcm_cfg.rm_mobile_rad),
                        "LT job (compression -> fold RM)",
                    )
                    jprint(
                        "[gait] LT compression -> RM 0 -> %.1f rad; "
                        "legs finish this hop; props OFF + stand at liftoff"
                        % float(self.lcm_cfg.rm_mobile_rad)
                    )
                elif (bool(self._lt_await_lo_stand)
                        and int(info.get("liftoff", 0)) != 0):
                    self._lt_await_lo_stand = False
                    self._lt_stand_t0 = time.time()
                    self._restore_hop_height("LT stand")
                    self._prop_enable = False
                    self._close_prop_base_window("LT stand")
                    jprint(
                        "[gait] LT STAND (liftoff after fold stance) -> "
                        "leg holds %.3f m (tau cap %.1f Nm, props OFF); "
                        "MOBILE when RM reaches %.1f rad"
                        % (float(self.lcm_cfg.switch_lt_leg_len_m),
                           float(self.lcm_cfg.switch_lt_tau_max_nm),
                           float(self.lcm_cfg.rm_mobile_rad))
                    )
                # RT: after P4 handoff, wait for the first push to finish
                # (liftoff) before unfolding RM +11.5 -> 0.
                elif bool(self._rm_rt_pending) and int(info.get("liftoff", 0)) != 0:
                    self._rm_rt_pending = False
                    self._rm_start(
                        float(self.lcm_cfg.rm_hopping_rad),
                        "RT job (push end -> unfold RM)",
                    )
                    jprint(
                        "[gait] RT: push end -> RM +11.5 -> %.1f rad"
                        % float(self.lcm_cfg.rm_hopping_rad)
                    )

                if self._lt_stand_t0 is not None:
                    # LT stand-and-fold: entered at liftoff AFTER RM fold
                    # started at compression and the legs finished that
                    # hop's push.  LEG holds for landing; RM may still be
                    # folding; props OFF for the whole stand.
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    tau_send, _err_m, _spd = self.core.compute_stand_swing_tau(
                        joint_pos=q_arr,
                        joint_vel=qd_arr,
                        leg_len_des_m=float(self.lcm_cfg.switch_lt_leg_len_m),
                        tau_max_nm=float(self.lcm_cfg.switch_lt_tau_max_nm),
                        imu_quat_wxyz=imu_quat,
                    )
                    tau_out_scale_applied = 1.0
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    pwm_us = np.full(6, pwm_min, dtype=float)
                    props_active = False
                    # Release to MOBILE once the fold drive is done. The
                    # 1 s stage-3 hold is insurance and may overlap the
                    # first wheel command after the gait change.
                    rm_done = not self._rm_drive_busy()
                    timed_out = (
                        time.time() - float(self._lt_stand_t0)
                    ) >= float(self.lcm_cfg.switch_lt_timeout_s)
                    if rm_done or timed_out:
                        self._lt_stand_t0 = None
                        self._gait_mode = "mobile"
                        self._reset_hop_estimator("LT STAND -> MOBILE")
                        self._reset_mobile_leg_stow()
                        props_active = False
                        jprint(
                            "[gait] LT STAND done (%s) -> MOBILE: "
                            "props OFF; leg stows after wheel motion"
                            % ("RM reached" if rm_done else "TIMEOUT")
                        )
                elif self._rt_stand_t0 is not None:
                    # RT first "hop" = long ground support:
                    #   - legs: P4-law tumbler (world-vertical foot) with
                    #     length ramp 0.448 -> 0.55 (lift without takeoff)
                    #   - props: ModeE allocator around hop_prop_base 1600
                    #     (attitude balance like stance; pwm_us from step)
                    #   - RM unfolds in parallel; 收腿/HOPPING when the
                    #     slowest arm is within switch_rt_stand_rm_near_deg
                    #     AND elapsed >= min_s (RM keeps driving to 0 /
                    #     timeout after the handoff).
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    stand_elapsed = time.time() - float(self._rt_stand_t0)
                    l_start = float(self.lcm_cfg.switch_rb_leg_len_m)
                    l_tgt = float(self.lcm_cfg.switch_rt_stand_leg_len_m)
                    rate = float(max(1e-3, float(
                        self.lcm_cfg.switch_rt_stand_extend_rate_mps
                    )))
                    if l_tgt >= l_start:
                        l_des = min(l_tgt, l_start + rate * stand_elapsed)
                    else:
                        l_des = max(l_tgt, l_start - rate * stand_elapsed)
                    tau_send, _err_m, _spd = self.core.compute_stand_swing_tau(
                        joint_pos=q_arr,
                        joint_vel=qd_arr,
                        leg_len_des_m=float(l_des),
                        tau_max_nm=float(self.lcm_cfg.switch_rt_stand_tau_max_nm),
                        imu_quat_wxyz=imu_quat,
                    )
                    tau_out_scale_applied = 1.0
                    # Keep ModeE pwm_us from step, but enforce the same
                    # hard per-motor floor used during the hopping RM-unfold
                    # window (hop_prop_base_pwm_min_us = 1600).
                    floor = float(getattr(
                        self.lcm_cfg, "hop_prop_base_pwm_min_us", 1600.0
                    ))
                    pwm_us = np.asarray(pwm_us, dtype=float).reshape(6)
                    for grp in self.modee_cfg.prop_pwm_idx_per_arm:
                        for idx in grp:
                            ii = int(idx)
                            if 0 <= ii < 6:
                                pwm_us[ii] = max(float(pwm_us[ii]), floor)
                    props_active = True
                    near_deg = float(getattr(
                        self.lcm_cfg, "switch_rt_stand_rm_near_deg", 10.0
                    ))
                    rm_near = (
                        int(self._rm_stage) == 2
                        and self._rm_max_abs_err_rad()
                        <= float(np.deg2rad(max(0.0, near_deg)))
                    ) or int(self._rm_stage) != 2
                    min_s = float(max(0.0, float(getattr(
                        self.lcm_cfg, "switch_rt_stand_min_s", 2.0
                    ))))
                    timed_out = stand_elapsed >= float(
                        self.lcm_cfg.switch_rt_stand_timeout_s
                    )
                    if (rm_near and stand_elapsed >= min_s) or timed_out:
                        self._rt_stand_t0 = None
                        self._gait_mode = "hopping"
                        # Fresh ModeE hop cycle: stand was NOT a hop.
                        # RM drive is intentionally LEFT RUNNING so the
                        # arms finish to 0 (or hit rm_drive_timeout).
                        # Zero leftover MOBILE/last-hop velocity so the
                        # first hop does not inherit a residual v_xy.
                        self._reset_hop_estimator(
                            "RT-STAND -> HOPPING", new_session=True
                        )
                        self._prop_enable = True
                        self._arm_rt_first_hops("RT-STAND")
                        self._arm_rt_first_hop_l0("RT-STAND")
                        why = (
                            "TIMEOUT" if timed_out else
                            ("RM within %.0fdeg (slowest); RM continues"
                             % float(getattr(
                                 self.lcm_cfg,
                                 "switch_rt_stand_rm_near_deg",
                                 10.0,
                             )))
                        )
                        jprint(
                            "[gait] RT STAND done (%s) -> HOPPING "
                            "(收腿/normal cycle; prop base stays until "
                            "RM@0 or timeout)"
                            % why
                        )
                elif self._switch_loop:
                    # RT (P4): first retract to the short alignment length
                    # while returning the foot WORLD-vertical. Once aligned
                    # (or timeout), extend and hold at switch_rb_leg_len_m
                    # (= 0.45 normal l0), then enter FLIGHT/HOPPING with
                    # RM unfold + prop base 1600.
                    q_arr = np.asarray(q, dtype=float).reshape(3)
                    qd_arr = np.asarray(qd, dtype=float).reshape(3)
                    now_t = time.time()
                    if (not bool(self._rb_p4_aligning)) and \
                       (self._rb_p4_t0 is not None) and \
                       (now_t - float(self._rb_p4_t0)) >= float(self.lcm_cfg.switch_rb_pushdelay_s):
                        # 2026-08-10 21:40 (user): P4 held at normal l0
                        # (0.45); enter FLIGHT/HOPPING.  RM unfolds and
                        # prop base 1600 run in parallel.  First hop is
                        # normal hopping (no height/spring/kd specials).
                        self._switch_loop = False
                        self._rb_p4_t0 = None
                        self._rb_p4_aligning = False
                        self._gait_mode = "hopping"
                        self._reset_hop_estimator(
                            "P4 -> HOPPING", new_session=True
                        )
                        self._prop_enable = True
                        self._open_prop_base_window()
                        self._arm_rt_first_hops("P4")
                        self._arm_rt_first_hop_l0("P4")
                        self._rm_start(
                            float(self.lcm_cfg.rm_hopping_rad),
                            "RT: unfold RM alongside hopping (flight)",
                        )
                        jprint(
                            "[gait] RT -> FLIGHT/HOPPING: hold was L=%.3f m "
                            "(= normal l0), RM +11.5 -> %.1f + prop base "
                            "%.0fus; normal hopping from here"
                            % (
                                float(self.lcm_cfg.switch_rb_leg_len_m),
                                float(self.lcm_cfg.rm_hopping_rad),
                                float(self.lcm_cfg.hop_prop_base_pwm_us),
                            )
                        )
                    p4_leg_len_m = (
                        float(self.lcm_cfg.switch_rb_align_leg_len_m)
                        if bool(self._rb_p4_aligning)
                        else float(self.lcm_cfg.switch_rb_leg_len_m)
                    )
                    tau_send, _err_m, _spd = self.core.compute_stand_swing_tau(
                        joint_pos=q_arr,
                        joint_vel=qd_arr,
                        leg_len_des_m=p4_leg_len_m,
                        tau_max_nm=float(self.lcm_cfg.switch_rb_tau_max_nm),
                        imu_quat_wxyz=imu_quat,
                    )
                    if bool(self._rb_p4_aligning):
                        align_elapsed_s = (
                            now_t - float(self._rb_p4_t0)
                            if self._rb_p4_t0 is not None else 0.0
                        )
                        align_reached = float(_err_m) <= float(
                            self.lcm_cfg.switch_rb_align_tol_m
                        )
                        align_timed_out = align_elapsed_s >= float(
                            self.lcm_cfg.switch_rb_align_timeout_s
                        )
                        if align_reached or align_timed_out:
                            self._rb_p4_aligning = False
                            self._rb_p4_t0 = now_t
                            jprint(
                                "[switch_loop] P4 upright/short stage done "
                                "(%s, err %.3fm) -> extend and hold L=%.3fm"
                                % (
                                    "reached" if align_reached else "TIMEOUT",
                                    float(_err_m),
                                    float(self.lcm_cfg.switch_rb_leg_len_m),
                                )
                            )
                    tau_out_scale_applied = 1.0
                    pwm_min = float(self.modee_cfg.pwm_min_us)
                    pwm_us = np.full(6, pwm_min, dtype=float)
                    # Force the upward prop baseline during P4 place.
                    prop_pwm_up = float(self.lcm_cfg.switch_rb_prop_base_pwm_us)
                    for grp in self.modee_cfg.prop_pwm_idx_per_arm:
                        for idx in grp:
                            ii = int(idx)
                            if 0 <= ii < 6:
                                pwm_us[ii] = prop_pwm_up
                    props_active = True
                elif self._gait_mode == "manipulation":
                    if str(getattr(self, "_manip_kind", "hold")) == "button":
                        manip_target_b = self._update_manip_button_target(q, dt)
                        button_auto = manip_target_b is not None
                        if manip_target_b is None:
                            manip_target_b = self._update_manip_foot_target(
                                gamepad_msg, q, dt
                            )
                        centering = (
                            bool(button_auto)
                            and str(getattr(self, "_btn_stage", ""))
                            in ("home", "wait_tag")
                        )
                        arm_pressing = (
                            bool(button_auto)
                            and str(getattr(self, "_btn_stage", ""))
                            in ("press", "hold", "done")
                        )
                        # WORK ENVELOPE (2026-08-14): every deployed
                        # Cartesian target obeys the user-defined range
                        # (z band, L cap, wheel keep-out sectors). Skip
                        # during 归中 — that is joint-space, not Cartesian.
                        if not centering and not arm_pressing:
                            manip_target_b = self._clamp_work_envelope(
                                manip_target_b
                            )
                            self._manip_foot_des_b = np.asarray(
                                manip_target_b, dtype=float
                            ).reshape(3)
                            # Floor guard: log 01:42 q2 dumped the foot
                            # to z=0.451 and 3 Nm sat against the ground.
                            # Freeze XY and lift if we are already too deep.
                            z_bot = float(getattr(
                                self.lcm_cfg, "work_z_bottom_m", 0.440
                            ))
                            try:
                                p_now, _ = self.core.fk.forward_kinematics(
                                    np.asarray(q, dtype=float).reshape(3)
                                )
                                p_now = np.asarray(
                                    p_now, dtype=float
                                ).reshape(3)
                                if (
                                    np.all(np.isfinite(p_now))
                                    and float(p_now[2]) > (z_bot - 0.005)
                                ):
                                    lift = p_now.copy()
                                    lift[2] = z_bot - 0.020
                                    manip_target_b = self._clamp_work_envelope(
                                        lift
                                    )
                                    self._manip_foot_des_b = (
                                        np.asarray(
                                            manip_target_b, dtype=float
                                        ).reshape(3)
                                    )
                            except Exception:
                                pass
                        boost = bool(
                            button_auto
                            and getattr(self, "_manip_press_boost", False)
                        )
                        if button_auto:
                            tau_cap = float(getattr(
                                self.lcm_cfg, "manip_button_tau_max_nm", 8.0
                            ))
                            kp_c = float(max(
                                50.0,
                                getattr(
                                    self.lcm_cfg, "mobile_leg_kp_hold", 16.0
                                ) * 15.0,
                            ))
                            kd_c = 4.0
                        else:
                            tau_cap = float(self.lcm_cfg.manip_tau_max_nm)
                            kp_c = float(self.lcm_cfg.manip_kp_z_n_m)
                            kd_c = float(self.lcm_cfg.manip_kd_z_n_s_m)
                        f_ff = None
                        _ = boost
                        if centering:
                            tau_send = self._button_center_tau(q, qd)
                            try:
                                p_now, _ = self.core.fk.forward_kinematics(
                                    np.asarray(q, dtype=float).reshape(3)
                                )
                                self._manip_err_m = float(np.linalg.norm(
                                    self._manip_home_foot()
                                    - np.asarray(p_now, dtype=float).reshape(3)
                                ))
                            except Exception:
                                self._manip_err_m = 0.0
                            self._manip_speed_mps = 0.0
                        else:
                            tau_send, self._manip_err_m, self._manip_speed_mps = (
                                self.core.compute_stand_swing_tau(
                                    joint_pos=np.asarray(q, dtype=float).reshape(3),
                                    joint_vel=np.asarray(qd, dtype=float).reshape(3),
                                    leg_len_des_m=float(
                                        np.linalg.norm(manip_target_b)
                                    ),
                                    tau_max_nm=tau_cap,
                                    foot_des_b=manip_target_b,
                                    kp_z=kp_c,
                                    kd_z=kd_c,
                                    axial_ff_n=0.0,
                                    cartesian_pd=bool(button_auto),
                                    f_ff_native=f_ff,
                                )
                            )
                        if not bool(control_enabled):
                            self._press_drive_active = False
                            self._press_drive_remain_m = 0.0
                        w_press = (
                            self._update_button_wheel_press(dt)
                            if bool(control_enabled) else None
                        )
                        if w_press is not None:
                            self._wheel_pending_cmd = np.asarray(
                                w_press, dtype=float
                            ).reshape(3)
                            self._wheel_pending_enable = (
                                bool(control_enabled)
                                and not self._rm_drive_busy()
                            )
                        else:
                            self._wheel_pending_cmd = np.zeros(3, dtype=float)
                            self._wheel_pending_enable = False
                    else:
                        tau_send = np.zeros(3, dtype=float)
                        wheels_on = bool(
                            control_enabled and not self._rm_drive_busy()
                        )
                        w_cmd = (
                            self._compute_wheel_cmd(
                                gamepad_msg, allow_yaw=True
                            )
                            if wheels_on else np.zeros(3, dtype=float)
                        )
                        self._wheel_pending_cmd = np.asarray(
                            w_cmd, dtype=float
                        ).reshape(3)
                        self._wheel_pending_enable = bool(wheels_on)
                        qh = np.asarray(q, dtype=float).reshape(3)
                        q_goal = np.asarray(
                            getattr(
                                self.lcm_cfg, "manip_hold_q",
                                (1.376, 1.378, -0.096),
                            ),
                            dtype=float,
                        ).reshape(3)
                        q_cmd = self._manip_hold_q_cmd
                        if q_cmd is None or (not np.all(np.isfinite(q_cmd))):
                            q_cmd = qh.copy()
                        rate = float(max(
                            0.0,
                            getattr(
                                self.lcm_cfg, "manip_hold_q_rate_rad_s", 0.0
                            ),
                        ))
                        delta = q_goal - q_cmd
                        dist = float(np.linalg.norm(delta))
                        step = rate * float(max(0.0, dt))
                        if dist > 1e-6 and step > 0.0:
                            q_cmd = q_cmd + delta * (min(step, dist) / dist)
                        else:
                            q_cmd = q_goal.copy()
                        self._manip_hold_q_cmd = np.asarray(
                            q_cmd, dtype=float
                        ).reshape(3)
                        self._manip_err_m = float(np.linalg.norm(q_goal - qh))
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
                        control_enabled and not self._rm_drive_busy()
                    )
                    # LT auto-approach servo overrides the sticks while
                    # active (it cancels itself on stick/B and may enter
                    # MANIPULATION on arrival).
                    w_auto = None
                    if wheels_on:
                        # Post-press reverse has priority over approach/sticks.
                        w_auto = self._update_post_press_backup(gamepad_msg, dt)
                        if w_auto is None:
                            if self._approach_kind == "box":
                                w_auto = self._update_box_approach(
                                    gamepad_msg, dt
                                )
                            else:
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
                        gamepad_msg=gamepad_msg,
                        dt=dt,
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
                    # Prop base window: collective rides inside ModeE
                    # (_open_prop_base_window). Hard per-motor PWM floor
                    # while open. Closes on first real hop LO (after TD)
                    # or hop_prop_base_timeout_s -- not on RM unfold done.
                    if bool(self._hop_prop_base_active):
                        floor = float(getattr(
                            self.lcm_cfg, "hop_prop_base_pwm_min_us", 1600.0
                        ))
                        pwm_us = np.asarray(pwm_us, dtype=float).reshape(6)
                        for grp in self.modee_cfg.prop_pwm_idx_per_arm:
                            for idx in grp:
                                ii = int(idx)
                                if 0 <= ii < 6:
                                    pwm_us[ii] = max(float(pwm_us[ii]), floor)
                        if int(info.get("touchdown", 0)) != 0:
                            self._hop_prop_base_saw_td = True
                        t0 = self._hop_prop_base_t0
                        timeout_s = float(max(
                            0.0,
                            getattr(
                                self.lcm_cfg, "hop_prop_base_timeout_s", 1.0
                            ),
                        ))
                        timed_out = (
                            timeout_s > 0.0
                            and t0 is not None
                            and (time.time() - float(t0)) >= timeout_s
                        )
                        first_hop_done = (
                            bool(self._hop_prop_base_saw_td)
                            and int(info.get("liftoff", 0)) != 0
                        )
                        if timed_out:
                            self._close_prop_base_window(
                                "timeout %.1fs" % timeout_s
                            )
                        elif first_hop_done:
                            self._close_prop_base_window("first hop LO")
                # RM M2006 sequence: refresh _rm_iq_des (rides inside hopper_cmd_lcmt below).
                self._update_rm()
                # MOBILE kiwi wheels: off only while RM is mid-drive
                # (stage 2). The 1 s endpoint hold may overlap the first
                # MOBILE wheel command -- that is the insurance hold.
                if self._gait_mode in ("mobile", "push") or (
                    self._gait_mode == "manipulation"
                    and str(getattr(self, "_manip_kind", "hold")) == "hold"
                ):
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
                        0.0,
                        getattr(
                            self.lcm_cfg, "mobile_leg_ak60_kd_move", 2.0
                        )
                        if str(getattr(self, "_manip_kind", "hold"))
                        == "button"
                        else self.lcm_cfg.manip_ak60_kd,
                    ))
                elif self._lt_stand_t0 is not None:
                    kd_use = float(max(0.0, float(
                        self.lcm_cfg.switch_lt_ak60_damp_kd
                    )))
                elif self._rt_stand_t0 is not None:
                    # LT-stand-proven damping for a leg carrying the body.
                    kd_use = float(max(0.0, float(getattr(
                        self.lcm_cfg, "switch_rt_stand_ak60_kd", 2.0
                    ))))
                else:
                    kd_use = float(kd_stance if in_stance else kd_flight)
                    # 第一跳 (n_flights_done<1) 支撑期额外阻尼：
                    # 切换后首次落地时 RM 臂还在展开，多给点耗散。
                    if (in_stance and self._gait_mode == "hopping"
                            and int(getattr(
                                self.core, "_n_flights_done", 1
                            )) < 1):
                        kd_use = float(max(kd_use, float(getattr(
                            self.lcm_cfg,
                            "ak60_first_hop_stance_damp_kd",
                            0.4,
                        ))))
                if (self._gait_mode == "manipulation"
                        and str(getattr(self, "_manip_kind", "hold"))
                        == "hold"):
                    q_hold = self._manip_hold_q_cmd
                    if q_hold is None or (not np.all(np.isfinite(q_hold))):
                        q_hold = np.asarray(
                            getattr(
                                self.lcm_cfg, "manip_hold_q",
                                (1.376, 1.378, -0.096),
                            ),
                            dtype=float,
                        ).reshape(3)
                    kp_hold = float(max(
                        0.0, getattr(self.lcm_cfg, "manip_hold_kp", 20.0)
                    ))
                    kd_hold = float(max(0.0, self.lcm_cfg.manip_ak60_kd))
                    self._publish_hopper_cmd(
                        np.zeros(3, dtype=float),
                        q_des=np.asarray(q_hold, dtype=float).reshape(3),
                        kp_joint=np.full(3, kp_hold, dtype=float),
                        kd_joint=np.full(3, kd_hold, dtype=float),
                    )
                elif kd_use > 0.0:
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
                    jprint(
                        self._format_status_line(
                            info=info,
                            q=q,
                            roll=roll,
                            pitch=pitch,
                            safe_armed=safe_armed,
                            now_t=now,
                        )
                    )

        finally:
            # Zero all outgoing commands so the robot doesn't keep acting on stale values.
            self._publish_zero_outputs()


