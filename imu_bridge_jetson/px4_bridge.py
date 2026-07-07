#!/usr/bin/env python3
"""px4_bridge: read Pixhawk attitude/IMU over MAVLink and publish hopper_imu_lcmt on LCM.

PX4 over USB only streams telemetry after it sees a GCS heartbeat, so we keep one
running in a background thread and explicitly request the message rates we need.

The Pixhawk can be mounted in any orientation (e.g. standing up). --mount-rpy applies
a fixed rotation R_mount (board frame -> robot/body frame) to every quantity so the
published data is already expressed in the robot frame:

  v_robot   = R_mount @ v_board                      (acc, gyro)
  q_robot   = q_board  (x) conj(q_mount)             (attitude)
  rpy_robot = euler(q_robot)

Default mount is pitch +90 deg (Pixhawk standing on its short edge): this maps a
board-frame gravity reading on -X back onto robot +Z.
"""
import os
import sys
import time
import math
import argparse
import threading

from pymavlink import mavutil


def _add_lcm_type_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("HOPPER_LCM_PY", ""),
        os.path.join(here, "..", "hopper_lcm_types", "lcm_types", "python"),
        os.path.expanduser("~/Hopper_srbRL/hopper_lcm_types/lcm_types/python"),
        os.path.expanduser("~/Hopper_srbRL/rl_deploy/lower_lcm/lcm_types/python"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)


_add_lcm_type_paths()
import lcm  # noqa: E402
from hopper_imu_lcmt import hopper_imu_lcmt  # noqa: E402
try:
    from motor_pwm_lcmt import motor_pwm_lcmt  # noqa: E402
except Exception:  # pragma: no cover - props optional if type missing
    motor_pwm_lcmt = None


# MAVLink message ids we want streamed
MSG_ATTITUDE = 30
MSG_ATTITUDE_QUATERNION = 31
MSG_HIGHRES_IMU = 105

# PX4 NSH-over-MAVLink shell (used to drive props via `actuator_test`).
SHELL_DEV = mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL
# flags=0: write to shell without requesting output back (fire-and-forget).
SHELL_FLAGS = 0

# Serialize all MAVLink sends (heartbeat / msg-interval / shell) across threads
# so concurrent writes never interleave and corrupt a frame.
_SEND_LOCK = threading.Lock()

DEG = math.pi / 180.0
RAD = 180.0 / math.pi


# ---- small rotation helpers (pure python, no numpy dependency) ----

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec(A, v):
    return [A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2] for i in range(3)]


def _transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def quat_to_R(q):
    """[w,x,y,z] -> 3x3 rotation matrix (body->world, matching PX4 ATTITUDE_QUATERNION)."""
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def euler_to_R(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll), angles in radians."""
    return _matmul(_rz(yaw), _matmul(_ry(pitch), _rx(roll)))


def R_to_quat(R):
    """3x3 rotation matrix -> quaternion [w, x, y, z]."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2][1] - R[1][2]) / s
        y = (R[0][2] - R[2][0]) / s
        z = (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    return [w, x, y, z]


def quat_mul(a, b):
    """Hamilton product, quaternions as [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def quat_to_euler(q):
    """[w,x,y,z] -> (roll, pitch, yaw) radians, ZYX convention."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def R_to_rpy_frd(R):
    """FRD roll/pitch/yaw from R_wb (body->world). Matches px4_dds_bridge."""
    R = [[float(R[i][j]) for j in range(3)] for i in range(3)]
    roll = math.atan2(-R[2][1], -R[2][2])
    pitch = math.asin(max(-1.0, min(1.0, R[2][0])))
    yaw = -math.atan2(R[1][0], R[0][0])
    return roll, pitch, yaw


def parse_rot(spec):
    """Build an SO(3) matrix from tokens like 'x180,y-90' as the matrix product
    Rx(180) @ Ry(-90) (left-to-right). Returns identity for empty spec."""
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    fns = {"x": _rx, "y": _ry, "z": _rz}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        ax = tok[0].lower()
        deg = float(tok[1:])
        R = _matmul(R, fns[ax](deg * DEG))
    return R


def parse_args():
    p = argparse.ArgumentParser(description="Pixhawk -> LCM IMU bridge")
    p.add_argument("--dev", default="/dev/ttyACM0", help="MAVLink serial device")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--channel", default="hopper_imu_lcmt", help="LCM channel name")
    p.add_argument("--rate", type=float, default=500.0, help="requested IMU stream rate (Hz). "
                   "Only used when this bridge also serves the IMU (NOT in --no-imu split mode).")
    p.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=255")
    p.add_argument("--print-hz", type=float, default=2.0, help="status print rate (0=off)")
    p.add_argument("--imu", dest="imu", action="store_true", default=True,
                   help="publish hopper_imu_lcmt from MAVLink ATTITUDE (default on)")
    p.add_argument("--no-imu", dest="imu", action="store_false",
                   help="props-only mode: IMU now comes from px4_dds_bridge over DDS/TELEM2, "
                        "so this process must NOT also publish hopper_imu_lcmt (avoids two "
                        "publishers fighting on the same channel). Frees the USB link to push "
                        "the prop DO_SET_ACTUATOR stream faster.")
    p.add_argument(
        "--mount-rpy", default="90,0,0",
        help="board->robot mount rotation as roll,pitch,yaw in degrees. This is the "
             "raw-level rotation: it rotates acc/gyro (R_mount @ v) AND the attitude "
             "quaternion (via R_mount^T in R_base), so all outputs stay consistent. "
             "90,0,0 = Rx(+90) applied to the raw board data.",
    )
    p.add_argument(
        "--raw", dest="raw", action="store_true", default=False,
        help="publish the RAW board-frame IMU with ONLY a single SO(3) rotation "
             "(from --rot or --mount-rpy) applied: acc=R@acc, gyro=R@gyro, "
             "quat=quat(R(quat_fc)@R^T). No acc negation, no R_total_fix, no ENU flip.",
    )
    p.add_argument(
        "--rot", default="",
        help="explicit SO(3) as a comma list of <axis><deg> tokens applied as a "
             "MATRIX PRODUCT left-to-right. e.g. 'x180,y-90' = Rx(180) @ Ry(-90). "
             "When set, overrides --mount-rpy. Axis is x/y/z.",
    )
    # ---- propeller path: subscribe motor_pwm_lcmt -> drive DShot via actuator_test ----
    p.add_argument("--props", dest="props", action="store_true", default=True,
                   help="enable motor_pwm_lcmt -> prop (DShot) bridging (default on)")
    p.add_argument("--no-props", dest="props", action="store_false",
                   help="disable prop bridging (IMU only)")
    p.add_argument("--pwm-channel", default="motor_pwm_lcmt",
                   help="LCM channel carrying prop PWM commands")
    p.add_argument("--prop-map", default="1,2,3",
                   help="pwm_values index -> actuator-set index, as a CSV of "
                        "'pwmIdx:setN' or just '1,2,3' meaning pwm[1]->set1, "
                        "pwm[2]->set2, pwm[3]->set3 (DO_SET_ACTUATOR param1..3).")
    p.add_argument("--pwm-min", type=float, default=1000.0, help="pwm_us at 0 throttle")
    p.add_argument("--pwm-max", type=float, default=2000.0, help="pwm_us at full throttle")
    p.add_argument("--prop-rate", type=float, default=150.0,
                   help="MAV_CMD_DO_SET_ACTUATOR re-stream rate (Hz) on the USB MAVLink link. "
                        "Keep at ~150Hz. Each tick is a command_long, and PX4 answers EVERY one "
                        "with a COMMAND_ACK on this SAME USB link -- so the cap is set by the ACK "
                        "return traffic, NOT by whether the IMU shares the link. At 400Hz+ the "
                        "ACK flood congests the link, PX4 throttles/drops, and the actuator "
                        "outputs FREEZE ('pwm stuck'). 150Hz (6.7ms) is far faster than the prop "
                        "time constant (~40ms). The PC still publishes motor_pwm_lcmt at the full "
                        "500Hz control rate; only this MAVLink re-stream is rate-limited.")
    p.add_argument("--prop-timeout", type=float, default=0.6,
                   help="(unused with DO_SET_ACTUATOR; kept for CLI compatibility)")
    p.add_argument("--prop-rx-timeout", type=float, default=0.4,
                   help="stop props if no motor_pwm_lcmt seen for this long (s)")
    p.add_argument("--prop-deadband", type=float, default=0.01,
                   help="throttle below this is treated as OFF")
    p.add_argument("--prop-arm-mode", type=int, default=3,
                   help="motor_pwm_lcmt.control_mode value that means 'props ON'. "
                        "Props spin ONLY when control_mode == this (default 3 = PWMPD). "
                        "Any other value (1=off, -1=DAMP) keeps props stopped.")
    return p.parse_args()


def _parse_prop_map(spec: str) -> dict:
    """Return {pwm_index: px4_motor_number}. Accepts 'm1,m2,m3' (pwm 1/2/3 -> M1/2/3)
    or explicit 'pwmIdx:motorN,...' pairs."""
    spec = (spec or "").strip()
    mapping = {}
    if ":" in spec:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            a, b = part.split(":")
            mapping[int(a)] = int(b)
    else:
        # shorthand: list of motor numbers assigned to pwm indices 1,2,3,...
        motors = [int(x) for x in spec.split(",") if x.strip() != ""]
        for i, mn in enumerate(motors):
            mapping[i + 1] = mn
    return mapping


def _nsh_send(m, line: str):
    """Fire-and-forget a single NSH command line over SERIAL_CONTROL (no response).
    A leading newline flushes any half-typed line and avoids the occasional
    dropped-first-character glitch seen on the PX4 mavlink shell."""
    data = ("\n" + line + "\n").encode()[:70]
    buf = list(data) + [0] * (70 - len(data))
    with _SEND_LOCK:
        m.mav.serial_control_send(SHELL_DEV, SHELL_FLAGS, 0, 0, len(data), buf)


def _start_prop_bridge(m, args, stop: "threading.Event"):
    """Subscribe motor_pwm_lcmt and drive PX4 outputs via MAV_CMD_DO_SET_ACTUATOR.

    pwm_values[i] (us) -> throttle in [0,1] -> actuator value 2*thr-1 in [-1,1] ->
    MAV_CMD_DO_SET_ACTUATOR.param<setIdx>. Requires AUX outputs assigned the function
    "Peripheral via Actuator Set N" (func 300+N) on the FC. The mapping pwm-index ->
    actuator-set index comes from --prop-map (default pwm[1..3] -> set 1..3).
    """
    prop_map = _parse_prop_map(args.prop_map)
    if not prop_map:
        print("!! --prop-map empty; prop bridge disabled", flush=True)
        return
    print(f">>> prop bridge: pwm_idx->motor {prop_map} "
          f"(pwm {args.pwm_min:.0f}->0, {args.pwm_max:.0f}->1)", flush=True)

    span = max(1.0, float(args.pwm_max) - float(args.pwm_min))
    state = {"pwm": [float(args.pwm_min)] * 6, "mode": 0, "rx_t": 0.0}
    lock = threading.Lock()

    def on_pwm(channel, data):
        try:
            mm = motor_pwm_lcmt.decode(data)
        except Exception:
            return
        with lock:
            state["pwm"] = list(mm.pwm_values)
            state["mode"] = int(mm.control_mode)
            state["rx_t"] = time.time()

    lc2 = lcm.LCM(args.lcm_url)
    lc2.subscribe(args.pwm_channel, on_pwm)

    def rx_loop():
        while not stop.is_set():
            try:
                lc2.handle_timeout(200)
            except Exception:
                time.sleep(0.05)

    # prop_map is {pwm_index: actuator_set_index}. With AUX1/2/3 assigned the PX4
    # function "Peripheral via Actuator Set 1/2/3" (func 301/302/303), the binary
    # command MAV_CMD_DO_SET_ACTUATOR param1..param6 drive sets 1..6 directly.
    # We therefore treat the old "motor number" 1/2/3 as the actuator-set index.
    set_idx = dict(prop_map)                        # {pwm_idx: set_index 1..6}
    SEND_HZ = max(20.0, float(args.prop_rate))      # binary cmd is cheap -> stream fast
    NAN = float("nan")
    # Use the numeric MAVLink command id (187): the named constant is missing from
    # some pymavlink dialects (e.g. v10.ardupilotmega on the Jetson), which would
    # otherwise crash the control thread the first time it fires.
    DO_SET_ACTUATOR = getattr(mavutil.mavlink, "MAV_CMD_DO_SET_ACTUATOR", 187)

    def _send_actuators(vals6):
        # vals6: list of 6 floats in [-1,1] (NaN = leave that set untouched).
        with _SEND_LOCK:
            m.mav.command_long_send(
                m.target_system, m.target_component,
                DO_SET_ACTUATOR, 0,
                vals6[0], vals6[1], vals6[2], vals6[3], vals6[4], vals6[5],
                0,                                   # param7 = actuator-set group index
            )

    def ctrl_loop():
        # GENERAL prop drive: one binary MAV_CMD_DO_SET_ACTUATOR per tick carries all
        # throttles. No NuttX shell, no text parsing, no per-motor process spawning ->
        # no jitter/"twitch" and no shell-reset glitches. Output maps value[-1,1] to the
        # AUX MIN..MAX (1000..2000us), so:
        #     throttle 0  -> value -1 -> 1000us (idle: ESC armed & quiet)
        #     throttle 1  -> value +1 -> 2000us (full)
        # control_mode is still the ON/OFF gate: props get >0 throttle only when
        # control_mode == prop_arm_mode (A button); otherwise they idle at value -1.
        # Streaming continuously also gives PX4 a steady signal; on stop/stale we idle.
        tick = 1.0 / SEND_HZ
        while not stop.is_set():
            now = time.time()
            with lock:
                pwm = list(state["pwm"])
                mode = int(state["mode"])
                rx_t = float(state["rx_t"])
            # Watchdog: stale stream -> idle (value -1), never leave props latched hot.
            alive = (rx_t > 0.0) and ((now - rx_t) <= float(args.prop_rx_timeout))
            armed = alive and (mode == int(args.prop_arm_mode))
            vals = [NAN] * 6                          # untouched by default
            for pwm_idx, sidx in set_idx.items():
                if 1 <= sidx <= 6:
                    if armed and (0 <= pwm_idx < len(pwm)):
                        thr = (float(pwm[pwm_idx]) - float(args.pwm_min)) / span
                    else:
                        thr = 0.0                     # idle
                    thr = max(0.0, min(1.0, thr))
                    vals[sidx - 1] = 2.0 * thr - 1.0  # [0,1] throttle -> [-1,1] value
            _send_actuators(vals)
            time.sleep(tick)
        # shutdown: command every used set to idle (-1) a few times so props stop quietly.
        try:
            idle = [NAN] * 6
            for sidx in set_idx.values():
                if 1 <= sidx <= 6:
                    idle[sidx - 1] = -1.0
            for _ in range(5):
                _send_actuators(idle)
                time.sleep(0.02)
        except Exception:
            pass

    threading.Thread(target=rx_loop, daemon=True).start()
    threading.Thread(target=ctrl_loop, daemon=True).start()
    print(f">>> prop bridge running: MAV_CMD_DO_SET_ACTUATOR @ {SEND_HZ:.0f}Hz "
          f"(sets {sorted(set_idx.values())}), value=2*thr-1, off=idle(-1), "
          f"rx-timeout {args.prop_rx_timeout:.2f}s", flush=True)


def main():
    args = parse_args()

    if args.rot.strip():
        R_mount = parse_rot(args.rot)               # explicit SO(3) product
        print(f">>> SO(3) rotation from --rot = {args.rot}", flush=True)
    else:
        mr = [float(v) * DEG for v in args.mount_rpy.split(",")]
        R_mount = euler_to_R(mr[0], mr[1], mr[2])   # board frame -> robot frame
        print(f">>> mount rotation (deg) roll,pitch,yaw = {args.mount_rpy}", flush=True)
    R_mount_T = _transpose(R_mount)

    print(f">>> connecting {args.dev} @ {args.baud} ...", flush=True)
    m = mavutil.mavlink_connection(
        args.dev, baud=args.baud, source_system=255, source_component=190
    )

    stop = threading.Event()

    def hb_loop():
        while not stop.is_set():
            with _SEND_LOCK:
                m.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
                )
            time.sleep(0.5)

    with _SEND_LOCK:
        m.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )
    hb = m.wait_heartbeat(timeout=8)
    if hb is None:
        print("!! no heartbeat from flight controller", flush=True)
        sys.exit(1)
    print(
        f">>> heartbeat ok sys={m.target_system} comp={m.target_component}",
        flush=True,
    )

    threading.Thread(target=hb_loop, daemon=True).start()

    def request_msg(msg_id, hz):
        interval_us = int(1e6 / hz) if hz > 0 else -1
        with _SEND_LOCK:
            m.mav.command_long_send(
                m.target_system, m.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, interval_us, 0, 0, 0, 0, 0,
            )

    def request_all_streams():
        for mid in (MSG_ATTITUDE, MSG_ATTITUDE_QUATERNION, MSG_HIGHRES_IMU):
            request_msg(mid, args.rate)

    if args.imu:
        request_all_streams()
        print(f">>> requested ATTITUDE/QUATERNION/HIGHRES_IMU @ {args.rate} Hz", flush=True)
    else:
        print(">>> --no-imu: IMU served by px4_dds_bridge (DDS/TELEM2); USB = props only",
              flush=True)

    # ---- propeller bridge: motor_pwm_lcmt -> actuator_test (DShot) ----
    if args.props and (motor_pwm_lcmt is not None):
        _start_prop_bridge(m, args, stop)
    elif args.props:
        print("!! props requested but motor_pwm_lcmt type not found; IMU only", flush=True)

    # Props-only mode: no IMU loop -- just keep the link/heartbeat alive while the
    # prop ctrl thread streams. Drain RX so the OS buffer never backs up.
    if not args.imu:
        try:
            while True:
                m.recv_match(blocking=True, timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            print("\n>>> px4_bridge stopped (props-only)", flush=True)
        return

    lc = lcm.LCM(args.lcm_url)
    msg = hopper_imu_lcmt()

    quat_fc = [1.0, 0.0, 0.0, 0.0]   # board attitude (raw from FC)
    gyro_b = [0.0, 0.0, 0.0]         # board-frame
    acc_b = [0.0, 0.0, 0.0]          # board-frame
    have_highres = False

    n_pub = 0
    last_print = time.time()
    # IMU stream watchdog: PX4 sometimes silently stops streaming ATTITUDE (e.g. the
    # one-shot SET_MESSAGE_INTERVAL at startup was lost, or the FC dropped the stream
    # after a reconnect). Without this, hopper_imu goes to 0 forever and the controller
    # runs on a FROZEN attitude -> frozen/choppy prop output. If no ATTITUDE for
    # IMU_WATCHDOG_S, re-assert the stream requests so it self-heals.
    IMU_WATCHDOG_S = 1.5
    last_att_t = time.time()

    try:
        while True:
            mv = m.recv_match(blocking=True, timeout=1.0)
            # Watchdog check (runs even on recv timeout): re-request streams if ATTITUDE stalls.
            if (time.time() - last_att_t) > IMU_WATCHDOG_S:
                request_all_streams()
                last_att_t = time.time()   # debounce: wait another IMU_WATCHDOG_S before retry
                print("!! ATTITUDE stream stalled -> re-requested SET_MESSAGE_INTERVAL", flush=True)
            if mv is None:
                continue
            t = mv.get_type()

            if t == "ATTITUDE":
                last_att_t = time.time()
                if not have_highres:
                    gyro_b = [mv.rollspeed, mv.pitchspeed, mv.yawspeed]

                # SINGLE SO(3) only: apply the one --rot rotation to the raw board data,
                # consistently on vectors and the attitude quaternion. No acc negation,
                # no ENU flip, no R_total_fix.
                #   rot ''      -> pure identity (true raw)
                #   rot 'y-90'  -> Ry(-90) on raw, consistently on vec + quat
                gyro_r = _matvec(R_mount, gyro_b)
                acc_r = _matvec(R_mount, acc_b)
                quat_r = R_to_quat(_matmul(quat_to_R(quat_fc), R_mount_T))
                rpy = list(quat_to_euler(quat_r))

                msg.quat = quat_r
                msg.rpy = rpy
                msg.gyro = gyro_r
                msg.acc = acc_r
                lc.publish(args.channel, msg.encode())
                n_pub += 1

            elif t == "ATTITUDE_QUATERNION":
                quat_fc = [mv.q1, mv.q2, mv.q3, mv.q4]

            elif t == "HIGHRES_IMU":
                have_highres = True
                acc_b = [mv.xacc, mv.yacc, mv.zacc]
                gyro_b = [mv.xgyro, mv.ygyro, mv.zgyro]

            if args.print_hz > 0:
                now = time.time()
                if now - last_print >= 1.0 / args.print_hz:
                    dt = now - last_print
                    hz = n_pub / dt if dt > 0 else 0.0
                    acc_r = _matvec(R_mount, acc_b)
                    r, p2, y = quat_to_euler(R_to_quat(_matmul(quat_to_R(quat_fc), R_mount_T)))
                    print(
                        "pub %5.1fHz | rpy(deg) % 7.2f % 7.2f % 7.2f | "
                        "acc % 6.2f % 6.2f % 6.2f  (robot frame)"
                        % (hz, r * RAD, p2 * RAD, y * RAD, acc_r[0], acc_r[1], acc_r[2]),
                        flush=True,
                    )
                    n_pub = 0
                    last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        # Explicitly stop props on shutdown (daemon ctrl thread may not flush in time).
        if args.props and (motor_pwm_lcmt is not None):
            try:
                for _ in range(3):
                    _nsh_send(m, "actuator_test stop")
                    time.sleep(0.05)
            except Exception:
                pass
        print("\n>>> px4_bridge stopped", flush=True)


if __name__ == "__main__":
    main()
