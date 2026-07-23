#!/usr/bin/env python3
"""px4_dds_bridge: PX4 <-> LCM bridge over uXRCE-DDS (TELEM2).

DOWNLINK (IMU): read PX4 attitude/IMU DDS topics and publish hopper_imu_lcmt --
the high-rate, contention-free replacement for the IMU half of px4_bridge.py.

UPLINK (props, 2026-07 "Plan B"): subscribe motor_pwm_lcmt on LCM and stream the
prop throttles to PX4 as VehicleCommand DO_SET_ACTUATOR (187) on
/fmu/in/vehicle_command -- the SAME command the USB MAVLink path sends, only the
transport changed. With this enabled the USB link carries nothing critical and
can be unplugged; run px4_bridge with --no-props to avoid double-streaming.

2026-07-06 BIDIRECTIONAL (DShot 3D): props moved to AUX1/2/3 as DShot600 with
PX4 DSHOT_3D_ENABLE=1 and AM32 3D mode saved in ESC EEPROM. The pwm convention
on motor_pwm_lcmt is now: 1000us = stop, >1000 = forward (same scale as before),
<1000 = reverse. See --bidir / _actuator_vals for the mapping details.

Why DDS instead of MAVLink-over-USB:
  The single USB-MAVLink link had to carry BOTH the IMU stream and the prop
  MAV_CMD_DO_SET_ACTUATOR stream, so the prop rate was capped (~150Hz because
  PX4 ACKs every command_long on the same link). TELEM2 at 3 Mbaud is dedicated
  and we simply never subscribe the DDS ack topic, so no ACK flood exists here.

Topics consumed (PX4 1.14 defaults, BEST_EFFORT QoS):
  /fmu/out/vehicle_attitude   VehicleAttitude  q=[w,x,y,z] body(FRD)->NED
  /fmu/out/sensor_combined    SensorCombined   gyro_rad, accelerometer_m_s2 (body FRD)

  Pure FRD pipeline (single proper rotation, no reflections):
      R_base  = R_ENU_NED @ R_fc @ R_mount^T
      R_wb    = R_base @ Rx(180deg)      # make body +Z point world-down
      gyro    = Rx(180deg) @ (R_mount @ raw gyro)
      acc     = Rx(180deg) @ (-(R_mount @ raw accel))
  quat/rpy/gyro/acc are all derived from this same FRD frame.

Official PX4 raw (see PX4 docs + mount_calibrate.py):
  board FRD (+X arrow, +Y right, +Z down); VehicleAttitude.q = board->NED; SensorCombined in board FRD.
Physical vertical mount: acc_b~[-g,0,0] at level => R_mount = Rz(psi)*Ry(+90deg) => mount_rpy = 0,90,psi.
This airframe: board Ry(+90deg) then Rz(+90deg) => mount_rpy = 0,90,90.
Live sign calibration: this mount gives the correct pitch direction; the displayed
rpy roll/yaw signs are flipped below to match the operator convention.
If X/Y still swapped after a mechanical change, re-run mount_solve.py (psi only).

Canonical transform matches pixhawk/px4_bridge.py (MAVLink path).

Run:
  source /opt/ros/humble/setup.bash
  source ~/px4_ws/install/setup.bash
  ROS_DOMAIN_ID=0 python3 px4_dds_bridge.py
"""
import os
import sys
import math
import time
import argparse
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleAttitude, SensorCombined
try:
    from px4_msgs.msg import VehicleCommand  # uplink: DO_SET_ACTUATOR over DDS
except Exception:  # pragma: no cover - old px4_msgs build without VehicleCommand
    VehicleCommand = None
try:
    # RM C610/M2006 relay (custom rm_c610 PX4 module, 2026-07). Needs the two
    # Rm*.msg files copied into ~/px4_ws/src/px4_msgs/msg + colcon rebuild.
    from px4_msgs.msg import RmEscCommand, RmEscStatus
except Exception:  # pragma: no cover - px4_msgs build without the Rm messages
    RmEscCommand = None
    RmEscStatus = None


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
    from motor_pwm_lcmt import motor_pwm_lcmt  # noqa: E402  (prop uplink)
except Exception:  # pragma: no cover
    motor_pwm_lcmt = None
try:
    from rm_esc_cmd_lcmt import rm_esc_cmd_lcmt    # noqa: E402  (M2006 uplink)
    from rm_esc_data_lcmt import rm_esc_data_lcmt  # noqa: E402  (M2006 downlink)
except Exception:  # pragma: no cover - lcm-gen not run yet for the rm types
    rm_esc_cmd_lcmt = None
    rm_esc_data_lcmt = None

DEG = math.pi / 180.0
RAD = 180.0 / math.pi


# ---- rotation helpers ----
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


# PX4 world is NED; controller / ModeE world is +Z up (ENU-style).
R_ENU_NED = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
R_FRD_FIX = _rx(math.pi)
# User-requested extra SO(3) rotation: right-hand +Z axis, +180 deg.
# Keeps Z-down unchanged while flipping X/Y directions.
R_USER_Z180 = _rz(math.pi)

def quat_to_R(q):
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def euler_to_R(roll, pitch, yaw):
    return _matmul(_rz(yaw), _matmul(_ry(pitch), _rx(roll)))


def R_to_quat(R):
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


def quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def R_to_rpy_frd(R):
    """FRD roll/pitch/yaw from R_wb (body->world).

    FRD convention used here:
      +X forward, +Y right, +Z down.
      - right tilt  -> roll  > 0
      - nose down   -> pitch < 0
      - yaw sign follows FRD (right turn positive => -atan2(R10,R00))
    """
    R = [[float(R[i][j]) for j in range(3)] for i in range(3)]
    roll = math.atan2(-R[2][1], -R[2][2])
    pitch = math.asin(max(-1.0, min(1.0, R[2][0])))
    yaw = -math.atan2(R[1][0], R[0][0])
    return roll, pitch, yaw


def parse_rot(spec):
    """Build SO(3) from tokens like 'z150,y-90' as the product Rz(150)@Ry(-90)
    (left-to-right). Identity for empty spec. IDENTICAL to pixhawk/px4_bridge.py."""
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    fns = {"x": _rx, "y": _ry, "z": _rz}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        R = _matmul(R, fns[tok[0].lower()](float(tok[1:]) * DEG))
    return R


def _parse_prop_map(spec: str) -> dict:
    """{pwm_index: actuator_set_index}. Accepts '1,2,3' (pwm 1/2/3 -> set 1/2/3)
    or explicit 'pwmIdx:setN,...' pairs. IDENTICAL semantics to px4_bridge.py."""
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
        motors = [int(x) for x in spec.split(",") if x.strip() != ""]
        for i, mn in enumerate(motors):
            mapping[i + 1] = mn
    return mapping


def robot_imu_mavlink_compat(quat_fc, gyro_b, acc_b, R_mount, R_mount_T):
    """EXACT replica of the MAVLink (px4_bridge.py --raw --rot) IMU transform, so the
    DDS path produces byte-for-byte the same controller frame as the old USB path.

    The DDS source data is identical to MAVLink's:
      quat_fc = VehicleAttitude.q  == ATTITUDE_QUATERNION  (board FRD -> NED, wxyz)
      gyro_b/acc_b = SensorCombined == HIGHRES_IMU         (board FRD)
    so applying the SAME single rotation reproduces the SAME output:
      gyro_r = R_mount @ gyro_b ; acc_r = R_mount @ acc_b
      quat_r = quat_to_R(quat_fc) @ R_mount^T ; rpy = quat_to_euler(quat_r)
    NO ENU flip / acc negation / Rx(180) / Rz(180) / FRD-rpy here (that is the OTHER
    pipeline, robot_imu_from_raw, which does NOT match the old USB frame)."""
    gyro_r = _matvec(R_mount, gyro_b)
    acc_r = _matvec(R_mount, acc_b)
    quat_r = R_to_quat(_matmul(quat_to_R(quat_fc), R_mount_T))
    rpy = list(quat_to_euler(quat_r))
    return quat_r, rpy, gyro_r, acc_r


def robot_imu_from_raw(quat_fc, gyro_b, acc_b, R_mount, R_mount_T):
    """PX4 raw -> robot FRD vectors + world attitude (quat/rpy from one R).

    PX4 raw:
      quat_fc  board FRD -> NED  (VehicleAttitude.q, Hamilton wxyz)
      gyro_b, acc_b  in board FRD  (SensorCombined)

    Base rotated values:
      gyro_base = R_mount @ gyro_board
      acc_base  = -(R_mount @ acc_board)
      R_base    = R_ENU_NED @ R_board_ned @ R_mount^T

    FRD values:
      R_wb  = R_base @ Rx(180deg) @ Rz(180deg)
      gyro  = Rz(180deg) @ Rx(180deg) @ gyro_base
      acc   = Rz(180deg) @ Rx(180deg) @ acc_base
    """
    gyro_base = _matvec(R_mount, gyro_b)
    sf_r = _matvec(R_mount, acc_b)
    acc_base = [-sf_r[0], -sf_r[1], -sf_r[2]]
    R_base = _matmul(R_ENU_NED, _matmul(quat_to_R(quat_fc), R_mount_T))
    R_total_fix = _matmul(R_FRD_FIX, R_USER_Z180)
    R_rw = _matmul(R_base, R_total_fix)
    gyro_r = _matvec(R_total_fix, gyro_base)
    acc_r = _matvec(R_total_fix, acc_base)
    quat_r = R_to_quat(R_rw)
    roll, pitch, yaw = R_to_rpy_frd(R_rw)
    return quat_r, [roll, pitch, yaw], gyro_r, acc_r


class DdsImuBridge(Node):
    def __init__(self, args):
        super().__init__("px4_dds_imu_bridge")
        # Two mutually-exclusive transforms:
        #   --rot (e.g. 'z150,y-90')  -> MAVLink-compatible path (EXACTLY matches the old
        #                                USB px4_bridge --raw --rot frame). PREFERRED so the
        #                                IMU coordinates are unchanged when moving to DDS.
        #   --mount-rpy (fallback)     -> the DDS-native ENU/FRD pipeline (robot_imu_from_raw).
        self._mav_compat = bool(args.rot.strip())
        if self._mav_compat:
            self.R_mount = parse_rot(args.rot)             # board -> robot (== USB --rot)
        else:
            mr = [float(v) * DEG for v in args.mount_rpy.split(",")]
            self.R_mount = euler_to_R(mr[0], mr[1], mr[2])  # board -> robot
        self.R_mount_T = _transpose(self.R_mount)
        self.args = args

        self.lc = lcm.LCM(args.lcm_url)
        self.msg = hopper_imu_lcmt()

        self.quat_fc = [1.0, 0.0, 0.0, 0.0]
        self.gyro_b = [0.0, 0.0, 0.0]
        self.acc_b = [0.0, 0.0, 0.0]
        self._have_attitude = False

        self.n_pub = 0
        self.t_last = self.get_clock().now()
        pub_hz = float(max(1.0, float(getattr(args, "publish_hz", 500.0))))
        self._pub_period_s = 1.0 / pub_hz

        # PX4 publishes BEST_EFFORT / VOLATILE / KEEP_LAST.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(SensorCombined, "/fmu/out/sensor_combined",
                                 self.on_sensor, qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude",
                                 self.on_attitude, qos)
        # Fixed-rate LCM publish (default 500 Hz) using latest DDS samples.
        # --no-imu = props-only mode: another device (e.g. Lpms IG1 via
        # hopper_driver) owns hopper_imu_lcmt, so we must not also publish it.
        if args.imu:
            self.create_timer(self._pub_period_s, self.on_publish_tick)
        if args.print_hz > 0:
            self.create_timer(1.0 / args.print_hz, self.on_print)

        _xf = (f"MAVLink-compat rot={args.rot}" if self._mav_compat
               else f"DDS-native mount(deg)={args.mount_rpy}")
        _imu_s = (f"publish '{args.channel}' @ {pub_hz:.0f}Hz" if args.imu
                  else "IMU publish OFF (--no-imu)")
        print(f">>> px4_dds_bridge up | {_xf} | {_imu_s} on {args.lcm_url}", flush=True)

        # ---- prop uplink (Plan B): motor_pwm_lcmt -> VehicleCommand DO_SET_ACTUATOR ----
        self._prop_enabled = False
        self._prop_stop = threading.Event()
        if args.props:
            if VehicleCommand is None:
                print("!! props requested but px4_msgs has no VehicleCommand -> prop "
                      "uplink DISABLED (rebuild px4_msgs in ~/px4_ws)", flush=True)
            elif motor_pwm_lcmt is None:
                print("!! props requested but motor_pwm_lcmt LCM type not found -> "
                      "prop uplink DISABLED", flush=True)
            else:
                self._setup_props(args)

        # ---- RM C610/M2006 relay (custom rm_c610 PX4 module, 2026-07) ----
        # UPLINK  : rm_esc_cmd_lcmt  (LCM) -> /fmu/in/rm_esc_command  (DDS)
        # DOWNLINK: /fmu/out/rm_esc_status (DDS) -> rm_esc_data_lcmt  (LCM)
        self._rm_enabled = False
        self._rm_stop = threading.Event()
        if args.rm_esc:
            if RmEscCommand is None or RmEscStatus is None:
                print("!! rm-esc requested but px4_msgs lacks RmEscCommand/RmEscStatus"
                      " -> relay DISABLED (copy Rm*.msg + rebuild px4_msgs)", flush=True)
            elif rm_esc_cmd_lcmt is None or rm_esc_data_lcmt is None:
                print("!! rm-esc requested but rm_esc_*_lcmt LCM types not found"
                      " -> relay DISABLED (re-run lcm-gen)", flush=True)
            else:
                self._setup_rm_esc(args)

    # ------------------------------------------------------------------
    # Prop uplink mapping.
    #
    # UNIDIRECTIONAL (--no-bidir, legacy PWM ESC mode, DSHOT_3D_ENABLE=0):
    #   pwm_us -> throttle = (pwm - pwm_min)/(pwm_max - pwm_min), clipped [0,1]
    #   actuator value = 2*thr - 1  in [-1,1]  (-1 = 1000us idle, +1 = 2000us full)
    #
    # BIDIRECTIONAL (--bidir, default since 2026-07-06; requires DSHOT_3D_ENABLE=1
    # on PX4 + AM32 3D mode saved in ESC EEPROM):
    #   actuator value = clip((pwm - 1000)/1000, -1, +1)
    #     pwm 1000 -> 0.0  = stop (inside the PX4 DSHOT_3D dead-band)
    #     pwm 2000 -> +1.0 = full forward   (SAME scale as before for pwm > 1000)
    #     pwm    0 -> -1.0 = full reverse   (pwm < 1000 = reverse, new)
    #   Idle/disarm value is 0.0, NOT -1.0: in 3D mode -1 means ~full REVERSE.
    #   NEVER run --no-bidir mapping while DSHOT_3D_ENABLE=1 (idle would reverse
    #   at high speed) -- that is why bidir is the default now.
    #
    # Props spin ONLY while control_mode == prop_arm_mode AND the LCM stream is
    # fresh (< prop_rx_timeout); otherwise every mapped set is commanded to idle.
    # Unused actuator sets are sent NaN = "leave untouched", same as MAVLink.
    # ------------------------------------------------------------------
    def _setup_props(self, args):
        self._prop_map = _parse_prop_map(args.prop_map)   # {pwm_idx: set_idx 1..6}
        if not self._prop_map:
            print("!! --prop-map empty; prop uplink disabled", flush=True)
            return
        self._prop_bidir = bool(args.bidir)
        self._prop_reverse = {
            int(x.strip()) for x in str(args.prop_reverse).split(",") if x.strip()
        }
        self._pwm_min = float(args.pwm_min)
        self._pwm_span = max(1.0, float(args.pwm_max) - float(args.pwm_min))
        self._prop_arm_mode = int(args.prop_arm_mode)
        self._prop_rx_timeout = float(args.prop_rx_timeout)
        self._do_set_actuator = int(getattr(
            VehicleCommand, "VEHICLE_CMD_DO_SET_ACTUATOR", 187))
        self._pwm_state = {"pwm": [self._pwm_min] * 6, "mode": 0, "rx_t": 0.0}
        self._pwm_lock = threading.Lock()
        # MAIN (PX4IO) outputs stay locked at the disarmed value (1500 = stop)
        # until the vehicle is armed, unlike the old AUX/FMU setup. So the
        # bridge force-arms PX4 while prop commands are active and disarms
        # after --auto-disarm-delay of idle. Disarmed outputs are 1500 = stop,
        # so arming by itself never spins anything.
        self._auto_arm = bool(args.auto_arm)
        self._arm_disarm_cmd = int(getattr(
            VehicleCommand, "VEHICLE_CMD_COMPONENT_ARM_DISARM", 400))
        self._auto_disarm_delay = float(args.auto_disarm_delay)
        self._armed_wanted = False
        self._idle_since = time.time()

        # /fmu/in/* uplink topics use default RELIABLE QoS (px4_ros_com convention).
        self._cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10)

        # Dedicated LCM instance + thread for the pwm subscription: self.lc is
        # used by the ROS timer thread for publishing and lcm handles are not
        # safe to share across threads.
        self._lc_pwm = lcm.LCM(args.lcm_url)
        self._lc_pwm.subscribe(args.pwm_channel, self._on_pwm_lcm)

        def rx_loop():
            while not self._prop_stop.is_set():
                try:
                    self._lc_pwm.handle_timeout(200)
                except Exception:
                    time.sleep(0.05)

        threading.Thread(target=rx_loop, daemon=True).start()

        prop_hz = max(20.0, float(args.prop_rate))
        self.create_timer(1.0 / prop_hz, self.on_prop_tick)
        self._prop_enabled = True
        _map_s = ("BIDIR pwm 1000=stop 2000=+1(fwd) 0=-1(rev)" if self._prop_bidir
                  else f"pwm {args.pwm_min:.0f}->0, {args.pwm_max:.0f}->1")
        print(f">>> prop uplink (DDS): '{args.pwm_channel}' -> /fmu/in/vehicle_command "
              f"DO_SET_ACTUATOR @ {prop_hz:.0f}Hz | pwm_idx->set {self._prop_map} "
              f"| reverse pwm_idx={sorted(self._prop_reverse)} "
              f"({_map_s}) | arm_mode="
              f"{self._prop_arm_mode} rx-timeout {self._prop_rx_timeout:.2f}s "
              f"| auto-arm={'ON' if self._auto_arm else 'off'} "
              f"(disarm after {self._auto_disarm_delay:.1f}s idle)", flush=True)

    # ------------------------------------------------------------------
    # RM C610/M2006 relay.
    #
    # UPLINK: rm_esc_cmd_lcmt.current_raw[3] (-10000..10000 = -10..+10A) is
    # re-streamed to /fmu/in/rm_esc_command at --rm-rate. If no LCM command
    # arrives within --rm-rx-timeout the relay streams zeros (and the PX4
    # rm_c610 module additionally zeroes after its own 100ms timeout, so the
    # motors are safe even if the whole bridge dies).
    #
    # DOWNLINK: every /fmu/out/rm_esc_status sample (100Hz, rate-limited in
    # dds_topics.yaml) is forwarded verbatim as rm_esc_data_lcmt.
    # ------------------------------------------------------------------
    def _setup_rm_esc(self, args):
        self._rm_rx_timeout = float(args.rm_rx_timeout)
        self._rm_state = {"cur": [0, 0, 0], "rx_t": 0.0}
        self._rm_lock = threading.Lock()
        self._rm_n_up = 0
        self._rm_n_down = 0

        # /fmu/in/* uplink uses default RELIABLE QoS; /fmu/out/* is BEST_EFFORT.
        self._rm_cmd_pub = self.create_publisher(
            RmEscCommand, "/fmu/in/rm_esc_command", 10)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(RmEscStatus, "/fmu/out/rm_esc_status",
                                 self._on_rm_status_dds, qos)

        # Dedicated LCM instance + thread for the command subscription
        # (lcm handles are not safe to share across threads; same pattern
        # as the pwm uplink).
        self._lc_rm = lcm.LCM(args.lcm_url)
        self._lc_rm.subscribe(args.rm_cmd_channel, self._on_rm_cmd_lcm)

        def rm_rx_loop():
            while not self._rm_stop.is_set():
                try:
                    self._lc_rm.handle_timeout(200)
                except Exception:
                    time.sleep(0.05)

        threading.Thread(target=rm_rx_loop, daemon=True).start()

        rm_hz = max(20.0, float(args.rm_rate))
        self.create_timer(1.0 / rm_hz, self.on_rm_cmd_tick)
        self._rm_data_channel = args.rm_data_channel
        self._rm_enabled = True
        print(f">>> rm-esc relay: '{args.rm_cmd_channel}' -> /fmu/in/rm_esc_command "
              f"@ {rm_hz:.0f}Hz (rx-timeout {self._rm_rx_timeout:.2f}s) | "
              f"/fmu/out/rm_esc_status -> '{args.rm_data_channel}'", flush=True)

    def _on_rm_cmd_lcm(self, channel, data):
        try:
            mm = rm_esc_cmd_lcmt.decode(data)
        except Exception:
            return
        cur = [max(-10000, min(10000, int(v))) for v in mm.current_raw[:3]]
        with self._rm_lock:
            self._rm_state["cur"] = cur
            self._rm_state["rx_t"] = time.time()

    def on_rm_cmd_tick(self):
        with self._rm_lock:
            cur = list(self._rm_state["cur"])
            rx_t = float(self._rm_state["rx_t"])
        alive = (rx_t > 0.0) and ((time.time() - rx_t) <= self._rm_rx_timeout)
        if not alive:
            cur = [0, 0, 0]
        msg = RmEscCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)  # us
        msg.current_raw = cur
        self._rm_cmd_pub.publish(msg)
        self._rm_n_up += 1

    def _on_rm_status_dds(self, m):
        # Runs on the (single-threaded) ROS executor, same thread as the IMU
        # publish timer, so sharing self.lc here is safe.
        out = rm_esc_data_lcmt()
        out.timestamp = int(m.timestamp)
        out.shaft_angle_rad = [float(v) for v in m.shaft_angle_rad[:3]]
        out.shaft_speed_rad_s = [float(v) for v in m.shaft_speed_rad_s[:3]]
        out.current_raw = [int(v) for v in m.current_raw[:3]]
        out.rpm = [int(v) for v in m.rpm[:3]]
        out.angle_raw = [int(v) for v in m.angle_raw[:3]]
        out.online_mask = int(m.online_mask) & 0x7F
        self.lc.publish(self._rm_data_channel, out.encode())
        self._rm_n_down += 1

    def _on_pwm_lcm(self, channel, data):
        try:
            mm = motor_pwm_lcmt.decode(data)
        except Exception:
            return
        with self._pwm_lock:
            self._pwm_state["pwm"] = list(mm.pwm_values)
            self._pwm_state["mode"] = int(mm.control_mode)
            self._pwm_state["rx_t"] = time.time()

    def _actuator_vals(self, force_idle=False):
        """Returns (vals6, active). active = a fresh prop-ON command stream exists."""
        NAN = float("nan")
        with self._pwm_lock:
            pwm = list(self._pwm_state["pwm"])
            mode = int(self._pwm_state["mode"])
            rx_t = float(self._pwm_state["rx_t"])
        now = time.time()
        alive = (rx_t > 0.0) and ((now - rx_t) <= self._prop_rx_timeout)
        armed = (not force_idle) and alive and (mode == self._prop_arm_mode)
        vals = [NAN] * 6
        for pwm_idx, sidx in self._prop_map.items():
            if 1 <= sidx <= 6:
                if self._prop_bidir:
                    # 3D/DShot mode: 0.0 = stop; sending -1 here would be full reverse
                    if armed and (0 <= pwm_idx < len(pwm)):
                        v = (float(pwm[pwm_idx]) - 1000.0) / 1000.0
                    else:
                        v = 0.0
                    if pwm_idx in self._prop_reverse:
                        v = -v
                    vals[sidx - 1] = max(-1.0, min(1.0, v))
                else:
                    if armed and (0 <= pwm_idx < len(pwm)):
                        thr = (float(pwm[pwm_idx]) - self._pwm_min) / self._pwm_span
                    else:
                        thr = 0.0
                    thr = max(0.0, min(1.0, thr))
                    vals[sidx - 1] = 2.0 * thr - 1.0
        return vals, armed

    def _send_actuator_cmd(self, vals6):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)  # us
        msg.command = self._do_set_actuator
        msg.param1 = float(vals6[0])
        msg.param2 = float(vals6[1])
        msg.param3 = float(vals6[2])
        msg.param4 = float(vals6[3])
        msg.param5 = float(vals6[4])
        msg.param6 = float(vals6[5])
        msg.param7 = 0.0                       # actuator-set group index
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 190
        msg.from_external = True
        self._cmd_pub.publish(msg)

    def _send_arm_cmd(self, arm: bool):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)  # us
        msg.command = self._arm_disarm_cmd
        msg.param1 = 1.0 if arm else 0.0
        msg.param2 = 21196.0  # force: bypass prearm checks (bench use, no RC/GPS)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 190
        msg.from_external = True
        self._cmd_pub.publish(msg)

    def _auto_arm_tick(self, active: bool):
        now = time.time()
        if active:
            self._idle_since = None
            if not self._armed_wanted:
                self._armed_wanted = True
                print(">>> props active -> ARM", flush=True)
                self._send_arm_cmd(True)
        else:
            if self._idle_since is None:
                self._idle_since = now
            if self._armed_wanted and (now - self._idle_since) >= self._auto_disarm_delay:
                self._armed_wanted = False
                print(f">>> props idle {self._auto_disarm_delay:.1f}s -> DISARM", flush=True)
                self._send_arm_cmd(False)

    def on_prop_tick(self):
        if self._prop_enabled:
            vals, active = self._actuator_vals()
            if self._auto_arm:
                self._auto_arm_tick(active)
            self._send_actuator_cmd(vals)

    def prop_shutdown(self):
        """Command every mapped set to idle a few times so props stop quietly."""
        if not self._prop_enabled:
            return
        self._prop_stop.set()
        try:
            idle, _ = self._actuator_vals(force_idle=True)
            for _ in range(5):
                self._send_actuator_cmd(idle)
                time.sleep(0.02)
            if self._auto_arm and self._armed_wanted:
                self._send_arm_cmd(False)
        except Exception:
            pass

    def rm_shutdown(self):
        """Stream zero current a few times so the M2006s coast to a stop."""
        if not self._rm_enabled:
            return
        self._rm_stop.set()
        try:
            for _ in range(5):
                msg = RmEscCommand()
                msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                msg.current_raw = [0, 0, 0]
                self._rm_cmd_pub.publish(msg)
                time.sleep(0.02)
        except Exception:
            pass

    def _imu(self):
        fn = robot_imu_mavlink_compat if self._mav_compat else robot_imu_from_raw
        return fn(self.quat_fc, self.gyro_b, self.acc_b, self.R_mount, self.R_mount_T)

    def on_sensor(self, m):
        self.gyro_b = [float(m.gyro_rad[0]), float(m.gyro_rad[1]), float(m.gyro_rad[2])]
        self.acc_b = [float(m.accelerometer_m_s2[0]), float(m.accelerometer_m_s2[1]),
                      float(m.accelerometer_m_s2[2])]

    def on_attitude(self, m):
        # PX4 VehicleAttitude.q = [w,x,y,z], body(FRD) -> NED.
        self.quat_fc = [float(m.q[0]), float(m.q[1]), float(m.q[2]), float(m.q[3])]
        self._have_attitude = True

    def on_publish_tick(self):
        if not bool(self._have_attitude):
            return
        quat_r, rpy, gyro_r, acc_r = self._imu()
        self.msg.quat = quat_r
        self.msg.rpy = rpy
        self.msg.gyro = gyro_r
        self.msg.acc = acc_r
        self.lc.publish(self.args.channel, self.msg.encode())
        self.n_pub += 1

    def on_print(self):
        now = self.get_clock().now()
        dt = (now - self.t_last).nanoseconds * 1e-9
        hz = self.n_pub / dt if dt > 0 else 0.0
        quat_r, rpy, _, acc_r = self._imu()
        r, p, y = rpy
        # Sanity: at rest R@acc ~= g_w=(0,0,-9.81), body+Z in world col2_z ~= -1 (FRD)
        w, xq, yq, zq = quat_r
        R_chk = quat_to_R([w, xq, yq, zq])
        col2z = R_chk[2][2]
        racc = _matvec(R_chk, acc_r)
        print(
            "pub %6.1fHz | rpy(deg) % 7.2f % 7.2f % 7.2f | acc % 6.2f % 6.2f % 6.2f | bodyZw %+.2f Racc_z %+.2f"
            % (hz, r * RAD, p * RAD, y * RAD, acc_r[0], acc_r[1], acc_r[2], col2z, racc[2]),
            flush=True,
        )
        self.n_pub = 0
        self.t_last = now


def parse_args():
    p = argparse.ArgumentParser(description="PX4 DDS -> LCM IMU bridge")
    p.add_argument("--channel", default="hopper_imu_lcmt")
    p.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=255")
    # board->robot: R_mount = Rz(psi) @ Ry(+90 deg).  See module docstring.
    # Default psi=0: vertical Pixhawk, board +Z -> robot +X, board +Y -> robot +Y.
    p.add_argument("--mount-rpy", default="0,90,90",
                   help="board->leg/controller mount roll,pitch,yaw deg (active Rz@Ry@Rx; this robot: 0,90,90). "
                        "Used ONLY by the DDS-native pipeline (when --rot is empty).")
    p.add_argument("--rot", default="",
                   help="MAVLink-compatible single SO(3), e.g. 'z150,y-90' (Rz@Ry, left-to-right). "
                        "When set, the IMU transform EXACTLY matches the old USB px4_bridge "
                        "--raw --rot frame -> IMU coordinates are UNCHANGED moving to DDS. "
                        "Takes precedence over --mount-rpy.")
    p.add_argument("--publish-hz", type=float, default=500.0,
                   help="hopper_imu_lcmt publish rate (Hz). Default: 500 (matches ModeE dt=0.002).")
    p.add_argument("--print-hz", type=float, default=2.0)
    p.add_argument("--imu", dest="imu", action="store_true", default=True,
                   help="publish hopper_imu_lcmt from the DDS attitude feed (default on)")
    p.add_argument("--no-imu", dest="imu", action="store_false",
                   help="props-only mode: do NOT publish hopper_imu_lcmt (use when the "
                        "Lpms/hopper_driver owns the IMU channel; prop uplink still runs)")
    # ---- prop uplink (Plan B): same semantics/defaults as px4_bridge.py ----
    p.add_argument("--props", dest="props", action="store_true", default=True,
                   help="enable motor_pwm_lcmt -> VehicleCommand DO_SET_ACTUATOR uplink "
                        "over DDS/TELEM2 (default on). Make sure px4_bridge (USB) runs "
                        "with --no-props, or is stopped, so only ONE prop stream exists.")
    p.add_argument("--no-props", dest="props", action="store_false",
                   help="disable the prop uplink (IMU downlink only)")
    p.add_argument("--pwm-channel", default="motor_pwm_lcmt",
                   help="LCM channel carrying prop PWM commands")
    p.add_argument("--prop-map", default="1:1,2:2,3:3",
                   help="pwm_values index -> actuator-set index (DO_SET_ACTUATOR "
                        "param1..6), as explicit 'pwmIdx:setN,...' pairs. "
                        "2026-07-18 physical map: M1 pwm[1]->set1->MAIN8 (+x,+y), "
                        "M2 pwm[2]->set2->MAIN2 (-y), M3 pwm[3]->set3->MAIN3 (-x,+y).")
    p.add_argument("--pwm-min", type=float, default=1000.0, help="pwm_us at 0 throttle")
    p.add_argument("--pwm-max", type=float, default=2000.0, help="pwm_us at full throttle")
    p.add_argument("--bidir", dest="bidir", action="store_true", default=True,
                   help="bidirectional (DShot 3D) mapping: pwm 1000=stop, 2000=full "
                        "forward, <1000=reverse; idle sends 0.0. REQUIRED while PX4 "
                        "DSHOT_3D_ENABLE=1 (default since 2026-07-06).")
    p.add_argument("--no-bidir", dest="bidir", action="store_false",
                   help="legacy unidirectional mapping (only for PWM ESC mode with "
                        "DSHOT_3D_ENABLE=0; idle sends -1.0)")
    p.add_argument("--prop-reverse", default="1,2",
                   help="comma-separated PWM indices whose bidirectional command is "
                        "inverted around stop. 2026-07-18 hardware: reverse M1/M2 "
                        "(pwm[1],pwm[2]); M3 remains normal.")
    p.add_argument("--prop-rate", type=float, default=250.0,
                   help="DO_SET_ACTUATOR re-stream rate (Hz) over DDS. Unlike the USB "
                        "MAVLink path there is no COMMAND_ACK return flood (we never "
                        "subscribe the ack topic), so this can exceed the old 150Hz cap. "
                        "250Hz is ~25KB/s of the 3Mbaud TELEM2 link.")
    p.add_argument("--prop-rx-timeout", type=float, default=0.4,
                   help="idle the props if no motor_pwm_lcmt seen for this long (s)")
    p.add_argument("--prop-arm-mode", type=int, default=3,
                   help="motor_pwm_lcmt.control_mode value that means 'props ON' "
                        "(default 3 = PWMPD); anything else idles the props")
    p.add_argument("--auto-arm", dest="auto_arm", action="store_true", default=True,
                   help="force-ARM PX4 while prop commands are active and DISARM "
                        "after --auto-disarm-delay of idle. Needed since 2026-07-18: "
                        "props on MAIN (PX4IO) outputs, which stay at the disarmed "
                        "1500us stop value until armed (default on)")
    p.add_argument("--no-auto-arm", dest="auto_arm", action="store_false",
                   help="never send ARM/DISARM from the bridge")
    p.add_argument("--auto-disarm-delay", type=float, default=2.0,
                   help="seconds of prop-idle before the bridge auto-DISARMs (s)")
    # ---- RM C610/M2006 relay (custom rm_c610 PX4 module) ----
    p.add_argument("--rm-esc", dest="rm_esc", action="store_true", default=True,
                   help="enable the rm_esc_cmd_lcmt <-> /fmu/in|out/rm_esc_* relay "
                        "(default on; auto-disables if px4_msgs/LCM types missing)")
    p.add_argument("--no-rm-esc", dest="rm_esc", action="store_false",
                   help="disable the RM C610/M2006 relay")
    p.add_argument("--rm-cmd-channel", default="rm_esc_cmd_lcmt",
                   help="LCM channel carrying M2006 current commands")
    p.add_argument("--rm-data-channel", default="rm_esc_data_lcmt",
                   help="LCM channel for M2006 feedback")
    p.add_argument("--rm-rate", type=float, default=200.0,
                   help="RmEscCommand re-stream rate over DDS (Hz); the PX4 "
                        "rm_c610 module zeroes currents after 100ms without input")
    p.add_argument("--rm-rx-timeout", type=float, default=0.2,
                   help="stream zero current if no rm_esc_cmd_lcmt for this long (s)")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = DdsImuBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.prop_shutdown()   # idle props before the DDS link goes away
        node.rm_shutdown()     # zero M2006 current before exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
