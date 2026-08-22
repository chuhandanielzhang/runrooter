#!/usr/bin/env python3
"""Trace the deployed-leg WORK ENVELOPE boundary with the foot.

Run ON the Jetson (PC only launches it):

  bash scripts/trace_workspace.sh

The script SSHs to the robot, stops hopper-upper, then this process
waits. Press gamepad X (driver PD) to start. Does NOT restart upper.
Logs land in hopper_controller/logs/ on the Jetson and are copied back
to the PC logs_local/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # .../hopper_controller
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT.parent / "hopper_lcm_types" / "lcm_types"))

import lcm  # noqa: E402
from forward_kinematics import ForwardKinematics  # noqa: E402
from python.hopper_cmd_lcmt import hopper_cmd_lcmt  # noqa: E402
from python.hopper_data_lcmt import hopper_data_lcmt  # noqa: E402
from python.gamepad_lcmt import gamepad_lcmt  # noqa: E402

LCM_URL = "udpm://239.255.76.67:7667?ttl=255"

# ---- envelope constants (keep in sync with ModeELCMConfig.work_*) -------
Z_TOP_M = 0.128          # two-q-down, q2=-1.2 (zero-side stop)
Z_BOTTOM_M = 0.440
LEN_MIN_M = 0.354        # FK(0,0,0) homing / 零位
LEN_MAX_M = 0.50
XY_MAX_M = 0.37          # FK(1.4,1.4,-1.2) r_xy
WHEEL_CENTER_M = 0.17
WHEEL_KEEP_M = 0.055
WHEEL_AZ_DRIVE_DEG = (180.0, 60.0, 300.0)   # ModeELCMConfig.wheel_azimuth_deg
DRIVE_YAW_FALLBACK_DEG = -32.9              # overridden by T_L_C.json below
CALIB_JSON = _ROOT / "tools" / "apriltags_print" / "calib" / "T_L_C.json"

# ---- MOBILE leg-stow swing parameters ------------------------------------
KP_JOINT = 20.0        # mobile_leg_kp
TAU_MAX_NM = 3.0       # mobile_leg_tau_max_nm
AK60_KD = 2.0          # mobile_leg_ak60_kd_move
CENTER_Q = 0.4         # mobile_leg_center_q (归中)
ARRIVE_RAD = 0.10      # mobile_leg_arrive_rad

RATE_HZ = 100.0
DQ_STEP_MAX_RAD = 0.03     # per-tick IK step clip
Q_CMD_LIM_RAD = (-1.2, 1.6)


def drive_yaw_rad() -> float:
    """Drive-frame yaw inside L, from the hand-eye calib (same file the
    controller loads); fallback to the config default."""
    try:
        R = np.asarray(
            json.loads(CALIB_JSON.read_text())["R"], dtype=float
        ).reshape(3, 3)
        # Same convention as lcm_controller __init__ (T_L_C -> drive yaw).
        return math.atan2(float(R[1, 2]), float(R[0, 2]))
    except Exception:
        return math.radians(DRIVE_YAW_FALLBACK_DEG)


def wheel_az_L(yaw: float) -> list[float]:
    return [math.radians(a) + yaw for a in WHEEL_AZ_DRIVE_DEG]


def wheel_centers_xy(yaw: float) -> list[tuple[float, float]]:
    return [
        (WHEEL_CENTER_M * math.cos(az), WHEEL_CENTER_M * math.sin(az))
        for az in wheel_az_L(yaw)
    ]


def r_cap(az: float, z: float, centers: list[tuple[float, float]]) -> float:
    """Max xy radius at this azimuth: outer disk minus wheel circles."""
    r_max = math.sqrt(max(0.0, LEN_MAX_M * LEN_MAX_M - z * z))
    if XY_MAX_M > 0.0:
        r_max = min(r_max, XY_MAX_M)
    ca, sa = math.cos(az), math.sin(az)
    k2 = WHEEL_KEEP_M * WHEEL_KEEP_M
    for cx, cy in centers:
        b = ca * cx + sa * cy
        disc = b * b - (cx * cx + cy * cy - k2)
        if disc <= 0.0:
            continue
        r1 = b - math.sqrt(disc)
        if 0.0 < r1 < r_max:
            r_max = r1
    return r_max


def envelope_path(centers: list[tuple[float, float]], *,
                  z_top: float, z_bot: float, start_az: float,
                  p_now: np.ndarray, both: bool = True,
                  loops: int = 1) -> np.ndarray:
    """Approach + bottom outline, then (optional) lift and top outline."""
    bot = boundary_waypoints(z_bot, centers, start_az=start_az)
    chunks = [p_now.reshape(1, 3), bot]
    for _ in range(max(0, loops - 1)):
        chunks.append(bot[1:])
    if both:
        top = boundary_waypoints(z_top, centers, start_az=start_az)
        chunks.append(top[:1])   # vertical edge at the same azimuth
        chunks.append(top)
        for _ in range(max(0, loops - 1)):
            chunks.append(top[1:])
    return np.vstack(chunks)


def boundary_waypoints(z: float, centers: list[tuple[float, float]],
                       start_az: float, step_deg: float = 1.0) -> np.ndarray:
    """Dense polygon along the boundary at height z, starting near
    start_az. Wheel bites appear as circular inner arcs."""
    n = int(round(360.0 / step_deg))
    pts = []
    for k in range(n + 1):
        az = start_az + math.radians(step_deg) * k
        r = r_cap(az, z, centers)
        pts.append((r * math.cos(az), r * math.sin(az), z))
    return np.asarray(pts, dtype=float)


def path_resample(way: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative arc length s_i for waypoint interpolation."""
    seg = np.linalg.norm(np.diff(way, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, way


def path_point(s_arr: np.ndarray, way: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, float(s_arr[-1])))
    i = int(np.searchsorted(s_arr, s, side="right")) - 1
    i = max(0, min(i, len(way) - 2))
    ds = float(s_arr[i + 1] - s_arr[i])
    t = 0.0 if ds <= 1e-12 else (s - float(s_arr[i])) / ds
    return way[i] * (1.0 - t) + way[i + 1] * t


def numeric_jacobian(fk: ForwardKinematics, q: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    J = np.zeros((3, 3), dtype=float)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        p1, _ = fk.forward_kinematics(q + d)
        p0, _ = fk.forward_kinematics(q - d)
        J[:, i] = (np.asarray(p1, dtype=float).reshape(3)
                   - np.asarray(p0, dtype=float).reshape(3)) / (2.0 * eps)
    return J


class Tee:
    """Print to stdout and a log file at the same time."""

    def __init__(self, path: Path):
        self._fp = path.open("w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self  # type: ignore[assignment]

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._fp.write(s)
        self._fp.flush()
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._fp.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        try:
            self._fp.close()
        except Exception:
            pass


class Robot:
    """LCM I/O: latest measured q + safe command publishing."""

    def __init__(self, lc: lcm.LCM):
        self.lc = lc
        self.q = None
        self.qd = np.zeros(3)
        self.tau_iq = np.zeros(3)
        self.t_q = 0.0
        self.x_now = 0
        self.x_prev = 0
        self.x_edge = False
        self.b_now = 0
        self.t_gp = 0.0
        self.foreign_cmds = 0
        self._lock = threading.Lock()
        lc.subscribe("hopper_data_lcmt", self._on_data)
        lc.subscribe("gamepad_lcmt", self._on_gamepad)
        self._sub_cmd = lc.subscribe("hopper_cmd_lcmt", self._on_cmd)
        self.running = True
        self._thr = threading.Thread(target=self._spin, daemon=True)
        self._thr.start()

    def _spin(self):
        while self.running:
            try:
                self.lc.handle_timeout(100)
            except Exception:
                time.sleep(0.05)

    def _on_data(self, _ch, data):
        msg = hopper_data_lcmt.decode(data)
        with self._lock:
            self.q = np.asarray(msg.q, dtype=float).reshape(3)
            try:
                self.qd = np.asarray(msg.qd, dtype=float).reshape(3)
            except Exception:
                self.qd = np.zeros(3)
            try:
                self.tau_iq = np.asarray(msg.tauIq, dtype=float).reshape(3)
            except Exception:
                self.tau_iq = np.zeros(3)
            self.t_q = time.time()

    def _on_gamepad(self, _ch, data):
        msg = gamepad_lcmt.decode(data)
        with self._lock:
            x = int(getattr(msg, "x", 0) or 0)
            self.b_now = int(getattr(msg, "b", 0) or 0)
            if x and not self.x_prev:
                self.x_edge = True
            self.x_prev = x
            self.x_now = x
            self.t_gp = time.time()

    def consume_x_edge(self) -> bool:
        with self._lock:
            hit = bool(self.x_edge)
            self.x_edge = False
            return hit

    def _on_cmd(self, _ch, _data):
        # Anything heard BEFORE we start publishing = another controller.
        self.foreign_cmds += 1

    def stop_listening_cmd(self):
        try:
            self.lc.unsubscribe(self._sub_cmd)
        except Exception:
            pass

    def get_state(self):
        with self._lock:
            if self.q is None:
                return None, None, None, self.t_q
            return (self.q.copy(), self.qd.copy(),
                    self.tau_iq.copy(), self.t_q)

    def send(self, tau: np.ndarray, kd: float):
        msg = hopper_cmd_lcmt()  # rm_iq_des zeros, rm_set_zero=0 (safe)
        msg.tau_ff = [float(np.clip(t, -TAU_MAX_NM, TAU_MAX_NM))
                      for t in np.asarray(tau, dtype=float).reshape(3)]
        msg.kd_joint = [float(kd)] * 3
        self.lc.publish("hopper_cmd_lcmt", msg.encode())

    def safe_stop(self, ticks: int = 60):
        """Ramp to zero torque with AK60 damping, then full zeros."""
        for _ in range(ticks):
            self.send(np.zeros(3), AK60_KD)
            time.sleep(1.0 / RATE_HZ)
        for _ in range(5):
            self.send(np.zeros(3), 0.0)
            time.sleep(0.01)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Trace the deployed-leg work envelope with the foot.")
    ap.add_argument("--z", type=float, default=None,
                    help="single-plane height, m below body (+down). "
                         "Default: trace BOTTOM (0.440) then TOP (0.128, q2=-1.2).")
    ap.add_argument("--speed", type=float, default=0.08,
                    help="foot speed along the outline, m/s (default 0.08)")
    ap.add_argument("--loops", type=int, default=1,
                    help="how many full outlines per plane (default 1)")
    ap.add_argument("--lcm-url", default=LCM_URL)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the outline, no robot commands")
    ap.add_argument("--log-dir", default=str(_ROOT / "logs_local"),
                    help="CSV + console log directory (default: logs_local)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_txt = log_dir / f"workspace_trace_{stamp}.log"
    log_csv = log_dir / f"workspace_trace_{stamp}.csv"
    tee = Tee(log_txt)
    print(f"[trace] console log -> {log_txt}")

    fk = ForwardKinematics()
    f0, _ = fk.forward_kinematics(np.zeros(3))
    l_home = float(np.linalg.norm(np.asarray(f0, dtype=float).reshape(3)))
    z_top = float(Z_TOP_M)
    both = args.z is None
    z_bot = float(Z_BOTTOM_M)
    z_plane = (z_top if both else float(np.clip(args.z, z_top, Z_BOTTOM_M)))
    yaw = drive_yaw_rad()
    centers = wheel_centers_xy(yaw)
    az_deg = [math.degrees(a) % 360.0 for a in wheel_az_L(yaw)]

    print(f"[trace] envelope: z {z_top:.3f}..{Z_BOTTOM_M:.3f} m, "
          f"L={l_home:.3f}..{LEN_MAX_M:.2f} m, swing r<={XY_MAX_M:.2f} m, "
          f"3 wheels at r={WHEEL_CENTER_M:.2f} m "
          f"az={'/'.join(f'{a:.1f}' for a in az_deg)} deg "
          f"keep={WHEEL_KEEP_M:.3f} m")
    r_top = math.sqrt(max(0.0, LEN_MAX_M ** 2 - z_top * z_top))
    r_bot = math.sqrt(max(0.0, LEN_MAX_M ** 2 - z_bot * z_bot))
    if both:
        print(f"[trace] BOTTOM z={z_bot:.3f} m r_free={r_bot:.3f} m, then "
              f"TOP z={z_top:.3f} m r_free={r_top:.3f} m, "
              f"speed={args.speed:.2f} m/s")
    else:
        r_free = math.sqrt(max(0.0, LEN_MAX_M ** 2 - z_plane * z_plane))
        print(f"[trace] outline at z={z_plane:.3f} m: free r={r_free:.3f} m, "
              f"speed={args.speed:.2f} m/s")

    if args.dry_run:
        zs = (z_bot, z_top) if both else (z_plane,)
        for zz in zs:
            way = boundary_waypoints(zz, centers, start_az=0.0, step_deg=5.0)
            print(f"  --- z={zz:.3f} m ---")
            for p in way[:: max(1, len(way) // 12)]:
                az = math.degrees(math.atan2(p[1], p[0])) % 360.0
                print(f"  az={az:6.1f} deg  r={math.hypot(p[0], p[1]):.3f} m")
        tee.close()
        return 0

    lc = lcm.LCM(args.lcm_url)
    robot = Robot(lc)

    print("[trace] waiting for hopper_data_lcmt from the Jetson driver...")
    t_wait = time.time()
    t_note = 0.0
    try:
        while True:
            q_meas, qd_meas, tau_iq, t_q = robot.get_state()
            if q_meas is not None and time.time() - t_q < 1.0:
                print(f"[trace] hopper_data ok  q={np.round(q_meas, 3)}")
                break
            now = time.time()
            if now - t_note > 2.0:
                t_note = now
                age = (now - t_wait)
                print(f"[trace] still no hopper_data ({age:.0f}s). "
                      "Need hopper-driver up, and PC multicast "
                      "(bash scripts/connect.sh). Ctrl-C to quit.")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[trace] quit while waiting for hopper_data")
        robot.running = False
        tee.close()
        return 1

    if robot.foreign_cmds > 0:
        print(f"[trace] WARNING: heard {robot.foreign_cmds} hopper_cmd msgs. "
              "Stop hopper-upper:  bash scripts/upper.sh stop")
    robot.stop_listening_cmd()
    robot.consume_x_edge()  # ignore X that happened before we were ready

    print("[trace] press gamepad X when ready (driver PD). "
          "B or Ctrl-C cancels. Upper stays stopped.")
    t_note = 0.0
    try:
        while True:
            if robot.consume_x_edge():
                print("[trace] X received -> 归中 first, then outline")
                break
            with robot._lock:
                b = int(robot.b_now)
            if b:
                print("[trace] B -> cancel, no motion")
                robot.running = False
                tee.close()
                return 1
            now = time.time()
            if now - t_note > 2.0:
                t_note = now
                gp = "ok" if time.time() - robot.t_gp < 1.0 else "NO gamepad_lcmt"
                print(f"[trace] waiting for X  (gamepad {gp})")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[trace] quit before X")
        robot.running = False
        tee.close()
        return 1

    q_meas, qd_meas, tau_iq, t_q = robot.get_state()
    if q_meas is None:
        print("[trace] ABORT: lost hopper_data after X")
        robot.running = False
        tee.close()
        return 1
    p_now = np.asarray(fk.forward_kinematics(q_meas)[0], float).reshape(3)
    print(f"[trace] start q={np.round(q_meas, 3)} foot={np.round(p_now, 3)}")

    # MOBILE first step: equal joints 归中, then outline from there.
    q_des = np.full(3, CENTER_Q, dtype=float)
    dt = 1.0 / RATE_HZ
    print(f"[trace] 归中 first (q={CENTER_Q:.2f}), same as MOBILE step 1")
    center_ok = False
    try:
        t_note = 0.0
        while True:
            t0 = time.time()
            q_meas, qd_meas, tau_iq, t_q = robot.get_state()
            if q_meas is None or time.time() - t_q > 2.0:
                print("[trace] ABORT: robot state stale during 归中")
                break
            with robot._lock:
                b = int(robot.b_now)
            if b:
                print("[trace] B -> cancel during 归中")
                break
            err = q_des - q_meas
            lag = float(np.max(np.abs(err)))
            if lag <= ARRIVE_RAD:
                p_now = np.asarray(
                    fk.forward_kinematics(q_meas)[0], float
                ).reshape(3)
                print(
                    f"[trace] CENTER done (|e|_max={lag:.3f} rad) "
                    f"q={np.round(q_meas, 3)} foot={np.round(p_now, 3)}"
                )
                center_ok = True
                break
            tau = np.clip(KP_JOINT * err, -TAU_MAX_NM, TAU_MAX_NM)
            robot.send(tau, AK60_KD)
            now = time.time()
            if now - t_note > 1.0:
                t_note = now
                print(f"[trace] 归中 lag={lag:.3f} rad")
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n[trace] Ctrl-C during 归中 -> safe stop")
    if not center_ok:
        robot.safe_stop()
        robot.running = False
        tee.close()
        return 1

    q_meas, qd_meas, tau_iq, t_q = robot.get_state()
    if q_meas is None:
        print("[trace] ABORT: lost hopper_data after 归中")
        robot.safe_stop()
        robot.running = False
        tee.close()
        return 1
    p_now = np.asarray(fk.forward_kinematics(q_meas)[0], float).reshape(3)

    # -- path: approach, bottom outline, optional lift + top outline -------
    start_az = math.atan2(p_now[1], p_now[0])
    way = envelope_path(
        centers, z_top=z_top, z_bot=(z_bot if both else z_plane),
        start_az=start_az, p_now=p_now, both=both, loops=args.loops)
    s_arr, way = path_resample(way)
    total = float(s_arr[-1])
    print(f"[trace] path {total:.2f} m -> ~{total / max(args.speed, 1e-3):.0f} s. "
          "Driver must be in PD (gamepad X). Ctrl-C = safe stop.")
    print(f"[trace] CSV -> {log_csv}")

    csv_fp = log_csv.open("w", newline="")
    csv_w = csv.writer(csv_fp)
    csv_w.writerow([
        "t", "s_m", "pct", "plane",
        "q0", "q1", "q2", "qd0", "qd1", "qd2",
        "q_cmd0", "q_cmd1", "q_cmd2",
        "tau0", "tau1", "tau2",
        "tauIq0", "tauIq1", "tauIq2",
        "foot_x", "foot_y", "foot_z",
        "tgt_x", "tgt_y", "tgt_z",
        "r_xy", "L_m", "lag_rad",
    ])

    q_cmd = q_meas.copy()
    s = 0.0
    dt = 1.0 / RATE_HZ
    t_warn = 0.0
    t_run0 = time.time()
    z_mid = 0.5 * (z_top + z_bot)
    try:
        while s < total:
            t0 = time.time()
            q_meas, qd_meas, tau_iq, t_q = robot.get_state()
            if q_meas is None or time.time() - t_q > 2.0:
                print("[trace] ABORT: robot state stale >2 s")
                break
            # resolved-rate IK on the commanded pose
            tgt = path_point(s_arr, way, s)
            p_cmd = np.asarray(
                fk.forward_kinematics(q_cmd)[0], float).reshape(3)
            p_meas = np.asarray(
                fk.forward_kinematics(q_meas)[0], float).reshape(3)
            dp = tgt - p_cmd
            J = numeric_jacobian(fk, q_cmd)
            dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3), J.T @ dp)
            dq = np.clip(dq, -DQ_STEP_MAX_RAD, DQ_STEP_MAX_RAD)
            q_cmd = np.clip(q_cmd + dq, Q_CMD_LIM_RAD[0], Q_CMD_LIM_RAD[1])
            # only advance along the path while the IK pose has caught up
            if float(np.linalg.norm(dp)) < 0.03:
                s += args.speed * dt
            # MOBILE swing params: joint-P as tau_ff + AK60 MIT damping
            tau = KP_JOINT * (q_cmd - q_meas)
            robot.send(tau, AK60_KD)
            lag = float(np.max(np.abs(q_cmd - q_meas)))
            pct = 100.0 * s / max(total, 1e-9)
            plane = "BOTTOM" if float(tgt[2]) >= z_mid - 1e-4 else "TOP"
            csv_w.writerow([
                f"{time.time() - t_run0:.4f}", f"{s:.5f}", f"{pct:.2f}", plane,
                *[f"{x:.6f}" for x in q_meas],
                *[f"{x:.6f}" for x in qd_meas],
                *[f"{x:.6f}" for x in q_cmd],
                *[f"{x:.6f}" for x in tau],
                *[f"{x:.6f}" for x in tau_iq],
                *[f"{x:.6f}" for x in p_meas],
                *[f"{x:.6f}" for x in tgt],
                f"{math.hypot(p_meas[0], p_meas[1]):.6f}",
                f"{float(np.linalg.norm(p_meas)):.6f}",
                f"{lag:.6f}",
            ])
            now = time.time()
            if now - t_warn > 2.0:
                t_warn = now
                csv_fp.flush()
                az = math.degrees(math.atan2(tgt[1], tgt[0])) % 360.0
                print(f"[trace] {pct:5.1f}%  {plane:6s}  az={az:6.1f} deg "
                      f"z={tgt[2]:.3f} r={math.hypot(tgt[0], tgt[1]):.3f} m "
                      f"lag={lag:.3f} rad"
                      + ("  << leg not following: gamepad X pressed?"
                         if lag > 0.4 else ""))
            time.sleep(max(0.0, dt - (time.time() - t0)))
        else:
            print("[trace] outline complete.")
    except KeyboardInterrupt:
        print("\n[trace] Ctrl-C -> safe stop")
    finally:
        robot.safe_stop()
        robot.running = False
        try:
            csv_fp.close()
        except Exception:
            pass
        print(f"[trace] CSV saved: {log_csv}")
        print("[trace] done. hopper-upper was left STOPPED; start it yourself:\n"
              "        bash scripts/upper.sh start")
        tee.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
