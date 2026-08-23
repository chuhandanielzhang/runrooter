#!/usr/bin/env python3
"""Trace the deployed-leg WORK ENVELOPE with the foot, ring by ring.

Run ON the Jetson (PC only launches it):

  bash scripts/trace_workspace.sh

The script SSHs to the robot, stops hopper-upper, then this process
waits. Press gamepad X (driver PD) to start.

Three rings bottom-up: BOTTOM at floor level, MID, TOP.  The geometry and
the path live in workspace_envelope.py; check_workspace_path.py replays the
whole thing offline against the joint stops before you drive it.

Logs land in hopper_controller/logs/ on the Jetson and are copied back
to the PC logs_local/.
"""

from __future__ import annotations

import argparse
import csv
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

# ---- envelope geometry (mirrored by ModeELCMConfig.work_*) ---------------
from workspace_envelope import (  # noqa: E402
    LEN_MAX_M,
    Q_SAFE_MAX,
    Q_SAFE_MIN,
    RINGS,
    SAFE_AZ_DEG,
    WHEEL_AZ_L_DEG,
    WHEEL_CENTER_M,
    WHEEL_KEEP_M,
    XY_FLOOR_R_M,
    Z_BOTTOM_M,
    Z_MID_M,
    Z_TOP_M,
    envelope_path,
    hole_inner_radius,
    keep_cross,
    keep_inside,
    ring_plan,
)

# ---- MOBILE leg-stow swing parameters ------------------------------------
KP_JOINT = 20.0        # mobile_leg_kp
TAU_MAX_NM = 3.0       # mobile_leg_tau_max_nm
AK60_KD = 2.0          # mobile_leg_ak60_kd_move
CENTER_Q = 0.4         # mobile_leg_center_q (归中)
ARRIVE_RAD = 0.10      # mobile_leg_arrive_rad

RATE_HZ = 100.0
DQ_STEP_MAX_RAD = 0.03     # per-tick IK step clip
# Was (-1.2, 1.6): the old rings asked for q=1.44 and q2=-1.2, so the leg
# stalled against its stops instead of following. Clip at the real ones.
Q_CMD_LIM_RAD = (Q_SAFE_MIN, Q_SAFE_MAX)


def path_resample(way: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative arc length s_i for waypoint interpolation."""
    seg = np.linalg.norm(np.diff(way, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, way


def path_point(s_arr: np.ndarray, way: np.ndarray,
               s: float) -> tuple[np.ndarray, int]:
    s = float(np.clip(s, 0.0, float(s_arr[-1])))
    i = int(np.searchsorted(s_arr, s, side="right")) - 1
    i = max(0, min(i, len(way) - 2))
    ds = float(s_arr[i + 1] - s_arr[i])
    t = 0.0 if ds <= 1e-12 else (s - float(s_arr[i])) / ds
    return way[i] * (1.0 - t) + way[i + 1] * t, i + (1 if t > 0.5 else 0)


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
    ap.add_argument("--rings", default="all",
                    help="comma list of rings to trace, bottom-up. "
                         f"Default all: {','.join(r[0] for r in RINGS)}")
    ap.add_argument("--speed", type=float, default=0.08,
                    help="foot speed along the outline, m/s (default 0.08)")
    ap.add_argument("--loops", type=int, default=1,
                    help="how many full outlines per ring (default 1)")
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
    want = str(args.rings).strip().lower()
    if want in ("", "all"):
        rings = RINGS
    else:
        names = [w.strip().upper() for w in want.split(",") if w.strip()]
        rings = tuple(r for r in RINGS if r[0] in names)
        if not rings:
            print(f"[trace] ABORT: --rings {args.rings} matched nothing of "
                  f"{[r[0] for r in RINGS]}")
            tee.close()
            return 1
    print(f"[trace] L={l_home:.3f}..{LEN_MAX_M:.2f} m, "
          f"q stops [{Q_SAFE_MIN:+.2f}, {Q_SAFE_MAX:+.2f}] rad, "
          f"floor-keep r<{XY_FLOOR_R_M:.2f} m at BOTTOM")
    print(f"[trace] 3 wheel barrels, az="
          f"{'/'.join(f'{a:.0f}' for a in WHEEL_AZ_L_DEG)} deg: "
          f"wheel disk r={WHEEL_CENTER_M:.2f}+/-{WHEEL_KEEP_M:.3f} m at the "
          "floor, opening to an ellipse and closing again above")
    for z in (Z_BOTTOM_M, Z_MID_M, Z_TOP_M):
        rc, a, b = keep_cross(z)
        print(f"[trace]   z={z:.3f} m: centre r={rc:.3f} a={a:.3f} "
              f"b={b:.3f} -> blocks r {rc - a:.3f}..{rc + a:.3f} m")
    for ring in ring_plan(rings):
        if ring["closed"]:
            duct = keep_cross(ring["z"])[0] - keep_cross(ring["z"])[1]
            how = f"closed loop, weaves through the r={duct:.3f} m duct"
        else:
            how = ("3 holes once each: " + ", ".join(
                f"{a1 - a0:.0f} deg at az {(a0 + a1) / 2 % 360:.0f}"
                for a0, a1 in ring["arcs"])
                + f", dipping to r={hole_inner_radius(ring['z'], ring['r_hole']):.3f} m"
                + "; retracts to the ring below between holes")
        shape = ("bulges at the holes"
                 if ring["r_hole"] - ring["r_wheel"] > 1e-4
                 else f"circle, capped by L<={LEN_MAX_M:.2f} m")
        print(f"[trace] ring {ring['name']:6s} z={ring['z']:.3f} m "
              f"r={ring['r_wheel']:.3f}..{ring['r_hole']:.3f} m "
              f"({shape})  {how}")
    print(f"[trace] height changes only at az="
          f"{'/'.join(f'{a:.0f}' for a in SAFE_AZ_DEG)} deg")
    print(f"[trace] speed={args.speed:.2f} m/s, loops={args.loops}")

    if args.dry_run:
        way, tags = envelope_path(np.array([0.0, 0.0, l_home]),
                                  rings=rings, loops=args.loops)
        s_arr, _ = path_resample(way)
        print(f"  path {len(way)} waypoints, {float(s_arr[-1]):.2f} m")
        bad = [i for i, p in enumerate(way) if keep_inside(*p)]
        print(f"  waypoints inside a keep-out: {len(bad)}")
        for i in range(0, len(way), max(1, len(way) // 24)):
            p = way[i]
            az = math.degrees(math.atan2(p[1], p[0])) % 360.0
            print(f"  {tags[i]:8s} az={az:6.1f} deg  "
                  f"r={math.hypot(p[0], p[1]):.3f} z={p[2]:.3f} m")
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

    # -- path: approach, then each ring bottom-up -------------------------
    way, tags = envelope_path(p_now, rings=rings, loops=args.loops)
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
    try:
        while s < total:
            t0 = time.time()
            q_meas, qd_meas, tau_iq, t_q = robot.get_state()
            if q_meas is None or time.time() - t_q > 2.0:
                print("[trace] ABORT: robot state stale >2 s")
                break
            # resolved-rate IK on the commanded pose
            tgt, i_way = path_point(s_arr, way, s)
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
            plane = tags[min(i_way, len(tags) - 1)]
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
