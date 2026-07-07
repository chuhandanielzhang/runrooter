#!/usr/bin/env python3
"""
Interactive 3-RSR plant viewer — print hip motor q, qd, torque in real time.

Hip motors = ctrl_joint_1/2/3  (qpos indices 7,10,13; qvel 6,9,12)
Torque    = actuator_force on ctrl_motor_1/2/3  (Nm, ±25 limit in XML)

Usage:
  cd Hopper-mujoco-standalone/3RSR_package_2
  python3 view_motors.py              # viewer + console HUD
  python3 view_motors.py --no-viewer  # console only (SSH / headless)
  python3 view_motors.py --test-m1    # hold motor1, others at home → show coupling
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three_leg_3rsr_closed.xml")
HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]
HIP_ACT = [0, 1, 2]  # ctrl_motor_1/2/3
HOME_KEY = "home"


def _fmt_row(label: str, v: np.ndarray, unit: str = "") -> str:
    v = np.asarray(v, dtype=float).reshape(3)
    return f"{label:6s} [{v[0]:+8.4f}, {v[1]:+8.4f}, {v[2]:+8.4f}]{unit}"


def read_motors(m: mujoco.MjModel, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.array([d.qpos[i] for i in HIP_QPOS])
    qd = np.array([d.qvel[i] for i in HIP_QVEL])
    tau = np.array([d.actuator_force[i] for i in HIP_ACT])
    return q, qd, tau


def print_hud(t: float, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, extra: str = "") -> None:
    deg = np.rad2deg(q)
    print(
        f"\r t={t:6.3f}s  "
        f"q(rad)=[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f}]  "
        f"q(deg)=[{deg[0]:+.1f},{deg[1]:+.1f},{deg[2]:+.1f}]  "
        f"qd=[{qd[0]:+.2f},{qd[1]:+.2f},{qd[2]:+.2f}]  "
        f"tau=[{tau[0]:+.2f},{tau[1]:+.2f},{tau[2]:+.2f}] Nm"
        + (f"  {extra}" if extra else ""),
        end="",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true", help="console only")
    ap.add_argument("--duration-s", type=float, default=0.0, help="0 = run until closed")
    ap.add_argument("--print-hz", type=float, default=10.0)
    ap.add_argument(
        "--test-m1",
        action="store_true",
        help="ramp hip1 only (+30 deg); hip2/3 held at home → see cross-coupling torques",
    )
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
    mujoco.mj_resetDataKeyframe(m, d, kid)
    home_ctrl = m.key_ctrl[kid].copy()
    d.ctrl[:] = home_ctrl
    dt = float(m.opt.timestep)

    viewer = None
    if not args.no_viewer:
        try:
            import mujoco.viewer as mj_viewer

            viewer = mj_viewer.launch_passive(m, d)
            print("[view_motors] MuJoCo viewer open. Drag to orbit; close window to exit.")
        except Exception as e:
            print(f"[view_motors] viewer failed ({e}); falling back to console.", file=sys.stderr)

    print("=== 3-RSR hip motors: ctrl_joint_1/2/3 ===")
    print("  q   = joint position [rad]  (motor 1,2,3)")
    print("  qd  = joint velocity [rad/s]")
    print("  tau = actuator force   [Nm]  (±25 Nm limit in XML)")
    if args.test_m1:
        print("  TEST: ramp motor1 to +30 deg; motors 2&3 held at home (parallel coupling visible in tau)")
    print()

    t0 = time.time()
    sim_t = 0.0
    last_print = 0.0
    print_dt = 1.0 / max(1e-3, float(args.print_hz))

    try:
        while True:
            if args.duration_s > 0 and sim_t >= args.duration_s:
                break
            if viewer is not None and not viewer.is_running():
                break

            # optional single-motor test
            if args.test_m1:
                ramp = float(np.clip(sim_t / 2.0, 0.0, 1.0))
                target1 = home_ctrl[0] + ramp * np.deg2rad(30.0)
                d.ctrl[0] = target1
                d.ctrl[1] = home_ctrl[1]
                d.ctrl[2] = home_ctrl[2]
            else:
                d.ctrl[:] = home_ctrl

            mujoco.mj_step(m, d)
            sim_t += dt

            if sim_t - last_print >= print_dt:
                last_print = sim_t
                q, qd, tau = read_motors(m, d)
                extra = ""
                if args.test_m1:
                    extra = f"|tau23|={np.linalg.norm(tau[1:3]):.2f}Nm (coupling)"
                print_hud(sim_t, q, qd, tau, extra)

            if viewer is not None:
                viewer.sync()
            elif not args.no_viewer:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass

    print()
    q, qd, tau = read_motors(m, d)
    print("\n--- final ---")
    print(_fmt_row("q", q, " rad"))
    print(_fmt_row("q(deg)", np.rad2deg(q)))
    print(_fmt_row("qd", qd, " rad/s"))
    print(_fmt_row("tau", tau, " Nm"))
    print(f"wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
