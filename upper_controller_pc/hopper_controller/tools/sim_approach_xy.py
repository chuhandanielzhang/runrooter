"""XY validation of the MOBILE auto-approach law (offline, PC-side).

Simulates the EXACT deployed control law (modee/approach_law.py — the
same module lcm_controller.py imports) on a holonomic kiwi base with a
forward camera, from a fan of initial poses/heading angles, and renders
an animated GIF plus a summary PNG.

Two panels:
  LEFT  "baseline": yaw servo aligns to the wall normal only (the law
        before 2026-08-08) — oblique starts sweep the tag out of the
        ~69 deg FOV and the run stalls on memory timeout.
  RIGHT "deployed": FOV-keeping bearing/normal blend + detection-age
        speed tiers + lateral-then-advance corridor gate — every start
        converges to the pre-goal pose square to the wall with the tag
        kept in view.

World model
  - Wall along x = WALL_X (plot: vertical line), inward normal +X.
  - Button/tag on the wall at y = 0; pre-goal hover point PRE_OFF in
    front of it. Perception returns the pre point and the inward normal
    in the CURRENT body frame at DET_HZ, only while the tag is inside
    the camera FOV and range window (plus optional pixel-ish noise).
  - Base kinematics: FRD body (x fwd, y right), field-calibrated wheel
    IK sense wz>0 = CW from above -> world heading psi_dot = -wz.
  - Dropouts replay the controller's own memory path: the remembered
    body-frame target is advanced with approach_law.dead_reckon_step.

Run:  python3 tools/sim_approach_xy.py   (from hopper_controller/)
Outputs: logs/sim_approach_xy.gif, logs/sim_approach_xy.png
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modee.approach_law import approach_twist, dead_reckon_step, wrap_pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Wedge

# ---- scenario constants ----------------------------------------------------
WALL_X = 2.0          # wall plane (m)
BUTTON = np.array([WALL_X, 0.0])
N_WALL = np.array([1.0, 0.0])       # inward normal (into the wall)
PRE_OFF = 0.10                      # pre-goal hover distance from wall (m)
PRE_W = BUTTON - PRE_OFF * N_WALL   # world pre-goal point
# Sideways-mounted D435: the WORLD-horizontal FOV is the color sensor's
# vertical FOV (~42 deg), not the 69 deg horizontal one.
FOV_HALF = math.radians(42.0 / 2.0)
RANGE_MIN, RANGE_MAX = 0.15, 4.0
DET_HZ = 15.0                       # perception packet rate
CTRL_HZ = 50.0
T_END = 60.0
NOISE_M = 0.004                     # detection noise, 1-sigma (m)
ARRIVE_POS_M = 0.015
ARRIVE_YAW_RAD = math.radians(3.0)

# Sim button and tag are co-located, so the lateral goal offset (which
# exists to keep the REAL tag, 16.5 cm left of the button, in frame)
# is zeroed here; everything else uses the deployed defaults.
CFG = SimpleNamespace(
    mobile_approach_goal_x_m=0.40,
    mobile_approach_goal_y_m=0.0,
)
MEMORY_S = 3.0                      # approach_memory_timeout_s
STALE_S = 1.5                       # manip_button_stale_s

rng = np.random.default_rng(7)


def body_axes(psi: float) -> tuple[np.ndarray, np.ndarray]:
    """World unit vectors of body x (forward) and body y (RIGHT)."""
    return (np.array([math.cos(psi), math.sin(psi)]),
            np.array([math.sin(psi), -math.cos(psi)]))


def to_body(vec_w: np.ndarray, psi: float) -> np.ndarray:
    xb, yb = body_axes(psi)
    return np.array([float(vec_w @ xb), float(vec_w @ yb)])


def simulate(x0: float, y0: float, psi0: float, fov_blend: bool) -> dict:
    """Run one approach; returns trajectory + metrics."""
    cfg = SimpleNamespace(**vars(CFG))
    if not fov_blend:
        # Baseline: kill the bearing blend (normal-only yaw, old law).
        cfg.mobile_approach_blend_near_m = 0.0
        cfg.mobile_approach_blend_far_m = 0.0

    p = np.array([x0, y0], dtype=float)
    psi = float(psi0)
    dt = 1.0 / CTRL_HZ
    det_period = 1.0 / DET_HZ
    since_det = 1e9         # time since last accepted detection
    next_det_t = 0.0
    mem = None              # [pre_x, pre_y, n_x, n_y] in body frame
    last_twist = (0.0, 0.0, 0.0)

    traj, yaws, vis_flags = [], [], []
    t_arrive = None
    t = 0.0
    while t < T_END:
        # ---- perception model ----
        d_tag_w = BUTTON - p
        tag_b = to_body(d_tag_w, psi)
        rng_m = float(np.hypot(*tag_b))
        bearing = math.atan2(tag_b[1], tag_b[0])
        visible = (abs(bearing) < FOV_HALF and RANGE_MIN < rng_m < RANGE_MAX)
        vis_flags.append(visible)
        if visible and t >= next_det_t:
            pre_b = to_body(PRE_W - p, psi) + rng.normal(0.0, NOISE_M, 2)
            # Sim co-locates tag with button; still pass tag_* so the
            # deployed center-first FOV path is exercised.
            tag_b = to_body(BUTTON - p, psi) + rng.normal(0.0, NOISE_M, 2)
            n_b = to_body(N_WALL, psi)
            mem = [pre_b[0], pre_b[1], n_b[0], n_b[1], tag_b[0], tag_b[1]]
            since_det = 0.0
            next_det_t = t + det_period
        else:
            since_det += dt
            if mem is not None and since_det <= MEMORY_S:
                vx_l, vy_l, wz_l = last_twist
                mem[0], mem[1] = dead_reckon_step(
                    mem[0], mem[1], vx_l, vy_l, wz_l, dt, is_point=True)
                mem[4], mem[5] = dead_reckon_step(
                    mem[4], mem[5], vx_l, vy_l, wz_l, dt, is_point=True)
                mem[2], mem[3] = dead_reckon_step(
                    mem[2], mem[3], vx_l, vy_l, wz_l, dt, is_point=False)

        # ---- deployed control law (shared module) ----
        if mem is None or since_det > MEMORY_S:
            vx = vy = wz = 0.0     # controller stops, waits for re-capture
        else:
            vx, vy, wz = approach_twist(
                mem[0], mem[1], mem[2], mem[3], since_det, cfg,
                tag_x=mem[4], tag_y=mem[5],
            )
        last_twist = (vx, vy, wz)

        # ---- base kinematics (wz>0 = CW from above) ----
        xb, yb = body_axes(psi)
        p = p + (vx * xb + vy * yb) * dt
        psi = wrap_pi(psi - wz * dt)

        traj.append(p.copy())
        yaws.append(psi)

        # ---- arrival check (true state, like the READY monitor) ----
        pre_b_true = to_body(PRE_W - p, psi)
        yaw_err = math.atan2(*to_body(N_WALL, psi)[::-1])
        gx = float(getattr(CFG, "mobile_approach_goal_x_m", 0.40))
        gy = float(getattr(CFG, "mobile_approach_goal_y_m", 0.0))
        if (np.hypot(pre_b_true[0] - gx, pre_b_true[1] - gy) < ARRIVE_POS_M
                and abs(yaw_err) < ARRIVE_YAW_RAD):
            t_arrive = t
            break
        t += dt

    traj = np.asarray(traj)
    gx = float(getattr(CFG, "mobile_approach_goal_x_m", 0.40))
    gy = float(getattr(CFG, "mobile_approach_goal_y_m", 0.0))
    return {
        "traj": traj,
        "yaws": np.asarray(yaws),
        "vis_frac": float(np.mean(vis_flags)) if vis_flags else 0.0,
        "t_arrive": t_arrive,
        "final_pos_err_cm": float(np.hypot(
            *(to_body(PRE_W - traj[-1], yaws[-1]) - [gx, gy]))) * 100.0,
        "final_yaw_err_deg": abs(math.degrees(
            math.atan2(*to_body(N_WALL, yaws[-1])[::-1]))),
    }


def main() -> None:
    # Fan of initial poses: lateral offsets x heading offsets relative to
    # the tag bearing. All start with the tag inside the 34.5 deg half
    # FOV (LT requires a detection to arm the approach). The last rows
    # are the oblique-lateral stress cases (large lateral offset, camera
    # aimed at the tag): squaring up to the wall immediately would sweep
    # the tag out of view.
    starts = []
    for y0 in (-0.6, 0.0, 0.6):
        for dpsi in (-15.0, 0.0, 15.0):
            psi0 = math.atan2(BUTTON[1] - y0, BUTTON[0] - 0.6) \
                + math.radians(dpsi)
            starts.append((0.6, y0, psi0, dpsi))
    for y0 in (-1.5, 1.5):
        psi0 = math.atan2(BUTTON[1] - y0, BUTTON[0] - 0.7)
        starts.append((0.7, y0, psi0, 0.0))
    runs = {}
    for mode, fov in (("baseline", False), ("deployed", True)):
        runs[mode] = [
            (s, simulate(s[0], s[1], s[2], fov)) for s in starts
        ]

    print(f"{'mode':9s} {'y0':>5s} {'dpsi':>6s} {'arrive':>8s} "
          f"{'pos_err':>8s} {'yaw_err':>8s} {'tag_vis':>7s}")
    for mode in ("baseline", "deployed"):
        for (x0, y0, psi0, dpsi), r in runs[mode]:
            ta = f"{r['t_arrive']:.1f}s" if r["t_arrive"] else "STALL"
            print(f"{mode:9s} {y0:+5.1f} {dpsi:+6.0f} {ta:>8s} "
                  f"{r['final_pos_err_cm']:7.1f}cm "
                  f"{r['final_yaw_err_deg']:7.1f}° "
                  f"{100*r['vis_frac']:6.0f}%")

    # ---- render ----
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "logs")
    os.makedirs(outdir, exist_ok=True)

    n_frames = 160
    max_len = max(len(r["traj"]) for m in runs for _, r in runs[m])
    stride = max(1, max_len // n_frames)
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(1, len(starts) - 1)) for i in range(len(starts))]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5))
    titles = {
        "baseline": "baseline: normal-only yaw (pre-fix)",
        "deployed": "deployed: FOV blend + age tiers (approach_law.py)",
    }

    def draw_static(ax, mode):
        ax.axvline(WALL_X, color="k", lw=3)
        ax.plot(*BUTTON, "ks", ms=9, label="button/tag")
        ax.plot(*PRE_W, "r*", ms=14, label="pre-goal")
        ax.set_xlim(-0.2, 2.4)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(titles[mode], fontsize=11)
        ax.set_xlabel("x (m)")

    artists = {}
    for ax, mode in zip(axes, ("baseline", "deployed")):
        draw_static(ax, mode)
        items = []
        for i, (_, r) in enumerate(runs[mode]):
            (ln,) = ax.plot([], [], "-", color=colors[i], lw=1.4, alpha=0.85)
            (hd,) = ax.plot([], [], ">", color=colors[i], ms=7)
            wd = Wedge((0, 0), 0.34, 0, 1, color=colors[i], alpha=0.16)
            ax.add_patch(wd)
            items.append((ln, hd, wd, r))
        artists[mode] = items
    axes[0].set_ylabel("y (m)")
    axes[1].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        f"MOBILE auto-approach: {len(starts)} initial poses "
        "(lateral x heading offsets)  ->  converge square to the wall",
        fontsize=12)

    def update(f):
        k = min((f + 1) * stride, max_len)
        out = []
        for mode in ("baseline", "deployed"):
            for ln, hd, wd, r in artists[mode]:
                j = min(k, len(r["traj"])) - 1
                tr = r["traj"][: j + 1]
                ln.set_data(tr[:, 0], tr[:, 1])
                hd.set_data([tr[-1, 0]], [tr[-1, 1]])
                psi = r["yaws"][j]
                hd.set_marker((3, 0, math.degrees(psi) - 90.0))
                wd.set_center((tr[-1, 0], tr[-1, 1]))
                wd.set_theta1(math.degrees(psi - FOV_HALF))
                wd.set_theta2(math.degrees(psi + FOV_HALF))
                out += [ln, hd, wd]
        return out

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=50, blit=True)
    gif_path = os.path.join(outdir, "sim_approach_xy.gif")
    ani.save(gif_path, writer=animation.PillowWriter(fps=20))
    update(n_frames - 1)
    png_path = os.path.join(outdir, "sim_approach_xy.png")
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"saved: {gif_path}\nsaved: {png_path}")


if __name__ == "__main__":
    main()
