#!/usr/bin/env python3
"""Leg linkage + foot workspace + camera frame chain for the manipulation task.

Left: 3-RSR delta leg drawn from the real closed-form FK (forward_kinematics.py)
      with the reachable foot workspace cloud and an EXAMPLE camera frustum.
Right: side-view calibration diagram -- the transform chain to record.

The camera pose here is a PLACEHOLDER: replace CAM_POS_LEG / CAM_PITCH_DEG with
the SolidWorks measurement (mount face + optical-center offset from the
camera datasheet), then refine with the foot-tip AprilTag procedure.

Usage:  python tools/plot_leg_frames.py
Output: logs/figs/leg_frames.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forward_kinematics import ForwardKinematics, DeltaLegConfig  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "logs" / "figs" / "leg_frames.png"

# ---- Camera on the 2-DOF gimbal (user dims 2026-07-25) ----
# Leg-base frame, +Z = leg extension = down.  Chain:
#   Tz(47.7+110 mm) -> Rz(q_yaw) -> Ry(q_pitch) -> Tx(30 mm) -> camera,
# optical axis along the 30 mm link.  D435 mount->optical-center offset
# still to be added from the datasheet (left IR / RGB origin).
GIMBAL_DROP_M = 0.0477 + 0.110   # plate center -> yaw axis, straight down
GIMBAL_LINK_M = 0.030            # pitch axis -> camera
Q_YAW_DEG = 0.0                  # preset: facing +X (forward)
Q_PITCH_DEG = 30.0               # preset: pitched down
CAM_HFOV_DEG, CAM_VFOV_DEG = 87.0, 58.0     # RealSense D435 depth FOV

_yaw = np.deg2rad(Q_YAW_DEG)
_pit = np.deg2rad(Q_PITCH_DEG)
# optical axis in leg frame (+Z down): forward tilted down by q_pitch
_axis = np.array([
    np.cos(_pit) * np.cos(_yaw),
    np.cos(_pit) * np.sin(_yaw),
    np.sin(_pit),
])
CAM_POS_LEG = np.array([0.0, 0.0, GIMBAL_DROP_M]) + GIMBAL_LINK_M * _axis
CAM_PITCH_DEG = Q_PITCH_DEG


def knee_points(theta, cfg):
    pts = np.zeros((3, 3))
    for i in range(3):
        a = np.deg2rad(120 * i)
        R = np.array([[np.cos(a), -np.sin(a), 0.0],
                      [np.sin(a), np.cos(a), 0.0],
                      [0.0, 0.0, 1.0]])
        base = np.array([0.0, cfg.r, 0.0])
        off = np.array([0.0, cfg.D * np.cos(theta[i]), cfg.D * np.sin(theta[i])])
        pts[:, i] = R @ (base + off)
    return pts


def hip_points(cfg):
    pts = np.zeros((3, 3))
    for i in range(3):
        a = np.deg2rad(120 * i)
        R = np.array([[np.cos(a), -np.sin(a), 0.0],
                      [np.sin(a), np.cos(a), 0.0],
                      [0.0, 0.0, 1.0]])
        pts[:, i] = R @ np.array([0.0, cfg.r, 0.0])
    return pts


def draw_axes(ax, origin, R, length, labels=("X", "Y", "Z"), lw=2.0):
    colors = ("tab:red", "tab:green", "tab:blue")
    for k in range(3):
        v = R[:, k] * length
        ax.plot([origin[0], origin[0] + v[0]],
                [origin[1], origin[1] + v[1]],
                [origin[2], origin[2] + v[2]], color=colors[k], lw=lw)
        ax.text(*(origin + v * 1.15), labels[k], color=colors[k], fontsize=9)


def main():
    cfg = DeltaLegConfig()
    fk = ForwardKinematics(cfg)

    # ---- workspace sweep ----
    grid = np.linspace(-0.2, 1.3, 17)
    pts = []
    for t1 in grid:
        for t2 in grid:
            for t3 in grid:
                p, chk = fk.forward_kinematics([t1, t2, t3])
                if np.all(np.isfinite(p)) and np.abs(chk).max() < 1e-9 \
                        and 0.30 < p[2] < 0.60:
                    pts.append(p)
    pts = np.array(pts)

    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    # workspace cloud
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=pts[:, 2],
               cmap="viridis", alpha=0.25, label="foot workspace")

    # linkage at a nominal manipulation pose
    th = [0.6, 0.6, 0.6]
    foot, _ = fk.forward_kinematics(th)
    hips = hip_points(cfg)
    knees = knee_points(th, cfg)
    ang = np.linspace(0, 2 * np.pi, 60)
    ax.plot(cfg.r * np.cos(ang), cfg.r * np.sin(ang), 0.0 * ang,
            color="k", lw=2.0)
    for i in range(3):
        ax.plot([hips[0, i], knees[0, i]], [hips[1, i], knees[1, i]],
                [hips[2, i], knees[2, i]], color="tab:orange", lw=3.5,
                label="upper arm D=0.158" if i == 0 else None)
        ax.plot([knees[0, i], foot[0]], [knees[1, i], foot[1]],
                [knees[2, i], foot[2]], color="tab:brown", lw=2.5,
                label="rod d=0.398" if i == 0 else None)
    ax.scatter(*foot, color="crimson", s=60, label="foot (end-effector)")

    # leg-base frame
    draw_axes(ax, np.zeros(3), np.eye(3), 0.09,
              labels=("X_leg", "Y_leg", "Z_leg"))

    # EXAMPLE camera frame + frustum
    pitch = np.deg2rad(CAM_PITCH_DEG)
    # camera optical axis: forward (+X_leg) tilted toward +Z_leg (ground)
    R_cam = np.array([
        [np.cos(pitch), 0.0, np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)],
    ])  # columns: cam X (right), Y (down-ish), Z... simplified axis triad
    z_opt = np.array([np.cos(pitch), 0.0, np.sin(pitch)])  # optical axis
    draw_axes(ax, CAM_POS_LEG, R_cam, 0.06,
              labels=("", "", "Z_cam"), lw=1.6)
    depth = 0.45
    hw = depth * np.tan(np.deg2rad(CAM_HFOV_DEG / 2))
    hh = depth * np.tan(np.deg2rad(CAM_VFOV_DEG / 2))
    y_c = np.array([0.0, 1.0, 0.0])
    x_c = np.cross(y_c, z_opt)
    ctr = CAM_POS_LEG + z_opt * depth
    corners = [ctr + sx * hw * y_c + sy * hh * x_c
               for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    for c in corners:
        ax.plot(*np.c_[CAM_POS_LEG, c], color="tab:purple", lw=0.8, alpha=0.7)
    loop = np.array(corners + [corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="tab:purple",
            lw=0.8, alpha=0.7)
    ax.text(*(CAM_POS_LEG + np.array([0.0, 0.0, -0.04])),
            f"D435 on gimbal (yaw {Q_YAW_DEG:.0f}, pitch {Q_PITCH_DEG:.0f})",
            color="tab:purple", fontsize=9)

    ax.set_xlabel("X_leg (m)")
    ax.set_ylabel("Y_leg (m)")
    ax.set_zlabel("Z_leg (m)  [+Z = extension = toward ground]")
    ax.invert_zaxis()  # draw the foot below the base, like the real robot
    ax.set_title("3-RSR leg (FK-exact) + foot workspace + camera frustum")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=18, azim=-60)

    # ---- right: calibration chain diagram (side view, X-Z) ----
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_aspect("equal")
    base = np.array([0.0, 0.0])
    cam = np.array([CAM_POS_LEG[0], -CAM_POS_LEG[2]])   # plot up-positive
    foot2 = np.array([foot[0], -foot[2]])
    tgt = np.array([0.42, -0.35])

    ax2.plot([-0.08, 0.08], [0, 0], color="k", lw=5)
    ax2.annotate("leg base {L}\n(FK root)", base + [-0.07, 0.03], fontsize=9)
    ax2.scatter(*cam, marker="s", s=90, color="tab:purple", zorder=5)
    ax2.annotate("camera {C}", cam + [0.02, 0.02], color="tab:purple",
                 fontsize=9)
    ax2.scatter(*foot2, s=90, color="crimson", zorder=5)
    ax2.annotate("foot (EE)\n= FK(theta)", foot2 + [0.02, -0.02],
                 color="crimson", fontsize=9)
    ax2.scatter(*tgt, marker="*", s=180, color="tab:blue", zorder=5)
    ax2.annotate("button target\np_C from depth px", tgt + [0.015, 0.01],
                 color="tab:blue", fontsize=9)
    ax2.plot([base[0], foot2[0]], [base[1], foot2[1]], color="tab:orange",
             lw=2, ls="-")
    ax2.annotate("", cam, base,
                 arrowprops=dict(arrowstyle="<-", color="tab:purple", lw=1.6))
    ax2.text(0.02, cam[1] * 0.45, "T_L_C\n(SolidWorks init\n+ foot-tag calib)",
             color="tab:purple", fontsize=9)
    ax2.plot([cam[0], tgt[0]], [cam[1], tgt[1]], color="tab:blue",
             lw=1.2, ls="--")
    ax2.plot([-0.15, 0.55], [tgt[1] - 0.06, tgt[1] - 0.06], color="gray",
             lw=3, alpha=0.5)
    ax2.text(-0.14, tgt[1] - 0.05, "ground", color="gray", fontsize=9)
    ax2.text(-0.14, 0.16,
             "p_L = T_L_C @ p_C\nfoot cmd: theta = IK(p_L)",
             fontsize=11, family="monospace",
             bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
    ax2.set_xlim(-0.18, 0.60)
    ax2.set_ylim(-0.50, 0.25)
    ax2.set_xlabel("X (m, forward)")
    ax2.set_ylabel("height (m)")
    ax2.set_title("Calibration chain: camera -> leg base -> foot")
    ax2.grid(True, alpha=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}  (workspace points: {len(pts)})")


if __name__ == "__main__":
    main()
