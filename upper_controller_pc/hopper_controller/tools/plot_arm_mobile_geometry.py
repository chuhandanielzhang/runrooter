#!/usr/bin/env python3
"""Arm + leg geometry for the MOBILE (M1/M2 folded, M3 flat) stance.

User-supplied arm segments (2026-07-25): COM -> fold joint 25 cm,
fold joint -> prop 30 cm, prop -> omni wheel 24 cm (total 79 cm).
ASSUMPTION to confirm: prop sits at 25+30=55 cm, wheel at the 79 cm tip.
Arm azimuths from modee/tools/spin_prop_test.py wiring notes:
M1 +30deg, M2 -90deg, M3 +150deg (body FRD, +X fwd, +Y right).

Left:  top view -- arms, fold joints, props, wheels, tip axis between the
       two folded-arm wheel contacts, COM offset, leg reach disc.
Right: side view along the tip axis -- M3 flat with prop up, M1/M2 wheels
       down, leg to ground, and the moment balance F_M3 * d_M3 = m*g * d_com.

Usage:  python tools/plot_arm_mobile_geometry.py
Output: logs/figs/arm_mobile_geometry.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "logs" / "figs" / "arm_mobile_geometry.png"

# ---- user-supplied geometry (m) ----
L_JOINT = 0.25      # COM -> fold joint
L_PROP = 0.30       # fold joint -> prop axis (ASSUMED position, confirm)
L_WHEEL = 0.24      # prop axis -> omni wheel tip
AZ = {"M1": 30.0, "M2": -90.0, "M3": 150.0}   # deg
FOLDED = ("M1", "M2")                          # arms folded ~90 deg down
MASS, G = 6.7, 9.81
LEG_REACH = 0.30    # lateral foot reach (from FK workspace plot)
LEG_GROUND = L_PROP + L_WHEEL  # folded outer segment length -> body height


def u(az_deg):
    a = np.deg2rad(az_deg)
    return np.array([np.cos(a), np.sin(a)])


def main():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7.2))

    # ================= TOP VIEW =================
    ax.set_aspect("equal")
    contacts = {}
    for name, az in AZ.items():
        d = u(az)
        p_joint = d * L_JOINT
        folded = name in FOLDED
        ax.plot([0, p_joint[0]], [0, p_joint[1]], color="k", lw=3)
        ax.scatter(*p_joint, marker="o", s=70, facecolor="white",
                   edgecolor="k", zorder=5)
        if folded:
            # outer segment points DOWN: wheel contact projects at the joint
            contacts[name] = p_joint
            ax.scatter(*p_joint, marker="s", s=140, color="tab:gray",
                       zorder=4, alpha=0.8)
            ax.annotate(f"{name} folded\nwheel @ {L_JOINT*100:.0f} cm",
                        p_joint + d * 0.05 + np.array([0.0, -0.02]),
                        fontsize=9, ha="left")
        else:
            p_prop = d * (L_JOINT + L_PROP)
            p_tip = d * (L_JOINT + L_PROP + L_WHEEL)
            ax.plot([p_joint[0], p_tip[0]], [p_joint[1], p_tip[1]],
                    color="k", lw=2)
            circ = plt.Circle(p_prop, 0.09, fill=False, color="tab:purple",
                              lw=2)
            ax.add_patch(circ)
            ax.scatter(*p_prop, marker="+", s=90, color="tab:purple")
            ax.scatter(*p_tip, marker="s", s=100, color="lightgray",
                       edgecolor="gray")
            ax.annotate(f"{name} FLAT: prop @ {(L_JOINT+L_PROP)*100:.0f} cm"
                        f" (thrust UP)", p_prop + [0.03, 0.10],
                        color="tab:purple", fontsize=9)

    # tip axis between the two wheel contacts
    p1, p2 = contacts["M1"], contacts["M2"]
    tdir = (p2 - p1) / np.linalg.norm(p2 - p1)
    lo, hi = p1 - tdir * 0.18, p2 + tdir * 0.18
    ax.plot([lo[0], hi[0]], [lo[1], hi[1]], color="crimson", ls="--", lw=2)
    ax.annotate("tip axis (2-wheel line)", (hi + [0.02, -0.05]),
                color="crimson", fontsize=10)

    # COM and perpendicular distances to the tip axis
    def dist_to_axis(p):
        return float(abs(np.cross(p2 - p1, p - p1)) / np.linalg.norm(p2 - p1))

    d_com = dist_to_axis(np.zeros(2))
    p_prop3 = u(AZ["M3"]) * (L_JOINT + L_PROP)
    d_m3 = dist_to_axis(p_prop3)
    f0 = MASS * G * d_com / max(1e-6, d_m3)

    ax.scatter(0, 0, marker="*", s=220, color="tab:orange", zorder=6)
    ax.annotate("COM (torso+leg)", (0.02, 0.02), fontsize=10,
                color="tab:orange")
    # foot reach disc
    ax.add_patch(plt.Circle((0, 0), LEG_REACH, fill=True, alpha=0.10,
                            color="tab:green"))
    ax.annotate("foot reach (top view)", (-LEG_REACH * 0.72, -LEG_REACH * 0.9),
                color="tab:green", fontsize=9)

    ax.set_title(
        f"TOP VIEW  |  d_com={d_com*100:.1f} cm   d_M3={d_m3*100:.1f} cm"
        f"   ->  static F_M3 = mg*d_com/d_M3 = {f0:.1f} N"
    )
    ax.set_xlabel("X body fwd (m)")
    ax.set_ylabel("Y body (m)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.95, 0.60)
    ax.set_ylim(-0.60, 0.60)

    # ================= SIDE VIEW =================
    # view along the tip axis: horizontal = distance from tip axis toward M3
    ax2.set_aspect("equal")
    h = LEG_GROUND  # arm-plane height above ground
    # ground
    ax2.axhline(0.0, color="gray", lw=3, alpha=0.6)
    ax2.text(-0.30, 0.012, "ground", color="gray", fontsize=9)
    # torso bar at height h, spanning from wheels (x=0) toward M3 side
    ax2.plot([-0.06, d_m3 + 0.10], [h, h], color="k", lw=5)
    # folded arm: from tip axis position at arm plane down to wheel
    ax2.plot([0, 0], [h, 0.045], color="k", lw=3)
    ax2.add_patch(plt.Circle((0, 0.045), 0.045, fill=False, color="k", lw=2))
    ax2.text(-0.28, 0.10, "M1/M2 folded\n(wheels)", fontsize=9)
    # COM
    ax2.scatter(d_com, h - 0.10, marker="*", s=240, color="tab:orange",
                zorder=6)
    ax2.annotate(f"COM\nd_com={d_com*100:.1f} cm", (d_com + 0.02, h - 0.14),
                 color="tab:orange", fontsize=9)
    ax2.annotate("", (d_com, h - 0.30), (d_com, h - 0.16),
                 arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2))
    ax2.text(d_com + 0.01, h - 0.26, "m g", color="tab:orange", fontsize=10)
    # M3 arm flat with prop thrust up
    ax2.scatter(d_m3, h, marker="+", s=140, color="tab:purple", zorder=6)
    ax2.annotate("", (d_m3, h + 0.18), (d_m3, h + 0.02),
                 arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2.5))
    ax2.text(d_m3 - 0.14, h + 0.20,
             f"M3 prop\nF0 = {f0:.1f} N", color="tab:purple", fontsize=10)
    # leg hanging to ground (foot near max extension)
    ax2.plot([d_com, d_com], [h - 0.02, 0.0], color="tab:brown", lw=2.5,
             ls="-")
    ax2.scatter(d_com, 0.0, s=70, color="crimson", zorder=6)
    ax2.annotate(f"leg to ground:\n{h*100:.0f} cm (FK max ext. 55 cm)",
                 (d_com + 0.03, 0.16), color="tab:brown", fontsize=9)
    # tilt axis marker
    ax2.annotate("tip axis (into page)", (0.01, h + 0.05), color="crimson",
                 fontsize=9)
    ax2.scatter(0, h, marker="x", s=90, color="crimson", zorder=6)

    ax2.set_title(
        "SIDE VIEW (along tip axis)  |  balance: F_M3*d_M3 = m*g*d_com"
    )
    ax2.set_xlabel("distance from tip axis toward M3 (m)")
    ax2.set_ylabel("height (m)")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(-0.35, 1.05)
    ax2.set_ylim(-0.06, 0.95)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")
    print(f"d_com = {d_com*100:.1f} cm, d_M3 = {d_m3*100:.1f} cm, "
          f"F0 = {f0:.2f} N  (m={MASS} kg)")


if __name__ == "__main__":
    main()
