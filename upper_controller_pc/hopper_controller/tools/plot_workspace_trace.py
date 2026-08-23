#!/usr/bin/env python3
"""Plot the work envelope and, if given one, a workspace_trace_*.csv run.

Left panel  XY, with each wheel barrel sliced at the three ring heights.
Right panel the barrel profile in (r, z) on a wheel ray, so the 3D shape is
            visible: wheel disk on the floor, opening to the widest layer,
            closing again symmetrically above.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from workspace_envelope import (  # noqa: E402
    KEEP_Z_CEIL,
    KEEP_Z_WIDE,
    Q_SAFE_MAX,
    RINGS,
    SAFE_AZ_DEG,
    WHEEL_AZ_L_DEG,
    WHEEL_CENTER_M,
    WHEEL_KEEP_M,
    XY_FLOOR_R_M,
    Z_BOTTOM_M,
    envelope_path,
    keep_cross,
    keep_near_r,
    keep_outline,
    reach_limit_r,
    ring_plan,
    wheel_centers_xy,
)

RING_COLOR = {"BOTTOM": "#1f77b4", "MID": "#ff7f0e", "TOP": "#2ca02c"}
PATH_COLOR = {"BOTTOM": "#1f77b4", "MID": "#ff7f0e", "TOP": "#2ca02c",
              "LIFT": "#7f7f7f", "APPROACH": "#7f7f7f", "START": "#7f7f7f"}


def _ring_label(p: dict) -> str:
    how = "duct loop" if p["closed"] else f"{len(p['arcs'])} arcs"
    return (f"{p['name']} z={p['z']:.3f} "
            f"r={p['r_wheel']:.3f}..{p['r_hole']:.3f} {how}")


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    out: dict[str, np.ndarray] = {}
    for key in ("t", "pct", "foot_x", "foot_y", "foot_z",
                "tgt_x", "tgt_y", "tgt_z", "lag_rad"):
        out[key] = np.asarray([float(r[key]) for r in rows], float)
    out["plane"] = np.asarray([r["plane"] for r in rows])
    return out


def _draw_xy(ax, d, marks) -> None:
    ax.set_aspect("equal", adjustable="box")

    ax.add_patch(plt.Circle((0.0, 0.0), XY_FLOOR_R_M, fill=True, alpha=0.16,
                            color="#8e44ad", lw=0.0))
    ax.add_patch(plt.Circle((0.0, 0.0), XY_FLOOR_R_M, fill=False, lw=1.2,
                            color="#8e44ad",
                            label=f"floor hit r<{XY_FLOOR_R_M:.2f} at z_bottom"))

    for i, (cx, cy) in enumerate(wheel_centers_xy()):
        ax.plot(cx, cy, "x", color="#c0392b", ms=5,
                label="wheel centre" if i == 0 else None)
    ax.plot(0.0, 0.0, "+", color="0.2", ms=10, mew=1.2)

    # barrel sliced at each ring height, plus the widest layer
    slices = [(name, z, RING_COLOR[name], "--") for name, z in RINGS]
    slices.append(("widest", KEEP_Z_WIDE, "#c0392b", "-"))
    for name, z, color, ls in slices:
        for j, az_w in enumerate(WHEEL_AZ_L_DEG):
            o = keep_outline(z, az_w)
            ax.plot(o[:, 0], o[:, 1], color=color, ls=ls, lw=1.1, alpha=0.85,
                    label=(f"keep-out @ z={z:.3f} ({name})"
                           if j == 0 else None))
            if name == "widest":
                ax.fill(o[:, 0], o[:, 1], color=color, alpha=0.07, lw=0)

    for a in SAFE_AZ_DEG:
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        ax.plot([0, 0.40 * ca], [0, 0.40 * sa], color="#2e7d32", ls=":",
                lw=1.0, label="safe lift azimuth" if a == SAFE_AZ_DEG[0] else None)

    # raw joint-stop reach boundary: 3-fold, bulging at each hole
    azs = np.radians(np.arange(0.0, 360.5, 1.0))
    for name, z in RINGS:
        rr = np.asarray([reach_limit_r(z, a) for a in azs])
        ax.plot(rr * np.cos(azs), rr * np.sin(azs), color=RING_COLOR[name],
                ls=(0, (1, 2)), lw=1.0, alpha=0.7,
                label=f"q={Q_SAFE_MAX:.2f} reach @ z={z:.3f}")

    way, tags = envelope_path(np.array([0.0, 0.0, 0.354]))
    for name in ("BOTTOM", "MID", "TOP"):
        m = np.asarray([t == name for t in tags])
        seg = way[m]
        ax.plot(seg[:, 0], seg[:, 1], ".", color=RING_COLOR[name], ms=2.2,
                alpha=0.85, label=f"plan {name}")

    if d is not None:
        lag_ok = d["lag_rad"] <= 0.4
        for name in ("BOTTOM", "MID", "TOP"):
            m = (d["plane"] == name) & lag_ok
            if np.any(m):
                ax.plot(d["foot_x"][m], d["foot_y"][m], color=RING_COLOR[name],
                        lw=1.8, label=f"foot {name}")
        bad = ~lag_ok
        if np.any(bad):
            ax.plot(d["foot_x"][bad], d["foot_y"][bad], color="#e74c3c",
                    lw=1.6, alpha=0.85, label="foot lag>0.4 rad")
        ax.plot(d["foot_x"][0], d["foot_y"][0], "o", color="0.15", ms=5)

    for x, y, _z, lab, col in marks:
        ax.plot(x, y, "o", color=col, ms=9, zorder=8, label=lab)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=6, framealpha=0.92, ncol=2)
    lim = 0.40
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def _draw_rz(ax, d, marks) -> None:
    """Barrel profile on the wheel ray: r across, z down the page.

    Only points that sit on a wheel ray belong in this slice, so the plan is
    shown by the carved waypoints (the ones hugging the inner face) plus the
    ring heights; everything else is at azimuths this slice does not cover.
    """
    zs = np.linspace(KEEP_Z_CEIL, Z_BOTTOM_M, 241)
    inner = np.asarray([keep_cross(z)[0] - keep_cross(z)[1] for z in zs])
    outer = np.asarray([keep_cross(z)[0] + keep_cross(z)[1] for z in zs])
    ax.fill_betweenx(zs, inner, outer, color="#c0392b", alpha=0.16, lw=0,
                     label="wheel barrel, on the wheel ray")
    ax.plot(inner, zs, color="#c0392b", lw=1.3)
    ax.plot(outer, zs, color="#c0392b", lw=1.3)
    ax.plot([inner[0], outer[0]], [KEEP_Z_CEIL, KEEP_Z_CEIL],
            color="#c0392b", lw=1.3)
    ax.axhline(Z_BOTTOM_M, color="#8e44ad", lw=1.2,
               label=f"floor z={Z_BOTTOM_M:.3f}")
    ax.plot([WHEEL_CENTER_M - WHEEL_KEEP_M, WHEEL_CENTER_M + WHEEL_KEEP_M],
            [Z_BOTTOM_M, Z_BOTTOM_M], color="#c0392b", lw=4.0,
            solid_capstyle="butt", label="wheel disk (floor)")
    ax.axhline(KEEP_Z_WIDE, color="#c0392b", ls=":", lw=1.0,
               label=f"widest layer z={KEEP_Z_WIDE:.3f}")
    ax.axhline(KEEP_Z_CEIL, color="#c0392b", ls=":", lw=1.0,
               label=f"mirrored top z={KEEP_Z_CEIL:.3f}")

    way, tags = envelope_path(np.array([0.0, 0.0, 0.354]))
    r = np.hypot(way[:, 0], way[:, 1])
    carved = np.asarray([
        t in RING_COLOR and abs(rr - keep_near_r(math.atan2(p[1], p[0]),
                                                 p[2])) < 1e-9
        for t, rr, p in zip(tags, r, way)
    ])
    for name in ("BOTTOM", "MID", "TOP"):
        m = np.asarray([t == name for t in tags])
        ax.axhline(float(way[m][:, 2].mean()), color=RING_COLOR[name],
                   lw=0.9, alpha=0.55, label=f"ring {name}")
        mc = m & carved
        if np.any(mc):
            ax.plot(r[mc], way[mc][:, 2], ".", color=RING_COLOR[name], ms=3.0,
                    label=f"{name} on the barrel face")

    if d is not None:
        ax.plot(np.hypot(d["foot_x"], d["foot_y"]), d["foot_z"], color="0.1",
                lw=0.7, alpha=0.45, label="foot (all azimuths)")

    ax.invert_yaxis()
    ax.set_xlabel("r_xy (m)")
    ax.set_ylabel("z (m, +Z down)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=6, framealpha=0.92)
    ax.set_xlim(0.0, 0.40)


def plot(d, out: Path, title: str, marks) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), dpi=140,
                             gridspec_kw={"width_ratios": [1.0, 0.78]})
    _draw_xy(axes[0], d, marks)
    _draw_rz(axes[1], d, marks)
    axes[0].set_title("XY (barrel sliced at each ring height)")
    axes[1].set_title("wheel ray cross-section")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _barrel_surface(az_w: float, n_z: int = 36, n_th: int = 48):
    """(X, Y, Zdisp) mesh of one keep-out barrel. Zdisp = -z so the floor
    sits at the bottom of a matplotlib 3D axes."""
    zs = np.linspace(KEEP_Z_CEIL, Z_BOTTOM_M, n_z)
    th = np.linspace(0.0, 2.0 * math.pi, n_th)
    c, s = math.cos(math.radians(az_w)), math.sin(math.radians(az_w))
    X = np.empty((n_z, n_th))
    Y = np.empty((n_z, n_th))
    Z = np.empty((n_z, n_th))
    for i, z in enumerate(zs):
        rc, a, b = keep_cross(z)
        if a <= 0.0:
            a = b = 1e-4
        for j, t in enumerate(th):
            du, dn = a * math.cos(t), b * math.sin(t)
            X[i, j] = (rc + du) * c - dn * s
            Y[i, j] = (rc + du) * s + dn * c
            Z[i, j] = -z
    return X, Y, Z


def plot_3d(d, out: Path, title: str) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9.6, 8.4), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    for i, az_w in enumerate(WHEEL_AZ_L_DEG):
        X, Y, Z = _barrel_surface(az_w)
        ax.plot_surface(X, Y, Z, color="#c0392b", alpha=0.18, linewidth=0,
                        antialiased=True, shade=True,
                        label="keep-out barrel" if i == 0 else None)
        ax.plot(X[0], Y[0], Z[0], color="#c0392b", lw=0.6, alpha=0.5)
        ax.plot(X[-1], Y[-1], Z[-1], color="#c0392b", lw=0.6, alpha=0.5)

    if d is not None:
        ax.plot(d["tgt_x"], d["tgt_y"], -d["tgt_z"], color="0.55", lw=0.7,
                alpha=0.55, label="planned")
        for name, color in RING_COLOR.items():
            m = d["plane"] == name
            if not np.any(m):
                continue
            ax.plot(d["foot_x"][m], d["foot_y"][m], -d["foot_z"][m],
                    color=color, lw=1.5, label=f"foot {name}")
        lift = np.isin(d["plane"], ("LIFT", "APPROACH", "START"))
        if np.any(lift):
            ax.plot(d["foot_x"][lift], d["foot_y"][lift], -d["foot_z"][lift],
                    color="#7f7f7f", lw=1.1, label="foot lift/approach")
        ax.plot([d["foot_x"][0]], [d["foot_y"][0]], [-d["foot_z"][0]],
                "o", color="0.15", ms=5, label="start")
        ax.plot([d["foot_x"][-1]], [d["foot_y"][-1]], [-d["foot_z"][-1]],
                "s", color="0.15", ms=5, label="end")
    else:
        way, tags = envelope_path(np.array([0.0, 0.0, 0.354]))
        tags = np.asarray(tags)
        for name, color in {**RING_COLOR,
                            "LIFT": "#7f7f7f",
                            "APPROACH": "#7f7f7f"}.items():
            m = tags == name
            if np.any(m):
                ax.plot(way[m, 0], way[m, 1], -way[m, 2],
                        color=color, lw=1.4, label=name)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("−z (m, floor at bottom)")
    lim = 0.40
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-Z_BOTTOM_M - 0.02, -KEEP_Z_CEIL + 0.06)
    try:
        ax.set_box_aspect((1, 1, 0.72))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="")
    p.add_argument("-o", "--out", default="")
    p.add_argument("--plan", action="store_true",
                   help="draw the envelope and planned path only, no log")
    p.add_argument("--mark", default="",
                   help="extra foot position(s) to point out, "
                        "'x,y,z[:label]' separated by ';'")
    args = p.parse_args()

    marks = []
    for i, spec in enumerate([s for s in args.mark.split(";") if s.strip()]):
        body, _, lab = spec.partition(":")
        x, y, z = (float(v) for v in body.split(","))
        marks.append((x, y, z, lab or f"mark {i + 1}",
                      ["#0b57d0", "#00897b", "#6a1b9a"][i % 3]))

    if args.plan or not args.csv:
        out = Path(args.out) if args.out else (
            _HERE.parent / "logs_local" / "workspace_envelope_plan.png")
        rings = ", ".join(_ring_label(p) for p in ring_plan())
        way, _ = envelope_path(np.array([0.0, 0.0, 0.354]))
        length = float(np.linalg.norm(np.diff(way, axis=0), axis=1).sum())
        plot(None, out,
             f"planned work envelope: {rings}\n"
             f"{len(way)} waypoints, {length:.2f} m", marks)
        print(f"saved {out}")
        return

    csv_path = Path(args.csv).expanduser().resolve()
    out = Path(args.out).expanduser() if args.out else csv_path.with_name(
        csv_path.stem + "_xy.png")
    d = load_csv(csv_path)
    counts = " / ".join(f"{n} {int(np.sum(d['plane'] == n))}"
                        for n, _z in RINGS)
    title = (f"{csv_path.stem}  {counts} samples  t={d['t'][-1]:.1f}s")
    plot(d, out, title, marks)
    print(f"saved {out}")
    out3d = out.with_name(out.stem.replace("_xy", "") + "_3d.png")
    if out3d == out:
        out3d = out.with_name(out.stem + "_3d.png")
    plot_3d(d, out3d, title)
    print(f"saved {out3d}")


if __name__ == "__main__":
    main()
