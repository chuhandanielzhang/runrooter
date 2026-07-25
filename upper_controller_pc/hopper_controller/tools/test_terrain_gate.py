#!/usr/bin/env python3
"""Synthetic-scene test for modee/terrain_gate.py (no hardware needed).

Simulates the D435 on the gimbal (pitch 30 deg, ~0.37 m above ground in
the MOBILE stance) approaching a step of configurable height, ray-casting
a depth image per frame, and running the TerrainGate.

Scenes: 12 cm step -> expect WHEEL far away, HOP once inside lookahead;
2 cm cable -> stays WHEEL; 40 cm wall -> STOP.

Usage:  python tools/test_terrain_gate.py
Output: logs/figs/terrain_gate_test.png + pass/fail prints
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modee.terrain_gate import (  # noqa: E402
    TerrainGate, TerrainGateConfig, depth_to_ground_points,
)

OUT = Path(__file__).resolve().parent.parent / "logs" / "figs" / "terrain_gate_test.png"

# Camera pose in ground frame (MOBILE stance, gimbal preset pitch 30 deg).
PITCH = np.deg2rad(30.0)
CAM_T = np.array([0.026, 0.0, 0.367])
# columns = camera axes in ground coords (X fwd, Y left, Z up):
# x_cam(image right) -> -Y ; y_cam(image down) -> down/back ; z_cam -> fwd/down
CAM_R = np.array([
    [0.0, -np.sin(PITCH), np.cos(PITCH)],
    [-1.0, 0.0, 0.0],
    [0.0, -np.cos(PITCH), -np.sin(PITCH)],
])
W, H, FX, FY, CX, CY = 848, 480, 425.0, 425.0, 424.0, 240.0


def render_depth(x_step_rel: float, h_step: float) -> np.ndarray:
    """Ray-cast heightfield: floor z=0 for x<x_step, z=h_step beyond."""
    us, vs = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    v_cam = np.stack([(us - CX) / FX, (vs - CY) / FY, np.ones_like(us)], -1)
    v_g = v_cam @ CAM_R.T                     # ray dir (per z_cam unit)
    depth = np.full((H, W), np.nan)
    for h_plane, cond in ((0.0, "lt"), (h_step, "ge")):
        dz = v_g[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            zc = (h_plane - CAM_T[2]) / dz
        x_hit = CAM_T[0] + zc * v_g[..., 0]
        ok = (zc > 0.05) & np.isfinite(zc)
        ok &= (x_hit < x_step_rel) if cond == "lt" else (x_hit >= x_step_rel)
        depth = np.where(ok & (~np.isfinite(depth) | (zc < depth)), zc, depth)
    # vertical face of the step
    vx = v_g[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        zc = (x_step_rel - CAM_T[0]) / vx
    z_hit = CAM_T[2] + zc * v_g[..., 2]
    ok = (zc > 0.05) & np.isfinite(zc) & (z_hit >= 0.0) & (z_hit <= h_step)
    depth = np.where(ok & (~np.isfinite(depth) | (zc < depth)), zc, depth)
    return depth


def drive_at_step(h_step: float, cfg: TerrainGateConfig):
    gate = TerrainGate(cfg)
    gate._latched = "WHEEL"  # start already rolling
    xs = np.arange(2.0, 0.35, -0.05)   # step distance ahead of robot
    latched, votes = [], []
    for xr in xs:
        d = render_depth(xr, h_step)
        pts = depth_to_ground_points(d, FX, FY, CX, CY, CAM_R, CAM_T, stride=4)
        m, v = gate.update(pts)
        latched.append(m)
        votes.append(v)
    return xs, latched, votes


def main():
    cfg = TerrainGateConfig()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    results = {}
    for ax, (h, label, expect) in zip(axes, [
        (0.02, "2 cm cable", "WHEEL"),
        (0.12, "12 cm step", "HOP"),
        (0.40, "40 cm wall", "STOP"),
    ]):
        xs, latched, votes = drive_at_step(h, cfg)
        final = latched[-1]
        results[label] = (final, expect)
        # profile snapshot when the step is 1.0 m ahead
        k = int(np.argmin(np.abs(xs - 1.0)))
        v = votes[k]
        ax.plot(v.profile_x, v.profile_z * 100, "o-", ms=3, lw=1.2,
                color="tab:blue", label="corridor profile @1.0 m")
        colors = {"WHEEL": "tab:green", "HOP": "tab:orange", "STOP": "crimson"}
        for xr, mm in zip(xs, latched):
            ax.scatter(xr, -3.0, marker="s", s=18, color=colors[mm])
        ax.axhline(cfg.step_wheel_max_m * 100, color="gray", ls=":",
                   label=f"wheel step max {cfg.step_wheel_max_m*100:.0f} cm")
        ax.set_title(f"{label}: latched={final} (expect {expect})")
        ax.set_xlabel("distance ahead (m)  [squares: latched mode vs step dist]")
        ax.set_ylabel("profile height (cm)")
        ax.set_ylim(-6, max(20, h * 120))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("TerrainGate synthetic test — D435 on gimbal, MOBILE stance")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")
    ok = True
    for label, (final, expect) in results.items():
        good = final == expect
        ok &= good
        print(f"  {label:12s} -> {final:6s} (expect {expect})"
              f"  {'PASS' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
