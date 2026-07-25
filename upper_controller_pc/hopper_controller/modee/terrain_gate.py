"""Continuous terrain gate: WHEEL / HOP / STOP from a depth point cloud.

Method (Wermelinger et al., IROS 2016 metrics on a 1-D corridor profile):
the cloud is expressed in a gravity-aligned GROUND frame (origin on the
floor under the leg-base center, X = heading, Z = up), cropped to a
forward corridor, binned by distance, and reduced to per-bin height
stats.  Three metrics decide the mode:

  step      max |median height jump| between adjacent bins
  slope     linear fit of the bin medians over distance
  roughness max in-bin height std

Rules (with a config'd wheel/hop envelope):
  step <= step_wheel_max  AND slope <= slope_wheel_max_deg AND
  roughness <= rough_wheel_max                    -> WHEEL
  step <= step_hop_max (and a landing bin exists) -> HOP
  otherwise / not enough valid depth              -> STOP

A frame vote only becomes the OUTPUT mode after `hysteresis_n`
consecutive identical votes (mode-switch debounce, M4-style switch
penalty at the frame level).

This module is deliberately I/O-free: feed it points from pyrealsense2,
a rosbag, or the synthetic test in tools/test_terrain_gate.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TerrainGateConfig:
    # Corridor crop, ground frame (m).
    corridor_half_width_m: float = 0.20
    range_min_m: float = 0.35          # D435 min-Z + margin
    range_max_m: float = 1.50
    bin_size_m: float = 0.05
    min_pts_per_bin: int = 12
    # Ground reference tolerance: points beyond +-this from the fitted
    # near-field floor are still terrain (steps), NOT outliers; only
    # points far ABOVE head height are dropped as non-terrain.
    max_height_m: float = 0.60
    # WHEEL envelope.
    step_wheel_max_m: float = 0.030    # ~omni wheel radius margin
    slope_wheel_max_deg: float = 10.0
    rough_wheel_max_m: float = 0.020
    # HOP envelope (positive obstacle or negative gap the hop can clear).
    step_hop_max_m: float = 0.25
    # A gap of >= this many consecutive empty bins is unknown -> STOP.
    gap_bins_stop: int = 4
    # Frames of agreement required to switch the latched mode.
    hysteresis_n: int = 5


@dataclass
class TerrainVote:
    mode: str                  # "WHEEL" | "HOP" | "STOP"
    step_m: float
    slope_deg: float
    rough_m: float
    step_dist_m: float         # distance of the worst step (nan if none)
    n_valid_bins: int
    n_steps: int = 0           # significant steps in the corridor (stairs
                               # pattern when >= 2)
    profile_x: np.ndarray = field(repr=False, default=None)
    profile_z: np.ndarray = field(repr=False, default=None)


class TerrainGate:
    def __init__(self, cfg: TerrainGateConfig | None = None):
        self.cfg = cfg or TerrainGateConfig()
        self._latched = "STOP"
        self._pending: str | None = None
        self._pending_n = 0

    # ---- per-frame analysis (stateless) ----
    def analyze(self, pts_ground: np.ndarray) -> TerrainVote:
        """pts_ground: (N,3) points in the ground frame (X fwd, Z up)."""
        c = self.cfg
        p = np.asarray(pts_ground, dtype=float).reshape(-1, 3)
        m = (
            np.isfinite(p).all(axis=1)
            & (np.abs(p[:, 1]) <= c.corridor_half_width_m)
            & (p[:, 0] >= c.range_min_m)
            & (p[:, 0] <= c.range_max_m)
            & (p[:, 2] <= c.max_height_m)
        )
        p = p[m]
        n_bins = int(np.ceil((c.range_max_m - c.range_min_m) / c.bin_size_m))
        med = np.full(n_bins, np.nan)
        std = np.full(n_bins, np.nan)
        xc = c.range_min_m + (np.arange(n_bins) + 0.5) * c.bin_size_m
        if len(p):
            bi = np.clip(
                ((p[:, 0] - c.range_min_m) / c.bin_size_m).astype(int),
                0, n_bins - 1,
            )
            for b in range(n_bins):
                z = p[bi == b, 2]
                if len(z) >= c.min_pts_per_bin:
                    med[b] = float(np.median(z))
                    std[b] = float(np.std(z))
        valid = np.isfinite(med)
        nv = int(valid.sum())

        # Unknown ground (long run of empty bins) -> STOP.
        run = mx = 0
        for v in valid:
            run = 0 if v else run + 1
            mx = max(mx, run)
        if nv < 3 or mx >= c.gap_bins_stop:
            return TerrainVote("STOP", float("nan"), float("nan"),
                               float("nan"), float("nan"), nv, 0, xc, med)

        iv = np.flatnonzero(valid)
        dz = np.diff(med[iv])
        step_per = np.abs(dz)
        k = int(np.argmax(step_per)) if len(step_per) else 0
        step = float(step_per[k]) if len(step_per) else 0.0
        step_x = float(0.5 * (xc[iv[k]] + xc[iv[k + 1]])) if len(step_per) \
            else float("nan")
        # stairs pattern: count all wheel-blocking jumps in the corridor
        n_steps = int(np.sum(step_per > c.step_wheel_max_m))
        slope = float(np.rad2deg(np.arctan(
            np.polyfit(xc[iv], med[iv], 1)[0]
        )))
        rough = float(np.nanmax(std[iv]))

        if (step <= c.step_wheel_max_m
                and abs(slope) <= c.slope_wheel_max_deg
                and rough <= c.rough_wheel_max_m):
            mode = "WHEEL"
        elif step <= c.step_hop_max_m:
            # single ledge OR stairs: both are hop territory as long as
            # each riser fits the hop envelope (the FSM hops them one by
            # one, re-gating after every landing)
            mode = "HOP"
        else:
            mode = "STOP"
        return TerrainVote(mode, step, slope, rough, step_x, nv, n_steps,
                           xc, med)

    # ---- debounced update (stateful) ----
    def update(self, pts_ground: np.ndarray) -> tuple[str, TerrainVote]:
        vote = self.analyze(pts_ground)
        if vote.mode == self._latched:
            self._pending, self._pending_n = None, 0
        elif vote.mode == self._pending:
            self._pending_n += 1
            if self._pending_n >= self.cfg.hysteresis_n:
                self._latched = vote.mode
                self._pending, self._pending_n = None, 0
        else:
            self._pending, self._pending_n = vote.mode, 1
        return self._latched, vote


# ---------------------------------------------------------------------------
# Semantic surface layer (RGB) + M4-style cost arbitration.
#
# Geometry answers "CAN the wheels roll here"; the surface class answers
# "SHOULD they" (grass/gravel look flat in depth but omni wheels slip).
# The classifier itself is pluggable -- anything that maps the corridor
# image patch to one of SURFACE_CLASSES works (pretrained segmentation on
# RUGD/GOOSE classes, a small patch CNN, or a stub returning PAVED
# indoors).  Costs are per-meter locomotion costs a la M4 (Nature Comm
# 2023, driving-vs-flying energy cost), plus a switch penalty applied by
# the FSM, so WHEEL is preferred on pavement, HOP takes over on slip-risk
# surfaces even when the geometry is wheelable.
# ---------------------------------------------------------------------------

SURFACE_CLASSES = ("PAVED", "GRASS", "GRAVEL", "UNKNOWN")

#: relative per-meter cost of wheeling / hopping on each surface
#: (calibrate later from power logs; ordering is what matters first)
SURFACE_COSTS = {
    #            wheel  hop
    "PAVED":   (1.0,   4.0),
    "GRASS":   (6.0,   4.5),   # small omni rollers jam in grass: hop wins
    "GRAVEL":  (2.5,   5.0),   # loose but rollable; bad landings, so wheel
    "UNKNOWN": (2.0,   4.5),
}


def arbitrate(vote: TerrainVote, surface: str = "UNKNOWN") -> str:
    """Fuse the geometric vote with the surface class (M4-style cost).

    Geometry keeps veto power (a wall is a wall); the surface class can
    only demote WHEEL to HOP when wheeling is geometrically possible but
    costlier than hopping (e.g. tall grass).
    """
    if vote.mode != "WHEEL":
        return vote.mode
    cw, ch = SURFACE_COSTS.get(str(surface).upper(),
                               SURFACE_COSTS["UNKNOWN"])
    return "WHEEL" if cw <= ch else "HOP"


def fit_dominant_plane(
    pts: np.ndarray, iters: int = 60, tol_m: float = 0.012,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """RANSAC dominant-plane fit.  Returns (unit normal n, d) with
    n.p + d = 0, oriented so the sensor origin is on the positive side
    (d > 0: normal points up toward the camera)."""
    p = np.asarray(pts, dtype=float).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 50:
        raise ValueError("not enough points for plane fit")
    rng = np.random.default_rng(seed)
    best_n, best_d, best_cnt = None, 0.0, -1
    for _ in range(iters):
        i = rng.choice(len(p), 3, replace=False)
        a, b, c = p[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(n @ a)
        cnt = int(np.sum(np.abs(p @ n + d) < tol_m))
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    # refine on inliers (least squares)
    m = np.abs(p @ best_n + best_d) < tol_m
    q = p[m]
    ctr = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - ctr, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    d = -float(n @ ctr)
    if d < 0.0:
        n, d = -n, -d
    return n, float(d)


def plane_frame(n: np.ndarray) -> np.ndarray:
    """Rotation (rows = plane-frame axes in camera coords) mapping camera
    points into a frame with Z = plane normal and X = camera optical
    forward projected onto the plane:  p_plane = R @ p_cam."""
    z = np.asarray(n, dtype=float)
    fwd = np.array([0.0, 0.0, 1.0])           # camera optical forward
    x = fwd - (fwd @ z) * z
    xn = np.linalg.norm(x)
    if xn < 1e-6:                              # camera staring straight down
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        xn = np.linalg.norm(x)
    x = x / xn
    y = np.cross(z, x)
    return np.stack([x, y, z])


def level_points_to_plane(
    pts_cam: np.ndarray, n: np.ndarray, d: float,
) -> np.ndarray:
    """Rotate/translate camera-frame points into a 'plane frame': Z up
    (plane normal), plane at z=0, X = camera forward projected onto the
    plane, origin under the camera.  Lets the TerrainGate corridor run
    with a handheld / arbitrarily-mounted camera (a global slope becomes
    unobservable -- fine for FLAT/NOT-FLAT)."""
    R = plane_frame(n)
    p = np.asarray(pts_cam, dtype=float).reshape(-1, 3) @ R.T
    p[:, 2] += float(d)                        # plane -> z = 0
    return p


def depth_to_ground_points(
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    R_ground_cam: np.ndarray,
    t_ground_cam: np.ndarray,
    stride: int = 4,
) -> np.ndarray:
    """Deproject a depth image (meters) and transform to the ground frame.

    Camera optical convention: +Z forward, +X image-right, +Y image-down.
    R/t: camera pose in the ground frame (gravity-aligned, X fwd, Z up).
    """
    d = np.asarray(depth_m, dtype=float)[::stride, ::stride]
    h, w = d.shape
    us = (np.arange(w) * stride).astype(float)
    vs = (np.arange(h) * stride).astype(float)
    uu, vv = np.meshgrid(us, vs)
    ok = np.isfinite(d) & (d > 0.05)
    z = d[ok]
    x = (uu[ok] - cx) / fx * z
    y = (vv[ok] - cy) / fy * z
    p_cam = np.stack([x, y, z], axis=1)
    return p_cam @ np.asarray(R_ground_cam, dtype=float).T \
        + np.asarray(t_ground_cam, dtype=float).reshape(1, 3)
