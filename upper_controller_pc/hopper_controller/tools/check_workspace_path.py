#!/usr/bin/env python3
"""Offline check of the planned trace: joint stops, keep-outs, floor.

Walks the path the tracer will drive with the same resolved-rate IK, so a
failure here is a failure on the robot.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from forward_kinematics import ForwardKinematics  # noqa: E402
from workspace_envelope import (  # noqa: E402
    KEEP_MARGIN_M,
    Q_SAFE_MAX,
    Q_SAFE_MIN,
    WHEEL_AZ_L_DEG,
    WHEEL_CENTER_M,
    WHEEL_KEEP_M,
    XY_FLOOR_R_M,
    Z_BOTTOM_M,
    envelope_path,
    keep_cross,
    keep_inside,
    ring_plan,
)

SEG_SAMPLE_M = 0.002


def numeric_jacobian(fk: ForwardKinematics, q: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        p1, _ = fk.forward_kinematics(q + d)
        p0, _ = fk.forward_kinematics(q - d)
        J[:, i] = (np.asarray(p1, float).reshape(3)
                   - np.asarray(p0, float).reshape(3)) / (2.0 * eps)
    return J


def keep_depth_m(p) -> float:
    """How far inside a barrel, in metres along the worst semi-axis. <=0 is
    outside."""
    rc, a, b = keep_cross(p[2])
    if a <= 0.0:
        return -1.0
    worst = -1.0
    for az_w in (270.0, 30.0, 150.0):
        c, s = math.cos(math.radians(az_w)), math.sin(math.radians(az_w))
        du = p[0] * c + p[1] * s - rc
        dn = -p[0] * s + p[1] * c
        v = math.hypot(du / a, dn / b)
        if v < 1.0:
            worst = max(worst, (1.0 - v) * min(a, b))
    return worst


LEG_R_M = WHEEL_KEEP_M + KEEP_MARGIN_M


def leg_depth_m(p, n: int = 120) -> float:
    """Worst penetration of the straight hip->foot segment, in metres.

    The foot clearing an obstacle is not enough: park it in the sliver
    outside one and the leg has to cross the wall to get there.

    The barrels are deliberately fat -- they are a foot envelope grown from
    a hand pose, not a model of the hardware -- so the leg is checked
    against the wheel itself instead: a vertical column of radius LEG_R_M on
    each wheel centre, covering the wheel and its fork.
    """
    p = np.asarray(p, float).reshape(3)
    worst = -1.0
    for k in range(1, n + 1):
        v = p * (k / n)
        for az_w in WHEEL_AZ_L_DEG:
            c, s = math.cos(math.radians(az_w)), math.sin(math.radians(az_w))
            d = math.hypot(v[0] - WHEEL_CENTER_M * c, v[1] - WHEEL_CENTER_M * s)
            worst = max(worst, LEG_R_M - d)
    return worst


def main() -> int:
    fk = ForwardKinematics()
    home = float(np.linalg.norm(
        np.asarray(fk.forward_kinematics(np.zeros(3))[0], float).reshape(3)))
    way, tags = envelope_path(np.array([0.0, 0.0, home]))
    length = float(np.linalg.norm(np.diff(way, axis=0), axis=1).sum())
    print(f"path {len(way)} waypoints, {length:.2f} m, "
          f"~{length / 0.08:.0f} s at 0.08 m/s")

    q = np.zeros(3)
    qlo, qhi = np.full(3, 9.0), np.full(3, -9.0)
    worst_err, worst_i, stops = 0.0, 0, 0
    for i, tgt in enumerate(way):
        for _ in range(60):
            p = np.asarray(fk.forward_kinematics(q)[0], float).reshape(3)
            e = tgt - p
            if float(np.linalg.norm(e)) < 2e-4:
                break
            dq = np.linalg.lstsq(numeric_jacobian(fk, q), e, rcond=None)[0]
            n = float(np.linalg.norm(dq))
            if n > 0.03:
                dq *= 0.03 / n
            q = np.clip(q + dq, Q_SAFE_MIN, Q_SAFE_MAX)
        p = np.asarray(fk.forward_kinematics(q)[0], float).reshape(3)
        err = float(np.linalg.norm(tgt - p))
        if err > worst_err:
            worst_err, worst_i = err, i
        qlo, qhi = np.minimum(qlo, q), np.maximum(qhi, q)
        if q.min() <= Q_SAFE_MIN + 1e-4 or q.max() >= Q_SAFE_MAX - 1e-4:
            stops += 1

    print(f"q range  {np.round(qlo, 3)} .. {np.round(qhi, 3)}  "
          f"(stops {Q_SAFE_MIN:+.2f}/{Q_SAFE_MAX:+.2f})")
    print(f"margin   lo {float(qlo.min() - Q_SAFE_MIN):+.3f} rad, "
          f"hi {float(Q_SAFE_MAX - qhi.max()):+.3f} rad, "
          f"{stops} waypoints on a stop")
    print(f"IK error worst {worst_err * 1e3:.2f} mm at #{worst_i} "
          f"({tags[worst_i]}) {np.round(way[worst_i], 3)}")

    bad = sum(1 for p in way if keep_inside(*p))
    deepest, where = 0.0, None
    for i in range(len(way) - 1):
        p0, p1 = way[i], way[i + 1]
        n = max(1, int(math.ceil(
            float(np.linalg.norm(p1 - p0)) / SEG_SAMPLE_M)))
        for k in range(n + 1):
            d = keep_depth_m(p0 + (p1 - p0) * k / n)
            if d > deepest:
                deepest, where = d, p0 + (p1 - p0) * k / n
    print(f"keep-out waypoints inside {bad}; deepest cut anywhere along the "
          f"path {deepest * 1e3:.2f} mm of the {KEEP_MARGIN_M * 1e3:.0f} mm "
          "margin"
          + (f" at {np.round(where, 3)}" if where is not None else ""))

    leg_bad, leg_deep, leg_where = 0, 0.0, None
    for p in way:
        d = leg_depth_m(p)
        if d > 1e-6:
            leg_bad += 1
        if d > leg_deep:
            leg_deep, leg_where = d, p
    print(f"leg      {leg_bad} waypoints drag the hip->foot segment into a "
          f"wheel column (r={LEG_R_M:.3f} m); deepest {leg_deep * 1e3:.2f} mm"
          + (f" at {np.round(leg_where, 3)}" if leg_where is not None else ""))

    floor = sum(1 for p in way
                if p[2] > Z_BOTTOM_M - 1e-6
                and math.hypot(p[0], p[1]) < XY_FLOOR_R_M)
    print(f"floor    {floor} waypoints inside r<{XY_FLOOR_R_M:.2f} at "
          f"z={Z_BOTTOM_M:.3f}")
    for ring in ring_plan():
        n = sum(1 for t in tags if t == ring["name"])
        how = ("closed loop through the ducts" if ring["closed"] else
               " + ".join(f"{a1 - a0:.0f} deg @ {(a0 + a1) / 2 % 360:.0f}"
                          for a0, a1 in ring["arcs"]))
        print(f"ring {ring['name']:6s} z={ring['z']:.3f} "
              f"r={ring['r_wheel']:.3f}..{ring['r_hole']:.3f} "
              f"{n:5d} waypoints  {how}")

    ok = (bad == 0 and floor == 0 and stops == 0 and leg_bad == 0
          and deepest < KEEP_MARGIN_M and worst_err < 2e-3)
    print("RESULT:", "ok" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
