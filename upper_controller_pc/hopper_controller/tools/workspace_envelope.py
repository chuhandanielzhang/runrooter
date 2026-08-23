"""Deployed-leg work envelope: the 3D wheel keep-outs and the trace path.

Shared by trace_workspace.py (drives the leg) and plot_workspace_trace.py
(draws it).  ModeELCMConfig.work_* mirrors the same numbers for the runtime
clamp.

Geometry, all in the LEG/L frame (+Z down, origin at the hip):

  Each wheel owns a barrel-shaped keep-out.  Its cross-section is an ellipse
  whose centre sits on the ray from the origin through the wheel centre:

    z = KEEP_Z_FLOOR (0.440, the wheel-contact plane)
        the wheel disk itself, r=0.170 +/- 0.055.
    z = KEEP_Z_WIDE (0.267)
        the widest layer.  Sized from a hand-posed foot that just clears the
        wheel: the ellipse centre is the perpendicular foot of that pose onto
        the ray, the tangential semi-axis is its distance to the ray, and the
        radial semi-axis reaches the inner edge of the wheel disk.
    in between and above
        an ellipse in (r, z) too, so the barrel opens from the disk up to the
        widest layer and closes again symmetrically.

  The radial semi-axis grows exactly as fast as the centre moves out, so the
  inner face of all three barrels is one straight cylinder at r=0.107 (with
  KEEP_MARGIN).  That duct is what lets the low rings weave between the
  wheels; higher up the leg cannot fold short enough to use it.
"""

from __future__ import annotations

import math

import numpy as np

# ---- heights -------------------------------------------------------------
Z_TOP_M = 0.220
Z_MID_M = 0.320
Z_BOTTOM_M = 0.440

LEN_MIN_M = 0.354        # FK(0,0,0) homing / 零位
LEN_MAX_M = 0.50
XY_MAX_M = 0.383         # FK(1.55,-0.97,1.55) r_xy, at z=0.162 only
XY_FLOOR_R_M = 0.10      # r < this at z_bottom hits the floor

# ---- wheel barrels -------------------------------------------------------
WHEEL_CENTER_M = 0.17
WHEEL_KEEP_M = 0.055
WHEEL_AZ_L_DEG = (270.0, 30.0, 150.0)   # bottom on x=0, then 120 deg apart

KEEP_Z_FLOOR = Z_BOTTOM_M
KEEP_Z_WIDE = 0.267          # height of the hand-posed foot that sized it
KEEP_CZ = KEEP_Z_FLOOR - KEEP_Z_WIDE
KEEP_Z_CEIL = KEEP_Z_WIDE - KEEP_CZ     # 0.094, the mirror of the floor
KEEP_R_WIDE = 0.229          # ellipse centre on the ray
KEEP_A_WIDE = 0.114          # radial semi-axis  (= KEEP_R_WIDE - 0.115)
# Tangential semi-axis, i.e. how wide the barrel is in AZIMUTH.  This is the
# constant that decides how far round a ring gets, because keep_near_r falls
# off a cliff: one degree outside the barrel's angular span the ray is clear
# (inf), one degree inside it drops to the r=0.115 duct wall, straight past
# what the leg can fold to.  So the arc length is set purely by this, and
# not at all by the ring radius.
# Was 0.150, from a hand pose beside the wheel at r=0.274, z=0.267.  That
# read 9 deg wider than the hardware and cost TOP two thirds of its travel:
# 2026-08-24 a hand pose at q=[-1.047, 1.464, 0.776], foot (0.162, 0.325,
# 0.151), az=63.5 deg, cleared the real wheel column by 31 mm yet the model
# called it blocked.  0.118 admits that pose with 3 deg to spare and is
# still 9 deg more conservative than the bare wheel column (21.8 deg), which
# is the allowance for the fork and the foot's own width.
KEEP_B_WIDE = 0.118          # tangential semi-axis
KEEP_MARGIN_M = 0.008        # inflate both semi-axes by this

# ---- joint stops the rings were solved against ---------------------------
# safe_q_min/safe_q_max in ModeELCMConfig; the measured retract stop is
# about -1.085 rad (-62 deg).  The extend stop used to be 1.40, a policy
# limit rather than a mechanical one: hand-posing the leg on 2026-08-23
# reached q=[-0.711, 1.457, 1.528] -> foot r=0.374 at z=0.221, which the
# 1.40 box could not touch.  1.55 leaves the 0.02 rad of solver margin the
# reach table keeps, so the envelope now stops exactly at that hand pose
# and no further.  Moving this needs safe_q_max, manip_q_max, XY_MAX_M and
# work_xy_max_m moved with it, and REACH_TABLE re-measured.
Q_SAFE_MIN = -1.06
Q_SAFE_MAX = 1.55

# Shortest L the *trace* is allowed to ask for.  Joints retract to 0.247 m;
# teleop still uses work_len_min_m = 归中 = 0.354.  MID's three holes sit at
# L = hypot(0.107, 0.320) = 0.337, so the trace has to fold 17 mm past 归中
# or it can only skate the outer lip of each hole.
LEN_FOLD_MIN_M = 0.330

# TOP cannot fold into the r=0.107 duct (IK bottoms out at r=0.288 on the
# hole rays, ~0.02 rad from the retract stop).  Each TOP hole is walked as
# an outer arc plus this inner pass.
TOP_HOLE_R_M = 0.288

# ---- outer reach boundary ------------------------------------------------
# Measured by IK bisection over the q box [Q_SAFE_MIN, Q_SAFE_MAX], keeping
# 0.02 rad of joint margin.  The leg is 3-fold symmetric, so this is indexed
# by the azimuth offset to the NEAREST hole centre (SAFE_AZ_DEG), 0..60 deg.
# It reaches furthest straight down a hole and least toward a wheel:
#   z=0.220   0.377 at a hole -> 0.362 at a wheel
#   z=0.320   0.347           -> 0.333
#   z=0.440   0.264           -> 0.250   (LEN_MAX_M caps this to 0.238)
# A constant ring radius throws that bulge away -- 43 mm of it at TOP -- so
# the rings ride this boundary instead.  Re-measure if Q_SAFE_MAX moves.
REACH_OFFSET_DEG: tuple = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
REACH_TABLE: dict = {
    0.220: (0.3769, 0.3746, 0.3726, 0.3708, 0.3691, 0.3675, 0.3663,
            0.3650, 0.3641, 0.3634, 0.3628, 0.3625, 0.3624),
    0.320: (0.3471, 0.3450, 0.3430, 0.3411, 0.3394, 0.3379, 0.3365,
            0.3354, 0.3345, 0.3338, 0.3331, 0.3329, 0.3328),
    0.440: (0.2641, 0.2620, 0.2600, 0.2581, 0.2565, 0.2550, 0.2536,
            0.2526, 0.2516, 0.2510, 0.2505, 0.2501, 0.2500),
}
REACH_MARGIN_M = 0.006       # Cartesian standoff from the measured boundary

# (name, z).  The outer radius comes from REACH_TABLE, so each ring is the
# real work envelope at that height rather than a constant circle.  Whether
# a ring closes into a full loop (weaves the three holes through the duct)
# is derived: see ring_arcs().  BOTTOM and MID close.  TOP cannot reach the
# duct, so each of its holes is an outer arc plus an inner pass at
# TOP_HOLE_R_M.
RINGS: tuple = (
    ("BOTTOM", Z_BOTTOM_M),
    ("MID", Z_MID_M),
    ("TOP", Z_TOP_M),
)

# Azimuths clear of every barrel at every radius and every height, so the
# foot may change z (and r) there.  Midway between the wheels; the widest
# layer subtends +/-43.6 deg, leaving +/-16.4 deg of slack.
SAFE_AZ_DEG = (90.0, 210.0, 330.0)

RING_STEP_DEG = 0.5          # 1.0 deg lets the carved arcs cut ~1.3 mm of corner
LINE_STEP_M = 0.008


def keep_scale(z: float) -> float:
    """Barrel profile in z: 1 at the widest layer, 0 at the floor and its
    mirror above."""
    t = (float(z) - KEEP_Z_WIDE) / KEEP_CZ
    return 0.0 if abs(t) >= 1.0 else math.sqrt(1.0 - t * t)


def keep_cross(z: float) -> tuple[float, float, float]:
    """(centre radius, radial semi-axis, tangential semi-axis) at height z,
    margin included.  Zero semi-axes outside the barrel's z span."""
    z = float(z)
    if not KEEP_Z_CEIL <= z <= KEEP_Z_FLOOR:
        return WHEEL_CENTER_M, 0.0, 0.0
    s = keep_scale(z)
    rc = WHEEL_CENTER_M + s * (KEEP_R_WIDE - WHEEL_CENTER_M)
    a = WHEEL_KEEP_M + s * (KEEP_A_WIDE - WHEEL_KEEP_M) + KEEP_MARGIN_M
    b = WHEEL_KEEP_M + s * (KEEP_B_WIDE - WHEEL_KEEP_M) + KEEP_MARGIN_M
    return rc, a, b


def keep_inside(x: float, y: float, z: float, tol: float = 1e-6) -> bool:
    """Strictly inside a barrel; points carved onto the surface do not count."""
    rc, a, b = keep_cross(z)
    if a <= 0.0:
        return False
    for az_w in WHEEL_AZ_L_DEG:
        c, s = math.cos(math.radians(az_w)), math.sin(math.radians(az_w))
        du = x * c + y * s - rc
        dn = -x * s + y * c
        if (du / a) ** 2 + (dn / b) ** 2 <= 1.0 - tol:
            return True
    return False


def keep_near_r(az: float, z: float) -> float:
    """Smallest r > 0 where the ray at azimuth az (rad) enters a barrel.

    inf when the ray misses all three.  Following this radius traces the
    inner face of the barrel, i.e. the carve.
    """
    rc, a, b = keep_cross(z)
    if a <= 0.0:
        return math.inf
    best = math.inf
    for az_w in WHEEL_AZ_L_DEG:
        d = az - math.radians(az_w)
        c, s = math.cos(d), math.sin(d)
        qa = (c / a) ** 2 + (s / b) ** 2
        qb = -2.0 * rc * c / (a * a)
        qc = (rc / a) ** 2 - 1.0
        disc = qb * qb - 4.0 * qa * qc
        if disc <= 0.0:
            continue
        r1 = (-qb - math.sqrt(disc)) / (2.0 * qa)
        if 0.0 < r1 < best:
            best = r1
    return best


def wheel_centers_xy() -> list[tuple[float, float]]:
    return [
        (WHEEL_CENTER_M * math.cos(math.radians(a)),
         WHEEL_CENTER_M * math.sin(math.radians(a)))
        for a in WHEEL_AZ_L_DEG
    ]


def keep_outline(z: float, az_w_deg: float, n: int = 181) -> np.ndarray:
    """XY outline of one barrel's cross-section at height z, for plotting."""
    rc, a, b = keep_cross(z)
    c, s = math.cos(math.radians(az_w_deg)), math.sin(math.radians(az_w_deg))
    t = np.linspace(0.0, 2.0 * math.pi, n)
    du, dn = a * np.cos(t), b * np.sin(t)
    return np.column_stack([
        (rc + du) * c - dn * s,
        (rc + du) * s + dn * c,
    ])


def hole_offset_deg(az: float) -> float:
    """Azimuth offset (deg, 0..60) from az (rad) to the nearest hole centre."""
    d = math.degrees(az)
    return min(abs((d - A + 180.0) % 360.0 - 180.0) for A in SAFE_AZ_DEG)


def _interp(xs, ys, x: float) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return float(ys[i] * (1.0 - t) + ys[i + 1] * t)
    return float(ys[-1])


def reach_limit_r(z: float, az: float) -> float:
    """Raw measured joint-stop boundary: no margin, no xy/length cap."""
    off = hole_offset_deg(az)
    zs = sorted(REACH_TABLE)
    per_z = [_interp(REACH_OFFSET_DEG, REACH_TABLE[zz], off) for zz in zs]
    return _interp(zs, per_z, float(z))


def reach_r(z: float, az: float) -> float:
    """Outer radius the ring rides at this height and azimuth.

    The measured joint-stop boundary less REACH_MARGIN_M, then capped by the
    xy and leg-length limits.  The length cap bites at BOTTOM: the bulge
    there measures 0.247, which is L=0.505, past LEN_MAX_M.
    """
    off = hole_offset_deg(az)
    zs = sorted(REACH_TABLE)
    per_z = [_interp(REACH_OFFSET_DEG, REACH_TABLE[zz], off) for zz in zs]
    r = _interp(zs, per_z, float(z)) - REACH_MARGIN_M
    r_len = math.sqrt(max(0.0, LEN_MAX_M ** 2 - float(z) ** 2))
    return min(r, XY_MAX_M, r_len)


def fold_r_min(z: float) -> float:
    """Smallest xy radius the envelope allows at height z (the LEN_FOLD_MIN_M
    sphere).  0 where the depth alone is already past it."""
    return math.sqrt(max(0.0, LEN_FOLD_MIN_M ** 2 - float(z) ** 2))


def hole_inner_radius(z: float, r_outer: float) -> float:
    """How far into a hole (SAFE_AZ gap) this height can walk.

    Closed rings weave through the duct and do not use this.  Open rings
    (TOP) get an extra inner pass at this radius so each of the three
    holes is actually entered, not just skimmed on the outer lip.
    """
    r_kin = TOP_HOLE_R_M if float(z) <= 0.230 else 0.0
    r = max(fold_r_min(z), r_kin)
    if r <= 0.0 or r >= float(r_outer) - 0.012:
        return float(r_outer)
    return r


def ring_radius(az: float, z: float, r_cap: float = math.inf) -> float:
    """Ring radius at this azimuth; nan where the ring cannot pass.

    The unobstructed radius is the measured reach boundary, so the ring
    bulges out at each hole and pulls back toward each wheel.

    Then the connected-from-the-origin rule (the same one _work_r_cap uses):
    a ray that meets a barrel is cut at the near intersection, so the ring
    peels off where the ray goes tangent and rides the barrel's inner face.

    That rule is not just about the foot.  The LEG is the segment from the
    hip to the foot, so parking the foot in the sliver OUTSIDE a barrel
    means the leg has to cross the barrel wall to get there -- a wheel
    strike no matter how thin the barrel is drawn.  Being cut at the near
    intersection is what keeps the whole leg on the origin's side.  Getting
    past a barrel therefore always costs a fold to LEN_FOLD_MIN_M, and a
    ring that cannot fold that far is cut into arcs instead.
    """
    r = min(reach_r(z, az), float(r_cap), keep_near_r(az, z))
    return r if r >= fold_r_min(z) - 1e-12 else math.nan


def ring_arcs(z: float, r_cap: float = math.inf,
              step_deg: float = RING_STEP_DEG
              ) -> tuple[list[tuple[float, float]], bool]:
    """(azimuth spans this ring can be traced over, whether it closes).

    A ring that can fold into every duct closes into one clover loop.
    Otherwise the barrels cut it into arcs centred on SAFE_AZ_DEG, and the
    ring below ferries the foot from one arc to the next.
    """
    n = int(round(360.0 / step_deg))
    if all(not math.isnan(ring_radius(math.radians(k * step_deg), z, r_cap))
           for k in range(n)):
        return [(SAFE_AZ_DEG[0], SAFE_AZ_DEG[0] + 360.0)], True
    arcs: list[tuple[float, float]] = []
    for A in SAFE_AZ_DEG:
        span = [0.0, 0.0]
        for sign in (1, -1):
            d = 0.0
            while d < 120.0:
                nxt = d + step_deg
                if math.isnan(ring_radius(math.radians(A + sign * nxt),
                                          z, r_cap)):
                    break
                d = nxt
            span[0 if sign < 0 else 1] = sign * d
        if span[1] - span[0] > step_deg:
            arcs.append((A + span[0], A + span[1]))
    return arcs, False


def arc_waypoints(z: float, az0_deg: float, az1_deg: float,
                  r_cap: float = math.inf,
                  step_deg: float = RING_STEP_DEG) -> np.ndarray:
    """Arc at height z from az0 to az1, riding the reach boundary."""
    n = max(1, int(round(abs(az1_deg - az0_deg) / step_deg)))
    pts = np.empty((n + 1, 3), dtype=float)
    for k in range(n + 1):
        az = math.radians(az0_deg + (az1_deg - az0_deg) * k / n)
        r = ring_radius(az, z, r_cap)
        if math.isnan(r):
            raise ValueError(
                f"ring z={z:.3f} is blocked at "
                f"az={math.degrees(az) % 360.0:.1f} deg")
        pts[k] = (r * math.cos(az), r * math.sin(az), z)
    return pts


def _line(p0, p1, step: float = LINE_STEP_M) -> np.ndarray:
    p0 = np.asarray(p0, float).reshape(3)
    p1 = np.asarray(p1, float).reshape(3)
    n = max(1, int(round(float(np.linalg.norm(p1 - p0)) / step)))
    return np.array([p0 + (p1 - p0) * k / n for k in range(1, n + 1)])


def _polar(r: float, az_deg: float, z: float) -> np.ndarray:
    a = math.radians(az_deg)
    return np.array([r * math.cos(a), r * math.sin(a), z])


def ring_plan(rings: tuple = RINGS) -> list[dict]:
    """Per ring: its arcs, whether it closes, and its radius range."""
    out = []
    for name, z in rings:
        arcs, closed = ring_arcs(z)
        rs = [reach_r(z, math.radians(a)) for a in range(0, 360)]
        out.append({"name": name, "z": z, "arcs": arcs, "closed": closed,
                    "r_hole": max(rs), "r_wheel": min(rs)})
    return out


def envelope_path(p_now, *, rings: tuple = RINGS,
                  loops: int = 1) -> tuple[np.ndarray, list[str]]:
    """Full trace, bottom-up, as (Nx3 waypoints, per-waypoint ring name).

    Every change of height happens on a SAFE_AZ_DEG ray, the only azimuths
    that clear the barrels at all radii.  A ring that closes is walked as one
    loop and becomes the ferry for the rings above it.  A ring the barrels
    cut into arcs is visited arc by arc, climbing from the ferry and coming
    back down to it, so the foot never has to cross a barrel.
    """
    p0 = np.asarray(p_now, float).reshape(3)
    pts: list[np.ndarray] = [p0]
    tag: list[str] = ["START"]

    def add(seq, name: str) -> None:
        for p in np.asarray(seq, float).reshape(-1, 3):
            pts.append(p)
            tag.append(name)

    plan = ring_plan(rings)
    a0 = SAFE_AZ_DEG[0]

    def at(ring: dict, az_deg: float, r_cap: float = math.inf) -> np.ndarray:
        """The point this ring occupies at that azimuth."""
        az = math.radians(az_deg)
        return _polar(ring_radius(az, ring["z"], r_cap), az_deg, ring["z"])

    # Approach: walk out of the floor disk first, then drop onto the lowest
    # ring, both on the safe ray.
    low = plan[0]
    p_low0 = at(low, a0)
    r_low0 = float(math.hypot(p_low0[0], p_low0[1]))
    add(_line(p0, _polar(max(XY_FLOOR_R_M, r_low0), a0, float(p0[2]))),
        "APPROACH")
    add(_line(pts[-1], p_low0), "APPROACH")

    ferry: dict | None = None
    stacked: list[dict] = []

    def walk_span(ring: dict, A: float, span: tuple,
                  r_cap: float = math.inf) -> None:
        """Walk one hole once: centre -> one lip -> the other lip -> centre."""
        z, name = ring["z"], ring["name"]
        add(arc_waypoints(z, A, span[1], r_cap), name)
        add(arc_waypoints(z, span[1], span[0], r_cap), name)
        add(arc_waypoints(z, span[0], A, r_cap), name)

    def hole_pass(ring: dict, A: float) -> None:
        """Dip radially into the hole on ray A and come back out.

        An arc already pulls in toward the barrels at its ends; this walks
        the opening at a fixed inner radius so the foot goes through it
        rather than only crossing its mouth.
        """
        z = ring["z"]
        p_out = at(ring, A)
        r_out = float(math.hypot(p_out[0], p_out[1]))
        r_in = hole_inner_radius(z, r_out)
        if r_in >= r_out - 0.012:
            return
        add(_line(pts[-1], _polar(r_in, A, z)), ring["name"])
        inner, _ = ring_arcs(z, r_in)
        iarc = next((s for s in inner
                     if s[0] - 1e-9 <= A <= s[1] + 1e-9), None)
        if iarc is not None:
            walk_span(ring, A, iarc, r_in)
        add(_line(pts[-1], p_out), ring["name"])

    def visit_stacked() -> None:
        """Climb the open rings hole by hole, from each safe azimuth.

        An open ring cannot fold into the ducts, so it can only ever be
        three separate arcs.  The foot retracts back down to the ferry ring
        between them and crosses the wheel inside the duct there -- riding
        round the outside instead would drag the leg through the wheel.
        """
        if not stacked or ferry is None:
            return
        for i, A in enumerate(SAFE_AZ_DEG):
            if i:
                add(arc_waypoints(ferry["z"], SAFE_AZ_DEG[i - 1], A),
                    ferry["name"])
            for ring in stacked:
                arc = next((s for s in ring["arcs"]
                            if s[0] - 1e-9 <= A <= s[1] + 1e-9), None)
                if arc is None:
                    continue
                add(_line(pts[-1], at(ring, A)), "LIFT")
                walk_span(ring, A, arc)
                hole_pass(ring, A)
            add(_line(pts[-1], at(ferry, A)), "LIFT")
        add(arc_waypoints(ferry["z"], SAFE_AZ_DEG[-1], SAFE_AZ_DEG[0] + 360.0),
            ferry["name"])
        stacked.clear()

    for ring in plan:
        if not ring["closed"]:
            stacked.append(ring)
            continue
        visit_stacked()
        add(_line(pts[-1], at(ring, a0)), "LIFT")
        for _ in range(max(1, loops)):
            add(arc_waypoints(ring["z"], a0, a0 + 360.0), ring["name"])
        ferry = ring
    visit_stacked()

    return np.asarray(pts, dtype=float), tag
