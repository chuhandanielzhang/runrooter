"""Pure MOBILE auto-approach control law (no LCM / numpy dependencies).

Shared verbatim between the deployed controller (modee/lcm_controller.py)
and the offline validator (tools/sim_approach_xy.py), so what the
simulation proves is exactly the code the robot runs.

Method (position-based visual servoing in the body frame):
  - Frames: FRD horizontal plane, x forward, y RIGHT; field-calibrated
    wheel-IK sense wz > 0 turns toward positive body-frame yaw
    references (2026-08-08 stick check).
  - Inputs: pre / tag / wall-normal in the SERVO (leg) frame L, plus
    optional drive/camera view of tag + button (pre) for FOV yaw.
  - TWO PHASES (user 2026-08-08):
      1) FAR: yaw keeps the TAG near camera center (don't lose it).
      2) NEAR: park the TAG only a little LEFT of center (tag_left_deg,
         default 6 deg) so the button sits nearer the middle — not a
         hard button-center that shoved the tag to -17 deg (log 23:00).
  - Translation speed scales with radial range: far-fast / near-slow.
  - Phase blend uses radial ||pre||_L. Detection-age tiers as before.
"""

from __future__ import annotations

import math


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def approach_twist(
    pre_x: float,
    pre_y: float,
    n_x: float,
    n_y: float,
    age_s: float,
    cfg,
    tag_x: float | None = None,
    tag_y: float | None = None,
    view_x: float | None = None,
    view_y: float | None = None,
    n_view_x: float | None = None,
    n_view_y: float | None = None,
    pre_view_x: float | None = None,
    pre_view_y: float | None = None,
) -> tuple[float, float, float]:
    """One servo tick -> body twist (vx, vy, wz) in the SERVO frame.

    Translation errors use ``pre_*`` / ``tag_*`` / ``n_*`` (leg frame L).
    Yaw FOV uses drive/camera ``view_*`` (tag) and ``pre_view_*`` (button).
    """
    gx = float(getattr(cfg, "mobile_approach_goal_x_m", 0.086))
    gy = float(getattr(cfg, "mobile_approach_goal_y_m", 0.063))
    kp_v = float(getattr(cfg, "mobile_approach_kp_v", 0.70))
    kp_w = float(getattr(cfg, "mobile_approach_kp_wz", 1.0))
    v_max = float(getattr(cfg, "mobile_approach_v_max_mps", 0.22))
    w_max = float(getattr(cfg, "mobile_approach_wz_max_rad_s", 0.45))
    fresh_s = float(getattr(cfg, "mobile_approach_fresh_s", 0.4))
    stale_s = float(getattr(cfg, "manip_button_stale_s", 1.5))
    # Radial handoff (meters of ||pre||_L): FAR above blend_far, NEAR
    # below blend_near. Defaults sit just outside the ready band.
    view_near = float(getattr(cfg, "mobile_approach_blend_near_m", 0.55))
    view_far = float(getattr(cfg, "mobile_approach_blend_far_m", 0.85))

    tx = float(pre_x) if tag_x is None else float(tag_x)
    ty = float(pre_y) if tag_y is None else float(tag_y)
    # Camera/drive bearings.
    tag_vx = tx if view_x is None else float(view_x)
    tag_vy = ty if view_y is None else float(view_y)
    btn_vx = float(pre_x) if pre_view_x is None else float(pre_view_x)
    btn_vy = float(pre_y) if pre_view_y is None else float(pre_view_y)

    if age_s <= fresh_s:
        scale = 1.0
    elif age_s <= stale_s:
        scale = float(getattr(cfg, "mobile_approach_slow_scale", 0.5))
    else:
        scale = float(getattr(cfg, "mobile_approach_memory_scale", 0.3))

    # Wall-normal sign guard: n_in must point INTO the wall (same half-
    # plane as the pre-goal).
    if float(n_x) * float(pre_x) + float(n_y) * float(pre_y) < 0.0:
        n_x, n_y = -float(n_x), -float(n_y)

    nh = math.hypot(float(n_x), float(n_y))
    nhx, nhy = ((float(n_x) / nh, float(n_y) / nh) if nh > 1e-9
                else (1.0, 0.0))

    # beta=1 FAR (center TAG); beta=0 NEAR (center BUTTON / press goal).
    # Use radial range so the handoff is not stuck on a large XY goal err
    # (log 22:54: |face|=0.51 but XY err=36 cm kept beta≈FAR).
    r_pre = math.hypot(float(pre_x), float(pre_y))
    if view_far > view_near:
        beta = _clip((r_pre - view_near) / (view_far - view_near), 0.0, 1.0)
    else:
        beta = 0.0

    # Far: tag → midline. Near: button hover → (gx, gy) for press pose.
    ex = beta * (tx - gx) + (1.0 - beta) * (float(pre_x) - gx)
    ey = beta * (ty - 0.0) + (1.0 - beta) * (float(pre_y) - gy)

    e_adv = ex * nhx + ey * nhy
    el_x = ex - e_adv * nhx
    el_y = ey - e_adv * nhy
    e_lat = math.hypot(el_x, el_y)
    lat_gate = float(getattr(cfg, "mobile_approach_lat_gate_m", 0.08))
    adv_scale = _clip(1.0 - e_lat / max(1e-6, lat_gate), 0.0, 1.0)

    yaw_tag = (
        math.atan2(tag_vy, tag_vx) if abs(tag_vx) + abs(tag_vy) > 1e-9
        else 0.0
    )
    # Desired tag bearing: FAR = 0 (center); NEAR = slight LEFT.
    tag_left = math.radians(
        float(getattr(cfg, "mobile_approach_tag_left_deg", 6.0))
    )
    yaw_tgt = -tag_left * (1.0 - beta)
    yaw_err = wrap_pi(yaw_tag - yaw_tgt)
    bear_gate = math.radians(
        float(getattr(cfg, "mobile_approach_bear_gate_deg", 12.0))
    )
    bear = abs(yaw_err)
    adv_scale *= _clip(1.0 - bear / max(1e-6, bear_gate), 0.0, 1.0)

    # Far-fast / near-slow on radial range (translation only).
    r_far = float(getattr(cfg, "mobile_approach_v_far_m", 0.90))
    r_near = float(getattr(cfg, "mobile_approach_v_near_m", 0.55))
    v_near_s = float(getattr(cfg, "mobile_approach_v_near_scale", 0.35))
    if r_far > r_near:
        dist_scale = _clip(
            v_near_s + (1.0 - v_near_s) * (r_pre - r_near) / (r_far - r_near),
            v_near_s, 1.0,
        )
    else:
        dist_scale = 1.0

    v_ax = kp_v * e_adv * adv_scale
    vx = _clip(kp_v * el_x + v_ax * nhx, -v_max, v_max) * scale * dist_scale
    vy = _clip(kp_v * el_y + v_ax * nhy, -v_max, v_max) * scale * dist_scale

    # Yaw: FAR keeps the tag in FOV; NEAR squares the camera to the wall
    # (user 2026-08-14: 按按钮要平行于墙面, depth plane supplies n).
    nvx = nhx if n_view_x is None else float(n_view_x)
    nvy = nhy if n_view_y is None else float(n_view_y)
    if float(nvx) * float(btn_vx) + float(nvy) * float(btn_vy) < 0.0:
        nvx, nvy = -nvx, -nvy
    nvh = math.hypot(nvx, nvy)
    if nvh > 1e-9:
        nvx, nvy = nvx / nvh, nvy / nvh
    yaw_n = math.atan2(nvy, nvx)
    yaw_ref = wrap_pi(beta * yaw_err + (1.0 - beta) * yaw_n)
    # Near: do not close distance while the base is still skewed to the
    # wall (same gate as box approach). Far keeps the old tag-bearing gate.
    yaw_ok = math.radians(
        float(getattr(cfg, "mobile_approach_yaw_ok_deg", 8.0))
    )
    yaw_stop = math.radians(
        float(getattr(cfg, "mobile_approach_yaw_stop_deg", 22.0))
    )
    if beta < 0.85:
        if abs(yaw_n) >= yaw_stop:
            adv_scale = 0.0
        else:
            adv_scale *= _clip(
                (yaw_stop - abs(yaw_n)) / max(1e-6, yaw_stop - yaw_ok),
                0.0, 1.0,
            )
    w_gain = kp_w * (1.0 + 0.35 * beta)
    wz = _clip(w_gain * yaw_ref, -w_max, w_max) * scale
    if age_s > stale_s:
        wz = 0.0
    return vx, vy, wz


def dead_reckon_step(
    x: float,
    y: float,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    is_point: bool = True,
) -> tuple[float, float]:
    """Advance one body-frame target by the commanded twist.

    When the body rotates by dtheta = wz*dt (wz>0 = CW from above in
    FRD), a world-fixed point expressed in body coordinates rotates the
    opposite way: p <- Rz(-dtheta) (p - v dt). Directions rotate only.
    """
    dth = float(wz) * float(dt)
    c, s = math.cos(dth), math.sin(dth)
    if is_point:
        x = float(x) - float(vx) * float(dt)
        y = float(y) - float(vy) * float(dt)
    return c * x + s * y, -s * x + c * y
