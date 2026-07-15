#!/usr/bin/env python3
"""Plot stance diagnostics from a ModeE csv log on ONE continuous timeline
(x = log time t_s), all hops together, stance windows shaded + numbered.

Rows:
  1. hiptorque tau_b_stance_des x/y (roll/pitch body torque) with +/- cap lines
  2. |hiptorque_xy| magnitude vs the cap (shows attitude-torque saturation)
  3. roll / pitch
  4. base height (+ apex markers)
  5. body velocity v_hat_w (vx, vy, vz) and raw leg meas v_meas_foot_w (dashed)
  6. swing-leg foot target vs actual in BODY frame (NO quaternion):
     foot_des_b (dashed) vs foot_b (solid), x/y/z -- flight only
Contact force is logged in world frame; it is rotated back to body with
rpy_hat so tilt does not leak fz into fx/fy (a -38 deg roll makes world fx
look like -100 N while body |fxy| is really capped at 20 N).

Usage (from hopper_controller/):
    python3 scripts/plot_hops.py                # latest log, ALL hops
    python3 scripts/plot_hops.py --hops 1 2 3   # only these hops (1-based)
    python3 scripts/plot_hops.py --log logs/modee_2116_snapshot.csv

Output: logs/hops_combined.png
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(HERE, "..", "logs", "modee_latest.csv")

# Pull the actual limits from the controller config so the plotted limit is
# always the one the controller enforced (|fxy| <= min(mu*fz, cap)).
try:
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from modee.core import ModeEConfig
    _cfg = ModeEConfig()
    FXY_CAP_N = float(_cfg.stance_fxy_max)
    STANCE_MU = float(_cfg.stance_mu)
    TAU_CAP_NM = float(_cfg.stance_tau_rp_max)
except Exception:
    FXY_CAP_N = 15.0
    STANCE_MU = 0.0
    TAU_CAP_NM = 2.0


def rot_wb(r, p, y):
    """World-from-body rotation for ZYX (yaw-pitch-roll) Euler angles."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--hops", type=int, nargs="+", default=None,
                    help="1-based hop indices to include (default: ALL hops)")
    ap.add_argument("--margin-s", type=float, default=0.3,
                    help="extra time plotted before first TD / after last LO")
    args = ap.parse_args()

    log_path = os.path.abspath(args.log)
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        raise SystemExit(f"empty log: {log_path}")

    col = lambda name: np.array([float(r[name]) for r in rows])
    t = col("t_s")
    st = np.array([int(float(r["stance"])) for r in rows])
    roll = col("rpy_hat_roll")
    pitch = col("rpy_hat_pitch")
    # Base height above ground: estimator position, world +Z DOWN -> negate.
    height = -col("p_hat_w2")

    def _vel3(prefix):
        if f"{prefix}0" in rows[0]:
            return np.stack([col(f"{prefix}0"), col(f"{prefix}1"), col(f"{prefix}2")], axis=1)
        return None

    v_hat = _vel3("v_hat_w")
    v_meas = _vel3("v_meas_foot_w")

    # Swing-leg tracking in BODY frame (no quaternion mapping):
    #   foot_des_b = commanded foot target, foot_b = measured foot position.
    foot_des_b = _vel3("foot_des_b")
    foot_b = _vel3("foot_b")

    # Stance hiptorque (body-frame attitude torque, already capped at TAU_CAP_NM).
    tau_hip = None
    if "tau_b_stance_des0" in rows[0]:
        tau_hip = np.stack(
            [col("tau_b_stance_des0"), col("tau_b_stance_des1"), col("tau_b_stance_des2")],
            axis=1,
        )
    v_des = None
    if "desired_vx_w" in rows[0] and "desired_vy_w" in rows[0]:
        v_des = np.stack([col("desired_vx_w"), col("desired_vy_w"), np.zeros(len(rows))], axis=1)

    if "f_contact_b0" in rows[0]:
        # New logs carry the controller's own BODY-frame GRF directly.
        fb = np.stack([col("f_contact_b0"), col("f_contact_b1"), col("f_contact_b2")], axis=1)
    else:
        # Old logs only have world frame: reconstruct body via R(rpy_hat)^T.
        fw = np.stack([col("f_contact_w0"), col("f_contact_w1"), col("f_contact_w2")], axis=1)
        yaw = col("rpy_hat_yaw")
        fb = np.empty_like(fw)
        for i in range(len(rows)):
            fb[i] = rot_wb(roll[i], pitch[i], yaw[i]).T @ fw[i]
        print("(old log: f_contact_b not logged, reconstructed from world + rpy_hat)")

    d = np.diff(st)
    tds = np.where(d == 1)[0] + 1
    los = np.where(d == -1)[0] + 1
    if st[0] == 1:
        tds = np.r_[0, tds]
    n_hops = len(tds)
    print(f"log: {log_path}\n{n_hops} touchdowns found")

    hops = list(range(1, n_hops + 1)) if args.hops is None \
        else [k for k in args.hops if 1 <= k <= n_hops]
    if not hops:
        raise SystemExit(f"no hops to plot (log has {n_hops}, asked {args.hops})")

    # (hop#, td_idx, lo_idx) for each selected hop
    segs = []
    for k in hops:
        a = int(tds[k - 1])
        b_cand = los[los > a]
        segs.append((k, a, int(b_cand[0]) if len(b_cand) else len(t) - 1))

    m0 = max(0, np.searchsorted(t, t[segs[0][1]] - args.margin_s))
    m1 = min(len(t), np.searchsorted(t, t[segs[-1][2]] + args.margin_s))
    sl = slice(m0, m1)
    ts = t[sl]

    fig, axs = plt.subplots(6, 1, figsize=(max(12, 3 * len(segs)), 15.5), sharex=True)

    def shade(ax, label_hops=False):
        for k, a, b in segs:
            ax.axvspan(t[a], t[b], color="tab:blue", alpha=0.10)
            if label_hops:
                ax.text((t[a] + t[b]) / 2, ax.get_ylim()[1] * 0.92, f"hop {k}",
                        ha="center", va="top", fontsize=9, color="tab:blue")

    # Hiptorque cap trace: only meaningful in stance (NaN in flight so nothing drawn).
    tau_cap = np.full(len(t), TAU_CAP_NM if TAU_CAP_NM > 0.0 else np.nan)
    tau_cap[st == 0] = np.nan
    tau_lbl = f"hiptorque cap {TAU_CAP_NM:.1f} Nm"

    ax = axs[0]
    if tau_hip is not None:
        ax.plot(ts, tau_hip[sl, 0], label="hiptorque_x (roll)", color="tab:blue")
        ax.plot(ts, tau_hip[sl, 1], label="hiptorque_y (pitch)", color="tab:orange")
        ax.plot(ts, tau_cap[sl], ls="--", c="r", lw=1.0, label=tau_lbl)
        ax.plot(ts, -tau_cap[sl], ls="--", c="r", lw=1.0)
    else:
        ax.text(0.5, 0.5, "tau_b_stance_des not in log", transform=ax.transAxes, ha="center")
    ax.axhline(0, color="k", lw=0.5)
    shade(ax, label_hops=True)
    ax.set_ylabel("hiptorque [Nm]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f"hops {hops[0]}..{hops[-1]} (shaded = stance, white = flight)")

    ax = axs[1]
    if tau_hip is not None:
        ax.plot(ts, np.hypot(tau_hip[sl, 0], tau_hip[sl, 1]), color="tab:red", label="|hiptorque_xy|")
        ax.plot(ts, tau_cap[sl], ls="--", c="r", lw=1.0, label=tau_lbl)
    else:
        ax.text(0.5, 0.5, "tau_b_stance_des not in log", transform=ax.transAxes, ha="center")
    shade(ax)
    ax.set_ylabel("|hiptorque| [Nm]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[2]
    ax.plot(ts, np.degrees(roll[sl]), label="roll", color="tab:purple")
    ax.plot(ts, np.degrees(pitch[sl]), label="pitch", color="tab:brown")
    ax.axhline(0, color="k", lw=0.5)
    shade(ax)
    ax.set_ylabel("deg")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # Row 4: base height, with the apex of each flight annotated.
    ax = axs[3]
    ax.plot(ts, height[sl], color="tab:cyan", label="base height (-p_hat_w z)")
    shade(ax)
    apexes = []
    for j, (k, a, b) in enumerate(segs):
        f_end = segs[j + 1][1] if j + 1 < len(segs) else m1 - 1
        if f_end <= b:
            continue
        i_apex = b + int(np.argmax(height[b:f_end]))
        z_td = height[a]                      # height when this stance began
        apex_rise = height[i_apex] - z_td     # hop height above touchdown level
        apexes.append((k, t[i_apex], height[i_apex], apex_rise))
        ax.plot(t[i_apex], height[i_apex], "v", color="tab:red", ms=7)
        ax.annotate(f"apex +{100*apex_rise:.1f}cm",
                    (t[i_apex], height[i_apex]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="tab:red")
    ax.set_ylabel("height [m]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # Row 5: body velocity (world frame, +Z down).
    ax = axs[4]
    if v_hat is not None:
        ax.plot(ts, v_hat[sl, 0], label="v_hat_x", color="tab:blue")
        ax.plot(ts, v_hat[sl, 1], label="v_hat_y", color="tab:orange")
        ax.plot(ts, v_hat[sl, 2], label="v_hat_z", color="tab:green", alpha=0.85)
    else:
        ax.text(0.5, 0.5, "v_hat_w not in log", transform=ax.transAxes, ha="center")
    if v_meas is not None:
        ax.plot(ts, v_meas[sl, 0], ls="--", lw=0.9, color="tab:blue", alpha=0.45, label="v_meas_x")
        ax.plot(ts, v_meas[sl, 1], ls="--", lw=0.9, color="tab:orange", alpha=0.45, label="v_meas_y")
    if v_des is not None:
        ax.plot(ts, v_des[sl, 0], ls=":", lw=1.0, color="k", alpha=0.6, label="v_des_x")
        ax.plot(ts, v_des[sl, 1], ls=":", lw=1.0, color="k", alpha=0.6, label="v_des_y")
    ax.axhline(0, color="k", lw=0.5)
    shade(ax)
    ax.set_ylim(-2, 2)
    ax.set_ylabel("velocity [m/s]")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Row 6: swing-leg foot tracking in BODY frame (NO quaternion). Target
    # foot_des_b is only computed in FLIGHT (NaN/held in stance), so the swing
    # tracking is read in the white (flight) bands. Solid = actual foot_b,
    # dashed = commanded foot_des_b, per axis.
    ax = axs[5]
    if foot_des_b is not None and foot_b is not None:
        cols3 = ("tab:blue", "tab:orange", "tab:green")
        names = ("x", "y", "z")
        for i in range(3):
            ax.plot(ts, foot_b[sl, i], color=cols3[i], lw=1.3, label=f"foot_{names[i]} (real)")
            ax.plot(ts, foot_des_b[sl, i], color=cols3[i], ls="--", lw=1.1,
                    alpha=0.8, label=f"foot_{names[i]} (target)")
    else:
        ax.text(0.5, 0.5, "foot_des_b / foot_b not in log", transform=ax.transAxes, ha="center")
    ax.axhline(0, color="k", lw=0.5)
    shade(ax)
    ax.set_ylabel("foot pos body [m]")
    ax.set_xlabel("log time t_s [s]")
    ax.legend(loc="lower right", fontsize=7, ncol=3)
    ax.grid(alpha=0.3)

    for k, a, b in segs:
        seg = slice(a, b)
        ap = next((x for x in apexes if x[0] == k), None)
        ap_s = f"apex +{100*ap[3]:.1f}cm (abs {ap[2]:.3f}m)" if ap else "apex n/a"
        v_lo = ""
        if v_hat is not None:
            v_lo = (f"  v_hat@LO [{v_hat[b,0]:+.2f},{v_hat[b,1]:+.2f},{v_hat[b,2]:+.2f}]")
        print(f"hop {k}: TD@{t[a]:7.3f}s Ts={(t[b]-t[a])*1000:4.0f}ms  "
              f"fx_b[{fb[seg,0].min():+6.1f}..{fb[seg,0].max():+6.1f} mean {fb[seg,0].mean():+6.1f}]  "
              f"fy_b[{fb[seg,1].min():+6.1f}..{fb[seg,1].max():+6.1f} mean {fb[seg,1].mean():+6.1f}]  "
              f"roll {np.degrees(roll[a]):+.1f}->{np.degrees(roll[b]):+.1f}deg  "
              f"pitch {np.degrees(pitch[a]):+.1f}->{np.degrees(pitch[b]):+.1f}deg  {ap_s}{v_lo}")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(log_path), "hops_combined.png")
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
