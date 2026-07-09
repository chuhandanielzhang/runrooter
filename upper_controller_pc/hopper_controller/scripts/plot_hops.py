#!/usr/bin/env python3
"""Plot stance diagnostics from a ModeE csv log on ONE continuous timeline
(x = log time t_s), all hops together, stance windows shaded + numbered.

Rows:
  1. BODY-frame fx/fy with the +/-20 N cap lines
  2. fz_b, the friction cone 0.3*fz, and |fxy_b| (shows which limit binds)
  3. roll / pitch
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
except Exception:
    FXY_CAP_N = 20.0
    STANCE_MU = 0.3


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

    fig, axs = plt.subplots(4, 1, figsize=(max(12, 3 * len(segs)), 11), sharex=True)

    def shade(ax, label_hops=False):
        for k, a, b in segs:
            ax.axvspan(t[a], t[b], color="tab:blue", alpha=0.10)
            if label_hops:
                ax.text((t[a] + t[b]) / 2, ax.get_ylim()[1] * 0.92, f"hop {k}",
                        ha="center", va="top", fontsize=9, color="tab:blue")

    # The enforced limit is fz-dependent: lim(t) = min(mu*fz(t), cap).
    # Only meaningful in stance; NaN elsewhere so nothing is drawn in flight.
    lim = np.full(len(t), np.inf)
    if STANCE_MU > 0.0:
        lim = np.minimum(lim, STANCE_MU * np.clip(fb[:, 2], 0.0, None))
    if FXY_CAP_N > 0.0:
        lim = np.minimum(lim, FXY_CAP_N)
    lim[~np.isfinite(lim)] = np.nan
    lim[st == 0] = np.nan
    lim_lbl = (f"limit min({STANCE_MU}*fz, {FXY_CAP_N:.0f}N)" if STANCE_MU > 0.0
               else f"limit {FXY_CAP_N:.0f}N")

    ax = axs[0]
    ax.plot(ts, fb[sl, 0], label="fx_b", color="tab:blue")
    ax.plot(ts, fb[sl, 1], label="fy_b", color="tab:orange")
    ax.plot(ts, lim[sl], ls="--", c="r", lw=1.0, label=lim_lbl)
    ax.plot(ts, -lim[sl], ls="--", c="r", lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    shade(ax, label_hops=True)
    ax.set_ylabel("body force [N]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f"hops {hops[0]}..{hops[-1]} (shaded = stance, white = flight)")

    ax = axs[1]
    ax.plot(ts, fb[sl, 2], label="fz_b", color="tab:green")
    ax.plot(ts, lim[sl], ls="--", c="r", lw=1.0, label=lim_lbl)
    ax.plot(ts, np.hypot(fb[sl, 0], fb[sl, 1]), color="tab:red", label="|fxy_b|")
    shade(ax)
    ax.set_ylabel("N")
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
    ax.set_xlabel("log time t_s [s]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    for k, a, b in segs:
        seg = slice(a, b)
        ap = next((x for x in apexes if x[0] == k), None)
        ap_s = f"apex +{100*ap[3]:.1f}cm (abs {ap[2]:.3f}m)" if ap else "apex n/a"
        print(f"hop {k}: TD@{t[a]:7.3f}s Ts={(t[b]-t[a])*1000:4.0f}ms  "
              f"fx_b[{fb[seg,0].min():+6.1f}..{fb[seg,0].max():+6.1f} mean {fb[seg,0].mean():+6.1f}]  "
              f"fy_b[{fb[seg,1].min():+6.1f}..{fb[seg,1].max():+6.1f} mean {fb[seg,1].mean():+6.1f}]  "
              f"roll {np.degrees(roll[a]):+.1f}->{np.degrees(roll[b]):+.1f}deg  "
              f"pitch {np.degrees(pitch[a]):+.1f}->{np.degrees(pitch[b]):+.1f}deg  {ap_s}")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(log_path), "hops_combined.png")
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
