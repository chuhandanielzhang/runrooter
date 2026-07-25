#!/usr/bin/env python3
"""Torque + Fz vs time plot (same style as torque_fz_vs_time_151345.png).

Usage:
    python tools/plot_torque_fz.py            # newest logs/modee_*.csv
    python tools/plot_torque_fz.py <log.csv>  # a specific log
Output: logs/figs/torque_fz_vs_time_<HHMMSS>.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path(__file__).resolve().parent.parent / "logs"


def newest_log() -> Path:
    logs = sorted(LOGS.glob("modee_2*.csv"), key=lambda p: p.stat().st_mtime)
    if not logs:
        sys.exit(f"no modee_*.csv logs in {LOGS}")
    return logs[-1]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_log()
    stamp = src.stem.split("_")[-1]
    out = LOGS / "figs" / f"torque_fz_vs_time_{stamp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(src)
    # Window: 200 samples before the first stance to 200 after the last.
    active = d["stance"].fillna(0).astype(bool).to_numpy()
    idx = np.flatnonzero(active)
    lo = max(0, int(idx[0]) - 200) if len(idx) else 0
    hi = min(len(d), int(idx[-1]) + 201) if len(idx) else len(d)
    x = d.iloc[lo:hi].copy()
    t0 = float(x["t_s"].iloc[0])
    t = x["t_s"].to_numpy() - t0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(14, 14), sharex=True, constrained_layout=True
    )
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, c in enumerate(colors):
        ax1.plot(t, x[f"tau_meas{i}"], color=c, lw=1.2, label=f"M{i+1} measured")
        ax1.plot(t, x[f"tau{i}"], color=c, lw=0.8, ls="--", alpha=0.75,
                 label=f"M{i+1} cmd")
    ax1.axhline(10.0, color="crimson", ls=":", lw=1.4, label="±10 Nm driver limit")
    ax1.axhline(-10.0, color="crimson", ls=":", lw=1.4)
    ax1.set_ylabel("Motor torque (Nm)")
    ax1.set_ylim(-11.5, 11.5)
    ax1.grid(True, alpha=0.25)
    ax1.legend(ncol=4, fontsize=8, loc="upper right")
    ax1.set_title(f"Motor Torque and Vertical Force vs Time — {src.name}")

    ax2.plot(t, x["f_ref_w2"], color="black", lw=1.4, label="Fz reference")
    ax2.plot(t, x["f_contact_w2"], color="tab:purple", lw=1.0, alpha=0.8,
             label="Fz allocated/contact")
    ax2.axhline(230.1, color="crimson", ls=":", lw=1.4,
                label="Fz budget 3.5mg ≈ 230 N")
    ax2.set_ylabel("Vertical force Fz (N)")
    ax2.grid(True, alpha=0.25)

    # Active prop channels are pwm1/2/3 (one per arm).
    for i, c in zip((1, 2, 3), colors):
        ax3.plot(t, x[f"pwm{i}"], color=c, lw=1.2, label=f"Prop arm {i} PWM")
    ax3.axhline(1700, color="crimson", ls=":", lw=1.2, label="PUSH base 1700 us")
    ax3.axhline(1100, color="gray", ls=":", lw=1.2, label="flight base 1100 us")
    ax3.set_ylabel("Prop PWM (us)")
    ax3.set_ylim(975, 2050)
    ax3.grid(True, alpha=0.25)
    ax3.legend(ncol=3, fontsize=8, loc="upper right")

    # Leg length + compression (actual vs plan).
    ax4.plot(t, x["leg_len_m"] * 100.0, color="tab:blue", lw=1.2,
             label="leg length")
    ax4b = ax4.twinx()
    ax4b.plot(t, x["comp_m"] * 100.0, color="tab:red", lw=1.2,
              label="compression (actual)")
    if "fbslip_x_c_plan_m" in x.columns:
        ax4b.plot(t, x["fbslip_x_c_plan_m"] * 100.0, color="tab:red",
                  lw=1.0, ls="--", alpha=0.7, label="compression plan x_c")
    if "fbslip_s_tgt_m" in x.columns:
        ax4b.plot(t, x["fbslip_s_tgt_m"] * 100.0, color="darkgreen",
                  lw=1.0, ls="-.", alpha=0.9, label="design-map s_tgt")
    ax4b.set_ylabel("Compression (cm)", color="tab:red")
    ax4b.tick_params(axis="y", labelcolor="tab:red")
    ax4b.set_ylim(bottom=0)
    ax4.set_ylabel("Leg length (cm)", color="tab:blue")
    ax4.tick_params(axis="y", labelcolor="tab:blue")
    ax4.set_xlabel(f"Time since {t0:.3f} s (s)")
    ax4.grid(True, alpha=0.25)
    h4, l4 = ax4.get_legend_handles_labels()
    h4b, l4b = ax4b.get_legend_handles_labels()
    ax4.legend(h4 + h4b, l4 + l4b, ncol=3, fontsize=8, loc="upper right")

    phase = x["phase"].astype(str).to_numpy()
    for label, face, alpha in [
        ("STANCE:COMP", "royalblue", 0.08),
        ("STANCE:PUSH", "darkorange", 0.10),
    ]:
        mask = phase == label
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        for a, b in zip(starts, ends):
            ax1.axvspan(t[a], t[b], color=face, alpha=alpha, lw=0)
            ax2.axvspan(t[a], t[b], color=face, alpha=alpha, lw=0,
                        label=label if a == starts[0] else None)
            ax3.axvspan(t[a], t[b], color=face, alpha=alpha, lw=0)
            ax4.axvspan(t[a], t[b], color=face, alpha=alpha, lw=0)

    handles, labels = ax2.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            H.append(h)
            L.append(l)
    ax2.legend(H, L, loc="upper left", fontsize=8)

    fig.savefig(out, dpi=140)
    print(out)
    # Open the figure in the system image viewer.
    import subprocess
    subprocess.Popen(
        ["xdg-open", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
