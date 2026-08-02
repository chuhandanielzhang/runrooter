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

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(
        7, 1, figsize=(14, 23), sharex=True, constrained_layout=True
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

    # Leg Fz only (f_ref is capped at nrc_leg_fz_max; prop residual is
    # NOT folded back into f_ref -- see prop_energy_fz / thrust_sum).
    ax2.plot(t, x["f_ref_w2"], color="black", lw=1.4, label="leg Fz cmd (f_ref)")
    ax2.plot(t, x["f_contact_w2"], color="tab:purple", lw=1.0, alpha=0.8,
             label="leg Fz contact")
    if "prop_energy_fz" in x.columns:
        ax2.plot(t, x["prop_energy_fz"], color="tab:green", lw=1.2,
                 label="prop Fz residual (energy)")
    # Delivered prop collective (sum of 3 arm thrusts) -- what the body
    # actually feels from the props on the world vertical (upright).
    if all(c in x.columns for c in ("thrust0", "thrust1", "thrust2")):
        tsum = (
            x["thrust0"].to_numpy(dtype=float)
            + x["thrust1"].to_numpy(dtype=float)
            + x["thrust2"].to_numpy(dtype=float)
        )
        ax2.plot(t, tsum, color="tab:red", lw=1.0, ls="--", alpha=0.85,
                 label="prop thrust_sum (delivered)")
        if "prop_energy_fz" in x.columns:
            ax2.plot(
                t,
                x["f_ref_w2"].to_numpy(dtype=float)
                + x["prop_energy_fz"].to_numpy(dtype=float),
                color="gray", lw=1.0, ls=":", alpha=0.9,
                label="leg+prop Fz cmd",
            )
    ax2.axhline(70.0, color="tab:blue", ls=":", lw=1.2,
                label="leg Fz ceiling 70 N")
    ax2.axhline(230.1, color="crimson", ls=":", lw=1.0, alpha=0.6,
                label="design F_max 3.5mg ≈ 230 N")
    ax2.set_ylabel("Vertical force Fz (N)")
    ax2.grid(True, alpha=0.25)

    # Active prop channels are pwm1/2/3 (one per arm; pwm0/4/5 idle).
    for i, c in zip((1, 2, 3), colors):
        ax3.plot(t, x[f"pwm{i}"], color=c, lw=1.2, label=f"Prop arm {i} PWM")
    ax3.axhline(1950, color="crimson", ls=":", lw=1.2, label="PWM max 1950 us")
    ax3.axhline(1250, color="gray", ls=":", lw=1.2, label="~idle base")
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
    ax4.grid(True, alpha=0.25)
    h4, l4 = ax4.get_legend_handles_labels()
    h4b, l4b = ax4b.get_legend_handles_labels()
    ax4.legend(h4 + h4b, l4 + l4b, ncol=3, fontsize=8, loc="upper right")

    # XY velocity estimate used by Raibert + flight tilt (world FRD).
    ax5.plot(t, x["v_hat_w0"], color="tab:blue", lw=1.2, label="vx_hat (world)")
    ax5.plot(t, x["v_hat_w1"], color="tab:orange", lw=1.2, label="vy_hat (world)")
    if "desired_vx_w" in x.columns and "desired_vy_w" in x.columns:
        ax5.plot(t, x["desired_vx_w"], color="tab:blue", lw=1.0, ls="--",
                 alpha=0.7, label="vx_des")
        ax5.plot(t, x["desired_vy_w"], color="tab:orange", lw=1.0, ls="--",
                 alpha=0.7, label="vy_des")
    ax5.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax5.set_ylabel("XY velocity (m/s)")
    ax5.grid(True, alpha=0.25)
    ax5.legend(ncol=4, fontsize=8, loc="upper right")

    # Body attitude (estimator roll/pitch; yaw is free and not plotted).
    roll_deg = np.degrees(x["rpy_hat_roll"].to_numpy(dtype=float))
    pitch_deg = np.degrees(x["rpy_hat_pitch"].to_numpy(dtype=float))
    ax6.plot(t, roll_deg, color="tab:blue", lw=1.2, label="roll")
    ax6.plot(t, pitch_deg, color="tab:orange", lw=1.2, label="pitch")
    if "fl_tilt_cmd_deg" in x.columns:
        ax6.plot(t, x["fl_tilt_cmd_deg"], color="tab:green", lw=1.0, ls="--",
                 alpha=0.8, label="fl_tilt_cmd")
    ax6.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax6.set_ylabel("Body attitude (deg)")
    ax6.grid(True, alpha=0.25)
    ax6.legend(ncol=3, fontsize=8, loc="upper right")

    # Foot placement: desired (Raibert) vs actual foot position in world XYZ.
    # Note: foot_b is body FRD; foot_des_w and foot_des_b are logged as well.
    ax7.set_title("Foot placement: desired vs actual (world XYZ, cm)")
    for i, (comp, c) in enumerate(zip(["x", "y", "z"], colors)):
        ax7.plot(t, x[f"foot_b{i}"] * 100.0, color=c, lw=1.2,
                 label=f"foot_{comp} actual (body)")
        if f"foot_des_w{i}" in x.columns:
            ax7.plot(t, x[f"foot_des_w{i}"] * 100.0, color=c, lw=1.0,
                     ls="--", alpha=0.75, label=f"foot_{comp} des (world)")
    ax7.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax7.set_ylabel("Foot pos (cm)")
    ax7.set_xlabel(f"Time since {t0:.3f} s (s)")
    ax7.grid(True, alpha=0.25)
    ax7.legend(ncol=3, fontsize=7, loc="upper right")

    phase = x["phase"].astype(str).to_numpy()
    axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7)
    for label, face, alpha in [
        ("STANCE:COMP", "royalblue", 0.08),
        ("STANCE:PUSH", "darkorange", 0.10),
    ]:
        mask = phase == label
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        for a, b in zip(starts, ends):
            for ax in axes:
                ax.axvspan(
                    t[a], t[b], color=face, alpha=alpha, lw=0,
                    label=(label if (ax is ax2 and a == starts[0]) else None),
                )

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
