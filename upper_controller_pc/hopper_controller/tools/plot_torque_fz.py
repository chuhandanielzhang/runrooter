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

    fig, (ax1, axF, ax2, ax3, ax4, ax5, ax6, axA, ax7) = plt.subplots(
        9, 1, figsize=(14, 29), sharex=True, constrained_layout=True
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
    ax1.set_title(f"Motor Torque / Prop Force / Attitude — {src.name}")

    # Leg XYZ force that maps to joint torque (f_tau_delta / J^T F),
    # plus world-frame contact / reference for stance.
    for i, (comp, c) in enumerate(zip(["x", "y", "z"], colors)):
        if f"f_tau_delta{i}" in x.columns:
            axF.plot(t, x[f"f_tau_delta{i}"], color=c, lw=1.3,
                     label=f"f_tau_{comp} (→JᵀF)")
        if f"f_contact_w{i}" in x.columns:
            axF.plot(t, x[f"f_contact_w{i}"], color=c, lw=1.0, ls="--",
                     alpha=0.7, label=f"f_contact_w{comp}")
        if f"f_ref_w{i}" in x.columns:
            axF.plot(t, x[f"f_ref_w{i}"], color=c, lw=0.9, ls=":",
                     alpha=0.85, label=f"f_ref_w{comp}")
    axF.axhline(0.0, color="gray", ls=":", lw=1.0)
    axF.set_ylabel("Leg force XYZ (N)")
    axF.grid(True, alpha=0.25)
    axF.legend(ncol=3, fontsize=7, loc="upper right")

    # Propeller world-frame force: actual = F_total_w - f_contact_w
    # (= z_thrust_w * thrust_sum).  Cmd = same axis * thrust_sum_ref.
    has_prop_f = all(
        c in x.columns
        for c in (
            "F_total_w0", "F_total_w1", "F_total_w2",
            "f_contact_w0", "f_contact_w1", "f_contact_w2",
            "thrust_sum", "thrust_sum_ref",
        )
    )
    if has_prop_f:
        f_prop_act = np.column_stack([
            x["F_total_w0"].to_numpy(dtype=float)
            - x["f_contact_w0"].to_numpy(dtype=float),
            x["F_total_w1"].to_numpy(dtype=float)
            - x["f_contact_w1"].to_numpy(dtype=float),
            x["F_total_w2"].to_numpy(dtype=float)
            - x["f_contact_w2"].to_numpy(dtype=float),
        ])
        tsum = x["thrust_sum"].to_numpy(dtype=float)
        tref = x["thrust_sum_ref"].to_numpy(dtype=float)
        scale = np.divide(
            tref, tsum,
            out=np.zeros_like(tref),
            where=np.abs(tsum) > 1e-6,
        )
        f_prop_cmd = f_prop_act * scale[:, None]
        for i, (comp, c) in enumerate(zip(["x", "y", "z"], colors)):
            ax2.plot(t, f_prop_cmd[:, i], color=c, lw=1.3,
                     label=f"prop F{comp} cmd")
            ax2.plot(t, f_prop_act[:, i], color=c, lw=1.0, ls="--",
                     alpha=0.85, label=f"prop F{comp} actual")
        ax2.plot(t, tref, color="black", lw=1.0, ls=":", alpha=0.7,
                 label="thrust_sum_ref (|cmd|)")
        ax2.plot(t, tsum, color="gray", lw=1.0, ls=":", alpha=0.7,
                 label="thrust_sum (|actual|)")
    else:
        # Fallback if an older log is missing F_total_w.
        if "thrust_sum_ref" in x.columns:
            ax2.plot(t, x["thrust_sum_ref"], color="black", lw=1.3,
                     label="thrust_sum_ref (cmd)")
        if "thrust_sum" in x.columns:
            ax2.plot(t, x["thrust_sum"], color="tab:red", lw=1.0, ls="--",
                     label="thrust_sum (actual)")
    ax2.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax2.set_ylabel("Prop force XYZ (N)")
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

    # Apex height used by the return map / height law (latched at apex).
    if "z_apex_actual_m" in x.columns:
        axA.plot(t, x["z_apex_actual_m"] * 100.0, color="tab:purple", lw=1.3,
                 label="z_apex_actual (held)")
    if "hop_height_m" in x.columns:
        axA.plot(t, x["hop_height_m"] * 100.0, color="black", lw=1.0, ls="--",
                 label="hop_height target")
    apex_mask = np.zeros(len(x), dtype=bool)
    if "apex" in x.columns:
        apex_mask = x["apex"].fillna(0).astype(bool).to_numpy()
    # Per-hop apex from the FLIGHT-TIME measurement (h = g*T^2/8 variant),
    # which core.py writes into z_apex_actual_m at each touchdown. The
    # in-flight "apex event" value comes from vz-crossing + position
    # integration, which drifts badly (log 200745 read 5.8 cm for a real
    # 2.5 cm hop) -- so mark the drift-free TD value at the flight midpoint.
    if all(c in x.columns for c in ("liftoff", "touchdown", "z_apex_actual_m")):
        za = x["z_apex_actual_m"].to_numpy(dtype=float)
        lo_idx = np.where(x["liftoff"].fillna(0).to_numpy(dtype=float) > 0.5)[0]
        td_idx = np.where(x["touchdown"].fillna(0).to_numpy(dtype=float) > 0.5)[0]
        hop_t, hop_h = [], []
        for i_lo in lo_idx:
            nxt = td_idx[td_idx > i_lo]
            if len(nxt) == 0:
                continue
            i_td = int(nxt[0])
            # z_apex_actual_m is overwritten by the flight-time formula ON
            # the TD row; read a row after to be safe against ordering.
            h_cm = za[min(i_td + 1, len(za) - 1)] * 100.0
            if np.isfinite(h_cm):
                hop_t.append(0.5 * (t[i_lo] + t[i_td]))
                hop_h.append(h_cm)
        if hop_t:
            axA.plot(hop_t, hop_h, "o", color="crimson", ms=7, zorder=5,
                     label="apex (flight-time, per hop)")
            for ta, ha in zip(hop_t, hop_h):
                axA.annotate(f"{ha:.1f}", (ta, ha), textcoords="offset points",
                             xytext=(0, 6), ha="center", fontsize=7,
                             color="crimson")
    axA.set_ylabel("Apex height (cm)")
    axA.grid(True, alpha=0.25)
    axA.legend(ncol=3, fontsize=8, loc="upper right")

    # Algorithm attitude estimate (rpy_hat from R_wb_hat) — this is what
    # attitude PD / fl_tilt / Raibert actually close on. IMU raw rpy is
    # overlaid for comparison; apex moments marked.
    roll_hat = np.degrees(x["rpy_hat_roll"].to_numpy(dtype=float))
    pitch_hat = np.degrees(x["rpy_hat_pitch"].to_numpy(dtype=float))
    ax6.plot(t, roll_hat, color="tab:blue", lw=1.3, label="roll_hat (alg)")
    ax6.plot(t, pitch_hat, color="tab:orange", lw=1.3, label="pitch_hat (alg)")
    if "imu_rpy_roll" in x.columns and "imu_rpy_pitch" in x.columns:
        ax6.plot(t, np.degrees(x["imu_rpy_roll"].to_numpy(dtype=float)),
                 color="tab:blue", lw=0.8, ls=":", alpha=0.7, label="roll_imu")
        ax6.plot(t, np.degrees(x["imu_rpy_pitch"].to_numpy(dtype=float)),
                 color="tab:orange", lw=0.8, ls=":", alpha=0.7, label="pitch_imu")
    if "fl_tilt_cmd_deg" in x.columns:
        ax6.plot(t, x["fl_tilt_cmd_deg"], color="tab:green", lw=1.0, ls="--",
                 alpha=0.8, label="fl_tilt_cmd")
    if np.any(apex_mask):
        ax6.plot(t[apex_mask], roll_hat[apex_mask], "o", color="tab:blue",
                 ms=7, zorder=5, label="roll @ apex")
        ax6.plot(t[apex_mask], pitch_hat[apex_mask], "o", color="tab:orange",
                 ms=7, zorder=5, label="pitch @ apex")
        for ta in t[apex_mask]:
            ax6.axvline(ta, color="crimson", ls=":", lw=0.9, alpha=0.55)
            axA.axvline(ta, color="crimson", ls=":", lw=0.9, alpha=0.55)
    ax6.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax6.set_ylabel("Attitude estimate (deg)")
    ax6.grid(True, alpha=0.25)
    ax6.legend(ncol=3, fontsize=7, loc="upper right")

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
    axes = (ax1, axF, ax2, ax3, ax4, ax5, ax6, axA, ax7)
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
