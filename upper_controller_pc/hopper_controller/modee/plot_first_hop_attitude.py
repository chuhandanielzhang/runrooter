#!/usr/bin/env python3
"""
Plot first hop attitude convergence: check if tau hits limits or kR is too small.

What it plots:
1) First hop: Tau_des (roll/pitch) vs time, with limits
2) First hop: actual tau (joint torques) vs time, with limits
3) First hop: roll/pitch error vs time
4) First hop: Tau_des vs roll/pitch error (to check if kR is sufficient)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np


def _default_log_dir() -> str:
    return os.path.expanduser(os.environ.get("MODEE_LOG_DIR", "~/hopper_logs/modee_csv"))


def _find_latest_csv(log_dir: str) -> str | None:
    if not os.path.isdir(log_dir):
        return None
    best_path = None
    best_mtime = None
    for name in os.listdir(log_dir):
        if not (name.startswith("modee_") and name.endswith(".csv")):
            continue
        path = os.path.join(log_dir, name)
        try:
            st = os.stat(path)
        except Exception:
            continue
        if best_mtime is None or st.st_mtime > best_mtime:
            best_mtime = st.st_mtime
            best_path = path
    return best_path


def _to_float_array(rows: List[Dict[str, str]], key: str, *, default: float = float("nan")) -> np.ndarray:
    arr = []
    for r in rows:
        try:
            v = float(r.get(key, default))
        except (ValueError, TypeError):
            v = default
        arr.append(v)
    return np.asarray(arr, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot first hop attitude convergence analysis.")
    ap.add_argument("--log", type=str, default=None, help="Path to a specific modee_*.csv file.")
    ap.add_argument("--log-dir", type=str, default=None, help="Directory to search for newest log.")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to save PNGs. Default: same directory as log.")
    ap.add_argument("--show", action="store_true", help="Show figures (requires a display).")
    args = ap.parse_args()

    log_dir = _default_log_dir() if args.log_dir is None else os.path.expanduser(str(args.log_dir))
    log_path = args.log
    if log_path is None:
        log_path = _find_latest_csv(log_dir)
        if log_path is None:
            print("[plot_first_hop_attitude] No modee_*.csv found.")
            print(f"  searched dir: {log_dir}")
            sys.exit(2)
    log_path = os.path.expanduser(str(log_path))
    if not os.path.isfile(log_path):
        print(f"[plot_first_hop_attitude] Log not found: {log_path}")
        sys.exit(2)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(log_path)
    out_dir = os.path.expanduser(str(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Load CSV
    with open(log_path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows: List[Dict[str, str]] = list(reader)

    if len(rows) == 0:
        print(f"[plot_first_hop_attitude] Empty log: {log_path}")
        sys.exit(2)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_first_hop_attitude] matplotlib not available. Install: pip install matplotlib")
        sys.exit(1)

    # Extract data
    t_s = _to_float_array(rows, "t_s")
    phase = [r.get("phase", "") for r in rows]
    stance = _to_float_array(rows, "stance")
    
    # Attitude (roll/pitch) from RPY estimate
    roll_deg = np.rad2deg(_to_float_array(rows, "rpy_hat_roll"))
    pitch_deg = np.rad2deg(_to_float_array(rows, "rpy_hat_pitch"))
    # Desired attitude is 0 (level)
    roll_des_deg = np.zeros_like(roll_deg)
    pitch_des_deg = np.zeros_like(pitch_deg)
    
    # Desired torques (from Tau_des_w: world frame, [x=roll, y=pitch, z=yaw])
    tau_des_w0 = _to_float_array(rows, "tau_des_w0")  # roll
    tau_des_w1 = _to_float_array(rows, "tau_des_w1")  # pitch
    tau_des_w2 = _to_float_array(rows, "tau_des_w2")  # yaw (unused)
    
    # Actual joint torques
    tau0 = _to_float_array(rows, "tau0")
    tau1 = _to_float_array(rows, "tau1")
    tau2 = _to_float_array(rows, "tau2")
    
    # Raw torques (before output limiting)
    tau_raw0 = _to_float_array(rows, "tau_raw0")
    tau_raw1 = _to_float_array(rows, "tau_raw1")
    tau_raw2 = _to_float_array(rows, "tau_raw2")
    
    # Limits
    tau_rp_max = 50.0  # stance_tau_rp_max
    tau_cmd_max = 20.0  # tau_cmd_max_nm (per joint)
    
    # Find first hop: from first TD to first APEX
    first_td_idx = None
    first_apex_idx = None
    for i, (p, s) in enumerate(zip(phase, stance)):
        if first_td_idx is None and s > 0.5:  # First touchdown
            first_td_idx = i
        if first_td_idx is not None and p == "APEX":
            first_apex_idx = i
            break
    
    if first_td_idx is None:
        print("[plot_first_hop_attitude] No touchdown found in log.")
        sys.exit(2)
    
    if first_apex_idx is None:
        first_apex_idx = len(rows) - 1
        print("[plot_first_hop_attitude] No apex found, using end of log.")
    
    # Extract first hop
    idx_first_hop = slice(first_td_idx, first_apex_idx + 1)
    t_first = t_s[idx_first_hop] - t_s[first_td_idx]
    phase_first = phase[idx_first_hop]
    stance_first = stance[idx_first_hop]
    
    roll_first = roll_deg[idx_first_hop]
    pitch_first = pitch_deg[idx_first_hop]
    roll_des_first = roll_des_deg[idx_first_hop]
    pitch_des_first = pitch_des_deg[idx_first_hop]
    
    roll_err_first = roll_first - roll_des_first
    pitch_err_first = pitch_first - pitch_des_first
    
    tau_des_roll_first = tau_des_w0[idx_first_hop]
    tau_des_pitch_first = tau_des_w1[idx_first_hop]
    tau_des_norm_first = np.sqrt(tau_des_roll_first**2 + tau_des_pitch_first**2)
    
    tau0_first = tau0[idx_first_hop]
    tau1_first = tau1[idx_first_hop]
    tau2_first = tau2[idx_first_hop]
    tau_raw0_first = tau_raw0[idx_first_hop]
    tau_raw1_first = tau_raw1[idx_first_hop]
    tau_raw2_first = tau_raw2[idx_first_hop]
    
    # Check if Tau_des hits limit
    tau_des_hits_limit = (
        np.any(np.abs(tau_des_roll_first) >= tau_rp_max * 0.95) or
        np.any(np.abs(tau_des_pitch_first) >= tau_rp_max * 0.95)
    )
    
    # Check if actual tau hits limit
    tau_hits_limit = (
        np.any(np.abs(tau_raw0_first) >= tau_cmd_max * 0.95) or
        np.any(np.abs(tau_raw1_first) >= tau_cmd_max * 0.95) or
        np.any(np.abs(tau_raw2_first) >= tau_cmd_max * 0.95)
    )
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"First Hop Attitude Convergence Analysis\n{os.path.basename(log_path)}", fontsize=12)
    
    # Plot 1: Tau_des vs time (with limits)
    ax = axes[0, 0]
    ax.plot(t_first, tau_des_roll_first, "r-", label="Tau_des_roll", linewidth=2)
    ax.plot(t_first, tau_des_pitch_first, "b-", label="Tau_des_pitch", linewidth=2)
    ax.axhline(+tau_rp_max, color="k", linestyle="--", alpha=0.5, label=f"Limit ±{tau_rp_max} Nm")
    ax.axhline(-tau_rp_max, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tau_des (Nm)")
    ax.set_title("Desired Torque (Tau_des) vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if tau_des_hits_limit:
        ax.text(0.02, 0.98, "⚠ Tau_des HITS LIMIT", transform=ax.transAxes,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    
    # Plot 2: Actual tau vs time (with limits)
    ax = axes[0, 1]
    ax.plot(t_first, tau_raw0_first, "r-", label="tau_raw0", linewidth=1.5, alpha=0.7)
    ax.plot(t_first, tau_raw1_first, "g-", label="tau_raw1", linewidth=1.5, alpha=0.7)
    ax.plot(t_first, tau_raw2_first, "b-", label="tau_raw2", linewidth=1.5, alpha=0.7)
    ax.plot(t_first, tau0_first, "r--", label="tau0 (limited)", linewidth=1, alpha=0.5)
    ax.plot(t_first, tau1_first, "g--", label="tau1 (limited)", linewidth=1, alpha=0.5)
    ax.plot(t_first, tau2_first, "b--", label="tau2 (limited)", linewidth=1, alpha=0.5)
    ax.axhline(+tau_cmd_max, color="k", linestyle="--", alpha=0.5, label=f"Limit ±{tau_cmd_max} Nm")
    ax.axhline(-tau_cmd_max, color="k", linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Torque (Nm)")
    ax.set_title("Actual Joint Torques vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if tau_hits_limit:
        ax.text(0.02, 0.98, "⚠ Tau HITS LIMIT", transform=ax.transAxes,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    
    # Plot 3: Roll/pitch error vs time
    ax = axes[1, 0]
    ax.plot(t_first, roll_err_first, "r-", label="Roll error", linewidth=2)
    ax.plot(t_first, pitch_err_first, "b-", label="Pitch error", linewidth=2)
    ax.axhline(0, color="k", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Attitude Error (deg)")
    ax.set_title("Roll/Pitch Error vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Tau_des vs attitude error (to check kR)
    ax = axes[1, 1]
    # Color by time
    scatter = ax.scatter(roll_err_first, pitch_err_first, c=t_first, cmap="viridis", s=20, alpha=0.6)
    ax.set_xlabel("Roll Error (deg)")
    ax.set_ylabel("Pitch Error (deg)")
    ax.set_title("Tau_des vs Attitude Error (colored by time)")
    ax.axhline(0, color="k", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.axvline(0, color="k", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Time (s)")
    
    # Overlay Tau_des vectors (scaled)
    scale = 0.1  # Scale factor for visualization
    for i in range(0, len(t_first), max(1, len(t_first) // 20)):  # Sample every 5%
        if not np.isnan(tau_des_roll_first[i]) and not np.isnan(tau_des_pitch_first[i]):
            ax.arrow(roll_err_first[i], pitch_err_first[i],
                    tau_des_roll_first[i] * scale, tau_des_pitch_first[i] * scale,
                    head_width=0.5, head_length=0.3, fc="red", ec="red", alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(out_dir, "first_hop_attitude.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_first_hop_attitude] Saved: {out_path}")
    
    # Print summary
    print("\n=== First Hop Attitude Analysis ===")
    print(f"First hop duration: {t_first[-1]:.3f} s")
    print(f"Tau_des max: roll={np.nanmax(np.abs(tau_des_roll_first)):.2f} Nm, pitch={np.nanmax(np.abs(tau_des_pitch_first)):.2f} Nm")
    print(f"Tau_des limit: ±{tau_rp_max} Nm")
    print(f"Tau_des hits limit: {tau_des_hits_limit}")
    print(f"Tau raw max: {np.nanmax(np.abs(tau_raw0_first)):.2f}, {np.nanmax(np.abs(tau_raw1_first)):.2f}, {np.nanmax(np.abs(tau_raw2_first)):.2f} Nm")
    print(f"Tau limit: ±{tau_cmd_max} Nm")
    print(f"Tau hits limit: {tau_hits_limit}")
    print(f"Max roll error: {np.nanmax(np.abs(roll_err_first)):.2f} deg")
    print(f"Max pitch error: {np.nanmax(np.abs(pitch_err_first)):.2f} deg")
    
    if tau_des_hits_limit:
        print("\n⚠ WARNING: Tau_des hits limit! Consider increasing stance_tau_rp_max or reducing attitude error.")
    if tau_hits_limit:
        print("\n⚠ WARNING: Actual tau hits limit! Consider increasing tau_cmd_max_nm or reducing Tau_des.")
    if not tau_des_hits_limit and not tau_hits_limit:
        max_err = max(np.nanmax(np.abs(roll_err_first)), np.nanmax(np.abs(pitch_err_first)))
        if max_err > 5.0:
            print(f"\n⚠ WARNING: Large attitude error ({max_err:.2f} deg) but tau not saturated.")
            print("  Consider increasing stance_kR (current: 120.0) for faster convergence.")
    
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()

