#!/usr/bin/env python3
"""
Plot ModeE CSV logs (produced by hopper_controller/modee/lcm_controller.py).

What it plots (time-domain):
1) Leg/foot convergence: foot_des_b (target) vs foot_b (measured) + error norm
2) Flight (S2S active): velocity convergence + S2S foot target (XY) vs time
3) Stance: velocity convergence (v_hat vs desired_v) vs time

Default behavior:
  - Find newest `modee_*.csv` under ~/hopper_logs/modee_csv (or MODEE_LOG_DIR)
  - Save PNGs next to the log (or --out-dir)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
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


def _to_float(rows: List[Dict[str, str]], key: str, default: float = float("nan")) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        try:
            out[i] = float(r.get(key, ""))
        except Exception:
            out[i] = float(default)
    return out


def _to_int(rows: List[Dict[str, str]], key: str, default: int = 0) -> np.ndarray:
    out = np.empty(len(rows), dtype=int)
    for i, r in enumerate(rows):
        try:
            out[i] = int(float(r.get(key, "")))
        except Exception:
            out[i] = int(default)
    return out


def _stack3(rows: List[Dict[str, str]], base: str) -> np.ndarray:
    return np.vstack(
        [
            _to_float(rows, f"{base}0"),
            _to_float(rows, f"{base}1"),
            _to_float(rows, f"{base}2"),
        ]
    ).T


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ModeE CSV logs (leg target vs real; velocity convergence).")
    ap.add_argument("--log", type=str, default=None, help="Path to a specific modee_*.csv file.")
    ap.add_argument("--log-dir", type=str, default=None, help="Directory to search for newest log.")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to save PNGs. Default: same directory as log.")
    ap.add_argument("--time", choices=["t_s", "wall_time_s"], default="t_s", help="Time axis to use.")
    ap.add_argument("--show", action="store_true", help="Show figures (requires a display).")
    ap.add_argument("--open", action="store_true", default=True, help="Auto-open PNGs with system default viewer (default: True).")
    ap.add_argument("--no-open", dest="open", action="store_false", help="Disable auto-opening PNGs.")
    ap.add_argument("--hop-peak-z", type=float, default=0.7, help="Desired apex height (m). Default: 0.7")
    args = ap.parse_args()

    log_dir = _default_log_dir() if args.log_dir is None else os.path.expanduser(str(args.log_dir))
    log_path = args.log
    if log_path is None:
        log_path = _find_latest_csv(log_dir)
        if log_path is None:
            print("[plot_leg_convergence] No modee_*.csv found.")
            print(f"  searched dir: {log_dir}")
            sys.exit(2)
    log_path = os.path.expanduser(str(log_path))
    if not os.path.isfile(log_path):
        print(f"[plot_leg_convergence] Log not found: {log_path}")
        sys.exit(2)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(log_path)
    out_dir = os.path.expanduser(str(out_dir))
    _ensure_dir(out_dir)

    # Load CSV
    with open(log_path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows: List[Dict[str, str]] = list(reader)

    if len(rows) == 0:
        print(f"[plot_leg_convergence] Empty log: {log_path}")
        sys.exit(2)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot_leg_convergence] matplotlib is required for plotting.")
        print(f"  error: {e}")
        sys.exit(2)

    # Time axis
    t = _to_float(rows, args.time)
    if not np.isfinite(t[0]):
        # fallback
        t = _to_float(rows, "wall_time_s")
    t = (t - float(t[0])) if np.isfinite(t[0]) else t

    stance = _to_int(rows, "stance")
    s2s_active = _to_int(rows, "s2s_active")
    s_stance = _to_float(rows, "s_stance")
    phases = [str(r.get("phase", "")) for r in rows]

    foot_b = _stack3(rows, "foot_b")
    foot_des_b = _stack3(rows, "foot_des_b")
    foot_err = foot_b - foot_des_b
    foot_err_norm = np.linalg.norm(foot_err, axis=1)

    v_hat = _stack3(rows, "v_hat_w")
    v_des = np.vstack([_to_float(rows, "desired_vx_w"), _to_float(rows, "desired_vy_w"), np.zeros(len(rows), dtype=float)]).T

    # ===== Auto-trim: remove data before first STANCE phase =====
    # Find first index where stance != 0
    first_stance_idx = None
    for i in range(len(stance)):
        if stance[i] != 0:
            first_stance_idx = i
            break
    
    if first_stance_idx is not None and first_stance_idx > 0:
        # Trim all arrays from first_stance_idx onwards
        t = t[first_stance_idx:]
        stance = stance[first_stance_idx:]
        s2s_active = s2s_active[first_stance_idx:]
        s_stance = s_stance[first_stance_idx:]
        phases = phases[first_stance_idx:]
        foot_b = foot_b[first_stance_idx:]
        foot_des_b = foot_des_b[first_stance_idx:]
        foot_err = foot_err[first_stance_idx:]
        foot_err_norm = foot_err_norm[first_stance_idx:]
        v_hat = v_hat[first_stance_idx:]
        v_des = v_des[first_stance_idx:]
        # Reset time to start from 0 after trimming
        t = (t - float(t[0])) if np.isfinite(t[0]) and len(t) > 0 else t
        print(f"[plot_leg_convergence] Trimmed {first_stance_idx} rows before first STANCE phase (now {len(t)} rows)")
    elif first_stance_idx is None:
        print(f"[plot_leg_convergence] No STANCE phase found in log, plotting all {len(t)} rows")
    # ===== Phase background shading helper =====
    # We shade each axis background by phase label to make it easy to see FLIGHT / STANCE:COMP / STANCE:PUSH segments.
    phase_colors = {
        "FLIGHT": ("tab:blue", 0.06),
        "STANCE:COMP": ("tab:orange", 0.08),
        "STANCE:PUSH": ("tab:red", 0.06),
        "STANCE": ("tab:orange", 0.05),
    }

    def _shade_by_phase(ax) -> None:
        # We use fill_between with NaN-safe y-limits after the axis has data plotted.
        ymin, ymax = ax.get_ylim()
        for ph, (c, a) in phase_colors.items():
            m = np.array([p == ph for p in phases], dtype=bool)
            if np.any(m):
                ax.fill_between(t, ymin, ymax, where=m, color=c, alpha=float(a), step="pre")
        ax.set_ylim(ymin, ymax)

    v_err_xy = (v_hat[:, 0:2] - v_des[:, 0:2]).astype(float)
    v_err_norm = np.linalg.norm(v_err_xy, axis=1)

    # ===== Figure 1: foot target vs measured =====
    fig1, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 9))
    names = ["x", "y", "z"]
    for i in range(3):
        axs[i].plot(t, foot_des_b[:, i], "k--", linewidth=1.5, label="target foot_des_b")
        axs[i].plot(t, foot_b[:, i], "b-", linewidth=1.2, label="measured foot_b")
        axs[i].set_ylabel(f"foot_{names[i]} (m)")
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc="best")
        _shade_by_phase(axs[i])
    axs[3].plot(t, foot_err_norm, "r-", linewidth=1.2, label="|foot_b - foot_des_b|")
    axs[3].set_ylabel("pos err (m)")
    axs[3].set_xlabel("time (s)")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="best")
    _shade_by_phase(axs[3])
    fig1.suptitle("Leg/foot convergence (BODY frame): target vs measured")

    # ===== Figure 2: Flight (S2S) velocity convergence + foot target (XY) =====
    m_flight_s2s = (stance == 0) & (s2s_active != 0)
    fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    for ax_i, (vel_i, pos_i, lab_v, lab_p) in enumerate(
        [
            (0, 0, "vx_w", "foot_des_bx"),
            (1, 1, "vy_w", "foot_des_by"),
        ]
    ):
        ax = axs2[ax_i]
        ax.plot(t, v_hat[:, vel_i], "b-", linewidth=1.2, label=f"v_hat {lab_v}")
        ax.plot(t, v_des[:, vel_i], "k--", linewidth=1.2, label=f"desired {lab_v}")
        ax.set_ylabel("vel (m/s)")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(t, foot_des_b[:, pos_i], color="tab:green", linewidth=1.0, alpha=0.9, label=f"{lab_p}")
        ax2.set_ylabel("foot target (m)")
        _shade_by_phase(ax)

        # One combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")

    axs2[-1].set_xlabel("time (s)")
    fig2.suptitle("Flight (S2S active): velocity vs desired + S2S foot target (XY)")

    # ===== Figure 3: Stance velocity convergence =====
    m_stance = stance != 0
    fig3, axs3 = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    axs3[0].plot(t, v_hat[:, 0], "b-", linewidth=1.2, label="v_hat vx_w")
    axs3[0].plot(t, v_des[:, 0], "k--", linewidth=1.2, label="desired vx_w")
    axs3[0].set_ylabel("vx (m/s)")
    axs3[0].grid(True, alpha=0.3)
    axs3[0].legend(loc="best")
    _shade_by_phase(axs3[0])

    axs3[1].plot(t, v_hat[:, 1], "b-", linewidth=1.2, label="v_hat vy_w")
    axs3[1].plot(t, v_des[:, 1], "k--", linewidth=1.2, label="desired vy_w")
    axs3[1].set_ylabel("vy (m/s)")
    axs3[1].grid(True, alpha=0.3)
    axs3[1].legend(loc="best")
    _shade_by_phase(axs3[1])

    axs3[2].plot(t, v_err_norm, "r-", linewidth=1.2, label="||v_hat_xy - v_des_xy||")
    axs3[2].plot(t, s_stance, color="tab:purple", linewidth=1.0, alpha=0.9, label="s_stance (0..1)")
    axs3[2].set_ylabel("err / s")
    axs3[2].set_xlabel("time (s)")
    axs3[2].grid(True, alpha=0.3)
    axs3[2].legend(loc="best")
    _shade_by_phase(axs3[2])

    fig3.suptitle("Stance: velocity convergence (highlighted in orange)")

    # ===== Figure 4: Takeoff vz convergence + Apex height vs desired =====
    liftoff = _to_int(rows, "liftoff")
    apex_evt = _to_int(rows, "apex")
    touchdown = _to_int(rows, "touchdown")
    p_hat_z = _to_float(rows, "p_hat_w2")
    v_hat_z = _to_float(rows, "v_hat_w2")
    
    # Apply trim if needed
    if first_stance_idx is not None and first_stance_idx > 0:
        liftoff = liftoff[first_stance_idx:]
        apex_evt = apex_evt[first_stance_idx:]
        touchdown = touchdown[first_stance_idx:]
        p_hat_z = p_hat_z[first_stance_idx:]
        v_hat_z = v_hat_z[first_stance_idx:]
    
    # Extract takeoff vz (at liftoff events) and apex heights (at apex events)
    takeoff_vz = []
    takeoff_t = []
    apex_z = []
    apex_t = []
    
    # Track touchdown z for each hop (to compute desired vz)
    last_td_z = None
    for i in range(len(t)):
        if touchdown[i] != 0:
            if np.isfinite(p_hat_z[i]):
                last_td_z = float(p_hat_z[i])
        if liftoff[i] != 0:
            if np.isfinite(v_hat_z[i]):
                takeoff_vz.append(float(v_hat_z[i]))
                takeoff_t.append(float(t[i]))
        if apex_evt[i] != 0:
            if np.isfinite(p_hat_z[i]):
                apex_z.append(float(p_hat_z[i]))
                apex_t.append(float(t[i]))
    
    # Compute desired takeoff vz from hop_peak_z and touchdown z
    # v_to = sqrt(2 * g * (hop_peak_z - z_td))
    hop_peak_z_desired = float(args.hop_peak_z)
    g = 9.81  # gravity
    desired_vz_list = []
    for i, t_vz in enumerate(takeoff_vz):
        # Find most recent touchdown z before this liftoff (search backwards)
        td_z = None
        for j in range(len(t) - 1, -1, -1):
            if t[j] <= takeoff_t[i] and touchdown[j] != 0 and np.isfinite(p_hat_z[j]):
                td_z = float(p_hat_z[j])
                break
        if td_z is not None and td_z < hop_peak_z_desired:
            dz = hop_peak_z_desired - td_z
            desired_vz_list.append(float(np.sqrt(2.0 * g * dz)))
        else:
            desired_vz_list.append(float("nan"))
    
    desired_vz_array = np.array(desired_vz_list, dtype=float)
    
    fig4, axs4 = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    
    # Top: Takeoff vz vs desired (with phase background)
    if len(takeoff_vz) > 0:
        takeoff_vz_arr = np.array(takeoff_vz, dtype=float)
        takeoff_t_arr = np.array(takeoff_t, dtype=float)
        axs4[0].plot(takeoff_t_arr, takeoff_vz_arr, "bo-", linewidth=1.5, markersize=6, label="takeoff vz (actual)")
        if np.any(np.isfinite(desired_vz_array)):
            axs4[0].plot(takeoff_t_arr, desired_vz_array, "r--", linewidth=1.5, label=f"desired vz (from hop_peak_z={hop_peak_z_desired:.2f}m)")
        axs4[0].set_ylabel("takeoff vz (m/s)")
        axs4[0].grid(True, alpha=0.3)
        axs4[0].legend(loc="best")
        # Add phase background shading (need to create a time series for shading)
        _shade_by_phase(axs4[0])
    else:
        axs4[0].text(0.5, 0.5, "No liftoff events found", ha="center", va="center", transform=axs4[0].transAxes)
        axs4[0].set_ylabel("takeoff vz (m/s)")
        _shade_by_phase(axs4[0])
    
    # Bottom: Apex height vs desired (with phase background)
    if len(apex_z) > 0:
        apex_z_arr = np.array(apex_z, dtype=float)
        apex_t_arr = np.array(apex_t, dtype=float)
        axs4[1].plot(apex_t_arr, apex_z_arr, "go-", linewidth=1.5, markersize=6, label="apex z (actual)")
        axs4[1].axhline(y=hop_peak_z_desired, color="r", linestyle="--", linewidth=1.5, label=f"desired hop_peak_z={hop_peak_z_desired:.2f}m")
        axs4[1].set_ylabel("apex height (m)")
        axs4[1].grid(True, alpha=0.3)
        axs4[1].legend(loc="best")
        _shade_by_phase(axs4[1])
    else:
        axs4[1].text(0.5, 0.5, "No apex events found", ha="center", va="center", transform=axs4[1].transAxes)
        axs4[1].set_ylabel("apex height (m)")
        _shade_by_phase(axs4[1])
    
    axs4[1].set_xlabel("time (s)")
    fig4.suptitle("Takeoff vz convergence + Apex height vs desired")

    # ===== Figure 5: Leg velocity (body) + Robot velocity (body) + Foot position (body) =====
    # Load quaternion for world->body transformation (wxyz format)
    q_hat_w = _to_float(rows, "q_hat_w")
    q_hat_x = _to_float(rows, "q_hat_x")
    q_hat_y = _to_float(rows, "q_hat_y")
    q_hat_z = _to_float(rows, "q_hat_z")
    q_hat = np.vstack([q_hat_w, q_hat_x, q_hat_y, q_hat_z]).T  # shape: (N, 4) [w, x, y, z]
    
    # Load foot velocity (body frame)
    foot_vrel_b = _stack3(rows, "foot_vrel_b")
    
    # Apply trim if needed
    if first_stance_idx is not None and first_stance_idx > 0:
        q_hat = q_hat[first_stance_idx:]
        foot_vrel_b = foot_vrel_b[first_stance_idx:]
    
    # Convert v_hat_w (world) to body frame
    def quat_to_R_wb(q_wxyz):
        """Convert quaternion (wxyz) to rotation matrix R_wb (world->body)"""
        w, x, y, z = q_wxyz
        n = (w*w + x*x + y*y + z*z)**0.5
        if n < 1e-12:
            return np.eye(3)
        w, x, y, z = w/n, x/n, y/n, z/n
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
        ], dtype=float)
    
    v_hat_b = np.zeros_like(v_hat)
    for i in range(len(t)):
        if np.all(np.isfinite(q_hat[i])) and np.all(np.isfinite(v_hat[i])):
            R_wb = quat_to_R_wb(q_hat[i])
            # v_body = R_wb^T @ v_world (body frame velocity)
            v_hat_b[i] = (R_wb.T @ v_hat[i].reshape(3)).reshape(3)
        else:
            v_hat_b[i] = np.nan
    
    fig5, axs5 = plt.subplots(6, 1, sharex=True, figsize=(12, 12))
    names = ["x", "y", "z"]
    
    # Top 3 subplots: velocities (body frame)
    for i in range(3):
        ax = axs5[i]
        ax.plot(t, foot_vrel_b[:, i], "g-", linewidth=1.5, label=f"leg vel_{names[i]} (body)")
        ax.plot(t, v_hat_b[:, i], "b-", linewidth=1.2, label=f"robot vel_{names[i]} (body)")
        ax.set_ylabel(f"vel_{names[i]} (m/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
        _shade_by_phase(ax)
    
    # Bottom 3 subplots: positions (body frame)
    for i in range(3):
        ax = axs5[i+3]
        ax.plot(t, foot_des_b[:, i], "k--", linewidth=1.5, label=f"target pos_{names[i]}")
        ax.plot(t, foot_b[:, i], "r-", linewidth=1.2, label=f"real pos_{names[i]}")
        ax.set_ylabel(f"pos_{names[i]} (m)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
        _shade_by_phase(ax)
    
    axs5[-1].set_xlabel("time (s)")
    fig5.suptitle("Leg velocity (body) + Robot velocity (body) + Foot position (body)")

    # Save
    base = os.path.splitext(os.path.basename(log_path))[0]
    p1 = os.path.join(out_dir, f"{base}_leg_pos_convergence.png")
    p2 = os.path.join(out_dir, f"{base}_flight_s2s_vel_target.png")
    p3 = os.path.join(out_dir, f"{base}_stance_vel_convergence.png")
    p4 = os.path.join(out_dir, f"{base}_takeoff_apex_convergence.png")
    p5 = os.path.join(out_dir, f"{base}_leg_robot_vel_pos_body.png")
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig1.savefig(p1, dpi=160)
    fig2.savefig(p2, dpi=160)
    fig3.savefig(p3, dpi=160)
    fig4.savefig(p4, dpi=160)
    fig5.savefig(p5, dpi=160)

    print("=" * 70)
    print("[plot_leg_convergence] Done")
    print(f"- log: {log_path}")
    print(f"- out: {out_dir}")
    print(f"- png: {p1}")
    print(f"- png: {p2}")
    print(f"- png: {p3}")
    print(f"- png: {p4}")
    print(f"- png: {p5}")
    print("=" * 70)

    # Auto-open PNGs with system default viewer (Linux: xdg-open)
    if bool(args.open):
        for png_path in [p1, p2, p3, p4, p5]:
            if os.path.isfile(png_path):
                try:
                    # Try xdg-open (Linux), fallback to other common viewers
                    if sys.platform.startswith("linux"):
                        subprocess.Popen(["xdg-open", png_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif sys.platform == "darwin":  # macOS
                        subprocess.Popen(["open", png_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif sys.platform.startswith("win"):  # Windows
                        os.startfile(png_path)
                except Exception:
                    pass  # Silently fail if viewer not available

    if bool(args.show):
        plt.show()


if __name__ == "__main__":
    main()


