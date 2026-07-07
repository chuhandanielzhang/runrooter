#!/usr/bin/env python3
"""
Quick check: does Tau_des hit stance_tau_rp_max limit in the latest log?
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
    ap = argparse.ArgumentParser(description="Check if Tau_des hits stance_tau_rp_max limit.")
    ap.add_argument("--log", type=str, default=None, help="Path to a specific modee_*.csv file.")
    ap.add_argument("--log-dir", type=str, default=None, help="Directory to search for newest log.")
    ap.add_argument("--tau-rp-max", type=float, default=800.0, help="stance_tau_rp_max value (default: 800.0).")
    args = ap.parse_args()

    log_dir = _default_log_dir() if args.log_dir is None else os.path.expanduser(str(args.log_dir))
    log_path = args.log
    if log_path is None:
        log_path = _find_latest_csv(log_dir)
        if log_path is None:
            print("[check_tau_limit] No modee_*.csv found.")
            print(f"  searched dir: {log_dir}")
            sys.exit(2)
    log_path = os.path.expanduser(str(log_path))
    if not os.path.isfile(log_path):
        print(f"[check_tau_limit] Log not found: {log_path}")
        sys.exit(2)

    # Load CSV
    with open(log_path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows: List[Dict[str, str]] = list(reader)

    if len(rows) == 0:
        print(f"[check_tau_limit] Empty log: {log_path}")
        sys.exit(2)

    # Extract data
    t_s = _to_float_array(rows, "t_s")
    phase = [r.get("phase", "") for r in rows]
    stance = _to_float_array(rows, "stance")
    
    # Desired torques (from Tau_des_w: world frame, [x=roll, y=pitch, z=yaw])
    tau_des_w0 = _to_float_array(rows, "tau_des_w0")  # roll
    tau_des_w1 = _to_float_array(rows, "tau_des_w1")  # pitch
    tau_des_w2 = _to_float_array(rows, "tau_des_w2")  # yaw (unused)
    
    # Attitude errors
    roll_deg = np.rad2deg(_to_float_array(rows, "rpy_hat_roll"))
    pitch_deg = np.rad2deg(_to_float_array(rows, "rpy_hat_pitch"))
    
    # Limits
    tau_rp_max = args.tau_rp_max
    
    # Check stance phase
    in_stance = stance > 0.5
    if not np.any(in_stance):
        print(f"[check_tau_limit] No stance phase found in log: {os.path.basename(log_path)}")
        print(f"  Total duration: {t_s[-1] - t_s[0]:.3f} s")
        print(f"  Phases: {set(phase)}")
        sys.exit(0)
    
    tau_des_roll_stance = tau_des_w0[in_stance]
    tau_des_pitch_stance = tau_des_w1[in_stance]
    roll_stance = roll_deg[in_stance]
    pitch_stance = pitch_deg[in_stance]
    t_stance = t_s[in_stance]
    
    # Check if Tau_des hits limit
    tau_des_roll_max = np.nanmax(np.abs(tau_des_roll_stance))
    tau_des_pitch_max = np.nanmax(np.abs(tau_des_pitch_stance))
    tau_des_hits_limit = (
        np.any(np.abs(tau_des_roll_stance) >= tau_rp_max * 0.95) or
        np.any(np.abs(tau_des_pitch_stance) >= tau_rp_max * 0.95)
    )
    
    # Find where it hits limit
    roll_limit_idx = np.where(np.abs(tau_des_roll_stance) >= tau_rp_max * 0.95)[0]
    pitch_limit_idx = np.where(np.abs(tau_des_pitch_stance) >= tau_rp_max * 0.95)[0]
    
    print(f"\n=== Tau_des Limit Check ===")
    print(f"Log: {os.path.basename(log_path)}")
    print(f"stance_tau_rp_max: ±{tau_rp_max} Nm")
    print(f"\nStance phase stats:")
    print(f"  Duration: {t_stance[-1] - t_stance[0]:.3f} s")
    print(f"  Tau_des_roll max: {tau_des_roll_max:.2f} Nm")
    print(f"  Tau_des_pitch max: {tau_des_pitch_max:.2f} Nm")
    print(f"  Roll error max: {np.nanmax(np.abs(roll_stance)):.2f} deg")
    print(f"  Pitch error max: {np.nanmax(np.abs(pitch_stance)):.2f} deg")
    
    if tau_des_hits_limit:
        print(f"\n⚠ WARNING: Tau_des HITS LIMIT!")
        if len(roll_limit_idx) > 0:
            print(f"  Roll hits limit at {len(roll_limit_idx)} points")
            print(f"    Max: {np.nanmax(np.abs(tau_des_roll_stance[roll_limit_idx])):.2f} Nm")
            print(f"    Time: {t_stance[roll_limit_idx[0]]:.3f} s")
            print(f"    Roll error: {roll_stance[roll_limit_idx[0]]:.2f} deg")
        if len(pitch_limit_idx) > 0:
            print(f"  Pitch hits limit at {len(pitch_limit_idx)} points")
            print(f"    Max: {np.nanmax(np.abs(tau_des_pitch_stance[pitch_limit_idx])):.2f} Nm")
            print(f"    Time: {t_stance[pitch_limit_idx[0]]:.3f} s")
            print(f"    Pitch error: {pitch_stance[pitch_limit_idx[0]]:.2f} deg")
        print(f"\n  Consider:")
        print(f"    - Increasing stance_tau_rp_max (current: {tau_rp_max})")
        print(f"    - Or reducing attitude error (increase stance_kR)")
    else:
        print(f"\n✓ Tau_des does NOT hit limit")
        print(f"  Headroom: roll={tau_rp_max - tau_des_roll_max:.2f} Nm, pitch={tau_rp_max - tau_des_pitch_max:.2f} Nm")
        if tau_des_roll_max > tau_rp_max * 0.8 or tau_des_pitch_max > tau_rp_max * 0.8:
            print(f"  ⚠ Getting close to limit (>80%)")
    
    # Overall stats
    print(f"\nOverall log stats:")
    print(f"  Total duration: {t_s[-1] - t_s[0]:.3f} s")
    print(f"  Phases: {set(phase)}")
    print(f"  Stance ratio: {np.sum(in_stance) / len(in_stance) * 100:.1f}%")


if __name__ == "__main__":
    main()


