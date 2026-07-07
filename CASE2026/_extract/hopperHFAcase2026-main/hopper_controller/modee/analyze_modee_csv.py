#!/usr/bin/env python3

"""
Analyze ModeE Python-side CSV logs produced by hopper_controller/modee/lcm_controller.py.

Default behavior:
  - Find the newest `modee_*.csv` under ~/hopper_logs/modee_csv (or MODEE_LOG_DIR)
  - Print a compact, high-signal summary useful for sim2real debugging.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _default_log_dir() -> str:
    return os.path.expanduser(os.environ.get("MODEE_LOG_DIR", "~/hopper_logs/modee_csv"))


def _find_latest_csv(log_dir: str) -> str | None:
    if not os.path.isdir(log_dir):
        return None
    # Prefer a fixed latest log name when available (overwrite mode).
    log_name = os.environ.get("MODEE_LOG_NAME", "modee_latest.csv")
    latest_path = os.path.join(log_dir, log_name)
    if os.path.isfile(latest_path):
        return latest_path
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
    out = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        v = r.get(key, "")
        try:
            out[i] = float(v)
        except Exception:
            out[i] = float(default)
    return out


def _to_int_array(rows: List[Dict[str, str]], key: str, *, default: int = 0) -> np.ndarray:
    out = np.empty(len(rows), dtype=int)
    for i, r in enumerate(rows):
        v = r.get(key, "")
        try:
            out[i] = int(float(v))
        except Exception:
            out[i] = int(default)
    return out


def _to_str_array(rows: List[Dict[str, str]], key: str, *, default: str = "") -> List[str]:
    out: List[str] = []
    for r in rows:
        out.append(str(r.get(key, default)))
    return out


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": float(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _print_stats(label: str, x: np.ndarray, *, scale: float = 1.0, unit: str = "") -> None:
    s = _stats(x)
    if s["n"] <= 0.0:
        print(f"- {label}: n=0")
        return
    print(
        f"- {label}: mean={s['mean']*scale:+.4f}{unit}  std={s['std']*scale:.4f}{unit}  "
        f"min={s['min']*scale:+.4f}{unit}  max={s['max']*scale:+.4f}{unit}"
    )


def _phase_mask(phases: List[str], want: str) -> np.ndarray:
    return np.array([p == want for p in phases], dtype=bool)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze ModeE CSV logs (Python-side).")
    ap.add_argument("--log", type=str, default=None, help="Path to a specific modee_*.csv file.")
    ap.add_argument("--log-dir", type=str, default=None, help="Directory to search for newest log. Default: MODEE_LOG_DIR or ~/hopper_logs/modee_csv")
    args = ap.parse_args()

    log_dir = _default_log_dir() if args.log_dir is None else os.path.expanduser(str(args.log_dir))
    log_path = args.log
    if log_path is None:
        log_path = _find_latest_csv(log_dir)
        if log_path is None:
            print(f"[analyze_modee_csv] No modee_*.csv found.")
            print(f"  searched dir: {log_dir}")
            print("  tip: run `python3 hopper_controller/run_modee.py`, then press gamepad Y once (should print LOG START).")
            sys.exit(2)
    log_path = os.path.expanduser(str(log_path))
    if not os.path.isfile(log_path):
        print(f"[analyze_modee_csv] Log not found: {log_path}")
        sys.exit(2)

    # Load CSV
    with open(log_path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        rows: List[Dict[str, str]] = list(reader)

    if len(rows) == 0:
        print(f"[analyze_modee_csv] Empty log: {log_path}")
        sys.exit(2)

    phases = _to_str_array(rows, "phase", default="")
    t_s = _to_float_array(rows, "t_s")
    wall = _to_float_array(rows, "wall_time_s")

    # Duration / rate
    dur_wall = float(wall[-1] - wall[0]) if np.isfinite(wall[0]) and np.isfinite(wall[-1]) else float("nan")
    dur_t = float(t_s[-1] - t_s[0]) if np.isfinite(t_s[0]) and np.isfinite(t_s[-1]) else float("nan")

    dt_wall = np.diff(wall)
    dt_t = np.diff(t_s)

    print("=" * 70)
    print("[ModeE CSV] Latest log analysis")
    print("=" * 70)
    print(f"- file: {log_path}")
    print(f"- rows: {len(rows)}")
    if np.isfinite(dur_wall):
        print(f"- duration (wall): {dur_wall:.3f} s  (~{len(rows)/max(1e-9, dur_wall):.1f} Hz)")
    if np.isfinite(dur_t):
        print(f"- duration (t_s):  {dur_t:.3f} s  (controller time)")
    if dt_wall.size:
        _print_stats("dt_wall", dt_wall, scale=1000.0, unit="ms")
    if dt_t.size:
        _print_stats("dt_t", dt_t, scale=1000.0, unit="ms")

    # Event counts
    touchdown = _to_int_array(rows, "touchdown")
    liftoff = _to_int_array(rows, "liftoff")
    apex = _to_int_array(rows, "apex")

    print("\n### Events")
    print(f"- touchdown count: {int(np.sum(touchdown != 0))}")
    print(f"- liftoff count:   {int(np.sum(liftoff != 0))}")
    print(f"- apex count:      {int(np.sum(apex != 0))}")

    # Phase summary
    unique_phases = sorted(set([p for p in phases if p]))
    print("\n### Phase summary")
    if not unique_phases:
        print("- phase column missing/empty")
    else:
        for ph in unique_phases:
            m = _phase_mask(phases, ph)
            frac = float(np.mean(m)) if m.size else 0.0
            print(f"- {ph:11s}: {int(np.sum(m))} rows ({100.0*frac:.1f}%)")

    # Key scalars
    q_shift = _to_float_array(rows, "q_shift_m")
    leg_len = _to_float_array(rows, "leg_len_m")
    qd_shift = _to_float_array(rows, "qd_shift_mps")
    comp_m = _to_float_array(rows, "comp_m")
    comp_tgt = _to_float_array(rows, "comp_tgt_m")
    J_inv_cond = _to_float_array(rows, "J_inv_cond")
    A_tau_f_cond = _to_float_array(rows, "A_tau_f_cond")

    tau0 = _to_float_array(rows, "tau0")
    tau1 = _to_float_array(rows, "tau1")
    tau2 = _to_float_array(rows, "tau2")

    fz = _to_float_array(rows, "f_tau_delta2")
    v_hat_x = _to_float_array(rows, "v_hat_w0")
    v_hat_y = _to_float_array(rows, "v_hat_w1")
    v_hat_z = _to_float_array(rows, "v_hat_w2")
    v_meas_x = _to_float_array(rows, "v_meas_foot_w0")
    v_meas_y = _to_float_array(rows, "v_meas_foot_w1")
    v_meas_z = _to_float_array(rows, "v_meas_foot_w2")
    stance = _to_int_array(rows, "stance")

    print("\n### Kinematics")
    _print_stats("leg_len_m", leg_len, unit="m")
    _print_stats("q_shift_m", q_shift, unit="m")
    _print_stats("qd_shift_mps", qd_shift, unit="m/s")
    _print_stats("J_inv_cond", J_inv_cond, unit="")
    _print_stats("A_tau_f_cond", A_tau_f_cond, unit="")

    print("\n### Compression")
    _print_stats("comp_m", comp_m, unit="m")
    _print_stats("comp_tgt_m", comp_tgt, unit="m")

    print("\n### Forces / torques")
    _print_stats("f_tau_delta_z", fz, unit="N")
    _print_stats("tau0", tau0, unit="Nm")
    _print_stats("tau1", tau1, unit="Nm")
    _print_stats("tau2", tau2, unit="Nm")

    m_stance = stance > 0.5
    if np.any(m_stance):
        dvx = v_hat_x[m_stance] - v_meas_x[m_stance]
        dvy = v_hat_y[m_stance] - v_meas_y[m_stance]
        dvz = v_hat_z[m_stance] - v_meas_z[m_stance]
        dvn = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
        print("\n### Velocity estimate (stance)")
        _print_stats("v_hat_w0", v_hat_x[m_stance], unit="m/s")
        _print_stats("v_hat_w1", v_hat_y[m_stance], unit="m/s")
        _print_stats("v_hat_w2", v_hat_z[m_stance], unit="m/s")
        _print_stats("v_meas_foot_w0", v_meas_x[m_stance], unit="m/s")
        _print_stats("v_meas_foot_w1", v_meas_y[m_stance], unit="m/s")
        _print_stats("v_meas_foot_w2", v_meas_z[m_stance], unit="m/s")
        _print_stats("v_hat - v_meas (x)", dvx, unit="m/s")
        _print_stats("v_hat - v_meas (y)", dvy, unit="m/s")
        _print_stats("v_hat - v_meas (z)", dvz, unit="m/s")
        _print_stats("||v_hat - v_meas||", dvn, unit="m/s")

    # Flight S2S target stats
    s2s_active = _to_int_array(rows, "s2s_active")
    tgt_bx = _to_float_array(rows, "foot_des_b0")
    tgt_by = _to_float_array(rows, "foot_des_b1")
    tgt_bz = _to_float_array(rows, "foot_des_b2")
    tgt_norm = np.sqrt(tgt_bx * tgt_bx + tgt_by * tgt_by + tgt_bz * tgt_bz)

    m_flight = _phase_mask(phases, "FLIGHT")
    if np.any(m_flight):
        print("\n### Flight target (S2S)")
        print(f"- s2s_active rows (in flight): {int(np.sum((s2s_active != 0) & m_flight))} / {int(np.sum(m_flight))}")
        _print_stats("tgt_bx", tgt_bx[m_flight], unit="m")
        _print_stats("tgt_by", tgt_by[m_flight], unit="m")
        _print_stats("tgt_bz", tgt_bz[m_flight], unit="m")
        _print_stats("|tgt_b|", tgt_norm[m_flight], unit="m")

    print("\nDone.")


if __name__ == "__main__":
    main()




