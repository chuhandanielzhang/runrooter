#!/usr/bin/env python3
"""Replay foot-tip apex clearance and hybrid-event timing from ModeE CSV.

This is diagnostic only.  It compares:
  1. the legacy constant-rho flight-time estimate in z_apex_actual_m,
  2. direct p_hat + FK foot clearance at the observed body apex,
  3. endpoint-constrained integration of allocated prop thrust, and
  4. endpoint-constrained integration of measured IMU specific force.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from modee.core import _endpoint_constrained_apex


def _rotation_wb(row: pd.Series) -> np.ndarray:
    w, x, y, z = (
        float(row["q_hat_w"]),
        float(row["q_hat_x"]),
        float(row["q_hat_y"]),
        float(row["q_hat_z"]),
    )
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _flight_arrays(seg: pd.DataFrame, mass: float, gravity: float):
    t = seg["t_s"].to_numpy(dtype=float)
    a_imu_up = []
    a_cmd_up = []
    r_foot_down = []
    for _, row in seg.iterrows():
        R = _rotation_wb(row)
        acc_b = row[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].to_numpy(dtype=float)
        foot_b = row[["foot_b0", "foot_b1", "foot_b2"]].to_numpy(dtype=float)
        # WORLD +Z is down.  IMU is specific force; inertial up acceleration
        # is -(R*a_specific)_z - g.
        a_imu_up.append(float(-(R @ acc_b)[2] - gravity))
        tilt = float(max(0.0, min(1.0, R[2, 2])))
        a_cmd_up.append(float(row["thrust_sum"]) * tilt / mass - gravity)
        r_foot_down.append(float((R @ foot_b)[2]))
    return (
        t,
        np.asarray(a_imu_up, dtype=float),
        np.asarray(a_cmd_up, dtype=float),
        np.asarray(r_foot_down, dtype=float),
    )


def analyze(path: Path, mass: float, gravity: float, target: float) -> None:
    df = pd.read_csv(path)
    stance = df["stance"].to_numpy(dtype=int)
    td_idx = np.where(np.diff(stance) == 1)[0] + 1
    lo_idx = np.where(np.diff(stance) == -1)[0] + 1
    print(f"log={path}  target_foot_clearance={100*target:.1f} cm")
    print(
        "hop  Tphase  qLO/qTD    legacy  direct  cmd-int  imu-int  "
        "Tmin(target)  unload/impact"
    )
    for hop, lo in enumerate(lo_idx, start=1):
        next_td = td_idx[td_idx > lo]
        if next_td.size == 0:
            continue
        td = int(next_td[0])
        seg = df.iloc[lo : td + 1]
        t, a_imu, a_cmd, rz = _flight_arrays(seg, mass, gravity)
        h_cmd = _endpoint_constrained_apex(t, a_cmd, rz)[2]
        h_imu = _endpoint_constrained_apex(t, a_imu, rz)[2]

        # Direct estimator: body apex from p_hat, plus FK leg-relative change.
        i_apex = int(np.argmin(seg["p_hat_w2"].to_numpy(dtype=float)))
        h_com = float(seg["p_hat_w2"].iloc[0] - seg["p_hat_w2"].iloc[i_apex])
        h_leg = float(rz[0] - rz[i_apex])
        h_direct = max(0.0, h_com + h_leg)
        legacy = float(df["z_apex_actual_m"].iloc[td])
        dz_down = float(rz[0] - rz[-1])  # landing COM is usually lower
        t_min = math.sqrt(2.0 * target / gravity) + math.sqrt(
            2.0 * max(0.0, target + dz_down) / gravity
        )
        unload = (
            float(df["lo_unload_delay_s"].iloc[td])
            if "lo_unload_delay_s" in df else float("nan")
        )
        impact = (
            float(df["td_impact_delay_s"].iloc[td])
            if "td_impact_delay_s" in df else float("nan")
        )
        print(
            f"{hop:3d}  {t[-1]-t[0]:6.3f}  "
            f"{df['q_shift_m'].iloc[lo]:+.3f}/{df['q_shift_m'].iloc[td]:+.3f}  "
            f"{100*legacy:6.2f}  {100*h_direct:6.2f}  "
            f"{100*h_cmd:7.2f}  {100*h_imu:7.2f}  "
            f"{t_min:12.3f}  {unload:6.3f}/{impact:6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--mass", type=float, default=5.61)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--target", type=float, default=0.05)
    args = parser.parse_args()
    analyze(args.csv, args.mass, args.gravity, args.target)


if __name__ == "__main__":
    main()
