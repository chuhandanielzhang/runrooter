import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from modee.controllers.motor_utils import MotorTableModel


@dataclass
class _Series:
    t: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    p_deg_s: np.ndarray
    q_deg_s: np.ndarray
    thrusts: np.ndarray  # (N,3) [RED,GREEN,BLUE]
    pwm_us: np.ndarray  # (N,3) [RED,GREEN,BLUE]


def _read_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty csv: {path}")

    n = len(rows)
    cols = {k: np.empty(n, dtype=float) for k in rows[0].keys()}

    for i, r in enumerate(rows):
        for k in cols.keys():
            try:
                cols[k][i] = float(r.get(k, ""))
            except Exception:
                cols[k][i] = np.nan
    return cols


def _get_first(cols: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    for n in names:
        if n in cols:
            return cols[n]
    return np.full_like(next(iter(cols.values())), np.nan, dtype=float)


def _build_series(cols: dict[str, np.ndarray], *, pwm_min: float, pwm_max: float) -> _Series:
    t = _get_first(cols, ["t_s"])

    # Prefer IMU rpy for readability (what you care about), fallback to estimated
    roll = _get_first(cols, ["imu_rpy_roll", "rpy_hat_roll"])
    pitch = _get_first(cols, ["imu_rpy_pitch", "rpy_hat_pitch"])
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)

    p = _get_first(cols, ["imu_gyro_x", "gyro_b0", "omega_b0"])
    q = _get_first(cols, ["imu_gyro_y", "gyro_b1", "omega_b1"])
    p_deg_s = np.degrees(p)
    q_deg_s = np.degrees(q)

    thrust0 = _get_first(cols, ["thrust0", "thrusts_arm0"])
    thrust1 = _get_first(cols, ["thrust1", "thrusts_arm1"])
    thrust2 = _get_first(cols, ["thrust2", "thrusts_arm2"])
    thrusts = np.vstack([thrust0, thrust1, thrust2]).T.astype(float)

    # If pwm columns exist, use them; otherwise reconstruct from thrust using MotorTableModel.
    pwm_cols = [k for k in cols.keys() if k.startswith("pwm_us")]
    if pwm_cols:
        # Attempt to map: pwm_us2->RED, pwm_us1->GREEN, pwm_us3->BLUE if present (your common wiring).
        # Otherwise, take the first 3 channels as-is.
        def _maybe(name: str) -> np.ndarray | None:
            return cols[name] if name in cols else None

        p_red = _maybe("pwm_us2")
        p_green = _maybe("pwm_us1")
        p_blue = _maybe("pwm_us3")
        if (p_red is not None) and (p_green is not None) and (p_blue is not None):
            pwm_us = np.vstack([p_red, p_green, p_blue]).T.astype(float)
        else:
            pwm_us = np.vstack([cols[pwm_cols[0]], cols[pwm_cols[1]], cols[pwm_cols[2]]]).T.astype(float)
    else:
        model = MotorTableModel.default_from_table()
        model.pwm_min_us = float(pwm_min)
        model.pwm_max_us = float(pwm_max)
        pwm_us = np.zeros_like(thrusts, dtype=float)
        for j in range(3):
            pwm_us[:, j] = model.pwm_from_thrust(thrusts[:, j])

    return _Series(
        t=t,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        p_deg_s=p_deg_s,
        q_deg_s=q_deg_s,
        thrusts=thrusts,
        pwm_us=pwm_us,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to modee_*.csv")
    ap.add_argument("--out", type=str, default=None, help="Output png path")
    ap.add_argument("--pwm-min", type=float, default=1000.0)
    ap.add_argument("--pwm-max", type=float, default=1300.0)
    ap.add_argument("--t0", type=float, default=None, help="Start time (s) for plot window")
    ap.add_argument("--t1", type=float, default=None, help="End time (s) for plot window")
    args = ap.parse_args()

    cols = _read_csv(str(args.log))
    s = _build_series(cols, pwm_min=float(args.pwm_min), pwm_max=float(args.pwm_max))

    if args.out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(args.log)), "_debug_plots", "prop_pwm_pitch_roll.png")
    else:
        out = str(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t = s.t.copy()
    mask = np.isfinite(t)
    if args.t0 is not None:
        mask &= t >= float(args.t0)
    if args.t1 is not None:
        mask &= t <= float(args.t1)
    if np.sum(mask) < 10:
        raise RuntimeError("plot window too small or invalid (need >=10 samples)")

    t = t[mask]
    roll_deg = s.roll_deg[mask]
    pitch_deg = s.pitch_deg[mask]
    p_deg_s = s.p_deg_s[mask]
    q_deg_s = s.q_deg_s[mask]
    pwm = s.pwm_us[mask, :]
    thrusts = s.thrusts[mask, :]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.2], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, roll_deg, label="roll (deg)")
    ax1.plot(t, pitch_deg, label="pitch (deg)")
    ax1.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("angle (deg)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t, p_deg_s, label="p=gyro_x (deg/s)")
    ax2.plot(t, q_deg_s, label="q=gyro_y (deg/s)")
    ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("rate (deg/s)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(t, pwm[:, 0], label="PWM RED", linewidth=1.2)
    ax3.plot(t, pwm[:, 1], label="PWM GREEN", linewidth=1.2)
    ax3.plot(t, pwm[:, 2], label="PWM BLUE", linewidth=1.2)
    ax3.set_ylabel("pwm (us)")
    ax3.set_xlabel("t (s)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.25)

    # annotate if we reconstructed PWM
    if not any(k.startswith("pwm_us") for k in cols.keys()):
        ax3.set_title("PWM reconstructed from thrust using MotorTableModel (no pwm_us* columns in log)")
    else:
        ax3.set_title("PWM from log columns")

    # include thrust stats in console
    ts = np.sum(thrusts, axis=1)
    print("=== window stats ===")
    print("t:", float(t[0]), "->", float(t[-1]), "s (N=", int(t.shape[0]), ")")
    print("thrust max (RED,GREEN,BLUE):", np.max(thrusts[:, 0]), np.max(thrusts[:, 1]), np.max(thrusts[:, 2]))
    print("thrust sum max:", float(np.max(ts)))
    print("pwm min/max:", float(np.min(pwm)), float(np.max(pwm)))

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("saved:", out)


if __name__ == "__main__":
    main()







