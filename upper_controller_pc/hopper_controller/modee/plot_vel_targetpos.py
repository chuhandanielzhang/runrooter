#!/usr/bin/env python3
"""Plot leg velocity, body velocity estimate, and world-frame targetpos from modee CSV log."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE_COLORS = {
    "FLIGHT": "#e8e8e8",
    "STANCE:COMP": "#cfe8ff",
    "STANCE:PUSH": "#ffe8cf",
}


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty log: {path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=float)

    phase = np.array([r["phase"] for r in rows], dtype=object)
    return {
        "t": col("t_s"),
        "phase": phase,
        "stance": col("stance").astype(int),
        "v_leg": np.column_stack([col("v_meas_foot_w0"), col("v_meas_foot_w1"), col("v_meas_foot_w2")]),
        "v_hat": np.column_stack([col("v_hat_w0"), col("v_hat_w1"), col("v_hat_w2")]),
        # Raibert target in WORLD (+Z down), before R_wb^T -> body rotation.
        "target_w": np.column_stack([col("foot_des_w0"), col("foot_des_w1"), col("foot_des_w2")]),
    }


def _shade_phases(ax, t: np.ndarray, phase: np.ndarray) -> None:
    if len(t) < 2:
        return
    dt_med = float(np.median(np.diff(t)))
    i = 0
    while i < len(t):
        ph = str(phase[i])
        j = i + 1
        while j < len(t) and str(phase[j]) == ph:
            j += 1
        t0 = float(t[i]) - 0.5 * dt_med
        t1 = float(t[j - 1]) + 0.5 * dt_med
        ax.axvspan(t0, t1, color=PHASE_COLORS.get(ph, "#f5f5f5"), alpha=0.55, lw=0)
        i = j


def _trim_leading_idle(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop leading FLIGHT idle before the first stance touchdown."""
    stance = data["stance"]
    i0 = int(np.argmax(stance == 1)) if np.any(stance == 1) else 0
    if i0 <= 0:
        return data
    out = {k: (v[i0:] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    return out


def plot_log(data: dict[str, np.ndarray], out: Path) -> None:
    data = _trim_leading_idle(data)
    t = data["t"]
    t0 = float(t[0])
    t_rel = t - t0

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    labels = ("x", "y")
    for i, ax in enumerate(axes[:2]):
        _shade_phases(ax, t_rel, data["phase"])
        ax.plot(t_rel, data["v_leg"][:, i], color="#1f77b4", lw=1.0, label="v_leg (leg kin.)")
        ax.plot(t_rel, data["v_hat"][:, i], color="#d62728", lw=1.0, ls="--", label="v_hat (body est.)")
        ax.set_ylabel(f"v_{labels[i]} [m/s]")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.set_title(
                "XY: leg velocity vs body estimate; targetpos in WORLD (no quaternion R_wb^T)"
            )

    for i, ax in enumerate(axes[2:]):
        _shade_phases(ax, t_rel, data["phase"])
        y = data["target_w"][:, i]
        ax.plot(t_rel, y, color="#2ca02c", lw=1.0, label="targetpos_w (Raibert, world)")
        ax.set_ylabel(f"target_{labels[i]} [m]")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.text(
                0.01,
                0.95,
                "FLIGHT only (NaN in stance)",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="#555",
            )

    axes[-1].set_xlabel(f"time [s]  (from first stance, t0={t0:.2f})")

    # phase legend patches
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, edgecolor="none", alpha=0.55, label=k) for k, c in PHASE_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs" / "modee_latest.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs" / "vel_targetpos_plot.png",
    )
    args = ap.parse_args()
    plot_log(_load_csv(args.csv), args.out)


if __name__ == "__main__":
    main()
