#!/usr/bin/env python3
"""
Generate the two headline figures:

  1) out/timeseries.png : one PAC-NMPC episode — base height, roll/pitch, and the
     online PAC bound (the per-solve certificate).
  2) out/bound_check.png: PAC certificate check — the online bound on the
     attitude-constraint violation probability vs. the EMPIRICAL violation
     frequency measured across Monte-Carlo episodes on random plants.
     The certificate holds if the empirical curve stays below the bound.

Usage:  python3 plot_results.py [--trials 8]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_demo import run_episode
from hopper_pac.sim import PlantUncertainty
from hopper_pac.srb_rollout import RolloutCostConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--duration", type=float, default=8.0)
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    # ---------------- Fig 1: single-episode timeseries ----------------
    r = run_episode(stochastic=True, plant_unc=PlantUncertainty(), duration_s=args.duration,
                    seed=1, verbose=True)
    log = r["log"]
    t = np.asarray(log["t"])
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, log["z"], lw=1.2)
    axes[0].set_ylabel("base z [m]")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, np.rad2deg(log["roll"]), label="roll", lw=1.0)
    axes[1].plot(t, np.rad2deg(log["pitch"]), label="pitch", lw=1.0)
    lim = np.rad2deg(RolloutCostConfig().att_limit_rad)
    axes[1].axhline(+lim, color="r", ls="--", lw=0.8)
    axes[1].axhline(-lim, color="r", ls="--", lw=0.8, label="constraint")
    axes[1].set_ylabel("attitude [deg]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[2].plot(t, log["bound"], color="purple", lw=1.0, label="PAC upper bound b(xi)")
    axes[2].plot(t, log["viol_hat"], color="gray", lw=0.8, label="elite empirical viol.")
    axes[2].set_ylabel("certificate")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.suptitle("PAC-NMPC hopping — MuJoCo (runtime-identical interface)")
    fig.tight_layout()
    fig.savefig("out/timeseries.png", dpi=140)
    print("fig -> out/timeseries.png")

    # ---------------- Fig 2: certificate vs empirical violations ----------------
    rng = np.random.default_rng(7)
    viol_frac = []       # per-episode: fraction of solves whose horizon saw a violation... (proxy)
    bound_mean = []      # per-episode mean online bound on cost+lambda*viol
    viol_hat_mean = []
    emp_violated = []    # per-episode: did the REAL plant violate the attitude envelope?
    lim_rad = RolloutCostConfig().att_limit_rad
    for i in range(args.trials):
        unc = PlantUncertainty(
            friction_mu=float(np.clip(rng.normal(0.3, 0.12), 0.10, 0.9)),
            ground_z_offset_m=float(rng.normal(0.0, 0.03)),
            payload_mass_kg=float(np.clip(rng.normal(0.0, 0.3), 0.0, 1.0)),
            thrust_scale=np.clip(rng.normal(1.0, 0.10, 6), 0.6, 1.3),
            gyro_noise_std=0.02,
        )
        r = run_episode(stochastic=True, plant_unc=unc, duration_s=args.duration,
                        seed=50 + i, verbose=True)
        log = r["log"]
        roll = np.asarray(log["roll"]); pitch = np.asarray(log["pitch"])
        emp = float(np.mean((np.abs(roll) > lim_rad) | (np.abs(pitch) > lim_rad)))
        emp_violated.append(emp)
        viol_hat_mean.append(float(np.nanmean(log["viol_hat"])))
        bound_mean.append(float(np.nanmean(log["bound"])))

    idx = np.arange(len(emp_violated))
    fig2, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.35
    ax.bar(idx - w / 2, viol_hat_mean, w, label="predicted viol. prob (elite mean)", color="tab:purple", alpha=0.8)
    ax.bar(idx + w / 2, emp_violated, w, label="empirical viol. freq (true plant)", color="tab:orange", alpha=0.8)
    ax.set_xlabel("Monte-Carlo episode (random true plant)")
    ax.set_ylabel("attitude-constraint violation")
    ax.set_title("PAC certificate check: predicted vs. empirical violations")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig("out/bound_check.png", dpi=140)
    print("fig -> out/bound_check.png")
    print(f"episodes with empirical violation: {int(np.sum(np.asarray(emp_violated) > 0))}/{len(emp_violated)}")
    print(f"mean predicted violation prob: {np.mean(viol_hat_mean):.3f}")
    print(f"mean empirical violation freq: {np.mean(emp_violated):.4f}")


if __name__ == "__main__":
    main()
