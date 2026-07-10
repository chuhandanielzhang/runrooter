#!/usr/bin/env python3
"""Fit bench_prop_id.py data -> k_thrust / tau_m / attitude FRF.

  ladder  Input: a small hand-made CSV `pwm,gram,volt` (one row per ladder
          level, scale reading in grams TOTAL weight, volt optional).
          Fits T_total = 3*k*(pwm-1000)^2 -> prints per-arm k_thrust and,
          if voltage present, k(V) linear fit for battery compensation.
          Example row: 1300, -412, 23.9   (weight CHANGE in grams; negative
          = lighter = upward thrust. Use --absolute if you logged absolute.)
  step    Input: the bench_step CSV. Detects pwm3 step edges, overlays the
          aligned gyro-x responses and fits a first-order rise -> tau_m
          (includes ESC+prop lag AND gimbal dynamics; use the INITIAL slope
          for control effectiveness).
  chirp   Input: the bench_chirp CSV. Empirical FRF (Welch CSD) from
          differential pwm -> gyro-x; prints -3dB bandwidth and -90deg phase
          crossover; saves bode plot.

Usage:
  python3 modee/tools/bench_prop_fit.py ladder scale_readings.csv
  python3 modee/tools/bench_prop_fit.py step  logs/bench_step_XXXX.csv
  python3 modee/tools/bench_prop_fit.py chirp logs/bench_chirp_XXXX.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

G = 9.81


def fit_ladder(path: str, absolute: bool) -> None:
    rows = np.genfromtxt(path, delimiter=",", names=("pwm", "gram", "volt"),
                         skip_header=0, invalid_raise=False)
    rows = np.atleast_1d(rows)
    pwm = np.asarray(rows["pwm"], dtype=float)
    gram = np.asarray(rows["gram"], dtype=float)
    volt = np.asarray(rows["volt"], dtype=float)
    if absolute:
        gram = gram - gram[np.argmin(pwm)]
    thrust_n = -gram * 1e-3 * G          # lighter (negative grams) = +thrust up
    d = np.maximum(0.0, pwm - 1000.0)
    # T_total = 3 k d^2  ->  k = sum(T d^2)/ (3 sum d^4)
    k = float(np.sum(thrust_n * d * d) / max(1e-9, 3.0 * np.sum(d ** 4)))
    print(f"levels: {len(pwm)}   max total thrust: {np.max(thrust_n):.2f} N")
    print(f"k_thrust (per arm) = {k:.4e}  N/us^2   [cfg prop_k_thrust]")
    resid = thrust_n - 3.0 * k * d * d
    print(f"fit residual rms   = {float(np.sqrt(np.mean(resid ** 2))):.3f} N")
    if np.all(np.isfinite(volt)) and np.ptp(volt) > 0.3:
        # per-level instantaneous k, linear in V
        ki = thrust_n / np.maximum(1e-9, 3.0 * d * d)
        m_ = np.isfinite(ki) & (d > 30.0)
        A = np.vstack([volt[m_], np.ones(m_.sum())]).T
        c = np.linalg.lstsq(A, ki[m_], rcond=None)[0]
        print(f"voltage comp: k(V) ~= {c[0]:.3e}*V + {c[1]:.3e}"
              f"  (k drops {100 * abs(c[0]) * 1.0 / max(1e-9, k):.1f}%/V)")
    out = os.path.splitext(path)[0] + "_fit.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d, thrust_n, "o", label="measured")
    dd = np.linspace(0, d.max(), 100)
    ax.plot(dd, 3.0 * k * dd * dd, "-", label=f"3k d^2, k={k:.3e}")
    ax.set_xlabel("pwm - 1000 [us]"); ax.set_ylabel("total thrust [N]")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"plot: {out}")


def _load_bench(path: str):
    d = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    t = np.asarray(d["t_s"], dtype=float)
    return d, t


def fit_step(path: str) -> None:
    d, t = _load_bench(path)
    p3 = np.asarray(d["pwm3"], dtype=float)
    gx = np.asarray(d["gx"], dtype=float)
    dp = np.diff(p3)
    edges = np.flatnonzero(np.abs(dp) > 30.0)
    # keep first edge of each burst
    edges = edges[np.insert(np.diff(edges) > 10, 0, True)]
    print(f"step edges found: {len(edges)}")
    taus, slopes = [], []
    fig, ax = plt.subplots(figsize=(7, 4))
    for e in edges:
        sgn = np.sign(dp[e])
        i0, i1 = e - 20, e + 200
        if i0 < 0 or i1 >= len(t):
            continue
        tt = t[i0:i1] - t[e]
        yy = sgn * (gx[i0:i1] - np.mean(gx[i0:e]))
        ax.plot(tt, yy, alpha=0.4)
        # 63% of the peak rate reached -> tau (first-order approx on the rise)
        pk = np.max(yy[20:120]) if np.max(yy[20:120]) > 1e-6 else None
        if pk:
            k63 = np.flatnonzero(yy[20:] >= 0.63 * pk)
            if len(k63):
                taus.append(float(tt[20 + k63[0]]))
        # initial angular accel (control effectiveness): slope over first 40 ms
        m_ = (tt >= 0.005) & (tt <= 0.045)
        if m_.sum() >= 3:
            slopes.append(float(np.polyfit(tt[m_], yy[m_], 1)[0]))
    if taus:
        print(f"tau_m (63% rise, incl. gimbal): median {np.median(taus)*1e3:.0f} ms"
              f"  (n={len(taus)}, spread {np.std(taus)*1e3:.0f} ms)")
    if slopes:
        print(f"initial ang. accel per doublet: median {np.median(slopes):.2f} rad/s^2"
              "  -> J = tau_step / alpha  (tau_step from the thrust map x arm length)")
    ax.set_xlabel("t since edge [s]"); ax.set_ylabel("gyro-x (aligned) [rad/s]")
    ax.grid(alpha=0.3)
    out = os.path.splitext(path)[0] + "_step.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"plot: {out}")


def fit_chirp(path: str) -> None:
    from scipy import signal as sp
    d, t = _load_bench(path)
    m_ = np.asarray(d["phase"], dtype=str) if d.dtype.names and "phase" in d.dtype.names else None
    p3 = np.asarray(d["pwm3"], dtype=float)
    p1 = np.asarray(d["pwm1"], dtype=float)
    gx = np.asarray(d["gx"], dtype=float)
    u = p3 - p1                      # differential command (us)
    fs = 1.0 / np.median(np.diff(t))
    f, Puy = sp.csd(u, gx, fs=fs, nperseg=1024)
    _, Puu = sp.welch(u, fs=fs, nperseg=1024)
    H = Puy / np.maximum(1e-12, Puu)
    mag = 20 * np.log10(np.maximum(1e-12, np.abs(H)))
    ph = np.unwrap(np.angle(H)) * 180 / np.pi
    band = (f > 0.3) & (f < 20.0)
    fb, mb = f[band], mag[band]
    # Reference = median of the low-frequency plateau (single noisy Welch
    # bins must not set it); -3 dB point requires 3 consecutive bins below.
    lo = (fb >= 0.4) & (fb <= 1.2)
    ref = float(np.median(mb[lo])) if lo.any() else float(mb[0])
    below = mb < ref - 3.0
    f3 = None
    for i in range(len(below) - 2):
        if below[i] and below[i + 1] and below[i + 2]:
            f3 = fb[i]
            break
    if f3 is not None:
        print(f"-3 dB bandwidth ~ {f3:.1f} Hz")
    else:
        print("-3 dB point not reached in band")
    try:
        f90 = f[band][np.flatnonzero(ph[band] < ph[band][0] - 90.0)[0]]
        print(f"-90 deg phase crossover ~ {f90:.1f} Hz  (usable ctrl bandwidth ~ 1/3 of this)")
    except IndexError:
        pass
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axs[0].semilogx(f[band], mag[band]); axs[0].set_ylabel("|gyro/pwm| [dB]"); axs[0].grid(alpha=0.3)
    axs[1].semilogx(f[band], ph[band]); axs[1].set_ylabel("phase [deg]")
    axs[1].set_xlabel("f [Hz]"); axs[1].grid(alpha=0.3)
    out = os.path.splitext(path)[0] + "_bode.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"fs={fs:.0f} Hz   plot: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ladder", "step", "chirp"])
    ap.add_argument("csv")
    ap.add_argument("--absolute", action="store_true",
                    help="ladder: scale column is absolute grams (not change)")
    a = ap.parse_args()
    if a.mode == "ladder":
        fit_ladder(a.csv, a.absolute)
    elif a.mode == "step":
        fit_step(a.csv)
    else:
        fit_chirp(a.csv)


if __name__ == "__main__":
    main()
