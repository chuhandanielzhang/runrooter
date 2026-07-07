#!/usr/bin/env python3
"""Numerical validation of the Mode 3 HLIP S2S foot-placement law (core.py).

Simulates the hybrid hopper template the law is derived from -- LIP stance
(integrated numerically, NOT the closed form, so integration error is real)
plus ballistic flight -- and compares, hop by hop:

  1. Raibert fixed gain  u = kv * v            (mode 2, kv = 0.15)
  2. Mode 3 S2S law      u = [(cosh-beta)v - (1-beta)v_des]/(lam*sinh)

under (a) nominal model, (b) +/-20% error in the z0 used by the law, and
(c) white noise on the velocity estimate. Prints hops-to-converge and the
steady-state error for each case.

Run:  python3 scripts/test_s2s_mode3.py
"""

import numpy as np

G = 9.81
Z0_TRUE = 0.35      # true pivot height (m)
TS_TRUE = 0.25      # true stance duration (s)
DT = 0.0005


def stance_map(v_td: float, u: float, z0: float, ts: float) -> float:
    """Integrate the LIP stance numerically: x'' = (g/z0) x, x(0) = -u, v(0) = v_td.
    Returns liftoff velocity."""
    x, v = -u, v_td
    n = int(round(ts / DT))
    for _ in range(n):
        a = (G / z0) * x
        v += a * DT
        x += v * DT
    return v


def run(law, v0: float, v_des: float, hops: int, noise_std: float = 0.0,
        seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = v0
    hist = [v]
    for _ in range(hops):
        v_meas = v + rng.normal(0.0, noise_std)
        u = law(v_meas, v_des)
        v = stance_map(v, u, Z0_TRUE, TS_TRUE)  # flight conserves XY velocity
        hist.append(v)
    return np.array(hist)


def raibert_law(kv: float):
    return lambda v, v_des: kv * v  # mode 2 (flight_kv=0.15, kr=0)


def s2s_law(beta: float, z0_assumed: float, ts_assumed: float):
    lam = np.sqrt(G / z0_assumed)
    lt = lam * ts_assumed
    den = lam * np.sinh(lt)
    return lambda v, v_des: ((np.cosh(lt) - beta) * v - (1.0 - beta) * v_des) / den


def hops_to_converge(hist: np.ndarray, v_des: float, tol: float = 0.05) -> int:
    err = np.abs(hist - v_des)
    for k in range(len(err)):
        if np.all(err[k:] < tol):
            return k
    return -1


def report(name: str, hist: np.ndarray, v_des: float) -> None:
    k = hops_to_converge(hist, v_des)
    ss = float(np.mean(np.abs(hist[-5:] - v_des)))
    kstr = f"{k:3d}" if k >= 0 else "  never"
    print(f"  {name:38s} hops-to-|err|<0.05: {kstr}   steady-state |err|: {ss:.4f}"
          f"   v: {hist[0]:+.2f} -> {hist[-1]:+.2f}")


def main() -> None:
    v0, hops = 0.8, 15

    print("=== (a) nominal model, hop in place (v_des = 0), v0 = 0.8 m/s ===")
    report("Raibert kv=0.15", run(raibert_law(0.15), v0, 0.0, hops), 0.0)
    report("S2S deadbeat (beta=0)", run(s2s_law(0.0, Z0_TRUE, TS_TRUE), v0, 0.0, hops), 0.0)
    report("S2S beta=0.2", run(s2s_law(0.2, Z0_TRUE, TS_TRUE), v0, 0.0, hops), 0.0)

    print("=== (b) law uses WRONG z0 (+/-20%), v_des = 0 ===")
    for z0a in (0.8 * Z0_TRUE, 1.2 * Z0_TRUE):
        report(f"S2S beta=0.2, z0 assumed {z0a:.2f}",
               run(s2s_law(0.2, z0a, TS_TRUE), v0, 0.0, hops), 0.0)

    print("=== (c) velocity-estimate noise sigma = 0.1 m/s, v_des = 0 ===")
    report("Raibert kv=0.15", run(raibert_law(0.15), v0, 0.0, hops, 0.1), 0.0)
    report("S2S deadbeat (beta=0)", run(s2s_law(0.0, Z0_TRUE, TS_TRUE), v0, 0.0, hops, 0.1), 0.0)
    report("S2S beta=0.2", run(s2s_law(0.2, Z0_TRUE, TS_TRUE), v0, 0.0, hops, 0.1), 0.0)
    report("S2S beta=0.4", run(s2s_law(0.4, Z0_TRUE, TS_TRUE), v0, 0.0, hops, 0.1), 0.0)

    print("=== (d) velocity TRACKING: v_des = +0.5 m/s from rest ===")
    report("Raibert kv=0.15 (kr=0: cannot track)", run(raibert_law(0.15), 0.0, 0.5, hops), 0.5)
    report("S2S beta=0.2", run(s2s_law(0.2, Z0_TRUE, TS_TRUE), 0.0, 0.5, hops), 0.5)


if __name__ == "__main__":
    main()
