#!/usr/bin/env python3
"""Simulate flight-phase prop PWM vs IMU roll/pitch tilt (current ModeE params).

Chain (flight, props ON, omega=0), matching ModeECore:
  IMU rpy -> SO(3) e_R -> Tau_des = -kR*e_R (PD)
  -> closed-form tri-rotor lstsq allocation -> PWM sqrt law

Usage:
  cd upper_controller_pc/hopper_controller
  python3 modee/tools/sim_pwm_vs_rpy.py [--out /tmp/pwm_vs_rpy.png]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow imports from hopper_controller/
HERE = os.path.dirname(os.path.abspath(__file__))
CTRL = os.path.abspath(os.path.join(HERE, "..", ".."))
if CTRL not in sys.path:
    sys.path.insert(0, CTRL)

from modee.core import ModeEConfig, ModeECore, _Rz, _vee_so3  # noqa: E402

DEG = math.pi / 180.0


def _Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _R_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->world, matches core._R_to_rpy_xyz (intrinsic XYZ)."""
    return _Rx(roll) @ _Ry(pitch) @ _Rz(yaw)


def simulate_pwm(
    roll_deg: float,
    pitch_deg: float,
    *,
    core: ModeECore,
    kR: float | None = None,
) -> dict:
    """Return PWM + intermediates for one attitude (flight, gyro=0)."""
    cfg = core.cfg
    roll = float(roll_deg) * DEG
    pitch = float(pitch_deg) * DEG
    yaw = 0.0
    R_wb = _R_from_rpy(roll, pitch, yaw)
    # Props push along body -Z (FRD +Z is down), so world thrust is -R[:,2].
    z_thrust_w = (-R_wb[:, 2]).copy()
    com_b = np.asarray(cfg.com_b, dtype=float).reshape(3)
    prop_r_w = (R_wb @ (core.prop_positions_b - com_b.reshape(1, 3)).T).T.copy()

    R_des = _Rz(yaw)
    E = (R_des.T @ R_wb) - (R_wb.T @ R_des)
    e_R = 0.5 * _vee_so3(E)
    e_R[2] = 0.0

    kR_use = float(cfg.flight_kR if kR is None else kR)
    kW = float(cfg.flight_kW)
    omega_b = np.zeros(3, dtype=float)

    tau_b = np.zeros(3, dtype=float)
    tau_b[0] = (-kR_use * float(e_R[0])) - (kW * float(omega_b[0]))
    tau_b[1] = (-kR_use * float(e_R[1])) - (kW * float(omega_b[1]))
    # Match core flight torque limiting: full-norm clip on body roll/pitch.
    tau_max = float(cfg.flight_tau_rp_max)
    n = float(np.linalg.norm(tau_b[:2]))
    if tau_max > 0.0 and n > tau_max and n > 1e-9:
        tau_b[:2] *= tau_max / n
    tau_w = (R_wb @ tau_b.reshape(3)).reshape(3)
    Tau_des = np.array([float(tau_w[0]), float(tau_w[1]), 0.0], dtype=float)

    mass = float(core.mass)
    g = float(core.gravity)
    thrust_sum_ref = mass * g * float(cfg.prop_base_thrust_ratio)
    thrust_sum_max = mass * g * float(cfg.thrust_total_ratio_max)

    thrusts = core._allocate_prop_thrust(
        tau_des_w=Tau_des,
        prop_r_w=prop_r_w,
        z_thrust_w=z_thrust_w,
        thrust_sum_ref=float(thrust_sum_ref),
        thrust_sum_max=float(thrust_sum_max),
        reverse_policy=str(getattr(cfg, "prop_flight_reverse", "fwd")),
    )
    pwm = core._pwm_from_arm_thrusts(thrusts)
    return {
        "pwm": pwm,
        "thrusts": thrusts,
        "e_R_deg": np.rad2deg(e_R[:2]),
        "Tau_des": Tau_des,
        "thrust_sum": float(np.sum(thrusts)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/pwm_vs_rpy.png")
    ap.add_argument("--sweep-deg", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=81)
    args = ap.parse_args()

    cfg = ModeEConfig()
    core = ModeECore(cfg)
    sweep = np.linspace(-args.sweep_deg, args.sweep_deg, args.steps)

    roll_pwm = {i: [] for i in (1, 2, 3)}
    pitch_pwm = {i: [] for i in (1, 2, 3)}
    roll_tau = []
    pitch_tau = []

    for deg in sweep:
        r = simulate_pwm(deg, 0.0, core=core)
        p = simulate_pwm(0.0, deg, core=core)
        for m in (1, 2, 3):
            roll_pwm[m].append(r["pwm"][m])
            pitch_pwm[m].append(p["pwm"][m])
        roll_tau.append(float(np.linalg.norm(r["Tau_des"][:2])))
        pitch_tau.append(float(np.linalg.norm(p["Tau_des"][:2])))

    level = simulate_pwm(0.0, 0.0, core=core)
    pwm0 = level["pwm"]

    cmp_pitch = []
    for kR in (10.0, 20.0, 40.0):
        r = simulate_pwm(0.0, 5.0, core=core, kR=kR)
        cmp_pitch.append((kR, r["pwm"][1], r["pwm"][2], r["pwm"][3]))

    rr = np.linspace(-15, 15, 31)
    pp = np.linspace(-15, 15, 31)
    heat = np.zeros((len(pp), len(rr)), dtype=float)
    for j, pd in enumerate(pp):
        for i, rd in enumerate(rr):
            s = simulate_pwm(rd, pd, core=core)
            heat[j, i] = float(np.max(s["pwm"][1:4] - pwm0[1:4]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Flight prop PWM vs IMU tilt  "
        f"(kR={cfg.flight_kR}, kW={cfg.flight_kW}, "
        f"base thrust={100*cfg.prop_base_thrust_ratio:.1f}% weight, "
        f"pwm_max={cfg.pwm_max_us:.0f}us)",
        fontsize=11,
    )

    ax = axes[0, 0]
    ax.plot(sweep, roll_pwm[1], label="M1 @ +30", color="#e74c3c")
    ax.plot(sweep, roll_pwm[2], label="M2 @ +150", color="#3498db")
    ax.plot(sweep, roll_pwm[3], label="M3 @ -90", color="#2ecc71")
    ax.axhline(pwm0[1], color="#e74c3c", ls="--", alpha=0.4)
    ax.set_xlabel("Roll (deg)  [pitch=0]")
    ax.set_ylabel("PWM (us)")
    ax.set_title("Roll sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(sweep, pitch_pwm[1], label="M1 @ +30", color="#e74c3c")
    ax.plot(sweep, pitch_pwm[2], label="M2 @ +150", color="#3498db")
    ax.plot(sweep, pitch_pwm[3], label="M3 @ -90", color="#2ecc71")
    ax.set_xlabel("Pitch (deg)  [roll=0]")
    ax.set_ylabel("PWM (us)")
    ax.set_title("Pitch sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(sweep, roll_tau, label="|Tau_des| roll sweep", color="#8e44ad")
    ax.plot(sweep, pitch_tau, label="|Tau_des| pitch sweep", color="#16a085")
    ax.set_xlabel("Tilt (deg)")
    ax.set_ylabel("Torque demand (Nm)")
    ax.set_title("Attitude torque demand (before allocation)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    im = ax.imshow(
        heat,
        origin="lower",
        extent=[rr[0], rr[-1], pp[0], pp[-1]],
        aspect="auto",
        cmap="YlOrRd",
    )
    ax.set_xlabel("Roll (deg)")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Max PWM rise above level (us)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    txt = (
        f"Level (0,0): M1={pwm0[1]:.0f}  M2={pwm0[2]:.0f}  M3={pwm0[3]:.0f} us\n"
        f"+5deg pitch: kR10 -> M1/2/3={cmp_pitch[0][1]:.0f}/{cmp_pitch[0][2]:.0f}/{cmp_pitch[0][3]:.0f} | "
        f"kR20 -> M1/2/3={cmp_pitch[1][1]:.0f}/{cmp_pitch[1][2]:.0f}/{cmp_pitch[1][3]:.0f} | "
        f"kR40 -> M1/2/3={cmp_pitch[2][1]:.0f}/{cmp_pitch[2][2]:.0f}/{cmp_pitch[2][3]:.0f} us"
    )
    fig.text(0.5, 0.01, txt, ha="center", fontsize=9, color="#444")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f">>> saved {out}")

    print(f"\n=== Anchor points (kR={cfg.flight_kR}, flight, gyro=0) ===")
    print(
        f"Level: pwm M1/2/3 = {pwm0[1]:.0f} / {pwm0[2]:.0f} / {pwm0[3]:.0f} us  "
        f"(base thrust each {level['thrusts'][0]:.3f} N)"
    )
    for tilt, name in [(5, "+5 roll"), (-5, "-5 roll"), (5, "+5 pitch"), (-5, "-5 pitch")]:
        if "roll" in name:
            s = simulate_pwm(tilt if "+" in name else -5, 0, core=core)
        else:
            s = simulate_pwm(0, tilt if "+" in name else -5, core=core)
        print(
            f"{name:10s}: pwm M1/2/3 = {s['pwm'][1]:.0f} / {s['pwm'][2]:.0f} / {s['pwm'][3]:.0f} us  "
            f"|Tau|={np.linalg.norm(s['Tau_des'][:2]):.2f} Nm  e_R={s['e_R_deg']}"
        )


if __name__ == "__main__":
    main()
