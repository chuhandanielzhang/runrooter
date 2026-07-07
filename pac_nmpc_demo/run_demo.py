#!/usr/bin/env python3
"""
Closed-loop PAC-NMPC hopping demo (isolated from the Jetson/PC runtime tree).

Plant     : MuJoCo hopper_serial.xml, same interface/coordinates as the Jetson stack
Controller: PAC-NMPC (50 Hz, Moore-style sampling SNMPC with PAC bound)
            + 500 Hz inner loop (leg GRF mapping / Raibert placement / prop PD)

Usage:
  python3 run_demo.py                       # single stochastic run, nominal plant
  python3 run_demo.py --deterministic      # baseline: NMPC without uncertainty sampling
  python3 run_demo.py --hard               # plant with adverse true parameters
  python3 run_demo.py --trials 10          # Monte-Carlo over random plants -> success rate
  python3 run_demo.py --video out.mp4      # also record video (offscreen render)
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hopper_pac.sim import HopperMujocoPlant, PlantUncertainty
from hopper_pac.controller import PACHoppingController
from hopper_pac.pac_nmpc import PACNMPC, PACNMPCConfig
from hopper_pac.conventions import CONTROL_DT


def run_episode(
    *,
    stochastic: bool,
    plant_unc: PlantUncertainty,
    duration_s: float = 8.0,
    seed: int = 0,
    video: str | None = None,
    verbose: bool = True,
    push_n: float = 0.0,           # if >0: random-direction horizontal pushes
    push_every_s: float = 2.5,
    push_dur_s: float = 0.10,
) -> dict:
    plant = HopperMujocoPlant(uncertainty=plant_unc, seed=seed, init_base_z=0.62)
    ctrl = PACHoppingController(
        nmpc=PACNMPC(PACNMPCConfig(stochastic=stochastic, seed=seed))
    )

    renderer = None
    frames = []
    if video:
        import mujoco
        renderer = mujoco.Renderer(plant.model, width=640, height=480)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth, cam.elevation, cam.distance = 90.0, -12.0, 2.6
        cam.lookat[:] = [0.0, 0.0, 0.6]

    sens = plant.reset(init_base_z=0.62)
    n_steps = int(duration_s / CONTROL_DT)
    v_des = np.zeros(2)
    rng_push = np.random.default_rng(seed + 9999)

    log = {k: [] for k in ("t", "z", "roll", "pitch", "bound", "viol_hat", "stance")}
    n_hops = 0
    prev_stance = False
    fell = False
    t_solve = []

    for k in range(n_steps):
        t = k * CONTROL_DT
        t0 = time.perf_counter()
        tau, pwm, info = ctrl.step(sens, v_des)
        t_solve.append(time.perf_counter() - t0)

        # impulsive pushes: random horizontal direction, fixed within each window
        push_f = None
        if push_n > 0.0 and t > 1.0:
            phase = (t + 1e-9) % push_every_s
            if phase < CONTROL_DT:
                ang = rng_push.uniform(0, 2 * np.pi)
                cur_push_dir = np.array([np.cos(ang), np.sin(ang), 0.0])
                run_episode._push_dir = cur_push_dir  # window-scoped state
            if phase < push_dur_s and getattr(run_episode, "_push_dir", None) is not None:
                push_f = push_n * run_episode._push_dir
        sens = plant.step(tau, pwm, extra_force_w=push_f)

        if info["stance"] and not prev_stance:
            n_hops += 1
        prev_stance = info["stance"]

        log["t"].append(t)
        log["z"].append(float(sens.gt_base_pos_w[2]))
        log["roll"].append(float(info["rpy"][0]))
        log["pitch"].append(float(info["rpy"][1]))
        log["bound"].append(float(info["bound"]))
        log["viol_hat"].append(float(info["viol_hat"]))
        log["stance"].append(bool(info["stance"]))

        if abs(info["rpy"][0]) > np.deg2rad(60) or abs(info["rpy"][1]) > np.deg2rad(60) \
           or sens.gt_base_pos_w[2] < 0.15:
            fell = True
            break

        if renderer is not None and k % 25 == 0:  # 20 fps
            cam.lookat[0], cam.lookat[1] = sens.gt_base_pos_w[0], sens.gt_base_pos_w[1]
            renderer.update_scene(plant.data, camera=cam)
            frames.append(renderer.render().copy())

    if video and frames:
        import imageio.v2 as imageio
        imageio.mimsave(video, frames, fps=20, macro_block_size=1)
        print(f"video -> {video}")

    result = {
        "fell": fell,
        "n_hops": n_hops,
        "t_survive": log["t"][-1] if log["t"] else 0.0,
        "roll_max_deg": float(np.rad2deg(np.max(np.abs(log["roll"])))) if log["roll"] else 0.0,
        "pitch_max_deg": float(np.rad2deg(np.max(np.abs(log["pitch"])))) if log["pitch"] else 0.0,
        "bound_mean": float(np.nanmean(log["bound"])) if log["bound"] else float("nan"),
        "solve_ms_mean": 1e3 * float(np.mean(t_solve)),
        "solve_ms_p99_outer": 1e3 * float(np.percentile(t_solve, 99)),
        "log": log,
    }
    if verbose:
        status = "FELL" if fell else "OK"
        print(
            f"[{status}] hops={n_hops:3d} survive={result['t_survive']:.1f}s "
            f"|roll|max={result['roll_max_deg']:.1f}deg |pitch|max={result['pitch_max_deg']:.1f}deg "
            f"bound={result['bound_mean']:.3f} solve(mean/p99)={result['solve_ms_mean']:.1f}/{result['solve_ms_p99_outer']:.1f}ms"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deterministic", action="store_true", help="baseline NMPC (no uncertainty sampling)")
    ap.add_argument("--hard", action="store_true", help="adverse true plant (low mu, ground offset, payload, weak props)")
    ap.add_argument("--trials", type=int, default=1, help="Monte-Carlo trials over random plants")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--push", type=float, default=0.0, help="push force (N), every 2.5 s for 0.1 s")
    args = ap.parse_args()

    stochastic = not args.deterministic
    tag = "PAC-NMPC (stochastic)" if stochastic else "NMPC (deterministic baseline)"
    print(f"=== {tag} ===")

    if args.trials <= 1:
        unc = PlantUncertainty()
        if args.hard:
            unc = PlantUncertainty(
                friction_mu=0.18, ground_z_offset_m=0.04, payload_mass_kg=0.5,
                thrust_scale=np.full(6, 0.8), gyro_noise_std=0.02,
            )
        run_episode(stochastic=stochastic, plant_unc=unc,
                    duration_s=args.duration, seed=args.seed, video=args.video,
                    push_n=args.push)
        return

    # Monte-Carlo: random true plants drawn from a "reality" distribution
    rng = np.random.default_rng(args.seed)
    ok = 0
    hops = []
    for i in range(args.trials):
        unc = PlantUncertainty(
            friction_mu=float(np.clip(rng.normal(0.3, 0.12), 0.10, 0.9)),
            ground_z_offset_m=float(rng.normal(0.0, 0.03)),
            payload_mass_kg=float(np.clip(rng.normal(0.0, 0.3), 0.0, 1.0)),
            thrust_scale=np.clip(rng.normal(1.0, 0.10, 6), 0.6, 1.3),
            gyro_noise_std=0.02,
        )
        r = run_episode(stochastic=stochastic, plant_unc=unc,
                        duration_s=args.duration, seed=args.seed + i, verbose=True,
                        push_n=args.push)
        ok += int(not r["fell"])
        hops.append(r["n_hops"])
    print(f"\nsuccess rate: {ok}/{args.trials} = {100.0*ok/args.trials:.0f}%   "
          f"hops mean={np.mean(hops):.1f}")


if __name__ == "__main__":
    main()
