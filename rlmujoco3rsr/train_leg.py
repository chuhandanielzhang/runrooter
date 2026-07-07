"""
PURE-LEG low-level policy for the closed-loop 3-RSR parallel leg (MJX + brax PPO).

Low level of the hierarchical stack (bottom-up training; the standard recipe in
ANYmal-Parkour / ASE / MCP). The leg is the only actuator: the 3 tilt servos and
3 rotor thrusts are frozen at zero (props/flight attitude belong to the high level).

The high level (SRB / centroidal QP, like Cao's case) drives this policy with three
hardware-meaningful setpoints:

    fz_cmd  : desired vertical foot push force in stance  (height / energy knob)
    vxy_cmd : desired horizontal base velocity            (direction / BALANCE knob,
                                                            realized via Raibert foot
                                                            placement -> keeps it upright
                                                            and steerable)
    L_cmd   : desired leg length in flight                (posture / landing prep)

v1 (scalar vertical force only) tipped over because it had no balance objective. v2
adds velocity tracking + Raibert foot placement (proven in the end-to-end hop policy)
so the high level can both inject energy AND steer/balance the leg. The policy is
still task-agnostic: it tracks setpoints, it does not "know" it is hopping.

Run:
    python train_leg.py --hw --dr --timesteps 60000000 --out leg_policy.params
"""
import argparse, functools, time
import jax, jax.numpy as jp
import numpy as np

from train_hop import (XML, HIP_QPOS, HIP_QVEL, HIP_HOME, MG, Z_STAND,
                       domain_randomize)

import mujoco
from mujoco import mjx
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

# ---- pure-leg interface ----
ACT_DIM = 3                    # 3 hips only (servos/thrust frozen at 0)
ACTION_SCALE = 1.10            # bigger stroke than the hop policy (0.80) -> higher jumps
FALL_Z = 0.24
F_MAX = 3.0 * MG               # N, max commanded vertical foot force (~3x weight)
V_MAX = 0.8                    # m/s, horizontal velocity command range
L_MIN, L_MAX = 0.28, 0.44      # m, commanded leg length (base_z - foot_z); home ~0.369
FOOT_Z_QPOS = 21               # foot free-body z in qpos
CMD_RESAMPLE_F = 0.30          # mean s between force-command switches (gait-fast)
CMD_RESAMPLE_V = 2.5           # mean s between velocity-command switches
CMD_RESAMPLE_L = 0.50          # mean s between length-command switches

# observation (hardware-available only), single frame =
#   grav(3) gyro(3) hip_q(3) hip_dq(3) v_hat_xy(2) contact(1)
#   fz_cmd(1) vxy_cmd(2) L_cmd(1) prev_act(3) = 22
LEG_FRAME = 22
HW_HIST = 5
HW_NOISE = jp.concatenate([
    jp.full((3,), 0.02),       # projected gravity
    jp.full((3,), 0.10),       # gyro
    jp.full((3,), 0.01),       # hip q
    jp.full((3,), 0.15),       # hip dq
    jp.full((2,), 0.10),       # v_hat
    jp.array([0.05]),          # contact
    jp.zeros(4),               # fz_cmd + vxy_cmd + L_cmd (exact, commanded)
    jp.zeros(ACT_DIM),         # prev action (exact)
])


class LegEnv(Env):
    def __init__(self, ctrl_dt=0.01, action_scale=ACTION_SCALE,
                 fall_z=FALL_Z, randomize=False, hw_obs=False):
        self._hw = hw_obs
        self._ctrl_dt = ctrl_dt
        self._mj = mujoco.MjModel.from_xml_path(XML)
        self._mjx = mjx.put_model(self._mj)
        self._n_sub = max(1, int(round(ctrl_dt / self._mj.opt.timestep)))
        self._home_q = jp.array(self._mj.key_qpos[0])
        self._home_ctrl = jp.array(self._mj.key_ctrl[0])
        self._action_scale = action_scale
        self._fall_z = fall_z
        self._randomize = randomize

    @property
    def sys(self):
        return self._mjx

    @sys.setter
    def sys(self, new_sys):
        self._mjx = new_sys

    def _frame(self, data, prev_action, fz_cmd, vxy_cmd, l_cmd) -> jp.ndarray:
        q, v = data.qpos, data.qvel
        w, x, y, z = q[3], q[4], q[5], q[6]
        grav = jp.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        gyro = v[3:6]
        R_T = jp.array([
            [1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)],
            [2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)],
        ])
        v_hat = R_T @ v[0:3]
        contact = (data.sensordata[-1] > 1e-6).astype(jp.float32)
        return jp.concatenate([grav, gyro, q[HIP_QPOS] - HIP_HOME, v[HIP_QVEL],
                               v_hat, contact[None],
                               fz_cmd[None], vxy_cmd, l_cmd[None], prev_action])

    def _sample_cmd(self, k):
        kf, kv, kvz, kl = jax.random.split(k, 4)
        fz_cmd = jax.random.uniform(kf, (), minval=0.0, maxval=1.0)   # -> [0,F_MAX]
        # 60% hop in place (vxy=0), 40% follow a random horizontal velocity
        zero_v = jax.random.uniform(kvz) < 0.6
        vxy_cmd = jp.where(zero_v, jp.zeros(2),
                           jax.random.uniform(kv, (2,), minval=-V_MAX, maxval=V_MAX))
        l_cmd = jax.random.uniform(kl, (), minval=L_MIN, maxval=L_MAX)
        return fz_cmd, vxy_cmd, l_cmd

    def reset(self, rng) -> State:
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        q = self._home_q
        q = q.at[2].add(0.02 * jax.random.normal(k1))
        q = q.at[HIP_QPOS].add(0.05 * jax.random.normal(k2, (3,)))
        data = mjx.make_data(self._mjx).replace(qpos=q, ctrl=self._home_ctrl)
        data = mjx.forward(self._mjx, data)
        fz_cmd, vxy_cmd, l_cmd = self._sample_cmd(k3)
        metrics = {"base_z": q[2], "f_err": jp.float32(0.0), "v_err": jp.float32(0.0)}
        info = {
            "rng": rng,
            "prev_action": jp.zeros(ACT_DIM),
            "prev_action2": jp.zeros(ACT_DIM),
            "prev_f": jp.float32(0.0),
            "t": jp.float32(0.0),
            "fz_cmd": fz_cmd,
            "vxy_cmd": vxy_cmd,
            "l_cmd": l_cmd,
            "delay": (jax.random.uniform(k4) < 0.5).astype(jp.float32) if self._randomize
                     else jp.float32(0.0),
        }
        if self._hw:
            frame = self._frame(data, jp.zeros(ACT_DIM), fz_cmd, vxy_cmd, l_cmd)
            info["hist"] = jp.tile(frame, HW_HIST)
            obs = info["hist"]
        else:
            obs = self._frame(data, jp.zeros(ACT_DIM), fz_cmd, vxy_cmd, l_cmd)
        return State(data, obs, jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        action = jp.clip(action, -1.0, 1.0)
        info = dict(state.info)
        rng, k_noise, k_rf, k_cf, k_rv, k_cv, k_cvz, k_rl, k_cl = jax.random.split(info["rng"], 9)
        info["rng"] = rng

        # in-episode piecewise-constant command resampling (force / velocity / length)
        rf = jax.random.uniform(k_rf) < (self._ctrl_dt / CMD_RESAMPLE_F)
        info["fz_cmd"] = jp.where(rf, jax.random.uniform(k_cf, (), minval=0.0, maxval=1.0),
                                  info["fz_cmd"])
        rv = jax.random.uniform(k_rv) < (self._ctrl_dt / CMD_RESAMPLE_V)
        new_zero_v = jax.random.uniform(k_cvz) < 0.5
        new_v = jp.where(new_zero_v, jp.zeros(2),
                         jax.random.uniform(k_cv, (2,), minval=-V_MAX, maxval=V_MAX))
        info["vxy_cmd"] = jp.where(rv, new_v, info["vxy_cmd"])
        rl = jax.random.uniform(k_rl) < (self._ctrl_dt / CMD_RESAMPLE_L)
        info["l_cmd"] = jp.where(rl, jax.random.uniform(k_cl, (), minval=L_MIN, maxval=L_MAX),
                                 info["l_cmd"])

        applied = jp.where(info["delay"] > 0.5, info["prev_action"], action)
        a_prev, a_prev2 = info["prev_action"], info["prev_action2"]
        info["prev_action2"] = info["prev_action"]
        info["prev_action"] = action

        hip_cmd = HIP_HOME + self._action_scale * applied[0:3]
        ctrl = (self._home_ctrl.at[0:3].set(hip_cmd)
                .at[3:6].set(0.0).at[6:9].set(0.0))

        def f(d, _):
            return mjx.step(self._mjx, d.replace(ctrl=ctrl)), None
        data, _ = jax.lax.scan(f, state.pipeline_state, None, length=self._n_sub)

        q, v = data.qpos, data.qvel
        base_z = q[2]
        w, x, y, zq = q[3], q[4], q[5], q[6]
        up = 1 - 2*(x*x + y*y)

        f_touch = data.sensordata[-1]
        in_contact = (f_touch > 1e-6).astype(jp.float32)
        leg_len = base_z - q[FOOT_Z_QPOS]
        hip_vel = v[HIP_QVEL]
        v_foot = v[18:21]

        fz_cmd, vxy_cmd, l_cmd = info["fz_cmd"], info["vxy_cmd"], info["l_cmd"]
        fz_des = fz_cmd * F_MAX

        # --- force tracking in stance (height/energy knob) ---
        f_err = (f_touch - fz_des) / MG
        r_force = 2.0 * in_contact * jp.exp(-2.0 * f_err ** 2)
        # --- horizontal velocity tracking (balance/direction knob) ---
        v_err = v[0:2] - vxy_cmd
        r_vel = 1.5 * jp.exp(-4.0 * jp.sum(v_err ** 2))
        # --- Raibert foot placement (only with an active velocity command) ---
        foot_offset = q[19:21] - q[0:2]
        raibert_target = 0.15 * v[0:2] + 0.10 * v_err
        cmd_active = jp.clip(jp.sqrt(jp.sum(vxy_cmd ** 2)) / 0.2, 0.0, 1.0)
        r_place = 0.6 * cmd_active * (1.0 - in_contact) * jp.exp(-30.0 * jp.sum((foot_offset - raibert_target) ** 2))
        # --- leg length tracking in flight (posture / landing prep) ---
        l_err = leg_len - l_cmd
        r_len = 1.0 * (1.0 - in_contact) * jp.exp(-120.0 * l_err ** 2)
        r_upright = 0.5 * up
        r_alive = 1.0

        # position anchor when commanded to stay (suppress random walk)
        zero_cmd = (jp.sum(vxy_cmd ** 2) < 1e-6).astype(jp.float32)
        c_pos = 0.10 * zero_cmd * jp.clip(jp.sum(q[0:2] ** 2), 0.0, 4.0)
        c_ctrl = 0.01 * jp.sum(action ** 2)
        c_smooth = 0.02 * jp.sum((action - 2.0 * a_prev + a_prev2) ** 2)
        c_grf_rate = 0.05 * jp.clip((f_touch - info["prev_f"]) / MG, 0.0, 10.0)
        info["prev_f"] = f_touch
        c_slip = 0.10 * in_contact * jp.clip(jp.sum(v_foot[0:2] ** 2), 0.0, 4.0)
        c_air_hold = 0.02 * (1.0 - in_contact) * jp.sum(hip_vel ** 2)

        reward = (r_alive + r_force + r_vel + r_place + r_len + r_upright
                  - c_pos - c_ctrl - c_smooth - c_grf_rate - c_slip - c_air_hold)

        info["t"] = info["t"] + self._ctrl_dt
        fell = (base_z < self._fall_z) | (up < 0.5) | jp.isnan(base_z)
        done = fell.astype(jp.float32)

        if self._hw:
            frame = self._frame(data, action, fz_cmd, vxy_cmd, l_cmd)
            if self._randomize:
                frame = frame + HW_NOISE * jax.random.uniform(k_noise, frame.shape, minval=-1.0, maxval=1.0)
            info["hist"] = jp.concatenate([info["hist"][LEG_FRAME:], frame])
            obs = info["hist"]
        else:
            obs = self._frame(data, action, fz_cmd, vxy_cmd, l_cmd)

        state.metrics.update(base_z=base_z,
                             f_err=in_contact * jp.abs(f_err),
                             v_err=jp.sqrt(jp.sum(v_err ** 2)))
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, info=info)

    @property
    def observation_size(self):
        return LEG_FRAME * HW_HIST if self._hw else LEG_FRAME

    @property
    def action_size(self):
        return ACT_DIM

    @property
    def backend(self):
        return "mjx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60_000_000)
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--episode_len", type=int, default=500)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--out", type=str, default="leg_policy.params")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend(), jax.devices())
    env = LegEnv(randomize=args.dr, hw_obs=args.hw)
    print(f"obs={env.observation_size} act={env.action_size} "
          f"n_substeps={env._n_sub} DR={args.dr} HW={args.hw}")

    times = [time.time()]
    def progress(step, metrics):
        times.append(time.time())
        r = metrics.get("eval/episode_reward", float("nan"))
        fe = metrics.get("eval/episode_f_err", float("nan"))
        ve = metrics.get("eval/episode_v_err", float("nan"))
        print(f"[{step:>10,}] reward={r:8.3f}  f_err={fe:.3f}  v_err={ve:.3f}  "
              f"({times[-1]-times[-2]:.1f}s)")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.timesteps,
        num_evals=10,
        episode_length=args.episode_len,
        num_envs=args.num_envs,
        batch_size=256,
        num_minibatches=32,
        unroll_length=20,
        num_updates_per_batch=4,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        normalize_observations=True,
        action_repeat=1,
        seed=0,
        randomization_fn=domain_randomize if args.dr else None,
    )
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    from brax.io import model
    model.save_params(args.out, params)
    print("saved", args.out)


if __name__ == "__main__":
    main()
