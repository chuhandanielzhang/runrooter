"""
Front-flip policy for the 3-RSR hopper (MJX + brax PPO).

Method: centroidal-angular-velocity (CAV) phase reward from
"Learning Impact-Rich Rotational Maneuvers via Centroidal Velocity Rewards"
(arXiv 2505.12222, one-leg hopper front flip, hardware-validated):
  - takeoff phase  [0, 0.5 s):    no rotation reward (crouch + jump)
  - aerial  phase  [0.5, 1.05 s): r = clip(alpha . w_com, -0.1, 10), alpha = +Y pitch
  - landing phase  [1.05, 2.0 s]: r = -0.5 * min(|L_com|, 2.5)  + upright recovery
Base link carries ~90% of total mass, so base angular velocity is used as the
CAV proxy. Props (TWR 0.82) may assist rotation via differential thrust.

Run:  python train_flip.py --timesteps 40000000
"""
import argparse, functools, time
import jax, jax.numpy as jp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        n = len(devices)
        x = jax.tree_util.tree_map(
            lambda a: jp.broadcast_to(jp.asarray(a)[None], (n,) + jp.shape(a)), x)
        if n == 1:
            return jax.device_put(x, devices[0])
        return jax.device_put_sharded(
            [jax.tree_util.tree_map(lambda a, i=i: a[i], x) for i in range(n)], devices)
    jax.device_put_replicated = _device_put_replicated

import mujoco
from mujoco import mjx
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

XML = "three_leg_3rsr_closed.xml"
HIP_QPOS = jp.array([7, 10, 13])
HIP_QVEL = jp.array([6, 9, 12])
HIP_HOME = -0.060299

ACT_DIM = 6                     # 3 hips + 3 prop thrusts
THRUST_MAX = 10.0               # full prop authority for the flip (TWR 0.82 total)
FRAME = 3 + 3 + 3 + 3 + ACT_DIM + 1   # grav gyro q dq prev_action phase = 19
HIST = 5

T_JUMP, T_LAND, T_END = 0.5, 1.05, 2.0
FLIP_AXIS = jp.array([0.0, 1.0, 0.0])     # pitch (front flip)
I_PITCH = 0.06                  # approx body pitch inertia for |L| (kg m^2)
W_DES = 2.0 * jp.pi / (T_LAND - T_JUMP)   # ~11.4 rad/s completes the flip in the air
RSI_PROB = 0.5                  # reference-state init: spawn mid-flip to learn landing

NOISE = jp.concatenate([
    jp.full((3,), 0.02), jp.full((3,), 0.10),
    jp.full((3,), 0.01), jp.full((3,), 0.15),
    jp.zeros(ACT_DIM), jp.zeros(1),
])


class FlipEnv(Env):
    def __init__(self, ctrl_dt=0.01, action_scale=0.80, randomize=False):
        self._ctrl_dt = ctrl_dt
        self._mj = mujoco.MjModel.from_xml_path(XML)
        self._mjx = mjx.put_model(self._mj)
        self._n_sub = max(1, int(round(ctrl_dt / self._mj.opt.timestep)))
        self._home_q = jp.array(self._mj.key_qpos[0])
        self._home_ctrl = jp.array(self._mj.key_ctrl[0])
        self._action_scale = action_scale
        self._randomize = randomize

    @property
    def sys(self):
        return self._mjx

    @sys.setter
    def sys(self, new_sys):
        self._mjx = new_sys

    def _frame(self, data, prev_action, t):
        q, v = data.qpos, data.qvel
        w, x, y, z = q[3], q[4], q[5], q[6]
        grav = jp.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        return jp.concatenate([grav, v[3:6], q[HIP_QPOS] - HIP_HOME, v[HIP_QVEL],
                               prev_action, jp.array([t / T_END])])

    def reset(self, rng) -> State:
        rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
        q = self._home_q
        q = q.at[2].add(0.02 * jax.random.normal(k1))
        q = q.at[HIP_QPOS].add(0.03 * jax.random.normal(k2, (3,)))
        # --- reference-state initialization (DeepMimic-style): half the episodes
        # start mid-flip in the air, already rotating, so the landing/recovery
        # part gets learned without first discovering takeoff+rotation ---
        rsi = jax.random.uniform(k3) < RSI_PROB
        th = jax.random.uniform(k4, (), minval=0.5, maxval=5.0)   # pitch progress [rad]
        q_rsi = q.at[2].set(0.95)
        q_rsi = q_rsi.at[3:7].set(jp.array([jp.cos(th/2), 0.0, jp.sin(th/2), 0.0]))
        q = jp.where(rsi, q_rsi, q)
        v0 = jp.zeros(self._mjx.nv)
        vz0 = jax.random.uniform(k5, (), minval=-1.0, maxval=2.0)
        v_rsi = v0.at[2].set(vz0).at[4].set(W_DES)
        v = jp.where(rsi, v_rsi, v0)
        t0 = jp.where(rsi, T_JUMP + th / W_DES, 0.0)
        data = mjx.make_data(self._mjx).replace(qpos=q, qvel=v, ctrl=self._home_ctrl)
        data = mjx.forward(self._mjx, data)
        info = {"rng": rng, "prev_action": jp.zeros(ACT_DIM),
                "t": jp.float32(t0), "pitch_acc": jp.where(rsi, th, jp.float32(0.0))}
        frame = self._frame(data, jp.zeros(ACT_DIM), jp.float32(0.0))
        info["hist"] = jp.tile(frame, HIST)
        metrics = {"pitch_total": jp.float32(0.0), "base_z": q[2],
                   "upright_end": jp.float32(0.0)}
        return State(data, info["hist"], jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        action = jp.clip(action, -1.0, 1.0)
        info = dict(state.info)
        rng, k_noise = jax.random.split(info["rng"], 2)
        info["rng"] = rng

        hip_cmd = HIP_HOME + self._action_scale * action[0:3]
        thrust_cmd = 0.5 * THRUST_MAX * (action[3:6] + 1.0)
        ctrl = self._home_ctrl.at[0:3].set(hip_cmd).at[6:9].set(thrust_cmd)

        def f(d, _):
            return mjx.step(self._mjx, d.replace(ctrl=ctrl)), None
        data, _ = jax.lax.scan(f, state.pipeline_state, None, length=self._n_sub)

        q, v = data.qpos, data.qvel
        base_z = q[2]
        w, x, y, zq = q[3], q[4], q[5], q[6]
        up = 1 - 2*(x*x + y*y)
        gyro = v[3:6]
        t_new = info["t"] + self._ctrl_dt
        info["t"] = t_new
        # accumulated pitch rotation (for metrics/success)
        info["pitch_acc"] = info["pitch_acc"] + gyro[1] * self._ctrl_dt

        in_takeoff = (t_new < T_JUMP).astype(jp.float32)
        in_air_ph = ((t_new >= T_JUMP) & (t_new < T_LAND)).astype(jp.float32)
        in_land = (t_new >= T_LAND).astype(jp.float32)

        # --- CAV phase reward, v2: TRACK the flip rate (exp kernel) instead of
        # "spin as fast as possible"; raw-rate term kept small for exploration ---
        w_pitch = jp.dot(FLIP_AXIS, gyro)
        r_cav = (in_air_ph * (4.0 * jp.exp(-0.10 * (w_pitch - W_DES) ** 2)
                              + 0.5 * jp.clip(w_pitch, -0.1, W_DES))
                 - in_land * 1.5 * jp.minimum(jp.abs(I_PITCH * w_pitch), 2.5))
        # jump high during takeoff/aerial: rotation needs airtime
        r_jump = 1.0 * (1.0 - in_land) * jp.clip(v[2], 0.0, 4.0) \
               + 1.0 * in_air_ph * jp.clip(base_z - 0.45, 0.0, 0.8)
        # landing recovery: upright + nominal height + low velocity
        r_recover = in_land * (2.0 * jp.clip(up, 0.0, 1.0)
                               + 1.0 * jp.exp(-8.0 * (base_z - 0.42) ** 2)
                               + 0.5 * jp.exp(-1.0 * jp.sum(v[0:3] ** 2)))
        # full-flip bonus held through a successful landing
        flip_done = (jp.abs(info["pitch_acc"]) > 5.8).astype(jp.float32)
        r_flip = 3.0 * in_land * flip_done * jp.clip(up, 0.0, 1.0)
        # regularizers (paper r_act / r_tau analogues)
        c_act = 0.05 * jp.sum((action - info["prev_action"]) ** 2)
        c_ctrl = 0.005 * jp.sum(action[0:3] ** 2)
        info["prev_action"] = action

        reward = r_cav + r_jump + r_recover + r_flip - c_act - c_ctrl

        # termination: only a hard crash; mid-air inversion is REQUIRED, so no
        # tilt cutoff during takeoff/aerial. In landing phase, ending up under
        # 0.15 m or still inverted past t=1.6 s ends the episode.
        crash = base_z < 0.12
        late_inverted = (t_new > 1.6) & (up < 0.0)
        done = (crash | late_inverted | jp.isnan(base_z)).astype(jp.float32)

        frame = self._frame(data, action, t_new)
        if self._randomize:
            frame = frame + NOISE * jax.random.uniform(k_noise, frame.shape, minval=-1.0, maxval=1.0)
        info["hist"] = jp.concatenate([info["hist"][FRAME:], frame])
        state.metrics.update(pitch_total=info["pitch_acc"], base_z=base_z,
                             upright_end=in_land * up)
        return state.replace(pipeline_state=data, obs=info["hist"],
                             reward=reward, done=done, info=info)

    @property
    def observation_size(self):
        return FRAME * HIST

    @property
    def action_size(self):
        return ACT_DIM

    @property
    def backend(self):
        return "mjx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=40_000_000)
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--out", type=str, default="flip_policy.params")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend(), jax.devices())
    env = FlipEnv(randomize=True)
    print(f"obs={env.observation_size} act={env.action_size}")

    times = [time.time()]
    def progress(step, metrics):
        times.append(time.time())
        r = metrics.get("eval/episode_reward", float("nan"))
        p = metrics.get("eval/episode_pitch_total", float("nan"))
        print(f"[{step:>10,}] reward={r:8.3f} pitch={p:6.2f}rad  ({times[-1]-times[-2]:.1f}s)")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.timesteps,
        num_evals=10,
        episode_length=int(T_END / 0.01),   # 2.0 s motion
        num_envs=args.num_envs,
        batch_size=256,
        num_minibatches=32,
        unroll_length=20,
        num_updates_per_batch=4,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.99,                   # short episode, value the landing
        normalize_observations=True,
        action_repeat=1,
        seed=0,
    )
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    from brax.io import model
    model.save_params(args.out, params)
    print("saved", args.out)


if __name__ == "__main__":
    main()
