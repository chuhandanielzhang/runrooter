"""Sim-to-sim validation (BRUCE Fig.7 style): policy trained in MJX (JAX solver)
is evaluated in classic CPU MuJoCo (different physics backend/solver).
If the policy survives this backend gap, it is more likely to survive sim2real.

Usage: python sim2sim_check.py [params_file]   (default hop_policy.params)
"""
import sys
import numpy as np
import jax
import mujoco

from train_hop import HopEnv, XML, HIP_HOME
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

PARAMS = sys.argv[1] if len(sys.argv) > 1 else "hop_policy.params"
HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]

env = HopEnv()
params = model.load_params(PARAMS)
ppo_net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
policy = jax.jit(ppo_networks.make_inference_fn(ppo_net)((params[0], params[1]),
                                                          deterministic=True))

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, key)
n_sub = 20  # 100 Hz control, same as training

def obs_cpu(d):
    q, v = d.qpos, d.qvel
    w, x, y, z = q[3:7]
    grav = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    return np.concatenate([
        [q[2]], grav, v[0:3], v[3:6],
        q[HIP_QPOS] - HIP_HOME, v[HIP_QVEL], [q[21]],
    ]).astype(np.float32)

rng = jax.random.PRNGKey(0)
base_z, touch = [], []
survived = 0
N = 400  # 4 s
for i in range(N):
    rng, k = jax.random.split(rng)
    act, _ = policy(obs_cpu(d), k)
    act = np.clip(np.array(act), -1, 1)
    d.ctrl[0:3] = HIP_HOME + 0.30 * act
    for _ in range(n_sub):
        mujoco.mj_step(m, d)
    base_z.append(d.qpos[2]); touch.append(d.sensordata[-1])
    up = 1 - 2*(d.qpos[4]**2 + d.qpos[5]**2)
    if d.qpos[2] < 0.30 or up < 0.6 or np.isnan(d.qpos[2]):
        survived = i
        break
else:
    survived = N

base_z, touch = np.array(base_z), np.array(touch)
airborne = touch < 1e-6
runs, cnt = [], 0
for a in airborne:
    cnt = cnt + 1 if a else (runs.append(cnt) or 0 if cnt else 0)
if cnt: runs.append(cnt)
flights = [r for r in runs if r >= 3]

print(f"params: {PARAMS}")
print(f"survived: {survived}/{N} ctrl steps ({survived*0.01:.2f} s)")
print(f"base_z range: [{base_z.min():.3f}, {base_z.max():.3f}]")
print(f"flight phases (>=30ms): {len(flights)}")
print("VERDICT:", "PASS - hops in classic MuJoCo too" if survived == N and len(flights) >= 2
      else "PARTIAL - survives but no real hopping" if survived == N
      else "FAIL - falls under backend change (sim2real gap would be worse)")
