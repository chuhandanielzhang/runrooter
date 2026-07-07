"""Render the sim2sim rollout (policy in classic CPU MuJoCo) to gif."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import numpy as np
import jax
import mujoco
import imageio.v2 as imageio

from train_hop import HopEnv, XML, HIP_HOME
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

PARAMS = sys.argv[1] if len(sys.argv) > 1 else "hop_policy.params"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sim2sim"
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
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))

def obs_cpu(d):
    q, v = d.qpos, d.qvel
    w, x, y, z = q[3:7]
    grav = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    return np.concatenate([
        [q[2]], grav, v[0:3], v[3:6],
        q[HIP_QPOS] - HIP_HOME, v[HIP_QVEL], [q[21]],
    ]).astype(np.float32)

ren = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 110

rng = jax.random.PRNGKey(0)
frames = []
for i in range(400):
    rng, k = jax.random.split(rng)
    act, _ = policy(obs_cpu(d), k)
    d.ctrl[0:3] = HIP_HOME + 0.30 * np.clip(np.array(act), -1, 1)
    for _ in range(20):
        mujoco.mj_step(m, d)
    up = 1 - 2*(d.qpos[4]**2 + d.qpos[5]**2)
    if d.qpos[2] < 0.30 or up < 0.6:
        print(f"fell at step {i}")
        break
    if i % 4 == 0:
        cam.lookat[:] = [d.qpos[0], d.qpos[1], max(0.5, d.qpos[2])]
        ren.update_scene(d, cam)
        frames.append(ren.render())

imageio.mimsave(f"{OUT}.gif", frames, fps=25)
imageio.mimsave(f"{OUT}.mp4", frames, fps=25, quality=8)
print(f"wrote {OUT}.gif / {OUT}.mp4 ({len(frames)} frames)")
