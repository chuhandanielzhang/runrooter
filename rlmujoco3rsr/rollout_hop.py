"""Render the trained hopping policy (hop_policy.params) to gif/mp4."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import jax, jax.numpy as jp
import numpy as np
import mujoco
import imageio.v2 as imageio

from train_hop import HopEnv, XML  # includes the jax compat shim
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

env = HopEnv()
params = model.load_params("hop_policy.params")

ppo_net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
make_policy = ppo_networks.make_inference_fn(ppo_net)
policy = make_policy((params[0], params[1]), deterministic=True)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_policy = jax.jit(policy)

rng = jax.random.PRNGKey(42)
state = jit_reset(rng)

qposes, base_zs = [], []
n_ctrl = 400  # 4 s at 100 Hz control
for i in range(n_ctrl):
    rng, k = jax.random.split(rng)
    act, _ = jit_policy(state.obs, k)
    state = jit_step(state, act)
    qposes.append(np.array(state.pipeline_state.qpos))
    base_zs.append(float(state.pipeline_state.qpos[2]))
    if state.done > 0.5:
        print(f"episode done at ctrl step {i}")
        break

base_zs = np.array(base_zs)
print(f"rollout: {len(qposes)} steps  base_z min={base_zs.min():.3f} max={base_zs.max():.3f} "
      f"mean={base_zs.mean():.3f}  hops above 0.65: {(base_zs > 0.65).sum()}")

# render collected qpos with CPU MuJoCo
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
ren = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 110

frames = []
for k, q in enumerate(qposes):
    if k % 4:  # 25 fps
        continue
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [q[0], q[1], max(0.5, q[2])]
    ren.update_scene(d, cam)
    frames.append(ren.render())

imageio.mimsave("hop_policy.gif", frames, fps=25)
imageio.mimsave("hop_policy.mp4", frames, fps=25, quality=8)
print(f"wrote hop_policy.gif / hop_policy.mp4 ({len(frames)} frames)")
