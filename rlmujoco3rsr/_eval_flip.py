"""Roll out flip_policy on classic CPU MuJoCo + gif."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import numpy as np
import jax
import mujoco
import imageio.v2 as imageio

from train_flip import (FlipEnv, XML, HIP_HOME, ACT_DIM, THRUST_MAX,
                        FRAME, HIST, T_END)
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

PARAMS = sys.argv[1] if len(sys.argv) > 1 else "flip_policy.params"
HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]
ACTION_SCALE = 0.80

env = FlipEnv()
params = model.load_params(PARAMS)
net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
policy = jax.jit(ppo_networks.make_inference_fn(net)((params[0], params[1]),
                                                      deterministic=True))
key = jax.random.PRNGKey(0)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
home_ctrl = m.key_ctrl[0].copy()

def frame_obs(prev_act, t):
    q, v = d.qpos, d.qvel
    w, x, y, z = q[3:7]
    grav = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    return np.concatenate([grav, v[3:6], q[HIP_QPOS] - HIP_HOME,
                           v[HIP_QVEL], prev_act, [t / T_END]]).astype(np.float32)

prev_act = np.zeros(ACT_DIM, np.float32)
hist = np.tile(frame_obs(prev_act, 0.0), HIST)
qposes, pitch_acc = [], 0.0
t = 0.0
for i in range(int(T_END * 100)):
    act, _ = policy(hist, key)
    act = np.clip(np.asarray(act), -1, 1)
    d.ctrl[0:3] = HIP_HOME + ACTION_SCALE * act[0:3]
    d.ctrl[3:6] = home_ctrl[3:6]
    d.ctrl[6:9] = 0.5 * THRUST_MAX * (act[3:6] + 1.0)
    for _ in range(20):
        mujoco.mj_step(m, d)
    pitch_acc += d.qvel[4] * 0.01
    t += 0.01
    prev_act = act
    hist = np.concatenate([hist[FRAME:], frame_obs(prev_act, t)])
    qposes.append(d.qpos.copy())

q = np.array(qposes)
up = 1 - 2*(q[:, 4]**2 + q[:, 5]**2)
print(f"total pitch rotation: {pitch_acc:.2f} rad ({np.degrees(pitch_acc):.0f} deg)  (full flip = +-6.28)")
print(f"apex z: {q[:, 2].max():.3f}  final z: {q[-1, 2]:.3f}  final upright: {up[-1]:.2f}")
print(f"min upright during motion: {up.min():.2f} (must go to -1 for a flip)")

dr = mujoco.MjData(m)
ren = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.distance = 2.8; cam.elevation = -10; cam.azimuth = 90
frames = []
for k, qq in enumerate(qposes):
    if k % 2:
        continue
    dr.qpos[:] = qq
    mujoco.mj_forward(m, dr)
    cam.lookat[:] = [qq[0], qq[1], max(0.6, qq[2])]
    ren.update_scene(dr, cam)
    frames.append(ren.render())
out = PARAMS.replace(".params", "") + ".gif"
imageio.mimsave(out, frames, fps=50)
print("wrote", out)
