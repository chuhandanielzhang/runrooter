"""Evaluate hop_policy_real.params (DR + HW obs + 9 Nm torque clamp):
rollout in MJX, record torques + hopping, render gif."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import jax
import numpy as np
import mujoco
import imageio.v2 as imageio

from train_hop import HopEnv, XML
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

env = HopEnv(hw_obs=True)
import sys
PARAMS = sys.argv[1] if len(sys.argv) > 1 else "hop_policy_real.params"
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
params = model.load_params(PARAMS)
ppo_net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
policy = jax.jit(ppo_networks.make_inference_fn(ppo_net)((params[0], params[1]),
                                                          deterministic=True))
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

rng = jax.random.PRNGKey(42)
state = jit_reset(rng)

qposes, base_zs, taus, touches, thrusts = [], [], [], [], []
for i in range(N_STEPS):
    rng, k = jax.random.split(rng)
    act, _ = policy(state.obs, k)
    state = jit_step(state, act)
    qposes.append(np.array(state.pipeline_state.qpos))
    base_zs.append(float(state.pipeline_state.qpos[2]))
    taus.append(np.array(state.pipeline_state.actuator_force[0:3]))
    thrusts.append(np.array(state.pipeline_state.ctrl[6:9]))
    touches.append(float(state.pipeline_state.sensordata[-1]))
    if state.done > 0.5:
        print(f"episode done at ctrl step {i}")
        break

base_zs = np.array(base_zs)
taus = np.abs(np.array(taus))
touches = np.array(touches)
air = touches < 1e-6
# count flight phases
flights = int(np.sum(np.diff(air.astype(int)) == 1))
print(f"rollout {len(base_zs)} steps  base_z min={base_zs.min():.3f} max={base_zs.max():.3f} mean={base_zs.mean():.3f}")
print(f"flight phases: {flights}   airtime fraction: {air.mean()*100:.0f}%")
print(f"|tau| peak={taus.max():.2f}  p99={np.percentile(taus,99):.2f}  rms={np.sqrt((taus**2).mean()):.2f} Nm")
print(f"time over 9 Nm: {(taus > 9.001).mean()*100:.2f}%   over 25 Nm: {(taus > 25.001).mean()*100:.2f}%")
thr = np.array(thrusts)
print(f"thrust per-arm mean={thr.mean():.2f} N  max={thr.max():.2f} N  total-mean={thr.sum(axis=1).mean():.1f} N (weight 36.8 N)")

# torque plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = np.arange(len(taus)) * 0.01
fig, ax = plt.subplots(figsize=(9, 4))
for j in range(3):
    ax.plot(t, taus[:, j], lw=0.8, label=f"hip {j+1}")
ax.axhline(9, c="r", ls="--", lw=1, label="AK60 peak 9 Nm")
ax.axhline(3, c="orange", ls="--", lw=1, label="AK60 rated 3 Nm")
ax.set_xlabel("time [s]"); ax.set_ylabel("|torque| [Nm]")
ax.set_title("hip torques, deployable policy (hop_policy_real)")
ax.legend(ncol=5, fontsize=8); ax.grid(alpha=0.3); ax.margins(x=0)
fig.tight_layout(); fig.savefig(PARAMS.replace(".params","") + "_torque.png", dpi=110)
print("wrote hop_policy_real_torque.png")

# render gif
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
ren = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 110
frames = []
for k, q in enumerate(qposes):
    if k % 4:
        continue
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [q[0], q[1], max(0.5, q[2])]
    ren.update_scene(d, cam)
    frames.append(ren.render())
base = PARAMS.replace(".params", "")
imageio.mimsave(base + ".gif", frames, fps=25)
imageio.mimsave(base + ".mp4", frames, fps=25, quality=8)
print(f"wrote {base}.gif / .mp4 ({len(frames)} frames)")
